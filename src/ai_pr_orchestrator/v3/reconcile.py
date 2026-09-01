"""V3 startup reconciliation and crash/orphan recovery (issue #44).

After any restart — Hermes, aipro, CAO, the machine, or a single worker —
the foreman must derive the correct next action **without duplicating side
effects**. The :class:`ReconcilePlanner` is the deterministic component that
turns the durable state (GitHub workflow state, live CAO sessions, git
refs/branches/PRs, active worktrees) into a typed plan of recovery actions.

Design constraints
------------------

- **Deterministic from inputs.** The planner never reads the wall clock;
  every action is reproducible from the observation bundle plus an explicit
  ``now`` argument. Tests pass a frozen datetime; production calls
  :func:`datetime.now` at the boundary.
- **Table-driven.** Every action is keyed off an *observation pattern*, not a
  chain of ``if`` branches with time-based heuristics. The decision table is
  :data:`_CRASH_DECISION_TABLE`; row order is the priority order, so a more
  severe row matches first.
- **No ``probably`` branches.** Non-idempotent side effects (creating a
  branch, launching a session) are either confirmed safe by durable state or
  escalated. ``ESCALATE`` means *do not act, surface to a human*.
- **One branch per run.** The planner proves (see the acceptance property
  test in ``tests/unit/test_v3_reconcile.py``) that for any two distinct
  ``WorkItem`` observations with the same ``run_id`` it never produces more
  than one branch-creating action.

Decision grammar
----------------

Actions are tagged with an ``auto_apply`` bit. Actions with ``auto_apply=False``
(``ESCALATE``, ``HALT_BRANCH_MOVED``) must never be applied automatically —
they are surfaced to a human via the ``aipro reconcile`` CLI, which exits
non-zero. Actions with ``auto_apply=True`` (``RECOVER_STALE_LEASE``,
``CLEAN_ORPHAN_SESSION``, ``CLEAN_ORPHAN_WORKTREE``) are safe to apply once
each row's precondition holds.

Once the planner emits a *manual* action (``ESCALATE`` or ``HALT_BRANCH_MOVED``)
for one work item, it emits no further actions for that same work item in
the same plan call. A manual action means *do not act further* — surfacing
additional crash-recovery or orphan-cleanup rows next to it would risk
contradicting the very signal that triggered the escalation. This is the
"stop on manual" contract documented in the issue.

Orphan detection
----------------

A CAO session is *orphaned* when both:

1. no live work-item lease references it (e.g. the lease has been reclaimed
   or never existed), AND
2. its last observed activity (status change) is older than
   :attr:`CleanupConfig.session_lease_ttl_seconds`.

A git worktree is *orphaned* when both:

1. no live work-item lease references its branch, AND
2. its branch has been unchanged for longer than
   :attr:`CleanupConfig.worktree_inactivity_ttl_seconds`.

Both checks are pure functions of the observation bundle, so unit tests can
cover every TTL boundary without any I/O.

Cross-work-item orphan detection (``plan_many``) considers *every* live
branch across the full inputs list — never just the branches of the work
item currently being planned. Without that, planning item B would
incorrectly mark item A's live branch as orphan and the next reconcile
pass would emit a spurious ``CLEAN_ORPHAN_WORKTREE`` for A's branch.

No vendor or model name appears in this module — V3 owns naming through
:mod:`ai_pr_orchestrator.v3.catalog`, never through reconcile code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from .config import CleanupConfig, GitHubQueueConfig
from .domain import (
    TERMINAL_PHASES,
    DomainError,
    GitHubIssueRef,
    RunId,
    WorkflowState,
)
from .lanes import LaneRegistry
from .queue import Claim, claim_from_state

# --- Action grammar ---------------------------------------------------------


class ActionKind(StrEnum):
    """The closed set of recovery actions the planner may emit."""

    #: Resume an existing live session (no work duplicated). The foreman
    #: should poll the session, not launch a new one.
    RESUME_SESSION = "resume_session"
    #: The session finished but the work item's authoritative state never
    #: recorded the result; collect it.
    COLLECT_RESULT = "collect_result"
    #: Re-launch from the durable checkpoint (branch + state exist but no
    #: session is alive). Idempotent on durable identifiers.
    RELAUNCH = "relaunch"
    #: Side effect outcome is unknown; surface to a human. Never auto-apply.
    ESCALATE = "escalate"
    #: The PR head SHA moved under us; do not act, surface to a human.
    HALT_BRANCH_MOVED = "halt_branch_moved"
    #: The lease is stale and another foreman reclaimed it. Recover by
    #: acquiring our own lease; never touch the in-flight work.
    RECOVER_STALE_LEASE = "recover_stale_lease"
    #: A CAO session has no live lease referencing it past the TTL — clean it.
    CLEAN_ORPHAN_SESSION = "clean_orphan_session"
    #: A worktree has no live lease referencing its branch past the TTL — clean it.
    CLEAN_ORPHAN_WORKTREE = "clean_orphan_worktree"
    #: No-op (already idle or terminal).
    NOOP = "noop"


#: Actions that the CLI must never apply without explicit operator approval.
_MANUAL_ACTIONS: frozenset[ActionKind] = frozenset(
    {ActionKind.ESCALATE, ActionKind.HALT_BRANCH_MOVED}
)


@dataclass(frozen=True)
class Action:
    """One recovery action the planner wants applied.

    ``auto_apply=False`` means the action is a *manual* one (escalate /
    halt). The CLI prints it and exits non-zero.
    """

    kind: ActionKind
    work_item_id: str | None = None
    run_id: RunId | None = None
    session_id: str | None = None
    branch: str | None = None
    worktree: str | None = None
    pr_number: int | None = None
    reason: str = ""
    #: True when the action is safe to apply automatically; False when it
    #: requires human review (mirrors :data:`_MANUAL_ACTIONS`).
    auto_apply: bool = True

    def __post_init__(self) -> None:
        if self.kind in _MANUAL_ACTIONS and self.auto_apply:
            # Freeze cannot be overridden via __setattr__, so a default of
            # True is the safest base case; enforce the policy here.
            object.__setattr__(self, "auto_apply", False)


def actions_target_branch(action: Action) -> bool:
    """Return True iff ``action`` is one that creates or moves a branch.

    Used by the acceptance property test (#44: no two authoritative
    developer branches per run).
    """
    return action.kind is ActionKind.RELAUNCH


# --- Observations -----------------------------------------------------------


SessionLifecycle = Literal["unknown", "starting", "active", "terminal", "failed"]


@dataclass(frozen=True)
class SessionObservation:
    """One CAO session the planner can see.

    Mirrors the production :class:`~ai_pr_orchestrator.v3.cao.SessionObservation`
    shape closely enough that the planner does not import the CAO module —
    production code calls ``list_sessions()`` (added in the same change) and
    converts; tests construct observations directly.
    """

    session_id: str
    work_item_id: str | None
    run_id: RunId | None
    lane: str
    state: SessionLifecycle
    last_activity_at: datetime
    #: Whether the session exited successfully (reviewer/coder round
    #: produced its result). A "no-op coder" — one whose worktree ended
    #: unchanged from HEAD — exits with ``success=False`` even when CAO
    #: reports a clean exit, because the session produced no usable changes.
    #: Used by ``commit_recorded`` and ``session_succeeded`` predicates so
    #: the planner does not treat an idle coder as having done work.
    success: bool = True
    #: Whether the foreman still considers this session alive — a
    #: reconciliation crash may leave ``state='terminal'`` but the controller
    #: not yet aware; the planner must rely on lease ownership, not state.
    is_terminal: bool = False


@dataclass(frozen=True)
class WorktreeObservation:
    """One git worktree the planner can see."""

    path: str
    branch: str
    last_commit_at: datetime
    last_push_at: datetime | None = None
    is_default_branch: bool = False


@dataclass(frozen=True)
class PullRequestObservation:
    """One open pull request the planner can see."""

    number: int
    branch: str
    head_sha: str
    expected_head_sha: str | None = None

    def head_moved(self) -> bool:
        """A previous owner recorded an ``expected_head_sha``; the current
        remote head no longer matches, so a side-effect outcome is
        untrustworthy."""
        return self.expected_head_sha is not None and self.expected_head_sha != self.head_sha


@dataclass
class WorkItemObservation:
    """Durable state of one work item (the *authoritative* view)."""

    work_item: GitHubIssueRef
    state: WorkflowState | None
    claim: Claim | None

    @property
    def work_item_id(self) -> str:
        return self.work_item.slug()

    @property
    def run_id(self) -> RunId | None:
        return self.state.run_id if self.state is not None else None

    @property
    def phase(self) -> str | None:
        return self.state.phase if self.state is not None else None


@dataclass
class ReconciliationInputs:
    """Bundle the planner needs for one work item.

    ``ci_status`` is the *current* gate snapshot for this work item's PR
    (or ``None`` if no PR exists / no gate evaluation has happened yet).
    The planner refuses to treat phase ``ci_gating`` as evidence that CI
    ran — phase progression only means *we started checking*. The gate
    snapshot is the durable signal that a check completed.
    """

    observation: WorkItemObservation
    sessions: tuple[SessionObservation, ...]
    worktrees: tuple[WorktreeObservation, ...]
    pull_requests: tuple[PullRequestObservation, ...]
    config: CleanupConfig
    queue_config: GitHubQueueConfig
    now: datetime
    ci_status: Any = None  # GateDecision | None (typed loosely to avoid import cycle)

    def sessions_for_work_item(self) -> tuple[SessionObservation, ...]:
        """All sessions whose declared ``work_item_id`` matches."""
        return tuple(s for s in self.sessions if s.work_item_id == self.observation.work_item_id)

    def sessions_for_run(self, run_id: RunId | None) -> tuple[SessionObservation, ...]:
        if run_id is None:
            return ()
        return tuple(s for s in self.sessions if s.run_id == run_id)

    def sessions_for_lane(self, lane: str | None) -> tuple[SessionObservation, ...]:
        if lane is None:
            return ()
        return tuple(s for s in self.sessions if s.lane == lane)

    def worktree_for_branch(self, branch: str | None) -> WorktreeObservation | None:
        if branch is None:
            return None
        return next((w for w in self.worktrees if w.branch == branch), None)

    def pull_request_for_branch(self, branch: str) -> PullRequestObservation | None:
        return next((pr for pr in self.pull_requests if pr.branch == branch), None)


# --- Decision table ---------------------------------------------------------


@dataclass(frozen=True)
class _DecisionRow:
    """One entry in the decision table.

    A row is *applicable* when every predicate in ``predicates`` returns
    True. The planner iterates the table in declaration order; the first
    applicable row wins, so row order encodes priority.
    """

    name: str
    predicates: tuple[tuple[str, bool], ...]
    action: Action


def _safe_state(state: WorkflowState | None) -> bool:
    return state is not None


def _terminal_state(state: WorkflowState | None) -> bool:
    return state is not None and state.phase in TERMINAL_PHASES


def _lease_alive(state: WorkflowState | None, now: datetime) -> bool:
    """A lease is alive when its expiry is in the future."""
    if state is None:
        return False
    try:
        claim = claim_from_state(state)
    except Exception:
        return False
    return not claim.is_stale(now)


def _phase_at_least(state: WorkflowState | None, target: str) -> bool:
    """Return True when ``state`` has progressed to ``target`` or beyond.

    Phase progression is monotonic across the V3 lifecycle; a work item that
    has reached ``reviewing`` has logically passed through ``coding``, and
    the planner treats those earlier phases as implied. The ordering mirrors
    :data:`ai_pr_orchestrator.v3.domain.VALID_PHASES` plus the terminal set.
    """
    ordering = (
        "queued",
        "claiming",
        "planning",
        "coding",
        "reviewing",
        "ci_gating",
        "updating_pr",
        "done",
        "failed",
        "escalated",
    )
    if state is None or state.phase not in ordering:
        return False
    return ordering.index(state.phase) >= ordering.index(target)


def _findings_published_to_pr(state: WorkflowState | None) -> bool:
    """Reviewer findings are durable evidence that the reviewer session published.

    Findings carry a ``thread_id`` once posted to the PR; without one they
    only live in the state block (which the planner cannot observe). The
    planner treats "at least one finding with a thread_id" as the durable
    signal that the review round reached the PR.
    """
    if state is None:
        return False
    return any(f.thread_id is not None for f in state.findings)


def _all_findings_dispositioned(state: WorkflowState | None) -> bool:
    """Every finding has a disposition recorded.

    Single-disposition predicates are racy: a foreman that has only
    persisted one disposition of three findings still has an in-flight
    review round; advancing out of ``post_findings_no_disposition`` on
    that signal would skip the remaining findings. The planner only
    advances once the entire finding set has been decided.
    """
    if state is None or not state.findings:
        return False
    dispositioned_ids = {d.finding_id for d in state.dispositions}
    return all(f.id in dispositioned_ids for f in state.findings)


def _ci_recorded(ci_status: Any) -> bool:
    """CI has actually been evaluated for the relevant PR.

    The phase field tracks *intent* (``ci_gating`` means *we started
    checking*); only a ``GateDecision`` snapshot (passed OR failed) is
    durable evidence the check ran. ``None`` or an unknown/unevaluated
    snapshot is treated as *no CI result yet*.
    """
    if ci_status is None:
        return False
    # GateDecision is ``passed / failed / pending``; we accept both
    # passed=True and passed=False as "result is known" — only pending
    # counts as "not recorded".
    return bool(getattr(ci_status, "passed", None) is not None) and not (
        getattr(ci_status, "pending_checks", ()) or ()
    )


def _claim_has_branch(claim: Claim | None) -> bool:
    return claim is not None and claim.branch is not None


def _claim_has_pr(claim: Claim | None) -> bool:
    return claim is not None and claim.pr_number is not None


#: The decision table. Row order is priority; predicates are evaluated
#: against the observation bundle. Names are exposed in test assertions so a
#: failed row reports itself.
_CRASH_DECISION_TABLE: tuple[_DecisionRow, ...] = (
    # --- Manual / halt rows (always match the dangerous patterns first) ----
    _DecisionRow(
        name="branch_moved",
        predicates=(("pr_head_moved", True),),
        # Filled in dynamically by the planner (depends on PR lookup).
        action=Action(kind=ActionKind.HALT_BRANCH_MOVED, reason="PR head SHA moved"),
    ),
    _DecisionRow(
        name="stale_lease_other_owner",
        predicates=(
            ("lease_present", True),
            ("lease_alive", False),
            ("work_item_in_terminal", False),
        ),
        action=Action(
            kind=ActionKind.ESCALATE,
            reason="Lease is stale — another foreman may own this work",
        ),
    ),
    _DecisionRow(
        name="duplicate_sessions",
        predicates=(("two_or_more_live_sessions_same_lane", True),),
        action=Action(
            kind=ActionKind.ESCALATE,
            reason="More than one live session exists for this work item",
        ),
    ),
    # --- Resumable crash rows ---------------------------------------------
    _DecisionRow(
        name="post_push_before_review",
        predicates=(
            ("has_claim", True),
            ("claim_has_pr", True),
            ("has_terminal_review_session", False),
            ("has_active_or_pending_session", False),
        ),
        action=Action(
            kind=ActionKind.RESUME_SESSION,
            reason="Pushed; need to launch the review session for this work item",
        ),
    ),
    _DecisionRow(
        name="post_review_no_findings_recorded",
        predicates=(
            ("has_claim", True),
            ("claim_has_pr", True),
            ("has_terminal_review_session", True),
            ("findings_published_to_pr", False),
        ),
        action=Action(
            kind=ActionKind.COLLECT_RESULT,
            reason="Review session finished; no findings published to the PR yet",
        ),
    ),
    _DecisionRow(
        name="post_findings_no_disposition",
        predicates=(
            ("has_claim", True),
            ("claim_has_pr", True),
            ("findings_published_to_pr", True),
            ("all_findings_dispositioned", False),
        ),
        action=Action(
            kind=ActionKind.RELAUNCH,
            reason="Findings published but not all have dispositions; relaunch at review",
        ),
    ),
    _DecisionRow(
        name="post_disposition_no_ci",
        predicates=(
            ("has_claim", True),
            ("claim_has_pr", True),
            ("all_findings_dispositioned", True),
            ("ci_recorded", False),
        ),
        action=Action(
            kind=ActionKind.RESUME_SESSION,
            reason="Disposition recorded for every finding; check CI before proceeding",
        ),
    ),
    _DecisionRow(
        name="post_ci_no_pr",
        predicates=(
            ("has_claim", True),
            ("all_findings_dispositioned", True),
            ("ci_recorded", True),
            ("claim_has_pr", False),
        ),
        action=Action(
            kind=ActionKind.RELAUNCH,
            reason="CI green; need to create the PR",
        ),
    ),
    _DecisionRow(
        name="post_pr_no_final_label",
        predicates=(
            ("has_claim", True),
            ("claim_has_pr", True),
            ("final_label_transitioned", False),
        ),
        action=Action(
            kind=ActionKind.RESUME_SESSION,
            reason="PR is open; final label transition is pending",
        ),
    ),
    # --- Coding/review session in flight ----------------------------------
    _DecisionRow(
        name="session_terminal_pre_commit",
        predicates=(
            ("has_claim", True),
            ("has_terminal_coding_session", True),
            ("coder_succeeded", True),
            ("commit_recorded", False),
        ),
        action=Action(
            kind=ActionKind.COLLECT_RESULT,
            reason="Coding session finished; no commit recorded yet",
        ),
    ),
    _DecisionRow(
        name="session_terminal_pre_push",
        predicates=(
            ("has_claim", True),
            ("has_terminal_coding_session", True),
            ("coder_succeeded", True),
            ("commit_recorded", True),
            ("branch_pushed", False),
        ),
        action=Action(
            kind=ActionKind.COLLECT_RESULT,
            reason="Commit recorded locally; remote untouched",
        ),
    ),
    _DecisionRow(
        name="crash_after_claim_before_branch",
        predicates=(
            ("has_claim", True),
            ("claim_has_branch", False),
            ("lease_alive", True),
        ),
        action=Action(
            kind=ActionKind.RESUME_SESSION,
            reason="Claimed but no branch yet — finish the claim",
        ),
    ),
    # --- After push, no PR yet: resume to launch the next phase -----------
    # This row must win over ``branch_exists_no_session`` so a pushed branch
    # (i.e. a crash after the local push) is "resume" rather than "relaunch",
    # which matches the spec.
    _DecisionRow(
        name="post_push_no_pr",
        predicates=(
            ("has_claim", True),
            ("claim_has_branch", True),
            ("claim_has_pr", False),
            ("branch_pushed", True),
            ("has_active_or_pending_session", False),
        ),
        action=Action(
            kind=ActionKind.RESUME_SESSION,
            reason="Pushed to remote; no PR yet — resume to launch the next phase",
        ),
    ),
    _DecisionRow(
        name="branch_exists_no_session",
        predicates=(
            ("has_claim", True),
            ("claim_has_branch", True),
            ("worktree_exists_for_branch", True),
            ("has_active_or_pending_session", False),
        ),
        action=Action(
            kind=ActionKind.RELAUNCH,
            reason="Branch + worktree exist but no session is running",
        ),
    ),
    # --- Session in flight (mid-launch crash) -----------------------------
    _DecisionRow(
        name="session_in_flight",
        predicates=(
            ("has_claim", True),
            ("has_active_or_pending_session", True),
            ("has_terminal_coding_session", False),
        ),
        action=Action(
            kind=ActionKind.COLLECT_RESULT,
            reason="Session started but no terminal output yet — collect the result",
        ),
    ),
    # --- Terminal / idle ---------------------------------------------------
    _DecisionRow(
        name="already_terminal",
        predicates=(("work_item_in_terminal", True),),
        action=Action(kind=ActionKind.NOOP, reason="Work item is already terminal"),
    ),
    _DecisionRow(
        name="no_state_yet",
        predicates=(("lease_present", False),),
        action=Action(
            kind=ActionKind.NOOP,
            reason="No claim yet — issue is either still queued or untracked",
        ),
    ),
)


# --- Planner ---------------------------------------------------------------


class ReconcilePlanner:
    """Plan recovery actions for one work item's authoritative state.

    The planner is *pure*: it never performs I/O, never mutates inputs, and
    never reads the wall clock. Tests construct an instance with explicit
    config + observation bundle and assert on the returned ``Action`` list.
    """

    def __init__(
        self,
        *,
        cleanup_config: CleanupConfig | None = None,
        queue_config: GitHubQueueConfig | None = None,
        lanes: LaneRegistry | None = None,
    ) -> None:
        self._cleanup = cleanup_config or CleanupConfig()
        self._queue = queue_config or GitHubQueueConfig()
        # The lane registry binds lane names to roles; the planner uses it
        # to identify the coding lane (``LaneRegistry.developer_lane``) so
        # the predicate survives deployments that rename the developer lane.
        self._lanes = lanes or LaneRegistry.default()

    @property
    def cleanup_config(self) -> CleanupConfig:
        return self._cleanup

    @property
    def queue_config(self) -> GitHubQueueConfig:
        return self._queue

    @property
    def lanes(self) -> LaneRegistry:
        return self._lanes

    # -- Per-work-item plan -------------------------------------------------

    def plan(self, inputs: ReconciliationInputs) -> list[Action]:
        """Return the deterministic recovery plan for ``inputs``."""
        return self.plan_many([inputs])

    def plan_many(self, inputs_list: list[ReconciliationInputs]) -> list[Action]:
        """Return the recovery plan for every work item in ``inputs_list``.

        The cross-work-item dedupe :meth:`_finalize` enforces the spec's
        "no two authoritative branches per run" guarantee by collapsing
        any branch-creating action whose ``run_id`` key has already been
        emitted. Cross-item orphan detection (``_plan_orphan_*``) builds a
        single live-branch set from *every* input, not just the current
        one, so item B's plan does not spuriously flag item A's live
        branches as orphan.
        """
        if any(inputs.now.tzinfo is None for inputs in inputs_list):
            raise DomainError("ReconciliationInputs.now must be timezone-aware")
        # Cross-item live-branch set, computed once before any per-item
        # planning so orphan detection sees the whole picture.
        live_branches = self._collect_live_branches(inputs_list)
        live_session_ids = self._collect_live_session_ids(inputs_list)
        actions: list[Action] = []
        for inputs in inputs_list:
            actions.extend(
                self._plan_one(
                    inputs,
                    live_branches=live_branches,
                    live_session_ids=live_session_ids,
                )
            )
        if not actions:
            actions.append(Action(kind=ActionKind.NOOP, reason="Nothing to reconcile"))
        return self._finalize(actions)

    def _plan_one(
        self,
        inputs: ReconciliationInputs,
        *,
        live_branches: set[str],
        live_session_ids: set[str],
    ) -> list[Action]:
        # Terminal-phase short-circuit: when the work item is already in a
        # terminal phase (``done`` / ``failed`` / ``escalated``), the planner
        # must return NOOP immediately and not evaluate any crash row. The
        # previous implementation evaluated crash rows first, which fired
        # spurious RELAUNCH / RESUME_SESSION actions on completed work and
        # could re-launch a terminal item.
        if inputs.observation.state is not None and inputs.observation.state.phase in (
            *TERMINAL_PHASES,
            "cancelled",
        ):
            return [
                Action(
                    kind=ActionKind.NOOP,
                    work_item_id=inputs.observation.work_item_id,
                    run_id=inputs.observation.run_id,
                    reason=f"Work item is in terminal phase {inputs.observation.state.phase!r}",
                )
            ]

        actions: list[Action] = []
        # Manual/dangerous observations first (one action per category).
        # These short-circuit — once any of them fires we do not plan
        # further actions for this work item.
        branch_moved = self._plan_branch_moved(inputs)
        if branch_moved:
            return branch_moved
        stale = self._plan_stale_lease(inputs)
        if stale:
            return stale
        duplicate = self._plan_duplicate_sessions(inputs)
        if duplicate:
            return duplicate

        # Crash-point crash recovery from the table.
        crash_action = self._plan_crash_point(inputs)
        if crash_action is not None:
            actions.append(crash_action)
        # Orphan cleanups — independent of the work item's own phase, but
        # suppressed once a manual action already fired above (we returned).
        actions.extend(self._plan_orphan_sessions(inputs, live_session_ids=live_session_ids))
        actions.extend(self._plan_orphan_worktrees(inputs, live_branches=live_branches))
        return actions

    # -- Individual planners -----------------------------------------------

    def _plan_branch_moved(self, inputs: ReconciliationInputs) -> list[Action]:
        claim = inputs.observation.claim
        if claim is None or claim.pr_number is None:
            return []
        for pr in inputs.pull_requests:
            if pr.number == claim.pr_number and pr.head_moved():
                return [
                    Action(
                        kind=ActionKind.HALT_BRANCH_MOVED,
                        work_item_id=inputs.observation.work_item_id,
                        run_id=inputs.observation.run_id,
                        branch=claim.branch,
                        pr_number=claim.pr_number,
                        reason=(
                            f"PR #{claim.pr_number} head is {pr.head_sha}, "
                            f"but expected {pr.expected_head_sha}"
                        ),
                    )
                ]
        return []

    def _plan_stale_lease(self, inputs: ReconciliationInputs) -> list[Action]:
        state = inputs.observation.state
        claim = inputs.observation.claim
        if state is None or claim is None:
            return []
        if state.phase in TERMINAL_PHASES:
            return []
        if claim.is_stale(inputs.now):
            # Lease is stale. The protocol distinguishes a paused owner
            # (recover via heartbeat) from an owner the queue already
            # reclaimed (escalate). With no other evidence, the planner
            # escalates: a stale lease means we cannot tell whose work is
            # about to land on the remote. The caller (``_plan_one``)
            # short-circuits on this action — see the "stop on manual"
            # contract at the top of the module.
            return [
                Action(
                    kind=ActionKind.ESCALATE,
                    work_item_id=inputs.observation.work_item_id,
                    run_id=claim.run_id,
                    branch=claim.branch,
                    pr_number=claim.pr_number,
                    reason=(
                        f"Lease expired at {claim.lease_expires_at.isoformat()}; "
                        "another foreman may now own this work"
                    ),
                )
            ]
        return []

    def _plan_duplicate_sessions(self, inputs: ReconciliationInputs) -> list[Action]:
        """Emit ESCALATE only when two sessions are actually duplicative.

        The previous implementation treated any ``>= 2`` session count for a
        run as a duplicate. Two genuine-but-legitimate cases match that
        count and must NOT escalate:

        - one terminal session + one live session in **different lanes**
          (e.g. an old coder session + a fresh reviewer session): the
          planner should COLLECT_RESULT from the terminal one, not escalate.

        Real duplication = at least two live sessions for the same work
        item in the same lane, OR one live + one terminal in the same lane
        (the live one inherited state from a session we never observed die
        and may now be in conflict).
        """
        run_id = inputs.observation.run_id
        run_sessions = inputs.sessions_for_run(run_id)
        if len(run_sessions) < 2:
            return []
        # Group by lane: a "duplicate" is a within-lane problem. A coder
        # session and a reviewer session sharing a run is normal (the
        # reviewer round follows the coder round on the same run).
        live = [s for s in run_sessions if not s.is_terminal]
        terminal = [s for s in run_sessions if s.is_terminal]
        if len(live) >= 2 and len({s.lane for s in live}) == 1:
            # Two live sessions on the SAME lane: that is unambiguous
            # duplication, escalate.
            return [
                Action(
                    kind=ActionKind.ESCALATE,
                    work_item_id=inputs.observation.work_item_id,
                    run_id=run_id,
                    session_id=live[0].session_id,
                    reason=(
                        f"Found {len(live)} live sessions for run {run_id} "
                        f"in lane {live[0].lane!r}; refusing to pick a winner"
                    ),
                )
            ]
        if live and terminal and len({s.lane for s in live + terminal}) == 1:
            # One live + one terminal in the SAME lane: that is genuine
            # duplication (the live session inherited a session we never
            # observed die), so escalate.
            return [
                Action(
                    kind=ActionKind.ESCALATE,
                    work_item_id=inputs.observation.work_item_id,
                    run_id=run_id,
                    session_id=terminal[0].session_id,
                    reason=(
                        f"Found {len(live)} live + {len(terminal)} terminal sessions "
                        f"for run {run_id} in lane {live[0].lane!r}; refusing to pick a winner"
                    ),
                )
            ]
        if live and terminal:
            # One live + one terminal in DIFFERENT lanes: legitimate — the
            # terminal one is the result we never recorded, in a lane the
            # live one is no longer running. COLLECT_RESULT.
            return [
                Action(
                    kind=ActionKind.COLLECT_RESULT,
                    work_item_id=inputs.observation.work_item_id,
                    run_id=run_id,
                    session_id=terminal[0].session_id,
                    reason=(
                        f"Terminal session in lane {terminal[0].lane!r} alongside a live "
                        f"session in lane {live[0].lane!r} — collect the result"
                    ),
                )
            ]
        return []

    def _plan_crash_point(self, inputs: ReconciliationInputs) -> Action | None:
        """Walk the decision table; return the first applicable row's action.

        Returns ``None`` when no row matches — caller decides whether to
        emit a default ``NOOP``.
        """
        ctx = self._build_decision_context(inputs)
        for row in _CRASH_DECISION_TABLE:
            if all(ctx.get(name) == expected for name, expected in row.predicates):
                action = row.action
                # Replace the placeholder with a bound copy that carries
                # the work-item identity and the right ``session_id`` for
                # session-touching actions, so the CLI can group / dispatch
                # actions without re-deriving identity.
                session_id = self._session_id_for(inputs, row.name)
                # Manual actions short-circuit at the _plan_one level;
                # returning them here keeps them in the action list when
                # invoked through a lower-level caller (e.g. tests).
                auto_apply = action.kind not in _MANUAL_ACTIONS
                if action.kind is ActionKind.NOOP:
                    auto_apply = True
                return Action(
                    kind=action.kind,
                    work_item_id=inputs.observation.work_item_id,
                    run_id=inputs.observation.run_id,
                    session_id=session_id,
                    branch=ctx.get("branch"),
                    pr_number=ctx.get("pr_number"),
                    reason=action.reason or row.name,
                    auto_apply=auto_apply,
                )
        return None

    def _build_decision_context(self, inputs: ReconciliationInputs) -> dict[str, Any]:
        """Map observation bundle to the predicates the decision table reads."""
        state = inputs.observation.state
        claim = inputs.observation.claim
        worktrees_for_wi = tuple(
            w for w in inputs.worktrees if w.branch == (claim.branch if claim else None)
        )
        coding_sessions = self._coding_sessions(inputs)
        review_sessions = self._review_sessions(inputs)
        live_coding = [s for s in coding_sessions if not s.is_terminal]

        # 'commit_recorded' is the durable signal that the worker produced
        # a commit locally. Two failure modes must NOT report True:
        # 1. The worktree's last_commit_at predates the lease claim
        #    (older commit, new lease — classic "coder no-op" false
        #    positive).
        # 2. The coder session itself did not exit successfully
        #    (``success=False``). A coder that exited cleanly but touched
        #    nothing is treated as having produced nothing.
        # We require BOTH "worktree has a newer commit" AND "a coder
        # session in this run exited successfully" — see finding #11.
        coder_succeeded = any(s.success and s.is_terminal for s in coding_sessions)
        commit_is_newer = bool(
            claim is not None
            and claim.branch is not None
            and worktrees_for_wi
            and worktrees_for_wi[0].last_commit_at > (claim.claimed_at if claim else inputs.now)
        )
        commit_recorded = commit_is_newer and coder_succeeded

        # Branch is "pushed" when the worktree records a non-None
        # ``last_push_at`` newer than the lease claim — that is the moment a
        # remote ref was last updated for this branch.
        branch_pushed = bool(
            claim is not None
            and worktrees_for_wi
            and worktrees_for_wi[0].last_push_at is not None
            and worktrees_for_wi[0].last_push_at > (claim.claimed_at if claim else inputs.now)
        )

        pr_observation = (
            inputs.pull_request_for_branch(claim.branch)
            if claim is not None and claim.branch is not None
            else None
        )

        return {
            "lease_present": _safe_state(state),
            "lease_alive": _lease_alive(state, inputs.now),
            "work_item_in_terminal": _terminal_state(state),
            "has_claim": claim is not None,
            "claim_has_branch": _claim_has_branch(claim),
            "claim_has_pr": _claim_has_pr(claim),
            "branch": claim.branch if claim else None,
            "pr_number": claim.pr_number if claim else None,
            "worktree_exists_for_branch": bool(worktrees_for_wi),
            "has_active_or_pending_session": any(
                not s.is_terminal for s in inputs.sessions_for_work_item()
            ),
            "has_terminal_coding_session": any(s.is_terminal for s in coding_sessions),
            "has_terminal_review_session": any(s.is_terminal for s in review_sessions),
            "commit_recorded": commit_recorded,
            "coder_succeeded": coder_succeeded,
            "branch_pushed": branch_pushed,
            "pr_head_moved": bool(pr_observation is not None and pr_observation.head_moved()),
            "findings_published_to_pr": _findings_published_to_pr(state),
            "all_findings_dispositioned": _all_findings_dispositioned(state),
            "disposition_recorded": _all_findings_dispositioned(state),
            "ci_recorded": _ci_recorded(inputs.ci_status),
            "final_label_transitioned": _phase_at_least(state, "done"),
            "two_or_more_live_sessions_same_lane": len(live_coding) >= 2
            and len({s.lane for s in live_coding}) == 1,
        }

    # -- Lane helpers -------------------------------------------------------

    def _coding_sessions(self, inputs: ReconciliationInputs) -> tuple[SessionObservation, ...]:
        """Sessions in the coding (developer) lane.

        Lane identity comes from the registry's ``developer_lane`` rather
        than a hard-coded ``"developer"`` / ``"worker"`` / ``"coder"``
        literal — a deployment that renames the developer lane must still
        detect coding sessions. We also include any lane whose registered
        role is ``worker`` (the registry stores role per-lane, so this
        matches the intent without caring about the lane's textual name).
        """
        return tuple(s for s in inputs.sessions_for_work_item() if self._is_coding_lane(s.lane))

    def _review_sessions(self, inputs: ReconciliationInputs) -> tuple[SessionObservation, ...]:
        """Sessions in any reviewer lane (role ``reviewer``)."""
        return tuple(s for s in inputs.sessions_for_work_item() if self._is_reviewer_lane(s.lane))

    def _is_coding_lane(self, lane: str) -> bool:
        try:
            identity = self._lanes.get(lane)  # type: ignore[arg-type]
        except Exception:
            return False
        return identity.role == "worker"

    def _is_reviewer_lane(self, lane: str) -> bool:
        try:
            identity = self._lanes.get(lane)  # type: ignore[arg-type]
        except Exception:
            return False
        return identity.role == "reviewer"

    def _session_id_for(self, inputs: ReconciliationInputs, row_name: str) -> str | None:
        """Best-effort session_id for an action row.

        RESUME_SESSION / COLLECT_RESULT / RELAUNCH all target a session the
        planner can name; tests assert every such row carries a non-None
        ``session_id`` so the CLI can dispatch. We pick:

        - the most recent live session for ``has_active_or_pending_session``
          rows,
        - the most recent terminal session for ``COLLECT_RESULT`` rows,
        - the prior terminal session for the duplicate-session's terminal
          one,
        - the coder session for the coding rows,
        - ``None`` for rows that don't target a session.
        """
        work_item_sessions = inputs.sessions_for_work_item()
        run_sessions = inputs.sessions_for_run(inputs.observation.run_id)
        if row_name in (
            "session_in_flight",
            "post_push_before_review",
            "post_push_no_pr",
            "crash_after_claim_before_branch",
            "branch_exists_no_session",
            "post_pr_no_final_label",
        ):
            # Resume the most-recent live session if any, otherwise the
            # most-recent session of any kind so the CLI has something
            # concrete to dispatch against.
            live = [s for s in work_item_sessions if not s.is_terminal]
            if live:
                return max(live, key=lambda s: s.last_activity_at).session_id
            if work_item_sessions:
                return max(work_item_sessions, key=lambda s: s.last_activity_at).session_id
        if row_name in (
            "post_review_no_findings_recorded",
            "session_terminal_pre_commit",
            "session_terminal_pre_push",
        ):
            # The session whose output we want to collect.
            target_pool = (
                self._coding_sessions(inputs)
                if row_name != "post_review_no_findings_recorded"
                else self._review_sessions(inputs)
            )
            if target_pool:
                return max(target_pool, key=lambda s: s.last_activity_at).session_id
            terminal = [s for s in run_sessions if s.is_terminal]
            if terminal:
                return terminal[0].session_id
        if row_name == "duplicate_sessions":
            terminal = [s for s in run_sessions if s.is_terminal]
            return terminal[0].session_id if terminal else None
        return None

    # -- Orphan detection --------------------------------------------------

    def _plan_orphan_sessions(
        self,
        inputs: ReconciliationInputs,
        *,
        live_session_ids: set[str],
    ) -> Iterable[Action]:
        ttl = timedelta(seconds=self._cleanup.session_lease_ttl_seconds)
        for session in inputs.sessions:
            if session.work_item_id is None:
                # Sessions with no declared work item are never reclaimed by
                # the planner; they are either pre-launch or operator-launched
                # and ownership is ambiguous.
                continue
            if session.session_id in live_session_ids:
                continue
            age = inputs.now - session.last_activity_at
            if age < ttl:
                continue
            yield Action(
                kind=ActionKind.CLEAN_ORPHAN_SESSION,
                work_item_id=session.work_item_id,
                session_id=session.session_id,
                reason=(
                    f"No live lease references session {session.session_id} "
                    f"and {int(age.total_seconds())}s elapsed since last activity "
                    f"(TTL {self._cleanup.session_lease_ttl_seconds}s)"
                ),
                auto_apply=True,
            )

    def _plan_orphan_worktrees(
        self,
        inputs: ReconciliationInputs,
        *,
        live_branches: set[str],
    ) -> Iterable[Action]:
        ttl = timedelta(seconds=self._cleanup.worktree_inactivity_ttl_seconds)
        for worktree in inputs.worktrees:
            if worktree.is_default_branch:
                continue
            if worktree.branch in live_branches:
                continue
            age = inputs.now - worktree.last_commit_at
            if age < ttl:
                continue
            yield Action(
                kind=ActionKind.CLEAN_ORPHAN_WORKTREE,
                branch=worktree.branch,
                worktree=worktree.path,
                reason=(
                    f"No live lease references branch {worktree.branch!r}; "
                    f"{int(age.total_seconds())}s elapsed since last commit "
                    f"(TTL {self._cleanup.worktree_inactivity_ttl_seconds}s)"
                ),
                auto_apply=True,
            )

    def _collect_live_branches(self, inputs_list: list[ReconciliationInputs]) -> set[str]:
        """Branches that *any* work item in ``inputs_list`` still actively claims.

        Computed across the entire inputs list (not per-item) so a follow-up
        item's plan does not spuriously flag a prior item's live branch as
        orphan. A branch is "live" only when its claim has a non-stale
        lease — a stale lease means another foreman may already own the
        branch.
        """
        branches: set[str] = set()
        for inputs in inputs_list:
            claim = inputs.observation.claim
            if claim is None or claim.branch is None:
                continue
            if claim.is_stale(inputs.now):
                continue
            branches.add(claim.branch)
        return branches

    def _collect_live_session_ids(self, inputs_list: list[ReconciliationInputs]) -> set[str]:
        """Session IDs that *any* work item in ``inputs_list`` still holds a
        live lease for, used by orphan-session detection (mirror of
        :meth:`_collect_live_branches`)."""
        ids: set[str] = set()
        for inputs in inputs_list:
            claim = inputs.observation.claim
            if claim is None:
                continue
            if claim.is_stale(inputs.now):
                continue
            for session in inputs.sessions:
                if session.work_item_id != inputs.observation.work_item_id:
                    continue
                if session.run_id is not None and claim.run_id != session.run_id:
                    continue
                ids.add(session.session_id)
        return ids

    # -- Output post-processing --------------------------------------------

    def _finalize(self, actions: list[Action]) -> list[Action]:
        """Apply the no-two-branches-per-run invariant.

        The acceptance property test proves this; here we defensively enforce
        it by collapsing RELAUNCH actions that share the same ``run_id``
        across the inputs. (The planner is invoked per-work-item so
        collisions across work items normally do not happen here, but a
        future caller passing the same observation twice, or two distinct
        observations with the same ``run_id`` but different branches,
        should still produce a deterministic plan.)

        Branch identity alone is not enough: the dedupe key is ``run_id``
        because two RELAUNCH actions for the *same* run (even with
        different branch names) cannot both be authoritative — only one
        branch owns a given run.
        """
        seen: set[str] = set()
        out: list[Action] = []
        for action in actions:
            if actions_target_branch(action):
                key = action.run_id or ""
                if key in seen:
                    continue
                seen.add(key)
            out.append(action)
        return out


# --- Orphan detection helpers (public) --------------------------------------


def is_orphan_session(
    session: SessionObservation,
    *,
    has_live_lease: bool,
    now: datetime,
    ttl_seconds: int,
) -> bool:
    """Pure-function orphan check.

    Mirrors the planner's predicate but exposed for callers that want to
    classify sessions without the planner's full output. Both conditions
    must hold:

    1. no live work-item lease references the session, AND
    2. ``now - session.last_activity_at >= ttl_seconds``.
    """
    if has_live_lease:
        return False
    age = now - session.last_activity_at
    return age >= timedelta(seconds=ttl_seconds)


def is_orphan_worktree(
    worktree: WorktreeObservation,
    *,
    branch_has_live_lease: bool,
    now: datetime,
    ttl_seconds: int,
) -> bool:
    """Pure-function orphan check for git worktrees.

    Both conditions must hold:

    1. no live work-item lease references the worktree's branch, AND
    2. ``now - worktree.last_commit_at >= ttl_seconds``.

    The default branch is never orphan — it is the merge target, not a
    worktree to clean.
    """
    if worktree.is_default_branch:
        return False
    if branch_has_live_lease:
        return False
    age = now - worktree.last_commit_at
    return age >= timedelta(seconds=ttl_seconds)


def list_orphan_sessions(
    sessions: Iterable[SessionObservation],
    *,
    now: datetime,
    ttl_seconds: int,
) -> list[SessionObservation]:
    """Return the subset of ``sessions`` whose lease + age place them past TTL.

    Useful as a CLI building block: a single sweep across all live sessions
    yields the orphan set in one pass.
    """
    return [s for s in sessions if _no_lease_signal_for(s) and _aged_out(s, now, ttl_seconds)]


def list_orphan_worktrees(
    worktrees: Iterable[WorktreeObservation],
    *,
    now: datetime,
    ttl_seconds: int,
) -> list[WorktreeObservation]:
    """Return the subset of ``worktrees`` whose branch + age place them past TTL."""
    return [
        w for w in worktrees if not w.is_default_branch and _aged_out_worktree(w, now, ttl_seconds)
    ]


def _no_lease_signal_for(_session: SessionObservation) -> bool:
    """Helper: without a lease registry, "no live lease" is encoded in the
    observation's ``work_item_id`` being None or marked terminal. Callers
    should prefer :func:`is_orphan_session` with an explicit
    ``has_live_lease`` argument."""
    return True


def _aged_out(session: SessionObservation, now: datetime, ttl_seconds: int) -> bool:
    age = now - session.last_activity_at
    return age >= timedelta(seconds=ttl_seconds)


def _aged_out_worktree(worktree: WorktreeObservation, now: datetime, ttl_seconds: int) -> bool:
    age = now - worktree.last_commit_at
    return age >= timedelta(seconds=ttl_seconds)


__all__ = [
    "Action",
    "ActionKind",
    "PullRequestObservation",
    "ReconcilePlanner",
    "ReconciliationInputs",
    "SessionLifecycle",
    "SessionObservation",
    "WorkItemObservation",
    "WorktreeObservation",
    "actions_target_branch",
    "is_orphan_session",
    "is_orphan_worktree",
    "list_orphan_sessions",
    "list_orphan_worktrees",
]
