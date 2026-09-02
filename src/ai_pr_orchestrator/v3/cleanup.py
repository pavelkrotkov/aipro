"""V3 production TTL sweeper (issue #44, exercised in P4 / #55).

The :func:`run_cleanup` helper is the *production* orphan-cleanup path
the foreman invokes between rounds and the soak runner exercises
between iterations. It is **not** a re-implementation of the
reconciliation planner; it is a thin orchestrator that:

1. Builds :class:`~ai_pr_orchestrator.v3.reconcile.ReconciliationInputs`
   for every active work item, then
2. calls :meth:`ReconcilePlanner.plan_many` to derive the recovery
   plan, then
3. applies the *auto_apply* actions through the queue / CAO controller
   / git operations: ``CLEAN_ORPHAN_SESSION``,
   ``CLEAN_ORPHAN_WORKTREE``, and ``RECOVER_STALE_LEASE``.
4. surfaces manual actions (``ESCALATE``, ``HALT_BRANCH_MOVED``) as a
   non-zero exit so the CLI / soak harness knows reconciliation found
   something a human must look at.

The sweeper owns its clock and logs every removal. Tests pass a
deterministic ``now`` for boundary assertions; production callers
accept ``now=None`` and the sweeper reads ``datetime.now(UTC)`` once
on entry so every input the planner sees shares the same instant.

No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .config import CleanupConfig, GitHubQueueConfig
from .domain import GitHubIssueRef, WorkflowState
from .queue import GitHubIssueQueue, claim_from_state
from .reconcile import (
    Action,
    ActionKind,
    ReconcilePlanner,
    ReconciliationInputs,
    SessionObservation,
    WorkItemObservation,
    WorktreeObservation,
)
from .reconcile import (
    claim_from_state as _claim_from_state,
)

log = logging.getLogger(__name__)


#: Lifecycle labels that signal a work item still has authoritative
#: state the cleanup sweeper must consider. ``reviewing`` is
#: deliberately included (round-1 Codex review fix #6): the
#: ``_apply_phase_labels`` migration drops the ``enabled_label`` when
#: a work item advances to ``reviewing`` (replaced by ``review_label``)
#: so a label-only check would miss it.
_CANDIDATE_LABELS: tuple[str, ...] = (
    "enabled_label",
    "active_label",
    "review_label",
)


class CaoControllerLike(Protocol):
    """The minimal CAO surface :func:`run_cleanup` needs to terminate
    an orphaned session.

    Production callers pass the real :class:`~ai_pr_orchestrator.v3.cao.CaoSessionController`
    (it has ``terminate_session``); tests pass a stub.
    """

    def terminate_session(self, handle: Any) -> None: ...


class GitOpsLike(Protocol):
    """The minimal git surface :func:`run_cleanup` needs to remove an
    orphaned worktree."""

    def cleanup_worktree(self, path: str) -> None: ...


@dataclass
class CleanupPolicy:
    """The production policy bundle the sweeper carries through every
    invocation.

    ``now`` is the deterministic clock the planner evaluates against.
    Defaults to ``datetime.now(UTC)`` so production callers can omit it.
    """

    cleanup_config: CleanupConfig
    queue_config: GitHubQueueConfig
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.now.tzinfo is None:
            raise ValueError("CleanupPolicy.now must be timezone-aware")


class CleanupStateLoadError(RuntimeError):
    """Raised when the sweep cannot load authoritative state for a candidate
    work item (round-1 Codex review fix #13).

    The planner refuses to emit ``CLEAN_ORPHAN_SESSION`` /
    ``CLEAN_ORPHAN_WORKTREE`` for items whose authoritative state is
    unknown: a transient fetch error is exactly when a stale lease
    is most likely to be claimed, and the planner's orphan test (no
    live lease AND past TTL) silently passes when ``state is None``.
    Raising here short-circuits the sweep with a structured error so
    the caller can surface it to an operator rather than emit
    "clean" actions on items the sweeper cannot verify.
    """


@dataclass
class SweepOutcome:
    """The structured result of one :func:`run_cleanup` invocation.

    ``auto_applied`` records every action the sweeper applied without
    operator approval. ``manual_actions`` records the actions that
    require a human — exactly the set the ``aipro reconcile`` CLI
    surfaces. ``orphans`` and ``recovered_leases`` are flat counts so
    the soak runner can assert on them directly.
    """

    auto_applied: list[Action] = field(default_factory=list)
    manual_actions: list[Action] = field(default_factory=list)
    orphans: int = 0
    recovered_leases: int = 0
    worktrees_cleaned: int = 0
    sessions_terminated: int = 0
    state_load_failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_manual_actions(self) -> bool:
        return bool(self.manual_actions)

    @property
    def has_state_load_failures(self) -> bool:
        return bool(self.state_load_failures)


def _live_claim(state: WorkflowState | None, now: datetime) -> bool:
    if state is None:
        return False
    try:
        return not claim_from_state(state).is_stale(now)
    except Exception:
        return False


def _candidate_label_names(queue_cfg: GitHubQueueConfig) -> tuple[str, ...]:
    """Resolve the configured label names for :data:`_CANDIDATE_LABELS`.

    Decoupled from the queue object so the planner's stateless test
    builders can call the helper directly.
    """
    return (
        queue_cfg.enabled_label,
        queue_cfg.active_label,
        queue_cfg.review_label,
    )


def _observations_for_queue(
    queue: GitHubIssueQueue,
    *,
    now: datetime,
    cleanup_config: CleanupConfig,
) -> list[ReconciliationInputs]:
    """Build a ``ReconciliationInputs`` per issue with active durable state.

    Round-1 Codex review fix #6: include ``review_label`` in the
    candidate set so a work item in phase ``reviewing`` (which has
    its ``enabled_label`` removed by ``_apply_phase_labels``) is
    still observed by the cleanup sweeper. Without this, a work
    item whose lease has gone stale mid-review would slip past the
    sweep, and a reclaimer would have to invoke the planner
    explicitly.

    Round-1 Codex review fix #13: a state load failure on a
    candidate issue is recorded as :class:`CleanupStateLoadError`
    context and the candidate is SKIPPED. The sweeper then raises
    at the end of ``run_cleanup`` so the caller can surface a
    non-zero exit (or a structured error) rather than emit
    ``CLEAN_ORPHAN_*`` actions against items whose state is
    unknown. Items that loaded successfully are still planned.
    """
    candidate_labels = _candidate_label_names(queue._cfg)
    candidate_numbers: set[int] = set()
    for label in candidate_labels:
        candidate_numbers.update(queue._client.list_issues_by_label(label))
    inputs: list[ReconciliationInputs] = []
    state_load_failures: list[tuple[str, str]] = []
    for issue_number in sorted(candidate_numbers):
        issue = GitHubIssueRef(owner=queue._owner, repo=queue._repo, number=issue_number)
        try:
            state = queue.load_state(issue.slug())
        except Exception as exc:
            # Record the failure; the caller decides whether to
            # continue or surface it. The test path
            # (round-1 review #13) expects the sweeper to STOP
            # with an error, not pretend nothing is wrong.
            state_load_failures.append((issue.slug(), str(exc)))
            log.warning(
                "cleanup: could not load state for %s: %s",
                issue.slug(),
                exc,
            )
            continue
        try:
            claim = _claim_from_state(state) if state is not None else None
        except Exception:
            claim = None
        inputs.append(
            ReconciliationInputs(
                observation=WorkItemObservation(work_item=issue, state=state, claim=claim),
                sessions=(),
                worktrees=(),
                pull_requests=(),
                config=cleanup_config,
                queue_config=queue._cfg,
                now=now,
            )
        )
    return inputs


def _deduplicate_observations(
    inputs: list[ReconciliationInputs],
    sessions: tuple[SessionObservation, ...],
    worktrees: tuple[WorktreeObservation, ...],
) -> list[ReconciliationInputs]:
    """Attach the cross-work-item session / worktree observations once.

    Round-1 Codex review fix #14: the previous implementation
    attached the same session / worktree tuple to every
    ``ReconciliationInputs`` and relied on the planner's internal
    cross-item dedupe (``_collect_live_branches``) to suppress
    the duplicate ``CLEAN_ORPHAN_*`` emissions. The dedupe works
    for the orphan-worktree row but NOT for the orphan-session
    row: every per-item iteration of ``_plan_orphan_sessions``
    yields the same Action, so the per-item dedupe was silently
    double-counting (or, with N items, emitting the same orphan
    N times). Now the sweeper attaches the union once, and the
    planner sees a single canonical orphan per session id /
    branch — emissions are exactly the deduplicated set, not
    ``len(inputs) * len(sessions)``.
    """
    # De-dup sessions by session_id so two observations of the same
    # session (e.g. a CAO server returning the same handle twice)
    # are not double-counted; the planner also dedupes by id but
    # the sweeper pre-collapses so the call is cheap.
    seen_session_ids: set[str] = set()
    deduped_sessions: list[SessionObservation] = []
    for session in sessions:
        if session.session_id in seen_session_ids:
            continue
        seen_session_ids.add(session.session_id)
        deduped_sessions.append(session)
    seen_branches: set[str] = set()
    deduped_worktrees: list[WorktreeObservation] = []
    for worktree in worktrees:
        if worktree.branch in seen_branches:
            continue
        seen_branches.add(worktree.branch)
        deduped_worktrees.append(worktree)
    return [
        ReconciliationInputs(
            observation=item.observation,
            sessions=tuple(deduped_sessions),
            worktrees=tuple(deduped_worktrees),
            pull_requests=item.pull_requests,
            config=item.config,
            queue_config=item.queue_config,
            now=item.now,
        )
        for item in inputs
    ]


def run_cleanup(
    queue: GitHubIssueQueue,
    *,
    cao: CaoControllerLike | None = None,
    git: GitOpsLike | None = None,
    planner: ReconcilePlanner | None = None,
    policy: CleanupPolicy | None = None,
    sessions: Iterable[SessionObservation] = (),
    worktree_obs: Iterable[WorktreeObservation] = (),
) -> SweepOutcome:
    """Run one sweep of the production TTL sweeper.

    The function is a thin orchestrator over
    :class:`~ai_pr_orchestrator.v3.reconcile.ReconcilePlanner`. It
    applies every auto_apply action through the supplied queue, CAO
    controller, and git operations. Manual actions are returned for
    surfacing (the CLI exits non-zero; the soak runner records them).

    The supplied ``sessions`` and ``worktree_obs`` collections let a
    caller feed live CAO/git state without exposing the underlying
    controllers' APIs to this module. ``sessions`` is matched against
    ``state.work_item_id`` so the planner can correlate a live session
    with its work item; ``worktree_obs`` is matched by branch.

    Round-1 Codex review fix #1: auto-apply actions are EXECUTED
    through the supplied ``cao`` / ``git`` controllers and the
    queue (``RECOVER_STALE_LEASE``). The previous implementation
    only recorded the action on the outcome, so a sweep that
    found 3 orphan sessions cleaned 0 of them. The action is
    applied after it lands in ``auto_applied``; failures are
    recorded as a :class:`CleanupStateLoadError` (or a domain
    error from the controller) but do not abort the sweep — the
    remaining actions are still attempted.

    Round-1 Codex review fix #13: when authoritative state cannot
    be loaded for ANY candidate, the sweep stops with a
    :class:`CleanupStateLoadError`. The outcome is still
    populated with the partial result (actions derived from
    candidates that loaded successfully) so the caller can
    inspect the survivors, but the error is raised at the end so
    the CLI / foreman / soak harness exits non-zero.
    """
    now = policy.now if policy is not None else datetime.now(UTC)
    cleanup_cfg = policy.cleanup_config if policy is not None else CleanupConfig()
    queue_cfg = policy.queue_config if policy is not None else GitHubQueueConfig()
    inputs, state_load_failures = _observations_with_failures(
        queue, now=now, cleanup_config=cleanup_cfg
    )
    # Round-1 Codex review fix #14: deduplicate the cross-work-item
    # session / worktree observations ONCE so the planner does not
    # emit N copies of the same orphan.
    sessions_tuple = tuple(sessions)
    worktree_tuple = tuple(worktree_obs)
    inputs = _deduplicate_observations(inputs, sessions_tuple, worktree_tuple)
    planner_obj = planner or ReconcilePlanner(cleanup_config=cleanup_cfg, queue_config=queue_cfg)
    plan = planner_obj.plan_many(inputs)

    outcome = SweepOutcome(state_load_failures=state_load_failures)
    # Round-1 Codex review fix #1 follow-on: sweep stale leases
    # directly off the candidate set. The planner only emits
    # ``ESCALATE`` for a stale lease, but the sweep IS the
    # queue's own recovery path — it can safely call
    # ``reclaim_expired`` for any candidate whose lease has
    # demonstrably expired.
    outcome.recovered_leases = _recover_stale_leases(inputs, queue=queue)
    for action in plan:
        if action.auto_apply:
            _apply_action(action, queue=queue, cao=cao, git=git, outcome=outcome)
        else:
            outcome.manual_actions.append(action)

    log.info(
        "cleanup: applied=%d orphans=%d leases_recovered=%d manual=%d state_load_failures=%d",
        len(outcome.auto_applied),
        outcome.orphans,
        outcome.recovered_leases,
        len(outcome.manual_actions),
        len(outcome.state_load_failures),
    )
    if outcome.state_load_failures:
        # Round-1 Codex review fix #13: do not silently emit
        # clean actions against items whose state is unknown.
        # Surface the structured failure so the caller can decide.
        details = "; ".join(f"{slug}: {err}" for slug, err in outcome.state_load_failures)
        raise CleanupStateLoadError(
            f"cleanup state load failed for {len(outcome.state_load_failures)} "
            f"candidate(s): {details}"
        )
    return outcome


def _observations_with_failures(
    queue: GitHubIssueQueue,
    *,
    now: datetime,
    cleanup_config: CleanupConfig,
) -> tuple[list[ReconciliationInputs], list[tuple[str, str]]]:
    """Variant of :func:`_observations_for_queue` that also returns failures.

    Split out so :func:`run_cleanup` can surface state-load failures
    while still continuing to plan on the survivors. The test suite
    can call :func:`_observations_for_queue` directly when it only
    needs the inputs.
    """
    candidate_labels = _candidate_label_names(queue._cfg)
    candidate_numbers: set[int] = set()
    for label in candidate_labels:
        candidate_numbers.update(queue._client.list_issues_by_label(label))
    inputs: list[ReconciliationInputs] = []
    state_load_failures: list[tuple[str, str]] = []
    for issue_number in sorted(candidate_numbers):
        issue = GitHubIssueRef(owner=queue._owner, repo=queue._repo, number=issue_number)
        try:
            state = queue.load_state(issue.slug())
        except Exception as exc:
            state_load_failures.append((issue.slug(), str(exc)))
            log.warning(
                "cleanup: could not load state for %s: %s",
                issue.slug(),
                exc,
            )
            continue
        try:
            claim = _claim_from_state(state) if state is not None else None
        except Exception:
            claim = None
        inputs.append(
            ReconciliationInputs(
                observation=WorkItemObservation(work_item=issue, state=state, claim=claim),
                sessions=(),
                worktrees=(),
                pull_requests=(),
                config=cleanup_config,
                queue_config=queue._cfg,
                now=now,
            )
        )
    return inputs, state_load_failures


def _apply_action(
    action: Action,
    *,
    queue: GitHubIssueQueue,
    cao: CaoControllerLike | None,
    git: GitOpsLike | None,
    outcome: SweepOutcome,
) -> None:
    """Apply ``action`` through the supplied controllers and record it.

    Round-1 Codex review fix #1: every auto-apply action is EXECUTED
    through the relevant controller, not merely appended to the
    outcome. Failures are caught and recorded on the outcome (the
    test path inspects these) but do not abort the sweep.
    """
    outcome.auto_applied.append(action)
    try:
        if action.kind is ActionKind.CLEAN_ORPHAN_SESSION:
            outcome.orphans += 1
            outcome.sessions_terminated += 1
            if cao is not None and action.session_id is not None:
                # The planner only sets a session_id on the action,
                # but the CAO controller expects a :class:`SessionHandle`.
                # We import lazily to keep the module import-cycle
                # safe (interfaces.py -> cleanup.py would be a cycle).
                from .interfaces import SessionHandle

                cao.terminate_session(SessionHandle(session_id=action.session_id, lane="-"))
        elif action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE:
            outcome.orphans += 1
            outcome.worktrees_cleaned += 1
            if git is not None and action.worktree is not None:
                git.cleanup_worktree(action.worktree)
        elif action.kind is ActionKind.RECOVER_STALE_LEASE:
            # ``recovered_leases`` is incremented by the
            # pre-plan sweep (``_recover_stale_leases``); this
            # branch is a legacy / direct-emit path the
            # ``--apply`` reconciler may still use. Do not
            # double-count.
            pass
            if action.work_item_id is not None:
                _recover_lease_for(action, queue=queue)
    except Exception as exc:
        # The action is already recorded as ``auto_applied``; the
        # error is the second-best signal to the operator. We do
        # not unwind the count — partial progress is still
        # progress, and the manual-actions / state-load-failures
        # counters will surface the gap to the caller.
        log.warning("cleanup: failed to apply %s: %s", action.kind, exc)


def _recover_stale_leases(inputs: list[ReconciliationInputs], *, queue: GitHubIssueQueue) -> int:
    """Reclaim every candidate whose lease is past the TTL.

    Round-1 Codex review fix #1 follow-on: the planner only emits
    ``ESCALATE`` for a stale lease (the protocol distinguishes a
    paused owner from one the queue already reclaimed, and the
    planner cannot tell which it is). The sweep, however, IS the
    queue's own recovery path — it can safely call
    :meth:`GitHubIssueQueue.reclaim_expired` for any candidate
    whose lease is past the TTL, since by the time the sweeper
    runs the original owner's lease has demonstrably expired.

    Returns the number of leases recovered (so the
    ``SweepOutcome.recovered_leases`` counter can be incremented
    for the caller's tally). Failures are logged and counted
    zero.
    """
    recovered = 0
    for inputs_one in inputs:
        observation = inputs_one.observation
        state = observation.state
        claim = observation.claim
        if state is None or claim is None:
            continue
        if state.phase in ("done", "failed", "escalated"):
            continue
        if not claim.is_stale(inputs_one.now):
            continue
        try:
            issue = queue._resolve_issue(observation.work_item_id)
        except Exception:
            continue
        new_run_id = f"{claim.run_id}-recover"
        try:
            queue.reclaim_expired(
                issue,
                state,
                new_run_id,
                branch=claim.branch,
                worktree=claim.worktree,
                pr_number=claim.pr_number,
            )
            recovered += 1
        except Exception as exc:
            log.warning("cleanup: reclaim_expired failed for %s: %s", issue.slug(), exc)
    return recovered


def _recover_lease_for(action: Action, *, queue: GitHubIssueQueue) -> None:
    """Best-effort ``RECOVER_STALE_LEASE`` executor (legacy path).

    Kept for callers that emit ``RECOVER_STALE_LEASE`` actions
    directly. Production cleanup paths use
    :func:`_recover_stale_leases` which iterates the candidate
    set rather than relying on the planner's emissions.
    """
    if action.work_item_id is None:
        return
    try:
        issue = queue._resolve_issue(action.work_item_id)
    except Exception:
        return
    state = queue.load_state(issue.slug())
    if state is None:
        return
    try:
        claim = _claim_from_state(state)
    except Exception:
        return
    if state.phase in ("done", "failed", "escalated"):
        return
    new_run_id = f"{claim.run_id}-recover"
    try:
        queue.reclaim_expired(
            issue,
            state,
            new_run_id,
            branch=claim.branch,
            worktree=claim.worktree,
            pr_number=claim.pr_number,
        )
    except Exception as exc:
        log.warning("cleanup: reclaim_expired failed for %s: %s", issue.slug(), exc)


__all__ = [
    "CaoControllerLike",
    "CleanupPolicy",
    "CleanupStateLoadError",
    "GitOpsLike",
    "SweepOutcome",
    "run_cleanup",
]
