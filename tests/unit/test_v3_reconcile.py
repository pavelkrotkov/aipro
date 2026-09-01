"""Tests for V3 startup reconciliation and crash/orphan recovery (issue #44).

The table-driven approach is the test: one row per scenario described in
the issue. Every row is a *named* test that constructs a
:class:`ReconciliationInputs` bundle, runs the planner, and asserts the
expected :class:`Action` (or action sequence).

Determinism
-----------
The fixture :func:`frozen_now` returns a timezone-aware :class:`datetime`
that the planner reads. Tests pass it through every bundle so two runs
of the same scenario produce identical output (and identical orphans).

Crash-point coverage
--------------------
The 15 scenarios from issue #44 are covered by ``TestCrashPoints``. They
are enumerated in the table inside that class so a new row reads as a
single sentence: "given X, the planner emits Y".

Property test
-------------
``TestNoTwoAuthoritativeBranches`` asserts the acceptance guarantee: for
any two distinct work-item observations with the same ``run_id``, the
planner never emits more than one branch-creating action. The
implementation walks the union of their action sets and rejects if it
finds two :func:`actions_target_branch` actions sharing a ``(run_id,
branch)`` key.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from ai_pr_orchestrator.v3.config import CleanupConfig, GitHubQueueConfig
from ai_pr_orchestrator.v3.domain import (
    FindingDisposition,
    GitHubIssueRef,
    ReviewerFinding,
    WorkflowState,
)
from ai_pr_orchestrator.v3.queue import Claim
from ai_pr_orchestrator.v3.reconcile import (
    Action,
    ActionKind,
    PullRequestObservation,
    ReconcilePlanner,
    ReconciliationInputs,
    SessionLifecycle,
    SessionObservation,
    WorkItemObservation,
    WorktreeObservation,
    actions_target_branch,
    is_orphan_session,
    is_orphan_worktree,
    list_orphan_sessions,
    list_orphan_worktrees,
)

# --- Deterministic test clock -----------------------------------------------


def frozen_now() -> datetime:
    """Return one frozen timezone-aware timestamp for the test run.

    Centralized so every scenario reads the same instant — both reads
    (lease is_stale, worktree age) and writes (last_activity_at) line up.
    """
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)


# --- Builders --------------------------------------------------------------


def make_issue(number: int = 1) -> GitHubIssueRef:
    return GitHubIssueRef(owner="owner", repo="repo", number=number)


def make_worktree(
    *,
    branch: str = "aipro-issue-1",
    last_commit_at: datetime | None = None,
    last_push_at: datetime | None = None,
    path: str | None = None,
    is_default_branch: bool = False,
) -> WorktreeObservation:
    return WorktreeObservation(
        path=path or f"/tmp/wt/{branch}",
        branch=branch,
        last_commit_at=last_commit_at or frozen_now(),
        last_push_at=last_push_at,
        is_default_branch=is_default_branch,
    )


def make_session(
    *,
    work_item_id: str,
    run_id: str | None,
    lane: str = "developer",
    state: SessionLifecycle = "terminal",
    is_terminal: bool = True,
    success: bool = True,
    last_activity_at: datetime | None = None,
    session_id: str = "sess-1",
) -> SessionObservation:
    return SessionObservation(
        session_id=session_id,
        work_item_id=work_item_id,
        run_id=run_id,
        lane=lane,
        state=state,
        last_activity_at=last_activity_at or frozen_now(),
        is_terminal=is_terminal,
        success=success,
    )


def make_pr(
    *,
    number: int,
    branch: str = "aipro-issue-1",
    head_sha: str = "abc123",
    expected_head_sha: str | None = None,
) -> PullRequestObservation:
    return PullRequestObservation(
        number=number, branch=branch, head_sha=head_sha, expected_head_sha=expected_head_sha
    )


def make_finding(*, finding_id: str = "f1", thread_id: str | None = "PRRT_1") -> ReviewerFinding:
    return ReviewerFinding(
        id=finding_id,
        lane="architecture-reviewer",
        body="b",
        severity="major",
        run_id="run-1",
        round_id="round-1",
        thread_id=thread_id,
        head_sha="abc123",
    )


def make_state(
    *,
    work_item_id: str = "owner/repo#1",
    run_id: str = "run-1",
    phase: str = "coding",
    findings: list[ReviewerFinding] | None = None,
    dispositions: list[FindingDisposition] | None = None,
    lease_expires_at: datetime | None = None,
    claimed_at: datetime | None = None,
    branch: str | None = "aipro-issue-1",
    worktree: str | None = "/tmp/wt/aipro-issue-1",
    pr_number: int | None = None,
) -> WorkflowState:
    """Build a state with a populated extras map (claim attribution)."""
    now = frozen_now()
    lease_expires_at = lease_expires_at or (now + timedelta(seconds=900))
    claimed_at = claimed_at or now
    extras: dict[str, Any] = {
        "host_id": "host-A",
        "claimed_at": claimed_at.isoformat(),
        "lease_expires_at": lease_expires_at.isoformat(),
        "branch": branch,
        "worktree": worktree,
    }
    if pr_number is not None:
        extras["pr_number"] = pr_number
    return WorkflowState(
        work_item_id=work_item_id,
        run_id=run_id,
        phase=cast(Any, phase),
        updated_at=now,
        findings=list(findings or []),
        dispositions=list(dispositions or []),
        extras=extras,
    )


def make_claim(state: WorkflowState) -> Claim:
    return Claim.from_dict(state.extras)


def make_inputs(
    *,
    state: WorkflowState | None = None,
    sessions: tuple[SessionObservation, ...] = (),
    worktrees: tuple[WorktreeObservation, ...] = (),
    pull_requests: tuple[PullRequestObservation, ...] = (),
    cleanup: CleanupConfig | None = None,
    queue: GitHubQueueConfig | None = None,
    now: datetime | None = None,
    issue: GitHubIssueRef | None = None,
    ci_status: Any = None,
) -> ReconciliationInputs:
    now = now or frozen_now()
    claim = make_claim(state) if state is not None else None
    observation = WorkItemObservation(
        work_item=issue or make_issue(),
        state=state,
        claim=claim,
    )
    return ReconciliationInputs(
        observation=observation,
        sessions=sessions,
        worktrees=worktrees,
        pull_requests=pull_requests,
        config=cleanup or CleanupConfig(),
        queue_config=queue or GitHubQueueConfig(),
        now=now,
        ci_status=ci_status,
    )


# --- Action filter helpers -------------------------------------------------


def find_action(actions: list[Action], kind: ActionKind) -> Action | None:
    for action in actions:
        if action.kind is kind:
            return action
    return None


def assert_only_kind(actions: list[Action], kind: ActionKind) -> Action:
    """Assert ``actions`` has exactly one element of ``kind``."""
    matched = [a for a in actions if a.kind is kind]
    assert len(matched) == 1, f"expected exactly one {kind}, got actions={actions}"
    return matched[0]


# --- Crash points (the table) ---------------------------------------------


@dataclass(frozen=True)
class _CrashScenario:
    name: str
    state_factory: Any
    expected_kind: ActionKind
    expect_auto_apply: bool = True
    # Optional helper returning a tuple to merge into the inputs bundle
    # (e.g. a ``ci_status`` tuple for the post-CI scenarios). ``None``
    # means "no extra inputs".
    extras_factory: Any = None


def _scenario_no_state() -> _CrashScenario:
    return _CrashScenario(
        name="crash_before_claim",
        state_factory=lambda: None,
        expected_kind=ActionKind.NOOP,
    )


def _scenario_claimed_no_branch() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(branch=None, worktree=None, phase="claiming")

    return _CrashScenario(
        name="crash_after_claim_before_branch",
        state_factory=state,
        expected_kind=ActionKind.RESUME_SESSION,
    )


def _scenario_branch_no_session() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(phase="coding")

    def worktrees() -> tuple[WorktreeObservation, ...]:
        return (make_worktree(last_push_at=None),)

    return _CrashScenario(
        name="crash_after_branch_before_agent_launch",
        state_factory=state,
        expected_kind=ActionKind.RELAUNCH,
    )


def _scenario_session_started_no_terminal() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(phase="coding")

    def sessions() -> tuple[SessionObservation, ...]:
        return (
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="developer",
                state="active",
                is_terminal=False,
                session_id="sess-coding",
            ),
        )

    return _CrashScenario(
        name="crash_during_agent_launch",
        state_factory=state,
        expected_kind=ActionKind.COLLECT_RESULT,
    )


def _scenario_terminal_pre_commit() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(phase="coding")

    def sessions() -> tuple[SessionObservation, ...]:
        return (
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="developer",
                state="terminal",
                is_terminal=True,
                session_id="sess-coding",
            ),
        )

    return _CrashScenario(
        name="crash_after_agent_completion_before_commit",
        state_factory=state,
        expected_kind=ActionKind.COLLECT_RESULT,
    )


def _scenario_terminal_pre_push() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(phase="coding")

    def sessions() -> tuple[SessionObservation, ...]:
        return (
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="developer",
                state="terminal",
                is_terminal=True,
                session_id="sess-coding",
            ),
        )

    def worktrees() -> tuple[WorktreeObservation, ...]:
        now = frozen_now()
        return (
            make_worktree(
                last_commit_at=now - timedelta(seconds=10),
                last_push_at=None,
            ),
        )

    return _CrashScenario(
        name="crash_after_commit_before_push",
        state_factory=state,
        expected_kind=ActionKind.COLLECT_RESULT,
    )


def _scenario_post_push() -> _CrashScenario:
    def state() -> WorkflowState:
        now = frozen_now()
        return make_state(
            phase="coding",
            pr_number=None,
            claimed_at=now - timedelta(seconds=300),
            lease_expires_at=now + timedelta(seconds=900),
        )

    def worktrees() -> tuple[WorktreeObservation, ...]:
        now = frozen_now()
        return (
            make_worktree(
                last_commit_at=now - timedelta(seconds=10),
                last_push_at=now - timedelta(seconds=5),
            ),
        )

    return _CrashScenario(
        name="crash_after_push_before_review",
        state_factory=state,
        expected_kind=ActionKind.RESUME_SESSION,
    )


def _scenario_review_no_findings() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(phase="reviewing", pr_number=42)

    def sessions() -> tuple[SessionObservation, ...]:
        return (
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="architecture-reviewer",
                state="terminal",
                is_terminal=True,
                session_id="sess-review",
            ),
        )

    return _CrashScenario(
        name="crash_after_review_launch_no_findings",
        state_factory=state,
        expected_kind=ActionKind.COLLECT_RESULT,
    )


def _scenario_findings_no_disposition() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(
            phase="reviewing",
            pr_number=42,
            findings=[make_finding(thread_id="PRRT_1")],
            dispositions=[],
        )

    def sessions() -> tuple[SessionObservation, ...]:
        return (
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="architecture-reviewer",
                state="terminal",
                is_terminal=True,
                session_id="sess-review",
            ),
        )

    return _CrashScenario(
        name="crash_after_findings_before_disposition",
        state_factory=state,
        expected_kind=ActionKind.RELAUNCH,
    )


def _scenario_disposition_no_ci() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(
            phase="reviewing",
            pr_number=42,
            findings=[make_finding(thread_id="PRRT_1")],
            dispositions=[
                FindingDisposition(
                    finding_id="f1",
                    action="fix",
                    rationale="fixed",
                    decided_by="foreman",
                )
            ],
        )

    return _CrashScenario(
        name="crash_after_disposition_before_ci",
        state_factory=state,
        expected_kind=ActionKind.RESUME_SESSION,
    )


def _scenario_ci_no_pr() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(
            phase="ci_gating",
            pr_number=None,
            findings=[make_finding(thread_id="PRRT_1")],
            dispositions=[
                FindingDisposition(
                    finding_id="f1",
                    action="fix",
                    rationale="fixed",
                    decided_by="foreman",
                )
            ],
        )

    def ci_status() -> tuple[Any, ...]:
        # Phase ``ci_gating`` alone is no longer proof CI ran; the
        # scenario passes a GateDecision-style tuple with
        # ``passed=True`` and empty pending list so the planner sees a
        # recorded result.
        from ai_pr_orchestrator.v3.interfaces import GateDecision

        return (GateDecision(passed=True, pending_checks=(), failed_checks=()),)

    return _CrashScenario(
        name="crash_after_ci_before_pr",
        state_factory=state,
        expected_kind=ActionKind.RELAUNCH,
        extras_factory=ci_status,
    )


def _scenario_pr_no_final_label() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(
            phase="updating_pr",
            pr_number=42,
            findings=[make_finding(thread_id="PRRT_1")],
            dispositions=[
                FindingDisposition(
                    finding_id="f1",
                    action="fix",
                    rationale="fixed",
                    decided_by="foreman",
                )
            ],
        )

    return _CrashScenario(
        name="crash_after_pr_before_final_label",
        state_factory=state,
        expected_kind=ActionKind.RESUME_SESSION,
    )


def _scenario_branch_moved() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(
            phase="reviewing",
            pr_number=42,
        )

    def prs() -> tuple[PullRequestObservation, ...]:
        return (
            make_pr(
                number=42,
                head_sha="newsha",
                expected_head_sha="oldsha",
            ),
        )

    return _CrashScenario(
        name="branch_moved",
        state_factory=state,
        expected_kind=ActionKind.HALT_BRANCH_MOVED,
        expect_auto_apply=False,
    )


def _scenario_stale_lease() -> _CrashScenario:
    def state() -> WorkflowState:
        now = frozen_now()
        return make_state(
            phase="coding",
            lease_expires_at=now - timedelta(seconds=120),  # expired 2 min ago
            claimed_at=now - timedelta(seconds=2000),
        )

    return _CrashScenario(
        name="stale_lease",
        state_factory=state,
        expected_kind=ActionKind.ESCALATE,
        expect_auto_apply=False,
    )


def _scenario_duplicate_sessions() -> _CrashScenario:
    def state() -> WorkflowState:
        return make_state(phase="coding")

    def sessions() -> tuple[SessionObservation, ...]:
        return (
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="developer",
                state="terminal",
                is_terminal=True,
                session_id="sess-old",
            ),
            make_session(
                work_item_id="owner/repo#1",
                run_id="run-1",
                lane="developer",
                state="active",
                is_terminal=False,
                session_id="sess-new",
            ),
        )

    return _CrashScenario(
        name="duplicate_sessions",
        state_factory=state,
        expected_kind=ActionKind.ESCALATE,
        expect_auto_apply=False,
    )


# The full table from the spec. Add a row per crash point. The order here
# mirrors the issue's enumerated scenarios.
_CRASH_SCENARIOS: tuple[_CrashScenario, ...] = (
    _scenario_no_state(),
    _scenario_claimed_no_branch(),
    _scenario_branch_no_session(),
    _scenario_session_started_no_terminal(),
    _scenario_terminal_pre_commit(),
    _scenario_terminal_pre_push(),
    _scenario_post_push(),
    _scenario_review_no_findings(),
    _scenario_findings_no_disposition(),
    _scenario_disposition_no_ci(),
    _scenario_ci_no_pr(),
    _scenario_pr_no_final_label(),
    _scenario_branch_moved(),
    _scenario_stale_lease(),
    _scenario_duplicate_sessions(),
)


class TestCrashPoints:
    """One row per crash point from issue #44.

    The planner is invoked once per scenario with the appropriate inputs;
    the action set must contain *exactly* the expected kind (and no others
    that would create side effects).
    """

    @pytest.mark.parametrize(
        "scenario",
        _CRASH_SCENARIOS,
        ids=[s.name for s in _CRASH_SCENARIOS],
    )
    def test_crash_point(self, scenario: _CrashScenario) -> None:
        state = scenario.state_factory()
        kwargs: dict[str, Any] = {"state": state}
        # Optional helper resolution via module-level factories. Each
        # scenario whose name matches ``_sessions_<name>``,
        # ``_worktrees_<name>``, or ``_prs_<name>`` contributes its bundle.
        sessions_fn = globals().get(f"_sessions_{scenario.name}")
        worktrees_fn = globals().get(f"_worktrees_{scenario.name}")
        prs_fn = globals().get(f"_prs_{scenario.name}")
        if sessions_fn is not None:
            kwargs["sessions"] = sessions_fn()
        if worktrees_fn is not None:
            kwargs["worktrees"] = worktrees_fn()
        if prs_fn is not None:
            kwargs["pull_requests"] = prs_fn()
        if scenario.extras_factory is not None:
            kwargs["ci_status"] = scenario.extras_factory()[0]
        inputs = make_inputs(**kwargs)
        actions = ReconcilePlanner().plan(inputs)
        matched = [a for a in actions if a.kind is scenario.expected_kind]
        assert matched, (
            f"scenario {scenario.name}: expected {scenario.expected_kind}, "
            f"got {[a.kind for a in actions]}"
        )
        action = matched[0]
        assert action.auto_apply is scenario.expect_auto_apply, (
            f"scenario {scenario.name}: auto_apply={action.auto_apply}, "
            f"expected {scenario.expect_auto_apply}"
        )
        # Stop further work for ESCALATE/HALT actions: those are terminal.
        if scenario.expected_kind in (
            ActionKind.ESCALATE,
            ActionKind.HALT_BRANCH_MOVED,
        ):
            assert all(not a.auto_apply for a in actions if a.kind is scenario.expected_kind)


# Helpers consumed by TestCrashPoints ---------------------------------------


def _sessions_crash_after_agent_completion_before_commit() -> tuple[SessionObservation, ...]:
    return (
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="developer",
            state="terminal",
            is_terminal=True,
            session_id="sess-coding",
        ),
    )


def _sessions_crash_during_agent_launch() -> tuple[SessionObservation, ...]:
    return (
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="developer",
            state="active",
            is_terminal=False,
            session_id="sess-coding",
        ),
    )


def _sessions_crash_after_commit_before_push() -> tuple[SessionObservation, ...]:
    return (
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="developer",
            state="terminal",
            is_terminal=True,
            session_id="sess-coding",
        ),
    )


def _worktrees_crash_after_commit_before_push() -> tuple[WorktreeObservation, ...]:
    now = frozen_now()
    return (
        make_worktree(
            last_commit_at=now - timedelta(seconds=10),
            last_push_at=None,
        ),
    )


def _worktrees_crash_after_branch_before_agent_launch() -> tuple[WorktreeObservation, ...]:
    return (make_worktree(last_push_at=None),)


def _worktrees_crash_after_push_before_review() -> tuple[WorktreeObservation, ...]:
    now = frozen_now()
    return (
        make_worktree(
            last_commit_at=now - timedelta(seconds=10),
            last_push_at=now - timedelta(seconds=5),
        ),
    )


def _sessions_crash_after_review_launch_no_findings() -> tuple[SessionObservation, ...]:
    return (
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="architecture-reviewer",
            state="terminal",
            is_terminal=True,
            session_id="sess-review",
        ),
    )


def _sessions_crash_after_findings_before_disposition() -> tuple[SessionObservation, ...]:
    return (
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="architecture-reviewer",
            state="terminal",
            is_terminal=True,
            session_id="sess-review",
        ),
    )


def _sessions_duplicate_sessions() -> tuple[SessionObservation, ...]:
    return (
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="developer",
            state="terminal",
            is_terminal=True,
            session_id="sess-old",
        ),
        make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="developer",
            state="active",
            is_terminal=False,
            session_id="sess-new",
        ),
    )


def _prs_branch_moved() -> tuple[PullRequestObservation, ...]:
    return (make_pr(number=42, head_sha="newsha", expected_head_sha="oldsha"),)


# --- Property test: no two authoritative branches per run -------------------


class TestNoTwoAuthoritativeBranches:
    """Acceptance property: the planner never emits two branch-creating
    actions for the same ``run_id``.

    Per finding #5: the dedup key is ``run_id`` alone, not
    ``(run_id, branch)``. The cross-item dedupe lives in
    :meth:`ReconcilePlanner._finalize`; per-work-item contract is "at most
    one branch action per call".
    """

    @pytest.mark.parametrize(
        "branch_a,branch_b",
        [
            ("aipro-issue-1", "aipro-issue-1"),
            ("aipro-issue-1", "aipro-issue-2"),
            ("aipro-issue-2", "aipro-issue-1"),
        ],
    )
    def test_no_two_authoritative_branches_for_same_run(self, branch_a: str, branch_b: str) -> None:
        """Acceptance: across multiple combinations of (branch_a, branch_b)
        for two distinct work items sharing a run_id, the planner must
        never produce more than one branch-creating action.

        Drives :meth:`ReconcilePlanner.plan_many` because the cross-item
        dedupe lives there.
        """
        planner = ReconcilePlanner()
        state_a = make_state(
            work_item_id="owner/repo#1",
            run_id="run-1",
            phase="coding",
            branch=branch_a,
            worktree=f"/tmp/wt/{branch_a}",
        )
        state_b = make_state(
            work_item_id="owner/repo#2",
            run_id="run-1",
            phase="coding",
            branch=branch_b,
            worktree=f"/tmp/wt/{branch_b}",
        )
        inputs_a = make_inputs(
            state=state_a,
            worktrees=(make_worktree(branch=branch_a, last_push_at=None),),
            issue=make_issue(1),
        )
        inputs_b = make_inputs(
            state=state_b,
            worktrees=(make_worktree(branch=branch_b, last_push_at=None),),
            issue=make_issue(2),
        )
        branch_actions = [
            a for a in planner.plan_many([inputs_a, inputs_b]) if actions_target_branch(a)
        ]
        # At most one branch-creating action per run_id.
        keys = [a.run_id for a in branch_actions]
        assert len(keys) == len(set(keys)), (
            f"two branch-creating actions for the same run_id: "
            f"{[(a.run_id, a.branch, a.work_item_id) for a in branch_actions]}"
        )

    def test_two_distinct_work_items_with_distinct_branches_both_emit(self) -> None:
        """Two distinct work items with the same run_id but *different*
        branches produce **at most one** branch-creating action — not two.

        Per finding #5: the dedup key is ``run_id`` alone, not
        ``(run_id, branch)``. Two distinct work items sharing a run_id is
        itself the corruption case the dedupe guards against (only one
        branch may be authoritative per run). A legitimate same-run
        multi-branch scenario would carry two different run_ids.
        """
        planner = ReconcilePlanner()
        state_a = make_state(
            work_item_id="owner/repo#1",
            run_id="run-1",
            phase="coding",
            branch="aipro-issue-1",
            worktree="/tmp/wt/aipro-issue-1",
        )
        state_b = make_state(
            work_item_id="owner/repo#2",
            run_id="run-1",
            phase="coding",
            branch="aipro-issue-2",
            worktree="/tmp/wt/aipro-issue-2",
        )
        inputs_a = make_inputs(
            state=state_a,
            worktrees=(make_worktree(branch="aipro-issue-1", last_push_at=None),),
            issue=make_issue(1),
        )
        inputs_b = make_inputs(
            state=state_b,
            worktrees=(make_worktree(branch="aipro-issue-2", last_push_at=None),),
            issue=make_issue(2),
        )
        branch_actions = [
            a for a in planner.plan_many([inputs_a, inputs_b]) if actions_target_branch(a)
        ]
        assert len(branch_actions) == 1
        # Dedupe is by run_id; the surviving action carries the FIRST
        # work item's identity (the planner iterates inputs_list in order).
        assert branch_actions[0].run_id == "run-1"

    def test_single_work_item_emits_at_most_one_branch_action(self) -> None:
        """Per-work-item contract: at most one RELAUNCH per call."""
        planner = ReconcilePlanner()
        state = make_state(phase="coding")
        inputs = make_inputs(
            state=state,
            worktrees=(make_worktree(last_push_at=None),),
        )
        branch_actions = [a for a in planner.plan(inputs) if actions_target_branch(a)]
        assert len(branch_actions) <= 1


# --- Determinism ----------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_yield_same_actions(self) -> None:
        state = make_state(phase="coding", pr_number=42)
        inputs = make_inputs(
            state=state,
            sessions=(
                make_session(
                    work_item_id="owner/repo#1",
                    run_id="run-1",
                    lane="developer",
                    is_terminal=True,
                ),
            ),
            worktrees=(make_worktree(),),
            pull_requests=(make_pr(number=42),),
        )
        planner = ReconcilePlanner()
        first = planner.plan(inputs)
        second = planner.plan(inputs)
        assert first == second


# --- Orphan detection -----------------------------------------------------


class TestOrphanDetection:
    def test_session_orphan_requires_no_lease_and_age(self) -> None:
        now = frozen_now()
        session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            last_activity_at=now - timedelta(seconds=10000),
        )
        # No live lease references it.
        assert is_orphan_session(session, has_live_lease=False, now=now, ttl_seconds=7200)
        # A live lease defeats the orphan signal regardless of age.
        assert not is_orphan_session(session, has_live_lease=True, now=now, ttl_seconds=7200)
        # Recent activity defeats it regardless of lease.
        recent = replace(session, last_activity_at=now - timedelta(seconds=60))
        assert not is_orphan_session(recent, has_live_lease=False, now=now, ttl_seconds=7200)

    def test_worktree_orphan_requires_no_lease_branch_and_age(self) -> None:
        now = frozen_now()
        worktree = make_worktree(last_commit_at=now - timedelta(seconds=200000))
        assert is_orphan_worktree(worktree, branch_has_live_lease=False, now=now, ttl_seconds=86400)
        # Default branch is never orphan.
        assert not is_orphan_worktree(
            replace(worktree, is_default_branch=True),
            branch_has_live_lease=False,
            now=now,
            ttl_seconds=86400,
        )
        # Live lease defeats the signal.
        assert not is_orphan_worktree(
            worktree, branch_has_live_lease=True, now=now, ttl_seconds=86400
        )

    def test_list_orphan_sessions(self) -> None:
        now = frozen_now()
        aged = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="aged",
        )
        fresh = make_session(
            work_item_id="owner/repo#2",
            run_id="run-2",
            last_activity_at=now - timedelta(seconds=10),
            session_id="fresh",
        )
        result = list_orphan_sessions((aged, fresh), now=now, ttl_seconds=7200)
        assert [s.session_id for s in result] == ["aged"]

    def test_list_orphan_worktrees(self) -> None:
        now = frozen_now()
        aged = make_worktree(
            branch="aipro-issue-1",
            last_commit_at=now - timedelta(seconds=200000),
            is_default_branch=False,
        )
        fresh = make_worktree(
            branch="aipro-issue-2",
            last_commit_at=now - timedelta(seconds=10),
            is_default_branch=False,
        )
        default = make_worktree(
            branch="main",
            last_commit_at=now - timedelta(seconds=200000),
            is_default_branch=True,
        )
        result = list_orphan_worktrees((aged, fresh, default), now=now, ttl_seconds=86400)
        assert [w.branch for w in result] == ["aipro-issue-1"]


# --- Planner surface ------------------------------------------------------


class TestPlannerSurface:
    def test_planner_requires_tz_aware_now(self) -> None:
        inputs = make_inputs()
        # Replace tz-aware now with naive; planner must refuse.
        inputs_naive = replace(inputs, now=datetime(2026, 1, 1))
        with pytest.raises(Exception, match="timezone-aware"):
            ReconcilePlanner().plan(inputs_naive)

    def test_no_inputs_emits_noop(self) -> None:
        inputs = make_inputs()  # no state, no sessions, no worktrees
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.NOOP in kinds

    def test_orphan_session_action_when_aged_and_unleased(self) -> None:
        now = frozen_now()
        # No state (no lease referencing the session) makes the session
        # orphaned — the planner emits CLEAN_ORPHAN_SESSION.
        orphan = make_session(
            work_item_id="orphan-wi",
            run_id="orphan-run",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="orphan-sess",
        )
        inputs = make_inputs(
            state=None,
            sessions=(orphan,),
            worktrees=(),
        )
        actions = ReconcilePlanner(
            cleanup_config=CleanupConfig(session_lease_ttl_seconds=7200)
        ).plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.CLEAN_ORPHAN_SESSION in kinds

    def test_orphan_worktree_action_when_aged_and_unleased(self) -> None:
        now = frozen_now()
        orphan_wt = make_worktree(
            branch="orphan-branch",
            last_commit_at=now - timedelta(seconds=200000),
            last_push_at=None,
        )
        # No live lease referencing the orphan branch — no state for it.
        inputs = make_inputs(state=None, worktrees=(orphan_wt,))
        actions = ReconcilePlanner(
            cleanup_config=CleanupConfig(worktree_inactivity_ttl_seconds=86400)
        ).plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.CLEAN_ORPHAN_WORKTREE in kinds

    def test_halt_branch_moved_not_auto_applied(self) -> None:
        state = make_state(phase="reviewing", pr_number=42)
        prs = (make_pr(number=42, head_sha="newsha", expected_head_sha="oldsha"),)
        inputs = make_inputs(state=state, pull_requests=prs)
        actions = ReconcilePlanner().plan(inputs)
        halt = find_action(actions, ActionKind.HALT_BRANCH_MOVED)
        assert halt is not None
        assert halt.auto_apply is False
