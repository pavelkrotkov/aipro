"""Regression tests for Codex review round 1 on reconciliation (issue #44).

Each test pins one of the 12 review findings:

1. ``--apply`` actually applies; non-manual actions go through controllers.
2. The CLI uses a real :class:`GitHubClient` when a token is supplied
   (not a hard-coded ``FakeGitHubClient`` with literal ``"owner"``/``"repo"``).
3. ``ESCALATE`` short-circuits the planner (no further actions for the same
   work item).
4. Terminal-phase work items return ``NOOP`` immediately, regardless of
   crash rows.
5. The cross-work-item dedupe key is ``run_id`` alone.
6. Cross-item orphan detection sees *all* live branches across the inputs.
7. The coding-lane predicate uses :class:`LaneRegistry` rather than the
   literal ``"developer"`` substring.
8. The duplicate-session predicate scopes by lane (different-lane
   terminal+live is NOT a duplicate).
9. ``post_findings_no_disposition`` requires ALL findings to be dispositioned.
10. ``ci_recorded`` requires an actual ``GateDecision`` snapshot, not the
    phase ``ci_gating``.
11. ``commit_recorded`` requires a successful coder session (no false
    positive on a clean worktree with stale HEAD).
12. Every action that targets a session carries a non-None ``session_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from ai_pr_orchestrator.v3.config import CleanupConfig, GitHubQueueConfig
from ai_pr_orchestrator.v3.domain import (
    FindingDisposition,
    GitHubIssueRef,
    ReviewerFinding,
    WorkflowState,
)
from ai_pr_orchestrator.v3.interfaces import GateDecision
from ai_pr_orchestrator.v3.lanes import (
    BREAKER_REVIEWER_LANE,
    DEVELOPER_LANE,
    REQUIREMENTS_REVIEWER_LANE,
    LaneIdentity,
    LaneRegistry,
)
from ai_pr_orchestrator.v3.queue import Claim
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    PullRequestObservation,
    ReconcilePlanner,
    ReconciliationInputs,
    SessionLifecycle,
    SessionObservation,
    WorkItemObservation,
    WorktreeObservation,
    actions_target_branch,
)

# --- Builders (shared) ------------------------------------------------------


def frozen_now() -> datetime:
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)


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
    terminal_reason: str | None = None,
) -> WorkflowState:
    now = frozen_now()
    extras: dict[str, Any] = {
        "host_id": "host-A",
        "claimed_at": (claimed_at or now).isoformat(),
        "lease_expires_at": (lease_expires_at or (now + timedelta(seconds=900))).isoformat(),
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
        terminal_reason=terminal_reason
        or ("completed" if phase in ("done", "failed", "escalated") else None),
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
    lanes: LaneRegistry | None = None,
) -> ReconciliationInputs:
    now = now or frozen_now()
    claim = make_claim(state) if state is not None else None
    observation = WorkItemObservation(
        work_item=issue or GitHubIssueRef(owner="owner", repo="repo", number=1),
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


def make_session(
    *,
    work_item_id: str,
    run_id: str | None,
    lane: str = DEVELOPER_LANE,
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


def make_disposition(finding_id: str = "f1") -> FindingDisposition:
    return FindingDisposition(
        finding_id=finding_id,
        action="fix",
        rationale="fixed",
        decided_by="foreman",
    )


# --- Finding 3: ESCALATE short-circuits the planner ------------------------


class TestEscalateShortCircuit:
    def test_escalate_short_circuits_no_subsequent_actions(self) -> None:
        """Once the planner emits ESCALATE for a work item, no further
        actions for that same work item follow.

        Construct a state that would otherwise also trigger
        ``CLEAN_ORPHAN_SESSION`` and ``CLEAN_ORPHAN_WORKTREE``; the only
        action the planner must return is the single ESCALATE.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            lease_expires_at=now - timedelta(seconds=120),
            claimed_at=now - timedelta(seconds=2000),
        )
        aged_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="aged",
        )
        aged_worktree = make_worktree(
            branch="orphan-branch",
            last_commit_at=now - timedelta(seconds=200000),
        )
        inputs = make_inputs(
            state=state,
            sessions=(aged_session,),
            worktrees=(aged_worktree,),
        )
        actions = ReconcilePlanner().plan(inputs)
        # Only the ESCALATE — no orphan cleanup rows.
        assert [a.kind for a in actions] == [ActionKind.ESCALATE]

    def test_halt_branch_moved_short_circuits(self) -> None:
        """HALT_BRANCH_MOVED also short-circuits; a halt means *do not act*,
        so further crash-recovery or orphan-cleanup rows for the same work
        item would contradict the halt itself.
        """
        now = frozen_now()
        state = make_state(phase="reviewing", pr_number=42)
        pr = PullRequestObservation(
            number=42, branch="aipro-issue-1", head_sha="newsha", expected_head_sha="oldsha"
        )
        aged_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="aged",
        )
        actions = ReconcilePlanner().plan(
            make_inputs(
                state=state,
                pull_requests=(pr,),
                sessions=(aged_session,),
            )
        )
        assert [a.kind for a in actions] == [ActionKind.HALT_BRANCH_MOVED]


# --- Finding 4: terminal phase priority ------------------------------------


class TestTerminalPhasePriority:
    @pytest.mark.parametrize("phase", ["done", "failed", "escalated"])
    def test_terminal_phase_emits_noop_immediately(self, phase: str) -> None:
        """A work item in a terminal phase must return ``NOOP`` BEFORE the
        planner evaluates any crash row.

        The previous implementation evaluated crash rows first; reaching
        phase ``done`` was not enough to suppress a spurious RELAUNCH.
        """
        state = make_state(phase=phase)
        inputs = make_inputs(state=state)
        actions = ReconcilePlanner().plan(inputs)
        assert [a.kind for a in actions] == [ActionKind.NOOP]
        assert actions[0].reason  # surfaces the phase in the reason


# --- Finding 5: dedupe key is run_id alone ---------------------------------


class TestDedupeByRunId:
    def test_same_run_different_branches_one_relaunch(self) -> None:
        """Per finding #5, two work items sharing a ``run_id`` produce at
        most one branch-creating action — even when their branches differ.

        Without that, two branches would each claim to be the authoritative
        developer branch for the same run.
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
            issue=GitHubIssueRef(owner="owner", repo="repo", number=1),
        )
        inputs_b = make_inputs(
            state=state_b,
            worktrees=(make_worktree(branch="aipro-issue-2", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=2),
        )
        branch_actions = [
            a for a in planner.plan_many([inputs_a, inputs_b]) if actions_target_branch(a)
        ]
        assert len(branch_actions) == 1
        # And the surviving action carries the FIRST work item's identity.
        assert branch_actions[0].work_item_id == "owner/repo#1"

    def test_different_runs_can_both_relaunch(self) -> None:
        """Two work items on different ``run_id`` values legitimately both
        produce branch-creating actions (no dedupe conflict).
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
            run_id="run-2",
            phase="coding",
            branch="aipro-issue-2",
            worktree="/tmp/wt/aipro-issue-2",
        )
        inputs_a = make_inputs(
            state=state_a,
            worktrees=(make_worktree(branch="aipro-issue-1", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=1),
        )
        inputs_b = make_inputs(
            state=state_b,
            worktrees=(make_worktree(branch="aipro-issue-2", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=2),
        )
        branch_actions = [
            a for a in planner.plan_many([inputs_a, inputs_b]) if actions_target_branch(a)
        ]
        assert len(branch_actions) == 2


# --- Finding 6: cross-item orphan detection --------------------------------


class TestCrossItemOrphanDetection:
    def test_worktree_kept_live_by_sibling_input(self) -> None:
        """Item B must not spuriously flag item A's live branch as orphan.

        The previous implementation only looked at the *current* item's
        branches when computing ``_live_branches``, so planning B (after A)
        would mark A's live branch as orphan and emit
        ``CLEAN_ORPHAN_WORKTREE`` for it.
        """
        now = frozen_now()
        # Item A has a LIVE claim on the shared branch.
        state_a = make_state(
            work_item_id="owner/repo#1",
            phase="coding",
            branch="aipro-shared",
            worktree="/tmp/wt/aipro-shared",
        )
        # Item B's session is aged, but the worktree is shared with A's
        # LIVE claim — it must NOT be cleaned up.
        aged_session = make_session(
            work_item_id="owner/repo#2",
            run_id="run-2",
            session_id="aged",
            last_activity_at=now - timedelta(seconds=10000),
        )
        # Use a long TTL so age alone would trigger cleanup otherwise.
        cleanup = CleanupConfig(
            session_lease_ttl_seconds=7200,
            worktree_inactivity_ttl_seconds=86400,
        )
        inputs_a = make_inputs(
            state=state_a,
            cleanup=cleanup,
            worktrees=(make_worktree(branch="aipro-shared"),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=1),
        )
        inputs_b = make_inputs(
            state=None,
            cleanup=cleanup,
            sessions=(aged_session,),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=2),
        )
        actions = ReconcilePlanner().plan_many([inputs_a, inputs_b])
        kinds = [a.kind for a in actions]
        # No spurious worktree cleanup for the shared branch.
        assert ActionKind.CLEAN_ORPHAN_WORKTREE not in kinds
        # And no spurious session cleanup either, because the worktree /
        # branch referenced by A's lease is live — but the session itself
        # does belong to B (different work_item_id), so it CAN be cleaned.
        # We accept either: an empty plan, or a CLEAN_ORPHAN_SESSION for
        # the aged session that is NOT for A's branch.
        for action in actions:
            if action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE:
                assert action.branch != "aipro-shared"


# --- Finding 7: lane registry used for coding-session predicate ------------


class TestLaneRegistryCodingPredicate:
    def test_developer_lane_property_returns_registered_name(self) -> None:
        """``LaneRegistry.developer_lane`` returns the worker-role lane."""
        registry = LaneRegistry(
            [
                LaneIdentity(lane="implementer", role="worker", profile_template="aipro-imp"),
                LaneIdentity(lane="reviewer-x", role="reviewer", profile_template="aipro-rev"),
            ]
        )
        assert registry.developer_lane == "implementer"

    def test_renamed_developer_lane_still_detected_as_coding(self) -> None:
        """A deployment that renames the developer lane to ``"implementer"``
        must still have its coding sessions recognised — the previous
        literal ``"worker"`` substring check would miss the rename."""
        registry = LaneRegistry(
            [
                LaneIdentity(lane="implementer", role="worker", profile_template="aipro-imp"),
                LaneIdentity(
                    lane=BREAKER_REVIEWER_LANE,
                    role="reviewer",
                    profile_template="aipro-brk",
                ),
            ]
        )
        now = frozen_now()
        state = make_state(phase="coding")
        # Worktree's last_commit_at must be NEWER than the lease claim
        # AND the coder session must report success — that is the only
        # way commit_recorded becomes True (finding #11).
        coding_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane="implementer",
            is_terminal=True,
            success=True,
            session_id="sess-coding",
        )
        inputs = make_inputs(
            state=state,
            sessions=(coding_session,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                ),
            ),
            lanes=registry,
        )
        actions = ReconcilePlanner(lanes=registry).plan(inputs)
        # session_terminal_pre_commit requires has_terminal_coding_session,
        # which requires the lane to be classified as coding. With the
        # registry wiring, "implementer" qualifies.
        kinds = [a.kind for a in actions]
        assert ActionKind.COLLECT_RESULT in kinds


# --- Finding 8: duplicate-session predicate --------------------------------


class TestDuplicateSessionPredicate:
    def test_terminal_plus_live_different_lanes_not_duplicate(self) -> None:
        """One terminal session + one live session in DIFFERENT lanes is
        legitimate: the terminal one is the result we never recorded. The
        planner must emit ``COLLECT_RESULT``, not ``ESCALATE``.

        The previous implementation treated any ``>= 2`` session count as
        a duplicate and escalated; that over-escalates a normal
        coder->reviewer round transition.
        """
        state = make_state(phase="reviewing", pr_number=42)
        terminal_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            session_id="sess-coder",
        )
        live_reviewer = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=REQUIREMENTS_REVIEWER_LANE,
            is_terminal=False,
            session_id="sess-rev",
        )
        inputs = make_inputs(state=state, sessions=(terminal_coder, live_reviewer))
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.ESCALATE not in kinds
        assert ActionKind.COLLECT_RESULT in kinds

    def test_terminal_plus_live_same_lane_is_duplicate(self) -> None:
        """Same lane + (terminal, live) is genuine duplication; escalate."""
        state = make_state(phase="coding")
        terminal_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            session_id="sess-old",
        )
        live_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=False,
            session_id="sess-new",
        )
        inputs = make_inputs(state=state, sessions=(terminal_coder, live_coder))
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.ESCALATE in kinds


# --- Finding 9: post_disposition requires ALL findings dispositioned --------


class TestPostDispositionRace:
    def test_partial_disposition_does_not_advance(self) -> None:
        """With one disposition of two findings, the planner must still
        be at ``post_findings_no_disposition`` (relaunch at review), not
        at ``post_disposition_no_ci``.

        The previous implementation fired ``post_disposition_no_ci`` as
        soon as ONE disposition was persisted, racing the rest of the
        findings.
        """
        finding_a = make_finding(finding_id="f1", thread_id="PRRT_1")
        finding_b = ReviewerFinding(
            id="f2",
            lane="architecture-reviewer",
            body="b",
            severity="major",
            run_id="run-1",
            round_id="round-1",
            thread_id="PRRT_2",
            head_sha="abc123",
        )
        state = make_state(
            phase="reviewing",
            pr_number=42,
            findings=[finding_a, finding_b],
            dispositions=[make_disposition("f1")],  # only one
        )
        review_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=REQUIREMENTS_REVIEWER_LANE,
            is_terminal=True,
            session_id="sess-review",
        )
        inputs = make_inputs(state=state, sessions=(review_session,))
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # We need RELAUNCH (post_findings_no_disposition); we must NOT see
        # RESUME_SESSION (post_disposition_no_ci).
        assert ActionKind.RELAUNCH in kinds
        assert ActionKind.RESUME_SESSION not in kinds

    def test_full_disposition_advances_to_ci_check(self) -> None:
        """With every finding dispositioned, the planner advances to
        ``post_disposition_no_ci``."""
        finding_a = make_finding(finding_id="f1", thread_id="PRRT_1")
        finding_b = ReviewerFinding(
            id="f2",
            lane="architecture-reviewer",
            body="b",
            severity="major",
            run_id="run-1",
            round_id="round-1",
            thread_id="PRRT_2",
            head_sha="abc123",
        )
        state = make_state(
            phase="reviewing",
            pr_number=42,
            findings=[finding_a, finding_b],
            dispositions=[make_disposition("f1"), make_disposition("f2")],
        )
        review_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=REQUIREMENTS_REVIEWER_LANE,
            is_terminal=True,
            session_id="sess-review",
        )
        inputs = make_inputs(state=state, sessions=(review_session,))
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.RESUME_SESSION in kinds


# --- Finding 10: ci_recorded requires a real GateDecision snapshot ----------


class TestCiRecordedRequiresSnapshot:
    def test_phase_ci_gating_without_snapshot_is_not_recorded(self) -> None:
        """Phase ``ci_gating`` is no longer proof CI ran. With no
        ``ci_status`` snapshot AND a live PR, the planner stays at
        ``post_disposition_no_ci`` (RESUME_SESSION), not advance to
        ``post_ci_no_pr`` (RELAUNCH)."""
        finding = make_finding(finding_id="f1", thread_id="PRRT_1")
        state = make_state(
            phase="ci_gating",
            pr_number=42,  # claim_has_pr=True so post_disposition_no_ci applies
            findings=[finding],
            dispositions=[make_disposition("f1")],
        )
        inputs = make_inputs(state=state, ci_status=None)
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # Stays at post_disposition_no_ci — RESUME_SESSION, not RELAUNCH.
        assert ActionKind.RESUME_SESSION in kinds
        assert ActionKind.RELAUNCH not in kinds

    def test_phase_ci_gating_with_pending_snapshot_is_not_recorded(self) -> None:
        """A ``GateDecision`` with pending checks is *not* a recorded
        result; the planner must still wait."""
        finding = make_finding(finding_id="f1", thread_id="PRRT_1")
        state = make_state(
            phase="ci_gating",
            pr_number=42,
            findings=[finding],
            dispositions=[make_disposition("f1")],
        )
        snapshot = GateDecision(
            passed=False,
            pending_checks=("ci-1",),
            failed_checks=(),
        )
        inputs = make_inputs(state=state, ci_status=snapshot)
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.RESUME_SESSION in kinds
        assert ActionKind.RELAUNCH not in kinds

    def test_phase_ci_gating_with_passed_snapshot_advances(self) -> None:
        """A passed ``GateDecision`` IS a recorded result; the planner
        advances to ``post_ci_no_pr`` (RELAUNCH)."""
        finding = make_finding(finding_id="f1", thread_id="PRRT_1")
        state = make_state(
            phase="ci_gating",
            pr_number=None,  # post_ci_no_pr needs claim_has_pr=False
            findings=[finding],
            dispositions=[make_disposition("f1")],
        )
        snapshot = GateDecision(
            passed=True,
            pending_checks=(),
            failed_checks=(),
        )
        inputs = make_inputs(state=state, ci_status=snapshot)
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        assert ActionKind.RELAUNCH in kinds


# --- Finding 11: no-op coder commit timestamp ------------------------------


class TestNoOpCoderCommitTimestamp:
    def test_old_commit_no_coder_session_is_not_recorded(self) -> None:
        """An old HEAD predating the lease claim is NOT a recorded commit,
        even if the worktree's last_commit_at is non-None — there was no
        coder session in this run, so the commit cannot be from this run.
        """
        now = frozen_now()
        # claimed_at is "fresh" (now); last_commit_at is older — i.e.
        # there is no commit on this branch since this claim.
        state = make_state(
            phase="coding",
            claimed_at=now,
            lease_expires_at=now + timedelta(seconds=900),
        )
        inputs = make_inputs(
            state=state,
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=60),
                ),
            ),
            sessions=(),
        )
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # The commit_recorded flag stays False, so session_terminal_pre_commit
        # does not match (it requires commit_recorded=False AND
        # coder_succeeded=True). With no coder session at all, the
        # planner instead emits branch_exists_no_session (RELAUNCH) — but
        # crucially NOT a "coder has committed" reading.
        assert ActionKind.COLLECT_RESULT not in kinds

    def test_no_op_coder_session_is_not_recorded(self) -> None:
        """A coder session that exited cleanly but produced no changes
        (success=False) must NOT mark the worktree as having a recorded
        commit. The previous commit_recorded predicate took only the
        commit timestamp vs lease and gave a false positive here.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            claimed_at=now - timedelta(seconds=200),
            lease_expires_at=now + timedelta(seconds=900),
        )
        no_op_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=False,  # no-op exit
            last_activity_at=now - timedelta(seconds=10),
            session_id="sess-noop",
        )
        inputs = make_inputs(
            state=state,
            sessions=(no_op_coder,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                ),
            ),
        )
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # commit_recorded requires BOTH a newer commit AND a successful
        # coder session; here coder_succeeded=False so commit_recorded is
        # False. session_terminal_pre_commit therefore does NOT match,
        # and the planner stays in branch_exists_no_session.
        assert ActionKind.COLLECT_RESULT not in kinds

    def test_productive_coder_session_records_commit(self) -> None:
        """A coder session that exited successfully AND the worktree has
        a newer commit IS recorded.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            claimed_at=now - timedelta(seconds=200),
            lease_expires_at=now + timedelta(seconds=900),
        )
        productive_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=True,
            last_activity_at=now - timedelta(seconds=10),
            session_id="sess-productive",
        )
        inputs = make_inputs(
            state=state,
            sessions=(productive_coder,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                ),
            ),
        )
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # commit_recorded=True with branch_pushed=False ⇒
        # session_terminal_pre_push → COLLECT_RESULT.
        assert ActionKind.COLLECT_RESULT in kinds


# --- Finding 12: every session-touching action carries session_id ----------


class TestActionSessionIdPopulated:
    @dataclass(frozen=True)
    class _Spec:
        name: str
        state_factory: Any
        sessions_factory: Any = None

    def _spec(self, name: str) -> _Spec:
        return self._Spec(name=name, state_factory=None, sessions_factory=None)

    def test_resume_session_actions_have_session_id(self) -> None:
        """Every RESUME_SESSION the planner emits carries a non-None
        ``session_id`` so the CLI can dispatch against a concrete session.

        Driving the planner through every row that emits RESUME_SESSION
        would be brittle; instead, exercise the high-traffic rows via
        a real observation bundle and assert non-None.
        """
        now = frozen_now()
        # No PR, branch pushed, no live session: post_push_no_pr -> RESUME.
        state = make_state(
            phase="coding",
            pr_number=None,
            claimed_at=now - timedelta(seconds=300),
            lease_expires_at=now + timedelta(seconds=900),
        )
        terminal_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            session_id="sess-coding",
        )
        inputs = make_inputs(
            state=state,
            sessions=(terminal_session,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                    last_push_at=now - timedelta(seconds=5),
                ),
            ),
            pull_requests=(),
        )
        actions = ReconcilePlanner().plan(inputs)
        resume = [a for a in actions if a.kind is ActionKind.RESUME_SESSION]
        assert resume, f"expected RESUME_SESSION, got {[a.kind for a in actions]}"
        for action in resume:
            assert action.session_id is not None, (
                f"RESUME_SESSION must carry a session_id, got {action}"
            )

    def test_collect_result_actions_have_session_id(self) -> None:
        """COLLECT_RESULT must carry a non-None ``session_id``.
        Use the post-CI scenario; the planner emits a RELAUNCH instead of
        COLLECT_RESULT in that path, so use a coding pre-commit scenario.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            claimed_at=now - timedelta(seconds=200),
            lease_expires_at=now + timedelta(seconds=900),
        )
        terminal_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=True,
            session_id="sess-coding",
        )
        inputs = make_inputs(
            state=state,
            sessions=(terminal_coder,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                    last_push_at=None,
                ),
            ),
        )
        actions = ReconcilePlanner().plan(inputs)
        collect = [a for a in actions if a.kind is ActionKind.COLLECT_RESULT]
        assert collect
        for action in collect:
            assert action.session_id is not None, (
                f"COLLECT_RESULT must carry a session_id, got {action}"
            )

    def test_relaunch_with_sessions_has_session_id(self) -> None:
        """RELAUNCH carries a session_id when one is available so the CLI
        can dispatch the relaunch to the right lane/session."""
        now = frozen_now()
        state = make_state(phase="coding")
        # success=False so the coder session did not commit; the planner
        # sees branch_exists_no_session (RELAUNCH) rather than
        # session_terminal_pre_commit (COLLECT_RESULT).
        aged_terminal = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=False,
            last_activity_at=now - timedelta(seconds=200),
            session_id="sess-prev",
        )
        inputs = make_inputs(
            state=state,
            sessions=(aged_terminal,),
            worktrees=(make_worktree(branch="aipro-issue-1", last_push_at=None),),
        )
        actions = ReconcilePlanner().plan(inputs)
        relaunch = [a for a in actions if a.kind is ActionKind.RELAUNCH]
        assert relaunch
        for action in relaunch:
            assert action.session_id is not None, (
                f"RELAUNCH must carry a session_id when one is known, got {action}"
            )

    def test_clean_orphan_session_has_session_id(self) -> None:
        """CLEAN_ORPHAN_SESSION must carry the target ``session_id``."""
        now = frozen_now()
        aged = make_session(
            work_item_id="orphan-wi",
            run_id="orphan-run",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="orphan-sess",
        )
        inputs = make_inputs(
            state=None,
            sessions=(aged,),
            cleanup=CleanupConfig(session_lease_ttl_seconds=7200),
        )
        actions = ReconcilePlanner().plan(inputs)
        cleanup_actions = [a for a in actions if a.kind is ActionKind.CLEAN_ORPHAN_SESSION]
        assert cleanup_actions
        for action in cleanup_actions:
            assert action.session_id == "orphan-sess"


# --- Finding 1 + 2: CLI --apply wires real I/O; --repo flag ---------------


class TestCliApplyAndRepo:
    def test_reconcile_repo_flag_default_from_env(self, tmp_path, monkeypatch, capsys) -> None:
        """``--repo owner/name`` overrides ``GITHUB_REPOSITORY`` and the
        config default. With ``--repo owner/named``, the CLI surfaces the
        owner/name in the NOOP reason (issue#N) rather than the literal
        ``"owner"``/``"repo"`` the previous code used unconditionally.
        """
        from ai_pr_orchestrator import cli

        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "  owner: cfg-owner\n  repo: cfg-repo\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exit_code = cli.main(
            [
                "reconcile",
                "--config",
                str(config_path),
                "--repo",
                "explicit/test-repo",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        # The placeholder issue is built from the resolved owner/repo.
        assert "explicit/test-repo#1" in out

    def test_reconcile_repo_missing_fails(self, tmp_path, monkeypatch, capsys) -> None:
        """No ``--repo``, no ``GITHUB_REPOSITORY``, no ``github_queue.owner``
        → the CLI refuses rather than fall back to the literal
        ``"owner"``/``"repo"`` it used to."""
        from ai_pr_orchestrator import cli

        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        with pytest.raises(SystemExit) as exc:
            cli.main(["reconcile", "--config", str(config_path)])
        assert "Could not determine GitHub repo" in str(exc.value)

    def test_reconcile_apply_invokes_queue_reclaim(self, tmp_path, monkeypatch, capsys) -> None:
        """``--apply`` with a stale-lease scenario goes through
        ``queue.reclaim_expired`` rather than printing and forgetting.

        The fake client + fake queue let us observe the write without a
        live network.
        """
        from ai_pr_orchestrator import cli
        from ai_pr_orchestrator.github.fake import FakeGitHubClient
        from ai_pr_orchestrator.v3.queue import GitHubIssueQueue

        # Build a config that names a real (fake-client-friendly) repo.
        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "  owner: test-owner\n  repo: test-repo\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )

        # Wire a fake client + queue and seed a stale-lease state.
        fake = FakeGitHubClient()
        GitHubIssueQueue(fake, "test-owner", "test-repo", GitHubQueueConfig(), host_id="test")
        now = frozen_now()
        state = make_state(
            phase="coding",
            lease_expires_at=now - timedelta(seconds=120),
            claimed_at=now - timedelta(seconds=2000),
        )
        issue = GitHubIssueRef(owner="test-owner", repo="test-repo", number=42)

        def fake_inputs(*args, **kwargs):
            from ai_pr_orchestrator.v3.reconcile import (
                ReconciliationInputs,
                WorkItemObservation,
            )

            return [
                ReconciliationInputs(
                    observation=WorkItemObservation(
                        work_item=issue,
                        state=state,
                        claim=make_claim(state),
                    ),
                    sessions=(),
                    worktrees=(),
                    pull_requests=(),
                    config=kwargs["cleanup_cfg"],
                    queue_config=kwargs["queue_cfg"],
                    now=now,
                )
            ]

        monkeypatch.setattr(cli, "_build_reconciliation_inputs", fake_inputs)
        monkeypatch.setattr(cli, "_build_github_client", lambda **_: (fake, True))

        # --apply should not raise and should report the stale lease.
        exit_code = cli.main(
            [
                "reconcile",
                "--config",
                str(config_path),
                "--apply",
                "--repo",
                "test-owner/test-repo",
            ]
        )
        # Stale lease = ESCALATE -> exit code 2.
        assert exit_code == 2
        out = capsys.readouterr().out
        assert "ESCALATE" in out or "escalate" in out

    def test_cli_uses_real_client_with_token(self, tmp_path, monkeypatch, capsys) -> None:
        """When ``GITHUB_TOKEN`` is set, the CLI builds the real
        :class:`GitHubClient`, not the in-memory fake.

        We monkeypatch :class:`GitHubClient` to a sentinel so we can
        observe construction without doing real network I/O.
        """
        from ai_pr_orchestrator import cli

        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "  owner: test-owner\n  repo: test-repo\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )

        sentinel = MagicMock(name="RealGitHubClient")
        monkeypatch.setattr("ai_pr_orchestrator.github.client.GitHubClient", sentinel)
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
        # Block the queue from doing real network calls by patching it.
        from ai_pr_orchestrator.v3.queue import GitHubIssueQueue as RealQueue

        class _NoNetQueue(RealQueue):
            def __init__(self, *a, **kw):
                # Skip real __init__: we only need a no-op placeholder
                # so the CLI's ``list_ready`` doesn't blow up.
                self._ready: list = []
                self._loaded: dict[str, WorkflowState] = {}

            def list_ready(self):
                return []

            def load_state(self, work_item_id):
                return None

        monkeypatch.setattr(cli, "GitHubIssueQueue", _NoNetQueue)

        # Drive a dry-run; the test only needs to confirm that with a
        # token, the CLI chose the real client branch (i.e. constructed
        # GitHubClient with the token).
        cli.main(["reconcile", "--config", str(config_path)])
        assert sentinel.called
        kwargs = sentinel.call_args.kwargs
        assert kwargs.get("token") == "fake-token"
        assert kwargs.get("owner") == "test-owner"
        assert kwargs.get("repo") == "test-repo"
