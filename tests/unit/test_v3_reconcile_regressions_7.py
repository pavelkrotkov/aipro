"""Regression test for Codex review round 1 finding #7 on reconciliation (issue #44).

--- Finding 7: lane registry used for coding-session predicate ------------.
"""

from __future__ import annotations

from datetime import timedelta

from _reconcile_builders import (
    frozen_now,
    make_inputs,
    make_session,
    make_state,
    make_worktree,
)

from ai_pr_orchestrator.v3.lanes import (
    BREAKER_REVIEWER_LANE,
    LaneIdentity,
    LaneRegistry,
)
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    ReconcilePlanner,
)


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
