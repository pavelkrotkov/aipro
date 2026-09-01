"""Regression test for Codex review round 1 finding #8 on reconciliation (issue #44).

--- Finding 8: duplicate-session predicate --------------------------------.
"""

from __future__ import annotations

from _reconcile_builders import (
    make_inputs,
    make_session,
    make_state,
)

from ai_pr_orchestrator.v3.lanes import (
    DEVELOPER_LANE,
    REQUIREMENTS_REVIEWER_LANE,
)
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    ReconcilePlanner,
)


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
