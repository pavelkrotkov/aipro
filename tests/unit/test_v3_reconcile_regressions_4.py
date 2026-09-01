"""Regression test for Codex review round 1 finding #4 on reconciliation (issue #44).

--- Finding 4: terminal phase priority ------------------------------------.
"""

from __future__ import annotations

import pytest
from _reconcile_builders import (
    make_inputs,
    make_state,
)

from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    ReconcilePlanner,
)


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
