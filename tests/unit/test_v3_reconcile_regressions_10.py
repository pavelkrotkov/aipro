"""Regression test for Codex review round 1 finding #10 on reconciliation (issue #44).

--- Finding 10: ci_recorded requires a real GateDecision snapshot ----------.
"""

from __future__ import annotations

from _reconcile_builders import (
    make_disposition,
    make_finding,
    make_inputs,
    make_state,
)

from ai_pr_orchestrator.v3.interfaces import GateDecision
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    ReconcilePlanner,
)


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
