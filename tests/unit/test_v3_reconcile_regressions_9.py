"""Regression test for Codex review round 1 finding #9 on reconciliation (issue #44).

--- Finding 9: post_disposition requires ALL findings dispositioned --------.
"""

from __future__ import annotations

from _reconcile_builders import (
    make_disposition,
    make_finding,
    make_inputs,
    make_session,
    make_state,
)

from ai_pr_orchestrator.v3.domain import ReviewerFinding
from ai_pr_orchestrator.v3.lanes import REQUIREMENTS_REVIEWER_LANE
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    ReconcilePlanner,
)


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
