"""Unit tests for V3 domain types."""

from __future__ import annotations

import pytest

from ai_pr_orchestrator.v3.domain import (
    DomainError,
    FailureSummary,
    FindingDisposition,
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
    ReviewerFinding,
    StagnationSummary,
    WorkflowState,
    WorkItem,
)


class TestGitHubIssueRef:
    def test_valid(self) -> None:
        ref = GitHubIssueRef(owner="pavelkrotkov", repo="aipro", number=41)
        assert ref.slug() == "pavelkrotkov/aipro#41"

    @pytest.mark.parametrize(
        "owner,repo,number",
        [
            ("", "aipro", 1),
            ("pavelkrotkov", "", 1),
            ("pavelkrotkov", "aipro", 0),
            ("pavelkrotkov", "aipro", -3),
        ],
    )
    def test_invalid(self, owner: str, repo: str, number: int) -> None:
        with pytest.raises(DomainError):
            GitHubIssueRef(owner=owner, repo=repo, number=number)


class TestWorkflowState:
    def test_valid_transient_phase(self) -> None:
        state = WorkflowState(work_item_id="wi-1", run_id="run-1", phase="coding")
        assert state.terminal_reason is None

    def test_terminal_phase_requires_reason(self) -> None:
        with pytest.raises(DomainError, match="terminal_reason"):
            WorkflowState(work_item_id="wi-1", run_id="run-1", phase="done")
        state = WorkflowState(
            work_item_id="wi-1", run_id="run-1", phase="done", terminal_reason="merged"
        )
        assert state.terminal_reason == "merged"

    def test_reason_on_transient_phase_is_invalid(self) -> None:
        with pytest.raises(DomainError, match="only valid in terminal phases"):
            WorkflowState(
                work_item_id="wi-1", run_id="run-1", phase="coding", terminal_reason="nope"
            )

    def test_invalid_phase(self) -> None:
        with pytest.raises(DomainError, match="Invalid phase"):
            WorkflowState(work_item_id="wi-1", run_id="run-1", phase="teleporting")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_round_trip(self) -> None:
        state = WorkflowState(
            work_item_id="wi-1", run_id="run-1", phase="reviewing", round_id="round-2"
        )
        data = state.to_dict()
        assert WorkflowState.from_dict(data) == state

    def test_from_dict_drops_unknown_fields(self) -> None:
        data = {
            "work_item_id": "wi-1",
            "run_id": "run-1",
            "phase": "queued",
            "future_field_added_by_newer_version": 42,
        }
        state = WorkflowState.from_dict(data)
        assert state.phase == "queued"


class TestWorkItem:
    def test_round_trip(self) -> None:
        item = WorkItem(
            id="wi-1",
            issue=GitHubIssueRef(owner="o", repo="r", number=7),
            title="Do the thing",
            labels=["v3-work"],
        )
        assert WorkItem.from_dict(item.to_dict()) == item


class TestLaneAndModel:
    def test_lane_identity_roles(self) -> None:
        lane = LaneIdentity(lane="rev-1", role="reviewer", profile_template="aipro-reviewer")
        assert lane.role == "reviewer"

    def test_lane_identity_invalid_role(self) -> None:
        with pytest.raises(DomainError):
            LaneIdentity(lane="x", role="janitor", profile_template="p")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_model_assignment(self) -> None:
        assignment = ModelAssignment(lane="worker-1", model_ref="coder-main")
        assert assignment.model_ref == "coder-main"

    @pytest.mark.parametrize("lane,model_ref", [("", "m"), ("lane", "")])
    def test_model_assignment_invalid(self, lane: str, model_ref: str) -> None:
        with pytest.raises(DomainError):
            ModelAssignment(lane=lane, model_ref=model_ref)


class TestFindings:
    def test_finding_round_trip(self) -> None:
        finding = ReviewerFinding(
            id="f1",
            lane="rev-1",
            body="Bug",
            severity="major",
            run_id="run-1",
            round_id="r1",
            line=12,
        )
        assert ReviewerFinding.from_dict(finding.to_dict()) == finding

    def test_finding_invalid_severity(self) -> None:
        with pytest.raises(DomainError, match="severity"):
            ReviewerFinding(
                id="f1",
                lane="rev",
                body="x",
                severity="catastrophic",  # ty: ignore[invalid-argument-type]
                run_id="r",
                round_id="r1",
            )  # type: ignore[arg-type]

    def test_finding_invalid_line(self) -> None:
        with pytest.raises(DomainError, match="line"):
            ReviewerFinding(
                id="f1", lane="rev", body="x", severity="minor", run_id="r", round_id="r1", line=0
            )

    def test_finding_empty_body(self) -> None:
        with pytest.raises(DomainError, match="body"):
            ReviewerFinding(
                id="f1", lane="rev", body="", severity="minor", run_id="r", round_id="r1"
            )

    def test_disposition(self) -> None:
        d = FindingDisposition(
            finding_id="f1", action="fix", rationale="real bug", decided_by="foreman"
        )
        assert FindingDisposition.from_dict(d.to_dict()) == d

    def test_disposition_invalid_action(self) -> None:
        with pytest.raises(DomainError):
            FindingDisposition(
                finding_id="f1",
                action="ignore_forever",  # ty: ignore[invalid-argument-type]
                rationale="x",
                decided_by="f",
            )  # type: ignore[arg-type]


class TestFailureAndStagnation:
    def test_failure_round_trip(self) -> None:
        f = FailureSummary(run_id="run-1", work_item_id="wi-1", kind="ci_failure", message="red")
        assert FailureSummary.from_dict(f.to_dict()) == f

    def test_failure_invalid_kind(self) -> None:
        with pytest.raises(DomainError, match="failure kind"):
            FailureSummary(run_id="r", work_item_id="w", kind="implosion", message="m")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_failure_invalid_count(self) -> None:
        with pytest.raises(DomainError):
            FailureSummary(
                run_id="r",
                work_item_id="w",
                kind="coder_failure",
                message="m",
                consecutive_failures=0,
            )

    def test_stagnation_round_trip(self) -> None:
        s = StagnationSummary(
            run_id="run-1", work_item_id="wi-1", rounds_without_progress=4, last_round_id="r4"
        )
        assert StagnationSummary.from_dict(s.to_dict()) == s

    def test_stagnation_invalid(self) -> None:
        with pytest.raises(DomainError):
            StagnationSummary(run_id="r", work_item_id="w", rounds_without_progress=0)
