"""Untrusted reviewer output crosses the real CAO/controller boundary."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ai_pr_orchestrator.v3.cao import CaoReviewerOutputError, session_name_for
from ai_pr_orchestrator.v3.domain import Evidence, ReviewerFinding
from ai_pr_orchestrator.v3.findings import FindingRegistry
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext
from ai_pr_orchestrator.v3.lanes import LaneRegistry


@pytest.fixture
def review_lane(cao_lane_executor, fake_cao, lane_registry, tmp_path):
    lane = lane_registry.get("requirements-reviewer")
    context = LaneExecutionContext("review-output", round_id="review-1")

    def execute(output, *, role_lane=lane):
        fake_cao.set_output(session_name_for(context.run_id, role_lane.lane), output)
        return cao_lane_executor.execute(role_lane, "review this change", str(tmp_path), context)

    return execute


@pytest.mark.parametrize(
    "output",
    ["", "   ", "There is a blocking bug", "```json\n[]\n```", "{}", "null", "[", "[null]"],
)
def test_unparsed_reviewer_output_fails_closed(review_lane, output):
    with pytest.raises(CaoReviewerOutputError, match="reviewer output"):
        review_lane(output)


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"severity": "critical"},
        {"body": 42},
        {"status": "resolved", "status_reason": "fine"},
        {"lane": "breaker-reviewer"},
        {"run_id": "other"},
        {"round_id": "review-0"},
        {"path": "../secret"},
        {"line": True},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": True},
        {"sources": {}},
        {"evidence": {}},
        {"evidence": [{"kind": "command", "text": "pytest", "extras": 42}]},
        {"evidence": [{"kind": "file", "path": "../secret"}]},
    ],
)
def test_invalid_finding_fields_fail_at_ingress(review_lane, invalid_fields):
    payload = {"id": "bug", "body": "Missing guard", "severity": "major", **invalid_fields}
    with pytest.raises(CaoReviewerOutputError, match="reviewer output"):
        review_lane(json.dumps([payload]))


def test_duplicate_finding_ids_fail_closed(review_lane):
    payload = {"id": "bug", "body": "Missing guard", "severity": "major"}
    with pytest.raises(CaoReviewerOutputError, match="duplicate finding ids"):
        review_lane(json.dumps([payload, payload]))


@pytest.mark.parametrize("created_at", ["1900-01-01T00:00:00+00:00", None])
def test_reviewer_cannot_supply_engine_timestamp(review_lane, created_at):
    payload = {"id": "bug", "body": "Missing guard", "severity": "major", "created_at": created_at}
    with pytest.raises(CaoReviewerOutputError, match="created_at"):
        review_lane(json.dumps([payload]))


def test_review_findings_reuse_domain_schema_and_current_turn(review_lane):
    finding = ReviewerFinding(
        id="bug-1",
        body="Missing guard",
        severity="major",
        lane="requirements-reviewer",
        run_id="review-output",
        round_id="review-1",
        path="src/example.py",
        line=3,
        evidence=[Evidence(kind="command", text="pytest tests/test_example.py")],
        falsification="Add the guard and rerun the failing test",
    )
    payload = finding.to_dict()
    del payload["created_at"]
    before = datetime.now(UTC)
    result = review_lane(json.dumps([payload]))
    received = result.findings[0]
    assert before <= received.created_at <= datetime.now(UTC)
    assert received == replace(finding, created_at=received.created_at)
    assert result.exit_code == 0
    prior = replace(finding, id="engine-earlier", created_at=datetime(2000, 1, 1, tzinfo=UTC))
    registry = FindingRegistry()
    registry.register(prior)
    registry.register(received)
    assert registry.deduplicate()[0].id == prior.id


def test_review_minimal_finding_uses_confirmed_attribution(review_lane):
    result = review_lane('[{"id":"bug-1","body":"Missing guard","severity":"major"}]')
    finding = result.findings[0]
    assert (finding.lane, finding.run_id, finding.round_id) == (
        "requirements-reviewer",
        "review-output",
        "review-1",
    )
    assert finding.status == "open"


def test_only_explicit_empty_array_is_clean(review_lane):
    result = review_lane("[]")
    assert result.findings == []
    assert result.output_summary == "[]"


def test_worker_plaintext_is_not_parsed_as_review(review_lane):
    result = review_lane(
        "implemented the change", role_lane=LaneRegistry.default().get("developer")
    )
    assert result.exit_code == 0
    assert result.findings == []
    assert result.output_summary == "implemented the change"
