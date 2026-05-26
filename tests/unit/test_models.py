"""Tests for core domain models and JSON serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from ai_pr_orchestrator.models import (
    AgentRunResult,
    CheckRun,
    CostTracker,
    Decision,
    Finding,
    FixTask,
    HandledFinding,
    ModelError,
    PlannedAction,
    PullRequest,
    ReviewerTrigger,
    ReviewThread,
    RuntimeState,
    TestResult,
    TokenUsage,
)

NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 5, 25, 13, 0, 0, tzinfo=UTC)


def _make_state(**overrides: Any) -> RuntimeState:
    defaults: dict[str, Any] = {
        "version": 1,
        "pr_number": 42,
        "head_sha": "abc123",
        "status": "init",
        "round_index": 0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return RuntimeState(**defaults)


def _roundtrip_json(data: dict) -> dict:
    return json.loads(json.dumps(data))


# --- RuntimeState round-trip ---


class TestRuntimeStateRoundTrip:
    def test_basic_round_trip(self) -> None:
        state = _make_state()
        restored = RuntimeState.from_dict(_roundtrip_json(state.to_dict()))

        assert restored.version == state.version
        assert restored.pr_number == state.pr_number
        assert restored.head_sha == state.head_sha
        assert restored.status == state.status
        assert restored.round_index == state.round_index

    def test_all_optional_fields_none(self) -> None:
        state = _make_state(base_sha=None, last_error=None, done_reason=None)
        restored = RuntimeState.from_dict(_roundtrip_json(state.to_dict()))

        assert restored.base_sha is None
        assert restored.last_error is None
        assert restored.done_reason is None

    def test_with_handled_findings(self) -> None:
        hf = HandledFinding(
            finding_id="f1",
            verdict="accepted",
            confidence="high",
            reason="Valid issue",
            reply="Fixed in abc123",
            should_resolve=True,
            changed_files=["src/foo.py"],
            handled_at=NOW,
        )
        state = _make_state(handled_findings={"f1": hf})
        restored = RuntimeState.from_dict(_roundtrip_json(state.to_dict()))

        assert "f1" in restored.handled_findings
        assert restored.handled_findings["f1"].verdict == "accepted"
        assert restored.handled_findings["f1"].handled_at == NOW

    def test_with_trigger_history(self) -> None:
        trigger = ReviewerTrigger(
            reviewer_name="gemini_github",
            round_index=0,
            timestamp=NOW,
            head_sha="abc123",
        )
        state = _make_state(trigger_history=[trigger])
        restored = RuntimeState.from_dict(_roundtrip_json(state.to_dict()))

        assert len(restored.trigger_history) == 1
        assert restored.trigger_history[0].reviewer_name == "gemini_github"

    def test_with_cost_tracker(self) -> None:
        cost = CostTracker(coder_invocations=1, reviewer_triggers=2, total_api_calls=10)
        state = _make_state(cost=cost)
        restored = RuntimeState.from_dict(_roundtrip_json(state.to_dict()))

        assert restored.cost.coder_invocations == 1
        assert restored.cost.reviewer_triggers == 2
        assert restored.cost.total_api_calls == 10

    def test_with_all_fields_populated(self) -> None:
        hf = HandledFinding(
            finding_id="f1",
            verdict="rejected",
            confidence="medium",
            reason="False positive",
            reply="Not a real issue",
            should_resolve=True,
            handled_at=LATER,
        )
        trigger = ReviewerTrigger(
            reviewer_name="gemini_github",
            round_index=1,
            timestamp=NOW,
            head_sha="abc123",
        )
        cost = CostTracker(
            coder_invocations=1,
            reviewer_triggers=1,
            total_api_calls=5,
            input_tokens=1000,
            output_tokens=500,
        )
        state = _make_state(
            base_sha="def456",
            round_index=1,
            status="handling",
            handled_findings={"f1": hf},
            trigger_history=[trigger],
            cost=cost,
            commits_made=["sha1", "sha2"],
            last_coder_round_index=1,
            last_error="some error",
            done_reason="completed",
        )
        restored = RuntimeState.from_dict(_roundtrip_json(state.to_dict()))

        assert restored.base_sha == "def456"
        assert restored.round_index == 1
        assert restored.status == "handling"
        assert restored.commits_made == ["sha1", "sha2"]
        assert restored.last_coder_round_index == 1
        assert restored.last_error == "some error"
        assert restored.done_reason == "completed"
        assert restored.cost.input_tokens == 1000


# --- Finding round-trip ---


class TestFindingRoundTrip:
    def test_minimal_fields(self) -> None:
        finding = Finding(id="f1", source="gemini", body="Fix this", created_at=NOW)
        restored = Finding.from_dict(_roundtrip_json(finding.to_dict()))

        assert restored.id == "f1"
        assert restored.source == "gemini"
        assert restored.body == "Fix this"
        assert restored.created_at == NOW

    def test_all_fields_populated(self) -> None:
        finding = Finding(
            id="f2",
            source="gemini_github",
            body="Potential null pointer",
            created_at=NOW,
            head_sha="abc123",
            thread_id="t1",
            comment_id="c1",
            path="src/main.py",
            line=42,
            severity="high",
            is_resolved=True,
            is_outdated=True,
            raw={"original": "data", "nested": {"key": "value"}},
        )
        restored = Finding.from_dict(_roundtrip_json(finding.to_dict()))

        assert restored.head_sha == "abc123"
        assert restored.thread_id == "t1"
        assert restored.comment_id == "c1"
        assert restored.path == "src/main.py"
        assert restored.line == 42
        assert restored.severity == "high"
        assert restored.is_resolved is True
        assert restored.is_outdated is True
        assert restored.raw == {"original": "data", "nested": {"key": "value"}}


# --- HandledFinding ---


class TestHandledFinding:
    def test_preserves_verdict_and_metadata(self) -> None:
        hf = HandledFinding(
            finding_id="f1",
            verdict="needs_human",
            confidence="low",
            reason="Complex logic change",
            reply="Needs manual review",
            should_resolve=False,
            changed_files=["a.py", "b.py"],
            handled_at=NOW,
        )
        restored = HandledFinding.from_dict(_roundtrip_json(hf.to_dict()))

        assert restored.verdict == "needs_human"
        assert restored.confidence == "low"
        assert restored.reason == "Complex logic change"
        assert restored.reply == "Needs manual review"
        assert restored.should_resolve is False
        assert restored.changed_files == ["a.py", "b.py"]
        assert restored.handled_at == NOW


# --- CostTracker ---


class TestCostTracker:
    def test_initializes_to_zeros(self) -> None:
        cost = CostTracker()

        assert cost.coder_invocations == 0
        assert cost.reviewer_triggers == 0
        assert cost.total_api_calls == 0
        assert cost.input_tokens == 0
        assert cost.output_tokens == 0

    def test_increments_correctly(self) -> None:
        cost = CostTracker()
        cost.coder_invocations += 1
        cost.reviewer_triggers += 2
        cost.total_api_calls += 5
        cost.input_tokens += 1000
        cost.output_tokens += 500

        assert cost.coder_invocations == 1
        assert cost.reviewer_triggers == 2
        assert cost.total_api_calls == 5
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500

    def test_allows_coder_invocations_at_limit(self) -> None:
        cost = CostTracker(coder_invocations=1)
        config = _fake_config(max_coder_invocations_per_run=1)

        assert cost.exceeds_limits(config) is False

    def test_exceeds_limits_above_coder_invocations(self) -> None:
        cost = CostTracker(coder_invocations=2)
        config = _fake_config(max_coder_invocations_per_run=1)

        assert cost.exceeds_limits(config) is True

    def test_allows_reviewer_triggers_at_limit(self) -> None:
        cost = CostTracker(reviewer_triggers=3)
        config = _fake_config(max_reviewer_triggers_per_run=3)

        assert cost.exceeds_limits(config) is False

    def test_exceeds_limits_above_reviewer_triggers(self) -> None:
        cost = CostTracker(reviewer_triggers=4)
        config = _fake_config(max_reviewer_triggers_per_run=3)

        assert cost.exceeds_limits(config) is True

    def test_exceeds_limits_on_token_count(self) -> None:
        cost = CostTracker(input_tokens=60000, output_tokens=40000)
        config = _fake_config(max_prompt_tokens=100000)

        assert cost.exceeds_limits(config) is True

    def test_within_limits(self) -> None:
        cost = CostTracker(coder_invocations=0, reviewer_triggers=1, input_tokens=100)
        config = _fake_config()

        assert cost.exceeds_limits(config) is False

    def test_round_trip(self) -> None:
        cost = CostTracker(
            coder_invocations=2,
            reviewer_triggers=3,
            total_api_calls=15,
            input_tokens=5000,
            output_tokens=2000,
        )
        restored = CostTracker.from_dict(_roundtrip_json(cost.to_dict()))

        assert restored == cost


# --- FixTask ---


class TestFixTask:
    def test_serializes_with_findings_and_diff(self) -> None:
        finding = Finding(id="f1", source="gemini", body="Issue here", created_at=NOW)
        task = FixTask(
            pr_number=42,
            head_sha="abc123",
            base_branch="main",
            findings=[finding],
            changed_files=["src/foo.py"],
            diff_text="--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new",
            output_file=".ai-orchestrator-result.json",
            repo_instructions="Follow PEP 8",
        )
        restored = FixTask.from_dict(_roundtrip_json(task.to_dict()))

        assert restored.pr_number == 42
        assert restored.head_sha == "abc123"
        assert len(restored.findings) == 1
        assert restored.findings[0].id == "f1"
        assert restored.diff_text.startswith("---")
        assert restored.repo_instructions == "Follow PEP 8"

    def test_optional_repo_instructions_none(self) -> None:
        task = FixTask(
            pr_number=1,
            head_sha="sha",
            base_branch="main",
            findings=[],
            changed_files=[],
            diff_text="",
            output_file="out.json",
        )
        restored = FixTask.from_dict(_roundtrip_json(task.to_dict()))

        assert restored.repo_instructions is None


# --- AgentRunResult ---


class TestAgentRunResult:
    def test_with_valid_decisions_round_trips(self) -> None:
        decision = Decision(
            finding_id="f1",
            verdict="accepted",
            confidence="high",
            reason="Valid bug",
            reply="Fixed the null check",
            should_resolve=True,
            thread_id="t1",
            changed_files=["src/main.py"],
        )
        test_result = TestResult(command="pytest tests/", result="passed", notes="All green")
        result = AgentRunResult(
            changed=True,
            summary="Fixed null pointer issue",
            decisions=[decision],
            needs_human=False,
            commit_message="fix: null check",
            tests=[test_result],
            token_usage=TokenUsage(input_tokens=5000, output_tokens=1000),
        )
        restored = AgentRunResult.from_dict(_roundtrip_json(result.to_dict()))

        assert restored.changed is True
        assert restored.summary == "Fixed null pointer issue"
        assert len(restored.decisions) == 1
        assert restored.decisions[0].verdict == "accepted"
        assert restored.decisions[0].thread_id == "t1"
        assert len(restored.tests) == 1
        assert restored.tests[0].result == "passed"
        assert restored.token_usage.input_tokens == 5000
        assert restored.commit_message == "fix: null check"


# --- PlannedAction ---


class TestPlannedAction:
    def test_preserves_type_and_payload(self) -> None:
        action = PlannedAction(
            type="reply_to_thread",
            payload={"thread_id": "t1", "body": "Fixed"},
        )
        restored = PlannedAction.from_dict(_roundtrip_json(action.to_dict()))

        assert restored.type == "reply_to_thread"
        assert restored.payload == {"thread_id": "t1", "body": "Fixed"}

    def test_empty_payload(self) -> None:
        action = PlannedAction(type="noop")
        restored = PlannedAction.from_dict(_roundtrip_json(action.to_dict()))

        assert restored.type == "noop"
        assert restored.payload == {}


# --- ReviewerTrigger ---


class TestReviewerTrigger:
    def test_records_all_fields(self) -> None:
        trigger = ReviewerTrigger(
            reviewer_name="gemini_github",
            round_index=1,
            timestamp=NOW,
            head_sha="abc123",
        )
        restored = ReviewerTrigger.from_dict(_roundtrip_json(trigger.to_dict()))

        assert restored.reviewer_name == "gemini_github"
        assert restored.round_index == 1
        assert restored.timestamp == NOW
        assert restored.head_sha == "abc123"


# --- Datetime serialization ---


class TestDatetimeSerialization:
    def test_datetime_serializes_as_iso_8601(self) -> None:
        finding = Finding(id="f1", source="test", body="body", created_at=NOW)
        data = finding.to_dict()

        assert data["created_at"] == "2026-05-25T12:00:00+00:00"

    def test_state_datetimes_are_iso_8601(self) -> None:
        state = _make_state()
        data = state.to_dict()

        assert data["created_at"] == "2026-05-25T12:00:00+00:00"
        assert data["updated_at"] == "2026-05-25T12:00:00+00:00"

    def test_trigger_timestamp_is_iso_8601(self) -> None:
        trigger = ReviewerTrigger(
            reviewer_name="test", round_index=0, timestamp=NOW, head_sha="sha"
        )
        data = trigger.to_dict()

        assert data["timestamp"] == "2026-05-25T12:00:00+00:00"

    def test_handled_finding_datetime_is_iso_8601(self) -> None:
        hf = HandledFinding(
            finding_id="f1",
            verdict="accepted",
            confidence="high",
            reason="r",
            reply="r",
            should_resolve=True,
            handled_at=NOW,
        )
        data = hf.to_dict()

        assert data["handled_at"] == "2026-05-25T12:00:00+00:00"


# --- Forward compatibility ---


class TestForwardCompatibility:
    def test_unknown_fields_ignored_in_finding(self) -> None:
        data = {
            "id": "f1",
            "source": "test",
            "body": "body",
            "created_at": NOW.isoformat(),
            "unknown_field": "should be ignored",
            "another_new_field": 42,
        }
        finding = Finding.from_dict(data)

        assert finding.id == "f1"
        assert not hasattr(finding, "unknown_field")

    def test_unknown_fields_ignored_in_state(self) -> None:
        state = _make_state()
        data = state.to_dict()
        data["future_feature"] = "new thing"
        restored = RuntimeState.from_dict(data)

        assert restored.version == state.version
        assert not hasattr(restored, "future_feature")

    def test_unknown_fields_ignored_in_cost_tracker(self) -> None:
        data = {"coder_invocations": 1, "new_counter": 99}
        cost = CostTracker.from_dict(data)

        assert cost.coder_invocations == 1

    def test_unknown_fields_ignored_in_planned_action(self) -> None:
        data = {"type": "noop", "payload": {}, "extra": True}
        action = PlannedAction.from_dict(data)

        assert action.type == "noop"

    def test_unknown_fields_ignored_in_decision(self) -> None:
        data = {
            "finding_id": "f1",
            "verdict": "accepted",
            "confidence": "high",
            "reason": "r",
            "reply": "r",
            "should_resolve": True,
            "new_field": "ignored",
        }
        decision = Decision.from_dict(data)

        assert decision.finding_id == "f1"


# --- Status validation ---


class TestStatusValidation:
    def test_rejects_invalid_status_on_construction(self) -> None:
        with pytest.raises(ModelError, match="Invalid status"):
            _make_state(status="invalid_status")

    def test_rejects_invalid_status_on_deserialization(self) -> None:
        state = _make_state()
        data = state.to_dict()
        data["status"] = "bogus"

        with pytest.raises(ModelError, match="Invalid status"):
            RuntimeState.from_dict(data)

    @pytest.mark.parametrize(
        "status",
        [
            "init",
            "triggering",
            "waiting",
            "collecting",
            "handling",
            "ci_wait",
            "done",
            "error",
            "needs_human",
        ],
    )
    def test_accepts_all_valid_statuses(self, status: str) -> None:
        state = _make_state(status=status)

        assert state.status == status


# --- GitHub snapshot types ---


class TestGitHubSnapshotTypes:
    def test_pull_request_round_trip(self) -> None:
        pr = PullRequest(
            number=42,
            head_sha="abc123",
            base_sha="def456",
            title="Fix bug",
            author_login="user",
            author_association="OWNER",
            labels=["ai-loop"],
            is_draft=False,
            is_fork=False,
            changed_files=["src/main.py"],
        )
        restored = PullRequest.from_dict(_roundtrip_json(pr.to_dict()))

        assert restored.number == 42
        assert restored.labels == ["ai-loop"]
        assert restored.changed_files == ["src/main.py"]

    def test_review_thread_round_trip(self) -> None:
        thread = ReviewThread(
            id="t1",
            is_resolved=False,
            is_outdated=True,
            comments=[{"author": "bot", "body": "Issue found"}],
        )
        restored = ReviewThread.from_dict(_roundtrip_json(thread.to_dict()))

        assert restored.id == "t1"
        assert restored.is_outdated is True
        assert len(restored.comments) == 1

    def test_check_run_round_trip(self) -> None:
        check = CheckRun(
            id="cr1",
            name="tests",
            status="completed",
            conclusion="success",
            head_sha="abc123",
        )
        restored = CheckRun.from_dict(_roundtrip_json(check.to_dict()))

        assert restored.id == "cr1"
        assert restored.conclusion == "success"


# --- Helpers ---


class _FakeSafety:
    def __init__(
        self,
        *,
        max_coder_invocations_per_run: int = 3,
        max_reviewer_triggers_per_run: int = 5,
        max_prompt_tokens: int = 100000,
    ):
        self.max_coder_invocations_per_run = max_coder_invocations_per_run
        self.max_reviewer_triggers_per_run = max_reviewer_triggers_per_run
        self.max_prompt_tokens = max_prompt_tokens


class _FakeConfig:
    def __init__(self, safety: _FakeSafety):
        self.safety = safety


def _fake_config(**kwargs: int) -> _FakeConfig:
    return _FakeConfig(safety=_FakeSafety(**kwargs))
