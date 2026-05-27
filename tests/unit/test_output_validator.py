"""Tests for coder output validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ai_pr_orchestrator.agents.output_validator import OutputValidationError, validate_agent_output
from ai_pr_orchestrator.models import Finding

NOW = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _finding(id_: str) -> Finding:
    return Finding(id=id_, source="gemini_github", body=f"body {id_}", created_at=NOW)


def _output(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "changed": False,
        "summary": "No code changes needed.",
        "needs_human": False,
        "decisions": [
            {
                "finding_id": "f1",
                "thread_id": "thread-1",
                "verdict": "rejected",
                "confidence": "high",
                "reason": "The existing implementation already handles this.",
                "reply": "This is already covered by the current guard.",
                "should_resolve": True,
                "changed_files": [],
            }
        ],
        "tests": [
            {
                "command": "uv run pytest",
                "result": "passed",
                "notes": "",
            }
        ],
        "token_usage": {"input_tokens": 12, "output_tokens": 34},
    }
    data.update(overrides)
    return data


def _validate(data: dict[str, object], findings: list[Finding] | None = None):
    return validate_agent_output(json.dumps(data), findings or [_finding("f1")])


def test_valid_json_with_required_fields_passes() -> None:
    result = _validate(_output())

    assert result.changed is False
    assert result.decisions is not None
    assert result.decisions[0].finding_id == "f1"


def test_invalid_json_returns_validation_error() -> None:
    with pytest.raises(OutputValidationError, match="valid JSON"):
        validate_agent_output("{not json", [_finding("f1")])


def test_missing_decisions_field_returns_validation_error() -> None:
    data = _output()
    data.pop("decisions")

    with pytest.raises(OutputValidationError, match="decisions"):
        _validate(data)


def test_missing_changed_field_returns_validation_error() -> None:
    data = _output()
    data.pop("changed")

    with pytest.raises(OutputValidationError, match="changed"):
        _validate(data)


def test_input_finding_without_corresponding_decision_returns_validation_error() -> None:
    with pytest.raises(OutputValidationError, match=r"missing decisions.*f2"):
        _validate(_output(), [_finding("f1"), _finding("f2")])


def test_decision_with_unknown_finding_id_returns_validation_error() -> None:
    data = _output(
        decisions=[
            {
                "finding_id": "unknown",
                "thread_id": None,
                "verdict": "rejected",
                "confidence": "high",
                "reason": "Reason",
                "reply": "Reply",
                "should_resolve": True,
                "changed_files": [],
            }
        ]
    )

    with pytest.raises(OutputValidationError, match=r"unknown finding IDs.*unknown"):
        _validate(data)


def test_invalid_verdict_returns_validation_error() -> None:
    data = _output(
        decisions=[
            {
                "finding_id": "f1",
                "thread_id": None,
                "verdict": "maybe",
                "confidence": "high",
                "reason": "Reason",
                "reply": "Reply",
                "should_resolve": True,
                "changed_files": [],
            }
        ]
    )

    with pytest.raises(OutputValidationError, match="verdict"):
        _validate(data)


def test_decision_with_empty_reason_returns_validation_error() -> None:
    data = _output(
        decisions=[
            {
                "finding_id": "f1",
                "thread_id": None,
                "verdict": "rejected",
                "confidence": "high",
                "reason": "",
                "reply": "Reply",
                "should_resolve": True,
                "changed_files": [],
            }
        ]
    )

    with pytest.raises(OutputValidationError, match="reason"):
        _validate(data)


def test_decision_with_empty_reply_returns_validation_error() -> None:
    data = _output(
        decisions=[
            {
                "finding_id": "f1",
                "thread_id": None,
                "verdict": "rejected",
                "confidence": "high",
                "reason": "Reason",
                "reply": "",
                "should_resolve": True,
                "changed_files": [],
            }
        ]
    )

    with pytest.raises(OutputValidationError, match="reply"):
        _validate(data)


def test_changed_true_is_accepted() -> None:
    result = _validate(_output(changed=True))

    assert result.changed is True


def test_needs_human_true_is_propagated() -> None:
    result = _validate(_output(needs_human=True))

    assert result.needs_human is True


def test_token_usage_fields_are_optional_and_default_to_zero() -> None:
    data = _output()
    data.pop("token_usage")

    result = _validate(data)

    assert result.token_usage is not None
    assert result.token_usage.input_tokens == 0
    assert result.token_usage.output_tokens == 0


def test_tests_array_is_optional() -> None:
    data = _output()
    data.pop("tests")

    result = _validate(data)

    assert result.tests == []


def test_extra_unknown_fields_are_ignored() -> None:
    result = _validate(_output(extra_future_field={"ok": True}))

    assert not hasattr(result, "extra_future_field")
