"""Validate coder JSON output against the agent contract."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from ai_pr_orchestrator.models import AgentRunResult, Finding

VALID_VERDICTS = {"accepted", "rejected", "needs_human"}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_TEST_RESULTS = {"passed", "failed", "not_run"}
MISSING = object()


class OutputValidationError(ValueError):
    """Raised when coder output violates the required schema."""


def validate_agent_output(output: str, findings: list[Finding]) -> AgentRunResult:
    """Parse and validate coder output JSON."""

    cleaned = _strip_markdown_code_block(output)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"Agent output must be valid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise OutputValidationError("Agent output must be a JSON object")

    _validate_top_level(raw)
    decisions = raw["decisions"]
    _validate_decisions(decisions, findings)
    _validate_tests(raw.get("tests", []))
    _validate_token_usage(raw.get("token_usage", {}))

    try:
        return AgentRunResult.from_dict(raw)
    except TypeError as exc:
        raise OutputValidationError(f"Agent output does not match schema: {exc}") from exc


def _validate_top_level(raw: dict[str, Any]) -> None:
    required = {"changed", "summary", "needs_human", "decisions"}
    missing = sorted(required - raw.keys())
    if missing:
        raise OutputValidationError(f"Agent output missing required fields: {', '.join(missing)}")

    if not isinstance(raw["changed"], bool):
        raise OutputValidationError("changed must be a boolean")
    if not isinstance(raw["summary"], str) or not raw["summary"].strip():
        raise OutputValidationError("summary must be a nonempty string")
    if not isinstance(raw["needs_human"], bool):
        raise OutputValidationError("needs_human must be a boolean")
    if not isinstance(raw["decisions"], list):
        raise OutputValidationError("decisions must be an array")

    commit_message = raw.get("commit_message")
    if commit_message is not None and not isinstance(commit_message, str):
        raise OutputValidationError("commit_message must be a string or null")


def _validate_decisions(decisions: list[Any], findings: list[Finding]) -> None:
    input_ids = {finding.id for finding in findings}
    decision_ids: list[str] = []

    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise OutputValidationError(f"decisions[{index}] must be an object")
        _validate_decision(index, decision)
        decision_ids.append(decision["finding_id"])

    seen_ids: set[str] = set()
    duplicate_ids_set: set[str] = set()
    for finding_id in decision_ids:
        if finding_id in seen_ids:
            duplicate_ids_set.add(finding_id)
        else:
            seen_ids.add(finding_id)
    duplicate_ids = sorted(duplicate_ids_set)
    if duplicate_ids:
        raise OutputValidationError(
            "Agent output has duplicate decisions for finding IDs: " + ", ".join(duplicate_ids)
        )

    decision_id_set = set(decision_ids)
    unknown_ids = sorted(decision_id_set - input_ids)
    if unknown_ids:
        raise OutputValidationError(
            "Agent output has unknown finding IDs: " + ", ".join(unknown_ids)
        )

    missing_ids = sorted(input_ids - decision_id_set)
    if missing_ids:
        raise OutputValidationError(
            "Agent output missing decisions for finding IDs: " + ", ".join(missing_ids)
        )


def _validate_decision(index: int, decision: dict[str, Any]) -> None:
    required = {
        "finding_id",
        "verdict",
        "confidence",
        "reason",
        "reply",
        "should_resolve",
        "changed_files",
    }
    missing = sorted(required - decision.keys())
    if missing:
        raise OutputValidationError(
            f"decisions[{index}] missing required fields: {', '.join(missing)}"
        )

    if not isinstance(decision["finding_id"], str) or not decision["finding_id"].strip():
        raise OutputValidationError(f"decisions[{index}].finding_id must be a nonempty string")
    if decision["verdict"] not in VALID_VERDICTS:
        raise OutputValidationError(
            f"decisions[{index}].verdict must be one of {sorted(VALID_VERDICTS)}"
        )
    if decision["confidence"] not in VALID_CONFIDENCE:
        raise OutputValidationError(
            f"decisions[{index}].confidence must be one of {sorted(VALID_CONFIDENCE)}"
        )
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise OutputValidationError(f"decisions[{index}].reason must be a nonempty string")
    if not isinstance(decision["reply"], str) or not decision["reply"].strip():
        raise OutputValidationError(f"decisions[{index}].reply must be a nonempty string")
    if not isinstance(decision["should_resolve"], bool):
        raise OutputValidationError(f"decisions[{index}].should_resolve must be a boolean")

    thread_id = decision.get("thread_id")
    if thread_id is not None and not isinstance(thread_id, str):
        raise OutputValidationError(f"decisions[{index}].thread_id must be a string or null")

    changed_files = decision["changed_files"]
    if not isinstance(changed_files, list) or not all(
        isinstance(path, str) for path in changed_files
    ):
        raise OutputValidationError(f"decisions[{index}].changed_files must be an array of strings")


def _validate_tests(tests: Any) -> None:
    if tests is None:
        return
    if not isinstance(tests, list):
        raise OutputValidationError("tests must be an array")
    for index, test in enumerate(tests):
        if not isinstance(test, dict):
            raise OutputValidationError(f"tests[{index}] must be an object")
        test_data = cast(dict[str, Any], test)
        command = test_data.get("command")
        result = test_data.get("result")
        if test_data.get("notes") is None:
            test_data["notes"] = ""
        notes = test_data["notes"]
        if not isinstance(command, str) or not command.strip():
            raise OutputValidationError(f"tests[{index}].command must be a nonempty string")
        if result not in VALID_TEST_RESULTS:
            raise OutputValidationError(
                f"tests[{index}].result must be one of {sorted(VALID_TEST_RESULTS)}"
            )
        if not isinstance(notes, str):
            raise OutputValidationError(f"tests[{index}].notes must be a string if provided")


def _validate_token_usage(token_usage: Any) -> None:
    if token_usage is None:
        return
    if not isinstance(token_usage, dict):
        raise OutputValidationError("token_usage must be an object")
    for field in ("input_tokens", "output_tokens"):
        if field in token_usage:
            value = token_usage[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise OutputValidationError(f"token_usage.{field} must be a non-negative integer")


def _strip_markdown_code_block(output: str) -> str:
    cleaned = output.strip()
    match = re.search(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned
