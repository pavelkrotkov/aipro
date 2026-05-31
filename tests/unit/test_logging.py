"""Tests for structured JSON logging and secret redaction."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterable
from typing import Any, cast

import pytest

from ai_pr_orchestrator.logging import (
    REDACTION_PLACEHOLDER,
    SecretRedactingFilter,
    SecretRedactor,
    collect_secret_values,
    log_action,
    log_error,
    log_state_transition,
    setup_logging,
)

# Each test gets its own logger name so setup_logging (which clears handlers on
# the named logger) never clobbers another test's configuration or the package
# logger used elsewhere in the suite.


def _configure(
    name: str,
    *,
    level: int | str = logging.INFO,
    secrets: Iterable[str] = (),
) -> tuple[io.StringIO, logging.Logger, SecretRedactor]:
    stream = io.StringIO()
    redactor = setup_logging(stream=stream, logger_name=name, level=level, secrets=secrets)
    return stream, logging.getLogger(name), redactor


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ---- JSON structure ----


def test_every_log_line_is_valid_json() -> None:
    stream, logger, _ = _configure("aipro_test_json")
    logger.info("plain message")
    logger.warning("another %s line", "formatted")
    log_action(logger, pr=1, action_type="add_label")

    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)  # raises if any line is not valid JSON
        assert parsed["message"]
        assert parsed["level"]
        assert parsed["logger"] == "aipro_test_json"


def test_state_transition_log_fields() -> None:
    stream, logger, _ = _configure("aipro_test_transition")
    log_state_transition(
        logger, pr=42, from_status="waiting", to_status="handling", head_sha="abc123"
    )
    (record,) = _records(stream)
    assert record["event"] == "state_transition"
    assert record["pr"] == 42
    assert record["from"] == "waiting"
    assert record["to"] == "handling"
    assert record["head_sha"] == "abc123"
    assert "ts" in record


def test_action_execution_log_fields() -> None:
    stream, logger, _ = _configure("aipro_test_action")
    log_action(logger, pr=7, action_type="push_branch")
    (record,) = _records(stream)
    assert record["event"] == "action"
    assert record["action_type"] == "push_branch"
    assert record["pr"] == 7


def test_error_log_fields_include_traceback() -> None:
    stream, logger, _ = _configure("aipro_test_error")
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        log_error(logger, error=exc, pr=3)
    (record,) = _records(stream)
    assert record["event"] == "error"
    assert "boom" in str(record["error"])
    assert record["pr"] == 3
    assert "Traceback" in str(record["traceback"])
    assert "RuntimeError: boom" in str(record["traceback"])


# ---- Secret redaction ----


def test_configured_secret_value_never_appears() -> None:
    # ``canary`` is a fixed sentinel test string, not a real credential; the
    # neutral name also keeps CodeQL's clear-text-logging heuristic from
    # flagging this deliberate redaction check.
    canary = "configured-canary-value"
    stream, logger, _ = _configure("aipro_test_secret_config", secrets=[canary])
    logger.info("connecting with %s now", canary)
    output = stream.getvalue()
    assert canary not in output
    assert REDACTION_PLACEHOLDER in output


def test_gh_token_value_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a non-token-shaped value so this exercises value-based redaction
    # (from collect_secret_values reading GH_TOKEN), not just pattern matching.
    monkeypatch.setenv("GH_TOKEN", "my-gh-token-sentinel-1234")
    secrets = collect_secret_values([])
    stream, logger, _ = _configure("aipro_test_gh_token", secrets=secrets)
    logger.info("Authorization: Bearer my-gh-token-sentinel-1234")
    output = stream.getvalue()
    assert "my-gh-token-sentinel-1234" not in output
    assert REDACTION_PLACEHOLDER in output


def test_redaction_works_for_multiline_output() -> None:
    canary = "leaked-canary-42"
    stream, logger, _ = _configure("aipro_test_multiline", secrets=[canary])
    logger.error("line one %s\nline two %s\nline three", canary, canary)
    record = _records(stream)[0]
    message = str(record["message"])
    assert "\n" in message  # the message really is multi-line
    assert canary not in message
    assert message.count(REDACTION_PLACEHOLDER) == 2


def test_secret_in_extra_field_is_redacted() -> None:
    canary = "canary-in-extra"
    stream, logger, _ = _configure("aipro_test_extra", secrets=[canary])
    logger.info("see field", extra={"detail": f"value={canary}"})
    output = stream.getvalue()
    assert canary not in output
    assert REDACTION_PLACEHOLDER in output


def test_secret_nested_in_extra_container_is_redacted() -> None:
    canary = "canary-nested-value"
    stream, logger, _ = _configure("aipro_test_extra_nested", secrets=[canary])
    logger.info(
        "see nested",
        extra={"detail": {"inner": canary}, "items": [f"x={canary}", "clean"]},
    )
    output = stream.getvalue()
    assert canary not in output
    assert REDACTION_PLACEHOLDER in output


def test_token_shaped_value_redacted_without_configuration() -> None:
    # A GitHub token shape is masked even if the operator never registered it,
    # and the token-kind prefix is preserved for debuggability.
    stream, logger, _ = _configure("aipro_test_token_pattern")
    logger.info("token ghp_abcdefghijklmnopqrstuvwxyz0123456789 leaked")
    output = stream.getvalue()
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in output
    assert f"ghp_{REDACTION_PLACEHOLDER}" in output


# ---- Configurable level ----


def test_log_level_is_configurable_with_int() -> None:
    stream, logger, _ = _configure("aipro_test_level_int", level=logging.WARNING)
    logger.info("should be filtered out")
    logger.warning("should appear")
    records = _records(stream)
    assert len(records) == 1
    assert records[0]["message"] == "should appear"


def test_log_level_is_configurable_with_string() -> None:
    stream, logger, _ = _configure("aipro_test_level_str", level="DEBUG")
    logger.debug("debug visible at DEBUG level")
    records = _records(stream)
    assert len(records) == 1
    assert records[0]["level"] == "DEBUG"


def test_unknown_level_name_falls_back_to_info() -> None:
    stream, logger, _ = _configure("aipro_test_level_bad", level="NOT_A_LEVEL")
    logger.info("info still emitted")
    logger.debug("debug suppressed")
    records = _records(stream)
    assert len(records) == 1
    assert records[0]["message"] == "info still emitted"


# ---- SecretRedactor / collect_secret_values units ----


def test_secret_redactor_matches_longest_first() -> None:
    redactor = SecretRedactor(["abcd", "abcdefgh"])
    # "abcdefgh" must be redacted whole, not leave "efgh" behind from masking
    # the shorter "abcd" first.
    assert redactor.redact("value=abcdefgh") == f"value={REDACTION_PLACEHOLDER}"


def test_secret_redactor_ignores_blank_secrets() -> None:
    redactor = SecretRedactor(["", "  "])
    assert redactor.redact("nothing to redact") == "nothing to redact"


def test_secret_redactor_ignores_too_short_secrets() -> None:
    # A 1-3 char "secret" must not be registered, or it would mass-redact common
    # substrings ("in", "or", ...) out of every line.
    redactor = SecretRedactor(["in", "abc"])
    assert redactor.redact("string containing abc") == "string containing abc"


def test_stringified_integer_level_is_honored() -> None:
    stream, logger, _ = _configure("aipro_test_level_strint", level="30")
    logger.info("info suppressed at WARNING")
    logger.warning("warning shown")
    records = _records(stream)
    assert len(records) == 1
    assert records[0]["message"] == "warning shown"


def test_collect_secret_values_includes_gh_token_and_skips_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GH_TOKEN", "tok-value")
    monkeypatch.setenv("MY_API_KEY", "key-value")
    monkeypatch.delenv("ABSENT_VAR", raising=False)
    values = collect_secret_values(["MY_API_KEY", "ABSENT_VAR"])
    assert "tok-value" in values
    assert "key-value" in values
    assert all(v for v in values)  # no empty entries for unset vars


# ---- Redaction robustness ----


class _LeakyObject:
    """A custom object whose ``str()`` exposes its value (only via default=str)."""

    def __init__(self, exposed: str) -> None:
        self._exposed = exposed

    def __str__(self) -> str:
        return f"<user value={self._exposed}>"


def test_secret_in_custom_object_str_is_redacted() -> None:
    # redact_recursive leaves the object as-is; the ``default`` conversion
    # redacts its __str__ before json escaping (and the final-string pass backs
    # that up).
    canary = "canary-in-custom-object"
    stream, logger, _ = _configure("aipro_test_custom_obj", secrets=[canary])
    logger.info("see obj", extra={"obj": _LeakyObject(canary)})
    output = stream.getvalue()
    assert canary not in output
    assert REDACTION_PLACEHOLDER in output


def test_secret_with_json_special_chars_in_custom_object_is_redacted() -> None:
    # A secret containing a quote and a newline: json.dumps escapes those
    # (`"` -> `\"`, newline -> `\n`), so a post-serialization-only pass would
    # miss the escaped form. Redacting inside ``default`` (pre-escape) closes it.
    canary = 'ZZ"can\naryZZ'
    stream, logger, _ = _configure("aipro_test_custom_obj_special", secrets=[canary])
    logger.info("see obj", extra={"obj": _LeakyObject(canary)})
    output = stream.getvalue()
    assert "ZZ" not in output  # neither the raw nor the JSON-escaped form leaks
    assert REDACTION_PLACEHOLDER in output


def test_circular_reference_in_extra_does_not_crash() -> None:
    stream, logger, _ = _configure("aipro_test_circular")
    cyclic: dict[str, object] = {"name": "node"}
    cyclic["self"] = cyclic  # circular reference
    logger.info("cyclic", extra={"detail": cyclic})
    # Must still emit exactly one valid JSON line (no RecursionError / no
    # json circular-reference crash); the cycle is replaced with a marker.
    record = _records(stream)[0]
    assert record["message"] == "cyclic"
    assert "<circular>" in json.dumps(record)


def test_shared_non_cyclic_reference_is_fully_redacted() -> None:
    # A container shared between two siblings (a DAG, not a cycle) must be
    # redacted in *both* places — the cycle guard only skips active ancestors.
    canary = "canary-shared-ref"
    shared = {"value": canary}
    redactor = SecretRedactor([canary])
    result = redactor.redact_recursive({"a": shared, "b": shared})
    assert result == {"a": {"value": REDACTION_PLACEHOLDER}, "b": {"value": REDACTION_PLACEHOLDER}}


def test_filter_survives_malformed_format_args() -> None:
    # A filter that raises is NOT caught by the logging machinery and would
    # crash the caller. getMessage() on a record with too few %-args raises;
    # the filter must swallow that and still return True.
    filt = SecretRedactingFilter(SecretRedactor())
    record = logging.LogRecord(
        name="n",
        level=logging.INFO,
        pathname="p",
        lineno=1,
        msg="needs %s and %s",
        args=("one",),
        exc_info=None,
    )
    assert filt.filter(record) is True


def test_malformed_format_args_are_redacted() -> None:
    # getMessage() fails (mismatched %-args), so args survive; ``args`` is
    # reserved, so the filter must redact them in the except branch or a secret
    # in an arg would leak via the handler's error path.
    canary = "canary-in-bad-args"
    filt = SecretRedactingFilter(SecretRedactor([canary]))
    record = logging.LogRecord(
        name="n",
        level=logging.INFO,
        pathname="p",
        lineno=1,
        msg="needs %s and %s",
        args=(canary,),  # one arg, two placeholders -> getMessage raises
        exc_info=None,
    )
    assert filt.filter(record) is True
    assert record.args == (REDACTION_PLACEHOLDER,)


def test_malformed_log_call_end_to_end_redacts_and_stays_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # End-to-end through setup_logging: a mismatched-args call must still emit a
    # single valid JSON line on the handler stream (no drop to handleError's
    # non-JSON stderr fallback) with the canary redacted.
    canary = "canary-e2e-bad-args"
    stream, logger, _ = _configure("aipro_test_bad_e2e", secrets=[canary])
    logger.info("token %s %s", canary)  # two placeholders, one arg
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    json.loads(lines[0])  # valid JSON, formatter never raised
    assert canary not in stream.getvalue()


def test_exc_info_without_active_exception_emits_no_traceback() -> None:
    # exc_info=True with no active exception becomes (None, None, None): truthy
    # but must not produce a bogus "NoneType: None" traceback field.
    stream, logger, _ = _configure("aipro_test_empty_excinfo")
    logger.info("no real exception", exc_info=True)
    (record,) = _records(stream)
    assert "traceback" not in record
    assert "NoneType" not in stream.getvalue()


def test_token_pattern_redacted_before_overlapping_literal_secret() -> None:
    # A literal secret that overlaps a token body must not break token masking
    # and leak the token's tail: the regex runs first.
    redactor = SecretRedactor(["234567"])
    out = redactor.redact("auth ghp_1234567890abcdefghij done")
    assert "7890abcdefghij" not in out
    assert "890abcdefghij" not in out
    assert f"ghp_{REDACTION_PLACEHOLDER}" in out


def test_redact_recursive_set_returns_list() -> None:
    # set -> list so json.dumps emits a native array, not a Python set repr.
    redactor = SecretRedactor(["canary-set-value"])
    result = redactor.redact_recursive({"canary-set-value", "clean"})
    assert isinstance(result, list)
    assert "canary-set-value" not in result
    assert REDACTION_PLACEHOLDER in result


def test_set_in_extra_serializes_as_json_array() -> None:
    canary = "canary-in-a-set"
    stream, logger, _ = _configure("aipro_test_set_extra", secrets=[canary])
    logger.info("set field", extra={"items": {canary, "clean"}})
    record = _records(stream)[0]
    assert isinstance(record["items"], list)  # JSON array, not a set-repr string
    assert canary not in stream.getvalue()


def test_coerce_level_handles_non_str_int_gracefully() -> None:
    from ai_pr_orchestrator.logging import _coerce_level

    # None (or any non-str/int) must fall back to INFO, not raise.
    assert _coerce_level(cast(Any, None)) == logging.INFO
    assert _coerce_level(logging.WARNING) == logging.WARNING
    assert _coerce_level("30") == 30


def test_mixed_key_types_in_extra_do_not_crash() -> None:
    # Without sort_keys, a dict with mixed int/str keys serializes fine instead
    # of raising TypeError.
    stream, logger, _ = _configure("aipro_test_mixed_keys")
    logger.info("mixed", extra={"detail": {1: "a", "b": 2}})
    record = _records(stream)[0]
    assert record["message"] == "mixed"


# ---- Dry-run tagging ----


def test_dry_run_flag_on_action_and_transition() -> None:
    stream, logger, _ = _configure("aipro_test_dry_flag")
    log_action(logger, pr=5, action_type="add_label", dry_run=True)
    log_state_transition(
        logger, pr=5, from_status="init", to_status="triggering", head_sha="s", dry_run=True
    )
    action_record, transition_record = _records(stream)
    assert action_record["dry_run"] is True
    assert transition_record["dry_run"] is True


def test_no_dry_run_flag_by_default() -> None:
    stream, logger, _ = _configure("aipro_test_no_dry_flag")
    log_action(logger, pr=5, action_type="add_label")
    (record,) = _records(stream)
    assert "dry_run" not in record
