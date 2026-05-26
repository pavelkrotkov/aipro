"""Tests for RuntimeState storage in PR comments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from ai_pr_orchestrator.models import RuntimeState
from ai_pr_orchestrator.state_storage import (
    MAX_STATE_JSON_BYTES,
    STATE_COMMENT_MARKER,
    StateComment,
    StateConflictError,
    StateSizeError,
    find_state_comment,
    parse_state_comment,
    prepare_state_comment_update,
    serialize_state_comment,
)

NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 5, 25, 13, 0, 0, tzinfo=UTC)


def _make_state(**overrides: Any) -> RuntimeState:
    defaults: dict[str, Any] = {
        "version": 1,
        "pr_number": 42,
        "head_sha": "abc123",
        "status": "waiting",
        "round_index": 2,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return RuntimeState(**defaults)


def test_serialize_state_comment_includes_visible_header_and_hidden_json() -> None:
    body = serialize_state_comment(_make_state())

    assert body.startswith("### AI PR Orchestrator State")
    assert "Status: `waiting`" in body
    assert "Round: `2`" in body
    assert "Head: `abc123`" in body
    assert f"Updated: `{NOW.isoformat()}`" in body
    assert STATE_COMMENT_MARKER in body
    assert '"pr_number":42' in body


def test_parse_state_comment_extracts_runtime_state() -> None:
    state = _make_state(status="handling", head_sha="def456")
    parsed = parse_state_comment(serialize_state_comment(state))

    assert parsed == state


def test_parse_state_comment_scans_past_invalid_marker_to_valid_state() -> None:
    state = _make_state(status="handling", head_sha="def456")
    body = f"<!-- aipro-state\nnot-json\n-->\n\n{serialize_state_comment(state)}"

    assert parse_state_comment(body) == state


def test_parse_state_comment_ignores_marker_prefix_matches() -> None:
    state = _make_state(status="handling", head_sha="def456")
    body = f"<!-- aipro-state-v2\n{{}}\n-->\n\n{serialize_state_comment(state)}"

    assert parse_state_comment(body) == state


def test_state_comment_round_trips_through_markdown() -> None:
    state = _make_state(base_sha="base123", commits_made=["sha1", "sha2"])

    assert parse_state_comment(serialize_state_comment(state)) == state


def test_state_comment_round_trips_with_html_comment_closer_in_string_field() -> None:
    state = _make_state(last_error="error with --> HTML comment closing tag")
    serialized = serialize_state_comment(state)
    hidden_payload = serialized.split(STATE_COMMENT_MARKER, maxsplit=1)[1].rsplit(
        "-->", maxsplit=1
    )[0]

    assert "-->" not in hidden_payload
    assert "--\\u003e" in hidden_payload
    assert parse_state_comment(serialized) == state


def test_find_state_comment_scans_comment_bodies_for_marker() -> None:
    state = _make_state()
    comments = [
        {"id": 100, "body": "ordinary comment"},
        {"id": 101, "body": serialize_state_comment(state)},
    ]

    found = find_state_comment(comments)

    assert found == StateComment(comment_id=101, body=comments[1]["body"], state=state)


def test_find_state_comment_returns_last_valid_state_comment() -> None:
    older = _make_state(head_sha="older")
    newer = _make_state(head_sha="newer")
    comments = [
        {"id": 100, "body": serialize_state_comment(older)},
        {"id": 101, "body": "ordinary comment"},
        {"id": 102, "body": serialize_state_comment(newer)},
    ]

    found = find_state_comment(comments)

    assert found == StateComment(comment_id=102, body=comments[2]["body"], state=newer)


def test_find_state_comment_returns_last_valid_state_comment_from_generator() -> None:
    older = _make_state(head_sha="older")
    newer = _make_state(head_sha="newer")
    comments = (
        comment
        for comment in [
            {"id": 100, "body": serialize_state_comment(older)},
            {"id": 101, "body": "ordinary comment"},
            {"id": 102, "body": serialize_state_comment(newer)},
        ]
    )

    found = find_state_comment(comments)

    assert found is not None
    assert found.comment_id == 102
    assert found.state == newer


def test_find_state_comment_accepts_parser_supported_marker_spacing() -> None:
    state = _make_state()
    body = serialize_state_comment(state).replace("<!-- aipro-state", "<!--\n  aipro-state")

    found = find_state_comment([{"id": 101, "body": body}])

    assert found == StateComment(comment_id=101, body=body, state=state)


def test_find_state_comment_returns_none_when_no_state_comment_exists() -> None:
    assert find_state_comment([{"id": 100, "body": "ordinary comment"}]) is None


def test_parse_state_comment_returns_none_for_corrupt_json() -> None:
    body = f"header\n\n{STATE_COMMENT_MARKER}\n{{not-json}}\n-->"

    assert parse_state_comment(body) is None


def test_parse_state_comment_returns_none_for_corrupt_nested_state() -> None:
    payload = _make_state().to_dict()
    payload["handled_findings"] = {"bad": None}
    body = f"{STATE_COMMENT_MARKER}\n{json.dumps(payload)}\n-->"

    assert parse_state_comment(body) is None


def test_parse_state_comment_returns_none_when_marker_is_missing() -> None:
    assert parse_state_comment("ordinary comment") is None


def test_parse_state_comment_returns_none_when_closing_comment_marker_is_missing() -> None:
    body = f"header\n\n{STATE_COMMENT_MARKER}\n{' ' * 10_000}"

    assert parse_state_comment(body) is None


def test_size_guard_refuses_to_serialize_payload_over_50kb() -> None:
    state = _make_state(last_error="x" * MAX_STATE_JSON_BYTES)

    with pytest.raises(StateSizeError):
        serialize_state_comment(state)


def test_size_guard_error_message_reports_payload_size() -> None:
    state = _make_state(last_error="x" * MAX_STATE_JSON_BYTES)

    with pytest.raises(StateSizeError) as exc_info:
        serialize_state_comment(state)

    message = str(exc_info.value)
    assert "RuntimeState JSON payload is" in message
    assert "limit is 50000 bytes" in message
    assert exc_info.value.payload_size > MAX_STATE_JSON_BYTES


def test_prepare_update_detects_stale_updated_at() -> None:
    existing = StateComment(
        comment_id=101,
        body=serialize_state_comment(_make_state(updated_at=LATER)),
        state=_make_state(updated_at=LATER),
    )

    with pytest.raises(StateConflictError):
        prepare_state_comment_update(
            existing,
            _make_state(updated_at=LATER),
            expected_updated_at=NOW,
        )


def test_prepare_update_preserves_comment_id() -> None:
    existing = StateComment(
        comment_id=101,
        body=serialize_state_comment(_make_state()),
        state=_make_state(),
    )
    new_state = _make_state(status="done", updated_at=LATER)

    updated = prepare_state_comment_update(existing, new_state, expected_updated_at=NOW)

    assert updated.comment_id == 101
    assert updated.state == new_state
    assert parse_state_comment(updated.body) == new_state


def test_prepare_update_treats_naive_expected_updated_at_as_utc() -> None:
    existing = StateComment(
        comment_id=101,
        body=serialize_state_comment(_make_state()),
        state=_make_state(),
    )
    new_state = _make_state(status="done", updated_at=LATER)

    updated = prepare_state_comment_update(
        existing,
        new_state,
        expected_updated_at=NOW.replace(tzinfo=None),
    )

    assert updated.comment_id == 101
