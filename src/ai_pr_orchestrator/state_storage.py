"""Store RuntimeState in hidden PR comment metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from ai_pr_orchestrator.models import ModelError, RuntimeState

MAX_STATE_JSON_BYTES = 50_000
STATE_COMMENT_MARKER = "<!-- aipro-state"

_STATE_START_RE = re.compile(r"<!--\s*aipro-state(?=\s|$)")


class StateStorageError(ValueError):
    """Raised when RuntimeState comment storage fails."""


class StateSizeError(StateStorageError):
    """Raised when a RuntimeState JSON payload is too large for comment storage."""

    def __init__(self, payload_size: int, max_size: int = MAX_STATE_JSON_BYTES) -> None:
        self.payload_size = payload_size
        self.max_size = max_size
        super().__init__(
            f"RuntimeState JSON payload is {payload_size} bytes; limit is {max_size} bytes"
        )


class StateConflictError(StateStorageError):
    """Raised when an update is based on stale RuntimeState data."""


@dataclass(frozen=True)
class StateComment:
    """A PR comment containing serialized RuntimeState."""

    comment_id: int | str
    body: str
    state: RuntimeState


def serialize_state_comment(
    state: RuntimeState, *, max_json_bytes: int = MAX_STATE_JSON_BYTES
) -> str:
    """Serialize RuntimeState as visible markdown plus a hidden JSON comment."""
    json_payload = _state_json(state)
    payload_size = len(json_payload.encode("utf-8"))
    if payload_size > max_json_bytes:
        raise StateSizeError(payload_size=payload_size, max_size=max_json_bytes)

    return "\n".join(
        [
            "### AI PR Orchestrator State",
            "",
            f"Status: `{state.status}`",
            f"Round: `{state.round_index}`",
            f"Head: `{state.head_sha}`",
            f"Updated: `{state.updated_at.isoformat()}`",
            "",
            STATE_COMMENT_MARKER,
            json_payload,
            "-->",
        ]
    )


def parse_state_comment(body: str) -> RuntimeState | None:
    """Extract RuntimeState from a PR comment body, returning None when absent or corrupt."""
    for match in _STATE_START_RE.finditer(body):
        end_index = body.find("-->", match.end())
        if end_index == -1:
            continue

        try:
            payload = json.loads(body[match.end() : end_index].strip())
            if not isinstance(payload, dict):
                continue
            return RuntimeState.from_dict(payload)
        except (
            json.JSONDecodeError,
            ModelError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
        ):
            continue
    return None


def find_state_comment(comments: Iterable[Mapping[str, Any]]) -> StateComment | None:
    """Find the last PR comment containing valid RuntimeState metadata."""
    if isinstance(comments, Sequence):
        comment_sequence = cast(Sequence[Mapping[str, Any]], comments)
        return _find_state_comment_from_latest(reversed(comment_sequence))

    latest: StateComment | None = None
    for comment in comments:
        state_comment = _state_comment_from_comment(comment)
        if state_comment is not None:
            latest = state_comment
    return latest


def _find_state_comment_from_latest(comments: Iterable[Mapping[str, Any]]) -> StateComment | None:
    for comment in comments:
        state_comment = _state_comment_from_comment(comment)
        if state_comment is not None:
            return state_comment
    return None


def _state_comment_from_comment(comment: Mapping[str, Any]) -> StateComment | None:
    body = comment.get("body")
    if not isinstance(body, str) or "aipro-state" not in body:
        return None
    state = parse_state_comment(body)
    if state is None or "id" not in comment:
        return None
    comment_id = comment["id"]
    if not isinstance(comment_id, (int, str)):
        return None
    return StateComment(comment_id=comment_id, body=body, state=state)


def prepare_state_comment_update(
    existing: StateComment,
    new_state: RuntimeState,
    *,
    expected_updated_at: datetime,
) -> StateComment:
    """Prepare an edit for an existing state comment after an optimistic lock check."""
    if _lock_timestamp(existing.state.updated_at) != _lock_timestamp(expected_updated_at):
        raise StateConflictError(
            "RuntimeState comment was updated by another process: "
            f"expected updated_at {expected_updated_at.isoformat()}, "
            f"found {existing.state.updated_at.isoformat()}"
        )
    return StateComment(
        comment_id=existing.comment_id,
        body=serialize_state_comment(new_state),
        state=new_state,
    )


def _state_json(state: RuntimeState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":")).replace(
        "-->", "--\\u003e"
    )


def _lock_timestamp(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        # RuntimeState deserialization treats naive timestamps as UTC; keep the lock input
        # contract aligned for callers that pass a previously parsed updated_at value.
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
