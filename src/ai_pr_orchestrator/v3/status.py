"""Read-only operator snapshots and deterministic next-step previews."""

from __future__ import annotations

from typing import Any

from ..logging import SecretRedactor, collect_secret_values
from .audit import workflow_snapshot
from .cao import (
    CaoControlPlaneError,
    CaoSessionController,
    CaoSessionNotFoundError,
    CaoTransportError,
    CaoUnavailableError,
)
from .domain import GitHubIssueRef, WorkflowState
from .queue import GitHubIssueQueue


def status_snapshot(
    queue: GitHubIssueQueue, controller: CaoSessionController, issue: GitHubIssueRef | None = None
) -> list[dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    issues = [issue] if issue is not None else queue.list_work_items()
    return [_work_status(queue, controller, item, cache) for item in issues]


def _work_status(queue, controller, issue, cache):
    state = queue.load_state(issue.slug())
    if state is None:
        return {"work_item_id": issue.slug(), "run_id": None, "phase": "queued", "lanes": []}
    work = workflow_snapshot(state)
    for lane in work["lanes"]:
        lane["runtime"] = _runtime(controller, state, lane, cache)
    return work


def _runtime(controller: CaoSessionController, state: WorkflowState, lane, cache):
    if "unavailable" in cache:
        return cache["unavailable"]
    try:
        observation = controller.inspect_session(lane["session_id"])
        metadata = observation.metadata
        actual = (metadata.context.run_id, metadata.context.work_item_id, metadata.lane.lane)
        if actual != (state.run_id, state.work_item_id, lane["lane"]):
            raise ValueError("CAO session attribution does not match durable work")
        return {
            "verified": True,
            "state": observation.state,
            "cao_status": observation.cao_status,
            "terminal_id": metadata.terminal_id,
            "profile": metadata.lane.profile_template,
            "detail": observation.detail,
        }
    except CaoSessionNotFoundError:
        return {"verified": True, "state": "disappeared"}
    except (CaoUnavailableError, CaoTransportError) as exc:
        cache["unavailable"] = _unverified(exc)
        return cache["unavailable"]
    except (CaoControlPlaneError, ValueError, KeyError, TypeError) as exc:
        return _unverified(exc)


def _unverified(exc: Exception) -> dict[str, Any]:
    return {
        "verified": False,
        "state": None,
        "reason": SecretRedactor(collect_secret_values(())).redact(str(exc)),
    }
