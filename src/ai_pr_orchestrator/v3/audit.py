"""Bounded workflow projections shared by operator status and JSON audit events."""

from __future__ import annotations

from typing import Any

from .cao import session_name_for
from .domain import ArchivedFinding, FindingDisposition, ReviewerFinding, WorkflowState

_FINDING_FIELDS = (
    "id",
    "lane",
    "severity",
    "path",
    "line",
    "thread_id",
    "confidence",
    "head_sha",
    "status",
    "status_reason",
    "conflict_group_id",
    "round_id",
    "run_id",
)
_DISPOSITION_FIELDS = (
    "finding_id",
    "action",
    "decided_by",
    "thread_id",
    "run_id",
    "round_id",
    "response_to_round_id",
)
_WORK_FIELDS = ("work_item_id", "run_id", "phase", "round_id", "terminal_reason")
_EXTRA_FIELDS = (
    "branch",
    "head_sha",
    "pr_number",
    "ci_wait_started_at",
    "ci_failure_hashes",
    "ci_status",
)


def workflow_snapshot(state: WorkflowState) -> dict[str, Any]:
    return {
        **{key: getattr(state, key) for key in _WORK_FIELDS},
        **{key: state.extras.get(key) for key in _EXTRA_FIELDS},
        "updated_at": state.updated_at.isoformat(),
        "lease": {
            key: state.extras.get(key) for key in ("host_id", "claimed_at", "lease_expires_at")
        },
        "lanes": lane_snapshots(state),
        "usage": {
            "coder_invocations": state.extras.get("coder_usage", {}).get("invocations"),
            "cash_authorizations": state.extras.get("spend_authorizations"),
            "actual_cash_usd": None,
            "actual_tokens": None,
            "estimated_prompt_tokens": state.extras.get("prompt_tokens"),
        },
        "findings": [_finding(state, finding) for finding in state.findings],
        "archived": [_archived(state, finding) for finding in state.archived],
        "dispositions": [_disposition(decision) for decision in state.dispositions],
    }


def lane_snapshots(state: WorkflowState) -> list[dict[str, Any]]:
    rows = {}
    model = state.extras.get("developer_model")
    if model:
        turn = state.extras.get("coder_turn", {})
        receipt = {
            "assignment": model,
            "profile": state.extras.get("developer_profile"),
            "fallbacks": state.extras.get("developer_fallbacks", []),
            "status": "completed" if turn.get("result") else "planned",
            "exit_code": turn.get("result", {}).get("exit_code"),
            "routing": state.extras.get("routing_decisions", {}).get(model["lane"]),
        }
        rows[model["lane"]] = _lane(
            state, model["lane"], "worker", receipt, turn.get("context", {}).get("round_id")
        )
    for wave in state.extras.get("review_waves", {}).values():
        for name, receipt in wave["lanes"].items():
            rows[name] = _lane(state, name, "reviewer", receipt, wave["round_id"])
    return list(rows.values())


def _lane(state: WorkflowState, name: str, role: str, receipt: dict, round_id: str | None):
    scope = state.work_item_id if role == "worker" else None
    return {
        "lane": name,
        "role": role,
        "round_id": round_id,
        "session_id": session_name_for(state.run_id, name, scope),
        "assignment": receipt["assignment"],
        "profile": receipt.get("profile"),
        "exit_code": receipt.get("exit_code", receipt.get("result", {}).get("exit_code")),
        "fallbacks": receipt["fallbacks"],
        "routing": receipt.get("routing"),
        "last_durable_state": "stopped" if state.extras.get("lanes_stopped") else receipt["status"],
    }


def _origin(state: WorkflowState, lane: str, round_id: str | None) -> dict[str, Any]:
    wave = state.extras.get("review_waves", {}).get(round_id, {})
    assignment = wave.get("lanes", {}).get(lane, {}).get("assignment", {})
    return {
        "lane": lane,
        "role": "reviewer",
        "round_id": round_id,
        "head_sha": wave.get("head_sha"),
        "model_ref": assignment.get("model_ref"),
    }


def _finding(state: WorkflowState, finding: ReviewerFinding) -> dict[str, Any]:
    return {
        **{key: getattr(finding, key) for key in _FINDING_FIELDS},
        "claim": (finding.claim or finding.body)[:1000],
        "origin": _origin(state, finding.lane, finding.round_id),
    }


def _archived(state: WorkflowState, finding: ArchivedFinding) -> dict[str, Any]:
    return {
        "finding_id": finding.finding_id,
        "severity": finding.severity,
        "status": finding.status,
        "status_reason": finding.status_reason,
        "summary": finding.summary[:1000],
        "origins": [_origin(state, source.lane, source.round_id) for source in finding.sources],
    }


def _disposition(decision: FindingDisposition) -> dict[str, Any]:
    return {
        **{key: getattr(decision, key) for key in _DISPOSITION_FIELDS},
        "rationale": decision.rationale[:1000],
    }
