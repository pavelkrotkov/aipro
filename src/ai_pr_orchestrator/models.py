"""Core domain models and JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Literal, get_args

Status = Literal[
    "init",
    "triggering",
    "waiting",
    "collecting",
    "handling",
    "ci_wait",
    "done",
    "error",
    "needs_human",
]

VALID_STATUSES: frozenset[str] = frozenset(get_args(Status))

ActionType = Literal[
    "post_pr_comment",
    "update_status_comment",
    "invoke_coder",
    "reply_to_thread",
    "resolve_thread",
    "commit_changes",
    "push_branch",
    "rollback_changes",
    "add_label",
    "remove_label",
    "post_final_summary",
    "noop",
]

Verdict = Literal["accepted", "rejected", "needs_human"]
Confidence = Literal["low", "medium", "high"]


class ModelError(ValueError):
    """Raised when model validation fails."""


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return _dt_to_str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


@dataclass
class Finding:
    id: str
    source: str
    body: str
    created_at: datetime
    head_sha: str | None = None
    thread_id: str | None = None
    comment_id: str | None = None
    path: str | None = None
    line: int | None = None
    severity: str | None = None
    is_resolved: bool = False
    is_outdated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "created_at" in filtered and isinstance(filtered["created_at"], str):
            filtered["created_at"] = _str_to_dt(filtered["created_at"])
        return cls(**filtered)


@dataclass
class HandledFinding:
    finding_id: str
    verdict: Verdict
    confidence: Confidence
    reason: str
    reply: str
    should_resolve: bool
    changed_files: list[str] = field(default_factory=list)
    handled_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandledFinding:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "handled_at" in filtered and isinstance(filtered["handled_at"], str):
            filtered["handled_at"] = _str_to_dt(filtered["handled_at"])
        return cls(**filtered)


@dataclass
class CostTracker:
    coder_invocations: int = 0
    reviewer_triggers: int = 0
    total_api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def exceeds_limits(self, config: Any) -> bool:
        safety = config.safety
        return (
            self.coder_invocations >= safety.max_coder_invocations_per_run
            or self.reviewer_triggers >= safety.max_reviewer_triggers_per_run
            or self.input_tokens + self.output_tokens >= safety.max_prompt_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostTracker:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ReviewerTrigger:
    reviewer_name: str
    round_index: int
    timestamp: datetime
    head_sha: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewerTrigger:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "timestamp" in filtered and isinstance(filtered["timestamp"], str):
            filtered["timestamp"] = _str_to_dt(filtered["timestamp"])
        return cls(**filtered)


def _validate_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise ModelError(f"Invalid status {status!r}, must be one of {sorted(VALID_STATUSES)}")


@dataclass
class RuntimeState:
    version: int
    pr_number: int
    head_sha: str
    status: Status
    round_index: int = 0
    base_sha: str | None = None
    handled_findings: dict[str, HandledFinding] = field(default_factory=dict)
    trigger_history: list[ReviewerTrigger] = field(default_factory=list)
    cost: CostTracker = field(default_factory=CostTracker)
    commits_made: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_error: str | None = None
    done_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_status(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeState:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "status" in filtered:
            _validate_status(filtered["status"])
        for dt_field in ("created_at", "updated_at"):
            if dt_field in filtered and isinstance(filtered[dt_field], str):
                filtered[dt_field] = _str_to_dt(filtered[dt_field])
        if "handled_findings" in filtered and isinstance(filtered["handled_findings"], dict):
            filtered["handled_findings"] = {
                k: HandledFinding.from_dict(v) for k, v in filtered["handled_findings"].items()
            }
        if "trigger_history" in filtered and isinstance(filtered["trigger_history"], list):
            filtered["trigger_history"] = [
                ReviewerTrigger.from_dict(t) for t in filtered["trigger_history"]
            ]
        if "cost" in filtered and isinstance(filtered["cost"], dict):
            filtered["cost"] = CostTracker.from_dict(filtered["cost"])
        return cls(**filtered)


@dataclass
class Decision:
    finding_id: str
    verdict: Verdict
    confidence: Confidence
    reason: str
    reply: str
    should_resolve: bool
    thread_id: str | None = None
    changed_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class TestResult:
    __test__ = False

    command: str
    result: Literal["passed", "failed", "not_run"]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class AgentRunResult:
    changed: bool
    summary: str
    decisions: list[Decision]
    needs_human: bool = False
    commit_message: str | None = None
    tests: list[TestResult] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRunResult:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "decisions" in filtered and isinstance(filtered["decisions"], list):
            filtered["decisions"] = [Decision.from_dict(d) for d in filtered["decisions"]]
        if "tests" in filtered and isinstance(filtered["tests"], list):
            filtered["tests"] = [TestResult.from_dict(t) for t in filtered["tests"]]
        if "token_usage" in filtered and isinstance(filtered["token_usage"], dict):
            filtered["token_usage"] = TokenUsage.from_dict(filtered["token_usage"])
        return cls(**filtered)


@dataclass
class FixTask:
    pr_number: int
    head_sha: str
    base_branch: str
    findings: list[Finding]
    changed_files: list[str]
    diff_text: str
    output_file: str
    repo_instructions: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixTask:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "findings" in filtered and isinstance(filtered["findings"], list):
            filtered["findings"] = [Finding.from_dict(f) for f in filtered["findings"]]
        return cls(**filtered)


@dataclass
class PlannedAction:
    type: ActionType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": _serialize(self.payload)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedAction:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class PullRequest:
    number: int
    head_sha: str
    base_sha: str
    title: str
    author_login: str
    author_association: str
    labels: list[str] = field(default_factory=list)
    is_draft: bool = False
    is_fork: bool = False
    changed_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PullRequest:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ReviewThread:
    id: str
    is_resolved: bool
    is_outdated: bool
    comments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewThread:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class CheckRun:
    id: str
    name: str
    status: str
    conclusion: str | None = None
    head_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckRun:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
