"""V3 domain model.

Pure, provider-independent value types for the thin policy engine. Everything
in this module is plain data: no I/O, no subprocesses, no GitHub/CAO/Hermes
imports. Types that may be persisted carry ``to_dict``/``from_dict`` round
trips; unknown fields in persisted payloads are dropped on load, which keeps
older readers forward compatible with newer writers.

Identifiers (``RunId``, ``RoundId``, ``WorkItemId``, ``LaneName``,
``ModelRef``) are typed aliases rather than opaque strings, so signatures
document intent without tying any type to a vendor or model name. ``ModelRef``
names an entry in the model catalog (a policy concept), never a specific
vendor/model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Literal

# --- Identifiers -----------------------------------------------------------

RunId = str
RoundId = str
WorkItemId = str
LaneName = str
#: A key into the V3 model catalog. Must never encode a vendor/model name;
#: the catalog owns the mapping to concrete model descriptors.
ModelRef = str
HeadSha = str

# --- Workflow phases -------------------------------------------------------

WorkItemPhase = Literal[
    "queued",
    "claiming",
    "planning",
    "coding",
    "reviewing",
    "ci_gating",
    "updating_pr",
    "escalated",
    "done",
    "failed",
]

VALID_PHASES: frozenset[str] = frozenset(
    (
        "queued",
        "claiming",
        "planning",
        "coding",
        "reviewing",
        "ci_gating",
        "updating_pr",
        "escalated",
        "done",
        "failed",
    )
)

TERMINAL_PHASES: frozenset[str] = frozenset(("done", "failed", "escalated"))

#: Phases that require a terminal reason when entered.
PHASES_REQUIRING_REASON: frozenset[str] = TERMINAL_PHASES

AgentLaneRole = Literal["foreman", "worker", "reviewer"]
VALID_LANE_ROLES: frozenset[str] = frozenset(("foreman", "worker", "reviewer"))

Severity = Literal["info", "minor", "major", "blocker"]
VALID_SEVERITIES: frozenset[str] = frozenset(("info", "minor", "major", "blocker"))

DispositionAction = Literal[
    "fix",
    "reject_wont_fix",
    "reply_deferred",
    "already_addressed",
    "escalate_human",
]
VALID_DISPOSITION_ACTIONS: frozenset[str] = frozenset(
    ("fix", "reject_wont_fix", "reply_deferred", "already_addressed", "escalate_human")
)

FailureKind = Literal[
    "coder_failure", "reviewer_failure", "ci_failure", "stagnation", "policy_violation", "unknown"
]
VALID_FAILURE_KINDS: frozenset[str] = frozenset(
    ("coder_failure", "reviewer_failure", "ci_failure", "stagnation", "policy_violation", "unknown")
)


class DomainError(ValueError):
    """Raised when a V3 domain object is constructed in an invalid state."""


# --- GitHub identity -------------------------------------------------------


@dataclass(frozen=True)
class GitHubIssueRef:
    """Authoritative GitHub identity of a work item (its origin issue)."""

    owner: str
    repo: str
    number: int

    def __post_init__(self) -> None:
        if not self.owner or not self.repo:
            raise DomainError("GitHubIssueRef requires non-empty owner and repo")
        if self.number <= 0:
            raise DomainError(f"GitHubIssueRef.number must be positive, got {self.number}")

    def slug(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"


# --- Work item -------------------------------------------------------------


@dataclass
class WorkItem:
    """A unit of autonomous coding work, identified by a GitHub issue."""

    id: WorkItemId
    issue: GitHubIssueRef
    title: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkItem:
        data = dict(data)
        if isinstance(data.get("issue"), dict):
            data["issue"] = GitHubIssueRef(**data["issue"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = _str_to_dt(data["created_at"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --- Workflow state --------------------------------------------------------


@dataclass
class WorkflowState:
    """Workflow phase/state of one work item in one run."""

    work_item_id: WorkItemId
    run_id: RunId
    phase: WorkItemPhase
    round_id: RoundId | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in VALID_PHASES:
            raise DomainError(
                f"Invalid phase {self.phase!r}, must be one of {sorted(VALID_PHASES)}"
            )
        if self.phase in PHASES_REQUIRING_REASON and not self.terminal_reason:
            raise DomainError(f"Phase {self.phase!r} requires a terminal_reason")
        if self.phase not in PHASES_REQUIRING_REASON and self.terminal_reason is not None:
            raise DomainError(
                f"terminal_reason is only valid in terminal phases, phase is {self.phase!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        data = dict(data)
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = _str_to_dt(data["updated_at"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --- Agent lanes -----------------------------------------------------------


@dataclass(frozen=True)
class LaneIdentity:
    """Identity of one agent lane: a name, its role, and the Hermes profile
    template it instantiates. The profile template is a policy-level name;
    how Hermes materializes it is outside V3's scope."""

    lane: LaneName
    role: AgentLaneRole
    profile_template: str

    def __post_init__(self) -> None:
        if not self.lane:
            raise DomainError("LaneIdentity.lane must be non-empty")
        if self.role not in VALID_LANE_ROLES:
            raise DomainError(
                f"Invalid lane role {self.role!r}, must be one of {sorted(VALID_LANE_ROLES)}"
            )
        if not self.profile_template:
            raise DomainError("LaneIdentity.profile_template must be non-empty")


@dataclass(frozen=True)
class ModelAssignment:
    """Model assigned to a lane, expressed as a catalog reference.

    The ref resolves through the model catalog/router configuration; no
    vendor or model name may appear here.
    """

    lane: LaneName
    model_ref: ModelRef

    def __post_init__(self) -> None:
        if not self.lane:
            raise DomainError("ModelAssignment.lane must be non-empty")
        if not self.model_ref:
            raise DomainError("ModelAssignment.model_ref must be non-empty")


# --- Reviewer findings -----------------------------------------------------


@dataclass
class ReviewerFinding:
    """A single finding produced by a reviewer lane."""

    id: str
    lane: LaneName
    body: str
    severity: Severity
    run_id: RunId
    round_id: RoundId
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    path: str | None = None
    line: int | None = None

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise DomainError(
                f"Invalid severity {self.severity!r}, must be one of {sorted(VALID_SEVERITIES)}"
            )
        if not self.body:
            raise DomainError("ReviewerFinding.body must be non-empty")
        if self.line is not None and self.line <= 0:
            raise DomainError(f"ReviewerFinding.line must be positive, got {self.line}")

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewerFinding:
        data = dict(data)
        if isinstance(data.get("created_at"), str):
            data["created_at"] = _str_to_dt(data["created_at"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FindingDisposition:
    """The policy decision applied to one reviewer finding."""

    finding_id: str
    action: DispositionAction
    rationale: str
    decided_by: LaneName

    def __post_init__(self) -> None:
        if self.action not in VALID_DISPOSITION_ACTIONS:
            raise DomainError(f"Invalid disposition action {self.action!r}")
        if not self.rationale:
            raise DomainError("FindingDisposition.rationale must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FindingDisposition:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --- Failure / stagnation --------------------------------------------------


@dataclass
class FailureSummary:
    """Compact summary of a failure, for escalation and audit purposes."""

    run_id: RunId
    work_item_id: WorkItemId
    kind: FailureKind
    message: str
    consecutive_failures: int = 1
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.kind not in VALID_FAILURE_KINDS:
            raise DomainError(f"Invalid failure kind {self.kind!r}")
        if self.consecutive_failures < 1:
            raise DomainError("consecutive_failures must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureSummary:
        data = dict(data)
        if isinstance(data.get("occurred_at"), str):
            data["occurred_at"] = _str_to_dt(data["occurred_at"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StagnationSummary:
    """Evidence that a work item stopped making progress across rounds."""

    run_id: RunId
    work_item_id: WorkItemId
    rounds_without_progress: int
    last_round_id: RoundId | None = None
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.rounds_without_progress < 1:
            raise DomainError("rounds_without_progress must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StagnationSummary:
        data = dict(data)
        if isinstance(data.get("observed_at"), str):
            data["observed_at"] = _str_to_dt(data["observed_at"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --- Serialization helpers -------------------------------------------------


def _serialize_dataclass(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, datetime):
            value = value.isoformat()
        elif isinstance(value, GitHubIssueRef):
            value = {"owner": value.owner, "repo": value.repo, "number": value.number}
        elif isinstance(value, list):
            value = [_nested_to_dict(v) for v in value]
        out[f.name] = value
    return out


def _nested_to_dict(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
