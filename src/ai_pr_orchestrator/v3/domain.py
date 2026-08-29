"""V3 domain model.

Pure, provider-independent value types for the thin policy engine. Everything
in this module is plain data: no I/O, no subprocesses, no GitHub/CAO/Hermes
imports. Types that may be persisted carry ``to_dict``/``from_dict`` round
trips; unknown fields in persisted payloads are preserved as an ``extras``
mapping and written back on serialization, so configs and state produced by
newer versions round-trip losslessly through older readers.

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

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "repo": self.repo, "number": self.number}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GitHubIssueRef:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class GitHubPullRequestRef:
    """Authoritative GitHub identity of one pull request.

    A head SHA alone does not identify a PR (the same SHA can back several
    open PRs), so CI/PR gating requires the PR number explicitly. The
    ``head_sha`` is carried alongside the number so the pair is always
    consistent.
    """

    owner: str
    repo: str
    number: int
    head_sha: str

    def __post_init__(self) -> None:
        if not self.owner or not self.repo:
            raise DomainError("GitHubPullRequestRef requires non-empty owner and repo")
        if self.number <= 0:
            raise DomainError(f"GitHubPullRequestRef.number must be positive, got {self.number}")
        if not self.head_sha:
            raise DomainError("GitHubPullRequestRef.head_sha must be non-empty")

    def slug(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "repo": self.repo,
            "number": self.number,
            "head_sha": self.head_sha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GitHubPullRequestRef:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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
            data["issue"] = GitHubIssueRef.from_dict(data["issue"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = _str_to_dt(data["created_at"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --- Workflow state --------------------------------------------------------


@dataclass(frozen=True)
class WorkflowState:
    """Workflow phase/state of one work item in one run.

    Immutable: phase changes go through :meth:`transition`, which re-runs the
    ``__post_init__`` invariants on the new instance. Unknown fields read from
    a persisted payload are preserved in ``extras`` and written back by
    ``to_dict`` so mixed-version rollouts never silently drop newer fields.
    """

    work_item_id: WorkItemId
    run_id: RunId
    phase: WorkItemPhase
    round_id: RoundId | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    terminal_reason: str | None = None
    #: Structured reviewer findings from the lane(s) of the current round,
    #: persisted so a supervisor stop never loses reviewer output (GitHub
    #: remains the authoritative store).
    findings: list[ReviewerFinding] = field(default_factory=list)
    #: Policy decisions applied to ``findings``; persisted alongside them.
    dispositions: list[FindingDisposition] = field(default_factory=list)
    #: Unknown fields from a newer writer, preserved for lossless round trips.
    extras: dict[str, Any] = field(default_factory=dict)

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

    def transition(
        self,
        phase: WorkItemPhase,
        *,
        round_id: RoundId | None = None,
        terminal_reason: str | None = None,
    ) -> WorkflowState:
        """Return a new state in ``phase``, re-validating all invariants.

        Terminal phases require ``terminal_reason``; leaving a terminal phase
        clears it. ``updated_at`` is refreshed on every transition.
        """
        return WorkflowState(
            work_item_id=self.work_item_id,
            run_id=self.run_id,
            phase=phase,
            round_id=round_id if round_id is not None else self.round_id,
            updated_at=datetime.now(UTC),
            terminal_reason=terminal_reason,
            findings=list(self.findings),
            dispositions=list(self.dispositions),
            extras=dict(self.extras),
        )

    def to_dict(self) -> dict[str, Any]:
        data = _serialize_dataclass(self)
        # The extras bucket itself is not part of the persisted payload;
        # its contents are merged at the top level. Reserved keys (validated
        # state fields) may never be overridden by extras — reject rather
        # than silently corrupting invariants at serialization time.
        reserved = {f.name for f in fields(self)} - {"extras"}
        collisions = sorted(reserved & set(self.extras))
        if collisions:
            raise DomainError(
                f"WorkflowState.extras may not override validated fields: {collisions}"
            )
        data.pop("extras", None)
        data.update(self.extras)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        data = dict(data)
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = _str_to_dt(data["updated_at"])
        if isinstance(data.get("findings"), list):
            data["findings"] = [
                f if isinstance(f, ReviewerFinding) else ReviewerFinding.from_dict(f)
                for f in data["findings"]
            ]
        if isinstance(data.get("dispositions"), list):
            data["dispositions"] = [
                d if isinstance(d, FindingDisposition) else FindingDisposition.from_dict(d)
                for d in data["dispositions"]
            ]
        known = {f.name for f in fields(cls)} - {"extras"}
        kwargs: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
            else:
                extras[key] = value
        return cls(**kwargs, extras=extras)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "role": self.role,
            "profile_template": self.profile_template,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaneIdentity:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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

    def to_dict(self) -> dict[str, Any]:
        return {"lane": self.lane, "model_ref": self.model_ref}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelAssignment:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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
    #: GitHub review-thread identifier this finding was filed on, if any.
    thread_id: str | None = None

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
    """The policy decision applied to one reviewer finding.

    ``thread_id`` identifies the GitHub review thread carrying the finding,
    and ``reply_body`` is the reply posted on that thread (needed for
    reply-before-resolve style policies); both are ``None`` when the
    disposition did not involve a GitHub thread or a reply.
    """

    finding_id: str
    action: DispositionAction
    rationale: str
    decided_by: LaneName
    thread_id: str | None = None
    reply_body: str | None = None

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
