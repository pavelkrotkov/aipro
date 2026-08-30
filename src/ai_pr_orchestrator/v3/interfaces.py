"""V3 core interfaces.

Protocols that decouple the thin policy engine from the systems it sits on
top of: GitHub (workflow state), CAO (session fabric), the model broker,
lane executors (Hermes profiles), and CI/PR gates.

Every protocol is structural (``@runtime_checkable``) and deliberately free
of any vendor, provider, or model name, so tests can supply in-memory fakes
without shelling out to CAO, Hermes, or GitHub, and so the core never
depends on a specific implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from .domain import (
    FindingDisposition,
    GitHubIssueRef,
    GitHubPullRequestRef,
    LaneIdentity,
    ModelAssignment,
    ReviewerFinding,
    RoundId,
    RunId,
    WorkflowState,
    WorkItem,
    WorkItemId,
)
from .telemetry import ProviderResourceSnapshot


class StateConflictError(RuntimeError):
    """Raised by :meth:`GitHubWorkflowStateStore.save_state` when the
    optimistic-concurrency precondition fails (another writer updated the
    state between our load and our save)."""


@dataclass(frozen=True)
class ModelLease:
    """A reservation of model capacity returned by the model broker."""

    lease_id: str
    assignment: ModelAssignment


@dataclass(frozen=True)
class LaneExecutionContext:
    """Typed run/round context for one lane execution.

    Carried explicitly through :meth:`LaneExecutor.execute` (and mirrored on
    :class:`SessionSpec`) so implementations never have to recover run or
    round identity from free-form prompt text or ambient state — which is
    what makes misattribution across overlapping rounds possible.
    """

    run_id: RunId
    round_id: RoundId | None = None
    work_item_id: WorkItemId | None = None


@dataclass(frozen=True)
class SessionSpec:
    """Declarative description of a session to start on the CAO fabric.

    ``image``/``command`` are opaque policy strings interpreted by the CAO
    adapter; V3 core never executes them. ``model_lease`` binds the session
    to a model reservation acquired from the model broker, so a lane can
    never run against capacity other than what was reserved for it: the
    lease's assignment lane must match the session's own lane. ``context``
    is **mandatory** — a spec without typed run/round identity cannot be
    constructed, so sessions are always attributable to their run/round.
    """

    lane: LaneIdentity
    run_id: RunId
    workdir: str
    env: dict[str, str]
    context: LaneExecutionContext
    image: str | None = None
    command: str | None = None
    model_lease: ModelLease | None = None

    def __post_init__(self) -> None:
        if self.context is None:
            raise ValueError(
                "SessionSpec requires a LaneExecutionContext: without typed "
                "run/round identity, lane findings cannot be attributed "
                "correctly across overlapping rounds"
            )
        if self.run_id != self.context.run_id:
            raise ValueError(
                f"SessionSpec run_id {self.run_id!r} does not match its context "
                f"run_id {self.context.run_id!r}: the session name and the durable "
                "metadata must carry the same run identity or attribution diverges"
            )
        if self.model_lease is not None and self.model_lease.assignment.lane != self.lane.lane:
            raise ValueError(
                f"SessionSpec lane {self.lane.lane!r} does not match model lease "
                f"reserved for lane {self.model_lease.assignment.lane!r}"
            )


@dataclass(frozen=True)
class SessionHandle:
    """Opaque handle to a running CAO session."""

    session_id: str
    lane: str


@dataclass(frozen=True)
class LaneResult:
    """Outcome of one lane execution.

    Reviewer lanes return their observations as structured
    :class:`~ai_pr_orchestrator.v3.domain.ReviewerFinding` values so they can
    flow into the policy engine unchanged; dispositions record the policy
    decisions applied to those findings.
    """

    session: SessionHandle
    exit_code: int
    output_summary: str
    changed_files: list[str]
    findings: list[ReviewerFinding] = field(default_factory=list)
    dispositions: list[FindingDisposition] = field(default_factory=list)


@dataclass(frozen=True)
class GateDecision:
    """Result of a CI/PR gate evaluation.

    Invariant: ``passed=True`` is only valid when no checks are failed or
    pending; a gate cannot pass with outstanding checks.
    """

    passed: bool
    pending_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    detail: str = ""

    def __post_init__(self) -> None:
        # Accept list-typed callers but store immutable tuples, so the
        # decision cannot be contradicted after construction.
        if isinstance(self.pending_checks, list):
            object.__setattr__(self, "pending_checks", tuple(self.pending_checks))
        if isinstance(self.failed_checks, list):
            object.__setattr__(self, "failed_checks", tuple(self.failed_checks))
        if self.passed and (self.failed_checks or self.pending_checks):
            raise ValueError(
                "GateDecision cannot be passed=True with failed or pending checks: "
                f"failed={self.failed_checks} pending={self.pending_checks}"
            )


@runtime_checkable
class GitHubWorkflowStateStore(Protocol):
    """Reads/writes the authoritative workflow state in GitHub.

    ``save_state`` uses optimistic concurrency, and a precondition is
    **mandatory on every write** — there is no default:

    - ``expected_updated_at=None`` means *create-only*: the store must save
      only if no state exists yet for the work item, and raise
      :class:`StateConflictError` otherwise. Use it for the initial claim.
    - ``expected_updated_at=<datetime>`` means *expect-that-version*: pass
      the ``updated_at`` of the state you loaded, and the store must raise
      :class:`StateConflictError` instead of saving if another writer
      persisted a newer version in the meantime.

    Two processes therefore cannot silently last-write-win over each other,
    and an update caller cannot accidentally skip the precondition by
    omitting it.
    """

    def load_work_item(self, issue: GitHubIssueRef) -> WorkItem: ...

    def load_state(self, work_item_id: str) -> WorkflowState | None: ...

    def save_state(self, state: WorkflowState, expected_updated_at: datetime | None) -> None: ...


@runtime_checkable
class CAOSessionController(Protocol):
    """Starts and stops agent sessions on the CAO execution fabric."""

    def start_session(self, spec: SessionSpec) -> SessionHandle: ...

    def poll_session(self, handle: SessionHandle) -> LaneResult | None:
        """Return the result if the session finished, else None."""
        ...

    def terminate_session(self, handle: SessionHandle) -> None: ...


@runtime_checkable
class ProviderTelemetrySource(Protocol):
    """Reports live quota/health for the resources it is configured to serve.

    ``snapshot`` is **total**: it never raises, and never signals failure by
    omission. A source that cannot reach its provider returns a snapshot with
    ``availability='unknown'`` and a stated reason, so a broken probe is
    reported as ignorance rather than as an exhausted or an empty quota.

    ``at`` is passed by the caller rather than read from the clock inside, so
    one fan-out over many resources evaluates every one of them against the
    same instant.
    """

    def resources(self) -> tuple[str, ...]: ...

    def snapshot(
        self, resource: str, *, at: datetime | None = None
    ) -> ProviderResourceSnapshot: ...


@runtime_checkable
class ModelBroker(Protocol):
    """Reserves and releases model capacity for lane assignments."""

    def reserve(self, assignment: ModelAssignment) -> ModelLease: ...

    def release(self, lease: ModelLease) -> None: ...


@runtime_checkable
class LaneExecutor(Protocol):
    """Executes one unit of work on an agent lane (a Hermes profile).

    ``lease`` is the model reservation for this execution (from
    :class:`ModelBroker`); passing it makes the lane-to-model binding part of
    the execution contract rather than ambient state. ``context`` carries the
    typed run/round identity of the work unit so implementations never have
    to recover it from free-form prompt text — findings are attributed to the
    run/round in ``context``, not to whatever the prompt happens to mention.
    ``context`` is **mandatory** (no default): a conforming implementation is
    never invoked without run/round identity, which closes the
    misattribution-across-overlapping-rounds hole.
    """

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease: ModelLease | None = None,
    ) -> LaneResult: ...


@runtime_checkable
class CIPRGate(Protocol):
    """Evaluates CI/PR gating policy for one pull request.

    The PR is identified by :class:`~ai_pr_orchestrator.v3.domain.GitHubPullRequestRef`
    (number + head SHA); a head SHA alone does not identify a PR because the
    same SHA can back several open pull requests.
    """

    def evaluate(self, issue: GitHubIssueRef, pr: GitHubPullRequestRef) -> GateDecision: ...


@runtime_checkable
class GitOperations(Protocol):
    """Repo lifecycle operations: branch, worktree, commit, push.

    All methods take explicit identity parameters (committer name/email) so
    the implementation never reads ambient git config — a forged or missing
    global identity must not silently change what lands on the remote. Every
    operation is deliberately one primitive: the foreman composes them, so
    tests can fake each step and the production implementation can shell out
    to git without embedding policy.
    """

    def default_branch(self) -> str: ...

    def create_branch(self, branch: str, from_ref: str) -> None: ...

    def create_worktree(self, path: str, branch: str) -> str:
        """Materialize ``branch`` at ``path``; return the resolved workdir."""

        ...

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        """Commit all changes in ``workdir``; return the new head SHA."""

        ...

    def commit_count(self, workdir: str, base_ref: str) -> int:
        """Commits in ``workdir`` not reachable from ``base_ref``."""

        ...

    def push(self, branch: str) -> None: ...

    def cleanup_worktree(self, path: str) -> None: ...

