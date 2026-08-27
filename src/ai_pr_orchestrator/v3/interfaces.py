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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .domain import (
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
    RunId,
    WorkflowState,
    WorkItem,
)


@dataclass(frozen=True)
class SessionSpec:
    """Declarative description of a session to start on the CAO fabric.

    ``image``/``command`` are opaque policy strings interpreted by the CAO
    adapter; V3 core never executes them.
    """

    lane: LaneIdentity
    run_id: RunId
    workdir: str
    env: dict[str, str]


@dataclass(frozen=True)
class SessionHandle:
    """Opaque handle to a running CAO session."""

    session_id: str
    lane: str


@dataclass(frozen=True)
class LaneResult:
    """Outcome of one lane execution."""

    session: SessionHandle
    exit_code: int
    output_summary: str
    changed_files: list[str]


@dataclass(frozen=True)
class ModelLease:
    """A reservation of model capacity returned by the model broker."""

    lease_id: str
    assignment: ModelAssignment


@dataclass(frozen=True)
class GateDecision:
    """Result of a CI/PR gate evaluation."""

    passed: bool
    pending_checks: list[str]
    failed_checks: list[str]
    detail: str = ""


@runtime_checkable
class GitHubWorkflowStateStore(Protocol):
    """Reads/writes the authoritative workflow state in GitHub."""

    def load_work_item(self, issue: GitHubIssueRef) -> WorkItem: ...

    def load_state(self, work_item_id: str) -> WorkflowState | None: ...

    def save_state(self, state: WorkflowState) -> None: ...


@runtime_checkable
class CAOSessionController(Protocol):
    """Starts and stops agent sessions on the CAO execution fabric."""

    def start_session(self, spec: SessionSpec) -> SessionHandle: ...

    def poll_session(self, handle: SessionHandle) -> LaneResult | None:
        """Return the result if the session finished, else None."""
        ...

    def terminate_session(self, handle: SessionHandle) -> None: ...


@runtime_checkable
class ModelBroker(Protocol):
    """Reserves and releases model capacity for lane assignments."""

    def reserve(self, assignment: ModelAssignment) -> ModelLease: ...

    def release(self, lease: ModelLease) -> None: ...


@runtime_checkable
class LaneExecutor(Protocol):
    """Executes one unit of work on an agent lane (a Hermes profile)."""

    def execute(self, lane: LaneIdentity, task_prompt: str, workdir: str) -> LaneResult: ...


@runtime_checkable
class CIPRGate(Protocol):
    """Evaluates CI/PR gating policy for a head SHA."""

    def evaluate(self, issue: GitHubIssueRef, head_sha: str) -> GateDecision: ...
