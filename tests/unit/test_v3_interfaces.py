"""Unit tests for V3 interfaces: protocols must be fakeable without CAO,
Hermes, or GitHub, and implementable with plain in-memory classes."""

from __future__ import annotations

from typing import Any

from ai_pr_orchestrator.v3.domain import (
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
    WorkflowState,
    WorkItem,
)
from ai_pr_orchestrator.v3.interfaces import (
    CAOSessionController,
    CIPRGate,
    GateDecision,
    GitHubWorkflowStateStore,
    LaneExecutor,
    LaneResult,
    ModelBroker,
    ModelLease,
    SessionHandle,
    SessionSpec,
)


class FakeGitHubStateStore:
    """In-memory stand-in for GitHub workflow state."""

    def __init__(self) -> None:
        self.items: dict[str, WorkItem] = {}
        self.states: dict[str, WorkflowState] = {}
        self.saved: list[WorkflowState] = []

    def load_work_item(self, issue: GitHubIssueRef) -> WorkItem:
        return self.items[issue.slug()]

    def load_state(self, work_item_id: str) -> WorkflowState | None:
        return self.states.get(work_item_id)

    def save_state(self, state: WorkflowState) -> None:
        self.saved.append(state)
        self.states[state.work_item_id] = state


class FakeCAOSessionController:
    def __init__(self) -> None:
        self.started: list[SessionSpec] = []
        self.terminated: list[SessionHandle] = []
        self._counter = 0

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        self.started.append(spec)
        self._counter += 1
        return SessionHandle(session_id=f"s{self._counter}", lane=spec.lane.lane)

    def poll_session(self, handle: SessionHandle) -> LaneResult | None:
        return LaneResult(session=handle, exit_code=0, output_summary="ok", changed_files=[])

    def terminate_session(self, handle: SessionHandle) -> None:
        self.terminated.append(handle)


class FakeModelBroker:
    def __init__(self) -> None:
        self.released: list[ModelLease] = []

    def reserve(self, assignment: ModelAssignment) -> ModelLease:
        return ModelLease(lease_id=f"lease-{assignment.lane}", assignment=assignment)

    def release(self, lease: ModelLease) -> None:
        self.released.append(lease)


class FakeLaneExecutor:
    def execute(self, lane: LaneIdentity, task_prompt: str, workdir: str) -> LaneResult:
        return LaneResult(
            session=SessionHandle(session_id="fake", lane=lane.lane),
            exit_code=0,
            output_summary=task_prompt,
            changed_files=[],
        )


class FakeCIPRGate:
    def evaluate(self, issue: GitHubIssueRef, head_sha: str) -> GateDecision:
        return GateDecision(passed=True, pending_checks=[], failed_checks=[])


def _run_fakes() -> dict[str, Any]:
    issue = GitHubIssueRef(owner="o", repo="r", number=1)
    store = FakeGitHubStateStore()
    item = WorkItem(id="wi-1", issue=issue)
    store.items[issue.slug()] = item
    store.save_state(WorkflowState(work_item_id="wi-1", run_id="run-1", phase="queued"))
    assert store.load_state("wi-1") is not None

    cao = FakeCAOSessionController()
    handle = cao.start_session(
        SessionSpec(
            lane=LaneIdentity(lane="worker-1", role="worker", profile_template="aipro-worker"),
            run_id="run-1",
            workdir="/tmp/x",
            env={},
        )
    )
    result = cao.poll_session(handle)
    assert result is not None and result.exit_code == 0
    cao.terminate_session(handle)

    broker = FakeModelBroker()
    lease = broker.reserve(ModelAssignment(lane="worker-1", model_ref="coder-main"))
    broker.release(lease)

    executor = FakeLaneExecutor()
    gate = FakeCIPRGate()
    decision = gate.evaluate(issue, "abc123")
    return {
        "store": store,
        "cao": cao,
        "broker": broker,
        "executor": executor,
        "gate": gate,
        "decision": decision,
    }


class TestProtocolsAreFakeable:
    def test_fakes_satisfy_protocols(self) -> None:
        out = _run_fakes()
        assert isinstance(out["store"], GitHubWorkflowStateStore)
        assert isinstance(out["cao"], CAOSessionController)
        assert isinstance(out["broker"], ModelBroker)
        assert isinstance(out["executor"], LaneExecutor)
        assert isinstance(out["gate"], CIPRGate)

    def test_fake_state_store_round_trip(self) -> None:
        out = _run_fakes()
        store: FakeGitHubStateStore = out["store"]
        state = store.load_state("wi-1")
        assert state is not None
        assert state.phase == "queued"
        assert len(store.saved) == 1

    def test_fake_cao_lifecycle(self) -> None:
        out = _run_fakes()
        cao: FakeCAOSessionController = out["cao"]
        assert len(cao.started) == 1
        assert len(cao.terminated) == 1

    def test_fake_broker_lifecycle(self) -> None:
        out = _run_fakes()
        broker: FakeModelBroker = out["broker"]
        assert [lease.lease_id for lease in broker.released] == ["lease-worker-1"]
