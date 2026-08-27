"""Unit tests for V3 interfaces: protocols must be fakeable without CAO,
Hermes, or GitHub, and implementable with plain in-memory classes."""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

import pytest

from ai_pr_orchestrator.v3.domain import (
    FindingDisposition,
    GitHubIssueRef,
    GitHubPullRequestRef,
    LaneIdentity,
    ModelAssignment,
    ReviewerFinding,
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
    StateConflictError,
)


class FakeGitHubStateStore:
    """In-memory stand-in for GitHub workflow state, with optimistic
    concurrency: save_state refuses to overwrite a newer version."""

    def __init__(self) -> None:
        self.items: dict[str, WorkItem] = {}
        self.states: dict[str, WorkflowState] = {}
        self.saved: list[WorkflowState] = []

    def load_work_item(self, issue: GitHubIssueRef) -> WorkItem:
        return self.items[issue.slug()]

    def load_state(self, work_item_id: str) -> WorkflowState | None:
        return self.states.get(work_item_id)

    def save_state(self, state: WorkflowState, expected_updated_at: datetime | None) -> None:
        current = self.states.get(state.work_item_id)
        if expected_updated_at is None:
            # None means create-only: refuse to overwrite an existing state.
            if current is not None:
                raise StateConflictError(
                    f"state for {state.work_item_id} already exists; create-only save refused"
                )
        elif current is not None and current.updated_at > expected_updated_at:
            raise StateConflictError(f"state for {state.work_item_id} was updated concurrently")
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
    def __init__(self) -> None:
        self.leases: list[ModelLease | None] = []

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        lease: ModelLease | None = None,
    ) -> LaneResult:
        self.leases.append(lease)
        return LaneResult(
            session=SessionHandle(session_id="fake", lane=lane.lane),
            exit_code=0,
            output_summary=task_prompt,
            changed_files=[],
        )


class FakeReviewerLaneExecutor:
    """A reviewer lane executor returning structured findings."""

    def __init__(self) -> None:
        self.leases: list[ModelLease | None] = []

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        lease: ModelLease | None = None,
    ) -> LaneResult:
        self.leases.append(lease)
        finding = ReviewerFinding(
            id="f1",
            lane=lane.lane,
            body="Unhandled error path",
            severity="major",
            run_id="run-1",
            round_id="r1",
        )
        return LaneResult(
            session=SessionHandle(session_id="fake", lane=lane.lane),
            exit_code=0,
            output_summary=task_prompt,
            changed_files=[],
            findings=[finding],
            dispositions=[
                FindingDisposition(
                    finding_id="f1", action="fix", rationale="real bug", decided_by="foreman"
                )
            ],
        )


class FakeCIPRGate:
    def __init__(self) -> None:
        self.seen: list[tuple[GitHubIssueRef, GitHubPullRequestRef]] = []

    def evaluate(self, issue: GitHubIssueRef, pr: GitHubPullRequestRef) -> GateDecision:
        self.seen.append((issue, pr))
        return GateDecision(passed=True, pending_checks=[], failed_checks=[])


def _run_fakes() -> dict[str, Any]:
    issue = GitHubIssueRef(owner="o", repo="r", number=1)
    store = FakeGitHubStateStore()
    item = WorkItem(id="wi-1", issue=issue)
    store.items[issue.slug()] = item
    store.save_state(WorkflowState(work_item_id="wi-1", run_id="run-1", phase="queued"), None)
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
    pr = GitHubPullRequestRef(owner="o", repo="r", number=9, head_sha="abc123")
    decision = gate.evaluate(issue, pr)
    return {
        "store": store,
        "cao": cao,
        "broker": broker,
        "executor": executor,
        "gate": gate,
        "decision": decision,
        "lease": lease,
        "issue": issue,
        "pr": pr,
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


class TestOptimisticConcurrency:
    def test_stale_save_raises_conflict(self) -> None:
        store = FakeGitHubStateStore()
        first = WorkflowState(work_item_id="wi-1", run_id="run-1", phase="queued")
        store.save_state(first, None)

        reader_a = store.load_state("wi-1")
        assert reader_a is not None
        # Process B saves a newer version in the meantime.
        newer = reader_a.transition("planning")
        store.save_state(newer, expected_updated_at=reader_a.updated_at)

        # Process A's save is based on the version it loaded; it must fail.
        stale = reader_a.transition("coding")
        with pytest.raises(StateConflictError, match="concurrently"):
            store.save_state(stale, expected_updated_at=reader_a.updated_at)

    def test_fresh_save_succeeds(self) -> None:
        store = FakeGitHubStateStore()
        state = WorkflowState(work_item_id="wi-1", run_id="run-1", phase="queued")
        store.save_state(state, None)
        updated = state.transition("planning")
        store.save_state(updated, expected_updated_at=state.updated_at)
        assert store.load_state("wi-1") == updated

    def test_none_means_create_only(self) -> None:
        store = FakeGitHubStateStore()
        first = WorkflowState(work_item_id="wi-1", run_id="run-1", phase="queued")
        store.save_state(first, None)
        # A second writer also observing "no state" (or accepting a default)
        # must not silently last-write-win over the first.
        with pytest.raises(StateConflictError, match="create-only"):
            store.save_state(
                WorkflowState(work_item_id="wi-1", run_id="run-1", phase="claiming"), None
            )
        # And the first write must still be intact.
        assert store.load_state("wi-1") == first

    def test_protocol_save_state_takes_no_default(self) -> None:
        # The precondition is mandatory: there is no default to accept.
        param = inspect.signature(GitHubWorkflowStateStore.save_state).parameters[
            "expected_updated_at"
        ]
        assert param.default is inspect.Parameter.empty


class TestGateDecisionInvariants:
    def test_passed_with_failed_checks_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="failed or pending"):
            GateDecision(passed=True, pending_checks=[], failed_checks=["tests"])

    def test_passed_with_pending_checks_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="failed or pending"):
            GateDecision(passed=True, pending_checks=["lint"], failed_checks=[])

    def test_consistent_decisions_are_valid(self) -> None:
        assert GateDecision(passed=True, pending_checks=[], failed_checks=[]).passed
        assert GateDecision(passed=False, pending_checks=[], failed_checks=["tests"]).failed_checks


class TestSessionSpecLeaseLaneBinding:
    def test_mismatched_lease_lane_is_rejected(self) -> None:
        lease = ModelLease(
            lease_id="lease-1", assignment=ModelAssignment(lane="reviewer-b", model_ref="rev")
        )
        with pytest.raises(ValueError, match="does not match model lease"):
            SessionSpec(
                lane=LaneIdentity(lane="worker-a", role="worker", profile_template="p"),
                run_id="run-1",
                workdir="/tmp/x",
                env={},
                model_lease=lease,
            )

    def test_matching_lease_lane_is_accepted(self) -> None:
        lease = ModelLease(
            lease_id="lease-1", assignment=ModelAssignment(lane="worker-a", model_ref="coder")
        )
        spec = SessionSpec(
            lane=LaneIdentity(lane="worker-a", role="worker", profile_template="p"),
            run_id="run-1",
            workdir="/tmp/x",
            env={},
            model_lease=lease,
        )
        assert spec.model_lease is lease


class TestModelBinding:
    def test_session_spec_carries_lease(self) -> None:
        out = _run_fakes()
        lease: ModelLease = out["lease"]
        spec = SessionSpec(
            lane=LaneIdentity(lane="worker-1", role="worker", profile_template="p"),
            run_id="run-1",
            workdir="/tmp/x",
            env={},
            model_lease=lease,
        )
        assert spec.model_lease is not None
        assert spec.model_lease.assignment.model_ref == "coder-main"
        # Default remains None so specs without a reservation stay valid.
        assert (
            SessionSpec(lane=spec.lane, run_id="run-1", workdir="/tmp/x", env={}).model_lease
            is None
        )

    def test_executor_receives_lease(self) -> None:
        out = _run_fakes()
        executor = FakeLaneExecutor()
        lane = LaneIdentity(lane="worker-1", role="worker", profile_template="p")
        executor.execute(lane, "task", "/tmp/x", lease=out["lease"])
        assert executor.leases == [out["lease"]]

    def test_reviewer_lane_returns_structured_findings(self) -> None:
        executor = FakeReviewerLaneExecutor()
        lane = LaneIdentity(lane="rev-1", role="reviewer", profile_template="p")
        result = executor.execute(lane, "review", "/tmp/x")
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.lane == "rev-1"
        assert finding.severity == "major"
        assert [d.finding_id for d in result.dispositions] == ["f1"]


class TestCIPRGateIdentity:
    def test_gate_receives_pr_identity(self) -> None:
        out = _run_fakes()
        gate: FakeCIPRGate = out["gate"]
        issue, pr = gate.seen[0]
        assert issue == out["issue"]
        assert pr == out["pr"]
        assert pr.head_sha == "abc123"
        assert pr.number == 9
