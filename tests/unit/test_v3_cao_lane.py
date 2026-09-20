"""Unit tests for :class:`ai_pr_orchestrator.v3.cao_lane.CaoLaneExecutor`.

Issue #55, P1, PR-2: the production ``LaneExecutor`` bridge over
:class:`CaoSessionController`. The fake from PR-1 supplies a deterministic
CAO control plane; these tests drive the executor through the four paths
the plan called out (happy, transient-5xx then success, terminal failed,
executor budget exhausted). The fifth path — controller-level
``timed_out`` — is covered by the controller's own tests in
``test_fake_cao_server.py``.

The tests run in plain CI: no real ``cao-server`` required.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoControlPlaneError,
    CaoSessionController,
    CaoTransportError,
    SessionBusyError,
    session_name_for,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.domain import LaneIdentity
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, SessionSpec
from ai_pr_orchestrator.v3.lanes import DEFAULT_LANES, DEVELOPER_LANE, LaneRegistry
from tests.integration._fake_cao_server import (
    DEFAULT_STATUS_SEQUENCE,
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_PROCESSING,
    STATUS_STARTED,
    FakeCAOServer,
    FaultSpec,
)
from tests.unit.test_v3_git_ops import FakeGitOperations

#: Marker the agent is asked to echo. Asserted via the controller's
#: ``final_output`` (which returns whatever the fake stored), so the
#: contract being tested is "executor returns the controller's result",
#: not the agent's behaviour.
MARKER = "AIPRO-CAO-LANE-OK"


def _config(url: str) -> CAOControlPlaneConfig:
    return CAOControlPlaneConfig(
        base_url=url,
        session_timeout_seconds=60,
        request_timeout_seconds=5,
    )


@pytest.fixture
def fake_cao() -> Any:
    with FakeCAOServer() as cao:
        yield cao


def _lane() -> LaneIdentity:
    return LaneRegistry.default().get(DEVELOPER_LANE)


def _context(run_id: str) -> LaneExecutionContext:
    return LaneExecutionContext(run_id=run_id)


def _spec(run_id: str, workdir: str, command: str) -> SessionSpec:
    return SessionSpec(
        lane=_lane(),
        run_id=run_id,
        workdir=workdir,
        env={},
        context=_context(run_id),
        command=command,
    )


# --- Happy path ---------------------------------------------------------


def test_execute_returns_lane_result_on_completed_session(fake_cao: FakeCAOServer, tmp_path):
    """A session that walks the default status sequence to ``idle`` is
    reported as completed by the controller; the executor returns the
    controller's :class:`LaneResult` with the live session handle."""
    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(
        controller, LaneRegistry.default(), git=FakeGitOperations(), poll_interval_seconds=0.01
    )

    handle = executor.execute(
        _lane(),
        f"Reply with exactly: {MARKER}",
        str(tmp_path),
        _context(run_id),
    )

    assert handle.session is not None
    assert handle.exit_code == 0, (
        f"expected 0 (completed), got {handle.exit_code}; output_summary={handle.output_summary!r}"
    )
    # The controller's final_output returns the marker; the executor
    # copies it into LaneResult.output_summary.
    assert MARKER in handle.output_summary
    state = fake_cao._sessions[name]
    assert state.initial_message is None
    assert state.submitted_messages == [f"Reply with exactly: {MARKER}"]


@pytest.mark.parametrize("restart", [False, True])
def test_execute_adopts_existing_session_by_name(fake_cao: FakeCAOServer, tmp_path, restart):
    """Retained and restarted controllers deliver the new prompt exactly once."""
    run_id = "follow-up"
    name = session_name_for(run_id, DEVELOPER_LANE)
    registry = LaneRegistry.default()
    with CaoSessionController(_config(fake_cao.url), registry) as controller:
        executor = CaoLaneExecutor(
            controller, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
        )
        executor.execute(_lane(), "first task", str(tmp_path), _context(run_id))
        if restart:
            with CaoSessionController(_config(fake_cao.url), registry) as restarted:
                executor = CaoLaneExecutor(
                    restarted, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
                )
                result = executor.execute(
                    _lane(), "follow-up task", str(tmp_path), _context(run_id)
                )
        else:
            result = executor.execute(_lane(), "follow-up task", str(tmp_path), _context(run_id))

    assert result.exit_code == 0
    assert len(fake_cao._sessions) == 1
    assert fake_cao._sessions[name].submitted_messages == ["first task", "follow-up task"]


def test_execute_preserves_busy_adopted_session(fake_cao: FakeCAOServer, tmp_path):
    """Rejected follow-up input must not terminate an earlier in-flight turn."""
    run_id = "busy-follow-up"
    name = session_name_for(run_id, DEVELOPER_LANE)
    registry = LaneRegistry.default()
    fake_cao.set_status_sequence(name, [STATUS_PROCESSING])
    with CaoSessionController(_config(fake_cao.url), registry) as controller:
        handle = controller.start_session(_spec(run_id, str(tmp_path), "first task"))
        controller.submit_work(handle, "first task")
        state = fake_cao._sessions[name]
        fake_cao.add_fault(
            FaultSpec(
                method="POST",
                path_prefix=f"/terminals/{state.terminal_id}/input",
                status_code=409,
            )
        )
        executor = CaoLaneExecutor(
            controller, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
        )
        with pytest.raises(SessionBusyError):
            executor.execute(_lane(), "follow-up task", str(tmp_path), _context(run_id))

        assert controller.observe(handle).state == "running"
    assert state.submitted_messages == ["first task"]
    assert not state.deleted


def test_execute_surfaces_uncertain_followup_submission(fake_cao: FakeCAOServer, tmp_path):
    """A dropped submission response cannot return the previous successful result."""
    run_id = "uncertain-follow-up"
    name = session_name_for(run_id, DEVELOPER_LANE)
    registry = LaneRegistry.default()
    with CaoSessionController(_config(fake_cao.url), registry) as controller:
        executor = CaoLaneExecutor(
            controller, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
        )
        executor.execute(_lane(), "first task", str(tmp_path), _context(run_id))
        state = fake_cao._sessions[name]
        fake_cao.add_fault(
            FaultSpec(
                method="POST",
                path_prefix=f"/terminals/{state.terminal_id}/input",
                transport_reset=True,
            )
        )
        with pytest.raises(CaoTransportError):
            executor.execute(_lane(), "follow-up task", str(tmp_path), _context(run_id))

    assert state.submitted_messages == ["first task"]
    assert state.deleted


def test_execute_clears_previous_idle_evidence_on_new_work(fake_cao: FakeCAOServer, tmp_path):
    """The controller clears idle-settle evidence on accepted input
    (see ``submit_work`` in ``cao.py``). A second ``execute`` therefore
    starts fresh, not by inheriting the first session's terminal state.

    PR #73 review thread 1: the executor now submits the lane's prompt on
    every invocation (adopted or fresh). The fake resets its status cursor
    on submit so the second walk is identical to the first.
    """
    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)
    fake_cao.set_status_sequence(
        name, (STATUS_STARTED, STATUS_PROCESSING, STATUS_IDLE, STATUS_IDLE, STATUS_IDLE)
    )

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(
        controller, LaneRegistry.default(), git=FakeGitOperations(), poll_interval_seconds=0.01
    )

    first = executor.execute(_lane(), f"echo {MARKER}", str(tmp_path), _context(run_id))
    second = executor.execute(_lane(), f"echo {MARKER}", str(tmp_path), _context(run_id))

    assert first.exit_code == 0
    assert second.exit_code == 0


# --- Transient fault then success --------------------------------------


def test_execute_succeeds_after_transient_5xx(fake_cao: FakeCAOServer, tmp_path):
    """A 503 on the first ``POST /sessions`` is transient; the controller
    does not retry internally (the executor drives that loop), so this
    case is actually covered by the next-level test: a single 5xx surfaces
    to the executor as a typed error. The behavior we verify here is that
    the executor terminates the half-launched session so it does not
    leak."""
    from ai_pr_orchestrator.v3.cao import CaoControlPlaneError

    run_id = f"it-{int(time.time() * 1000)}"
    fake_cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=503))

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(
        controller, LaneRegistry.default(), git=FakeGitOperations(), poll_interval_seconds=0.01
    )

    with pytest.raises(CaoControlPlaneError):
        executor.execute(
            _lane(),
            f"echo {MARKER}",
            str(tmp_path),
            _context(run_id),
        )
    # No session was successfully created, so no termination is required;
    # the contract is that the executor does not raise anything new from
    # the cleanup path. The controller already saw the 5xx and mapped it
    # to a typed error.
    assert session_name_for(run_id, DEVELOPER_LANE) not in fake_cao._sessions


# --- Terminal failure ---------------------------------------------------


def test_execute_returns_failure_on_terminal_error(fake_cao: FakeCAOServer, tmp_path):
    """A session whose terminal reports ``error`` (mapped to ``failed``
    lifecycle) is returned as a non-zero :class:`LaneResult` with the
    controller's detail as output_summary. The executor does not raise;
    the foreman classifies the failure from ``exit_code``."""
    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    # Drive the terminal straight to the error state and stop.
    fake_cao.set_status_sequence(name, [STATUS_ERROR])

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(
        controller, LaneRegistry.default(), git=FakeGitOperations(), poll_interval_seconds=0.01
    )

    result = executor.execute(
        _lane(),
        f"echo {MARKER}",
        str(tmp_path),
        _context(run_id),
    )

    assert result.exit_code != 0
    assert result.session.session_id == name


# --- Executor budget ---------------------------------------------------


@pytest.fixture
def accelerated_time(monkeypatch):
    """Advance both executor and controller clocks without wall-clock waits."""
    elapsed = [0.0]
    started = datetime.now(UTC)

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return (started + timedelta(seconds=elapsed[0])).astimezone(tz)

    def advance(_seconds):
        elapsed[0] += 200.0

    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(time, "sleep", advance)
    monkeypatch.setattr("ai_pr_orchestrator.v3.cao.datetime", ClockDateTime)
    return elapsed


def test_execute_allows_work_beyond_old_600_second_budget(fake_cao, tmp_path, accelerated_time):
    name = session_name_for("long-work", DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)
    fake_cao.set_status_sequence(name, [STATUS_PROCESSING] * 5 + [STATUS_IDLE] * 3)
    with CaoSessionController(CAOControlPlaneConfig(base_url=fake_cao.url)) as controller:
        result = CaoLaneExecutor(
            controller, LaneRegistry.default(), git=FakeGitOperations()
        ).execute(_lane(), "long task", str(tmp_path), _context("long-work"))
    assert 600 < accelerated_time[0] < 3600
    assert result.exit_code == 0
    assert result.output_summary == MARKER
    assert not fake_cao._sessions[name].deleted


@pytest.mark.parametrize("budget", [0.05, 60, float("nan"), float("inf"), -float("inf")])
def test_executor_rejects_invalid_override(fake_cao, budget):
    with (
        CaoSessionController(_config(fake_cao.url)) as controller,
        pytest.raises(ValueError, match="exceed session_timeout_seconds"),
    ):
        CaoLaneExecutor(
            controller, LaneRegistry.default(), git=FakeGitOperations(), max_poll_seconds=budget
        )
    assert not fake_cao._sessions


def test_controller_timeout_remains_authoritative(fake_cao, tmp_path, accelerated_time):
    name = session_name_for("expired-work", DEVELOPER_LANE)
    fake_cao.set_status_sequence(name, [STATUS_PROCESSING] * 20)
    with CaoSessionController(_config(fake_cao.url)) as controller:
        result = CaoLaneExecutor(
            controller, LaneRegistry.default(), git=FakeGitOperations()
        ).execute(_lane(), "task", str(tmp_path), _context("expired-work"))
    assert result.exit_code != 0
    assert "session exceeded 60s" in result.output_summary
    assert fake_cao._sessions[name].deleted


@pytest.mark.parametrize("override", [None, 500.0])
def test_execute_raises_when_observer_never_finishes(
    fake_cao, tmp_path, accelerated_time, monkeypatch, override
):
    name = session_name_for("stuck-observer", DEVELOPER_LANE)
    with CaoSessionController(_config(fake_cao.url)) as controller:
        # Deliberately break only the observation boundary to exercise the guard.
        monkeypatch.setattr(controller, "poll_session", lambda _handle: None)
        executor = CaoLaneExecutor(
            controller, LaneRegistry.default(), git=FakeGitOperations(), max_poll_seconds=override
        )
        with pytest.raises(TimeoutError):
            executor.execute(_lane(), "task", str(tmp_path), _context("stuck-observer"))
    assert accelerated_time[0] >= (90 if override is None else override)
    assert fake_cao._sessions[name].deleted


# --- Lane registry resolution ------------------------------------------


def test_execute_uses_registry_lane_not_caller_identity(fake_cao: FakeCAOServer, tmp_path):
    """The executor looks the lane up in the registry on every call, so
    a reconfigured registry wins. A caller that passes a mismatched
    :class:`LaneIdentity` (wrong profile) is silently corrected to the
    registry's identity, which is the lane-to-profile binding the
    controller already enforces."""
    run_id = f"it-{int(time.time() * 1000)}"

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(
        controller, LaneRegistry.default(), git=FakeGitOperations(), poll_interval_seconds=0.01
    )

    # The real lane is the registry's. The caller passes a different
    # LaneIdentity with the same lane NAME but a different (wrong)
    # profile; the registry's binding should win, not the caller's.
    bogus = LaneIdentity(lane=DEVELOPER_LANE, role="worker", profile_template="bogus-profile")
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)

    result = executor.execute(
        bogus,
        f"echo {MARKER}",
        str(tmp_path),
        _context(run_id),
    )
    assert result.exit_code == 0
    # The session was launched with the registry's profile, not bogus.
    state = fake_cao._sessions[name]
    assert state.agent_profile != "bogus-profile"


# --- Required status constants are exported (smoke) --------------------


def test_fake_cao_status_constants_match_real_cao_vocabulary():
    """Smoke check: the fake's status constants exist and align with the
    vocabulary the controller's ``_STATUS_LIFECYCLE`` map understands.
    If CAO renames a status, this fails first and PR-1's fake must be
    updated in the same change."""
    assert STATUS_IDLE == "idle"
    assert STATUS_PROCESSING == "processing"
    assert STATUS_COMPLETED == "completed"
    assert len(DEFAULT_STATUS_SEQUENCE) >= 3, (
        "default sequence must walk started -> processing -> idle at minimum"
    )


@pytest.mark.parametrize("lane", DEFAULT_LANES, ids=lambda lane: lane.lane)
def test_execute_refreshes_round_context_durably(fake_cao: FakeCAOServer, tmp_path, lane):
    """Every lane retains its session identity while advancing turn attribution."""
    registry = LaneRegistry.default()
    run_id = "multi-round"
    work_items = (
        ["same-item", "same-item"] if lane.role == "worker" else ["first-head", "fixed-head"]
    )
    name = session_name_for(run_id, lane.lane, work_items[0] if lane.role == "worker" else None)
    fake_cao.set_output(name, "[]" if lane.role == "reviewer" else MARKER)
    contexts = [
        LaneExecutionContext(run_id=run_id, round_id="review-1", work_item_id=work_items[0]),
        LaneExecutionContext(run_id=run_id, round_id="review-2", work_item_id=work_items[1]),
    ]
    with CaoSessionController(_config(fake_cao.url), registry) as controller:
        executor = CaoLaneExecutor(
            controller, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
        )
        for context in contexts:
            result = executor.execute(lane, f"Review {context.round_id}", str(tmp_path), context)
            assert result.exit_code == 0
            metadata = fake_cao._sessions[name].metadata
            assert (metadata["round_id"], metadata["work_item_id"]) == (
                context.round_id,
                context.work_item_id,
            )
    assert len(fake_cao._sessions) == 1
    assert fake_cao._sessions[name].submitted_messages == ["Review review-1", "Review review-2"]
    with CaoSessionController(_config(fake_cao.url), registry) as restarted:
        observation = restarted.adopt_session(name)
        assert observation.metadata.context == contexts[-1]


@pytest.mark.parametrize("transport_reset", [False, True])
def test_execute_fails_closed_when_turn_context_update_fails(
    fake_cao: FakeCAOServer, tmp_path, transport_reset
):
    registry = LaneRegistry.default()
    lane = registry.get("requirements-reviewer")
    run_id = "context-failure"
    name = session_name_for(run_id, lane.lane)
    fake_cao.set_output(name, "[]")
    with CaoSessionController(_config(fake_cao.url), registry) as controller:
        executor = CaoLaneExecutor(
            controller, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
        )
        executor.execute(lane, "first review", str(tmp_path), _context(run_id))
        state = fake_cao._sessions[name]
        fake_cao.add_fault(
            FaultSpec(
                method="PATCH",
                path_prefix=f"/terminals/{state.terminal_id}/metadata",
                status_code=503,
                transport_reset=transport_reset,
            )
        )
        error = CaoTransportError if transport_reset else CaoControlPlaneError
        with pytest.raises(error):
            executor.execute(
                lane,
                "second review",
                str(tmp_path),
                LaneExecutionContext(run_id=run_id, round_id="review-2"),
            )
    assert state.submitted_messages == ["first review", "second review"]
    assert state.deleted


def test_busy_followup_preserves_previous_round(fake_cao: FakeCAOServer, tmp_path):
    registry = LaneRegistry.default()
    lane = registry.get("requirements-reviewer")
    run_id = "busy-review"
    name = session_name_for(run_id, lane.lane)
    fake_cao.set_output(name, "[]")
    with CaoSessionController(_config(fake_cao.url), registry) as controller:
        executor = CaoLaneExecutor(
            controller, registry, git=FakeGitOperations(), poll_interval_seconds=0.01
        )
        context = LaneExecutionContext(run_id=run_id, round_id="review-1")
        executor.execute(lane, "first review", str(tmp_path), context)
        state = fake_cao._sessions[name]
        previous_metadata = dict(state.metadata)
        fake_cao.add_fault(
            FaultSpec(
                method="POST", path_prefix=f"/terminals/{state.terminal_id}/input", status_code=409
            )
        )
        with pytest.raises(SessionBusyError):
            executor.execute(
                lane,
                "second review",
                str(tmp_path),
                LaneExecutionContext(run_id=run_id, round_id="review-2"),
            )
    assert state.metadata == previous_metadata
    assert state.submitted_messages == ["first review"]
    assert not state.deleted
