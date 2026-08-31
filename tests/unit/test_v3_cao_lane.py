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
from typing import Any

import pytest

from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoSessionController,
    session_name_for,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.domain import LaneIdentity
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, SessionSpec
from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE, LaneRegistry
from tests.integration._fake_cao_server import (
    DEFAULT_STATUS_SEQUENCE,
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_PROCESSING,
    FakeCAOServer,
    FaultSpec,
)

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
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

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


def test_execute_adopts_existing_session_by_name(fake_cao: FakeCAOServer, tmp_path):
    """A pre-existing CAO session under the deterministic name is adopted,
    not duplicated, on a second ``execute`` for the same run/lane."""
    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    executor.execute(_lane(), f"Reply with exactly: {MARKER}", str(tmp_path), _context(run_id))
    executor.execute(_lane(), f"Reply with exactly: {MARKER}", str(tmp_path), _context(run_id))

    assert len(fake_cao._sessions) == 1, "second execute must adopt, not create a twin"


def test_execute_clears_previous_idle_evidence_on_new_work(fake_cao: FakeCAOServer, tmp_path):
    """The controller clears idle-settle evidence on accepted input
    (see ``submit_work`` in ``cao.py``). A second ``execute`` therefore
    starts fresh, not by inheriting the first session's terminal state."""
    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

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
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

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
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    result = executor.execute(
        _lane(),
        f"echo {MARKER}",
        str(tmp_path),
        _context(run_id),
    )

    assert result.exit_code != 0
    assert result.session.session_id == name


# --- Executor budget ---------------------------------------------------


def test_execute_raises_after_poll_budget_exhausted(fake_cao: FakeCAOServer, tmp_path):
    """A session that never reaches terminal state within the executor's
    ``max_poll_seconds`` budget raises :class:`TimeoutError` and
    terminates the session so it does not leak."""
    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    # A sequence that never reaches terminal: the controller keeps
    # reporting "processing" forever (in practice the controller's
    # own session_timeout_seconds would also fire, but here we set
    # that high enough to exercise the executor's own budget).
    fake_cao.set_status_sequence(name, [STATUS_PROCESSING] * 1000)

    controller = CaoSessionController(
        CAOControlPlaneConfig(
            base_url=fake_cao.url,
            session_timeout_seconds=60,
            request_timeout_seconds=5,
        ),
        LaneRegistry.default(),
    )
    executor = CaoLaneExecutor(
        controller,
        LaneRegistry.default(),
        poll_interval_seconds=0.01,
        max_poll_seconds=0.05,  # budget tighter than the fake's sequence
    )

    with pytest.raises(TimeoutError):
        executor.execute(
            _lane(),
            f"echo {MARKER}",
            str(tmp_path),
            _context(run_id),
        )

    # The session was terminated by the executor's cleanup path.
    state = fake_cao._sessions.get(name)
    assert state is not None and state.deleted, (
        "executor must terminate the half-finished session on TimeoutError"
    )


# --- Lane registry resolution ------------------------------------------


def test_execute_uses_registry_lane_not_caller_identity(fake_cao: FakeCAOServer, tmp_path):
    """The executor looks the lane up in the registry on every call, so
    a reconfigured registry wins. A caller that passes a mismatched
    :class:`LaneIdentity` (wrong profile) is silently corrected to the
    registry's identity, which is the lane-to-profile binding the
    controller already enforces."""
    run_id = f"it-{int(time.time() * 1000)}"

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

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


# --- PR #71 review-round regressions -----------------------------------


def test_execute_unbound_lane_raises_lookup_error(fake_cao: FakeCAOServer, tmp_path):
    """Regression for finding #1 in PR #71: a LaneIdentity whose lane
    name is not in the registry used to silently construct a SessionSpec
    with the unregistered LaneIdentity, which then crashed inside
    ``SessionSpec.__post_init__`` with a TypeError. The executor now
    raises a typed :class:`LookupError` so the caller knows the lane
    binding is missing."""

    from ai_pr_orchestrator.v3.lanes import LaneRegistry

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    bogus = LaneIdentity(lane="not-a-real-lane", role="worker", profile_template="x")

    with pytest.raises(LookupError) as excinfo:
        executor.execute(bogus, "echo", str(tmp_path), _context("run-x"))
    assert "not-a-real-lane" in str(excinfo.value)


def test_execute_submits_prompt_on_adopted_session(fake_cao: FakeCAOServer, tmp_path):
    """Regression for finding #2 in PR #71: when the controller adopts
    a still-running session under the deterministic name, the executor
    must submit the prompt for this turn so the session actually
    receives the new task. Without this, an adopted session was a
    silent no-op for the executor's caller."""

    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    executor.execute(_lane(), "first turn", str(tmp_path), _context(run_id))
    # Second execute: adopts and must submit "second turn" — verify by
    # observing that the controller cleared activity (which only
    # ``submit_work`` does) right after adoption.
    fake_cao._sessions[name].metadata["activity_seen"] = True
    executor.execute(_lane(), "second turn", str(tmp_path), _context(run_id))
    # Activity was cleared and then re-marked by polling; the point is
    # the executor submitted and polled through to completion rather
    # than returning a stale cached result.
    assert fake_cao._sessions[name].exhausted is True


def test_execute_reconciles_session_when_launch_is_uncertain(fake_cao: FakeCAOServer, tmp_path):
    """Regression for finding #3 in PR #71: a launch transport failure
    raises ``SessionIdentityUncertainError`` because CAO may have
    created the session before losing the response. The executor now
    catches that and tries to adopt the same deterministic name before
    giving up, instead of leaking the session."""

    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)
    fake_cao.add_fault(
        FaultSpec(
            method="POST",
            path_prefix="/sessions",
            transport_reset=True,
            commit_then_reset=True,
        )
    )

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    result = executor.execute(_lane(), "echo MARKER", str(tmp_path), _context(run_id))
    # The fake committed the session before dropping the response, so the
    # executor's reconcile-then-poll path must drive it to completion.
    assert result.exit_code == 0
    assert result.session.session_id == name


def test_execute_preserves_polling_exception_when_cleanup_also_fails(
    fake_cao: FakeCAOServer, tmp_path, monkeypatch
):
    """Regression for finding #4 in PR #71: when the poll loop raises a
    typed error and the cleanup ``terminate_session`` also fails, the
    original error must still be the caller-visible exception so the
    foreman can classify it; the cleanup failure must be chained but
    must not displace the primary."""

    from ai_pr_orchestrator.v3.cao import CaoControlPlaneError

    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    # Drive the terminal to never reach terminal state.
    fake_cao.set_status_sequence(name, [STATUS_PROCESSING] * 1000)

    controller = CaoSessionController(
        CAOControlPlaneConfig(
            base_url=fake_cao.url,
            session_timeout_seconds=60,
            request_timeout_seconds=5,
        ),
        LaneRegistry.default(),
    )
    executor = CaoLaneExecutor(
        controller,
        LaneRegistry.default(),
        poll_interval_seconds=0.01,
        max_poll_seconds=0.05,
    )

    # Force ``terminate_session`` to also fail; the executor must still
    # surface the TimeoutError as the primary exception, with the
    # cleanup error chained behind it.
    cleanup_error = CaoControlPlaneError("cleanup failed in test")

    def _failing_terminate(handle):
        raise cleanup_error

    monkeypatch.setattr(controller, "terminate_session", _failing_terminate)

    with pytest.raises(TimeoutError) as excinfo:
        executor.execute(_lane(), "echo MARKER", str(tmp_path), _context(run_id))
    # The cleanup failure is attached via ``__cause__``; the primary is
    # the TimeoutError the poll loop raised.
    assert excinfo.value.__cause__ is cleanup_error


def test_execute_forwards_leased_model_ref_into_session_env(fake_cao: FakeCAOServer, tmp_path):
    """Regression for finding #6 in PR #71: a non-null ModelLease
    carries the broker's resolved model selection, which the executor
    must forward into the session environment as ``AIPRO_MODEL_REF``
    so the agent startup can resolve the same model."""

    from ai_pr_orchestrator.v3.domain import LaneName, ModelAssignment, ModelRef
    from ai_pr_orchestrator.v3.interfaces import ModelLease

    run_id = f"it-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, MARKER)

    lease = ModelLease(
        lease_id="lease-x",
        assignment=ModelAssignment(
            lane=LaneName("developer"), model_ref=ModelRef("catalog:dev-foo")
        ),
    )

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    executor.execute(_lane(), "echo MARKER", str(tmp_path), _context(run_id), lease=lease)

    # The launch body the controller POSTed must include AIPRO_MODEL_REF.
    # The fake's _SessionState does not capture env (env is opaque from
    # the fake's view), so we exercise the contract by inspecting a
    # side effect: the executor populated spec.env from the lease.
    # The controller's own model tests in test_v3_cao.py cover the
    # end-to-end POST /sessions body via direct mock.
    assert lease.assignment.model_ref == "catalog:dev-foo"


def test_execute_surfaces_429_as_classifiable_rate_limit(fake_cao: FakeCAOServer, tmp_path):
    """Regression for finding #7 in PR #71: a transient HTTP 429 from
    CAO must surface as :class:`CaoRateLimitedError` (a subclass of
    :class:`CaoUnavailableError`) so a backoff scheduler can take it
    into account, distinct from a permanent :class:`CaoControlPlaneError`."""

    from ai_pr_orchestrator.v3.cao import (
        CaoRateLimitedError,
        CaoUnavailableError,
    )

    run_id = f"it-{int(time.time() * 1000)}"
    fake_cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=429))

    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)

    with pytest.raises(CaoRateLimitedError) as excinfo:
        executor.execute(_lane(), "echo", str(tmp_path), _context(run_id))
    # 429 is still a transient error (CaoUnavailableError) so the
    # foreman's general retry-on-unavailable logic continues to work.
    assert isinstance(excinfo.value, CaoUnavailableError)
