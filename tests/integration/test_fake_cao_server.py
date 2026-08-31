"""Integration test: the real ``CaoSessionController`` against ``FakeCAOServer``.

Issue #55, P1 (E2E harness). Validates that the fake matches the real CAO
contract on the surface the controller exercises. If this test breaks, the
fake has drifted from CAO's response shapes and the E2E scenarios in #55
must be re-checked against a real ``cao-server`` before merging.

Runs in plain CI (no real CAO required): opt-in only for the real-network
CAO test in ``test_v3_cao_local.py``.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoSessionController,
    session_name_for,
)
from ai_pr_orchestrator.v3.domain import LaneIdentity  # noqa: F401
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, SessionSpec
from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE, LaneRegistry
from tests.integration._fake_cao_server import (
    DEFAULT_STATUS_SEQUENCE,
    STATUS_IDLE,  # noqa: F401
    STATUS_PROCESSING,  # noqa: F401
    STATUS_STARTED,
    FakeCAOServer,
)

#: Sentinel token the agent is asked to echo. The controller's
#: ``final_output`` returns whatever the fake stored on the session, so the
#: assertion is on the contract (final_output returns what CAO returned) and
#: not on the agent's behaviour.
MARKER = "AIPRO-FAKE-CAO-OK"


def _control_plane(url: str) -> CAOControlPlaneConfig:
    return CAOControlPlaneConfig(
        base_url=url,
        session_timeout_seconds=60,
        request_timeout_seconds=5,
    )


@pytest.fixture
def fake_cao() -> Any:
    with FakeCAOServer() as cao:
        yield cao


def _spec(run_id: str, workdir: str) -> SessionSpec:
    lane = LaneRegistry.default().get(DEVELOPER_LANE)
    return SessionSpec(
        lane=lane,
        run_id=run_id,
        workdir=workdir,
        env={},
        context=LaneExecutionContext(run_id=run_id),
        command=f"Reply with exactly: {MARKER}",
    )


def test_post_sessions_creates_session_with_byte_identical_name(fake_cao: FakeCAOServer, tmp_path):
    """POST /sessions returns the requested session_name verbatim so a
    second ``start_session`` for the same spec adopts instead of creating a
    twin."""
    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    run_id = f"it-{int(time.time() * 1000)}"
    expected = session_name_for(run_id, DEVELOPER_LANE)

    handle = controller.start_session(_spec(run_id, str(tmp_path)))

    assert handle.session_id == expected
    assert fake_cao._sessions[expected].terminal_id  # server issued a terminal id


def test_relaunch_same_spec_adopts_instead_of_creating_twin(fake_cao: FakeCAOServer, tmp_path):
    """A second ``start_session`` for the same deterministic name must
    return the existing handle, not a new one."""
    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    run_id = f"it-{int(time.time() * 1000)}"

    first = controller.start_session(_spec(run_id, str(tmp_path)))
    second = controller.start_session(_spec(run_id, str(tmp_path)))

    assert first.session_id == second.session_id
    assert len(fake_cao._sessions) == 1, "relaunch must not create a second session"


def test_metadata_round_trips_through_launch_and_get(fake_cao: FakeCAOServer, tmp_path):
    """Metadata pushed in the launch body must come back through
    ``GET /terminals/{tid}`` so ``adopt_session`` can rebuild session
    metadata after a process restart. This test exercises the real HTTP
    round-trip via the live fake server, not just the in-memory state."""
    run_id = f"it-{int(time.time() * 1000)}"
    expected_name = session_name_for(run_id, DEVELOPER_LANE)

    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    handle = controller.start_session(_spec(run_id, str(tmp_path)))

    terminal_id = fake_cao._sessions[handle.session_id].terminal_id
    response = httpx.get(f"{fake_cao.url}/terminals/{terminal_id}", timeout=5.0)
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_name"] == expected_name
    assert payload["metadata"], "metadata must round-trip through the fake"
    assert payload["metadata"].get("workdir") == str(tmp_path)


def test_observe_walks_status_sequence_to_completed(fake_cao: FakeCAOServer, tmp_path):
    """The controller's ``observe`` walks the terminal's status sequence
    and reports ``completed`` once the idle-settle threshold is met."""
    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    run_id = f"it-{int(time.time() * 1000)}"

    handle = controller.start_session(_spec(run_id, str(tmp_path)))

    # First observe sees the first status in DEFAULT (started).
    first = controller.observe(handle)
    assert first.cao_status == STATUS_STARTED
    assert first.state == "started"

    # Walk the sequence; the last two are idle, which the controller's
    # _lifecycle rule maps onto completed after the idle-settle threshold.
    observations = [first]
    for _ in range(len(DEFAULT_STATUS_SEQUENCE) - 1):
        obs = controller.observe(handle)
        observations.append(obs)
        if obs.is_terminal:
            break
    final = observations[-1]
    assert final.state == "completed", (
        f"expected 'completed' after idle-settle, got {final.state!r} "
        f"from sequence {[o.cao_status for o in observations]}"
    )


def test_final_output_returns_what_cao_returned(fake_cao: FakeCAOServer, tmp_path):
    """``final_output`` returns CAO's last-response field verbatim, never
    the controller's own read of the terminal."""
    run_id = f"it-{int(time.time() * 1000)}"
    fake_cao.set_output(session_name_for(run_id, DEVELOPER_LANE), MARKER)

    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    handle = controller.start_session(_spec(run_id, str(tmp_path)))

    output = controller.final_output(handle)
    assert output == MARKER


def test_terminate_session_makes_subsequent_lookup_404(fake_cao: FakeCAOServer, tmp_path):
    """``terminate_session`` deletes the session so a later ``adopt_session``
    by name raises ``CaoSessionNotFoundError``."""
    from ai_pr_orchestrator.v3.cao import CaoSessionNotFoundError

    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    run_id = f"it-{int(time.time() * 1000)}"
    expected = session_name_for(run_id, DEVELOPER_LANE)

    handle = controller.start_session(_spec(run_id, str(tmp_path)))
    controller.terminate_session(handle)

    with pytest.raises(CaoSessionNotFoundError):
        controller.adopt_session(expected)


def test_fault_injection_returns_injected_status(fake_cao: FakeCAOServer, tmp_path):
    """A configured ``FaultSpec`` causes the matching request to return the
    injected HTTP status; the controller maps non-2xx to a control-plane
    error."""
    from ai_pr_orchestrator.v3.cao import CaoSessionNotFoundError
    from tests.integration._fake_cao_server import FaultSpec

    fake_cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=503))

    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    run_id = f"it-{int(time.time() * 1000)}"

    with pytest.raises(Exception) as excinfo:
        controller.start_session(_spec(run_id, str(tmp_path)))
    # 5xx (not 404/409) surfaces as a generic CaoControlPlaneError, NOT
    # CaoSessionNotFoundError; the assertion guards against regression.
    assert not isinstance(excinfo.value, CaoSessionNotFoundError)


def test_empty_status_sequence_returns_500_not_indexerror(
    fake_cao: FakeCAOServer, tmp_path
):
    """Regression for finding #1 in PR #70: a session scripted with an
    empty status sequence must surface as an HTTP 500 (an explicit
    misconfiguration) rather than raising ``IndexError`` inside the
    handler thread. The controller's HTTP error handling then bubbles up
    a control-plane error to the caller."""
    from tests.integration._fake_cao_server import _SessionState

    run_id = f"it-{int(time.time() * 1000)}"
    expected = session_name_for(run_id, DEVELOPER_LANE)

    # Force-launch a session whose status_sequence is empty.
    with fake_cao._lock:
        fake_cao._sessions[expected] = _SessionState(
            session_name=expected,
            terminal_id="term-empty-0001",
            workdir=str(tmp_path),
            agent_profile="",
            initial_message=None,
            metadata={},
            status_sequence=(),
        )

    response = httpx.get(f"{fake_cao.url}/terminals/term-empty-0001", timeout=5.0)
    assert response.status_code == 500
    assert "empty status sequence" in response.text
