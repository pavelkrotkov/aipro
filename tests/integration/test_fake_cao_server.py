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


def test_patch_metadata_is_persisted_and_visible_to_later_get(
    fake_cao: FakeCAOServer, tmp_path: Any
):
    """Regression for round-2 finding #4 in PR #71: a PATCH that flips
    ``activity_seen=True`` or updates per-turn context (``round_id``,
    ``work_item_id``) must be persisted on the server side so a
    controller that adopts the session after a restart can read the
    updated metadata. Unknown / deleted terminals return 404.

    The previous fake acknowledged every PATCH with 204 but discarded
    its body, so activity changes written by ``_mark_active`` and
    ``_clear_activity`` never reached the session state returned by
    later GETs and the idle-settle guard never armed during soak
    recovery tests.
    """

    run_id = f"it-{int(time.time() * 1000)}"
    controller = CaoSessionController(_control_plane(fake_cao.url), LaneRegistry.default())
    handle = controller.start_session(_spec(run_id, str(tmp_path)))
    terminal_id = fake_cao._sessions[handle.session_id].terminal_id

    # PATCH a metadata key.
    patch_resp = httpx.patch(
        f"{fake_cao.url}/terminals/{terminal_id}/metadata",
        json={"activity_seen": True, "round_id": 3},
        timeout=5.0,
    )
    assert patch_resp.status_code == 204

    # The next GET round-trips the merged metadata.
    get_resp = httpx.get(f"{fake_cao.url}/terminals/{terminal_id}", timeout=5.0)
    assert get_resp.status_code == 200
    payload = get_resp.json()
    assert payload["metadata"].get("activity_seen") is True
    assert payload["metadata"].get("round_id") == 3

    # PATCH to a deleted terminal must return 404.
    controller.terminate_session(handle)
    bad_resp = httpx.patch(
        f"{fake_cao.url}/terminals/{terminal_id}/metadata",
        json={"activity_seen": False},
        timeout=5.0,
    )
    assert bad_resp.status_code == 404


def test_status_200_fault_falls_through_to_real_handler(fake_cao: FakeCAOServer, tmp_path):
    """Regression for round-2 finding #5 in PR #71: a ``FaultSpec`` whose
    only effect is ``status_code=200`` is documented as a no-op and the
    dispatcher must continue to the real handler so the client
    receives the route's normal 200-series response. The previous
    implementation returned the fault object (because the
    ``status_code != 200`` branch was False) and the dispatcher then
    short-circuited, returning an empty response and making the
    controller treat baseline scenarios as transport failures.

    The same rule covers delay-only faults: a synthetic latency without
    a status override must not drop the response.
    """

    from tests.integration._fake_cao_server import FaultSpec

    # A 200-status "no-op" fault on /sessions.
    fake_cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=200))

    def _post(name: str, workdir: str) -> httpx.Response:
        return httpx.post(
            f"{fake_cao.url}/sessions",
            params={
                "session_name": name,
                "working_directory": workdir,
                "agent_profile": "",
            },
            json={
                "metadata": {
                    "workdir": workdir,
                    "lane": "developer",
                    "round_id": "1",
                    "work_item_id": "1",
                    "run_id": name,
                }
            },
            timeout=5.0,
        )

    workdir = str(tmp_path)
    first = _post("noop-1", workdir)
    second = _post("noop-2", workdir)
    # Both POSTs return the route's real 200 (session created), NOT a
    # dropped response. A regression that returns the fault object
    # would short-circuit the dispatcher and the response would be
    # empty / connection-reset.
    assert first.status_code == 200, (
        f"status_code=200 fault must fall through to the real handler; got {first.status_code}"
    )
    assert second.status_code == 200
    # Both sessions must have been actually created.
    assert "noop-1" in fake_cao._sessions
    assert "noop-2" in fake_cao._sessions

    # A delay-only fault (no status, no transport_reset) also must not
    # drop the response.
    fake_cao.clear_faults()
    fake_cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", delay_seconds=0.01))
    delayed = _post("delayed", str(tmp_path / "delayed"))
    assert delayed.status_code == 200, (
        f"delay-only fault must fall through to the real handler; got {delayed.status_code}"
    )


def test_commit_then_reset_still_drops_response(fake_cao: FakeCAOServer, tmp_path):
    """The commit_then_reset branch must keep its existing behaviour:
    the dispatcher runs the handler for its state-mutation effect and
    then drops the response so the client sees a transport error. This
    test guards the carve-out in test_status_200_fault_falls_through_to_real_handler
    so the no-op fix does not regress the committed-but-unacknowledged
    branch (PR #71 #5 fix must not regress finding #7 from PR #70)."""

    from tests.integration._fake_cao_server import FaultSpec

    fake_cao.add_fault(
        FaultSpec(
            method="POST",
            path_prefix="/sessions",
            transport_reset=True,
            commit_then_reset=True,
        )
    )

    workdir = str(tmp_path)
    # The server closes the connection without writing a response, so
    # httpx raises RemoteProtocolError. Catch it and assert that the
    # fake applied the handler's state mutation anyway.
    response_status: int | None = None
    response_empty = False
    raised: BaseException | None = None
    try:
        response = httpx.post(
            f"{fake_cao.url}/sessions",
            params={
                "session_name": "ctr-1",
                "working_directory": workdir,
                "agent_profile": "",
            },
            json={
                "metadata": {
                    "workdir": workdir,
                    "lane": "developer",
                    "round_id": "1",
                    "work_item_id": "1",
                    "run_id": "ctr-1",
                }
            },
            timeout=5.0,
        )
        response_status = response.status_code
        response_empty = not response.content
    except httpx.RemoteProtocolError as exc:
        raised = exc

    assert raised is not None or response_status != 200 or response_empty, (
        "commit_then_reset must still drop the response; got "
        f"status={response_status} empty={response_empty}"
    )
    assert "ctr-1" in fake_cao._sessions, (
        "commit_then_reset must still apply the handler's state mutation"
    )


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
