"""Real-local CAO control-plane integration test.

Opt-in only: it needs a `cao` binary, a running CAO server, and a provisioned
agent profile, so it stays skipped in CI and on a laptop that has none of
those. Enable with::

    AIPO_CAO_INTEGRATION=1 uv run pytest tests/integration/test_v3_cao_local.py

It runs the `developer` lane, so the `aipro-developer` agent profile must be
provisioned; see ``docs/V3_CAO.md``. Override the endpoint with
``AIPO_CAO_BASE_URL`` (default derived from ``CAO_API_PORT``).

The session it launches is deliberately trivial and deterministic — the agent
is asked to echo one fixed token — so the assertion is about the control
plane's lifecycle contract, not about model behaviour.
"""

from __future__ import annotations

import os
import shutil
import time

import httpx
import pytest

from ai_pr_orchestrator.v3.cao import CaoSessionController, session_name_for
from ai_pr_orchestrator.v3.config import CAOControlPlaneConfig
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, SessionSpec
from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE, LaneRegistry

MARKER = "AIPRO-CAO-OK"
POLL_INTERVAL_SECONDS = 2.0
POLL_BUDGET_SECONDS = 180.0


def _base_url() -> str:
    explicit = os.environ.get("AIPO_CAO_BASE_URL")
    if explicit:
        return explicit
    return f"http://localhost:{os.environ.get('CAO_API_PORT', '9889')}"


def _control_plane_is_up(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url}/sessions", timeout=5.0)
    except httpx.HTTPError:
        return False
    return response.is_success


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AIPO_CAO_INTEGRATION") != "1",
        reason="set AIPO_CAO_INTEGRATION=1 to run against a real local CAO",
    ),
    pytest.mark.skipif(shutil.which("cao") is None, reason="cao binary not on PATH"),
]


@pytest.fixture
def controller(tmp_path):
    base_url = _base_url()
    if not _control_plane_is_up(base_url):
        pytest.skip(f"no CAO control plane answering at {base_url}")
    config = CAOControlPlaneConfig(base_url=base_url, session_timeout_seconds=600)
    with CaoSessionController(config, LaneRegistry.default()) as controller:
        yield controller, tmp_path


def test_trivial_session_runs_to_completion_and_is_reconcilable(controller):
    session_controller, workdir = controller
    lane = LaneRegistry.default().get(DEVELOPER_LANE)
    run_id = f"it-{int(time.time())}"
    spec = SessionSpec(
        lane=lane,
        run_id=run_id,
        workdir=str(workdir),
        env={},
        context=LaneExecutionContext(run_id=run_id),
        command=f"Reply with exactly this token and nothing else: {MARKER}",
    )

    handle = session_controller.start_session(spec)
    try:
        assert handle.session_id == session_name_for(run_id, DEVELOPER_LANE)

        # Reconcile: a second start must adopt, never create a twin.
        assert session_controller.start_session(spec).session_id == handle.session_id

        # A fresh controller sees only the durable name, exactly as a restarted
        # process would, and must rebuild the full metadata from CAO.
        with CaoSessionController(
            CAOControlPlaneConfig(base_url=_base_url()), LaneRegistry.default()
        ) as restarted:
            adopted = restarted.adopt_session(handle.session_id)
        assert adopted.metadata.lane == lane
        assert adopted.metadata.context.run_id == run_id
        assert adopted.metadata.workdir == str(workdir)

        result = _poll_until_terminal(session_controller, handle)
        assert result.exit_code == 0
        assert MARKER in result.output_summary
    finally:
        session_controller.terminate_session(handle)


def _poll_until_terminal(session_controller, handle):
    deadline = time.monotonic() + POLL_BUDGET_SECONDS
    while time.monotonic() < deadline:
        result = session_controller.poll_session(handle)
        if result is not None:
            return result
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"session {handle.session_id!r} did not finish within {POLL_BUDGET_SECONDS}s")
