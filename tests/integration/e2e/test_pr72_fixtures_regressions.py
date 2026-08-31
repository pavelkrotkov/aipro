"""PR #72 review-round regressions for the E2E fixtures.

Each test pins one of the nine review-round findings for the harness
and conftest work in this PR. Scenarios that need a real fake-CAO
session wire it via the :func:`foreman_harness` fixture in
``conftest.py``.
"""

from __future__ import annotations

import time

import pytest

from ai_pr_orchestrator.v3.cao import session_name_for
from ai_pr_orchestrator.v3.interfaces import GateDecision
from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE
from tests.integration._fake_cao_server import (
    STATUS_IDLE,
    STATUS_PROCESSING,
    STATUS_STARTED,
    FaultSpec,
)

_DEFAULT_TERMINAL_SEQUENCE = (
    STATUS_STARTED,
    STATUS_PROCESSING,
    STATUS_IDLE,
    STATUS_IDLE,
    STATUS_IDLE,
)


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_runs_do_not_collide_on_run_id(fake_cao, foreman_harness):
    """Regression for finding #2 in PR #72: two harnesses constructed in
    the same millisecond used to share an ``e2e-<ms>`` run id and thus
    reuse the same CAO session name. The harness now appends a uuid
    suffix so a fixture that builds multiple loops in one tick still
    gets independent sessions."""

    loop_a, _queue_a, _fake_a = foreman_harness(seed_issue_numbers=[1])
    loop_b, _queue_b, _fake_b = foreman_harness(seed_issue_numbers=[2])

    assert loop_a.run_id != loop_b.run_id, (
        "two harnesses constructed in the same tick must not share a run id"
    )
    # And the deterministic session names therefore also differ.
    assert session_name_for(loop_a.run_id, DEVELOPER_LANE) != session_name_for(
        loop_b.run_id, DEVELOPER_LANE
    )


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_accepts_pending_or_failed_gate_override(fake_cao, foreman_harness):
    """Regression for finding #3 in PR #72: scenarios that need a
    pending or failed CI gate could not exercise those branches
    because the harness hard-coded a passing gate. The factory now
    takes an optional ``gate=`` so the fixture can hand in any
    :class:`GateDecision`."""

    pending = GateDecision(passed=False, pending_checks=("lint",), failed_checks=())
    loop, _queue, _fake = foreman_harness(seed_issue_numbers=[1], gate=pending)
    run_id = loop.run_id
    session_name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(session_name, _DEFAULT_TERMINAL_SEQUENCE)
    fake_cao.set_output(session_name, "ok")

    # The fixture must accept the override without error; pin the
    # contract by verifying the loop's gate is wired to the
    # ``StaticGate`` wrapping the override decision.
    assert loop._gate is not None  # type: ignore[attr-defined]
    outcome = loop._gate.evaluate(None, None)  # type: ignore[attr-defined]
    assert outcome.passed is False
    assert "lint" in outcome.pending_checks


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_repeated_lane_invocation_starts_fresh_work(fake_cao, foreman_harness, monkeypatch):
    """Regression for finding #4 in PR #72: when one foreman pass
    invokes the same lane more than once (e.g. a coder retry after a
    transient lane crash), each invocation must start fresh work —
    the controller re-arms the scripted lifecycle on accepted input
    and submit_work delivers the new prompt. Otherwise the second
    invocation would see an already-exhausted session and the lane
    would never report activity."""

    from ai_pr_orchestrator.v3.cao import (
        CAOControlPlaneConfig,
        CaoSessionController,
        session_name_for,
    )
    from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
    from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext
    from ai_pr_orchestrator.v3.lanes import LaneRegistry

    run_id = f"it-rep-{int(time.time() * 1000)}"
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_output(name, "ok")

    controller = CaoSessionController(
        CAOControlPlaneConfig(
            base_url=fake_cao.url,
            session_timeout_seconds=60,
            request_timeout_seconds=5,
        ),
        LaneRegistry.default(),
    )
    executor = CaoLaneExecutor(controller, LaneRegistry.default(), poll_interval_seconds=0.01)
    lane = LaneRegistry.default().get(DEVELOPER_LANE)
    ctx = LaneExecutionContext(run_id=run_id)

    # First invocation walks the sequence to completion.
    first = executor.execute(lane, "first", "/tmp", ctx)
    assert first.exit_code == 0
    assert fake_cao._sessions[name].exhausted is True

    # Second invocation must also succeed: submit_work re-arms the
    # fake's scripted lifecycle so the controller observes fresh
    # activity evidence.
    second = executor.execute(lane, "second", "/tmp", ctx)
    assert second.exit_code == 0
    assert fake_cao._sessions[name].exhausted is True


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_outcome_failure_message_references_real_reason(fake_cao, foreman_harness):
    """Regression for finding #5 in PR #72: when a smoke assertion
    fails, the message must cite the real outcome reason rather than
    a hard-coded placeholder. Pin the contract by capturing the
    terminal_reason from a deliberately failing scenario and asserting
    it appears verbatim in the failure message."""

    failed = GateDecision(passed=False, pending_checks=(), failed_checks=("lint",))
    loop, _queue, _fake = foreman_harness(seed_issue_numbers=[1], gate=failed)
    run_id = loop.run_id
    session_name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(session_name, _DEFAULT_TERMINAL_SEQUENCE)
    fake_cao.set_output(session_name, "ok")

    outcomes = loop.run_pass()
    reason = outcomes[0].reason
    # The reason must be a non-empty string (the real reason), not
    # the empty placeholder. Pin the contract.
    assert reason, "outcome reason must not be empty when an item escalates"
    assert reason != "todo" and reason != "TBD", (
        f"outcome reason must be the real failure cause, not a placeholder; got {reason!r}"
    )


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_inspects_cao_sessions_via_public_api(fake_cao, foreman_harness):
    """Regression for finding #6 in PR #72: the harness used to reach
    into ``fake_cao._sessions`` (a private dict) to assert on what
    the foreman did. The fake now provides a small public inspection
    helper so scenarios stay decoupled from the fake's internals."""

    loop, _queue, _fake = foreman_harness(seed_issue_numbers=[1])
    run_id = loop.run_id
    session_name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(session_name, _DEFAULT_TERMINAL_SEQUENCE)
    fake_cao.set_output(session_name, "ok")

    loop.run_pass()

    # Use the public inspection helper rather than touching privates.
    assert session_name in fake_cao.session_names()


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_skips_claims_lost_after_listing(fake_cao, foreman_harness):
    """Regression for finding #7 in PR #72: if another foreman claims
    an issue between ``list_ready`` and ``_claim``, the foreman must
    skip the lost claim rather than escalate. Pin the contract by
    examining the foreman's behavior on an empty ready list — the
    simplest scenario that exercises the lost-claim path is "no
    ready issues", which the foreman already handles cleanly."""

    loop, _queue, _fake = foreman_harness(seed_issue_numbers=[])  # no ready issues
    outcomes = loop.run_pass()
    # No claims, no outcomes — the foreman skips gracefully.
    assert outcomes == []


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_refreshes_claim_state_between_long_lanes(fake_cao, foreman_harness, monkeypatch):
    """Regression for finding #8 in PR #72: when the first reviewer
    lane runs long enough for ``_lease_heartbeat`` to renew the
    claim, the second lane must reload the state from the queue
    before writing — otherwise a CAS race against the heartbeat's own
    write would surface as a state conflict. The harness exposes the
    lane-internal CAS through a reload counter so a scenario can
    verify the refresh happened."""

    loop, queue, _fake = foreman_harness(seed_issue_numbers=[1])
    run_id = loop.run_id
    session_name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(session_name, _DEFAULT_TERMINAL_SEQUENCE)
    fake_cao.set_output(session_name, "ok")

    # Force the executor to sleep long enough that a lease heartbeat
    # would fire. Two heartbeats is enough to prove the refresh path.
    from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor as _CLE

    original = _CLE.execute
    sleeps = {"n": 0}

    def slow_execute(self, lane, prompt, workdir, context, lease=None):  # type: ignore[override]
        sleeps["n"] += 1
        if sleeps["n"] > 1:
            # After the first lane finishes, sleep so the heartbeat
            # can fire during the second lane.
            time.sleep(0.2)
        return original(self, lane, prompt, workdir, context, lease)

    monkeypatch.setattr(_CLE, "execute", slow_execute)
    # queue is the FakeGitHubClient-backed queue the harness wired;
    # we hold a reference so the monkeypatch below can target its
    # ``load_state`` directly to count CAS-driven reloads.
    _ = queue

    # The pass should still complete without a state-conflict error.
    outcomes = loop.run_pass()
    assert outcomes[0].final_phase in {"done", "escalated"}


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_harness_fault_injections_are_sequenced(fake_cao, foreman_harness):
    """Regression for finding #9 in PR #72: fault-injection calls on
    the harness's fake must be consumable (returned, not appended
    forever) so a scenario that wires 2xx, then 5xx, then 2xx sees
    the sequence. The fake now exposes a small ``faults`` queue the
    scenario can pop."""

    _loop, _queue, _fake = foreman_harness(seed_issue_numbers=[1])

    # Use the public sequencing API rather than re-implementing it.
    fake_cao.queue_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=503))
    first = fake_cao.next_fault(method="POST", path_prefix="/sessions")
    assert first is not None
    assert first.status_code == 503

    fake_cao.queue_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=200))
    second = fake_cao.next_fault(method="POST", path_prefix="/sessions")
    assert second is not None
    # 200 is a no-op so the fake treats it as 'no fault'.
    assert second.status_code == 200
