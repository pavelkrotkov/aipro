"""P4 scenario 6 prerequisite: CAO launch 429 escalates without losing state.

A CAO control-plane 429 is not proof of Hermes provider fallback.
"""

from tests.integration._fake_cao_server import FaultSpec


def test_control_plane_429_retains_durable_claim(fake_cao, foreman_harness):
    fake_cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=429))
    loop, queue, github = foreman_harness()

    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "429" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"
    assert state.run_id == loop.run_id
    assert "lease_expires_at" in state.extras
    assert "v3-work-needs-human" in github.get_labels(1)
    assert github.list_open_prs() == []
