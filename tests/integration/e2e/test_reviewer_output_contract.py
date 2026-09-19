"""Production reviewer parsing must gate PR creation, not just populate a test double."""

import pytest

from ai_pr_orchestrator.v3.cao import session_name_for


@pytest.mark.parametrize("output", ["", "There is a blocking defect in the patch"])
def test_unstructured_review_escalates_without_creating_pr(fake_cao, foreman_harness, output):
    loop, queue, github = foreman_harness()
    name = session_name_for(loop.run_id, "requirements-reviewer")
    fake_cao.set_output(name, output)

    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "Invalid reviewer output" in outcome.reason
    assert queue.load_state("owner/repo#1").phase == "escalated"
    assert github.list_open_prs() == []
    assert "Return only a JSON array" in fake_cao._sessions[name].submitted_messages[0]


def test_structured_blocker_reaches_real_policy(fake_cao, foreman_harness, lane_registry):
    loop, queue, github = foreman_harness()
    for lane in lane_registry:
        if lane.role == "reviewer":
            fake_cao.set_output(session_name_for(loop.run_id, lane.lane), "[]")
    fake_cao.set_output(
        session_name_for(loop.run_id, "requirements-reviewer"),
        '[{"id":"guard","body":"Blocking defect: missing guard","severity":"blocker"}]',
    )

    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert github.list_open_prs() == []
    assert "coder invocation budget exhausted with open findings" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert [(f.id, f.status) for f in state.findings] == [("guard", "open")]
    assert state.dispositions == []  # No coder response exists at the exhausted budget.
