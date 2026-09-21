"""Scenario 2: real CAO fix claim -> explicit reviewer acceptance -> CI/PR."""

from ai_pr_orchestrator.v3.cao import session_name_for
from tests.integration._harness import developer_output, developer_session_name, script_protocol


def test_scenario_2_blocking_finding_triggers_fix_round(fake_cao, foreman_harness):
    loop, _, github = foreman_harness()
    script_protocol(fake_cao, loop, proposal="fix")
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 2
    assert outcome.review_rounds == 2
    assert len(github.list_open_prs()) == 1


def test_scenario_2_finding_disposition_history_persists(fake_cao, foreman_harness):
    loop, queue, _ = foreman_harness(seed_issue_numbers=[2])
    script_protocol(fake_cao, loop, proposal="fix", issue_number=2)
    assert loop.run_pass()[0].final_phase == "done"
    state = queue.load_state("owner/repo#2")
    assert state.phase == "done"
    assert [(d.finding_id, d.action) for d in state.dispositions] == [
        ("guard", "fix"),
        ("guard", "accept"),
    ]
    assert all(d.rationale for d in state.dispositions)
    assert state.findings == []
    assert [(f.finding_id, f.status) for f in state.archived] == [("guard", "accepted")]


def test_scenario_2_no_finding_terminates_quickly(fake_cao, foreman_harness, lane_registry):
    loop, _, _ = foreman_harness(seed_issue_numbers=[3])
    for lane in lane_registry:
        name = (
            developer_session_name(loop.run_id, 3)
            if lane.role == "worker"
            else session_name_for(loop.run_id, lane.lane)
        )
        fake_cao.set_output(name, developer_output() if lane.role == "worker" else "[]")
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 1
    assert outcome.review_rounds == 1
