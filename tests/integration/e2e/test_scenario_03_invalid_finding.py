"""Scenario 3: real CAO rebuttal -> independent acceptance -> durable closure."""

from tests.integration._harness import script_protocol


def test_scenario_3_invalid_finding_triggers_rebuttal(fake_cao, foreman_harness):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop, proposal="rebut")
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 2
    assert outcome.review_rounds == 2
    state = queue.load_state("owner/repo#1")
    assert [d.action for d in state.dispositions] == ["rebut", "accept"]
    assert state.dispositions[0].decided_by != state.dispositions[1].decided_by
    assert state.dispositions[1].response_to_round_id == state.dispositions[0].round_id
    assert [f.finding_id for f in state.archived] == ["guard"]
    assert state.findings == []
    assert len(github.list_open_prs()) == 1
