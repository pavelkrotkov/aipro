"""Conflicting reviewer claims require different-family independent adjudication."""

import json
from dataclasses import replace

from ai_pr_orchestrator.v3.cao import session_name_for
from tests.integration._harness import developer_output, developer_session_name


def test_scenario_4_reviewer_disagreement_picks_stronger(fake_cao, foreman_harness):
    loop, queue, github = foreman_harness()
    loop._cfg = replace(
        loop._cfg,
        safety=replace(
            loop._cfg.safety,
            max_coder_invocations_per_run=2,
            max_commits_per_run=2,
            max_reviewer_triggers_per_run=6,
        ),
    )
    ids = ("eager", "lazy")
    lanes = ("requirements-reviewer", "breaker-reviewer")
    proposals = [
        {"finding_id": fid, "action": "rebut", "rationale": "reproduction disproves this premise"}
        for fid in ids
    ]
    fake_cao.set_output_sequence(
        developer_session_name(loop.run_id),
        [developer_output(), developer_output(dispositions=proposals)],
    )
    for fid, lane in zip(ids, lanes, strict=True):
        finding = {
            "id": fid,
            "body": "Conflicting validation requirement",
            "severity": "major",
            "path": "src/validation.py",
            "line": 7,
            "claim": fid,
        }
        fake_cao.set_output_sequence(
            session_name_for(loop.run_id, lane), [json.dumps([finding]), "[]"]
        )
    decisions = [
        {
            "finding_id": fid,
            "action": "accept",
            "rationale": "independent experiment confirms rebuttal",
            "response_to_round_id": "response-1",
        }
        for fid in ids
    ]
    fake_cao.set_output_sequence(
        session_name_for(loop.run_id, "architecture-reviewer"),
        ["[]", json.dumps({"findings": [], "dispositions": decisions})],
    )

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 2 and outcome.review_rounds == 2
    assert len(github.list_open_prs()) == 1
    state = queue.load_state("owner/repo#1")
    assert [(d.finding_id, d.action) for d in state.dispositions] == [
        ("eager", "rebut"),
        ("lazy", "rebut"),
        ("eager", "accept"),
        ("lazy", "accept"),
    ]
    assert {d.decided_by for d in state.dispositions if d.action == "accept"} == {
        "architecture-reviewer"
    }
    assert not state.findings and len(state.archived) == 2
    first = state.extras["review_waves"]["review-1"]["lanes"]
    final = state.extras["review_waves"]["review-2"]["lanes"]["architecture-reviewer"]
    catalog = loop._executor._catalog
    adjudicator = catalog.get(final["assignment"]["model_ref"])
    origins = [catalog.get(first[lane]["assignment"]["model_ref"]) for lane in lanes]
    assert all(adjudicator.family != origin.family for origin in origins)
    assert final["routing"]["demand"]["independent_of"]
    assert adjudicator.quality_by_role["reviewer"] >= 4
