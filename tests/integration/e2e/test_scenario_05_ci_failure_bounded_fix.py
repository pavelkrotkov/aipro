"""P4 scenario 5: a real CAO developer fixes red CI and gets re-reviewed."""

from dataclasses import replace

from ai_pr_orchestrator.v3.cao import session_name_for
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from tests.integration._harness import developer_output, developer_session_name


def test_ci_failure_is_fixed_and_re_reviewed_through_cao(fake_cao, foreman_harness, monkeypatch):
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
    check_runs = github.get_check_runs
    green = check_runs("sha")
    github.set_check_runs(
        "sha", [replace(green[0], conclusion="failure", summary="regression failed")]
    )
    observed = []

    def checks(ref):
        result = check_runs(ref)
        observed.append(result)
        github.set_check_runs(ref, green)
        return result

    monkeypatch.setattr(github, "get_check_runs", checks)
    developer = developer_session_name(loop.run_id)
    fake_cao.set_output_sequence(
        developer, [developer_output(), developer_output(summary="fixed failing CI")]
    )
    for lane in LaneRegistry.default():
        if lane.role == "reviewer":
            fake_cao.set_output(session_name_for(loop.run_id, lane.lane), "[]")

    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 2
    assert outcome.review_rounds == 2
    assert len(observed) == 2 and observed[0][0].conclusion == "failure"
    assert len(fake_cao._sessions[developer].submitted_messages) == 2
    assert queue.load_state("owner/repo#1").phase == "done"
    assert len(github.list_open_prs()) == 1
