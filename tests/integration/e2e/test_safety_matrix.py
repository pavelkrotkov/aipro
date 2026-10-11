"""Safety and stagnation gates through the production CAO-backed foreman."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ai_pr_orchestrator.v3.cao import session_name_for
from tests.integration._harness import developer_output, developer_session_name, script_protocol


def _clean(fake_cao, loop):
    fake_cao.set_output(developer_session_name(loop.run_id), developer_output())
    for lane in loop._lanes:
        if lane.role == "reviewer":
            fake_cao.set_output(session_name_for(loop.run_id, lane.lane), "[]")


@pytest.mark.parametrize("fork,disallow", [(False, True), (True, True), (True, False)])
def test_pr_fork_policy_is_load_bearing(fake_cao, foreman_harness, monkeypatch, fork, disallow):
    loop, queue, github = foreman_harness()
    _clean(fake_cao, loop)
    loop._cfg = replace(loop._cfg, safety=replace(loop._cfg.safety, disallow_forks=disallow))
    create = github.create_pr

    def returned_pr(*args, **kwargs):
        pr = replace(create(*args, **kwargs), is_fork=fork)
        github.seed_pr(pr)
        return pr

    monkeypatch.setattr(github, "create_pr", returned_pr)
    outcome = loop.run_pass()[0]
    blocked = fork and disallow
    assert outcome.final_phase == ("escalated" if blocked else "done"), outcome.reason
    if blocked:
        assert "comes from a fork" in outcome.reason
    assert queue.load_state("owner/repo#1").phase == outcome.final_phase
    assert not queue.list_active() and not loop._git.worktrees
    assert all(session.deleted for session in fake_cao._sessions.values())


@pytest.mark.parametrize(
    "guard,reason,invocations,commits",
    [
        ("max_prompt_tokens", "prompt token budget exceeded", 0, 0),
        ("max_total_iterations", "unresolved findings after 1", 1, 1),
        ("max_coder_invocations_per_run", "coder invocation budget exhausted", 1, 1),
        ("max_reviewer_triggers_per_run", "reviewer trigger budget exhausted", 1, 0),
    ],
)
def test_dispatch_and_commit_budgets_bound_remediation(
    fake_cao,
    foreman_harness,
    guard,
    reason,
    invocations,
    commits,
):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop, proposal="fix")
    loop._cfg = replace(loop._cfg, safety=replace(loop._cfg.safety, **{guard: 1}))
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated", outcome.reason
    assert reason in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state.phase == "escalated" and state.extras["lanes_stopped"]
    developer = fake_cao._sessions.get(developer_session_name(loop.run_id))
    assert (len(developer.submitted_messages) if developer else 0) == invocations
    assert len(loop._git.commits) == commits
    assert not github.list_open_prs() and not queue.list_active()
    assert not loop._git.worktrees
    assert all(session.deleted for session in fake_cao._sessions.values())


def test_repeated_patch_stops_before_more_ci_or_dispatch(fake_cao, foreman_harness):
    loop, queue, github = foreman_harness()
    _clean(fake_cao, loop)
    loop._cfg = replace(
        loop._cfg,
        safety=replace(
            loop._cfg.safety,
            max_coder_invocations_per_run=5,
            max_total_iterations=5,
            max_commits_per_run=5,
            max_reviewer_triggers_per_run=15,
        ),
        escalation=replace(loop._cfg.escalation, stagnation_rounds_threshold=2),
    )
    github.set_check_runs(
        "sha",
        [
            replace(
                github.get_check_runs("sha")[0], conclusion="failure", summary="regression failed"
            )
        ],
    )
    outcome = loop.run_pass()[0]
    state = queue.load_state("owner/repo#1")
    assert outcome.final_phase == state.phase == "escalated"
    assert "no converging signal" in outcome.reason
    assert len(state.extras["review_signatures"]) == 2
    assert len(set(state.extras["review_signatures"])) == 1
    assert len(state.extras["ci_failure_hashes"]) == 1
    assert len(fake_cao._sessions[developer_session_name(loop.run_id)].submitted_messages) == 2
    assert len(github.list_open_prs()) == 1
    assert not queue.list_active() and not loop._git.worktrees
    assert all(session.deleted for session in fake_cao._sessions.values())


def test_expired_wall_clock_prevents_review_after_worker(fake_cao, foreman_harness, monkeypatch):
    loop, queue, github = foreman_harness()
    _clean(fake_cao, loop)
    loop._cfg = replace(loop._cfg, escalation=replace(loop._cfg.escalation, max_run_seconds=1))
    execute = loop._executor.execute

    def elapsed_worker(*args, **kwargs):
        result = execute(*args, **kwargs)
        state = queue.load_state("owner/repo#1")
        extras = {
            **state.extras,
            "run_started_at": (datetime.now(UTC) - timedelta(seconds=30)).isoformat(),
        }
        queue.save_state(replace(state, extras=extras), expected_updated_at=state.updated_at)
        return result

    monkeypatch.setattr(loop._executor, "execute", elapsed_worker)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated" and "wall-clock deadline" in outcome.reason
    assert len(fake_cao._sessions) == 1
    assert all(session.deleted for session in fake_cao._sessions.values())
    assert not github.list_open_prs() and not queue.list_active()
    assert not loop._git.worktrees


def test_every_native_launch_strips_coordinator_authority(fake_cao, foreman_harness):
    loop, queue, _ = foreman_harness()
    _clean(fake_cao, loop)
    loop._executor._env = {
        "GITHUB_TOKEN": "invented-github-canary",
        "GH_TOKEN": "invented-gh-canary",
        "SSH_AUTH_SOCK": "/invented/coordinator/socket",
        "GH_CONFIG_DIR": "/invented/gh",
    }
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", outcome.reason
    assert len(fake_cao._sessions) == 4
    for session in fake_cao._sessions.values():
        assert all(
            session.env_vars[key] == "" for key in ("GITHUB_TOKEN", "GH_TOKEN", "SSH_AUTH_SOCK")
        )
        assert session.env_vars["GH_CONFIG_DIR"] == "/nonexistent/aipro-no-github-auth"
        assert session.env_vars["GIT_CONFIG_VALUE_4"] == ""
        assert session.env_vars["GIT_TERMINAL_PROMPT"] == "0"
        assert session.deleted
    assert not queue.list_active()
