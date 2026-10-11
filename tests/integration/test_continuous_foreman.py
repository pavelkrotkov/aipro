"""Unattended scheduling and cold recovery through real CAO HTTP calls."""

from dataclasses import replace

import pytest

from ai_pr_orchestrator.v3.cao import CaoSessionController, CaoTransportError, session_name_for
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.ci_gate import GateDecision
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop, PRReconciliationError
from ai_pr_orchestrator.v3.model_router import build_model_broker
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue, LabelSyncError
from ai_pr_orchestrator.v3.scheduler import ContinuousForeman
from tests.integration._harness import FakeGitOperations, developer_output, developer_session_name
from tests.unit.test_v3_foreman import RecordingGate, RecordingGit


def _scheduler(template, queue, executor, fake_cao, *, broker=None):
    def factory(run_id):
        for issue in queue.list_work_items():
            fake_cao.set_output(developer_session_name(run_id, issue.number), developer_output())
        for lane in template._lanes:
            if lane.role == "reviewer":
                fake_cao.set_output(session_name_for(run_id, lane.lane), "[]")
        return ForemanPolicyLoop(
            queue,
            broker or build_model_broker(template._cfg),
            template._lanes,
            executor,
            template._gate,
            template._git,
            template._cfg,
            run_id=run_id,
            worktree_root="/wt",
            committer_name="Test",
            committer_email="test@example.invalid",
        )

    return ContinuousForeman(queue, factory)


def _lose_accepted_input(controller, fake_cao, profile, monkeypatch):
    request = controller._request
    lost = []

    def send(method, url, **kwargs):
        response = request(method, url, **kwargs)
        if method == "POST" and url.endswith("/input"):
            terminal_id = url.split("/")[2]
            state = next(s for s in fake_cao._sessions.values() if s.terminal_id == terminal_id)
            if state.agent_profile == profile and not lost:
                lost.append(terminal_id)
                raise CaoTransportError("accepted response lost")
        return response

    monkeypatch.setattr(controller, "_request", send)


def _assert_completed_once(queue, fake_cao, issue_number):
    state = queue.load_state(f"owner/repo#{issue_number}")
    assert state.phase == "done"
    assert state.extras["coder_usage"]["invocations"] == 1
    worker = fake_cao._sessions[developer_session_name(state.run_id, issue_number)]
    assert len(worker.submitted_messages) == 1
    assert worker.deleted
    reviewers = [
        fake_cao._sessions[session_name_for(state.run_id, lane)]
        for lane in ("requirements-reviewer", "breaker-reviewer", "architecture-reviewer")
    ]
    assert all(len(s.submitted_messages) == 1 and s.deleted for s in reviewers)
    assert state.extras["prompt_tokens"] == sum(
        (len(prompt) + 3) // 4
        for session in [worker, *reviewers]
        for prompt in session.submitted_messages
    )
    return state


def test_unattended_two_issues_are_sequential_with_three_reviewers(fake_cao, foreman_harness):
    template, queue, fake = foreman_harness([1, 2], git_ops=RecordingGit())
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    assert scheduler.step()[0].issue.number == 1
    first = _assert_completed_once(queue, fake_cao, 1)
    assert queue.load_state("owner/repo#2") is None
    assert scheduler.step()[0].issue.number == 2
    second = _assert_completed_once(queue, fake_cao, 2)
    assert first.run_id != second.run_id
    assert scheduler.step() == []
    assert len(fake.list_open_prs()) == 2
    assert template._git.branches == ["main", "aipro-issue-1", "aipro-issue-2"]


@pytest.mark.parametrize("profile", ["aipro-developer", "aipro-requirements-reviewer"])
def test_cold_restart_collects_existing_turn_and_preserves_capacity(
    fake_cao, foreman_harness, cao_controller, monkeypatch, profile
):
    template, queue, fake = foreman_harness([1, 2], git_ops=RecordingGit())
    _lose_accepted_input(cao_controller, fake_cao, profile, monkeypatch)
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    with pytest.raises(CaoTransportError):
        scheduler.step()
    active = queue.load_state("owner/repo#1")
    assert queue.load_state("owner/repo#2") is None
    fresh_queue = GitHubIssueQueue(fake, "owner", "repo", host_id=queue.host_id)
    with CaoSessionController(cao_controller._config, template._lanes) as fresh_controller:
        executor = CaoLaneExecutor(
            fresh_controller,
            template._lanes,
            git=FakeGitOperations(),
            catalog=template._executor._catalog,
            poll_interval_seconds=0.01,
        )
        resumed = _scheduler(template, fresh_queue, executor, fake_cao)
        outcome = resumed.step()[0]
        assert outcome.final_phase == "done", outcome.reason
    finished = _assert_completed_once(fresh_queue, fake_cao, 1)
    assert finished.run_id == active.run_id
    assert queue.load_state("owner/repo#2") is None
    assert template._git.branches == ["main", "aipro-issue-1"]
    assert len(fake.list_open_prs()) == 1


def test_needs_human_is_not_retried_with_a_stale_ready_label(fake_cao, foreman_harness):
    template, queue, fake = foreman_harness([1], git_ops=RecordingGit())
    state = queue.claim(queue.list_ready()[0], "needs-human")
    state = queue.mark_needs_human(queue.list_active()[0], state, reason="operator decision")
    fake.add_label(1, "v3-work")
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    assert scheduler.step() == []
    assert queue.load_state("owner/repo#1") == state
    assert not fake_cao._sessions


def test_foreign_active_owner_blocks_next_ready_issue(fake_cao, foreman_harness):
    template, queue, fake = foreman_harness([1, 2], git_ops=RecordingGit())
    foreign = GitHubIssueQueue(fake, "owner", "repo", host_id="other-machine")
    state = foreign.claim(foreign.list_ready()[0], "foreign-run")
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    assert scheduler.step() == []
    assert queue.load_state("owner/repo#1") == state
    assert queue.load_state("owner/repo#2") is None
    assert not fake_cao._sessions


def test_pending_ci_releases_only_a_stopped_developer(fake_cao, foreman_harness):
    gate = RecordingGate(
        [
            GateDecision(passed=False, pending_checks=("build",), failed_checks=()),
            GateDecision(passed=True, pending_checks=(), failed_checks=()),
        ]
    )
    template, queue, _ = foreman_harness([1, 2], git_ops=RecordingGit())
    template._gate = gate
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    outcomes = scheduler.step()
    assert [outcome.final_phase for outcome in outcomes] == ["ci_gating", "done"]
    first = queue.load_state("owner/repo#1")
    assert first.phase == "queued" and first.extras["lanes_stopped"]
    assert fake_cao._sessions[developer_session_name(first.run_id, 1)].deleted
    _assert_completed_once(queue, fake_cao, 2)
    scheduler.step()
    _assert_completed_once(queue, fake_cao, 1)


@pytest.mark.parametrize("pending", [False, True])
def test_unconfirmed_developer_stop_keeps_capacity_reserved(
    fake_cao, foreman_harness, monkeypatch, pending
):
    gate = RecordingGate(
        [
            GateDecision(
                passed=not pending, pending_checks=("build",) if pending else (), failed_checks=()
            )
        ]
    )
    template, queue, _ = foreman_harness([1, 2], git_ops=RecordingGit())
    template._gate = gate
    terminate = template._executor.terminate_session

    def stop(handle):
        if handle.lane == "developer":
            raise CaoTransportError("termination response lost")
        terminate(handle)

    monkeypatch.setattr(template._executor, "terminate_session", stop)
    with pytest.raises(PRReconciliationError, match="termination is unconfirmed"):
        _scheduler(template, queue, template._executor, fake_cao).step()
    first = queue.load_state("owner/repo#1")
    assert first.phase == ("ci_gating" if pending else "updating_pr")
    assert not first.extras["lanes_stopped"]
    assert queue.load_state("owner/repo#2") is None
    assert not fake_cao._sessions[developer_session_name(first.run_id, 1)].deleted


def test_cancelled_run_does_not_release_an_unconfirmed_developer(fake_cao, foreman_harness):
    template, queue, _ = foreman_harness([1, 2], git_ops=RecordingGit())
    issue = queue.list_ready()[0]
    state = queue.claim(issue, "cancelled-run")
    claimed = state
    state = replace(state, extras={**state.extras, "coder_turn": {}})
    queue.save_state(state, expected_updated_at=claimed.updated_at)
    with pytest.raises(LabelSyncError, match="unconfirmed developer"):
        queue.cancel(issue, state, reason="operator cancellation")
    assert queue._cfg.active_label in queue._client.get_labels(1)
    with pytest.raises(RuntimeError, match="unconfirmed developer"):
        _scheduler(template, queue, template._executor, fake_cao).step()
    assert queue.load_state("owner/repo#2") is None


def test_unconfirmed_reviewer_stop_keeps_capacity_reserved(fake_cao, foreman_harness, monkeypatch):
    template, queue, _ = foreman_harness([1, 2], git_ops=RecordingGit())
    terminate = template._executor.terminate_session

    def stop(handle):
        if handle.lane == "requirements-reviewer":
            raise CaoTransportError("reviewer termination response lost")
        terminate(handle)

    monkeypatch.setattr(template._executor, "terminate_session", stop)
    with pytest.raises(PRReconciliationError, match="termination is unconfirmed"):
        _scheduler(template, queue, template._executor, fake_cao).step()
    assert queue.load_state("owner/repo#1").phase == "reviewing"
    assert queue.load_state("owner/repo#2") is None


def test_multi_issue_soak_has_no_duplicate_or_unmanaged_side_effects(
    fake_cao, foreman_harness, request
):
    count = request.config.getoption("--soak-issues")
    template, queue, github = foreman_harness(list(range(1, count + 1)), git_ops=RecordingGit())
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    states = []
    for number in range(1, count + 1):
        outcome = scheduler.step()
        assert len(outcome) == 1 and outcome[0].issue.number == number
        states.append(_assert_completed_once(queue, fake_cao, number))
        assert not queue.list_active()
        assert not template._git.worktrees
        assert all(session.deleted for session in fake_cao._sessions.values())
        assert len(github.list_open_prs()) == number
    assert scheduler.step() == []
    assert len({state.run_id for state in states}) == count
    assert len(template._git.branches) == count + 1
    assert len(set(template._git.branches)) == count + 1
    assert len(fake_cao._sessions) == count * 4
    assert all(queue.load_state(state.work_item_id) == state for state in states)


@pytest.mark.parametrize("association", ["NONE", "CONTRIBUTOR", ""])
def test_untrusted_issue_is_quarantined_before_resources(fake_cao, foreman_harness, association):
    template, queue, github = foreman_harness([1], git_ops=RecordingGit())
    github.seed_issue(1, labels=["v3-work", "v3-work-enabled"], author_association=association)
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    outcome = scheduler.step()[0]
    assert outcome.final_phase == "escalated"
    state = queue.load_state("owner/repo#1")
    assert "host_id" not in state.extras
    assert not fake_cao._sessions and not github.list_open_prs()
    assert template._git.branches == ["main"] and not template._git.worktrees
    assert scheduler.step() == []


@pytest.mark.parametrize("role", ["worker", "reviewer"])
@pytest.mark.parametrize("revocation", ["label", "author"])
def test_opt_out_during_late_lane_start_stops_every_session(
    fake_cao, foreman_harness, monkeypatch, role, revocation
):
    import time

    template, queue, github = foreman_harness([1], git_ops=RecordingGit())
    template._cfg = replace(
        template._cfg, github_queue=replace(template._cfg.github_queue, lease_seconds=1)
    )
    execute = template._executor.execute

    def delayed(lane, *args, **kwargs):
        if lane.role == role:
            if revocation == "label":
                github.remove_label(1, "v3-work-enabled")
            else:
                github.seed_issue(1, labels=github.get_labels(1), author_association="NONE")
            time.sleep(0.7)
        return execute(lane, *args, **kwargs)

    monkeypatch.setattr(template._executor, "execute", delayed)
    outcome = _scheduler(template, queue, template._executor, fake_cao).step()[0]
    assert outcome.final_phase == "cancelled"
    state = queue.load_state("owner/repo#1")
    assert state.extras["lanes_stopped"]
    assert all(session.deleted for session in fake_cao._sessions.values())
    assert not github.list_open_prs() and not queue.list_active()


def test_cold_restart_after_opt_out_cancels_without_replaying_input(
    fake_cao, foreman_harness, cao_controller, monkeypatch
):
    template, queue, github = foreman_harness([1, 2], git_ops=RecordingGit())
    _lose_accepted_input(cao_controller, fake_cao, "aipro-developer", monkeypatch)
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    with pytest.raises(CaoTransportError):
        scheduler.step()
    github.remove_label(1, "v3-work-enabled")
    outcome = scheduler.step()[0]
    assert outcome.final_phase == "cancelled"
    first = queue.load_state("owner/repo#1")
    worker = fake_cao._sessions[developer_session_name(first.run_id, 1)]
    assert worker.deleted and len(worker.submitted_messages) == 1
    assert queue.load_state("owner/repo#2") is None
    assert not github.list_open_prs() and not queue.list_active()


@pytest.mark.parametrize(
    ("limit_adjustment", "corruption"),
    [(0, None), (-1, None), (0, "missing"), (0, -1), (0, True)],
)
def test_cold_review_recovery_preserves_prompt_budget(
    fake_cao, foreman_harness, cao_controller, monkeypatch, limit_adjustment, corruption
):
    template, queue, github = foreman_harness([1], git_ops=RecordingGit())
    _lose_accepted_input(cao_controller, fake_cao, "aipro-requirements-reviewer", monkeypatch)
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    with pytest.raises(CaoTransportError):
        scheduler.step()
    checkpoint = queue.load_state("owner/repo#1")
    prompts = [checkpoint.extras["coder_turn"]["prompt"]]
    for wave in checkpoint.extras["review_waves"].values():
        prompts.extend(wave["prompts"].values())
    spent = sum((len(prompt) + 3) // 4 for prompt in prompts)
    assert checkpoint.extras["prompt_tokens"] == spent
    if corruption is not None:
        extras = {**checkpoint.extras}
        if corruption == "missing":
            extras.pop("prompt_tokens")
        else:
            extras["prompt_tokens"] = corruption
        queue.save_state(
            replace(checkpoint, extras=extras), expected_updated_at=checkpoint.updated_at
        )
    template._cfg = replace(
        template._cfg,
        safety=replace(template._cfg.safety, max_prompt_tokens=spent + limit_adjustment),
    )
    outcome = scheduler.step()[0]
    state = queue.load_state("owner/repo#1")
    if corruption is None:
        assert state.extras["prompt_tokens"] == spent
    successful = limit_adjustment == 0 and corruption is None
    assert outcome.final_phase == ("done" if successful else "escalated")
    assert len(github.list_open_prs()) == int(successful)
    assert all(
        len(session.submitted_messages) == 1 and session.deleted
        for session in fake_cao._sessions.values()
    )
