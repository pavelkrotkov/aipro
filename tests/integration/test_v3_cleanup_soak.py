"""Sequential foreman soak with real Git and production TTL cleanup."""

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_pr_orchestrator.v3.cao import CaoTransportError
from ai_pr_orchestrator.v3.cleanup import CleanupSweeper
from ai_pr_orchestrator.v3.git_ops import GitOpsError, GitWorktreeOps
from tests.integration.test_continuous_foreman import (
    _assert_completed_once,
    _lose_accepted_input,
    _resume_cold,
    _scheduler,
)
from tests.unit.test_v3_review_checkout import _repo


@pytest.mark.parametrize("restart", [None, "aipro-developer", "aipro-requirements-reviewer"])
def test_native_git_soak_sweeps_retained_checkouts_after_every_issue(
    tmp_path,
    fake_cao,
    foreman_harness,
    cao_controller,
    monkeypatch,
    request,
    restart,
):
    root = _repo(tmp_path)
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "clone", "--bare", str(root), str(remote)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)], cwd=root, check=True, capture_output=True
    )
    git = GitWorktreeOps(root)
    count = request.config.getoption("--soak-issues")
    template, queue, github = foreman_harness(list(range(1, count + 1)), git_ops=git)
    template._worktree_root = str(tmp_path / "worktrees")
    template._executor._git = git
    record_head = github.record_branch_head

    def green_ci(branch, head):
        record_head(branch, head)
        github.seed_check_run(head, "tests", "completed", "success")

    monkeypatch.setattr(github, "record_branch_head", green_ci)
    send = cao_controller._request

    def tool_effect(method, url, **kwargs):
        response = send(method, url, **kwargs)
        if method == "POST" and url.endswith("/input"):
            terminal_id = url.split("/")[2]
            session = next(s for s in fake_cao._sessions.values() if s.terminal_id == terminal_id)
            if session.agent_profile == "aipro-developer":
                number = session.metadata["work_item_id"].split("#")[1]
                (Path(session.workdir) / "implemented.py").write_text(f"issue_number = {number}\n")
        return response

    monkeypatch.setattr(cao_controller, "_request", tool_effect)
    retained = set()
    remove = git.cleanup_worktree

    def temporary_cleanup_failure(path):
        if path not in retained:
            retained.add(path)
            raise GitOpsError("temporary checkout cleanup failure")
        remove(path)

    monkeypatch.setattr(git, "cleanup_worktree", temporary_cleanup_failure)
    scheduler = _scheduler(template, queue, template._executor, fake_cao)
    sweeper = CleanupSweeper(queue, cao_controller, git, template._cfg.cleanup)
    human = tmp_path / "human"
    human.mkdir()
    (human / "keep.py").write_text("irreplaceable = True\n")
    states = []
    for number in range(1, count + 1):
        if number == 1 and restart is not None:
            _lose_accepted_input(cao_controller, fake_cao, restart, monkeypatch)
            with pytest.raises(CaoTransportError):
                scheduler.step()
            active = queue.load_state("owner/repo#1")
            assert queue.load_state("owner/repo#2") is None
            outcome = _resume_cold(template, queue, fake_cao, git)
            assert queue.load_state("owner/repo#1").run_id == active.run_id
        else:
            outcome = scheduler.step()
        assert len(outcome) == 1 and outcome[0].final_phase == "done", outcome
        state = _assert_completed_once(queue, fake_cao, number)
        states.append(state)
        if number == 1:
            with pytest.raises(GitOpsError):
                git.create_worktree(
                    str(human),
                    state.extras["branch"],
                    run_id=state.run_id,
                    work_item_id=state.work_item_id,
                )
        assert Path(state.extras["worktree"]).exists()
        assert sweeper.sweep(now=datetime.now(UTC) + timedelta(hours=25)) == [
            {
                "kind": "clean_orphan_worktree",
                "work_item_id": state.work_item_id,
                "run_id": state.run_id,
                "worktree": state.extras["worktree"],
            }
        ]
        assert not Path(state.extras["worktree"]).exists()
        inventory = git.checkout_inventory()
        assert len(inventory) == 2
        assert {r["path"] for r in inventory} == {str(root), str(human)}
        assert next(r for r in inventory if r["path"] == str(human))["materialized"] is False
        assert (human / "keep.py").read_text() == "irreplaceable = True\n"
        assert not queue.list_active()
        assert len(github.list_open_prs()) == number
    assert scheduler.step() == []
    assert len({state.run_id for state in states}) == count
    assert len({state.extras["head_sha"] for state in states}) == count
    assert len(fake_cao._sessions) == count * 4
    assert all(queue.load_state(state.work_item_id) == state for state in states)
    assert (
        len(git._run("for-each-ref", "--format=%(refname)", "refs/heads").splitlines()) == count + 1
    )
