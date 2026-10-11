"""Production TTL sweep over native Git and the CAO HTTP boundary."""

import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.cleanup import CleanupSweeper
from ai_pr_orchestrator.v3.config import CleanupConfig
from ai_pr_orchestrator.v3.domain import GitHubIssueRef, ModelAssignment
from ai_pr_orchestrator.v3.git_ops import GitWorktreeOps
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, SessionSpec
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from tests.unit.test_v3_review_checkout import _repo


def _resources(tmp_path, controller, *, terminal=True):
    root = _repo(tmp_path)
    git = GitWorktreeOps(root)
    github = FakeGitHubClient()
    github.seed_issue(1, labels=["v3-work", "v3-work-enabled"])
    queue = GitHubIssueQueue(github, "owner", "repo", host_id="cleanup-host")
    issue = GitHubIssueRef("owner", "repo", 1)
    state = queue.claim(issue, "cleanup-run", branch="aipro-issue-1")
    state = replace(state, phase="reviewing", round_id="round-1")
    git.create_branch("aipro-issue-1", "main")
    developer = git.create_worktree(
        str(tmp_path / "developer"), "aipro-issue-1", run_id=state.run_id, work_item_id=issue.slug()
    )
    head = git.head_sha(developer)
    paths, sessions, lanes = [developer], [], {}
    for lane in LaneRegistry.default():
        workdir = (
            developer
            if lane.role == "worker"
            else git.create_review_checkout(
                str(tmp_path / lane.lane),
                developer,
                head,
                run_id=state.run_id,
                work_item_id=issue.slug(),
            )
        )
        if lane.role == "reviewer":
            paths.append(workdir)
            lanes[lane.lane] = {
                "worktree": workdir,
                "assignment": ModelAssignment(lane.lane, "ref-" + lane.lane).to_dict(),
                "fallbacks": [],
                "profile": lane.profile_template,
            }
        context = LaneExecutionContext(state.run_id, state.round_id, issue.slug())
        sessions.append(
            controller.start_session(
                SessionSpec(
                    lane, state.run_id, workdir, {}, context, command="invented model boundary"
                )
            )
        )
    updated = replace(
        state,
        extras={
            **state.extras,
            "worktree": developer,
            "developer_session": sessions[0].session_id,
            "coder_turn": {},
            "lanes_stopped": True,
            "review_waves": {
                state.round_id: {"round_id": state.round_id, "head_sha": head, "lanes": lanes}
            },
        },
    )
    queue.save_state(updated, expected_updated_at=state.updated_at)
    if terminal:
        updated = queue.complete(issue, updated, reason="closed job with retained resources")
    return git, queue, updated, paths, sessions


@pytest.mark.parametrize("terminal", [False, True])
def test_sweep_removes_only_verified_terminal_resources_after_inactivity(
    tmp_path, cao_controller, fake_cao, terminal
):
    git, queue, state, paths, sessions = _resources(tmp_path, cao_controller, terminal=terminal)
    human = tmp_path / "human"
    subprocess.run(
        ["git", "worktree", "add", "-b", "human", str(human)],
        cwd=git._root,
        check=True,
        capture_output=True,
    )
    (human / "tracked.txt").write_text("irreplaceable human edits")
    sweeper = CleanupSweeper(queue, cao_controller, git, CleanupConfig())
    base = datetime.now(UTC)
    first = sweeper.sweep(now=base + timedelta(hours=3))
    assert len(first) == (4 if terminal else 0)
    assert all(Path(path).exists() for path in paths)
    recent = (base + timedelta(hours=24)).timestamp()
    os.utime(Path(paths[0]) / "tracked.txt", (recent, recent))
    second = sweeper.sweep(now=base + timedelta(hours=25))
    assert Path(paths[0]).exists()
    assert len(second) == (3 if terminal else 0)
    third = sweeper.sweep(now=base + timedelta(hours=49))
    assert len(third) == (1 if terminal else 0)
    assert all(Path(path).exists() is not terminal for path in paths)
    assert all(fake_cao._sessions[handle.session_id].deleted is terminal for handle in sessions)
    assert (human / "tracked.txt").read_text() == "irreplaceable human edits"
    assert queue.load_state(state.work_item_id) == state
    assert sweeper.sweep(now=base + timedelta(hours=50)) == []


@pytest.mark.parametrize(
    "failure",
    [
        "empty_inventory",
        "accepted_delete",
        "young_orphan",
        "changed_authority",
        "renamed_worker",
        "renamed_reviewer",
    ],
)
def test_unverified_runtime_or_changed_authority_cannot_release_checkout(
    tmp_path, cao_controller, fake_cao, monkeypatch, failure
):
    import httpx

    from ai_pr_orchestrator.v3.cao import CaoSessionNotReadyError
    from ai_pr_orchestrator.v3.interfaces import StateConflictError
    from tests.integration._fake_cao_server import FaultSpec

    git, queue, _state, paths, sessions = _resources(tmp_path, cao_controller)
    request = cao_controller._request
    if failure.startswith("renamed_"):
        survivor = sessions[0 if failure == "renamed_worker" else 1]
        for handle in sessions:
            if handle != survivor:
                cao_controller.terminate_session(handle)
        sessions = [survivor]
        cao_controller._lanes = LaneRegistry(
            replace(lane, lane="new-" + lane.lane) for lane in LaneRegistry.default()
        )
        cao_controller._sessions.clear()
    if failure in ("empty_inventory", "renamed_worker", "renamed_reviewer"):

        def missing_inventory(method, url, **kwargs):
            if method == "GET" and url == "/sessions":
                return httpx.Response(200, json=[])
            return request(method, url, **kwargs)

        monkeypatch.setattr(cao_controller, "_request", missing_inventory)
    elif failure == "accepted_delete":
        fake_cao.add_fault(FaultSpec(method="DELETE", path_prefix="/sessions/", status_code=204))
    elif failure == "changed_authority":
        verify = queue.verify_cleanup_state

        def changed(authority):
            queue.save_state(
                replace(authority, phase="queued", terminal_reason=None),
                expected_updated_at=authority.updated_at,
            )
            verify(authority)

        monkeypatch.setattr(queue, "verify_cleanup_state", changed)
    now = datetime.now(UTC) + timedelta(hours=1 if failure == "young_orphan" else 25)
    with pytest.raises((CaoSessionNotReadyError, StateConflictError)):
        CleanupSweeper(queue, cao_controller, git, CleanupConfig()).sweep(now=now)
    assert all(Path(path).exists() for path in paths)
    assert not any(fake_cao._sessions[handle.session_id].deleted for handle in sessions)
    assert (Path(paths[0]) / "tracked.txt").read_text() == "base\n"


def test_foreign_repository_runtime_is_retained_during_owned_cleanup(
    tmp_path, cao_controller, fake_cao
):
    git, queue, state, _, sessions = _resources(tmp_path, cao_controller)
    lane = LaneRegistry.default().get("developer")
    foreign_path = tmp_path / "foreign-runtime"
    foreign_path.mkdir()
    context = LaneExecutionContext("foreign-run", "round-1", "other/repo#99")
    foreign = cao_controller.start_session(
        SessionSpec(
            lane,
            context.run_id,
            str(foreign_path),
            {},
            context,
            command="foreign invented boundary",
        )
    )
    actions = CleanupSweeper(queue, cao_controller, git, CleanupConfig()).sweep(
        now=datetime.now(UTC) + timedelta(hours=25)
    )
    assert len(actions) == 8
    assert all(fake_cao._sessions[handle.session_id].deleted for handle in sessions)
    assert not fake_cao._sessions[foreign.session_id].deleted and foreign_path.exists()
    assert queue.load_state(state.work_item_id) == state


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory read permissions")
def test_unreadable_activity_retains_every_checkout(tmp_path, cao_controller):
    from ai_pr_orchestrator.v3.git_ops import GitOpsError

    git, queue, state, paths, _ = _resources(tmp_path, cao_controller)
    private = Path(paths[0]) / "private"
    private.mkdir()
    (private / "uninspected.py").write_text("irreplaceable = True\n")
    private.chmod(0)
    try:
        with pytest.raises(GitOpsError, match="activity inventory is incomplete"):
            CleanupSweeper(queue, cao_controller, git, CleanupConfig()).sweep(
                now=datetime.now(UTC) + timedelta(hours=25)
            )
        assert all(Path(path).exists() for path in paths)
        assert queue.load_state(state.work_item_id) == state
    finally:
        private.chmod(0o700)
    assert (private / "uninspected.py").read_text() == "irreplaceable = True\n"
