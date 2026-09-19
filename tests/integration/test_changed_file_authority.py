"""Actual Git observations cross the CAO boundary before policy and commit."""

import subprocess
from pathlib import Path

import pytest

from ai_pr_orchestrator.v3.cao import CAOControlPlaneConfig, CaoSessionController, session_name_for
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.git_ops import GitOpsError, GitWorktreeOps
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from tests.integration._fake_cao_server import FakeCAOServer
from tests.unit.test_v3_foreman import _foreman, _gate, _ready_fake


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    (root / "tracked").write_text("initial")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    git(root, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/heads/main")
    git(root, "checkout", "-b", "issue")
    return root


def test_changed_paths_preserve_names_and_include_all_git_states(repo):
    ops = GitWorktreeOps(repo)
    (repo / "committed").write_text("committed")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "worker commit")
    (repo / "staged").write_text("staged")
    git(repo, "add", "staged")
    (repo / "tracked").write_text("unstaged")
    names = [" space ", "new\nline", ".github/workflows/new.yml"]
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("untracked")
    assert set(ops.changed_files(str(repo), "main")) == {
        "committed",
        "staged",
        "tracked",
        *names,
    }
    assert "committed" not in ops.changed_files(str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-m", "all changes")
    assert ops.changed_files(str(repo)) == []
    git(repo, "checkout", "main")
    (repo / "main-only").write_text("base advanced")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base advance")
    git(repo, "checkout", "issue")
    assert "main-only" not in ops.changed_files(str(repo), "main")


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("direction", ["into", "out"])
def test_renames_include_both_paths(repo, committed, direction):
    protected = ".github/workflows/ci.yml"
    (repo / ".github/workflows").mkdir(parents=True)
    if direction == "out":
        git(repo, "mv", "tracked", protected)
        git(repo, "commit", "-m", "baseline workflow")
        git(repo, "branch", "-f", "main", "HEAD")
        git(repo, "mv", protected, "ordinary")
        expected = {protected, "ordinary"}
    else:
        git(repo, "mv", "tracked", protected)
        expected = {"tracked", protected}
    if committed:
        git(repo, "commit", "-m", "rename")
    assert set(GitWorktreeOps(repo).changed_files(str(repo), "main")) == expected


@pytest.mark.parametrize("lane_name", ["developer", "requirements-reviewer"])
@pytest.mark.parametrize("committed", [False, True])
def test_actual_executor_returns_git_paths_over_http(repo, lane_name, committed):
    path = repo / ".github/workflows/ci.yml"
    path.parent.mkdir(parents=True)
    path.write_text("prohibited")
    if committed:
        git(repo, "add", ".")
        git(repo, "commit", "-m", "agent commit")
    lanes = LaneRegistry.default()
    with (
        FakeCAOServer() as cao,
        CaoSessionController(
            CAOControlPlaneConfig(base_url=cao.url),
            lanes,
        ) as controller,
    ):
        cao.set_output(session_name_for("run", lane_name), "[]")
        executor = CaoLaneExecutor(
            controller, lanes, git=GitWorktreeOps(repo), poll_interval_seconds=0
        )
        result = executor.execute(
            lanes.get(lane_name), "task", str(repo), LaneExecutionContext("run")
        )
        assert result.changed_files == [".github/workflows/ci.yml"]


@pytest.fixture
def foreman_runtime(repo, tmp_path, monkeypatch):
    lanes = LaneRegistry.default()
    ops = GitWorktreeOps(repo)
    pushed = []
    monkeypatch.setattr(ops, "push", pushed.append)
    catalog = ModelCatalog(
        tuple(
            ModelCatalogEntry(f"ref-{lane.lane}", "test-model", provider="test") for lane in lanes
        )
    )
    with (
        FakeCAOServer() as cao,
        CaoSessionController(
            CAOControlPlaneConfig(base_url=cao.url),
            lanes,
        ) as controller,
    ):
        for lane in lanes:
            cao.set_output(session_name_for("run-1", lane.lane), "[]")
        executor = CaoLaneExecutor(
            controller, lanes, git=ops, catalog=catalog, poll_interval_seconds=0
        )
        gate = _gate()
        loop, queue = _foreman(_ready_fake(), executor, gate, git=ops)
        loop._worktree_root = str(tmp_path / "worktrees")
        yield loop, queue, controller, gate, pushed


@pytest.mark.parametrize("editing_lane", ["developer", "requirements-reviewer"])
def test_foreman_rejects_actual_cao_workflow_edits_before_push(
    foreman_runtime, monkeypatch, editing_lane
):
    loop, queue, controller, gate, pushed = foreman_runtime
    submit = controller.submit_work

    def simulate_worker_edit(handle, prompt):
        submit(handle, prompt)  # Real HTTP input delivery still runs.
        if handle.lane == editing_lane:
            path = Path(loop._worktree_root) / "issue-1/.github/workflows/ci.yml"
            path.parent.mkdir(parents=True)
            path.write_text("prohibited")

    monkeypatch.setattr(controller, "submit_work", simulate_worker_edit)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase in {"failed", "escalated"}
    assert ".github/workflows/" in outcome.reason
    assert pushed == []
    assert gate.evaluated == []
    assert queue.load_state("owner/repo#1").phase in {"failed", "escalated"}


def test_actual_executor_fails_closed_when_git_cannot_observe(repo):
    lanes = LaneRegistry.default()
    git(repo, "branch", "-D", "main")
    with (
        FakeCAOServer() as cao,
        CaoSessionController(
            CAOControlPlaneConfig(base_url=cao.url),
            lanes,
        ) as controller,
    ):
        executor = CaoLaneExecutor(
            controller, lanes, git=GitWorktreeOps(repo), poll_interval_seconds=0
        )
        with pytest.raises(GitOpsError):
            executor.execute(lanes.get("developer"), "task", str(repo), LaneExecutionContext("run"))
        assert cao.session_names() == []


@pytest.mark.parametrize(
    "commits,editing_lane,expected",
    [
        (1, None, "done"),
        (1, "developer", "escalated"),
        (1, "requirements-reviewer", "escalated"),
        (2, None, "escalated"),
    ],
)
def test_commit_budget_uses_current_worktree_after_review(
    foreman_runtime,
    monkeypatch,
    commits,
    editing_lane,
    expected,
):
    loop, _, controller, gate, pushed = foreman_runtime
    submit = controller.submit_work

    def simulate_worker_edit(handle, prompt):
        submit(handle, prompt)
        worktree = Path(loop._worktree_root) / "issue-1"
        if handle.lane == "developer":
            for number in range(commits):
                (worktree / "tracked").write_text(str(number))
                git(worktree, "add", ".")
                git(worktree, "commit", "-m", "agent commit")
        if handle.lane == editing_lane:
            (worktree / "pending").write_text("must consume a commit")

    monkeypatch.setattr(controller, "submit_work", simulate_worker_edit)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == expected
    if expected == "escalated":
        assert "commit budget exhausted" in outcome.reason
        assert pushed == []
        assert gate.evaluated == []
    else:
        assert len(pushed) == 1
        assert len(gate.evaluated) == 1
