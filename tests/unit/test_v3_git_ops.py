"""Tests for V3 git operations (issue #55): protocol conformance and the
subprocess implementation against a real throwaway repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_pr_orchestrator.v3.git_ops import GitOpsError, GitWorktreeOps
from ai_pr_orchestrator.v3.interfaces import GitOperations


class FakeGitOperations:
    """In-memory GitOperations recording the calls the foreman makes."""

    def __init__(self, default: str = "main") -> None:
        self.default = default
        self.calls: list[tuple] = []
        self.branches = {default}
        self.worktrees: dict[str, str] = {}
        self.commits: dict[str, list[str]] = {}
        self.pushed: list[str] = []
        self._seq = 0

    def default_branch(self) -> str:
        self.calls.append(("default_branch",))
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.calls.append(("create_branch", branch, from_ref))
        self.branches.add(branch)

    def create_worktree(self, path: str, branch: str) -> str:
        self.calls.append(("create_worktree", path, branch))
        self.worktrees[path] = branch
        return path

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        self.calls.append(("commit", workdir, message, name, email))
        self._seq += 1
        sha = f"sha-{self._seq}"
        self.commits.setdefault(workdir, []).append(sha)
        return sha

    def commit_count(self, workdir: str, base_ref: str) -> int:
        self.calls.append(("commit_count", workdir, base_ref))
        return len(self.commits.get(workdir, []))

    def push(self, branch: str) -> None:
        self.calls.append(("push", branch))
        self.pushed.append(branch)

    def cleanup_worktree(self, path: str) -> None:
        self.calls.append(("cleanup_worktree", path))
        self.worktrees.pop(path, None)


def test_fake_satisfies_git_operations_protocol():
    assert isinstance(FakeGitOperations(), GitOperations)


# --- GitWorktreeOps against a real temp repo ---------------------------------


@pytest.fixture()
def real_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str, cwd: Path = root) -> str:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
        return ""

    git("init", "-b", "main")
    git(
        "-c",
        "user.name=T",
        "-c",
        "user.email=t@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    (root / "file.txt").write_text("hello\n")
    git("add", ".")
    git("-c", "user.name=T", "-c", "user.email=t@example.com", "commit", "-m", "add file")
    # origin HEAD ref so default_branch() works without a real remote
    git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/heads/main")
    return root


def test_default_branch_and_branch_creation(real_repo: Path):
    ops = GitWorktreeOps(real_repo)
    assert ops.default_branch() == "main"
    ops.create_branch("feat/x", "main")
    heads = subprocess.run(
        ["git", "branch", "--list", "feat/x"],
        cwd=real_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "feat/x" in heads


def test_worktree_commit_push_round_trip(real_repo: Path, tmp_path: Path):
    ops = GitWorktreeOps(real_repo)
    base = ops.default_branch()
    ops.create_branch("feat/y", base)
    wt = tmp_path / "wt"
    workdir = ops.create_worktree(str(wt), "feat/y")
    assert Path(workdir).is_dir()

    (wt / "out.txt").write_text("work\n")
    sha = ops.commit(workdir, "do work", name="Pavel Krotkov", email="pavel.krotkov@gmail.com")
    assert len(sha) == 40
    assert ops.commit_count(workdir, base) == 1

    # author identity came from the explicit parameters, not ambient config
    author = subprocess.run(
        ["git", "log", "-1", "--format=%an <%ae>"],
        cwd=wt,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert author == "Pavel Krotkov <pavel.krotkov@gmail.com>"

    ops.cleanup_worktree(str(wt))
    assert not wt.exists()


def test_push_failure_raises_gitops_error(real_repo: Path):
    ops = GitWorktreeOps(real_repo)
    with pytest.raises(GitOpsError, match="push"):
        ops.push("no-such-branch")
