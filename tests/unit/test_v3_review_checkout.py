"""Independent reviewer checkout invariants for issue #49."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_pr_orchestrator.v3.git_ops import GitOpsError, GitWorktreeOps


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "tracked.txt").write_text("base\n")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=T",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-m",
        "base",
    )
    _git(root, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/heads/main")
    return root


def test_review_checkouts_are_independent_detached_repositories(tmp_path: Path):
    root = _repo(tmp_path)
    ops = GitWorktreeOps(root)
    ops.create_branch("feature", "main")
    developer = Path(ops.create_worktree(str(tmp_path / "developer"), "feature"))
    (developer / "tracked.txt").write_text("snapshot\n")
    head = ops.commit(str(developer), "snapshot", name="T", email="t@example.com")

    review_a = Path(ops.create_review_checkout(str(tmp_path / "review-a"), str(developer), head))
    review_b = Path(ops.create_review_checkout(str(tmp_path / "review-b"), str(developer), head))
    try:
        for review in (review_a, review_b):
            assert _git(review, "rev-parse", "HEAD") == head
            assert _git(review, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
            assert _git(review, "remote") == ""
            assert (review / ".git").is_dir()
            assert not (review / ".git" / "objects" / "info" / "alternates").exists()

        _git(review_a, "update-ref", "refs/heads/reviewer-experiment", head)
        (review_a / "tracked.txt").write_text("experiment\n")

        assert (developer / "tracked.txt").read_text() == "snapshot\n"
        assert (review_b / "tracked.txt").read_text() == "snapshot\n"
        assert "reviewer-experiment" not in _git(
            developer, "for-each-ref", "--format=%(refname:short)"
        )
        assert "reviewer-experiment" not in _git(
            review_b, "for-each-ref", "--format=%(refname:short)"
        )
        local_config = _git(review_a, "config", "--local", "--list").lower()
        assert "credential" not in local_config
        assert "remote." not in local_config
        assert "hookspath" not in local_config
    finally:
        ops.cleanup_review_checkout(str(review_a), head)
        ops.cleanup_review_checkout(str(review_b), head)

    assert not review_a.exists()
    assert not review_b.exists()


def test_review_cleanup_refuses_unowned_path(tmp_path: Path):
    root = _repo(tmp_path)
    unknown = tmp_path / "unknown"
    (unknown / ".git").mkdir(parents=True)
    with pytest.raises(GitOpsError, match="unowned review checkout"):
        GitWorktreeOps(root).cleanup_review_checkout(str(unknown), _git(root, "rev-parse", "HEAD"))
    assert unknown.exists()
