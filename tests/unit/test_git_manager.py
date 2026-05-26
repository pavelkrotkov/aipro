"""Tests for the git manager (GitRepo)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_pr_orchestrator.git.repo import GitError, GitRepo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path: Path) -> GitRepo:
    """Create a real temporary git repo and return a GitRepo wrapping it."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    _git_cfg(tmp_path, "user.email", "test@test.com")
    _git_cfg(tmp_path, "user.name", "Test")
    _git_cfg(tmp_path, "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return GitRepo(tmp_path)


# ---------------------------------------------------------------------------
# Worktree detection
# ---------------------------------------------------------------------------


class TestIsClean:
    def test_clean_worktree_returns_true(self, repo: GitRepo) -> None:
        assert repo.is_clean() is True

    def test_modified_file_returns_false(self, repo: GitRepo) -> None:
        (repo.path / "README.md").write_text("changed\n")
        assert repo.is_clean() is False

    def test_untracked_file_returns_false(self, repo: GitRepo) -> None:
        (repo.path / "new.txt").write_text("hello\n")
        assert repo.is_clean() is False


# ---------------------------------------------------------------------------
# Commit behaviour
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commits_staged_changes(self, repo: GitRepo) -> None:
        (repo.path / "file.txt").write_text("data\n")
        sha = repo.commit("test commit", "Bot", "bot@example.com")

        assert sha is not None
        assert len(sha) == 40

        log = subprocess.run(
            ["git", "log", "-1", "--format=%an <%ae> %s"],
            cwd=repo.path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Bot <bot@example.com>" in log.stdout
        assert "test commit" in log.stdout

    def test_no_commit_when_clean(self, repo: GitRepo) -> None:
        sha = repo.commit("nothing", "Bot", "bot@example.com")
        assert sha is None

    def test_one_commit_per_invocation(self, repo: GitRepo) -> None:
        (repo.path / "a.txt").write_text("a\n")
        (repo.path / "b.txt").write_text("b\n")
        before = _commit_count(repo.path)
        repo.commit("batch", "Bot", "bot@example.com")
        assert _commit_count(repo.path) == before + 1


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


class TestPush:
    def test_push_calls_git_push(self, repo: GitRepo) -> None:
        with patch.object(repo, "_run") as mock_run:
            repo.push("my-branch")
        mock_run.assert_called_once_with("push", "origin", "my-branch")

    def test_no_force_push(self, repo: GitRepo) -> None:
        with patch.object(repo, "_run") as mock_run:
            repo.push("my-branch")
        args = mock_run.call_args[0]
        assert "--force" not in args
        assert "-f" not in args


# ---------------------------------------------------------------------------
# HEAD SHA race detection
# ---------------------------------------------------------------------------


class TestHeadShaRace:
    def test_fetch_remote_head_returns_sha(self, repo: GitRepo) -> None:
        _add_origin(repo)
        sha = repo.fetch_remote_head("main")
        assert len(sha) == 40

    def test_fetch_remote_head_returns_none_for_nonexistent_branch(self, repo: GitRepo) -> None:
        _add_origin(repo)
        assert repo.fetch_remote_head("non-existent-branch") is None

    def test_race_detected_when_sha_differs(self, repo: GitRepo) -> None:
        _add_origin(repo)
        assert repo.check_remote_head_matches("main", "0" * 40) is False

    def test_race_detected_when_branch_does_not_exist(self, repo: GitRepo) -> None:
        _add_origin(repo)
        assert repo.check_remote_head_matches("non-existent-branch", "0" * 40) is False

    def test_safe_when_sha_matches(self, repo: GitRepo) -> None:
        _add_origin(repo)
        head = repo.get_head_sha()
        assert repo.check_remote_head_matches("main", head) is True


# ---------------------------------------------------------------------------
# get_head_sha / get_diff
# ---------------------------------------------------------------------------


class TestHeadSha:
    def test_get_head_sha_returns_40_char_hex(self, repo: GitRepo) -> None:
        sha = repo.get_head_sha()
        assert len(sha) == 40
        int(sha, 16)

    def test_get_diff_returns_diff_output(self, repo: GitRepo) -> None:
        base = repo.get_head_sha()
        (repo.path / "file.txt").write_text("diff content\n")
        repo.commit("add file", "Bot", "bot@example.com")
        diff = repo.get_diff(base)
        assert "diff content" in diff


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_restores_tracked_files(self, repo: GitRepo) -> None:
        (repo.path / "README.md").write_text("dirty\n")
        assert repo.is_clean() is False
        repo.rollback()
        assert repo.is_clean() is True
        assert (repo.path / "README.md").read_text() == "init\n"

    def test_rollback_leaves_worktree_clean(self, repo: GitRepo) -> None:
        (repo.path / "README.md").write_text("dirty\n")
        repo.rollback()
        assert repo.is_clean() is True

    def test_rollback_preserves_untracked_files(self, repo: GitRepo) -> None:
        (repo.path / "untracked.txt").write_text("keep me\n")
        (repo.path / "README.md").write_text("dirty\n")
        repo.rollback()
        assert (repo.path / "untracked.txt").read_text() == "keep me\n"

    def test_rollback_discards_staged_changes(self, repo: GitRepo) -> None:
        (repo.path / "README.md").write_text("dirty\n")
        subprocess.run(["git", "add", "README.md"], cwd=repo.path, check=True)
        assert repo.is_clean() is False
        repo.rollback()
        assert repo.is_clean() is True
        assert (repo.path / "README.md").read_text() == "init\n"


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestSafety:
    def test_commit_uses_configured_prefix(self, repo: GitRepo) -> None:
        prefix = "fix: address AI review feedback"
        (repo.path / "file.txt").write_text("data\n")
        repo.commit(f"{prefix} — iteration 1", "Bot", "bot@example.com")
        log = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=repo.path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert log.stdout.strip().startswith(prefix)

    def test_requires_clean_worktree_concept(self, repo: GitRepo) -> None:
        """is_clean() gives callers a way to enforce the clean-worktree precondition."""
        assert repo.is_clean() is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_git_error_on_bad_command(self, repo: GitRepo) -> None:
        with pytest.raises(GitError, match="failed"):
            repo._run("not-a-real-command")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _commit_count(path: Path) -> int:
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _git_cfg(path: Path, key: str, value: str) -> None:
    subprocess.run(["git", "config", key, value], cwd=path, check=True, capture_output=True)


def _add_origin(repo: GitRepo) -> None:
    """Set up a bare clone as origin so fetch/push operations work."""
    # Rename default branch to main first so the bare clone has it.
    subprocess.run(
        ["git", "branch", "-M", "main"],
        cwd=repo.path,
        check=True,
        capture_output=True,
    )
    bare = repo.path.parent / f"{repo.path.name}-origin.git"
    subprocess.run(
        ["git", "clone", "--bare", str(repo.path), str(bare)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(bare)],
        cwd=repo.path,
        check=True,
        capture_output=True,
    )
