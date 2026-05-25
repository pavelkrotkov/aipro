"""Git repository operations: worktree status, commit, push, rollback, and HEAD SHA detection."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """Raised when a git operation fails."""


class GitRepo:
    """Thin wrapper around git CLI operations for a local repository."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self._path,
                capture_output=True,
                text=True,
                check=check,
            )
        except subprocess.CalledProcessError as exc:
            raise GitError(
                f"git {' '.join(args)} failed (exit {exc.returncode}): {exc.stderr.strip()}"
            ) from exc

    def is_clean(self) -> bool:
        """Return True if the working tree has no modified or untracked files."""
        result = self._run("status", "--porcelain")
        return result.stdout.strip() == ""

    def commit(self, message: str, author_name: str, author_email: str) -> str | None:
        """Stage all changes and commit. Returns the commit SHA, or None if nothing to commit."""
        if self.is_clean():
            return None
        self._run("add", "--all")
        self._run(
            "commit",
            "-m",
            message,
            "--author",
            f"{author_name} <{author_email}>",
        )
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip()

    def push(self, branch: str) -> None:
        """Push the current branch to origin. Never force-pushes."""
        self._run("push", "origin", branch)

    def fetch_remote_head(self, branch: str) -> str:
        """Fetch and return the SHA of the remote branch HEAD."""
        self._run("fetch", "origin", branch)
        result = self._run("rev-parse", f"origin/{branch}")
        return result.stdout.strip()

    def get_head_sha(self) -> str:
        """Return the SHA of the current HEAD."""
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip()

    def get_diff(self, base: str) -> str:
        """Return the diff between the given base ref and HEAD."""
        result = self._run("diff", base, "HEAD")
        return result.stdout

    def rollback(self) -> None:
        """Discard all working-tree changes (tracked files only), restoring pre-coder state."""
        self._run("checkout", ".")

    def check_remote_head_matches(self, branch: str, expected_sha: str) -> bool:
        """Return True if the remote HEAD for *branch* matches *expected_sha*."""
        remote_sha = self.fetch_remote_head(branch)
        return remote_sha == expected_sha
