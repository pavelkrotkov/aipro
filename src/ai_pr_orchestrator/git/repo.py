"""Git repository operations: worktree status, commit, push, rollback, and HEAD SHA detection."""

from __future__ import annotations

import os
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

    def _run(
        self,
        *args: str,
        check: bool = True,
        timeout: int = 30,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        run_env: dict[str, str] | None = None
        if env:
            run_env = os.environ.copy()
            run_env.update(env)
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self._path,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout,
                env=run_env,
            )
        except subprocess.CalledProcessError as exc:
            raise GitError(
                f"git {' '.join(args)} failed (exit {exc.returncode}): {exc.stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from exc
        except OSError as exc:
            raise GitError(f"Failed to execute git: {exc}") from exc

    def is_clean(self) -> bool:
        """Return True if the working tree has no modified or untracked files."""
        result = self._run("status", "--porcelain")
        return result.stdout.strip() == ""

    def commit(self, message: str, author_name: str, author_email: str) -> str | None:
        """Stage all changes (including new files) and commit.

        Returns the commit SHA, or None if nothing to commit.
        """
        if self.is_clean():
            return None
        self._run("add", "--all")
        committer_env = {
            "GIT_COMMITTER_NAME": author_name,
            "GIT_COMMITTER_EMAIL": author_email,
        }
        self._run(
            "commit",
            "-m",
            message,
            "--author",
            f"{author_name} <{author_email}>",
            env=committer_env,
        )
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip()

    def push(self, branch: str) -> None:
        """Push the current HEAD to the remote branch. Never force-pushes."""
        self._run("push", "origin", f"HEAD:refs/heads/{branch}")

    def fetch_remote_head(self, branch: str) -> str | None:
        """Fetch and return the SHA of the remote branch HEAD, or None if it doesn't exist."""
        try:
            self._run("fetch", "origin", branch)
            result = self._run("rev-parse", f"origin/{branch}")
            return result.stdout.strip()
        except GitError as exc:
            msg = str(exc).lower()
            if "couldn't find remote ref" in msg or "ambiguous argument" in msg:
                return None
            raise

    def get_head_sha(self) -> str:
        """Return the SHA of the current HEAD."""
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip()

    def get_diff(self, base: str) -> str:
        """Return the diff between the given base ref and HEAD."""
        result = self._run("diff", base, "HEAD")
        return result.stdout

    def rollback(self) -> None:
        """Discard all working-tree changes and remove untracked files, restoring pre-coder state.

        Note: this also removes untracked files that existed before the coder ran.
        """
        self._run("reset", "--hard", "HEAD")
        self._run("clean", "-fd")

    def check_remote_head_matches(self, branch: str, expected_sha: str) -> bool:
        """Return True if the remote HEAD for *branch* matches *expected_sha*.

        Returns False if the remote branch does not exist.
        """
        remote_sha = self.fetch_remote_head(branch)
        if remote_sha is None:
            return False
        return remote_sha == expected_sha
