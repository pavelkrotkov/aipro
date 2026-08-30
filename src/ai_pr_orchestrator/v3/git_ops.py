"""V3 git worktree operations (issue #55).

Subprocess-backed implementation of
:class:`~ai_pr_orchestrator.v3.interfaces.GitOperations`: branch off the
repository's default branch, materialize an isolated worktree, commit with an
explicit identity, and push.

Every git invocation carries an explicit identity (``-c user.name``/
``user.email``) or takes one as a parameter: the implementation never reads
ambient global git config, so a machine's default identity cannot silently
author V3 commits. All failures surface as :class:`GitOpsError` with the
command and its stderr attached — policy callers must not need to parse
subprocess results.

No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitOpsError(RuntimeError):
    """Raised when a git operation fails; message names the command and stderr."""


class GitWorktreeOps:
    """Production :class:`GitOperations` implementation on top of the git CLI.

    ``repo_root`` anchors every operation; worktrees are created *inside* the
    repository checkout by default (``.worktrees/<branch>``), matching the
    cleanup-TTL sweep's expectations. Each call is one primitive so the
    foreman composes the lifecycle and can fake any step in tests.
    """

    def __init__(self, repo_root: str | Path) -> None:
        self._root = Path(repo_root)

    # --- primitives ---------------------------------------------------------

    def default_branch(self) -> str:
        out = self._run("symbolic-ref", "refs/remotes/origin/HEAD")
        # "refs/remotes/origin/main\n" -> "main"
        return out.strip().rsplit("/", 1)[-1]

    def create_branch(self, branch: str, from_ref: str) -> None:
        self._run("branch", branch, from_ref)

    def create_worktree(self, path: str, branch: str) -> str:
        workdir = str(Path(path))
        self._run("worktree", "add", workdir, branch)
        return workdir

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        cwd = self._workdir(workdir)
        self._run(
            "-C",
            str(cwd),
            "-c",
            f"user.name={name}",
            "-c",
            f"user.email={email}",
            "add",
            "-A",
        )
        # commit-tree via commit; capture the SHA rather than trusting output
        # formatting.
        self._run(
            "-C",
            str(cwd),
            "-c",
            f"user.name={name}",
            "-c",
            f"user.email={email}",
            "commit",
            "-m",
            message,
            "-q",
        )
        out = self._run("-C", str(cwd), "rev-parse", "HEAD")
        return out.strip()

    def commit_count(self, workdir: str, base_ref: str) -> int:
        out = self._run("-C", str(self._workdir(workdir)), "rev-list", "--count", f"{base_ref}..HEAD")
        return int(out.strip())

    def push(self, branch: str) -> None:
        self._run("push", "-u", "origin", branch)

    def cleanup_worktree(self, path: str) -> None:
        self._run("worktree", "remove", str(Path(path)), "--force")

    # --- helpers -------------------------------------------------------------

    def _workdir(self, workdir: str) -> Path:
        path = Path(workdir)
        if not path.is_dir():
            raise GitOpsError(f"workdir {workdir} does not exist")
        return path

    def _run(self, *args: str) -> str:
        cmd = ("git", *args)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise GitOpsError(f"git binary not found: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise GitOpsError(
                f"git {' '.join(args)} failed (exit {exc.returncode}): "
                f"{exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc
        return proc.stdout
