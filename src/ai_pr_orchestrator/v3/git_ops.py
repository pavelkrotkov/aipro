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

import hashlib
import os
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

    def __init__(
        self, repo_root: str | Path, *, timeout: float = 60.0, git_timeout: float | None = None
    ) -> None:
        self._root = Path(repo_root)
        #: Per-invocation subprocess bound. Every git call shares this budget so a
        #: hung remote (credential prompt, stalled SSH, lock contention) cannot pin
        #: a lane worker forever; a breach raises GitOpsError naming the command.
        self._timeout = git_timeout if git_timeout is not None else timeout
        #: Non-interactive child environment: never prompt for credentials and
        #: make SSH/host-key interaction fail fast instead of hanging a lane.
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_SSH_COMMAND"] = env.get("GIT_SSH_COMMAND", "ssh") + " -o BatchMode=yes"
        self._env = env

    # --- primitives ---------------------------------------------------------

    def default_branch(self) -> str:
        out = self._run("symbolic-ref", "refs/remotes/origin/HEAD")
        # "refs/remotes/origin/main\n" -> "main"
        return out.strip().rsplit("/", 1)[-1]

    def create_branch(self, branch: str, from_ref: str) -> None:
        # Idempotent: re-claiming a requeued item may find the branch already
        # present from a prior pass; recreating it would exit 1.
        if self._run("branch", "--list", branch).strip():
            return
        self._run("branch", branch, from_ref)

    def create_worktree(self, path: str, branch: str) -> str:
        workdir = Path(path)
        if not workdir.is_absolute():
            # git resolves a relative <path> against the invocation cwd (the
            # repo root), but the caller uses the returned path from its own
            # process cwd: hand back an absolute path rooted at the repo
            # (round-2 #11).
            workdir = self._root / workdir
        self._run("worktree", "add", str(workdir), branch)
        return str(workdir)

    def head_sha(self, workdir: str) -> str:
        return self._run("-C", str(self._workdir(workdir)), "rev-parse", "HEAD").strip()

    def repo_instructions(self, workdir: str) -> str:
        root = self._workdir(workdir)
        parts = []
        for name in ("AGENTS.md", "CLAUDE.md"):
            path = root / name
            if path.is_file():
                parts.append(f"{name}:\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def write_issue_description(self, workdir: str, description: str) -> tuple[str, str]:
        """Cache complete input in worktree-private metadata, never staged content."""
        git_dir = self._run("-C", str(self._workdir(workdir)), "rev-parse", "--absolute-git-dir")
        content = description.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        path = Path(git_dir.strip()) / f"aipro-issue-{digest}.md"
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            pass  # Identical inputs reuse a cache; never overwrite an in-flight input.
        if path.read_bytes() != content:
            raise GitOpsError(f"issue description cache does not match fetched body: {path}")
        return str(path), digest

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        cwd = self._workdir(workdir)
        # A lane run that made no edits is a no-op, not an error: ``git commit``
        # with nothing staged exits 1, which would surface as a spurious failure.
        # Check ``status --porcelain`` (untracked files included, so ``add -A``
        # semantics) and short-circuit to the current HEAD when the worktree is
        # clean.
        porcelain = self._run("-C", str(cwd), "status", "--porcelain", "-uall")
        if not porcelain.strip():
            out = self._run("-C", str(cwd), "rev-parse", "HEAD")
            return out.strip()
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
        out = self._run(
            "-C", str(self._workdir(workdir)), "rev-list", "--count", f"{base_ref}..HEAD"
        )
        return int(out.strip())

    def changed_files(self, workdir: str, base_ref: str | None = None) -> list[str]:
        """The paths the worktree has touched relative to ``base_ref``.

        PR #73 review thread 8 / issue #78: the production CAO controller
        reports ``changed_files=[]`` for every completed session, so the
        foreman's policy and budget checks (``_policy_violation``,
        ``_commit_and_push``) cannot detect a coder or reviewer editing
        ``.github/workflows/`` until the commit itself runs. This method
        derives the changed paths from the worktree so the policy layer
        sees the real diff, not an empty signal.
        """
        cwd = str(self._workdir(workdir))
        committed = ""
        if base_ref is not None:
            committed = self._run(
                "-C", cwd, "diff", "--name-only", "--no-renames", "-z", f"{base_ref}...HEAD"
            )
        pending = self._run("-C", cwd, "status", "--porcelain", "-uall", "--no-renames", "-z")
        paths = [path for path in committed.split("\x00") if path]
        paths.extend(record[3:] for record in pending.split("\x00") if record)
        return list(dict.fromkeys(paths))

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
        # Non-interactive + bounded: a git call must never hang waiting on a
        # credential/terminal prompt or a stalled SSH agent. We close stdin and
        # force batch behaviour so the only way a call ends is success, a clean
        # failure, or the explicit timeout below (any of which surfaces as
        # GitOpsError rather than an unbounded block).
        env = dict(self._env)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=env,
                timeout=self._timeout,
            )
        except FileNotFoundError as exc:
            raise GitOpsError(f"git binary not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitOpsError(
                f"git {' '.join(args)} timed out after {self._timeout}s "
                f"(stdout: {exc.stdout or 'none'}; stderr: {exc.stderr or 'none'})"
            ) from exc
        if proc.returncode != 0:
            raise GitOpsError(
                f"git {' '.join(args)} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout
