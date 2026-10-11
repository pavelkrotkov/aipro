"""V3 git worktree operations (issue #55).

Subprocess-backed implementation of
:class:`~ai_pr_orchestrator.v3.interfaces.GitOperations`: branch off the
repository's default branch, materialize an isolated worktree, commit with an
explicit identity, and push.

Git author identity is explicit, and authoritative invocations ignore ambient
Git configuration and reject executable repository settings. Hooks are disabled.
Worker isolation of credentials and shared Git metadata remains a deployment
requirement; subprocess hardening is not an OS security boundary.

All failures surface as :class:`GitOpsError` with the command and stderr
attached — policy callers must not need to parse subprocess results.

No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
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
        env = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSH_AUTH_SOCK")
            if key in os.environ
        }
        env.update(
            GIT_TERMINAL_PROMPT="0",
            GIT_SSH_COMMAND="ssh -o BatchMode=yes",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
        )
        self._env = env
        try:
            self._origin_url = self._run(
                "config", "--local", "--no-includes", "--get", "remote.origin.url"
            ).strip()
        except GitOpsError:
            self._origin_url = ""

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

    def create_worktree(self, path: str, branch: str, *, run_id: str, work_item_id: str) -> str:
        workdir = self._checkout_path(path)
        self._record_checkout(workdir, "worker", branch, run_id, work_item_id)
        self._run("worktree", "add", str(workdir), branch)
        self._verify_developer_repository(workdir)
        self._mark_materialized(workdir)
        return str(workdir)

    def create_review_checkout(
        self, path: str, source_workdir: str, head_sha: str, *, run_id: str, work_item_id: str
    ) -> str:
        """Clone an independent, remote-free repository for one reviewer."""
        workdir = self._checkout_path(path)
        source = self._workdir(source_workdir)
        self.verify_checkout(str(source), run_id, work_item_id)
        self._record_checkout(workdir, "reviewer", head_sha, run_id, work_item_id)
        self._run("clone", "--no-local", "--no-checkout", str(source), str(workdir))
        self._run("-C", str(workdir), "remote", "remove", "origin")
        self._run("-C", str(workdir), "checkout", "--detach", head_sha)
        actual = self.head_sha(str(workdir))
        if actual != head_sha:
            raise GitOpsError(f"review checkout {workdir} expected HEAD {head_sha}, found {actual}")
        self._mark_materialized(workdir)
        return str(workdir)

    def head_sha(self, workdir: str) -> str:
        return self._run("-C", str(self._workdir(workdir)), "rev-parse", "HEAD").strip()

    def current_branch(self, workdir: str) -> str:
        return self._run(
            "-C", str(self._workdir(workdir)), "rev-parse", "--abbrev-ref", "HEAD"
        ).strip()

    def repo_instructions(self, workdir: str) -> str:
        root = self._workdir(workdir)
        parts = []
        for name in ("AGENTS.md", "CLAUDE.md"):
            path = root / name
            if path.is_symlink():
                raise GitOpsError(f"repository instruction file must not be a symlink: {path}")
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

    def patch_digest(self, workdir: str, base_ref: str) -> str:
        diff = self._run(
            "-C",
            str(self._workdir(workdir)),
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            f"{base_ref}...HEAD",
        )
        return hashlib.sha256(diff.encode("utf-8")).hexdigest()

    def push(self, branch: str) -> None:
        self._run("push", "-u", "origin", branch)

    def push_sha(self, head_sha: str, branch: str) -> None:
        self._run("push", "origin", f"{head_sha}:refs/heads/{branch}")

    def cleanup_worktree(self, path: str) -> None:
        workdir = self._checkout_path(path)
        if workdir.is_symlink() or workdir.resolve() == self._root.resolve():
            raise GitOpsError(f"refusing to remove unowned developer checkout: {workdir}")
        if not workdir.exists():
            return
        self._verify_developer_cleanup(workdir)
        self._run("worktree", "remove", str(workdir), "--force")
        self._checkout_index(workdir).unlink()

    def _verify_developer_cleanup(self, workdir: Path) -> None:
        branch = self.current_branch(str(workdir))
        self._verify_checkout_record(workdir, "worker", branch)
        self._verify_developer_repository(workdir)

    def _verify_developer_repository(self, workdir: Path) -> None:
        git_dir = Path(
            self._run("-C", str(workdir), "rev-parse", "--absolute-git-dir").strip()
        ).resolve()
        common = Path(
            self._run("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
        ).resolve()
        if not git_dir.is_relative_to(common / "worktrees"):
            raise GitOpsError(f"developer checkout does not belong to this repository: {workdir}")

    def cleanup_review_checkout(self, path: str, head_sha: str) -> None:
        workdir = self._checkout_path(path)
        if workdir.is_symlink() or workdir.resolve() == self._root.resolve():
            raise GitOpsError(f"refusing to remove unowned review checkout: {workdir}")
        if not workdir.exists():
            return
        self._verify_checkout_record(workdir, "reviewer", head_sha)
        shutil.rmtree(workdir)
        self._checkout_index(workdir).unlink()

    def _verify_checkout_record(self, workdir: Path, role: str, ref: str) -> None:
        record = self._checkout_record(workdir)
        if record.get("materialized") is not True or (record.get("role"), record.get("ref")) != (
            role,
            ref,
        ):
            raise GitOpsError(f"refusing to remove a different {role} checkout: {workdir}")

    def verify_checkout(self, path: str, run_id: str, work_item_id: str) -> None:
        record = self._checkout_record(self._checkout_path(path))
        if not record.get("materialized") or (record["run_id"], record["work_item_id"]) != (
            run_id,
            work_item_id,
        ):
            raise GitOpsError("checkout belongs to a different issue or run")

    def _checkout_index(self, workdir: Path) -> Path:
        common = self._run("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
        key = hashlib.sha256(str(workdir.resolve()).encode()).hexdigest()
        return Path(common) / "aipro-checkouts" / (key + ".json")

    def _record_checkout(self, workdir, role, ref, run_id, work_item_id) -> None:
        index = self._checkout_index(workdir)
        index.parent.mkdir(exist_ok=True)
        record = {
            "path": str(workdir.resolve()),
            "role": role,
            "ref": ref,
            "run_id": run_id,
            "work_item_id": work_item_id,
            "created_at": datetime.now(UTC).isoformat(),
            "materialized": False,
        }
        with index.open("x", encoding="utf-8") as target:
            json.dump(record, target)

    def _checkout_record(self, workdir: Path) -> dict:
        index = self._checkout_index(workdir)
        if index.is_symlink() or not index.is_file():
            raise GitOpsError(f"checkout has no protected ownership record: {workdir}")
        record = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("path") != str(workdir.resolve()):
            raise GitOpsError(f"checkout ownership record has a different path: {workdir}")
        return record

    def _mark_materialized(self, workdir: Path) -> None:
        index = self._checkout_index(workdir)
        record = {**self._checkout_record(workdir), "materialized": True}
        index.write_text(json.dumps(record), encoding="utf-8")

    def checkout_inventory(self) -> list[dict]:
        common = self._run("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
        records = [
            self._read_checkout_index(index)
            for index in sorted((Path(common) / "aipro-checkouts").glob("*.json"))
        ]
        known = {record["path"] for record in records}
        unowned = [
            {"path": path, "role": "unowned"}
            for path in self._worktree_paths()
            if path not in known
        ]
        return records + unowned

    def _read_checkout_index(self, index: Path) -> dict:
        if index.is_symlink():
            raise GitOpsError("checkout inventory contains a symlink")
        record = json.loads(index.read_text(encoding="utf-8"))
        self._validate_checkout_record(record)
        if self._checkout_index(Path(record["path"])) != index:
            raise GitOpsError("checkout inventory has a different protected identity")
        return record

    @staticmethod
    def _validate_checkout_record(record) -> None:
        if not isinstance(record, dict):
            raise GitOpsError("checkout inventory contains invalid metadata")
        for key in ("path", "role", "ref", "run_id", "work_item_id", "created_at"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise GitOpsError("checkout inventory contains invalid metadata")
        if (
            record["role"] not in ("worker", "reviewer")
            or type(record.get("materialized")) is not bool
        ):
            raise GitOpsError("checkout inventory contains invalid ownership")

    def retire_checkout(self, record: dict, run_id: str, work_item_id: str) -> None:
        if not Path(record["path"]).exists():
            self.forget_missing_checkout(record)
            return
        self.verify_checkout(record["path"], run_id, work_item_id)
        if record["role"] == "worker":
            self.cleanup_worktree(record["path"])
        else:
            self.cleanup_review_checkout(record["path"], record["ref"])

    def checkout_expired(self, record: dict, since: datetime, now: datetime, ttl: int) -> bool:
        from .reconcile import WorktreeObservation, is_orphan_worktree

        if (now - since).total_seconds() < ttl:
            return False
        observation = WorktreeObservation(
            record["path"], record["ref"], max(since, self.checkout_activity(record))
        )
        return is_orphan_worktree(
            observation, branch_has_live_lease=False, now=now, ttl_seconds=ttl
        )

    def forget_missing_checkout(self, record: dict) -> None:
        workdir = Path(record["path"])
        if workdir.exists() or workdir.is_symlink():
            raise GitOpsError("cannot forget a checkout that still exists")
        if self._checkout_record(workdir) != record:
            raise GitOpsError("checkout ownership changed during cleanup")
        self._checkout_index(workdir).unlink()

    def checkout_activity(self, record: dict) -> datetime:
        workdir = Path(record["path"])
        latest = datetime.fromisoformat(record["created_at"]).timestamp()
        if workdir.is_symlink():
            raise GitOpsError("checkout activity cannot follow a symlink")
        for root, dirs, files in os.walk(workdir, followlinks=False):
            latest = max(
                latest, Path(root).lstat().st_mtime, self._entry_activity(Path(root), dirs + files)
            )
        return datetime.fromtimestamp(latest, UTC)

    @staticmethod
    def _entry_activity(root: Path, names: list[str]) -> float:
        return max(((root / name).lstat().st_mtime for name in names), default=0)

    def _worktree_paths(self) -> list[str]:
        output = self._run("worktree", "list", "--porcelain", "-z")
        return [
            field.removeprefix("worktree ")
            for field in output.split("\0")
            if field.startswith("worktree ")
        ]

    # --- helpers -------------------------------------------------------------

    def _checkout_path(self, path: str) -> Path:
        workdir = Path(path)
        return workdir if workdir.is_absolute() else self._root / workdir

    def _workdir(self, workdir: str) -> Path:
        path = Path(workdir)
        if not path.is_dir():
            raise GitOpsError(f"workdir {workdir} does not exist")
        return path

    def _check_authority(self, args: tuple[str, ...]) -> None:
        command = args[2:] if args[0] == "-C" else args
        if command[0] == "config":
            return
        cwd = args[1] if args[0] == "-C" else str(self._root)
        self._check_executable_config(cwd)
        if command[0] == "push":
            if not self._origin_url:
                raise GitOpsError("push refused: no trusted origin URL at initialization")
            origin = self._run(
                "config", "--local", "--no-includes", "--get", "remote.origin.url"
            ).strip()
            if origin != self._origin_url:
                raise GitOpsError("origin URL changed since GitWorktreeOps initialization")

    def _check_executable_config(self, cwd: str) -> None:
        config = self._run("-C", cwd, "config", "--list", "--no-includes", "-z")
        executable = (
            "filter.",
            "url.",
            "include.",
            "includeif.",
            "credential.",
            "gpg.",
            "diff.",
            "merge.",
            "alias.",
            "http.",
            "https.",
            "protocol.",
            "push.",
            "commit.gpgsign",
            "core.sshcommand",
            "core.worktree",
            "core.alternaterefscommand",
            "core.gitproxy",
            "core.pager",
            "core.editor",
            "core.askpass",
            "sendemail.",
            "sequence.editor",
            "transfer.",
            "remote.origin.push",
            "remote.origin.receivepack",
            "remote.origin.uploadpack",
            "remote.origin.vcs",
            "remote.origin.mirror",
            "remote.origin.proxy",
        )
        unsafe = next(
            (
                entry.partition("\n")[0]
                for entry in config.split("\0")
                if entry.startswith(executable)
            ),
            None,
        )
        if unsafe:
            raise GitOpsError(f"unsafe Git configuration in {cwd}: {unsafe}")

    def _run(self, *args: str) -> str:
        self._check_authority(args)
        cmd = ("git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args)
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
