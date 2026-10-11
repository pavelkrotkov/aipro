"""Tests for V3 git operations (issue #55): protocol conformance and the
subprocess implementation against a real throwaway repository."""

from __future__ import annotations

import hashlib
import subprocess
import sys
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
        # PR #73 review thread 8: track per-worktree touched paths so the
        # foreman's ``_policy_violation`` call (which now derives changed
        # files from the worktree) can be exercised in unit tests.
        self.touched: dict[str, list[str]] = {}
        self.pushed: list[str] = []
        self._seq = 0

    def default_branch(self) -> str:
        self.calls.append(("default_branch",))
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.calls.append(("create_branch", branch, from_ref))
        self.branches.add(branch)

    def create_worktree(self, path: str, branch: str, *, run_id: str, work_item_id: str) -> str:
        self.calls.append(("create_worktree", path, branch))
        self.worktrees[path] = branch
        return path

    def create_review_checkout(
        self, path: str, source_workdir: str, head_sha: str, *, run_id: str, work_item_id: str
    ) -> str:
        self.calls.append(("create_review_checkout", path, source_workdir, head_sha))
        self.worktrees[path] = f"detached:{head_sha}"
        return path

    def head_sha(self, workdir: str) -> str:
        self.calls.append(("head_sha", workdir))
        return self.commits.get(workdir, ["sha"])[-1]

    def verify_checkout(self, path: str, run_id: str, work_item_id: str) -> None:
        pass

    def current_branch(self, workdir: str) -> str:
        self.calls.append(("current_branch", workdir))
        return self.worktrees.get(workdir, self.default)

    def repo_instructions(self, workdir: str) -> str:
        self.calls.append(("repo_instructions", workdir))
        return ""

    def write_issue_description(self, workdir: str, description: str) -> tuple[str, str]:
        raise NotImplementedError("use real GitWorktreeOps for issue input delivery tests")

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        self.calls.append(("commit", workdir, message, name, email))
        self._seq += 1
        sha = f"sha-{self._seq}"
        self.commits.setdefault(workdir, []).append(sha)
        return sha

    def commit_count(self, workdir: str, base_ref: str) -> int:
        self.calls.append(("commit_count", workdir, base_ref))
        return len(self.commits.get(workdir, []))

    def changed_files(self, workdir: str, base_ref: str | None = None) -> list[str]:
        self.calls.append(("changed_files", workdir, base_ref))
        return list(self.touched.get(workdir, []))

    def patch_digest(self, workdir: str, base_ref: str) -> str:
        self.calls.append(("patch_digest", workdir, base_ref))
        return "|".join(sorted(self.touched.get(workdir, [])))

    def push(self, branch: str) -> None:
        self.calls.append(("push", branch))
        self.pushed.append(branch)

    def push_sha(self, head_sha: str, branch: str) -> None:
        self.calls.append(("push_sha", head_sha, branch))
        self.pushed.append(branch)

    def cleanup_worktree(self, path: str) -> None:
        self.calls.append(("cleanup_worktree", path))
        self.worktrees.pop(path, None)

    def cleanup_review_checkout(self, path: str, head_sha: str) -> None:
        self.calls.append(("cleanup_review_checkout", path, head_sha))
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


def test_repo_instructions_reads_root_agent_files(real_repo: Path):
    (real_repo / "AGENTS.md").write_text("agent rules\n")
    (real_repo / "CLAUDE.md").write_text("claude rules\n")
    instructions = GitWorktreeOps(real_repo).repo_instructions(str(real_repo))
    assert instructions == "AGENTS.md:\nagent rules\n\n\nCLAUDE.md:\nclaude rules\n"


def test_repo_instructions_reject_symlinks_outside_worktree(real_repo: Path):
    secret = real_repo.parent / "host-secret"
    secret.write_text("do not disclose\n")
    (real_repo / "AGENTS.md").symlink_to(secret)
    with pytest.raises(GitOpsError, match="must not be a symlink"):
        GitWorktreeOps(real_repo).repo_instructions(str(real_repo))


def test_default_branch_and_branch_creation(real_repo: Path):
    ops = GitWorktreeOps(real_repo)
    assert ops.default_branch() == "main"
    assert ops.current_branch(str(real_repo)) == "main"
    assert (
        ops.head_sha(str(real_repo))
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=real_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
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
    workdir = ops.create_worktree(str(wt), "feat/y", run_id="test-run", work_item_id="owner/repo#1")
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


def test_patch_digest_tracks_content_not_commit_metadata(real_repo: Path):
    ops = GitWorktreeOps(real_repo)
    ops.create_branch("feature", "main")
    subprocess.run(
        ["git", "checkout", "feature"],
        cwd=real_repo,
        check=True,
        capture_output=True,
    )
    (real_repo / "file.txt").write_text("changed\n")
    first_sha = ops.commit(str(real_repo), "first message", name="T", email="t@example.com")
    first = ops.patch_digest(str(real_repo), "main")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@example.com",
            "commit",
            "--amend",
            "-m",
            "metadata only",
        ],
        cwd=real_repo,
        check=True,
        capture_output=True,
    )
    assert ops.head_sha(str(real_repo)) != first_sha
    assert ops.patch_digest(str(real_repo), "main") == first
    (real_repo / "file.txt").write_text("different content\n")
    ops.commit(str(real_repo), "different", name="T", email="t@example.com")
    assert ops.patch_digest(str(real_repo), "main") != first


def test_push_failure_raises_gitops_error(real_repo: Path):
    ops = GitWorktreeOps(real_repo)
    with pytest.raises(GitOpsError, match="push"):
        ops.push("no-such-branch")


def test_commit_on_clean_worktree_is_a_noop(real_repo: Path, tmp_path: Path):
    """A lane run with no edits must not surface ``git commit`` exit-1 as failure."""
    ops = GitWorktreeOps(real_repo)
    base = ops.default_branch()
    ops.create_branch("feat/noop", base)
    wt = tmp_path / "wt-noop"
    workdir = ops.create_worktree(
        str(wt), "feat/noop", run_id="test-run", work_item_id="owner/repo#1"
    )
    head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True, check=True
    ).stdout.strip()

    # No changes staged: commit short-circuits to the current HEAD, no error.
    sha = ops.commit(workdir, "noop", name="Pavel Krotkov", email="pavel.krotkov@gmail.com")

    assert sha == head_before
    count = subprocess.run(
        ["git", "rev-list", "--count", f"{base}..HEAD"],
        cwd=wt,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert count == "0"  # nothing was committed


def test_subprocesses_are_noninteractive_and_bounded(real_repo: Path):
    """git calls must never prompt for credentials or hang without a bound."""
    ops = GitWorktreeOps(real_repo)
    assert ops._env["GIT_TERMINAL_PROMPT"] == "0"
    assert "BatchMode=yes" in ops._env["GIT_SSH_COMMAND"]


def test_timeout_surfaces_as_gitops_error(real_repo: Path):
    ops = GitWorktreeOps(real_repo, timeout=0.0001)
    with pytest.raises(GitOpsError, match="timed out"):
        ops.default_branch()


def test_relative_worktree_path_returns_absolute_path(real_repo: Path):
    """A relative worktree path is resolved against the repo root and handed
    back ABSOLUTE, so callers can use the returned path from any cwd (#11)."""
    import os

    ops = GitWorktreeOps(real_repo)
    base = ops.default_branch()
    ops.create_branch("feat/rel", base)
    returned = ops.create_worktree(
        "rel-wt", "feat/rel", run_id="test-run", work_item_id="owner/repo#1"
    )
    try:
        assert os.path.isabs(returned)
        assert Path(returned) == real_repo / "rel-wt"
        assert Path(returned).is_dir()  # usable from the caller's own cwd
    finally:
        ops.cleanup_worktree(returned)


@pytest.mark.parametrize(
    "body", ["x" * 8_001, "Requirements ü. " * 5_000 + "Final AC: preserve me."]
)
def test_bounded_prompts_deliver_complete_issue_context(real_repo: Path, tmp_path: Path, body):
    from tests.unit.test_v3_foreman import ISSUE, ScriptedExecutor, _foreman, _gate, _ready_fake

    ops = GitWorktreeOps(real_repo)
    ops.create_branch("issue84", "main")
    workdir = ops.create_worktree(
        str(tmp_path / "worker"), "issue84", run_id="test-run", work_item_id="owner/repo#1"
    )
    fake = _ready_fake()
    fake._issue_bodies[1] = body
    loop, _ = _foreman(fake, ScriptedExecutor(), _gate(), git=ops)
    prompts = [
        loop._coder_prompt(ISSUE, (), workdir),
        loop._reviewer_prompt(
            ISSUE,
            "review-1",
            workdir,
            lane_name="requirements-reviewer",
            head_sha=ops.head_sha(workdir),
            issue_body=body,
        ),
    ]
    path, digest = ops.write_issue_description(workdir, body)
    reference = f"{path!r} ({len(body.encode())} UTF-8 bytes; SHA-256 {hashlib.sha256(body.encode()).hexdigest()})"
    for prompt in prompts:
        assert len(prompt) < 2_000
        assert reference in prompt
    # The worker reads the complete bytes from its actual cwd, not a mocked path.
    observed = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
            path,
        ],
        cwd=workdir,
    )
    assert observed == body.encode()
    assert digest == hashlib.sha256(observed).hexdigest()


def test_issue_input_stays_unstaged_and_shares_worktree_cleanup(real_repo: Path, tmp_path: Path):
    ops = GitWorktreeOps(real_repo)
    ops.create_branch("issue84", "main")
    workdir = ops.create_worktree(
        str(tmp_path / "worker"), "issue84", run_id="test-run", work_item_id="owner/repo#1"
    )
    path, _ = ops.write_issue_description(workdir, "Complete requirements.")
    assert ops.changed_files(workdir) == []
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=True)
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=workdir) == b""
    ops.commit(workdir, "no issue data", name="T", email="t@example.com")
    assert ops.commit_count(workdir, "main") == 0
    assert Path(path).exists()  # retained with a pending/busy worktree
    ops.cleanup_worktree(workdir)
    assert not Path(path).exists()


def test_issue_input_versions_do_not_replace_inflight_context(real_repo: Path, tmp_path: Path):
    ops = GitWorktreeOps(real_repo)
    ops.create_branch("issue84", "main")
    workdir = ops.create_worktree(
        str(tmp_path / "worker"), "issue84", run_id="test-run", work_item_id="owner/repo#1"
    )
    first, _ = ops.write_issue_description(workdir, "Original criteria.")
    second, _ = ops.write_issue_description(workdir, "Updated criteria.")
    assert first != second
    assert Path(first).read_text() == "Original criteria."
    assert Path(second).read_text() == "Updated criteria."
    Path(first).write_text("truncated")
    with pytest.raises(GitOpsError, match="does not match"):
        ops.write_issue_description(workdir, "Original criteria.")


def test_push_sha_uses_explicit_refspec(real_repo: Path, monkeypatch: pytest.MonkeyPatch):
    ops = GitWorktreeOps(real_repo)
    calls = []
    monkeypatch.setattr(ops, "_run", lambda *args, **kwargs: calls.append(args) or "")
    ops.push_sha("abc123", "reviewed")
    assert calls == [("push", "origin", "abc123:refs/heads/reviewed")]


def test_authoritative_git_ignores_worker_hooks_and_inherited_controls(
    real_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    marker = tmp_path / "hook-ran"
    hook = real_repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\nprintf executed > '{marker}'\n")
    hook.chmod(0o755)
    monkeypatch.setenv("AIPRO_AUTHORITY_CANARY", "private")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(hook.parent))
    ops = GitWorktreeOps(real_repo)
    assert "AIPRO_AUTHORITY_CANARY" not in ops._env
    assert "GIT_CONFIG_COUNT" not in ops._env
    assert "GIT_CONFIG_KEY_0" not in ops._env
    (real_repo / "file.txt").write_text("changed\n")
    ops.commit(str(real_repo), "safe", name="T", email="t@example.com")
    assert not marker.exists()


def test_authoritative_git_refuses_worker_filters(real_repo: Path, tmp_path: Path):
    ops = GitWorktreeOps(real_repo)
    marker = tmp_path / "filter-ran"
    (real_repo / ".gitattributes").write_text("*.txt filter=probe\n")
    command = ["git", "config", "--local", "filter.probe.clean", f"sh -c 'touch {marker}; cat'"]
    subprocess.run(command, cwd=real_repo, check=True)
    (real_repo / "file.txt").write_text("changed\n")
    with pytest.raises(GitOpsError, match=r"unsafe Git configuration.*filter\.probe\.clean"):
        ops.commit(str(real_repo), "unsafe", name="T", email="t@example.com")
    assert not marker.exists()


def test_authoritative_push_pins_remote_and_rejects_rewrites(real_repo: Path, tmp_path: Path):
    trusted = tmp_path / "trusted.git"
    redirected = tmp_path / "redirected.git"
    for bare in (trusted, redirected):
        subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(trusted)], cwd=real_repo, check=True)
    ops = GitWorktreeOps(real_repo)
    sha = ops.head_sha(str(real_repo))
    ops.push_sha(sha, "main")
    command = ["git", "-C", str(trusted), "rev-parse", "main"]
    assert subprocess.check_output(command, text=True).strip() == sha

    config_key = f"url.{redirected}.insteadOf"
    command = ["git", "config", "--local", config_key, str(trusted)]
    subprocess.run(command, cwd=real_repo, check=True)
    with pytest.raises(GitOpsError, match=r"unsafe Git configuration.*url\."):
        ops.push_sha(sha, "main")
    subprocess.run(["git", "config", "--local", "--unset", config_key], cwd=real_repo, check=True)
    command = ["git", "remote", "set-url", "origin", str(redirected)]
    subprocess.run(command, cwd=real_repo, check=True)
    with pytest.raises(GitOpsError, match="origin URL changed"):
        ops.push_sha(sha, "main")
    command = ["git", "-C", str(redirected), "show-ref", "--verify", "refs/heads/main"]
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_cleanup_refuses_unowned_dirty_checkout_and_removes_owned_checkout(real_repo, tmp_path):
    ops = GitWorktreeOps(real_repo)
    human = tmp_path / "human"
    subprocess.run(
        ["git", "worktree", "add", "-b", "human", str(human)],
        cwd=real_repo,
        check=True,
        capture_output=True,
    )
    (human / "file.txt").write_text("unsaved human edits")
    with pytest.raises(GitOpsError):
        ops.cleanup_worktree(str(human))
    assert (human / "file.txt").read_text() == "unsaved human edits"
    with pytest.raises(GitOpsError):
        ops.cleanup_worktree(str(real_repo))
    link = tmp_path / "human-link"
    link.symlink_to(human, target_is_directory=True)
    with pytest.raises(GitOpsError):
        ops.cleanup_worktree(str(link))
    ops.create_branch("owned", "main")
    owned = Path(
        ops.create_worktree(
            str(tmp_path / "owned"), "owned", run_id="test-run", work_item_id="owner/repo#1"
        )
    )
    (owned / "file.txt").write_text("unfinished owned edits")
    subprocess.run(
        ["git", "checkout", "-b", "changed-identity"], cwd=owned, check=True, capture_output=True
    )
    with pytest.raises(GitOpsError):
        ops.cleanup_worktree(str(owned))
    assert owned.exists()
    subprocess.run(["git", "checkout", "owned"], cwd=owned, check=True, capture_output=True)
    ops.cleanup_worktree(str(owned))
    ops.cleanup_worktree(str(owned))
    assert not owned.exists() and human.exists()


def test_checkout_inventory_binds_runs_and_failed_creation_cannot_authorize_human_work(
    real_repo, tmp_path
):
    ops = GitWorktreeOps(real_repo)
    ops.create_branch("owned-inventory", "main")
    owned = ops.create_worktree(
        str(tmp_path / "owned-inventory"),
        "owned-inventory",
        run_id="run-1",
        work_item_id="owner/repo#1",
    )
    ops.verify_checkout(owned, "run-1", "owner/repo#1")
    with pytest.raises(GitOpsError, match="different issue or run"):
        ops.verify_checkout(owned, "run-2", "owner/repo#1")
    review = ops.create_review_checkout(
        str(tmp_path / "review-inventory"),
        owned,
        ops.head_sha(owned),
        run_id="run-1",
        work_item_id="owner/repo#1",
    )
    inventory = ops.checkout_inventory()
    assert {(r["role"], r.get("run_id")) for r in inventory} == {
        ("worker", "run-1"),
        ("reviewer", "run-1"),
        ("unowned", None),
    }
    assert {r["path"] for r in inventory} == {str(real_repo), owned, review}
    human = tmp_path / "failed-allocation"
    subprocess.run(
        ["git", "worktree", "add", "-b", "human-inventory", str(human)],
        cwd=real_repo,
        check=True,
        capture_output=True,
    )
    (human / "file.txt").write_text("irreplaceable human edits")
    with pytest.raises(GitOpsError):
        ops.create_worktree(
            str(human), "human-inventory", run_id="run-3", work_item_id="owner/repo#3"
        )
    with pytest.raises(GitOpsError):
        ops.cleanup_worktree(str(human))
    assert (human / "file.txt").read_text() == "irreplaceable human edits"
    assert (
        next(r for r in ops.checkout_inventory() if r["path"] == str(human))["materialized"]
        is False
    )
