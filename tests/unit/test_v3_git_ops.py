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

    def create_worktree(self, path: str, branch: str) -> str:
        self.calls.append(("create_worktree", path, branch))
        self.worktrees[path] = branch
        return path

    def head_sha(self, workdir: str) -> str:
        self.calls.append(("head_sha", workdir))
        return self.commits.get(workdir, ["sha"])[-1]

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


def test_repo_instructions_reads_root_agent_files(real_repo: Path):
    (real_repo / "AGENTS.md").write_text("agent rules\n")
    (real_repo / "CLAUDE.md").write_text("claude rules\n")
    instructions = GitWorktreeOps(real_repo).repo_instructions(str(real_repo))
    assert instructions == "AGENTS.md:\nagent rules\n\n\nCLAUDE.md:\nclaude rules\n"


def test_default_branch_and_branch_creation(real_repo: Path):
    ops = GitWorktreeOps(real_repo)
    assert ops.default_branch() == "main"
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


def test_commit_on_clean_worktree_is_a_noop(real_repo: Path, tmp_path: Path):
    """A lane run with no edits must not surface ``git commit`` exit-1 as failure."""
    ops = GitWorktreeOps(real_repo)
    base = ops.default_branch()
    ops.create_branch("feat/noop", base)
    wt = tmp_path / "wt-noop"
    workdir = ops.create_worktree(str(wt), "feat/noop")
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
    returned = ops.create_worktree("rel-wt", "feat/rel")
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
    workdir = ops.create_worktree(str(tmp_path / "worker"), "issue84")
    fake = _ready_fake()
    fake._issue_bodies[1] = body
    loop, _ = _foreman(fake, ScriptedExecutor(), _gate(), git=ops)
    prompts = [
        loop._coder_prompt(ISSUE, (), workdir),
        loop._reviewer_prompt(ISSUE, "review-1", workdir),
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
    workdir = ops.create_worktree(str(tmp_path / "worker"), "issue84")
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
    workdir = ops.create_worktree(str(tmp_path / "worker"), "issue84")
    first, _ = ops.write_issue_description(workdir, "Original criteria.")
    second, _ = ops.write_issue_description(workdir, "Updated criteria.")
    assert first != second
    assert Path(first).read_text() == "Original criteria."
    assert Path(second).read_text() == "Updated criteria."
    Path(first).write_text("truncated")
    with pytest.raises(GitOpsError, match="does not match"):
        ops.write_issue_description(workdir, "Original criteria.")
