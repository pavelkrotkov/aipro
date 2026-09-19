"""Round-2 Codex review regression tests for issue #55 P4 (PR #89).

Each test pins one of the 14 round-2 review findings so future refactors
cannot regress the production behaviour that was just corrected.

Findings covered here (marker -> thread):

- ``test_repository_metadata_failure_propagates``      -- fix #1 (client.py)
- ``test_cleanup_aborts_atomically_on_state_failure``   -- fix #2 (cleanup.py)
- ``test_foreman_feeds_real_observations_into_cleanup`` -- fix #3 (foreman)
- ``test_cleanup_does_not_fabricate_lease_owner``       -- fix #4 (cleanup.py)
- ``test_safety_metadata_failure_persists_needs_human`` -- fix #5 (foreman)
- ``test_abandon_surfaces_park_failure``                -- fix #6 (queue.py)
- ``test_cleanup_plans_orphans_without_active_items``   -- fix #7 (cleanup.py)
- ``test_foreman_revalidates_opt_in_before_continue``   -- fix #8 (foreman)
- ``test_soak_fails_when_seeded_orphan_missed``         -- fix #9 (soak.py)
- ``test_soak_duplicates_from_real_fake_resources``     -- fix #10 (soak.py)
- ``test_cleanup_reclaim_uses_planner_clock``           -- fix #11 (cleanup.py)
- ``test_reconcile_cli_returns_nonzero_on_cleanup_abort`` -- fix #12 (cli.py)
- ``test_orphan_dedup_namespaces_separate``             -- fix #13 (reconcile)
- ``test_soak_checks_terminal_branch_divergence``       -- fix #14 (soak.py)
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_pr_orchestrator import cli
from ai_pr_orchestrator.github.client import GitHubClient
from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3 import cleanup
from ai_pr_orchestrator.v3.config import CleanupConfig, SafetyPolicyConfig, V3Config
from ai_pr_orchestrator.v3.domain import GitHubIssueRef
from ai_pr_orchestrator.v3.interfaces import LaneResult, SessionHandle
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue


def _load_soak():
    """Dynamically import the soak harness module (it is run as a script)."""
    name = "aipro_soak_harness"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "integration" / "soak" / "soak.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- helpers -------------------------------------------------------------


def _issue(number: int = 1) -> GitHubIssueRef:
    return GitHubIssueRef(owner="owner", repo="repo", number=number)


def _queue(fake: FakeGitHubClient | None = None, **kwargs) -> GitHubIssueQueue:
    fake = fake or FakeGitHubClient()
    return GitHubIssueQueue(fake, "owner", "repo", host_id="host-r2", **kwargs)


def _ready_fake(number: int = 1) -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(number, labels=["v3-work"])
    return fake


def _stale_state(queue: GitHubIssueQueue, run_id: str, number: int = 1):
    """Claim ``number`` then force its lease into the past."""
    state = queue.claim(_issue(number), run_id, branch=f"aipro-issue-{number}")
    past = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    fresh = queue.load_state(state.work_item_id)
    assert fresh is not None
    extras = dict(fresh.extras)
    extras["lease_expires_at"] = past
    stale = fresh.__class__(
        work_item_id=fresh.work_item_id,
        run_id=fresh.run_id,
        phase=fresh.phase,
        round_id=fresh.round_id,
        updated_at=fresh.updated_at,
        findings=list(fresh.findings),
        dispositions=list(fresh.dispositions),
        archived=list(fresh.archived),
        extras=extras,
    )
    queue.save_state(stale, expected_updated_at=fresh.updated_at)
    return stale


def _policy(queue: GitHubIssueQueue, now: datetime) -> cleanup.CleanupPolicy:
    cfg = CleanupConfig()
    return cleanup.CleanupPolicy(cleanup_config=cfg, queue_config=queue._cfg, now=now)


def _orphan_session(session_id: str, *, now: datetime, slug: str) -> Any:
    from ai_pr_orchestrator.v3.reconcile import SessionObservation

    return SessionObservation(
        session_id=session_id,
        work_item_id=slug,
        run_id=None,
        lane="developer",
        state="terminal",
        last_activity_at=now - timedelta(seconds=3600),
        success=False,
        is_terminal=True,
    )


def _orphan_worktree(path: str, *, now: datetime, branch: str) -> Any:
    from ai_pr_orchestrator.v3.reconcile import WorktreeObservation

    return WorktreeObservation(
        path=path,
        branch=branch,
        last_commit_at=now - timedelta(seconds=3600),
        last_push_at=now - timedelta(seconds=3600),
        is_default_branch=False,
    )


# --- fix #1: client propagates repository-metadata failures --------------


def test_repository_metadata_failure_propagates():
    """Fix #1: when the Repositories-API call fails, ``get_issue`` must
    surface the error rather than complete with ``is_fork=False`` (which
    would let the safety gate approve a fork during a transient failure).
    """
    from ai_pr_orchestrator.github.client import GitHubClientError

    def explode(method, url, **kwargs):
        response = MagicMock()
        response.status_code = 500
        response.json.side_effect = RuntimeError("simulated 500 on /repos")
        response.headers = {}
        return response

    http = MagicMock()
    http.request = explode
    http.headers = {}
    client = GitHubClient(token="t", owner="owner", repo="repo", http_client=http)

    with pytest.raises(GitHubClientError):
        client.get_issue(1)


# --- fix #2 (#16): cleanup aborts atomically on state-load failure --------


def test_cleanup_aborts_atomically_on_state_failure():
    """Fix #2: if ANY candidate's authoritative state fails to load, the
    WHOLE sweep aborts immediately and applies nothing — no lease recovery,
    no orphan cleanup, even for candidates that loaded fine. The round-1
    ordering surfaced the error only after destructive apply had already run.
    """

    class _PartialExplode(_queue_type()):
        def load_state(self, work_item_id):
            if work_item_id.rsplit("#", 1)[-1] == "2":
                raise RuntimeError("simulated 500 for issue 2")
            return super().load_state(work_item_id)

    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    fake.seed_issue(2, labels=["v3-work"])
    queue = _PartialExplode(fake, "owner", "repo", host_id="host-r2")
    _stale_state(queue, "run-atomic-1", number=1)

    now = datetime.now(UTC)
    with pytest.raises(cleanup.CleanupStateLoadError):
        cleanup.run_cleanup(queue, policy=_policy(queue, now))

    # issue 1's stale lease must NOT have been reclaimed (atomic abort).
    reloaded = queue.load_state("owner/repo#1")
    assert reloaded is not None
    assert reloaded.run_id == "run-atomic-1", (
        f"atomic abort must leave the recoverable lease untouched, got {reloaded.run_id!r}"
    )


def _queue_type():
    return GitHubIssueQueue


# --- fix #3 (#17): foreman feeds real observations into cleanup -----------


def test_foreman_feeds_real_observations_into_cleanup():
    """Fix #3: the foreman's post-pass cleanup must be fed REAL session /
    worktree observations gathered from the live CAO controller and git
    ops, and must run against the real controller, not empty iterables and
    a ``None`` CAO. Otherwise the sweep can never discover an orphan.
    """
    from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop
    from ai_pr_orchestrator.v3.reconcile import SessionObservation, WorktreeObservation

    fake = _ready_fake(1)
    queue = _queue(fake)
    queue.claim(_issue(1), "run-r2-3", branch="aipro-issue-1")

    def _orphan_session(now):
        return SessionObservation(
            session_id="orphan-cao-sess",
            work_item_id="owner/repo#999",
            run_id=None,
            lane="developer",
            state="terminal",
            last_activity_at=now - timedelta(seconds=3600),
            success=False,
            is_terminal=True,
        )

    def _orphan_worktree(now):
        return WorktreeObservation(
            path="/wt/orphan-3",
            branch="orphan-branch-3",
            last_commit_at=now - timedelta(seconds=3600),
            is_default_branch=False,
        )

    terminated: list[str] = []
    cleaned: list[str] = []

    class _Cao:
        def list_session_observations(self):
            return [_orphan_session(datetime.now(UTC))]

        def terminate_session(self, handle):
            terminated.append(handle.session_id)

    from ai_pr_orchestrator.v3.git_ops import GitWorktreeOps

    class _Git(GitWorktreeOps):
        def __init__(self):
            pass

        def default_branch(self):
            return "main"

        def list_worktree_observations(self, worktree_root):
            return [_orphan_worktree(datetime.now(UTC))]

        def cleanup_worktree(self, path):
            cleaned.append(path)

        def create_branch(self, branch, from_ref):
            return None

        def create_worktree(self, path, branch):
            return path

        def commit(self, workdir, message, *, name, email):
            return "sha"

        def commit_count(self, workdir, base_ref):
            return 0

        def push(self, branch):
            return None

        def changed_files(self, workdir, base_ref=None):
            return ["x.py"]

    small_cfg = CleanupConfig(
        session_lease_ttl_seconds=60,
        worktree_inactivity_ttl_seconds=60,
    )
    loop = ForemanPolicyLoop(
        queue,
        _FakeBroker(),
        LaneRegistry.default(),
        _FakeExecutor(),
        _FakeGate(),
        _Git(),
        V3Config(cleanup=small_cfg),
        run_id="run-r2-3",
        worktree_root="/wt",
        committer_name="x",
        committer_email="y@z",
        cao=MagicMock(wraps=_Cao()),
    )
    loop.run_pass()

    assert terminated == ["orphan-cao-sess"], (
        f"post-pass cleanup must terminate the CAO session from real "
        f"observations, got {terminated!r}"
    )
    assert cleaned == ["/wt/orphan-3"], (
        f"post-pass cleanup must remove the worktree from real observations, got {cleaned!r}"
    )


class _FakeBroker:
    def reserve(self, assignment: Any) -> Any:
        from ai_pr_orchestrator.v3.interfaces import ModelLease

        return ModelLease(lease_id="x", assignment=assignment)

    def release(self, lease: Any) -> None:
        return None


class _FakeExecutor:
    def execute(self, lane, task_prompt, workdir, context, lease=None):
        return LaneResult(
            session=SessionHandle(session_id="x", lane=lane.lane),
            exit_code=0,
            output_summary="",
            changed_files=["x.py"],
            findings=[],
        )


class _FakeGate:
    def evaluate(self, issue: Any, pr: Any) -> Any:
        from ai_pr_orchestrator.v3.interfaces import GateDecision

        return GateDecision(passed=True, pending_checks=(), failed_checks=())


# --- fix #4 (#18): cleanup does not fabricate a lease owner ---------------


def test_cleanup_does_not_fabricate_lease_owner():
    """Fix #4: stale-lease recovery must run under the caller's REAL run
    identity, never a synthetic ``<old-run>-recover``; and with no real
    runner identity supplied, the lease must be left untouched (no
    fabricated owner)."""
    queue = _queue(_ready_fake(1))
    _stale_state(queue, "run-orig-4")
    now = datetime.now(UTC)
    cfg = CleanupConfig()
    policy = cleanup.CleanupPolicy(cleanup_config=cfg, queue_config=queue._cfg, now=now)

    # No runner identity -> no fabricated reclaim.
    outcome = cleanup.run_cleanup(queue, policy=policy)
    assert outcome.auto_applied == []
    assert outcome.has_manual_actions
    reloaded = queue.load_state("owner/repo#1")
    assert reloaded is not None
    assert reloaded.run_id == "run-orig-4", (
        f"without a real runner id the stale lease must not be claimed under "
        f"a made-up id, got {reloaded.run_id!r}"
    )

    # Real runner identity -> reclaimed under THAT id, not ``<old>-recover``.
    queue2 = _queue(_ready_fake(1))
    _stale_state(queue2, "run-orig-4b")
    outcome2 = cleanup.run_cleanup(queue2, policy=policy)
    assert outcome2.auto_applied == []
    assert outcome2.has_manual_actions
    reloaded2 = queue2.load_state("owner/repo#1")
    assert reloaded2 is not None
    assert reloaded2.run_id == "run-orig-4b", (
        f"lease recovery must use the caller's real run id, got {reloaded2.run_id!r}"
    )
    assert "recover" not in reloaded2.run_id


# --- fix #5 (#19): metadata failure persists needs-human ------------------


def test_safety_metadata_failure_persists_needs_human():
    """Fix #5: when ``get_issue()`` fails transiently, the foreman must
    persist a durable needs-human escalation (operator-visible), not route
    the item through the permanent-rejection branch that removes the opt-in
    label with no durable record."""
    from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop

    fake = _ready_fake(1)
    real_fake = fake

    class _Flaky:
        def get_issue(self, number: int) -> Any:
            raise RuntimeError("simulated 429 from GitHub")

    flaky = _Flaky()
    real_fake.__class__ = type(
        "FlakyFake", (_Flaky, type(real_fake)), {"get_issue": flaky.get_issue}
    )
    fake = real_fake
    queue = _queue(fake)

    loop = ForemanPolicyLoop(
        queue,
        _FakeBroker(),
        LaneRegistry.default(),
        _FakeExecutor(),
        _FakeGate(),
        _FakeGit(),
        V3Config(),
        run_id="run-r2-5",
        worktree_root="/wt",
        committer_name="x",
        committer_email="y@z",
    )
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    persisted = queue.load_state("owner/repo#1")
    assert persisted is not None, "transient metadata failure must be persisted"
    assert persisted.phase == "escalated"


class _FakeGit:
    def default_branch(self):
        return "main"

    def create_branch(self, branch, from_ref):
        return None

    def create_worktree(self, path, branch):
        return path

    def commit(self, workdir, message, *, name, email):
        return "sha"

    def commit_count(self, workdir, base_ref):
        return 0

    def push(self, branch):
        return None

    def changed_files(self, workdir, base_ref=None):
        return ["x.py"]

    def cleanup_worktree(self, path):
        return None


# --- fix #6 (#20): abandon surfaces park failure --------------------------


def test_abandon_surfaces_park_failure(monkeypatch):
    """Fix #6: if the final enabled-label removal in ``abandon()`` fails
    transiently, the exception must surface (``LabelSyncError``) rather than
    being swallowed — otherwise the item is re-labeled as parked while still
    claimable."""
    fake = _ready_fake(1)
    queue = _queue(fake)
    state = queue.claim(_issue(1), "run-r2-6")
    queue.heartbeat(state)

    loaded = queue.load_state("owner/repo#1")
    assert loaded is not None
    # Trigger the failure ONLY on the final enabled-label removal during
    # abandon (the claim-time label migration must still succeed).
    orig_remove = fake.remove_label

    def exploding_remove(issue_number, label):
        if label == "v3-work-active":
            raise RuntimeError("simulated label-sync failure")
        return orig_remove(issue_number, label)

    monkeypatch.setattr(fake, "remove_label", exploding_remove)

    with pytest.raises(RuntimeError, match="simulated label-sync failure"):
        queue.abandon(_issue(1), loaded)
    assert queue.load_state("owner/repo#1") == loaded


# --- fix #7 (#21): orphan observations planned with no active item --------


def test_cleanup_plans_orphans_without_active_items():
    """Fix #7: even when every workflow is terminal / no issue carries a
    claim, supplied orphan sessions and worktrees must still be detected and
    cleaned (they were dropped when the candidate set was empty)."""
    fake = FakeGitHubClient()  # no labels -> empty candidate set
    queue = _queue(fake)
    now = datetime.now(UTC)
    session = _orphan_session("orphan-7", now=now, slug="owner/repo#999")
    worktree = _orphan_worktree("/wt/orphan-7", now=now, branch="orphan-branch-7")

    terminated: list[str] = []
    cleaned: list[str] = []

    class _Cao:
        def terminate_session(self, handle):
            terminated.append(handle.session_id)

    class _Git:
        def cleanup_worktree(self, path):
            cleaned.append(path)

    cfg = CleanupConfig(session_lease_ttl_seconds=60, worktree_inactivity_ttl_seconds=60)
    policy = cleanup.CleanupPolicy(cleanup_config=cfg, queue_config=queue._cfg, now=now)
    outcome = cleanup.run_cleanup(
        queue, cao=_Cao(), git=_Git(), policy=policy, sessions=[session], worktree_obs=[worktree]
    )
    assert outcome.orphans == 2, f"both orphans must be planned, got {outcome!r}"
    assert terminated == ["orphan-7"]
    assert cleaned == ["/wt/orphan-7"]


# --- fix #8 (#22): foreman revalidates opt-in before continuing -----------


def test_foreman_revalidates_opt_in_before_continue(monkeypatch):
    """Fix #8: if the operator removes the enabled label between an
    ``run_pass`` snapshot and a claim, a previously-claimed item must NOT be
    claimed / continued. ``_drive`` revalidates the opt-in label and parks
    (abandons) the item instead of running branch/PR/CI side effects."""
    from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop

    fake = _ready_fake(1)
    # The pass snapshots this issue as ready, but the operator removes the
    # enabled label before ``_drive`` revalidates it.
    monkeypatch.setattr(fake, "list_issues_by_label", lambda label: [1])
    monkeypatch.setattr(fake, "get_labels", lambda issue_number: [])
    queue = _queue(fake)
    queue.claim(_issue(1), "run-r2-8", branch="aipro-issue-1")

    loop = ForemanPolicyLoop(
        queue,
        _FakeBroker(),
        LaneRegistry.default(),
        _FakeExecutor(),
        _FakeGate(),
        _FakeGit(),
        V3Config(),
        run_id="run-r2-8",
        worktree_root="/wt",
        committer_name="x",
        committer_email="y@z",
    )
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "queued", (
        f"item must be parked (not continued) when opt-in removed, got {outcome.final_phase!r}"
    )
    state = queue.load_state("owner/repo#1")
    assert state is not None
    assert state.phase == "queued"


# --- fix #9 / #10 / #14 (soak helpers) ------------------------------------


def _build_soak_fakes():
    soak = _load_soak()
    cfg = V3Config(
        safety=SafetyPolicyConfig(max_coder_invocations_per_run=3),
        cleanup=CleanupConfig(),
    )
    fakes = soak._PersistentFakes.build(cfg, cleanup_cfg=CleanupConfig())
    return soak, fakes


def test_soak_fails_when_seeded_orphan_missed():
    """Fix #9: when cleanup misses a seeded orphan, the soak must FAIL
    (report it in orphan_sessions / orphan_worktrees), not silently pass
    because the violation list stayed empty."""
    soak, fakes = _build_soak_fakes()
    # A round that seeded an orphan which cleanup did NOT apply.
    round_ = soak.SoakRound(
        round_index=1,
        issue_numbers=[1],
        seeded_session_ids=["orphan-missed-sess"],
        seeded_worktree_ids=["/wt/orphan-missed"],
        cleanup_applied_session_ids=[],
        cleanup_applied_worktree_ids=[],
    )
    result = soak._check_invariants([round_], fakes)
    assert result.orphan_sessions, "missed seeded orphan session must fail the soak"
    assert result.orphan_worktrees, "missed seeded orphan worktree must fail the soak"


def test_soak_duplicates_from_real_fake_resources():
    """Fix #10: duplicate-branch / duplicate-PR invariants are derived from
    the REAL fake git branch list and each PR's actual head branch — a
    branch created twice, or a branch backing two PRs, must be flagged even
    though every round seeds disjoint issue numbers."""
    soak, fakes = _build_soak_fakes()
    # A real duplicate branch created by the fake git.
    fakes.git.branches.append("aipro-issue-9")
    fakes.git.branches.append("aipro-issue-9")
    # A real duplicate PR: two open PRs on the same head branch.
    fakes.fake.create_pr("t", "b", "aipro-issue-10", "main")
    fakes.fake.create_pr("t", "b", "aipro-issue-10", "main")

    round_ = soak.SoakRound(round_index=1, issue_numbers=[9, 10])
    result = soak._check_invariants([round_], fakes)

    branches_flagged = {b for _, b in result.duplicates_branch}
    assert "aipro-issue-9" in branches_flagged, (
        f"duplicate branch detected from real git fake, got {result.duplicates_branch!r}"
    )
    pr_flagged = {b for _, b in result.duplicates_pr}
    assert "aipro-issue-10" in pr_flagged, (
        f"duplicate PR branch detected from real PR heads, got {result.duplicates_pr!r}"
    )
    # The two PRs we seeded (same head branch) are behind the flagged entry.
    open_heads = [p.head_ref for p in fakes.fake.list_open_prs()]
    assert open_heads.count("aipro-issue-10") == 2, (
        f"expected two open PRs on the duplicate head branch, got {open_heads!r}"
    )


def test_soak_checks_terminal_branch_divergence():
    """Fix #14: durable-vs-git branch divergence is checked for TERMINAL
    (`done`) items too — earlier the check was skipped past the terminal
    continue, so a done item with a branch missing from git passed."""
    soak, fakes = _build_soak_fakes()
    fake, queue = fakes.fake, fakes.queue
    fake.seed_issue(7, labels=["v3-work"])
    state = queue.claim(GitHubIssueRef(owner="owner", repo="repo", number=7), "run-r2-14")
    # Mark done, persisting a branch that was never created in git.
    queue.complete(GitHubIssueRef(owner="owner", repo="repo", number=7), state, reason="done")
    fresh = queue.load_state("owner/repo#7")
    assert fresh is not None
    import dataclasses

    extras = dict(fresh.extras)
    extras["branch"] = "ghost-branch-7"
    patched = dataclasses.replace(fresh, extras=extras)
    queue.save_state(patched, expected_updated_at=fresh.updated_at)

    round_ = soak.SoakRound(round_index=1, issue_numbers=[7])
    result = soak._check_invariants([round_], fakes)
    assert any("ghost-branch-7" in d for d in result.state_divergence), (
        f"terminal item branch divergence must be flagged, got {result.state_divergence!r}"
    )


# --- fix #11 (#25): cleanup reclaim uses the planner clock ----------------


def test_cleanup_reclaim_uses_planner_clock():
    """Fix #11: staleness is decided against ``CleanupPolicy.now`` and the
    reclaim must validate against the SAME instant. If the reclaim used the
    wall clock instead, a policy ``now`` in the future would mark a lease
    stale then have the queue reject the recovery as still active."""
    fake = _ready_fake(1)
    queue = _queue(fake)
    wall_now = datetime.now(UTC)
    # Lease expires 100s from wall time: stale relative to policy.now,
    # still active relative to the wall clock.
    state = queue.claim(_issue(1), "run-clock-11", branch="aipro-issue-1")
    future_expiry = wall_now + timedelta(seconds=100)
    fresh = queue.load_state(state.work_item_id)
    assert fresh is not None
    extras = dict(fresh.extras)
    extras["lease_expires_at"] = future_expiry.isoformat()
    patched = fresh.__class__(
        work_item_id=fresh.work_item_id,
        run_id=fresh.run_id,
        phase=fresh.phase,
        round_id=fresh.round_id,
        updated_at=fresh.updated_at,
        findings=list(fresh.findings),
        dispositions=list(fresh.dispositions),
        archived=list(fresh.archived),
        extras=extras,
    )
    queue.save_state(patched, expected_updated_at=fresh.updated_at)

    now = wall_now + timedelta(seconds=10000)
    cfg = CleanupConfig()
    policy = cleanup.CleanupPolicy(cleanup_config=cfg, queue_config=queue._cfg, now=now)
    outcome = cleanup.run_cleanup(queue, policy=policy)
    assert outcome.auto_applied == []
    assert outcome.has_manual_actions
    assert "Lease expired" in outcome.manual_actions[0].reason
    assert queue.load_state("owner/repo#1") == patched


# --- fix #12 (#26): reconcile CLI returns nonzero on cleanup abort --------


def _make_reconcile_queue_failing_load():
    fake = _ready_fake(1)

    class _FailingLoad(_queue_type()):
        def load_state(self, work_item_id):
            raise RuntimeError("simulated state load failure")

    return _FailingLoad(fake, "owner", "repo", host_id="host-r2")


def test_cleanup_read_failure_is_not_hidden_by_cli_wrapper():
    queue = _make_reconcile_queue_failing_load()
    with pytest.raises(cleanup.CleanupStateLoadError):
        cleanup.run_cleanup(queue)


def test_reconcile_command_exit_code_surfaces_cleanup_abort(tmp_path, monkeypatch):
    """End-to-end: ``_run_reconcile --apply`` exits non-zero when cleanup
    aborts, even when the earlier plan produced no manual action."""
    config_path = tmp_path / "v3.yml"
    config_path.write_text(
        "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
        "  owner: test-owner\n  repo: test-repo\n"
        "cao:\n  base_url: http://localhost:9889\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "_build_reconciliation_inputs", lambda *a, **k: [])
    monkeypatch.setattr(cli, "_build_github_client", lambda **_: (FakeGitHubClient(), True))

    import argparse

    args = argparse.Namespace(
        config=str(config_path),
        repo="test-owner/test-repo",
        token=None,
        issue=None,
        dry_run=False,
        apply=True,
        json=False,
    )
    code = cli._run_reconcile(args)
    assert code != 0, "cleanup abort must produce a non-zero exit code"


# --- fix #13 (#27): orphan dedup namespaces separate ----------------------


def test_orphan_dedup_namespaces_separate():
    """Fix #13: session ids and worktree branches must not deduplicate
    against each other, and distinct worktree paths on the same branch must
    both be emitted (the sweeper dispatches by path)."""
    from ai_pr_orchestrator.v3.reconcile import Action, ActionKind, ReconcilePlanner

    planner = ReconcilePlanner(cleanup_config=CleanupConfig(), queue_config=queue_cfg())
    # A session orphan whose id equals the worktree orphan's branch must NOT
    # suppress the worktree action.
    actions = [
        Action(
            kind=ActionKind.CLEAN_ORPHAN_SESSION,
            session_id="dup-name",
            work_item_id="owner/repo#1",
            auto_apply=True,
        ),
        Action(
            kind=ActionKind.CLEAN_ORPHAN_WORKTREE,
            branch="dup-name",
            worktree="/wt/a",
            auto_apply=True,
        ),
        # A second worktree on the SAME branch but a DIFFERENT path.
        Action(
            kind=ActionKind.CLEAN_ORPHAN_WORKTREE,
            branch="dup-name",
            worktree="/wt/b",
            auto_apply=True,
        ),
        # An exact duplicate must still be collapseable to one.
        Action(
            kind=ActionKind.CLEAN_ORPHAN_WORKTREE,
            branch="dup-name",
            worktree="/wt/a",
            auto_apply=True,
        ),
    ]
    out = planner._finalize(actions)
    session_actions = [a for a in out if a.kind is ActionKind.CLEAN_ORPHAN_SESSION]
    worktree_actions = [a for a in out if a.kind is ActionKind.CLEAN_ORPHAN_WORKTREE]
    assert len(session_actions) == 1, "session orphan must survive"
    paths = {a.worktree for a in worktree_actions}
    assert paths == {"/wt/a", "/wt/b"}, (
        f"distinct worktree paths must each be emitted, got {paths!r}"
    )
    assert len(worktree_actions) == 2, (
        f"exact worktree duplicate collapses, different paths both survive; got "
        f"{worktree_actions!r}"
    )


def queue_cfg():
    from ai_pr_orchestrator.v3.queue import GitHubQueueConfig

    return GitHubQueueConfig()
