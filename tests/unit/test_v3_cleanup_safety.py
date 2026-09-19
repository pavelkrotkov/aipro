"""Destructive cleanup must prove ownership, inactivity and successful removal."""

import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import httpx
import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.cao import (
    CaoMetadataError,
    CaoSessionController,
    CaoSessionMetadata,
    session_name_for,
)
from ai_pr_orchestrator.v3.cleanup import CleanupPolicy, CleanupStateLoadError, run_cleanup
from ai_pr_orchestrator.v3.config import (
    CAOControlPlaneConfig,
    CleanupConfig,
    GitHubQueueConfig,
    V3Config,
)
from ai_pr_orchestrator.v3.domain import GitHubIssueRef
from ai_pr_orchestrator.v3.git_ops import GitOpsError, GitWorktreeOps
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, SessionHandle
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from ai_pr_orchestrator.v3.reconcile import SessionObservation, WorktreeObservation
from tests.integration.soak import soak

NOW = datetime.now(UTC)
OLD = NOW - timedelta(days=10)
ISSUE = GitHubIssueRef("owner", "repo", 1)


def _seed_terminal_owner(queue, number=999, branch="orphan-branch"):
    from ai_pr_orchestrator.v3.domain import GitHubIssueRef, WorkflowState

    state = WorkflowState(
        work_item_id=f"owner/repo#{number}",
        run_id=f"terminal-{number}",
        phase="failed",
        terminal_reason="finished cleanup fixture",
        extras={"branch": branch},
    )
    queue.save_state(state, expected_updated_at=None)
    queue.repair_labels(GitHubIssueRef("owner", "repo", number), state)


def queue():
    client = FakeGitHubClient()
    client.seed_issue(1, labels=["v3-work"])
    return client, GitHubIssueQueue(client, "owner", "repo", host_id="test")


def resources():
    return (
        SessionObservation("session", ISSUE.slug(), "original", "developer", "active", OLD),
        WorktreeObservation("/wt/issue-1", "aipro-issue-1", OLD),
    )


def test_stale_claim_cannot_be_reclaimed_or_deleted_by_sweep():
    _, q = queue()
    original = q.claim(ISSUE, "original", branch="aipro-issue-1", worktree="/wt/issue-1", now=OLD)
    session, worktree = resources()
    cao, git = Mock(), Mock()
    outcome = run_cleanup(q, cao=cao, git=git, sessions=[session], worktree_obs=[worktree])
    assert q.load_state(ISSUE.slug()) == original
    assert outcome.auto_applied == []
    assert any(a.kind.value == "escalate" for a in outcome.manual_actions)
    cao.terminate_session.assert_not_called()
    git.cleanup_worktree.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_active_state_blocks_every_deletion(missing):
    _, q = queue()
    state = q.claim(ISSUE, "original", branch="aipro-issue-1", now=OLD)
    if missing:
        q.load_state = Mock(return_value=None)
    else:
        q.save_state(
            replace(state, extras={**state.extras, "lease_expires_at": "invalid"}),
            expected_updated_at=state.updated_at,
        )
    session, worktree = resources()
    cao, git = Mock(), Mock()
    with pytest.raises(CleanupStateLoadError):
        run_cleanup(q, cao=cao, git=git, sessions=[session], worktree_obs=[worktree])
    cao.terminate_session.assert_not_called()
    git.cleanup_worktree.assert_not_called()


@pytest.mark.parametrize(
    "controller", [None, Mock(terminate_session=Mock(side_effect=RuntimeError("unavailable")))]
)
def test_unconfirmed_removal_is_not_success(controller):
    _, q = queue()
    _seed_terminal_owner(q, 1, "aipro-issue-1")
    session, _ = resources()
    result = run_cleanup(q, cao=controller, sessions=[session])
    assert result.auto_applied == []
    assert result.sessions_terminated == result.orphans == 0
    assert result.has_manual_actions


def test_two_paths_on_same_branch_both_cleaned():
    _, q = queue()
    _seed_terminal_owner(q, 1, "aipro-issue-1")
    _, worktree = resources()
    git = Mock()
    result = run_cleanup(
        q, git=git, worktree_obs=[worktree, replace(worktree, path="/wt/other"), worktree]
    )
    assert result.worktrees_cleaned == 2
    assert [c.args[0] for c in git.cleanup_worktree.call_args_list] == ["/wt/issue-1", "/wt/other"]


def test_foreign_repository_session_is_never_deleted():
    _, q = queue()
    session, _ = resources()
    cao = Mock()
    run_cleanup(q, cao=cao, sessions=[replace(session, work_item_id="other/repo#1")])
    cao.terminate_session.assert_not_called()


def test_actual_git_preserves_unowned_dirty_and_fresh_worktrees(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
        )

    git("init", "-b", "main")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "base",
    )
    git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    owned_root = tmp_path / "owned"
    owned_root.mkdir()
    human, dirty, clean = tmp_path / "human", owned_root / "issue-1", owned_root / "issue-2"
    for path, branch in (
        (human, "human-feature"),
        (dirty, "aipro-issue-1"),
        (clean, "aipro-issue-2"),
    ):
        git("worktree", "add", "-b", branch, str(path))
    (human / "unsaved").write_text("human work")
    (dirty / "unsaved").write_text("agent work")
    ops = GitWorktreeOps(repo)
    observed = ops.list_worktree_observations(str(owned_root))
    assert [o.path for o in observed] == [str(clean)]
    _, q = queue()
    _seed_terminal_owner(q, 2, "aipro-issue-2")
    run_cleanup(q, git=ops, worktree_obs=observed)
    assert human.exists() and dirty.exists() and clean.exists()
    with pytest.raises(GitOpsError):
        ops.cleanup_worktree(str(dirty))
    assert (dirty / "unsaved").read_text() == "agent work"
    # A clean, positively owned tree past the configured TTL is removable.
    later = CleanupPolicy(CleanupConfig(), GitHubQueueConfig(), NOW + timedelta(days=20))
    result = run_cleanup(q, git=ops, worktree_obs=observed, policy=later)
    assert result.worktrees_cleaned == 1
    assert not clean.exists()
    assert human.exists() and dirty.exists()


@pytest.mark.parametrize("activity", [NOW.isoformat(), None])
def test_cold_cao_inventory_uses_remote_metadata_and_activity(activity):
    lane = LaneRegistry.default().get("developer")
    name = session_name_for("original", lane.lane)
    metadata = CaoSessionMetadata(
        name,
        "terminal",
        lane,
        "/wt/issue-1",
        LaneExecutionContext("original", work_item_id=ISSUE.slug()),
        launched_at=OLD,
    )
    deleted = []

    def handle(request):
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(204)
        if request.url.path == "/sessions":
            return httpx.Response(200, json=[{"name": name}, {"name": "cao-human"}])
        if request.url.path.startswith("/sessions/"):
            return httpx.Response(200, json={"terminals": [{"id": "terminal"}]})
        return httpx.Response(
            200,
            json={
                "metadata": metadata.to_dict(),
                "status": "processing",
                "last_active": activity,
            },
        )

    client = httpx.Client(base_url="http://cao", transport=httpx.MockTransport(handle))
    with CaoSessionController(CAOControlPlaneConfig(), client=client) as controller:
        if activity is None:
            with pytest.raises(CaoMetadataError, match="last_active"):
                controller.list_session_observations()
            assert deleted == []
            return
        observations = controller.list_session_observations()
        assert len(observations) == 1
        assert observations[0].last_activity_at == NOW
        assert observations[0].session_id == name
        assert observations[0].work_item_id == ISSUE.slug()
        controller.terminate_session(SessionHandle(name, lane.lane))
    assert deleted == [f"/sessions/{name}"]


def test_soak_rejects_no_work_and_unperformed_cleanup():
    with patch.object(soak.ForemanPolicyLoop, "run_pass", return_value=[]):
        assert soak.main(["--runs", "1"]) == 1
    with patch.object(soak._PersistentFakes, "terminate_session", return_value=None):
        assert soak.main(["--runs", "1"]) == 1
    with patch.object(soak._StaticGit, "cleanup_worktree", return_value=None):
        assert soak.main(["--runs", "1"]) == 1


def test_opt_out_after_coder_stops_push_and_preserves_checkpoint():
    fakes = soak._PersistentFakes.build(V3Config(), cleanup_cfg=CleanupConfig())
    fakes.fake.seed_issue(1, labels=["v3-work"])
    execute = fakes.executor.execute

    def opt_out(*args, **kwargs):
        result = execute(*args, **kwargs)
        fakes.fake.remove_label(1, "v3-work-active")
        return result

    with patch.object(fakes.executor, "execute", side_effect=opt_out):
        result = fakes.loop.run_pass()[0]
    assert result.final_phase == "queued"
    assert fakes.git.pushed == []
    assert fakes.fake.list_open_prs() == []
    state = fakes.queue.load_state(ISSUE.slug())
    assert state is not None
    assert state.extras["worktree"] == "/wt/issue-1"
    assert state.extras["branch"] == "aipro-issue-1"
    assert "lease_expires_at" not in state.extras
    assert fakes.queue.list_ready() == []


def test_parked_unlabeled_checkpoint_retains_resources():
    _, q = queue()
    state = q.claim(ISSUE, "original", branch="aipro-issue-1", worktree="/wt/issue-1", now=OLD)
    parked = q.abandon(ISSUE, state)
    assert q.list_tracked() == []
    session, worktree = resources()
    cao, git = Mock(), Mock()
    result = run_cleanup(q, cao=cao, git=git, sessions=[session], worktree_obs=[worktree])
    assert result.auto_applied == []
    assert q.load_state(ISSUE.slug()) == parked
    cao.terminate_session.assert_not_called()
    git.cleanup_worktree.assert_not_called()


@pytest.mark.parametrize("labels", [[], ["v3-work"]])
def test_missing_ready_or_unlabeled_owner_never_authorizes_cleanup(labels):
    client, q = queue()
    client.seed_issue(1, labels=labels)
    session, worktree = resources()
    cao, git = Mock(), Mock()
    with pytest.raises(CleanupStateLoadError, match="no authoritative state"):
        run_cleanup(q, cao=cao, git=git, sessions=[session], worktree_obs=[worktree])
    cao.terminate_session.assert_not_called()
    git.cleanup_worktree.assert_not_called()


def test_unclaimed_ready_issue_without_resources_does_not_block_sweep():
    _, q = queue()
    result = run_cleanup(q)
    assert result.auto_applied == []
    assert result.manual_actions == []


def test_managed_looking_worktree_without_durable_branch_owner_is_preserved():
    _, q = queue()
    _seed_terminal_owner(q, 1, "aipro-issue-1")
    _, worktree = resources()
    git = Mock()
    with pytest.raises(CleanupStateLoadError, match="branch owner"):
        run_cleanup(q, git=git, worktree_obs=[replace(worktree, branch="aipro-issue-unrecorded")])
    git.cleanup_worktree.assert_not_called()
