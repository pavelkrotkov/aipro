"""Regression test for Codex review round 1 finding #6 on reconciliation (issue #44).

--- Finding 6: cross-item orphan detection --------------------------------.
"""

from __future__ import annotations

from datetime import timedelta

from _reconcile_builders import (
    frozen_now,
    make_inputs,
    make_session,
    make_state,
    make_worktree,
)

from ai_pr_orchestrator.v3.domain import GitHubIssueRef
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    CleanupConfig,
    ReconcilePlanner,
)


class TestCrossItemOrphanDetection:
    def test_worktree_kept_live_by_sibling_input(self) -> None:
        """Item B must not spuriously flag item A's live branch as orphan.

        The previous implementation only looked at the *current* item's
        branches when computing ``_live_branches``, so planning B (after A)
        would mark A's live branch as orphan and emit
        ``CLEAN_ORPHAN_WORKTREE`` for it.
        """
        now = frozen_now()
        # Item A has a LIVE claim on the shared branch.
        state_a = make_state(
            work_item_id="owner/repo#1",
            phase="coding",
            branch="aipro-shared",
            worktree="/tmp/wt/aipro-shared",
        )
        # Item B's session is aged, but the worktree is shared with A's
        # LIVE claim — it must NOT be cleaned up.
        aged_session = make_session(
            work_item_id="owner/repo#2",
            run_id="run-2",
            session_id="aged",
            last_activity_at=now - timedelta(seconds=10000),
        )
        # Use a long TTL so age alone would trigger cleanup otherwise.
        cleanup = CleanupConfig(
            session_lease_ttl_seconds=7200,
            worktree_inactivity_ttl_seconds=86400,
        )
        inputs_a = make_inputs(
            state=state_a,
            cleanup=cleanup,
            worktrees=(make_worktree(branch="aipro-shared"),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=1),
        )
        inputs_b = make_inputs(
            state=None,
            cleanup=cleanup,
            sessions=(aged_session,),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=2),
        )
        actions = ReconcilePlanner().plan_many([inputs_a, inputs_b])
        kinds = [a.kind for a in actions]
        # No spurious worktree cleanup for the shared branch.
        assert ActionKind.CLEAN_ORPHAN_WORKTREE not in kinds
        # And no spurious session cleanup either, because the worktree /
        # branch referenced by A's lease is live — but the session itself
        # does belong to B (different work_item_id), so it CAN be cleaned.
        # We accept either: an empty plan, or a CLEAN_ORPHAN_SESSION for
        # the aged session that is NOT for A's branch.
        for action in actions:
            if action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE:
                assert action.branch != "aipro-shared"
