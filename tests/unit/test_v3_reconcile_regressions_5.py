"""Regression test for Codex review round 1 finding #5 on reconciliation (issue #44).

--- Finding 5: dedupe key is run_id alone ---------------------------------.
"""

from __future__ import annotations

from _reconcile_builders import (
    make_inputs,
    make_state,
    make_worktree,
)

from ai_pr_orchestrator.v3.domain import GitHubIssueRef
from ai_pr_orchestrator.v3.reconcile import (
    ReconcilePlanner,
    actions_target_branch,
)


class TestDedupeByRunId:
    def test_same_run_different_branches_one_relaunch(self) -> None:
        """Per finding #5, two work items sharing a ``run_id`` produce at
        most one branch-creating action — even when their branches differ.

        Without that, two branches would each claim to be the authoritative
        developer branch for the same run.
        """
        planner = ReconcilePlanner()
        state_a = make_state(
            work_item_id="owner/repo#1",
            run_id="run-1",
            phase="coding",
            branch="aipro-issue-1",
            worktree="/tmp/wt/aipro-issue-1",
        )
        state_b = make_state(
            work_item_id="owner/repo#2",
            run_id="run-1",
            phase="coding",
            branch="aipro-issue-2",
            worktree="/tmp/wt/aipro-issue-2",
        )
        inputs_a = make_inputs(
            state=state_a,
            worktrees=(make_worktree(branch="aipro-issue-1", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=1),
        )
        inputs_b = make_inputs(
            state=state_b,
            worktrees=(make_worktree(branch="aipro-issue-2", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=2),
        )
        branch_actions = [
            a for a in planner.plan_many([inputs_a, inputs_b]) if actions_target_branch(a)
        ]
        assert len(branch_actions) == 1
        # And the surviving action carries the FIRST work item's identity.
        assert branch_actions[0].work_item_id == "owner/repo#1"

    def test_different_runs_can_both_relaunch(self) -> None:
        """Two work items on different ``run_id`` values legitimately both
        produce branch-creating actions (no dedupe conflict).
        """
        planner = ReconcilePlanner()
        state_a = make_state(
            work_item_id="owner/repo#1",
            run_id="run-1",
            phase="coding",
            branch="aipro-issue-1",
            worktree="/tmp/wt/aipro-issue-1",
        )
        state_b = make_state(
            work_item_id="owner/repo#2",
            run_id="run-2",
            phase="coding",
            branch="aipro-issue-2",
            worktree="/tmp/wt/aipro-issue-2",
        )
        inputs_a = make_inputs(
            state=state_a,
            worktrees=(make_worktree(branch="aipro-issue-1", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=1),
        )
        inputs_b = make_inputs(
            state=state_b,
            worktrees=(make_worktree(branch="aipro-issue-2", last_push_at=None),),
            issue=GitHubIssueRef(owner="owner", repo="repo", number=2),
        )
        branch_actions = [
            a for a in planner.plan_many([inputs_a, inputs_b]) if actions_target_branch(a)
        ]
        assert len(branch_actions) == 2
