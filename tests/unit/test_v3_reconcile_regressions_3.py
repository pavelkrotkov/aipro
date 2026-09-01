"""Regression test for Codex review round 1 finding #3 on reconciliation (issue #44).

--- Finding 3: ESCALATE short-circuits the planner ------------------------.
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

from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    PullRequestObservation,
    ReconcilePlanner,
)


class TestEscalateShortCircuit:
    def test_escalate_short_circuits_no_subsequent_actions(self) -> None:
        """Once the planner emits ESCALATE for a work item, no further
        actions for that same work item follow.

        Construct a state that would otherwise also trigger
        ``CLEAN_ORPHAN_SESSION`` and ``CLEAN_ORPHAN_WORKTREE``; the only
        action the planner must return is the single ESCALATE.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            lease_expires_at=now - timedelta(seconds=120),
            claimed_at=now - timedelta(seconds=2000),
        )
        aged_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="aged",
        )
        aged_worktree = make_worktree(
            branch="orphan-branch",
            last_commit_at=now - timedelta(seconds=200000),
        )
        inputs = make_inputs(
            state=state,
            sessions=(aged_session,),
            worktrees=(aged_worktree,),
        )
        actions = ReconcilePlanner().plan(inputs)
        # Only the ESCALATE — no orphan cleanup rows.
        assert [a.kind for a in actions] == [ActionKind.ESCALATE]

    def test_halt_branch_moved_short_circuits(self) -> None:
        """HALT_BRANCH_MOVED also short-circuits; a halt means *do not act*,
        so further crash-recovery or orphan-cleanup rows for the same work
        item would contradict the halt itself.
        """
        now = frozen_now()
        state = make_state(phase="reviewing", pr_number=42)
        pr = PullRequestObservation(
            number=42, branch="aipro-issue-1", head_sha="newsha", expected_head_sha="oldsha"
        )
        aged_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="aged",
        )
        actions = ReconcilePlanner().plan(
            make_inputs(
                state=state,
                pull_requests=(pr,),
                sessions=(aged_session,),
            )
        )
        assert [a.kind for a in actions] == [ActionKind.HALT_BRANCH_MOVED]
