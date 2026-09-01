"""Regression test for Codex review round 1 finding #11 on reconciliation (issue #44).

--- Finding 11: no-op coder commit timestamp ------------------------------.
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

from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind,
    ReconcilePlanner,
)


class TestNoOpCoderCommitTimestamp:
    def test_old_commit_no_coder_session_is_not_recorded(self) -> None:
        """An old HEAD predating the lease claim is NOT a recorded commit,
        even if the worktree's last_commit_at is non-None — there was no
        coder session in this run, so the commit cannot be from this run.
        """
        now = frozen_now()
        # claimed_at is "fresh" (now); last_commit_at is older — i.e.
        # there is no commit on this branch since this claim.
        state = make_state(
            phase="coding",
            claimed_at=now,
            lease_expires_at=now + timedelta(seconds=900),
        )
        inputs = make_inputs(
            state=state,
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=60),
                ),
            ),
            sessions=(),
        )
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # The commit_recorded flag stays False, so session_terminal_pre_commit
        # does not match (it requires commit_recorded=False AND
        # coder_succeeded=True). With no coder session at all, the
        # planner instead emits branch_exists_no_session (RELAUNCH) — but
        # crucially NOT a "coder has committed" reading.
        assert ActionKind.COLLECT_RESULT not in kinds

    def test_no_op_coder_session_is_not_recorded(self) -> None:
        """A coder session that exited cleanly but produced no changes
        (success=False) must NOT mark the worktree as having a recorded
        commit. The previous commit_recorded predicate took only the
        commit timestamp vs lease and gave a false positive here.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            claimed_at=now - timedelta(seconds=200),
            lease_expires_at=now + timedelta(seconds=900),
        )
        no_op_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=False,  # no-op exit
            last_activity_at=now - timedelta(seconds=10),
            session_id="sess-noop",
        )
        inputs = make_inputs(
            state=state,
            sessions=(no_op_coder,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                ),
            ),
        )
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # commit_recorded requires BOTH a newer commit AND a successful
        # coder session; here coder_succeeded=False so commit_recorded is
        # False. session_terminal_pre_commit therefore does NOT match,
        # and the planner stays in branch_exists_no_session.
        assert ActionKind.COLLECT_RESULT not in kinds

    def test_productive_coder_session_records_commit(self) -> None:
        """A coder session that exited successfully AND the worktree has
        a newer commit IS recorded.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            claimed_at=now - timedelta(seconds=200),
            lease_expires_at=now + timedelta(seconds=900),
        )
        productive_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=True,
            last_activity_at=now - timedelta(seconds=10),
            session_id="sess-productive",
        )
        inputs = make_inputs(
            state=state,
            sessions=(productive_coder,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                ),
            ),
        )
        actions = ReconcilePlanner().plan(inputs)
        kinds = [a.kind for a in actions]
        # commit_recorded=True with branch_pushed=False ⇒
        # session_terminal_pre_push → COLLECT_RESULT.
        assert ActionKind.COLLECT_RESULT in kinds
