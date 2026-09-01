"""Regression test for Codex review round 1 finding #12 on reconciliation (issue #44).

--- Finding 12: every session-touching action carries session_id ----------.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

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
    CleanupConfig,
    ReconcilePlanner,
)


class TestActionSessionIdPopulated:
    @dataclass(frozen=True)
    class _Spec:
        name: str
        state_factory: Any
        sessions_factory: Any = None

    def _spec(self, name: str) -> _Spec:
        return self._Spec(name=name, state_factory=None, sessions_factory=None)

    def test_resume_session_actions_have_session_id(self) -> None:
        """Every RESUME_SESSION the planner emits carries a non-None
        ``session_id`` so the CLI can dispatch against a concrete session.

        Driving the planner through every row that emits RESUME_SESSION
        would be brittle; instead, exercise the high-traffic rows via
        a real observation bundle and assert non-None.
        """
        now = frozen_now()
        # No PR, branch pushed, no live session: post_push_no_pr -> RESUME.
        state = make_state(
            phase="coding",
            pr_number=None,
            claimed_at=now - timedelta(seconds=300),
            lease_expires_at=now + timedelta(seconds=900),
        )
        terminal_session = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            session_id="sess-coding",
        )
        inputs = make_inputs(
            state=state,
            sessions=(terminal_session,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                    last_push_at=now - timedelta(seconds=5),
                ),
            ),
            pull_requests=(),
        )
        actions = ReconcilePlanner().plan(inputs)
        resume = [a for a in actions if a.kind is ActionKind.RESUME_SESSION]
        assert resume, f"expected RESUME_SESSION, got {[a.kind for a in actions]}"
        for action in resume:
            assert action.session_id is not None, (
                f"RESUME_SESSION must carry a session_id, got {action}"
            )

    def test_collect_result_actions_have_session_id(self) -> None:
        """COLLECT_RESULT must carry a non-None ``session_id``.
        Use the post-CI scenario; the planner emits a RELAUNCH instead of
        COLLECT_RESULT in that path, so use a coding pre-commit scenario.
        """
        now = frozen_now()
        state = make_state(
            phase="coding",
            claimed_at=now - timedelta(seconds=200),
            lease_expires_at=now + timedelta(seconds=900),
        )
        terminal_coder = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=True,
            session_id="sess-coding",
        )
        inputs = make_inputs(
            state=state,
            sessions=(terminal_coder,),
            worktrees=(
                make_worktree(
                    branch="aipro-issue-1",
                    last_commit_at=now - timedelta(seconds=10),
                    last_push_at=None,
                ),
            ),
        )
        actions = ReconcilePlanner().plan(inputs)
        collect = [a for a in actions if a.kind is ActionKind.COLLECT_RESULT]
        assert collect
        for action in collect:
            assert action.session_id is not None, (
                f"COLLECT_RESULT must carry a session_id, got {action}"
            )

    def test_relaunch_with_sessions_has_session_id(self) -> None:
        """RELAUNCH carries a session_id when one is available so the CLI
        can dispatch the relaunch to the right lane/session."""
        now = frozen_now()
        state = make_state(phase="coding")
        # success=False so the coder session did not commit; the planner
        # sees branch_exists_no_session (RELAUNCH) rather than
        # session_terminal_pre_commit (COLLECT_RESULT).
        aged_terminal = make_session(
            work_item_id="owner/repo#1",
            run_id="run-1",
            lane=DEVELOPER_LANE,
            is_terminal=True,
            success=False,
            last_activity_at=now - timedelta(seconds=200),
            session_id="sess-prev",
        )
        inputs = make_inputs(
            state=state,
            sessions=(aged_terminal,),
            worktrees=(make_worktree(branch="aipro-issue-1", last_push_at=None),),
        )
        actions = ReconcilePlanner().plan(inputs)
        relaunch = [a for a in actions if a.kind is ActionKind.RELAUNCH]
        assert relaunch
        for action in relaunch:
            assert action.session_id is not None, (
                f"RELAUNCH must carry a session_id when one is known, got {action}"
            )

    def test_clean_orphan_session_has_session_id(self) -> None:
        """CLEAN_ORPHAN_SESSION must carry the target ``session_id``."""
        now = frozen_now()
        aged = make_session(
            work_item_id="orphan-wi",
            run_id="orphan-run",
            last_activity_at=now - timedelta(seconds=10000),
            session_id="orphan-sess",
        )
        inputs = make_inputs(
            state=None,
            sessions=(aged,),
            cleanup=CleanupConfig(session_lease_ttl_seconds=7200),
        )
        actions = ReconcilePlanner().plan(inputs)
        cleanup_actions = [a for a in actions if a.kind is ActionKind.CLEAN_ORPHAN_SESSION]
        assert cleanup_actions
        for action in cleanup_actions:
            assert action.session_id == "orphan-sess"
