"""TTL cleanup fenced by protected ownership and terminal GitHub state."""

from datetime import datetime

from ..logging import emit_cleanup_event
from .cao import CaoSessionController
from .config import CleanupConfig
from .git_ops import GitWorktreeOps
from .queue import GitHubIssueQueue


class CleanupSweeper:
    def __init__(
        self,
        queue: GitHubIssueQueue,
        controller: CaoSessionController,
        git: GitWorktreeOps,
        config: CleanupConfig,
    ):
        self.queue, self.controller, self.git, self.config = queue, controller, git, config

    def sweep(self, *, now: datetime) -> list[dict]:
        sessions = self.controller.session_inventory()
        checkouts = self.git.checkout_inventory()
        refs = {m.context.work_item_id for m in sessions}
        refs.update(r.get("work_item_id") for r in checkouts)
        states = self.queue.cleanup_states(refs)
        return self._sessions(sessions, states, now) + self._checkouts(checkouts, states, now)

    def _sessions(self, sessions, states, now):
        actions = []
        for meta, state in self.queue.cleanup_session_targets(sessions, states):
            self.controller.require_session_expired(
                meta, state.updated_at, now, self.config.session_lease_ttl_seconds
            )
            self.queue.verify_cleanup_state(state)
            if not self.queue.cleanup_session_reference(state, meta.workdir, meta.lane.role):
                raise RuntimeError("orphan session attribution is unverified")
            self.controller.stop_verified_session(meta)
            actions.append(
                emit_cleanup_event(state, "clean_orphan_session", session_id=meta.session_name)
            )
        return actions

    def _checkouts(self, records, states, now):
        actions = []
        targets = [
            (record, state)
            for record, state in self.queue.cleanup_checkout_targets(records, states)
            if self.git.checkout_expired(
                record, state.updated_at, now, self.config.worktree_inactivity_ttl_seconds
            )
        ]
        for record, state in targets:
            self.queue.verify_cleanup_state(state)
            self.controller.require_run_stopped(
                state.run_id,
                state.work_item_id,
                developer_session=state.extras.get("developer_session"),
                reviewer_lanes=(
                    lane
                    for wave in state.extras.get("review_waves", {}).values()
                    for lane in wave["lanes"]
                ),
            )
            self.git.retire_checkout(record, state.run_id, state.work_item_id)
            actions.append(
                emit_cleanup_event(state, "clean_orphan_worktree", worktree=record["path"])
            )
        return actions
