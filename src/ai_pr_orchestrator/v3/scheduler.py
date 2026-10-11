"""Single-developer scheduling and recovery from authoritative GitHub state."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from .domain import TERMINAL_PHASES, GitHubIssueRef
from .foreman import ForemanPolicyLoop, WorkItemOutcome
from .queue import GitHubIssueQueue


class ContinuousForeman:
    def __init__(
        self, queue: GitHubIssueQueue, loop_factory: Callable[[str], ForemanPolicyLoop]
    ) -> None:
        self.queue = queue
        self.loop_factory = loop_factory

    def step(self) -> list[WorkItemOutcome]:
        active = self.queue.list_active()
        if len(active) > 1:
            raise RuntimeError("Multiple active developer issues require reconciliation")
        if active:
            return self._resume(active[0])
        return self._process_ready()

    def _process_ready(self) -> list[WorkItemOutcome]:
        outcomes = []
        for issue in self.queue.list_ready():
            state = self.queue.load_state(issue.slug())
            run_id = state.run_id if state is not None else uuid.uuid4().hex
            outcome = self.loop_factory(run_id).run_work(issue)
            outcomes.append(outcome)
            if outcome.final_phase != "ci_gating":
                break
        return outcomes

    def _resume(self, issue: GitHubIssueRef) -> list[WorkItemOutcome]:
        state = self.queue.load_state(issue.slug())
        if state is None:
            raise RuntimeError("Active issue has no authoritative run state")
        if state.phase in TERMINAL_PHASES:
            self.queue.repair_labels(issue, state)
            return []
        if self.queue.is_claim_stale(state):
            state = self.queue.reclaim_expired(
                issue,
                state,
                state.run_id,
                branch=state.extras.get("branch"),
                worktree=state.extras.get("worktree"),
                pr_number=state.extras.get("pr_number"),
            )
        if state.extras.get("host_id") != self.queue.host_id:
            return []
        return [self.loop_factory(state.run_id).resume_work(issue, state)]
