"""Apply the existing reconciliation planner's orphan actions after authoritative reads.

Lease recovery remains the reconciliation owner's decision. This sweep never
reclaims work or interprets a failed read as evidence that a resource is orphaned.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from .config import CleanupConfig, GitHubQueueConfig
from .domain import TERMINAL_PHASES
from .interfaces import SessionHandle
from .queue import GitHubIssueQueue, claim_from_state
from .reconcile import (
    Action,
    ActionKind,
    ReconcilePlanner,
    ReconciliationInputs,
    SessionObservation,
    WorkItemObservation,
    WorktreeObservation,
)


class CaoControllerLike(Protocol):
    def terminate_session(self, handle: SessionHandle) -> None: ...


class GitOpsLike(Protocol):
    def cleanup_worktree(self, path: str) -> None: ...


@dataclass
class CleanupPolicy:
    cleanup_config: CleanupConfig
    queue_config: GitHubQueueConfig
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.now.tzinfo is None:
            raise ValueError("CleanupPolicy.now must be timezone-aware")


class CleanupStateLoadError(RuntimeError):
    """Authoritative state is unavailable; no destructive action is permitted."""


@dataclass
class SweepOutcome:
    auto_applied: list[Action] = field(default_factory=list)
    manual_actions: list[Action] = field(default_factory=list)
    orphans: int = 0
    worktrees_cleaned: int = 0
    sessions_terminated: int = 0
    state_load_failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_manual_actions(self) -> bool:
        return bool(self.manual_actions)

    @property
    def has_state_load_failures(self) -> bool:
        return bool(self.state_load_failures)


def _observations(queue: GitHubIssueQueue) -> list[WorkItemObservation]:
    observations = []
    for issue in queue.list_tracked():
        try:
            state = queue.load_state(issue.slug())
            claim = None
            if state is not None and state.phase not in (*TERMINAL_PHASES, "queued"):
                claim = claim_from_state(state)
            observations.append(WorkItemObservation(issue, state, claim))
        except Exception as exc:
            raise CleanupStateLoadError(f"Cannot verify {issue.slug()}: {exc}") from exc
    return observations


def run_cleanup(
    queue: GitHubIssueQueue,
    *,
    cao: CaoControllerLike | None = None,
    git: GitOpsLike | None = None,
    planner: ReconcilePlanner | None = None,
    policy: CleanupPolicy | None = None,
    sessions: Iterable[SessionObservation] = (),
    worktree_obs: Iterable[WorktreeObservation] = (),
) -> SweepOutcome:
    """Surface recovery decisions and apply only confirmed orphan cleanup.

    Missing/failed controllers leave a manual action, never a successful removal.
    """
    policy = policy or CleanupPolicy(CleanupConfig(), GitHubQueueConfig())
    observations = _observations(queue)
    sessions, worktrees = tuple(sessions), tuple(worktree_obs)
    planner = planner or ReconcilePlanner(
        cleanup_config=policy.cleanup_config, queue_config=policy.queue_config
    )
    inputs = [
        ReconciliationInputs(
            observation=item,
            sessions=sessions,
            worktrees=(),
            pull_requests=(),
            config=policy.cleanup_config,
            queue_config=policy.queue_config,
            now=policy.now,
        )
        for item in observations
    ]
    outcome = SweepOutcome()
    outcome.manual_actions = [a for a in planner.plan_many(inputs) if not a.auto_apply]
    actions = planner.plan_orphans(observations, sessions, worktrees, now=policy.now)
    for action in actions:
        # Re-read all ownership immediately before deletion. A newly claimed item
        # makes its earlier orphan action ineligible; failed reads abort the sweep.
        current = planner.plan_orphans(_observations(queue), sessions, worktrees, now=policy.now)
        if action in current:
            _apply_action(action, cao=cao, git=git, outcome=outcome)
    return outcome


def _apply_action(
    action: Action,
    *,
    cao: CaoControllerLike | None,
    git: GitOpsLike | None,
    outcome: SweepOutcome,
) -> None:
    try:
        if action.kind is ActionKind.CLEAN_ORPHAN_SESSION:
            if cao is None or action.session_id is None:
                raise RuntimeError("Session cleanup requires a CAO controller and session id")
            cao.terminate_session(SessionHandle(session_id=action.session_id, lane="-"))
            outcome.sessions_terminated += 1
        elif action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE:
            if git is None or action.worktree is None:
                raise RuntimeError("Worktree cleanup requires git operations and a path")
            git.cleanup_worktree(action.worktree)
            outcome.worktrees_cleaned += 1
        else:
            raise ValueError(f"Unsupported cleanup action: {action.kind}")
    except Exception as exc:
        outcome.manual_actions.append(
            Action(
                kind=ActionKind.ESCALATE,
                work_item_id=action.work_item_id,
                session_id=action.session_id,
                worktree=action.worktree,
                reason=f"Cleanup unconfirmed: {exc}",
            )
        )
        return
    outcome.auto_applied.append(action)
    outcome.orphans += 1
