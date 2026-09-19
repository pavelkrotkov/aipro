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
from .domain import TERMINAL_PHASES, GitHubIssueRef
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

    @property
    def has_manual_actions(self) -> bool:
        return bool(self.manual_actions)


def _observations(
    queue: GitHubIssueQueue,
    sessions: tuple[SessionObservation, ...],
    worktrees: tuple[WorktreeObservation, ...],
) -> list[WorkItemObservation]:
    issue_numbers = tuple(
        int(w.branch.removeprefix("aipro-issue-"))
        for w in worktrees
        if w.branch.removeprefix("aipro-issue-").isdigit()
    )
    slugs = tuple(s.work_item_id for s in sessions if s.work_item_id is not None)
    issues = queue.list_tracked(work_item_ids=slugs, issue_numbers=issue_numbers)
    return [_observe_item(queue, issue) for issue in issues]


def _observe_item(queue: GitHubIssueQueue, issue: GitHubIssueRef) -> WorkItemObservation:
    """Read state and claim atomically from the sweep's perspective.

    A malformed active claim or missing state behind a lifecycle label is
    unknown ownership, not proof that its resources can be deleted.
    """
    try:
        state = queue.load_state(issue.slug())
        if state is None:
            if not queue.is_enabled(issue) and queue.load_work_item(issue).labels:
                raise ValueError("lifecycle item has no authoritative state")
            return WorkItemObservation(issue, None, None)
        claim = None if state.phase in (*TERMINAL_PHASES, "queued") else claim_from_state(state)
        return WorkItemObservation(issue, state, claim)
    except Exception as exc:
        raise CleanupStateLoadError(f"Cannot verify {issue.slug()}: {exc}") from exc


def run_cleanup(
    queue: GitHubIssueQueue,
    *,
    cao: CaoControllerLike | None = None,
    git: GitOpsLike | None = None,
    policy: CleanupPolicy | None = None,
    sessions: Iterable[SessionObservation] = (),
    worktree_obs: Iterable[WorktreeObservation] = (),
) -> SweepOutcome:
    """Surface recovery decisions and apply only confirmed orphan cleanup.

    Missing/failed controllers leave a manual action, never a successful removal.
    """
    policy = policy or CleanupPolicy(CleanupConfig(), GitHubQueueConfig())
    sessions, worktrees = tuple(sessions), tuple(worktree_obs)
    observations = _observations(queue, sessions, worktrees)
    known_ids = {item.work_item_id for item in observations}
    sessions = tuple(s for s in sessions if s.work_item_id in known_ids)
    planner = ReconcilePlanner(
        cleanup_config=policy.cleanup_config, queue_config=policy.queue_config
    )
    outcome = SweepOutcome(
        manual_actions=_manual_recovery_actions(planner, observations, sessions, policy)
    )
    actions = planner.plan_orphans(observations, sessions, worktrees, now=policy.now)
    for action in actions:
        # Re-read all ownership immediately before deletion. A newly claimed item
        # makes its earlier orphan action ineligible; failed reads abort the sweep.
        current = planner.plan_orphans(
            _observations(queue, sessions, worktrees), sessions, worktrees, now=policy.now
        )
        if action in current:
            _apply_action(action, cao=cao, git=git, outcome=outcome)
    return outcome


def _manual_recovery_actions(
    planner: ReconcilePlanner,
    observations: list[WorkItemObservation],
    sessions: tuple[SessionObservation, ...],
    policy: CleanupPolicy,
) -> list[Action]:
    """Recovery remains a planner decision; this sweep executes only orphan removal."""
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
    return [action for action in planner.plan_many(inputs) if not action.auto_apply]


def _apply_action(
    action: Action,
    *,
    cao: CaoControllerLike | None,
    git: GitOpsLike | None,
    outcome: SweepOutcome,
) -> None:
    try:
        _remove_orphan(action, cao=cao, git=git)
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
    outcome.sessions_terminated += action.kind is ActionKind.CLEAN_ORPHAN_SESSION
    outcome.worktrees_cleaned += action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE


def _remove_orphan(
    action: Action, *, cao: CaoControllerLike | None, git: GitOpsLike | None
) -> None:
    """The destructive boundary returns only after its controller confirms success."""
    if action.kind is ActionKind.CLEAN_ORPHAN_SESSION:
        if cao is None or action.session_id is None:
            raise RuntimeError("Session cleanup requires a CAO controller and session id")
        cao.terminate_session(SessionHandle(session_id=action.session_id, lane="-"))
    elif action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE:
        if git is None or action.worktree is None:
            raise RuntimeError("Worktree cleanup requires git operations and a path")
        git.cleanup_worktree(action.worktree)
    else:
        raise ValueError(f"Unsupported cleanup action: {action.kind}")
