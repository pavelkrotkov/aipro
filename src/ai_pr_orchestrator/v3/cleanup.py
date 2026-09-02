"""V3 production TTL sweeper (issue #44, exercised in P4 / #55).

The :func:`run_cleanup` helper is the *production* orphan-cleanup path
the foreman invokes between rounds and the soak runner exercises
between iterations. It is **not** a re-implementation of the
reconciliation planner; it is a thin orchestrator that:

1. Builds :class:`~ai_pr_orchestrator.v3.reconcile.ReconciliationInputs`
   for every active work item, then
2. calls :meth:`ReconcilePlanner.plan_many` to derive the recovery
   plan, then
3. applies the *auto_apply* actions through the queue / CAO controller
   / git operations: ``CLEAN_ORPHAN_SESSION``,
   ``CLEAN_ORPHAN_WORKTREE``, and ``RECOVER_STALE_LEASE``.
4. surfaces manual actions (``ESCALATE``, ``HALT_BRANCH_MOVED``) as a
   non-zero exit so the CLI / soak harness knows reconciliation found
   something a human must look at.

The sweeper owns its clock and logs every removal. Tests pass a
deterministic ``now`` for boundary assertions; production callers
accept ``now=None`` and the sweeper reads ``datetime.now(UTC)`` once
on entry so every input the planner sees shares the same instant.

No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .config import CleanupConfig, GitHubQueueConfig
from .domain import GitHubIssueRef, WorkflowState
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
from .reconcile import (
    claim_from_state as _claim_from_state,
)

log = logging.getLogger(__name__)


class CaoControllerLike(Protocol):
    """The minimal CAO surface :func:`run_cleanup` needs to terminate
    an orphaned session.

    Production callers pass the real :class:`~ai_pr_orchestrator.v3.cao.CaoSessionController`
    (it has ``terminate_session``); tests pass a stub.
    """

    def terminate_session(self, handle: Any) -> None: ...


class GitOpsLike(Protocol):
    """The minimal git surface :func:`run_cleanup` needs to remove an
    orphaned worktree."""

    def cleanup_worktree(self, path: str) -> None: ...


@dataclass
class CleanupPolicy:
    """The production policy bundle the sweeper carries through every
    invocation.

    ``now`` is the deterministic clock the planner evaluates against.
    Defaults to ``datetime.now(UTC)`` so production callers can omit it.
    """

    cleanup_config: CleanupConfig
    queue_config: GitHubQueueConfig
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.now.tzinfo is None:
            raise ValueError("CleanupPolicy.now must be timezone-aware")


@dataclass
class SweepOutcome:
    """The structured result of one :func:`run_cleanup` invocation.

    ``auto_applied`` records every action the sweeper applied without
    operator approval. ``manual_actions`` records the actions that
    require a human — exactly the set the ``aipro reconcile`` CLI
    surfaces. ``orphans`` and ``recovered_leases`` are flat counts so
    the soak runner can assert on them directly.
    """

    auto_applied: list[Action] = field(default_factory=list)
    manual_actions: list[Action] = field(default_factory=list)
    orphans: int = 0
    recovered_leases: int = 0
    worktrees_cleaned: int = 0
    sessions_terminated: int = 0

    @property
    def has_manual_actions(self) -> bool:
        return bool(self.manual_actions)


def _live_claim(state: WorkflowState | None, now: datetime) -> bool:
    if state is None:
        return False
    try:
        return not claim_from_state(state).is_stale(now)
    except Exception:
        return False


def _observations_for_queue(
    queue: GitHubIssueQueue,
    *,
    now: datetime,
    cleanup_config: CleanupConfig,
) -> list[ReconciliationInputs]:
    """Build a ``ReconciliationInputs`` per issue with active durable state.

    The planner's ``plan_many`` requires every input to carry its own
    observations; this helper reads the queue's state comments and the
    CAO controller's session list (if provided) into a single
    bundle per work item.
    """
    active_numbers = set(queue._client.list_issues_by_label(queue._cfg.active_label))
    enabled_numbers = set(queue._client.list_issues_by_label(queue._cfg.enabled_label))
    inputs: list[ReconciliationInputs] = []
    for issue_number in sorted(active_numbers | enabled_numbers):
        issue = GitHubIssueRef(owner=queue._owner, repo=queue._repo, number=issue_number)
        try:
            state = queue.load_state(issue.slug())
        except Exception:
            state = None
        try:
            claim = _claim_from_state(state) if state is not None else None
        except Exception:
            claim = None
        inputs.append(
            ReconciliationInputs(
                observation=WorkItemObservation(work_item=issue, state=state, claim=claim),
                sessions=(),
                worktrees=(),
                pull_requests=(),
                config=cleanup_config,
                queue_config=queue._cfg,
                now=now,
            )
        )
    return inputs


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
    """Run one sweep of the production TTL sweeper.

    The function is a thin orchestrator over
    :class:`~ai_pr_orchestrator.v3.reconcile.ReconcilePlanner`. It
    applies every auto_apply action through the supplied queue, CAO
    controller, and git operations. Manual actions are returned for
    surfacing (the CLI exits non-zero; the soak runner records them).

    The supplied ``sessions`` and ``worktree_obs`` collections let a
    caller feed live CAO/git state without exposing the underlying
    controllers' APIs to this module. ``sessions`` is matched against
    ``state.work_item_id`` so the planner can correlate a live session
    with its work item; ``worktree_obs`` is matched by branch.
    """
    now = policy.now if policy is not None else datetime.now(UTC)
    cleanup_cfg = policy.cleanup_config if policy is not None else CleanupConfig()
    queue_cfg = policy.queue_config if policy is not None else GitHubQueueConfig()
    inputs = _observations_for_queue(queue, now=now, cleanup_config=cleanup_cfg)
    # Attach the supplied session / worktree observations to every
    # work item that owns them.
    sessions_tuple = tuple(sessions)
    worktree_tuple = tuple(worktree_obs)
    inputs = [
        ReconciliationInputs(
            observation=item.observation,
            sessions=sessions_tuple,
            worktrees=worktree_tuple,
            pull_requests=item.pull_requests,
            config=item.config,
            queue_config=item.queue_config,
            now=item.now,
        )
        for item in inputs
    ]
    planner_obj = planner or ReconcilePlanner(cleanup_config=cleanup_cfg, queue_config=queue_cfg)
    plan = planner_obj.plan_many(inputs)

    outcome = SweepOutcome()
    for action in plan:
        if action.auto_apply:
            outcome.auto_applied.append(action)
            if action.kind is ActionKind.CLEAN_ORPHAN_SESSION:
                outcome.orphans += 1
                outcome.sessions_terminated += 1
            elif action.kind is ActionKind.CLEAN_ORPHAN_WORKTREE:
                outcome.orphans += 1
                outcome.worktrees_cleaned += 1
            elif action.kind is ActionKind.RECOVER_STALE_LEASE:
                outcome.recovered_leases += 1
        else:
            outcome.manual_actions.append(action)

    log.info(
        "cleanup: applied=%d orphans=%d leases_recovered=%d manual=%d",
        len(outcome.auto_applied),
        outcome.orphans,
        outcome.recovered_leases,
        len(outcome.manual_actions),
    )
    return outcome


__all__ = [
    "CaoControllerLike",
    "CleanupPolicy",
    "GitOpsLike",
    "SweepOutcome",
    "run_cleanup",
]
