"""Shared builders for reconciliation regression tests (issue #44).

Each test pins one of the 12 review findings:

1. ``--apply`` actually applies; non-manual actions go through controllers.
2. The CLI uses a real :class:`GitHubClient` when a token is supplied
   (not a hard-coded ``FakeGitHubClient`` with literal ``"owner"``/``"repo"``).
3. ``ESCALATE`` short-circuits the planner (no further actions for the same
   work item).
4. Terminal-phase work items return ``NOOP`` immediately, regardless of
   crash rows.
5. The cross-work-item dedupe key is ``run_id`` alone.
6. Cross-item orphan detection sees *all* live branches across the inputs.
7. The coding-lane predicate uses :class:`LaneRegistry` rather than the
   literal ``"developer"`` substring.
8. The duplicate-session predicate scopes by lane (different-lane
   terminal+live is NOT a duplicate).
9. ``post_findings_no_disposition`` requires ALL findings to be dispositioned.
10. ``ci_recorded`` requires an actual ``GateDecision`` snapshot, not the
    phase ``ci_gating``.
11. ``commit_recorded`` requires a successful coder session (no false
    positive on a clean worktree with stale HEAD).
12. Every action that targets a session carries a non-None ``session_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ai_pr_orchestrator.v3.config import CleanupConfig, GitHubQueueConfig
from ai_pr_orchestrator.v3.domain import (
    FindingDisposition,
    GitHubIssueRef,
    ReviewerFinding,
    WorkflowState,
)
from ai_pr_orchestrator.v3.lanes import (
    DEVELOPER_LANE,
    LaneRegistry,
)
from ai_pr_orchestrator.v3.queue import Claim
from ai_pr_orchestrator.v3.reconcile import (
    PullRequestObservation,
    ReconciliationInputs,
    SessionLifecycle,
    SessionObservation,
    WorkItemObservation,
    WorktreeObservation,
)

# --- Builders (shared) ------------------------------------------------------


def frozen_now() -> datetime:
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)


def make_state(
    *,
    work_item_id: str = "owner/repo#1",
    run_id: str = "run-1",
    phase: str = "coding",
    findings: list[ReviewerFinding] | None = None,
    dispositions: list[FindingDisposition] | None = None,
    lease_expires_at: datetime | None = None,
    claimed_at: datetime | None = None,
    branch: str | None = "aipro-issue-1",
    worktree: str | None = "/tmp/wt/aipro-issue-1",
    pr_number: int | None = None,
    terminal_reason: str | None = None,
) -> WorkflowState:
    now = frozen_now()
    extras: dict[str, Any] = {
        "host_id": "host-A",
        "claimed_at": (claimed_at or now).isoformat(),
        "lease_expires_at": (lease_expires_at or (now + timedelta(seconds=900))).isoformat(),
        "branch": branch,
        "worktree": worktree,
    }
    if pr_number is not None:
        extras["pr_number"] = pr_number
    return WorkflowState(
        work_item_id=work_item_id,
        run_id=run_id,
        phase=cast(Any, phase),
        updated_at=now,
        findings=list(findings or []),
        dispositions=list(dispositions or []),
        extras=extras,
        terminal_reason=terminal_reason
        or ("completed" if phase in ("done", "failed", "escalated") else None),
    )


def make_claim(state: WorkflowState) -> Claim:
    return Claim.from_dict(state.extras)


def make_inputs(
    *,
    state: WorkflowState | None = None,
    sessions: tuple[SessionObservation, ...] = (),
    worktrees: tuple[WorktreeObservation, ...] = (),
    pull_requests: tuple[PullRequestObservation, ...] = (),
    cleanup: CleanupConfig | None = None,
    queue: GitHubQueueConfig | None = None,
    now: datetime | None = None,
    issue: GitHubIssueRef | None = None,
    ci_status: Any = None,
    lanes: LaneRegistry | None = None,
) -> ReconciliationInputs:
    now = now or frozen_now()
    claim = make_claim(state) if state is not None else None
    observation = WorkItemObservation(
        work_item=issue or GitHubIssueRef(owner="owner", repo="repo", number=1),
        state=state,
        claim=claim,
    )
    return ReconciliationInputs(
        observation=observation,
        sessions=sessions,
        worktrees=worktrees,
        pull_requests=pull_requests,
        config=cleanup or CleanupConfig(),
        queue_config=queue or GitHubQueueConfig(),
        now=now,
        ci_status=ci_status,
    )


def make_session(
    *,
    work_item_id: str,
    run_id: str | None,
    lane: str = DEVELOPER_LANE,
    state: SessionLifecycle = "terminal",
    is_terminal: bool = True,
    success: bool = True,
    last_activity_at: datetime | None = None,
    session_id: str = "sess-1",
) -> SessionObservation:
    return SessionObservation(
        session_id=session_id,
        work_item_id=work_item_id,
        run_id=run_id,
        lane=lane,
        state=state,
        last_activity_at=last_activity_at or frozen_now(),
        is_terminal=is_terminal,
        success=success,
    )


def make_worktree(
    *,
    branch: str = "aipro-issue-1",
    last_commit_at: datetime | None = None,
    last_push_at: datetime | None = None,
    path: str | None = None,
    is_default_branch: bool = False,
) -> WorktreeObservation:
    return WorktreeObservation(
        path=path or f"/tmp/wt/{branch}",
        branch=branch,
        last_commit_at=last_commit_at or frozen_now(),
        last_push_at=last_push_at,
        is_default_branch=is_default_branch,
    )


def make_finding(*, finding_id: str = "f1", thread_id: str | None = "PRRT_1") -> ReviewerFinding:
    return ReviewerFinding(
        id=finding_id,
        lane="architecture-reviewer",
        body="b",
        severity="major",
        run_id="run-1",
        round_id="round-1",
        thread_id=thread_id,
        head_sha="abc123",
    )


def make_disposition(finding_id: str = "f1") -> FindingDisposition:
    return FindingDisposition(
        finding_id=finding_id,
        action="fix",
        rationale="fixed",
        decided_by="foreman",
    )
