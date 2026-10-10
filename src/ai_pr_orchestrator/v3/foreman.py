"""V3 foreman policy loop (issue #55).

The policy engine that decides "what next" for one work item, replacing the
V1 monolithic runner loop. Given a queue (authoritative GitHub state), a lane
registry, a lane executor, a model broker, a CI/PR gate, and git operations,
it drives one issue through the full lifecycle:

    claim → coding → review rounds → CI gating → PR → done

with explicit escalation paths:

- **Budgets.** Coder invocations and review/CI iterations are counted
  against :class:`~ai_pr_orchestrator.v3.config.SafetyPolicyConfig`; a budget
  exhausted mid-flight escalates to ``needs-human`` rather than looping.
- **Stagnation.** ``EscalationPolicyConfig.stagnation_rounds_threshold``
  consecutive review rounds that add no new findings escalate.
- **Coder failures.** ``max_consecutive_coder_failures`` failed lane
  executions in a row escalate; a single failure fails the item.
- **Policy violations.** ``disallow_workflow_file_changes`` rejects any lane
  output that touches files under ``.github/workflows/``.

Review rounds respect ``ReviewPolicyConfig.max_review_rounds`` and
``require_coder_reply_before_resolve``: a disposition settles a finding with
a coder reply body attached, so a thread is never resolved silently.
Conflicting findings (``conflict_group_id`` set by the finding registry) are
adjudicated — every member of a conflict group is dispositioned explicitly,
never dropped.

The loop is deterministic and does no I/O of its own beyond the interfaces it
is constructed with, so the whole lifecycle is testable with in-memory
fakes. No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol, cast

import httpx

from ai_pr_orchestrator.github.protocol import GitHubClient

from .broker import BrokerError, TaskDemand
from .cao import SessionBusyError
from .catalog import MAX_TASK_DIFFICULTY
from .config import V3Config
from .domain import (
    TERMINAL_PHASES,
    ArchivedFinding,
    FindingDisposition,
    GitHubIssueRef,
    GitHubPullRequestRef,
    LaneIdentity,
    ModelAssignment,
    ReviewerFinding,
    WorkflowState,
)
from .findings import SEVERITY_RANK, FindingRegistry, normalize_claim
from .interfaces import (
    CIPRGate,
    GateDecision,
    GitHubWorkflowStateStore,
    GitOperations,
    LaneExecutionContext,
    LaneExecutor,
    LaneResult,
    ModelBroker,
    ModelLease,
)
from .lanes import (
    ARCHITECTURE_REVIEWER_LANE,
    BREAKER_REVIEWER_LANE,
    DEVELOPER_LANE,
    REQUIREMENTS_REVIEWER_LANE,
    LaneRegistry,
)


class ForemanQueue(GitHubWorkflowStateStore, Protocol):
    """The state-store reads/writes plus the queue verbs the foreman uses.

    Structural on purpose: the production :class:`GitHubIssueQueue` satisfies
    it, and tests may supply partial in-memory fakes.
    """

    def list_ready(self) -> list[GitHubIssueRef]: ...

    def claim(
        self,
        issue: GitHubIssueRef,
        run_id: str,
        *,
        branch: str | None = None,
        worktree: str | None = None,
        pr_number: int | None = None,
        now: datetime | None = None,
    ) -> WorkflowState: ...

    def transition(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        phase: str,
        *,
        terminal_reason: str | None = None,
        round_id: str | None = None,
    ) -> WorkflowState: ...

    def mark_needs_human(
        self, issue: GitHubIssueRef, state: WorkflowState, *, reason: str
    ) -> WorkflowState: ...

    def fail(
        self, issue: GitHubIssueRef, state: WorkflowState, *, reason: str
    ) -> WorkflowState: ...

    def heartbeat(self, state: WorkflowState, *, now: datetime | None = None) -> WorkflowState: ...


@dataclass
class WorkItemOutcome:
    """What the foreman did to one work item in one pass."""

    issue: GitHubIssueRef
    final_phase: str
    reason: str = ""
    review_rounds: int = 0
    coder_invocations: int = 0
    gate: GateDecision | None = None
    escalated: bool = False


class ForemanQueueError(RuntimeError):
    """Raised when the foreman is handed a queue missing the claim verbs."""


class PRReconciliationError(RuntimeError):
    """PR outcome is uncertain; retain the run for GitHub reconciliation."""


class _ForemanEscalation(RuntimeError):
    """Internal control-flow signal: this work item must escalate now.

    Raised inside a review round (stagnation, reviewer-lane crash, or a
    persistence failure) so the escalation propagates out of the deep
    recursion and is persisted once by ``_drive``'s handler — it must never
    be swallowed into an ordinary "no findings" round report, which would
    let the lifecycle resume (or even complete) from a state that was
    explicitly escalated.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _RoundReport:
    """Result of one review round."""

    remaining: tuple[ReviewerFinding, ...]
    rounds: int
    stagnant_rounds: int
    triggers: int
    head_sha: str


@dataclass(frozen=True)
class _ReviewSeat:
    lane: LaneIdentity
    worktree: str
    lease: ModelLease
    fallbacks: tuple[str, ...]
    requests: tuple[tuple[str, str | None], ...]


def _parse_timestamp(value: object) -> datetime:
    """Parse a persisted timestamp (ISO string or datetime) to an aware datetime."""
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


class ForemanPolicyLoop:
    """Drives claimed work items through the V3 lifecycle.

    The queue is accepted structurally: it must provide the
    :class:`GitHubWorkflowStateStore` reads/writes plus ``list_ready``,
    ``claim``, ``transition``, and (for escalation) ``mark_needs_human`` and
    ``fail``. Anything missing fails on first use with an ``AttributeError``
    naming the method — a more useful diagnostic than a nominal-type check at
    construction, and it lets tests substitute partial fakes.
    """

    def __init__(
        self,
        queue: ForemanQueue,
        broker: ModelBroker,
        lanes: LaneRegistry,
        lane_executor: LaneExecutor,
        ci_gate: CIPRGate,
        git_ops: GitOperations,
        config: V3Config,
        *,
        run_id: str,
        worktree_root: str,
        committer_name: str,
        committer_email: str,
    ) -> None:
        self._queue = queue
        self._broker = broker
        self._lanes = lanes
        self._executor = lane_executor
        self._gate = ci_gate
        self._git = git_ops
        self._cfg = config
        self._run_id = run_id
        self._worktree_root = worktree_root
        self._name = committer_name
        self._email = committer_email

    @property
    def run_id(self) -> str:
        """The run identity this foreman instance owns.

        Public so E2E harnesses can script a CAO fake's per-session
        state against the same run id the controller will derive the
        deterministic session name from.
        """
        return self._run_id

    # --- Public API ----------------------------------------------------------

    def run_pass(
        self, *, now: datetime | None = None, max_items: int | None = None
    ) -> list[WorkItemOutcome]:
        """Claim and drive every ready issue; one outcome per item.

        An issue whose claim fails (already claimed by a competing foreman)
        is skipped rather than fatal: contention is normal queue behaviour.
        One item crashing escalates *that item* only — and persists the crash
        (``mark_needs_human``) so the authoritative issue is never left on an
        active phase with a stranded claim. The pass continues only after
        that authoritative write succeeds; persistence failures propagate.
        Busy CAO sessions propagate for reconciliation without terminalizing
        their active run or deleting its checkout.
        """
        issues = self._list_ready()
        if max_items is not None:
            issues = issues[:max_items]
        outcomes: list[WorkItemOutcome] = []
        for issue in issues:
            existing = self._queue.load_state(issue.slug())
            resume_at_gate = existing is not None and (
                existing.phase in ("queued", "ci_gating")
                and existing.extras.get("pr_number") is not None
            )
            try:
                outcomes.append(
                    self._drive(issue, existing, now=now, resume_at_gate=resume_at_gate)
                )
            except (SessionBusyError, PRReconciliationError):
                # Uncertain effects require reconciliation, not terminal cleanup.
                raise
            except Exception as exc:
                reason = f"foreman error: {exc}"
                self._persist_crash(issue, reason)
                cancelled = self._cancelled_outcome(issue)
                outcome = cancelled or WorkItemOutcome(
                    issue=issue, final_phase="escalated", reason=reason, escalated=True
                )
                outcomes.append(outcome)
            # CI-only claims own no local checkout; a retained path can belong
            # to unrelated work on this host. Keep its attribution for recovery.
            if not resume_at_gate:
                self._cleanup_terminal_worktree(issue)
        return outcomes

    def _cancelled_outcome(self, issue: GitHubIssueRef) -> WorkItemOutcome | None:
        state = self._queue.load_state(issue.slug())
        if state is not None and state.run_id == self._run_id and state.phase == "cancelled":
            return WorkItemOutcome(
                issue=issue, final_phase="cancelled", reason=state.terminal_reason or ""
            )
        return None

    def _persist_crash(self, issue: GitHubIssueRef, reason: str) -> None:
        """Persist an in-pass crash against the authoritative issue.

        Only this run's non-terminal claim may be escalated. Authoritative
        read/write failures propagate so the caller cannot report a terminal
        outcome while the durable claim remains active.
        """
        state = self._queue.load_state(issue.slug())
        if state is None or state.phase in TERMINAL_PHASES:
            return
        if state.run_id != self._run_id:
            return
        self._queue.mark_needs_human(issue, state, reason=reason)

    def _cleanup_terminal_worktree(self, issue: GitHubIssueRef) -> None:
        """Release this run's checkout only after confirming durable termination."""
        try:
            state = self._queue.load_state(issue.slug())
        except Exception:
            logging.getLogger(__name__).warning(
                "Retaining worktree for %s: cleanup state verification failed",
                issue.slug(),
                exc_info=True,
            )
            return
        if (
            state is None
            or state.run_id != self._run_id
            or state.phase not in TERMINAL_PHASES
            or state.phase == "cancelled"
        ):
            return
        worktree = state.extras.get("worktree")
        if worktree:
            with suppress(Exception):
                self._git.cleanup_worktree(worktree)

    # --- Lifecycle -----------------------------------------------------------

    def _drive(
        self,
        issue: GitHubIssueRef,
        existing: WorkflowState | None,
        *,
        now: datetime | None,
        resume_at_gate: bool,
    ) -> WorkItemOutcome:
        branch = f"aipro-issue-{issue.number}"

        # Claim FIRST — resources are created only once the claim is won, so a
        # lost claim (contention) never leaks a branch/worktree behind it.
        extras = existing.extras if existing is not None else {}
        branch = extras.get("branch") or branch
        pr_number = extras.get("pr_number")
        state = self._claim(
            issue,
            branch=branch,
            worktree=extras.get("worktree"),
            pr_number=pr_number,
            now=now,
        )

        worktree: str | None = state.extras.get("worktree")
        try:
            if not worktree and not resume_at_gate:
                base = self._git.default_branch()
                self._git.create_branch(branch, base)
                worktree = self._git.create_worktree(
                    f"{self._worktree_root}/issue-{issue.number}", branch
                )
            if worktree and not resume_at_gate:
                state = self._verify_checkout(issue, state, branch, worktree)
            return self._run_loop(
                issue, state, worktree, branch, now=now, resume_at_gate=resume_at_gate
            )
        except _ForemanEscalation as exc:
            return self._cancelled_outcome(issue) or self._escalate(
                issue, state, exc.reason, now=now
            )

    def _run_loop(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str | None,
        branch: str,
        *,
        now: datetime | None,
        resume_at_gate: bool = False,
    ) -> WorkItemOutcome:
        safety = self._cfg.safety
        coder_invocations = 0 if resume_at_gate else self._coder_usage(state)
        review_rounds = self._saved_review_round(state)
        stagnant_rounds = 0
        reviewer_triggers = review_rounds * len(self._reviewer_lanes())
        self._prompt_tokens = 0
        state = self._enforce_deadline(issue, state, now=now)
        fix_findings = tuple(f for f in state.findings if f.status == "open")
        result: LaneResult | None = None
        head_sha: str | None = None

        while True:
            cancelled = self._cancelled_outcome(issue)
            if cancelled is not None:
                return cancelled
            state = self._enforce_deadline(issue, state, now=now)
            if resume_at_gate:
                # PR #73 review thread 16 / issue #85: requeued pending-CI
                # items are claimable by a different foreman host whose
                # local worktree path does not exist. Skip the
                # commit/push step and re-evaluate the retained PR head
                # against the recorded PR number — the worktree is not
                # the source of truth for a CI-only resume.
                result = None
            else:
                if worktree is None:
                    raise _ForemanEscalation("coding requires a local worktree")
                requests = tuple((f.id, None) for f in fix_findings if f.lane != "ci")
                if not self._resume_proposals(state):
                    state, result, coder_invocations = self._run_coder(
                        issue, state, worktree, fix_findings, now=now
                    )
                    violation = self._policy_violation(result)
                    if violation:
                        return self._fail(issue, state, violation, now=now)
                    if requests:
                        state = self._record_proposals(state, result)
                # --- Review rounds --------------------------------------------
                state = self._enforce_deadline(issue, state, now=now)
                report = self._review(
                    issue,
                    state,
                    worktree,
                    branch,
                    review_rounds,
                    stagnant_rounds,
                    reviewer_triggers,
                    now=now,
                )
                reviewer_triggers = report.triggers
                review_rounds = report.rounds
                head_sha = report.head_sha
                stagnant_rounds = report.stagnant_rounds
                state = self._enforce_deadline(issue, self._load(issue, state), now=now)
                if report.remaining:
                    if review_rounds >= safety.max_total_iterations:
                        return self._escalate(
                            issue,
                            state,
                            f"unresolved findings after {review_rounds} review rounds",
                            now=now,
                        )
                    fix_findings = report.remaining
                    continue  # fixes dispositioned; run the coder again
                if self._clean_no_change(state, worktree):
                    state = self._transition(
                        issue,
                        state,
                        "done",
                        terminal_reason="developer reported no changes",
                        now=now,
                    )
                    return WorkItemOutcome(
                        issue=issue,
                        final_phase="done",
                        reason="developer reported no changes",
                        review_rounds=review_rounds,
                        coder_invocations=coder_invocations,
                    )

            # --- CI gate ------------------------------------------------------
            state = self._transition(issue, state, "ci_gating", now=now)
            state = self._heartbeat(issue, state, now=now)
            if not resume_at_gate:
                assert worktree is not None  # Coding requires a checkout above.
                if head_sha is None:
                    raise _ForemanEscalation("review completed without a recorded snapshot")
                self._publish_snapshot(worktree, branch, head_sha)
                state = self._persist_resources(issue, state, branch=branch, worktree=worktree)
            try:
                pr = self._ensure_pr(issue, state, branch, head_sha)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404 and state.extras.get("pr_number") is not None:
                    raise _ForemanEscalation(
                        f"recorded PR #{state.extras['pr_number']} could not be found"
                    ) from exc
                raise PRReconciliationError(f"PR reconciliation required: {exc}") from exc
            except Exception as exc:
                raise PRReconciliationError(f"PR reconciliation required: {exc}") from exc
            # _ensure_pr may persist the recorded PR number (advancing the CAS
            # version): reload so the subsequent transitions expect the
            # authoritative state.
            state = self._load(issue, state)
            decision = self._gate.evaluate(issue, pr)
            self._validate_ci_head(pr)
            if decision.passed:
                state = self._transition(issue, state, "updating_pr", now=now)
                state = self._transition(issue, state, "done", terminal_reason="ci green", now=now)
                return WorkItemOutcome(
                    issue=issue,
                    final_phase="done",
                    reason="ci green",
                    review_rounds=review_rounds,
                    coder_invocations=coder_invocations,
                    gate=decision,
                )
            if decision.pending_checks:
                # Not a failure and not ours to busy-wait. Requeue onto the
                # enabled label so a later pass re-selects it (list_ready only
                # sees the enabled label) and re-evaluates the same head. The
                # worktree/branch are retained for that reuse. The wait start
                # is persisted on the first pending requeue so a head that
                # never settles cannot pend forever: once
                # ci_wait_timeout_seconds is exceeded the item escalates
                # (round-2 #5).
                now_dt = now or datetime.now(UTC)
                started = state.extras.get("ci_wait_started_at")
                if started is not None:
                    elapsed = (now_dt - _parse_timestamp(started)).total_seconds()
                    if elapsed > self._cfg.ci_policy.ci_wait_timeout_seconds:
                        return self._escalate(
                            issue,
                            state,
                            "CI checks pending longer than "
                            f"ci_wait_timeout_seconds "
                            f"({self._cfg.ci_policy.ci_wait_timeout_seconds}s)",
                            now=now,
                        )
                else:
                    state = self._save_fresh(
                        issue,
                        state,
                        extras={**state.extras, "ci_wait_started_at": now_dt.isoformat()},
                    )
                state = self._transition(issue, state, "queued", now=now)
                return WorkItemOutcome(
                    issue=issue,
                    final_phase="ci_gating",
                    reason="CI checks pending: " + ", ".join(decision.pending_checks),
                    review_rounds=review_rounds,
                    coder_invocations=coder_invocations,
                    gate=decision,
                )
            if resume_at_gate:
                return self._escalate(
                    issue, state, "CI-only resume cannot safely dispatch local remediation", now=now
                )
            # Real CI failures become findings for the next coding round —
            # unless the review budget is spent and reviews keep reporting
            # nothing new: that is stagnation, not fixable work.
            if state.extras.get("ci_wait_started_at") is not None:
                # The wait is over (the checks settled): drop the stale wait
                # start so a later pending requeue measures a fresh window.
                extras = {k: v for k, v in state.extras.items() if k != "ci_wait_started_at"}
                state = self._save_fresh(issue, state, extras=extras)
            if stagnant_rounds >= self._cfg.escalation.stagnation_rounds_threshold:
                return self._escalate(
                    issue,
                    state,
                    f"no converging signal after {review_rounds} review rounds",
                    now=now,
                )
            if review_rounds >= safety.max_total_iterations:
                return self._escalate(
                    issue, state, f"CI failing after {review_rounds} iterations", now=now
                )
            if not decision.failed_checks:
                # A gate that is neither passing, pending, nor naming a failed
                # check (e.g. "no checks reported and green required") has no
                # signal to turn into findings — loop back would spin forever.
                return self._escalate(
                    issue, state, f"CI gate cannot progress: {decision.detail}", now=now
                )
            failure = self._signal_hash(
                [
                    *sorted(normalize_claim(name) for name in decision.failed_checks),
                    normalize_claim(decision.detail),
                ]
            )
            seen = state.extras.get("ci_failure_hashes", [])
            if not isinstance(seen, list):
                raise _ForemanEscalation("malformed durable CI failure history")
            if failure in seen:
                return self._escalate(
                    issue,
                    state,
                    "no converging signal: repeated CI failure after remediation",
                    now=now,
                )
            state = self._save_fresh(
                issue,
                state,
                extras={**state.extras, "ci_failure_hashes": [*seen, failure]},
            )
            fix_findings = tuple(
                ReviewerFinding(
                    id=f"ci-failed-{name}",
                    lane="ci",
                    body=f"CI check {name} failed. {decision.detail[:1000]}",
                    severity="major",
                    run_id=self._run_id,
                    round_id=state.round_id or "ci",
                )
                for name in decision.failed_checks
            )
            # Loop back to coding with the CI-check findings.

    def _enforce_deadline(
        self, issue: GitHubIssueRef, state: WorkflowState, *, now: datetime | None
    ) -> WorkflowState:
        current = now or datetime.now(UTC)
        started = state.extras.get("run_started_at")
        if started is None:
            return self._save_fresh(
                issue,
                state,
                extras={**state.extras, "run_started_at": current.isoformat()},
            )
        try:
            elapsed = (current - _parse_timestamp(started)).total_seconds()
        except (TypeError, ValueError) as exc:
            raise _ForemanEscalation("malformed durable run start time") from exc
        if elapsed >= self._cfg.escalation.max_run_seconds:
            raise _ForemanEscalation("run wall-clock deadline exhausted")
        return state

    @staticmethod
    def _signal_hash(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _review_signature(
        self,
        worktree: str,
        findings: tuple[ReviewerFinding, ...],
        decisions: list[FindingDisposition],
    ) -> str:
        return self._signal_hash(
            {
                "patch": self._git.patch_digest(worktree, self._git.default_branch()),
                "findings": sorted(
                    (
                        normalize_claim(f.claim or f.body),
                        f.path or "",
                        f.line or 0,
                        f.severity,
                    )
                    for f in findings
                ),
                "decisions": sorted(
                    (d.finding_id, d.action, normalize_claim(d.rationale)) for d in decisions
                ),
            },
        )

    def _coder_usage(self, state: WorkflowState) -> int:
        usage = state.extras.get("coder_usage")
        if usage is None:
            if state.phase != "claiming" or any(
                (state.round_id, state.findings, state.dispositions, state.archived)
            ):
                raise _ForemanEscalation("coder budget usage is unknown; reconciliation required")
            return 0
        if not isinstance(usage, dict) or set(usage) != {"run_id", "invocations"}:
            raise _ForemanEscalation("malformed durable coder usage")
        count = usage["invocations"]
        if type(count) is not int or count < 0:
            raise _ForemanEscalation("malformed durable coder usage")
        if usage["run_id"] != self._run_id:
            raise _ForemanEscalation("coder usage belongs to another run; reconciliation required")
        return count

    def _run_coder(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str,
        findings: tuple[ReviewerFinding, ...],
        *,
        now: datetime | None,
    ) -> tuple[WorkflowState, LaneResult, int]:
        """Run the existing bounded coder attempts until success or escalation.

        Invocation budget spans successful turns; consecutive failures reset only
        after success. Failed attempts do not consume a reviewer round.
        """
        failures = 0
        invocations = self._coder_usage(state)
        cap = self._cfg.safety.max_coder_invocations_per_run
        requests = tuple((f.id, None) for f in findings if f.lane != "ci")
        while True:
            if invocations >= cap:
                raise _ForemanEscalation("coder invocation budget exhausted with open findings")
            state = self._transition(
                issue,
                state,
                "coding",
                round_id=f"response-{self._saved_review_round(state)}",
                now=now,
            )
            state = self._heartbeat(issue, state, now=now)
            # Reserve the attempt durably before dispatch; uncertain outcomes retain the charge.
            invocations += 1
            charged = replace(
                state,
                updated_at=datetime.now(UTC),
                extras={
                    **state.extras,
                    "coder_usage": {"run_id": self._run_id, "invocations": invocations},
                },
            )
            self._queue.save_state(charged, expected_updated_at=state.updated_at)
            state = charged
            state = self._save_fresh(
                issue,
                state,
                extras={**state.extras, "head_sha": self._git.head_sha(worktree)},
            )
            worker_lane = self._worker_lane()
            result = self._run_lane(
                worker_lane,
                worktree,
                state,
                self._coder_prompt(issue, findings, worktree, state.dispositions, state),
                requests,
                issue=issue,
            )
            state = self._load(issue, state)
            checkout_violation = self._developer_head_violation(state, worktree)
            if checkout_violation:
                raise _ForemanEscalation(checkout_violation)
            if result.exit_code == 0:
                violation = self._developer_report_violation(result)
                if violation:
                    raise _ForemanEscalation(violation)
                state = self._save_fresh(
                    issue,
                    state,
                    extras={**state.extras, "developer_report": json.loads(result.output_summary)},
                )
                return state, result, invocations
            failures += 1
            if invocations >= cap:
                raise _ForemanEscalation("coder budget exhausted on failing attempts")
            if failures >= self._cfg.escalation.max_consecutive_coder_failures:
                raise _ForemanEscalation(f"coder failed {failures}x consecutively")
            state = self._advance_worker_route(issue, state, worker_lane)

    # --- Review ---------------------------------------------------------------

    def _review(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        developer_worktree: str,
        branch: str,
        review_rounds: int,
        stagnant_rounds: int,
        reviewer_triggers: int,
        *,
        now: datetime | None,
    ) -> _RoundReport:
        """Run one blind, isolated reviewer wave against one immutable snapshot."""
        state, round_id, reviewer_lanes = self._begin_review_round(
            issue, state, review_rounds, reviewer_triggers, now=now
        )
        issue_body = self._read_issue_body(issue)
        state, head_sha = self._snapshot(issue, state, developer_worktree, branch)
        policy = self._cfg.review_policy
        registry = FindingRegistry(
            findings=deepcopy(state.findings),
            archived=list(state.archived),
            require_coder_reply_before_resolve=policy.require_coder_reply_before_resolve,
            quarantine_unknown_head_sha=False,
        )
        proposals = self._pending_proposals(state)
        state, seats = self._plan_review_seats(
            issue, state, round_id, head_sha, reviewer_lanes, registry, proposals, issue_body
        )
        try:
            prompts = self._materialize_review_seats(
                issue,
                round_id,
                head_sha,
                developer_worktree,
                issue_body,
                registry,
                proposals,
                seats,
            )
            outcomes, heartbeat_error = self._run_review_seats(state, head_sha, seats, prompts)
        finally:
            for seat in seats:
                self._release(seat.lease)

        waves = deepcopy(cast(dict[str, dict[str, object]], state.extras.get("review_waves", {})))
        wave = waves[round_id]
        registry, dispositions, completed, failures = self._collect_review_outcomes(
            seats, outcomes, state, head_sha, registry, proposals, wave
        )
        if heartbeat_error is not None:
            failures.append(heartbeat_error)
        remaining, had_signal = self._settle_review_findings(registry, dispositions, round_id)
        registry.compact()

        self._persist_round(issue, state, round_id, registry, dispositions)
        signature = self._review_signature(developer_worktree, remaining, dispositions)
        seen = state.extras.get("review_signatures", [])
        if not isinstance(seen, list):
            raise _ForemanEscalation("malformed durable review progress history")
        waves[round_id] = wave
        self._save_fresh(
            issue,
            state,
            extras={
                **state.extras,
                "review_waves": waves,
                "review_signatures": [*seen, signature],
            },
        )
        if heartbeat_error is None:
            failures.extend(self._cleanup_review_results(completed, head_sha))
        if failures:
            raise _ForemanEscalation("review wave incomplete: " + "; ".join(failures))

        rounds = review_rounds + 1
        stagnant = stagnant_rounds + 1 if signature in seen or not had_signal else 0
        if rounds > 1 and stagnant >= self._cfg.escalation.stagnation_rounds_threshold:
            raise _ForemanEscalation("review rounds produced no converging signal")
        return _RoundReport(
            remaining,
            rounds,
            stagnant,
            reviewer_triggers + len(reviewer_lanes),
            head_sha,
        )

    def _begin_review_round(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        review_rounds: int,
        reviewer_triggers: int,
        *,
        now: datetime | None,
    ) -> tuple[WorkflowState, str, list[str]]:
        policy = self._cfg.review_policy
        if review_rounds >= policy.max_review_rounds:
            raise _ForemanEscalation(
                "review-round cap exhausted while fix findings pending "
                f"(limit {policy.max_review_rounds}); refusing unverified changes"
            )
        reviewer_lanes = self._reviewer_lanes()
        budget_left = self._cfg.safety.max_reviewer_triggers_per_run - reviewer_triggers
        if not reviewer_lanes:
            raise _ForemanEscalation("review round requires at least one reviewer lane")
        if budget_left < len(reviewer_lanes):
            raise _ForemanEscalation(
                f"reviewer trigger budget exhausted ({reviewer_triggers}/"
                f"{self._cfg.safety.max_reviewer_triggers_per_run}); review round "
                f"requires all {len(reviewer_lanes)} reviewer lanes"
            )
        round_id = f"review-{review_rounds + 1}"
        state = self._transition(issue, state, "reviewing", round_id=round_id, now=now)
        return self._heartbeat(issue, state, now=now), round_id, reviewer_lanes

    def _collect_review_outcomes(
        self,
        seats: list[_ReviewSeat],
        outcomes: dict[str, LaneResult | Exception],
        state: WorkflowState,
        head_sha: str,
        registry: FindingRegistry,
        proposals: dict[str, FindingDisposition],
        wave: dict[str, object],
    ) -> tuple[
        FindingRegistry,
        list[FindingDisposition],
        list[tuple[_ReviewSeat, LaneResult]],
        list[str],
    ]:
        dispositions: list[FindingDisposition] = []
        completed: list[tuple[_ReviewSeat, LaneResult]] = []
        failures: list[str] = []
        for seat in seats:
            registry, decisions, result, failure = self._consume_review_outcome(
                seat, outcomes.get(seat.lane.lane), state, head_sha, registry, proposals, wave
            )
            dispositions.extend(decisions)
            if result is not None:
                completed.append((seat, result))
            if failure is not None:
                failures.append(f"{seat.lane.lane}: {failure}")
        return registry, dispositions, completed, failures

    def _consume_review_outcome(
        self,
        seat: _ReviewSeat,
        outcome: LaneResult | Exception | None,
        state: WorkflowState,
        head_sha: str,
        registry: FindingRegistry,
        proposals: dict[str, FindingDisposition],
        wave: dict[str, object],
    ) -> tuple[FindingRegistry, list[FindingDisposition], LaneResult | None, str | None]:
        receipts = cast(dict[str, dict[str, object]], wave["lanes"])
        receipt = receipts[seat.lane.lane]
        if isinstance(outcome, Exception):
            receipt.update(status="uncertain", error=str(outcome))
            return registry, [], None, str(outcome)
        if outcome is None:
            error = "reviewer returned no result"
            receipt.update(status="uncertain", error=error)
            return registry, [], None, error

        error = self._review_result_error(seat, outcome, state, head_sha)
        decisions: list[FindingDisposition] = []
        updated = registry
        if error is None:
            candidate = deepcopy(registry)
            try:
                self._apply_review_result(candidate, outcome, proposals)
            except Exception as exc:
                error = f"invalid review result: {exc}"
            else:
                updated = candidate
                decisions = list(outcome.dispositions)
        receipt.update(
            status="completed" if error is None else "invalid",
            session=outcome.session.session_id,
            exit_code=outcome.exit_code,
            findings=len(outcome.findings),
            error=error,
        )
        return updated, decisions, outcome, error

    def _settle_review_findings(
        self,
        registry: FindingRegistry,
        dispositions: list[FindingDisposition],
        round_id: str,
    ) -> tuple[tuple[ReviewerFinding, ...], bool]:
        conflicts = registry.detect_conflicts()
        conflict_members = {fid for ids in conflicts.values() for fid in ids}
        open_findings = [f for f in registry.findings if f.status == "open"]
        had_signal = bool(open_findings or dispositions)
        for finding in open_findings:
            if not self._deferable_finding(finding, conflict_members):
                continue
            _, disposition = registry.apply_disposition(
                finding.id,
                "reply_deferred",
                rationale="minor finding deferred by policy",
                decided_by="foreman",
            )
            dispositions.append(replace(disposition, run_id=self._run_id, round_id=round_id))
        return tuple(f for f in registry.findings if f.status == "open"), had_signal

    @staticmethod
    def _deferable_finding(finding: ReviewerFinding, conflict_members: set[str]) -> bool:
        return (
            finding.id not in conflict_members
            and SEVERITY_RANK[finding.severity] < SEVERITY_RANK["major"]
            and finding.thread_id is None
        )

    def _plan_review_seats(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        round_id: str,
        head_sha: str,
        reviewer_lanes: list[str],
        registry: FindingRegistry,
        proposals: dict[str, FindingDisposition],
        issue_body: str,
    ) -> tuple[WorkflowState, list[_ReviewSeat]]:
        seats: list[_ReviewSeat] = []
        peers: list[str] = []
        in_flight: dict[str, int] = {}
        registry.detect_conflicts()
        try:
            for lane_name in reviewer_lanes:
                lane = self._lanes.get(lane_name)
                requests = self._review_requests(lane_name, registry, proposals, state)
                independent_of = self._adjudication_peers(state, registry, requests)
                lease, fallbacks = self._reserve(
                    lane,
                    state,
                    peers=tuple(peers),
                    in_flight=in_flight,
                    independent_of=independent_of,
                )
                worktree = f"{self._worktree_root}/issue-{issue.number}-{round_id}-{lane_name}"
                seats.append(_ReviewSeat(lane, worktree, lease, fallbacks, requests))
                ref = lease.assignment.model_ref
                peers.append(ref)
                in_flight[ref] = in_flight.get(ref, 0) + 1
            wave: dict[str, object] = {
                "run_id": self._run_id,
                "round_id": round_id,
                "head_sha": head_sha,
                "issue_sha256": hashlib.sha256(issue_body.encode("utf-8")).hexdigest(),
                "lanes": {
                    seat.lane.lane: {
                        "worktree": seat.worktree,
                        "assignment": seat.lease.assignment.to_dict(),
                        "fallbacks": list(seat.fallbacks),
                        "status": "planned",
                    }
                    for seat in seats
                },
            }
            waves = deepcopy(
                cast(dict[str, dict[str, object]], state.extras.get("review_waves", {}))
            )
            waves[round_id] = wave
            extras = state.extras
            for seat in seats:
                extras = self._authorize_spend(
                    replace(state, extras=extras), seat.lease, f"{round_id}:{seat.lane.lane}"
                )
            state = self._save_fresh(issue, state, extras={**extras, "review_waves": waves})
        except Exception:
            for seat in seats:
                self._release(seat.lease)
            raise
        return state, seats

    @staticmethod
    def _needs_independent_review(finding: ReviewerFinding, state: WorkflowState) -> bool:
        return bool(finding.conflict_group_id) or any(
            d.finding_id == finding.id and d.action == "fix" and d.response_to_round_id
            for d in state.dispositions
        )

    def _adjudication_peers(
        self,
        state: WorkflowState,
        registry: FindingRegistry,
        requests: tuple[tuple[str, str | None], ...],
    ) -> tuple[str, ...]:
        requested = {fid for fid, _ in requests}
        findings = [
            f
            for f in registry.findings
            if f.id in requested and self._needs_independent_review(f, state)
        ]
        groups = {f.conflict_group_id for f in findings if f.conflict_group_id}
        sources = [
            f
            for f in registry.findings
            if f in findings or (f.conflict_group_id and f.conflict_group_id in groups)
        ]
        waves = state.extras.get("review_waves", {})
        refs: set[str] = set()
        for finding in sources:
            wave = waves.get(finding.round_id, {})
            lane = wave.get("lanes", {}).get(finding.lane, {})
            ref = lane.get("assignment", {}).get("model_ref")
            if not isinstance(ref, str) or not ref:
                raise _ForemanEscalation("independent adjudication lacks origin model provenance")
            refs.add(ref)
        return tuple(sorted(refs))

    def _review_requests(
        self,
        lane_name: str,
        registry: FindingRegistry,
        proposals: dict[str, FindingDisposition],
        state: WorkflowState,
    ) -> tuple[tuple[str, str | None], ...]:
        return tuple(
            (finding.id, proposals[finding.id].round_id)
            for finding in registry.findings
            if finding.id in proposals and self._adjudicator(finding, registry, state) == lane_name
        )

    def _materialize_review_seats(
        self,
        issue: GitHubIssueRef,
        round_id: str,
        head_sha: str,
        developer_worktree: str,
        issue_body: str,
        registry: FindingRegistry,
        proposals: dict[str, FindingDisposition],
        seats: list[_ReviewSeat],
    ) -> dict[str, str]:
        prompts: dict[str, str] = {}
        created_paths: list[str] = []
        try:
            for seat in seats:
                created = self._git.create_review_checkout(
                    seat.worktree, developer_worktree, head_sha
                )
                if created != seat.worktree:
                    raise _ForemanEscalation(
                        f"review checkout path changed: planned {seat.worktree}, created {created}"
                    )
                created_paths.append(created)
                prompt = self._reviewer_prompt(
                    issue,
                    round_id,
                    seat.worktree,
                    lane_name=seat.lane.lane,
                    head_sha=head_sha,
                    issue_body=issue_body,
                )
                if seat.requests:
                    prompt += self._adjudication_prompt(registry, proposals, seat.requests)
                self._charge_prompt(prompt)
                prompts[seat.lane.lane] = prompt
        except Exception:
            for path in created_paths:
                with suppress(Exception):
                    self._git.cleanup_review_checkout(path, head_sha)
            raise
        return prompts

    def _run_review_seats(
        self,
        state: WorkflowState,
        head_sha: str,
        seats: list[_ReviewSeat],
        prompts: dict[str, str],
    ) -> tuple[dict[str, LaneResult | Exception], str | None]:
        outcomes: dict[str, LaneResult | Exception] = {}
        heartbeat_error: str | None = None
        try:
            with (
                self._lease_heartbeat(state),
                ThreadPoolExecutor(
                    max_workers=len(seats), thread_name_prefix="aipro-review"
                ) as pool,
            ):
                futures = {
                    pool.submit(
                        self._executor.execute,
                        seat.lane,
                        prompts[seat.lane.lane],
                        seat.worktree,
                        LaneExecutionContext(
                            run_id=self._run_id,
                            round_id=state.round_id,
                            work_item_id=state.work_item_id,
                            disposition_requests=seat.requests,
                            head_sha=head_sha,
                        ),
                        seat.lease,
                    ): seat.lane.lane
                    for seat in seats
                }
                for future in as_completed(futures):
                    lane_name = futures[future]
                    try:
                        outcomes[lane_name] = future.result()
                    except Exception as exc:
                        outcomes[lane_name] = exc
        except _ForemanEscalation as exc:
            heartbeat_error = exc.reason
        return outcomes, heartbeat_error

    def _review_result_error(
        self,
        seat: _ReviewSeat,
        result: LaneResult,
        state: WorkflowState,
        head_sha: str,
    ) -> str | None:
        if result.exit_code != 0:
            return f"reviewer lane failed with exit {result.exit_code}"
        context = LaneExecutionContext(
            self._run_id,
            state.round_id,
            state.work_item_id,
            seat.requests,
            head_sha,
        )
        try:
            context.validate_dispositions(result.dispositions, seat.lane)
        except ValueError as exc:
            return str(exc)
        for finding in result.findings:
            if (finding.lane, finding.run_id, finding.round_id) != (
                seat.lane.lane,
                self._run_id,
                state.round_id,
            ):
                return "finding has forged or stale reviewer attribution"
            if finding.head_sha != head_sha:
                return "finding head_sha does not match the reviewed snapshot"
        return None

    def _cleanup_review_results(
        self, completed: list[tuple[_ReviewSeat, LaneResult]], head_sha: str
    ) -> list[str]:
        failures: list[str] = []
        for seat, result in completed:
            try:
                self._executor.terminate_session(result.session)
            except Exception as exc:
                failures.append(f"{seat.lane.lane} session cleanup failed: {exc}")
                continue
            try:
                self._git.cleanup_review_checkout(seat.worktree, head_sha)
            except Exception as exc:
                failures.append(f"{seat.lane.lane} checkout cleanup failed: {exc}")
        return failures

    @staticmethod
    def _apply_review_result(
        registry: FindingRegistry,
        result: LaneResult,
        proposals: dict[str, FindingDisposition],
    ) -> None:
        for decision in result.dispositions:
            if decision.action == "accept":
                registry.apply_disposition(
                    decision.finding_id,
                    "accept",
                    rationale=decision.rationale,
                    decided_by=decision.decided_by,
                    reply_body=proposals[decision.finding_id].rationale,
                )
        archived_ids = {a.finding_id for a in registry.archived}
        for finding in result.findings:
            if finding.id in archived_ids:
                raise _ForemanEscalation("reviewer reused an archived finding id")
            registry.reconcile(finding)

    def _reviewer_lanes(self) -> list[str]:
        return self._cfg.review_policy.reviewer_lanes or [
            lane.lane for lane in self._lanes if lane.role == "reviewer"
        ]

    def _adjudicator(
        self, finding: ReviewerFinding, registry: FindingRegistry, state: WorkflowState
    ) -> str:
        lanes = self._reviewer_lanes()
        if not lanes or any(self._lanes.get(name).role != "reviewer" for name in lanes):
            raise _ForemanEscalation("independent reviewer required for adjudication")
        if not self._needs_independent_review(finding, state):
            return finding.lane if finding.lane in lanes else lanes[0]
        origins = {finding.lane}
        origins.update(
            f.lane
            for f in registry.findings
            if finding.conflict_group_id and f.conflict_group_id == finding.conflict_group_id
        )
        candidate = next((name for name in lanes if name not in origins), None)
        if candidate is None:
            raise _ForemanEscalation("strong independent adjudicator unavailable for dispute")
        return candidate

    @staticmethod
    def _saved_review_round(state: WorkflowState) -> int:
        turns = [state.round_id, *(d.round_id for d in state.dispositions)]
        return max(
            (int(t.split("-")[-1]) for t in turns if t and t.startswith(("review-", "response-"))),
            default=0,
        )

    def _resume_proposals(self, state: WorkflowState) -> bool:
        pending = self._pending_proposals(state)
        if pending and any(f.thread_id for f in state.findings if f.id in pending):
            raise _ForemanEscalation("pending thread reply requires reconciliation before resume")
        return bool(pending)

    def _pending_proposals(self, state: WorkflowState) -> dict[str, FindingDisposition]:
        latest = {d.finding_id: d for d in state.dispositions}
        pending = {}
        for finding in state.findings:
            decision = latest.get(finding.id)
            if decision is None:
                continue
            if finding.status != "open" or decision.response_to_round_id is not None:
                continue
            if (decision.run_id, decision.decided_by, decision.action) not in {
                (self._run_id, self._worker_lane().lane, "fix"),
                (self._run_id, self._worker_lane().lane, "rebut"),
            }:
                raise _ForemanEscalation("pending proposal lacks current run/turn provenance")
            if not decision.round_id:
                raise _ForemanEscalation("pending proposal lacks current run/turn provenance")
            pending[finding.id] = decision
        return pending

    def _record_proposals(self, state: WorkflowState, result: LaneResult) -> WorkflowState:
        requests = tuple((f.id, None) for f in state.findings if f.status == "open")
        LaneExecutionContext(
            self._run_id, state.round_id, disposition_requests=requests
        ).validate_dispositions(result.dispositions, self._worker_lane())
        by_id = {f.id: f for f in state.findings}
        proposals = [
            replace(d, thread_id=by_id[d.finding_id].thread_id, reply_body=d.rationale)
            for d in result.dispositions
        ]
        self._reject_repeated_rebuttals(state, proposals)
        history = self._merge_dispositions(state.dispositions, proposals)
        # Save real coder evidence before any reviewer or thread side effect.
        saved = replace(state, dispositions=history, updated_at=datetime.now(UTC))
        self._queue.save_state(saved, expected_updated_at=state.updated_at)
        for proposal in proposals:
            if proposal in state.dispositions:
                continue  # Exact replay cannot repeat a thread mutation.
            finding = by_id[proposal.finding_id]
            if finding.thread_id:
                self._post_thread_reply(finding.thread_id, proposal.rationale)
        return saved

    @classmethod
    def _reject_repeated_rebuttals(
        cls, state: WorkflowState, proposals: list[FindingDisposition]
    ) -> None:
        rejected = {
            (d.finding_id, d.response_to_round_id)
            for d in state.dispositions
            if d.action == "fix" and d.response_to_round_id is not None
        }
        for proposal in proposals:
            if proposal.action != "rebut":
                continue
            digest = cls._signal_hash(normalize_claim(proposal.rationale))
            if any(
                old.action == "rebut"
                and old.finding_id == proposal.finding_id
                and (old.finding_id, old.round_id) in rejected
                and cls._signal_hash(normalize_claim(old.rationale)) == digest
                for old in state.dispositions
            ):
                raise _ForemanEscalation("repeated rejected rebuttal without new evidence")

    @staticmethod
    def _adjudication_prompt(
        registry: FindingRegistry,
        proposals: dict[str, FindingDisposition],
        requests: tuple[tuple[str, str | None], ...],
    ) -> str:
        findings = {f.id: f for f in registry.findings}
        packet = [
            {"finding": findings[fid].to_dict(), "proposal": proposals[fid].to_dict()}
            for fid, _ in requests
        ]
        return (
            "\nFor this adjudication return an object instead of the array: "
            '{"findings":[],"dispositions":[{"finding_id":"...","action":"accept",'
            '"rationale":"independent evidence","response_to_round_id":"response-N"}]}. '
            "Supply exactly one decision per requested finding. Accept confirms the coder response; "
            "fix rejects it and requests further remediation. Echo each proposal round_id as "
            "response_to_round_id; never accept your own or a different turn's response.\n"
            + json.dumps(packet)
        )

    def _persist_round(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        round_id: str,
        registry: FindingRegistry,
        dispositions: list[FindingDisposition],
    ) -> None:
        """Merge this round's findings/dispositions into durable state.

        Rounds accumulate rather than replace: a restart must reconcile every
        review round, so prior findings/dispositions/archive are preserved and
        this round's immutable decisions are appended without losing earlier rationale.

        Persistence is restart-safety-critical: any failure to save raises
        ``_ForemanEscalation`` so the pass aborts and escalates instead of
        silently dropping authoritative state.
        """
        fresh = self._load(issue, state)
        if fresh is None:
            raise _ForemanEscalation(
                f"cannot persist review round {round_id}: no authoritative state for {issue.slug()}"
            )
        archived_ids = {a.finding_id for a in registry.archived}
        merged_findings = [f for f in registry.findings if f.id not in archived_ids]
        merged_dispositions = self._merge_dispositions(fresh.dispositions, dispositions)
        merged_archived = self._merge_archived(fresh.archived, registry.archived)
        current = (fresh.findings, fresh.dispositions, fresh.archived)
        desired = (merged_findings, merged_dispositions, merged_archived)
        if current == desired and fresh.round_id == round_id:
            return  # Reconcile an already committed decision without another write.
        if current != (state.findings, state.dispositions, state.archived):
            raise _ForemanEscalation(
                "finding history changed during review; reconciliation required"
            )
        try:
            updated = replace(
                fresh,
                updated_at=datetime.now(UTC),
                round_id=round_id,
                findings=merged_findings,
                dispositions=merged_dispositions,
                archived=merged_archived,
            )
            self._queue.save_state(updated, expected_updated_at=fresh.updated_at)
        except Exception as exc:
            raise _ForemanEscalation(
                f"persist review round {round_id} for {issue.slug()} failed: {exc}"
            ) from exc

    def _save_fresh(
        self, issue: GitHubIssueRef, state: WorkflowState, **changes: object
    ) -> WorkflowState:
        """Load the authoritative state, apply ``changes`` with a FRESH
        ``updated_at``, and save under the loaded version's CAS precondition.

        Every state write must advance ``updated_at``: a ``replace`` that
        keeps the loaded timestamp leaves the CAS version unchanged, so a
        stale concurrent writer could still pass the precondition and
        clobber this write (round-2 #4).
        """
        fresh = self._load(issue, state)
        if fresh.phase in TERMINAL_PHASES:
            raise _ForemanEscalation(f"cannot modify terminal work item {fresh.phase!r}")
        if "updated_at" not in changes:
            changes = {**changes, "updated_at": datetime.now(UTC)}
        updated = replace(fresh, **changes)  # type: ignore[arg-type]
        self._queue.save_state(updated, expected_updated_at=fresh.updated_at)
        return updated

    def _post_thread_reply(self, thread_id: str, body: str) -> None:
        """Post the recorded coder reply to the GitHub review thread.

        Persistence-critical: a failure to post is a failure of the
        disposition itself, so it propagates (as an escalation) rather than
        leaving a disposition that claims a reply the thread never received.
        """
        client = getattr(self._queue, "_client", None)
        reply = getattr(client, "reply_to_review_thread", None)
        if reply is None:
            raise _ForemanEscalation(
                f"cannot reply to review thread {thread_id}: the queue client "
                "does not support review-thread replies"
            )
        try:
            reply(thread_id, body)
        except Exception as exc:
            raise _ForemanEscalation(
                f"posting coder reply to review thread {thread_id} failed: {exc}"
            ) from exc

    @staticmethod
    def _merge_dispositions(
        existing: list[FindingDisposition],
        current: list[FindingDisposition],
    ) -> list[FindingDisposition]:
        merged = list(existing)
        for decision in current:
            key = (decision.run_id, decision.round_id, decision.decided_by, decision.finding_id)
            prior = next(
                (d for d in merged if (d.run_id, d.round_id, d.decided_by, d.finding_id) == key),
                None,
            )
            if prior is None:
                merged.append(decision)
            elif prior != decision:
                raise _ForemanEscalation("conflicting replay of immutable finding disposition")
        return merged

    @staticmethod
    def _merge_archived(
        existing: list[ArchivedFinding], current: list[ArchivedFinding]
    ) -> list[ArchivedFinding]:
        seen = {record.finding_id for record in existing}
        merged = list(existing)
        for record in current:
            if record.finding_id not in seen:
                merged.append(record)
                seen.add(record.finding_id)
        return merged

    # --- Lane execution ---------------------------------------------------------

    @staticmethod
    def _spend_charges(state: WorkflowState) -> dict[str, object]:
        ledger = state.extras.get("spend_authorizations")
        if ledger is None:
            return {}
        if not isinstance(ledger, dict) or ledger.get("run_id") != state.run_id:
            raise _ForemanEscalation("spend guard: missing or foreign run attribution")
        charges = ledger.get("charges")
        if not isinstance(charges, dict):
            raise _ForemanEscalation("spend guard: malformed durable authorizations")
        return charges

    @staticmethod
    def _cash_total(charges: dict[str, object]) -> float:
        total = 0.0
        for row in charges.values():
            if not isinstance(row, dict):
                raise _ForemanEscalation("spend guard: invalid persisted cash-charge ceiling")
            amount = cast(dict[str, object], row).get("max_charge_usd")
            if (
                not isinstance(amount, (int, float))
                or isinstance(amount, bool)
                or not math.isfinite(amount)
                or amount < 0
            ):
                raise _ForemanEscalation("spend guard: invalid persisted cash-charge ceiling")
            total += amount
        return total

    def _authorize_spend(
        self, state: WorkflowState, lease: ModelLease, dispatch_id: str
    ) -> dict[str, object]:
        limit = self._cfg.safety.max_run_spend_usd
        if limit is None:
            return state.extras
        if lease.max_charge_usd is None:
            raise _ForemanEscalation("spend guard: broker has no guaranteed cash-charge ceiling")
        charged = self._spend_charges(state)
        entry = {
            "model_ref": lease.assignment.model_ref,
            "max_charge_usd": lease.max_charge_usd,
        }
        if dispatch_id in charged and charged[dispatch_id] != entry:
            raise _ForemanEscalation("spend guard: conflicting dispatch authorization")
        charged = {**charged, dispatch_id: entry}
        if self._cash_total(charged) > limit:
            raise _ForemanEscalation("spend guard: maximum run cash charge exhausted")
        return {
            **state.extras,
            "spend_authorizations": {"run_id": state.run_id, "charges": charged},
        }

    def _charge_prompt(self, prompt: str) -> None:
        estimated = (len(prompt) + 3) // 4
        if self._prompt_tokens + estimated > self._cfg.safety.max_prompt_tokens:
            raise _ForemanEscalation(
                f"prompt token budget exceeded: {self._prompt_tokens + estimated} > "
                f"max {self._cfg.safety.max_prompt_tokens}"
            )
        self._prompt_tokens += estimated

    def _run_lane(
        self,
        lane: LaneIdentity,
        worktree: str,
        state: WorkflowState,
        prompt: str,
        disposition_requests: tuple[tuple[str, str | None], ...] = (),
        *,
        issue: GitHubIssueRef | None = None,
    ) -> LaneResult:
        lease, fallbacks = self._reserve(lane, state)
        context = LaneExecutionContext(
            run_id=self._run_id,
            round_id=state.round_id,
            work_item_id=state.work_item_id,
            disposition_requests=disposition_requests,
            head_sha=cast(str | None, state.extras.get("head_sha")),
        )
        try:
            self._charge_prompt(prompt)
        except Exception:
            self._release(lease)
            raise
        try:
            if issue is not None and lane.role == "worker":
                extras = self._authorize_spend(state, lease, f"coder:{self._coder_usage(state)}")
                state = self._save_fresh(
                    issue,
                    state,
                    extras={
                        **extras,
                        "developer_model": lease.assignment.to_dict(),
                        "developer_fallbacks": list(fallbacks),
                    },
                )
            with self._lease_heartbeat(state):
                result = self._executor.execute(lane, prompt, worktree, context, lease)
            if issue is not None and lane.role == "worker":
                fresh = self._load(issue, state)
                self._save_fresh(
                    issue,
                    fresh,
                    extras={**fresh.extras, "developer_session": result.session.session_id},
                )
            return result
        finally:
            self._release(lease)

    @contextlib.contextmanager
    def _lease_heartbeat(self, state: WorkflowState):
        """Renew the claim lease while a blocking lane runs (round-2 #1).

        A lane that outlasts ``lease_seconds`` would otherwise let
        ``reclaim_expired`` hand the issue to another foreman mid-edit. A
        daemon thread heartbeats at a third of the lease interval; the first
        heartbeat failure aborts the keeper and the lane's outcome is
        escalated rather than continued on a lost claim.
        """
        heartbeat = getattr(self._queue, "heartbeat", None)
        interval = max(0.05, self._cfg.github_queue.lease_seconds / 3)
        if heartbeat is None:
            yield
            return
        stop = threading.Event()
        failures: list[str] = []
        current = state

        def _keep() -> None:
            nonlocal current
            while not stop.wait(interval):
                try:
                    current = heartbeat(current)
                except Exception as exc:
                    failures.append(str(exc))
                    return

        worker = threading.Thread(target=_keep, daemon=True, name="foreman-lease-heartbeat")
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=5.0)
        # A failed heartbeat invalidates success, but must not mask a lane error.
        if failures:
            raise _ForemanEscalation(f"claim lease heartbeat failed: {failures[0]}")

    def _reserve(
        self,
        lane: LaneIdentity,
        state: WorkflowState,
        *,
        peers: tuple[str, ...] = (),
        in_flight: dict[str, int] | None = None,
        independent_of: tuple[str, ...] = (),
    ) -> tuple[ModelLease, tuple[str, ...]]:
        saved = self._saved_worker_route(lane, state)
        if saved is not None:
            assignment, fallbacks = saved
            return self._broker.reserve(assignment), fallbacks
        ref = self._cfg.model_router.lane_assignments.get(lane.lane)
        if ref and not independent_of:
            assignment = ModelAssignment(lane=lane.lane, model_ref=ref)
            return self._broker.reserve(assignment), ()
        select = getattr(self._broker, "select", None)
        if select is None:
            raise ForemanQueueError(f"broker cannot resolve a model for lane {lane.lane!r}")
        decision = select(
            TaskDemand(
                lane=lane.lane,
                role=lane.role,
                difficulty=MAX_TASK_DIFFICULTY if independent_of else 1,
                peers=peers,
                independent_of=independent_of,
                require_bounded_spend=self._cfg.safety.max_run_spend_usd is not None,
                in_flight=in_flight or {},
            )
        )
        if decision.assignment is None:
            raise ForemanQueueError(f"no model available for lane {lane.lane!r}: {decision.reason}")
        return self._reserve_decision(lane, decision.assignment, tuple(decision.fallbacks))

    def _reserve_decision(
        self,
        lane: LaneIdentity,
        primary: ModelAssignment,
        fallbacks: tuple[str, ...],
    ) -> tuple[ModelLease, tuple[str, ...]]:
        refs = (primary.model_ref, *fallbacks)
        errors: list[str] = []
        for index, ref in enumerate(refs):
            assignment = ModelAssignment(lane=lane.lane, model_ref=ref)
            try:
                return self._broker.reserve(assignment), refs[index + 1 :]
            except BrokerError as exc:
                errors.append(str(exc))
        raise ForemanQueueError(f"no reservable model for lane {lane.lane!r}: " + "; ".join(errors))

    @staticmethod
    def _saved_worker_route(
        lane: LaneIdentity, state: WorkflowState
    ) -> tuple[ModelAssignment, tuple[str, ...]] | None:
        raw = state.extras.get("developer_model") if lane.role == "worker" else None
        if raw is None:
            return None
        assignment = ModelAssignment.from_dict(raw)
        if assignment.lane != lane.lane:
            raise ForemanQueueError("durable developer model belongs to another lane")
        fallbacks = state.extras.get("developer_fallbacks", [])
        if not isinstance(fallbacks, list) or not all(isinstance(ref, str) for ref in fallbacks):
            raise ForemanQueueError("malformed durable developer fallback chain")
        return assignment, tuple(fallbacks)

    def _advance_worker_route(
        self, issue: GitHubIssueRef, state: WorkflowState, lane: LaneIdentity
    ) -> WorkflowState:
        saved = self._saved_worker_route(lane, state)
        if saved is None or not saved[1]:
            return state
        _, fallbacks = saved
        assignment = ModelAssignment(lane=lane.lane, model_ref=fallbacks[0])
        return self._save_fresh(
            issue,
            state,
            extras={
                **state.extras,
                "developer_model": assignment.to_dict(),
                "developer_fallbacks": list(fallbacks[1:]),
            },
        )

    def _release(self, lease: ModelLease) -> None:
        self._broker.release(lease)

    # --- Queue adapters -----------------------------------------------------------

    def _list_ready(self) -> list[GitHubIssueRef]:
        list_ready = getattr(self._queue, "list_ready", None)
        if list_ready is None:
            raise ForemanQueueError(
                "queue does not expose list_ready(); the foreman needs the "
                "queue verbs, not just the state-store protocol"
            )
        return list(list_ready())

    def _claim(
        self,
        issue: GitHubIssueRef,
        *,
        branch: str | None,
        worktree: str | None,
        pr_number: int | None,
        now: datetime | None,
    ) -> WorkflowState:
        return self._queue.claim(
            issue,
            self._run_id,
            branch=branch,
            worktree=worktree,
            pr_number=pr_number,
            now=now,
        )

    def _heartbeat(
        self, issue: GitHubIssueRef, state: WorkflowState, *, now: datetime | None = None
    ) -> WorkflowState:
        """Refresh the queue lease before a lane or gate step.

        A lane execution or CI poll can outlast the configured lease; without
        a heartbeat a competing foreman could reclaim and launch duplicate
        work. A lost claim (heartbeat raises) aborts the item — propagating to
        the crash escalator rather than continuing on a reclaimed item.
        """
        heartbeat = getattr(self._queue, "heartbeat", None)
        if heartbeat is None:
            return state
        return heartbeat(state, now=now)

    def _transition(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        phase: str,
        *,
        round_id: str | None = None,
        terminal_reason: str | None = None,
        now: datetime | None = None,
    ) -> WorkflowState:
        del now  # the queue stamps updated_at itself; a caller clock cannot be injected here
        return self._queue.transition(
            issue, state, phase, round_id=round_id, terminal_reason=terminal_reason
        )

    def _load(self, issue: GitHubIssueRef, fallback: WorkflowState) -> WorkflowState:
        fresh = self._queue.load_state(issue.slug())
        return fresh if fresh is not None else fallback

    def _persist_resources(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        *,
        branch: str,
        worktree: str,
    ) -> WorkflowState:
        """Record the materialized branch/worktree on the authoritative claim.

        Called only after the claim is already won, so resources are created
        post-claim (no leak) and the durable claim carries the lease
        attribution another foreman would need to reuse or reclaim them.
        """
        fresh = self._load(issue, state)
        extras = dict(fresh.extras)
        extras["branch"] = branch
        extras["worktree"] = worktree
        extras["head_sha"] = self._git.head_sha(worktree)
        return self._save_fresh(issue, fresh, extras=extras)

    def _verify_checkout(
        self, issue: GitHubIssueRef, state: WorkflowState, branch: str, worktree: str
    ) -> WorkflowState:
        actual_branch = self._git.current_branch(worktree)
        if actual_branch != branch:
            raise _ForemanEscalation(
                f"unexpected developer branch movement: expected {branch}, found {actual_branch}"
            )
        actual = self._git.head_sha(worktree)
        same_checkout = (
            state.extras.get("branch") == branch and state.extras.get("worktree") == worktree
        )
        expected = state.extras.get("head_sha")
        if same_checkout and expected is not None and expected != actual:
            raise _ForemanEscalation(
                f"unexpected developer HEAD movement: expected {expected}, found {actual}"
            )
        if same_checkout and expected == actual:
            return state
        return self._persist_resources(issue, state, branch=branch, worktree=worktree)

    # --- Policy helpers -------------------------------------------------------------

    def _worker_lane(self) -> LaneIdentity:
        """The configured worker lane (F12): the first role-``worker`` lane, or
        the canonical developer lane as a fallback."""
        for lane in self._lanes:
            if lane.role == "worker":
                return lane
        return self._lanes.get(DEVELOPER_LANE)

    def _policy_violation(self, result: LaneResult) -> str | None:
        if self._cfg.safety.disallow_workflow_file_changes and any(
            f.startswith(".github/workflows/") for f in result.changed_files
        ):
            return "policy violation: lane modified files under .github/workflows/"
        return None

    def _coder_prompt(
        self,
        issue: GitHubIssueRef,
        findings: tuple[ReviewerFinding, ...],
        worktree: str,
        dispositions: list[FindingDisposition] | None = None,
        state: WorkflowState | None = None,
    ) -> str:
        title = self._read_issue_title(issue)
        description = self._read_issue_description(issue, worktree)
        instructions = self._git.repo_instructions(worktree)
        extras = state.extras if state is not None else {}
        lines = [
            f"Implement {issue.slug()}: {title}",
            f"Authoritative branch: {extras.get('branch', 'unknown')}",
            f"Expected HEAD: {extras.get('head_sha', 'unknown')}",
            "Do not commit or push; aipro validates, commits and pushes after your turn.",
            "Run the relevant focused tests and repository checks before reporting success.",
            "Return only JSON with exactly: summary (nonempty string), tests (array of "
            "{command,result,notes}; result is passed|failed|not_run), concerns (string array), "
            "no_changes (boolean), dispositions (array). Set no_changes=true only when the "
            "authoritative worktree has no edits relative to the default branch; explain why "
            "in summary.",
        ]
        if instructions:
            lines.append(f"Repository instructions:\n{instructions}")
        if description:
            lines.append(f"Issue description:\n{description}")
        finding_text = self._finding_prompt(findings, dispositions)
        if finding_text:
            lines.append(finding_text)
        return "\n".join(lines)

    @staticmethod
    def _finding_prompt(
        findings: tuple[ReviewerFinding, ...],
        dispositions: list[FindingDisposition] | None,
    ) -> str:
        if not findings:
            return ""
        latest = {d.finding_id: d for d in dispositions or []}
        lines = ["Address these review findings:"]
        lines.extend(f"- {f.id} [{f.severity}] {f.body}" for f in findings)
        lines.extend(
            f"Prior decision for {f.id}: {latest[f.id].rationale}"
            for f in findings
            if f.id in latest
        )
        if any(f.lane != "ci" for f in findings):
            lines.append(
                "For every requested review finding, include exactly one disposition with "
                "finding_id, action=fix|rebut, and a falsifiable rationale; you cannot accept "
                "your own response."
            )
        return "\n".join(lines)

    def _read_issue_title(self, issue: GitHubIssueRef) -> str:
        client = getattr(self._queue, "_client", None)
        get_title = getattr(client, "get_issue_title", None)
        title = get_title(issue.number) if get_title is not None else None
        return title or issue.slug()

    def _read_issue_body(self, issue: GitHubIssueRef) -> str:
        client = getattr(self._queue, "_client", None)
        get_body = getattr(client, "get_issue_body", None)
        if get_body is None:
            raise ForemanQueueError("queue cannot fetch the authoritative issue description")
        return get_body(issue.number) or ""

    def _format_issue_description(self, description: str, worktree: str) -> str:
        """Deliver one fetched issue body completely in the lane's own checkout."""
        if len(description) <= 8_000:
            return description
        path, digest = self._git.write_issue_description(worktree, description)
        return (
            f"Read the COMPLETE issue description and acceptance criteria from {path!r} "
            f"({len(description.encode('utf-8'))} UTF-8 bytes; SHA-256 {digest}). "
            "Read all pages, including the final criteria; do not rely on a preview. "
            "If the file cannot be read completely or its digest differs, report that failure "
            "instead of implementing or returning a clean review."
        )

    def _read_issue_description(self, issue: GitHubIssueRef, worktree: str) -> str:
        return self._format_issue_description(self._read_issue_body(issue), worktree)

    def _developer_head_violation(self, state: WorkflowState, worktree: str) -> str | None:
        expected_branch = state.extras.get("branch")
        actual_branch = self._git.current_branch(worktree)
        if expected_branch is None or actual_branch != expected_branch:
            return (
                f"unexpected developer branch movement: expected {expected_branch}, "
                f"found {actual_branch}"
            )
        expected = state.extras.get("head_sha")
        actual = self._git.head_sha(worktree)
        if expected is None or actual != expected:
            return f"unexpected developer HEAD movement: expected {expected}, found {actual}"
        return None

    def _clean_no_change(self, state: WorkflowState, worktree: str) -> bool:
        report = state.extras.get("developer_report")
        if not isinstance(report, dict) or report.get("no_changes") is not True:
            return False
        if self._git.changed_files(worktree):
            return False
        return self._git.commit_count(worktree, self._git.default_branch()) == 0

    @staticmethod
    def _developer_report_violation(result: LaneResult) -> str | None:
        try:
            report = json.loads(result.output_summary)
        except (json.JSONDecodeError, TypeError) as exc:
            return f"invalid developer completion report: {exc}"
        keys = {"summary", "tests", "concerns", "no_changes", "dispositions"}
        if not isinstance(report, dict) or set(report) != keys:
            return f"invalid developer completion report: expected exactly {sorted(keys)}"
        if not isinstance(report["summary"], str) or not report["summary"].strip():
            return "invalid developer completion report: summary must be nonempty"
        if not isinstance(report["no_changes"], bool):
            return "invalid developer completion report: no_changes must be boolean"
        if not isinstance(report["dispositions"], list):
            return "invalid developer completion report: dispositions must be an array"
        concern_error = ForemanPolicyLoop._concern_report_violation(report["concerns"])
        test_error = ForemanPolicyLoop._test_report_violation(report["tests"])
        return (
            concern_error
            or test_error
            or ForemanPolicyLoop._no_change_violation(report["no_changes"], result)
        )

    @staticmethod
    def _concern_report_violation(concerns: object) -> str | None:
        if not isinstance(concerns, list) or any(
            not isinstance(item, str) or not item.strip() for item in concerns
        ):
            return "invalid developer completion report: concerns must be nonempty strings"
        return None

    @staticmethod
    def _test_report_violation(tests: object) -> str | None:
        if not isinstance(tests, list):
            return "invalid developer completion report: tests must be an array"
        for test in tests:
            error = ForemanPolicyLoop._one_test_violation(test)
            if error:
                return error
        return None

    @staticmethod
    def _one_test_violation(test: object) -> str | None:
        if not isinstance(test, dict) or set(test) != {"command", "result", "notes"}:
            return "invalid developer completion report: malformed test result"
        record = cast(dict[str, object], test)
        if not isinstance(record["command"], str) or not record["command"].strip():
            return "invalid developer completion report: test command must be nonempty"
        if record["result"] not in {"passed", "failed", "not_run"}:
            return "invalid developer completion report: unknown test result"
        if not isinstance(record["notes"], str):
            return "invalid developer completion report: test notes must be a string"
        if record["result"] == "failed":
            return f"developer reported failing test: {record['command']}"
        return None

    @staticmethod
    def _no_change_violation(no_changes: object, result: LaneResult) -> str | None:
        if result.changed_files and no_changes is True:
            return "developer reported no_changes=true despite authoritative worktree edits"
        if not result.changed_files and no_changes is not True:
            return "developer made no changes but did not report no_changes=true"
        return None

    @staticmethod
    def _review_mandate(lane_name: str) -> str:
        mandates = {
            REQUIREMENTS_REVIEWER_LANE: (
                "Requirements prosecutor: try to prove the implementation does not satisfy "
                "the issue; trace each material requirement to observable behavior."
            ),
            BREAKER_REVIEWER_LANE: (
                "Breaker: hunt edge cases, regressions, missing tests, and state/concurrency "
                "failures; temporary tests and experimental patches are encouraged."
            ),
            ARCHITECTURE_REVIEWER_LANE: (
                "Architecture skeptic: challenge unnecessary complexity, hidden coupling, "
                "maintainability, security, performance, and materially simpler designs."
            ),
        }
        return mandates.get(lane_name, "Independently challenge correctness and maintainability.")

    def _reviewer_prompt(
        self,
        issue: GitHubIssueRef,
        round_id: str,
        worktree: str,
        *,
        lane_name: str,
        head_sha: str,
        issue_body: str,
    ) -> str:
        lines = [
            f"Review immutable snapshot {head_sha} for issue {issue.slug()} ({round_id}).",
            self._review_mandate(lane_name),
            "This first-pass review is blind: do not assume sibling reviewer findings or a "
            "developer self-assessment. Judge the issue specification and repository state.",
            "This checkout is disposable. You may edit files and run tests to investigate, "
            "but do not commit or push; experimental changes are discarded after evidence "
            "is collected.",
            "Return only a JSON array of new findings, or [] when there are none; no Markdown. "
            "Each finding needs id, body and severity (info, minor, major or blocker). "
            'Example: [{"id":"missing-guard","body":"Describe the defect","severity":"major"}]. '
            "Use ReviewerFinding fields for optional location/evidence. Omit created_at, "
            "head_sha, lane/run_id/round_id and durable policy/provenance fields; the "
            "controller supplies snapshot and turn attribution.",
        ]
        description = self._format_issue_description(issue_body, worktree)
        if description:
            lines.append(f"Issue description:\n{description}")
        return "\n".join(lines)

    def _snapshot(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str,
        branch: str,
    ) -> tuple[WorkflowState, str]:
        """Commit developer output once so every reviewer sees one immutable head."""
        cap = self._cfg.safety.max_commits_per_run
        count = self._git.commit_count(worktree, self._git.default_branch())
        pending = bool(self._git.changed_files(worktree))
        if count > cap or (count == cap and pending):
            raise _ForemanEscalation(
                f"commit budget exhausted: {count} commits on {branch} already "
                f">= max_commits_per_run ({cap})"
            )
        sha = self._git.commit(
            worktree,
            f"[aipro] work for {issue.slug()} ({self._run_id})",
            name=self._name,
            email=self._email,
        )
        state = self._save_fresh(issue, state, extras={**state.extras, "head_sha": sha})
        return state, sha

    def _publish_snapshot(self, worktree: str, branch: str, head_sha: str) -> None:
        """Publish exactly the reviewed SHA, never post-review working-tree state."""
        actual_branch = self._git.current_branch(worktree)
        if actual_branch != branch:
            raise _ForemanEscalation(
                f"unexpected developer branch movement: expected {branch}, found {actual_branch}"
            )
        actual_head = self._git.head_sha(worktree)
        if actual_head != head_sha:
            raise _ForemanEscalation(
                f"unexpected developer HEAD movement after review: expected {head_sha}, "
                f"found {actual_head}"
            )
        if self._git.changed_files(worktree):
            raise _ForemanEscalation("developer worktree changed after the reviewed snapshot")
        self._git.push_sha(head_sha, branch)

    def _ensure_pr(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        branch: str,
        head_sha: str | None,
    ) -> GitHubPullRequestRef:
        """Refresh a known PR or reconcile by branch before creating one.

        Lookup and persistence failures propagate: a missing response never
        proves the PR does not exist, including after a successful create.
        """
        client = getattr(self._queue, "_client", None)
        pr_number = state.extras.get("pr_number")
        if pr_number is not None:
            pr = cast(GitHubClient, client).get_pr(int(pr_number))
        else:
            pr = self._discover_open_pr(client, branch)
            if pr is None and hasattr(client, "create_pr"):
                pr = client.create_pr(
                    f"[aipro] issue #{issue.number}: automated change",
                    self._pr_body(issue, state),
                    head=branch,
                    base=self._git.default_branch(),
                )
            if pr is not None:
                self._record_pr(issue, state, pr.number)
        if pr is not None:
            body = self._with_pr_summary(pr.body, self._pr_body(issue, state))
            if body != pr.body:
                expected_head = pr.head_sha
                pr = cast(GitHubClient, client).update_pr_body(pr.number, body)
                if pr.head_sha != expected_head:
                    raise PRReconciliationError("PR head changed during summary update")
            return GitHubPullRequestRef(
                owner=issue.owner,
                repo=issue.repo,
                number=pr.number,
                head_sha=pr.head_sha,
            )
        # Deterministic fallback for fakes that track PRs 1:1 with issues.
        number = issue.number
        self._record_pr(issue, state, number)
        return GitHubPullRequestRef(
            owner=issue.owner,
            repo=issue.repo,
            number=number,
            head_sha=head_sha or f"head-{number}",
        )

    @staticmethod
    def _pr_body(issue: GitHubIssueRef, state: WorkflowState) -> str:
        report = state.extras.get("developer_report", {})
        waves = state.extras.get("review_waves", {})
        reviewers = [
            f"{turn}/{lane}: {record['assignment']['model_ref']}"
            for turn, wave in sorted(waves.items())
            for lane, record in sorted(wave["lanes"].items())
        ]
        developer = state.extras.get("developer_model", {})
        models = ", ".join([f"developer: {developer.get('model_ref', 'unknown')}", *reviewers])
        dispositions = ", ".join(sorted(d.action for d in state.dispositions)) or "none"
        tests = ", ".join(
            f"{test['command']} ({test['result']})" for test in report.get("tests", [])
        ) or "not reported"
        detail = "\n".join(
            (
                f"Refs #{issue.number}",
                f"Run: `{state.run_id}` | Head: `{state.extras.get('head_sha', '')}`",
                f"Implementation: {report.get('summary', 'not reported')}",
                f"Tests: {tests}",
                f"Review: {len(state.findings)} active, {len(state.archived)} archived; "
                f"dispositions: {dispositions}",
                f"Models: {models}",
            )
        )
        return "<!-- aipro-summary:start -->\n" + detail + "\n<!-- aipro-summary:end -->"

    @staticmethod
    def _with_pr_summary(body: str, summary: str) -> str:
        start, end = "<!-- aipro-summary:start -->", "<!-- aipro-summary:end -->"
        if start not in body:
            return (body.rstrip() + "\n\n" + summary).lstrip()
        prefix, _, remainder = body.partition(start)
        _, marker, suffix = remainder.partition(end)
        if not marker:
            raise PRReconciliationError("malformed aipro PR summary markers")
        return prefix + summary + suffix

    def _validate_ci_head(self, pr: GitHubPullRequestRef) -> None:
        client = getattr(self._queue, "_client", None)
        if client is None:
            return
        try:
            actual = client.get_pr(pr.number).head_sha
        except Exception as exc:
            raise PRReconciliationError(f"PR head recheck failed: {exc}") from exc
        if actual != pr.head_sha:
            raise PRReconciliationError(
                f"PR head changed during CI gate: expected {pr.head_sha}, got {actual}"
            )

    def _discover_open_pr(self, client, branch: str):
        """Reconcile the authoritative branch, excluding same-named fork PRs."""
        if client is None or not hasattr(client, "list_open_prs"):
            return None
        for pr in client.list_open_prs():
            if (
                pr.head_ref == branch
                and pr.base_ref == self._git.default_branch()
                and not pr.is_fork
            ):
                return pr
        return None

    def _record_pr(self, issue: GitHubIssueRef, state: WorkflowState, pr_number: int) -> None:
        fresh = self._load(issue, state)
        extras = dict(fresh.extras)
        extras["pr_number"] = pr_number
        self._save_fresh(issue, fresh, extras=extras)

    def _escalate(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> WorkItemOutcome:
        # PR #73 review thread 10 / issue #80: a transient GitHub or CAS
        # failure during ``mark_needs_human`` must not be silently dropped —
        # the durable issue would otherwise remain on its active phase while
        # the foreman reports success. Raise so the caller can decide
        # whether to retry; the prior ``with suppress(Exception)`` swallowed
        # the error and led to worktree cleanup against a non-terminal
        # state.
        self._queue.mark_needs_human(issue, self._load(issue, state), reason=reason)
        return WorkItemOutcome(issue=issue, final_phase="escalated", reason=reason, escalated=True)

    def _fail(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> WorkItemOutcome:
        # Same fix as ``_escalate`` (thread 10): a failed ``queue.fail`` is
        # an authoritative-write failure and must propagate rather than be
        # silently absorbed.
        self._queue.fail(issue, self._load(issue, state), reason=reason)
        return WorkItemOutcome(issue=issue, final_phase="failed", reason=reason)
