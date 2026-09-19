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
import json
import logging
import threading
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol, cast

import httpx

from ai_pr_orchestrator.github.protocol import GitHubClient

from .broker import TaskDemand
from .cao import SessionBusyError
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
from .findings import SEVERITY_RANK, FindingRegistry
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
from .lanes import DEVELOPER_LANE, LaneRegistry


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
                outcomes.append(
                    WorkItemOutcome(
                        issue=issue,
                        final_phase="escalated",
                        reason=reason,
                        escalated=True,
                    )
                )
            # CI-only claims own no local checkout; a retained path can belong
            # to unrelated work on this host. Keep its attribution for recovery.
            if not resume_at_gate:
                self._cleanup_terminal_worktree(issue)
        return outcomes

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
        if state is None or state.run_id != self._run_id or state.phase not in TERMINAL_PHASES:
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
                state = self._persist_resources(issue, state, branch=branch, worktree=worktree)
            return self._run_loop(
                issue, state, worktree, branch, now=now, resume_at_gate=resume_at_gate
            )
        except _ForemanEscalation as exc:
            return self._escalate(issue, state, exc.reason, now=now)

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
        coder_invocations = 0
        coder_failures = 0
        review_rounds = self._saved_review_round(state)
        stagnant_rounds = 0
        reviewer_triggers = review_rounds * len(self._reviewer_lanes())
        self._prompt_tokens = 0
        fix_findings = tuple(f for f in state.findings if f.status == "open")
        result: LaneResult | None = None
        head_sha: str | None = None

        while True:
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
                    # --- Coding ---------------------------------------------------
                    # The cap is checked BEFORE launching: once it is reached with
                    # open findings, no further coder invocation may start
                    # (round-2 #6).
                    if fix_findings and coder_invocations >= safety.max_coder_invocations_per_run:
                        return self._escalate(
                            issue,
                            state,
                            "coder invocation budget exhausted with open findings",
                            now=now,
                        )
                    state = self._transition(
                        issue, state, "coding", round_id=f"response-{review_rounds}", now=now
                    )
                    state = self._heartbeat(issue, state, now=now)
                    result = self._run_lane(
                        self._worker_lane(),
                        worktree,
                        state,
                        self._coder_prompt(issue, fix_findings, worktree, state.dispositions),
                        requests,
                    )
                    coder_invocations += 1
                    if result.exit_code != 0:
                        coder_failures += 1
                        # A failed attempt consumes invocation budget too; it must not
                        # bypass the cap by leaning only on the consecutive-failure
                        # threshold (which counts differently).
                        if coder_invocations >= safety.max_coder_invocations_per_run:
                            return self._escalate(
                                issue,
                                state,
                                "coder budget exhausted on failing attempts",
                                now=now,
                            )
                        if coder_failures >= self._cfg.escalation.max_consecutive_coder_failures:
                            return self._escalate(
                                issue,
                                state,
                                f"coder failed {coder_failures}x consecutively",
                                now=now,
                            )
                        # Below both thresholds the failure is retried: transient lane
                        # crashes must not kill the item, and every retry stays within
                        # the invocation budget.
                        continue
                    coder_failures = 0
                    # A background lease heartbeat may have advanced the CAS
                    # version while the lane ran: reload before further writes.
                    state = self._load(issue, state)
                    violation = self._policy_violation(result)
                    if violation:
                        return self._fail(issue, state, violation, now=now)

                    if requests:
                        state = self._record_proposals(state, result)
                # --- Review rounds --------------------------------------------
                report = self._review(
                    issue,
                    state,
                    worktree,
                    review_rounds,
                    stagnant_rounds,
                    reviewer_triggers,
                    now=now,
                )
                reviewer_triggers = report.triggers
                review_rounds = report.rounds
                stagnant_rounds = report.stagnant_rounds
                state = self._load(issue, state)
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

            # --- CI gate ------------------------------------------------------
            state = self._transition(issue, state, "ci_gating", now=now)
            state = self._heartbeat(issue, state, now=now)
            if not resume_at_gate:
                assert worktree is not None  # Coding requires a checkout above.
                head_sha = self._commit_and_push(
                    issue,
                    state,
                    worktree,
                    branch,
                )
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
            fix_findings = tuple(
                ReviewerFinding(
                    id=f"ci-failed-{name}",
                    lane="ci",
                    body=f"CI check {name} failed",
                    severity="major",
                    run_id=self._run_id,
                    round_id=state.round_id or "ci",
                )
                for name in decision.failed_checks
            )
            # Loop back to coding with the CI-check findings.

    # --- Review ---------------------------------------------------------------

    def _review(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str,
        review_rounds: int,
        stagnant_rounds: int,
        reviewer_triggers: int,
        *,
        now: datetime | None,
    ) -> _RoundReport:
        """Run one review round; dispositions what it found.

        A reviewer lane that crashes is a *failed review round* — raised as an
        escalation — so a crashed reviewer never masquerades as "no findings".
        Reviewer triggers are capped at ``max_reviewer_triggers_per_run``.
        """
        policy = self._cfg.review_policy
        # Every call follows coder work, including CI fixes, and requires review.
        if review_rounds >= policy.max_review_rounds:
            raise _ForemanEscalation(
                "review-round cap exhausted while fix findings pending "
                f"(limit {policy.max_review_rounds}); refusing unverified changes"
            )
        reviewer_lanes = self._reviewer_lanes()
        budget_left = max(0, self._cfg.safety.max_reviewer_triggers_per_run - reviewer_triggers)
        rounds = review_rounds + 1
        round_id = f"review-{rounds}"
        if not reviewer_lanes or budget_left < len(reviewer_lanes):
            # Partial reviewer coverage cannot establish a clean review.
            raise _ForemanEscalation(
                f"reviewer trigger budget exhausted ({reviewer_triggers}/"
                f"{self._cfg.safety.max_reviewer_triggers_per_run}); review round "
                f"{round_id} requires all {len(reviewer_lanes)} reviewer lanes"
            )
        state = self._transition(issue, state, "reviewing", round_id=round_id, now=now)
        state = self._heartbeat(issue, state, now=now)

        registry = FindingRegistry(
            findings=list(state.findings),
            archived=list(state.archived),
            require_coder_reply_before_resolve=policy.require_coder_reply_before_resolve,
            quarantine_unknown_head_sha=False,
        )
        proposals = self._pending_proposals(state)
        dispositions: list[FindingDisposition] = []
        triggers = reviewer_triggers
        for lane_name in reviewer_lanes:
            requests = tuple(
                (f.id, proposals[f.id].round_id)
                for f in registry.findings
                if f.id in proposals and self._adjudicator(f) == lane_name
            )
            prompt = self._reviewer_prompt(issue, round_id, worktree)
            if requests:
                prompt += self._adjudication_prompt(registry, proposals, requests)
            result = self._review_lane(lane_name, worktree, state, prompt, requests)
            self._apply_review_result(registry, result, proposals)
            dispositions.extend(result.dispositions)
            triggers += 1
        # Preserve pending identities: deduplication may not rename an adjudicated finding.
        if not state.findings:
            registry.deduplicate()
        conflicts = registry.detect_conflicts()
        open_findings = [f for f in registry.findings if f.status == "open"]

        conflict_members = {fid for ids in conflicts.values() for fid in ids}
        for finding in open_findings:
            if (
                finding.id in conflict_members
                or SEVERITY_RANK[finding.severity] >= SEVERITY_RANK["major"]
            ):
                continue
            # Defer nonblocking findings explicitly, without pretending the coder replied.
            if finding.thread_id:
                continue
            _, disposition = registry.apply_disposition(
                finding.id,
                "reply_deferred",
                rationale="minor finding deferred by policy",
                decided_by="foreman",
            )
            dispositions.append(replace(disposition, run_id=self._run_id, round_id=round_id))
        remaining = tuple(f for f in registry.findings if f.status == "open")
        registry.compact()
        self._persist_round(issue, state, round_id, registry, dispositions)
        stagnant = stagnant_rounds + 1 if not open_findings and not dispositions else 0
        if rounds > 1 and stagnant >= self._cfg.escalation.stagnation_rounds_threshold:
            raise _ForemanEscalation("review rounds produced no converging signal")
        return _RoundReport(remaining, rounds, stagnant, triggers)

    def _review_lane(
        self,
        lane_name: str,
        worktree: str,
        state: WorkflowState,
        prompt: str,
        requests: tuple[tuple[str, str | None], ...],
    ) -> LaneResult:
        lane = self._lanes.get(lane_name)
        result = self._run_lane(lane, worktree, state, prompt, requests)
        if result.exit_code != 0:
            raise _ForemanEscalation(
                f"reviewer lane {lane_name!r} failed (exit {result.exit_code})"
            )
        violation = self._policy_violation(result)
        if violation:
            raise _ForemanEscalation(violation)
        LaneExecutionContext(
            self._run_id, state.round_id, disposition_requests=requests
        ).validate_dispositions(result.dispositions, lane)
        return result

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
            registry.register(finding)

    def _reviewer_lanes(self) -> list[str]:
        return self._cfg.review_policy.reviewer_lanes or [
            lane.lane for lane in self._lanes if lane.role == "reviewer"
        ]

    def _adjudicator(self, finding: ReviewerFinding) -> str:
        lanes = self._reviewer_lanes()
        if not lanes or any(self._lanes.get(name).role != "reviewer" for name in lanes):
            raise _ForemanEscalation("independent reviewer required for adjudication")
        return finding.lane if finding.lane in lanes else lanes[0]

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

    def _run_lane(
        self,
        lane: LaneIdentity,
        worktree: str,
        state: WorkflowState,
        prompt: str,
        disposition_requests: tuple[tuple[str, str | None], ...] = (),
    ) -> LaneResult:
        lease = self._reserve(lane)
        context = LaneExecutionContext(
            run_id=self._run_id,
            round_id=state.round_id,
            work_item_id=state.work_item_id,
            disposition_requests=disposition_requests,
        )
        # Prompt-token budget: charge before executing so a lane is never
        # launched past the configured cap — escalate rather than exceed.
        estimated = (len(prompt) + 3) // 4
        if self._prompt_tokens + estimated > self._cfg.safety.max_prompt_tokens:
            self._release(lease)
            raise _ForemanEscalation(
                f"prompt token budget exceeded: {self._prompt_tokens + estimated} > "
                f"max {self._cfg.safety.max_prompt_tokens}"
            )
        self._prompt_tokens += estimated
        try:
            with self._lease_heartbeat(state):
                return self._executor.execute(lane, prompt, worktree, context, lease)
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

    def _reserve(self, lane: LaneIdentity) -> ModelLease:
        ref = self._cfg.model_router.lane_assignments.get(lane.lane)
        if ref:
            return self._broker.reserve(ModelAssignment(lane=lane.lane, model_ref=ref))
        decision = getattr(self._broker, "select", None)
        if decision is None:
            raise ForemanQueueError(f"broker cannot resolve a model for lane {lane.lane!r}")
        d = decision(TaskDemand(lane=lane.lane, role=lane.role))
        if d.assignment is None:
            raise ForemanQueueError(f"no model available for lane {lane.lane!r}: {d.reason}")
        return self._broker.reserve(d.assignment)

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
        return self._save_fresh(issue, fresh, extras=extras)

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
    ) -> str:
        lines = [f"Implement issue {issue.slug()} in the current worktree."]
        description = self._read_issue_description(issue, worktree)
        if description:
            lines.append(f"Issue description:\n{description}")
        if findings:
            lines.append("Address these review findings:")
            lines.extend(f"- {f.id} [{f.severity}] {f.body}" for f in findings)
            latest = {d.finding_id: d for d in dispositions or []}
            lines.extend(
                f"Prior decision for {f.id}: {latest[f.id].rationale}"
                for f in findings
                if f.id in latest
            )
            if any(f.lane != "ci" for f in findings):
                lines.append(
                    'Return only {"dispositions":[{"finding_id":"...","action":"fix",'
                    '"rationale":"change/test or falsifiable rebuttal evidence"}]}. '
                    "Exactly one fix or rebut per requested review finding; you cannot accept your own response."
                )
        return "\n".join(lines)

    def _read_issue_description(self, issue: GitHubIssueRef, worktree: str) -> str:
        """Shared complete requirements, inline up to 8,000 characters, else file-linked."""
        client = getattr(self._queue, "_client", None)
        get_body = getattr(client, "get_issue_body", None)
        if get_body is None:
            raise ForemanQueueError("queue cannot fetch the authoritative issue description")
        description = get_body(issue.number) or ""
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

    def _reviewer_prompt(self, issue: GitHubIssueRef, round_id: str, worktree: str) -> str:
        lines = [
            f"Review the changes for issue {issue.slug()} ({round_id}); report structured findings.",
            "Return only a JSON array of new findings, or [] when there are none; no Markdown. "
            "Each finding needs id, body and severity (info, minor, major or blocker). "
            'Example: [{"id":"missing-guard","body":"Describe the defect","severity":"major"}]. '
            "Use ReviewerFinding fields for optional location/evidence. Omit created_at, lane/run_id/round_id "
            "and durable policy/provenance fields; the controller supplies turn attribution.",
        ]
        description = self._read_issue_description(issue, worktree)
        if description:
            lines.append(f"Issue description:\n{description}")
        return "\n".join(lines)

    def _commit_and_push(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str,
        branch: str,
    ) -> str:
        """Commit lane output and push it to the remote branch.

        Called immediately before the PR is (re)opened so the PR targets a
        committed, pushed head — never an uncommitted/local branch. A lane
        that made no edits commits nothing (a no-op returning the current
        HEAD) and pushes the branch so it exists remotely.

        ``max_commits_per_run`` is consulted BEFORE committing: when the
        branch already carries the cap's worth of commits over the default
        branch and this round still has changes to commit, the budget is
        exceeded and the item escalates instead of silently growing past the
        cap (round-2 #7). A visit with nothing to commit consumes no budget.
        """
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
        self._git.push(branch)
        return sha

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
                    f"Automated work for {issue.slug()}.",
                    head=branch,
                    base=self._git.default_branch(),
                )
            if pr is not None:
                self._record_pr(issue, state, pr.number)
        if pr is not None:
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
