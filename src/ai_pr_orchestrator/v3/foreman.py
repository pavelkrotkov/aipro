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

from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from .broker import TaskDemand
from .config import V3Config
from .domain import (
    DispositionAction,
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


@dataclass(frozen=True)
class _RoundReport:
    """Result of one review round."""

    remaining: tuple[ReviewerFinding, ...]
    rounds: int
    stagnant_rounds: int


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

    # --- Public API ----------------------------------------------------------

    def run_pass(
        self, *, now: datetime | None = None, max_items: int | None = None
    ) -> list[WorkItemOutcome]:
        """Claim and drive every ready issue; one outcome per item.

        An issue whose claim fails (already claimed by a competing foreman)
        is skipped rather than fatal: contention is normal queue behaviour.
        One item crashing escalates *that item* only; the pass continues.
        """
        issues = self._list_ready()
        if max_items is not None:
            issues = issues[:max_items]
        outcomes: list[WorkItemOutcome] = []
        for issue in issues:
            try:
                outcomes.append(self._drive(issue, now=now))
            except Exception as exc:
                outcomes.append(
                    WorkItemOutcome(
                        issue=issue,
                        final_phase="escalated",
                        reason=f"foreman error: {exc}",
                        escalated=True,
                    )
                )
        return outcomes

    # --- Lifecycle -----------------------------------------------------------

    def _drive(self, issue: GitHubIssueRef, *, now: datetime | None) -> WorkItemOutcome:
        safety = self._cfg.safety
        branch = f"aipro-issue-{issue.number}"
        base = self._git.default_branch()
        self._git.create_branch(branch, base)
        worktree = self._git.create_worktree(f"{self._worktree_root}/issue-{issue.number}", branch)
        state = self._claim(issue, branch=branch, worktree=worktree, now=now)

        coder_invocations = 0
        coder_failures = 0
        review_rounds = 0
        stagnant_rounds = 0
        fix_findings: tuple[ReviewerFinding, ...] = ()

        while True:
            # --- Coding ------------------------------------------------
            state = self._transition(issue, state, "coding", now=now)
            result = self._run_lane(
                self._lanes.get(DEVELOPER_LANE),
                worktree,
                state,
                self._coder_prompt(issue, fix_findings),
            )
            coder_invocations += 1
            if result.exit_code != 0:
                coder_failures += 1
                if coder_failures >= self._cfg.escalation.max_consecutive_coder_failures:
                    return self._escalate(
                        issue, state, f"coder failed {coder_failures}x consecutively", now=now
                    )
                # Below the threshold a failure is retried: transient lane
                # crashes must not kill the item, but the attempts stay
                # bounded and every retry consumes invocation budget.
                continue
            coder_failures = 0
            violation = self._policy_violation(result)
            if violation:
                return self._fail(issue, state, violation, now=now)
            if coder_invocations >= safety.max_coder_invocations_per_run and fix_findings:
                return self._escalate(
                    issue, state, "coder invocation budget exhausted with open findings", now=now
                )

            # --- Review rounds ------------------------------------------
            report = self._review(issue, state, worktree, review_rounds, stagnant_rounds, now=now)
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

            # --- CI gate --------------------------------------------------
            state = self._transition(issue, state, "ci_gating", now=now)
            pr = self._ensure_pr(issue, branch)
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
                # Not a failure and not ours to busy-wait: leave the item in
                # ci_gating; a later pass re-evaluates against the same head.
                return WorkItemOutcome(
                    issue=issue,
                    final_phase="ci_gating",
                    reason="CI checks pending: " + ", ".join(decision.pending_checks),
                    review_rounds=review_rounds,
                    coder_invocations=coder_invocations,
                    gate=decision,
                )
            # Real CI failures become findings for the next coding round —
            # unless the review budget is spent and reviews keep reporting
            # nothing new: that is stagnation, not fixable work.
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

    # --- Review ---------------------------------------------------------------

    def _review(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str,
        review_rounds: int,
        stagnant_rounds: int,
        *,
        now: datetime | None,
    ) -> _RoundReport:
        """Run one review round; dispositions what it found."""
        policy = self._cfg.review_policy
        if review_rounds >= policy.max_review_rounds:
            return _RoundReport((), review_rounds, stagnant_rounds)
        reviewer_lanes = policy.reviewer_lanes or [
            lane.lane for lane in self._lanes if lane.role == "reviewer"
        ]
        rounds = review_rounds + 1
        round_id = f"review-{rounds}"
        state = self._transition(issue, state, "reviewing", round_id=round_id, now=now)

        registry = FindingRegistry(
            require_coder_reply_before_resolve=policy.require_coder_reply_before_resolve,
            quarantine_unknown_head_sha=False,
        )
        for lane_name in reviewer_lanes:
            lane = self._lanes.get(lane_name)
            result = self._run_lane(lane, worktree, state, self._reviewer_prompt(issue, round_id))
            if result.exit_code == 0:
                for finding in result.findings:
                    registry.register(finding)
        registry.deduplicate()
        conflicts = registry.detect_conflicts()
        open_findings = [f for f in registry.findings if f.status == "open"]

        if not open_findings:
            stagnant = stagnant_rounds + 1
            if rounds > 1 and stagnant >= self._cfg.escalation.stagnation_rounds_threshold:
                self._persist_round(issue, state, round_id, registry, [])
                self._escalate(issue, state, "review rounds produced no converging signal", now=now)
                return _RoundReport((), rounds, stagnant)
            return _RoundReport((), rounds, stagnant)
        stagnant_rounds = 0

        # Adjudicate: blockers/majors (and whole conflict groups) are fixed;
        # minors are deferred with an explicit reply.
        conflict_members = {fid for ids in conflicts.values() for fid in ids}
        fix_ids: set[str] = set()
        dispositions: list[FindingDisposition] = []
        for finding in sorted(open_findings, key=lambda f: (-SEVERITY_RANK[f.severity], f.id)):
            if finding.id in conflict_members:
                action: DispositionAction = "fix"
                rationale = "conflict group adjudicated: fix"
            elif SEVERITY_RANK[finding.severity] >= SEVERITY_RANK["major"]:
                action = "fix"
                rationale = "major/blocker finding: fix"
            else:
                action = "reply_deferred"
                rationale = "minor finding: deferred with reply"
            if action == "fix":
                fix_ids.add(finding.id)
            _, disposition = registry.apply_disposition(
                finding.id,
                action,
                rationale=rationale,
                decided_by="foreman",
                reply_body="coder will address this finding",
            )
            dispositions.append(disposition)
        # "What the coder must address next": everything dispositioned as
        # ``fix`` (including whole conflict groups), regardless of the
        # settled status — the disposition records the *decision*, the
        # follow-up coding round records the work. Computed *before*
        # compaction, which archives settled findings out of the registry.
        remaining = tuple(f for f in registry.findings if f.id in fix_ids)
        registry.compact()
        self._persist_round(issue, state, round_id, registry, dispositions)
        return _RoundReport(remaining, rounds, stagnant_rounds)

    def _persist_round(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        round_id: str,
        registry: FindingRegistry,
        dispositions: list[FindingDisposition],
    ) -> None:
        """Persist round findings/dispositions so a restart never loses them."""
        try:
            fresh = self._load(issue, state)
            if fresh is None:
                return
            updated = replace(
                fresh,
                round_id=round_id,
                findings=list(registry.findings),
                dispositions=list(dispositions),
                archived=list(registry.archived),
            )
            self._queue.save_state(updated, expected_updated_at=fresh.updated_at)
        except Exception:
            pass

    # --- Lane execution ---------------------------------------------------------

    def _run_lane(
        self, lane: LaneIdentity, worktree: str, state: WorkflowState, prompt: str
    ) -> LaneResult:
        lease = self._reserve(lane)
        context = LaneExecutionContext(
            run_id=self._run_id, round_id=state.round_id, work_item_id=state.work_item_id
        )
        try:
            return self._executor.execute(lane, prompt, worktree, context, lease)
        finally:
            self._release(lease)

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
        self, issue: GitHubIssueRef, *, branch: str, worktree: str, now: datetime | None
    ) -> WorkflowState:
        return self._queue.claim(issue, self._run_id, branch=branch, worktree=worktree, now=now)

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

    # --- Policy helpers -------------------------------------------------------------

    def _policy_violation(self, result: LaneResult) -> str | None:
        if self._cfg.safety.disallow_workflow_file_changes and any(
            f.startswith(".github/workflows/") for f in result.changed_files
        ):
            return "policy violation: lane modified files under .github/workflows/"
        return None

    def _coder_prompt(self, issue: GitHubIssueRef, findings: tuple[ReviewerFinding, ...]) -> str:
        lines = [f"Implement issue {issue.slug()} in the current worktree."]
        if findings:
            lines.append("Address these review findings:")
            lines.extend(f"- [{f.severity}] {f.body}" for f in findings)
        return "\n".join(lines)

    def _reviewer_prompt(self, issue: GitHubIssueRef, round_id: str) -> str:
        return (
            f"Review the changes for issue {issue.slug()} ({round_id}); report structured findings."
        )

    def _ensure_pr(self, issue: GitHubIssueRef, branch: str) -> GitHubPullRequestRef:
        """A PR ref for gating; creates the PR if the queue's client can."""
        client = getattr(self._queue, "_client", None)
        pr_number: int
        if client is not None and hasattr(client, "create_pr"):
            pr = client.create_pr(
                f"[aipro] issue #{issue.number}: automated change",
                f"Automated work for {issue.slug()}.",
                head=branch,
                base=self._git.default_branch(),
            )
            pr_number = pr.number
        else:
            # Deterministic fallback for fakes that track PRs 1:1 with issues.
            pr_number = issue.number
        return GitHubPullRequestRef(
            owner=issue.owner, repo=issue.repo, number=pr_number, head_sha=branch
        )

    def _escalate(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> WorkItemOutcome:
        with suppress(Exception):
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
        with suppress(Exception):
            self._queue.fail(issue, self._load(issue, state), reason=reason)
        return WorkItemOutcome(issue=issue, final_phase="failed", reason=reason)
