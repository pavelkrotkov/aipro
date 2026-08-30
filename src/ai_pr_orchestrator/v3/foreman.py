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
    TERMINAL_PHASES,
    ArchivedFinding,
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
        self._prompt_tokens = 0
        self._head_sha: str = ""

    # --- Public API ----------------------------------------------------------

    def run_pass(
        self, *, now: datetime | None = None, max_items: int | None = None
    ) -> list[WorkItemOutcome]:
        """Claim and drive every ready issue; one outcome per item.

        An issue whose claim fails (already claimed by a competing foreman)
        is skipped rather than fatal: contention is normal queue behaviour.
        One item crashing escalates *that item* only — and persists the crash
        (``mark_needs_human``) so the authoritative issue is never left on an
        active phase with a stranded claim. The pass continues.
        """
        issues = self._list_ready()
        if max_items is not None:
            issues = issues[:max_items]
        outcomes: list[WorkItemOutcome] = []
        for issue in issues:
            try:
                outcomes.append(self._drive(issue, now=now))
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
        return outcomes

    def _persist_crash(self, issue: GitHubIssueRef, reason: str) -> None:
        """Persist an in-pass crash against the authoritative issue.

        Best-effort: if the issue was claimed *by this run* (state exists, is
        non-terminal, and carries our run id) the crash is recorded as
        ``needs-human`` so the item is recovered. A competing foreman's claim
        (contention, or a stale pre-run state) is left alone — it is not ours
        to escalate.
        """
        load = getattr(self._queue, "load_state", None)
        mark = getattr(self._queue, "mark_needs_human", None)
        if load is None or mark is None:
            return
        try:
            state = load(issue.slug())
        except Exception:
            return
        if state is None or state.phase in TERMINAL_PHASES:
            return
        if state.run_id != self._run_id:
            # Another foreman owns this claim (or it is a stale pre-run state);
            # escalating it would mis-mark someone else's live work.
            return
        with suppress(Exception):
            mark(issue, self._load(issue, state), reason=reason)

    # --- Lifecycle -----------------------------------------------------------

    def _drive(self, issue: GitHubIssueRef, *, now: datetime | None) -> WorkItemOutcome:
        branch = f"aipro-issue-{issue.number}"

        # Claim FIRST — resources are created only once the claim is won, so a
        # lost claim (contention) never leaks a branch/worktree behind it.
        existing = self._load_optional(issue)
        if existing is not None:
            branch = existing.extras.get("branch") or branch
        state = self._claim(
            issue,
            branch=branch,
            worktree=existing.extras.get("worktree") if existing else None,
            pr_number=existing.extras.get("pr_number") if existing else None,
            now=now,
        )

        worktree: str | None = state.extras.get("worktree")
        outcome: WorkItemOutcome | None = None
        try:
            if not worktree:
                base = self._git.default_branch()
                self._git.create_branch(branch, base)
                worktree = self._git.create_worktree(
                    f"{self._worktree_root}/issue-{issue.number}", branch
                )
                state = self._persist_resources(issue, state, branch=branch, worktree=worktree)
            outcome = self._run_loop(issue, state, worktree, branch, now=now)
        except _ForemanEscalation as exc:
            outcome = self._escalate(issue, state, exc.reason, now=now)
        finally:
            # Terminal outcomes release the worktree; the pending-CI/requeue
            # path deliberately retains it so a later pass reuses the checkout.
            if worktree and outcome is not None and outcome.final_phase in TERMINAL_PHASES:
                with suppress(Exception):
                    self._git.cleanup_worktree(worktree)
        assert outcome is not None
        return outcome

    def _run_loop(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        worktree: str,
        branch: str,
        *,
        now: datetime | None,
    ) -> WorkItemOutcome:
        safety = self._cfg.safety
        coder_invocations = 0
        coder_failures = 0
        review_rounds = 0
        stagnant_rounds = 0
        reviewer_triggers = 0
        self._prompt_tokens = 0
        fix_findings: tuple[ReviewerFinding, ...] = ()

        while True:
            # --- Coding ------------------------------------------------------
            state = self._transition(issue, state, "coding", now=now)
            state = self._heartbeat(issue, state, now=now)
            result = self._run_lane(
                self._worker_lane(),
                worktree,
                state,
                self._coder_prompt(issue, fix_findings),
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
            violation = self._policy_violation(result)
            if violation:
                return self._fail(issue, state, violation, now=now)
            if coder_invocations >= safety.max_coder_invocations_per_run and fix_findings:
                return self._escalate(
                    issue,
                    state,
                    "coder invocation budget exhausted with open findings",
                    now=now,
                )

            # --- Review rounds ----------------------------------------------
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
            sha = self._commit_and_push(issue, state, worktree, branch)
            self._head_sha = sha
            pr = self._ensure_pr(issue, state, branch, worktree)
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
                # worktree/branch are retained for that reuse.
                state = self._transition(issue, state, "queued", now=now)
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
        if review_rounds >= policy.max_review_rounds:
            return _RoundReport((), review_rounds, stagnant_rounds, reviewer_triggers)
        reviewer_lanes = policy.reviewer_lanes or [
            lane.lane for lane in self._lanes if lane.role == "reviewer"
        ]
        budget_left = max(0, self._cfg.safety.max_reviewer_triggers_per_run - reviewer_triggers)
        active_lanes = list(reviewer_lanes[:budget_left])
        rounds = review_rounds + 1
        round_id = f"review-{rounds}"
        state = self._transition(issue, state, "reviewing", round_id=round_id, now=now)
        state = self._heartbeat(issue, state, now=now)

        registry = FindingRegistry(
            require_coder_reply_before_resolve=policy.require_coder_reply_before_resolve,
            quarantine_unknown_head_sha=False,
        )
        triggers = reviewer_triggers
        for lane_name in active_lanes:
            lane = self._lanes.get(lane_name)
            result = self._run_lane(lane, worktree, state, self._reviewer_prompt(issue, round_id))
            triggers += 1
            if result.exit_code != 0:
                raise _ForemanEscalation(
                    f"reviewer lane {lane_name!r} failed (exit {result.exit_code})"
                )
            for finding in result.findings:
                registry.register(finding)
        registry.deduplicate()
        conflicts = registry.detect_conflicts()
        open_findings = [f for f in registry.findings if f.status == "open"]

        if not open_findings:
            stagnant = stagnant_rounds + 1
            if rounds > 1 and stagnant >= self._cfg.escalation.stagnation_rounds_threshold:
                # Persist the empty round so restart reconciliation sees it,
                # then escalate *through* the loop (never as a plain empty
                # report, which _drive would resume from as if unscathed).
                self._persist_round(issue, state, round_id, registry, [])
                raise _ForemanEscalation("review rounds produced no converging signal")
            return _RoundReport((), rounds, stagnant, triggers)
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
        return _RoundReport(remaining, rounds, stagnant_rounds, triggers)

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
        this round's are appended (newest disposition wins per finding).

        Persistence is restart-safety-critical: any failure to save raises
        ``_ForemanEscalation`` so the pass aborts and escalates instead of
        silently dropping authoritative state.
        """
        fresh = self._load(issue, state)
        if fresh is None:
            raise _ForemanEscalation(
                f"cannot persist review round {round_id}: no authoritative state for {issue.slug()}"
            )
        merged_findings = self._merge_round_findings(fresh.findings, registry.findings)
        merged_dispositions = self._merge_dispositions(fresh.dispositions, dispositions)
        merged_archived = self._merge_archived(fresh.archived, registry.archived)
        updated = replace(
            fresh,
            round_id=round_id,
            findings=merged_findings,
            dispositions=merged_dispositions,
            archived=merged_archived,
        )
        try:
            self._queue.save_state(updated, expected_updated_at=fresh.updated_at)
        except Exception as exc:
            raise _ForemanEscalation(
                f"persist review round {round_id} for {issue.slug()} failed: {exc}"
            ) from exc

    @staticmethod
    def _merge_round_findings(
        existing: list[ReviewerFinding], current: list[ReviewerFinding]
    ) -> list[ReviewerFinding]:
        seen = {f.id for f in existing}
        merged = list(existing)
        for finding in current:
            if finding.id not in seen:
                merged.append(finding)
                seen.add(finding.id)
        return merged

    @staticmethod
    def _merge_dispositions(
        existing: list[FindingDisposition],
        current: list[FindingDisposition],
    ) -> list[FindingDisposition]:
        # Newest disposition per finding wins, on top of the accumulated
        # prior rounds. A round's disposition always settles a finding that is
        # in this round's registry, so the merge is keyed on finding_id.
        dealt: set[str] = set()
        for d in current:
            dealt.add(d.finding_id)
        merged = [d for d in existing if d.finding_id not in dealt]
        merged.extend(current)
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
        self, lane: LaneIdentity, worktree: str, state: WorkflowState, prompt: str
    ) -> LaneResult:
        lease = self._reserve(lane)
        context = LaneExecutionContext(
            run_id=self._run_id, round_id=state.round_id, work_item_id=state.work_item_id
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

    def _load_optional(self, issue: GitHubIssueRef) -> WorkflowState | None:
        load = getattr(self._queue, "load_state", None)
        if load is None:
            return None
        try:
            return load(issue.slug())
        except Exception:
            return None

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
        updated = replace(fresh, extras=extras)
        self._queue.save_state(updated, expected_updated_at=fresh.updated_at)
        return updated

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

    def _coder_prompt(self, issue: GitHubIssueRef, findings: tuple[ReviewerFinding, ...]) -> str:
        lines = [f"Implement issue {issue.slug()} in the current worktree."]
        description = self._read_issue_description(issue)
        if description:
            lines.append(f"Issue description:\n{description}")
        if findings:
            lines.append("Address these review findings:")
            lines.extend(f"- [{f.severity}] {f.body}" for f in findings)
        return "\n".join(lines)

    def _read_issue_description(self, issue: GitHubIssueRef) -> str:
        """The issue body (description + any acceptance criteria) for the coder."""
        client = getattr(self._queue, "_client", None)
        get_body = getattr(client, "get_issue_body", None)
        if get_body is None:
            return ""
        try:
            return get_body(issue.number) or ""
        except Exception:
            return ""

    def _reviewer_prompt(self, issue: GitHubIssueRef, round_id: str) -> str:
        return (
            f"Review the changes for issue {issue.slug()} ({round_id}); report structured findings."
        )

    def _commit_and_push(
        self, issue: GitHubIssueRef, state: WorkflowState, worktree: str, branch: str
    ) -> str:
        """Commit lane output and push it to the remote branch.

        Called immediately before the PR is (re)opened so the PR targets a
        committed, pushed head — never an uncommitted/local branch. A lane
        that made no edits commits nothing (a no-op returning the current
        HEAD) and pushes the branch so it exists remotely.
        """
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
        worktree: str,
    ) -> GitHubPullRequestRef:
        """The PR ref for gating; create the PR if the queue's client can.

        Reuses an already-recorded PR number (F7): a second ``create_pr`` for
        the same branch is rejected by GitHub, so every later ci_gating visit
        must gate against the *same* open PR. ``head_sha`` is the PR's commit
        SHA from the client (F19), refreshed from the live PR on reuse, never
        the branch name.
        """
        client = getattr(self._queue, "_client", None)
        pr_number = state.extras.get("pr_number")
        if pr_number is not None:
            live = self._live_pr_head(client, int(pr_number))
            if live is not None:
                return GitHubPullRequestRef(
                    owner=issue.owner,
                    repo=issue.repo,
                    number=int(pr_number),
                    head_sha=live,
                )
        if client is not None and hasattr(client, "create_pr"):
            pr = client.create_pr(
                f"[aipro] issue #{issue.number}: automated change",
                f"Automated work for {issue.slug()}.",
                head=branch,
                base=self._git.default_branch(),
            )
            self._record_pr(issue, state, pr.number)
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
            head_sha=self._head_sha or f"head-{number}",
        )

    def _live_pr_head(self, client, pr_number: int) -> str | None:
        """The live PR's commit SHA (F19): refresh when the head moves."""
        get_pr = getattr(client, "get_pr", None)
        if get_pr is None:
            return None
        try:
            return get_pr(pr_number).head_sha
        except Exception:
            return None

    def _record_pr(self, issue: GitHubIssueRef, state: WorkflowState, pr_number: int) -> None:
        fresh = self._load(issue, state)
        extras = dict(fresh.extras)
        extras["pr_number"] = pr_number
        self._queue.save_state(replace(fresh, extras=extras), expected_updated_at=fresh.updated_at)

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
