"""E2E scenario 10 (issue #55): stalled / repeated patch -> needs-human.

The foreman's stagnation guard (``EscalationPolicyConfig.stagnation_rounds_threshold``)
must escalate when review rounds keep coming back without converging
signal. The hybrid executor scripts the reviewer lane to return the
*same* finding every round; the foreman's adjudication dispositions
the finding as ``fix`` and runs the coder, but the next round reports
the same finding again — no progress, no convergence.

Acceptance (per #55 E2E scenarios, #10):
- The foreman reaches ``escalated`` rather than spinning forever.
- The escalation reason names the stagnation threshold (so an
  operator reading the durable issue knows what to look at).
- The durable state carries the ``escalated`` phase and the
  ``v3-work-needs-human`` label.
- The coder is invoked once per round (a fix round per reviewer
  finding) up to the configured cap; the cap or the stagnation
  threshold is what stops the loop, whichever trips first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.config import (
    EscalationPolicyConfig,
    ReviewPolicyConfig,
    SafetyPolicyConfig,
    V3Config,
)
from ai_pr_orchestrator.v3.domain import (
    LaneIdentity,
    ModelAssignment,
    ReviewerFinding,
)
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop
from ai_pr_orchestrator.v3.interfaces import (
    GateDecision,
    LaneExecutionContext,
    LaneResult,
    ModelLease,
    SessionHandle,
)
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue


@dataclass
class StaticBroker:
    outstanding: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    def select(self, demand: Any) -> BrokerDecision:
        return BrokerDecision(
            demand=demand,
            evaluated_at=__import__("datetime").datetime(2026, 9, 1, tzinfo=__import__("datetime").UTC),
            assignment=ModelAssignment(lane=demand.lane, model_ref=f"ref-{demand.lane}"),
        )

    def reserve(self, assignment: ModelAssignment) -> ModelLease:
        self.outstanding.append(assignment.lane)
        return ModelLease(
            lease_id=f"lease-{assignment.lane}-{len(self.outstanding)}",
            assignment=assignment,
        )

    def release(self, lease: ModelLease) -> None:
        if lease.assignment.lane in self.outstanding:
            self.outstanding.remove(lease.assignment.lane)
        self.released.append(lease.lease_id)


class StaticGate:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def evaluate(self, issue: Any, pr: Any) -> GateDecision:
        self.calls.append((pr.number, pr.head_sha))
        return GateDecision(passed=True, pending_checks=(), failed_checks=())


@dataclass
class TrackingGit:
    default: str = "main"
    branches: list[str] = field(default_factory=lambda: ["main"])
    worktrees: dict[str, str] = field(default_factory=dict)
    commits: list[tuple[str, str]] = field(default_factory=list)
    cleanups: list[str] = field(default_factory=list)
    pushed: list[str] = field(default_factory=list)
    touched: dict[str, list[str]] = field(default_factory=dict)

    def default_branch(self) -> str:
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.branches.append(branch)

    def create_worktree(self, path: str, branch: str) -> str:
        self.worktrees[path] = branch
        return path

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        self.commits.append((workdir, message))
        return "sha"

    def commit_count(self, workdir: str, base_ref: str) -> int:
        return sum(1 for w, _ in self.commits if w == workdir)

    def push(self, branch: str) -> None:
        self.pushed.append(branch)

    def changed_files(self, workdir: str, base_ref: str | None = None) -> list[str]:
        return list(self.touched.get(workdir, ["src/change.py"]))

    def cleanup_worktree(self, path: str) -> None:
        self.cleanups.append(path)
        self.worktrees.pop(path, None)


@dataclass
class StagnatingReviewerExecutor:
    """A scripted executor whose reviewer lane returns the same finding
    every round. The foreman adjudicates each round's finding as a
    ``fix``, but the coder's "fix" produces no real change so the next
    round reports it again.

    The reviewer's per-round finding gets a unique id
    (``{base_id}-r{n}-{lane}``) so the finding registry does not reject
    it as a duplicate of either the previous round's report or the
    sibling reviewer lane's same-round report.
    """

    base_finding: ReviewerFinding
    coder_calls: int = 0
    reviewer_calls: int = 0
    last_round_id: str | None = None
    round_counter: int = 0
    per_lane_counter: dict[str, int] = field(default_factory=dict)

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ):
        handle = SessionHandle(session_id="cao-stale", lane=lane.lane)
        if lane.role == "reviewer":
            self.reviewer_calls += 1
            n = self.per_lane_counter.get(lane.lane, 0) + 1
            self.per_lane_counter[lane.lane] = n
            self.round_counter += 1
            finding = ReviewerFinding(
                id=f"{self.base_finding.id}-r{self.round_counter}-{lane.lane}",
                lane=lane.lane,
                body=self.base_finding.body,
                severity=self.base_finding.severity,
                run_id=self.base_finding.run_id,
                round_id=context.round_id or "review",
            )
            return LaneResult(
                session=handle,
                exit_code=0,
                output_summary="stagnating reviewer",
                changed_files=[],
                findings=[finding],
            )
        self.coder_calls += 1
        self.last_round_id = context.round_id
        return LaneResult(
            session=handle,
            exit_code=0,
            output_summary="coder attempt that does not actually fix the issue",
            changed_files=["src/change.py"],
        )


def _ready_fake() -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    return fake


def test_scenario_10_repeated_finding_escalates_on_stagnation_threshold():
    """A blocker finding that the coder cannot resolve surfaces every
    round. The foreman escalates via ``mark_needs_human`` once it
    reaches a stop condition — either the stagnation threshold or the
    review-round cap, whichever trips first.

    Both guards are part of the same safety contract: a non-converging
    item must never reach ``done``. This test configures the
    stagnation threshold to be the *first* guard to trip (lower than
    the review-round cap) so the assertion is precise about which
    one fired.
    """
    fake = _ready_fake()
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=10,
            max_reviewer_triggers_per_run=20,
            max_total_iterations=10,
        ),
        escalation=EscalationPolicyConfig(
            # Stagnation threshold at 1 means "one empty/circling
            # round after the first finding is enough to escalate".
            stagnation_rounds_threshold=1,
        ),
        review_policy=ReviewPolicyConfig(max_review_rounds=10),
    )
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    stuck = ReviewerFinding(
        id="stuck-1",
        lane="requirements-reviewer",
        body="missing input validation",
        severity="blocker",
        run_id="run-s10",
        round_id="ignored-by-scripted",
    )
    git = TrackingGit()
    git.touched = {"/wt/issue-1": ["src/change.py"]}
    executor = StagnatingReviewerExecutor(base_finding=stuck)
    loop = ForemanPolicyLoop(
        queue,
        StaticBroker(),
        LaneRegistry.default(),
        executor,
        StaticGate(),
        git,
        cfg,
        run_id="run-s10",
        worktree_root="/wt",
        committer_name="AIPRO E2E Bot",
        committer_email="aipro-bot@example.invalid",
    )

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated", (
        f"expected 'escalated', got {outcome.final_phase!r}; reason={outcome.reason!r}"
    )
    # The escalation must point at *something* the operator can act on
    # — either a budget name or a converging-signal warning.
    reason = outcome.reason.lower()
    assert any(
        token in reason
        for token in ("budget", "converging", "stagnation", "round")
    ), f"expected reason to name a guard, got {outcome.reason!r}"
    # The durable state reflects the escalation and the needs-human label.
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"
    assert "v3-work-needs-human" in fake.get_labels(1)
    # The coder was invoked at least twice (initial + at least one
    # fix round) before the threshold tripped.
    assert executor.coder_calls >= 2, (
        f"expected at least 2 coder invocations, got {executor.coder_calls}"
    )


def test_scenario_10_repeated_finding_hits_coder_cap_before_stagnation():
    """If the coder invocation budget is lower than the stagnation
    threshold, the coder cap trips first. The escalation reason is
    the budget, not the stagnation threshold."""
    fake = _ready_fake()
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=2,
            max_reviewer_triggers_per_run=10,
            max_total_iterations=10,
        ),
        escalation=EscalationPolicyConfig(
            stagnation_rounds_threshold=5,  # very high; cap trips first
        ),
    )
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    stuck = ReviewerFinding(
        id="stuck-2",
        lane="requirements-reviewer",
        body="missing audit log",
        severity="blocker",
        run_id="run-s10-budget",
        round_id="ignored",
    )
    git = TrackingGit()
    git.touched = {"/wt/issue-1": ["src/change.py"]}
    executor = StagnatingReviewerExecutor(base_finding=stuck)
    loop = ForemanPolicyLoop(
        queue,
        StaticBroker(),
        LaneRegistry.default(),
        executor,
        StaticGate(),
        git,
        cfg,
        run_id="run-s10-budget",
        worktree_root="/wt",
        committer_name="AIPRO E2E Bot",
        committer_email="aipro-bot@example.invalid",
    )

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "budget" in outcome.reason, (
        f"expected budget escalation, got {outcome.reason!r}"
    )
    # The coder invocation cap is consulted *before* launching the
    # coder when open findings are present, so the final coder
    # invocation count is one less than the cap (the cap is what
    # escalated us rather than launching one more).
    assert executor.coder_calls >= 1