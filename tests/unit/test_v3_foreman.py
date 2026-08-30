"""Tests for the V3 foreman policy loop (issue #55): a full fake lifecycle.

The foreman runs against the *real* ``GitHubIssueQueue`` over the fake GitHub
client, so claim/transition/label semantics are production semantics; only
the lane executor, broker, gate, and git ops are faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, cast

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.config import EscalationPolicyConfig, SafetyPolicyConfig, V3Config
from ai_pr_orchestrator.v3.domain import (
    GitHubIssueRef,
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

ISSUE = GitHubIssueRef(owner="owner", repo="repo", number=1)
NOW = datetime(2026, 8, 29, tzinfo=UTC)
HANDLE = SessionHandle(session_id="s", lane="l")


# --- Fakes --------------------------------------------------------------------


class FakeBroker:
    def __init__(self) -> None:
        self.outstanding: list[str] = []
        self.released: list[str] = []

    def select(self, demand) -> BrokerDecision:
        return BrokerDecision(
            demand=demand,
            evaluated_at=NOW,
            assignment=ModelAssignment(lane=demand.lane, model_ref=f"ref-{demand.lane}"),
        )

    def reserve(self, assignment: ModelAssignment) -> ModelLease:
        self.outstanding.append(assignment.lane)
        return ModelLease(
            lease_id=f"lease-{assignment.lane}-{len(self.outstanding)}", assignment=assignment
        )

    def release(self, lease: ModelLease) -> None:
        self.outstanding.remove(lease.assignment.lane)
        self.released.append(lease.lease_id)


@dataclass
class ScriptedExecutor:
    """Developer succeeds; reviewer findings are scripted per round."""

    reviewer_findings_by_round: dict[int, list[ReviewerFinding]] = field(default_factory=dict)
    developer_exit: int = 0
    developer_files: list[str] = field(default_factory=lambda: ["src/x.py"])
    calls: list[tuple[str, str]] = field(default_factory=list)
    round_counter: dict[str, int] = field(default_factory=dict)

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ) -> LaneResult:
        self.calls.append((lane.lane, workdir))
        if lane.role == "reviewer":
            n = self.round_counter.get(lane.lane, 0) + 1
            self.round_counter[lane.lane] = n
            scripted = self.reviewer_findings_by_round.get(n, [])
            # Each reviewer lane reports its own copy with a lane-unique id,
            # mirroring independent reviewers; dedup merges identical claims.
            findings = [replace(f, id=f"{f.id}-{lane.lane}", lane=lane.lane) for f in scripted]
            return LaneResult(
                session=HANDLE,
                exit_code=0,
                output_summary="",
                changed_files=[],
                findings=list(findings),
            )
        return LaneResult(
            session=HANDLE,
            exit_code=self.developer_exit,
            output_summary="",
            changed_files=list(self.developer_files),
        )


class StaticGate:
    def __init__(self, decision: GateDecision) -> None:
        self.decision = decision
        self.evaluated: list[str] = []

    def evaluate(self, issue, pr) -> GateDecision:
        self.evaluated.append(pr.number)
        return self.decision


class FakeGitOperations:
    """In-memory GitOperations: records branches/worktrees, no subprocess."""

    def __init__(self, default: str = "main") -> None:
        self.default = default
        self.branches = [default]
        self.worktrees: dict[str, str] = {}

    def default_branch(self) -> str:
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.branches.append(branch)

    def create_worktree(self, path: str, branch: str) -> str:
        self.worktrees[path] = branch
        return path

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        return "sha"

    def commit_count(self, workdir: str, base_ref: str) -> int:
        return 0

    def push(self, branch: str) -> None:
        pass

    def cleanup_worktree(self, path: str) -> None:
        self.worktrees.pop(path, None)


def _finding(idx: int, severity: Any = "major") -> ReviewerFinding:
    return ReviewerFinding(
        id=f"f-{idx}",
        lane="requirements-reviewer",
        body=f"finding {idx}",
        severity=cast(Any, severity),
        run_id="run-1",
        round_id="review-1",
    )


def _gate(decision: GateDecision | None = None) -> StaticGate:
    return StaticGate(decision or GateDecision(passed=True, pending_checks=(), failed_checks=()))


def _foreman(
    fake: FakeGitHubClient,
    executor: ScriptedExecutor,
    gate: StaticGate,
    config: V3Config | None = None,
):
    cfg = config or V3Config()
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    loop = ForemanPolicyLoop(
        queue,
        FakeBroker(),
        LaneRegistry.default(),
        executor,
        gate,
        FakeGitOperations(),
        cfg,
        run_id="run-1",
        worktree_root="/wt",
        committer_name="Pavel Krotkov",
        committer_email="pavel.krotkov@gmail.com",
    )
    return loop, queue


def _ready_fake() -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    return fake


# --- Tests ---------------------------------------------------------------------


def test_clean_lifecycle_claim_to_done():
    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = _gate()
    loop, queue = _foreman(fake, executor, gate)
    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "done"
    assert outcome.gate is not None and outcome.gate.passed
    assert outcome.coder_invocations == 1 and outcome.review_rounds == 1
    # authoritative state persisted
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "done"
    assert state.terminal_reason == "ci green"
    # labels moved through the lifecycle
    assert "v3-work-done" in fake.get_labels(1)
    assert "v3-work" not in fake.get_labels(1)
    # git ops were used for branch + worktree
    assert ("developer", "/wt/issue-1") in executor.calls


def test_finding_triggers_fix_round_then_done():
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(1)]})
    config = V3Config(safety=SafetyPolicyConfig(max_coder_invocations_per_run=3))
    loop, queue = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "done"
    assert outcome.coder_invocations == 2  # fix round happened
    assert outcome.review_rounds == 2
    state = queue.load_state("owner/repo#1")
    assert state is not None
    # the finding was dispositioned (fix → accepted), not silently dropped
    assert any(d.action == "fix" for d in state.dispositions)
    assert state.dispositions[0].reply_body  # coder reply recorded


def test_minor_findings_are_deferred_not_fixed():
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(2, severity="minor")]})
    loop, _ = _foreman(fake, executor, _gate())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    # a deferred minor finding did not trigger an extra coding round
    assert outcome.coder_invocations == 1


def test_workflow_file_change_is_a_policy_violation():
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_files=[".github/workflows/ci.yml"])
    loop, queue = _foreman(fake, executor, _gate())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "failed"
    assert "policy violation" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "failed"


def test_coder_invocation_budget_escalates():
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(3)]})
    config = V3Config(safety=SafetyPolicyConfig(max_coder_invocations_per_run=1))
    loop, queue = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "budget" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"
    assert "v3-work-needs-human" in fake.get_labels(1)


def test_coder_exit_code_retries_then_escalates():
    """A failing lane retries until the consecutive-failure threshold."""
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_exit=3)
    loop, queue = _foreman(fake, executor, _gate())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "consecutively" in outcome.reason
    dev_calls = [c for c in executor.calls if c[0] == "developer"]
    assert len(dev_calls) == 3  # default threshold, each attempt retried
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


def test_pending_ci_parks_item_in_ci_gating():
    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=()))
    loop, queue = _foreman(fake, executor, gate)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "ci_gating"
    assert "pending" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "ci_gating"


def test_leases_are_released_even_when_lanes_fail():
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_exit=1)
    broker = FakeBroker()
    queue = GitHubIssueQueue(fake, "owner", "repo", host_id="host-A")
    loop = ForemanPolicyLoop(
        queue,
        broker,
        LaneRegistry.default(),
        executor,
        _gate(),
        FakeGitOperations(),
        V3Config(),
        run_id="run-1",
        worktree_root="/wt",
        committer_name="N",
        committer_email="e@x",
    )
    loop.run_pass()
    assert broker.outstanding == []  # every lease released despite the failure
    assert broker.released


def test_max_items_limits_claims():
    fake = _ready_fake()
    fake.seed_issue(2, labels=["v3-work"])
    executor = ScriptedExecutor()
    loop, _ = _foreman(fake, executor, _gate())
    outcomes = loop.run_pass(max_items=1)
    assert len(outcomes) == 1
    assert len(executor.calls) >= 1


def test_stagnation_threshold_escalates():
    """Reviewer keeps returning nothing while work never converges."""
    from ai_pr_orchestrator.v3.config import SafetyPolicyConfig
    from ai_pr_orchestrator.v3.config import V3Config as Cfg

    fake = _ready_fake()
    # Round 1: findings exist (so we enter a fix round); rounds 2+ empty.
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(9)], 2: [], 3: []})
    config = Cfg(
        safety=SafetyPolicyConfig(max_coder_invocations_per_run=20, max_total_iterations=9),
        escalation=EscalationPolicyConfig(stagnation_rounds_threshold=2),
    )
    gate = _gate(GateDecision(passed=False, pending_checks=(), failed_checks=("build",)))
    loop, queue = _foreman(fake, executor, gate, config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "converging" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


def test_consecutive_coder_failures_escalate():
    from ai_pr_orchestrator.v3.config import V3Config as Cfg

    fake = _ready_fake()
    executor = ScriptedExecutor(developer_exit=1)
    config = Cfg(escalation=EscalationPolicyConfig(max_consecutive_coder_failures=3))
    loop, queue = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "consecutively" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"
