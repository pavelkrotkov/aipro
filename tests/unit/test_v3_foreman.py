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
from ai_pr_orchestrator.v3.config import (
    EscalationPolicyConfig,
    HermesLanesConfig,
    LaneProfileConfig,
    SafetyPolicyConfig,
    V3Config,
)
from ai_pr_orchestrator.v3.domain import (
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
    ReviewerFinding,
    WorkflowState,
)
from ai_pr_orchestrator.v3.findings import FindingRegistry
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop, ForemanQueue, _ForemanEscalation
from ai_pr_orchestrator.v3.interfaces import (
    GateDecision,
    LaneExecutionContext,
    LaneResult,
    ModelLease,
    SessionHandle,
    StateConflictError,
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
    reviewer_exit: int = 0
    developer_files: list[str] = field(default_factory=lambda: ["src/x.py"])
    calls: list[tuple[str, str]] = field(default_factory=list)
    round_counter: dict[str, int] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ) -> LaneResult:
        self.calls.append((lane.lane, workdir))
        self.prompts.append(task_prompt)
        if lane.role == "reviewer":
            n = self.round_counter.get(lane.lane, 0) + 1
            self.round_counter[lane.lane] = n
            scripted = self.reviewer_findings_by_round.get(n, [])
            # Each reviewer lane reports its own copy with a lane-unique id,
            # mirroring independent reviewers; dedup merges identical claims.
            findings = [replace(f, id=f"{f.id}-{lane.lane}", lane=lane.lane) for f in scripted]
            return LaneResult(
                session=HANDLE,
                exit_code=self.reviewer_exit,
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
    gate: Any,
    config: V3Config | None = None,
    git: Any = None,
    lanes: LaneRegistry | None = None,
):
    cfg = config or V3Config()
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    loop = ForemanPolicyLoop(
        queue,
        FakeBroker(),
        lanes or LaneRegistry.default(),
        executor,
        gate,
        git if git is not None else FakeGitOperations(),
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
    """A failing lane retries until the consecutive-failure threshold — while
    staying inside the invocation budget (F13 counts failures against it)."""
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_exit=3)
    config = V3Config(
        safety=SafetyPolicyConfig(max_coder_invocations_per_run=5),
        escalation=EscalationPolicyConfig(max_consecutive_coder_failures=3),
    )
    loop, queue = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "consecutively" in outcome.reason
    dev_calls = [c for c in executor.calls if c[0] == "developer"]
    assert len(dev_calls) == 3  # threshold reached within the budget
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


def test_failed_coder_attempts_consume_invocation_budget():
    """A failing lane may not bypass max_coder_invocations_per_run: with a
    budget of 1, a single sketched attempt escalates rather than retrying
    toward the (larger) consecutive-failure threshold."""
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_exit=1)
    config = V3Config(
        safety=SafetyPolicyConfig(max_coder_invocations_per_run=1),
        escalation=EscalationPolicyConfig(max_consecutive_coder_failures=5),
    )
    loop, queue = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "budget exhausted" in outcome.reason
    dev_calls = [c for c in executor.calls if c[0] == "developer"]
    assert len(dev_calls) == 1
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
    # Requeued onto the enabled label so a LATER pass re-selects it (list_ready
    # only returns the enabled label), and a second run_pass picks it up again.
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "queued"
    assert "v3-work" in fake.get_labels(1)
    again = loop.run_pass()[0]
    assert again.final_phase == "ci_gating"  # re-claimed and re-evaluated


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
    config = Cfg(
        safety=SafetyPolicyConfig(max_coder_invocations_per_run=5),
        escalation=EscalationPolicyConfig(max_consecutive_coder_failures=3),
    )
    loop, queue = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "consecutively" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


# --- Round-1 remediation: remaining findings --------------------------------


class RecordingGate:
    def __init__(self, decisions) -> None:
        self.decisions = list(decisions)
        self.seen: list[tuple[int, str]] = []

    def evaluate(self, issue, pr) -> GateDecision:
        self.seen.append((pr.number, pr.head_sha))
        if len(self.decisions) > 1:
            return self.decisions.pop(0)
        return self.decisions[0]


class RecordingGit(FakeGitOperations):
    def __init__(self, default: str = "main") -> None:
        super().__init__(default)
        self.cleanups: list[str] = []
        self.pushed: list[str] = []
        self.commits: list[tuple[str, str]] = []

    def cleanup_worktree(self, path: str) -> None:
        self.cleanups.append(path)
        super().cleanup_worktree(path)

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        self.commits.append((workdir, message))
        return super().commit(workdir, message, name=name, email=email)

    def push(self, branch: str) -> None:
        self.pushed.append(branch)


def test_ci_failure_loops_back_to_coding_then_done():
    """A failed CI check becomes findings for the next coding round; when the
    gate later goes green the item completes (F1 explicit loop-back)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = RecordingGate(
        [
            GateDecision(passed=False, pending_checks=(), failed_checks=("build",)),
            GateDecision(passed=True, pending_checks=(), failed_checks=()),
        ]
    )
    cfg = V3Config(safety=SafetyPolicyConfig(max_coder_invocations_per_run=3))
    loop, _ = _foreman(fake, executor, gate, cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    assert outcome.coder_invocations == 2  # second coding round from the CI finding
    assert len(gate.seen) == 2


def test_degenerate_gate_escalates_not_loops():
    """A gate that is neither passing, pending, nor naming a failed check (e.g.
    'no checks reported and green required') has no signal to loop on and must
    escalate, never spin forever (F1)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = GateDecision(passed=False, pending_checks=(), failed_checks=(), detail="no checks")
    loop, queue = _foreman(fake, executor, _gate(gate))
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "cannot progress" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


def test_claim_contention_does_not_leak_resources_or_mark_needs_human():
    """When another foreman already owns the claim, we must not create a
    branch/worktree behind a lost claim and must not escalate the competitor's
    work (F2 resource order + F9 crash persistence)."""
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    other = GitHubIssueQueue(fake, "owner", "repo", host_id="host-other")
    other.claim(ISSUE, "other-run", now=NOW)

    executor = ScriptedExecutor()
    git = RecordingGit()
    gate = _gate()
    cfg = V3Config()
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    loop = ForemanPolicyLoop(
        queue,
        FakeBroker(),
        LaneRegistry.default(),
        executor,
        gate,
        git,
        cfg,
        run_id="run-1",
        worktree_root="/wt",
        committer_name="Pavel Krotkov",
        committer_email="pavel.krotkov@gmail.com",
    )
    outcomes = loop.run_pass()
    # the item is already claimed by another host, so list_ready does not return
    # it and no work/resource materialization happens on our side
    assert outcomes == []  # no outcomes: the item was not re-queued
    assert executor.calls == []  # the developer lane never ran
    assert git.worktrees == {}
    # the competitor's claim is left alone — not escalated by us
    state = other.load_state("owner/repo#1")
    assert state is not None and state.phase != "escalated"


def test_reviewer_lane_crash_is_escalation_not_absence():
    """A reviewer lane that exits nonzero is a failed review round: it must
    escalate, never quietly read as 'no findings' and sail on to CI/done (F10)."""
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_exit=1)
    loop, queue = _foreman(fake, executor, _gate())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "reviewer lane" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


def test_worker_lane_resolves_from_configured_lanes():
    """The coder lane is the configured role-worker lane, not a hard-coded
    'developer' (F12)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    hermes = HermesLanesConfig(
        lanes=[
            LaneProfileConfig(name="coder", role="worker", profile_template="t-coder"),
            LaneProfileConfig(name="req", role="reviewer", profile_template="t-req"),
        ]
    )
    cfg = V3Config(hermes_lanes=hermes)
    lanes = LaneRegistry.from_config(hermes)
    loop, _ = _foreman(fake, executor, _gate(), cfg, lanes=lanes)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    assert executor.calls and executor.calls[0][0] == "coder"


def test_reviewer_triggers_capped_per_run():
    """max_reviewer_triggers_per_run bounds reviewer lane launches across the
    whole run (F14)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    cfg = V3Config(safety=SafetyPolicyConfig(max_reviewer_triggers_per_run=1))
    loop, _ = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    reviewer_calls = [c for c in executor.calls if c[0] != "developer"]
    assert len(reviewer_calls) == 1


def test_issue_body_is_included_in_coder_prompt():
    """The coder prompt carries the issue description/acceptance criteria, not
    just the slug (F16)."""
    fake = _ready_fake()
    fake._issue_bodies[1] = "Add feature X. AC: it must be fast."
    executor = ScriptedExecutor()
    loop, _ = _foreman(fake, executor, _gate())
    loop.run_pass()[0]
    dev_prompts = [
        p
        for call, p in zip(executor.calls, executor.prompts, strict=False)
        if call[0] == "developer"
    ]
    assert dev_prompts
    assert "Add feature X. AC: it must be fast." in dev_prompts[0]


def test_terminal_worktree_is_cleanup_but_pending_is_retained():
    """done/failed/escalated release the worktree; the pending-CI/requeue path
    deliberately retains it (F17)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    git = RecordingGit()
    loop, _ = _foreman(fake, executor, _gate(), git=git)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    assert "/wt/issue-1" in git.cleanups

    fake = _ready_fake()
    executor = ScriptedExecutor()
    git = RecordingGit()
    gate = _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=()))
    loop, _ = _foreman(fake, executor, gate, git=git)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "ci_gating"
    assert git.cleanups == []  # pending path retains the checkout for reuse


def test_prompt_token_budget_is_enforced_before_execution():
    """A lane is never launched once max_prompt_tokens would be exceeded;
    the foreman escalates instead (F18)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    cfg = V3Config(safety=SafetyPolicyConfig(max_prompt_tokens=1))
    loop, queue = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "token budget" in outcome.reason
    assert executor.calls == []  # never executed a lane
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "escalated"


def test_pr_ref_carries_commit_sha_not_branch():
    """The PR ref handed to the gate carries a commit SHA from create_pr, never
    the branch name (F19)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = RecordingGate([GateDecision(passed=True, pending_checks=(), failed_checks=())])
    loop, _ = _foreman(fake, executor, gate)
    loop.run_pass()[0]
    number, head_sha = gate.seen[0]
    assert head_sha
    assert head_sha != "aipro-issue-1"
    assert number != 1  # PR number from the shared sequence, not the issue number


def test_review_history_accumulates_across_rounds():
    """Each review round appends to — rather than replaces — the durable
    findings/dispositions, so restart reconciliation sees every round (F11)."""
    fake = _ready_fake()
    executor = ScriptedExecutor(
        reviewer_findings_by_round={1: [_finding(1)], 2: [_finding(2)], 3: []}
    )
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=5, max_reviewer_triggers_per_run=10
        ),
        escalation=EscalationPolicyConfig(stagnation_rounds_threshold=5),
    )
    loop, queue = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    state = queue.load_state("owner/repo#1")
    assert state is not None
    fixed_ids = {d.finding_id for d in state.dispositions if d.action == "fix"}
    # finding ids are rewritten per lane (e.g. "f-1-requirements-reviewer")
    assert any(rid.startswith("f-1") for rid in fixed_ids)
    assert any(rid.startswith("f-2") for rid in fixed_ids)


def test_persist_failure_aborts_the_round_as_escalation():
    """If review state cannot be persisted, abort/escalate rather than silently
    continuing with an unpersisted round (F15)."""

    class BoomQueue:
        state: WorkflowState | None = None

        def __init__(self, st):
            self.state = st

        def load_state(self, wid):
            return self.state

        def save_state(self, state, expected_updated_at):
            raise StateConflictError("boom")

    # A minimal claimed state so _persist_round can load it.
    st = WorkflowState(work_item_id=ISSUE.slug(), run_id="run-1", phase="reviewing")
    queue = cast("ForemanQueue", BoomQueue(st))
    gate = _gate()
    loop = ForemanPolicyLoop(
        queue,
        FakeBroker(),
        LaneRegistry.default(),
        ScriptedExecutor(),
        gate,
        RecordingGit(),
        V3Config(),
        run_id="run-1",
        worktree_root="/wt",
        committer_name="Pavel Krotkov",
        committer_email="pavel.krotkov@gmail.com",
    )
    try:
        loop._persist_round(ISSUE, st, "review-1", FindingRegistry(), [])
    except _ForemanEscalation as exc:
        assert "persist" in str(exc)
        return
    raise AssertionError("expected _ForemanEscalation")


def test_gate_is_committed_and_pushed_before_pr_open():
    """Lane output is committed and pushed before the PR is (re)opened so the PR
    targets a real remote head (F4)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    git = RecordingGit()
    loop, _ = _foreman(fake, executor, _gate(), git=git)
    loop.run_pass()[0]
    assert git.commits  # lane output was committed
    assert "aipro-issue-1" in git.pushed  # and pushed to the remote branch
