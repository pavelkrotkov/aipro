"""Tests for the V3 foreman policy loop (issue #55): a full fake lifecycle.

The foreman runs against the *real* ``GitHubIssueQueue`` over the fake GitHub
client, so claim/transition/label semantics are production semantics; only
the lane executor, broker, gate, and git ops are faked.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoSessionController,
    SessionBusyError,
    session_name_for,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.config import (
    EscalationPolicyConfig,
    HermesLanesConfig,
    LaneProfileConfig,
    ReviewPolicyConfig,
    SafetyPolicyConfig,
    V3Config,
)
from ai_pr_orchestrator.v3.domain import (
    FindingDisposition,
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
    LaneExecutor,
    LaneResult,
    ModelLease,
    SessionHandle,
    SessionSpec,
    StateConflictError,
)
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from tests.integration._fake_cao_server import STATUS_PROCESSING, FakeCAOServer, FaultSpec
from tests.unit.test_v3_git_ops import real_repo as real_repo

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
            fallbacks=(f"fallback-{demand.lane}",),
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
    reviewer_files: list[str] = field(default_factory=list)
    developer_sleep: float = 0.0
    developer_test_result: str = "passed"
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
                changed_files=list(self.reviewer_files),
                findings=list(findings),
                dispositions=[
                    FindingDisposition(
                        finding_id=fid,
                        action="accept",
                        rationale="independently verified coder correction",
                        decided_by=lane.lane,
                        run_id=context.run_id,
                        round_id=context.round_id,
                        response_to_round_id=turn,
                    )
                    for fid, turn in context.disposition_requests
                ],
            )
        if self.developer_sleep:
            import time

            time.sleep(self.developer_sleep)
        dispositions = [
            FindingDisposition(
                finding_id=fid,
                action="fix",
                rationale="fixed guard; regression test passes",
                decided_by=lane.lane,
                run_id=context.run_id,
                round_id=context.round_id,
            )
            for fid, _ in context.disposition_requests
        ]
        output = {
            "summary": "implemented requested change",
            "tests": [{"command": "pytest -q", "result": self.developer_test_result, "notes": ""}],
            "concerns": [],
            "no_changes": not self.developer_files,
            "dispositions": [
                {"finding_id": d.finding_id, "action": d.action, "rationale": d.rationale}
                for d in dispositions
            ],
        }
        return LaneResult(
            session=HANDLE,
            exit_code=self.developer_exit,
            output_summary=json.dumps(output),
            changed_files=list(self.developer_files),
            dispositions=dispositions,
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
        self.touched: dict[str, list[str]] = {}

    def default_branch(self) -> str:
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.branches.append(branch)

    def create_worktree(self, path: str, branch: str) -> str:
        self.worktrees[path] = branch
        return path

    def head_sha(self, workdir: str) -> str:
        return "sha"

    def current_branch(self, workdir: str) -> str:
        return self.worktrees.get(workdir, "aipro-issue-1")

    def repo_instructions(self, workdir: str) -> str:
        return ""

    def write_issue_description(self, workdir: str, description: str) -> tuple[str, str]:
        raise NotImplementedError("use real GitWorktreeOps for issue input delivery tests")

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        return "sha"

    def commit_count(self, workdir: str, base_ref: str) -> int:
        return 0

    def push(self, branch: str) -> None:
        pass

    def changed_files(self, workdir: str, base_ref: str | None = None) -> list[str]:
        # PR #73 review thread 8 / issue #78: the foreman's policy layer
        # now derives changed files from the worktree. The unit-test fake
        # records no real git state, so it returns an empty list unless
        # a test sets ``self.touched`` explicitly.
        return list(getattr(self, "touched", {}).get(workdir, []))

    def cleanup_worktree(self, path: str) -> None:
        self.worktrees.pop(path, None)


def _finding(idx: int, severity: Any = "major") -> ReviewerFinding:
    return ReviewerFinding(
        id=f"f-{idx}",
        lane="requirements-reviewer",
        body=f"finding {idx}",
        severity=severity,
        run_id="run-1",
        round_id="review-1",
    )


def _gate(decision: GateDecision | None = None) -> StaticGate:
    return StaticGate(decision or GateDecision(passed=True, pending_checks=(), failed_checks=()))


def _foreman(
    fake: FakeGitHubClient,
    executor: LaneExecutor,
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
    loop, queue = _foreman(
        fake, executor, gate, V3Config(review_policy=ReviewPolicyConfig(max_review_rounds=1))
    )
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
    config = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3,
            # Two review rounds x 3 reviewer lanes: the fix round must still
            # be reviewable (an exhausted trigger budget escalates).
            max_reviewer_triggers_per_run=6,
        )
    )
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


def test_developer_no_change_is_explicit_and_can_complete():
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_files=[])
    gate = _gate()
    git = RecordingGit()
    loop, queue = _foreman(fake, executor, gate, git=git)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "done"
    assert outcome.reason == "developer reported no changes"
    report = queue.load_state("owner/repo#1").extras["developer_report"]
    assert report["no_changes"] is True
    assert fake.list_open_prs() == []
    assert git.pushed == []
    assert gate.evaluated == []


def test_developer_no_change_report_cannot_contradict_edits():
    result = LaneResult(
        session=HANDLE,
        exit_code=0,
        output_summary=json.dumps(
            {
                "summary": "no changes needed",
                "tests": [],
                "concerns": [],
                "no_changes": True,
                "dispositions": [],
            }
        ),
        changed_files=["src/x.py"],
    )
    violation = ForemanPolicyLoop._developer_report_violation(result)
    assert violation == "developer reported no_changes=true despite authoritative worktree edits"


def test_failed_developer_attempt_advances_durable_fallback_route():
    class FailOnceExecutor(ScriptedExecutor):
        def __init__(self):
            super().__init__()
            self.worker_models = []
            self.worker_attempts = 0

        def execute(self, lane, task_prompt, workdir, context, lease=None):
            if lane.role != "worker":
                return super().execute(lane, task_prompt, workdir, context, lease)
            self.worker_attempts += 1
            assert lease is not None
            self.worker_models.append(lease.assignment.model_ref)
            self.developer_exit = 1 if self.worker_attempts == 1 else 0
            return super().execute(lane, task_prompt, workdir, context, lease)

    fake = _ready_fake()
    executor = FailOnceExecutor()
    cfg = V3Config(safety=SafetyPolicyConfig(max_coder_invocations_per_run=2))
    loop, queue = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "done"
    assert executor.worker_models == ["ref-developer", "fallback-developer"]
    state = queue.load_state(ISSUE.slug())
    assert state.extras["developer_model"]["model_ref"] == "fallback-developer"
    assert state.extras["developer_fallbacks"] == []


def test_developer_reported_test_failure_stops_before_push():
    fake = _ready_fake()
    executor = ScriptedExecutor(developer_test_result="failed")
    git = RecordingGit()
    loop, _ = _foreman(fake, executor, _gate(), git=git)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "developer reported failing test" in outcome.reason
    assert git.pushed == []


def test_developer_head_movement_is_rejected_before_controller_commit():
    class MovedHeadGit(RecordingGit):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def head_sha(self, workdir: str) -> str:
            self.reads += 1
            return "sha" if self.reads <= 3 else "agent-commit"

    fake = _ready_fake()
    git = MovedHeadGit()
    loop, _ = _foreman(fake, ScriptedExecutor(), _gate(), git=git)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "unexpected developer HEAD movement" in outcome.reason
    assert git.commits == []
    assert git.pushed == []


def test_failed_developer_attempt_cannot_reset_trusted_head_for_retry():
    class MovedHeadGit(RecordingGit):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def head_sha(self, workdir: str) -> str:
            self.reads += 1
            return "sha" if self.reads <= 3 else "agent-commit"

    fake = _ready_fake()
    executor = ScriptedExecutor(developer_exit=1)
    git = MovedHeadGit()
    loop, _ = _foreman(fake, executor, _gate(), git=git)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "unexpected developer HEAD movement" in outcome.reason
    assert [lane for lane, _ in executor.calls if lane == "developer"] == ["developer"]
    assert git.pushed == []


def test_developer_branch_switch_is_rejected_before_controller_commit():
    class SwitchedBranchGit(RecordingGit):
        def __init__(self):
            super().__init__()
            self.branch_reads = 0

        def current_branch(self, workdir: str) -> str:
            self.branch_reads += 1
            return "aipro-issue-1" if self.branch_reads == 1 else "other"

    fake = _ready_fake()
    git = SwitchedBranchGit()
    loop, _ = _foreman(fake, ScriptedExecutor(), _gate(), git=git)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "unexpected developer branch movement" in outcome.reason
    assert git.commits == []
    assert git.pushed == []


def test_checkout_adoption_rejects_persisted_head_movement():
    git = RecordingGit()
    loop, _ = _foreman(_ready_fake(), ScriptedExecutor(), _gate(), git=git)
    state = WorkflowState(
        ISSUE.slug(),
        loop.run_id,
        "coding",
        extras={
            "branch": "aipro-issue-1",
            "worktree": "/wt/issue-1",
            "head_sha": "trusted",
        },
    )

    with pytest.raises(_ForemanEscalation, match="expected trusted, found sha"):
        loop._verify_checkout(ISSUE, state, "aipro-issue-1", "/wt/issue-1")


def test_developer_resources_and_task_packet_are_durable():
    class InstructionGit(RecordingGit):
        def repo_instructions(self, workdir: str) -> str:
            return "AGENTS.md:\nkeep it small"

    fake = FakeGitHubClient()
    fake.seed_issue(
        1,
        labels=["v3-work"],
        title="Implement durable developer lane",
        body="Acceptance: preserve issue context.",
    )
    executor = ScriptedExecutor()
    loop, queue = _foreman(fake, executor, _gate(), git=InstructionGit())
    assert loop.run_pass()[0].final_phase == "done"

    state = queue.load_state("owner/repo#1")
    assert state.extras["developer_model"] == {
        "lane": "developer",
        "model_ref": "ref-developer",
    }
    assert state.extras["developer_fallbacks"] == ["fallback-developer"]
    assert state.extras["developer_session"] == HANDLE.session_id
    assert state.extras["branch"] == "aipro-issue-1"
    assert state.extras["head_sha"] == "sha"
    prompt = executor.prompts[0]
    for expected in (
        "Implement owner/repo#1: Implement durable developer lane",
        "Acceptance: preserve issue context.",
        "AGENTS.md:\nkeep it small",
        "Authoritative branch: aipro-issue-1",
        "Expected HEAD: sha",
        "Do not commit or push",
        "summary",
        "tests",
        "concerns",
        "no_changes",
    ):
        assert expected in prompt


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
    assert state.extras["coder_usage"] == {"run_id": loop.run_id, "invocations": 3}


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
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=20,
            max_total_iterations=9,
            max_reviewer_triggers_per_run=15,
        ),
        review_policy=ReviewPolicyConfig(max_review_rounds=5),
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

    def write_issue_description(self, workdir: str, description: str) -> tuple[str, str]:
        raise NotImplementedError("use real GitWorktreeOps for issue input delivery tests")

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
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3,
            max_reviewer_triggers_per_run=6,  # two rounds x 3 reviewer lanes
        )
    )
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


@pytest.mark.parametrize("budget,completed_rounds", [(0, 0), (1, 0), (2, 0), (4, 1), (5, 1)])
def test_reviewer_triggers_capped_per_run(budget, completed_rounds):
    """A budget must cover every required reviewer before any lane starts."""
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(1)]})
    gate = _gate()
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3, max_reviewer_triggers_per_run=budget
        )
    )
    loop, queue = _foreman(fake, executor, gate, cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "reviewer trigger budget" in outcome.reason
    assert len([c for c in executor.calls if c[0] != "developer"]) == 3 * completed_rounds
    assert gate.evaluated == []
    assert fake.list_open_prs() == []
    state = queue.load_state(ISSUE.slug())
    assert state is not None and state.phase == "escalated"


@pytest.mark.parametrize("findings,gate_calls", [({1: [_finding(1)]}, 0), ({}, 1)])
def test_review_round_cap_never_accepts_unreviewed_fixes(findings, gate_calls):
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round=findings)
    gate = RecordingGate(
        [
            GateDecision(passed=False, pending_checks=(), failed_checks=("build",)),
            GateDecision(passed=True, pending_checks=(), failed_checks=()),
        ]
    )
    git = RecordingGit()
    cfg = V3Config(
        review_policy=ReviewPolicyConfig(max_review_rounds=1),
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3,
            max_commits_per_run=3,
            max_reviewer_triggers_per_run=6,
        ),
    )
    loop, queue = _foreman(fake, executor, gate, cfg, git)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "review-round cap exhausted while fix findings pending" in outcome.reason
    assert executor.calls.count(("developer", "/wt/issue-1")) == 2
    assert len(executor.calls) == 5  # two coder turns and three reviewers
    assert len(gate.seen) == gate_calls
    assert len(git.commits) == len(git.pushed) == len(gate.seen)
    assert len(fake.list_open_prs()) == gate_calls
    state = queue.load_state(ISSUE.slug())
    assert state is not None and state.phase == "escalated"
    assert state.terminal_reason == outcome.reason
    assert "v3-work-needs-human" in fake.get_labels(1)


@pytest.mark.parametrize("body", ["", "Add feature X. AC: it must be fast.", "x" * 8_000])
def test_issue_body_is_included_in_coder_and_all_reviewer_prompts(body):
    fake = _ready_fake()
    fake._issue_bodies[1] = body
    executor = ScriptedExecutor()
    loop, _ = _foreman(fake, executor, _gate())
    assert loop.run_pass()[0].final_phase == "done"
    assert len(executor.prompts) == 4  # coder and all three independent reviewers
    for prompt in executor.prompts:
        assert body in prompt
    for prompt in executor.prompts[1:]:
        assert "Return only a JSON array" in prompt


@pytest.mark.parametrize("successful_reads", [0, 1])
def test_issue_description_fetch_failure_prevents_contextless_execution(
    monkeypatch, successful_reads
):
    fake = _ready_fake()
    calls = 0

    def get_body(number):
        nonlocal calls
        calls += 1
        if calls > successful_reads:
            raise RuntimeError("authoritative issue fetch failed")
        return "Complete requirements."

    monkeypatch.setattr(fake, "get_issue_body", get_body)
    executor = ScriptedExecutor()
    loop, _ = _foreman(fake, executor, _gate())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "authoritative issue fetch failed" in outcome.reason
    assert len(executor.calls) == successful_reads
    assert fake.list_open_prs() == []


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


# --- Round-2 remediation -------------------------------------------------------


def test_lease_is_heartbeated_while_lane_runs():
    """A lane that outlasts the lease interval gets its lease renewed during
    execution — reclaim_expired cannot hand the issue away mid-edit (#1)."""

    from ai_pr_orchestrator.v3.config import GitHubQueueConfig

    fake = _ready_fake()
    executor = ScriptedExecutor()
    executor.developer_sleep = 0.5
    cfg = V3Config(github_queue=GitHubQueueConfig(lease_seconds=1))  # interval ≈ 0.33s
    loop, queue = _foreman(fake, executor, _gate(), cfg)
    heartbeats: list[WorkflowState] = []
    original = queue.heartbeat

    def counting_heartbeat(state, *, now=None):
        refreshed = original(state, now=now)
        heartbeats.append(refreshed)
        return refreshed

    queue.heartbeat = counting_heartbeat  # type: ignore[method-assign]
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    # One pre-lane heartbeat plus at least one DURING the 0.5s lane (interval
    # is a third of the 1s lease).
    assert len(heartbeats) >= 2


def test_pending_ci_requeue_goes_straight_to_gate():
    """A requeued pending-CI item with a retained PR must not relaunch
    coding/review — it re-evaluates the same head at the CI gate (#2)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=()))
    loop, _queue = _foreman(fake, executor, gate)
    first = loop.run_pass()[0]
    assert first.final_phase == "ci_gating"
    assert len([c for c in executor.calls if c[0] == "developer"]) == 1

    second = loop.run_pass()[0]
    assert second.final_phase == "ci_gating"
    # No new coding or review lanes on the re-claim pass.
    assert len([c for c in executor.calls if c[0] == "developer"]) == 1
    assert len(gate.evaluated) == 2


def test_reviewer_budget_exhausted_escalates_not_unreviewed():
    """With the trigger budget spent, a round that needs reviewers escalates
    instead of reporting a fake 'no findings' (#3)."""
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(4)]})
    cfg = V3Config(
        safety=SafetyPolicyConfig(max_coder_invocations_per_run=3)  # triggers stay at 3
    )
    loop, _ = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "reviewer trigger budget" in outcome.reason


def test_every_state_write_stamps_a_fresh_updated_at():
    """_persist_round must advance updated_at so the CAS version moves — a
    stale concurrent writer may not pass the precondition (#4)."""

    class RecorderQueue:
        def __init__(self, st):
            self.state = st
            self.saved: list[tuple[WorkflowState, object]] = []

        def load_state(self, wid):
            return self.state

        def save_state(self, state, expected_updated_at):
            self.saved.append((state, expected_updated_at))

    st = WorkflowState(work_item_id=ISSUE.slug(), run_id="run-1", phase="reviewing")
    recorder = RecorderQueue(st)
    queue = cast("ForemanQueue", recorder)
    loop = ForemanPolicyLoop(
        queue,
        FakeBroker(),
        LaneRegistry.default(),
        ScriptedExecutor(),
        _gate(),
        RecordingGit(),
        V3Config(),
        run_id="run-1",
        worktree_root="/wt",
        committer_name="N",
        committer_email="e@x",
    )
    loop._persist_round(ISSUE, st, "review-1", FindingRegistry(), [])
    assert recorder.saved
    saved, expected = recorder.saved[0]
    assert expected == st.updated_at  # CAS against the loaded version
    assert saved.updated_at > st.updated_at  # version advanced


def test_pending_ci_beyond_timeout_escalates():
    """The pending-CI wait start is persisted and, once
    ci_wait_timeout_seconds is exceeded, the item escalates instead of
    requeueing forever (#5)."""
    from datetime import UTC as UTC_
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=()))
    loop, queue = _foreman(fake, executor, gate)
    first = loop.run_pass()[0]
    assert first.final_phase == "ci_gating"
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.extras.get("ci_wait_started_at")  # wait persisted

    # Backdate the wait start beyond the configured 1800s timeout.
    stale = replace(
        state,
        extras={
            **state.extras,
            "ci_wait_started_at": (_dt.now(UTC_) - _td(seconds=3600)).isoformat(),
        },
    )
    queue.save_state(stale, expected_updated_at=state.updated_at)

    second = loop.run_pass()[0]
    assert second.final_phase == "escalated"
    assert "ci_wait_timeout_seconds" in second.reason


def test_coder_cap_is_checked_before_launching():
    """With open findings and the cap already reached, no further coder
    invocation is launched — the item escalates first (#6)."""
    fake = _ready_fake()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(5)]})
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=1,
            max_reviewer_triggers_per_run=6,
        )
    )
    loop, _ = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "budget exhausted with open findings" in outcome.reason
    assert len([c for c in executor.calls if c[0] == "developer"]) == 1  # never relaunched


def test_commit_cap_exceeded_escalates_but_noop_passes():
    """max_commits_per_run is consulted BEFORE committing: a round that would
    push past the cap escalates; a visit with nothing to commit does not (#7)."""

    class CountingGit(RecordingGit):
        def __init__(self):
            super().__init__()
            self.count = 0

        def commit_count(self, workdir, base_ref):
            return self.count

    # Cap reached and changes pending -> escalate before committing.
    fake = _ready_fake()
    git = CountingGit()
    git.count = 1  # default cap is 1
    git.touched = {"/wt/issue-1": ["src/change.py"]}
    loop, _ = _foreman(fake, ScriptedExecutor(), _gate(), git=git)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "commit budget exhausted" in outcome.reason

    # Cap reached but nothing to commit -> the no-op commit path still gates.
    fake = _ready_fake()
    git = CountingGit()
    git.count = 1
    loop, _ = _foreman(fake, ScriptedExecutor(developer_files=[]), _gate(), git=git)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"


def test_crash_path_cleans_up_worktree(monkeypatch):
    """A crash releases its worktree only after the terminal write succeeds."""

    class CrashingExecutor(ScriptedExecutor):
        def execute(self, lane, task_prompt, workdir, context, lease=None):
            raise RuntimeError("lane exploded")

    fake = _ready_fake()
    git = RecordingGit()
    loop, queue = _foreman(fake, CrashingExecutor(), _gate(), git=git)
    cleanup = git.cleanup_worktree

    def require_terminal_before_cleanup(path):
        state = queue.load_state(ISSUE.slug())
        assert state is not None and state.phase == "escalated"
        cleanup(path)

    monkeypatch.setattr(git, "cleanup_worktree", require_terminal_before_cleanup)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "/wt/issue-1" in git.cleanups  # leaked no worktree on the crash path


def test_reviewer_workflow_file_change_is_a_policy_violation():
    """A reviewer lane editing .github/workflows/ hits the same policy check
    as the coder — it escalates instead of being staged and pushed (#9)."""
    fake = _ready_fake()
    executor = ScriptedExecutor()
    executor.reviewer_files = [".github/workflows/evil.yml"]
    loop, _ = _foreman(fake, executor, _gate())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "policy violation" in outcome.reason


def test_coder_reply_is_actually_posted_to_the_review_thread():
    """require_coder_reply_before_resolve posts the reply to the GitHub
    thread — the disposition and the thread agree (#10)."""
    fake = _ready_fake()
    fake.seed_thread("T-1", pr_number=1)
    finding = _finding(6)
    finding = replace(finding, thread_id="T-1")
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [finding]})
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=2,
            max_reviewer_triggers_per_run=6,
        )
    )
    loop, _ = _foreman(fake, executor, _gate(), cfg)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    thread = fake._threads["T-1"]
    assert len(thread.comments) == 1
    assert thread.comments[0].body == "fixed guard; regression test passes"


def test_failed_thread_reply_is_an_escalation():
    """A review-thread reply that cannot be posted aborts the round as an
    escalation — never a disposition claiming a reply that was not sent (#10)."""
    fake = _ready_fake()
    fake.seed_thread("T-2", pr_number=1)
    finding = replace(_finding(7), thread_id="T-2")
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [finding]})

    def _impl(thread_id: str, body: str) -> dict[str, Any] | None:
        raise RuntimeError("github down")

    setattr(fake, "reply_to_review_thread", _impl)  # noqa: B010
    config = V3Config(safety=SafetyPolicyConfig(max_coder_invocations_per_run=2))
    loop, _ = _foreman(fake, executor, _gate(), config)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "reply" in outcome.reason


@pytest.mark.parametrize(
    "developer_exit,developer_files", [(1, []), (0, [".github/workflows/evil.yml"])]
)
@pytest.mark.parametrize("error_type", [RuntimeError, StateConflictError])
def test_terminal_write_failure_retains_active_worktree(
    monkeypatch, developer_exit, developer_files, error_type
):
    fake = _ready_fake()
    git = RecordingGit()
    executor = ScriptedExecutor(developer_exit=developer_exit, developer_files=developer_files)
    loop, queue = _foreman(fake, executor, _gate(), git=git)
    save = queue.save_state

    def reject_terminal(state, expected_updated_at):
        if state.phase in ("failed", "escalated"):
            raise error_type("terminal persistence unavailable")
        save(state, expected_updated_at)

    monkeypatch.setattr(queue, "save_state", reject_terminal)
    with pytest.raises(error_type, match="terminal persistence unavailable"):
        loop.run_pass()
    state = queue.load_state(ISSUE.slug())
    assert state is not None and state.phase == "coding"
    assert git.cleanups == []
    assert state.extras["worktree"] in git.worktrees


def test_crash_state_read_failure_propagates_and_retains_worktree(monkeypatch):
    fake = _ready_fake()
    git = RecordingGit()
    loop, queue = _foreman(fake, ScriptedExecutor(), _gate(), git=git)
    load = queue.load_state

    def fail_read(work_item_id):
        raise RuntimeError("authoritative state unavailable")

    def crash(*args, **kwargs):
        monkeypatch.setattr(queue, "load_state", fail_read)
        raise RuntimeError("lane exploded")

    monkeypatch.setattr(loop._executor, "execute", crash)
    with pytest.raises(RuntimeError, match="authoritative state unavailable"):
        loop.run_pass()
    state = load(ISSUE.slug())
    assert state is not None and state.phase == "coding"
    assert git.cleanups == []
    assert state.extras["worktree"] in git.worktrees


def test_cleanup_read_failure_retains_worktree_and_continues_pass(monkeypatch, caplog):
    fake = _ready_fake()
    fake.seed_issue(2, labels=["v3-work"])
    git = RecordingGit()
    loop, queue = _foreman(fake, ScriptedExecutor(), _gate(), git=git)
    load = queue.load_state

    def reject_first_terminal_read(work_item_id):
        state = load(work_item_id)
        if work_item_id == ISSUE.slug() and state is not None and state.phase == "done":
            raise RuntimeError("cleanup verification unavailable")
        return state

    monkeypatch.setattr(queue, "load_state", reject_first_terminal_read)
    outcomes = loop.run_pass()
    assert [(outcome.issue.number, outcome.final_phase) for outcome in outcomes] == [
        (1, "done"),
        (2, "done"),
    ]
    assert "/wt/issue-1" in git.worktrees
    assert git.cleanups == ["/wt/issue-2"]
    assert "owner/repo#1" in caplog.text
    assert "cleanup verification unavailable" in caplog.text


def test_busy_cao_submission_preserves_active_run_despite_heartbeat_failure(monkeypatch):
    registry = LaneRegistry.default()
    lane = registry.get("developer")
    git = RecordingGit()
    name = session_name_for("run-1", lane.lane, ISSUE.slug())
    with FakeCAOServer() as cao, httpx.Client(base_url=cao.url) as client:
        cao.set_status_sequence(name, [STATUS_PROCESSING])
        with CaoSessionController(
            CAOControlPlaneConfig(base_url=cao.url), registry, client=client
        ) as controller:
            handle = controller.start_session(
                SessionSpec(
                    lane=lane,
                    run_id="run-1",
                    workdir="/wt/issue-1",
                    env={},
                    context=LaneExecutionContext(run_id="run-1", work_item_id=ISSUE.slug()),
                    model_lease=ModelLease(
                        lease_id="previous",
                        assignment=ModelAssignment(lane=lane.lane, model_ref="ref-developer"),
                    ),
                )
            )
            controller.submit_work(handle, "earlier in-flight task")
            session = cao._sessions[name]
            cao.add_fault(
                FaultSpec(
                    method="POST",
                    path_prefix=f"/terminals/{session.terminal_id}/input",
                    status_code=409,
                )
            )
            executor = CaoLaneExecutor(
                controller,
                registry,
                git=FakeGitOperations(),
                catalog=ModelCatalog(
                    (ModelCatalogEntry("ref-developer", "test-model", provider="test"),)
                ),
            )
            cfg = V3Config()
            cfg = replace(cfg, github_queue=replace(cfg.github_queue, lease_seconds=0.15))
            loop, queue = _foreman(_ready_fake(), executor, _gate(), config=cfg, git=git)
            before_submission = []
            heartbeat_failed = threading.Event()
            heartbeat = queue.heartbeat

            def fail_background_heartbeat(state, **kwargs):
                if threading.current_thread().name == "foreman-lease-heartbeat":
                    heartbeat_failed.set()
                    raise RuntimeError("heartbeat network outage")
                return heartbeat(state, **kwargs)

            monkeypatch.setattr(queue, "heartbeat", fail_background_heartbeat)

            def capture_state(request):
                if request.url.path.endswith("/input"):
                    assert heartbeat_failed.wait(2), "background heartbeat must fail before HTTP409"
                    before_submission.append(queue.load_state(ISSUE.slug()))

            client.event_hooks["request"].append(capture_state)
            with pytest.raises(SessionBusyError):
                loop.run_pass()

            assert controller.observe(handle).state == "running"
            assert session.submitted_messages == ["earlier in-flight task"]
            assert queue.load_state(ISSUE.slug()) == before_submission[0]
    assert before_submission[0].phase == "coding"
    assert before_submission[0].extras["host_id"] == "host-A"
    assert git.worktrees == {"/wt/issue-1": "aipro-issue-1"}


def test_successful_lane_cannot_hide_failed_heartbeat(monkeypatch):
    cfg = V3Config()
    cfg = replace(cfg, github_queue=replace(cfg.github_queue, lease_seconds=0.15))
    loop, queue = _foreman(_ready_fake(), ScriptedExecutor(), _gate(), config=cfg)
    failed = threading.Event()

    def fail_heartbeat(state):
        failed.set()
        raise RuntimeError("heartbeat network outage")

    monkeypatch.setattr(queue, "heartbeat", fail_heartbeat)
    state = WorkflowState(work_item_id=ISSUE.slug(), run_id="run-1", phase="coding")
    with (
        pytest.raises(_ForemanEscalation, match="claim lease heartbeat failed"),
        loop._lease_heartbeat(state),
    ):
        assert failed.wait(2)


@pytest.mark.parametrize("failure", ["discovery", "create_response", "record"])
def test_uncertain_pr_outcome_preserves_state_and_reconciles(monkeypatch, failure):
    from ai_pr_orchestrator.v3.foreman import PRReconciliationError

    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = RecordingGate([GateDecision(passed=True, pending_checks=(), failed_checks=())])
    git = RecordingGit()
    loop, queue = _foreman(fake, executor, gate, git=git)
    create_pr = fake.create_pr
    save_state = queue.save_state
    created = []

    def create(*args, **kwargs):
        pr = create_pr(*args, **kwargs)
        created.append(pr)
        if failure == "create_response":
            raise RuntimeError("lost PR response")
        return pr

    def save(state, expected_updated_at):
        if state.extras.get("pr_number") is not None:
            raise StateConflictError("lost PR record")
        return save_state(state, expected_updated_at)

    def unavailable():
        raise RuntimeError("PR discovery unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(fake, "create_pr", create)
        if failure == "discovery":
            patch.setattr(fake, "list_open_prs", unavailable)
        if failure == "record":
            patch.setattr(queue, "save_state", save)
        with pytest.raises(PRReconciliationError) as error:
            loop.run_pass()
    assert error.value.__cause__ is not None
    state = queue.load_state(ISSUE.slug())
    assert state is not None and state.phase == "ci_gating"
    assert state.extras.get("pr_number") is None
    assert git.cleanups == []
    assert gate.seen == []
    assert len(fake.list_open_prs()) == (0 if failure == "discovery" else 1)
    calls = list(executor.calls)

    # Recovery policy entry: scheduling/reclaiming the retained run belongs to
    # the foreman coordinator, not another retry loop in PR creation.
    outcome = loop._run_loop(
        ISSUE,
        state,
        state.extras["worktree"],
        state.extras["branch"],
        now=None,
        resume_at_gate=True,
    )
    assert outcome.final_phase == "done"
    assert executor.calls == calls
    prs = fake.list_open_prs()
    assert len(prs) == 1
    assert gate.seen == [(prs[0].number, prs[0].head_sha)]
    persisted = queue.load_state(ISSUE.slug())
    assert persisted is not None and persisted.extras["pr_number"] == prs[0].number
    if created:
        assert prs[0] == created[0]


def test_recorded_pr_refresh_failure_never_discovers_or_creates(monkeypatch):
    from ai_pr_orchestrator.v3.foreman import PRReconciliationError

    fake = _ready_fake()
    executor = ScriptedExecutor()
    gate = RecordingGate(
        [
            GateDecision(passed=False, pending_checks=("build",), failed_checks=()),
            GateDecision(passed=True, pending_checks=(), failed_checks=()),
        ]
    )
    git = RecordingGit()
    loop, queue = _foreman(fake, executor, gate, git=git)
    assert loop.run_pass()[0].final_phase == "ci_gating"
    original = fake.list_open_prs()[0]
    calls = list(executor.calls)

    def unavailable(*args, **kwargs):
        raise RuntimeError("PR refresh unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("recorded PR identity must not reach discovery or creation")

    with monkeypatch.context() as patch:
        patch.setattr(fake, "get_pr", unavailable)
        patch.setattr(fake, "list_open_prs", forbidden)
        patch.setattr(fake, "create_pr", forbidden)
        with pytest.raises(PRReconciliationError, match="PR refresh unavailable"):
            loop.run_pass()
    state = queue.load_state(ISSUE.slug())
    assert state is not None and state.phase == "ci_gating"
    assert state.extras["pr_number"] == original.number
    assert git.cleanups == []
    fake._prs[original.number] = replace(original, head_sha="updated-head")
    outcome = loop._run_loop(
        ISSUE,
        state,
        state.extras["worktree"],
        state.extras["branch"],
        now=None,
        resume_at_gate=True,
    )
    assert outcome.final_phase == "done"
    assert executor.calls == calls
    assert gate.seen[-1] == (original.number, "updated-head")
    assert len(fake.list_open_prs()) == 1


@pytest.mark.parametrize("mismatch", [{"is_fork": True}, {"base_ref": "release"}])
def test_pr_discovery_excludes_other_repository_or_base(mismatch):
    fake = _ready_fake()
    wrong = fake.create_pr("wrong PR", "", head="aipro-issue-1", base="main")
    fake._prs[wrong.number] = replace(wrong, **mismatch)
    matching = fake.create_pr("matching PR", "", head="aipro-issue-1", base="main")
    gate = RecordingGate([GateDecision(passed=True, pending_checks=(), failed_checks=())])
    loop, queue = _foreman(fake, ScriptedExecutor(), gate)
    assert loop.run_pass()[0].final_phase == "done"
    assert gate.seen == [(matching.number, matching.head_sha)]
    assert len(fake.list_open_prs()) == 2
    state = queue.load_state(ISSUE.slug())
    assert state is not None and state.extras["pr_number"] == matching.number


@pytest.mark.parametrize("retained_path", ["absent", "missing", "unrelated"])
@pytest.mark.parametrize("settled", ["green", "pending", "failed"])
def test_cold_ci_resume_needs_only_recorded_pr(monkeypatch, tmp_path, retained_path, settled):
    fake = _ready_fake()
    first, queue = _foreman(
        fake,
        ScriptedExecutor(),
        _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=())),
    )
    assert first.run_pass()[0].final_phase == "ci_gating"
    state = queue.load_state(ISSUE.slug())
    extras = {k: v for k, v in state.extras.items() if k != "worktree"}
    if retained_path != "absent":
        extras["worktree"] = str(tmp_path / "other-host-checkout")
    if retained_path == "unrelated":
        (tmp_path / "other-host-checkout").mkdir()
    queue.save_state(replace(state, extras=extras), expected_updated_at=state.updated_at)
    original = fake.get_pr(extras["pr_number"])
    fake._prs[original.number] = replace(original, head_sha="live-pr-head")
    decision = GateDecision(
        passed=settled == "green",
        pending_checks=("build",) if settled == "pending" else (),
        failed_checks=("build",) if settled == "failed" else (),
    )
    gate = RecordingGate([decision])
    executor = ScriptedExecutor()
    git = RecordingGit()
    resumed, resumed_queue = _foreman(fake, executor, gate, git=git)
    resumed_queue._host_id = "host-B"

    def forbidden(*args, **kwargs):
        pytest.fail("CI-only resume must not materialize, commit, push, or recreate a PR")

    for method in ("default_branch", "create_branch", "create_worktree", "commit", "push"):
        monkeypatch.setattr(git, method, forbidden)
    monkeypatch.setattr(fake, "create_pr", forbidden)
    monkeypatch.setattr(fake, "list_open_prs", forbidden)
    outcome = resumed.run_pass()[0]
    expected = {"green": "done", "pending": "ci_gating", "failed": "escalated"}
    assert outcome.final_phase == expected[settled]
    assert gate.seen == [(original.number, "live-pr-head")]
    assert git.cleanups == []
    assert executor.calls == []
    persisted = resumed_queue.load_state(ISSUE.slug())
    assert persisted.extras["pr_number"] == original.number
    assert persisted.extras["host_id"] == "host-B"


@pytest.mark.parametrize("status", [404, 403])
def test_cold_ci_resume_missing_pr_escalates_only_on_definite_404(monkeypatch, status):
    from ai_pr_orchestrator.github.client import GitHubClient
    from ai_pr_orchestrator.v3.foreman import PRReconciliationError

    fake = _ready_fake()
    first, _ = _foreman(
        fake,
        ScriptedExecutor(),
        _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=())),
    )
    first.run_pass()
    gate = RecordingGate([])
    executor = ScriptedExecutor()
    resumed, queue = _foreman(fake, executor, gate)

    def forbidden(*args, **kwargs):
        pytest.fail("known PR lookup must never discover or create another PR")

    monkeypatch.setattr(fake, "create_pr", forbidden)
    monkeypatch.setattr(fake, "list_open_prs", forbidden)
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={}))
    with httpx.Client(transport=transport) as http:
        client = GitHubClient("test", "owner", "repo", http_client=http)
        monkeypatch.setattr(fake, "get_pr", client.get_pr)
        if status == 404:
            outcome = resumed.run_pass()[0]
            assert outcome.final_phase == "escalated"
            assert "could not be found" in outcome.reason
        else:
            with pytest.raises(PRReconciliationError):
                resumed.run_pass()
    assert gate.seen == []
    assert executor.calls == []
    assert queue.load_state(ISSUE.slug()).phase == ("escalated" if status == 404 else "ci_gating")


@pytest.mark.parametrize("settled", ["green", "failed", "missing-pr", "timeout"])
def test_ci_resume_preserves_unrelated_real_worktree(monkeypatch, real_repo, tmp_path, settled):
    from ai_pr_orchestrator.v3.git_ops import GitWorktreeOps

    git = GitWorktreeOps(real_repo)
    git.create_branch("human-work", "main")
    checkout = tmp_path / "human-work"
    git.create_worktree(str(checkout), "human-work")
    (checkout / "file.txt").write_text("unsaved tracked change")
    (checkout / "unsaved-human-work").write_text("irreplaceable")
    fake = _ready_fake()
    first, queue = _foreman(
        fake,
        ScriptedExecutor(),
        _gate(GateDecision(passed=False, pending_checks=("build",), failed_checks=())),
    )
    first.run_pass()
    state = queue.load_state(ISSUE.slug())
    extras = {**state.extras, "worktree": str(checkout)}
    if settled == "timeout":
        extras["ci_wait_started_at"] = "2000-01-01T00:00:00+00:00"
    queue.save_state(replace(state, extras=extras), expected_updated_at=state.updated_at)
    gate = _gate(
        GateDecision(
            passed=settled == "green",
            pending_checks=("build",) if settled == "timeout" else (),
            failed_checks=("build",) if settled == "failed" else (),
        )
    )
    resumed, queue = _foreman(fake, ScriptedExecutor(), gate, git=git)
    queue._host_id = "host-B"
    if settled == "missing-pr":

        def missing(number):
            response = httpx.Response(
                404, request=httpx.Request("GET", f"https://api.github.com/pulls/{number}")
            )
            response.raise_for_status()

        monkeypatch.setattr(fake, "get_pr", missing)
    outcome = resumed.run_pass()[0]
    assert outcome.final_phase == ("done" if settled == "green" else "escalated")
    assert (checkout / "file.txt").read_text() == "unsaved tracked change"
    assert (checkout / "unsaved-human-work").read_text() == "irreplaceable"
    assert queue.load_state(ISSUE.slug()).extras["worktree"] == str(checkout)
    assert resumed._executor.calls == []


def test_initial_state_read_failure_cannot_authorize_cleanup(monkeypatch):
    git = RecordingGit()
    loop, queue = _foreman(_ready_fake(), ScriptedExecutor(), _gate(), git=git)

    def unavailable(work_item_id):
        raise RuntimeError("initial authoritative read failed")

    monkeypatch.setattr(queue, "load_state", unavailable)
    with pytest.raises(RuntimeError, match="initial authoritative read failed"):
        loop.run_pass()
    assert git.cleanups == []
    assert git.worktrees == {}
    assert loop._executor.calls == []


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"run_id": "run-1", "invocations": True},
        {"run_id": "run-1", "invocations": -1},
        {"run_id": "other", "invocations": 1},
    ],
)
def test_reconstructed_coder_budget_never_guesses_missing_or_invalid_usage(usage):
    loop, _ = _foreman(_ready_fake(), ScriptedExecutor(), _gate())
    state = WorkflowState(
        "owner/repo#1",
        loop.run_id,
        "coding",
        extras={} if usage is None else {"coder_usage": usage},
    )
    restored = WorkflowState.from_dict(state.to_dict())
    with pytest.raises(_ForemanEscalation, match=r"coder (budget|usage)|malformed"):
        loop._coder_usage(restored)
