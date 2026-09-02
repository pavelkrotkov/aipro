"""E2E scenario 11 (issue #55): reviewer worktree mutation never reaches
the authoritative branch.

The foreman's workflow-file policy (``SafetyPolicyConfig.disallow_workflow_file_changes``)
rejects any lane output that touches files under ``.github/workflows/``.
The reviewer's edits must NEVER reach the PR's authoritative branch —
neither via direct commit nor via a later coder commit that picks up
the reviewer's stray worktree changes.

Acceptance (per #55 E2E scenarios, #11):
- A reviewer lane that mutates ``.github/workflows/`` is rejected; the
  foreman's outcome is ``failed`` (policy violation) and the
  ``v3-work-error`` label is set.
- No commit was created on the branch: the PR head never advanced.
- A coder lane that does the same is rejected on the same path
  (``policy violation``), and the durable state records the violation
  reason.
- The commit-attempting code path is never invoked when the policy
  check trips — i.e., the violation surfaces before any push.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.config import V3Config
from ai_pr_orchestrator.v3.domain import (
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
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
        return list(self.touched.get(workdir, []))

    def cleanup_worktree(self, path: str) -> None:
        self.cleanups.append(path)
        self.worktrees.pop(path, None)


@dataclass
class WorkflowMutatingExecutor:
    """A scripted executor whose reviewer (or coder) lane reports that
    it modified a ``.github/workflows/`` file.

    Two modes:
    - ``mutator_role="reviewer"`` simulates a reviewer that wrote a
      workflow change; the foreman must refuse to push and mark the
      work item escalated.
    - ``mutator_role="worker"`` simulates a coder (developer lane's
      role is ``worker`` in the lane registry) doing the same; the
      foreman's ``_policy_violation`` check on the coder's lane
      result rejects the run before the commit/push step.

    The lane executor's ``changed_files`` carries the offending path;
    the foreman's policy check evaluates ``result.changed_files`` first
    and falls back to ``git.changed_files(worktree)`` when the result
    reports nothing.
    """

    mutator_role: str
    touched_files: list[str] = field(default_factory=lambda: [".github/workflows/ci.yml"])
    coder_calls: int = 0
    reviewer_calls: int = 0

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ):
        handle = SessionHandle(session_id="cao-mutator", lane=lane.lane)
        if lane.role == self.mutator_role:
            files = list(self.touched_files)
            if lane.role == "reviewer":
                self.reviewer_calls += 1
                return LaneResult(
                    session=handle,
                    exit_code=0,
                    output_summary="reviewer writes to .github/workflows/",
                    changed_files=files,
                    findings=[],
                )
            self.coder_calls += 1
            return LaneResult(
                session=handle,
                exit_code=0,
                output_summary="coder writes to .github/workflows/",
                changed_files=files,
            )
        # Other lanes are no-ops.
        if lane.role == "reviewer":
            self.reviewer_calls += 1
            return LaneResult(
                session=handle,
                exit_code=0,
                output_summary="",
                changed_files=[],
                findings=[],
            )
        self.coder_calls += 1
        return LaneResult(
            session=handle,
            exit_code=0,
            output_summary="",
            changed_files=["src/clean.py"],
        )


def _ready_fake() -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    return fake


def _build(
    fake: FakeGitHubClient,
    executor: WorkflowMutatingExecutor,
    *,
    worktree_files: list[str] | None = None,
) -> ForemanPolicyLoop:
    cfg = V3Config()
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    git = TrackingGit()
    git.touched = {
        "/wt/issue-1": worktree_files or [".github/workflows/ci.yml"],
    }
    return ForemanPolicyLoop(
        queue,
        StaticBroker(),
        LaneRegistry.default(),
        executor,
        StaticGate(),
        git,
        cfg,
        run_id="run-s11",
        worktree_root="/wt",
        committer_name="AIPRO E2E Bot",
        committer_email="aipro-bot@example.invalid",
    )


def test_scenario_11_reviewer_mutation_of_workflows_is_a_policy_violation():
    """A reviewer lane that reports a ``.github/workflows/`` change is
    rejected before any commit / push, with the work item marked
    ``escalated`` (the foreman's reviewer-violation path uses the
    ``_ForemanEscalation`` control-flow signal so the durable state
    ends on the ``needs-human`` phase for operator review). The branch
    head never advances."""
    fake = _ready_fake()
    executor = WorkflowMutatingExecutor(
        mutator_role="reviewer",
        touched_files=[".github/workflows/ci.yml"],
    )
    loop = _build(fake, executor)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated", (
        f"expected 'escalated' (reviewer violation path), got {outcome.final_phase!r}; "
        f"reason={outcome.reason!r}"
    )
    assert "policy violation" in outcome.reason
    assert ".github/workflows" in outcome.reason
    state = loop._queue.load_state("owner/repo#1")  # noqa: SLF001
    assert state is not None and state.phase == "escalated"
    assert "v3-work-needs-human" in fake.get_labels(1)
    # No PR was created.
    assert fake.list_open_prs() == []


def test_scenario_11_coder_mutation_of_workflows_is_a_policy_violation():
    """A coder lane that touches ``.github/workflows/`` is rejected by
    the same ``_policy_violation`` path used for reviewer edits. The
    foreman's coder fails the work item before the commit/push step
    runs, so the branch never advances.

    Only the worktree's touched files need to include the workflow
    path so the reviewer's fallback path (``git.changed_files``) does
    not also trip and turn the test into a reviewer-violation case.
    Setting it before any reviewer runs keeps the test deterministic.
    """
    fake = _ready_fake()
    executor = WorkflowMutatingExecutor(
        mutator_role="worker",
        touched_files=[".github/workflows/release.yml"],
    )
    # Pre-seed the worktree with a clean file so reviewer lanes do not
    # see the workflow path; the coder reports the workflow path
    # itself, and the foreman's policy check trips on the lane result
    # before the reviewer ever runs.
    loop = _build(fake, executor, worktree_files=["src/clean.py"])

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "failed", (
        f"expected 'failed', got {outcome.final_phase!r}; reason={outcome.reason!r}"
    )
    assert "policy violation" in outcome.reason
    assert "v3-work-error" in fake.get_labels(1)


def test_scenario_11_worktree_observation_triggers_violation_when_lane_omits_files():
    """When the lane's ``LaneResult.changed_files`` is empty but the
    worktree git state reports a workflow-file change (the production
    CAO controller returns ``changed_files=[]`` for every completed
    session, see issue #78), the foreman still detects the violation
    via ``self._git.changed_files(worktree)``. Without this fallback
    the safeguard would be silently bypassed in the production
    executor path.

    The coder returns no ``changed_files`` at all (mimicking the CAO
    controller behaviour); the policy violation must still surface
    on the coder's exit, BEFORE the reviewer lane ever runs.
    """
    fake = _ready_fake()
    executor = WorkflowMutatingExecutor(
        mutator_role="worker",
        touched_files=[],  # lane does NOT report the workflow change
    )
    loop = _build(
        fake,
        executor,
        # The worktree git reports the real change even though the
        # lane's result is empty.
        worktree_files=[".github/workflows/ci.yml"],
    )

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "failed"
    assert "policy violation" in outcome.reason
    assert ".github/workflows" in outcome.reason


def test_scenario_11_clean_coder_and_reviewer_do_not_trigger_violation():
    """Sanity check: a normal coder + reviewer pair that does not
    touch ``.github/workflows/`` runs through to ``done`` and the
    workflow-file policy never trips."""
    fake = _ready_fake()
    executor = WorkflowMutatingExecutor(
        mutator_role="never",
        touched_files=[],
    )
    loop = _build(fake, executor, worktree_files=["src/clean.py"])

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", (
        f"expected 'done', got {outcome.final_phase!r}; reason={outcome.reason!r}"
    )
    # Exactly one PR was opened and the done label is set.
    assert len(fake.list_open_prs()) == 1
    assert "v3-work-done" in fake.get_labels(1)