"""E2E scenario 12 (issue #55): two queued issues -> sequential developer
execution with parallel review fan-out.

The foreman claims multiple issues in a single pass, but the coder
lanes run sequentially (the coder is single-threaded per
:class:`~ai_pr_orchestrator.v3.lanes.LaneRegistry`). Reviewer lanes
across rounds run in parallel within their own round (each round
spawns one lane per reviewer profile).

The test:

1. Seeds two issues on the enabled label.
2. Drives the foreman through one ``run_pass``.
3. Asserts: both issues reach ``done``, exactly one PR per issue,
   exactly one coder invocation per issue (no interleaving that would
   spawn duplicate branches or PRs), and the reviewer lane ran
   multiple times (the fan-out across the three default reviewer
   profiles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.config import SafetyPolicyConfig, V3Config
from ai_pr_orchestrator.v3.domain import (
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
            evaluated_at=__import__("datetime").datetime(
                2026, 9, 1, tzinfo=__import__("datetime").UTC
            ),
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
class MultiIssueExecutor:
    """A scripted executor that records per-issue coder / reviewer
    invocations and returns no findings (the no-finding default for
    reviewer lanes). Coder calls are recorded with the workdir so the
    test can assert no duplicate worktree paths were spawned.
    """

    coder_calls: list[tuple[str, str]] = field(default_factory=list)
    reviewer_calls: list[tuple[str, str]] = field(default_factory=list)

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ):
        handle = SessionHandle(session_id="cao-multi", lane=lane.lane)
        if lane.role == "reviewer":
            self.reviewer_calls.append((lane.lane, workdir))
            return LaneResult(
                session=handle,
                exit_code=0,
                output_summary="",
                changed_files=[],
                findings=[],
            )
        self.coder_calls.append((lane.lane, workdir))
        return LaneResult(
            session=handle,
            exit_code=0,
            output_summary="",
            changed_files=["src/change.py"],
        )


def _ready_fake() -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    fake.seed_issue(2, labels=["v3-work"])
    return fake


def test_scenario_12_two_queued_issues_both_reach_done():
    """Both seeded issues reach ``done`` in a single ``run_pass`` with
    exactly one coder invocation and one PR per issue. The reviewer
    lanes run multiple times — the fan-out covers all three default
    reviewer profiles — but the coder lane is single-threaded per
    issue (no interleaving across the two)."""
    fake = _ready_fake()
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=2,
            max_reviewer_triggers_per_run=10,
        )
    )
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    executor = MultiIssueExecutor()
    git = TrackingGit()
    git.touched = {
        "/wt/issue-1": ["src/change.py"],
        "/wt/issue-2": ["src/change.py"],
    }
    loop = ForemanPolicyLoop(
        queue,
        StaticBroker(),
        LaneRegistry.default(),
        executor,
        StaticGate(),
        git,
        cfg,
        run_id="run-s12",
        worktree_root="/wt",
        committer_name="AIPRO E2E Bot",
        committer_email="aipro-bot@example.invalid",
    )

    outcomes = loop.run_pass()
    assert len(outcomes) == 2, (
        f"expected 2 outcomes (one per seeded issue), got {len(outcomes)}: "
        f"{[(o.issue.number, o.final_phase) for o in outcomes]}"
    )
    for outcome in outcomes:
        assert outcome.final_phase == "done", (
            f"expected 'done' for issue {outcome.issue.number}, got {outcome.final_phase!r}"
        )
        assert outcome.coder_invocations == 1
        assert outcome.review_rounds == 1

    # Exactly one PR per issue branch (no duplicates). The fake
    # client allocates PR numbers sequentially, so the test cannot
    # rely on ``pr.number == issue_number`` — match on ``head_ref``
    # instead, which is the branch the foreman opened.
    prs_for_1 = [pr for pr in fake.list_open_prs() if pr.head_ref == "aipro-issue-1"]
    prs_for_2 = [pr for pr in fake.list_open_prs() if pr.head_ref == "aipro-issue-2"]
    assert len(prs_for_1) == 1, f"expected 1 PR for issue 1's branch, got {prs_for_1}"
    assert len(prs_for_2) == 1, f"expected 1 PR for issue 2's branch, got {prs_for_2}"
    # Exactly one coder invocation per issue (no interleaved retries).
    coder_workdirs = sorted({w for _, w in executor.coder_calls})
    assert coder_workdirs == ["/wt/issue-1", "/wt/issue-2"], (
        f"unexpected coder workdirs: {coder_workdirs}"
    )
    # Reviewer fan-out: the default registry has 3 reviewer profiles;
    # across both issues and one round, that's 6 invocations.
    assert len(executor.reviewer_calls) == 6, (
        f"expected 6 reviewer calls (3 profiles x 2 issues), got {len(executor.reviewer_calls)}"
    )
    # Each issue reaches done with the done label.
    assert "v3-work-done" in fake.get_labels(1)
    assert "v3-work-done" in fake.get_labels(2)


def test_scenario_12_repeated_passes_do_not_duplicate_prs():
    """A second ``run_pass`` after both items reached ``done`` must be
    a no-op (no ready items), confirming the durable state is the
    authoritative source of work-item selection."""
    fake = _ready_fake()
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=2,
            max_reviewer_triggers_per_run=10,
        )
    )
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    executor = MultiIssueExecutor()
    git = TrackingGit()
    git.touched = {
        "/wt/issue-1": ["src/change.py"],
        "/wt/issue-2": ["src/change.py"],
    }
    loop = ForemanPolicyLoop(
        queue,
        StaticBroker(),
        LaneRegistry.default(),
        executor,
        StaticGate(),
        git,
        cfg,
        run_id="run-s12",
        worktree_root="/wt",
        committer_name="AIPRO E2E Bot",
        committer_email="aipro-bot@example.invalid",
    )
    first = loop.run_pass()
    assert len(first) == 2
    second = loop.run_pass()
    assert second == [], f"second pass should be no-op, got {len(second)} outcomes"
    # Exactly two PRs total (no duplicates).
    assert len(fake.list_open_prs()) == 2
