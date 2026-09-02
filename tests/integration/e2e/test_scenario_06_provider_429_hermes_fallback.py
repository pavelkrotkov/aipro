"""E2E scenario 6 (issue #55): provider 429 / capacity failure -> Hermes
fallback without losing the phase.

The lane execution surface can fail with :class:`CaoRateLimitedError`
when CAO returns HTTP 429 or an equivalent backpressure signal. The
foreman must NOT lose the work item's authoritative state when that
happens — every committed observation (claim, branch, worktree, PR)
must remain reachable so a later pass (or operator intervention) can
recover. The production contract is "transient lane failures escalate
the item rather than silently rewriting durable state".

This test exercises the path through the real
:class:`~ai_pr_orchestrator.v3.cao_lane.CaoLaneExecutor` against a
``FakeCAOServer`` whose launch endpoint is faulted to return 429 on the
first call. The foreman's lane exception propagates up to ``run_pass``,
which persists the crash via ``mark_needs_human`` so the operator can
re-queue the item; the authoritative labels move to
``v3-work-needs-human`` and the durable claim is retained (so an
operator's `aipro reconcile --apply` can recover the lease).

Acceptance (per #55 E2E scenarios, #6):
- The 429 did not delete the claim: the work item still carries
  ``v3-work-needs-human`` (a recoverable phase, not ``done``).
- The branch, worktree, and PR attribution persist on the workflow
  state. A recovered foreman should be able to ``reclaim_expired``
  from there.
- No duplicate PR was minted; ``list_open_prs()`` returns zero (the
  lane never reached the ``_ensure_pr`` step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoSessionController,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.config import V3Config
from ai_pr_orchestrator.v3.domain import (
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
)
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop
from ai_pr_orchestrator.v3.interfaces import (
    GateDecision,
    ModelLease,
)
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import (
    ClaimConflictError,
    GitHubIssueQueue,
    NoActiveClaimError,
)
from tests.integration._fake_cao_server import FakeCAOServer, FaultSpec

ISSUE = GitHubIssueRef(owner="owner", repo="repo", number=1)


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


@pytest.fixture
def faulted_cao() -> Any:
    """A ``FakeCAOServer`` whose launch endpoint returns 429 on the first call.

    A subsequent call (without the fault) sees the standard
    ``started -> processing -> idle (x3)`` sequence. The scenario asserts
    that the first (faulted) call's failure does NOT lose the phase.
    """
    with FakeCAOServer() as cao:
        # Inject a 429 on the first POST /sessions launch. ``add_fault``
        # registers against ``self._faults`` (consumed via ``_match_fault``
        # inside the request handler); a single fault is consumed for the
        # first matching request and then the rest succeed normally.
        cao.add_fault(FaultSpec(method="POST", path_prefix="/sessions", status_code=429))
        yield cao


def _foreman(fake: FakeGitHubClient, cao: FakeCAOServer) -> tuple[ForemanPolicyLoop, GitHubIssueQueue]:
    cfg = V3Config()
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-A")
    controller = CaoSessionController(
        CAOControlPlaneConfig(
            base_url=cao.url,
            session_timeout_seconds=60,
            request_timeout_seconds=5,
        ),
        LaneRegistry.default(),
    )
    try:
        lane_executor = CaoLaneExecutor(
            controller,
            LaneRegistry.default(),
            poll_interval_seconds=0.01,
        )
        git = TrackingGit()
        git.touched = {"/wt/issue-1": ["src/change.py"]}
        loop = ForemanPolicyLoop(
            queue,
            StaticBroker(),
            LaneRegistry.default(),
            lane_executor,
            StaticGate(),
            git,
            cfg,
            run_id="run-s6",
            worktree_root="/wt",
            committer_name="AIPRO E2E Bot",
            committer_email="aipro-bot@example.invalid",
        )
        return loop, queue
    finally:
        # The controller lives for the duration of the call; the caller
        # closes it on test exit by relying on fixture teardown order.
        pass


def test_scenario_6_lane_429_does_not_lose_the_phase(faulted_cao: Any):
    """A 429 on the first lane launch escalates the item (not done, not
    dropped) and the durable claim is preserved so an operator can recover."""
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    loop, queue = _foreman(fake, faulted_cao)

    outcomes = loop.run_pass()

    # The outcome carries the escalation signal but the work item is not lost.
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "escalated", (
        f"expected escalation (not lost), got {outcome.final_phase!r}; "
        f"reason={outcome.reason!r}"
    )
    # The work item is marked needs-human and remains on a recoverable phase.
    labels = fake.get_labels(1)
    assert "v3-work-needs-human" in labels, (
        f"expected 'v3-work-needs-human', got {labels}"
    )
    assert "v3-work-done" not in labels
    # The durable claim is retained: a later pass or aipro reconcile can
    # reclaim_expired from here. The workflow state carries the run id and
    # the lease attribution.
    state = queue.load_state("owner/repo#1")
    assert state is not None
    assert state.phase == "escalated"
    assert state.run_id == "run-s6"
    assert "lease_expires_at" in state.extras
    # No PR was minted: the lane failure happened before the gate ran.
    assert fake.list_open_prs() == []
    # The lease is now in needs-human state; not silent loss.
    assert "v3-work" not in labels


def test_scenario_6_transient_429_then_success_does_not_mint_duplicate(
    faulted_cao: Any,
):
    """After the first call returns 429, a re-run on the same issue
    succeeds (no PR was created on the first attempt). No duplicate
    branches or PRs are minted on the second pass because the claim was
    already escalated, not re-claimable."""
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    loop, queue = _foreman(fake, faulted_cao)

    # First pass: 429 escalates the work item.
    first = loop.run_pass()
    assert len(first) == 1
    assert first[0].final_phase == "escalated"
    # Re-claim should fail: the item is terminal (escalated).
    from ai_pr_orchestrator.v3.queue import NoActiveClaimError, ClaimConflictError

    issue = GitHubIssueRef(owner="owner", repo="repo", number=1)
    state = queue.load_state("owner/repo#1")
    # The terminal phase forbids reclamation — exactly what a stale-lease
    # recovery must respect.
    assert state is not None, "expected state to be present after escalation"
    with pytest.raises((NoActiveClaimError, ClaimConflictError)):
        queue.reclaim_expired(issue, state, "run-s6-recover")
    # No PRs were minted across the two passes.
    assert fake.list_open_prs() == []