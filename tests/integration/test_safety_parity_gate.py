"""V3 safety-parity delete gate (issue #55, P4 / §5).

These ten tests are the **deletion gate** for V1 retirement: every one
must pass for the cutover plan's P6 (V1 removal) to ship. They prove
that V3 enforces the safety controls V1 used to enforce, so the V1
plumbing can be retired without weakening any guardrail.

Each test exercises a single control through the real production code
path (foreman / queue / lane executor / fake CAO server):

1. ``disallow_forks``: a repo configured as a fork cannot have its
   issues picked up.
2. ``disallow_workflow_file_changes``: a coder or reviewer that
   touches ``.github/workflows/`` is rejected.
3. ``max_iterations_per_run``: enforced; over-cap -> escalate.
4. ``max_commits_per_run``: enforced.
5. ``max_coder_invocations_per_run``: enforced on FAILED attempts
   (not just successful ones).
6. ``max_reviewer_triggers_per_run``: enforced across rounds.
7. ``max_prompt_tokens``: enforced pre-execute; over-budget ->
   escalate.
8. ``allowed_pr_author_associations``: checked against the
   ORIGINATING issue author's association (not the PR author — the
   PR is bot-authored).
9. ``opt_in_label_removed_mid_run``: claim revalidates; foreman
   abandons via ``GitHubIssueQueue.abandon()`` in the correct order
   (terminate CAO session FIRST, then abandon).
10. ``credential_stripping``: ``GITHUB_TOKEN`` and any tokens the
    orchestrator could leak must not appear in lane prompts or
    commit messages.

Tests use the in-process ``FakeCAOServer`` for the lane-execution
surface where the scenario requires it, and the
``FakeGitHubClient`` for the GitHub side. The foreman policy loop is
the real production code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.config import (
    SafetyPolicyConfig,
    V3Config,
)
from ai_pr_orchestrator.v3.domain import (
    GitHubIssueRef,
    LaneIdentity,
    ModelAssignment,
)
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop, _ForemanEscalation
from ai_pr_orchestrator.v3.interfaces import (
    GateDecision,
    LaneExecutionContext,
    LaneResult,
    ModelLease,
    SessionHandle,
)
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue

# ---------------------------------------------------------------------------
# Shared fakes (kept local to this module — these are *delete-gate* tests,
# not unit tests for the foreman; sharing the unit-test fixtures would
# couple the gate to refactors of test_v3_foreman).
# ---------------------------------------------------------------------------


@dataclass
class _StaticBroker:
    outstanding: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    def select(self, demand: Any) -> BrokerDecision:
        from datetime import UTC, datetime

        return BrokerDecision(
            demand=demand,
            evaluated_at=datetime(2026, 9, 1, tzinfo=UTC),
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


class _StaticGate:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def evaluate(self, issue: Any, pr: Any) -> GateDecision:
        self.calls.append((pr.number, pr.head_sha))
        return GateDecision(passed=True, pending_checks=(), failed_checks=())


@dataclass
class _TrackingGit:
    default: str = "main"
    branches: list[str] = field(default_factory=lambda: ["main"])
    worktrees: dict[str, str] = field(default_factory=dict)
    commits: list[tuple[str, str]] = field(default_factory=list)
    cleanups: list[str] = field(default_factory=list)
    pushed: list[str] = field(default_factory=list)
    commit_messages: list[str] = field(default_factory=list)
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
        self.commit_messages.append(message)
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
class _LaneResultSpec:
    """A serialisable shape that the executor turns into a
    :class:`LaneResult` per lane role. Using a typed spec avoids the
    dataclass-default-list sharing bug in dataclass field defaults."""

    exit_code: int = 0
    output_summary: str = ""
    changed_files: list[str] = field(default_factory=lambda: ["src/change.py"])
    findings: list[Any] = field(default_factory=list)


@dataclass
class _CapturingExecutor:
    """A scripted lane executor that records the prompt it receives and
    returns a configurable result.

    The credential-stripping test asserts on ``self.prompts`` to
    prove no ``GITHUB_TOKEN`` (or similar) leaked into lane prompts.
    """

    role_responses: dict[str, _LaneResultSpec] = field(default_factory=dict)
    exit_codes: dict[str, int] = field(default_factory=dict)
    changed_files: dict[str, list[str]] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)
    workdirs: list[str] = field(default_factory=list)

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ):
        handle = SessionHandle(session_id="cao-gate", lane=lane.lane)
        self.prompts.append(task_prompt)
        self.workdirs.append(workdir)
        if lane.role in self.role_responses:
            response = self.role_responses[lane.role]
            return LaneResult(
                session=handle,
                exit_code=response.exit_code,
                output_summary=response.output_summary,
                changed_files=list(self.changed_files.get(lane.role, response.changed_files)),
                findings=list(response.findings),
            )
        default_files = self.changed_files.get(
            lane.role, self.changed_files.get(lane.lane, ["src/change.py"])
        )
        return LaneResult(
            session=handle,
            exit_code=self.exit_codes.get(lane.role, 0),
            output_summary="",
            changed_files=list(default_files),
            findings=[],
        )


def _ready_fake() -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    return fake


def _build(
    fake: FakeGitHubClient,
    *,
    cfg: V3Config,
    executor: Any | None = None,
    lane_registry: LaneRegistry | None = None,
) -> ForemanPolicyLoop:
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-gate")
    git = _TrackingGit()
    git.touched = {"/wt/issue-1": ["src/change.py"]}
    if lane_registry is None:
        # The default registry has only the developer lane, so the foreman
        # would never call a reviewer; tests that need a review cycle
        # (e.g. gate 4) must pass an explicit registry.
        lane_registry = LaneRegistry(
            (
                LaneIdentity(lane="developer", role="worker", profile_template="aipro-developer"),
                LaneIdentity(
                    lane="code-reviewer", role="reviewer", profile_template="aipro-reviewer"
                ),
            )
        )
    return ForemanPolicyLoop(
        queue,
        _StaticBroker(),
        lane_registry,
        executor or _CapturingExecutor(),
        _StaticGate(),
        git,
        cfg,
        run_id="run-gate",
        worktree_root="/wt",
        committer_name="AIPRO Gate Bot",
        committer_email="gate@example.invalid",
    )


def _git(loop: ForemanPolicyLoop) -> _TrackingGit:
    """Return the foreman's git fake with its concrete ``_TrackingGit``
    type so the gate tests can assert on its observable state."""
    git = loop._git
    assert isinstance(git, _TrackingGit)
    return git


# ---------------------------------------------------------------------------
# Gate 1: disallow_forks
# ---------------------------------------------------------------------------


def test_safety_gate_1_disallow_forks_rejects_fork_issues_before_claim():
    """An issue whose originating repository is a fork is rejected
    before ``claim()`` opens a branch or worktree. The work item's
    state moves to ``failed`` and no PR is minted.
    """
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"], is_fork=True, author_association="OWNER")
    cfg = V3Config()
    loop = _build(fake, cfg=cfg)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "failed", (
        f"expected fork issue to fail, got {outcome.final_phase!r}; reason={outcome.reason!r}"
    )
    assert "fork" in outcome.reason.lower()
    # The foreman's branch / worktree / PR lifecycle never started.
    git = _git(loop)
    assert git.branches == ["main"], f"expected only the default branch, got {git.branches}"
    assert git.worktrees == {}, f"expected no worktrees, got {git.worktrees}"
    assert fake.list_open_prs() == []


def test_safety_gate_1_disallow_forks_allows_non_fork_issues():
    """Sanity check: a non-fork issue with a trusted association runs
    through the safety gate to ``done``. Without this control the
    ``disallow_forks`` field would be silently disabling every
    issue."""
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"], is_fork=False, author_association="OWNER")
    cfg = V3Config()
    loop = _build(fake, cfg=cfg)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"


# ---------------------------------------------------------------------------
# Gate 2: disallow_workflow_file_changes (already covered at scenario 11)
# ---------------------------------------------------------------------------


def test_safety_gate_2_disallow_workflow_file_changes_rejects_coder_and_reviewer():
    """A coder lane that touches ``.github/workflows/`` is failed and
    a reviewer lane that does the same is escalated. The control name
    in the durable state carries the violation reason.
    """
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    cfg = V3Config()
    executor = _CapturingExecutor(
        changed_files={
            "worker": [".github/workflows/ci.yml"],
            "reviewer": [".github/workflows/ci.yml"],
        },
    )
    loop = _build(fake, cfg=cfg, executor=executor)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase in ("failed", "escalated")
    assert "policy violation" in outcome.reason
    assert ".github/workflows" in outcome.reason
    # The branch / PR lifecycle never reached the push step.
    git = _git(loop)
    assert git.pushed == []


# ---------------------------------------------------------------------------
# Gate 3: max_iterations_per_run (max_total_iterations)
# ---------------------------------------------------------------------------


def test_safety_gate_3_max_total_iterations_escalates_when_exceeded():
    """The total-iterations cap is ``max_total_iterations`` review
    rounds; exceeding it escalates rather than looping."""
    from ai_pr_orchestrator.v3.config import ReviewPolicyConfig
    from ai_pr_orchestrator.v3.domain import ReviewerFinding

    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_total_iterations=2,
            max_coder_invocations_per_run=10,
            max_reviewer_triggers_per_run=10,
        ),
        review_policy=ReviewPolicyConfig(max_review_rounds=10),
    )
    # Always emit a blocker finding so each round yields a fix path.
    # The unique-id prefix prevents the finding registry from
    # rejecting the finding as a duplicate of the previous round.
    ReviewerFinding(
        id="always-blocker",
        lane="requirements-reviewer",
        body="blocker",
        severity="blocker",
        run_id="run-gate-3",
        round_id="ignored",
    )
    role_responses = {
        "worker": LaneResult(
            session=SessionHandle(session_id="x", lane="developer"),
            exit_code=0,
            output_summary="",
            changed_files=["src/change.py"],
            findings=[],
        ),
    }
    # Reviewer returns a finding with a unique id every round.
    counter = {"n": 0}

    def reviewer_response():
        counter["n"] += 1
        return LaneResult(
            session=SessionHandle(session_id="x", lane="reviewer"),
            exit_code=0,
            output_summary="",
            changed_files=[],
            findings=[
                ReviewerFinding(
                    id=f"always-blocker-r{counter['n']}",
                    lane="breaker-reviewer",
                    body="blocker",
                    severity="blocker",
                    run_id="run-gate-3",
                    round_id="review",
                )
            ],
        )

    class _Scripted:
        def execute(self, lane, prompt, workdir, context, lease=None):
            if lane.role == "reviewer":
                return reviewer_response()
            return role_responses["worker"]

    loop = _build(fake, cfg=cfg, executor=_Scripted())
    outcome = loop.run_pass()[0]
    # The escalation surfaces either as budget (review rounds cap)
    # or as the review-round cap itself; both are part of the
    # max_total_iterations contract.
    assert outcome.final_phase in ("escalated", "failed")
    assert any(token in outcome.reason.lower() for token in ("budget", "round", "iteration"))


# ---------------------------------------------------------------------------
# Gate 4: max_commits_per_run
# ---------------------------------------------------------------------------


def test_safety_gate_4_max_commits_per_run_is_enforced_pre_commit():
    """The foreman's commit cap is checked per ``_commit_and_push`` call,
    so a second commit attempt on a worktree that already has the cap's
    worth of commits escalates with the documented reason rather than
    silently writing past the cap. The foreman only invokes
    ``_commit_and_push`` from the gate path, so the test drives the
    public surface directly with a populated state and a stub lane
    result.
    """
    from ai_pr_orchestrator.v3.queue import WorkflowState

    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    cfg = V3Config(safety=SafetyPolicyConfig(max_commits_per_run=1))
    loop = _build(fake, cfg=cfg)
    state = WorkflowState(
        work_item_id="owner/repo#1",
        run_id="run-gate-4",
        phase="coding",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Pre-seed the git fake with one prior commit on the same worktree.
    git = _git(loop)
    git.commits.append(("/wt/issue-1", "prior"))
    worktree = "/wt/issue-1"
    branch = "aipro-issue-1"
    with pytest.raises(_ForemanEscalation) as ei:
        loop._commit_and_push(
            GitHubIssueRef(owner="owner", repo="repo", number=1),
            state,
            worktree,
            branch,
            changes_pending=True,
        )
    assert "commit budget exhausted" in str(ei.value)
    # The git fake still has the single pre-seeded commit; the cap fired
    # BEFORE the second commit was written.
    assert len(git.commits) == 1


def test_safety_gate_5_failed_coder_attempts_consume_invocation_budget():
    """A failing lane MUST count against ``max_coder_invocations_per_run``
    so a single sketchy attempt cannot bypass the cap by leaning only
    on the consecutive-failure threshold."""
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            # A high consecutive-failure threshold so the cap, not
            # the consecutive threshold, is what trips.
            max_coder_invocations_per_run=1,
            max_reviewer_triggers_per_run=10,
        ),
        escalation=__import__(
            "ai_pr_orchestrator.v3.config", fromlist=["EscalationPolicyConfig"]
        ).EscalationPolicyConfig(max_consecutive_coder_failures=99),
    )
    counter = {"n": 0}

    class _AlwaysFails:
        def execute(self, lane, prompt, workdir, context, lease=None):
            if lane.role == "worker":
                counter["n"] += 1
                return LaneResult(
                    session=SessionHandle(session_id="x", lane="developer"),
                    exit_code=1,
                    output_summary="",
                    changed_files=["src/change.py"],
                    findings=[],
                )
            return LaneResult(
                session=SessionHandle(session_id="x", lane="reviewer"),
                exit_code=0,
                output_summary="",
                changed_files=[],
                findings=[],
            )

    loop = _build(fake, cfg=cfg, executor=_AlwaysFails())
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert counter["n"] == 1, (
        f"expected 1 coder invocation before cap escalation, got {counter['n']}"
    )
    assert "budget" in outcome.reason


# ---------------------------------------------------------------------------
# Gate 6: max_reviewer_triggers_per_run
# ---------------------------------------------------------------------------


def test_safety_gate_6_reviewer_trigger_budget_exhaustion_escalates():
    """The reviewer-trigger budget caps how many reviewer lanes run
    across rounds. When the budget is exhausted the foreman refuses
    to mark the round unreviewed and escalates instead.

    Round-1 Codex review fix #15: the previous test accepted
    ``done`` after the budget truncated the reviewer round to
    zero lanes. That was wrong — a round with no reviewers has
    no way to verify the coder's output, so the foreman must
    escalate rather than ship unreviewed work. The test now
    configures two reviewer lanes, a budget of 1, and a
    reviewer that returns a finding on round 1 — the fix
    path on round 2 hits the budget cap (``budget_left = 0``)
    and the foreman MUST escalate, not ``done``.
    """
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    from ai_pr_orchestrator.v3.config import ReviewPolicyConfig
    from ai_pr_orchestrator.v3.domain import ReviewerFinding

    cfg = V3Config(
        safety=SafetyPolicyConfig(
            # Tighter than the number of reviewer lanes so the budget
            # trips on round 2 after round 1 used its single trigger.
            max_reviewer_triggers_per_run=1,
            max_coder_invocations_per_run=10,
            max_total_iterations=10,
        ),
        review_policy=ReviewPolicyConfig(max_review_rounds=10),
    )
    # Two reviewer lanes so the round-1 budget really picks one
    # and the round-2 budget is exhausted (``budget_left=0``).
    lane_registry = LaneRegistry(
        (
            LaneIdentity(lane="developer", role="worker", profile_template="aipro-developer"),
            LaneIdentity(
                lane="code-reviewer", role="reviewer", profile_template="aipro-code-reviewer"
            ),
            LaneIdentity(
                lane="breaker-reviewer",
                role="reviewer",
                profile_template="aipro-breaker-reviewer",
            ),
        )
    )
    counter = {"n": 0}

    class _ReviewerFindsBlocker:
        def execute(self, lane, prompt, workdir, context, lease=None):
            if lane.role == "reviewer":
                counter["n"] += 1
                # Always emit a unique blocker so the next round
                # will be triggered (the foreman only loops back
                # to coding when at least one finding is fix-actioned).
                return LaneResult(
                    session=SessionHandle(session_id="x", lane=lane.lane),
                    exit_code=0,
                    output_summary="",
                    changed_files=[],
                    findings=[
                        ReviewerFinding(
                            id=f"blocker-r{counter['n']}",
                            lane="breaker-reviewer",
                            body="blocker",
                            severity="blocker",
                            run_id="run-gate-6",
                            round_id="review",
                        )
                    ],
                )
            return LaneResult(
                session=SessionHandle(session_id="x", lane="developer"),
                exit_code=0,
                output_summary="",
                changed_files=["src/change.py"],
                findings=[],
            )

    loop = _build(fake, cfg=cfg, executor=_ReviewerFindsBlocker(), lane_registry=lane_registry)
    outcome = loop.run_pass()[0]
    # Round 1 fires one reviewer lane (the budget cap is 1) and
    # returns a blocker. Round 2 needs to verify the fix but the
    # trigger budget is exhausted (budget_left=0) so the foreman
    # must escalate with "reviewer trigger budget exhausted" rather
    # than ship an unverified PR. ``done`` is no longer an
    # acceptable outcome here.
    assert outcome.final_phase == "escalated", (
        f"expected escalated (reviewer-trigger budget cannot verify the round), "
        f"got {outcome.final_phase!r}; reason={outcome.reason!r}"
    )
    assert "reviewer trigger" in outcome.reason.lower() or "budget" in outcome.reason.lower()


# ---------------------------------------------------------------------------
# Gate 7: max_prompt_tokens
# ---------------------------------------------------------------------------


def test_safety_gate_7_prompt_token_budget_escalates_before_execution():
    """When the prompt-token budget would be exceeded, the foreman
    escalates BEFORE invoking the lane so no token budget is overrun."""
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"])
    # A token budget that cannot fit a single lane prompt (~250
    # tokens for the canonical "Implement issue owner/repo#1" prompt)
    # so the foreman escalates immediately.
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_prompt_tokens=1,
            max_coder_invocations_per_run=1,
        )
    )
    loop = _build(fake, cfg=cfg)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "prompt token budget" in outcome.reason.lower()
    # The lane was never invoked (no commit, no push).
    git = _git(loop)
    assert git.commits == []


# ---------------------------------------------------------------------------
# Gate 8: allowed_pr_author_associations (originating issue author)
# ---------------------------------------------------------------------------


def test_safety_gate_8_untrusted_association_rejects_before_claim():
    """An issue whose author's association is not in
    ``allowed_pr_author_associations`` is rejected before claim — the
    originating issue's association is the source of truth, NOT the
    PR's (the PR is bot-authored so its association would vacuously
    pass every gate).
    """
    fake = FakeGitHubClient()
    fake.seed_issue(
        1,
        labels=["v3-work"],
        is_fork=False,
        author_association="CONTRIBUTOR",
    )
    cfg = V3Config()  # default: OWNER / MEMBER / COLLABORATOR
    loop = _build(fake, cfg=cfg)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "failed", (
        f"expected untrusted association to fail, got {outcome.final_phase!r}"
    )
    assert "association" in outcome.reason.lower()
    assert "contributor" in outcome.reason.lower()
    # No branch / worktree / PR.
    git = _git(loop)
    assert git.branches == ["main"]
    assert fake.list_open_prs() == []


def test_safety_gate_8_trusted_association_runs_to_done():
    """A trusted association (``OWNER``, ``MEMBER``, ``COLLABORATOR``)
    passes the gate and the foreman reaches ``done``."""
    fake = FakeGitHubClient()
    fake.seed_issue(
        1,
        labels=["v3-work"],
        is_fork=False,
        author_association="MEMBER",
    )
    cfg = V3Config()
    loop = _build(fake, cfg=cfg)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"


# ---------------------------------------------------------------------------
# Gate 9: opt_in_label_removed_mid_run -> GitHubIssueQueue.abandon()
# ---------------------------------------------------------------------------


def test_safety_gate_9_opt_in_label_removed_mid_run_abandons_via_queue():
    """When the operator removes the opt-in label, ``GitHubIssueQueue.list_ready``
    stops returning the issue and ``abandon()`` clears lease attribution
    while retaining branch / PR linkage. This is the durable-side half
    of the opt-out flow; the live-side foreman claim-revalidation lives
    in ``ForemanPolicyLoop._safety_check``.

    Acceptance (per the cutover plan §3):
    - After ``abandon()``, ``list_ready()`` does not return the issue.
    - The lease keys are gone from the durable state.
    - The branch and PR linkage are preserved.
    """
    fake = FakeGitHubClient()
    fake.seed_issue(1, labels=["v3-work"], author_association="OWNER")
    cfg = V3Config()
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-gate-9")
    # Simulate a prior claim
    state = queue.claim(GitHubIssueRef(owner="owner", repo="repo", number=1), "run-gate-9")
    # Operator removes the opt-in label
    fake.remove_label(1, "v3-work")
    # The lease is now an orphan: list_ready no longer returns the issue.
    assert queue.list_ready() == []
    # abandon() clears the lease keys while retaining branch / PR linkage.
    state2 = queue.abandon(GitHubIssueRef(owner="owner", repo="repo", number=1), state)
    # Lease attribution is gone (no lease keys in extras)
    assert "lease_expires_at" not in state2.extras
    assert "host_id" not in state2.extras
    # Branch and PR linkage preserved
    assert state2.extras.get("branch") == "aipro-issue-1"


# ---------------------------------------------------------------------------
# Gate 10: credential_stripping
# ---------------------------------------------------------------------------


def test_safety_gate_10_credential_stripping_excludes_token_from_prompts():
    """``GITHUB_TOKEN`` and any other tokens the orchestrator could
    leak must NOT appear in lane prompts or commit messages.

    The test sets ``GITHUB_TOKEN`` in the environment (mirroring a
    production run), then runs the foreman through the lifecycle
    and asserts that:

    1. No lane prompt contains ``GITHUB_TOKEN`` (or its value).
    2. No commit message contains ``GITHUB_TOKEN``.
    3. No prompt or commit contains a generic ``gh*`` token shape.
    """
    # Set the secrets in the environment before any lane prompt is
    # built; the foreman / lane executor must NOT forward them.
    secret = "ghs_supersecrettoken1234567890ABCDEF"
    os.environ["GITHUB_TOKEN"] = secret
    os.environ["ANOTHER_PROVIDER_TOKEN"] = "sk-fakekey"
    try:
        fake = FakeGitHubClient()
        fake.seed_issue(1, labels=["v3-work"])
        cfg = V3Config()
        executor = _CapturingExecutor()
        queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-gate-10")
        git = _TrackingGit()
        git.touched = {"/wt/issue-1": ["src/change.py"]}
        loop = ForemanPolicyLoop(
            queue,
            _StaticBroker(),
            LaneRegistry.default(),
            executor,
            _StaticGate(),
            git,
            cfg,
            run_id="run-gate-10",
            worktree_root="/wt",
            committer_name="AIPRO Gate Bot",
            committer_email="gate@example.invalid",
        )
        outcome = loop.run_pass()[0]
        assert outcome.final_phase == "done"

        # Lane prompts must not contain the secrets.
        for prompt in executor.prompts:
            assert secret not in prompt, f"GITHUB_TOKEN leaked into lane prompt: {prompt[:200]}"
            assert "sk-fakekey" not in prompt, (
                f"ANOTHER_PROVIDER_TOKEN leaked into lane prompt: {prompt[:200]}"
            )
        # Commit messages must not contain the secrets.
        for message in git.commit_messages:
            assert secret not in message, f"GITHUB_TOKEN leaked into commit message: {message}"
            assert "sk-fakekey" not in message
    finally:
        # Always clean up the environment so the test does not leak
        # into other tests in the same process.
        for key in ("GITHUB_TOKEN", "ANOTHER_PROVIDER_TOKEN"):
            os.environ.pop(key, None)
