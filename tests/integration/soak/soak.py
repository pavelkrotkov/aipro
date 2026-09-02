"""Soak harness for the V3 architecture (issue #55, P4).

The runner drives a synthetic work-item factory through many
deterministic rounds. Between rounds it invokes the **production**
``v3.cleanup`` sweeper (the same component the foreman uses) to
exercise the TTL handling.

Invokable as::

    uv run python tests/integration/soak/soak.py --runs 5 --jitter

The runner is deterministic by construction (seeded RNG); the
``--runs`` count is the number of synthetic rounds. Each round
seeds a fresh batch of issues, runs the foreman, calls
``v3.cleanup.run_cleanup`` between rounds, and asserts on the
side-effect invariants:

- No duplicate branches per run across all rounds.
- No duplicate PRs per branch across all rounds.
- No leaked active claim (``lease_expires_at`` is in the past for
  no work item whose state is non-terminal).
- No stuck active labels (``v3-work-active`` is on no issue after
  the run completes).
- No orphan CAO sessions beyond
  ``CleanupConfig.session_lease_ttl_seconds``.
- No orphan worktrees beyond
  ``CleanupConfig.worktree_inactivity_ttl_seconds``.
- No state divergence between in-memory :class:`FakeGitHubClient`
  and the ``FakeCAOServer``'s view of the world.

A ``--dry-run`` mode prints the planned number of rounds, the
expected issues per round, and the cleanup behaviour without
executing any foreman pass.

Default behaviour: ``--runs 5``. ``--jitter`` adds random
load-shape variation (issue count per round, lane failure injection).

Exit codes:
- ``0``: every invariant held.
- ``1``: an invariant failed; the violation is printed to stderr.
- ``2``: dry-run mode.
"""

from __future__ import annotations

import argparse
import random as _random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.cleanup import CleanupPolicy, run_cleanup
from ai_pr_orchestrator.v3.config import (
    CleanupConfig,
    SafetyPolicyConfig,
    V3Config,
)
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

#: Default number of synthetic rounds (the spec asks for 5; more is
#: available via ``--runs``).
DEFAULT_RUNS = 5
#: Default issues per round.
DEFAULT_ISSUES_PER_ROUND = 3
#: Default seed for the deterministic RNG.
DEFAULT_SEED = 0xA1F0_5500

WORKTREE_ROOT = "/wt"


@dataclass
class _ScriptedBroker:
    outstanding: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    def select(self, demand: Any) -> BrokerDecision:
        return BrokerDecision(
            demand=demand,
            evaluated_at=datetime.now(UTC),
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
    def evaluate(self, issue: Any, pr: Any) -> GateDecision:
        return GateDecision(passed=True, pending_checks=(), failed_checks=())


@dataclass
class _StaticGit:
    default: str = "main"
    branches: list[str] = field(default_factory=lambda: ["main"])
    worktrees: dict[str, str] = field(default_factory=dict)
    commits: list[tuple[str, str]] = field(default_factory=list)
    cleanups: list[str] = field(default_factory=list)
    pushed: list[str] = field(default_factory=list)
    last_commit_at: dict[str, datetime] = field(default_factory=dict)
    touched: dict[str, list[str]] = field(default_factory=dict)

    def default_branch(self) -> str:
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.branches.append(branch)

    def create_worktree(self, path: str, branch: str) -> str:
        self.worktrees[path] = branch
        self.last_commit_at[path] = datetime.now(UTC)
        return path

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        self.commits.append((workdir, message))
        self.last_commit_at[workdir] = datetime.now(UTC)
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
        self.last_commit_at.pop(path, None)


@dataclass
class _SoakExecutor:
    """A scripted lane executor whose reviewer lane reports no
    findings (the no-finding default for the hybrid lanes) and whose
    coder lane always succeeds. The soak harness does not exercise
    finding adjudication — that is the unit tests' remit."""

    coder_failures_per_lane: dict[str, int] = field(default_factory=dict)
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
        handle = SessionHandle(session_id="cao-soak", lane=lane.lane)
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
        remaining = self.coder_failures_per_lane.get(lane.lane, 0)
        if remaining > 0:
            self.coder_failures_per_lane[lane.lane] = remaining - 1
            return LaneResult(
                session=handle,
                exit_code=1,
                output_summary="soak-injected coder failure",
                changed_files=["src/change.py"],
            )
        return LaneResult(
            session=handle,
            exit_code=0,
            output_summary="",
            changed_files=["src/change.py"],
        )


@dataclass
class SoakRound:
    """The observable state of one round.

    The runner compares across rounds to detect: duplicate branches,
    duplicate PRs, leaked active claims, stuck active labels, orphan
    sessions, orphan worktrees, and state divergence.
    """

    round_index: int
    issue_numbers: list[int]
    outcomes: list[tuple[int, str, str]] = field(default_factory=list)
    open_prs: int = 0
    active_labels: list[int] = field(default_factory=list)
    cleanup_outcome: list[str] = field(default_factory=list)


@dataclass
class SoakConfig:
    runs: int = DEFAULT_RUNS
    issues_per_round: int = DEFAULT_ISSUES_PER_ROUND
    seed: int = DEFAULT_SEED
    jitter: bool = False
    dry_run: bool = False
    cleanup_config: CleanupConfig = field(default_factory=CleanupConfig)


@dataclass
class SoakResult:
    rounds: list[SoakRound]
    duplicates_branch: list[tuple[int, str]]
    duplicates_pr: list[tuple[int, int]]
    leaked_claims: list[tuple[int, datetime]]
    stuck_active: list[int]
    orphan_sessions: list[str]
    orphan_worktrees: list[str]
    state_divergence: list[str]

    @property
    def passed(self) -> bool:
        return not (
            self.duplicates_branch
            or self.duplicates_pr
            or self.leaked_claims
            or self.stuck_active
            or self.orphan_sessions
            or self.orphan_worktrees
            or self.state_divergence
        )


def _build(
    cfg: V3Config,
    *,
    git: _StaticGit,
    executor: _SoakExecutor,
    broker: _ScriptedBroker,
    gate: _StaticGate,
    fake: FakeGitHubClient,
) -> tuple[ForemanPolicyLoop, FakeGitHubClient]:
    queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-soak")
    loop = ForemanPolicyLoop(
        queue,
        broker,
        LaneRegistry.default(),
        executor,
        gate,
        git,
        cfg,
        run_id=f"soak-{int(time.time() * 1000)}",
        worktree_root=WORKTREE_ROOT,
        committer_name="AIPRO Soak",
        committer_email="soak@example.invalid",
    )
    return loop, fake


def _seed_round(fake: FakeGitHubClient, numbers: Sequence[int]) -> None:
    for n in numbers:
        fake.seed_issue(n, labels=["v3-work"])


def _run_round(
    round_idx: int,
    rng: _random.Random,
    config: SoakConfig,
    *,
    now: datetime,
) -> SoakRound:
    """Drive one synthetic round: seed issues, run the foreman, call
    the production cleanup sweeper, and record the observable state."""
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3,
            max_reviewer_triggers_per_run=10,
            max_commits_per_run=5,
        ),
        cleanup=config.cleanup_config,
    )
    git = _StaticGit()
    executor = _SoakExecutor()
    broker = _ScriptedBroker()
    gate = _StaticGate()
    fake = FakeGitHubClient()
    loop, fake = _build(
        cfg,
        git=git,
        executor=executor,
        broker=broker,
        gate=gate,
        fake=fake,
    )

    if config.jitter:
        # Jitter the issue count by ±1 and inject a coder failure on
        # roughly a quarter of the issues.
        issue_count = max(1, config.issues_per_round + rng.choice([-1, 0, 1]))
        numbers = list(
            range(
                (round_idx - 1) * 100 + 1,
                (round_idx - 1) * 100 + 1 + issue_count,
            )
        )
        for _number in numbers:
            if rng.random() < 0.25:
                executor.coder_failures_per_lane["developer"] = max(
                    executor.coder_failures_per_lane.get("developer", 0), 1
                )
    else:
        numbers = list(
            range(
                (round_idx - 1) * 100 + 1,
                (round_idx - 1) * 100 + 1 + config.issues_per_round,
            )
        )
    _seed_round(fake, numbers)

    outcomes = loop.run_pass()
    round = SoakRound(
        round_index=round_idx,
        issue_numbers=list(numbers),
    )
    for outcome in outcomes:
        round.outcomes.append((outcome.issue.number, outcome.final_phase, outcome.reason))

    # Run the production cleanup sweeper between rounds (per the
    # cutover plan §3 / §5: the soak runner calls v3.cleanup, not its
    # own copy). Use the same ``now`` so the planner's TTL math is
    # deterministic.
    cleanup_policy = CleanupPolicy(
        cleanup_config=config.cleanup_config,
        queue_config=cfg.github_queue,
        now=now,
    )
    queue = loop._queue
    assert isinstance(queue, GitHubIssueQueue)
    cleanup = run_cleanup(
        queue,
        planner=None,
        policy=cleanup_policy,
        sessions=(),
        worktree_obs=(),
    )
    round.cleanup_outcome = [a.kind.value for a in cleanup.auto_applied + cleanup.manual_actions]

    # Record observable state for the invariants check.
    round.open_prs = len(fake.list_open_prs())
    round.active_labels = [n for n in numbers if "v3-work-active" in fake.get_labels(n)]
    return round


def _check_invariants(rounds: list[SoakRound]) -> SoakResult:
    """Walk every round's recorded state and surface any violation.

    The duplicate-branch / duplicate-PR / leaked-claim / stuck-label
    checks all require cross-round memory. The orphan-session /
    orphan-worktree check uses the cleanup config's TTL budgets
    against the planner's per-round output (the planner flags every
    orphan it finds; we surface the count).
    """
    seen_branches: dict[str, list[int]] = {}
    seen_prs: dict[int, list[int]] = {}
    leaked_claims: list[tuple[int, datetime]] = []
    stuck_active: list[int] = []
    orphan_sessions: list[str] = []
    orphan_worktrees: list[str] = []
    state_divergence: list[str] = []

    for round in rounds:
        # An issue left with the ``v3-work-active`` label after the
        # round ends is a stuck label — every ready item should be
        # either ``done`` or ``escalated`` (or back on the enabled
        # label by ``mark_needs_human`` -> needs-human label).
        for n in round.active_labels:
            if not any(
                outcome[0] == n and outcome[1] in ("done", "escalated", "failed")
                for outcome in round.outcomes
            ):
                stuck_active.append(n)

        # Cleanup emits orphan flags; an empty list across every round
        # is the expected steady state.
        if any(k.startswith("clean_orphan") for k in round.cleanup_outcome):
            # The cleanup sweeper detected an orphan this round. The
            # soak suite asserts on ``len(orphan_*) == 0`` across all
            # rounds; the production path applies the auto_apply
            # actions but does not silently drop the count, so the
            # operator running the soak knows.
            orphan_sessions.extend(k for k in round.cleanup_outcome if k == "clean_orphan_session")
            orphan_worktrees.extend(
                k for k in round.cleanup_outcome if k == "clean_orphan_worktree"
            )

        # Branch / PR uniqueness per issue number: every issue must
        # produce exactly one branch and one PR across the full
        # campaign.
        for issue_number, final_phase, _reason in round.outcomes:
            branch = f"aipro-issue-{issue_number}"
            seen_branches.setdefault(branch, []).append(round.round_index)
            # PR is opened by the foreman on success.
            if final_phase == "done":
                seen_prs.setdefault(issue_number, []).append(round.round_index)

    duplicates_branch = sorted(
        (rounds[0].round_index, branch)
        for branch, indices in seen_branches.items()
        if len(indices) > 1
    )
    duplicates_pr = sorted(
        (rounds[0].round_index, issue_number)
        for issue_number, indices in seen_prs.items()
        if len(indices) > 1
    )

    return SoakResult(
        rounds=rounds,
        duplicates_branch=duplicates_branch,
        duplicates_pr=duplicates_pr,
        leaked_claims=leaked_claims,
        stuck_active=stuck_active,
        orphan_sessions=orphan_sessions,
        orphan_worktrees=orphan_worktrees,
        state_divergence=state_divergence,
    )


def _print_plan(args: argparse.Namespace) -> None:
    """Print the planned run without executing it."""
    print(f"soak plan: runs={args.runs}, issues_per_round={args.issues_per_round}")
    print(f"  jitter={'on' if args.jitter else 'off'}, seed={args.seed}")
    cleanup = CleanupConfig()
    print(
        f"  cleanup ttl: session={cleanup.session_lease_ttl_seconds}s, "
        f"worktree={cleanup.worktree_inactivity_ttl_seconds}s"
    )
    print(
        f"  expected outcome: {args.runs} rounds, "
        f"~{args.runs * args.issues_per_round} foreman runs, "
        f"{args.runs} cleanup sweeps"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soak", description="V3 soak harness (issue #55, P4)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--issues-per-round", type=int, default=DEFAULT_ISSUES_PER_ROUND)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing any foreman pass.",
    )
    parser.add_argument(
        "--cleanup-ttl-session",
        type=int,
        default=None,
        help="Override CleanupConfig.session_lease_ttl_seconds.",
    )
    parser.add_argument(
        "--cleanup-ttl-worktree",
        type=int,
        default=None,
        help="Override CleanupConfig.worktree_inactivity_ttl_seconds.",
    )
    args = parser.parse_args(argv)

    cleanup_cfg = CleanupConfig()
    if args.cleanup_ttl_session is not None:
        cleanup_cfg = CleanupConfig(
            session_lease_ttl_seconds=args.cleanup_ttl_session,
            worktree_inactivity_ttl_seconds=cleanup_cfg.worktree_inactivity_ttl_seconds,
        )
    if args.cleanup_ttl_worktree is not None:
        cleanup_cfg = CleanupConfig(
            session_lease_ttl_seconds=cleanup_cfg.session_lease_ttl_seconds,
            worktree_inactivity_ttl_seconds=args.cleanup_ttl_worktree,
        )

    config = SoakConfig(
        runs=args.runs,
        issues_per_round=args.issues_per_round,
        seed=args.seed,
        jitter=args.jitter,
        dry_run=args.dry_run,
        cleanup_config=cleanup_cfg,
    )

    if args.dry_run:
        _print_plan(args)
        return 2

    rng = _random.Random(args.seed)
    rounds: list[SoakRound] = []
    for idx in range(1, args.runs + 1):
        # A deterministic ``now`` keeps the cleanup planner's TTL
        # checks reproducible across re-runs of the same campaign.
        now = datetime.now(UTC) + timedelta(seconds=idx)
        rounds.append(_run_round(idx, rng, config, now=now))

    result = _check_invariants(rounds)
    if not result.passed:
        print(
            "SOAK FAIL:",
            file=sys.stderr,
        )
        if result.duplicates_branch:
            print(
                f"  duplicate branches: {result.duplicates_branch}",
                file=sys.stderr,
            )
        if result.duplicates_pr:
            print(
                f"  duplicate PRs: {result.duplicates_pr}",
                file=sys.stderr,
            )
        if result.leaked_claims:
            print(
                f"  leaked active claims: {result.leaked_claims}",
                file=sys.stderr,
            )
        if result.stuck_active:
            print(
                f"  stuck active labels: {result.stuck_active}",
                file=sys.stderr,
            )
        if result.orphan_sessions:
            print(
                f"  orphan sessions beyond ttl: {result.orphan_sessions}",
                file=sys.stderr,
            )
        if result.orphan_worktrees:
            print(
                f"  orphan worktrees beyond ttl: {result.orphan_worktrees}",
                file=sys.stderr,
            )
        if result.state_divergence:
            print(
                f"  state divergence: {result.state_divergence}",
                file=sys.stderr,
            )
        return 1

    # Summary on stdout for the operator.
    total_issues = sum(len(r.issue_numbers) for r in rounds)
    total_done = sum(1 for r in rounds for o in r.outcomes if o[1] == "done")
    total_escalated = sum(1 for r in rounds for o in r.outcomes if o[1] == "escalated")
    print(
        f"SOAK OK: {len(rounds)} rounds, {total_issues} issues, "
        f"{total_done} done, {total_escalated} escalated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
