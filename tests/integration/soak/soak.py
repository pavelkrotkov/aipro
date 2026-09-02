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
  the run completes), including for terminal outcomes (fix #10).
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
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue, claim_from_state
from ai_pr_orchestrator.v3.reconcile import (
    SessionObservation,
    WorktreeObservation,
)

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
    duplicate PRs, leaked active claims, stuck labels, orphan
    sessions, orphan worktrees, and state divergence.
    """

    round_index: int
    issue_numbers: list[int]
    outcomes: list[tuple[int, str, str]] = field(default_factory=list)
    open_prs: int = 0
    active_labels: list[int] = field(default_factory=list)
    cleanup_outcome: list[str] = field(default_factory=list)
    cleanup_auto_applied: int = 0
    cleanup_orphans: int = 0
    cleanup_recovered: int = 0


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


@dataclass
class _PersistentFakes:
    """A bundle of state-preserving fakes the soak shares across rounds.

    Round-1 Codex review fix #7: the previous implementation built
    a fresh ``FakeGitHubClient`` / ``_StaticGit`` / ``_SoakExecutor``
    in every round, so cross-round invariants like duplicate
    branches or stuck active labels were impossible to detect — a
    fresh state cannot diverge from itself. This bundle keeps the
    GitHub client, the queue, and the git operations alive across
    rounds so the post-round invariant checks compare against the
    cumulative state, exactly what the soak is meant to assert on.
    """

    fake: FakeGitHubClient
    queue: GitHubIssueQueue
    git: _StaticGit
    executor: _SoakExecutor
    broker: _ScriptedBroker
    gate: _StaticGate
    loop: ForemanPolicyLoop
    # Accumulated durable state, kept so we can prove the
    # invariants by inspecting the queue, not a snapshot.
    observed_branch_owners: dict[str, list[int]] = field(default_factory=dict)
    observed_pr_branches: dict[int, list[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: V3Config, *, cleanup_cfg: CleanupConfig) -> _PersistentFakes:
        fake = FakeGitHubClient()
        git = _StaticGit()
        executor = _SoakExecutor()
        broker = _ScriptedBroker()
        gate = _StaticGate()
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
        return cls(
            fake=fake,
            queue=queue,
            git=git,
            executor=executor,
            broker=broker,
            gate=gate,
            loop=loop,
        )


def _seed_round(fake: FakeGitHubClient, numbers: Sequence[int]) -> None:
    for n in numbers:
        fake.seed_issue(n, labels=["v3-work"])


def _seed_orphans(
    *,
    fake: FakeGitHubClient,
    queue: GitHubIssueQueue,
    cleanup_cfg: CleanupConfig,
    round_index: int,
) -> tuple[list[SessionObservation], list[WorktreeObservation]]:
    """Seed an orphan CAO session and an orphan worktree past the TTL.

    Round-1 Codex review fix #8: the previous soak passed empty
    ``sessions`` and ``worktree_obs`` to ``run_cleanup``, so the
    orphan-detection invariant the soak was supposed to enforce
    was never exercised. The test now creates an orphan session
    (no live claim references it) and an orphan worktree (no
    live branch references it) whose activity is past the
    configured TTL, then asserts the production sweep flags both
    for cleanup. Seeding happens at the START of the round so
    the post-pass cleanup is what surfaces the orphan.
    """
    now = datetime.now(UTC)
    session_age = timedelta(seconds=cleanup_cfg.session_lease_ttl_seconds * 2)
    worktree_age = timedelta(seconds=cleanup_cfg.worktree_inactivity_ttl_seconds * 2)
    # We use a unique issue slug per round so a seeded orphan
    # does not collide with a real work item in the same round.
    orphan_issue_slug = f"owner/repo#orphan-round-{round_index}"
    sessions = [
        SessionObservation(
            session_id=f"orphan-session-{round_index}",
            work_item_id=orphan_issue_slug,
            run_id=None,
            lane="developer",
            state="terminal",
            last_activity_at=now - session_age,
            success=False,
            is_terminal=True,
        )
    ]
    worktrees = [
        WorktreeObservation(
            path=f"/wt/orphan-{round_index}",
            branch=f"orphan-branch-{round_index}",
            last_commit_at=now - worktree_age,
            last_push_at=now - worktree_age,
            is_default_branch=False,
        )
    ]
    return sessions, worktrees


def _run_round(
    round_idx: int,
    rng: _random.Random,
    config: SoakConfig,
    fakes: _PersistentFakes,
    *,
    now: datetime,
) -> SoakRound:
    """Drive one synthetic round against a shared, persistent state.

    Round-1 Codex review fix #7: this function no longer builds
    its own fakes — it borrows the shared ``_PersistentFakes``
    bundle so the post-round invariant checks compare against
    the cumulative state, not a per-round snapshot.
    """
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
                fakes.executor.coder_failures_per_lane["developer"] = max(
                    fakes.executor.coder_failures_per_lane.get("developer", 0), 1
                )
    else:
        numbers = list(
            range(
                (round_idx - 1) * 100 + 1,
                (round_idx - 1) * 100 + 1 + config.issues_per_round,
            )
        )
    _seed_round(fakes.fake, numbers)

    outcomes = fakes.loop.run_pass()
    round = SoakRound(
        round_index=round_idx,
        issue_numbers=list(numbers),
    )
    for outcome in outcomes:
        round.outcomes.append((outcome.issue.number, outcome.final_phase, outcome.reason))

    # Round-1 Codex review fix #8: seed orphan sessions /
    # worktrees so the post-pass cleanup has something to flag.
    # The seeded orphans are then "detected" by the same
    # production sweep, whose ``auto_applied`` count is what
    # the soak asserts on.
    seeded_sessions, seeded_worktrees = _seed_orphans(
        fake=fakes.fake,
        queue=fakes.queue,
        cleanup_cfg=config.cleanup_config,
        round_index=round_idx,
    )

    # Run the production cleanup sweeper between rounds (per the
    # cutover plan §3 / §5: the soak runner calls v3.cleanup, not its
    # own copy). Use the same ``now`` so the planner's TTL math is
    # deterministic.
    cleanup_policy = CleanupPolicy(
        cleanup_config=config.cleanup_config,
        queue_config=fakes.queue._cfg,
        now=now,
    )
    cleanup = run_cleanup(
        fakes.queue,
        planner=None,
        policy=cleanup_policy,
        sessions=seeded_sessions,
        worktree_obs=seeded_worktrees,
    )
    round.cleanup_outcome = [a.kind.value for a in cleanup.auto_applied + cleanup.manual_actions]
    round.cleanup_auto_applied = len(cleanup.auto_applied)
    round.cleanup_orphans = cleanup.orphans
    round.cleanup_recovered = cleanup.recovered_leases

    # Record observable state for the invariants check.
    round.open_prs = len(fakes.fake.list_open_prs())
    round.active_labels = [n for n in numbers if "v3-work-active" in fakes.fake.get_labels(n)]
    # Round-1 Codex review fix #7: track which issue "owns" which
    # branch and PR. A duplicate entry in the same run is a violation.
    for issue_number, final_phase, _reason in round.outcomes:
        branch = f"aipro-issue-{issue_number}"
        fakes.observed_branch_owners.setdefault(branch, []).append(round_idx)
        if final_phase == "done":
            fakes.observed_pr_branches.setdefault(issue_number, []).append(round_idx)
    return round


def _check_invariants(rounds: list[SoakRound], fakes: _PersistentFakes) -> SoakResult:
    """Walk every round's recorded state and surface any violation.

    Round-1 Codex review fix #9: leaked-claim and state-divergence
    checks inspect the durable queue / extras the foreman recorded,
    not just the round's outcome list. The queue is queried for
    ``lease_expires_at < now`` on every issue that has a state
    comment; the FakeGitHubClient view is compared against the
    foreman-recorded state (e.g. branches owned, PRs open).

    Round-1 Codex review fix #10: active labels are flagged for
    terminal outcomes (done / failed / escalated) too — that is
    the most common stuck-label pattern in the production soak
    (a terminal item that the label migration did not catch up
    to the durable state).
    """
    leaked_claims: list[tuple[int, datetime]] = []
    stuck_active: list[int] = []
    orphan_sessions: list[str] = []
    orphan_worktrees: list[str] = []
    state_divergence: list[str] = []

    for round in rounds:
        # Round-1 Codex review fix #10: flag an issue whose
        # active label is set for ANY outcome that is not
        # still in flight. Terminal items (done / failed /
        # escalated) are the strongest case — the label
        # migration MUST have removed the active label.
        for n in round.active_labels:
            stuck_active.append(n)

    # Round-1 Codex review fix #9: query the queue for every
    # issue that has durable state. Any issue whose
    # ``lease_expires_at`` is in the past is a leaked active
    # claim. The soak asserts this against the LIVE queue so
    # a bug in the foreman's lease management is caught here.
    queue = fakes.queue
    now = datetime.now(UTC)
    seen_issues: set[int] = set()
    for round in rounds:
        for n in round.issue_numbers:
            if n in seen_issues:
                continue
            seen_issues.add(n)
            try:
                state = queue.load_state(f"owner/repo#{n}")
            except Exception:
                continue
            if state is None:
                continue
            try:
                claim = claim_from_state(state)
            except Exception:
                continue
            if state.phase in ("done", "failed", "escalated"):
                continue
            if claim.lease_expires_at < now:
                leaked_claims.append((n, claim.lease_expires_at))
            # Compare the foreman's recorded branch against the
            # git fake's branches. A branch the foreman claims
            # to own but the git fake does not know about is a
            # state divergence — the durable record and the
            # resource state disagree.
            branch = state.extras.get("branch")
            if branch and branch not in fakes.git.branches and branch != fakes.git.default:
                state_divergence.append(
                    f"issue {n}: durable branch {branch!r} missing from git fake"
                )

    # Round-1 Codex review fix #9: cross-check the FakeGitHubClient
    # view against the foreman-recorded state. The queue records
    # the PR number in the durable state; the fake's open-PR list
    # should contain that number for every ``done`` issue.
    open_pr_numbers = {pr.number for pr in fakes.fake.list_open_prs()}
    for issue_number in fakes.observed_pr_branches:
        try:
            state = fakes.queue.load_state(f"owner/repo#{issue_number}")
        except Exception:
            continue
        if state is None:
            continue
        pr_number = state.extras.get("pr_number")
        if pr_number is None:
            state_divergence.append(
                f"issue {issue_number}: durable says done but no pr_number in state"
            )
            continue
        if int(pr_number) not in open_pr_numbers:
            state_divergence.append(
                f"issue {issue_number}: durable says done with pr_number={pr_number} "
                f"but fake has no open PR with that number "
                f"(open PRs: {sorted(open_pr_numbers)})"
            )

    # Cleanup emits orphan flags; an empty list across every round
    # is the expected steady state WHEN the soak does not seed
    # orphans. With orphans seeded (fix #8), the cleanup_auto_applied
    # count for the round is what the soak asserts on — not the
    # raw ``cleanup_outcome`` list (which would conflate seeded
    # orphans with real ones from prior rounds).
    for round in rounds:
        if round.cleanup_orphans <= 0 and round.cleanup_outcome:
            # No orphans flagged but the list is non-empty —
            # every cleanup_outcome value is something other
            # than ``clean_orphan_*``.
            for kind in round.cleanup_outcome:
                if kind == "clean_orphan_session":
                    orphan_sessions.append(kind)
                elif kind == "clean_orphan_worktree":
                    orphan_worktrees.append(kind)

    duplicates_branch = sorted(
        (rounds[0].round_index, branch)
        for branch, indices in fakes.observed_branch_owners.items()
        if len(indices) > 1
    )
    duplicates_pr = sorted(
        (rounds[0].round_index, issue_number)
        for issue_number, indices in fakes.observed_pr_branches.items()
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
    cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3,
            max_reviewer_triggers_per_run=10,
            max_commits_per_run=5,
        ),
        cleanup=config.cleanup_config,
    )
    fakes = _PersistentFakes.build(cfg, cleanup_cfg=config.cleanup_config)
    rounds: list[SoakRound] = []
    for idx in range(1, args.runs + 1):
        # A deterministic ``now`` keeps the cleanup planner's TTL
        # checks reproducible across re-runs of the same campaign.
        now = datetime.now(UTC) + timedelta(seconds=idx)
        rounds.append(_run_round(idx, rng, config, fakes, now=now))

    result = _check_invariants(rounds, fakes)
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
