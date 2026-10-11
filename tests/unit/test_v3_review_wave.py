"""Reviewer-wave orchestration for issue #49."""

from __future__ import annotations

import threading

from ai_pr_orchestrator.v3.broker import BrokerError
from ai_pr_orchestrator.v3.interfaces import StateConflictError
from tests.unit.test_v3_foreman import (
    FakeBroker,
    RecordingGit,
    ScriptedExecutor,
    _finding,
    _foreman,
    _gate,
    _ready_fake,
)


class RecordingBroker(FakeBroker):
    def __init__(self) -> None:
        super().__init__()
        self.demands = []

    def select(self, demand):
        self.demands.append(demand)
        return super().select(demand)


class ConcurrentExecutor(ScriptedExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.barrier = threading.Barrier(3)
        self.seen = {}
        self.lock = threading.Lock()

    def execute(self, lane, task_prompt, workdir, context, lease=None, *, resume=False):
        if lane.role == "reviewer":
            self.barrier.wait(timeout=2)
            with self.lock:
                self.seen[lane.lane] = (workdir, context, task_prompt, lease)
        return super().execute(lane, task_prompt, workdir, context, lease)


def test_default_reviewers_are_blind_isolated_concurrent_and_diverse():
    fake = _ready_fake()
    fake._issue_bodies[1] = "Acceptance: preserve the exact task specification."
    executor = ConcurrentExecutor()
    broker = RecordingBroker()
    git = RecordingGit()
    loop, queue = _foreman(fake, executor, _gate(), git=git, broker=broker)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done"
    assert set(executor.seen) == {
        "requirements-reviewer",
        "breaker-reviewer",
        "architecture-reviewer",
    }

    seen = executor.seen
    workdirs = {row[0] for row in seen.values()}
    assert len(workdirs) == 3
    assert "/wt/issue-1" not in workdirs
    assert {row[1].head_sha for row in seen.values()} == {"sha"}
    assert all(
        "Acceptance: preserve the exact task specification." in row[2] for row in seen.values()
    )
    assert all("implemented requested change" not in row[2] for row in seen.values())
    assert "Requirements prosecutor:" in seen["requirements-reviewer"][2]
    assert "Breaker:" in seen["breaker-reviewer"][2]
    assert "Architecture skeptic:" in seen["architecture-reviewer"][2]

    demands = [d for d in broker.demands if d.role == "reviewer"]
    assert [d.peers for d in demands] == [
        (),
        ("ref-requirements-reviewer",),
        ("ref-requirements-reviewer", "ref-breaker-reviewer"),
    ]
    assert demands[1].in_flight == {"ref-requirements-reviewer": 1}
    assert demands[2].in_flight == {
        "ref-requirements-reviewer": 1,
        "ref-breaker-reviewer": 1,
    }

    state = queue.load_state("owner/repo#1")
    assert state is not None
    wave = state.extras["review_waves"]["review-1"]
    lanes = wave["lanes"]
    assert wave["head_sha"] == "sha"
    assert {v["status"] for v in lanes.values()} == {"completed"}
    assert {name: record["assignment"]["model_ref"] for name, record in lanes.items()} == {
        "requirements-reviewer": "ref-requirements-reviewer",
        "breaker-reviewer": "ref-breaker-reviewer",
        "architecture-reviewer": "ref-architecture-reviewer",
    }
    assert all(path not in git.worktrees for path in workdirs)
    assert {"developer", "requirements-reviewer", "breaker-reviewer", "architecture-reviewer"} <= {
        handle.lane for handle in executor.terminated
    }
    assert executor.terminated[-1].lane == "developer"


class BreakerCrashExecutor(ScriptedExecutor):
    def execute(self, lane, task_prompt, workdir, context, lease=None, *, resume=False):
        if lane.lane == "breaker-reviewer":
            raise RuntimeError("breaker exploded")
        return super().execute(lane, task_prompt, workdir, context, lease)


def test_reviewer_failure_preserves_sibling_results_and_retains_uncertain_checkout():
    fake = _ready_fake()
    executor = BreakerCrashExecutor()
    git = RecordingGit()
    loop, queue = _foreman(fake, executor, _gate(), git=git)

    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "breaker exploded" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state is not None
    wave = state.extras["review_waves"]["review-1"]
    assert wave["lanes"]["requirements-reviewer"]["status"] == "completed"
    assert wave["lanes"]["architecture-reviewer"]["status"] == "completed"
    assert wave["lanes"]["breaker-reviewer"]["status"] == "uncertain"
    assert "/wt/issue-1-review-1-breaker-reviewer" in git.worktrees
    assert "/wt/issue-1-review-1-requirements-reviewer" not in git.worktrees
    assert "/wt/issue-1-review-1-architecture-reviewer" not in git.worktrees
    assert git.pushed == []


def test_result_persistence_failure_retains_review_resources(monkeypatch):
    fake = _ready_fake()
    executor = ScriptedExecutor()
    git = RecordingGit()
    loop, queue = _foreman(fake, executor, _gate(), git=git)
    save = queue.save_state

    def reject_results(state, expected_updated_at):
        waves = state.extras.get("review_waves")
        if isinstance(waves, dict) and any(
            lane.get("status") == "completed"
            for wave in waves.values()
            if isinstance(wave, dict)
            for lane in wave.get("lanes", {}).values()
        ):
            raise StateConflictError("review result persistence unavailable")
        save(state, expected_updated_at)

    monkeypatch.setattr(queue, "save_state", reject_results)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert len({handle.lane for handle in executor.terminated}) == 4
    assert executor.terminated[-1].lane == "developer"
    retained = {path for path in git.worktrees if "-review-1-" in path}
    assert retained == {
        "/wt/issue-1-review-1-requirements-reviewer",
        "/wt/issue-1-review-1-breaker-reviewer",
        "/wt/issue-1-review-1-architecture-reviewer",
    }


class SaturatedPrimaryBroker(FakeBroker):
    def reserve(self, assignment):
        if assignment.model_ref.startswith("ref-"):
            raise BrokerError("primary at concurrency cap")
        return super().reserve(assignment)


def test_review_wave_uses_broker_fallbacks_when_primary_capacity_is_full():
    fake = _ready_fake()
    git = RecordingGit()
    loop, queue = _foreman(
        fake,
        ScriptedExecutor(),
        _gate(),
        git=git,
        broker=SaturatedPrimaryBroker(),
    )
    assert loop.run_pass()[0].final_phase == "done"
    state = queue.load_state("owner/repo#1")
    assert state is not None
    lanes = state.extras["review_waves"]["review-1"]["lanes"]
    assert {record["assignment"]["model_ref"] for record in lanes.values()} == {
        "fallback-requirements-reviewer",
        "fallback-breaker-reviewer",
        "fallback-architecture-reviewer",
    }


def test_partial_checkout_failure_cleans_previously_materialized_reviews():
    class FailingGit(RecordingGit):
        def create_review_checkout(self, path, source_workdir, head_sha):
            if "breaker-reviewer" in path:
                raise RuntimeError("review checkout creation failed")
            return super().create_review_checkout(path, source_workdir, head_sha)

    git = FailingGit()
    loop, queue = _foreman(_ready_fake(), ScriptedExecutor(), _gate(), git=git)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "review checkout creation failed" in outcome.reason
    assert not any("-review-1-" in path for path in git.worktrees)
    assert queue.load_state("owner/repo#1").extras["review_waves"]["review-1"]["lanes"]


def test_failed_finding_persistence_does_not_claim_completed_reviews(monkeypatch):
    git = RecordingGit()
    executor = ScriptedExecutor(reviewer_findings_by_round={1: [_finding(1)]})
    loop, queue = _foreman(_ready_fake(), executor, _gate(), git=git)
    save = queue.save_state

    def reject_findings(state, expected_updated_at):
        if state.findings:
            raise StateConflictError("finding persistence failed")
        save(state, expected_updated_at)

    monkeypatch.setattr(queue, "save_state", reject_findings)
    outcome = loop.run_pass()[0]

    assert outcome.final_phase == "escalated"
    assert "finding persistence failed" in outcome.reason
    wave = queue.load_state("owner/repo#1").extras["review_waves"]["review-1"]
    assert {record["status"] for record in wave["lanes"].values()} == {"planned"}
    assert len({handle.lane for handle in executor.terminated}) == 4
    assert executor.terminated[-1].lane == "developer"
    assert len([path for path in git.worktrees if "-review-1-" in path]) == 3
