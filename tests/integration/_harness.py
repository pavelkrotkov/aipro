"""Test-only scaffolding for the V3 E2E / soak harness (issue #55, P1).

This module is **not** part of the production code path; it is the
scaffolding the E2E scenarios in ``tests/integration/e2e/`` and the soak
harness in ``tests/integration/soak/`` build on top of. It mirrors the
test-only ``FakeBroker`` / ``StaticGate`` / ``FakeGitOperations`` from
``tests/unit/test_v3_foreman.py`` so the E2E fixtures do not depend on
imports into a unit test file. Per the cutover plan (rev 5, §1
principle 1), the **only** legitimate faking surface in V3 is the CAO
HTTP transport; this scaffolding is for glue around it, not a
replacement for the lane executor.

The one exception is ``HybridLaneExecutor``: it routes reviewer lanes
to a scripted findings list and worker lanes to the real
``CaoLaneExecutor``. CAO today returns agent output as unstructured
text, not as structured ``ReviewerFinding`` objects; without parsing,
the E2E scenarios that depend on reviewer findings (2, 3, 4 in #55)
cannot drive the foreman's adjudication path. The hybrid executor
exists to keep the worker path real (proving the wiring) and the
reviewer path scripted (matching today's actual behaviour, where the
agent's text is parsed by a separate layer not yet present in V3).
Scenarios that assert on review logic continue to use the unit-test
layer's ``ScriptedExecutor`` for full coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.domain import LaneIdentity, ModelAssignment
from ai_pr_orchestrator.v3.interfaces import (
    GateDecision,
    LaneExecutionContext,
    LaneResult,
    ModelLease,
    ReviewerFinding,
    SessionHandle,
)

NOW = datetime.now(UTC)


@dataclass
class FakeBroker:
    """In-memory :class:`ModelBroker` that always picks a deterministic
    per-lane model. The foreman's lane-execution contract is preserved
    (the broker produces a :class:`ModelLease` the executor can carry),
    but the selection logic is hard-coded for E2E determinism.
    """

    outstanding: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    def select(self, demand: Any) -> BrokerDecision:
        return BrokerDecision(
            demand=demand,
            evaluated_at=NOW,
            assignment=ModelAssignment(lane=demand.lane, model_ref=f"ref-{demand.lane}"),
        )

    def reserve(self, assignment: ModelAssignment) -> ModelLease:
        self.outstanding.append(assignment.lane)
        return ModelLease(
            lease_id=f"lease-{assignment.lane}-{len(self.outstanding)}",
            assignment=assignment,
        )

    def release(self, lease: ModelLease) -> None:
        self.outstanding.remove(lease.assignment.lane)
        self.released.append(lease.lease_id)


@dataclass
class StaticGate:
    """In-memory :class:`CIPRGate` that returns a pre-set decision.

    Use one per scenario, configured to either pass or fail on demand.
    """

    decision: GateDecision

    def evaluate(self, issue: Any, pr: Any) -> GateDecision:
        return self.decision


@dataclass
class FakeGitOperations:
    default: str = "main"
    branches: list[str] = field(default_factory=lambda: ["main"])
    worktrees: dict[str, str] = field(default_factory=dict)
    commits: list[tuple[str, str]] = field(default_factory=list)

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
        return None

    def cleanup_worktree(self, path: str) -> None:
        self.worktrees.pop(path, None)


@dataclass
class HybridLaneExecutor:
    """A test-only :class:`LaneExecutor` that routes by role.

    Worker lanes (developer) are delegated to a real
    :class:`CaoLaneExecutor` so the E2E wiring is exercised end-to-end.
    Reviewer lanes are routed to a per-round scripted findings list —
    the same shape the unit tests use — because the production code
    does not yet parse agent output into structured
    :class:`ReviewerFinding` objects. This is the second legitimate
    faking surface in V3, alongside the CAO HTTP transport: reviewer
    result parsing is a separate layer that does not exist today, so
    the E2E scenarios that depend on it must inject the parsed result
    some other way. The hybrid keeps the worker path real and the
    reviewer path deterministic.

    Parameters
    ----------
    worker:
        A real :class:`CaoLaneExecutor` for worker lanes.
    reviewer_findings_by_round:
        Mapping of ``round_number`` to a list of findings to return.
        Each reviewer lane appends its lane name to the finding's
        ``id`` (matching the unit tests' ``ScriptedExecutor``).
    """

    worker: CaoLaneExecutor
    reviewer_findings_by_round: dict[int, list[ReviewerFinding]] = field(default_factory=dict)
    round_counter: dict[str, int] = field(default_factory=dict)
    reviewer_session_handle: SessionHandle | None = None

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease: ModelLease | None = None,
    ) -> LaneResult:
        if lane.role == "reviewer":
            n = self.round_counter.get(lane.lane, 0) + 1
            self.round_counter[lane.lane] = n
            scripted = self.reviewer_findings_by_round.get(n, [])
            findings = [
                ReviewerFinding(
                    id=f"{f.id}-{lane.lane}",
                    lane=lane.lane,
                    body=f.body,
                    severity=f.severity,
                    run_id=context.run_id,
                    round_id=context.round_id or f"review-{n}",
                )
                for f in scripted
            ]
            return LaneResult(
                session=self.reviewer_session_handle or _NULL_HANDLE,
                exit_code=0,
                output_summary="",
                changed_files=[],
                findings=findings,
            )
        return self.worker.execute(lane, task_prompt, workdir, context, lease)


#: A stand-in :class:`SessionHandle` for reviewer lanes that never
#: touched CAO. The foreman only checks ``session.session_id`` for
#: persistence, not for round continuity, so a synthetic placeholder
#: is sufficient. Real worker lanes always return a real handle via
#: :class:`CaoLaneExecutor`.
_NULL_HANDLE = SessionHandle(session_id="scripted-reviewer", lane="reviewer")


__all__ = [
    "FakeBroker",
    "FakeGitOperations",
    "HybridLaneExecutor",
    "StaticGate",
]
