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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_pr_orchestrator.v3.broker import BrokerDecision
from ai_pr_orchestrator.v3.domain import ModelAssignment
from ai_pr_orchestrator.v3.interfaces import GateDecision, ModelLease

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


__all__ = [
    "FakeBroker",
    "FakeGitOperations",
    "StaticGate",
]
