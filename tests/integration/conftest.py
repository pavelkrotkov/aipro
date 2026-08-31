"""Shared fixtures for the V3 E2E / soak harness (issue #55, P1).

This ``conftest.py`` is auto-discovered by pytest for any test file under
``tests/integration/`` and ``tests/integration/e2e/``. It wires the real
production code paths to the in-process :class:`FakeCAOServer` from
PR-1, so individual scenario files (PRs 4-6) can declare small,
focused tests that exercise observable behaviour — branches, PRs,
labels — without re-implementing the wiring.

The foreman loop, ``GitHubIssueQueue``, the lane executor, and the
broker/gate are all **real production code**. Only the CAO HTTP
transport is faked; that is the boundary the cutover plan (rev 5, §1
principle 1) carved out as the only legitimate faking surface.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoSessionController,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.config import V3Config
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop
from ai_pr_orchestrator.v3.interfaces import GateDecision
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from tests.integration._fake_cao_server import FakeCAOServer
from tests.integration._harness import (
    FakeBroker,
    FakeGitOperations,
    HybridLaneExecutor,
    StaticGate,
)

#: Tag used on issues the harness seeds; the queue reads it as the
#: "enabled" label. Matches :class:`V3Config.github_queue.enabled_label`
#: default (``"v3-work"``) so the smoke test does not need a custom
#: queue configuration.
E2E_WORK_TAG = "v3-work"


def _config(url: str) -> CAOControlPlaneConfig:
    return CAOControlPlaneConfig(
        base_url=url,
        session_timeout_seconds=60,
        request_timeout_seconds=5,
    )


@pytest.fixture
def fake_cao() -> Iterator[FakeCAOServer]:
    """A live ``FakeCAOServer`` for the duration of one test.

    Bind on entry, shut down on exit. The harness scripts status
    sequences and outputs *before* calling the executor; see
    ``script_session`` helpers in individual scenario files.
    """
    with FakeCAOServer() as cao:
        yield cao


@pytest.fixture
def cao_controller(fake_cao: FakeCAOServer) -> Iterator[CaoSessionController]:
    """Real :class:`CaoSessionController` pointing at the fake.

    Closed at the end of the test so the background HTTP client's
    socket is released even if a scenario crashed.
    """
    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    try:
        yield controller
    finally:
        controller.close()


@pytest.fixture
def cao_lane_executor(cao_controller: CaoSessionController) -> CaoLaneExecutor:
    """Real :class:`CaoLaneExecutor` wired to the fake-backed controller.

    Poll interval is small so scenarios do not slow CI down; the soak
    harness can override this with a larger value to drive wall-clock
    behaviour.
    """
    return CaoLaneExecutor(cao_controller, LaneRegistry.default(), poll_interval_seconds=0.01)


@pytest.fixture
def lane_registry() -> LaneRegistry:
    """The default lane registry."""
    return LaneRegistry.default()


@pytest.fixture
def foreman_harness(cao_lane_executor: CaoLaneExecutor, lane_registry: LaneRegistry):
    """Factory that wires the foreman loop with a fresh ``FakeGitHubClient``.

    Returns a callable ``(seed_issue_numbers=None)`` that builds a
    fresh ``FakeGitHubClient``, seeds the requested issue numbers with
    :data:`E2E_WORK_TAG`, and yields a ``(loop, queue, fake)`` triple.
    Each call is independent; the fakes are not shared between
    scenarios.

    The harness is intentionally minimal: scenarios that need richer
    configuration (a custom broker, a custom gate, a custom git-ops
    fake) can build their own. The smoke test
    (``test_e2e_smoke.py``) is the reference for the happy wiring.
    """

    def _build(
        seed_issue_numbers: list[int] | None = None,
        hybrid_findings: dict[int, list[Any]] | None = None,
        config: V3Config | None = None,
    ) -> tuple[ForemanPolicyLoop, GitHubIssueQueue, FakeGitHubClient]:
        fake = FakeGitHubClient()
        if seed_issue_numbers is None:
            seed_issue_numbers = [1]
        for n in seed_issue_numbers:
            fake.seed_issue(n, labels=[E2E_WORK_TAG])

        cfg = config or V3Config()
        # Two-review-round scenarios (e.g. #55 scenario 2) need a
        # wider coder / reviewer budget than the V3Config() default
        # (1 coder invocation, 3 reviewer triggers) — three reviewer
        # lanes x one round already hits the default's reviewer cap,
        # and a fix round needs at least 2 coder invocations. The
        # scenario tests that want the default behaviour pass
        # ``config=V3Config()`` explicitly.
        if config is None and hybrid_findings is not None:
            from ai_pr_orchestrator.v3.config import SafetyPolicyConfig

            cfg = V3Config(
                safety=SafetyPolicyConfig(
                    max_coder_invocations_per_run=5,
                    max_reviewer_triggers_per_run=20,
                )
            )

        executor: Any = cao_lane_executor
        if hybrid_findings is not None:
            executor = HybridLaneExecutor(
                worker=cao_lane_executor,
                reviewer_findings_by_round=hybrid_findings,
            )

        queue = GitHubIssueQueue(fake, "owner", "repo", cfg.github_queue, host_id="host-e2e")
        loop = ForemanPolicyLoop(
            queue,
            FakeBroker(),
            lane_registry,
            executor,
            StaticGate(GateDecision(passed=True, pending_checks=(), failed_checks=())),
            FakeGitOperations(),
            cfg,
            run_id=f"e2e-{int(time.time() * 1000)}",
            worktree_root="/wt",
            committer_name="Pavel Krotkov",
            committer_email="pavel.krotkov@gmail.com",
        )
        return loop, queue, fake

    return _build


@pytest.fixture
def clock() -> FakeClock:
    """A monotonic clock the soak harness can advance to make time-based
    scenarios deterministic without wall-clock waits.

    The foreman's lifecycle currently keys off the controller's own
    time-based rules (idle-settle, ``session_timeout_seconds``); this
    clock is here for forward-compatibility with scenarios that
    exercise the foreman's own time-based escalations. Keep the
    surface minimal.
    """
    return FakeClock()


class FakeClock:
    """Monotonic clock with manual advance for the E2E/soak harness."""

    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


__all__ = [
    "E2E_WORK_TAG",
    "FakeClock",
]
