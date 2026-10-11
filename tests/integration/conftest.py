"""Real CAO execution, policy broker and CI gate over external test boundaries."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.cao import (
    CAOControlPlaneConfig,
    CaoSessionController,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.ci_gate import CIPRGateImpl
from ai_pr_orchestrator.v3.config import ModelRouterConfig, V3Config
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop
from ai_pr_orchestrator.v3.interfaces import GateDecision, GitOperations
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from ai_pr_orchestrator.v3.model_router import build_model_broker
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from tests.integration._fake_cao_server import FakeCAOServer
from tests.integration._harness import FakeGitOperations, StaticGate

E2E_WORK_TAG = "v3-work"


def _config(url: str) -> CAOControlPlaneConfig:
    return CAOControlPlaneConfig(
        base_url=url,
        session_timeout_seconds=60,
        request_timeout_seconds=5,
    )


@pytest.fixture
def fake_cao() -> Iterator[FakeCAOServer]:
    with FakeCAOServer() as cao:
        yield cao


@pytest.fixture
def cao_controller(fake_cao: FakeCAOServer) -> Iterator[CaoSessionController]:
    controller = CaoSessionController(_config(fake_cao.url), LaneRegistry.default())
    try:
        yield controller
    finally:
        controller.close()


@pytest.fixture
def cao_lane_executor(cao_controller: CaoSessionController) -> CaoLaneExecutor:
    lanes = LaneRegistry.default()
    catalog = ModelCatalog(
        tuple(
            ModelCatalogEntry(
                ref=f"ref-{lane.lane}",
                descriptor="test-model",
                provider=lane.lane,
                family=lane.lane,
                vendor=lane.lane,
                resource_class="free_tier",
                cost_class="free",
                input_price_per_mtok=0,
                output_price_per_mtok=0,
                capabilities=("tools", "coding"),
                roles=("worker", "reviewer"),
                quality_by_role={"worker": 5, "reviewer": 5},
            )
            for lane in lanes
        )
    )
    return CaoLaneExecutor(
        cao_controller, lanes, git=FakeGitOperations(), catalog=catalog, poll_interval_seconds=0.01
    )


@pytest.fixture
def lane_registry() -> LaneRegistry:
    return LaneRegistry.default()


DEFAULT_COMMITTER_NAME = "AIPRO E2E Bot"
DEFAULT_COMMITTER_EMAIL = "aipro-bot@example.invalid"


@pytest.fixture
def foreman_harness(
    cao_lane_executor: CaoLaneExecutor,
    lane_registry: LaneRegistry,
):

    def _build(
        seed_issue_numbers: list[int] | None = None,
        *,
        committer_name: str = DEFAULT_COMMITTER_NAME,
        committer_email: str = DEFAULT_COMMITTER_EMAIL,
        gate: GateDecision | None = None,
        broker: Any | None = None,
        git_ops: GitOperations | FakeGitOperations | None = None,
    ) -> tuple[ForemanPolicyLoop, GitHubIssueQueue, FakeGitHubClient]:
        fake = FakeGitHubClient()
        if seed_issue_numbers is None:
            seed_issue_numbers = [1]
        for n in seed_issue_numbers:
            fake.seed_issue(n, labels=[E2E_WORK_TAG, "v3-work-enabled"])

        config = V3Config(
            model_router=ModelRouterConfig(catalog=list(cao_lane_executor._catalog.entries))
        )
        fake.seed_check_run("sha", "tests", "completed", "success")
        ci_gate = CIPRGateImpl(fake, config.ci_policy) if gate is None else StaticGate(gate)
        if broker is None:
            broker = build_model_broker(config)
        if git_ops is None:
            git_ops = FakeGitOperations()
        executor = cao_lane_executor

        from tests.unit.test_v3_foreman import _record_fake_push

        _record_fake_push(fake, git_ops)
        queue = GitHubIssueQueue(fake, "owner", "repo", V3Config().github_queue, host_id="host-e2e")
        run_id = f"e2e-{uuid.uuid4().hex}"
        loop = ForemanPolicyLoop(
            queue,
            broker,
            lane_registry,
            executor,
            ci_gate,
            git_ops,
            config,
            run_id=run_id,
            worktree_root="/wt",
            committer_name=committer_name,
            committer_email=committer_email,
        )
        return loop, queue, fake

    return _build


def pytest_addoption(parser):
    parser.addoption(
        "--soak-issues",
        type=int,
        default=6,
        help="Sequential issues in the CAO-backed soak (at least 2)",
    )


def pytest_configure(config):
    if config.getoption("--soak-issues") < 2:
        raise pytest.UsageError("--soak-issues must be at least 2")
