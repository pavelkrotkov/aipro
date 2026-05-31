"""Tests for dry-run mode: the runner plans actions without mutating anything."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from ai_pr_orchestrator.config import (
    CiConfig,
    Config,
    GitConfig,
    MainCoderConfig,
    ReviewerConfig,
    SafetyConfig,
)
from ai_pr_orchestrator.github import models as gh_models
from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.models import AgentRunResult, Finding, RuntimeState
from ai_pr_orchestrator.runner import ParsedEvent, Runner, RunnerContext
from ai_pr_orchestrator.state_storage import find_state_comment, serialize_state_comment

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


# ----- Minimal test doubles -----


@dataclass
class StubCoder:
    name: str = "stub-coder"

    def run_fix_task(self, task: Any) -> AgentRunResult:  # pragma: no cover - never called
        raise AssertionError("dry-run must not invoke the coder")


@dataclass
class StubReviewer:
    name: str = "fake"
    findings: list[Finding] = field(default_factory=list)

    def matches_author(self, login: str) -> bool:
        return False

    def build_trigger_comment(self, round_index: int, head_sha: str) -> str:
        return "/fake review"

    def collect_findings(
        self, pr_number: int, head_sha: str, trigger_timestamp: datetime
    ) -> list[Finding]:
        return list(self.findings)

    def has_responded(self, pr_number: int, trigger_timestamp: datetime) -> bool:
        return bool(self.findings)


def make_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "main_coder": MainCoderConfig(provider="codex_cli"),
        "reviewers": {
            "fake": ReviewerConfig(bot_logins=["fake-bot[bot]"], trigger_comment="/fake")
        },
        "ci": CiConfig(require_green_before_done=True, ignored_checks=["AI PR Review Loop"]),
        "safety": SafetyConfig(only_run_on_labeled_prs=False),
        "git": GitConfig(base_branch="main"),
    }
    defaults.update(overrides)
    return Config(**defaults)


def seed_pr(gh: FakeGitHubClient, *, labels: list[str] | None = None) -> gh_models.PullRequest:
    pr = gh_models.PullRequest(
        number=1,
        title="Test PR",
        body="",
        state="open",
        head_sha="head-1",
        head_ref="feature/x",
        base_ref="main",
        author="pavel",
        labels=labels or [],
        author_association="OWNER",
    )
    gh.seed_pr(pr)
    return pr


def build_ctx(gh: FakeGitHubClient, *, config: Config | None = None) -> RunnerContext:
    return RunnerContext(
        github=gh,
        coder=StubCoder(),
        reviewers={"fake": StubReviewer()},
        git=None,
        config=config or make_config(),
        clock=lambda: NOW,
        dry_run=True,
    )


def _comment_count(gh: FakeGitHubClient, pr_number: int) -> int:
    return len(gh.get_pr_comments(pr_number))


# ----- Tests -----


def test_dry_run_reads_state_and_plans_actions(capsys: pytest.CaptureFixture[str]) -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha=pr.head_sha,
        status="triggering",
        round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    rc = Runner(build_ctx(gh)).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    assert "DRY-RUN PR #1" in out
    assert "1 action(s) planned" in out
    assert "would post a comment triggering reviewer 'fake'" in out


def test_dry_run_performs_no_write_operations(capsys: pytest.CaptureFixture[str]) -> None:
    """A ci_wait -> done transition would normally post a summary and relabel.

    In dry-run those mutations must be planned but never executed.
    """
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    gh.seed_check_run("head-1", "build", "completed", "success")
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha=pr.head_sha,
        status="ci_wait",
        round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))
    comments_before = _comment_count(gh, pr.number)

    rc = Runner(build_ctx(gh)).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    # The plan names the mutations it would make...
    assert "would post the final summary comment" in out
    assert "would add label 'ai-loop-done'" in out
    assert "would remove label 'ai-loop'" in out
    # ...but none were actually performed: no new comment, labels untouched.
    assert _comment_count(gh, pr.number) == comments_before
    assert gh.get_pr(pr.number).labels == ["ai-loop"]


def test_dry_run_read_operations_work_normally() -> None:
    """Reads (get_pr, get_check_runs, get_commit_statuses) feed the plan.

    A passing seeded check drives ci_wait -> done, which only happens if the
    runner successfully read the check state.
    """
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    gh.seed_check_run("head-1", "build", "completed", "success")
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha=pr.head_sha,
        status="ci_wait",
        round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    Runner(build_ctx(gh)).run(pr.number)
    # State comment was NOT advanced to done (no persistence in dry-run).
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in gh.get_pr_comments(pr.number)])
    assert sc is not None
    assert sc.state.status == "ci_wait"


def test_dry_run_does_not_post_initial_state_comment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fresh PR with no state comment must not get one posted in dry-run."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])

    rc = Runner(build_ctx(gh)).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    assert _comment_count(gh, pr.number) == 0  # no init comment posted
    assert "DRY-RUN PR #1" in out


def test_dry_run_returns_exit_code_zero() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha=pr.head_sha,
        status="triggering",
        round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    assert Runner(build_ctx(gh)).run(pr.number) == 0


def test_dry_run_with_event_also_works(capsys: pytest.CaptureFixture[str]) -> None:
    """The event (e.g. from --event-path) flows into the snapshot in dry-run."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    gh.seed_check_run("head-1", "build", "completed", "success")
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha=pr.head_sha,
        status="ci_wait",
        round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))
    comments_before = _comment_count(gh, pr.number)

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha="head-1")
    rc = Runner(build_ctx(gh)).run(pr.number, event=event)
    out = capsys.readouterr().out

    assert rc == 0
    assert "would post the final summary comment" in out
    assert _comment_count(gh, pr.number) == comments_before


def test_dry_run_with_stale_event_plans_noop(capsys: pytest.CaptureFixture[str]) -> None:
    """A check event for a superseded SHA plans a stale-CI noop, no mutations."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha="head-1",
        status="ci_wait",
        round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha="stale-sha")
    rc = Runner(build_ctx(gh)).run(pr.number, event=event)
    out = capsys.readouterr().out

    assert rc == 0
    assert "would do nothing (noop: stale_ci_event)" in out
