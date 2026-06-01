"""Tests for dry-run mode: the runner plans actions without mutating anything."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_pr_orchestrator.config import (
    CiConfig,
    Config,
    GitConfig,
    MainCoderConfig,
    ReviewerConfig,
    ReviewPhaseConfig,
    SafetyConfig,
)
from ai_pr_orchestrator.github import models as gh_models
from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.models import AgentRunResult, Finding, ReviewerTrigger, RuntimeState
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
    responded: bool = False

    def matches_author(self, login: str) -> bool:
        return False

    def build_trigger_comment(self, round_index: int, head_sha: str) -> str:
        return "/fake review"

    def collect_findings(
        self, pr_number: int, head_sha: str, trigger_timestamp: datetime
    ) -> list[Finding]:
        return list(self.findings)

    def has_responded(self, pr_number: int, trigger_timestamp: datetime) -> bool:
        return self.responded or bool(self.findings)


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


def build_ctx(
    gh: FakeGitHubClient,
    *,
    config: Config | None = None,
    reviewers: dict[str, Any] | None = None,
    clock: Any = None,
) -> RunnerContext:
    return RunnerContext(
        github=gh,
        coder=StubCoder(),
        reviewers={"fake": StubReviewer()} if reviewers is None else reviewers,
        git=None,
        config=config or make_config(),
        clock=clock or (lambda: NOW),
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


# ----- Dry-run plan accuracy for runner-only branches -----


def _waiting_state(*, trigger_ts: datetime) -> RuntimeState:
    return RuntimeState(
        version=1,
        pr_number=1,
        head_sha="head-1",
        status="waiting",
        round_index=1,
        created_at=trigger_ts,
        updated_at=trigger_ts,
        trigger_history=[
            ReviewerTrigger(
                reviewer_name="fake", round_index=1, timestamp=trigger_ts, head_sha="head-1"
            )
        ],
    )


def test_dry_run_reports_orphaned_coder_bailout(capsys: pytest.CaptureFixture[str]) -> None:
    # A persisted handling state whose coder result is lost would bail to
    # needs_human in a real run; the plan must report that, not noop.
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = RuntimeState(
        version=1,
        pr_number=pr.number,
        head_sha=pr.head_sha,
        status="handling",
        round_index=1,
        last_coder_round_index=1,
        created_at=NOW,
        updated_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    rc = Runner(build_ctx(gh)).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    assert "-> 'needs_human'" in out
    assert "would post the final summary comment" in out
    assert "waiting_for_coder" not in out
    assert _comment_count(gh, pr.number) == 1  # no mutation


def test_dry_run_reports_zero_findings_completion(capsys: pytest.CaptureFixture[str]) -> None:
    # Reviewer responded with no findings -> a real poll completes the phase;
    # the plan must reflect that instead of the plain waiting transition.
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = _waiting_state(trigger_ts=NOW)
    gh.seed_comment(pr.number, serialize_state_comment(state))
    reviewer = StubReviewer(name="fake", responded=True)

    rc = Runner(build_ctx(gh, reviewers={"fake": reviewer})).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    assert "status 'waiting' ->" in out
    assert "-> 'waiting'" not in out  # the phase advanced, not a noop


def test_dry_run_reports_reviewer_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    # An already-expired waiting state plans needs_human (reviewer timeout),
    # anchored to the persisted trigger timestamp — not a noop.
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = _waiting_state(trigger_ts=NOW - timedelta(hours=2))
    gh.seed_comment(pr.number, serialize_state_comment(state))
    cfg = make_config(
        review_phase=ReviewPhaseConfig(
            poll_interval_seconds=10, reviewer_timeout_seconds=60, phase_timeout_seconds=120
        )
    )

    rc = Runner(build_ctx(gh, config=cfg, reviewers={"fake": StubReviewer()})).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    assert "-> 'needs_human'" in out
    assert _comment_count(gh, pr.number) == 1  # no mutation


def test_dry_run_reports_missing_reviewer_adapter(capsys: pytest.CaptureFixture[str]) -> None:
    # A reviewer triggered this round but absent from ctx.reviewers plans the
    # needs_human bailout a real poll would reach.
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, labels=["ai-loop"])
    state = _waiting_state(trigger_ts=NOW)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    rc = Runner(build_ctx(gh, reviewers={})).run(pr.number)
    out = capsys.readouterr().out

    assert rc == 0
    assert "-> 'needs_human'" in out
    assert _comment_count(gh, pr.number) == 1  # no mutation


# ----- CLI entry point -----


def test_cli_run_dry_run_exits_zero_without_runtime_wiring(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Runtime context construction (_build_runtime_context) is still a stub
    # (left to a follow-up issue); until then `aipro dry-run` must exit 0 (not
    # the generic 1 a missing context yields for a real run), per the issue
    # acceptance criteria.
    from ai_pr_orchestrator import runner as runner_mod

    monkeypatch.setattr(runner_mod, "load_config", make_config)
    # Stub setup_logging so the real call doesn't reconfigure the global package
    # logger (propagate=False) and leak into other tests' caplog capture.
    monkeypatch.setattr(runner_mod, "setup_logging", lambda **kwargs: None)
    rc = runner_mod.run(pr_number=1, dry_run=True)
    err = capsys.readouterr().err

    assert rc == 0
    assert "Dry-run" in err
    assert "not\nimplemented yet" in err or "not implemented yet" in err
