from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from ai_pr_orchestrator.config import (
    CiConfig,
    Config,
    GitConfig,
    MainCoderConfig,
    NotificationsConfig,
    ReviewerConfig,
    ReviewPhaseConfig,
    SafetyConfig,
)
from ai_pr_orchestrator.github import models as gh_models
from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.models import (
    AgentRunResult,
    CostTracker,
    Decision,
    Finding,
    FixTask,
    ReviewerTrigger,
    RuntimeState,
    Status,
    TestResult,
    TokenUsage,
    Verdict,
)
from ai_pr_orchestrator.runner import ParsedEvent, Runner, RunnerContext
from ai_pr_orchestrator.state_storage import find_state_comment, serialize_state_comment

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


class TrackingGitHubClient(FakeGitHubClient):
    def __init__(self) -> None:
        super().__init__(now=NOW)
        self.posted_bodies: list[str] = []
        self.edited_bodies: list[str] = []
        self.state_statuses: list[str] = []
        self.replies: list[tuple[str, str]] = []
        self.resolved_threads: list[str] = []
        self.added_labels: list[str] = []
        self.removed_labels: list[str] = []

    def post_comment(self, issue_number: int, body: str) -> gh_models.Comment:
        comment = super().post_comment(issue_number, body)
        self.posted_bodies.append(body)
        self._record_state(body)
        return comment

    def edit_comment(self, comment_id: int, body: str) -> gh_models.Comment:
        comment = super().edit_comment(comment_id, body)
        self.edited_bodies.append(body)
        self._record_state(body)
        return comment

    def reply_to_review_thread(self, thread_id: str, body: str) -> dict[str, Any] | None:
        self.replies.append((thread_id, body))
        return super().reply_to_review_thread(thread_id, body)

    def resolve_review_thread(self, thread_id: str) -> dict[str, Any] | None:
        self.resolved_threads.append(thread_id)
        return super().resolve_review_thread(thread_id)

    def add_label(self, issue_number: int, label: str) -> list[dict[str, Any]]:
        self.added_labels.append(label)
        return super().add_label(issue_number, label)

    def remove_label(self, issue_number: int, label: str) -> None:
        self.removed_labels.append(label)
        return super().remove_label(issue_number, label)

    def _record_state(self, body: str) -> None:
        state_comment = find_state_comment([{"id": 0, "body": body}])
        if state_comment is not None:
            self.state_statuses.append(state_comment.state.status)


@dataclass
class FakeCoderAdapter:
    result: AgentRunResult
    name: str = "fake-coder"
    on_run: Callable[[FixTask], None] | None = None
    calls: list[FixTask] = field(default_factory=list)

    def run_fix_task(self, task: FixTask) -> AgentRunResult:
        self.calls.append(task)
        if self.on_run is not None:
            self.on_run(task)
        return self.result


@dataclass
class FakeReviewerAdapter:
    name: str = "fake"
    findings: list[Finding] = field(default_factory=list)
    responded: bool = False
    on_collect: Callable[[], None] | None = None
    collect_calls: int = 0
    trigger_comments: list[tuple[int, str]] = field(default_factory=list)

    def matches_author(self, login: str) -> bool:
        return login == "fake-reviewer[bot]"

    def build_trigger_comment(self, round_index: int, head_sha: str) -> str:
        self.trigger_comments.append((round_index, head_sha))
        return (
            f"/fake review\n\n<!-- aipro-review-trigger fake round={round_index} sha={head_sha} -->"
        )

    def collect_findings(
        self, pr_number: int, head_sha: str, trigger_timestamp: datetime
    ) -> list[Finding]:
        del pr_number, head_sha, trigger_timestamp
        self.collect_calls += 1
        if self.on_collect is not None:
            self.on_collect()
        return list(self.findings)

    def has_responded(self, pr_number: int, trigger_timestamp: datetime) -> bool:
        del pr_number, trigger_timestamp
        return self.responded


@dataclass
class FakeGitRepo:
    head_sha: str = "head-1"
    clean: bool = True
    remote_heads: list[str | Exception | None] = field(default_factory=lambda: ["head-1"])
    commit_sha: str = "fix-sha"
    commits: list[tuple[str, str, str]] = field(default_factory=list)
    pushes: list[str] = field(default_factory=list)
    rollbacks: int = 0

    def is_clean(self) -> bool:
        return self.clean

    def commit(self, message: str, author_name: str, author_email: str) -> str | None:
        self.commits.append((message, author_name, author_email))
        self.head_sha = self.commit_sha
        self.clean = True
        return self.commit_sha

    def push(self, branch: str) -> None:
        self.pushes.append(branch)

    def rollback(self) -> None:
        self.rollbacks += 1
        self.clean = True

    def fetch_remote_head(self, branch: str) -> str | None:
        del branch
        if not self.remote_heads:
            return self.head_sha
        value = self.remote_heads.pop(0) if len(self.remote_heads) > 1 else self.remote_heads[0]
        if value is None:
            return self.head_sha
        if isinstance(value, Exception):
            raise value
        return value

    def get_head_sha(self) -> str:
        return self.head_sha

    def get_diff(self, base: str) -> str:
        return f"diff --git a/src/app.py b/src/app.py\n# base: {base}\n"


class FakeClock:
    def __init__(self, start: datetime = NOW) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeSleeper:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


def make_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "main_coder": MainCoderConfig(provider="codex_cli"),
        "reviewers": {
            "fake": ReviewerConfig(
                bot_logins=["fake-reviewer[bot]"],
                trigger_comment="/fake review",
            )
        },
        "review_phase": ReviewPhaseConfig(
            poll_interval_seconds=1,
            reviewer_timeout_seconds=5,
            phase_timeout_seconds=10,
        ),
        "git": GitConfig(commit_message_prefix="fix: address AI review feedback"),
        "ci": CiConfig(require_green_before_done=True, ignored_checks=["aipro-self"]),
        "safety": SafetyConfig(
            only_run_on_labeled_prs=True,
            disallow_forks=True,
            disallow_workflow_file_changes=True,
            max_total_iterations=5,
            max_coder_invocations_per_run=1,
            max_commits_per_run=1,
            max_reviewer_triggers_per_run=3,
            max_prompt_tokens=100000,
        ),
    }
    defaults.update(overrides)
    return Config(**defaults)


def seed_pr(
    gh: TrackingGitHubClient,
    *,
    number: int = 14,
    head_sha: str = "head-1",
    labels: list[str] | None = None,
    is_fork: bool = False,
    changed_files: list[str] | None = None,
) -> gh_models.PullRequest:
    pr = gh_models.PullRequest(
        number=number,
        title="Integration PR",
        body="",
        state="open",
        head_sha=head_sha,
        head_ref="feature/integration",
        base_ref="main",
        author="pavel",
        labels=["ai-loop"] if labels is None else labels,
        is_fork=is_fork,
        changed_files=changed_files or ["src/app.py"],
        author_association="OWNER",
    )
    gh.seed_pr(pr)
    return pr


def finding(id_: str = "f1", *, thread_id: str = "thread-1") -> Finding:
    return Finding(
        id=id_,
        source="fake",
        body="Please fix this",
        created_at=NOW,
        head_sha="head-1",
        thread_id=thread_id,
        path="src/app.py",
        line=12,
    )


def decision(
    finding_id: str = "f1",
    *,
    verdict: Verdict = "accepted",
    reply: str = "Fixed in the follow-up commit.",
    should_resolve: bool = True,
    thread_id: str = "thread-1",
) -> Decision:
    return Decision(
        finding_id=finding_id,
        verdict=verdict,
        confidence="high",
        reason="valid finding",
        reply=reply,
        should_resolve=should_resolve,
        thread_id=thread_id,
        changed_files=["src/app.py"],
    )


def coder_result(**overrides: Any) -> AgentRunResult:
    defaults: dict[str, Any] = {
        "changed": False,
        "summary": "done",
        "decisions": [decision()],
        "token_usage": TokenUsage(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return AgentRunResult(**defaults)


def context(
    *,
    gh: TrackingGitHubClient,
    coder: FakeCoderAdapter | None = None,
    reviewer: FakeReviewerAdapter | None = None,
    git: FakeGitRepo | None = None,
    config: Config | None = None,
    clock: FakeClock | None = None,
    dry_run: bool = False,
) -> RunnerContext:
    use_clock = clock or FakeClock()
    return RunnerContext(
        github=gh,
        coder=cast(Any, coder or FakeCoderAdapter(result=coder_result(decisions=[]))),
        reviewers={"fake": reviewer or FakeReviewerAdapter(responded=True)},
        git=cast(Any, git),
        config=config or make_config(),
        clock=use_clock,
        sleeper=FakeSleeper(use_clock),
        dry_run=dry_run,
    )


def current_state(gh: TrackingGitHubClient, pr_number: int = 14) -> RuntimeState:
    state_comment = find_state_comment(
        [{"id": c.id, "body": c.body} for c in gh.get_pr_comments(pr_number)]
    )
    assert state_comment is not None
    return state_comment.state


def seed_state(
    gh: TrackingGitHubClient,
    pr_number: int,
    *,
    status: Status,
    head_sha: str = "head-1",
    round_index: int = 1,
    trigger_history: list[ReviewerTrigger] | None = None,
    cost: CostTracker | None = None,
    ci_wait_started_at: datetime | None = None,
    done_reason: str | None = None,
) -> RuntimeState:
    state = RuntimeState(
        version=1,
        pr_number=pr_number,
        head_sha=head_sha,
        status=status,
        round_index=round_index,
        trigger_history=trigger_history or [],
        cost=cost or CostTracker(),
        ci_wait_started_at=ci_wait_started_at,
        created_at=NOW,
        updated_at=NOW,
        done_reason=done_reason,
    )
    gh.seed_comment(pr_number, serialize_state_comment(state))
    return state


def trigger(round_index: int = 1, *, timestamp: datetime = NOW) -> ReviewerTrigger:
    return ReviewerTrigger(
        reviewer_name="fake",
        round_index=round_index,
        timestamp=timestamp,
        head_sha="head-1",
    )


def assert_statuses_in_order(actual: list[str], expected: list[str]) -> None:
    cursor = 0
    for status in actual:
        if cursor < len(expected) and status == expected[cursor]:
            cursor += 1
    assert cursor == len(expected), actual


def test_full_happy_path_exits_for_ci_then_resumes_to_done() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    gh.seed_thread("thread-1", pr.number, path="src/app.py")
    reviewer = FakeReviewerAdapter(findings=[finding()])
    git = FakeGitRepo(head_sha=pr.head_sha, remote_heads=[pr.head_sha], commit_sha="fix-sha")
    coder = FakeCoderAdapter(
        coder_result(changed=True, commit_message=None),
        on_run=lambda _task: setattr(git, "clean", False),
    )
    cfg = make_config()

    first_result = Runner(context(gh=gh, reviewer=reviewer, coder=coder, git=git, config=cfg)).run(
        pr.number
    )

    assert first_result == 0
    assert current_state(gh).status == "ci_wait"
    assert current_state(gh).head_sha == "fix-sha"
    assert_statuses_in_order(
        gh.state_statuses,
        ["init", "triggering", "waiting", "handling", "ci_wait"],
    )
    assert any("aipro-review-trigger fake" in body for body in gh.posted_bodies)
    assert len(coder.calls) == 1
    task = coder.calls[0]
    assert task.pr_number == pr.number
    assert task.head_sha == pr.head_sha
    assert [item.id for item in task.findings] == ["f1"]
    assert task.changed_files == ["src/app.py"]
    assert gh.replies == [("thread-1", "Fixed in the follow-up commit.")]
    assert gh.resolved_threads == ["thread-1"]
    assert git.commits[0][0] == cfg.git.commit_message_prefix
    assert git.pushes == ["feature/integration"]
    assert not any(
        body.startswith("AI PR Orchestrator: status `done`") for body in gh.posted_bodies
    )

    gh.seed_pr(replace(pr, head_sha="fix-sha"))
    gh.seed_check_run("fix-sha", name="tests", status="completed", conclusion="success")
    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha="fix-sha")

    second_result = Runner(context(gh=gh, reviewer=reviewer, coder=coder, git=git, config=cfg)).run(
        pr.number, event=event
    )

    assert second_result == 0
    final = current_state(gh)
    assert final.status == "done"
    assert final.done_reason == "ci_passed"
    assert "ai-loop-done" in gh.get_pr(pr.number).labels
    assert "ai-loop" not in gh.get_pr(pr.number).labels
    assert any("reason: `ci_passed`" in body for body in gh.posted_bodies)
    assert gh.state_statuses[-1] == "done"


def test_reviewer_response_with_no_findings_finishes_without_coder() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    reviewer = FakeReviewerAdapter(findings=[], responded=True)
    coder = FakeCoderAdapter(coder_result(decisions=[]))

    assert Runner(context(gh=gh, reviewer=reviewer, coder=coder)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "done"
    assert state.done_reason == "no_findings"
    assert coder.calls == []
    assert any("reason: `no_findings`" in body for body in gh.posted_bodies)


def test_coder_needs_human_posts_summary_with_mentions() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    reviewer = FakeReviewerAdapter(findings=[finding()])
    coder = FakeCoderAdapter(coder_result(needs_human=True, decisions=[]))
    cfg = make_config(notifications=NotificationsConfig(mention_on_needs_human=["alice", "@bob"]))

    assert Runner(context(gh=gh, reviewer=reviewer, coder=coder, config=cfg)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "needs_human"
    assert state.last_error == "coder_needs_human"
    summary = next(body for body in gh.posted_bodies if body.startswith("AI PR Orchestrator"))
    assert "cc: @alice @bob" in summary
    assert "ai-loop-error" in gh.get_pr(pr.number).labels


def test_cost_limit_routes_to_needs_human_with_cost_summary() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    cfg = make_config(
        safety=SafetyConfig(
            only_run_on_labeled_prs=True,
            max_reviewer_triggers_per_run=0,
            disallow_forks=True,
            disallow_workflow_file_changes=True,
        )
    )

    assert Runner(context(gh=gh, config=cfg)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "needs_human"
    assert state.last_error == "cost_limit_reached"
    assert any("reason: `cost_limit_reached`" in body for body in gh.posted_bodies)


def test_reviewer_timeout_routes_to_needs_human() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    old_trigger = trigger(timestamp=NOW - timedelta(minutes=10))
    seed_state(gh, pr.number, status="waiting", trigger_history=[old_trigger])
    reviewer = FakeReviewerAdapter(findings=[], responded=False)

    assert Runner(context(gh=gh, reviewer=reviewer)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "needs_human"
    assert state.last_error == "reviewer_timeout"


def test_test_regression_rolls_back_and_needs_human() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    gh.seed_thread("thread-1", pr.number, path="src/app.py")
    reviewer = FakeReviewerAdapter(findings=[finding()])
    git = FakeGitRepo(head_sha=pr.head_sha, remote_heads=[pr.head_sha])
    coder = FakeCoderAdapter(
        coder_result(
            changed=True,
            tests=[TestResult(command="pytest", result="failed", notes="unit failed")],
        ),
        on_run=lambda _task: setattr(git, "clean", False),
    )

    assert Runner(context(gh=gh, reviewer=reviewer, coder=coder, git=git)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "needs_human"
    assert state.last_error == "test_regression"
    assert git.rollbacks == 1
    assert git.commits == []


def test_ci_failure_after_resume_routes_to_needs_human() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh, head_sha="fix-sha")
    seed_state(
        gh,
        pr.number,
        status="ci_wait",
        head_sha="fix-sha",
        ci_wait_started_at=NOW,
    )
    gh.seed_check_run("fix-sha", name="tests", status="completed", conclusion="failure")
    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha="fix-sha")

    assert Runner(context(gh=gh)).run(pr.number, event=event) == 0

    state = current_state(gh)
    assert state.status == "needs_human"
    assert state.last_error == "ci_failed"
    assert "ai-loop-error" in gh.get_pr(pr.number).labels


@pytest.mark.parametrize(
    ("pr_kwargs", "expected_status", "expected_error"),
    [
        ({"is_fork": True}, "error", "fork_pr_not_allowed"),
        (
            {"changed_files": [".github/workflows/ci.yml", "src/app.py"]},
            "needs_human",
            "workflow_file_changed",
        ),
    ],
)
def test_safety_paths_stop_before_coder(
    pr_kwargs: dict[str, Any],
    expected_status: str,
    expected_error: str,
) -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh, **pr_kwargs)
    coder = FakeCoderAdapter(coder_result(decisions=[]))

    assert Runner(context(gh=gh, coder=coder)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == expected_status
    assert state.last_error == expected_error
    assert coder.calls == []


def test_label_removed_mid_run_completes_with_label_removed_reason() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    reviewer = FakeReviewerAdapter(
        findings=[],
        responded=False,
        on_collect=lambda: gh.remove_label(pr.number, "ai-loop"),
    )

    assert Runner(context(gh=gh, reviewer=reviewer)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "done"
    assert state.done_reason == "label_removed"
    assert "ai-loop-done" in gh.get_pr(pr.number).labels


def test_rerun_after_done_is_noop() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh, labels=["ai-loop-done"])
    seed_state(gh, pr.number, status="done", round_index=1, done_reason="ci_passed")
    comments_before = gh.get_pr_comments(pr.number)
    bodies_before = [comment.body for comment in comments_before]

    assert Runner(context(gh=gh)).run(pr.number) == 0

    assert [comment.body for comment in gh.get_pr_comments(pr.number)] == bodies_before
    assert gh.edited_bodies == []


def test_rerun_after_partial_waiting_state_picks_up_without_retriggering() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    gh.seed_thread("thread-1", pr.number, path="src/app.py")
    seed_state(gh, pr.number, status="waiting", trigger_history=[trigger()])
    reviewer = FakeReviewerAdapter(findings=[finding()])
    coder = FakeCoderAdapter(coder_result(changed=False))

    assert Runner(context(gh=gh, reviewer=reviewer, coder=coder)).run(pr.number) == 0

    state = current_state(gh)
    assert state.status == "done"
    assert state.done_reason == "completed"
    assert not any("aipro-review-trigger fake" in body for body in gh.posted_bodies)
    assert len(coder.calls) == 1


def test_collecting_resume_uses_real_collection_and_handling_flow() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    gh.seed_thread("thread-1", pr.number, path="src/app.py")
    seed_state(gh, pr.number, status="collecting", trigger_history=[trigger()])
    reviewer = FakeReviewerAdapter(findings=[finding()])
    coder = FakeCoderAdapter(coder_result(changed=False))

    assert Runner(context(gh=gh, reviewer=reviewer, coder=coder)).run(pr.number) == 0

    assert current_state(gh).status == "done"
    assert reviewer.collect_calls >= 1
    assert len(coder.calls) == 1


def test_head_sha_race_discards_coder_output_and_restarts() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    seed_state(gh, pr.number, status="handling", trigger_history=[trigger()])
    reviewer = FakeReviewerAdapter(findings=[finding()])

    def clear_findings_after_stale_reset() -> None:
        if reviewer.collect_calls > 1:
            reviewer.findings = []
            reviewer.responded = True

    reviewer.on_collect = clear_findings_after_stale_reset
    git = FakeGitRepo(head_sha=pr.head_sha, remote_heads=[pr.head_sha, "head-2"])
    coder = FakeCoderAdapter(
        coder_result(changed=True),
        on_run=lambda _task: setattr(git, "clean", False),
    )

    assert Runner(context(gh=gh, reviewer=reviewer, coder=coder, git=git)).run(pr.number) == 0

    assert len(coder.calls) == 1
    assert git.commits == []
    assert any(
        status == "init" for status in gh.state_statuses[gh.state_statuses.index("handling") + 1 :]
    )
    assert current_state(gh).last_error in {None, "stale_coder_output_discarded"}


def test_ci_wait_exits_process_and_later_check_event_completes() -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    seed_state(
        gh,
        pr.number,
        status="ci_wait",
        head_sha=pr.head_sha,
        ci_wait_started_at=NOW,
    )
    gh.seed_check_run(pr.head_sha, name="tests", status="in_progress", conclusion=None)

    assert Runner(context(gh=gh)).run(pr.number) == 0
    assert current_state(gh).status == "ci_wait"

    gh.set_check_runs(
        pr.head_sha,
        [gh_models.CheckRun(id=1, name="tests", status="completed", conclusion="success")],
    )
    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha=pr.head_sha)

    assert Runner(context(gh=gh)).run(pr.number, event=event) == 0
    assert current_state(gh).status == "done"


def test_dry_run_plans_without_mutations(capsys: pytest.CaptureFixture[str]) -> None:
    gh = TrackingGitHubClient()
    pr = seed_pr(gh)
    seed_state(gh, pr.number, status="triggering", round_index=1)
    posted_before = list(gh.posted_bodies)
    edited_before = list(gh.edited_bodies)

    assert Runner(context(gh=gh, dry_run=True)).run(pr.number) == 0

    out = capsys.readouterr().out
    assert "DRY-RUN PR #14" in out
    assert "would post a comment triggering reviewer 'fake'" in out
    assert gh.posted_bodies == posted_before
    assert gh.edited_bodies == edited_before
