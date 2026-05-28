"""Tests for the orchestrator runner loop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from ai_pr_orchestrator.models import (
    AgentRunResult,
    Decision,
    Finding,
    FixTask,
    RuntimeState,
    TokenUsage,
)
from ai_pr_orchestrator.runner import ParsedEvent, Runner, RunnerContext
from ai_pr_orchestrator.state_storage import find_state_comment, serialize_state_comment

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


# ----- Test doubles -----


@dataclass
class FakeCoderAdapter:
    name: str = "fake-coder"
    result: AgentRunResult = field(
        default_factory=lambda: AgentRunResult(
            changed=False, summary="no-op", decisions=[], token_usage=TokenUsage()
        )
    )
    calls: list[FixTask] = field(default_factory=list)

    def run_fix_task(self, task: FixTask) -> AgentRunResult:
        self.calls.append(task)
        return self.result


@dataclass
class FakeReviewerAdapter:
    name: str
    bot_login: str = "fake-bot[bot]"
    trigger_comment: str = "/fake review"
    findings_by_call: list[list[Finding]] = field(default_factory=list)
    call_count: int = 0

    def matches_author(self, login: str) -> bool:
        return login == self.bot_login

    def build_trigger_comment(self, round_index: int, head_sha: str) -> str:
        return self.trigger_comment

    def collect_findings(
        self, pr_number: int, head_sha: str, trigger_timestamp: datetime
    ) -> list[Finding]:
        idx = self.call_count
        self.call_count += 1
        if idx < len(self.findings_by_call):
            return list(self.findings_by_call[idx])
        if self.findings_by_call:
            return list(self.findings_by_call[-1])
        return []


@dataclass
class FakeGitRepo:
    head_sha: str = "head-1"
    remote_head_sha: str | None = None
    clean: bool = True
    diff_text: str = ""
    commit_count: int = 0
    push_count: int = 0
    rollback_count: int = 0
    commit_calls: list[tuple[str, str, str]] = field(default_factory=list)
    push_calls: list[str] = field(default_factory=list)
    commit_sha_to_return: str | None = "new-commit-sha"

    def is_clean(self) -> bool:
        return self.clean

    def commit(self, message: str, author_name: str, author_email: str) -> str | None:
        self.commit_count += 1
        self.commit_calls.append((message, author_name, author_email))
        self.clean = True
        return self.commit_sha_to_return

    def push(self, branch: str) -> None:
        self.push_count += 1
        self.push_calls.append(branch)

    def rollback(self) -> None:
        self.rollback_count += 1
        self.clean = True

    def fetch_remote_head(self, branch: str) -> str | None:
        return self.remote_head_sha

    def get_head_sha(self) -> str:
        return self.head_sha

    def get_diff(self, base: str) -> str:
        return self.diff_text


class FakeClock:
    def __init__(self, start: datetime = NOW) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def __call__(self) -> datetime:
        return self.now


class FakeSleeper:
    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self.clock = clock

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.clock is not None:
            self.clock.advance(seconds)


# ----- Helpers -----


def make_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "main_coder": MainCoderConfig(provider="codex_cli"),
        "reviewers": {
            "fake": ReviewerConfig(
                bot_logins=["fake-bot[bot]"],
                trigger_comment="/fake review",
            )
        },
        "review_phase": ReviewPhaseConfig(
            poll_interval_seconds=5,
            reviewer_timeout_seconds=60,
            phase_timeout_seconds=120,
        ),
        "ci": CiConfig(
            require_green_before_done=False,
            ignored_checks=["aipro-self"],
        ),
        "safety": SafetyConfig(
            only_run_on_labeled_prs=False,
            disallow_forks=True,
            disallow_workflow_file_changes=False,
            max_total_iterations=3,
            max_coder_invocations_per_run=1,
            max_commits_per_run=1,
            max_reviewer_triggers_per_run=3,
            max_prompt_tokens=100000,
        ),
        "git": GitConfig(base_branch="main"),
    }
    defaults.update(overrides)
    return Config(**defaults)


def seed_pr(
    gh: FakeGitHubClient,
    *,
    number: int = 1,
    head_sha: str = "head-1",
    head_ref: str = "feature/x",
    base_ref: str = "main",
    author: str = "pavel",
    labels: list[str] | None = None,
) -> gh_models.PullRequest:
    pr = gh_models.PullRequest(
        number=number,
        title="Test PR",
        body="",
        state="open",
        head_sha=head_sha,
        head_ref=head_ref,
        base_ref=base_ref,
        author=author,
        labels=labels or [],
    )
    gh.seed_pr(pr)
    return pr


def build_ctx(
    *,
    gh: FakeGitHubClient,
    coder: FakeCoderAdapter | None = None,
    reviewers: dict[str, Any] | None = None,
    git: FakeGitRepo | None = None,
    config: Config | None = None,
    clock: FakeClock | None = None,
    sleeper: FakeSleeper | None = None,
) -> RunnerContext:
    cfg = config or make_config()
    return RunnerContext(
        github=gh,
        coder=coder or FakeCoderAdapter(),
        reviewers=reviewers or {"fake": FakeReviewerAdapter(name="fake")},
        git=git,
        config=cfg,
        clock=clock or FakeClock(),
        sleeper=sleeper or FakeSleeper(),
    )


def initial_state(pr_number: int = 1, head_sha: str = "head-1", **overrides: Any) -> RuntimeState:
    defaults: dict[str, Any] = {
        "version": 1,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "status": "init",
        "round_index": 0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return RuntimeState(**defaults)


# ----- Main loop tests -----


def test_run_loads_state_and_runs_to_terminal_done() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="waiting", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]])
    ctx = build_ctx(
        gh=gh,
        reviewers={"fake": reviewer},
        config=make_config(),
    )
    # Trick: waiting state without findings -> we need reviewer_responded path.
    # Simpler: use `triggering` state so loop progresses to waiting then handling
    # via collected findings.
    finding = Finding(id="f1", source="fake", body="b", created_at=NOW, thread_id="t1")
    reviewer.findings_by_call = [[finding]]

    coder = FakeCoderAdapter(
        result=AgentRunResult(
            changed=False,
            summary="ok",
            decisions=[
                Decision(
                    finding_id="f1",
                    verdict="accepted",
                    confidence="high",
                    reason="ok",
                    reply="thanks",
                    should_resolve=True,
                    thread_id="t1",
                )
            ],
            token_usage=TokenUsage(),
        )
    )
    gh.seed_thread("t1", pr.number)
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, coder=coder, config=make_config())

    result = Runner(ctx).run(pr.number)
    assert result == 0

    final_comments = gh.get_pr_comments(pr.number)
    state_comment = find_state_comment([{"id": c.id, "body": c.body} for c in final_comments])
    assert state_comment is not None
    assert state_comment.state.status == "done"


def test_run_terminates_on_each_terminal_status() -> None:
    for terminal in ("done", "error", "needs_human"):
        gh = FakeGitHubClient(now=NOW)
        pr = seed_pr(gh)
        state = initial_state(status=terminal, head_sha=pr.head_sha, done_reason="completed")
        gh.seed_comment(pr.number, serialize_state_comment(state))

        ctx = build_ctx(gh=gh)
        assert Runner(ctx).run(pr.number) == 0


def test_at_most_one_coder_invocation_per_run() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    finding = Finding(id="f1", source="fake", body="b", created_at=NOW, thread_id="t1")
    gh.seed_thread("t1", pr.number)
    state = initial_state(status="handling", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[finding]])
    coder = FakeCoderAdapter(
        result=AgentRunResult(
            changed=False,
            summary="ok",
            decisions=[
                Decision(
                    finding_id="f1",
                    verdict="accepted",
                    confidence="high",
                    reason="ok",
                    reply="thanks",
                    should_resolve=True,
                    thread_id="t1",
                )
            ],
            token_usage=TokenUsage(),
        )
    )
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, coder=coder)

    assert Runner(ctx).run(pr.number) == 0
    assert len(coder.calls) == 1


def test_at_most_one_commit_per_run() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    finding = Finding(id="f1", source="fake", body="b", created_at=NOW, thread_id="t1")
    gh.seed_thread("t1", pr.number)
    state = initial_state(status="handling", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[finding]])
    coder = FakeCoderAdapter(
        result=AgentRunResult(
            changed=True,
            summary="ok",
            decisions=[
                Decision(
                    finding_id="f1",
                    verdict="accepted",
                    confidence="high",
                    reason="ok",
                    reply="fixed",
                    should_resolve=True,
                    thread_id="t1",
                )
            ],
            commit_message="fix: stuff",
            token_usage=TokenUsage(),
        )
    )
    git = FakeGitRepo(head_sha=pr.head_sha, remote_head_sha=pr.head_sha, clean=False)
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, coder=coder, git=git)

    assert Runner(ctx).run(pr.number) == 0
    assert git.commit_count == 1


def test_run_initializes_fresh_state_when_no_state_comment_exists() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]])
    ctx = build_ctx(
        gh=gh,
        reviewers={"fake": reviewer},
        config=make_config(
            review_phase=ReviewPhaseConfig(
                poll_interval_seconds=1,
                reviewer_timeout_seconds=1,
                phase_timeout_seconds=1,
            )
        ),
    )
    # Force timeout to terminate
    clock = FakeClock()
    sleeper = FakeSleeper(clock=clock)
    ctx = replace(ctx, clock=clock, sleeper=sleeper)

    Runner(ctx).run(pr.number)

    comments = gh.get_pr_comments(pr.number)
    state_comment = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert state_comment is not None
    assert state_comment.state.pr_number == pr.number


# ----- Action execution tests -----


def _setup_action_test(
    monkeypatch: pytest.MonkeyPatch, planned_actions: list[Any], terminal_status: str = "done"
) -> tuple[FakeGitHubClient, FakeGitRepo, FakeCoderAdapter, RunnerContext]:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="init", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))
    git = FakeGitRepo(head_sha=pr.head_sha, remote_head_sha=pr.head_sha, clean=False)
    coder = FakeCoderAdapter()

    call_count = {"n": 0}
    actions_iter = iter([planned_actions, []])

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        try:
            actions = next(actions_iter)
        except StopIteration:
            actions = []
        if call_count["n"] >= 2:
            return replace(s, status=terminal_status, done_reason="completed", updated_at=now), []
        return s, actions

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)

    ctx = build_ctx(gh=gh, coder=coder, git=git)
    return gh, git, coder, ctx


def test_post_pr_comment_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("post_pr_comment", {"body": "hello"})]
    gh, _git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    bodies = [c.body for c in gh.get_pr_comments(1)]
    assert any(b == "hello" for b in bodies)


def test_update_status_comment_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("update_status_comment", {"status": "triggering"})]
    gh, _git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    comments = gh.get_pr_comments(1)
    state_comment = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert state_comment is not None


def test_reply_to_thread_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    gh.seed_thread("t1", pr.number)
    state = initial_state(status="init", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    call_count = {"n": 0}
    planned = [PlannedAction("reply_to_thread", {"thread_id": "t1", "body": "reply!"})]

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(gh=gh)
    Runner(ctx).run(pr.number)

    threads = gh.get_review_threads(pr.number)
    assert any(any(c.body == "reply!" for c in t.comments) for t in threads)


def test_resolve_thread_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    gh.seed_thread("t1", pr.number)
    state = initial_state(status="init", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    call_count = {"n": 0}
    planned = [PlannedAction("resolve_thread", {"thread_id": "t1"})]

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(gh=gh)
    Runner(ctx).run(pr.number)

    threads = gh.get_review_threads(pr.number)
    assert any(t.id == "t1" and t.is_resolved for t in threads)


def test_commit_changes_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("commit_changes", {"message": "fix: x"})]
    _gh, git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    assert git.commit_count == 1
    assert git.commit_calls[0][0] == "fix: x"


def test_commit_changes_records_sha_in_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("commit_changes", {"message": "fix: x"})]
    gh, _git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    comments = gh.get_pr_comments(1)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert "new-commit-sha" in sc.state.commits_made


def test_push_branch_action_uses_head_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("push_branch", {})]
    _gh, git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    assert git.push_count == 1
    assert git.push_calls[0] == "feature/x"


def test_add_label_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("add_label", {"label": "ai-loop-done"})]
    gh, _git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    pr_after = gh.get_pr(1)
    assert "ai-loop-done" in pr_after.labels


def test_remove_label_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    seed_pr(gh, labels=["ai-loop"])
    state = initial_state(status="init", round_index=1)
    gh.seed_comment(1, serialize_state_comment(state))
    git = FakeGitRepo(head_sha="head-1", remote_head_sha="head-1")

    call_count = {"n": 0}
    planned = [PlannedAction("remove_label", {"label": "ai-loop"})]

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(gh=gh, git=git)
    Runner(ctx).run(1)
    pr_after = gh.get_pr(1)
    assert "ai-loop" not in pr_after.labels


def test_noop_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("noop", {"reason": "nothing"})]
    gh, git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    pre_comments = len(gh.get_pr_comments(1))
    Runner(ctx).run(1)
    assert git.commit_count == 0
    assert git.push_count == 0
    # status comment may be re-edited via save_state, but no new comments posted by noop.
    # We just verify no GitHub side effects from the action itself.
    assert len(gh.get_pr_comments(1)) == pre_comments


def test_post_final_summary_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("post_final_summary", {"reason": "completed"})]
    gh, _git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    comments = gh.get_pr_comments(1)
    assert any("completed" in c.body for c in comments)


def test_rollback_changes_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    actions = [PlannedAction("rollback_changes", {"reason": "regression"})]
    _gh, git, _coder, ctx = _setup_action_test(monkeypatch, actions)
    Runner(ctx).run(1)
    assert git.rollback_count == 1


def test_invoke_coder_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    finding = Finding(id="f1", source="fake", body="b", created_at=NOW, thread_id="t1")
    gh.seed_thread("t1", pr.number)
    state = initial_state(status="handling", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[finding]])
    coder = FakeCoderAdapter(
        result=AgentRunResult(
            changed=False,
            summary="ok",
            decisions=[],
            token_usage=TokenUsage(),
        )
    )
    git = FakeGitRepo(head_sha=pr.head_sha, remote_head_sha=pr.head_sha)
    call_count = {"n": 0}

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return s, [
                PlannedAction(
                    "invoke_coder",
                    {"finding_ids": ["f1"], "head_sha": pr.head_sha, "pr_number": pr.number},
                )
            ]
        # Verify snapshot has coder_result after invocation
        assert snap.coder_result is not None
        return replace(s, status="done", done_reason="completed", updated_at=now), []

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, coder=coder, git=git)
    Runner(ctx).run(pr.number)
    assert len(coder.calls) == 1


def test_actions_executed_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    seed_pr(gh, labels=["ai-loop"])
    state = initial_state(status="init", round_index=1)
    gh.seed_comment(1, serialize_state_comment(state))
    git = FakeGitRepo(head_sha="head-1", remote_head_sha="head-1")

    call_count = {"n": 0}
    planned = [
        PlannedAction("post_pr_comment", {"body": "first"}),
        PlannedAction("add_label", {"label": "second"}),
        PlannedAction("post_pr_comment", {"body": "third"}),
    ]

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(gh=gh, git=git)
    Runner(ctx).run(1)

    comments = gh.get_pr_comments(1)
    text_comments = [c.body for c in comments if c.body in ("first", "third")]
    assert text_comments == ["first", "third"]


def test_duplicate_commit_skipped_when_already_committed(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    seed_pr(gh)
    state = initial_state(status="init", round_index=1, commits_made=["existing-sha"])
    gh.seed_comment(1, serialize_state_comment(state))
    git = FakeGitRepo(head_sha="head-1", remote_head_sha="head-1", clean=False)

    call_count = {"n": 0}
    planned = [PlannedAction("commit_changes", {"message": "dup"})]

    def fake_transition(s: RuntimeState, snap: Any, cfg: Any, now: datetime) -> tuple[
        RuntimeState, list[Any]
    ]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(
        gh=gh,
        git=git,
        config=make_config(safety=SafetyConfig(
            only_run_on_labeled_prs=False,
            max_commits_per_run=1,
            max_coder_invocations_per_run=1,
        )),
    )
    Runner(ctx).run(1)
    assert git.commit_count == 0


# ----- Reviewer polling tests -----


def test_polling_calls_sleeper_at_configured_interval() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="waiting", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    # First call returns nothing, second returns a finding to terminate polling.
    finding = Finding(id="f1", source="fake", body="b", created_at=NOW)
    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[], [finding]])
    clock = FakeClock()
    sleeper = FakeSleeper(clock=clock)
    coder = FakeCoderAdapter(
        result=AgentRunResult(
            changed=False, summary="ok", decisions=[], token_usage=TokenUsage()
        )
    )
    cfg = make_config(
        review_phase=ReviewPhaseConfig(
            poll_interval_seconds=7,
            reviewer_timeout_seconds=600,
            phase_timeout_seconds=900,
        )
    )
    ctx = build_ctx(
        gh=gh,
        reviewers={"fake": reviewer},
        coder=coder,
        clock=clock,
        sleeper=sleeper,
        config=cfg,
    )
    Runner(ctx).run(pr.number)

    assert 7 in sleeper.calls


def test_polling_stops_when_findings_appear() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="waiting", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    finding = Finding(id="f1", source="fake", body="b", created_at=NOW)
    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[finding]])
    clock = FakeClock()
    sleeper = FakeSleeper(clock=clock)
    coder = FakeCoderAdapter(
        result=AgentRunResult(
            changed=False, summary="ok", decisions=[], token_usage=TokenUsage()
        )
    )
    ctx = build_ctx(
        gh=gh, reviewers={"fake": reviewer}, coder=coder, clock=clock, sleeper=sleeper
    )
    Runner(ctx).run(pr.number)
    # Findings appeared on first poll, so we should NOT have slept at all.
    assert sleeper.calls == []


def test_polling_times_out_with_reviewer_timeout() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="waiting", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]])
    clock = FakeClock()
    sleeper = FakeSleeper(clock=clock)
    cfg = make_config(
        review_phase=ReviewPhaseConfig(
            poll_interval_seconds=10,
            reviewer_timeout_seconds=20,
            phase_timeout_seconds=120,
        )
    )
    ctx = build_ctx(
        gh=gh, reviewers={"fake": reviewer}, clock=clock, sleeper=sleeper, config=cfg
    )
    Runner(ctx).run(pr.number)

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "reviewer_timeout"


def test_polling_times_out_with_phase_timeout() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="waiting", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]])
    clock = FakeClock()
    sleeper = FakeSleeper(clock=clock)
    # phase_timeout < reviewer_timeout to make phase budget bind first
    cfg = make_config(
        review_phase=ReviewPhaseConfig(
            poll_interval_seconds=5,
            reviewer_timeout_seconds=600,
            phase_timeout_seconds=10,
        )
    )
    ctx = build_ctx(
        gh=gh, reviewers={"fake": reviewer}, clock=clock, sleeper=sleeper, config=cfg
    )
    Runner(ctx).run(pr.number)

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"


# ----- CI resume tests -----


def test_ci_resume_all_green_completes() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    gh.seed_check_run(pr.head_sha, name="tests", status="completed", conclusion="success")
    state = initial_state(
        status="ci_wait", round_index=1, head_sha=pr.head_sha, ci_wait_started_at=NOW
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha=pr.head_sha)
    ctx = build_ctx(gh=gh, config=make_config(ci=CiConfig(require_green_before_done=True)))
    result = Runner(ctx).run(pr.number, event=event)
    assert result == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "done"


def test_ci_resume_failed_check_needs_human() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    gh.seed_check_run(pr.head_sha, name="tests", status="completed", conclusion="failure")
    state = initial_state(
        status="ci_wait", round_index=1, head_sha=pr.head_sha, ci_wait_started_at=NOW
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha=pr.head_sha)
    ctx = build_ctx(gh=gh, config=make_config(ci=CiConfig(require_green_before_done=True)))
    assert Runner(ctx).run(pr.number, event=event) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"


def test_ci_resume_pending_check_stays_in_ci_wait() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    gh.seed_check_run(pr.head_sha, name="tests", status="in_progress", conclusion=None)
    state = initial_state(
        status="ci_wait", round_index=1, head_sha=pr.head_sha, ci_wait_started_at=NOW
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha=pr.head_sha)
    ctx = build_ctx(gh=gh, config=make_config(ci=CiConfig(require_green_before_done=True)))
    assert Runner(ctx).run(pr.number, event=event) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "ci_wait"


def test_ci_resume_non_matching_sha_does_not_advance() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, head_sha="head-2")
    state = initial_state(
        status="ci_wait", round_index=1, head_sha="head-2", ci_wait_started_at=NOW
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha="something-else")
    ctx = build_ctx(gh=gh, config=make_config(ci=CiConfig(require_green_before_done=True)))
    assert Runner(ctx).run(pr.number, event=event) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "ci_wait"


def test_ci_ignored_checks_filtered() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    gh.seed_check_run(pr.head_sha, name="tests", status="completed", conclusion="success")
    gh.seed_check_run(pr.head_sha, name="aipro-self", status="completed", conclusion="failure")
    state = initial_state(
        status="ci_wait", round_index=1, head_sha=pr.head_sha, ci_wait_started_at=NOW
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    event = ParsedEvent(event_type="check_run", pr_number=pr.number, head_sha=pr.head_sha)
    cfg = make_config(
        ci=CiConfig(require_green_before_done=True, ignored_checks=["aipro-self"])
    )
    ctx = build_ctx(gh=gh, config=cfg)
    assert Runner(ctx).run(pr.number, event=event) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "done"
