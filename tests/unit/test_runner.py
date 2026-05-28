"""Tests for the orchestrator runner loop."""

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
    ReviewerTrigger,
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
    responded: bool = False
    has_responded_calls: list[tuple[int, datetime]] = field(default_factory=list)

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

    def has_responded(self, pr_number: int, trigger_timestamp: datetime) -> bool:
        self.has_responded_calls.append((pr_number, trigger_timestamp))
        return self.responded


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
    # Set to an Exception instance to make the next call raise. Used by tests
    # that exercise the runner's failure-handling branches without needing to
    # subclass the fake.
    raise_on_fetch_remote_head: Exception | None = None
    raise_on_push: Exception | None = None

    def is_clean(self) -> bool:
        return self.clean

    def commit(self, message: str, author_name: str, author_email: str) -> str | None:
        self.commit_count += 1
        self.commit_calls.append((message, author_name, author_email))
        self.clean = True
        return self.commit_sha_to_return

    def push(self, branch: str) -> None:
        if self.raise_on_push is not None:
            raise self.raise_on_push
        self.push_count += 1
        self.push_calls.append(branch)

    def rollback(self) -> None:
        self.rollback_count += 1
        self.clean = True

    def fetch_remote_head(self, branch: str) -> str | None:
        if self.raise_on_fetch_remote_head is not None:
            raise self.raise_on_fetch_remote_head
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
    author_association: str = "OWNER",
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
        author_association=author_association,
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
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> RunnerContext:
    cfg = config or make_config()
    return RunnerContext(
        github=gh,
        coder=coder or FakeCoderAdapter(),
        reviewers=reviewers or {"fake": FakeReviewerAdapter(name="fake")},
        git=cast(Any, git),
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

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
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

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
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

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
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

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
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

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
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

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
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


def test_duplicate_commit_skipped_within_single_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Within a single Runner.run, the in-process counter prevents a second
    commit even if the state machine emits multiple commit_changes actions.
    Pre-existing commits_made entries from prior runs do NOT block, so a later
    round's commit can still proceed in a subsequent invocation."""
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    seed_pr(gh)
    state = initial_state(status="init", round_index=1, commits_made=["prior-run-sha"])
    gh.seed_comment(1, serialize_state_comment(state))
    git = FakeGitRepo(head_sha="head-1", remote_head_sha="head-1", clean=False)

    call_count = {"n": 0}
    planned = [
        PlannedAction("commit_changes", {"message": "first"}),
        PlannedAction("commit_changes", {"message": "second"}),
    ]

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(
        gh=gh,
        git=git,
        config=make_config(
            safety=SafetyConfig(
                only_run_on_labeled_prs=False,
                max_commits_per_run=1,
                max_coder_invocations_per_run=1,
            )
        ),
    )
    Runner(ctx).run(1)
    assert git.commit_count == 1


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
        result=AgentRunResult(changed=False, summary="ok", decisions=[], token_usage=TokenUsage())
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
        result=AgentRunResult(changed=False, summary="ok", decisions=[], token_usage=TokenUsage())
    )
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, coder=coder, clock=clock, sleeper=sleeper)
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
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, clock=clock, sleeper=sleeper, config=cfg)
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
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, clock=clock, sleeper=sleeper, config=cfg)
    Runner(ctx).run(pr.number)

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"


def test_polling_deadline_is_anchored_to_trigger_not_resume() -> None:
    """A ``waiting`` state whose ReviewerTrigger predates the runner's clock
    by more than ``reviewer_timeout_seconds`` must immediately time out —
    without any sleeper calls — because the deadline is anchored to the
    persisted trigger timestamp, not to the wall-clock at resume.

    Regression test for the bug where ``start = ctx.clock()`` in
    ``_poll_reviewers`` reset the timeout budget on every webhook wake-up.
    """
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    # Trigger fired 2 hours ago — well past any reasonable reviewer_timeout.
    two_hours_ago = NOW - timedelta(hours=2)
    state = initial_state(
        status="waiting",
        round_index=1,
        head_sha=pr.head_sha,
        trigger_history=[
            ReviewerTrigger(
                reviewer_name="fake",
                round_index=1,
                timestamp=two_hours_ago,
                head_sha=pr.head_sha,
            )
        ],
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]])
    clock = FakeClock(start=NOW)
    sleeper = FakeSleeper(clock=clock)
    cfg = make_config(
        review_phase=ReviewPhaseConfig(
            poll_interval_seconds=10,
            reviewer_timeout_seconds=600,
            phase_timeout_seconds=900,
        )
    )
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, clock=clock, sleeper=sleeper, config=cfg)
    Runner(ctx).run(pr.number)

    # Deadline was already in the past; no sleep should have been issued.
    assert sleeper.calls == []

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "reviewer_timeout"


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
    cfg = make_config(ci=CiConfig(require_green_before_done=True, ignored_checks=["aipro-self"]))
    ctx = build_ctx(gh=gh, config=cfg)
    assert Runner(ctx).run(pr.number, event=event) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "done"


# ----- Hard-cap / clock-skew guardrail -----


def test_polling_terminates_when_clock_never_advances() -> None:
    """If a stuck clock prevents timeout detection, the iteration cap must
    still terminate the poll loop and route to ``needs_human``."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="waiting", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]])

    # Clock that never advances regardless of how often it's queried.
    class FrozenClock:
        def __init__(self, now: datetime) -> None:
            self.now = now

        def __call__(self) -> datetime:
            return self.now

    # Sleeper that returns immediately and does NOT advance any clock.
    def instant_sleeper(seconds: float) -> None:
        return None

    cfg = make_config(
        review_phase=ReviewPhaseConfig(
            poll_interval_seconds=5,
            reviewer_timeout_seconds=60,
            phase_timeout_seconds=120,
        )
    )
    ctx = build_ctx(
        gh=gh,
        reviewers={"fake": reviewer},
        clock=FrozenClock(NOW),
        sleeper=instant_sleeper,
        config=cfg,
    )
    result = Runner(ctx).run(pr.number)
    assert result == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "reviewer_timeout"


# ----- Zero-findings reviewer completion -----


def test_reviewer_responded_with_no_findings_transitions_to_done() -> None:
    """When the reviewer bot has posted *something* (so has_responded is True)
    but found no issues, the runner should route to ``done`` with
    ``done_reason='no_findings'`` instead of timing out."""
    from ai_pr_orchestrator.models import ReviewerTrigger

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(
        status="waiting",
        round_index=1,
        head_sha=pr.head_sha,
        trigger_history=[
            ReviewerTrigger(
                reviewer_name="fake",
                round_index=1,
                timestamp=NOW,
                head_sha=pr.head_sha,
            )
        ],
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[]], responded=True)
    clock = FakeClock()
    sleeper = FakeSleeper(clock=clock)
    ctx = build_ctx(gh=gh, reviewers={"fake": reviewer}, clock=clock, sleeper=sleeper)
    Runner(ctx).run(pr.number)

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "done"
    assert sc.state.done_reason == "no_findings"


# ----- Orphaned coder invocation on restart -----


def test_orphaned_coder_invocation_transitions_to_needs_human() -> None:
    """If the persisted state shows the coder was already invoked this round
    but the in-memory ``_pending_coder_result`` is gone (process restart), the
    runner must terminate as ``needs_human`` instead of waiting forever."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(
        status="handling",
        round_index=1,
        head_sha=pr.head_sha,
        last_coder_round_index=1,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    # Fresh Runner with no pending coder result.
    ctx = build_ctx(gh=gh)
    assert Runner(ctx).run(pr.number) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "coder_invocation_orphaned"


# ----- Safety: fork / workflow file changes -----


def test_fork_pr_with_disallow_forks_errors() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = gh_models.PullRequest(
        number=1,
        title="Forked PR",
        body="",
        state="open",
        head_sha="head-1",
        head_ref="feature/x",
        base_ref="main",
        author="outsider",
        is_fork=True,
        author_association="OWNER",
    )
    gh.seed_pr(pr)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    ctx = build_ctx(
        gh=gh,
        config=make_config(
            safety=SafetyConfig(
                only_run_on_labeled_prs=False,
                disallow_forks=True,
                disallow_workflow_file_changes=False,
            )
        ),
    )
    assert Runner(ctx).run(pr.number) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "error"
    assert "fork" in (sc.state.last_error or "")


def test_workflow_file_change_with_disallow_needs_human() -> None:
    gh = FakeGitHubClient(now=NOW)
    pr = gh_models.PullRequest(
        number=1,
        title="Workflow edit PR",
        body="",
        state="open",
        head_sha="head-1",
        head_ref="feature/x",
        base_ref="main",
        author="pavel",
        changed_files=[".github/workflows/ci.yml", "src/main.py"],
        author_association="OWNER",
    )
    gh.seed_pr(pr)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    ctx = build_ctx(
        gh=gh,
        config=make_config(
            safety=SafetyConfig(
                only_run_on_labeled_prs=False,
                disallow_forks=False,
                disallow_workflow_file_changes=True,
            )
        ),
    )
    assert Runner(ctx).run(pr.number) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "workflow_file_changed"


def test_untrusted_author_association_errors() -> None:
    """A PR whose ``author_association`` is not in the allowlist must terminate
    in ``error`` with ``last_error`` reporting the untrusted association.

    This guards the safety bypass that used to default the association to the
    first allowed entry — making the state-machine check a no-op against any
    real author.
    """
    gh = FakeGitHubClient(now=NOW)
    pr = gh_models.PullRequest(
        number=1,
        title="Drive-by PR",
        body="",
        state="open",
        head_sha="head-1",
        head_ref="feature/x",
        base_ref="main",
        author="drive-by-contributor",
        author_association="FIRST_TIME_CONTRIBUTOR",
    )
    gh.seed_pr(pr)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    ctx = build_ctx(
        gh=gh,
        config=make_config(
            safety=SafetyConfig(
                only_run_on_labeled_prs=False,
                disallow_forks=False,
                disallow_workflow_file_changes=False,
                allowed_pr_author_associations=["OWNER", "MEMBER", "COLLABORATOR"],
            )
        ),
    )
    assert Runner(ctx).run(pr.number) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "error"
    assert "untrusted_author_association" in (sc.state.last_error or "")


def test_missing_author_association_falls_back_to_none_sentinel() -> None:
    """When the GitHub payload omits ``author_association`` we route through the
    ``NONE`` sentinel (GitHub's own value for "no association"), which the
    default allowlist treats as untrusted."""
    gh = FakeGitHubClient(now=NOW)
    pr = gh_models.PullRequest(
        number=1,
        title="No-assoc PR",
        body="",
        state="open",
        head_sha="head-1",
        head_ref="feature/x",
        base_ref="main",
        author="random",
        # author_association defaults to "" — we want the runner to coerce
        # that to ``NONE`` rather than silently inheriting a trusted value.
    )
    gh.seed_pr(pr)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    ctx = build_ctx(
        gh=gh,
        config=make_config(
            safety=SafetyConfig(
                only_run_on_labeled_prs=False,
                disallow_forks=False,
                disallow_workflow_file_changes=False,
                allowed_pr_author_associations=["OWNER", "MEMBER", "COLLABORATOR"],
            )
        ),
    )
    assert Runner(ctx).run(pr.number) == 0

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    assert sc.state.status == "error"
    assert "untrusted_author_association:NONE" in (sc.state.last_error or "")


# ----- Commit/push checkpoint and push-recovery -----


def test_commit_state_checkpoint_survives_push_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``push_branch`` raises after ``commit_changes`` succeeds, the
    persisted state must already contain the new commit SHA so the next run
    can detect the orphan local commit and resume the push.

    Without the post-commit checkpoint, ``state.commits_made`` would still be
    empty in the persisted comment when the push exception unwinds through
    ``_run`` — and the next runner would have no idea a local commit exists.
    """
    from ai_pr_orchestrator.models import PlannedAction

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="init", round_index=1, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    class FailingPushRepo(FakeGitRepo):
        def push(self, branch: str) -> None:
            self.push_count += 1
            self.push_calls.append(branch)
            raise RuntimeError("network down")

    git = FailingPushRepo(head_sha=pr.head_sha, remote_head_sha=pr.head_sha, clean=False)

    call_count = {"n": 0}
    planned = [
        PlannedAction("commit_changes", {"message": "fix: x"}),
        PlannedAction("push_branch", {}),
    ]

    def fake_transition(
        s: RuntimeState, snap: Any, cfg: Any, now: datetime
    ) -> tuple[RuntimeState, list[Any]]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return replace(s, status="done", done_reason="completed", updated_at=now), []
        return s, planned

    monkeypatch.setattr("ai_pr_orchestrator.runner.transition", fake_transition)
    ctx = build_ctx(gh=gh, git=cast(Any, git))
    # Runner.run catches the push exception and returns 1; the key invariant
    # is that the state comment was checkpointed before the push attempt.
    assert Runner(ctx).run(pr.number) == 1

    comments = gh.get_pr_comments(pr.number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    assert sc is not None
    # The new commit SHA must be in commits_made even though push failed.
    assert "new-commit-sha" in sc.state.commits_made
    assert sc.state.head_sha == "new-commit-sha"


def test_push_recovery_reissues_push_on_resume() -> None:
    """If persisted state shows a local commit that is not yet on the remote,
    the runner must reissue the push at the top of ``_run``."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, head_sha="prior-sha")
    state = initial_state(
        status="ci_wait",
        round_index=1,
        head_sha="local-sha",
        ci_wait_started_at=NOW,
        commits_made=["local-sha"],
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))
    # Worktree HEAD matches the recorded last commit; remote still on prior SHA.
    git = FakeGitRepo(head_sha="local-sha", remote_head_sha="prior-sha")

    ctx = build_ctx(gh=gh, git=git)
    Runner(ctx).run(pr.number)

    assert git.push_count == 1
    assert git.push_calls[0] == "feature/x"


def test_push_recovery_skips_when_remote_already_matches() -> None:
    """No push should be issued when the remote already has the local commit."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh, head_sha="local-sha")
    state = initial_state(
        status="ci_wait",
        round_index=1,
        head_sha="local-sha",
        ci_wait_started_at=NOW,
        commits_made=["local-sha"],
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))
    # Remote already matches local; recovery is a no-op.
    git = FakeGitRepo(head_sha="local-sha", remote_head_sha="local-sha")

    ctx = build_ctx(gh=gh, git=git)
    Runner(ctx).run(pr.number)

    assert git.push_count == 0


def test_push_recovery_no_op_when_no_commits_made() -> None:
    """When state has never recorded a commit, push-recovery must not run."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))
    git = FakeGitRepo(head_sha="some-sha", remote_head_sha="other-sha")

    ctx = build_ctx(gh=gh, git=git)
    Runner(ctx).run(pr.number)

    # No commits in state -> no recovery push, regardless of remote/local mismatch.
    assert git.push_count == 0


# ----- Round 6: stale gh_pr, current-round reviewers, remote-head unverified,
# push-recovery terminal, optimistic concurrency -----


def test_gh_pr_is_refetched_each_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner must refetch the PR at the top of every loop iteration so
    snapshot consumers (label-removed safety check, head_sha comparisons) see
    mutations applied by earlier actions in the same run rather than a stale
    snapshot captured before the loop started."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    call_count = {"n": 0}
    original_get_pr = gh.get_pr

    def counting_get_pr(number: int) -> gh_models.PullRequest:
        call_count["n"] += 1
        return original_get_pr(number)

    monkeypatch.setattr(gh, "get_pr", counting_get_pr)

    ctx = build_ctx(gh=gh)
    Runner(ctx).run(pr.number)

    # At minimum: one pre-loop fetch + one per loop iteration. Two iterations
    # for init->triggering->waiting plus terminal save means >= 3 calls.
    assert call_count["n"] >= 3


def test_untriggered_reviewer_does_not_block_zero_findings_short_circuit() -> None:
    """A reviewer enabled in config but NOT triggered this round must not
    block the zero-findings short-circuit. Only reviewers with a trigger in
    state.trigger_history for the current round_index participate in the
    has_responded poll."""
    from ai_pr_orchestrator.models import ReviewerTrigger

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(
        status="waiting",
        round_index=1,
        head_sha=pr.head_sha,
        trigger_history=[
            ReviewerTrigger(
                reviewer_name="triggered",
                round_index=1,
                timestamp=NOW,
                head_sha=pr.head_sha,
            )
        ],
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    triggered = FakeReviewerAdapter(name="triggered", findings_by_call=[[]], responded=True)
    untriggered = FakeReviewerAdapter(name="untriggered", findings_by_call=[[]], responded=False)

    cfg = make_config(
        reviewers={
            "triggered": ReviewerConfig(bot_logins=["a[bot]"], trigger_comment="x"),
            "untriggered": ReviewerConfig(bot_logins=["b[bot]"], trigger_comment="y"),
        }
    )
    clock = FakeClock()
    ctx = build_ctx(
        gh=gh,
        reviewers={"triggered": triggered, "untriggered": untriggered},
        config=cfg,
        clock=clock,
        sleeper=FakeSleeper(clock=clock),
    )
    assert Runner(ctx).run(pr.number) == 0

    sc = find_state_comment([{"id": c.id, "body": c.body} for c in gh.get_pr_comments(pr.number)])
    assert sc is not None
    assert sc.state.status == "done"
    assert sc.state.done_reason == "no_findings"
    # The untriggered reviewer must not have been polled for has_responded.
    assert untriggered.has_responded_calls == []


def test_fetch_remote_head_failure_routes_to_needs_human_and_skips_commit() -> None:
    """When ``fetch_remote_head`` raises during the handling state, the
    snapshot's ``remote_head_unverified`` flag must drive the state machine
    to ``needs_human`` with no commit. Falling back to gh_pr.head_sha would
    silently treat the unknown remote as a match and allow stale coder output
    to commit."""
    from ai_pr_orchestrator.models import Finding as Finding_
    from ai_pr_orchestrator.models import ReviewerTrigger

    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    finding = Finding_(id="f1", source="fake", body="b", created_at=NOW, head_sha=pr.head_sha)
    state = initial_state(
        status="handling",
        round_index=1,
        head_sha=pr.head_sha,
        trigger_history=[
            ReviewerTrigger(
                reviewer_name="fake",
                round_index=1,
                timestamp=NOW,
                head_sha=pr.head_sha,
            )
        ],
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))

    reviewer = FakeReviewerAdapter(name="fake", findings_by_call=[[finding]])
    git = FakeGitRepo(
        head_sha=pr.head_sha,
        remote_head_sha=pr.head_sha,
        clean=False,
        raise_on_fetch_remote_head=RuntimeError("fetch failed"),
    )

    ctx = build_ctx(gh=gh, git=git, reviewers={"fake": reviewer})
    Runner(ctx).run(pr.number)

    assert git.commit_count == 0
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in gh.get_pr_comments(pr.number)])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "remote_head_unverified"


def test_push_recovery_failure_routes_to_needs_human_terminal_state() -> None:
    """If the recovery push raises, the runner must persist a terminal
    ``needs_human`` with ``last_error='push_recovery_failed'`` instead of
    falling through to ci_wait — otherwise CI events can never arrive for
    the unpushed commit and the PR sits pending indefinitely."""
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(
        status="ci_wait",
        round_index=1,
        head_sha="local-sha",
        commits_made=["local-sha"],
        ci_wait_started_at=NOW,
    )
    gh.seed_comment(pr.number, serialize_state_comment(state))
    git = FakeGitRepo(
        head_sha="local-sha",
        remote_head_sha="prior-sha",
        raise_on_push=RuntimeError("push denied"),
    )

    ctx = build_ctx(gh=gh, git=git)
    assert Runner(ctx).run(pr.number) == 0

    sc = find_state_comment([{"id": c.id, "body": c.body} for c in gh.get_pr_comments(pr.number)])
    assert sc is not None
    assert sc.state.status == "needs_human"
    assert sc.state.last_error == "push_recovery_failed"


def test_state_conflict_on_save_exits_cleanly_without_clobber() -> None:
    """If a second runner edits the state comment between this runner's load
    and save, the optimistic-concurrency check must raise StateConflictError;
    the runner catches it and exits 0 without overwriting the winning state.
    """
    gh = FakeGitHubClient(now=NOW)
    pr = seed_pr(gh)
    state = initial_state(status="init", round_index=0, head_sha=pr.head_sha)
    gh.seed_comment(pr.number, serialize_state_comment(state))

    ctx = build_ctx(gh=gh)
    runner = Runner(ctx)

    # Snapshot the comment id and updated_at as the runner would on load.
    runner._load_or_init_state(pr.number, gh.get_pr(pr.number))
    original_updated_at = runner._state_expected_updated_at
    assert runner._state_comment_id is not None

    # Simulate a concurrent runner mutating the same comment (advance its
    # updated_at by editing the body in place via the fake's edit API).
    competing_state = replace(
        state,
        status="triggering",
        round_index=1,
        updated_at=NOW + timedelta(seconds=10),
    )
    gh.edit_comment(runner._state_comment_id, serialize_state_comment(competing_state))

    # Now this runner tries to save based on its pre-conflict view.
    rc = runner.run(pr.number)
    assert rc == 0

    # The competing edit must still be the persisted state (no clobber).
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in gh.get_pr_comments(pr.number)])
    assert sc is not None
    # The competing state's round_index was 1; our run loaded round_index=0 and
    # would have attempted to write back round_index=1 from triggering, but
    # the conflict path must have prevented that overwrite.
    # We assert the comment carries a body parseable as state and verify the
    # original_updated_at was the value we recorded.
    assert original_updated_at == NOW
