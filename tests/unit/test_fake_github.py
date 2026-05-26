"""Tests for the in-memory FakeGitHubClient."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient, StaleEditError
from ai_pr_orchestrator.github.models import (
    CheckRun,
    Comment,
    PullRequest,
    ReviewComment,
    ReviewThread,
)
from ai_pr_orchestrator.models import RuntimeState
from ai_pr_orchestrator.state_storage import (
    find_state_comment,
    serialize_state_comment,
)

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _make_pr(number: int = 42, **overrides) -> PullRequest:
    defaults = dict(
        number=number,
        title="Test PR",
        body="PR body",
        state="open",
        head_sha="abc123",
        head_ref="feature",
        base_ref="main",
        author="author",
    )
    defaults.update(overrides)
    return PullRequest(**defaults)


def _make_client(**kwargs) -> FakeGitHubClient:
    return FakeGitHubClient(now=NOW, **kwargs)


# --- get_pr returns a PR with expected fields ---


def test_get_pr_returns_seeded_pr() -> None:
    client = _make_client()
    pr = _make_pr(42, labels=["bug", "v1"])
    client.seed_pr(pr)

    result = client.get_pr(42)

    assert isinstance(result, PullRequest)
    assert result.number == 42
    assert result.title == "Test PR"
    assert result.head_sha == "abc123"
    assert result.labels == ["bug", "v1"]
    assert result.author == "author"


def test_get_pr_raises_on_missing() -> None:
    client = _make_client()
    with pytest.raises(KeyError, match="PR #99"):
        client.get_pr(99)


# --- post_comment creates a comment retrievable by get_pr_comments ---


def test_post_comment_creates_retrievable_comment() -> None:
    client = _make_client()
    comment = client.post_comment(42, "hello world")

    assert isinstance(comment, Comment)
    assert comment.body == "hello world"
    assert comment.id >= 1

    comments = client.get_pr_comments(42)
    assert len(comments) == 1
    assert comments[0].body == "hello world"
    assert comments[0].id == comment.id


def test_post_comment_isolates_by_issue_number() -> None:
    client = _make_client()
    client.post_comment(1, "on issue 1")
    client.post_comment(2, "on issue 2")

    assert len(client.get_pr_comments(1)) == 1
    assert len(client.get_pr_comments(2)) == 1
    assert client.get_pr_comments(1)[0].body == "on issue 1"


# --- edit_comment updates body and updated_at ---


def test_edit_comment_updates_body_and_timestamp() -> None:
    client = _make_client()
    original = client.post_comment(42, "original")

    edited = client.edit_comment(original.id, "updated")

    assert edited.body == "updated"
    assert edited.updated_at != original.updated_at

    comments = client.get_pr_comments(42)
    assert comments[0].body == "updated"


def test_edit_comment_raises_on_missing() -> None:
    client = _make_client()
    with pytest.raises(KeyError, match="Comment #999"):
        client.edit_comment(999, "nope")


# --- get_review_threads returns threads with comments ---


def test_get_review_threads_returns_seeded_threads() -> None:
    client = _make_client()
    rc = ReviewComment(
        id="RC_1",
        body="fix this",
        author="reviewer",
        path="src/main.py",
        created_at=NOW.isoformat(),
    )
    client.seed_thread("RT_1", 42, path="src/main.py", comments=[rc])

    threads = client.get_review_threads(42)

    assert len(threads) == 1
    assert isinstance(threads[0], ReviewThread)
    assert threads[0].id == "RT_1"
    assert not threads[0].is_resolved
    assert len(threads[0].comments) == 1
    assert threads[0].comments[0].body == "fix this"


def test_get_review_threads_isolates_by_pr() -> None:
    client = _make_client()
    client.seed_thread("RT_1", 1)
    client.seed_thread("RT_2", 2)

    assert len(client.get_review_threads(1)) == 1
    assert len(client.get_review_threads(2)) == 1
    assert client.get_review_threads(1)[0].id == "RT_1"


# --- reply_to_review_thread appends a reply ---


def test_reply_to_review_thread_appends_comment() -> None:
    client = _make_client()
    client.seed_thread("RT_1", 42, path="file.py")

    result = client.reply_to_review_thread("RT_1", "done, fixed")

    assert result is not None
    assert "comment" in result

    threads = client.get_review_threads(42)
    assert len(threads[0].comments) == 1
    assert threads[0].comments[0].body == "done, fixed"


def test_reply_to_missing_thread_raises() -> None:
    client = _make_client()
    with pytest.raises(KeyError, match="Thread RT_missing"):
        client.reply_to_review_thread("RT_missing", "oops")


# --- resolve_review_thread marks as resolved ---


def test_resolve_review_thread_marks_resolved() -> None:
    client = _make_client()
    client.seed_thread("RT_1", 42)

    result = client.resolve_review_thread("RT_1")

    assert result is not None
    assert result["thread"]["isResolved"] is True

    threads = client.get_review_threads(42)
    assert threads[0].is_resolved is True


# --- resolve already-resolved thread is idempotent ---


def test_resolve_already_resolved_is_idempotent() -> None:
    client = _make_client()
    client.seed_thread("RT_1", 42, is_resolved=True)

    result = client.resolve_review_thread("RT_1")

    assert result is not None
    assert result["thread"]["isResolved"] is True
    assert client.get_review_threads(42)[0].is_resolved is True


# --- add_label / remove_label ---


def test_add_label_modifies_pr_labels() -> None:
    client = _make_client()
    client.seed_pr(_make_pr(42))

    result = client.add_label(42, "enhancement")

    assert {"name": "enhancement"} in result
    pr = client.get_pr(42)
    assert "enhancement" in pr.labels


def test_add_duplicate_label_is_idempotent() -> None:
    client = _make_client()
    client.seed_pr(_make_pr(42, labels=["bug"]))

    client.add_label(42, "bug")

    pr = client.get_pr(42)
    assert pr.labels.count("bug") == 1


def test_remove_label() -> None:
    client = _make_client()
    client.seed_pr(_make_pr(42, labels=["bug", "v1"]))

    client.remove_label(42, "bug")

    pr = client.get_pr(42)
    assert "bug" not in pr.labels
    assert "v1" in pr.labels


def test_remove_nonexistent_label_is_noop() -> None:
    client = _make_client()
    client.seed_pr(_make_pr(42, labels=["bug"]))

    client.remove_label(42, "nonexistent")

    pr = client.get_pr(42)
    assert pr.labels == ["bug"]


# --- get_check_runs returns check runs for a ref ---


def test_get_check_runs_returns_seeded_runs() -> None:
    client = _make_client()
    client.seed_check_run("abc123", "tests", "completed", "success")
    client.seed_check_run("abc123", "lint", "completed", "failure")

    runs = client.get_check_runs("abc123")

    assert len(runs) == 2
    assert all(isinstance(r, CheckRun) for r in runs)
    assert runs[0].name == "tests"
    assert runs[0].conclusion == "success"
    assert runs[1].name == "lint"
    assert runs[1].conclusion == "failure"


def test_get_check_runs_empty_ref() -> None:
    client = _make_client()
    assert client.get_check_runs("no-such-ref") == []


# --- Different check run statuses ---


def test_check_run_statuses() -> None:
    client = _make_client()
    client.seed_check_run("ref1", "ci", "completed", "success")
    client.seed_check_run("ref1", "deploy", "completed", "failure")
    client.seed_check_run("ref1", "build", "in_progress", None)

    runs = client.get_check_runs("ref1")

    by_name = {r.name: r for r in runs}
    assert by_name["ci"].status == "completed"
    assert by_name["ci"].conclusion == "success"
    assert by_name["deploy"].conclusion == "failure"
    assert by_name["build"].status == "in_progress"
    assert by_name["build"].conclusion is None


# --- Pagination: >30 comments ---


def test_pagination_many_comments() -> None:
    client = _make_client(page_size=30)

    for i in range(35):
        client.post_comment(42, f"comment {i}")

    comments = client.get_pr_comments(42)
    assert len(comments) == 35
    assert comments[0].body == "comment 0"
    assert comments[34].body == "comment 34"


# --- Optimistic concurrency ---


def test_optimistic_edit_succeeds_with_matching_timestamp() -> None:
    client = _make_client()
    original = client.post_comment(42, "original")

    edited = client.edit_comment_optimistic(
        original.id, "updated", expected_updated_at=original.updated_at
    )

    assert edited.body == "updated"


def test_optimistic_edit_fails_with_stale_timestamp() -> None:
    client = _make_client()
    original = client.post_comment(42, "original")

    client.edit_comment(original.id, "someone else edited")

    with pytest.raises(StaleEditError, match="updated_at mismatch"):
        client.edit_comment_optimistic(
            original.id, "my edit", expected_updated_at=original.updated_at
        )


# --- State comment round-trip ---


def test_state_comment_round_trip() -> None:
    client = _make_client()

    state = RuntimeState(
        version=1,
        pr_number=42,
        status="collecting",
        head_sha="abc123",
        round_index=1,
        updated_at=NOW,
    )
    body = serialize_state_comment(state)
    comment = client.post_comment(42, body)

    comments = client.get_pr_comments(42)
    comment_dicts = [{"id": c.id, "body": c.body} for c in comments]
    found = find_state_comment(comment_dicts)

    assert found is not None
    assert found.state.status == "collecting"
    assert found.state.head_sha == "abc123"
    assert found.state.round_index == 1

    new_state = RuntimeState(
        version=1,
        pr_number=42,
        status="handling",
        head_sha="abc123",
        round_index=2,
        updated_at=datetime.now(UTC),
    )
    new_body = serialize_state_comment(new_state)
    client.edit_comment(comment.id, new_body)

    comments = client.get_pr_comments(42)
    comment_dicts = [{"id": c.id, "body": c.body} for c in comments]
    found = find_state_comment(comment_dicts)

    assert found is not None
    assert found.state.status == "handling"
    assert found.state.round_index == 2


# --- Seeding: pre-load test scenarios ---


def test_seeding_full_scenario() -> None:
    client = _make_client()

    client.seed_pr(_make_pr(10, labels=["ai-review"]))
    client.seed_comment(10, "initial review comment", user="reviewer-bot")
    rc = ReviewComment(
        id="RC_seed",
        body="please fix",
        author="reviewer",
        path="app.py",
        created_at=NOW.isoformat(),
    )
    client.seed_thread("RT_seed", 10, path="app.py", comments=[rc])
    client.seed_check_run("abc123", "ci", "completed", "success")

    pr = client.get_pr(10)
    assert pr.labels == ["ai-review"]

    comments = client.get_pr_comments(10)
    assert len(comments) == 1
    assert comments[0].user == "reviewer-bot"

    threads = client.get_review_threads(10)
    assert len(threads) == 1
    assert threads[0].comments[0].body == "please fix"

    runs = client.get_check_runs("abc123")
    assert len(runs) == 1
    assert runs[0].conclusion == "success"


def test_seed_comment_with_custom_id() -> None:
    client = _make_client()
    comment = client.seed_comment(1, "seeded", comment_id=100)

    assert comment.id == 100

    new = client.post_comment(1, "new")
    assert new.id == 101


def test_multiple_prs_isolated() -> None:
    client = _make_client()
    client.seed_pr(_make_pr(1))
    client.seed_pr(_make_pr(2))
    client.post_comment(1, "on pr 1")
    client.post_comment(2, "on pr 2")

    assert len(client.get_pr_comments(1)) == 1
    assert len(client.get_pr_comments(2)) == 1
