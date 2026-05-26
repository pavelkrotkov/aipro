"""Tests for the GitHub httpx client."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
import respx

from ai_pr_orchestrator.github.client import (
    GitHubClient,
    GitHubClientError,
    _parse_next_link,
    _redact_token,
)
from ai_pr_orchestrator.github.models import CheckRun, Comment, PullRequest, ReviewThread

BASE = "https://api.github.com"
GQL = "https://api.github.com/graphql"
OWNER = "test-owner"
REPO = "test-repo"
TOKEN = "ghp_testtoken1234567890"


def _make_client(*, dry_run: bool = False) -> GitHubClient:
    return GitHubClient(
        token=TOKEN,
        owner=OWNER,
        repo=REPO,
        dry_run=dry_run,
    )


def _pr_json(number: int = 42) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Test PR",
        "body": "PR body",
        "state": "open",
        "head": {"sha": "abc123", "ref": "feature"},
        "base": {"ref": "main"},
        "user": {"login": "author"},
        "draft": False,
        "mergeable": True,
        "labels": [{"name": "bug"}, {"name": "v1"}],
    }


def _comment_json(comment_id: int = 1) -> dict[str, Any]:
    return {
        "id": comment_id,
        "body": "A comment",
        "user": {"login": "commenter"},
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }


# --- REST: get_pr ---


@respx.mock
def test_get_pr() -> None:
    route = respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/42").mock(
        return_value=httpx.Response(200, json=_pr_json())
    )
    with _make_client() as client:
        pr = client.get_pr(42)

    assert route.called
    assert isinstance(pr, PullRequest)
    assert pr.number == 42
    assert pr.title == "Test PR"
    assert pr.head_sha == "abc123"
    assert pr.labels == ["bug", "v1"]
    assert pr.author == "author"


# --- REST: post_comment ---


@respx.mock
def test_post_comment() -> None:
    route = respx.post(f"{BASE}/repos/{OWNER}/{REPO}/issues/42/comments").mock(
        return_value=httpx.Response(201, json=_comment_json())
    )
    with _make_client() as client:
        comment = client.post_comment(42, "hello")

    assert route.called
    assert isinstance(comment, Comment)
    assert comment.id == 1
    assert comment.body == "A comment"


# --- REST: edit_comment ---


@respx.mock
def test_edit_comment() -> None:
    route = respx.patch(f"{BASE}/repos/{OWNER}/{REPO}/issues/comments/99").mock(
        return_value=httpx.Response(200, json=_comment_json(99))
    )
    with _make_client() as client:
        comment = client.edit_comment(99, "updated")

    assert route.called
    assert isinstance(comment, Comment)
    assert comment.id == 99


# --- REST: add_label ---


@respx.mock
def test_add_label() -> None:
    route = respx.post(f"{BASE}/repos/{OWNER}/{REPO}/issues/42/labels").mock(
        return_value=httpx.Response(200, json=[{"name": "new-label"}])
    )
    with _make_client() as client:
        result = client.add_label(42, "new-label")

    assert route.called
    assert result == [{"name": "new-label"}]


# --- REST: remove_label ---


@respx.mock
def test_remove_label() -> None:
    route = respx.delete(f"{BASE}/repos/{OWNER}/{REPO}/issues/42/labels/old-label").mock(
        return_value=httpx.Response(204)
    )
    with _make_client() as client:
        client.remove_label(42, "old-label")

    assert route.called


# --- REST: get_check_runs ---


@respx.mock
def test_get_check_runs() -> None:
    route = respx.get(f"{BASE}/repos/{OWNER}/{REPO}/commits/abc123/check-runs").mock(
        return_value=httpx.Response(
            200,
            json={
                "total_count": 1,
                "check_runs": [
                    {
                        "id": 1,
                        "name": "tests",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://example.com",
                    }
                ],
            },
        )
    )
    with _make_client() as client:
        runs = client.get_check_runs("abc123")

    assert route.called
    assert len(runs) == 1
    assert isinstance(runs[0], CheckRun)
    assert runs[0].name == "tests"
    assert runs[0].conclusion == "success"


# --- GraphQL: get_review_threads ---


@respx.mock
def test_get_review_threads() -> None:
    route = respx.post(GQL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": [
                                    {
                                        "id": "RT_1",
                                        "isResolved": False,
                                        "isOutdated": False,
                                        "path": "src/main.py",
                                        "comments": {
                                            "nodes": [
                                                {
                                                    "id": "RC_1",
                                                    "body": "fix this",
                                                    "author": {"login": "reviewer"},
                                                    "path": "src/main.py",
                                                    "createdAt": "2025-01-01T00:00:00Z",
                                                }
                                            ]
                                        },
                                    }
                                ],
                            }
                        }
                    }
                }
            },
        )
    )
    with _make_client() as client:
        threads = client.get_review_threads(42)

    assert route.called
    assert len(threads) == 1
    assert isinstance(threads[0], ReviewThread)
    assert threads[0].id == "RT_1"
    assert not threads[0].is_resolved
    assert threads[0].comments[0].body == "fix this"


# --- GraphQL: reply_to_review_thread ---


@respx.mock
def test_reply_to_review_thread() -> None:
    route = respx.post(GQL).mock(
        return_value=httpx.Response(
            200, json={"data": {"addPullRequestReviewComment": {"comment": {"id": "RC_new"}}}}
        )
    )
    with _make_client() as client:
        result = client.reply_to_review_thread("RT_1", "fixed")

    assert route.called
    assert result is not None


# --- GraphQL: resolve_review_thread ---


@respx.mock
def test_resolve_review_thread() -> None:
    route = respx.post(GQL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"resolveReviewThread": {"thread": {"id": "RT_1", "isResolved": True}}}},
        )
    )
    with _make_client() as client:
        result = client.resolve_review_thread("RT_1")

    assert route.called
    assert result is not None


# --- GraphQL error handling ---


@respx.mock
def test_graphql_error_raises() -> None:
    respx.post(GQL).mock(
        return_value=httpx.Response(200, json={"errors": [{"message": "something went wrong"}]})
    )
    with _make_client() as client, pytest.raises(GitHubClientError, match="GraphQL error"):
        client.get_review_threads(42)


# --- Rate limiting: 429 with Retry-After ---


@respx.mock
def test_429_retries_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    route = respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}),
            httpx.Response(200, json=_pr_json(1)),
        ]
    )
    with _make_client() as client:
        pr = client.get_pr(1)

    assert route.call_count == 2
    assert sleeps == [2.0]
    assert pr.number == 1


# --- Rate limiting: 403 with rate limit headers ---


@respx.mock
def test_403_rate_limit_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    route = respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        side_effect=[
            httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "0"},
            ),
            httpx.Response(200, json=_pr_json(1)),
        ]
    )
    with _make_client() as client:
        pr = client.get_pr(1)

    assert route.call_count == 2
    assert len(sleeps) == 1
    assert pr.number == 1


# --- Rate limiting: gives up after max retries ---


@respx.mock
def test_rate_limit_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        return_value=httpx.Response(429, headers={"retry-after": "1"}),
    )
    with _make_client() as client, pytest.raises(GitHubClientError, match="Request failed after"):
        client.get_pr(1)

    assert len(sleeps) == 3  # MAX_RETRIES - 1


# --- Rate limiting: 403 with retry-after (secondary/abuse limit) ---


@respx.mock
def test_403_with_retry_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    route = respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        side_effect=[
            httpx.Response(403, headers={"retry-after": "3"}),
            httpx.Response(200, json=_pr_json(1)),
        ]
    )
    with _make_client() as client:
        pr = client.get_pr(1)

    assert route.call_count == 2
    assert sleeps == [3.0]
    assert pr.number == 1


# --- 403 without rate limit is not retried ---


@respx.mock
def test_403_not_rate_limited_raises_immediately() -> None:
    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        return_value=httpx.Response(403),
    )
    with _make_client() as client, pytest.raises(httpx.HTTPStatusError):
        client.get_pr(1)


# --- Pagination: REST Link header ---


@respx.mock
def test_rest_pagination_follows_link_header() -> None:
    page1_url = f"{BASE}/repos/{OWNER}/{REPO}/commits/abc/check-runs"
    page2_url = f"{BASE}/repos/{OWNER}/{REPO}/commits/abc/check-runs?page=2"
    call_count = 0

    def pagination_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "page=2" not in str(request.url):
            return httpx.Response(
                200,
                json={
                    "total_count": 2,
                    "check_runs": [
                        {"id": 1, "name": "a", "status": "completed", "conclusion": "success"}
                    ],
                },
                headers={"link": f'<{page2_url}>; rel="next"'},
            )
        return httpx.Response(
            200,
            json={
                "total_count": 2,
                "check_runs": [
                    {"id": 2, "name": "b", "status": "completed", "conclusion": "failure"}
                ],
            },
        )

    respx.get(page1_url).mock(side_effect=pagination_handler)
    respx.get(page2_url).mock(side_effect=pagination_handler)
    with _make_client() as client:
        runs = client.get_check_runs("abc")

    assert call_count == 2
    assert len(runs) == 2
    assert runs[0].name == "a"
    assert runs[1].name == "b"


# --- Pagination: GraphQL pageInfo ---


@respx.mock
def test_graphql_pagination() -> None:
    call_count = 0

    def graphql_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "pageInfo": {
                                        "hasNextPage": True,
                                        "endCursor": "cursor1",
                                    },
                                    "nodes": [
                                        {
                                            "id": "RT_1",
                                            "isResolved": False,
                                            "isOutdated": False,
                                            "path": "a.py",
                                            "comments": {"nodes": []},
                                        }
                                    ],
                                }
                            }
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": [
                                    {
                                        "id": "RT_2",
                                        "isResolved": True,
                                        "isOutdated": False,
                                        "path": "b.py",
                                        "comments": {"nodes": []},
                                    }
                                ],
                            }
                        }
                    }
                }
            },
        )

    respx.post(GQL).mock(side_effect=graphql_handler)
    with _make_client() as client:
        threads = client.get_review_threads(1)

    assert call_count == 2
    assert len(threads) == 2
    assert threads[0].id == "RT_1"
    assert threads[1].id == "RT_2"


# --- Dry-run: read operations execute ---


@respx.mock
def test_dry_run_allows_reads() -> None:
    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/42").mock(
        return_value=httpx.Response(200, json=_pr_json())
    )
    with _make_client(dry_run=True) as client:
        pr = client.get_pr(42)

    assert pr.number == 42


# --- Dry-run: write operations are skipped ---


def test_dry_run_skips_post(caplog: pytest.LogCaptureFixture) -> None:
    with _make_client(dry_run=True) as client, caplog.at_level(logging.INFO):
        result = client.post_comment(42, "hello")

    assert result is None
    assert "DRY-RUN" in caplog.text
    assert "POST" in caplog.text


def test_dry_run_skips_patch(caplog: pytest.LogCaptureFixture) -> None:
    with _make_client(dry_run=True) as client, caplog.at_level(logging.INFO):
        result = client.edit_comment(99, "updated")

    assert result is None
    assert "DRY-RUN" in caplog.text


def test_dry_run_skips_delete(caplog: pytest.LogCaptureFixture) -> None:
    with _make_client(dry_run=True) as client, caplog.at_level(logging.INFO):
        client.remove_label(42, "old")

    assert "DRY-RUN" in caplog.text
    assert "DELETE" in caplog.text


def test_dry_run_skips_graphql_reply(caplog: pytest.LogCaptureFixture) -> None:
    with _make_client(dry_run=True) as client, caplog.at_level(logging.INFO):
        result = client.reply_to_review_thread("RT_1", "reply")

    assert result == {"comment": {"id": "dry-run"}}
    assert "DRY-RUN" in caplog.text


def test_dry_run_skips_graphql_resolve(caplog: pytest.LogCaptureFixture) -> None:
    with _make_client(dry_run=True) as client, caplog.at_level(logging.INFO):
        result = client.resolve_review_thread("RT_1")

    assert result is not None
    assert result["thread"]["isResolved"] is True
    assert "DRY-RUN" in caplog.text


# --- Token redaction ---


def test_token_redacted_in_error_messages() -> None:
    msg = "Auth failed with token ghp_abcd1234567890abcdef"
    assert "ghp_abcd****" in _redact_token(msg)
    assert "1234567890abcdef" not in _redact_token(msg)


def test_fine_grained_pat_redacted() -> None:
    msg = "token github_pat_abcd1234567890abcdef"
    assert "github_pat_abcd****" in _redact_token(msg)
    assert "1234567890abcdef" not in _redact_token(msg)


def test_all_token_prefixes_redacted() -> None:
    for prefix in ("gho_", "ghu_", "ghr_", "ghp_", "ghs_"):
        msg = f"token {prefix}abcd1234567890"
        redacted = _redact_token(msg)
        assert f"{prefix}abcd****" in redacted


def test_github_client_error_redacts_token() -> None:
    err = GitHubClientError("Failed with ghp_abcd1234567890abcdef in header")
    assert "ghp_abcd****" in str(err)
    assert "1234567890abcdef" not in str(err)


def test_token_not_in_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG), _make_client(dry_run=True) as client:
        client.post_comment(1, "test")

    for record in caplog.records:
        assert TOKEN not in record.getMessage()


# --- Link header parsing ---


def test_parse_next_link_with_next() -> None:
    header = '<https://api.github.com/repos/o/r/pulls?page=2>; rel="next", <https://api.github.com/repos/o/r/pulls?page=5>; rel="last"'
    assert _parse_next_link(header) == "https://api.github.com/repos/o/r/pulls?page=2"


def test_parse_next_link_no_next() -> None:
    header = '<https://api.github.com/repos/o/r/pulls?page=1>; rel="prev"'
    assert _parse_next_link(header) is None


def test_parse_next_link_empty() -> None:
    assert _parse_next_link("") is None


# --- Context manager ---


def test_context_manager() -> None:
    with _make_client(dry_run=True) as client:
        assert client is not None


# --- Respects x-ratelimit-reset ---


@respx.mock
def test_rate_limit_uses_reset_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.time", lambda: 1000.0)

    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1005"},
            ),
            httpx.Response(200, json=_pr_json(1)),
        ]
    )
    with _make_client() as client:
        client.get_pr(1)

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(5.0, abs=0.1)


# --- 5xx retries ---


@respx.mock
def test_502_retries_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    route = respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(200, json=_pr_json(1)),
        ]
    )
    with _make_client() as client:
        pr = client.get_pr(1)

    assert route.call_count == 2
    assert len(sleeps) == 1
    assert pr.number == 1


@respx.mock
def test_5xx_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        return_value=httpx.Response(503),
    )
    with _make_client() as client, pytest.raises(GitHubClientError, match="Request failed after"):
        client.get_pr(1)

    assert len(sleeps) == 3


# --- Network error retries ---


@respx.mock
def test_network_error_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    call_count = 0

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_pr_json(1))

    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(side_effect=flaky_handler)
    with _make_client() as client:
        pr = client.get_pr(1)

    assert call_count == 2
    assert len(sleeps) == 1
    assert pr.number == 1


@respx.mock
def test_network_error_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ai_pr_orchestrator.github.client.time.sleep", sleeps.append)

    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/1").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with _make_client() as client, pytest.raises(GitHubClientError, match="Network error after"):
        client.get_pr(1)

    assert len(sleeps) == 3


# --- GraphQL null repository/PR errors ---


@respx.mock
def test_graphql_null_repository_raises() -> None:
    respx.post(GQL).mock(return_value=httpx.Response(200, json={"data": {"repository": None}}))
    with (
        _make_client() as client,
        pytest.raises(GitHubClientError, match="not found or inaccessible"),
    ):
        client.get_review_threads(1)


@respx.mock
def test_graphql_null_pull_request_raises() -> None:
    respx.post(GQL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"repository": {"pullRequest": None}}},
        )
    )
    with (
        _make_client() as client,
        pytest.raises(GitHubClientError, match="Pull request #1 not found"),
    ):
        client.get_review_threads(1)


# --- Null user fields ---


@respx.mock
def test_get_pr_with_deleted_user() -> None:
    pr_data = _pr_json()
    pr_data["user"] = None
    respx.get(f"{BASE}/repos/{OWNER}/{REPO}/pulls/42").mock(
        return_value=httpx.Response(200, json=pr_data)
    )
    with _make_client() as client:
        pr = client.get_pr(42)

    assert pr.author == "ghost"


@respx.mock
def test_parse_comment_with_deleted_user() -> None:
    comment_data = _comment_json()
    comment_data["user"] = None
    respx.post(f"{BASE}/repos/{OWNER}/{REPO}/issues/1/comments").mock(
        return_value=httpx.Response(201, json=comment_data)
    )
    with _make_client() as client:
        comment = client.post_comment(1, "test")

    assert comment is not None
    assert comment.user == "ghost"
