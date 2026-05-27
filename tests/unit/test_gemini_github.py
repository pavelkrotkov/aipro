"""Tests for the GeminiGitHubReviewerAdapter."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.github.models import ReviewComment
from ai_pr_orchestrator.models import Finding
from ai_pr_orchestrator.reviewers.gemini_github import GeminiGitHubReviewerAdapter

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
BOT_LOGIN = "gemini-code-assist[bot]"
HEAD_SHA = "abc123def456"
PR_NUMBER = 42


def _make_adapter(client: FakeGitHubClient | None = None) -> GeminiGitHubReviewerAdapter:
    if client is None:
        client = FakeGitHubClient(now=NOW)
    return GeminiGitHubReviewerAdapter(
        name="gemini_github",
        bot_logins=[BOT_LOGIN, "google-gemini-code-assist[bot]"],
        trigger_text="@gemini-code-assist review",
        github=client,
    )


# --- matches_author tests ---


def test_matches_author_primary_bot() -> None:
    adapter = _make_adapter()
    assert adapter.matches_author("gemini-code-assist[bot]") is True


def test_matches_author_secondary_bot() -> None:
    adapter = _make_adapter()
    assert adapter.matches_author("google-gemini-code-assist[bot]") is True


def test_matches_author_other_bot() -> None:
    adapter = _make_adapter()
    assert adapter.matches_author("some-other-bot[bot]") is False


def test_matches_author_human() -> None:
    adapter = _make_adapter()
    assert adapter.matches_author("human-user") is False


def test_matches_author_case_insensitive() -> None:
    adapter = _make_adapter()
    assert adapter.matches_author("Gemini-Code-Assist[bot]") is True


# --- build_trigger_comment tests ---


def test_trigger_comment_includes_machine_marker() -> None:
    adapter = _make_adapter()
    comment = adapter.build_trigger_comment(round_index=1, head_sha=HEAD_SHA)
    assert "<!-- ai-orchestrator:reviewer=gemini_github -->" in comment


def test_trigger_comment_includes_round_and_sha() -> None:
    adapter = _make_adapter()
    comment = adapter.build_trigger_comment(round_index=3, head_sha=HEAD_SHA)
    assert "Round 3" in comment
    assert HEAD_SHA[:8] in comment


def test_trigger_comment_includes_trigger_text() -> None:
    adapter = _make_adapter()
    comment = adapter.build_trigger_comment(round_index=1, head_sha=HEAD_SHA)
    assert "@gemini-code-assist review" in comment


# --- collect_findings tests ---


def test_collect_findings_returns_normalized_findings() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_1",
        pr_number=PR_NUMBER,
        path="src/main.py",
        comments=[
            ReviewComment(
                id="RC_1",
                body="This function has a bug",
                author=BOT_LOGIN,
                path="src/main.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, Finding)
    assert f.source == "gemini_github"
    assert f.body == "This function has a bug"
    assert f.head_sha == HEAD_SHA


def test_collect_findings_only_from_matched_bot() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_bot",
        pr_number=PR_NUMBER,
        path="src/a.py",
        comments=[
            ReviewComment(
                id="RC_bot",
                body="Bot finding",
                author=BOT_LOGIN,
                path="src/a.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    client.seed_thread(
        "T_human",
        pr_number=PR_NUMBER,
        path="src/b.py",
        comments=[
            ReviewComment(
                id="RC_human",
                body="Human comment",
                author="human-user",
                path="src/b.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert len(findings) == 1
    assert findings[0].thread_id == "T_bot"


def test_collect_findings_ignores_comments_before_trigger() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_old",
        pr_number=PR_NUMBER,
        path="src/old.py",
        comments=[
            ReviewComment(
                id="RC_old",
                body="Old finding",
                author=BOT_LOGIN,
                path="src/old.py",
                created_at="2025-06-01T11:00:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert len(findings) == 0


def test_collect_findings_outdated_thread() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_outdated",
        pr_number=PR_NUMBER,
        path="src/stale.py",
        is_outdated=True,
        comments=[
            ReviewComment(
                id="RC_outdated",
                body="Outdated finding",
                author=BOT_LOGIN,
                path="src/stale.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert len(findings) == 1
    assert findings[0].is_outdated is True


def test_collect_findings_ignores_own_trigger_comments() -> None:
    client = FakeGitHubClient(now=NOW)
    marker = "<!-- ai-orchestrator:reviewer=gemini_github -->"
    client.seed_thread(
        "T_trigger",
        pr_number=PR_NUMBER,
        path="src/trigger.py",
        comments=[
            ReviewComment(
                id="RC_trigger",
                body=f"{marker}\n@gemini-code-assist review\n\n_Round 1 · HEAD: `abc123de`_",
                author=BOT_LOGIN,
                path="src/trigger.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert len(findings) == 0


def test_collect_findings_deterministic_id() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_42",
        pr_number=PR_NUMBER,
        path="src/det.py",
        comments=[
            ReviewComment(
                id="RC_42",
                body="Some finding",
                author=BOT_LOGIN,
                path="src/det.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert findings[0].id == "gemini_github:T_42"


def test_collect_findings_includes_path_body_thread_comment() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_meta",
        pr_number=PR_NUMBER,
        path="src/meta.py",
        comments=[
            ReviewComment(
                id="RC_meta",
                body="Check this logic",
                author=BOT_LOGIN,
                path="src/meta.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    f = findings[0]
    assert f.path == "src/meta.py"
    assert f.body == "Check this logic"
    assert f.thread_id == "T_meta"
    assert f.comment_id == "RC_meta"
    assert f.raw == {"comment_id": "RC_meta", "thread_id": "T_meta"}


def test_collect_findings_empty_review() -> None:
    client = FakeGitHubClient(now=NOW)
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert findings == []


def test_collect_findings_only_first_comment_per_thread() -> None:
    client = FakeGitHubClient(now=NOW)
    client.seed_thread(
        "T_multi",
        pr_number=PR_NUMBER,
        path="src/multi.py",
        comments=[
            ReviewComment(
                id="RC_first",
                body="First bot comment",
                author=BOT_LOGIN,
                path="src/multi.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
            ReviewComment(
                id="RC_second",
                body="Second bot comment",
                author=BOT_LOGIN,
                path="src/multi.py",
                created_at="2025-06-01T12:10:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    findings = adapter.collect_findings(PR_NUMBER, HEAD_SHA, trigger_ts)

    assert len(findings) == 1
    assert findings[0].comment_id == "RC_first"
    assert findings[0].body == "First bot comment"
