"""Tests for the ReviewerAdapter protocol."""

from datetime import UTC, datetime

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.github.models import ReviewComment
from ai_pr_orchestrator.reviewers.base import ReviewerAdapter
from ai_pr_orchestrator.reviewers.gemini_github import GeminiGitHubReviewerAdapter

BOT_LOGIN = "gemini-code-assist[bot]"
PR_NUMBER = 42


def _make_adapter(client: FakeGitHubClient) -> GeminiGitHubReviewerAdapter:
    return GeminiGitHubReviewerAdapter(
        name="gemini_github",
        bot_logins=[BOT_LOGIN],
        trigger_text="@gemini-code-assist review",
        github=client,
    )


def test_gemini_github_satisfies_reviewer_adapter_protocol() -> None:
    client = FakeGitHubClient()
    adapter = GeminiGitHubReviewerAdapter(
        name="gemini_github",
        bot_logins=["gemini-code-assist[bot]"],
        trigger_text="@gemini-code-assist review",
        github=client,
    )

    assert isinstance(adapter, ReviewerAdapter)


def test_has_responded_true_when_bot_comment_after_trigger() -> None:
    client = FakeGitHubClient(now=datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC))
    client.seed_thread(
        "T_1",
        pr_number=PR_NUMBER,
        path="src/main.py",
        comments=[
            ReviewComment(
                id="RC_1",
                body="A finding",
                author=BOT_LOGIN,
                path="src/main.py",
                created_at="2025-06-01T12:05:00+00:00",
            ),
        ],
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    assert adapter.has_responded(PR_NUMBER, trigger_ts) is True


def test_has_responded_false_when_no_bot_comments() -> None:
    client = FakeGitHubClient(now=datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC))
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    assert adapter.has_responded(PR_NUMBER, trigger_ts) is False


def test_has_responded_skips_own_machine_marker_comments() -> None:
    """The orchestrator's own trigger comments must not count as a response,
    otherwise every fresh round would be considered already-responded."""
    client = FakeGitHubClient(now=datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC))
    marker = "<!-- ai-orchestrator:reviewer=gemini_github -->"
    # Marker found in a top-level PR (issue) comment that the orchestrator
    # itself posted, authored by the bot account.
    client.seed_comment(
        PR_NUMBER,
        body=f"{marker}\n@gemini-code-assist review",
        user=BOT_LOGIN,
        created_at=datetime(2025, 6, 1, 12, 5, 0, tzinfo=UTC),
    )
    adapter = _make_adapter(client)
    trigger_ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    assert adapter.has_responded(PR_NUMBER, trigger_ts) is False
