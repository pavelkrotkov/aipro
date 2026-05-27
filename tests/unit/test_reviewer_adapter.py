"""Tests for the ReviewerAdapter protocol."""

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.reviewers.base import ReviewerAdapter
from ai_pr_orchestrator.reviewers.gemini_github import GeminiGitHubReviewerAdapter


def test_gemini_github_satisfies_reviewer_adapter_protocol() -> None:
    client = FakeGitHubClient()
    adapter = GeminiGitHubReviewerAdapter(
        name="gemini_github",
        bot_logins=["gemini-code-assist[bot]"],
        trigger_text="@gemini-code-assist review",
        github=client,
    )

    assert isinstance(adapter, ReviewerAdapter)
