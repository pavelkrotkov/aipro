"""Gemini GitHub reviewer adapter."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from ai_pr_orchestrator.github.protocol import GitHubClient
from ai_pr_orchestrator.models import Finding


class GeminiGitHubReviewerAdapter:
    """Triggers Gemini via PR comment and normalizes its review findings."""

    def __init__(
        self,
        name: str,
        bot_logins: list[str],
        trigger_text: str,
        github: GitHubClient,
    ) -> None:
        self.name = name
        self._bot_logins = [login.lower() for login in bot_logins]
        self._trigger_text = trigger_text
        self.github = github

    @property
    def _machine_marker(self) -> str:
        return f"<!-- ai-orchestrator:reviewer={self.name} -->"

    def matches_author(self, login: str) -> bool:
        return login.lower() in self._bot_logins

    def build_trigger_comment(self, round_index: int, head_sha: str) -> str:
        return (
            f"{self._machine_marker}\n"
            f"{self._trigger_text}\n"
            f"\n"
            f"_Round {round_index} · HEAD: `{head_sha[:8]}`_"
        )

    def collect_findings(
        self,
        pr_number: int,
        head_sha: str,
        trigger_timestamp: datetime,
    ) -> list[Finding]:
        threads = self.github.get_review_threads(pr_number)
        findings: list[Finding] = []

        for thread in threads:
            for comment in thread.comments:
                if not self.matches_author(comment.author):
                    continue

                created_at = datetime.fromisoformat(comment.created_at)
                if created_at < trigger_timestamp:
                    continue

                if self._machine_marker in comment.body:
                    continue

                findings.append(
                    Finding(
                        id=f"{self.name}:{thread.id}:{comment.id}",
                        source=self.name,
                        body=comment.body,
                        created_at=created_at,
                        head_sha=head_sha,
                        thread_id=thread.id,
                        comment_id=comment.id,
                        path=thread.path,
                        line=None,
                        severity=None,
                        is_resolved=thread.is_resolved,
                        is_outdated=thread.is_outdated,
                        raw={"comment_id": comment.id, "thread_id": thread.id},
                    )
                )
                break  # Only take the first matching comment per thread

        return findings

    def has_responded(self, pr_number: int, trigger_timestamp: datetime) -> bool:
        """Return True if the bot has posted any comment after the trigger.

        Looks at both review-thread comments and top-level PR (issue)
        comments. The orchestrator's own machine-marker comments are skipped
        so that the trigger itself does not count as a response.
        """
        for author, body, created_at_str in self._iter_bot_candidates(pr_number):
            if not self.matches_author(author):
                continue
            if self._machine_marker in body:
                continue
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                continue
            if created_at >= trigger_timestamp:
                return True

        return False

    def _iter_bot_candidates(self, pr_number: int) -> Iterable[tuple[str, str, str]]:
        """Yield (author, body, created_at) tuples from all comment sources."""
        for thread in self.github.get_review_threads(pr_number):
            for comment in thread.comments:
                yield comment.author, comment.body, comment.created_at
        for pr_comment in self.github.get_pr_comments(pr_number):
            yield pr_comment.user, pr_comment.body, pr_comment.created_at
