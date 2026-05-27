"""Gemini GitHub reviewer adapter."""

from __future__ import annotations

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
                        id=f"{self.name}:{thread.id}",
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
