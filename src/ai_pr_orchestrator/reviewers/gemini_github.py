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
        """Return True if the bot has signalled completion after the trigger.

        The runner uses this probe to decide the zero-findings completion case
        (``snapshot.findings == []`` but the reviewer is done →
        ``done``/``no_findings``). Two completion signals are recognised, both
        chosen so they cannot mark the phase complete while dropping actionable
        feedback:

        1. An **APPROVED** pull-request review submitted after the trigger. An
           approval is an explicit no-issues verdict — there is no inline
           feedback to lose — so it is a safe, reachable completion signal. This
           is the normal clean-run path: Gemini approves with zero findings.
        2. A review-thread comment after the trigger. This is the same source
           ``collect_findings`` reads, so it is symmetric — any thread comment
           that could trip this would already have become a ``Finding`` (making
           this branch largely redundant), but it is kept so a thread comment
           that post-dates collection is still treated as a response.

        Deliberately NOT counted: top-level PR comments and non-APPROVED review
        bodies (``COMMENTED``/``CHANGES_REQUESTED``). ``collect_findings`` does
        not turn those into ``Finding``s, so counting them would let the runner
        reach ``no_findings`` while silently dropping feedback the bot posted
        there. A reviewer that responds only that way falls through to the poll
        timeout → ``needs_human`` so the feedback is seen.

        The orchestrator's own machine-marker comments are skipped so the
        trigger itself does not count as a response.
        """
        # Signal 1: an APPROVED review after the trigger.
        for review in self.github.get_pull_request_reviews(pr_number):
            if not self.matches_author(review.author):
                continue
            if review.state != "APPROVED":
                continue
            if self._is_after_trigger(review.submitted_at, trigger_timestamp):
                return True

        # Signal 2: a (non-marker) review-thread comment after the trigger.
        for thread in self.github.get_review_threads(pr_number):
            for comment in thread.comments:
                if not self.matches_author(comment.author):
                    continue
                if self._machine_marker in comment.body:
                    continue
                if self._is_after_trigger(comment.created_at, trigger_timestamp):
                    return True

        return False

    @staticmethod
    def _is_after_trigger(timestamp_str: str, trigger_timestamp: datetime) -> bool:
        try:
            return datetime.fromisoformat(timestamp_str) >= trigger_timestamp
        except (ValueError, TypeError):
            return False
