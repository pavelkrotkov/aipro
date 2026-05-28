"""Protocol for AI reviewer adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ai_pr_orchestrator.models import Finding


@runtime_checkable
class ReviewerAdapter(Protocol):
    """Triggers an AI reviewer and collects its normalized findings."""

    name: str

    def matches_author(self, login: str) -> bool:
        """Return True if the given GitHub login belongs to this reviewer's bot."""
        ...

    def build_trigger_comment(self, round_index: int, head_sha: str) -> str:
        """Build the PR comment body that triggers this reviewer."""
        ...

    def collect_findings(
        self,
        pr_number: int,
        head_sha: str,
        trigger_timestamp: datetime,
    ) -> list[Finding]:
        """Collect and normalize findings posted by this reviewer after the trigger."""
        ...

    def has_responded(self, pr_number: int, trigger_timestamp: datetime) -> bool:
        """Return True once the reviewer bot has posted *any* comment after
        ``trigger_timestamp`` — even one that produced zero findings.

        This lets the runner distinguish "reviewer finished and found nothing"
        (a happy-path completion) from "reviewer never responded" (a timeout
        that needs human attention). Implementations should ignore the
        orchestrator's own trigger comments by checking for the adapter's
        machine marker.
        """
        ...
