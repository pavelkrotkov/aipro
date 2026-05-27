"""Protocol for primary coding-agent adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_pr_orchestrator.models import AgentRunResult, FixTask


@runtime_checkable
class CoderAdapter(Protocol):
    """Runs a coding agent against a structured fix task."""

    name: str

    def run_fix_task(self, task: FixTask) -> AgentRunResult:
        """Invoke the coder and return its structured result."""
        ...
