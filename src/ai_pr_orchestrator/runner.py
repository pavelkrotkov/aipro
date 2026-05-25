"""Runner facade used by the CLI.

The orchestration loop is implemented in later issues. These functions provide
stable CLI hand-off points that tests and future modules can target.
"""

from pathlib import Path


def run(*, pr_number: int, dry_run: bool, event_path: Path | None = None) -> int:
    """Run the orchestrator for a pull request."""
    _ = (pr_number, dry_run, event_path)
    return 0


def inspect(*, pr_number: int) -> int:
    """Inspect the orchestrator inputs for a pull request."""
    _ = pr_number
    return 0
