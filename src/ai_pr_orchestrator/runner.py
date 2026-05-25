"""Runner facade used by the CLI.

The orchestration loop is implemented in later issues. These functions provide
stable CLI hand-off points that tests and future modules can target.
"""

import sys
from pathlib import Path

NOT_IMPLEMENTED_MESSAGE = "AI PR Orchestrator runner is not implemented yet."


def run(*, pr_number: int, dry_run: bool, event_path: Path | None = None) -> int:
    """Run the orchestrator for a pull request."""
    _ = (pr_number, dry_run, event_path)
    print(NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return 2


def inspect(*, pr_number: int) -> int:
    """Inspect the orchestrator inputs for a pull request."""
    _ = pr_number
    print(NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return 2
