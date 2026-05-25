"""Command-line entry point for the AI PR orchestrator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path

from ai_pr_orchestrator import runner


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return runner.inspect(pr_number=args.pr)

    pr_number = args.pr
    event_path = Path(args.event_path) if args.event_path else None
    if event_path is not None:
        pr_number = _pr_number_from_event(event_path)
    elif pr_number is None:
        # argparse enforces this today; keep the guard for direct parser changes.
        raise SystemExit("Either --pr or --event-path must be provided")

    return runner.run(
        pr_number=pr_number,
        dry_run=args.command == "dry-run",
        event_path=event_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aipro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the PR review loop")
    _add_run_arguments(run_parser)

    dry_run_parser = subparsers.add_parser("dry-run", help="Inspect intended actions")
    _add_run_arguments(dry_run_parser)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a pull request")
    inspect_parser.add_argument(
        "--pr", type=_positive_int, required=True, help="Pull request number"
    )

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pr", type=_positive_int, help="Pull request number")
    group.add_argument("--event-path", help="Path to a GitHub event JSON file")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _pr_number_from_event(event_path: Path) -> int:
    try:
        with event_path.open(encoding="utf-8") as event_file:
            event = json.load(event_file)
    except OSError as exc:
        raise SystemExit(f"Failed to read event file {event_path}: {exc}") from exc
    except JSONDecodeError as exc:
        raise SystemExit(f"Event file {event_path} is not valid JSON: {exc}") from exc

    try:
        pr_number = event["pull_request"]["number"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"{event_path} does not contain pull_request.number") from exc
    if not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number <= 0:
        raise SystemExit(f"{event_path} pull_request.number must be a positive integer")
    return pr_number
