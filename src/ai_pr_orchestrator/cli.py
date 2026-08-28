"""Command-line entry point for the AI PR orchestrator."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from pathlib import Path

from ai_pr_orchestrator import runner
from ai_pr_orchestrator.v3._schema import SchemaError
from ai_pr_orchestrator.v3.catalog import (
    MAX_TASK_DIFFICULTY,
    MIN_TASK_DIFFICULTY,
    ModelCatalogEntry,
    load_model_catalog,
)
from ai_pr_orchestrator.v3.config import load_v3_config, resolve_model_catalog
from ai_pr_orchestrator.v3.domain import VALID_LANE_ROLES
from ai_pr_orchestrator.v3.telemetry_hermes import build_telemetry


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "catalog":
        return _run_catalog(args)

    if args.command == "telemetry":
        return _run_telemetry(args)

    if args.command == "inspect":
        return runner.inspect(pr_number=args.pr)

    pr_number = args.pr
    event_path = Path(args.event_path) if args.event_path else None
    if event_path is not None:
        # When both are supplied, the event's PR number wins if present;
        # otherwise (e.g. a ``status`` event) fall back to the explicit --pr.
        pr_number = _pr_number_from_event(event_path, fallback_pr=args.pr)
    elif pr_number is None:
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

    catalog_parser = subparsers.add_parser(
        "catalog", help="List currently eligible model catalog candidates"
    )
    source = catalog_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--catalog", help="Path to a shared model catalog file")
    source.add_argument(
        "--config", help="Path to a V3 config whose model_router catalog should be resolved"
    )
    catalog_parser.add_argument(
        "--role", choices=sorted(VALID_LANE_ROLES), help="Only candidates suitable for this role"
    )
    catalog_parser.add_argument(
        "--difficulty",
        type=_difficulty,
        default=MIN_TASK_DIFFICULTY,
        help=f"Task difficulty ({MIN_TASK_DIFFICULTY}-{MAX_TASK_DIFFICULTY})",
    )
    catalog_parser.add_argument(
        "--all", action="store_true", help="List every entry, not just eligible ones"
    )
    catalog_parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Report live quota, health, and freshness for every configured resource",
    )
    telemetry_parser.add_argument(
        "--config", required=True, help="Path to a V3 config declaring a telemetry section"
    )
    telemetry_parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of a table"
    )

    return parser


def _run_telemetry(args: argparse.Namespace) -> int:
    """Print normalized telemetry for every configured resource in one pass.

    Diagnostic only: it always exits 0 once the config loads, because an
    unavailable or exhausted resource is a finding to report, not a failure of
    the command. Credentials cannot appear in the output — snapshots have no
    field that holds one, and provider error text is redacted on the way in.
    """
    # One timestamp for the whole listing, so two rows cannot disagree about
    # what time it is when a window resets mid-scan.
    now = datetime.now(UTC)
    try:
        config_path = Path(args.config)
        config = load_v3_config(config_path)
        catalog = resolve_model_catalog(config, base_dir=config_path.parent)
        registry, _ledger = build_telemetry(config.telemetry, catalog=catalog)
    except SchemaError as exc:
        raise SystemExit(str(exc)) from exc

    snapshots = registry.snapshot_all(at=now)

    if args.json:
        print(
            json.dumps(
                {
                    "evaluated_at": now.isoformat(),
                    "resources": [snap.to_dict() for snap in snapshots],
                },
                indent=2,
            )
        )
        return 0

    if not snapshots:
        print("No telemetry resources are configured.")
        return 0

    print(f"{'RESOURCE':<20} {'AVAILABILITY':<13} {'CLASS':<13} {'AGE':>8}  SOURCE")
    for snap in snapshots:
        stale = "" if snap.is_stale(now) is not True else "  [STALE]"
        age = f"{int(snap.age(now).total_seconds())}s"
        print(
            f"{snap.resource:<20} {snap.availability:<13} {snap.resource_class:<13} "
            f"{age:>8}  {snap.source or '-'}{stale}"
        )
        for window in snap.windows:
            used = "unknown" if window.used_fraction is None else f"{window.used_fraction:.0%}"
            reset = window.reset_at.isoformat() if window.reset_at else "no reset reported"
            ttr = window.time_to_reset(now)
            resets_in = "" if ttr is None else f" (in {_format_duration(ttr)})"
            print(f"    window {window.label:<18} used {used:>7}  resets {reset}{resets_in}")
        if snap.cash_balance is not None:
            print(f"    balance {snap.cash_balance:.2f} {snap.currency}".rstrip())
        if snap.expires_at is not None:
            print(f"    expires {snap.expires_at.isoformat()}")
        for detail in snap.details:
            print(f"    detail {detail}")
        if snap.health.total:
            failure_rate = snap.health.failure_rate
            print(
                f"    health {snap.health.total} recent request(s), "
                f"{failure_rate:.0%} failed, "
                f"{snap.health.consecutive_failures} consecutive"
                + (", throttled" if snap.health.is_throttled(now) else "")
            )
        if snap.reason:
            print(f"    reason {snap.reason}")
    return 0


def _format_duration(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    hours, remainder = divmod(total, 3600)
    return f"{hours}h{remainder // 60:02d}m" if hours else f"{remainder // 60}m"


def _run_catalog(args: argparse.Namespace) -> int:
    """Print catalog candidates with normalized effective price/resource class."""
    # One timestamp for the whole listing. Reading the clock per entry would
    # let a promotion expire mid-scan, so the filter and the row it produces
    # could disagree about whether the same entry is eligible.
    now = datetime.now(UTC)
    try:
        if args.catalog:
            catalog = load_model_catalog(args.catalog)
        else:
            config_path = Path(args.config)
            catalog = resolve_model_catalog(
                load_v3_config(config_path), base_dir=config_path.parent
            )
        entries = (
            list(catalog.entries)
            if args.all
            else catalog.eligible(role=args.role, difficulty=args.difficulty, at=now)
        )
    except SchemaError as exc:
        raise SystemExit(str(exc)) from exc

    rows = [
        {
            "ref": entry.ref,
            "resource_class": entry.resource_class,
            "cost_class": entry.cost_class,
            # Distinguish the three cases the broker must not conflate:
            # priced, free, and simply unknown.
            "effective_input_price_per_mtok": _price(entry, now, index=0),
            "effective_output_price_per_mtok": _price(entry, now, index=1),
            "promotion_active": entry.promotion_active(now),
            "eligible": entry.is_eligible(role=args.role, difficulty=args.difficulty, at=now),
            "provider": entry.provider,
            "family": entry.family,
            "vendor": entry.vendor,
            "enabled": entry.enabled,
        }
        for entry in entries
    ]

    if args.json:
        print(json.dumps({"evaluated_at": now.isoformat(), "candidates": rows}, indent=2))
        return 0

    if not rows:
        print("Catalog is empty." if args.all else "No eligible catalog candidates.")
        return 0

    # Under --all the table mixes dispatchable and undispatchable entries, so
    # it has to say which is which or an operator reads an unusable resource
    # as available.
    header = f"{'REF':<24} {'RESOURCE':<13} {'COST':<7} {'IN/MTOK':>9} {'OUT/MTOK':>9}  "
    header += f"{'PROMO':<5}  ELIGIBLE" if args.all else "PROMO"
    print(header)
    for row in rows:
        in_price = (
            "unknown"
            if row["effective_input_price_per_mtok"] is None
            else f"{row['effective_input_price_per_mtok']:.4f}"
        )
        out_price = (
            "unknown"
            if row["effective_output_price_per_mtok"] is None
            else f"{row['effective_output_price_per_mtok']:.4f}"
        )
        promo = "yes" if row["promotion_active"] else "-"
        line = (
            f"{row['ref']:<24} {row['resource_class']:<13} {row['cost_class']:<7} "
            f"{in_price:>9} {out_price:>9}  "
        )
        line += f"{promo:<5}  {'yes' if row['eligible'] else 'no'}" if args.all else promo
        print(line)
    return 0


def _price(entry: ModelCatalogEntry, now: datetime, *, index: int) -> float | None:
    prices = entry.effective_prices(now)
    return None if prices is None else prices[index]


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    # ``--pr`` and ``--event-path`` are NOT mutually exclusive: a ``status``
    # webhook carries a commit SHA but no PR number, so operators must pass
    # ``--event-path`` (to forward the SHA into the stale-CI-event guard) *and*
    # ``--pr`` (to identify the PR). main() requires at least one of the two.
    parser.add_argument("--pr", type=_positive_int, help="Pull request number")
    parser.add_argument("--event-path", help="Path to a GitHub event JSON file")


def _difficulty(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be an integer within {MIN_TASK_DIFFICULTY}-{MAX_TASK_DIFFICULTY}"
        ) from exc
    if not MIN_TASK_DIFFICULTY <= parsed <= MAX_TASK_DIFFICULTY:
        raise argparse.ArgumentTypeError(
            f"must be an integer within {MIN_TASK_DIFFICULTY}-{MAX_TASK_DIFFICULTY}"
        )
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _pr_number_from_event(event_path: Path, *, fallback_pr: int | None = None) -> int:
    try:
        with event_path.open(encoding="utf-8") as event_file:
            event = json.load(event_file)
    except OSError as exc:
        raise SystemExit(f"Failed to read event file {event_path}: {exc}") from exc
    except JSONDecodeError as exc:
        raise SystemExit(f"Event file {event_path} is not valid JSON: {exc}") from exc

    # The caller's payload is explicit; infer the event type from its keys
    # rather than an ambient GITHUB_EVENT_NAME, which may describe a different
    # trigger (e.g. "push" on push-to-main CI runs) and would mask a payload
    # that clearly carries ``pull_request``. The hint is honored only when it
    # agrees with the inferred type or nothing could be inferred.
    parsed = runner.parse_event_payload_first(event, event_name=os.environ.get("GITHUB_EVENT_NAME"))
    pr_number = parsed.pr_number
    if pr_number is None:
        # ``status`` webhooks carry a commit SHA but no PR number. Mapping SHA →
        # PR requires a live GitHub client the CLI doesn't construct yet, so the
        # operator supplies the PR via ``--pr``; we honor it here as a fallback
        # while still forwarding the event (and its head_sha) into Runner.run,
        # keeping the stale-CI-event guard intact.
        if fallback_pr is not None:
            return fallback_pr
        if parsed.event_type == "status" and parsed.head_sha is not None:
            raise SystemExit(
                f"status events carry a commit SHA ({parsed.head_sha}) but no PR "
                "number; resolving SHA -> PR is not wired yet. Pass --pr "
                "explicitly (alongside --event-path) to run on this PR."
            )
        raise SystemExit(f"Could not determine pull request number from event file {event_path}")
    if not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number <= 0:
        raise SystemExit(f"{event_path} pull_request.number must be a positive integer")
    return pr_number
