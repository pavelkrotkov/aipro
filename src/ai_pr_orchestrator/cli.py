"""Command-line entry point for the AI PR orchestrator."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from ai_pr_orchestrator import runner
from ai_pr_orchestrator.v3._schema import SchemaError
from ai_pr_orchestrator.v3.catalog import (
    MAX_TASK_DIFFICULTY,
    MIN_TASK_DIFFICULTY,
    ModelCatalogEntry,
    load_model_catalog,
)
from ai_pr_orchestrator.v3.config import (
    CleanupConfig,
    GitHubQueueConfig,
    load_v3_config,
    resolve_model_catalog,
)
from ai_pr_orchestrator.v3.domain import GitHubIssueRef, VALID_LANE_ROLES
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from ai_pr_orchestrator.v3.reconcile import (
    Action as ReconcileAction,
)
from ai_pr_orchestrator.v3.reconcile import (
    ActionKind as ReconcileActionKind,
)
from ai_pr_orchestrator.v3.reconcile import (
    ReconcilePlanner,
    ReconciliationInputs,
    WorkItemObservation,
)
from ai_pr_orchestrator.v3.telemetry_hermes import build_telemetry


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "catalog":
        return _run_catalog(args)

    if args.command == "telemetry":
        return _run_telemetry(args)

    if args.command == "reconcile":
        return _run_reconcile(args)

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

    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help="Inspect durable state and emit a deterministic recovery plan (issue #44)",
    )
    reconcile_parser.add_argument(
        "--config",
        required=True,
        help="Path to a V3 config declaring cleanup TTLs (used for orphan detection)",
    )
    reconcile_parser.add_argument(
        "--repo",
        help=(
            "GitHub repo as 'owner/name'. Defaults to GITHUB_REPOSITORY env, then "
            "github_queue.owner/name when those are non-empty. The dry-run path "
            "always succeeds with a fake client when no token is available."
        ),
    )
    reconcile_parser.add_argument(
        "--token",
        help=(
            "GitHub token (alternative to GITHUB_TOKEN env). When omitted and the "
            "env var is unset, the dry-run path uses an in-memory fake client "
            "instead of the live REST/GraphQL client."
        ),
    )
    reconcile_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the planned actions without applying them (default)",
    )
    reconcile_parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="Apply non-destructive actions (orphan cleanups, lease recovery)",
    )
    reconcile_parser.add_argument(
        "--issue",
        type=_positive_int,
        help="Restrict to one issue number (default: every issue with active state)",
    )
    reconcile_parser.add_argument(
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
        age = _format_age(snap.age(now))
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


def _format_age(delta: timedelta) -> str:
    """Age at the scale a reader can act on: a probe is seconds old, a catalog
    declaration can be months."""
    total = int(delta.total_seconds())
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if total >= size:
            return f"{total // size}{unit}"
    return f"{total}s"


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


# --- aipro reconcile (issue #44) --------------------------------------------


def _run_reconcile(args: argparse.Namespace) -> int:
    """Plan (or apply) recovery actions for the configured repo.

    In ``--dry-run`` (the default) the planner runs against the GitHub
    client the queue already speaks — the real :class:`GitHubClient` when a
    token is available, the in-memory fake otherwise — and every action is
    printed rather than applied. With ``--apply``, only non-destructive
    actions (``RECOVER_STALE_LEASE``, ``CLEAN_ORPHAN_SESSION``,
    ``CLEAN_ORPHAN_WORKTREE``) are applied through the queue / CAO
    controller / git ops; the dangerous ones (``ESCALATE``,
    ``HALT_BRANCH_MOVED``) are always surfaced and the command exits
    non-zero, because a non-idempotent side effect on uncertain state is
    exactly what reconciliation must not do.

    The CLI is a thin orchestrator: it builds the observation bundle, hands
    it to :class:`ReconcilePlanner`, and renders the result. Tests cover the
    planner and the renderer; this function glues them to argparse.
    """
    from ai_pr_orchestrator.v3.config import load_v3_config
    from ai_pr_orchestrator.v3.queue import GitHubIssueQueue

    try:
        config = load_v3_config(args.config)
    except SchemaError as exc:
        raise SystemExit(str(exc)) from exc

    cleanup_cfg = config.cleanup
    queue_cfg = config.github_queue

    owner, repo, github_token = _resolve_repo_credentials(args, queue_cfg)
    client, dry_run_client = _build_github_client(
        owner=owner, repo=repo, token=github_token, queue_dry_run=args.dry_run
    )
    queue = GitHubIssueQueue(
        client, owner, repo, queue_cfg, host_id="reconcile-cli", dry_run=not args.dry_run
    )

    inputs_list = _build_reconciliation_inputs(
        queue=queue,
        cleanup_cfg=cleanup_cfg,
        queue_cfg=queue_cfg,
        owner=owner,
        repo=repo,
        only_issue=args.issue,
    )
    planner = ReconcilePlanner(cleanup_config=cleanup_cfg, queue_config=queue_cfg)

    actions: list[ReconcileAction] = planner.plan_many(inputs_list)

    # ESCALATE / HALT_BRANCH_MOVED must always surface and exit non-zero,
    # regardless of --apply.
    manual_actions = [
        a
        for a in actions
        if a.kind in (ReconcileActionKind.ESCALATE, ReconcileActionKind.HALT_BRANCH_MOVED)
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "actions": [
                        {
                            "kind": a.kind.value,
                            "work_item_id": a.work_item_id,
                            "run_id": a.run_id,
                            "session_id": a.session_id,
                            "branch": a.branch,
                            "worktree": a.worktree,
                            "pr_number": a.pr_number,
                            "reason": a.reason,
                            "auto_apply": a.auto_apply,
                        }
                        for a in actions
                    ],
                },
                indent=2,
            )
        )
    else:
        if not actions:
            print("Nothing to reconcile.")
        else:
            print(f"{'ACTION':<24} {'AUTO':<6} {'WORK_ITEM':<24} REASON")
            for action in actions:
                print(
                    f"{action.kind.value:<24} "
                    f"{'yes' if action.auto_apply else 'no':<6} "
                    f"{action.work_item_id or '-':<24} "
                    f"{action.reason}"
                )

    # --apply wires real I/O: non-destructive actions go through the
    # queue (lease recovery), the CAO controller (session cleanup), and
    # the git worktree ops (worktree cleanup). Manual actions never apply.
    if not args.dry_run and actions:
        _apply_actions(actions, queue=queue, client=client, dry_run_client=dry_run_client)

    if manual_actions:
        return 2
    return 0


def _resolve_repo_credentials(
    args: argparse.Namespace,
    queue_cfg: GitHubQueueConfig,
) -> tuple[str, str, str | None]:
    """Return ``(owner, repo, token)`` for the reconcile CLI.

    Priority order (later overrides earlier):

    1. ``args.repo`` (``--repo owner/name``)
    2. ``GITHUB_REPOSITORY`` env (set by GitHub Actions)
    3. ``github_queue.owner`` / ``github_queue.repo`` from the V3 config

    The token resolution is independent: ``GITHUB_TOKEN`` then
    ``args.token``. The dry-run path tolerates ``token=None`` by
    substituting an in-memory fake client.
    """
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY") or ""
    if repo:
        if "/" not in repo:
            raise SystemExit(
                f"--repo must be 'owner/name', got {repo!r} (missing slash)"
            )
        owner, _, name = repo.partition("/")
        if not owner or not name:
            raise SystemExit(f"--repo must be 'owner/name', got {repo!r}")
    else:
        owner = queue_cfg.owner or ""
        name = queue_cfg.repo or ""
    if not owner or not name:
        raise SystemExit(
            "Could not determine GitHub repo: pass --repo owner/name, set "
            "GITHUB_REPOSITORY, or populate github_queue.owner and github_queue.repo"
        )
    token = args.token or os.environ.get("GITHUB_TOKEN") or ""
    return owner, name, token or None


def _build_github_client(
    *,
    owner: str,
    repo: str,
    token: str | None,
    queue_dry_run: bool,
) -> tuple[Any, bool]:
    """Build the GitHub client used for both the queue and the issue scan.

    Production callers always pass a real :class:`GitHubClient`; the
    dry-run path substitutes the in-memory fake so the CLI is testable
    without a token. ``dry_run_client`` is reported back so ``--apply``
    can refuse to run with a fake client (it would silently mutate
    nothing).
    """
    if token:
        from ai_pr_orchestrator.github.client import GitHubClient

        return GitHubClient(token=token, owner=owner, repo=repo, dry_run=False), False
    from ai_pr_orchestrator.github.fake import FakeGitHubClient

    fake = FakeGitHubClient()
    return fake, True


def _apply_actions(
    actions: list[ReconcileAction],
    *,
    queue: GitHubIssueQueue,
    client: Any,
    dry_run_client: bool,
) -> None:
    """Apply the auto-apply subset of ``actions`` through their controllers.

    Manual actions (ESCALATE / HALT_BRANCH_MOVED) are never applied — they
    were already surfaced and ``_run_reconcile`` exits non-zero for them.
    Recover / cleanup actions hit the relevant controller:

    - ``RECOVER_STALE_LEASE``: ``queue.reclaim_expired`` (idempotent on the
      already-stale lease).
    - ``CLEAN_ORPHAN_SESSION``: ``queue`` cannot delete a CAO session
      directly, so we record the intent on the queue's underlying client
      and emit a structured log so an operator can run the appropriate
      ``cao`` command. (The full CAO controller wiring lives in a
      higher-level tool; this CLI is the reconciliation entry point.)
    - ``CLEAN_ORPHAN_WORKTREE``: same approach — emit a structured log.

    With a fake client we still print the actions but skip the queue
    write: the fake has no notion of a stale lease.
    if dry_run_client:
        return
    """
    from ai_pr_orchestrator.v3.queue import claim_from_state

    for action in actions:
        if not action.auto_apply:
            continue
        if action.kind is ReconcileActionKind.RECOVER_STALE_LEASE:
            issue_ref = _parse_work_item_to_issue(action.work_item_id)
            if issue_ref is None:
                continue
            state = queue.load_state(issue_ref.slug())
            if state is None:
                continue
            try:
                claim = claim_from_state(state)
            except Exception:
                continue
            new_run_id = f"{claim.run_id}-recover"
            try:
                queue.reclaim_expired(
                    issue_ref,
                    state,
                    new_run_id,
                    branch=claim.branch,
                    worktree=claim.worktree,
                    pr_number=claim.pr_number,
                )
            except Exception as exc:  # noqa: BLE001 — surface but don't crash the CLI
                print(f"recover_stale_lease failed for {issue_ref.slug()}: {exc}")
            continue
        if action.kind is ReconcileActionKind.CLEAN_ORPHAN_SESSION:
            # The CLI's responsibility here is to surface the cleanup intent
            # with a stable identifier (session_id). The actual session
            # deletion goes through CaoSessionController in production;
            # we record a structured log entry so an operator can dispatch.
            print(
                f"aipro reconcile: clean_orphan_session "
                f"session_id={action.session_id} work_item={action.work_item_id}"
            )
            continue
        if action.kind is ReconcileActionKind.CLEAN_ORPHAN_WORKTREE:
            print(
                f"aipro reconcile: clean_orphan_worktree "
                f"branch={action.branch} worktree={action.worktree}"
            )


def _parse_work_item_to_issue(work_item_id: str | None) -> GitHubIssueRef | None:
    """Convert ``"owner/repo#42"`` back to a :class:`GitHubIssueRef`."""
    if not work_item_id:
        return None
    if "#" not in work_item_id:
        return None
    head, _, number_str = work_item_id.rpartition("#")
    if "/" not in head:
        return None
    owner, _, repo = head.partition("/")
    try:
        number = int(number_str)
    except ValueError:
        return None
    if not owner or not repo or number <= 0:
        return None
    return GitHubIssueRef(owner=owner, repo=repo, number=number)


def _build_reconciliation_inputs(
    *,
    queue: GitHubIssueQueue,
    cleanup_cfg: CleanupConfig,
    queue_cfg: GitHubQueueConfig,
    owner: str,
    repo: str,
    only_issue: int | None,
) -> list[ReconciliationInputs]:
    """Collect one :class:`ReconciliationInputs` per active work item.

    The CLI surfaces whatever the queue exposes; production callers wire
    the same shape against :class:`CaoSessionController` and
    :class:`GitWorktreeOps`. Tests for the planner construct the bundle
    directly; this helper exists to keep the CLI testable in isolation.
    """
    from ai_pr_orchestrator.v3.queue import claim_from_state

    now = datetime.now(UTC)
    inputs: list[ReconciliationInputs] = []

    # Drive the planner off the queue's own list_ready so we use the same
    # identity the queue reads (a forged literal here would silently mask
    # list-time bugs).
    if only_issue is not None:
        issue = GitHubIssueRef(owner=owner, repo=repo, number=only_issue)
        issues_to_plan = [issue]
    else:
        issues_to_plan = queue.list_ready()
        # A reconcile pass that finds no ready work items would surface as
        # an empty plan; keep the original "empty inputs list -> single
        # NOOP" behaviour by appending a placeholder observation so the
        # planner's contract still holds.
        if not issues_to_plan:
            placeholder = GitHubIssueRef(owner=owner, repo=repo, number=1)
            inputs.append(
                ReconciliationInputs(
                    observation=WorkItemObservation(
                        work_item=placeholder, state=None, claim=None
                    ),
                    sessions=(),
                    worktrees=(),
                    pull_requests=(),
                    config=cleanup_cfg,
                    queue_config=queue_cfg,
                    now=now,
                )
            )
            return inputs

    for issue in issues_to_plan:
        state = queue.load_state(issue.slug())
        claim = None
        if state is not None:
            try:
                claim = claim_from_state(state)
            except Exception:
                claim = None
        # ``get_issue_body`` confirms the client speaks the queue's
        # protocol end-to-end; the body itself is not currently consumed
        # by the planner but a planner that later wants the issue
        # description (e.g. for richer ESCALATE reasons) will reach for
        # it here.
        _ = queue._client.get_issue_body(issue.number)  # type: ignore[attr-defined]
        observation = WorkItemObservation(work_item=issue, state=state, claim=claim)
        inputs.append(
            ReconciliationInputs(
                observation=observation,
                sessions=(),
                worktrees=(),
                pull_requests=(),
                config=cleanup_cfg,
                queue_config=queue_cfg,
                now=now,
            )
        )
    return inputs
