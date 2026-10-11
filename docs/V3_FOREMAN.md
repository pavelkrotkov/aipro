# Continuous V3 foreman

`aipro foreman` runs the existing policy loop continuously. The Hermes
`aipro-foreman` profile supervises this service through its skill, including
from the existing Hermes gateway. Hermes does not duplicate scheduling in a
conversation or cron job. GitHub state and live CAO metadata reconstruct the
run; local process memory is disposable.

One developer issue runs at a time. The policy loop launches its independent
reviewer seats concurrently, within broker capacity. Completed work moves to
the next issue immediately. Pending CI is polled without monopolizing the
developer slot. Needs-human state is never automatically retried, even if a
stale ready label remains.

Queue an issue with both `v3-work` and the stable `v3-work-enabled` label
(configurable under `github_queue`). Remove the stable label to cancel active
work; the coordinator retains capacity until every attributed session stops.
The originating issue author must satisfy `safety.allowed_pr_author_associations`.
Pin `github_queue.state_author_login` to the bot writer for GitHub App tokens;
otherwise it resolves through the authenticated user. Drain and migrate machine
comments before changing writers. See [acceptance boundaries](V3_SOAK.md).

## Prerequisites

- Python 3.12+, a primary Git checkout with a `.git` directory, and the
  installed `aipro` entry point. The coordinator owns commit/push credentials
  and a GitHub token with the configured queue/PR permissions. Verify its SSH
  origin and noninteractive push authentication; workers receive neither.
- CAO 2.5.0 with durable terminal metadata, separately managed from this
  service. Follow [CAO provisioning](V3_CAO.md) for the pinned Hermes image,
  whole-process container wrapper, separate lane homes, consent bootstrap,
  and profiles. Keep CAO's unauthenticated control endpoint loopback-only.
- A shared V3 YAML file with `github_queue`, `cao`, `hermes_lanes`, safety/CI
  policy, and `model_router.catalog_path` or inline catalog. Relative catalog
  paths resolve beside that YAML file. The configured catalog must be nonempty.
  Account usage and provider health use the existing `telemetry` section and
  Hermes bridge; model selection refreshes telemetry when constructing each
  issue/recovery loop. Configure reserves, prices, promotion dates, quality,
  and provider concurrency in the shared router, not in CAO profiles.
- One coordinator deployment per repository. A native lock inside coordinator
  Git metadata excludes a second local process. GitHub leases block an active
  foreign owner; multiple active developer records halt new claims.

## Install and run

From the checkout, install the package with the repository's usual `uv sync`
workflow. Install the reusable supervisor profile into Hermes:

```sh
hermes profile install "$AIPRO_CHECKOUT/profiles/aipro-foreman"
```

The distribution contains no account credentials or model pin. Select the
profile explicitly in your Hermes CLI/gateway configuration; do not change
unrelated profiles. Set its `AIPRO_CHECKOUT` and `AIPRO_V3_CONFIG` variables.

The supplied user unit defaults to `$HOME/dev/aipro`. For another checkout,
edit its `WorkingDirectory` and executable path before installing it.
Create `$HOME/.config/aipro/foreman.env`, mode 600, with your own values:

```ini
AIPRO_CHECKOUT=/home/you/dev/aipro
AIPRO_V3_CONFIG=/home/you/.config/aipro/v3.yml
AIPRO_REPOSITORY=owner/repo
AIPRO_COMMITTER_NAME="AIPRO Foreman"
AIPRO_COMMITTER_EMAIL=foreman@example.invalid
GITHUB_TOKEN=your-coordinator-token
```

The `${NAME}` arguments in the unit preserve values containing spaces as
single arguments, per [systemd's command-line rules](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml).

```sh
install -d "$HOME/.config/systemd/user"
install -m 644 deploy/aipro-foreman.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now aipro-foreman
systemctl --user status aipro-foreman
journalctl --user -u aipro-foreman -f
```

For operation after logout, enable user lingering through your host's normal
administration procedure. Start, stop, and restart use `systemctl --user
start|stop|restart aipro-foreman`. CAO must run in a separate service/process:
stopping the coordinator preserves CAO workers for adoption. `--once` performs
one scheduling/recovery pass for diagnostics; it can drive one full issue and
is not a dry run. Do not run it alongside the service.

## Recovery and operator decisions

Startup and every pass scan active/review state before ready work. A same-host
run is adopted under the native singleton lock. An unexpired foreign lease
blocks claims; only an expired lease is reclaimed, preserving the run id,
phase, budget usage, branch, model assignments, and reviewer receipts.
Late writes from the previous owner fail ownership checks.

Coder dispatch intent and invocation reservation are persisted together.
Restart collects its matching CAO turn or stored result without another
charge. Reviewer waves persist their immutable head, issue digest, prompts,
assignments, checkout paths, and individual results before cleanup. Recovery
reuses these seats and results; it does not create another wave. All attributed developer and reviewer sessions are confirmed stopped before
a terminal transition or CI requeue releases capacity. Repeated termination
is idempotent, including a crash after CAO accepted termination. An accepted
delete without native disappearance keeps capacity occupied; timeout and model
replacement use the same verified stop. Cancellation derives historical lane
identities from the durable assignment and reviewer waves after a registry rename.

Unknown launches/submissions/PR effects stay active and emit
`reconciliation_required`. A missing checkpoint, missing expected session,
changed review input/head, conflicting owner, or unconfirmed cleanup requires
inspection. The service does not fabricate absent inventories or blindly
relaunch. Older active runs without dispatch intent require manual migration.
A cancelled run with unconfirmed worker termination retains its active label
and blocks capacity until an operator confirms termination; cancellation alone
does not authorize reuse of its lane home.
Use [operator diagnostics](V3_OBSERVABILITY.md) to inspect durable state,
live attribution, routing, findings, and CI. Inspect actual CAO/GitHub/Git
state before changing it;
`aipro reconcile --apply` remains unsupported.

For needs-human work, fix the stated cause and explicitly transition its
GitHub state to queued before restoring the ready label. Merely adding a
label does not erase the terminal decision. Preserve the original run id
when resuming the same issue's history.

The fake-HTTP integration harness covers two sequential issues and cold
developer/reviewer recovery. It does not establish live model reasoning or
the separate #55 soak/cutover criteria.

Prompt budget estimates (`ceil(prompt characters / 4)`) are persisted before
worker or reviewer dispatch and restored on restart. Collecting an existing
turn does not charge its prompt again. Tightening the cap below already
reserved prompt usage escalates the run. Older active dispatch checkpoints
without a counter, or malformed counters, require manual accounting before
recovery; restarting cannot reset the budget. Status reports these estimates
separately from actual provider token usage.


## Retained resource cleanup

Each service pass runs the production TTL sweeper before scheduling. It uses
the service's UTC clock, the configured session TTL (default two hours) and
checkout inactivity TTL (default 24 hours). Only terminal runs owned by this
coordinator with confirmed lane stops authorize cleanup; expired active leases
do not. GitHub authority is rechecked before each removal.

The coordinator records checkout path, role, branch/head, run and issue in its
protected Git administration directory. Allocation intent alone cannot authorize
deletion: a record becomes materialized only after successful native creation.
Existing unmarked checkouts and unknown or mismatched ownership require manual
inspection. Unmaterialized allocation records for terminal runs are retained
for manual review without blocking unrelated work; their paths are never deleted. Human checkouts are retained. Reviewer-local metadata cannot grant
cleanup authority.

The sweeper discovers attributed native CAO sessions, developer worktrees and
reviewer clones. It confirms session disappearance after termination and probes
every current and saved historical run lane before deleting a checkout, even when CAO lists
no sessions. Unknown inventories or unconfirmed stops emit
`reconciliation_required` and hold new work. Recent file activity postpones
checkout removal. An unreadable activity inventory retains all checkouts;
review deletion requires the native descriptor-based symlink protection. Each confirmed removal emits a correlated, redacted
`owned_resource_cleanup` event and appears in the service's `cleanup_outcomes`.
