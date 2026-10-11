# V1 migration and rollback

V3 is the deployment default. P5 ships its replacement configuration, trigger
and runbook before P6 removes V1. V1 `run`, `dry-run`, `inspect`, provider adapters,
`examples/sample-config.yml` and `examples/target-repo-workflow.yml` are deprecated
in P5. Do not install them for new deployments. P6 removes them without aliases
or a compatibility loader.

## Switch an existing repository

1. Disable the V1 review workflow and let its in-flight jobs finish. Inspect the
   actual PRs and native agents; cancellation of an Actions job alone does not
   prove its subprocesses stopped. Record existing branches and PR numbers.
2. Provision the contained CAO/Hermes lanes and persistent coordinator using
   [the README](../README.md) and [foreman runbook](V3_FOREMAN.md). Copy the new
   V3 policy/catalog to distinct paths; V1 YAML and review state cannot be loaded
   as V3 state. Select current accounts/models and exact target CI check names.
3. Replace the target's V1 workflow with the thin V3 queue template. Remove old
   `ai-loop` opt-ins; select explicit GitHub issues for V3 work. Existing V1 PRs
   need an operator decision rather than automatic conversion or duplicate PRs.
4. Inspect `aipro status` and `aipro route explain`, then opt in one authorized
   issue. Verify its durable state, review results, CI, and native teardown before
   adding further work. No host service is activated by installing this package.

## Drain V3 before rollback

Stop the foreman service and disable the queue trigger before changing code.
Stopping the coordinator preserves CAO workers for recovery; it does not stop them.
Inspect durable state with `aipro status` and native CAO metadata. For every
active run, terminate all attributed developer and historical reviewer sessions
through CAO, then confirm disappearance by name. An empty session inventory,
accepted DELETE, expired lease, or elapsed TTL is insufficient proof.

Use the native controller's `require_run_stopped(run_id, work_item_id,
developer_session=..., reviewer_lanes=...)` with the saved developer session
and all historical reviewer-wave lanes. After successful verification, persist
`lanes_stopped: true` in the current trusted state with its `updated_at`
precondition. The Python queue API `queue.abandon(issue, current_state)` then
requeues that owned claim and removes lease attribution while retaining its run,
branch, checkout, PR, budgets and review history. Reload state before retrying;
`abandon` rejects stale state, foreign ownership and unconfirmed stops. Keep the
service stopped while draining so queued records cannot immediately be claimed.
There is no CLI drain/apply command; `aipro reconcile` is a read-only plan.

Confirm zero active claims and no attributed live sessions before reverting.
Preserve GitHub state comments and linkage for later V3 recovery. V1 cannot read
these records: keep their issues out of the V1 loop, and reconcile their existing
PRs manually before any fallback dispatch.

## Ordered rollback

Revert P6 (V1 retirement), then P5 (V3 default/config/trigger/docs), using coherent
Git commits and normal CI review. Reverting only P6 restores files without
restoring the old default. Reinstall the target repository's V1 workflow/config
from the restored version only after drain and manual PR reconciliation.
Keep the V3 trigger and service disabled. Do not erase V3 comments or reset budgets.

After P7, reopen tracking epic #16 for resumed V1 work and #56 for unfinished V3
cutover work as appropriate. Issue state, target-repository workflow installation,
and host service configuration are external changes; a Git revert does not undo
those changes or reopen issues.
