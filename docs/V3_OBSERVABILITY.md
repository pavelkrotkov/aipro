# V3 operator diagnostics

Use the same shared V3 config as the foreman. Status and step require the
coordinator's GitHub token (`GITHUB_TOKEN` or `--token`); an absent token is an
error, not an empty simulated queue.

```sh
aipro status --config "$AIPRO_V3_CONFIG" --repo owner/repo
aipro status --config "$AIPRO_V3_CONFIG" --repo owner/repo --json
aipro step --config "$AIPRO_V3_CONFIG" --repo owner/repo --dry-run --json
aipro route explain --config "$AIPRO_V3_CONFIG" --role reviewer --difficulty 3 --json
aipro telemetry --config "$AIPRO_V3_CONFIG" --json
aipro reconcile --config "$AIPRO_V3_CONFIG" --repo owner/repo --dry-run --json
```

Status scans open issues carrying any configured lifecycle label, including
queued, active/review, needs-human, done, and error. `--issue N` inspects a
specific issue, including a closed one. Status does not adopt sessions, mark
activity, enforce timeouts, submit work, or change GitHub state. An idle CAO
reading alone is not proof that a turn completed. The runtime attribution must
match the configured lane/profile and durable run/work-item identity.

When CAO is unavailable, the command keeps the last durable phase, PR/head,
model/fallback assignments, profile, findings, dispositions, usage reservations,
and reason. Each runtime row has `verified: false` and a redacted reason.
A definite named-session 404 is reported as verified `disappeared`. Requests
are bounded to five seconds; one transport failure marks the remaining runtime
rows unverified without repeatedly waiting on the unavailable control plane.

Step is always a read-only preview. It reports `claim`, `poll_ci`,
`reconcile_active`, `halt`, or `idle` using the scheduler's queue rules. It
never claims work or launches a model. `--issue` filters displayed status;
the proposed step still respects capacity across the whole repository.
Reconcile also scans active/review issues, not only ready issues. Its runtime
and Git inventories remain incomplete, so non-noop actions are not approved
for automatic application; `--apply` remains unsupported.

## Version 1 JSON

Status/step return `schema_version`, `repository`, `evaluated_at`, `work_items`,
and `resources`. A never-claimed row has a null run id, queued phase, and empty
lanes. Claimed rows additionally carry round/head/PR, lease, last CI snapshot
and failure hashes, lane records, findings/archives/dispositions, usage, and
terminal reason. Step adds `dry_run`, `next_step`, and `inventory_complete`.
Telemetry and route explanation also use `schema_version: 1`; reconciliation
adds `dry_run` and `inventory_complete` to its versioned action envelope.

Lane records distinguish `last_durable_state` from the current `runtime`.
`routing` preserves the actual broker decision made at dispatch, including
rankings, candidate exclusions, score components, demand, and evaluation time.
Configured model pins identify their source instead of inventing scores.
Older records may have null routing/profile fields. `route explain` is a
hypothetical selection against current telemetry, not a reconstruction of an
older decision.

Finding origins carry reviewer role/lane, model, round, and reviewed head.
Dispositions retain finding ids and response-round links, including archived
findings. Claims, summaries, and rationales are bounded to 1,000 characters;
full evidence remains in GitHub's authoritative record. Prompts, raw model
responses, environment mappings, and source/test artifacts are excluded from
operator projections.

`cash_authorizations` contains maximum permitted charges per dispatch. Those
ceilings are not actual spend. `actual_cash_usd` and `actual_tokens` are null
because CAO's current result contract does not report them. Resource telemetry
shows provider-reported quota windows, utilization/reset times, balances, and
health where available; missing measurements remain unknown.

## JSON audit stream

The service writes JSON events to the existing journal. `workflow_state` is
emitted after verified durable saves; `model_routing` records selection;
`recovery_action` identifies checkpoint resume or expired-lease recovery.
Scheduler outcomes, reconciliation requirements, and audit errors also have
versioned event names. Workflow and routing events carry issue/run/round/head
correlation, with lane/role on routing and finding-origin records. Repeated
snapshots can be correlated by `updated_at`; journal retention does not define
business truth.

Shared redaction runs on native values before JSON escaping. It masks GitHub,
OpenRouter, Anthropic, OpenAI/Codex, authorization/OAuth fields, credential
names, and configured credential environment values, including custom keys.
Numeric token/quota counts remain numeric. Structured dictionary keys and
exception messages are filtered too; source prompts are not logged.
