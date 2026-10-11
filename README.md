# AI PR Orchestrator

aipro V3 runs an issue-to-PR loop through a persistent foreman, Hermes agents,
and CAO 2.5. GitHub issue comments hold authoritative run state. The coordinator
owns Git/GitHub mutations; one developer and three independent reviewer lanes
run through contained Hermes processes. Successful work produces a PR after
review and CI; merging remains a repository decision.

## Install in a target repository

1. On the coordinator host, clone this package and a primary checkout of the
   target repository. Install Python 3.12+ and run `uv sync --locked` in the
   package checkout. Keep coordinator push credentials separate from lane homes.
2. Follow [CAO provisioning](docs/V3_CAO.md) to install CAO 2.5, bind its control
   API to loopback, pin the Hermes container image, and provision the four
   independent lane wrappers and profiles. Verify containment before enabling work.
3. Copy [v3-config.yml](examples/v3-config.yml) and
   [model-catalog.yml](examples/model-catalog.yml) beside each other on the host.
   Replace illustrative model descriptors, providers, prices, promotions, quality
   and concurrency with verified account values. The catalog is shared across
   lanes; relative paths resolve beside the policy YAML. Match catalog providers
   to telemetry rows and reserve resource names. Set `telemetry.hermes_python`
   to the host Hermes interpreter. Credentials belong in dedicated Hermes homes,
   never the policy YAML. Custom catalog endpoints are currently unsupported
   by the lane bridge.
4. Set the target repository's exact `ci_policy.required_checks` names and the
   trusted issue-author associations. For a GitHub App, pin
   `github_queue.state_author_login` to its exact bot writer login. Create the
   configured lifecycle labels: `v3-work`, `v3-work-enabled`, `v3-work-active`,
   `v3-work-review`, `v3-work-needs-human`, `v3-work-done`, `v3-work-error`.
5. Install the Hermes supervisor profile and persistent user service using the
   [foreman runbook](docs/V3_FOREMAN.md). Its coordinator token needs repository
   contents/PR/issues write and checks/status read access. Verify noninteractive
   origin push authentication. Run one coordinator per target repository.
6. Copy [v3-queue-workflow.yml](examples/v3-queue-workflow.yml) into the target's
   `.github/workflows/`. It only adds the ready label; it runs no agents and
   requires only issues write permission. If you rename queue labels, edit
   both the workflow and host policy.

Inspect policy and live state before opting in:

```sh
.venv/bin/aipro catalog --config /path/to/v3-config.yml
.venv/bin/aipro telemetry --config /path/to/v3-config.yml --json
.venv/bin/aipro route explain --config /path/to/v3-config.yml --role worker --difficulty 3
.venv/bin/aipro status --config /path/to/v3-config.yml --repo owner/repo --json
```

Add `v3-work-enabled` to an issue to queue it through the installed trigger,
then inspect status and the service journal. Without the trigger, add both
`v3-work-enabled` and `v3-work`. The foreman independently verifies the
originating author's authorization. Remove the stable opt-in label to cancel;
capacity remains reserved until native lane termination is confirmed.
Needs-human and cancelled records require an explicit operator state decision;
re-adding a label does not reset their history or budgets.

See [operator diagnostics](docs/V3_OBSERVABILITY.md),
[acceptance evidence and limits](docs/V3_SOAK.md), and
[V1 migration and rollback](docs/V1_MIGRATION.md). V1 commands and examples
are retired; the supported CLI operates only on V3.

## Development

```sh
uv sync --locked --all-extras --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pre-commit install
```
