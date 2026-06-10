# AI PR Orchestrator

AI PR Orchestrator is a Python-based GitHub PR orchestration tool for coordinating
AI coding agents and AI reviewers.

The current repository contains the project plan, minimal package scaffolding, and
repo hygiene automation.

## Installing in a target repository

> **Preview:** these artifacts let you *configure* a target repo, but the loop is
> not yet executable end-to-end. The orchestrator's runtime wiring
> (`_build_runtime_context` — GitHub client, git, coder, and reviewer adapters)
> is a tracked follow-up and not implemented yet, so `aipro run` currently exits
> with "AI PR Orchestrator runner is not implemented yet." You will also need to
> install your coder CLI in the workflow and keep `PATH` in `main_coder.env`
> before the coder phase can run.

To set up the review loop on your own repository:

1. Copy [`examples/target-repo-workflow.yml`](examples/target-repo-workflow.yml)
   to `.github/workflows/ai-review-loop.yml` and replace `YOUR_ORG` in the
   install step with the org hosting this package.
2. Copy [`examples/sample-config.yml`](examples/sample-config.yml) to
   `.github/ai-review-loop.yml` and adjust it to taste. Only `main_coder` is
   required; everything else falls back to built-in defaults.
3. Add the required repository secrets:
   - `CODEX_API_KEY` — credential for the coder.
   - `AI_ORCHESTRATOR_GITHUB_TOKEN` *(optional)* — a PAT or app token for the
     orchestrator; falls back to the workflow's `GITHUB_TOKEN`.
4. Add the `ai-loop` label to a PR to opt it into the loop.

The workflow grants the least-privilege scopes the orchestrator needs
(`contents: write`, `pull-requests: write`, `issues: write`, `checks: read`,
`statuses: read`) and serializes runs
per PR via a `concurrency` group with `cancel-in-progress: false`. The
orchestrator token is never forwarded to the coder: it is not listed in
`main_coder.env`, and the coder adapter strips `GH_TOKEN`/`GITHUB_TOKEN` from the
coder subprocess environment.

## Development

Install the project and development tools:

```sh
uv sync --all-extras --dev
```

Run the test suite:

```sh
uv run pytest
```

Run linting and formatting:

```sh
uv run ruff check .
uv run ruff format .
```

Run type checking:

```sh
uv run ty check .
```

Install pre-commit hooks:

```sh
uv run pre-commit install
```
