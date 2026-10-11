# V3 deterministic acceptance and soak

Run from an installed development checkout:

```sh
uv run pytest tests/integration/e2e tests/integration/test_continuous_foreman.py
uv run pytest tests/integration/test_continuous_foreman.py::test_multi_issue_soak_has_no_duplicate_or_unmanaged_side_effects tests/integration/test_v3_cleanup_soak.py --soak-issues=40
```

CI runs six sequential issues by default; `--soak-issues` accepts at least two.
The soak uses the production continuous scheduler, foreman, model broker,
GitHub queue/serialized state, CAO controller/executor and GitHub CI adapter.
External GitHub/Git operations and model responses are faked. Each issue must
reach `done`, submit exactly one developer turn and three reviewer turns, and
stop its sessions. The test checks unique run ids/branches, one PR per issue,
no active claims, empty tracked worktrees, no live sessions and unchanged
completed durable records. A controller that does no work cannot pass.

The E2E suite covers clean work, fix/rebut/independent acceptance, conflicting
claims with stronger different-family adjudication, bounded CI repair,
subscription reserve/surplus dispatch, promotion expiry, cold turn recovery,
and sequential scheduling. Scenario 10 verifies repeated patch fingerprints
route to needs-human before more CI or dispatch. Provider fallback is separate from CAO control-plane
backpressure. Native Hermes was tested on `hermes` against a local synthetic
429 provider and fallback, with invented keys and a bounded run.

Issue dispatch requires both `v3-work` (ready) and `v3-work-enabled` (stable
opt-in). Lifecycle transitions preserve the stable label. Removing it cancels
active work only after all attributed sessions are confirmed stopped; an
unconfirmed stop retains capacity. Originating issue authors must match
`safety.allowed_pr_author_associations` before resources are created and while
work runs. Unknown metadata and PR numbers cannot authorize issue work.

The safety matrix exercises fork policy, prompt/iteration/invocation/reviewer
budgets, elapsed wall-clock limits and credential stripping through actual
lane dispatch. Real-Git authority tests cover commit budgets and workflow-file
restriction. Accepted native deletion without disappearance retains active
capacity; historical developer/reviewer lanes are stopped after a registry
rename. Native timeout and replacement stops use the same verified teardown.

Only comments from `github_queue.state_author_login` supply machine state.
An empty value resolves the authenticated GitHub user's login. GitHub App
installations must pin their bot login explicitly. Preserve that login across
credential rotation; drain and migrate authoritative comments before changing
writers. Comments from other accounts remain human input.

The native Git soak also runs the production sweeper after each issue. It uses
real branches, commits, reviewer clones and pushes to a local bare remote;
model tool effects, CAO HTTP and GitHub are the external boundaries. A failed
immediate checkout removal leaves a protected artifact, which the sweeper
must remove after its inactivity TTL. Completed durable records stay unchanged. Cold developer and reviewer cases
construct fresh queue/controller/executor objects and recover existing native
Git artifacts without another submission, branch or PR.

Cleanup tests retain active work and human checkouts, delay recently modified
files, and reject changed GitHub authority, empty runtime inventories with live
sessions, and acknowledged deletions that leave sessions alive.

P5 shipped the replacement configuration, thin queue trigger and migration
runbook in PR #133 after the acceptance/safety/cleanup gates. P6 retires V1
only after those changes pass CI on main. Historical epic #16 is closed in P7
after the retirement itself passes CI. See [ordered rollback](V1_MIGRATION.md).

The native Hermes/CAO 2.5 cleanup canary also confirmed the owned pinned-image
container was absent before checkout deletion, removed two protected checkouts
after TTL, and repeated idempotently. Evidence is attached to PR #131. These
native boundary checks and deterministic model fixtures establish acceptance;
funded live-provider reasoning and 24/7 host activation remain optional deployment
checks, not claims made by this repository cutover.
