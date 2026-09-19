# V3 deterministic soak and remaining cutover gates

Run `uv run python tests/integration/soak/soak.py --runs 5 --jitter` from the
repository. `--runs` and `--issues-per-round` control campaign size; injected
coder failures are bounded and must still reach `done`. A successful run requires
an outcome and readable durable state for every seeded issue, no duplicate
branches/PRs, no stuck claims/active labels, consistent persisted branch/PR links,
and actual removal of seeded resources from persistent resource stores. Missing
or failing cleanup controllers cannot count as successful removal. Attributed
resources require readable durable ownership records even when their issues are
ready or unlabeled; the soak seeds terminal records for its cleanup fixtures.

This deterministic policy harness uses fake GitHub, Git, CI, broker and lane
boundaries. It does **not** certify production CAO/Hermes execution or replace
real-provider smoke tests. It runs the production reconciliation planner and
cleanup application code. Stale leases are surfaced for reconciliation; cleanup
never independently reclaims them. Unfinished checkpoints retain their resources.

Automatic Git cleanup observes only clean `aipro-issue-*` branches under the
foreman's explicitly owned worktree root. Workspace activity bounds the TTL;
an inherited commit date does not. Git removal is non-force, so a tree made dirty
after observation is retained with a visible cleanup failure. Unowned worktrees
are never candidates. CAO inventory comes from the remote session list and
terminal `last_active`, using durable aipro attribution. Missing attribution or
activity blocks automatic cleanup.

## Acceptance evidence

| #55 scenario | Current evidence | Remaining acceptance |
| --- | --- | --- |
| 1 | Happy-path policy/CAO tests | Production findings and changed paths (#76/#78) |
| 2–4 | Finding/remediation tests; existing xfails | Reliable CAO turns (#74/#75), durable independent rebuttal/adjudication (#87) |
| 5 | CI-failure bounded-fix policy test | Actual multi-turn CAO execution and complete re-review (#74/#75/#76/#83) |
| 6 | CAO launch HTTP 429 preserves state and escalates | Hermes **provider** fallback with phase continuity; a control-plane 429 is not this scenario |
| 7–8 | Real deterministic broker selection tests | Broker choice reaches Hermes (#77) |
| 9 | Planner recovery decisions and failed-restart regression | Successful cold reconstruction from GitHub + live CAO, pending-PR-only resume (#85), durable terminal persistence (#80), PR reconciliation (#81) |
| 10 | Stagnation/cap policy tests | Production repeated turns and fail-closed findings (#74/#75/#76) |
| 11 | Workflow-mutation rejection and authoritative Git operations | Real CAO reviewer-worktree isolation/credential boundary and authoritative changed paths (#78) |
| 12 | Two issues finish with three review calls each | Demonstrated temporal review concurrency and fresh real CAO workers per issue |

Safety-parity tests verify individual controls at their stated boundary. Prompt
secret exclusion is not evidence that worker process environments strip authority.
Queue abandonment tests alone are not proof of live CAO termination ordering.
The independent execution-boundary tests remain required before retiring V1.

## Operational limitations and cutover status

The foreman now accepts the CAO controller explicitly. The supported external
Hermes runner must supply it; there is no in-repository production constructor
wiring to certify yet (#53 integration). `aipro reconcile --apply` is always rejected because this CLI has no complete
runtime inventory or execution-controller/local-worktree binding. Authentication
or a NOOP from the planning-only view does not authorize successful apply. Previous output-only recovery/cleanup was not
real application. Dry-run planning remains available; use the configured foreman
for supported cleanup, and explicit reconciliation for manual actions.

P4 and #55 are **incomplete**. #74–#87 must be reconciled against actual merged
code, then all deterministic scenarios, restart cases and soak must be rerun.
Unknown side-effect outcomes must reconcile. No skipped/xfail test is acceptance
evidence. No V1 deletion, historical #16 closure or epic #56 completion is
permitted by this policy harness alone. P5/P6 documentation/cutover work must wait
for the full production acceptance gate.
