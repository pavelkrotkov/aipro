# V3 deterministic acceptance and soak

Run from an installed development checkout:

```sh
uv run pytest tests/integration/e2e tests/integration/test_continuous_foreman.py
uv run pytest tests/integration/test_continuous_foreman.py::test_multi_issue_soak_has_no_duplicate_or_unmanaged_side_effects --soak-issues=40
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
and sequential scheduling. Provider fallback is separate from CAO control-plane
backpressure. Native Hermes was tested on `hermes` against a local synthetic
429 provider and fallback, with invented keys and a bounded run.

These tests do not establish cutover readiness by themselves. In particular,
originating-issue authorization and stable opt-in cancellation need production
guards and boundary tests; orphan cleanup needs verified attribution/inventories.
The replacement config, queue trigger and migration docs must ship before V1
retirement. Keep V1 and historical epic #16 until the full cutover gate passes.
