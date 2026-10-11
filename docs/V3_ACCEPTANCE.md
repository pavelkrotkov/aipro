# V3 cutover acceptance record

Recorded 2026-10-11 for the repository cutover in #55 and epic #56. V1 retirement
landed in [PR #134](https://github.com/pavelkrotkov/aipro/pull/134), main commit
`b9aecde668ebe95bf50173c7fb2f13a14ee38484`, after the replacement configuration,
trigger and migration runbook in [PR #133](https://github.com/pavelkrotkov/aipro/pull/133).
This is a verification record; deployments must inspect their actual code and state.

| Acceptance area | Evidence |
| --- | --- |
| Scenarios 1–5: clean work, fix, rebuttal, independent adjudication, bounded CI repair | [E2E suite](../tests/integration/e2e/), production foreman/broker/controller/executor/CI adapters |
| Scenario 6: provider fallback and control-plane backpressure | Native Hermes synthetic 429-to-fallback check in [PR #127](https://github.com/pavelkrotkov/aipro/pull/127); separate CAO HTTP failure tests |
| Scenarios 7–8: reserve/surplus and promotion expiry | E2E dispatch through the actual broker and shared catalog |
| Scenarios 9 and 12: cold recovery and sequential issues with blind parallel review | [Continuous scheduler tests](../tests/integration/test_continuous_foreman.py) and cold native-Git soak |
| Scenario 10: repeated patch becomes needs-human | [Safety matrix](../tests/integration/e2e/test_safety_matrix.py), without further dispatch/CI |
| Scenario 11: reviewer mutation stays isolated | Native developer worktrees, independent reviewer clones, and contained Hermes authority checks |
| Authorization, opt-out, budgets, wall-clock, fork/workflow restrictions and credential stripping | Safety matrix, [native changed-file tests](../tests/integration/test_changed_file_authority.py), and trusted-state/cancellation regressions |
| TTL cleanup without duplicate effects or unsafe capacity release | [Production-sweeper soak](../tests/integration/test_v3_cleanup_soak.py), failed immediate cleanup, retained human resources and confirmed native stops |
| Sole supported runtime and migration | Fresh-process package import guard, retired-command rejection, built-wheel check, [README](../README.md) and [migration runbook](V1_MIGRATION.md) |

The 40-issue native-Git soak passed for normal, cold-developer and cold-reviewer
cases (120 issues), preserving completed records and producing one branch/PR
per issue without unmanaged sessions/checkouts beyond TTL. Final retirement CI
passed **1,362 tests, three environment-gated skips**; lint, format, types,
workflow security, dependency checks and CodeQL passed. The source distribution
and wheel built locally. Six changed retirement functions have max CC 2 and
minimum function MI 80.56; existing module-level maintainability debt remains.

A separate [native Hermes/CAO 2.5 cleanup check](https://github.com/pavelkrotkov/aipro/pull/131#issuecomment-6105911405)
confirmed the owned pinned-image container was absent before deleting two
protected checkouts, then repeated idempotently. Native checks used invented
credentials. GitHub/model responses and the CAO HTTP boundary are fixtures in
the deterministic suite; native Git operations are real. Funded provider
reasoning and 24/7 host activation are optional deployment checks and are not
claimed by this acceptance record. Sample model/account values require operator verification.

Q and Greptile reviews completed; valid findings were fixed in batched follow-ups.
Requested J-Bot reviews failed before review because `COMMANDCODE_ACCESS_KEY`
was missing; those failures are not independent-review evidence.

P7 closes historical epic #16 with a pointer to #56 after green cutover CI,
without rewriting its V1 history, then completes #56. Historical PR #89 is
superseded and must be closed rather than merged as proof. Rollback requires
native drain and ordered P6/P5 reverts; reopening #16/#56 and restoring target
workflows or host services are explicit external actions, as documented in
[V1_MIGRATION.md](V1_MIGRATION.md).
