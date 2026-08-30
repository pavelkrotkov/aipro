# aipro V3 Cutover Plan (issue #55)

Status: **proposed**. Review this before any V1 code is removed. This is the
human-authorization gate the bootstrap requires for retiring the V1 runtime.

## 0. Goal

Prove the V3 architecture end to end under realistic failures, then remove the
superseded V1 execution plumbing so the repo has one supported autonomous path:
**Hermes supervisor + CAO session fabric + GitHub-as-authoritative-state + aipro
policy (V3)**. V1 is not deleted until every V3 acceptance gate passes; V1 stays
fully runnable throughout so we can fall back at any point.

## 1. Phasing — each phase lands as its own PR and is independently shippable

| Phase | Deliverable | Removes / adds |
| --- | --- | --- |
| **P1 Cutover plan** *(this doc)* | Reviewable migration plan + exact V1→V3 inventory | none (docs only) |
| **P2 E2E harness foundation** | `tests/e2e/` deterministic harness + fake-LLM lane/CAO stubs; scenarios 1–4 | adds only |
| **P3 Soak/failure scenarios** | scenarios 5–12 + `tests/e2e/soak.py` runner with cleanup TTL checks | adds only |
| **P4 Default architecture** | Make Hermes+CAO the documented default; README runbook update | docs |
| **P5 V1 retirement** | Delete `coders/`, `reviewers/`, `runner.py` monolithic loop, V1 execution workflow template | removes V1 path |
| **P6 Migration notes + epic close** | `docs/V1_MIGRATION.md`, close V1 epic #16 pointing to #56 | docs |

P2–P3 are additive; the destructive steps (P4–P6) only run after the E2E/soak
acceptance criteria pass on CI, and only with your sign-off on the Phase → Go
checkpoint.

## 2. V1 → V3 inventory (from the issue-#41 migration table)

| V1 module | Cutover action | Notes |
| --- | --- | --- |
| `runner.py` | DELETE at P5 | Its scheduling/policy skeleton is reimplemented against V3 interfaces (foreman policy, not yet in repo — see §4) |
| `coders/`, `reviewers/` | DELETE at P5 | Direct CLI-provider process management replaced by CAO sessions + Hermes worker/reviewer lanes |
| `state_storage.py`, `state_machine.py` | WRAP→REPLACE at P5 | GitHub hidden-comment persistence reused behind `GitHubWorkflowStateStore`; V1 `RuntimeState` payload retired |
| `models.py` (V1 types) | REPLACE at P5 | Superseded by `v3.domain`; removed once no V1 consumer remains |
| V1 execution workflow (`examples/target-repo-workflow.yml`) | DELETE at P5 | GitHub Actions stays only for CI checks (tests.yml, python-quality.yml) and the thin trigger |
| `config.py` (V1 schema) | REPLACE at P5 | `v3.config` becomes the sole schema; CLI `aipro` script points at V3 |
| `decision_application.py`, `logging.py` | KEEP | Deterministic, no process management |
| `github/`, `git/`, `models.py` utils | KEEP (adapted) | Transport + helpers for V3 adapters/lanes |
| `cli.py` | REWIRE | Entrypoint becomes V3 policy operations (dry-run broker, queue status, state inspect) |

**Shims:** V1 config is not reloadable at P5 (no compat loader); the migration
note calls out that `aipro run` (V1 runner) is deprecated → removed, and the V3
`aipro` surface replaces it.

## 3. E2E / soak harness design (P2–P3)

Deterministic by construction: **fake/scripted lanes** implement
`v3.interfaces.LaneExecutor` and `CAOSessionController` with scripted outcomes,
and a **fake provider broker** returns a fixed catalog + scripted telemetry. No
LLM call anywhere; real-provider smoke is an opt-in flag `AIPO_E2E_LIVE=1`.

Harness covers the 12 scenarios from issue #55 (numbered as written): clean
issue→3 reviews→CI→PR; blocking-finding fix flow; rebuttal + independent
acceptance; reviewer disagreement→adjudication; CI failure bounded fix; 429
→Hermes fallback (lease preserved); broker pull-forward near reset; promotion
expiry changes next assignment; foreman/CAO restart→reconciliation; stalled
patch→needs-human; reviewer worktree isolation; two queued issues with parallel
review fan-out.

### Soak runner (`tests/e2e/soak.py`)
Configurable `--runs N --jitter`; drives the harness over many synthetic issues
and asserts the issue-#55 soak criteria are never violated: no duplicate
branch/PR creation, no leaked active claims, no stuck active labels, no orphan
sessions/worktrees beyond the documented cleanup TTL, and no state divergence
between the in-memory harness and GitHub-authoritative state. A local
state-directory TTL sweep runs between rounds. Optional real-provider smoke
behind `AIPO_E2E_LIVE=1`.

## 4. What must exist before V1 can be deleted (gates)

V1 removal is **blocked** until the repo actually contains the V3 execution
glue that `v3` currently only *defines as interfaces*:

- A **foreman policy loop** (the "what next?" decision over
  `GitHubWorkflowStateStore` + broker + lane registry) — currently a bootstrap
  script, not yet a `v3` module. **Open gap.**
- A **`CaoLaneExecutor`** exercising `CaoSessionController` + a Hermes profile
  per lane, and **`CIPRGate`**/CI adapter.
- A **catalog resolver + broker wiring** (`PolicyBroker`) from `v3.config`.

P2–P3 will surface exactly which glue is missing by driving these interfaces;
the E2E harness is intentionally the forcing function. If a gate can't be met,
the plan descends to a smaller cutover (keep V1 execution behind a `--legacy`
flag rather than deleting) rather than deleting a still-reachable path. **No V1
code is deleted until every gate here is green on CI.**

## 5. Rollback

Because each destructive phase is a separate revertible commit that also ships
migration notes, rolling back means reverting the P5 commit; V1 retained its
tests and imports until that commit. GitHub Actions CI keeps running V1 tests
until P5, so regressions are visible before the destructive step.

## 6. Issue-scoped acceptance mapping

Every issue-#55 acceptance criterion is traced to a phase in
`tests/e2e/README.md` (written in P2), with the check each PR must satisfy.