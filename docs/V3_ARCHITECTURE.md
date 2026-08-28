# aipro V3 Architecture: Thin Autonomous Coding Policy Engine

*Status: target architecture defined by issue #41. The V1 runtime remains the
only executable path until cutover; this document defines the seam.*

## 1. What aipro becomes

aipro V3 is a **thin, deterministic policy engine** layered above two existing
platforms, not a process/runtime orchestrator of its own:

- **Hermes Agent (supervisor/foreman)**: the persistent control loop. It
  decides *what happens next* — which work item to claim, which lane to run,
  when to iterate, when to escalate.
- **CAO**: the session/process execution fabric and agent lifecycle manager.
  It starts, observes, and terminates agent sessions.
- **Hermes worker/reviewer profiles**: the actual coding and review agents,
  each with isolated Hermes state.
- **GitHub**: the authoritative business/workflow state and audit trail.
  Queue membership, phase, findings, decisions, and history live there.
- **aipro (V3)**: deterministic policy only — queue state, model allocation,
  review protocol, limits, recovery, CI/PR gates.

**Ownership shift, stated explicitly:** CAO and Hermes replace aipro's direct
CLI-provider process management. V1's subprocess-spawning coder/reviewer
adapters (`coders/`, `reviewers/`) exist because aipro once had no execution
fabric underneath it. In V3 that responsibility moves to CAO + Hermes; aipro
never spawns, supervises, or tears down an agent process.

## 2. Responsibility map

| Responsibility | Owner | Notes |
| --- | --- | --- |
| Work queue (which issues are eligible, claim ordering) | aipro policy, state in GitHub | GitHub labels/comments are the durable state |
| Workflow phase/state machine (V3 phases) | aipro policy | `v3.domain.WorkflowState`; persisted to GitHub |
| Agent session lifecycle (start/poll/kill) | CAO | aipro only declares attachment points (`v3.config.CAOControlPlaneConfig`) |
| Foreman control loop ("what next?") | Hermes supervisor | consumes aipro policy output |
| Coding/review agent behavior | Hermes worker/reviewer profiles | isolated Hermes state per lane |
| Model allocation (which model per lane) | aipro policy via model catalog/router | refs only; brokers resolve to concrete models |
| Model capacity reservation | model broker (V3 interface) | implementation bridges to providers |
| Review protocol (rounds, dispositions, thread rules) | aipro policy | `v3.config.ReviewPolicyConfig` |
| Limits and budgets | aipro policy | enforced before lanes execute |
| CI/PR gates | aipro policy, evaluated via `CIPRGate` | GitHub supplies check status |
| Human escalation | aipro policy | triggers on failures/stagnation thresholds |
| Audit trail | GitHub | aipro writes decisions; never a local source of truth |

## 3. V3 package boundary

All V3 code lives under `ai_pr_orchestrator.v3` so it can coexist with the V1
runtime until cutover:

- `v3.domain` — provider-independent, serializable domain types
  (`WorkItem`, `GitHubIssueRef`, `WorkflowState`, `LaneIdentity`,
  `ModelAssignment`, `ReviewerFinding`, `FindingDisposition`,
  `FailureSummary`, `StagnationSummary`).
- `v3.config` — the V3 policy schema (`V3Config` and sections: GitHub queue,
  CAO control plane, Hermes lanes/profile templates, model catalog/router,
  review roles/limits, CI/PR policy, safety rails, human escalation). The
  `safety` section carries the V1 safety surface across the cutover: fork
  and workflow-file restrictions, per-run budgets (iterations, commits,
  lane invocations, prompt tokens), and PR-author association allowlists.
  Unknown keys are preserved as `extras` for forward compatibility.
- `v3.catalog` — the shared machine-level model catalog: candidate metadata
  (price, promotion window, capability, role suitability, quality tier,
  lineage, limits) plus eligibility queries. It states facts and never ranks;
  quota, health, and diversity belong to the broker. See
  `docs/V3_MODEL_CATALOG.md`.
- `v3._schema` — the mapping/dataclass coercion rules shared by `v3.config`
  and `v3.catalog`: shape validation before construction, rejection of `None`
  for non-optional fields, and `extras` preservation.
- `v3.telemetry` — live resource state: quota windows, provider health, and
  freshness, as a pure domain with no vendor knowledge. Missing telemetry
  yields *unknown*, never *zero*, and health is kept structurally separate
  from quota so a 429 can never be recorded as an exhausted allowance. See
  `docs/V3_TELEMETRY.md`.
- `v3.telemetry_hermes` — the Hermes account-usage adapter, which runs a
  constant bridge script under Hermes' own interpreter rather than importing
  its pinned dependency tree. The only telemetry module that names a vendor.
- `v3.interfaces` — protocols for `GitHubWorkflowStateStore`,
  `CAOSessionController`, `ProviderTelemetrySource`, `ModelBroker`,
  `LaneExecutor`, and `CIPRGate`. All are structural and fakeable in tests
  without CAO, Hermes, or GitHub.
- `v3.lanes` — the lane registry, sole authority on the lane-to-profile
  binding. Lane names *and* profile templates are unique: every concurrent
  lane owns an independent agent profile/home.
- `v3.cao` — the CAO control-plane adapter. It never spawns a process, parses
  a terminal, or picks a model. See `docs/V3_CAO.md` for the CAO version
  floor, the endpoints it relies on, and Hermes lane profile provisioning.

`v3.cao` and `v3.telemetry_hermes` are the only V3 modules that perform I/O.
The first speaks HTTP to CAO; the second spawns exactly one subprocess, a
constant bridge script under Hermes' own interpreter. Neither ever executes an
agent.

No V3 core type or interface names a specific vendor or model. `ModelRef`
points into the catalog; catalog `descriptor` strings are opaque to the core
and interpreted only by the model broker adapter.

The catalog is normally a machine-level file shared by every lane, referenced
from `model_router.catalog_path`; inline entries remain available for tests
and single-repo setups. Declaring both is rejected as ambiguous.

## 4. Migration table (V1 → V3 cutover)

| V1 module | Decision | Rationale / successor |
| --- | --- | --- |
| `runner.py` | **REPLACE** | Its loop becomes the Hermes foreman loop. The scheduling/policy skeleton it embeds is reimplemented against V3 interfaces. |
| `coders/` (CLI coder adapters) | **DELETE** | Direct CLI-provider process management is replaced by CAO sessions + Hermes worker lanes. |
| `reviewers/` (in-process reviewers) | **DELETE** | Reviewer agents become Hermes reviewer profiles executed by CAO; findings flow through `ReviewerFinding`/`FindingDisposition`. |
| `state_storage.py` | **WRAP** | The GitHub hidden-comment persistence mechanics survive behind `GitHubWorkflowStateStore`; the V1 `RuntimeState` payload is replaced by V3 state at cutover. |
| `state_machine.py` | **WRAP then REPLACE** | Transition guard logic is absorbed into the V3 phase model (`WorkflowState` validation) and foreman policy; the V1 status enum retires at cutover. |
| `models.py` (V1 domain types) | **REPLACE** (progressively) | V3 types in `v3.domain` supersede `Finding`/`HandledFinding`/`RuntimeState`; kept until the last V1 consumer is gone. |
| V1 execution template (`examples/target-repo-workflow.yml`, the GitHub Actions workflow that shells out to the coder CLI) | **DELETE** | Execution moves to CAO/Hermes; GitHub Actions remains only for CI checks and the thin trigger/queue listener. |
| `config.py` (V1 schema) | **WRAP** until cutover, then **REPLACE** by `v3.config` | V1 and V3 configs coexist; V3 loader is independent. |
| `decision_application.py`, `logging.py` | **KEEP** | Deterministic helpers with no process management; reused by V3. |
| `github/` (API client) | **KEEP** (adapted) | Transport for the `GitHubWorkflowStateStore` adapter. |
| `git/` (branch/commit helpers) | **KEEP** (adapted) | Invoked by lanes/foreman rather than the runner. |

## 5. Non-goals of the seam (issue #41)

- No model routing logic or foreman loop is implemented here. (The CAO
  adapter arrived separately in issue #42; see `docs/V3_CAO.md`.)
- No V1 behavior is deleted yet.
- CAO and Hermes internals are not redesigned.
