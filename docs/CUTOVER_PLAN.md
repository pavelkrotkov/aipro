# aipro V3 Cutover Plan (issue #55) — rev 4

Status: **proposed (rev 4, addressing review round 3)**. Review before any V1
code is removed. This is the human-authorization gate for retiring V1.

## 0. Goal

Prove V3 end-to-end against the **real** adapters, then remove superseded V1
execution plumbing so the repo has one supported autonomous path: **Hermes
supervisor + CAO session fabric + GitHub-as-authoritative-state + aipro V3
policy**. V1 stays runnable and CI-tested throughout; fallback requires **two
ordered reverts (P6 then P5)** after draining in-flight V3 work, so a
rollback is always possible but never instantaneous.

## 1. Principles (changed in rev 2)

1. **The E2E suite exercises the real production adapters.** The foreman loop,
   `CaoLaneExecutor`, `CIPRGate`, and `PolicyBroker` are in the harness path;
   only the *external* boundary is faked — the CAO/LLM transport (a
   `FakeCAOServer` speaking the real CAO HTTP contract) and telemetry/clock. A
   fake `LaneExecutor` is used **only** for unit-level lane-policy tests, never
   as the E2E executor.
2. **No phase advances the default or deletes V1 until the production glue
   exists and its own acceptance gate is green on CI.** The plan cannot stall at
   "missing glue" and cannot ship a default that can't execute.
3. **Every destructive step ships its counterpart first**: the V3 trigger
   replaces the V1 workflow *before* the V1 workflow is deleted; the V3 sample
   config replaces the V1 sample *before* V1 config is made sole; migration
   notes ship *with* the removal that breaks users, not after.
4. **Safety parity is an explicit E2E gate.** The safeguards currently declared
   only as `v3.config.safety` fields must be proven enforced before V1's
   enforcement is retired.

## 2. Phasing — each phase is a separate PR, independently shippable

| Phase | Deliverable | Adds / removes | Gate to advance |
| --- | --- | --- | --- |
| **P1** Plan (rev 4) | this doc | docs only | review sign-off |
| **P2** Production glue | Foreman policy loop (`v3.foreman` over `GitHubWorkflowStateStore`+broker+lanes+lane-registry), `CaoLaneExecutor` (real `CaoSessionController` + Hermes lane), `CIPRGate`+GitHub CI adapter, GitHub `issue-content`/`pr-create` protocol ops, `git` worktree/branch/commit/push interface+impl, catalog-resolver+`PolicyBroker` wiring, **`GitHubIssueQueue.abandon()` (requeue+clear-claim drain op)**, **`v3.cleanup` production TTL sweeper** (owns CAO-session + worktree TTLs, invoked by the foreman) | adds only | `uv run pytest` + new unit tests green |
| **P3** E2E harness foundation | `tests/e2e/` deterministic harness; **real adapters** ref`d against a `FakeCAOServer` (real CAO HTTP contract) + `FakeGitHub` boundary + scripted telemetry/clock; scenarios 1–4 (clean issue→draft PR→3 reviews→CI→PR-ready — **draft PR created before the CI gate since `CIPRGate.evaluate()` requires a PR ref**; blocking-fix; rebuttal/independent-acceptance; disagreement→adjudication) | adds only | scenarios 1–4 pass repeatedly |
| **P4** Soak + failure + safety | `tests/e2e/soak.py` runner **exercising the `v3.cleanup` production sweeper (not its own copy)**; scenarios 5–12; **safety-parity scenarios** (fork reject, workflow-file restriction, iteration/commit/invocation budgets, `max_prompt_tokens`, `allowed_pr_author_associations`, credential-stripping) | adds only | full soak pass, no dup side effects, safety enforced |
| **P5** Default + trigger + docs | V3 thin queue trigger workflow (adds before delete); make Hermes+CAO default; README runbook; **`docs/V1_MIGRATION.md`**; **`examples/v3-config.yml`** (new path, distinct from the V1 sample it supersedes) + update README references | adds, then docs react | P2–P4 green + your sign-off |
| **P6** V1 retirement | DELETE `coders/`, `reviewers/`, `runner.py` monolithic loop, V1 `workflow.yml`, and now-unused V1 `models`/`state_*`/`config.py` + **the now-superseded `examples/sample-config.yml` (V1 path only)**; adapt `agents/`, `decision_application.py` consumers; rewire `aipro` CLI to V3; migration notes finalize. **The `examples/v3-config.yml` sample survives.** | removes V1 path | P5 shipped; CI green with notes |
| **P7** Epic close | close V1 epic #16 pointing to #56; reopen-and-mode notes vs issues **not** handled by a git revert | docs | P6 shipped |

**Deletion gate (before P6):** the complete production path must exist (P2),
the E2E/soak/safety suite must run the real adapters green on CI (P3–P4), and
the replacement trigger/sample/docs/migration notes must already be shipped
(P5). P6 then deletes only what is provably superseded.

## 3. Intent of each gate (rev 4)

- **P2 IS the missing-glue gate.** Rev-2 lesson: P2/P3 must not be tests-only. The
  foreman loop, `CaoLaneExecutor`, `CIPRGate`, broker wiring, and the repo/PR
  lifecycle ops (issue read, branch/worktree, commit, push, PR create) are the
  production capabilities §5 of the old rev identified as absent; they are
  built here, with their own acceptance tests, *before* any default change.
- **P3 tests real adapters against a fake transport**, so CAO restart/
  reconciliation and worktree-isolation scenarios traverse the real
  `CaoSessionController` ↔ `CaoLaneExecutor` integration, not a scripted pair.
- **P4 runs broker scenarios through the real `PolicyBroker`** with only
  telemetry and the clock scripted — a broken ranking/wiring cannot pass.
- **GitHub is also faked at the transport boundary** (`FakeGitHub`, same
  principle as `FakeCAOServer`): the real `GitHubIssueQueue`, `CIPRGateImpl`,
  and PR/issue adapters run against it, so the E2E path stays deterministic
  without mutating live GitHub and without dropping the production adapters.
- **Cleanup TTL (defined here, owned in P2 by `v3.cleanup`, enforced in P4):**
  CAO sessions are orphaned past `CAOControlPlaneConfig.session_lease_ttl_seconds`
  (default 2h) without renewal; git worktrees are orphaned past 24h of
  inactivity. `v3.cleanup` is a **production component the foreman invokes**; the
  P4 soak suite calls that same component each round (it does not reimplement
  cleanup). The sweeper owns its clock and logs each removal. (Config fields for
  the two TTLs are added in P2 under `v3.config`.)
- **Draining active claims (defined here, implemented P2)**: rollback/shrink
  requires reaching **zero active leases**. `GitHubIssueQueue.reclaim_expired`
  does not drain — it re-claims and extends the lease — so a
  `GitHubIssueQueue.abandon(issue, state)` operation (requeue to `queued` and
  clear claim attribution, idempotent) is the drain primitive. The foreman
  stops heartbeat renewal and calls `abandon` on each active item.
- **Safety-parity scenarios (P4)** mirror the V1-permitted operations:
  `disallow_forks`, `disallow_workflow_file_changes`, per-run budgets
  (`max_*`), **`max_prompt_tokens`**, **`allowed_pr_author_associations`**,
  **opt-out label removal stops active work** (V1's `only_run_on_labeled_prs`
  parity: a separate stable opt-in label — distinct from the per-phase
  `enabled_label`, which `_apply_phase_labels` always removes once `claim()`
  moves the issue into the active lifecycle label — is required for the
  foreman to claim an item, and removal of that opt-in label mid-run
  causes the foreman to abandon active items via `GitHubIssueQueue.abandon()`),
  and credential stripping — each is
  exercised as a pass/fail E2E case through the real policy path, not just
  asserted in config. V1's enforcement is not removed (P6) until each passes.

## 4. V1 → V3 inventory (complete, all consumers accounted for)

| V1 module / file | Action | Reverse dependencies handled |
| --- | --- | --- |
| `runner.py` | DELETE (P6) | foreman policy loop absorbs its responsibilities |
| `coders/`, `reviewers/` | DELETE (P6) | replaced by CAO/Hermes lanes |
| `models.py`, `state_machine.py`, `state_storage.py` | DELETE (P6) after rewire | `decision_application.py`, `agents/output_validator.py`, `agents/prompt_builder.py`, `logging.py` import V1 types → **adapted to V3 types in P2/P6, never left broken** |
| `decision_application.py`, `logging.py` | KEEP (adapted to V3 types) | explicit adapter change in P6 |
| `agents/` (output_validator, prompt_builder) | KEEP (adapted) | see inventory above |
| `github/`, `git/` | KEEP (extended in P2: issue-content, pr-create, worktree/branch/commit/push) | V3 adapters |
| `config.py` (V1), `examples/sample-config.yml` | DELETE (P6); **V3 sample added P5, references updated** | README must copy a V3-valid config after P5 |
| V1 execution workflow (`examples/target-repo-workflow.yml`) | DELETE (P6) **after** V3 trigger shipped (P5) | GitHub Actions CI only after |
| `cli.py` `aipro` script | REWIRE (P6) | V3 policy ops (broker dry-run, queue status, state inspect); `aipro run` deprecated in P5 notes, removed in P6 |

## 5. E2E/soak harness (P3–P4)

Deterministic by construction: real foreman + real `CaoLaneExecutor` + real
`CIPRGate` + real `PolicyBroker`; **faked only at the external boundary** —
`FakeCAOServer` (speaks the real CAO HTTP contract), scripted telemetry, a
fixed clock, and real-provider smoke behind `AIPO_E2E_LIVE=1`. The 12
issue-#55 scenarios are traced in `tests/e2e/README.md` (written P3); the
added safety scenarios are documented in P4. `soak.py` runs `--runs N
--jitter`, sweeps orphans by the §3 TTLs, and fails on any duplicate branch/PR,
leaked active claim, stuck active label, orphan beyond TTL, or state
divergence.

## 6. Rollback (rev 3 — full-path, including active V3 state)

- **Full fallback requires reverting BOTH destructive PRs in reverse order,
  after draining** — P6 (V1 removal) then P5 (V3 default/trigger/config/docs).
  Reverting P6 alone restores the old V1 files while P5 still sets V3 as the
  default and ships the V3 sample/README, so V1 would not actually be the
  usable path. Each destructive phase's PR bundles the removal *and* its
  counterpart docs/config so a discarded intermediate state is never valid.
- **In-flight V3 work:** before any fallback from an active V3 state, **quiesce
  then drain** — stop new claims, stop heartbeat renewal, then **explicitly
  terminate and confirm every active CAO session has stopped** (a TTL sweep is
  *not* sufficient: a session mid-commit could keep pushing after its claim is
  released), and only then call `GitHubIssueQueue.abandon()` on each item to
  reach zero active leases. `abandon()` **clears lease ownership while
  retaining durable branch/PR linkage** (branch name, PR number), so a later
  V3 claimant recovers the existing branch/PR instead of duplicating them.
  GitHub issue-comment state is left intact. **V1 cannot read V3 state**, so
  draining is a hard precondition to any revert that could strand a claim or
  start duplicate work.
- Reopening a closed epic/tracking issue is a repo mutation, not a git revert;
  the P7 procedure lists the exact issues to reopen. There is no compatibility
  loader: V1↔V3 config/state conversion is out of scope and stated as such.

## 7. Issue-scoped acceptance mapping

Traced per criterion to a phase + the specific check each PR must satisfy in
`tests/e2e/README.md` (written P3). The delete-of-V1 criterion ("no supported
production path imports the old provider adapters") resolves to P6 only after
P2–P5 gates.