"""V3 thin policy-engine seam.

This subpackage defines the target architecture of aipro V3: a deterministic
autonomous coding policy engine layered above Hermes Agent (supervisor +
worker/reviewer profiles) and CAO (session/process execution fabric).

V3 is intentionally narrow:

- **Domain types** (:mod:`ai_pr_orchestrator.v3.domain`) describe work items,
  workflow phases, agent lanes, model assignments, reviewer findings, and
  failure/stagnation summaries. They are pure data: provider independent,
  typed, and serializable.
- **Configuration** (:mod:`ai_pr_orchestrator.v3.config`) validates the V3
  policy schema with forward-compatible handling of unknown fields.
- **Interfaces** (:mod:`ai_pr_orchestrator.v3.interfaces`) are protocols that
  decouple the policy engine from GitHub, CAO, Hermes, and CI systems. Every
  protocol can be faked in unit tests without shelling out to any external
  tool.
- **Lanes** (:mod:`ai_pr_orchestrator.v3.lanes`) is the registry binding each
  lane to its own isolated agent profile.
- **CAO adapter** (:mod:`ai_pr_orchestrator.v3.cao`) drives agent sessions over
  CAO's HTTP control plane. It never spawns a process itself.
- **Telemetry** (:mod:`ai_pr_orchestrator.v3.telemetry`) reports live quota,
  provider health, and freshness per resource. It is pure domain; missing
  telemetry yields *unknown*, never *zero*.

Only two modules reach outside the process: the CAO adapter, and the Hermes
account-usage probe (:mod:`ai_pr_orchestrator.v3.telemetry_hermes`), which runs
a constant bridge script under Hermes' own interpreter so its pinned dependency
tree and credential side effects stay out of ours. Nothing else here executes
processes, calls providers, or talks to GitHub.

The V1 runtime remains fully functional and untouched until cutover; see
``docs/V3_ARCHITECTURE.md`` for the ownership map and migration table,
``docs/V3_CAO.md`` for the CAO version floor and profile provisioning, and
``docs/V3_TELEMETRY.md`` for the telemetry contract.
"""
