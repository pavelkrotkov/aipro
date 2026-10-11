# aipro V3 architecture

V3 is the supported issue-to-PR runtime. Hermes supervises the persistent
service; its deterministic aipro loop schedules work and enforces policy.
CAO 2.5 owns contained Hermes worker lifecycle. GitHub is the authoritative
business-state store; process memory and local telemetry are disposable.

| Component | Responsibility |
| --- | --- |
| `v3.scheduler`, `v3.foreman` | One developer issue, parallel blind review, remediation/adjudication, limits and recovery |
| `v3.queue`, `github/` | Authorized issue opt-in, trusted state writer, leases, serialized history and GitHub mutations |
| `v3.catalog`, `v3.model_router`, `v3.broker` | Shared metadata, dynamic model/lineage choice, quota reserves, capacity and fallback policy |
| `v3.telemetry`, `v3.telemetry_hermes` | Typed resource state and Hermes account-usage bridge; missing evidence is unknown |
| `v3.cao`, `v3.cao_lane`, Hermes profiles | HTTP session attribution/turns, coding/review reasoning, native provider fallback |
| `v3.git_ops`, `v3.ci_gate` | Coordinator-only commit/push/PR authority and exact-head CI evidence |
| `v3.cleanup`, `v3.reconcile`, `v3.status`, `v3.audit` | Verified owned cleanup, recovery plans and operator diagnostics |
| `v3.domain`, `v3.findings`, `v3.config` | Serializable types, deterministic findings and validated policy |

The coordinator performs native Git operations and calls GitHub/CAO APIs.
The account-usage adapter runs a fixed bridge under Hermes' interpreter.
Agent processes are launched and stopped through CAO, never direct vendor CLIs
inside aipro. Lane wrappers contain the whole Hermes process tree and use
independent homes; reviewer laboratories are detached clones of one immutable
head. The coordinator owns Git metadata, credentials and authoritative mutations.

An issue is claimed before resource creation. Dispatch intent, reservations,
model assignment, immutable review inputs and results persist in trusted GitHub
state. Review clearance creates or updates a draft PR before CI evaluation;
only green required checks at the expected head permit readiness/done. Unknown
side-effect outcomes retain capacity and require reconciliation. Cold recovery
adopts existing turns and artifacts rather than replaying ambiguous submissions.

One native coordinator lock per target checkout supplements GitHub leases.
GitHub comment updates use best-effort timestamp preconditions and readback;
there is no transactional compare-and-swap API. Keep one deployment per repository.
Protected Git allocation records prove artifact ownership and never replace
GitHub business state. Cleanup requires fresh authority and native lane absence;
active or human resources cannot be reclaimed merely because a TTL elapsed.

V1's runner, provider adapters, domain/state/config and unused helpers are
retired. `v3.git_ops`, V3 finding logic and CAO lanes replace their responsibilities;
shared GitHub and logging utilities remain. No compatibility loader or direct
provider execution path survives. See [migration and rollback](V1_MIGRATION.md),
[the foreman runbook](V3_FOREMAN.md) and [acceptance evidence](V3_SOAK.md).

### Complete issue input for coding and review

The foreman fetches the authoritative issue body for both lane roles; retrieval
failures stop the run rather than substitute empty requirements. Bodies up to
8,000 characters are included verbatim. Larger bodies are linked in the prompt
as a complete UTF-8 file with byte length and SHA-256, without sentence truncation
or omission of trailing acceptance criteria.

`GitWorktreeOps` caches that input in the linked worktree's private Git metadata,
using a content-addressed filename. It cannot enter `git add -A` or a commit;
retained worktrees retain the input, and normal worktree removal deletes it.
The cache is disposable input, never authoritative workflow state. Updated issue
bodies receive distinct files so an existing prompt's input is not overwritten.

CAO/Hermes must share the worktree and its Git metadata filesystem with the
foreman. Profiles must enable file/terminal tools and read the complete input,
using pagination or terminal byte-range reads for oversized lines. An inaccessible
or mismatched file must be reported as a failure, never a clean review. A remote
or container backend mounting only tracked worktree content does not satisfy
this contract; it must expose the linked Git metadata too. No GitHub credentials
are needed in the worker to read these inputs.

## Review-loop terminal states and bounded spending

Review clearance hands work to the CI/PR gate (`ci_gating`); a pending check
parks the work item in `queued` and is **not** a terminal result. The actual
terminal phases are `done` (CI green or reviewed, explicit no-change),
`escalated` (needs human), `failed`, and `cancelled`. Cancel is a deliberate
`GitHubIssueQueue.cancel(..., reason=...)` transition: it is durable, removes
lifecycle labels, preserves finding/disposition history, and cannot be
reclaimed or reopened. Stop owned CAO sessions before cancelling; checkout
cleanup is deferred rather than deleting files under a possibly live agent.

`safety.max_run_spend_usd` is an optional **hard admission ceiling**, not an
invoice or estimated-usage report. When configured, each developer and reviewer
dispatch must carry a broker-verified maximum incremental cash charge; that
ceiling is reserved in the GitHub state **before** launching the agent, and
replays cannot reset the ledger. Any unknown charge, missing provenance, or
exhausted ceiling escalates without dispatch. The current production broker
can prove a zero ceiling only for permanently free models; it deliberately
does not assume metered, promotional, or prepaid subscription capacity has a
hard per-session dollar maximum. Consequently a configured USD ceiling rejects
those sessions until their execution layer can supply an enforceable ceiling.
With no USD ceiling configured, existing model-broker quota and prompt/round/
invocation limits continue to apply unchanged.
