# aipro V3 on the CAO Control Plane

*Companion to `docs/V3_ARCHITECTURE.md`. Covers the CAO version floor, the
control-plane contract `ai_pr_orchestrator.v3.cao` depends on, and how to
provision the Hermes lane profiles.*

## 1. Minimum CAO version: 2.5.0

**aipro V3 requires CAO >= 2.5.0.** The earliest verified release is
[official v2.5.0](https://github.com/awslabs/cli-agent-orchestrator/tree/a5ccbe2624aabadfbdf64642c5f1e364db299ec3),
commit `a5ccbe2624aabadfbdf64642c5f1e364db299ec3`. For a reproducible installation:

```sh
uv tool install 'git+https://github.com/awslabs/cli-agent-orchestrator.git@a5ccbe2624aabadfbdf64642c5f1e364db299ec3'
cao-server --host 127.0.0.1 --port 9889 --terminal tmux
```

Manage `cao-server` separately from the foreman, with the same non-root
coordinator user and installed lane wrappers on its `PATH`. Keep its
`CAO_HOME_DIR` stable across restarts so terminal metadata survives.

The previous 2.4.x floor was incorrect: official 2.4.0 and 2.4.1 lack both
creation attribution fields (`group`/`metadata`) and the metadata PATCH route.
Their successful session launch is not evidence that restart recovery works.
The adapter records the corrected floor as
`ai_pr_orchestrator.v3.cao.MINIMUM_CAO_VERSION`; it does not infer a server
version from `/health`. Missing metadata PATCH raises the existing
`CaoSessionNotFoundError`; sessions without attribution cannot be adopted.
Upgrade the control plane rather than retrying a missing capability.

### Verified runtime contract (issue #94)

On 2026-09-19, an isolated installation of the exact 2.5.0 commit above passed
real HTTP/tmux checks using aipro controller revision
`3712d3d5df8d8e726c33e4ff00e236452330bf8c`:

- Creation attribution survived GET; `update_turn_context` persisted round 2.
- PATCH used `{"metadata": {...}}` and replaced the complete metadata record.
- A fresh controller and a restarted CAO server adopted the same session and
  terminal with the updated attribution intact.
- A CAO-launched process received the explicit session environment and read
  all 220,030 bytes of a Unicode issue artifact from linked-worktree-private
  Git metadata, with a matching SHA-256.

The profile command was a harmless local process launched by CAO's Hermes
provider, **not live Hermes reasoning**. This verifies transport, metadata,
restart adoption, environment delivery and filesystem access. It does not
verify model consumption, reviewer reasoning, task completion or #55 cutover.
Owned sessions and the private server were stopped after the check.

The negative regression fixture `tests/fixtures/cao-2.4.1-metadata-contract.json`
was captured from official 2.4.1 commit
`752be53f19c15b328a67a4e64c3d3c87c38b2d23`: its actual FastAPI OpenAPI schema
omits creation metadata and the PATCH route, and an in-process ASGI request
returns the recorded 404. It is not a response invented by FakeCAOServer.
To recapture, install that commit in an isolated environment, import
`cli_agent_orchestrator.api.main.app`, inspect
`app.openapi()["components"]["schemas"]["CreateSessionBody"]["properties"]`
and `app.openapi()["paths"]`, then use `httpx.ASGITransport(app=app)` to PATCH
`/terminals/aipro94-probe/metadata` with `{"metadata":{"round_id":"round2"}}`.
Set `CAO_HOME_DIR` to a scratch directory before importing. No live worker or
provider credentials are needed for this missing-route check.

## 2. What aipro asks of CAO, and what it never does

aipro V3 never spawns, supervises, or tears down an agent process. It asks CAO
to do so over HTTP, and it reads back only typed fields:

| Operation | Route |
| --- | --- |
| Launch a named session in a working directory | `POST /sessions` |
| Reconcile a session by durable name | `GET /sessions/{session_name}` |
| Observe lifecycle, read attribution metadata | `GET /terminals/{terminal_id}` |
| Record activity evidence in durable metadata | `PATCH /terminals/{terminal_id}/metadata` |
| Submit work / follow-up to a live session | `POST /terminals/{terminal_id}/input` |
| Read the agent's final answer | `GET /terminals/{terminal_id}/output?mode=last` |
| Stop and clean up | `DELETE /sessions/{session_name}` |

Two prohibitions are structural, not stylistic:

- **No terminal scraping.** Lifecycle comes from the `status` field
  (`idle`/`processing`/`completed`/`waiting_user_answer`/`error`); the final
  answer comes from CAO's own provider-aware last-response extraction. aipro
  never reads a scrollback, a prompt glyph, or a tmux pane. Anything that would
  require it belongs in CAO, behind a typed field.
- **No model selection.** The adapter sends no `model` and no `provider`
  parameter. The provider is resolved by CAO from the agent profile; the
  model, when a broker leased one, travels in `SessionSpec.env` and is recorded
  in the session's `ModelAssignment` for audit. No vendor or model name appears
  anywhere in `ai_pr_orchestrator/v3/`.

### Configuring the attachment point

```yaml
cao:
  base_url: http://localhost:9889   # CAO_API_PORT default
  request_timeout_seconds: 30.0     # per HTTP call
  session_timeout_seconds: 3600     # how long a session may run
```

`request_timeout_seconds` and `session_timeout_seconds` are deliberately
separate budgets: a session legitimately runs for an hour while every
individual control-plane call is expected to answer in seconds.

## 3. Session identity and restart recovery

A session's name is a pure function of its run and lane —
`session_name_for(run_id, lane)` yields
`cao-aipro-{run_id}-{lane}-{digest}`, sanitized to tmux's charset and
64-character cap. The digest is a SHA-256 over the run id and lane joined by
a delimiter that cannot occur in either, so the run/lane boundary is
unambiguous and two distinct (run, lane) pairs can never produce the same
name — even when the identifiers are built from the same safe character set.
The `cao-` prefix is supplied by aipro because
CAO adds it otherwise, and a lookup under the unprefixed name would miss and
launch a duplicate.

At launch, aipro writes its own attribution record into CAO's per-terminal
`metadata`: lane, profile, workdir, run/round/work-item ids, model assignment,
launch time. That makes CAO the single source of truth for session state — a
restarted aipro that knows only the session *name* can call
`adopt_session(name)` and rebuild everything else. A session carrying no aipro
attribution is refused rather than adopted.

Activity evidence is durable too. The first status response that shows a
post-start state (`processing`/`waiting_user_answer`) is recorded as
`activity_seen: true` in the terminal's metadata via the PATCH route above;
on adoption a restarted process restores that flag and the idle-settle rule
applies as before. Accepted input is deliberately *not* evidence: the control
plane accepts input while a provider is still starting, so only an observed
post-start state may let an idle reading complete a session.

`SessionSpec.image` is rejected by the adapter: CAO's `POST /sessions` launch
parameters expose no image override, so the adapter cannot forward it and
refuses rather than silently running the session somewhere else.

### Never blind-retry a launch

`start_session` distinguishes two transport failures:

- `CaoUnavailableError` — the connection was refused before any bytes landed,
  so nothing happened. Retry is safe.
- `SessionIdentityUncertainError` — the request reached the wire and the answer
  was lost (read timeout, mid-flight drop). CAO may or may not have created the
  session. Retrying risks a second live agent on the same lane and worktree, so
  the error carries `.session_name` and the caller must reconcile with
  `adopt_session()` instead.

Before input, `submit_work` writes the turn context, SHA-256 input fingerprint,
and `activity_seen: false` together. If that write fails, no input is sent.
The fingerprint covers the prompt and full execution context. On restart,
`CaoLaneExecutor.execute` collects a matching turn without resending input;
a different turn requires the previous one to have completed. Transport
failures preserve the worker and workflow state for reconciliation.

A recorded fingerprint proves dispatch intent, not delivery. If a crash
happens between metadata preparation and input, collection cannot fabricate
completion from the previous turn's idle/completed status. It needs new
observed activity, otherwise the existing timeout applies. Do not resend a
possibly delivered turn. A definitive input rejection restores the previous
metadata; a failed restoration requires reconciliation.

Older sessions without an explicit `input_digest` are refused: reconcile or
stop them before upgrading. Do not add a fingerprint to an unknown in-flight
turn to force adoption.

## 4. Provisioning the Hermes lane profiles

V3 runs four lanes, each of which **must own an independent agent profile**.
The lane registry (`ai_pr_orchestrator.v3.lanes`) enforces this: two lanes
sharing a profile would put two concurrent agent processes on one profile state
directory and corrupt it.

| Lane | Role | CAO agent profile |
| --- | --- | --- |
| `developer` | worker | `aipro-developer` |
| `requirements-reviewer` | reviewer | `aipro-requirements-reviewer` |
| `breaker-reviewer` | reviewer | `aipro-breaker-reviewer` |
| `architecture-reviewer` | reviewer | `aipro-architecture-reviewer` |

### Reviewer checkout isolation

Each review wave first commits the developer output and records that immutable
head SHA. The three reviewer sessions then run concurrently, when broker
capacity permits, in separate independent clones detached at that same SHA.
They do not use linked Git worktrees: independent clones avoid the shared Git
metadata and writable refs that would let one reviewer's experimental commit
or ref update contaminate the developer checkout or a sibling reviewer. The
review clones have no source remote, and the lane environment also disables
push credentials. Reviewer assignments and owned checkout paths are persisted
before launch; findings/results are persisted before a successful session is
terminated and its checkout is deleted.

This is repository isolation, not a hostile-process sandbox. A reviewer process
running as the same unrestricted OS user could still walk to another checkout
by absolute path or inspect other same-user processes. Deployments that need
protection against a malicious reviewer must add OS/container isolation around
the CAO/Hermes lane.

### Step 1 — wrap the whole lane process

On a Linux Docker host, install `scripts/aipro-container` on the coordinator's
`PATH`. Pull the official Hermes image once and pin its returned digest in the
lane wrappers; do not resolve a mutable tag at each session launch:

```sh
install -m 755 scripts/aipro-container "$HOME/.local/bin/aipro-container"
docker pull nousresearch/hermes-agent:stable
docker image inspect nousresearch/hermes-agent:stable --format '{{index .RepoDigests 0}}'
```

Create one wrapper per lane, using the digest printed above:

```sh
#!/bin/sh
export HERMES_HOME="$HOME/.aipro/hermes/developer"
export AIPRO_HERMES_IMAGE='nousresearch/hermes-agent@sha256:REPLACE_WITH_VERIFIED_DIGEST'
exec aipro-container hermes "$@" --model "${AIPRO_MODEL:?broker model required}" \
    --provider "${AIPRO_PROVIDER:?broker provider required}"
```

Repeat for the three reviewer wrappers with distinct homes. Create these homes
from scratch, with only the model credentials/configuration each lane needs;
never copy the coordinator's home, GitHub credentials, SSH keys, agent sockets,
Docker socket, or general Hermes profile. Run the coordinator as a non-root user.

Initialize each fresh lane home's first-run choices using the pinned image before
CAO starts its interactive session. For the verified image below, disable shared
metrics through Hermes's own consent function (which also records the current
offer version); setting only `enabled: false` leaves a first-run prompt:

```sh
HERMES_HOME="$HOME/.aipro/hermes/developer" AIPRO_HERMES_IMAGE="$VERIFIED_IMAGE" \
    aipro-container python3 -c \
    'from hermes_cli.observability.shared_metrics_consent import save_consent; save_consent(False, False)'
```

Run this from a Git checkout, and repeat for each distinct lane home.

The wrapper runs the complete Hermes process tree in a separate container,
without host environment inheritance, host PID/network namespaces or capabilities.
Only source and the lane home are writable; the checkout's Git entry and common
Git metadata are read-only. This supports linked developer worktrees and independent
review clones while preventing worker hooks/config/ref/index changes. Worker-side
Git staging/committing is deliberately unavailable; the coordinator owns those steps.
The image root is read-only, with container-private `/tmp` for tests and tools.

Keep CAO and every other unauthenticated coordinator control endpoint bound to
host loopback. The container uses Docker bridge networking for model APIs; it is
not an egress allowlist and must not be given access to an unauthenticated host
control endpoint. Provision required toolchains in the pinned image, not with
privileged host mounts.

Validate the exact deployed image before enabling a lane:

```sh
AIPRO_ISOLATION_IMAGE='nousresearch/hermes-agent@sha256:VERIFIED_DIGEST' \
    uv run pytest tests/integration/test_container_authority.py
```

This check uses invented credential canaries, verifies Git metadata is unwritable,
and starts the real Hermes executable under the same wrapper. It does not by itself
prove CAO turn handling, live model behavior, cold-start composition, or all #52
acceptance criteria; those still require the provisioned CAO/foreman integration.

On 2026-10-10, the native check passed on the Tailscale `hermes` host with
`nousresearch/hermes-agent@sha256:9774f4f39a9bb8c2f68ce728ed5e99ddbad282163be56764afacf88ed952b784`.
A separate temporary CAO 2.5.0 server also launched real Hermes through this wrapper
with invented model credentials. Canaries executed inside that CAO-owned container
confirmed that host credential files, sockets, coordinator process environments,
writable Git metadata and the host-loopback CAO endpoint were unavailable. The
owned session, container and server were removed afterward. No model turn was
submitted; this establishes the deployed launch boundary, not live reasoning,
foreman recovery or the #55 soak.

### Step 2 — one CAO agent profile per lane

Agent profiles are markdown files with YAML frontmatter, read from
`$CAO_HOME_DIR/agent-store/{name}.md` (default
`~/.aws/cli-agent-orchestrator/agent-store/`). Create
`agent-store/aipro-developer.md`:

```markdown
---
name: aipro-developer
description: aipro V3 developer lane
provider: hermes
role: developer
hermesProfile: aipro-hermes-developer
---

You implement the assigned work item in the checked-out worktree.
```

and one file per reviewer lane, with `role: reviewer`, its own
`hermesProfile` wrapper, and the review brief in the body.

**Do not set `model:` in these profiles.** The model is chosen by aipro's
broker and passed per session; a model pinned in the profile would silently
override that decision, and it would put a vendor name in the deployment's
lane definition where aipro cannot audit it.

### Per-session model binding

Construct `CaoLaneExecutor` with the same resolved `ModelCatalog` used by the
broker (`model_router.resolve_catalog(config, base_dir=...)`). The executor
resolves a lease's policy `model_ref` to its catalog `descriptor` and `provider`;
these become `AIPRO_MODEL` and `AIPRO_PROVIDER` in the CAO session environment.
Optional constructor `env` defaults are copied and preserved, with the selected
model/provider taking precedence. Coordinator process environment is never copied.

The container lane wrapper translates those variables to Hermes's
supported `chat --model` and `--provider` arguments. The native
`scripts/aipro-hermes` bridge supplies the same translation for control-plane
smoke tests but does not provide process isolation. Hermes does not natively
consume `AIPRO_MODEL`, `LLM_MODEL`, or `OPENAI_MODEL`: merely setting a variable
without translating it to CLI arguments does not select the model. The lane wrapper
continues to own `HERMES_HOME`; do not pin `model:` in the CAO profile.

Leased aliases absent from the supplied catalog, missing explicit providers, and
custom `endpoint` entries fail before session launch. This bridge cannot bind a
custom endpoint; configure a supported Hermes provider rather than silently
billing a default endpoint. No lease leaves the supplied environment unchanged.
Adoption retains the existing process's model; a different model assignment is
rejected by the controller, not applied retroactively. Keep the catalog binding
for a running assignment stable throughout its session.

The launch arguments are verified against
[CAO's Hermes provider](https://github.com/awslabs/cli-agent-orchestrator/blob/0ef89f358fa2b475b0025ccb86f8ad20bbadca38/src/cli_agent_orchestrator/providers/hermes.py)
and [Hermes's chat parser](https://github.com/NousResearch/hermes-agent/blob/03fee43ca344ead7245a3b0ae20d38de0ae75642/hermes_cli/_parser.py).

### Step 3 — verify

```sh
curl -s localhost:9889/agents/profiles/aipro-developer
```

Each of the four names must resolve. A missing profile surfaces at launch as a
CAO `ProviderError`, not as a config error, so check them up front.

### Overriding the lane set

A deployment that needs different lanes writes the `hermes_lanes` config
section; `LaneRegistry.from_config` uses it verbatim and still enforces unique
names and unique profiles. An empty section falls back to the four lanes above.

## 5. Testing against a real CAO

The unit suite (`tests/unit/test_v3_cao.py`) fakes the HTTP transport with
`respx` and needs no CAO. `tests/integration/test_v3_cao_local.py` drives a
real local CAO and is skipped unless you opt in:

```sh
AIPO_CAO_INTEGRATION=1 uv run pytest tests/integration/test_v3_cao_local.py
```

It additionally skips when the `cao` binary is absent or no control plane
answers, and it needs the `aipro-developer` profile from step 2. Override the
endpoint with `AIPO_CAO_BASE_URL`.

## Reviewer result contract

On a successful lane with `role == "reviewer"`, `poll_session` parses the
normalized `LaneResult.output_summary` (CAO's extracted last response). It never
parses terminal scrollback. The response must be a JSON array of new
`ReviewerFinding` objects; `[]` explicitly reports no findings. Plain text,
empty output, Markdown fences, malformed JSON and invalid fields raise
`CaoReviewerOutputError`, so the foreman escalates instead of opening a PR.
Worker-lane text and unsuccessful lane results are not parsed as reviews.

Each object requires `id`, `body` and `severity` (`info`, `minor`, `major` or
`blocker`). Location, confidence, claim, evidence, falsification and reproduction
fields use the existing finding schema and its constructor validation. For example:

```json
[{"id":"missing-guard","body":"Missing permission check","severity":"major","path":"src/service.py","line":12}]
```

The controller supplies omitted `lane`, `run_id` and `round_id` from the confirmed
current session turn. Supplied attribution must match. Findings must be `open`:
reviewers cannot supply settled dispositions, existing thread IDs, conflict groups
or merged provenance. `created_at` must be omitted: the domain constructor assigns
the engine timestamp used for canonical finding selection. Those fields belong
to durable policy processing. IDs must be
unique within a response. Primitive types are checked before domain construction;
booleans are not line numbers, and non-JSON NaN/Infinity values are rejected.
Legacy durable-state decoding remains unchanged and is not used for this ingress.

`head_sha`, when supplied, is retained as finding evidence; this parser cannot
verify it because session execution context currently has no authoritative reviewed
head. Head validation remains a separate policy/execution-context requirement.
