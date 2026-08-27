# aipro V3 on the CAO Control Plane

*Companion to `docs/V3_ARCHITECTURE.md`. Covers the CAO version floor, the
control-plane contract `ai_pr_orchestrator.v3.cao` depends on, and how to
provision the Hermes lane profiles.*

## 1. Minimum CAO version: 2.4.x

**aipro V3 requires CAO >= 2.4.0 (developed against 2.4.1).** 2.4 is the first
line whose HTTP control plane exposes every route the adapter needs, in
particular per-terminal `metadata` (the durable attribution record that makes
restart reconcile possible) and the typed terminal `status` field.

The adapter records the floor as `ai_pr_orchestrator.v3.cao.MINIMUM_CAO_VERSION`.

## 2. What aipro asks of CAO, and what it never does

aipro V3 never spawns, supervises, or tears down an agent process. It asks CAO
to do so over HTTP, and it reads back only typed fields:

| Operation | Route |
| --- | --- |
| Launch a named session in a working directory | `POST /sessions` |
| Reconcile a session by durable name | `GET /sessions/{session_name}` |
| Observe lifecycle, read attribution metadata | `GET /terminals/{terminal_id}` |
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
`session_name_for(run_id, lane)` yields `cao-aipro-{run_id}-{lane}`, sanitized
to tmux's charset and 64-character cap (long names keep a SHA-256 suffix so
they stay collision-resistant). The `cao-` prefix is supplied by aipro because
CAO adds it otherwise, and a lookup under the unprefixed name would miss and
launch a duplicate.

At launch, aipro writes its own attribution record into CAO's per-terminal
`metadata`: lane, profile, workdir, run/round/work-item ids, model assignment,
launch time. That makes CAO the single source of truth for session state — a
restarted aipro that knows only the session *name* can call
`adopt_session(name)` and rebuild everything else. A session carrying no aipro
attribution is refused rather than adopted.

### Never blind-retry a launch

`start_session` distinguishes two transport failures:

- `CaoUnavailableError` — the connection was refused before any bytes landed,
  so nothing happened. Retry is safe.
- `SessionIdentityUncertainError` — the request reached the wire and the answer
  was lost (read timeout, mid-flight drop). CAO may or may not have created the
  session. Retrying risks a second live agent on the same lane and worktree, so
  the error carries `.session_name` and the caller must reconcile with
  `adopt_session()` instead.

The same rule applies to `submit_work`: on `CaoTransportError`, observe the
session before resending, because a resent instruction can double-apply.

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

### Step 1 — one isolated Hermes home per lane

CAO's Hermes provider launches whatever command the profile's `hermesProfile`
field names. Use that indirection to give each lane its own state directory:
create one small wrapper per lane on `PATH`, e.g. `aipro-hermes-developer`:

```sh
#!/bin/sh
export HERMES_HOME="$HOME/.aipro/hermes/developer"
exec hermes "$@"
```

Repeat for `aipro-hermes-requirements-reviewer`,
`aipro-hermes-breaker-reviewer`, and `aipro-hermes-architecture-reviewer`,
each pointing at a distinct directory. (If your Hermes build isolates state
via a different variable, set that one instead — the requirement is only that
no two lanes share a state directory.)

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
