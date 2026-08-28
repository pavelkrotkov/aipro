# V3 Provider Telemetry

*Delivered by issue #46. Builds on the shared model catalog (#45). Consumed by
the model broker (#47).*

## 1. The one rule

**Missing telemetry means *unknown*, never *zero*.**

The catalog (#45) says what *could* be used and what it costs. This layer says
what is *actually left* right now. The broker (#47) then chooses. The failure
this layer exists to prevent is the broker reading a failed probe as free
capacity and routing real work at a dead or exhausted account.

So the availability vocabulary has four values, not two:

| Value | Means | Broker should |
| --- | --- | --- |
| `available` | Positive evidence of headroom | Use it |
| `exhausted` | Positive evidence there is none | Skip it until reset |
| `unavailable` | A durable reason it cannot serve (expired credential, disabled entry) | Skip it until fixed |
| `unknown` | We could not find out | Not assume anything |

`unavailable` and `unknown` are separate on purpose. "Your token expired" is
actionable; "the probe timed out" is not. Collapsing them loses the only
signal an operator can act on.

Both non-usable states are **required to carry a `reason`** —
`ProviderResourceSnapshot` rejects construction without one. A snapshot that
says "no" without saying why is a support ticket.

## 2. Health is not quota

A `429` says the provider is refusing you *right now*. It does not say your
weekly allowance is spent. Treating the two as one is how a transient throttle
gets recorded as an exhausted subscription and takes a resource out of
rotation for a week.

So they are different types with different lifetimes:

| | `QuotaWindow` | `ProviderHealth` |
| --- | --- | --- |
| Answers | How much allowance is left, and when does it return? | Is this endpoint answering us? |
| Source | The provider's own usage API | Our own recent request outcomes |
| Lifetime | Hours to days | Seconds to minutes |
| A 429 affects | Nothing | `retry_after`, `is_throttled`, `failure_rate` |

This is enforced, not merely documented. `ProviderResourceSnapshot.__post_init__`
refuses `availability="exhausted"` unless there is positive evidence — a
`cash_balance` of exactly zero, or `any_window_spent()`:

> A failed or empty probe is 'unknown', not 'exhausted'.

`any_window_spent()` is true when **any measured** window is at
`used_fraction >= 1.0`. Unmeasured windows are ignored rather than assumed
either spent or empty — §1 again.

Whether one spent window should stop the whole resource depends on what that
window applies to, and providers mix the two kinds. Anthropic's OAuth usage API
returns `five_hour` and `seven_day`, which constrain every request, beside
`seven_day_opus` and `seven_day_sonnet`, which constrain one model each.
Spending the Opus allowance really does leave Sonnet capacity.

**We cannot tell them apart.** Hermes maps those API keys onto display labels
(`"Current session"`, `"Opus week"`) and drops the keys, so applicability
reaches us only as prose — the same structural loss as the cash balance in §6.
Recovering it would mean matching on rendered text that has already changed
between Hermes builds.

So a spent window of unknown applicability is assumed to constrain everything.
The asymmetry is deliberate and follows §1: over-reporting capacity sends the
broker at a resource that cannot serve it, while under-reporting only idles a
resource until reset — visibly, with `spent_windows()` naming exactly which
window is responsible. Fixing this properly needs the applicability carried
upstream through Hermes, not a label heuristic here.

It symmetrically refuses `available` when the evidence says the resource is
spent, so a snapshot cannot contradict itself in either direction.
`spent_windows()` exposes the individual spent windows for a broker that wants
to defer a lane until the tightest one resets.

## 3. Unknown is not zero, at every level

The rule recurs at each layer, because a `0.0` default anywhere would
reintroduce the bug the type system is meant to close:

- `QuotaWindow.used_fraction` is `None` when unmeasured, and
  `remaining_fraction` is then `None` too — *not* `1.0`.
- `ProviderHealth.failure_rate` is `None` with zero observations — *not* `0.0`,
  which would read as "perfectly healthy" for a resource never contacted.
- `ProviderResourceSnapshot.is_stale()` is `None` when no TTL is configured —
  *not* `False`, which would read as "confirmed fresh".
- `cash_balance` is `None` when Hermes does not expose a number (see §6).

`used_fraction` has a floor of 0 but deliberately **no ceiling**: providers do
report over 100%. That is data, not a schema violation. `remaining_fraction`
clamps at `0.0` so nothing downstream sees negative headroom.

## 4. Shape

```
TelemetryRegistry            fan-out; one timestamp per collection
  ├── HermesTelemetrySource  live probe of subscription/gateway accounts
  └── CatalogTelemetrySource declared perishable capacity from the catalog
        (both write into one shared ProviderHealthLedger)
```

`telemetry.py` is pure domain and imports nothing vendor-specific.
`telemetry_hermes.py` is the only module that knows Hermes exists.

Two source implementations, not one. The catalog source is not a mock: free
tiers and promotions are perishable capacity whose availability is a *declared
fact* rather than a live measurement, so they need no probe — but the broker
must still see them through the same interface as a subscription. That is what
makes the seam real rather than hypothetical, and it is what motivates
`expires_at` (a promotion *ends*, permanently) being distinct from a window's
`reset_at` (an allowance *returns*, periodically).

`expires_at` is set only while the promotion is actually running. A catalog
entry whose promotion window has already closed reports `expires_at=None` with
detail `promotion inactive`, rather than a timestamp in the past that reads as
"expires soon" to anything sorting on that column.

`snapshot_all()` evaluates the whole fan-out at **one timestamp**, so two rows
cannot disagree about whether a window has reset mid-scan.

Every source is wrapped so `snapshot()` is **total**: it never raises. A source
that throws yields an `unknown` snapshot naming the exception. A diagnostic that
crashes on the resource you needed to diagnose is worse than useless.

## 5. Why the Hermes adapter is a subprocess

Hermes has its own virtualenv with exact-pinned dependencies, and
`agent.account_usage` transitively imports its auth, adapter, credential-pool,
and runtime-provider modules — which resolve credentials as an import side
effect. Importing that into the orchestrator's interpreter would couple our
dependency resolution to Hermes' pins and run its credential machinery inside
our process.

So the bridge runs a **constant script** (`BRIDGE_SCRIPT`) under Hermes' own
interpreter, with `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV` scrubbed from the
environment, and provider names passed as `argv` — no shell, so a configured
provider string can never become a command. This is the "narrow pinned local
adapter" the issue sanctions, and it is the reason no credential ever needs to
appear in V3 config.

`HermesSubprocessProbe.probe()` never raises. A missing checkout, a broken
venv, a timeout, or unparseable output all return per-provider error entries,
so a machine without Hermes yields `unknown` telemetry rather than a crash.

The subprocess runs with `cwd` set to `hermes_home` when that directory exists.
`python -c` puts the working directory on `sys.path`, so inheriting the
orchestrator's cwd would let a local file named e.g. `agent.py` shadow Hermes'
own package.

### Our failures are not the provider's failures

A probe can fail for two unrelated reasons, and only one of them is evidence
about the provider:

| Failure | Example | Health ledger |
| --- | --- | --- |
| The provider answered badly | `401`, `429`, `5xx`, per-request timeout | Recorded |
| We never reached the provider | no interpreter, import error, unparseable output, **our own process deadline** | **Not recorded** (`local_error`) |

A `probe_timeout_seconds` expiry is a `local_error`, not a per-provider
timeout. The bridge walks the providers sequentially inside one subprocess, so
a deadline on the whole process cannot say which provider hung: blaming all of
them would charge a timeout to providers that already answered and to providers
never contacted at all.

Recording a local error as a provider failure would drive `failure_rate` and
`consecutive_failures` up for an endpoint that was never contacted, and the
broker would route away from a healthy account because *our* install is broken.
`local_error` kinds map to no `RequestOutcome` at all; the resource is simply
`unknown`.

### Fidelity gates the verdict

When a provider returns no usage payload, what that means depends on which path
produced it. Through the **private** path (`fidelity == "private"`) a bare
`None` is a real negative answer, so the resource is `unavailable`. Through the
**public** fallback it is not: `fetch_account_usage` ends in a blanket
`except Exception: return None`, so `None` there is indistinguishable from a
network blip. That case degrades to `unknown` instead — asserting `unavailable`
from a value that cannot distinguish "expired credential" from "transient
error" would manufacture the exact certainty §1 forbids.

`Retry-After` is parsed as either delta-seconds or an HTTP-date (RFC 9110
permits both); an HTTP-date is converted to seconds from now and floored at 0.
Non-finite delays are dropped at both ends: `float()` accepts `"NaN"` and
`"Infinity"`, `json.dumps` emits those literals verbatim and `json.loads`
accepts them back, and `timedelta()` then raises — turning a rate limit into a
generic source failure.

### The private-function deviation

The bridge calls Hermes' **private** `_fetch_<provider>_account_usage`
functions, falling back to the public `fetch_account_usage` when a private name
is absent.

The public function ends in a blanket `except Exception: return None`, which
collapses every distinct failure into the same answer. Verified live: through
the public path an expired Codex credential reported bare `None`; through the
bridge the same account reported `401 Unauthorized`. That is the difference
between `unknown` and `unavailable` — between "try again later" and "go
re-authenticate", which is precisely the distinction §1 exists to preserve.

The public fallback keeps the adapter working if the private names change; the
cost of the change is reduced fidelity, not a crash.

## 6. What Hermes cannot tell us

**Cash balances are not structurally available.** Both
`_fetch_codex_account_usage` and `_fetch_openrouter_account_usage` compute a
numeric balance and then discard it into a formatted `details` string
(`"Credits balance: $7.19"`).

`cash_balance` therefore stays `None` for every Hermes-sourced resource, and
the `details` strings are preserved verbatim. Parsing the number back out of
rendered text would be exactly the scraping this data path exists to avoid, and
it would silently produce wrong numbers the moment the format changed. The fix
belongs upstream: Hermes exposing the balance as a field.

**Gateway usage has no reset window.** OpenRouter reports spend, not an
allowance that returns, so its snapshot has no windows and `next_reset_at()` is
`None`. Nothing is fabricated to fill the column.

Together those two make OpenRouter `unknown`, not `available`. Hermes' own
`AccountUsageSnapshot.available` is `bool(windows or details)` — it answers "is
there a panel worth rendering?", not "may we dispatch work here?". Copying it
made a reply whose only content was the prose `"Credits balance: $0.00"` report
as `available`. `available` now requires a **measured** window with room in it,
which is the only positive evidence of headroom Hermes gives us structurally.
Unmeasured windows and prose are not headroom.

`observed_at` comes from Hermes' own `fetched_at`, not our clock, so freshness
is measured from when the data was actually read.

## 7. Redaction

Provider error text can quote a request that carried a credential. Every
free-text field (`reason`, `detail`, `details`) is passed through
`redact_secrets()` on the way in — bearer tokens, `sk-`/`gh*_` keys, and
userinfo in URLs become `«REDACTED»`. Snapshots have no field that holds a
credential, so the `telemetry` command cannot print one.

## 8. Configuration

See `examples/v3-telemetry.yml` for a worked sample.

```yaml
telemetry:
  hermes_home: /opt/hermes            # bridge runs <home>/venv/bin/python
  snapshot_ttl_seconds: 300
  health_window_size: 50
  resources:
    - name: anthropic-sub
      provider: anthropic
      resource_class: subscription
    - name: openrouter
      provider: openrouter
      resource_class: metered
      ttl_seconds: 60                 # balances move faster than allowances
```

`name` is the policy-level id the broker and operator use; `provider` is the
upstream key Hermes resolves credentials for. They are separate so renaming a
provider upstream does not rewrite routing decisions.

Two resources may **not** share a `provider`. Hermes resolves credentials per
provider from ambient machine state and has no way to select between two
accounts on one provider, so both rows would report the same allowance twice —
and a broker summing them would see double the capacity that exists. The Hermes
source rejects the duplicate at construction rather than reporting one account's
numbers under another account's name.

`snapshot_ttl_seconds`, `probe_timeout_seconds`, and per-resource `ttl_seconds`
must be finite. YAML's `.nan` and `.inf` otherwise slip past a bare `> 0` check
— `.nan` because every comparison with it is false, `.inf` because it is
genuinely greater than zero — and would make freshness unevaluable.

`hermes_home`/`hermes_python` are intentionally **optional**: a config must
validate identically in CI, where no Hermes install exists. A missing
interpreter degrades every resource to `unknown` at collection time — the
correct answer — rather than failing to load the policy.

`include_catalog_resources` (default `true`) also reports free-tier and
promotional catalog entries. A telemetry resource named the same as one of
those entries is rejected at config load, naming the key at fault: two sources
owning one resource would make its telemetry ambiguous.

## 9. Inspecting it

```console
$ aipro telemetry --config examples/v3-telemetry.yml
RESOURCE             AVAILABILITY  CLASS              AGE  SOURCE
anthropic-sub        available     subscription        0s  hermes:oauth_usage_api
    window Current session    used     37%  resets 2026-08-28T22:59:59+00:00 (in 4h04m)
    window Current week       used     79%  resets 2026-08-29T17:59:59+00:00 (in 23h04m)
    health 1 recent request(s), 0% failed, 0 consecutive
codex-sub            unavailable   subscription        0s  hermes
    health 1 recent request(s), 100% failed, 1 consecutive
    reason Hermes usage probe for 'openai-codex' failed (auth_failure): 401 Unauthorized
openrouter           unknown       metered             0s  hermes:credits_api
    detail Credits balance: $7.16
    health 1 recent request(s), 0% failed, 0 consecutive
    reason Hermes reported account usage for 'openrouter' with no measured quota
           window or structured balance, so remaining capacity cannot be determined
promo-free-generalist available     metered             0s  catalog
    expires 2027-09-30T00:00:00+00:00
    detail promotion active
```

`--json` emits the same data machine-readably, under a single `evaluated_at`.

The command **always exits 0** once the config loads. An exhausted or
unavailable resource is a finding to report, not a failure of the diagnostic.
Only a malformed config is an error.
