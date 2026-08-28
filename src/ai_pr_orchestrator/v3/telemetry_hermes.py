"""Hermes account-usage adapter: the subscription-quota telemetry source.

Hermes already fetches and normalizes subscription usage for the providers V3
runs on, and renders it through ``/usage``. This module consumes that
*structured* data — never the rendered text — and translates it into the
provider-independent types in :mod:`ai_pr_orchestrator.v3.telemetry`.

**Why a subprocess.** Hermes lives in its own virtualenv with exact-pinned
dependencies, and ``agent.account_usage`` transitively imports its auth,
credential-pool and runtime-provider modules. Importing it into aipro's
interpreter would mean adopting Hermes' entire dependency closure and its
credential-resolution side effects. So the adapter runs a small fixed script
under *Hermes' own* interpreter and reads JSON back: the "narrow pinned local
adapter" the issue calls for. Providers are passed as argv to a constant
script — there is no shell and no interpolation.

**Why the script calls the private per-provider functions.** Hermes' public
``fetch_account_usage`` ends in a blanket ``except Exception: return None``, so
every distinct failure — expired credentials, a 429 on the usage endpoint, a
DNS failure, an unsupported provider — arrives as the same ``None``. That is
precisely the collapse this issue forbids: the router must tell *unavailable*
from *unknown*. Calling ``_fetch_<provider>_account_usage`` lets the exception
reach the bridge, which classifies it by HTTP status and reports it. This was
verified against a live install: the public path reported ``None`` for Codex,
while the bridge reported the actual ``401 Unauthorized``. The bridge falls
back to the public function when a private one is absent, so a Hermes refactor
degrades fidelity instead of breaking the probe.

**What Hermes cannot give us.** Cash balances are computed and then formatted
into human-facing ``details`` strings (``"Credits balance: $7.19"``); the
numbers are not exposed structurally. ``cash_balance`` therefore stays ``None``
for every Hermes-sourced resource. Parsing it back out of the rendered string
would be exactly the scraping this data path exists to avoid, and recovering
it properly needs an upstream change that surfaces the number.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .catalog import VALID_RESOURCE_CLASSES, ModelCatalog
from .config import TelemetryConfig
from .telemetry import (
    CatalogTelemetrySource,
    ProviderHealthLedger,
    ProviderResourceSnapshot,
    QuotaWindow,
    TelemetryError,
    TelemetryRegistry,
    any_window_spent,
    coerce_aware,
    redact_secrets,
    unknown_snapshot,
)

#: Default freshness budget for a Hermes-sourced snapshot. Usage endpoints are
#: rate-limited and the underlying numbers move slowly, so re-probing per
#: routing decision would cost far more than the precision it buys.
DEFAULT_TELEMETRY_TTL_SECONDS = 300.0

#: How the bridge's failure classification maps onto a health outcome.
#: ``local_error`` is deliberately absent: see :data:`_LOCAL_FAILURE_KINDS`.
_KIND_TO_OUTCOME: Mapping[str, str] = {
    "auth_failure": "auth_failure",
    "rate_limited": "rate_limited",
    "server_error": "server_error",
    "timeout": "timeout",
    "client_error": "client_error",
    "error": "transport_error",
}

#: Failures that happened on our side of the wire — no interpreter, the bridge
#: could not import Hermes, the subprocess died, the output was unparseable.
#: No request reached the provider, so recording these as health would give a
#: never-contacted provider a 100% failure rate and slander it for our own
#: misconfiguration. They still degrade telemetry to ``unknown``.
_LOCAL_FAILURE_KINDS: frozenset[str] = frozenset(("local_error",))

#: Failure kinds that tell us the resource is genuinely unusable rather than
#: merely unmeasured. An auth failure is durable and needs a human; everything
#: else may well be gone by the next probe, so it degrades to ``unknown``.
_UNAVAILABLE_KINDS: frozenset[str] = frozenset(("auth_failure",))


BRIDGE_SCRIPT = '''\
"""Emit Hermes account-usage snapshots as JSON. Run under Hermes' interpreter."""
import json
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def _iso(value):
    return value.isoformat() if value is not None else None


def _status_of(exc):
    return getattr(getattr(exc, "response", None), "status_code", None)


def _classify(exc):
    status = _status_of(exc)
    if status in (401, 403):
        return "auth_failure"
    if status == 429:
        return "rate_limited"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    if "Timeout" in type(exc).__name__:
        return "timeout"
    if isinstance(status, int):
        return "client_error"
    return "error"


def _retry_after(exc):
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        # float() happily accepts "NaN" and "Infinity", and json.dumps emits
        # them verbatim, so such a header would survive the bridge and blow up
        # in timedelta() on our side -- turning a rate limit into a generic
        # source failure.
        if seconds != seconds or seconds in (float("inf"), float("-inf")):
            return None
        return max(0.0, seconds)
    # RFC 9110 allows an HTTP-date instead of delay-seconds. Dropping that
    # form would leave is_throttled() false while the provider is still
    # refusing us, so the broker would retry straight into the backoff.
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


PRIVATE = {
    "openai-codex": "_fetch_codex_account_usage",
    "anthropic": "_fetch_anthropic_account_usage",
    "openrouter": "_fetch_openrouter_account_usage",
}


def probe(provider):
    try:
        from agent import account_usage as mod
    except Exception as exc:
        # Local: we never reached the provider, so this must not be filed as
        # provider health.
        return {"ok": False, "provider": provider, "kind": "local_error", "message": repr(exc)}

    key = provider.strip().lower()
    fn = getattr(mod, PRIVATE.get(key, ""), None)
    try:
        if fn is None:
            # No private entry point on this Hermes build: fall back to the
            # public API, accepting that it cannot report a failure reason.
            snapshot = mod.fetch_account_usage(provider)
            fidelity = "public"
        elif key == "openrouter":
            snapshot = fn(None, None)
            fidelity = "private"
        else:
            snapshot = fn()
            fidelity = "private"
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "kind": _classify(exc),
            "message": "{0}: {1}".format(type(exc).__name__, exc),
            "status_code": _status_of(exc),
            "retry_after_seconds": _retry_after(exc),
        }
    if snapshot is None:
        return {"ok": True, "provider": provider, "fidelity": fidelity, "snapshot": None}
    # Inside the per-provider boundary on purpose. Hermes owns this shape and
    # can rename a field; reading it outside would let one changed provider
    # abort the whole results list, so a healthy account's telemetry would be
    # thrown away along with the broken one's. Local, not provider health:
    # the request succeeded, it is our projection of it that failed.
    try:
        return {
            "ok": True,
            "provider": provider,
            "fidelity": fidelity,
            "snapshot": {
                "provider": snapshot.provider,
                "source": snapshot.source,
                "fetched_at": _iso(snapshot.fetched_at),
                "plan": snapshot.plan,
                "windows": [
                    {
                        "label": w.label,
                        "used_percent": w.used_percent,
                        "reset_at": _iso(w.reset_at),
                        "detail": w.detail,
                    }
                    for w in snapshot.windows
                ],
                "details": list(snapshot.details),
                "unavailable_reason": snapshot.unavailable_reason,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "kind": "local_error",
            "message": "unreadable account usage shape: {0}: {1}".format(
                type(exc).__name__, exc
            ),
        }


print(json.dumps({"results": [probe(p) for p in sys.argv[1:]]}))
'''


# --- Configuration ---------------------------------------------------------


@dataclass(frozen=True)
class HermesResource:
    """One configured subscription resource served by the Hermes data path.

    ``name`` is the policy-level id the broker and the operator use;
    ``provider`` is Hermes' own provider key. Keeping them separate means
    renaming a provider upstream does not rewrite every routing decision.

    ``provider`` is canonicalized to match what :data:`BRIDGE_SCRIPT` does with
    it (``strip().lower()``), so the identity we deduplicate on, the argv we
    send, and the key we read the result back under are all the same string.

    Two resources may not share a ``provider``; :class:`HermesTelemetrySource`
    rejects that, because Hermes cannot select between two accounts on one
    provider.
    """

    name: str
    provider: str
    resource_class: str = "subscription"
    #: Overrides the source-level TTL for this resource only.
    ttl_seconds: float | None = None
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", self.provider.strip().lower())
        if not self.name:
            raise TelemetryError("telemetry resource name must be non-empty")
        if not self.provider:
            raise TelemetryError(f"telemetry resource {self.name!r} provider must be non-empty")
        if self.resource_class not in VALID_RESOURCE_CLASSES:
            raise TelemetryError(
                f"telemetry resource {self.name!r} has unknown resource_class "
                f"{self.resource_class!r}, must be one of {sorted(VALID_RESOURCE_CLASSES)}"
            )
        if self.ttl_seconds is not None and (
            not math.isfinite(self.ttl_seconds) or self.ttl_seconds <= 0
        ):
            raise TelemetryError(
                f"telemetry resource {self.name!r} ttl_seconds must be a finite number > 0, "
                f"got {self.ttl_seconds}"
            )


@runtime_checkable
class AccountUsageProbe(Protocol):
    """Fetches raw Hermes account-usage results for a set of providers.

    The seam that keeps tests off the subprocess: normalization is pure and is
    exercised against recorded payloads, while the one implementation that
    spawns an interpreter has almost no logic left to get wrong.
    """

    def probe(self, providers: Sequence[str]) -> Mapping[str, dict[str, Any]]: ...


class HermesSubprocessProbe:
    """Runs :data:`BRIDGE_SCRIPT` under Hermes' own interpreter.

    Never raises: a missing checkout, a broken venv, a timeout, or unparseable
    output all come back as per-provider error entries, so a machine without
    Hermes yields *unknown* telemetry rather than a crashed diagnostic.
    """

    def __init__(
        self,
        *,
        hermes_home: str | Path | None = None,
        python_executable: str | Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        # Absolute from the start. We hand the interpreter *and* a different
        # cwd to `subprocess.run`, and the OS resolves a relative executable
        # against the new cwd — so a relative `hermes_home` would pass the
        # `exists()` preflight here and then fail to launch over there.
        #
        # Lexically absolute, not `resolve()`d: a venv's `bin/python` is a
        # symlink to the base interpreter, and following it lands outside the
        # venv, where `sys.prefix` no longer points at the venv's
        # site-packages. Verified against the real install — Hermes then fails
        # to import `httpx`.
        self._hermes_home = Path(os.path.abspath(hermes_home)) if hermes_home else None
        self._python_executable = (
            Path(os.path.abspath(python_executable)) if python_executable else None
        )
        self._timeout_seconds = timeout_seconds

    def interpreter(self) -> Path | None:
        """The interpreter to run the bridge under, if one can be located."""
        if self._python_executable is not None:
            return self._python_executable
        if self._hermes_home is not None:
            return self._hermes_home / "venv" / "bin" / "python"
        return None

    def _working_directory(self) -> str | None:
        """The Hermes checkout, when we know it and it exists."""
        if self._hermes_home is not None and self._hermes_home.is_dir():
            return str(self._hermes_home)
        return None

    def probe(self, providers: Sequence[str]) -> Mapping[str, dict[str, Any]]:
        interpreter = self.interpreter()
        if interpreter is None:
            return self._all_failed(
                providers,
                "no Hermes interpreter configured: set telemetry.hermes_home or "
                "telemetry.hermes_python",
            )
        if not interpreter.exists():
            return self._all_failed(providers, f"Hermes interpreter {interpreter} does not exist")
        working_directory = self._working_directory()
        try:
            # No shell: a constant script plus provider names as argv, so a
            # configured provider string cannot become a command.
            #
            # cwd is the Hermes checkout, not ours. `python -c` puts the
            # working directory on sys.path, so inheriting aipro's would both
            # miss `agent` in a checkout that was never installed into the venv
            # and let any stray directory named `agent` shadow the real one.
            # When we have no checkout to point at, there is nothing to inherit
            # safely either, so the path entry is dropped instead.
            completed = subprocess.run(
                [str(interpreter), "-c", BRIDGE_SCRIPT, *providers],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                env=_bridge_env(isolate_sys_path=working_directory is None),
                cwd=working_directory,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Deliberately a local_error, not a per-provider timeout. The
            # bridge walks the providers sequentially inside one subprocess, so
            # a deadline on the whole process cannot say which provider hung:
            # blaming all of them would charge a timeout to providers that
            # already answered and to providers never contacted at all.
            return self._all_failed(
                providers, f"Hermes usage probe timed out after {self._timeout_seconds}s"
            )
        except OSError as exc:
            return self._all_failed(providers, f"could not run the Hermes usage probe: {exc}")
        if completed.returncode != 0:
            return self._all_failed(
                providers,
                f"Hermes usage probe exited {completed.returncode}: "
                f"{redact_secrets(completed.stderr.strip())[-500:]}",
            )
        try:
            payload = json.loads(completed.stdout)
            results = payload["results"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return self._all_failed(
                providers, f"Hermes usage probe returned unreadable output: {exc}"
            )
        # Structurally valid JSON can still be the wrong shape — `{"results":
        # null}` from a half-written bridge. Iterating it outside this check
        # raises past the local-error path, and the caller's blanket handler
        # then files our own corrupt output as a provider transport_error.
        if not isinstance(results, list):
            return self._all_failed(
                providers,
                "Hermes usage probe returned unreadable output: 'results' must be a list, "
                f"got {type(results).__name__}",
            )
        return {
            str(result.get("provider")): result
            for result in results
            if isinstance(result, dict) and result.get("provider")
        }

    @staticmethod
    def _all_failed(
        providers: Sequence[str], message: str, *, kind: str = "local_error"
    ) -> dict[str, dict[str, Any]]:
        # Defaults to ``local_error``: everything this helper reports went
        # wrong before or around the request, not inside it.
        return {
            provider: {"ok": False, "provider": provider, "kind": kind, "message": message}
            for provider in providers
        }


def _bridge_env(*, isolate_sys_path: bool) -> dict[str, str]:
    """Environment for the bridge, scrubbed of our own interpreter's paths.

    ``PYTHONPATH``/``PYTHONHOME``/``VIRTUAL_ENV`` inherited from aipro's venv
    would let our dependencies shadow Hermes' pinned ones inside its own
    interpreter, which is the failure this adapter exists to avoid.

    ``isolate_sys_path`` covers the case where we know the interpreter but not
    the checkout, so there is no Hermes directory to point ``cwd`` at. The
    subprocess would then inherit *ours*, and ``python -c`` puts the working
    directory on ``sys.path`` — any ``agent.py`` lying in it shadows Hermes'
    real package. ``PYTHONSAFEPATH`` (3.11+, and ignored by older interpreters)
    drops that entry, leaving the interpreter's own environment to supply
    ``agent``. It is *not* set when we do have a checkout, because there the
    entry is load-bearing: a checkout never installed into its venv is
    importable only via ``cwd``.
    """
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    if isolate_sys_path:
        env["PYTHONSAFEPATH"] = "1"
    else:
        # Set, not merely left alone. A machine that exports PYTHONSAFEPATH
        # globally would otherwise drop the cwd we deliberately chose, and a
        # checkout never installed into its venv is importable only from there
        # -- so the configured hermes_home would fail to import `agent` at all.
        env.pop("PYTHONSAFEPATH", None)
    return env


# --- Normalization (pure) --------------------------------------------------


def normalize_probe_result(
    result: Mapping[str, Any] | None,
    *,
    resource: HermesResource,
    at: datetime,
    ttl_seconds: float | None = None,
) -> ProviderResourceSnapshot:
    """Translate one raw bridge result into a normalized snapshot.

    Pure and total: any shape it cannot make sense of becomes an ``unknown``
    snapshot with a stated reason, because the one thing a telemetry layer
    must never do is turn a parsing problem into a quota claim.
    """
    ttl = resource.ttl_seconds if resource.ttl_seconds is not None else ttl_seconds
    degrade = _unknown_for(resource, at=at, ttl=ttl)

    if result is None:
        return degrade(f"Hermes reported nothing for provider {resource.provider!r}")
    if not result.get("ok", False):
        return _normalize_failure(result, resource=resource, at=at, ttl=ttl)

    raw = result.get("snapshot")
    if raw is None:
        if str(result.get("fidelity") or "") != "private":
            # The public entry point buries every failure in a blanket
            # ``except Exception: return None``, so its None is ambiguous: a
            # timeout and a missing credential are indistinguishable. Calling
            # that 'unavailable' would turn a transient blip into a durable
            # verdict and suppress the resource indefinitely.
            return degrade(
                f"Hermes returned no account-usage data for {resource.provider!r} via its "
                "public entry point, which reports every failure identically; the cause "
                "cannot be determined"
            )
        # Through the private entry point a failure raises, so reaching here
        # with no snapshot means Hermes genuinely has nothing for this
        # provider: no credentials configured, or not supported. Both are
        # durable conditions a human has to fix.
        return ProviderResourceSnapshot(
            resource=resource.name,
            observed_at=at,
            availability="unavailable",
            resource_class=resource.resource_class,
            ttl_seconds=ttl,
            source="hermes",
            reason=(
                f"Hermes has no account-usage data for provider {resource.provider!r}: "
                "either no credentials are configured for it or Hermes does not support it"
            ),
        )

    try:
        return _normalize_snapshot(raw, resource=resource, at=at, ttl=ttl)
    except (TelemetryError, TypeError, ValueError, AttributeError, KeyError) as exc:
        return degrade(
            f"Hermes account-usage payload for {resource.provider!r} could not be "
            f"normalized: {type(exc).__name__}: {exc}"
        )


def _normalize_snapshot(
    raw: Mapping[str, Any],
    *,
    resource: HermesResource,
    at: datetime,
    ttl: float | None,
) -> ProviderResourceSnapshot:
    windows = tuple(_normalize_window(w) for w in raw.get("windows") or ())
    details = tuple(str(d) for d in raw.get("details") or ())
    unavailable_reason = raw.get("unavailable_reason") or ""
    # Freshness is measured from when Hermes actually read the provider, not
    # from when we happened to ask, or a cached snapshot would look newer than
    # its contents every time it is served.
    observed_at = coerce_aware(raw.get("fetched_at"), "account usage fetched_at") or at

    measured = [w for w in windows if w.used_fraction is not None]

    if unavailable_reason:
        availability, reason = "unavailable", str(unavailable_reason)
    elif any_window_spent(windows):
        availability, reason = "exhausted", ""
    elif measured:
        # Positive evidence, and the only kind Hermes gives us structurally: a
        # window the provider measured that still has room in it.
        availability, reason = "available", ""
    elif windows or details:
        # Hermes' own `AccountUsageSnapshot.available` is `bool(windows or
        # details)`, which answers "is there a panel worth rendering?" — not
        # "may we dispatch work here?". Copying it made an OpenRouter reply
        # whose unparsed text reads "Credits balance: $0.00" report as
        # `available`. Unmeasured windows and prose are not headroom.
        availability, reason = (
            "unknown",
            f"Hermes reported account usage for {resource.provider!r} with no measured "
            "quota window or structured balance, so remaining capacity cannot be "
            "determined; only unstructured details were returned",
        )
    else:
        availability, reason = (
            "unknown",
            f"Hermes returned an empty account-usage snapshot for {resource.provider!r}",
        )

    source = raw.get("source")
    return ProviderResourceSnapshot(
        resource=resource.name,
        observed_at=observed_at,
        availability=availability,
        resource_class=resource.resource_class,
        windows=windows,
        ttl_seconds=ttl,
        plan=str(raw.get("plan") or ""),
        source=f"hermes:{source}" if source else "hermes",
        details=details,
        reason=reason,
    )


def _normalize_window(raw: Mapping[str, Any]) -> QuotaWindow:
    used_percent = raw.get("used_percent")
    used_fraction = None
    if isinstance(used_percent, (int, float)) and not isinstance(used_percent, bool):
        used_fraction = float(used_percent) / 100.0
    try:
        reset_at = coerce_aware(raw.get("reset_at"), "quota window reset_at")
    except TelemetryError:
        # A reset we cannot parse is an unknown reset. Dropping the window
        # entirely would lose the usage figure, which is the more valuable half.
        reset_at = None
    return QuotaWindow(
        label=str(raw["label"]),
        used_fraction=used_fraction,
        reset_at=reset_at,
        detail=str(raw.get("detail") or ""),
    )


def _normalize_failure(
    result: Mapping[str, Any],
    *,
    resource: HermesResource,
    at: datetime,
    ttl: float | None,
) -> ProviderResourceSnapshot:
    kind = str(result.get("kind") or "error")
    message = str(result.get("message") or "no detail reported")
    reason = f"Hermes usage probe for {resource.provider!r} failed ({kind}): {message}"
    availability = "unavailable" if kind in _UNAVAILABLE_KINDS else "unknown"
    return ProviderResourceSnapshot(
        resource=resource.name,
        observed_at=at,
        availability=availability,
        resource_class=resource.resource_class,
        ttl_seconds=ttl,
        source="hermes",
        reason=reason,
    )


def _unknown_for(resource: HermesResource, *, at: datetime, ttl: float | None):
    def degrade(reason: str) -> ProviderResourceSnapshot:
        return unknown_snapshot(
            resource.name,
            reason=reason,
            at=at,
            resource_class=resource.resource_class,
            ttl_seconds=ttl,
            source="hermes",
        )

    return degrade


def _earliest_spent_reset(
    results: Iterable[Mapping[str, Any]], *, after: datetime
) -> datetime | None:
    """When the first spent window in these results comes back, if it says.

    Read from the raw payload rather than a normalized snapshot because the
    cache is keyed by provider and normalization needs a resource. Only spent
    windows count: a window with headroom left does not make a cached reading
    wrong when it rolls over.

    Only resets strictly after ``after`` qualify. A provider lagging its own
    rollover, or a little clock skew, can report a spent window whose reset has
    already passed; arming the cache on that would make a freshly stored
    reading expire on arrival, and one ``snapshot_all()`` over N resources
    would fire N probes instead of one. A past reset that the provider is still
    reporting as spent is not evidence a re-probe would say anything different,
    so the TTL governs.
    """
    resets: list[datetime] = []
    for result in results:
        snapshot = result.get("snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        windows = snapshot.get("windows")
        if not isinstance(windows, list):
            continue
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            used = window.get("used_percent")
            if not isinstance(used, (int, float)) or used < 100.0:
                continue
            try:
                reset = coerce_aware(window.get("reset_at"), "account usage reset_at")
            except TelemetryError:
                continue
            if reset is not None and reset > after:
                resets.append(reset)
    return min(resets) if resets else None


def probe_outcome(result: Mapping[str, Any] | None) -> tuple[str, float | None] | None:
    """The health outcome implied by a probe result, if a request was made.

    Returns ``None`` when nothing was actually requested (an unsupported or
    unconfigured provider), so "we never called it" does not get filed as a
    provider fault.
    """
    if result is None:
        return None
    if result.get("ok", False):
        if result.get("snapshot") is None:
            return None
        return ("success", None)
    kind = str(result.get("kind") or "error")
    if kind in _LOCAL_FAILURE_KINDS:
        return None
    retry_after = result.get("retry_after_seconds")
    # `json.loads` accepts the NaN/Infinity literals `json.dumps` emits, so the
    # finiteness check belongs here too: this is where the value becomes a
    # timedelta, and a bridge from a different build is not ours to trust.
    seconds = (
        float(retry_after)
        if isinstance(retry_after, (int, float)) and math.isfinite(retry_after)
        else None
    )
    return (_KIND_TO_OUTCOME.get(kind, "transport_error"), seconds)


# --- Source ----------------------------------------------------------------


class HermesTelemetrySource:
    """Telemetry for every configured Hermes-backed subscription resource.

    One probe serves the whole configured set, independent of whichever
    provider the current Hermes session happens to be using — a session-scoped
    view cannot answer "what does this machine have available", which is the
    question the broker and the diagnostic both ask.
    """

    def __init__(
        self,
        resources: Sequence[HermesResource],
        *,
        probe: AccountUsageProbe,
        ttl_seconds: float | None = DEFAULT_TELEMETRY_TTL_SECONDS,
        ledger: ProviderHealthLedger | None = None,
    ) -> None:
        self._resources: dict[str, HermesResource] = {}
        by_provider: dict[str, str] = {}
        for resource in resources:
            if resource.name in self._resources:
                raise TelemetryError(
                    f"duplicate telemetry resource name {resource.name!r}; "
                    "resource names must be unique"
                )
            # Hermes' account-usage API is keyed by provider and resolves that
            # provider's credentials globally, so it cannot distinguish two
            # accounts on one provider. Serving both from a single result would
            # report one account's allowance for the other — and the broker
            # would route work at an exhausted account on the strength of it.
            # Refusing is the only honest option until Hermes takes an account
            # identity.
            if resource.provider in by_provider:
                raise TelemetryError(
                    f"telemetry resources {by_provider[resource.provider]!r} and "
                    f"{resource.name!r} both use provider {resource.provider!r}, but "
                    "Hermes resolves credentials per provider and cannot tell two "
                    "accounts apart; both rows would report the same account"
                )
            by_provider[resource.provider] = resource.name
            self._resources[resource.name] = resource
        self._probe = probe
        self._ttl_seconds = ttl_seconds
        self._ledger = ledger
        self._cache: Mapping[str, dict[str, Any]] = {}
        self._probed_at: datetime | None = None
        self._expires_at: datetime | None = None

    def resources(self) -> tuple[str, ...]:
        return tuple(self._resources)

    def snapshot(self, resource: str, *, at: datetime | None = None) -> ProviderResourceSnapshot:
        now = at or datetime.now(UTC)
        declared = self._resources.get(resource)
        if declared is None:
            return unknown_snapshot(
                resource,
                reason=f"{resource!r} is not a configured Hermes telemetry resource",
                at=now,
                source="hermes",
            )
        results = self._results(now)
        return normalize_probe_result(
            results.get(declared.provider),
            resource=declared,
            at=now,
            ttl_seconds=self._ttl_seconds,
        )

    def _results(self, now: datetime) -> Mapping[str, dict[str, Any]]:
        """Return probe results, refreshing them when the cache has aged out.

        Cached across resources so a diagnostic listing every subscription
        costs one subprocess and one round of provider calls, not one per
        resource. The refresh interval is the *strictest* configured TTL, so
        no resource is ever served data older than it asked for.

        A spent window's reset also expires the cache, ahead of the TTL. The
        provider tells us exactly when the allowance returns, and holding a
        100%-used reading past that moment keeps a resource out of rotation
        for the remainder of the TTL after it became usable again — five
        minutes, by default, for a window that reset in thirty seconds.
        """
        if (
            self._probed_at is not None
            and now - self._probed_at <= self._refresh_interval()
            and not (self._expires_at is not None and now >= self._expires_at)
        ):
            return self._cache
        providers = tuple(dict.fromkeys(r.provider for r in self._resources.values()))
        started = time.monotonic()
        try:
            results = dict(self._probe.probe(providers))
        except Exception as exc:
            # local_error, not "error". The probe raising means we never
            # established that any provider misbehaved -- the failure is on our
            # side of the adapter. Filing it as transport health would charge a
            # failure to every configured resource for a bug in this process.
            results = {
                provider: {
                    "ok": False,
                    "provider": provider,
                    "kind": "local_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
                for provider in providers
            }
        elapsed = timedelta(seconds=max(0.0, time.monotonic() - started))
        self._cache = results
        self._probed_at = now
        self._expires_at = _earliest_spent_reset(results.values(), after=now)
        self._record_health(results, now, elapsed=elapsed)
        return results

    def _refresh_interval(self) -> timedelta:
        ttls = [
            r.ttl_seconds if r.ttl_seconds is not None else self._ttl_seconds
            for r in self._resources.values()
        ]
        known = [ttl for ttl in ttls if ttl is not None]
        return timedelta(seconds=min(known)) if known else timedelta(0)

    def _record_health(
        self,
        results: Mapping[str, dict[str, Any]],
        now: datetime,
        *,
        elapsed: timedelta = timedelta(0),
    ) -> None:
        """Feed probe outcomes into the health ledger.

        The usage endpoint is a real request against the provider, so its
        outcome is genuine health evidence. It is recorded against the health
        ledger only — quota windows are rebuilt from the payload and are never
        touched by an outcome.

        ``Retry-After`` counts from when the provider answered, but ``now`` was
        read before the subprocess even started, and the bridge walks providers
        sequentially. On a slow probe the whole delay could elapse in transit,
        leaving ``is_throttled()`` false against a provider still refusing us.
        The probe's own duration is therefore added back. That over-waits by
        the gap between the rate-limited reply and the subprocess exiting,
        which is the safe direction to be wrong; being exact would mean the
        bridge reporting an absolute deadline.
        """
        if self._ledger is None:
            return
        for resource in self._resources.values():
            outcome = probe_outcome(results.get(resource.provider))
            if outcome is None:
                continue
            name, retry_after_seconds = outcome
            self._ledger.record(
                resource.name,
                name,
                at=now,
                retry_after=(
                    now + elapsed + timedelta(seconds=retry_after_seconds)
                    if retry_after_seconds is not None
                    else None
                ),
            )


def build_telemetry(
    telemetry: TelemetryConfig,
    *,
    catalog: ModelCatalog | None = None,
) -> tuple[TelemetryRegistry, ProviderHealthLedger]:
    """Assemble the configured telemetry sources into one registry.

    Returns the ledger alongside the registry because the two share it: the
    Hermes source writes probe outcomes into it, and the registry reads it back
    when attaching health to each snapshot.
    """
    ledger = ProviderHealthLedger(window_size=telemetry.health_window_size)
    sources: list[Any] = []
    if telemetry.resources:
        sources.append(
            HermesTelemetrySource(
                resources=[
                    HermesResource(
                        name=resource.name,
                        provider=resource.provider,
                        resource_class=resource.resource_class,
                        ttl_seconds=resource.ttl_seconds,
                    )
                    for resource in telemetry.resources
                ],
                probe=HermesSubprocessProbe(
                    hermes_home=telemetry.hermes_home,
                    python_executable=telemetry.hermes_python,
                    timeout_seconds=telemetry.probe_timeout_seconds,
                ),
                ttl_seconds=telemetry.snapshot_ttl_seconds,
                ledger=ledger,
            )
        )
    if catalog is not None and telemetry.include_catalog_resources:
        # Deliberately untimed. `snapshot_ttl_seconds` budgets how long a *probe*
        # may be served before it is re-measured; a catalog entry is a declaration
        # that no probe will ever refresh, so that budget can only ever be blown.
        # Handing it over marked every provenanced entry permanently stale — noise,
        # not signal. The snapshot's `age` still reports the declaration's real
        # age; staleness stays unanswerable rather than being answered wrongly.
        sources.append(CatalogTelemetrySource(catalog))
    return TelemetryRegistry(sources=sources, ledger=ledger), ledger
