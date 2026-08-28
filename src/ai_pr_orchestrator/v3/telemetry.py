"""Live provider resource telemetry: quota, health, and freshness.

The model broker (#47) has to tell four very different situations apart before
it can route work: cash cost, subscription allowance that expires unused,
a transient rate limit, and a provider that is simply dead. This module is the
normalized vocabulary for that, plus the fan-out that collects it.

Nothing here talks to a provider. Adapters do that (see
:mod:`ai_pr_orchestrator.v3.telemetry_hermes`) and hand back the value types
declared below, so the policy engine never learns a vendor's payload shape.

**The one rule the types enforce.** Missing telemetry means *unknown*, never
*zero*. A provider we could not reach has unknown headroom; treating that as
an exhausted quota would silently drain the candidate pool, and treating it as
a full quota would route work into a black hole. So:

- :class:`ProviderResourceSnapshot` refuses to be constructed as ``exhausted``
  without positive evidence (a spent window, or a zero cash balance);
- ``unavailable`` and ``unknown`` both require a stated ``reason``;
- an unmeasured :class:`QuotaWindow` reports ``used_fraction=None``, and
  ``remaining_fraction`` is likewise ``None`` rather than ``0.0`` or ``1.0``;
- :meth:`ProviderResourceSnapshot.is_stale` returns ``None`` (not ``False``)
  when no TTL was configured, because "we never decided" is not "it is fresh".

**Health is not quota.** Request outcomes accumulate in a
:class:`ProviderHealthLedger` and are attached with
:meth:`ProviderResourceSnapshot.with_health`, which rebuilds the snapshot with
its quota windows untouched. A burst of 429s therefore throttles a resource
(``health.is_throttled``) without ever rewriting what the provider told us
about its allowance — otherwise a rate limit would be indistinguishable from a
spent subscription, and the resource would drop out of rotation until a reset
that was never actually needed.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from statistics import fmean, median
from types import MappingProxyType
from typing import Any, Literal

from ._schema import SchemaError, to_mapping
from .catalog import VALID_RESOURCE_CLASSES, ModelCatalog

# --- Vocabularies ----------------------------------------------------------

#: What the router may conclude about a resource right now.
#:
#: ``available``   we have evidence of headroom;
#: ``exhausted``   we have evidence the allowance is spent — recoverable at
#:                 the window's ``reset_at``;
#: ``unavailable`` unusable for a reason that is *not* quota (no credentials,
#:                 auth rejected, provider not supported). Not time-boxed by a
#:                 quota reset, so waiting will not fix it;
#: ``unknown``     we could not determine the state. Distinct from all three
#:                 above, and the only honest answer after a failed probe.
ResourceAvailability = Literal["available", "exhausted", "unavailable", "unknown"]
VALID_AVAILABILITIES: frozenset[str] = frozenset(
    ("available", "exhausted", "unavailable", "unknown")
)

#: Outcome of one request against a provider, as recorded for health stats.
#: ``transport_error`` is the catch-all for a failure that never produced an
#: HTTP status (DNS, connection reset, an unclassified client exception) —
#: filing those under ``timeout`` or ``server_error`` would attribute a fault
#: to the provider that may well be ours.
RequestOutcome = Literal[
    "success",
    "rate_limited",
    "auth_failure",
    "server_error",
    "timeout",
    "client_error",
    "transport_error",
]
VALID_REQUEST_OUTCOMES: frozenset[str] = frozenset(
    (
        "success",
        "rate_limited",
        "auth_failure",
        "server_error",
        "timeout",
        "client_error",
        "transport_error",
    )
)

#: Default number of recent outcomes kept per resource. Health is meant to
#: describe *now*, so the ledger is a bounded ring rather than a lifetime tally.
DEFAULT_HEALTH_WINDOW = 50


class TelemetryError(SchemaError):
    """Raised when telemetry data is malformed or self-contradictory."""


# --- Redaction -------------------------------------------------------------

_REDACTED = "«REDACTED»"

#: Key names whose value is credential material wherever it appears.
_SECRET_KEYS = (
    r"bearer|(?:api|access|refresh|private|secret|auth)[_-]?(?:key|token)|"
    r"token|secrets?|passwords?|passwd|credentials?"
)

#: Namespace prefixes, which is how these names actually arrive. The vendor
#: form of an API key is ``OPENAI_API_KEY``/``ANTHROPIC_API_KEY``, and OAuth
#: calls its shared secret ``client_secret`` — in every one of those a leading
#: ``\b`` fails, because ``_`` is a word character so there is no boundary
#: before ``API`` or ``secret``. Each prefix segment must end in a separator,
#: so ``notatoken`` is not treated as a namespaced ``token``.
_SECRET_KEY = rf"(?:[A-Za-z0-9]+[_.-])*(?:{_SECRET_KEYS})"

#: Key/value form. The separator must allow punctuation, not just whitespace:
#: a provider that echoes ``?api_key=plainsecret``, ``access_token: abc``, or a
#: JSON body where the key itself is quoted would otherwise sail straight
#: through to an operator's terminal. The key name is kept and only the value
#: masked, so the message stays useful.
_SECRET_KEY_VALUE = re.compile(
    rf"(?i)\b({_SECRET_KEY})\b([\"']?\s*[:=]\s*|\s+)([\"']?)[^\s\"'&,;)\]}}]+"
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Recognizable token shapes, which carry no key name to anchor on.
    re.compile(r"\bsk-[A-Za-z0-9._-]{6,}"),
    re.compile(r"\b(gh[pousr]|github_pat)_[A-Za-z0-9_]{6,}"),
    # userinfo in a URL, e.g. https://user:secret@host/path
    re.compile(r"(?<=//)[^/\s:@]+:[^/\s@]+(?=@)"),
)


def redact_secrets(text: str) -> str:
    """Mask credential material in operator-facing text.

    Snapshots have no field that holds a credential, so the only way one can
    reach an operator's terminal is inside a provider error string that echoed
    a header, a query string, or a URL. This is that last filter, applied where
    such strings enter the telemetry types.
    """
    text = _SECRET_KEY_VALUE.sub(rf"\1\2\3{_REDACTED}", text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


# --- Quota windows ---------------------------------------------------------


@dataclass(frozen=True)
class QuotaWindow:
    """One allowance window on a resource, e.g. "Current week".

    A subscription commonly exposes several of these at once (a rolling
    session window *and* a weekly window, sometimes per-model). They are kept
    separate rather than collapsed to a single percentage because they expire
    at different times: the binding constraint is whichever window is tightest
    *for the work being scheduled*, which only the broker can decide.

    ``reset_at`` is the authoritative form and is always timezone-aware;
    :meth:`time_to_reset` is derived from it on demand rather than stored, so
    a snapshot cached for five minutes cannot report a countdown from when it
    was taken.
    """

    label: str
    #: Fraction of the allowance consumed, ``0.0`` to ``1.0``. ``None`` means the
    #: provider did not say — never assume ``0.0``.
    used_fraction: float | None = None
    reset_at: datetime | None = None
    detail: str = ""
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise TelemetryError("quota window label must be non-empty")
        if self.used_fraction is not None:
            value = self.used_fraction
            # No upper bound: providers legitimately report >100% on a window
            # they let you overshoot. Clamping here would erase real data; the
            # only thing that must not go negative is remaining headroom.
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TelemetryError(
                    f"quota window {self.label!r} used_fraction must be a number, "
                    f"got {type(value).__name__}"
                )
            if not math.isfinite(value) or value < 0:
                raise TelemetryError(
                    f"quota window {self.label!r} used_fraction must be a finite "
                    f"value >= 0, got {value}"
                )
        if self.reset_at is not None:
            _require_aware(self.reset_at, f"quota window {self.label!r} reset_at")
        if self.detail:
            object.__setattr__(self, "detail", redact_secrets(self.detail))

    @property
    def remaining_fraction(self) -> float | None:
        """Headroom left, or ``None`` when usage was never reported."""
        if self.used_fraction is None:
            return None
        return max(0.0, 1.0 - self.used_fraction)

    @property
    def is_exhausted(self) -> bool:
        """True only on positive evidence that the allowance is spent."""
        return self.used_fraction is not None and self.used_fraction >= 1.0

    def time_to_reset(self, at: datetime | None = None) -> timedelta | None:
        """Time until this window rolls over, or ``None`` if it never says.

        Clamped at zero: a reset already in the past means the window has
        rolled and the snapshot is simply behind, not that time is negative.
        """
        if self.reset_at is None:
            return None
        now = at or datetime.now(UTC)
        return max(timedelta(0), self.reset_at - now)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "label": self.label,
            "used_fraction": self.used_fraction,
            "remaining_fraction": self.remaining_fraction,
            "is_exhausted": self.is_exhausted,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
            "detail": self.detail,
        }
        out.update(self.extras)
        return out


def any_window_spent(windows: Sequence[QuotaWindow]) -> bool:
    """True when *any* window the provider measured reports full usage.

    Whether one spent window stops the whole resource depends on what that
    window applies to, and providers do mix the two kinds: Anthropic's OAuth
    usage API returns ``five_hour`` and ``seven_day`` (which constrain every
    request) beside ``seven_day_opus`` and ``seven_day_sonnet`` (which
    constrain one model each). Spending the Opus allowance really does leave
    Sonnet capacity.

    We cannot tell them apart. Hermes maps those API keys onto display labels
    and drops the keys, so a window reaches us as prose — the same structural
    loss as the cash balance in :mod:`~ai_pr_orchestrator.v3.telemetry_hermes`.
    Recovering the distinction would mean matching on rendered text that has
    already changed between Hermes builds.

    So a spent window of unknown scope is assumed to constrain everything. The
    asymmetry is deliberate and matches this module's rule: over-reporting
    capacity sends the broker at a resource that cannot serve it, while
    under-reporting only idles a resource until reset, visibly, with
    :meth:`ProviderResourceSnapshot.spent_windows` naming exactly which window
    is responsible. Fixing this properly needs the applicability upstream, not
    a label heuristic here.

    Unmeasured windows are ignored rather than assumed empty or full.
    """
    return any(w.is_exhausted for w in windows)


# --- Health ----------------------------------------------------------------


@dataclass(frozen=True)
class ProviderHealth:
    """Recent request outcomes for one resource.

    Deliberately separate from quota: this is what *we* observed calling the
    provider, whereas quota is what the provider says about our allowance. The
    two have different provenance and different lifetimes, and conflating them
    is how a transient 429 ends up looking like a spent subscription.
    """

    observed_at: datetime | None = None
    #: Counts keyed by :data:`RequestOutcome`. Absent keys are zero.
    outcomes: Mapping[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    latency_samples_ms: tuple[float, ...] = ()
    #: When a 429 told us to back off until. A throttle, not a quota fact.
    retry_after: datetime | None = None
    last_outcome: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", MappingProxyType(dict(self.outcomes)))
        object.__setattr__(self, "latency_samples_ms", tuple(self.latency_samples_ms))
        unknown = sorted(set(self.outcomes) - VALID_REQUEST_OUTCOMES)
        if unknown:
            raise TelemetryError(
                f"unknown request outcome(s) {unknown}, must be among "
                f"{sorted(VALID_REQUEST_OUTCOMES)}"
            )
        if self.observed_at is not None:
            _require_aware(self.observed_at, "health observed_at")
        if self.retry_after is not None:
            _require_aware(self.retry_after, "health retry_after")

    @property
    def total(self) -> int:
        return sum(self.outcomes.values())

    @property
    def successes(self) -> int:
        return self.outcomes.get("success", 0)

    @property
    def failures(self) -> int:
        return self.total - self.successes

    @property
    def failure_rate(self) -> float | None:
        """Share of recent requests that failed, or ``None`` if we never called.

        ``None`` rather than ``0.0``: an untried provider is not a healthy one.
        """
        if self.total == 0:
            return None
        return self.failures / self.total

    @property
    def mean_latency_ms(self) -> float | None:
        return fmean(self.latency_samples_ms) if self.latency_samples_ms else None

    @property
    def p50_latency_ms(self) -> float | None:
        return median(self.latency_samples_ms) if self.latency_samples_ms else None

    def is_throttled(self, at: datetime | None = None) -> bool:
        """True while a provider-issued back-off is still in force."""
        if self.retry_after is None:
            return False
        return (at or datetime.now(UTC)) < self.retry_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "outcomes": dict(self.outcomes),
            "total": self.total,
            "failure_rate": self.failure_rate,
            "consecutive_failures": self.consecutive_failures,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "retry_after": self.retry_after.isoformat() if self.retry_after else None,
            "last_outcome": self.last_outcome,
        }


class ProviderHealthLedger:
    """Bounded, in-process tally of recent request outcomes per resource.

    Health describes the recent past, so the ledger keeps a fixed-size ring
    per resource and recomputes :class:`ProviderHealth` on read. A provider
    that failed all morning and has been fine since should not stay penalized
    forever, which a lifetime counter would guarantee.

    This is the only mutable object in the telemetry surface, and it writes
    nowhere near quota data — see :meth:`ProviderResourceSnapshot.with_health`.
    """

    def __init__(self, *, window_size: int = DEFAULT_HEALTH_WINDOW) -> None:
        if window_size < 1:
            raise TelemetryError(f"health window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._events: dict[str, deque[tuple[str, float | None]]] = {}
        self._observed_at: dict[str, datetime] = {}
        self._retry_after: dict[str, datetime] = {}

    def record(
        self,
        resource: str,
        outcome: str,
        *,
        at: datetime | None = None,
        latency_ms: float | None = None,
        retry_after: datetime | None = None,
    ) -> None:
        """Record one request outcome against ``resource``."""
        if outcome not in VALID_REQUEST_OUTCOMES:
            raise TelemetryError(
                f"unknown request outcome {outcome!r}, must be one of "
                f"{sorted(VALID_REQUEST_OUTCOMES)}"
            )
        if latency_ms is not None and (not math.isfinite(latency_ms) or latency_ms < 0):
            raise TelemetryError(f"latency_ms must be a finite value >= 0, got {latency_ms}")
        now = at or datetime.now(UTC)
        _require_aware(now, "health outcome timestamp")
        events = self._events.setdefault(resource, deque(maxlen=self._window_size))
        events.append((outcome, latency_ms))
        self._observed_at[resource] = now
        if retry_after is not None:
            _require_aware(retry_after, "retry_after")
            self._retry_after[resource] = retry_after

    def health(self, resource: str) -> ProviderHealth:
        """Current health for ``resource``; empty (not an error) if unseen."""
        events = self._events.get(resource)
        if not events:
            return ProviderHealth()
        counts: dict[str, int] = {}
        for outcome, _ in events:
            counts[outcome] = counts.get(outcome, 0) + 1
        consecutive = 0
        for outcome, _ in reversed(events):
            if outcome == "success":
                break
            consecutive += 1
        return ProviderHealth(
            observed_at=self._observed_at.get(resource),
            outcomes=counts,
            consecutive_failures=consecutive,
            latency_samples_ms=tuple(ms for _, ms in events if ms is not None),
            retry_after=self._retry_after.get(resource),
            last_outcome=events[-1][0],
        )

    def forget(self, resource: str) -> None:
        """Drop all recorded history for ``resource``."""
        self._events.pop(resource, None)
        self._observed_at.pop(resource, None)
        self._retry_after.pop(resource, None)


# --- Snapshot --------------------------------------------------------------


@dataclass(frozen=True)
class ProviderResourceSnapshot:
    """Normalized view of one configured resource at one instant.

    ``expires_at`` is deliberately not a window ``reset_at``: a reset means the
    allowance comes back, an expiry means the resource stops being what it
    claims to be (a promotion ends, a free tier closes). The broker treats
    those oppositely — one is worth waiting for, the other is worth spending
    before it evaporates.
    """

    resource: str
    observed_at: datetime
    availability: str = "unknown"
    resource_class: str = "subscription"
    windows: tuple[QuotaWindow, ...] = ()
    cash_balance: float | None = None
    currency: str = ""
    expires_at: datetime | None = None
    health: ProviderHealth = field(default_factory=ProviderHealth)
    #: Freshness budget in seconds. ``None`` means no TTL was configured, and
    #: :meth:`is_stale` then answers ``None`` rather than pretending it is fresh.
    ttl_seconds: float | None = None
    plan: str = ""
    #: Provenance, e.g. ``hermes:oauth_usage_api``. Operator-facing only.
    source: str = ""
    #: Provider-supplied operator lines, kept verbatim and never parsed for
    #: routing data — that would be scraping rendered text.
    details: tuple[str, ...] = ()
    #: Why the resource is ``unavailable`` or ``unknown``. Mandatory for both.
    reason: str = ""
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resource:
            raise TelemetryError("snapshot resource must be non-empty")
        _require_aware(self.observed_at, f"snapshot {self.resource!r} observed_at")

        object.__setattr__(self, "windows", tuple(self.windows))
        object.__setattr__(self, "details", tuple(redact_secrets(d) for d in self.details))
        if self.reason:
            object.__setattr__(self, "reason", redact_secrets(self.reason))

        if self.availability not in VALID_AVAILABILITIES:
            raise TelemetryError(
                f"snapshot {self.resource!r} has unknown availability "
                f"{self.availability!r}, must be one of {sorted(VALID_AVAILABILITIES)}"
            )
        if self.resource_class not in VALID_RESOURCE_CLASSES:
            raise TelemetryError(
                f"snapshot {self.resource!r} has unknown resource_class "
                f"{self.resource_class!r}, must be one of {sorted(VALID_RESOURCE_CLASSES)}"
            )

        labels = [w.label for w in self.windows]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise TelemetryError(
                f"snapshot {self.resource!r} has duplicate window labels {duplicates}; "
                "each allowance window must be addressable on its own"
            )

        if self.cash_balance is not None and (
            not isinstance(self.cash_balance, (int, float))
            or isinstance(self.cash_balance, bool)
            or not math.isfinite(self.cash_balance)
            or self.cash_balance < 0
        ):
            raise TelemetryError(
                f"snapshot {self.resource!r} cash_balance must be a finite value >= 0, "
                f"got {self.cash_balance!r}"
            )
        if self.ttl_seconds is not None and (
            not math.isfinite(self.ttl_seconds) or self.ttl_seconds <= 0
        ):
            raise TelemetryError(
                f"snapshot {self.resource!r} ttl_seconds must be a finite value > 0, "
                f"got {self.ttl_seconds}"
            )
        if self.expires_at is not None:
            _require_aware(self.expires_at, f"snapshot {self.resource!r} expires_at")

        # A spent window of unknown applicability is assumed to constrain the
        # whole resource; see :func:`any_window_spent` for why we cannot tell
        # a shared window from a model-specific one.
        spent = any_window_spent(self.windows) or self.cash_balance == 0.0
        # The invariant this whole module exists for: a probe that failed can
        # never present as an exhausted quota, because 'exhausted' demands a
        # window the provider itself reported as spent.
        if self.availability == "exhausted" and not spent:
            raise TelemetryError(
                f"snapshot {self.resource!r} claims availability 'exhausted' without "
                "evidence: no quota window reports full usage and there is no zero cash "
                "balance. A failed or empty probe is 'unknown', not 'exhausted'."
            )
        if self.availability == "available" and spent:
            raise TelemetryError(
                f"snapshot {self.resource!r} claims availability 'available' but a quota "
                "window reports full usage; the snapshot contradicts itself"
            )
        if self.availability in ("unavailable", "unknown") and not self.reason:
            raise TelemetryError(
                f"snapshot {self.resource!r} is {self.availability!r} but states no reason; "
                "the router and the operator both need to know which it is and why"
            )

    # --- Queries -----------------------------------------------------------

    def age(self, at: datetime | None = None) -> timedelta:
        return max(timedelta(0), (at or datetime.now(UTC)) - self.observed_at)

    def is_stale(self, at: datetime | None = None) -> bool | None:
        """Whether the snapshot has outlived its TTL, or ``None`` if untimed."""
        if self.ttl_seconds is None:
            return None
        return self.age(at).total_seconds() > self.ttl_seconds

    def window(self, label: str) -> QuotaWindow | None:
        return next((w for w in self.windows if w.label == label), None)

    def tightest_window(self) -> QuotaWindow | None:
        """The measured window with the least headroom, or ``None``.

        Unmeasured windows are skipped rather than treated as empty, so an
        unreported window cannot become the binding constraint.
        """
        measured = [w for w in self.windows if w.used_fraction is not None]
        if not measured:
            return None
        return max(measured, key=lambda w: w.used_fraction or 0.0)

    def next_reset_at(self) -> datetime | None:
        """Earliest absolute reset across all windows, or ``None``."""
        resets = [w.reset_at for w in self.windows if w.reset_at is not None]
        return min(resets) if resets else None

    def spent_windows(self) -> tuple[QuotaWindow, ...]:
        """Windows the provider reports as fully used.

        A resource stays ``available`` while any measured window has headroom,
        so a model-aware caller must check whether the window *it* depends on
        is in here rather than reading availability alone.
        """
        return tuple(w for w in self.windows if w.is_exhausted)

    def with_health(self, health: ProviderHealth) -> ProviderResourceSnapshot:
        """Return a copy carrying ``health``, with quota data untouched.

        Rebuilding rather than mutating is what makes "request outcomes never
        corrupt durable quota data" a property of the type instead of a
        convention callers have to remember.
        """
        return ProviderResourceSnapshot(
            **{f.name: getattr(self, f.name) for f in fields(self) if f.name != "health"},
            health=health,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extras":
                continue
            out[f.name] = to_mapping(getattr(self, f.name))
        # `to_mapping` walks dataclasses field-by-field, which would silently
        # drop the derived values these two publish (remaining_fraction,
        # failure_rate, latency percentiles). The JSON output is the machine
        # contract, so route them through their own serializers.
        out["windows"] = [w.to_dict() for w in self.windows]
        out["health"] = self.health.to_dict()
        tightest = self.tightest_window()
        next_reset = self.next_reset_at()
        out["tightest_window"] = tightest.label if tightest is not None else None
        out["next_reset_at"] = next_reset.isoformat() if next_reset is not None else None
        out["spent_windows"] = [w.label for w in self.spent_windows()]
        out.update(self.extras)
        return out


def unknown_snapshot(
    resource: str,
    *,
    reason: str,
    at: datetime | None = None,
    resource_class: str = "subscription",
    ttl_seconds: float | None = None,
    source: str = "",
) -> ProviderResourceSnapshot:
    """The honest answer when telemetry could not be obtained.

    Every adapter degrades through here rather than inventing a value, which
    is what keeps "probe failed" from ever reaching the broker as "no quota".
    """
    return ProviderResourceSnapshot(
        resource=resource,
        observed_at=at or datetime.now(UTC),
        availability="unknown",
        resource_class=resource_class,
        ttl_seconds=ttl_seconds,
        source=source,
        reason=reason or "telemetry unavailable for an unstated reason",
    )


# --- Sources ---------------------------------------------------------------


class CatalogTelemetrySource:
    """Telemetry for the resources the shared model catalog itself describes.

    Free tiers and promotions are perishable capacity whose availability is a
    declared fact, not a live measurement — so they need no probe, but the
    broker still has to see them through the same interface as a subscription.
    That is what makes this a second real implementation of the source seam
    rather than a hypothetical one.
    """

    def __init__(self, catalog: ModelCatalog, *, ttl_seconds: float | None = None) -> None:
        self._catalog = catalog
        self._ttl_seconds = ttl_seconds

    def resources(self) -> tuple[str, ...]:
        return tuple(
            entry.ref
            for entry in self._catalog.entries
            if entry.resource_class == "free_tier"
            or entry.cost_class == "free"
            or entry.promotional
        )

    def snapshot(self, resource: str, *, at: datetime | None = None) -> ProviderResourceSnapshot:
        now = at or datetime.now(UTC)
        entry = self._catalog.get(resource)
        if entry is None:
            return unknown_snapshot(
                resource,
                reason=f"{resource!r} is not present in the model catalog",
                at=now,
                ttl_seconds=self._ttl_seconds,
                source="catalog",
            )
        details: list[str] = []
        reason = ""
        if not entry.enabled:
            availability = "unavailable"
            reason = f"catalog entry {entry.ref!r} is disabled"
        elif not entry.has_known_price(now):
            # A lapsed promotion with no list price is unusable for the same
            # reason the catalog excludes it: nothing can budget against an
            # unknown price.
            availability = "unavailable"
            reason = (
                f"catalog entry {entry.ref!r} has no determinable price at "
                f"{now.isoformat()} (promotion ended and no list price is declared)"
            )
        else:
            availability = "available"
        if entry.promotional:
            details.append(
                "promotion active" if entry.promotion_active(now) else "promotion inactive"
            )
        # The verdict is computed at `now`, but the *declaration* it rests on is
        # only as fresh as the last time an operator confirmed it. Stamping
        # `now` here made a months-old promotion report age zero and
        # `is_stale() is False` forever — a live measurement's freshness
        # attached to a hand-written file. Where the entry declares no
        # provenance, staleness is unanswerable rather than false, so the
        # snapshot carries no TTL (§3: unknown is not zero).
        return ProviderResourceSnapshot(
            resource=entry.ref,
            observed_at=entry.source_updated_at or now,
            availability=availability,
            resource_class=entry.resource_class,
            # Only a *live* promotion is perishable. Carrying promo_ends_at for
            # one that has not started, or that lapsed back to a list price,
            # would show the broker ordinary paid capacity as something to
            # spend before it evaporates.
            expires_at=entry.promo_ends_at if entry.promotion_active(now) else None,
            ttl_seconds=self._ttl_seconds if entry.source_updated_at is not None else None,
            source="catalog",
            details=tuple(details),
            reason=reason,
        )


class TelemetryRegistry:
    """Fans one query out over every configured telemetry source.

    Answers the diagnostic question "what does the machine have available
    right now" in a single call, which is what the CLI prints and what the
    broker will rank. Sources are asked in registration order and a resource
    may only be claimed once, so the effective telemetry for a resource is
    never ambiguous.
    """

    def __init__(
        self,
        sources: Sequence[Any],
        *,
        ledger: ProviderHealthLedger | None = None,
    ) -> None:
        self._sources = tuple(sources)
        self._ledger = ledger
        owner: dict[str, Any] = {}
        for source in self._sources:
            for resource in source.resources():
                if resource in owner:
                    raise TelemetryError(
                        f"resource {resource!r} is claimed by more than one telemetry "
                        "source; the effective telemetry would be ambiguous"
                    )
                owner[resource] = source
        self._owner = owner

    def resources(self) -> tuple[str, ...]:
        return tuple(self._owner)

    def snapshot(self, resource: str, *, at: datetime | None = None) -> ProviderResourceSnapshot:
        now = at or datetime.now(UTC)
        source = self._owner.get(resource)
        if source is None:
            return self._attach_health(
                unknown_snapshot(
                    resource,
                    reason=f"{resource!r} is not served by any configured telemetry source",
                    at=now,
                )
            )
        try:
            snap = source.snapshot(resource, at=now)
        except Exception as exc:
            # A misbehaving adapter must not take the whole diagnostic down,
            # and must not be able to report anything but 'unknown'.
            snap = unknown_snapshot(
                resource,
                reason=f"telemetry source failed: {type(exc).__name__}: {exc}",
                at=now,
            )
        return self._attach_health(snap)

    def snapshot_all(self, *, at: datetime | None = None) -> tuple[ProviderResourceSnapshot, ...]:
        """Snapshot every configured resource against a single timestamp.

        One clock read for the whole fan-out: reading it per resource would
        let a promotion expire or a window reset mid-scan, so two rows of the
        same listing could disagree about what time it is.
        """
        now = at or datetime.now(UTC)
        return tuple(self.snapshot(resource, at=now) for resource in self._owner)

    def _attach_health(self, snap: ProviderResourceSnapshot) -> ProviderResourceSnapshot:
        if self._ledger is None:
            return snap
        return snap.with_health(self._ledger.health(snap.resource))


# --- Helpers ---------------------------------------------------------------


def _require_aware(value: datetime, where: str) -> None:
    """Reject naive datetimes; every telemetry instant is absolute."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise TelemetryError(
            f"{where} must be timezone-aware: a reset or observation time without an "
            f"offset cannot be compared across providers, got {value.isoformat()}"
        )


def coerce_aware(value: Any, where: str) -> datetime | None:
    """Parse an ISO 8601 string/datetime into an aware UTC datetime.

    Naive input is read as UTC, matching how the model catalog reads its
    timestamps, so an adapter never has to decide what a bare local time meant.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise TelemetryError(f"{where} is not a valid ISO 8601 timestamp: {value!r}") from exc
    if not isinstance(value, datetime):
        raise TelemetryError(f"{where} must be a timestamp, got {type(value).__name__}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value
