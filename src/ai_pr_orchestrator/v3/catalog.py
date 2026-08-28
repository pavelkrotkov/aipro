"""Shared model catalog: the machine-level source of truth about candidates.

One catalog describes every model/resource the developer, reviewer, and
adjudicator lanes may run on, so vendor and model choices evolve by editing
one file instead of every CAO profile and role prompt. Nothing here selects a
model — that is the broker's job. This module only answers two questions:

- *what candidates exist, and what is true about them* (schema + validation);
- *which of them are usable for this role, at this difficulty, right now*
  (:meth:`ModelCatalog.eligible`).

Deliberately excluded, because they belong to the broker rather than the
catalog: live quota, provider health, cross-lane diversity, and the shadow
value of perishable subscription capacity. The catalog reports declared cash
price and resource class; it never ranks.

**Reload semantics.** Entries and catalogs are frozen. Reloading a file
produces a *new* :class:`ModelCatalog`, leaving any catalog already handed to
a running phase untouched — so a catalog edit affects future assignments only
and can never re-point a session that is already executing.

**Not Hermes fallback config.** A catalog entry is policy metadata: what a
candidate costs, what it is good at, when its promotion expires. The ordered
chain Hermes walks when a provider returns 429 or dies mid-session is
operational configuration and lives with Hermes. The broker (#47) derives a
fallback chain from this metadata; the catalog does not store one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ._schema import SchemaError, build_dataclass, to_mapping
from .domain import VALID_LANE_ROLES

# --- Vocabularies ----------------------------------------------------------

#: How capacity on this resource is paid for. This is orthogonal to price:
#: ``subscription`` capacity has a declared list price but is already bought,
#: which is what makes it perishable (see #47).
VALID_RESOURCE_CLASSES: frozenset[str] = frozenset(("subscription", "metered", "free_tier"))

#: Coarse cash-cost bucket, used for filtering and operator display when
#: exact per-token pricing is unknown or volatile.
VALID_COST_CLASSES: frozenset[str] = frozenset(("free", "low", "medium", "high"))

#: Capability flags. ``tools`` means the model can drive tool calls at all;
#: ``coding`` means it is fit to author changes; ``long_context`` and
#: ``vision`` are declared for completeness.
VALID_CAPABILITIES: frozenset[str] = frozenset(
    ("tools", "coding", "reasoning", "vision", "long_context")
)

#: Task difficulty is a small closed integer scale so config stays comparable
#: across entries. 1 is trivial, 5 is the hardest work the system attempts.
MIN_TASK_DIFFICULTY = 1
MAX_TASK_DIFFICULTY = 5

#: Quality tiers use the same closed scale as difficulty so the broker can
#: compare "how good is this model for this role" against "how hard is this
#: task" without a mapping table.
MIN_QUALITY = 1
MAX_QUALITY = 5


class ModelCatalogError(SchemaError):
    """Raised when the model catalog is malformed or self-contradictory."""


# --- Entry -----------------------------------------------------------------


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One candidate model/resource.

    ``ref`` is the policy-level key that the rest of V3 uses
    (:data:`~ai_pr_orchestrator.v3.domain.ModelRef`); ``descriptor`` is an
    opaque provider-owned string that only the broker/execution adapter
    interprets. V3 core never parses ``descriptor``, which is what keeps
    vendor and model names out of routing logic.

    Entries are deeply immutable: the collection fields are normalized to
    tuples and mapping proxies at construction. A shallow ``frozen=True``
    would only stop rebinding, leaving a caller free to append a role to an
    entry already handed to a running phase — which would change its
    eligibility mid-flight and skip the invariants below entirely.
    """

    ref: str
    descriptor: str
    #: Hermes provider id. Free-form on purpose: adding a provider must not
    #: require changing this module.
    provider: str = ""
    #: Custom/gateway base URL when the provider is not reached at its
    #: default endpoint (e.g. an OpenAI-compatible aggregator).
    endpoint: str | None = None
    resource_class: str = "metered"
    cost_class: str = "medium"
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    #: True when this entry is currently offered below its list price (or
    #: free). The window bounds it; an unbounded promotion is open-ended.
    promotional: bool = False
    promo_starts_at: datetime | None = None
    promo_ends_at: datetime | None = None
    capabilities: tuple[str, ...] = ()
    #: Lane roles this entry may serve. Empty means "suitable for any role".
    roles: tuple[str, ...] = ()
    #: Difficulty floor: do not spend this candidate on work easier than
    #: this. Lets an expensive/scarce entry be reserved for hard tasks
    #: without encoding that rule in the broker.
    min_task_difficulty: int = MIN_TASK_DIFFICULTY
    #: Lineage, used by the broker's adversarial-diversity penalty. Two
    #: entries sharing a family are not independent reviewers.
    family: str = ""
    vendor: str = ""
    max_context_tokens: int | None = None
    enabled: bool = True
    #: Free-form data/training-policy constraint (e.g. "no-training").
    data_policy: str = ""
    #: Manual quality tier per lane role. A role present here must also be
    #: listed in ``roles``.
    quality_by_role: Mapping[str, int] = field(default_factory=dict)
    max_concurrency: int | None = None
    notes: str = ""
    #: When the volatile fields (pricing, promotion) were last confirmed.
    #: Promotions expire quietly; this is how an operator spots stale data.
    source_updated_at: datetime | None = None
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ref:
            raise ModelCatalogError("catalog entry ref must be non-empty")
        if not self.descriptor:
            raise ModelCatalogError(f"catalog entry {self.ref!r} descriptor must be non-empty")

        # Normalize the collection fields to immutable containers before any
        # invariant runs, so what is validated is what the entry will hold.
        for name in ("capabilities", "roles"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "quality_by_role", MappingProxyType(dict(self.quality_by_role)))

        for name in ("promo_starts_at", "promo_ends_at", "source_updated_at"):
            object.__setattr__(self, name, _coerce_dt(getattr(self, name), self.ref, name))

        if self.resource_class not in VALID_RESOURCE_CLASSES:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} has unknown resource_class "
                f"{self.resource_class!r}, must be one of {sorted(VALID_RESOURCE_CLASSES)}"
            )
        if self.cost_class not in VALID_COST_CLASSES:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} has unknown cost_class "
                f"{self.cost_class!r}, must be one of {sorted(VALID_COST_CLASSES)}"
            )

        for name in ("input_price_per_mtok", "output_price_per_mtok"):
            price = getattr(self, name)
            # NaN/infinity would pass a bare ``< 0`` test, then propagate into
            # budget comparisons that silently do the wrong thing and into
            # CLI JSON output that is not valid JSON.
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ModelCatalogError(
                    f"catalog entry {self.ref!r} {name} must be a finite value >= 0, got {price}"
                )

        # A declared price contradicts a free cost class / free tier. Left
        # unchecked, the broker would treat the entry as free while the
        # invoice says otherwise.
        if self.cost_class == "free" or self.resource_class == "free_tier":
            for name in ("input_price_per_mtok", "output_price_per_mtok"):
                price = getattr(self, name)
                if price:
                    raise ModelCatalogError(
                        f"catalog entry {self.ref!r} declares cost_class="
                        f"{self.cost_class!r}/resource_class={self.resource_class!r} "
                        f"but a non-zero {name} ({price})"
                    )

        if (
            self.promo_starts_at is not None
            and self.promo_ends_at is not None
            and self.promo_ends_at <= self.promo_starts_at
        ):
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} promo_ends_at ({self.promo_ends_at.isoformat()}) "
                f"must be after promo_starts_at ({self.promo_starts_at.isoformat()})"
            )
        if not self.promotional and (
            self.promo_starts_at is not None or self.promo_ends_at is not None
        ):
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} declares a promotion window but promotional is false"
            )
        # "Temporarily free" and "permanently free" are different claims, and
        # the permanent one wins in pricing: cost_class 'free'/free_tier price
        # at zero whether or not the promotion is still running. An entry
        # asserting both looks time-boxed but would in fact stay free and
        # eligible forever once its window closed.
        if self.promotional and (self.cost_class == "free" or self.resource_class == "free_tier"):
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} is promotional but also classified permanently "
                f"free (cost_class={self.cost_class!r}, resource_class={self.resource_class!r}); "
                "its promotion could never expire. Drop the promotion, or declare the class "
                "the entry reverts to when the window closes."
            )

        unknown_caps = sorted(set(self.capabilities) - VALID_CAPABILITIES)
        if unknown_caps:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} has unknown capabilities {unknown_caps}, "
                f"must be among {sorted(VALID_CAPABILITIES)}"
            )
        # Every lane in this system reaches the repository through tools, so a
        # candidate that claims it can author changes but cannot call tools
        # could never actually serve a lane.
        if "coding" in self.capabilities and "tools" not in self.capabilities:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} declares the 'coding' capability without "
                "'tools': lanes author changes through tool calls, so this combination "
                "cannot execute"
            )

        unknown_roles = sorted(set(self.roles) - VALID_LANE_ROLES)
        if unknown_roles:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} lists unknown roles {unknown_roles}, "
                f"must be among {sorted(VALID_LANE_ROLES)}"
            )

        if not MIN_TASK_DIFFICULTY <= self.min_task_difficulty <= MAX_TASK_DIFFICULTY:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} min_task_difficulty must be within "
                f"{MIN_TASK_DIFFICULTY}..{MAX_TASK_DIFFICULTY}, got {self.min_task_difficulty}"
            )

        for role, quality in self.quality_by_role.items():
            if role not in VALID_LANE_ROLES:
                raise ModelCatalogError(
                    f"catalog entry {self.ref!r} scores unknown role {role!r} in quality_by_role"
                )
            if self.roles and role not in self.roles:
                raise ModelCatalogError(
                    f"catalog entry {self.ref!r} scores role {role!r} in quality_by_role "
                    f"but does not list it in roles ({sorted(self.roles)})"
                )
            if not MIN_QUALITY <= quality <= MAX_QUALITY:
                raise ModelCatalogError(
                    f"catalog entry {self.ref!r} quality for role {role!r} must be within "
                    f"{MIN_QUALITY}..{MAX_QUALITY}, got {quality}"
                )

        if self.max_context_tokens is not None and self.max_context_tokens < 1:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} max_context_tokens must be >= 1, "
                f"got {self.max_context_tokens}"
            )
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ModelCatalogError(
                f"catalog entry {self.ref!r} max_concurrency must be >= 1, "
                f"got {self.max_concurrency}"
            )

    # --- Queries -----------------------------------------------------------

    def promotion_active(self, at: datetime | None = None) -> bool:
        """True when a promotion is in force at ``at`` (default: now)."""
        if not self.promotional:
            return False
        now = at or datetime.now(UTC)
        if self.promo_starts_at is not None and now < self.promo_starts_at:
            return False
        return not (self.promo_ends_at is not None and now >= self.promo_ends_at)

    def has_known_price(self, at: datetime | None = None) -> bool:
        """True when the cash cost of this entry at ``at`` is determinable."""
        if self.promotion_active(at) or self.cost_class == "free":
            return True
        if self.resource_class == "free_tier":
            return True
        return self.input_price_per_mtok is not None and self.output_price_per_mtok is not None

    def effective_prices(self, at: datetime | None = None) -> tuple[float, float] | None:
        """Normalized cash price per Mtok at ``at`` as ``(input, output)``.

        An active promotion or a free cost class/resource class prices at
        zero. Otherwise the declared list price applies, and ``None`` means
        the price is simply unknown — which the broker must treat as a
        distinct case from "free", never as zero.

        Subscription entries report their declared *list* price here. Their
        marginal cash cost is arguably zero because the allowance is already
        bought, but that is an economic judgement about perishable capacity
        and belongs to the broker (#47), not to the catalog.
        """
        if self.promotion_active(at) or self.cost_class == "free":
            return (0.0, 0.0)
        if self.resource_class == "free_tier":
            return (0.0, 0.0)
        if self.input_price_per_mtok is None or self.output_price_per_mtok is None:
            return None
        return (self.input_price_per_mtok, self.output_price_per_mtok)

    def quality_for(self, role: str) -> int | None:
        """Declared quality tier for ``role``, or ``None`` if unscored."""
        return self.quality_by_role.get(role)

    def is_eligible(
        self,
        *,
        role: str | None = None,
        difficulty: int = MIN_TASK_DIFFICULTY,
        at: datetime | None = None,
    ) -> bool:
        """True when this entry may be dispatched for ``role``/``difficulty``.

        Eligibility is only about what the catalog itself knows. Quota,
        provider health, and diversity are the broker's filters, applied on
        top of this set.

        A promotional entry outside its window is *not* automatically
        ineligible: if it declares a list price it simply reverts to it. It
        drops out only when the expiry leaves its cost unknown, because
        dispatching at an unknown price is what the reserve/budget policy
        cannot reason about.
        """
        if not self.enabled:
            return False
        if role is not None and self.roles and role not in self.roles:
            return False
        if difficulty < self.min_task_difficulty:
            return False
        return self.has_known_price(at)

    # --- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extras":
                continue
            out[f.name] = to_mapping(getattr(self, f.name))
        out.update(self.extras)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCatalogEntry:
        return build_dataclass(cls, data, ModelCatalogError)


# --- Catalog ---------------------------------------------------------------


@dataclass(frozen=True)
class ModelCatalog:
    """An immutable set of catalog entries keyed by ``ref``."""

    entries: tuple[ModelCatalogEntry, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.entries, list):
            object.__setattr__(self, "entries", tuple(self.entries))
        refs = [entry.ref for entry in self.entries]
        duplicates = sorted({ref for ref in refs if refs.count(ref) > 1})
        if duplicates:
            raise ModelCatalogError(f"Duplicate model catalog refs: {duplicates}")

    def refs(self) -> list[str]:
        return [entry.ref for entry in self.entries]

    def get(self, ref: str) -> ModelCatalogEntry | None:
        return next((entry for entry in self.entries if entry.ref == ref), None)

    def eligible(
        self,
        *,
        role: str | None = None,
        difficulty: int = MIN_TASK_DIFFICULTY,
        at: datetime | None = None,
    ) -> list[ModelCatalogEntry]:
        """Entries dispatchable for ``role`` at ``difficulty``, in file order.

        File order is preserved rather than ranked: ranking needs live quota
        and health, which the catalog does not have.
        """
        if role is not None and role not in VALID_LANE_ROLES:
            raise ModelCatalogError(
                f"unknown role {role!r}, must be one of {sorted(VALID_LANE_ROLES)}"
            )
        if not MIN_TASK_DIFFICULTY <= difficulty <= MAX_TASK_DIFFICULTY:
            raise ModelCatalogError(
                f"difficulty must be within {MIN_TASK_DIFFICULTY}..{MAX_TASK_DIFFICULTY}, "
                f"got {difficulty}"
            )
        return [
            entry
            for entry in self.entries
            if entry.is_eligible(role=role, difficulty=difficulty, at=at)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"models": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCatalog:
        if not isinstance(data, dict):
            raise ModelCatalogError(
                f"model catalog must be a mapping at the top level, got {type(data).__name__}"
            )
        raw = data.get("models", [])
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ModelCatalogError(
                f"model catalog 'models' must be a list, got {type(raw).__name__}"
            )
        unknown = sorted(set(data) - {"models"})
        if unknown:
            raise ModelCatalogError(f"unknown top-level keys in model catalog: {unknown}")
        return cls(entries=tuple(ModelCatalogEntry.from_dict(item) for item in raw))


def load_model_catalog(path: str | Path) -> ModelCatalog:
    """Load and validate a shared model catalog from a YAML file.

    Returns a fresh immutable catalog on every call, so an operator may edit
    the file and reload without disturbing a catalog already in use by a
    running phase.
    """
    catalog_path = Path(path)
    try:
        content = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelCatalogError(f"Failed to read model catalog {catalog_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ModelCatalogError(f"Invalid YAML in model catalog {catalog_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ModelCatalogError(
            f"Model catalog {catalog_path} must be a YAML mapping at the top level"
        )
    return ModelCatalog.from_dict(raw)


def _coerce_dt(value: Any, ref: str, field_name: str) -> datetime | None:
    """Normalize a YAML timestamp/ISO string to an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ModelCatalogError(
                f"catalog entry {ref!r} {field_name} is not a valid ISO 8601 timestamp: {value!r}"
            ) from exc
    if not isinstance(value, datetime):
        raise ModelCatalogError(
            f"catalog entry {ref!r} {field_name} must be a timestamp, got {type(value).__name__}"
        )
    # Naive timestamps are read as UTC so promotion windows compare against
    # an aware "now" without raising.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value
