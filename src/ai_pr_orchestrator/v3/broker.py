"""Model broker: which candidate should run this unit of work, and why.

The catalog says what exists and what is true about it; telemetry says what is
live right now. Neither ranks. This module is the ranking, and it is the only
place in V3 that makes an economic judgement.

**Every rejection is named.** "No eligible model" with no cause is
unactionable, so a candidate that is filtered out is returned alongside the
ones that survived, carrying the reason it lost. That is why
:class:`BrokerDecision` publishes ``rejected`` as well as ``ranked``: the
dry-run command prints both, and an operator fixes the config from the reason
rather than by bisecting the catalog.

**Unknown is not favourable.** Inherited from
:mod:`~ai_pr_orchestrator.v3.telemetry`: an entry unscored for the role does
not get the benefit of the doubt, and a resource whose telemetry probe failed
is penalized as unmeasured rather than being scored as if it had headroom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from types import MappingProxyType
from typing import Any

from ._schema import SchemaError
from .catalog import (
    MAX_QUALITY,
    MAX_TASK_DIFFICULTY,
    MIN_QUALITY,
    MIN_TASK_DIFFICULTY,
    ModelCatalog,
    ModelCatalogEntry,
)
from .domain import (
    VALID_LANE_ROLES,
    LaneName,
    ModelAssignment,
    ModelLease,
    ModelRef,
)
from .telemetry import ProviderResourceSnapshot


class BrokerError(SchemaError):
    """Raised when a broker request is malformed."""


def _utc(dt: datetime) -> datetime:
    """Return ``dt`` made timezone-aware (UTC). Naive input is assumed UTC.

    Normalizing at the API boundary keeps every decision record absolute and
    comparable: a naive ``at`` would otherwise blow up later against aware
    promotion/reset timestamps or publish a decision with no offset at all.
    """

    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


#: A resource may legitimately have no telemetry at all — pay-as-you-go
#: capacity has no allowance to probe — so every lookup is optional.
Snapshot = ProviderResourceSnapshot | None


@dataclass(frozen=True)
class _WindowView:
    """One measured quota window with its configured reserve applied."""

    label: str
    reserve: float
    remaining: float
    available: float
    hours_to_reset: float | None


def _binding(views: Sequence[_WindowView]) -> _WindowView | None:
    """The window with the least headroom left to spend.

    All windows are evaluated together and the tightest one governs immediate
    dispatch, because a roomy weekly allowance cannot be drawn through a
    session window that is already at its reserve.
    """
    return min(views, key=lambda v: v.available) if views else None


# --- Demand ----------------------------------------------------------------


@dataclass(frozen=True)
class TaskDemand:
    """One unit of work to place on a model.

    Carries everything the decision depends on, so the broker itself holds no
    per-run state and the same demand plus the same telemetry always yields
    the same answer.
    """

    lane: LaneName
    role: str
    difficulty: int = MIN_TASK_DIFFICULTY
    #: Models already placed in the same adversarial set. Two reviewers from
    #: one lineage are not two independent opinions, so a seat prefers a
    #: candidate unlike the ones already seated.
    peers: tuple[ModelRef, ...] = ()
    #: The model this phase is already running on. Kept while it remains
    #: dispatchable, so a phase does not change models mid-thought. Pass
    #: ``None`` at a phase boundary or after an operational failure, which is
    #: what makes re-evaluation the caller's explicit decision.
    incumbent: ModelRef | None = None
    #: Dispatches currently outstanding per model, for concurrency caps.
    in_flight: Mapping[ModelRef, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "peers", tuple(self.peers))
        object.__setattr__(self, "in_flight", MappingProxyType(dict(self.in_flight)))
        if not self.lane:
            raise BrokerError("TaskDemand.lane must be non-empty")
        if self.role not in VALID_LANE_ROLES:
            raise BrokerError(
                f"unknown role {self.role!r}, must be one of {sorted(VALID_LANE_ROLES)}"
            )
        if not MIN_TASK_DIFFICULTY <= self.difficulty <= MAX_TASK_DIFFICULTY:
            raise BrokerError(
                f"difficulty must be within {MIN_TASK_DIFFICULTY}..{MAX_TASK_DIFFICULTY}, "
                f"got {self.difficulty}"
            )


# --- Decision --------------------------------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    """Why a candidate scored what it did, component by component.

    Components are normalized to roughly ``0.0``-``1.0`` where higher is
    better, and ``total`` is their configured weighted sum. Raw components are
    published rather than only their weighted contributions, so re-reading a
    decision under different weights does not require re-running the broker.
    """

    quality: float = 0.0
    cash_cost: float = 0.0
    quota_pressure: float = 0.0
    health: float = 0.0
    #: Signed on purpose only in the sense that it is a bonus and never a
    #: penalty: allowance that would otherwise reset unused. Hoarding a scarce
    #: allowance is a filter, not a negative score here.
    perishability: float = 0.0
    #: Allowance withheld from the binding window by configured reserve.
    #: Reported rather than weighted: the reserve's effect is to *exclude*, so
    #: what a reader needs is how close the decision came to that edge.
    reserve_effect: float = 0.0
    #: Subtracted, not added: how much this candidate resembles the ones
    #: already seated in the same adversarial review set.
    diversity_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "cash_cost": self.cash_cost,
            "quota_pressure": self.quota_pressure,
            "health": self.health,
            "perishability": self.perishability,
            "reserve_effect": self.reserve_effect,
            "diversity_penalty": self.diversity_penalty,
            "total": self.total,
        }


@dataclass(frozen=True)
class Candidate:
    """One catalog entry as the broker saw it for one demand."""

    ref: ModelRef
    eligible: bool
    reason: str = ""
    score: ScoreBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "eligible": self.eligible,
            "reason": self.reason,
            "score": self.score.to_dict() if self.score else None,
        }


@dataclass(frozen=True)
class BrokerDecision:
    """The placement, the ranking behind it, and every rejection."""

    demand: TaskDemand
    evaluated_at: datetime
    assignment: ModelAssignment | None = None
    #: Ordered chain for Hermes to walk when the primary fails at runtime.
    #: Separate from the assignment on purpose: Hermes handles operational
    #: failure, the broker handles economic scheduling, and a runtime retry
    #: must not quietly re-make an economic decision.
    fallbacks: tuple[ModelRef, ...] = ()
    ranked: tuple[Candidate, ...] = ()
    rejected: tuple[Candidate, ...] = ()
    #: True when the assignment is the incumbent kept in place rather than the
    #: top of ``ranked``.
    sticky: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.demand.lane,
            "role": self.demand.role,
            "difficulty": self.demand.difficulty,
            "evaluated_at": self.evaluated_at.isoformat(),
            "assignment": self.assignment.to_dict() if self.assignment else None,
            "fallbacks": list(self.fallbacks),
            "sticky": self.sticky,
            "ranked": [c.to_dict() for c in self.ranked],
            "rejected": [c.to_dict() for c in self.rejected],
            "reason": self.reason,
        }

    def render(self) -> str:
        """Human-readable dry-run view: ranked candidates with score components.

        The rejected are printed alongside the ranked on purpose — "no eligible
        model" with no cause is unactionable, and this is the surface an
        operator reads to fix the catalog or the policy instead of bisecting.
        """

        lines: list[str] = []
        head = (
            f"{self.demand.lane}/{self.demand.role} difficulty {self.demand.difficulty}"
            f" @ {self.evaluated_at.isoformat()}"
        )
        lines.append(head)
        if self.assignment is not None:
            sticky = " (sticky incumbent)" if self.sticky else ""
            fb = f" fallbacks: {', '.join(self.fallbacks)}" if self.fallbacks else ""
            lines.append(f"primary: {self.assignment.model_ref}{sticky}{fb}")
        else:
            lines.append(f"primary: NONE — {self.reason}")
        lines.append("ranked:")
        for c in self.ranked:
            if c.score is None:
                lines.append(f"  {c.ref} (no score)")
                continue
            s = c.score
            lines.append(
                f"  {c.ref}: total {s.total:.3f} "
                f"(quality {s.quality:.2f}, cost {s.cash_cost:.2f}, "
                f"quota {s.quota_pressure:.2f}, health {s.health:.2f}, "
                f"perish {s.perishability:.2f}, diversity -{s.diversity_penalty:.2f})"
            )
        if self.rejected:
            lines.append("rejected:")
            for c in self.rejected:
                lines.append(f"  {c.ref}: {c.reason}")
        return "\n".join(lines)


# --- Broker ----------------------------------------------------------------


class PolicyBroker:
    """Ranks catalog candidates for a demand against one telemetry scan.

    Snapshots are supplied at construction rather than fetched per call: a
    broker instance is *policy as of one scan*, which is what makes routing
    reproducible from a decision record.
    """

    def __init__(
        self,
        catalog: ModelCatalog,
        config: Any,
        *,
        snapshots: Sequence[ProviderResourceSnapshot] = (),
        resource_by_provider: Mapping[str, str] | None = None,
    ) -> None:
        self._catalog = catalog
        self._config = config
        self._snapshots = {snap.resource: snap for snap in snapshots}
        # Catalog entries are keyed by policy ref, telemetry by resource name,
        # and the two are deliberately different vocabularies. The join is the
        # Hermes provider id: telemetry config forbids two resources sharing
        # one provider, which is what makes this lookup single-valued.
        self._resource_by_provider = dict(resource_by_provider or {})
        self._reserves = {r.resource: r for r in config.reserves}
        # Atomic capacity reservation state (ModelBroker protocol).
        self._cap_lock = Lock()
        self._outstanding: dict[ModelRef, int] = {}

    def select(self, demand: TaskDemand, *, at: datetime | None = None) -> BrokerDecision:
        now = _utc(at or datetime.now(UTC))
        ranked: list[Candidate] = []
        rejected: list[Candidate] = []

        for entry in self._catalog.entries:
            reason = self._reject(entry, demand, now)
            if reason:
                rejected.append(Candidate(ref=entry.ref, eligible=False, reason=reason))
            else:
                ranked.append(
                    Candidate(ref=entry.ref, eligible=True, score=self._score(entry, demand, now))
                )

        # Python's sort is stable, so candidates that score alike keep catalog
        # order. That is the tiebreak: it is declared in a file an operator
        # controls, which makes a decision reproducible from its inputs.
        ranked.sort(key=lambda c: -c.score.total)

        chosen = self._choose(ranked, demand)
        return BrokerDecision(
            demand=demand,
            evaluated_at=now,
            assignment=(ModelAssignment(lane=demand.lane, model_ref=chosen) if chosen else None),
            fallbacks=self._fallback_chain(ranked, chosen),
            ranked=tuple(ranked),
            rejected=tuple(rejected),
            sticky=chosen is not None and chosen == demand.incumbent,
            reason="" if chosen else self._no_candidate_reason(rejected, demand),
        )

    def _choose(self, ranked: Sequence[Candidate], demand: TaskDemand) -> ModelRef | None:
        """The model to run on: the incumbent if it is still dispatchable.

        Keeping the incumbent is not a scoring bonus but an override, because
        "do not churn mid-phase" must not be defeasible by a competitor that
        happens to score a hair higher this turn. The caller ends stickiness
        by not passing an incumbent.
        """
        if demand.incumbent is not None and any(c.ref == demand.incumbent for c in ranked):
            return demand.incumbent
        return ranked[0].ref if ranked else None

    # --- ModelBroker protocol: atomic capacity reservation -------------------

    def reserve(self, assignment: ModelAssignment) -> ModelLease:
        """Atomically claim one concurrency slot for ``assignment``.

        ``select()`` ranks against a caller-supplied ``in_flight`` snapshot and
        is deliberately stateless; *this* is the serialization point. The
        outstanding count is checked and incremented under one lock, so two
        lanes scheduling concurrently cannot both take a model's last slot.
        The caller must :meth:`release` the lease when its dispatch ends.
        """
        with self._cap_lock:
            entry = self._catalog.get(assignment.model_ref)
            cap = entry.max_concurrency if entry else None
            if cap is not None and self._outstanding.get(assignment.model_ref, 0) >= cap:
                raise BrokerError(
                    f"model {assignment.model_ref!r} is at its concurrency cap ({cap}); "
                    "cannot reserve another slot"
                )
            self._outstanding[assignment.model_ref] = (
                self._outstanding.get(assignment.model_ref, 0) + 1
            )
        return ModelLease(model_ref=assignment.model_ref, lane=assignment.lane)

    def release(self, lease: ModelLease) -> None:
        """Return the slot taken by :meth:`reserve` (idempotent per lease)."""
        with self._cap_lock:
            current = self._outstanding.get(lease.model_ref, 0)
            if current > 0:
                self._outstanding[lease.model_ref] = current - 1

    def _fallback_chain(
        self, ranked: Sequence[Candidate], chosen: ModelRef | None
    ) -> tuple[ModelRef, ...]:
        """Runtime alternates for the primary, best first.

        Each link comes from a provider not already in the chain when
        ``require_distinct_fallback_provider`` is set: the chain is walked
        *because* a provider failed, and another model behind the same
        credentials and the same endpoint is the least likely thing to work.
        """
        if chosen is None:
            return ()
        config = self._config
        seen_providers = {self._provider_of(chosen)}
        chain: list[ModelRef] = []
        for candidate in ranked:
            if candidate.ref == chosen or len(chain) >= config.max_fallbacks:
                continue
            provider = self._provider_of(candidate.ref)
            if config.require_distinct_fallback_provider and provider in seen_providers:
                continue
            seen_providers.add(provider)
            chain.append(candidate.ref)
        return tuple(chain)

    def _provider_of(self, ref: ModelRef) -> str:
        """The failure domain a candidate dispatches through.

        An entry that declares no provider cannot be assumed to have its own
        endpoint or credentials: the conservative assumption is that all such
        entries share one unknown domain, so a provider outage defeats a chain
        built from them just as it would a chain of same-provider models.
        Claiming diversity the configuration does not evidence would be worse
        than admitting there is none to claim.
        """
        entry = self._catalog.get(ref)
        return (entry.provider if entry else "") or "<unset-provider>"

    # --- Scoring -----------------------------------------------------------

    def _score(self, entry: ModelCatalogEntry, demand: TaskDemand, now: datetime) -> ScoreBreakdown:
        config = self._config
        snap = self._snapshot_for(entry)
        views = self._windows(snap, now)
        binding = _binding(views)

        quality = self._quality_component(entry, demand)
        cash_cost = self._cash_cost_component(entry, now)
        quota_pressure = self._quota_component(entry, snap, binding)
        health = self._health_component(snap)
        perishability = self._perishability_component(snap, views)
        diversity_penalty = self._diversity_penalty(entry, demand)
        return ScoreBreakdown(
            quality=quality,
            cash_cost=cash_cost,
            quota_pressure=quota_pressure,
            health=health,
            perishability=perishability,
            diversity_penalty=diversity_penalty,
            reserve_effect=binding.reserve if binding else 0.0,
            total=(
                config.weight_quality * quality
                + config.weight_cash_cost * cash_cost
                + config.weight_quota_pressure * quota_pressure
                + config.weight_health * health
                + config.weight_perishability * perishability
                - config.weight_diversity * diversity_penalty
            ),
        )

    def _diversity_penalty(self, entry: ModelCatalogEntry, demand: TaskDemand) -> float:
        """Normalized similarity to the peers already chosen, in ``0.0``..``1.0``.

        Adversarial review wants independent opinions: two models from one
        lineage tend to share blind spots, so a shared family costs more than a
        merely shared vendor, and both cost more than nothing. The raw
        component is published unweighted; :attr:`BrokerConfig.weight_diversity`
        scales it in the total, so tuning diversity never requires re-deriving
        what similarity meant.
        """
        if not demand.peers:
            return 0.0
        catalog = self._catalog
        penalty = 0.0
        for peer in demand.peers:
            peer_entry = catalog.get(peer)
            if peer_entry is None:
                continue
            if entry.family and entry.family == peer_entry.family:
                penalty += 1.0  # strongest tie: one lineage
            elif entry.vendor and entry.vendor == peer_entry.vendor:
                penalty += 0.5
        # Bounded component: cap at the strongest possible disagreement cost.
        return min(1.0, penalty)

    def _quota_component(
        self, entry: ModelCatalogEntry, snap: Snapshot, binding: _WindowView | None
    ) -> float:
        """Headroom on the window that binds, from ``0.0`` to ``1.0``.

        Four cases that must not collapse into each other: a *metered*
        candidate with no allowance to run out (pay-as-you-go) is unconstrained
        and scores ``1.0``; a subscription whose allowance we could not
        measure scores the configured unknown value — unobserved is not the
        same as unlimited, and an unmeasured subscription must not beat a
        measured one nor skip reserve and scarcity filtering; a measured one
        scores its real headroom above reserve.
        """
        if snap is None:
            # An unmeasured subscription is unknown, not unlimited.
            return self._config.unknown_quota_score if self._is_subscription(entry) else 1.0
        if binding is None:
            return self._config.unknown_quota_score
        return max(0.0, min(1.0, binding.available))

    def _is_subscription(self, entry: ModelCatalogEntry) -> bool:
        return entry.resource_class == "subscription"

    def _health_component(self, snap: Snapshot) -> float:
        if snap is None:
            return self._config.unknown_health_score
        rate = snap.health.failure_rate
        if rate is None or snap.health.total < self._config.min_health_samples:
            # Too few calls to mean anything. Scoring it as perfect would let
            # an untried candidate outrank a proven one; scoring it as failed
            # would mean nothing new ever got tried.
            return self._config.unknown_health_score
        return max(0.0, 1.0 - rate)

    def _perishability_component(self, snap: Snapshot, views: Sequence[_WindowView]) -> float:
        """How much allowance is about to reset unused, per resource, not per window.

        With several simultaneously constraining windows the surplus usable is
        the *minimum* across them — a roomy session window cannot be drawn
        through a weekly window that is nearly spent — so the bonus is bounded
        by the tightest resetting window rather than the roomiest. Otherwise a
        hard task could be routed here on the strength of expiring capacity it
        could not actually consume.

        Purely a bonus. The mirror case — an allowance too scarce to spend on
        easy work — is handled by a filter with a stated reason rather than a
        negative score here, so it cannot be cancelled out by a good price.
        """
        config = self._config
        burn = (
            config.projected_burn_fraction_per_hour.get(snap.resource, 0.0)
            if snap is not None
            else 0.0
        )
        resetting = [(v, v.hours_to_reset) for v in views if v.hours_to_reset is not None]
        if not resetting:
            return self._promo_perishability(snap, config)
        # Jointly usable surplus after projected burn: the tightest window's
        # available headroom governs what any dispatch can actually draw.
        joint_surplus = min(max(0.0, v.available - burn * h) for v, h in resetting)
        # Urgency follows the soonest reset among the constraining windows.
        urgency = max(max(0.0, 1.0 - h / config.pull_forward_horizon_hours) for _, h in resetting)
        promo = self._promo_perishability(snap, config)
        return max(joint_surplus * urgency, promo)

    def _promo_perishability(self, snap: Snapshot, config: Any) -> float:
        """Perishable value of a promotion that is about to end.

        A promotion's expiry is the same economic event as a quota reset:
        capacity that vanishes unused is capacity wasted. The snapshot carries
        ``expires_at`` precisely so this policy can see the deadline.
        """
        if snap is None or snap.expires_at is None:
            return 0.0
        hours = (snap.expires_at - datetime.now(UTC)).total_seconds() / 3600.0
        if hours <= 0.0:
            return 0.0
        urgency = max(0.0, 1.0 - hours / config.pull_forward_horizon_hours)
        return urgency  # the whole promotion is surplus about to vanish

    def _quality_component(self, entry: ModelCatalogEntry, demand: TaskDemand) -> float:
        quality = entry.quality_for(demand.role)
        if quality is None:
            # The floor filter has already run, so an unscored entry that got
            # here has a configured default to stand in for it.
            quality = self._config.default_quality or MIN_QUALITY
        return (quality - MIN_QUALITY) / (MAX_QUALITY - MIN_QUALITY)

    def _cash_cost_component(self, entry: ModelCatalogEntry, now: datetime) -> float:
        """How little new money this candidate costs, from ``0.0`` to ``1.0``.

        Subscription capacity prices at zero here, however high its list
        price. The allowance is already bought, so dispatching against it
        moves no money — which is exactly the economic judgement
        :meth:`~.catalog.ModelCatalogEntry.effective_prices` documents as the
        broker's to make. What stops that from spending a subscription
        recklessly is the reserve and quota-pressure policy, not a pretend
        cash price.
        """
        config = self._config
        if entry.resource_class == "subscription":
            blended = 0.0
        else:
            prices = entry.effective_prices(now)
            # The price filter has already rejected unpriced entries, so
            # ``None`` cannot reach here.
            assert prices is not None
            blended = (
                prices[0] * config.expected_input_mtok + prices[1] * config.expected_output_mtok
            )
        # Mapped against a fixed reference price rather than normalized across
        # the candidate set: a set-relative score would change every entry's
        # number when an unrelated model is added to the catalog, so two
        # decisions could not be compared.
        return 1.0 / (1.0 + blended / config.reference_price_per_mtok)

    # --- Filters -----------------------------------------------------------

    def _reject(self, entry: ModelCatalogEntry, demand: TaskDemand, now: datetime) -> str:
        """Why ``entry`` cannot serve ``demand``, or ``""`` if it can."""
        if not entry.enabled:
            return f"catalog entry {entry.ref!r} is disabled"
        if entry.roles and demand.role not in entry.roles:
            return f"does not serve role {demand.role!r} (declares roles {sorted(entry.roles)})"
        if demand.difficulty < entry.min_task_difficulty:
            return (
                f"is reserved for difficulty >= {entry.min_task_difficulty}, "
                f"this task is difficulty {demand.difficulty}"
            )
        if not entry.has_known_price(now) and not self._is_subscription(entry):
            # Token prices are mandatory for *metered* capacity: without one,
            # no budget or reserve policy can reason about spending it. A
            # subscription's marginal cash cost is zero by definition (the
            # allowance is already bought), so it needs no token prices to be
            # dispatchable — fixed-price plans rarely expose them.
            return (
                "has no determinable price at "
                f"{now.isoformat()}: its promotion ended and it declares no list price, "
                "so no budget or reserve policy can reason about spending it"
            )
        cap = entry.max_concurrency
        if cap is not None and demand.in_flight.get(entry.ref, 0) >= cap:
            return (
                f"is at its concurrency cap ({cap} outstanding, cap {cap}); wait for "
                "an outstanding dispatch to finish or raise the catalog limit"
            )
        return self._reject_on_quality(entry, demand) or self._reject_on_telemetry(
            entry, demand, now
        )

    def _reject_on_telemetry(
        self, entry: ModelCatalogEntry, demand: TaskDemand, now: datetime
    ) -> str:
        """Why live telemetry says ``entry`` cannot serve ``demand`` now.

        A candidate with no telemetry at all is not rejected here: absence of
        a probe is not evidence against it.
        """
        snap = self._snapshot_for(entry)
        if snap is None:
            return ""
        if snap.availability == "exhausted":
            spent = ", ".join(w.label for w in snap.spent_windows())
            if spent:
                return (
                    f"has spent its allowance on window(s) [{spent}]; a spent window blocks "
                    "dispatch however much headroom the longer windows still report"
                )
            # No windows reported: the exhaustion came from the cash balance,
            # and naming the real cause is the point of the rejection.
            return (
                "is exhausted: its cash balance is zero and it declares no quota "
                "windows; top up the account or use a different resource"
            )
        if snap.availability == "unavailable":
            # Not time-boxed by a reset, so waiting will not fix it — the
            # operator has to act, which means they need the provider's words.
            return f"is unusable: {snap.reason}"

        health = snap.health
        if health.is_throttled(now) and health.retry_after is not None:
            return (
                f"is backing off until {health.retry_after.isoformat()} after a provider "
                "rate limit; this is a throttle, not a spent allowance"
            )
        rate = health.failure_rate
        config = self._config
        if (
            rate is not None
            and health.total >= config.min_health_samples
            and rate > config.max_failure_rate
        ):
            return (
                f"failed {rate:.0%} of its last {health.total} requests, beyond the "
                f"{config.max_failure_rate:.0%} threshold"
            )

        views = self._windows(snap, now)
        breached = [v for v in views if v.available <= 0.0]
        if breached:
            worst = min(breached, key=lambda v: v.available)
            return (
                f"window {worst.label!r} has {worst.remaining:.0%} of its allowance left, "
                f"at or below the configured reserve of {worst.reserve:.0%}; the reserve is "
                "held for the operator rather than spent by the broker"
            )
        return self._reject_on_scarcity(demand, views)

    def _reject_on_scarcity(self, demand: TaskDemand, views: Sequence[_WindowView]) -> str:
        """Withhold a nearly-spent allowance from work that does not need it.

        Every simultaneously constraining window is checked: a near-reset
        exemption on one window must not admit work that also draws on a
        *different* scarce window whose reset is far away, or the easy dispatch
        consumes allowance the policy meant to hold for hard tasks.

        Only when no scarce window is far from resetting is the gate lifted:
        the same remainder is worth spending freely when it is about to roll
        over, because holding it back then simply wastes it.
        """
        config = self._config
        scarce = [v for v in views if v.available < config.scarcity_threshold]
        if not scarce:
            return ""
        if demand.difficulty >= config.scarcity_difficulty_floor:
            return ""
        # The gate lifts only if *every* scarce window resets within the
        # pull-forward horizon (a window with no known reset never lifts it).
        near = all(
            v.hours_to_reset is not None and v.hours_to_reset <= config.pull_forward_horizon_hours
            for v in scarce
        )
        if near:
            return ""
        worst = min(scarce, key=lambda v: v.available)
        when = (
            "reports no reset"
            if worst.hours_to_reset is None
            else f"does not reset for {worst.hours_to_reset:.0f}h"
        )
        return (
            f"has only {worst.available:.0%} of window {worst.label!r} left above "
            f"reserve and {when}: that allowance is scarce and is held for work at "
            f"difficulty >= {config.scarcity_difficulty_floor} "
            f"(this task is difficulty {demand.difficulty})"
        )

    # --- Telemetry lookup --------------------------------------------------

    def _snapshot_for(self, entry: ModelCatalogEntry) -> Snapshot:
        # The explicit provider mapping is authoritative: a telemetry resource
        # that happens to share a paid ref's *name* is a different account, and
        # attaching its quota/health here would route work on the wrong
        # capacity. Direct ref matching is reserved for catalog-owned telemetry
        # where the resource was deliberately named after the ref.
        resource = self._resource_by_provider.get(entry.provider)
        if resource:
            snap = self._snapshots.get(resource)
            if snap is not None:
                return snap
        return self._snapshots.get(entry.ref)

    def _windows(self, snap: Snapshot, now: datetime) -> tuple[_WindowView, ...]:
        """Every *measured* window, with its reserve already subtracted.

        Unmeasured windows are dropped rather than assumed empty or full, so
        a window the provider declined to report can never become the binding
        constraint.
        """
        if snap is None:
            return ()
        reserve_config = self._reserves.get(snap.resource)
        views = []
        for w in snap.windows:
            remaining = w.remaining_fraction
            if remaining is None:
                continue
            reserve = 0.0
            if reserve_config is not None:
                reserve = reserve_config.windows.get(w.label, reserve_config.fraction)
            ttr = w.time_to_reset(now)
            views.append(
                _WindowView(
                    label=w.label,
                    reserve=reserve,
                    remaining=remaining,
                    available=remaining - reserve,
                    hours_to_reset=None if ttr is None else ttr.total_seconds() / 3600.0,
                )
            )
        return tuple(views)

    def _reject_on_quality(self, entry: ModelCatalogEntry, demand: TaskDemand) -> str:
        floor = self._quality_floor(demand)
        quality = entry.quality_for(demand.role)
        if quality is None:
            quality = self._config.default_quality
        if quality is None:
            return (
                f"is unscored for role {demand.role!r}: add it to quality_by_role in the "
                "catalog, or set broker.default_quality, so the quality floor can be "
                "checked instead of assumed"
            )
        if quality < floor:
            return (
                f"scores quality {quality} for role {demand.role!r}, below the floor "
                f"{floor} required at difficulty {demand.difficulty}"
            )
        return ""

    def _quality_floor(self, demand: TaskDemand) -> int:
        """The quality tier this demand requires.

        Difficulty and quality share one closed scale precisely so this
        comparison needs no mapping table (see :mod:`.catalog`). A role may
        raise the floor above it — adjudication work wants a stronger model
        than its nominal difficulty implies.
        """
        return max(
            demand.difficulty,
            self._config.min_quality_by_role.get(demand.role, MIN_TASK_DIFFICULTY),
        )

    def _no_candidate_reason(self, rejected: Sequence[Candidate], demand: TaskDemand) -> str:
        where = f"role {demand.role!r} at difficulty {demand.difficulty}"
        if not rejected:
            return (
                f"no model can serve {where}: the catalog declares no entries at all. "
                "Populate model_router.catalog or model_router.catalog_path."
            )
        detail = "; ".join(f"{c.ref}: {c.reason}" for c in rejected)
        return f"no model in the catalog can serve {where}. Rejected — {detail}"
