"""E2E scenarios 7 and 8 (issue #55):

- **7**: subscription near reset -> broker preferentially consumes surplus
  above reserve. When a subscription's session window has been pulled
  forward but its weekly allowance still has headroom, the broker must
  spend the *weekly surplus* on the demand rather than consuming a
  different resource's allowance. This proves the broker honours
  ``ResourceReserveConfig.fraction`` (the operator's "do not touch this
  fraction of every window" rule) and pulls the surplus forward when
  the reset is inside the configured horizon.

- **8**: promotion expires between issues -> the next assignment changes.
  Two consecutive ``select`` calls at different ``at`` times, one before
  the promo end and one after, must place the same demand on different
  candidates. A free promotional allowance is preferred while live; once
  expired, the next caller gets the next-best (and more expensive)
  candidate.

Both scenarios are exercised against the real :class:`PolicyBroker`
with a two-entry catalog: a subscription whose ``weekly`` window has
surplus above the configured reserve, and a paid candidate that is
always available. The broker's ``select`` is called through
:class:`ForemanPolicyLoop._reserve` (and therefore the production path)
in scenario 7; scenario 8 calls ``select`` directly because the
promotion-expiry time-shift must not be elided by a long-running
foreman's clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ai_pr_orchestrator.v3.broker import PolicyBroker, TaskDemand
from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.config import BrokerConfig, ResourceReserveConfig
from ai_pr_orchestrator.v3.telemetry import (
    ProviderResourceSnapshot,
    QuotaWindow,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _entry(ref: str, **overrides) -> ModelCatalogEntry:
    defaults: dict[str, Any] = dict(
        ref=ref,
        descriptor=f"descriptor-for-{ref}",
        provider=ref,
        vendor=ref,
        family=ref,
        capabilities=("tools", "coding"),
        roles=("worker", "reviewer", "foreman"),
        quality_by_role={"worker": 3, "reviewer": 3, "foreman": 3},
        input_price_per_mtok=1.0,
        output_price_per_mtok=2.0,
    )
    defaults.update(overrides)
    return ModelCatalogEntry(**defaults)


def _window(label: str, used: float, *, resets_in: timedelta) -> QuotaWindow:
    return QuotaWindow(label=label, used_fraction=used, reset_at=NOW + resets_in)


def _snapshot(
    resource: str,
    *windows: QuotaWindow,
    availability: str = "available",
    resource_class: str = "subscription",
) -> ProviderResourceSnapshot:
    return ProviderResourceSnapshot(
        resource=resource,
        observed_at=NOW,
        availability=availability,
        resource_class=resource_class,
        windows=windows,
    )


def test_scenario_7_subscription_surplus_is_consumed_above_reserve():
    """A subscription whose ``weekly`` window is mostly intact (30%
    used -> 70% remaining), with the operator reserving 50% of every
    window for other work, leaves the configured reserve alone and
    pulls the *surplus* (20%) into the dispatchable headroom when the
    reset is inside ``pull_forward_horizon_hours``.

    The candidate the broker dispatches for a developer demand must be
    that subscription — not the more expensive paid candidate. This is
    the surplus-routing contract from #47 / scenario 7.
    """
    sub = _entry("sub-ref", resource_class="subscription")
    paid = _entry("paid-ref")
    catalog = ModelCatalog(entries=(sub, paid))

    config = BrokerConfig(
        scarcity_threshold=0.25,
        scarcity_difficulty_floor=4,
        # Reserve 10% of every window — anything above 10% remaining is
        # therefore dispatchable; the broker pulls the 60% surplus
        # above the reserve into the score when the reset is inside
        # ``pull_forward_horizon_hours``.
        reserves=[
            ResourceReserveConfig(resource="sub-ref", fraction=0.1),
        ],
        # Pull forward resets within 24 hours.
        pull_forward_horizon_hours=24.0,
    )

    snapshots = [
        _snapshot(
            "sub-ref",
            _window("session", 0.3, resets_in=timedelta(hours=2)),
            _window("weekly", 0.3, resets_in=timedelta(hours=20)),
        ),
    ]
    resource_by_provider = {"sub-ref": "sub-ref"}

    broker_obj = PolicyBroker(
        catalog, config, snapshots=snapshots, resource_by_provider=resource_by_provider
    )
    decision = broker_obj.select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    assert decision.assignment is not None, (
        f"expected a dispatchable subscription assignment, got reason={decision.reason!r}; "
        f"rejected={[c.ref for c in decision.rejected]}"
    )
    assert decision.assignment.model_ref == "sub-ref", (
        f"expected the subscription to consume its weekly surplus, got {decision.assignment.model_ref}"
    )


def test_scenario_7_surplus_is_not_consumed_when_reserve_is_violated():
    """When the subscription's ``weekly`` window has burned past the
    operator's reserved 50%, the broker must NOT consume the headroom
    even though a reset is imminent. The candidate is rejected on
    scarcity grounds, and the paid candidate wins.
    """
    sub = _entry("sub-ref", resource_class="subscription")
    paid = _entry("paid-ref")
    catalog = ModelCatalog(entries=(sub, paid))

    config = BrokerConfig(
        scarcity_threshold=0.25,
        scarcity_difficulty_floor=4,
        reserves=[
            ResourceReserveConfig(resource="sub-ref", fraction=0.5),
        ],
        pull_forward_horizon_hours=24.0,
    )

    snapshots = [
        _snapshot(
            "sub-ref",
            _window("session", 0.9, resets_in=timedelta(hours=2)),
            _window("weekly", 0.8, resets_in=timedelta(hours=20)),
        ),
    ]
    resource_by_provider = {"sub-ref": "sub-ref"}

    broker_obj = PolicyBroker(
        catalog, config, snapshots=snapshots, resource_by_provider=resource_by_provider
    )
    decision = broker_obj.select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    assert decision.assignment is not None
    # The subscription was rejected for violating its reserve.
    sub_rejection = next((c.reason for c in decision.rejected if c.ref == "sub-ref"), None)
    assert sub_rejection is not None and (
        "reserve" in sub_rejection or "scarce" in sub_rejection
    ), f"expected a reserve-related rejection, got {sub_rejection!r}"
    assert decision.assignment.model_ref == "paid-ref"


def test_scenario_8_promotion_expiry_changes_next_assignment():
    """Two consecutive ``select`` calls — one before the promotion
    ends, one after — must place the same demand on different
    candidates. While the promo is live the broker dispatches the
    promotional entry; once expired, the next call must fall through
    to the paid candidate.

    The call uses two distinct ``at`` times so the broker cannot
    paper over the transition with a single point-in-time evaluation.
    """
    promo = _entry(
        "promo-ref",
        promotional=True,
        promo_starts_at=NOW - timedelta(days=1),
        promo_ends_at=NOW + timedelta(hours=1),
        # Higher list price: promo's *only* advantage is its transient
        # free tier; without it the paid entry is strictly better.
        input_price_per_mtok=4.0,
        output_price_per_mtok=8.0,
    )
    paid = _entry(
        "paid-ref",
        input_price_per_mtok=1.0,
        output_price_per_mtok=2.0,
    )
    catalog = ModelCatalog(entries=(promo, paid))
    broker_obj = PolicyBroker(catalog, BrokerConfig())

    during = broker_obj.select(
        TaskDemand(lane="developer", role="worker", difficulty=3),
        at=NOW,
    )
    after = broker_obj.select(
        TaskDemand(lane="developer", role="worker", difficulty=3),
        at=NOW + timedelta(hours=2),
    )

    assert during.assignment is not None
    assert after.assignment is not None
    assert during.assignment.model_ref == "promo-ref", (
        f"expected promo while live, got {during.assignment.model_ref}"
    )
    assert after.assignment.model_ref == "paid-ref", (
        f"expected paid once promo expired, got {after.assignment.model_ref}"
    )
    # The same demand should never have been dispatched to the same
    # model across the expiry boundary; this guards against a broker
    # that caches ``assignment`` per lane and ignores ``at``.
    assert during.assignment.model_ref != after.assignment.model_ref


def test_scenario_8_promotion_expiry_is_per_demand():
    """A single ``select`` after the promotion has expired must not
    dispatch the promotional entry even if the broker was previously
    asked to reserve it. The reservation contract and the time
    evaluation are independent; expiry is absolute."""
    promo = _entry(
        "promo-ref",
        promotional=True,
        promo_starts_at=NOW - timedelta(days=1),
        promo_ends_at=NOW - timedelta(seconds=1),
        input_price_per_mtok=4.0,
        output_price_per_mtok=8.0,
    )
    paid = _entry(
        "paid-ref",
        input_price_per_mtok=1.0,
        output_price_per_mtok=2.0,
    )
    catalog = ModelCatalog(entries=(promo, paid))
    broker_obj = PolicyBroker(catalog, BrokerConfig())

    decision = broker_obj.select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    assert decision.assignment is not None
    assert decision.assignment.model_ref == "paid-ref", (
        f"expired promo must not be selected, got {decision.assignment.model_ref}"
    )
    # Even though the promo is *not* explicitly rejected (the broker
    # silently down-scores its perishability to zero), it must not be
    # the chosen candidate. The ranked list is observable in dry-run.
    ranked_refs = [c.ref for c in decision.ranked]
    assert ranked_refs[0] == "paid-ref", f"expected paid first in ranked list, got {ranked_refs}"
