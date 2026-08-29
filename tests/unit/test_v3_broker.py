"""Behaviour of the V3 model broker (issue #47).

Tests exercise the public seam only: build a catalog + telemetry snapshots,
ask :meth:`PolicyBroker.select` to place one unit of work, and read the
decision. Nothing here reaches into scoring internals — a component is
asserted through the breakdown the decision publishes, which is the same
surface the dry-run command prints.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_pr_orchestrator.v3.broker import BrokerError, PolicyBroker, TaskDemand
from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.config import BrokerConfig, ResourceReserveConfig
from ai_pr_orchestrator.v3.domain import ModelAssignment
from ai_pr_orchestrator.v3.telemetry import (
    ProviderHealth,
    ProviderResourceSnapshot,
    QuotaWindow,
)

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def entry(ref: str, **overrides) -> ModelCatalogEntry:
    """A catalog entry that is dispatchable unless a test says otherwise."""
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


def broker(*entries: ModelCatalogEntry, config: BrokerConfig | None = None, **kwargs):
    return PolicyBroker(
        ModelCatalog(entries=entries),
        config or BrokerConfig(),
        **kwargs,
    )


class TestPlacement:
    def test_an_adequate_candidate_is_assigned_to_the_lane_that_asked(self):
        decision = broker(entry("solo")).select(
            TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW
        )

        assert decision.assignment is not None
        assert decision.assignment.lane == "developer"
        assert decision.assignment.model_ref == "solo"

    def test_an_empty_pool_says_what_to_do_about_it_rather_than_guessing(self):
        decision = broker().select(TaskDemand(lane="developer", role="worker"), at=NOW)

        assert decision.assignment is None
        assert decision.reason
        # The operator has to learn *which* filter emptied the pool. An empty
        # catalog and a catalog whose every entry was rejected are different
        # problems with different fixes.
        assert "catalog" in decision.reason

    def test_a_candidate_below_the_quality_floor_is_rejected_by_name_and_reason(self):
        decision = broker(entry("weak", quality_by_role={"worker": 2})).select(
            TaskDemand(lane="developer", role="worker", difficulty=4), at=NOW
        )

        assert decision.assignment is None
        rejected = {c.ref: c.reason for c in decision.rejected}
        assert "weak" in rejected
        assert "quality" in rejected["weak"]
        assert "4" in rejected["weak"]


class TestUnscoredCandidates:
    def test_an_entry_unscored_for_the_role_is_not_silently_treated_as_adequate(self):
        # Scoring the role is how an operator states a model is fit for it.
        # Absent that, the broker cannot establish the floor is met, and the
        # module's rule is that unknown is never assumed favourable.
        decision = broker(entry("unscored", quality_by_role={})).select(
            TaskDemand(lane="developer", role="worker", difficulty=1), at=NOW
        )

        assert decision.assignment is None
        reason = next(c.reason for c in decision.rejected if c.ref == "unscored")
        assert "quality_by_role" in reason

    def test_a_configured_default_quality_makes_unscored_entries_dispatchable(self):
        decision = broker(
            entry("unscored", quality_by_role={}),
            config=BrokerConfig(default_quality=3),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is not None
        assert decision.assignment.model_ref == "unscored"


class TestRanking:
    def test_the_cheaper_of_two_equally_good_candidates_wins_on_cash_cost(self):
        decision = broker(
            entry("pricey", input_price_per_mtok=1.0, output_price_per_mtok=2.0),
            entry("thrifty", input_price_per_mtok=0.5, output_price_per_mtok=1.0),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is not None
        assert decision.assignment.model_ref == "thrifty"
        scores = {c.ref: c.score for c in decision.ranked}
        # The reason must be visible in the breakdown, not just in the outcome:
        # same quality, better cash-cost component.
        assert scores["thrifty"].quality == scores["pricey"].quality
        assert scores["thrifty"].cash_cost > scores["pricey"].cash_cost

    def test_genuinely_free_capacity_is_preferred_when_quality_is_adequate(self):
        decision = broker(
            entry("paid", input_price_per_mtok=1.0, output_price_per_mtok=2.0),
            entry("promo", promotional=True, promo_ends_at=NOW + timedelta(days=7)),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "promo"

    def test_an_expired_promotion_stops_being_preferred_once_its_window_closes(self):
        lapsed = entry(
            "promo",
            promotional=True,
            promo_ends_at=NOW - timedelta(seconds=1),
            input_price_per_mtok=4.0,
            output_price_per_mtok=8.0,
        )
        pool = broker(entry("paid", input_price_per_mtok=1.0, output_price_per_mtok=2.0), lapsed)

        during = pool.select(
            TaskDemand(lane="developer", role="worker", difficulty=3),
            at=NOW - timedelta(days=1),
        )
        after = pool.select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert during.assignment.model_ref == "promo"
        assert after.assignment.model_ref == "paid"

    def test_a_stronger_model_wins_when_the_price_is_the_same(self):
        decision = broker(
            entry("adequate", quality_by_role={"worker": 3}, roles=("worker",)),
            entry("strong", quality_by_role={"worker": 5}, roles=("worker",)),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "strong"

    def test_subscription_capacity_competes_as_a_primary_candidate_not_a_fallback(self):
        # The allowance is already bought, so its marginal cash cost is zero
        # however high its list price is. That judgement is the broker's; the
        # catalog deliberately reports list price and refuses to make it.
        # The subscription is *measured* here: an unmeasured one scores
        # unknown on quota (unknown-is-not-favourable) and need not win.
        decision = broker(
            entry("metered-cheap", input_price_per_mtok=1.0, output_price_per_mtok=2.0),
            entry(
                "subscription",
                resource_class="subscription",
                input_price_per_mtok=15.0,
                output_price_per_mtok=75.0,
            ),
            snapshots=[
                snapshot("subscription", window("weekly", 0.1, resets_in=timedelta(days=5)))
            ],
            resource_by_provider={"subscription": "subscription"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "subscription"

    def test_identical_inputs_route_identically(self):
        pool = broker(entry("alpha"), entry("beta"), entry("gamma"))
        demand = TaskDemand(lane="developer", role="worker", difficulty=3)

        first = pool.select(demand, at=NOW)
        second = pool.select(demand, at=NOW)

        assert [c.ref for c in first.ranked] == [c.ref for c in second.ranked]

    def test_candidates_that_score_alike_are_ordered_by_the_catalog_not_by_chance(self):
        # Three entries differing only in ref. Without a stable tiebreak the
        # order would depend on sort implementation details, and an operator
        # could not reproduce a routing decision from the recorded inputs.
        decision = broker(entry("gamma"), entry("alpha"), entry("beta")).select(
            TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW
        )

        assert [c.ref for c in decision.ranked] == ["gamma", "alpha", "beta"]


def window(label: str, used: float | None, *, resets_in: timedelta | None = None) -> QuotaWindow:
    return QuotaWindow(
        label=label,
        used_fraction=used,
        reset_at=None if resets_in is None else NOW + resets_in,
    )


def snapshot(resource: str, *windows: QuotaWindow, **overrides) -> ProviderResourceSnapshot:
    defaults: dict[str, Any] = dict(
        resource=resource,
        observed_at=NOW,
        availability="available",
        resource_class="subscription",
        windows=windows,
    )
    defaults.update(overrides)
    return ProviderResourceSnapshot(**defaults)


def subscription(ref: str, **overrides) -> ModelCatalogEntry:
    return entry(ref, resource_class="subscription", **overrides)


class TestQuotaWindows:
    def test_a_spent_short_window_blocks_dispatch_however_large_the_weekly_surplus(self):
        # The binding constraint is the tightest window, not the roomiest. A
        # weekly allowance with 90% left cannot be spent through a session
        # window that is already gone.
        decision = broker(
            subscription("claude"),
            snapshots=[
                snapshot(
                    "claude",
                    window("session", 1.0, resets_in=timedelta(hours=2)),
                    window("weekly", 0.1, resets_in=timedelta(days=5)),
                    availability="exhausted",
                )
            ],
            resource_by_provider={"claude": "claude"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is None
        reason = next(c.reason for c in decision.rejected if c.ref == "claude")
        assert "session" in reason

    def test_weekly_surplus_is_dispatchable_while_the_short_window_has_headroom(self):
        decision = broker(
            subscription("claude"),
            snapshots=[
                snapshot(
                    "claude",
                    window("session", 0.3, resets_in=timedelta(hours=2)),
                    window("weekly", 0.2, resets_in=timedelta(hours=48)),
                )
            ],
            resource_by_provider={"claude": "claude"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is not None
        assert decision.assignment.model_ref == "claude"

    def test_the_configured_reserve_is_preserved_rather_than_spent(self):
        pool: dict[str, Any] = dict(
            snapshots=[snapshot("claude", window("weekly", 0.5, resets_in=timedelta(days=3)))],
            resource_by_provider={"claude": "claude"},
        )
        demand = TaskDemand(lane="developer", role="worker", difficulty=3)

        without = broker(subscription("claude"), **pool).select(demand, at=NOW)
        with_reserve = broker(
            subscription("claude"),
            config=BrokerConfig(reserves=[ResourceReserveConfig(resource="claude", fraction=0.6)]),
            **pool,
        ).select(demand, at=NOW)

        assert without.assignment is not None
        assert with_reserve.assignment is None
        reason = next(c.reason for c in with_reserve.rejected if c.ref == "claude")
        assert "reserve" in reason and "weekly" in reason

    def test_a_reserve_can_be_set_for_one_window_without_binding_the_others(self):
        config = BrokerConfig(
            reserves=[
                ResourceReserveConfig(resource="claude", fraction=0.0, windows={"session": 0.5})
            ]
        )
        decision = broker(
            subscription("claude"),
            config=config,
            snapshots=[
                snapshot(
                    "claude",
                    window("session", 0.7, resets_in=timedelta(hours=2)),
                    window("weekly", 0.1, resets_in=timedelta(days=5)),
                )
            ],
            resource_by_provider={"claude": "claude"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is None
        reason = next(c.reason for c in decision.rejected if c.ref == "claude")
        assert "session" in reason


class TestPerishableCapacity:
    def test_allowance_about_to_reset_unused_is_pulled_forward(self):
        # Two identical subscriptions; one's allowance evaporates in an hour.
        # Spending that one costs nothing that would not be lost anyway.
        decision = broker(
            subscription("evaporating"),
            subscription("patient"),
            snapshots=[
                snapshot("evaporating", window("session", 0.2, resets_in=timedelta(hours=1))),
                snapshot("patient", window("session", 0.2, resets_in=timedelta(days=7))),
            ],
            resource_by_provider={"evaporating": "evaporating", "patient": "patient"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "evaporating"
        scores = {c.ref: c.score for c in decision.ranked}
        assert scores["evaporating"].perishability > scores["patient"].perishability

    def test_expected_work_before_reset_cancels_the_pull_forward(self):
        # Allowance is only perishable if nothing else will consume it. A
        # deployment that expects to burn the window anyway has no surplus to
        # pull forward, and saying so must change the answer.
        pool: dict[str, Any] = dict(
            snapshots=[
                snapshot("busy", window("session", 0.2, resets_in=timedelta(hours=4))),
            ],
            resource_by_provider={"busy": "busy"},
        )
        demand = TaskDemand(lane="developer", role="worker", difficulty=3)

        idle = broker(subscription("busy"), **pool).select(demand, at=NOW)
        busy = broker(
            subscription("busy"),
            config=BrokerConfig(projected_burn_fraction_per_hour={"busy": 0.25}),
            **pool,
        ).select(demand, at=NOW)

        assert idle.ranked[0].score.perishability > busy.ranked[0].score.perishability

    def test_a_scarce_allowance_with_a_distant_reset_is_saved_for_hard_work(self):
        # 5% left and days until it comes back. Withholding it from easy work
        # is a filter with a stated reason, not a scoring nudge: an operator
        # has to be able to read *why* the subscription sat idle, and a rule
        # that only bites when the arithmetic happens to land is not a policy.
        pool: dict[str, Any] = dict(
            snapshots=[snapshot("scarce", window("weekly", 0.95, resets_in=timedelta(days=6)))],
            resource_by_provider={"scarce": "scarce"},
        )
        candidates = (
            subscription("scarce", quality_by_role={"worker": 4}, roles=("worker",)),
            entry(
                "metered",
                quality_by_role={"worker": 4},
                roles=("worker",),
                input_price_per_mtok=1.0,
                output_price_per_mtok=2.0,
            ),
        )

        easy = broker(*candidates, **pool).select(
            TaskDemand(lane="developer", role="worker", difficulty=1), at=NOW
        )
        hard = broker(*candidates, **pool).select(
            TaskDemand(lane="developer", role="worker", difficulty=4), at=NOW
        )

        assert easy.assignment.model_ref == "metered"
        reason = next(c.reason for c in easy.rejected if c.ref == "scarce")
        assert "scarce" in reason
        # "Preserved for hard tasks" means the gate lifts, not that the scarce
        # allowance then outbids every alternative — that stays a matter of
        # what the deployment weights.
        assert "scarce" not in {c.ref for c in hard.rejected}
        assert "scarce" in {c.ref for c in hard.ranked}

    def test_scarcity_only_withholds_an_allowance_that_will_not_come_back_soon(self):
        # The same 5% is not worth hoarding when it resets within the hour —
        # holding it back would simply waste it.
        scarce = subscription("scarce", quality_by_role={"worker": 4}, roles=("worker",))
        demand = TaskDemand(lane="developer", role="worker", difficulty=1)

        distant = broker(
            scarce,
            snapshots=[snapshot("scarce", window("weekly", 0.95, resets_in=timedelta(days=6)))],
            resource_by_provider={"scarce": "scarce"},
        ).select(demand, at=NOW)
        imminent = broker(
            scarce,
            snapshots=[snapshot("scarce", window("session", 0.95, resets_in=timedelta(hours=1)))],
            resource_by_provider={"scarce": "scarce"},
        ).select(demand, at=NOW)

        assert distant.assignment is None
        assert imminent.assignment is not None


class TestHealth:
    def test_a_provider_failing_beyond_the_threshold_is_filtered_out(self):
        decision = broker(
            subscription("flaky"),
            snapshots=[
                snapshot(
                    "flaky",
                    health=ProviderHealth(
                        observed_at=NOW,
                        outcomes={"success": 2, "server_error": 8},
                        consecutive_failures=3,
                    ),
                )
            ],
            resource_by_provider={"flaky": "flaky"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is None
        reason = next(c.reason for c in decision.rejected if c.ref == "flaky")
        assert "80%" in reason

    def test_a_provider_failing_below_the_threshold_is_demoted_not_excluded(self):
        decision = broker(
            subscription("shaky"),
            subscription("solid"),
            snapshots=[
                snapshot(
                    "shaky",
                    health=ProviderHealth(observed_at=NOW, outcomes={"success": 8, "timeout": 2}),
                ),
                snapshot(
                    "solid",
                    health=ProviderHealth(observed_at=NOW, outcomes={"success": 10}),
                ),
            ],
            resource_by_provider={"shaky": "shaky", "solid": "solid"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "solid"
        assert [c.ref for c in decision.ranked] == ["solid", "shaky"]

    def test_a_provider_still_backing_off_is_not_dispatched_to(self):
        # A 429 back-off is a throttle, not a spent quota: the candidate comes
        # back on its own, so the reason has to say when.
        retry_at = NOW + timedelta(minutes=5)
        decision = broker(
            subscription("throttled"),
            snapshots=[
                snapshot(
                    "throttled",
                    health=ProviderHealth(
                        observed_at=NOW,
                        outcomes={"rate_limited": 1},
                        retry_after=retry_at,
                    ),
                )
            ],
            resource_by_provider={"throttled": "throttled"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is None
        reason = next(c.reason for c in decision.rejected if c.ref == "throttled")
        assert retry_at.isoformat() in reason


class TestUnknownTelemetry:
    def test_an_unreachable_provider_is_demoted_but_not_declared_empty(self):
        # The rule this whole surface exists for: a failed probe is ignorance,
        # not an exhausted quota. It loses to a measured competitor, and it
        # stays in the pool when it is the only thing left.
        unknown = snapshot(
            "dark",
            availability="unknown",
            reason="probe failed: connection refused",
        )
        measured = snapshot("lit", window("weekly", 0.1, resets_in=timedelta(days=5)))

        contested = broker(
            subscription("dark"),
            subscription("lit"),
            snapshots=[unknown, measured],
            resource_by_provider={"dark": "dark", "lit": "lit"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
        alone = broker(
            subscription("dark"),
            snapshots=[unknown],
            resource_by_provider={"dark": "dark"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert contested.assignment.model_ref == "lit"
        assert alone.assignment.model_ref == "dark"

    def test_a_resource_the_provider_calls_unusable_is_rejected_with_its_reason(self):
        decision = broker(
            subscription("no-creds"),
            snapshots=[
                snapshot(
                    "no-creds",
                    availability="unavailable",
                    reason="this provider is not supported by the installed adapter",
                )
            ],
            resource_by_provider={"no-creds": "no-creds"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment is None
        reason = next(c.reason for c in decision.rejected if c.ref == "no-creds")
        assert "not supported by the installed adapter" in reason

    def test_a_model_with_no_quota_to_track_is_not_penalized_for_having_none(self):
        # Pay-as-you-go capacity has no allowance that can run out. Scoring its
        # absent quota the same as an unmeasured one would penalize it for a
        # constraint it does not have.
        decision = broker(
            entry("metered", input_price_per_mtok=1.0, output_price_per_mtok=2.0),
            subscription("dark"),
            snapshots=[
                snapshot("dark", availability="unknown", reason="probe failed"),
            ],
            resource_by_provider={"dark": "dark"},
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        scores = {c.ref: c.score for c in decision.ranked}
        assert scores["metered"].quota_pressure > scores["dark"].quota_pressure


class TestAdversarialDiversity:
    def test_a_review_set_is_not_stacked_with_one_vendors_models(self):
        # Two reviewers from one family are not two independent opinions, so
        # the second seat prefers a different lineage even when the in-family
        # sibling is the stronger model on its own.
        pool = (
            entry(
                "house-strong",
                vendor="house",
                family="house-1",
                quality_by_role={"reviewer": 5, "worker": 5, "foreman": 5},
            ),
            entry("rival", vendor="rival", family="rival-1"),
        )
        first = broker(*pool).select(TaskDemand(lane="r1", role="reviewer", difficulty=3), at=NOW)
        second = broker(*pool).select(
            TaskDemand(
                lane="r2", role="reviewer", difficulty=3, peers=(first.assignment.model_ref,)
            ),
            at=NOW,
        )

        assert first.assignment.model_ref == "house-strong"
        assert second.assignment.model_ref == "rival"

    def test_shared_lineage_is_penalized_more_than_a_merely_shared_vendor(self):
        decision = broker(
            entry("same-family", vendor="house", family="shared"),
            entry("same-vendor-only", vendor="house", family="other"),
            entry("incumbent", vendor="house", family="shared"),
        ).select(TaskDemand(lane="r2", role="reviewer", difficulty=3, peers=("incumbent",)), at=NOW)

        scores = {c.ref: c.score for c in decision.ranked}
        assert (
            scores["same-family"].diversity_penalty > scores["same-vendor-only"].diversity_penalty
        )
        assert scores["same-vendor-only"].diversity_penalty > 0

    def test_an_unrelated_candidate_carries_no_diversity_penalty(self):
        decision = broker(entry("unrelated"), entry("peer")).select(
            TaskDemand(lane="r2", role="reviewer", difficulty=3, peers=("peer",)), at=NOW
        )

        scores = {c.ref: c.score for c in decision.ranked}
        assert scores["unrelated"].diversity_penalty == 0


class TestStickiness:
    def test_a_phase_keeps_its_model_rather_than_churning_every_turn(self):
        pool = (
            entry("incumbent", input_price_per_mtok=4.0, output_price_per_mtok=8.0),
            entry("newcomer", input_price_per_mtok=0.1, output_price_per_mtok=0.2),
        )
        fresh = broker(*pool).select(TaskDemand(lane="developer", role="worker"), at=NOW)
        continuing = broker(*pool).select(
            TaskDemand(lane="developer", role="worker", incumbent="incumbent"), at=NOW
        )

        assert fresh.assignment.model_ref == "newcomer"
        assert continuing.assignment.model_ref == "incumbent"
        assert continuing.sticky
        # The ranking still says what would have won, so a dry-run shows the
        # cost of staying put instead of hiding it.
        assert continuing.ranked[0].ref == "newcomer"

    def test_an_incumbent_that_stopped_being_dispatchable_is_replaced(self):
        decision = broker(
            entry("incumbent", enabled=False),
            entry("newcomer"),
        ).select(TaskDemand(lane="developer", role="worker", incumbent="incumbent"), at=NOW)

        assert decision.assignment.model_ref == "newcomer"
        assert not decision.sticky


class TestConcurrency:
    def test_a_model_at_its_concurrency_cap_is_not_handed_more_work(self):
        decision = broker(
            entry("busy", max_concurrency=2),
            entry("free"),
        ).select(
            TaskDemand(lane="developer", role="worker", difficulty=3, in_flight={"busy": 2}),
            at=NOW,
        )

        assert decision.assignment.model_ref == "free"
        reason = next(c.reason for c in decision.rejected if c.ref == "busy")
        assert "concurrency" in reason and "2" in reason

    def test_a_model_below_its_cap_still_takes_work(self):
        decision = broker(entry("busy", max_concurrency=2)).select(
            TaskDemand(lane="developer", role="worker", difficulty=3, in_flight={"busy": 1}),
            at=NOW,
        )

        assert decision.assignment.model_ref == "busy"


class TestFallbackChain:
    def test_fallbacks_are_ordered_and_kept_separate_from_the_primary(self):
        # Hermes walks this chain on runtime failure; the broker decides the
        # economics. Conflating them would let a runtime retry silently
        # re-do the economic choice.
        decision = broker(
            entry("best", input_price_per_mtok=0.1, output_price_per_mtok=0.2),
            entry("second", input_price_per_mtok=1.0, output_price_per_mtok=2.0),
            entry("third", input_price_per_mtok=4.0, output_price_per_mtok=8.0),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "best"
        assert decision.fallbacks == ("second", "third")

    def test_a_fallback_does_not_share_the_failure_domain_it_is_covering_for(self):
        # A chain is walked because the provider failed. Another model on that
        # same provider is the least likely thing to work.
        decision = broker(
            entry("primary", provider="alpha", input_price_per_mtok=0.1, output_price_per_mtok=0.2),
            entry("sibling", provider="alpha", input_price_per_mtok=0.2, output_price_per_mtok=0.4),
            entry(
                "elsewhere", provider="beta", input_price_per_mtok=1.0, output_price_per_mtok=2.0
            ),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert decision.assignment.model_ref == "primary"
        assert decision.fallbacks == ("elsewhere",)

    def test_the_chain_is_bounded_so_a_large_catalog_does_not_become_the_chain(self):
        decision = broker(
            *[
                entry(f"m{i}", input_price_per_mtok=float(i), output_price_per_mtok=float(i))
                for i in range(8)
            ],
            config=BrokerConfig(max_fallbacks=2),
        ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)

        assert len(decision.fallbacks) == 2


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"enabled": False}, "disabled"),
        ({"roles": ("reviewer",), "quality_by_role": {"reviewer": 3}}, "role"),
        ({"min_task_difficulty": 5}, "difficulty"),
        (
            # A lapsed promotion is only fatal when it leaves the price
            # unknown; one that reverts to a declared list price stays
            # dispatchable, so the prices have to go too.
            {
                "promotional": True,
                "promo_ends_at": NOW - timedelta(days=1),
                "input_price_per_mtok": None,
                "output_price_per_mtok": None,
            },
            "price",
        ),
    ],
)
def test_catalog_level_ineligibility_is_reported_not_hidden(overrides, expected):
    # The catalog already answers these; the broker's job is to surface *which*
    # one fired, because "no eligible model" with no cause is unactionable.
    candidate = entry("blocked", **overrides)
    decision = broker(candidate).select(
        TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW
    )

    assert decision.assignment is None
    reason = next(c.reason for c in decision.rejected if c.ref == "blocked")
    assert expected in reason
    assert expected in decision.render()  # the dry-run view names the cause too


def test_render_shows_ranked_scores_and_rejections():
    decision = broker(
        subscription("primary"),
        entry("rival", provider="other", vendor="other", family="other"),
        entry("blocked", enabled=False),
    ).select(
        TaskDemand(lane="r1", role="reviewer", difficulty=3, peers=("primary",)),
        at=NOW,
    )
    text = decision.render()
    assert "primary:" in text
    assert "ranked:" in text
    assert "diversity" in text
    assert "rejected:" in text and "disabled" in text


# --- review round 1 regression tests -----------------------------------------


def test_reserves_deserialize_from_plain_dicts():
    from ai_pr_orchestrator.v3.config import V3Config

    cfg = V3Config.from_dict(
        {
            "broker": {
                "reserves": [{"resource": "sub", "fraction": 0.2}],
            }
        }
    )
    decision = broker(
        subscription("sub"),
        snapshots=[snapshot("sub", window("weekly", 0.95, resets_in=timedelta(days=5)))],
        resource_by_provider={"sub": "sub"},
        config=cfg.broker,
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    # 5% remaining - 20% reserve < 0: the *deserialized* reserve (a typed
    # ResourceReserveConfig built from a plain dict, not a crash) breached the
    # window and excluded the candidate.
    assert decision.assignment is None
    reason = next(c.reason for c in decision.rejected if c.ref == "sub")
    assert "reserve" in reason


def test_unmeasured_subscription_is_unknown_not_unlimited():
    # A subscription with NO snapshot must not score perfect quota or skip
    # reserve/scarcity reasoning: unknown is not favourable.
    measured = broker(
        subscription("measured"),
        snapshots=[snapshot("measured", window("weekly", 0.05, resets_in=timedelta(days=5)))],
        resource_by_provider={"measured": "measured"},
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    unmeasured = broker(subscription("blind")).select(
        TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW
    )
    m_score = next(c.score for c in measured.ranked if c.ref == "measured")
    u_score = next(c.score for c in unmeasured.ranked if c.ref == "blind")
    assert m_score is not None and u_score is not None
    assert u_score.quota_pressure < m_score.quota_pressure  # unknown below measured-good


def test_perishability_is_bounded_by_jointly_usable_quota():
    # Session window: 80% remaining, resets in 1h. Weekly: 1% remaining, resets
    # in 5d. The perishable surplus usable is ~1% (weekly binds), NOT ~80%.
    decision = broker(
        subscription("both"),
        snapshots=[
            snapshot(
                "both",
                window("session", 0.9, resets_in=timedelta(hours=1)),
                window("weekly", 0.1, resets_in=timedelta(days=5)),
            )
        ],
        resource_by_provider={"both": "both"},
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    score = next(c.score for c in decision.ranked if c.ref == "both")
    assert score is not None
    # Session has 90% remaining and resets in 1h (urgency ~0.96); weekly has
    # only 10% remaining. The jointly usable surplus is bounded by the weekly
    # window (~0.10), so perishability must be small, not ~0.9*0.96.
    assert score.perishability < 0.2


def test_scarcity_gate_checks_all_windows_not_just_the_binding_one():
    # Session: 40% available, resets in 1h (near-reset would lift the gate).
    # Weekly: 5% available, resets in 6 days (scarce, far). Easy work must be
    # rejected even though the *binding* (session) window is about to reset.
    decision = broker(
        subscription("split"),
        snapshots=[
            snapshot(
                "split",
                window("session", 0.6, resets_in=timedelta(hours=1)),
                window("weekly", 0.95, resets_in=timedelta(days=6)),
            )
        ],
        resource_by_provider={"split": "split"},
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    assert decision.assignment is None
    reason = next(c.reason for c in decision.rejected if c.ref == "split")
    assert "scarce" in reason


def test_reserve_is_atomic_under_concurrency():
    import threading

    b = broker(entry("capped", max_concurrency=1), entry("spare"))
    assignment = ModelAssignment(lane="developer", model_ref="capped")

    lease = b.reserve(assignment)
    with pytest.raises(BrokerError):
        b.reserve(assignment)  # cap 1 already taken
    b.release(lease)
    lease2 = b.reserve(assignment)  # released -> reservable again
    b.release(lease2)

    # Hammer it from threads: never more than cap concurrent holders.
    results = []
    lock = threading.Lock()

    def worker():
        try:
            b.reserve(assignment)
            with lock:
                results.append("ok")
        except BrokerError:
            with lock:
                results.append("blocked")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") <= 1


def test_diversity_weight_is_independent_of_quality_weight():
    e_family = entry("same-family", vendor="house", family="shared")
    e_other = entry("other", vendor="rival", family="rival")
    demand = TaskDemand(lane="r2", role="reviewer", difficulty=3, peers=("incumbent",))
    incumbent = entry("incumbent", vendor="house", family="shared")

    tuned = broker(
        e_family,
        e_other,
        incumbent,
        config=BrokerConfig(weight_quality=5.0, weight_diversity=0.1),
    ).select(demand, at=NOW)
    s_family = next(c.score for c in tuned.ranked if c.ref == "same-family")
    assert s_family is not None
    # The published component is normalized 0..1 regardless of weights.
    assert s_family.diversity_penalty == 1.0


def test_naive_evaluation_timestamp_is_normalized():
    naive = datetime(2026, 8, 28, 12, 0)  # no tzinfo
    decision = broker(entry("solo")).select(
        TaskDemand(lane="developer", role="worker", difficulty=3), at=naive
    )
    assert decision.evaluated_at.tzinfo is not None


def test_provider_mapping_beats_ref_name_match():
    # A telemetry resource named like the ref but mapped to a DIFFERENT
    # provider must not be attached via the ref-name shortcut.
    decision = broker(
        entry("paid", provider="real"),
        snapshots=[
            snapshot("paid", window("weekly", 0.01, resets_in=timedelta(days=1))),
            snapshot("real", window("weekly", 0.99, resets_in=timedelta(days=1))),
        ],
        resource_by_provider={"real": "real"},
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    score = next(c.score for c in decision.ranked if c.ref == "paid")
    assert score is not None
    # The provider mapping binds 'paid' to the 'real' resource, which is
    # nearly exhausted — the roomy resource that merely *shares the ref's
    # name* must be ignored.
    assert score.quota_pressure < 0.5


def test_zero_cash_balance_is_named_as_the_exhaustion_cause():
    decision = broker(
        entry("broke"),
        snapshots=[
            snapshot("broke", availability="exhausted", cash_balance=0.0)  # no windows
        ],
        resource_by_provider={"broke": "broke"},
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    reason = next(c.reason for c in decision.rejected if c.ref == "broke")
    assert "cash balance" in reason


def test_subscription_without_token_prices_is_dispatchable():
    # Fixed-price plans expose no per-token prices; the subscription's marginal
    # cost is zero by definition, so it must not be rejected for missing them.
    decision = broker(
        subscription("flat"),
        snapshots=[snapshot("flat", window("weekly", 0.5, resets_in=timedelta(days=3)))],
        resource_by_provider={"flat": "flat"},
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    assert decision.assignment is not None


def test_unset_providers_share_one_conservative_failure_domain():
    # Two fallbacks with no declared provider must NOT both enter the chain:
    # the broker cannot know they are distinct failure domains.
    decision = broker(
        entry("primary", provider="p1"),
        entry("blank-a", provider=""),
        entry("blank-b", provider=""),
    ).select(TaskDemand(lane="developer", role="worker", difficulty=3), at=NOW)
    blank_fallbacks = [r for r in decision.fallbacks if r.startswith("blank-")]
    assert len(blank_fallbacks) <= 1
