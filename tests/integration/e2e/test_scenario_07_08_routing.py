"""Real phase dispatch responds to quota reserve and catalog promotion changes."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_pr_orchestrator.v3.catalog import ModelCatalog
from ai_pr_orchestrator.v3.config import ResourceReserveConfig
from ai_pr_orchestrator.v3.model_router import build_model_broker
from ai_pr_orchestrator.v3.telemetry import ProviderResourceSnapshot, QuotaWindow, TelemetryRegistry
from tests.integration.test_continuous_foreman import _assert_completed_once, _scheduler


def _wire(loop, entries, snapshots=()):
    catalog = ModelCatalog(tuple(entries))
    loop._cfg = replace(
        loop._cfg, model_router=replace(loop._cfg.model_router, catalog=list(entries))
    )
    loop._executor._catalog = catalog
    by_resource = {s.resource: s for s in snapshots}
    source = SimpleNamespace(
        resources=lambda: tuple(by_resource), snapshot=lambda resource, **_: by_resource[resource]
    )
    loop._broker = build_model_broker(
        loop._cfg, telemetry_source=TelemetryRegistry([source]), catalog=catalog
    )


@pytest.mark.parametrize("reserve,expected", [(0.1, "near-reset"), (0.9, None)])
def test_scenario_7_dispatch_consumes_only_surplus_above_reserve(
    fake_cao, foreman_harness, reserve, expected
):
    template, queue, _ = foreman_harness([1, 2])
    base = template._executor._catalog.entries
    worker = next(e for e in base if e.ref == "ref-developer")
    subscriptions = [
        replace(
            worker,
            ref=name,
            provider=name,
            resource_class="subscription",
            cost_class="medium",
            input_price_per_mtok=1,
            output_price_per_mtok=1,
            roles=("worker",),
            quality_by_role={"worker": 5},
        )
        for name in ("near-reset", "patient")
    ]
    now = datetime.now(UTC)
    snapshots = [
        ProviderResourceSnapshot(
            resource=e.ref,
            observed_at=now,
            availability="available",
            resource_class="subscription",
            windows=(
                QuotaWindow("session", 0.2, now + delta),
                QuotaWindow("weekly", 0.2, now + delta),
            ),
        )
        for e, delta in zip(subscriptions, (timedelta(hours=1), timedelta(days=7)), strict=True)
    ]
    template._cfg = replace(
        template._cfg,
        broker=replace(
            template._cfg.broker,
            reserves=[ResourceReserveConfig(resource="near-reset", fraction=reserve)],
        ),
    )
    _wire(template, [*base, *subscriptions], snapshots)
    scheduler = _scheduler(template, queue, template._executor, fake_cao, broker=template._broker)
    assert scheduler.step()[0].final_phase == "done"
    state = _assert_completed_once(queue, fake_cao, 1)
    selected = state.extras["developer_model"]["model_ref"]
    routing = state.extras["routing_decisions"]["developer"]
    if expected:
        assert selected == expected
        scores = {row["ref"]: row["score"] for row in routing["ranked"]}
        assert scores["near-reset"]["perishability"] > scores["patient"]["perishability"]
    else:
        assert selected != "near-reset"
        assert any(
            row["ref"] == "near-reset" and "reserve" in row["reason"] for row in routing["rejected"]
        )


def test_scenario_8_expired_promotion_changes_next_issue_assignment(fake_cao, foreman_harness):
    template, queue, github = foreman_harness([1, 2])
    base = template._executor._catalog.entries
    worker = next(e for e in base if e.ref == "ref-developer")
    promo = replace(
        worker,
        ref="promo",
        provider="promo",
        resource_class="metered",
        cost_class="low",
        input_price_per_mtok=4,
        output_price_per_mtok=8,
        promotional=True,
        promo_ends_at=datetime.now(UTC) + timedelta(hours=1),
        roles=("worker",),
        quality_by_role={"worker": 5},
    )
    _wire(template, [*base, promo])
    first = _scheduler(template, queue, template._executor, fake_cao)
    assert first.step()[0].final_phase == "done"
    before = _assert_completed_once(queue, fake_cao, 1)
    _wire(template, [*base, replace(promo, promo_ends_at=datetime.now(UTC) - timedelta(seconds=1))])
    second = _scheduler(template, queue, template._executor, fake_cao)
    assert second.step()[0].final_phase == "done"
    after = _assert_completed_once(queue, fake_cao, 2)
    assert before.extras["developer_model"]["model_ref"] == "promo"
    assert after.extras["developer_model"]["model_ref"] != "promo"
    assert before.run_id != after.run_id and len(github.list_open_prs()) == 2
    assert queue.load_state("owner/repo#1") == before
