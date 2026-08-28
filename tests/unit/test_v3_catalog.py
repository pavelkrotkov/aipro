"""Unit tests for the V3 shared model catalog.

No test names a real vendor model. The catalog must be able to *express*
promotional free models, gateway endpoints, and subscription-backed
resources, but nothing in the catalog or routing code may special-case one,
so the fixtures use structural names only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_pr_orchestrator.v3.catalog import (
    ModelCatalog,
    ModelCatalogEntry,
    ModelCatalogError,
    load_model_catalog,
)

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def entry(**overrides: object) -> ModelCatalogEntry:
    base: dict[str, object] = {"ref": "candidate-a", "descriptor": "opaque-provider-string"}
    base.update(overrides)
    return ModelCatalogEntry(**base)  # ty: ignore[invalid-argument-type]


class TestEntryValidation:
    def test_minimal_entry_is_valid(self) -> None:
        assert entry().ref == "candidate-a"

    @pytest.mark.parametrize("ref,descriptor", [("", "d"), ("r", "")])
    def test_ref_and_descriptor_required(self, ref: str, descriptor: str) -> None:
        with pytest.raises(ModelCatalogError):
            ModelCatalogEntry(ref=ref, descriptor=descriptor)

    def test_unknown_resource_class_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown resource_class"):
            entry(resource_class="barter")

    def test_unknown_cost_class_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown cost_class"):
            entry(cost_class="cheapish")

    @pytest.mark.parametrize("field", ["input_price_per_mtok", "output_price_per_mtok"])
    def test_negative_price_rejected(self, field: str) -> None:
        with pytest.raises(ModelCatalogError, match="must be >= 0"):
            entry(**{field: -0.01})

    def test_free_cost_class_with_a_price_is_contradictory(self) -> None:
        with pytest.raises(ModelCatalogError, match="non-zero"):
            entry(cost_class="free", input_price_per_mtok=3.0)

    def test_free_tier_with_a_price_is_contradictory(self) -> None:
        with pytest.raises(ModelCatalogError, match="non-zero"):
            entry(resource_class="free_tier", output_price_per_mtok=1.0)

    def test_promo_window_must_be_ordered(self) -> None:
        with pytest.raises(ModelCatalogError, match="must be after"):
            entry(promotional=True, promo_starts_at=NOW, promo_ends_at=NOW - timedelta(days=1))

    def test_promo_window_without_promotional_flag_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="promotional is false"):
            entry(promo_ends_at=NOW)

    def test_malformed_timestamp_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="ISO 8601"):
            entry(promotional=True, promo_ends_at="the day after tomorrow")

    def test_non_timestamp_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="must be a timestamp"):
            entry(promotional=True, promo_ends_at=17)

    def test_naive_timestamp_is_read_as_utc(self) -> None:
        parsed = entry(promotional=True, promo_ends_at="2026-06-02T00:00:00")
        assert parsed.promo_ends_at == datetime(2026, 6, 2, tzinfo=UTC)

    def test_unknown_capability_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown capabilities"):
            entry(capabilities=["telepathy"])

    def test_coding_without_tools_is_impossible(self) -> None:
        # Lanes reach the repository through tool calls, so this combination
        # could never actually execute.
        with pytest.raises(ModelCatalogError, match="without 'tools'"):
            entry(capabilities=["coding"])

    def test_unknown_role_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown roles"):
            entry(roles=["adjudicator"])

    def test_quality_for_unlisted_role_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="does not list it in roles"):
            entry(roles=["worker"], quality_by_role={"reviewer": 4})

    def test_quality_out_of_range_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="quality for role"):
            entry(roles=["worker"], quality_by_role={"worker": 9})

    def test_quality_for_unknown_role_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown role"):
            entry(quality_by_role={"nobody": 3})

    @pytest.mark.parametrize("difficulty", [0, 6])
    def test_min_task_difficulty_out_of_range_rejected(self, difficulty: int) -> None:
        with pytest.raises(ModelCatalogError, match="min_task_difficulty"):
            entry(min_task_difficulty=difficulty)

    def test_non_positive_context_window_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="max_context_tokens"):
            entry(max_context_tokens=0)

    def test_non_positive_concurrency_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="max_concurrency"):
            entry(max_concurrency=0)

    def test_unknown_keys_are_preserved_for_forward_compatibility(self) -> None:
        parsed = ModelCatalogEntry.from_dict(
            {"ref": "r", "descriptor": "d", "future_knob": {"a": 1}}
        )
        assert parsed.extras == {"future_knob": {"a": 1}}
        assert parsed.to_dict()["future_knob"] == {"a": 1}

    def test_descriptor_is_never_parsed(self) -> None:
        assert entry(descriptor="anything::at-all/v9").descriptor == "anything::at-all/v9"


class TestPromotionWindow:
    def test_promotion_inactive_before_start(self) -> None:
        promo = entry(promotional=True, promo_starts_at=NOW + timedelta(hours=1), cost_class="low")
        assert promo.promotion_active(NOW) is False

    def test_promotion_active_inside_window(self) -> None:
        promo = entry(
            promotional=True,
            promo_starts_at=NOW - timedelta(hours=1),
            promo_ends_at=NOW + timedelta(hours=1),
        )
        assert promo.promotion_active(NOW) is True

    def test_promotion_inactive_at_and_after_end(self) -> None:
        promo = entry(promotional=True, promo_ends_at=NOW)
        assert promo.promotion_active(NOW) is False
        assert promo.promotion_active(NOW + timedelta(seconds=1)) is False

    def test_unbounded_promotion_is_active(self) -> None:
        assert entry(promotional=True).promotion_active(NOW) is True

    def test_active_promotion_prices_at_zero(self) -> None:
        promo = entry(
            promotional=True,
            promo_ends_at=NOW + timedelta(days=1),
            input_price_per_mtok=5.0,
            output_price_per_mtok=15.0,
        )
        assert promo.effective_prices(NOW) == (0.0, 0.0)

    def test_expired_promotion_reverts_to_list_price(self) -> None:
        promo = entry(
            promotional=True,
            promo_ends_at=NOW - timedelta(days=1),
            input_price_per_mtok=5.0,
            output_price_per_mtok=15.0,
        )
        assert promo.effective_prices(NOW) == (5.0, 15.0)
        assert promo.is_eligible(at=NOW) is True

    def test_expired_promotion_without_list_price_becomes_ineligible(self) -> None:
        # Its cash cost is now unknown, which the reserve/budget policy
        # cannot reason about — that is different from being free.
        promo = entry(promotional=True, promo_ends_at=NOW - timedelta(days=1))
        assert promo.effective_prices(NOW) is None
        assert promo.has_known_price(NOW) is False
        assert promo.is_eligible(at=NOW) is False

    def test_unknown_price_is_not_reported_as_zero(self) -> None:
        assert entry().effective_prices(NOW) is None


class TestEligibility:
    def test_disabled_entry_is_ineligible(self) -> None:
        assert entry(enabled=False, cost_class="free").is_eligible(at=NOW) is False

    def test_empty_roles_means_any_role(self) -> None:
        candidate = entry(cost_class="free")
        assert candidate.is_eligible(role="worker", at=NOW) is True
        assert candidate.is_eligible(role="reviewer", at=NOW) is True

    def test_role_filtering(self) -> None:
        reviewer_only = entry(cost_class="free", roles=["reviewer"])
        assert reviewer_only.is_eligible(role="reviewer", at=NOW) is True
        assert reviewer_only.is_eligible(role="worker", at=NOW) is False

    def test_difficulty_floor_reserves_an_entry_for_hard_work(self) -> None:
        premium = entry(
            cost_class="high",
            input_price_per_mtok=15.0,
            output_price_per_mtok=75.0,
            min_task_difficulty=4,
        )
        assert premium.is_eligible(difficulty=3, at=NOW) is False
        assert premium.is_eligible(difficulty=4, at=NOW) is True

    def test_free_tier_and_free_cost_class_price_at_zero(self) -> None:
        assert entry(resource_class="free_tier", cost_class="free").effective_prices(NOW) == (
            0.0,
            0.0,
        )

    def test_subscription_reports_list_price_not_zero(self) -> None:
        # Whether already-bought subscription capacity should be treated as
        # marginally free is a broker judgement (#47), not a catalog fact.
        subscription = entry(
            resource_class="subscription",
            cost_class="high",
            input_price_per_mtok=3.0,
            output_price_per_mtok=15.0,
        )
        assert subscription.effective_prices(NOW) == (3.0, 15.0)
        assert subscription.is_eligible(at=NOW) is True


class TestCatalog:
    def test_duplicate_refs_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="Duplicate model catalog refs"):
            ModelCatalog(entries=(entry(ref="dup"), entry(ref="dup")))

    def test_get_and_refs(self) -> None:
        catalog = ModelCatalog(entries=(entry(ref="a"), entry(ref="b")))
        assert catalog.refs() == ["a", "b"]
        assert catalog.get("b") is not None
        assert catalog.get("missing") is None

    def test_eligible_preserves_file_order_and_filters(self) -> None:
        catalog = ModelCatalog(
            entries=(
                entry(ref="free-promo", cost_class="free", roles=["worker", "reviewer"]),
                entry(ref="disabled", cost_class="free", enabled=False),
                entry(ref="hard-only", cost_class="free", min_task_difficulty=5),
                entry(ref="reviewer-only", cost_class="free", roles=["reviewer"]),
            )
        )
        assert [e.ref for e in catalog.eligible(role="worker", difficulty=2, at=NOW)] == [
            "free-promo"
        ]
        assert [e.ref for e in catalog.eligible(role="reviewer", difficulty=5, at=NOW)] == [
            "free-promo",
            "hard-only",
            "reviewer-only",
        ]

    def test_eligible_rejects_unknown_role(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown role"):
            ModelCatalog().eligible(role="adjudicator")

    @pytest.mark.parametrize("difficulty", [0, 6])
    def test_eligible_rejects_out_of_range_difficulty(self, difficulty: int) -> None:
        with pytest.raises(ModelCatalogError, match="difficulty must be within"):
            ModelCatalog().eligible(difficulty=difficulty)

    def test_round_trip(self) -> None:
        catalog = ModelCatalog(
            entries=(
                entry(
                    ref="a",
                    promotional=True,
                    promo_ends_at=NOW,
                    capabilities=["tools", "coding"],
                    roles=["worker"],
                    quality_by_role={"worker": 4},
                    source_updated_at=NOW,
                ),
            )
        )
        assert ModelCatalog.from_dict(catalog.to_dict()) == catalog

    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ModelCatalogError, match="unknown top-level keys"):
            ModelCatalog.from_dict({"models": [], "modles": []})

    def test_models_must_be_a_list(self) -> None:
        with pytest.raises(ModelCatalogError, match="must be a list"):
            ModelCatalog.from_dict({"models": {"a": 1}})


class TestLoading:
    def test_loads_the_documented_shapes(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text(
            """
models:
  # A promotional free model, offered below list price for a window.
  - ref: promo-free-coder
    descriptor: opaque-promo-descriptor
    provider: gateway
    resource_class: free_tier
    cost_class: free
    promotional: true
    promo_ends_at: 2026-12-31T00:00:00Z
    capabilities: [tools, coding]
    roles: [worker, reviewer]
    quality_by_role: {worker: 3, reviewer: 3}
    family: family-x
    vendor: vendor-x
  # A metered model behind a custom aggregator endpoint.
  - ref: gateway-metered
    descriptor: opaque-gateway-descriptor
    provider: custom-gateway
    endpoint: https://gateway.example/v1
    resource_class: metered
    cost_class: low
    input_price_per_mtok: 0.4
    output_price_per_mtok: 1.2
    capabilities: [tools, coding, long_context]
    max_context_tokens: 262144
  # Subscription-backed capacity, a normal primary candidate.
  - ref: subscription-primary
    descriptor: opaque-subscription-descriptor
    provider: subscription-provider
    resource_class: subscription
    cost_class: high
    input_price_per_mtok: 3.0
    output_price_per_mtok: 15.0
    capabilities: [tools, coding, reasoning]
    roles: [worker, reviewer, foreman]
    quality_by_role: {worker: 5, reviewer: 5, foreman: 4}
    min_task_difficulty: 2
    max_concurrency: 2
    family: family-y
    vendor: vendor-y
    data_policy: no-training
    source_updated_at: 2026-05-30T09:00:00Z
""",
            encoding="utf-8",
        )
        catalog = load_model_catalog(path)
        assert catalog.refs() == ["promo-free-coder", "gateway-metered", "subscription-primary"]
        gateway = catalog.get("gateway-metered")
        subscription = catalog.get("subscription-primary")
        assert gateway is not None and subscription is not None
        assert gateway.endpoint == "https://gateway.example/v1"
        assert subscription.quality_for("worker") == 5
        # The difficulty floor keeps the subscription entry off trivial work,
        # but it returns for difficulty 2. `gateway-metered` declares no roles
        # and so suits any.
        assert [e.ref for e in catalog.eligible(role="worker", difficulty=1, at=NOW)] == [
            "promo-free-coder",
            "gateway-metered",
        ]
        assert [e.ref for e in catalog.eligible(role="worker", difficulty=2, at=NOW)] == [
            "promo-free-coder",
            "gateway-metered",
            "subscription-primary",
        ]

    def test_reload_yields_an_independent_catalog(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text("models: [{ref: a, descriptor: d}]", encoding="utf-8")
        first = load_model_catalog(path)
        path.write_text("models: [{ref: b, descriptor: d}]", encoding="utf-8")
        second = load_model_catalog(path)
        # The already-loaded catalog is untouched, so an edit can never
        # re-point a phase that is already running.
        assert first.refs() == ["a"]
        assert second.refs() == ["b"]

    def test_empty_file_is_an_empty_catalog(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text("", encoding="utf-8")
        assert load_model_catalog(path).refs() == []

    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ModelCatalogError, match="Failed to read model catalog"):
            load_model_catalog(tmp_path / "nope.yml")

    def test_invalid_yaml_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text("models: [", encoding="utf-8")
        with pytest.raises(ModelCatalogError, match="Invalid YAML"):
            load_model_catalog(path)

    def test_non_mapping_top_level_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text("- a\n- b", encoding="utf-8")
        with pytest.raises(ModelCatalogError, match="must be a YAML mapping"):
            load_model_catalog(path)

    def test_duplicate_refs_in_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text(
            "models: [{ref: a, descriptor: d}, {ref: a, descriptor: e}]", encoding="utf-8"
        )
        with pytest.raises(ModelCatalogError, match="Duplicate model catalog refs"):
            load_model_catalog(path)

    def test_malformed_field_shape_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.yml"
        path.write_text("models: [{ref: a, descriptor: d, capabilities: tools}]", encoding="utf-8")
        with pytest.raises(ModelCatalogError, match="capabilities"):
            load_model_catalog(path)

    def test_shipped_sample_catalog_is_valid(self) -> None:
        sample = Path(__file__).resolve().parents[2] / "examples" / "model-catalog.yml"
        catalog = load_model_catalog(sample)
        assert catalog.refs(), "sample catalog should declare candidates"
