"""Tests for the V3 provider telemetry domain."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.telemetry import (
    CatalogTelemetrySource,
    ProviderHealth,
    ProviderHealthLedger,
    ProviderResourceSnapshot,
    QuotaWindow,
    TelemetryError,
    TelemetryRegistry,
    redact_secrets,
    unknown_snapshot,
    windows_fully_spent,
)

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def require[T](value: T | None) -> T:
    """Assert an optional lookup found something, and narrow it for the checker."""
    assert value is not None
    return value


def snapshot(**overrides) -> ProviderResourceSnapshot:
    kwargs: dict[str, Any] = {
        "resource": "anthropic-sub",
        "observed_at": NOW,
        "availability": "available",
    }
    kwargs.update(overrides)
    return ProviderResourceSnapshot(**kwargs)


class TestQuotaWindow:
    def test_requires_a_label(self):
        with pytest.raises(TelemetryError, match="label"):
            QuotaWindow(label="")

    def test_rejects_a_naive_reset_at(self):
        with pytest.raises(TelemetryError, match="timezone-aware"):
            QuotaWindow(label="Weekly", reset_at=datetime(2026, 8, 28, 12, 0))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5])
    def test_rejects_a_nonsensical_used_fraction(self, bad):
        with pytest.raises(TelemetryError, match="used_fraction"):
            QuotaWindow(label="Weekly", used_fraction=bad)

    def test_unknown_usage_is_not_zero_usage(self):
        window = QuotaWindow(label="Weekly")
        assert window.used_fraction is None
        assert window.remaining_fraction is None
        assert window.is_exhausted is False

    def test_over_full_usage_is_preserved_but_never_negative_remaining(self):
        # Providers do report >100%; that is data, not a schema violation.
        window = QuotaWindow(label="Weekly", used_fraction=1.05)
        assert window.used_fraction == 1.05
        assert window.remaining_fraction == 0.0
        assert window.is_exhausted is True

    def test_time_to_reset_is_derived_and_never_negative(self):
        window = QuotaWindow(label="Weekly", reset_at=NOW + timedelta(hours=2))
        assert window.time_to_reset(NOW) == timedelta(hours=2)
        assert window.time_to_reset(NOW + timedelta(hours=5)) == timedelta(0)

    def test_time_to_reset_is_unknown_without_an_absolute_reset(self):
        assert QuotaWindow(label="API key quota").time_to_reset(NOW) is None


class TestSnapshotInvariants:
    def test_requires_a_timezone_aware_observed_at(self):
        with pytest.raises(TelemetryError, match="timezone-aware"):
            ProviderResourceSnapshot(resource="r", observed_at=datetime(2026, 8, 28, 12, 0))

    def test_rejects_an_unknown_availability(self):
        with pytest.raises(TelemetryError, match="availability"):
            snapshot(availability="probably-fine")

    def test_exhausted_requires_evidence(self):
        # The whole point of the type: a telemetry failure can never be
        # dressed up as an exhausted quota.
        with pytest.raises(TelemetryError, match="evidence"):
            snapshot(availability="exhausted")

    def test_exhausted_is_accepted_with_a_spent_window(self):
        snap = snapshot(
            availability="exhausted",
            windows=(QuotaWindow(label="Weekly", used_fraction=1.0),),
        )
        assert snap.availability == "exhausted"

    def test_exhausted_is_accepted_with_a_zero_cash_balance(self):
        assert snapshot(availability="exhausted", cash_balance=0.0).availability == "exhausted"

    def test_available_cannot_contradict_a_spent_window(self):
        with pytest.raises(TelemetryError, match="available"):
            snapshot(
                availability="available",
                windows=(QuotaWindow(label="Weekly", used_fraction=1.0),),
            )

    def test_one_spent_window_among_several_is_not_exhaustion(self):
        # A spent 5-hour session alongside a 74% week is normal operation: the
        # session returns within hours. Calling that 'exhausted' would take a
        # resource with real headroom out of rotation until the weekly reset.
        snap = snapshot(
            availability="available",
            windows=(
                QuotaWindow(label="Current session", used_fraction=1.0),
                QuotaWindow(label="Current week", used_fraction=0.74),
            ),
        )
        assert snap.availability == "available"
        assert [w.label for w in snap.spent_windows()] == ["Current session"]

    def test_exhausted_needs_every_measured_window_spent(self):
        with pytest.raises(TelemetryError, match="evidence"):
            snapshot(
                availability="exhausted",
                windows=(
                    QuotaWindow(label="Current session", used_fraction=1.0),
                    QuotaWindow(label="Current week", used_fraction=0.74),
                ),
            )

    def test_unmeasured_windows_are_ignored_not_assumed_spent(self):
        snap = snapshot(
            availability="exhausted",
            windows=(
                QuotaWindow(label="Current week", used_fraction=1.0),
                QuotaWindow(label="Opus week"),
            ),
        )
        assert snap.availability == "exhausted"

    def test_windows_alone_cannot_be_evidence_when_none_are_measured(self):
        assert windows_fully_spent(()) is False
        assert windows_fully_spent((QuotaWindow(label="Opus week"),)) is False
        assert windows_fully_spent((QuotaWindow(label="W", used_fraction=1.0),)) is True

    @pytest.mark.parametrize("availability", ["unavailable", "unknown"])
    def test_non_usable_states_must_say_why(self, availability):
        with pytest.raises(TelemetryError, match="reason"):
            snapshot(availability=availability)

    def test_duplicate_window_labels_are_rejected(self):
        with pytest.raises(TelemetryError, match="duplicate"):
            snapshot(
                windows=(
                    QuotaWindow(label="Weekly", used_fraction=0.1),
                    QuotaWindow(label="Weekly", used_fraction=0.2),
                )
            )

    def test_rejects_an_unknown_resource_class(self):
        with pytest.raises(TelemetryError, match="resource_class"):
            snapshot(resource_class="vibes")

    @pytest.mark.parametrize("bad", [float("nan"), -1.0])
    def test_rejects_a_nonsensical_cash_balance(self, bad):
        with pytest.raises(TelemetryError, match="cash_balance"):
            snapshot(cash_balance=bad)

    def test_windows_are_deeply_immutable(self):
        snap = snapshot(windows=[QuotaWindow(label="Weekly", used_fraction=0.5)])
        assert isinstance(snap.windows, tuple)
        with pytest.raises((AttributeError, TypeError)):
            snap.windows.append(QuotaWindow(label="Session"))  # ty: ignore[unresolved-attribute]


class TestSnapshotQueries:
    def test_multiple_windows_are_preserved_separately(self):
        snap = snapshot(
            windows=(
                QuotaWindow(label="Current session", used_fraction=0.63),
                QuotaWindow(label="Current week", used_fraction=0.71),
                QuotaWindow(label="Opus week", used_fraction=0.2),
            )
        )
        assert [w.label for w in snap.windows] == [
            "Current session",
            "Current week",
            "Opus week",
        ]
        assert require(snap.window("Opus week")).used_fraction == 0.2
        assert snap.window("nope") is None

    def test_tightest_window_is_the_one_with_least_headroom(self):
        snap = snapshot(
            windows=(
                QuotaWindow(label="Current session", used_fraction=0.63),
                QuotaWindow(label="Current week", used_fraction=0.91),
                QuotaWindow(label="Unmeasured"),
            )
        )
        assert require(snap.tightest_window()).label == "Current week"

    def test_tightest_window_is_unknown_when_no_window_reports_usage(self):
        assert snapshot(windows=(QuotaWindow(label="Unmeasured"),)).tightest_window() is None

    def test_next_reset_is_the_earliest_absolute_reset(self):
        snap = snapshot(
            windows=(
                QuotaWindow(label="Current week", reset_at=NOW + timedelta(days=1)),
                QuotaWindow(label="Current session", reset_at=NOW + timedelta(hours=5)),
                QuotaWindow(label="API key quota"),
            )
        )
        assert snap.next_reset_at() == NOW + timedelta(hours=5)

    def test_staleness_is_marked_once_the_ttl_lapses(self):
        snap = snapshot(ttl_seconds=300)
        assert snap.is_stale(NOW + timedelta(seconds=299)) is False
        assert snap.is_stale(NOW + timedelta(seconds=301)) is True
        assert snap.age(NOW + timedelta(seconds=60)) == timedelta(seconds=60)

    def test_without_a_ttl_staleness_is_unknown_rather_than_false(self):
        assert snapshot(ttl_seconds=None).is_stale(NOW + timedelta(days=9)) is None

    def test_round_trips_through_a_mapping(self):
        snap = snapshot(
            windows=(QuotaWindow(label="Current week", used_fraction=0.71, reset_at=NOW),),
            ttl_seconds=300,
            details=("Credits balance: $7.19",),
        )
        data = snap.to_dict()
        assert data["windows"][0]["reset_at"] == NOW.isoformat()
        assert data["availability"] == "available"

    def test_serialization_keeps_the_derived_answers_the_types_compute(self):
        # A consumer reading --json must not have to re-derive what the domain
        # already knows; a generic dataclass walk drops these silently.
        ledger = ProviderHealthLedger()
        ledger.record("anthropic-sub", "success", at=NOW, latency_ms=12.0)
        snap = snapshot(
            windows=(
                QuotaWindow(label="Current session", used_fraction=1.0, reset_at=NOW),
                QuotaWindow(label="Current week", used_fraction=0.74),
            ),
        ).with_health(ledger.health("anthropic-sub"))
        data = snap.to_dict()

        assert data["windows"][0]["remaining_fraction"] == 0.0
        assert data["windows"][0]["is_exhausted"] is True
        assert data["windows"][1]["remaining_fraction"] == pytest.approx(0.26)
        assert data["health"]["total"] == 1
        assert data["health"]["failure_rate"] == 0.0
        assert data["health"]["mean_latency_ms"] == 12.0
        assert data["health"]["p50_latency_ms"] == 12.0
        assert data["tightest_window"] == "Current session"
        assert data["next_reset_at"] == NOW.isoformat()
        assert data["spent_windows"] == ["Current session"]


class TestHealthDoesNotCorruptQuota:
    def test_recording_outcomes_leaves_quota_windows_untouched(self):
        snap = snapshot(windows=(QuotaWindow(label="Current week", used_fraction=0.71),))
        ledger = ProviderHealthLedger()
        for _ in range(3):
            ledger.record("anthropic-sub", "server_error", at=NOW)
        updated = snap.with_health(ledger.health("anthropic-sub"))
        assert updated.windows == snap.windows
        assert updated.availability == "available"
        assert updated.health.consecutive_failures == 3

    def test_a_transient_429_throttles_without_claiming_exhaustion(self):
        # A rate limit is a health signal. Only quota evidence may make a
        # resource 'exhausted', or a burst of 429s would look like a spent
        # subscription and take the resource out of rotation until reset.
        ledger = ProviderHealthLedger()
        ledger.record(
            "anthropic-sub", "rate_limited", at=NOW, retry_after=NOW + timedelta(seconds=30)
        )
        snap = snapshot(
            windows=(QuotaWindow(label="Current week", used_fraction=0.1),),
        ).with_health(ledger.health("anthropic-sub"))
        assert snap.availability == "available"
        assert snap.health.is_throttled(NOW) is True
        assert snap.health.is_throttled(NOW + timedelta(minutes=1)) is False

    def test_failure_rate_is_unknown_before_any_request(self):
        assert ProviderHealth().failure_rate is None

    def test_ledger_forgets_beyond_its_window(self):
        ledger = ProviderHealthLedger(window_size=3)
        for outcome in ("timeout", "timeout", "success", "success", "success"):
            ledger.record("r", outcome, at=NOW)
        health = ledger.health("r")
        assert health.total == 3
        assert health.failure_rate == 0.0
        assert health.consecutive_failures == 0

    def test_consecutive_failures_reset_on_success(self):
        ledger = ProviderHealthLedger()
        ledger.record("r", "timeout", at=NOW)
        ledger.record("r", "timeout", at=NOW)
        assert ledger.health("r").consecutive_failures == 2
        ledger.record("r", "success", at=NOW)
        assert ledger.health("r").consecutive_failures == 0

    def test_latency_statistics_are_summarized(self):
        ledger = ProviderHealthLedger()
        for latency in (10.0, 20.0, 60.0):
            ledger.record("r", "success", at=NOW, latency_ms=latency)
        health = ledger.health("r")
        assert health.p50_latency_ms == 20.0
        assert health.mean_latency_ms == 30.0

    def test_rejects_an_unknown_outcome(self):
        with pytest.raises(TelemetryError, match="outcome"):
            ProviderHealthLedger().record("r", "exploded", at=NOW)

    def test_health_for_an_unseen_resource_is_empty_not_an_error(self):
        assert ProviderHealthLedger().health("never-called").total == 0


class TestCatalogTelemetrySource:
    def _catalog(self) -> ModelCatalog:
        return ModelCatalog(
            entries=(
                ModelCatalogEntry(
                    ref="free-tier-helper",
                    descriptor="vendor/free",
                    resource_class="free_tier",
                    cost_class="free",
                ),
                ModelCatalogEntry(
                    ref="promo-generalist",
                    descriptor="vendor/promo",
                    resource_class="metered",
                    cost_class="low",
                    promotional=True,
                    promo_ends_at=NOW + timedelta(days=30),
                    input_price_per_mtok=1.0,
                    output_price_per_mtok=2.0,
                ),
                ModelCatalogEntry(ref="plain-metered", descriptor="vendor/plain"),
            )
        )

    def test_only_free_and_promotional_entries_are_telemetry_resources(self):
        source = CatalogTelemetrySource(self._catalog())
        assert source.resources() == ("free-tier-helper", "promo-generalist")

    def test_a_promotion_end_is_an_expiry_not_a_quota_reset(self):
        snap = CatalogTelemetrySource(self._catalog()).snapshot("promo-generalist", at=NOW)
        assert snap.availability == "available"
        assert snap.expires_at == NOW + timedelta(days=30)
        assert snap.next_reset_at() is None

    def test_an_entry_whose_price_became_unknown_is_unavailable_with_a_reason(self):
        catalog = ModelCatalog(
            entries=(
                ModelCatalogEntry(
                    ref="lapsed-promo",
                    descriptor="vendor/lapsed",
                    promotional=True,
                    promo_ends_at=NOW - timedelta(days=1),
                ),
            )
        )
        snap = CatalogTelemetrySource(catalog).snapshot("lapsed-promo", at=NOW)
        assert snap.availability == "unavailable"
        assert snap.reason

    def test_a_closed_promotion_reports_no_expiry_rather_than_a_past_one(self):
        # expires_at means "capacity disappears then". A timestamp in the past
        # reads as "expiring soon" to anything sorting on the column.
        catalog = ModelCatalog(
            entries=(
                ModelCatalogEntry(
                    ref="lapsed-promo",
                    descriptor="vendor/lapsed",
                    cost_class="low",
                    promotional=True,
                    promo_ends_at=NOW - timedelta(days=1),
                    input_price_per_mtok=1.0,
                    output_price_per_mtok=2.0,
                ),
            )
        )
        snap = CatalogTelemetrySource(catalog).snapshot("lapsed-promo", at=NOW)
        assert snap.expires_at is None
        assert "promotion inactive" in snap.details

    def test_an_unknown_ref_degrades_to_unknown_rather_than_raising(self):
        snap = CatalogTelemetrySource(self._catalog()).snapshot("not-here", at=NOW)
        assert snap.availability == "unknown"
        assert "not-here" in snap.reason


class TestTelemetryRegistry:
    def test_fans_out_over_every_configured_resource_in_one_call(self):
        registry = TelemetryRegistry(sources=[CatalogTelemetrySource(self._two_entry_catalog())])
        snapshots = registry.snapshot_all(at=NOW)
        assert [s.resource for s in snapshots] == ["free-a", "free-b"]
        assert {s.observed_at for s in snapshots} == {NOW}

    def _two_entry_catalog(self) -> ModelCatalog:
        return ModelCatalog(
            entries=(
                ModelCatalogEntry(ref="free-a", descriptor="v/a", cost_class="free"),
                ModelCatalogEntry(ref="free-b", descriptor="v/b", cost_class="free"),
            )
        )

    def test_a_source_that_raises_degrades_to_unknown_not_to_zero_quota(self):
        class Exploding:
            def resources(self):
                return ("boom",)

            def snapshot(self, resource, *, at=None):
                raise RuntimeError("provider library blew up")

        snap = TelemetryRegistry(sources=[Exploding()]).snapshot_all(at=NOW)[0]
        assert snap.availability == "unknown"
        assert "provider library blew up" in snap.reason
        assert snap.windows == ()

    def test_health_is_attached_from_the_ledger(self):
        ledger = ProviderHealthLedger()
        ledger.record("free-a", "success", at=NOW, latency_ms=12.0)
        registry = TelemetryRegistry(
            sources=[CatalogTelemetrySource(self._two_entry_catalog())], ledger=ledger
        )
        by_ref = {s.resource: s for s in registry.snapshot_all(at=NOW)}
        assert by_ref["free-a"].health.total == 1
        assert by_ref["free-b"].health.total == 0

    def test_duplicate_resources_across_sources_are_rejected(self):
        source = CatalogTelemetrySource(self._two_entry_catalog())
        with pytest.raises(TelemetryError, match="free-a"):
            TelemetryRegistry(sources=[source, source])


class TestRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer sk-ant-oat01-abcdefghijklmnop",
            "failed for https://user:hunter2@gateway.example/v1/credits",
            "key sk-proj-AAAABBBBCCCCDDDDEEEE rejected",
        ],
    )
    def test_credential_material_never_survives_redaction(self, text):
        cleaned = redact_secrets(text)
        for secret in ("sk-ant-oat01-abcdefghijklmnop", "hunter2", "sk-proj-AAAABBBBCCCCDDDDEEEE"):
            assert secret not in cleaned
        assert "REDACTED" in cleaned

    @pytest.mark.parametrize(
        ("text", "secret"),
        [
            ("api_key=9f8e7d6c5b4a3210 rejected", "9f8e7d6c5b4a3210"),
            ('{"access_token": "eyJhbGciOiJIUzI1NiJ9"}', "eyJhbGciOiJIUzI1NiJ9"),
            ("X-Api-Key: ZmFrZS1zZWNyZXQtdmFsdWU", "ZmFrZS1zZWNyZXQtdmFsdWU"),
            ("password: hunter2 was refused", "hunter2"),
            ("refresh_token 1//0gAbCdEfGhIjKl", "1//0gAbCdEfGhIjKl"),
        ],
    )
    def test_credentials_are_caught_by_key_name_not_only_by_vendor_prefix(self, text, secret):
        # Provider error text quotes whole requests; a value is credential
        # material because of the key it sits under, whatever its shape.
        cleaned = redact_secrets(text)
        assert secret not in cleaned
        assert "REDACTED" in cleaned

    def test_ordinary_text_is_left_alone(self):
        assert redact_secrets("Current week 71% used") == "Current week 71% used"
        assert redact_secrets("Credits balance: $7.19") == "Credits balance: $7.19"


def test_unknown_snapshot_is_never_mistaken_for_an_empty_quota():
    snap = unknown_snapshot("some-resource", reason="probe timed out", at=NOW)
    assert snap.availability == "unknown"
    assert snap.windows == ()
    assert snap.cash_balance is None
    assert snap.reason == "probe timed out"
