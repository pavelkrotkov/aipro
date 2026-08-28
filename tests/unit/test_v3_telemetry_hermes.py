"""Tests for the Hermes account-usage telemetry adapter.

The payloads below are recorded from a live run of the bridge script against
a real Hermes install (credentials elided), so the normalization is tested
against shapes Hermes actually emits rather than shapes we wish it emitted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_pr_orchestrator.v3.telemetry import (
    ProviderHealthLedger,
    TelemetryError,
    TelemetryRegistry,
)
from ai_pr_orchestrator.v3.telemetry_hermes import (
    BRIDGE_SCRIPT,
    HermesResource,
    HermesTelemetrySource,
    normalize_probe_result,
)

NOW = datetime(2026, 8, 28, 14, 0, tzinfo=UTC)


def require[T](value: T | None) -> T:
    """Assert an optional lookup found something, and narrow it for the checker."""
    assert value is not None
    return value


# --- Recorded payloads -----------------------------------------------------

CODEX_OK = {
    "ok": True,
    "provider": "openai-codex",
    "fidelity": "private",
    "snapshot": {
        "provider": "openai-codex",
        "source": "usage_api",
        "fetched_at": "2026-08-28T13:55:41.974743+00:00",
        "title": "Account limits",
        "plan": "Plus",
        "windows": [
            {
                "label": "Session",
                "used_percent": 42.0,
                "reset_at": "2026-08-28T17:19:59.906613+00:00",
                "detail": None,
            },
            {
                "label": "Weekly",
                "used_percent": 88.5,
                "reset_at": "2026-09-01T00:00:00+00:00",
                "detail": None,
            },
        ],
        "details": ["You have 2 resets banked - use /usage reset to activate"],
        "unavailable_reason": None,
    },
}

ANTHROPIC_OK = {
    "ok": True,
    "provider": "anthropic",
    "fidelity": "private",
    "snapshot": {
        "provider": "anthropic",
        "source": "oauth_usage_api",
        "fetched_at": "2026-08-28T13:55:41.974743+00:00",
        "title": "Account limits",
        "plan": None,
        "windows": [
            {
                "label": "Current session",
                "used_percent": 63.0,
                "reset_at": "2026-08-28T17:19:59.906613+00:00",
                "detail": None,
            },
            {
                "label": "Current week",
                "used_percent": 71.0,
                "reset_at": "2026-08-29T17:59:59.906633+00:00",
                "detail": None,
            },
            {
                "label": "Opus week",
                "used_percent": 12.0,
                "reset_at": "2026-08-29T17:59:59.906633+00:00",
                "detail": None,
            },
            {
                "label": "Sonnet week",
                "used_percent": 55.0,
                "reset_at": "2026-08-29T17:59:59.906633+00:00",
                "detail": None,
            },
        ],
        "details": [],
        "unavailable_reason": None,
    },
}

OPENROUTER_OK = {
    "ok": True,
    "provider": "openrouter",
    "fidelity": "private",
    "snapshot": {
        "provider": "openrouter",
        "source": "credits_api",
        "fetched_at": "2026-08-28T13:55:42.739057+00:00",
        "title": "Account limits",
        "plan": None,
        "windows": [],
        "details": [
            "Credits balance: $7.19",
            "API key usage: $2.81 total \u2022 $0.66 today \u2022 $2.81 this week",
        ],
        "unavailable_reason": None,
    },
}

CODEX_AUTH_FAILURE = {
    "ok": False,
    "provider": "openai-codex",
    "kind": "auth_failure",
    "message": (
        "HTTPStatusError: Client error '401 Unauthorized' for url "
        "'https://chatgpt.com/backend-api/wham/usage'"
    ),
    "status_code": 401,
}

NO_DATA = {"ok": True, "provider": "nope-provider", "fidelity": "public", "snapshot": None}


def _anthropic_snapshot() -> dict[str, Any]:
    snapshot = ANTHROPIC_OK["snapshot"]
    assert isinstance(snapshot, dict)
    return snapshot


def _normalize(result, **kwargs):
    resource = kwargs.pop("resource", HermesResource(name="res", provider="openai-codex"))
    return normalize_probe_result(result, resource=resource, at=NOW, **kwargs)


class TestNormalizingRecordedPayloads:
    def test_codex_session_and_weekly_windows_are_kept_separate(self):
        snap = _normalize(CODEX_OK)
        assert snap.availability == "available"
        assert [w.label for w in snap.windows] == ["Session", "Weekly"]
        assert require(snap.window("Session")).used_fraction == pytest.approx(0.42)
        assert require(snap.window("Weekly")).used_fraction == pytest.approx(0.885)
        assert require(snap.tightest_window()).label == "Weekly"
        assert snap.plan == "Plus"
        assert snap.source == "hermes:usage_api"

    def test_absolute_reset_times_survive_as_aware_datetimes(self):
        snap = _normalize(CODEX_OK)
        session = require(snap.window("Session"))
        assert session.reset_at == datetime(2026, 8, 28, 17, 19, 59, 906613, tzinfo=UTC)
        # Derived, not stored: computed against the query time, not fetch time.
        assert session.time_to_reset(NOW) == session.reset_at - NOW
        assert snap.next_reset_at() == session.reset_at

    def test_observed_at_is_the_providers_fetch_time_not_our_clock(self):
        # Freshness must be measured from when the data was actually read.
        snap = _normalize(CODEX_OK)
        assert snap.observed_at == datetime(2026, 8, 28, 13, 55, 41, 974743, tzinfo=UTC)
        assert snap.age(NOW) > timedelta(0)

    def test_anthropic_model_specific_weekly_windows_are_preserved(self):
        snap = _normalize(
            ANTHROPIC_OK, resource=HermesResource(name="anthropic-sub", provider="anthropic")
        )
        assert [w.label for w in snap.windows] == [
            "Current session",
            "Current week",
            "Opus week",
            "Sonnet week",
        ]
        assert require(snap.window("Opus week")).used_fraction == pytest.approx(0.12)
        assert require(snap.window("Sonnet week")).used_fraction == pytest.approx(0.55)
        assert snap.availability == "available"

    def test_openrouter_has_no_windows_and_therefore_no_reset(self):
        snap = _normalize(
            OPENROUTER_OK,
            resource=HermesResource(
                name="openrouter", provider="openrouter", resource_class="metered"
            ),
        )
        assert snap.windows == ()
        assert snap.next_reset_at() is None
        assert snap.availability == "available"

    def test_openrouter_cash_balance_stays_unknown_rather_than_scraped(self):
        # Hermes only renders the balance into a human-facing detail string;
        # parsing it back out would be scraping rendered text, which is the
        # thing this data path exists to avoid. Unknown is the honest answer.
        snap = _normalize(
            OPENROUTER_OK,
            resource=HermesResource(
                name="openrouter", provider="openrouter", resource_class="metered"
            ),
        )
        assert snap.cash_balance is None
        assert "Credits balance: $7.19" in snap.details


class TestFailureModesStayDistinguishable:
    def test_a_full_window_is_exhausted_not_merely_unavailable(self):
        payload = _with_window_usage(CODEX_OK, "Weekly", 100.0)
        snap = _normalize(payload)
        assert snap.availability == "exhausted"
        assert require(snap.window("Weekly")).is_exhausted is True
        # Still recoverable, and the snapshot says exactly when.
        assert require(snap.window("Weekly")).reset_at is not None

    def test_an_auth_failure_is_unavailable_not_unknown(self):
        # It is durable and needs a human; a retry will not fix it.
        snap = _normalize(CODEX_AUTH_FAILURE)
        assert snap.availability == "unavailable"
        assert "401" in snap.reason

    def test_a_transient_429_is_unknown_and_never_exhausted(self):
        snap = _normalize({"ok": False, "provider": "p", "kind": "rate_limited", "message": "429"})
        assert snap.availability == "unknown"
        assert snap.windows == ()

    @pytest.mark.parametrize("kind", ["server_error", "timeout", "error"])
    def test_transport_failures_degrade_to_unknown(self, kind):
        snap = _normalize({"ok": False, "provider": "p", "kind": kind, "message": "boom"})
        assert snap.availability == "unknown"
        assert "boom" in snap.reason

    def test_no_credentials_is_unavailable_with_an_actionable_reason(self):
        snap = _normalize(NO_DATA)
        assert snap.availability == "unavailable"
        assert "no account-usage data" in snap.reason

    def test_hermes_own_unavailable_reason_is_carried_through(self):
        payload = {
            "ok": True,
            "provider": "anthropic",
            "snapshot": {
                **_anthropic_snapshot(),
                "windows": [],
                "details": [],
                "unavailable_reason": (
                    "Anthropic account limits are only available for OAuth-backed accounts."
                ),
            },
        }
        snap = _normalize(payload, resource=HermesResource(name="a", provider="anthropic"))
        assert snap.availability == "unavailable"
        assert "OAuth-backed" in snap.reason

    def test_an_empty_snapshot_is_unknown_rather_than_available(self):
        payload = {
            "ok": True,
            "provider": "anthropic",
            "snapshot": {**_anthropic_snapshot(), "windows": [], "details": []},
        }
        snap = _normalize(payload, resource=HermesResource(name="a", provider="anthropic"))
        assert snap.availability == "unknown"

    def test_a_malformed_payload_degrades_to_unknown_rather_than_raising(self):
        snap = _normalize({"ok": True, "snapshot": {"windows": "not-a-list"}})
        assert snap.availability == "unknown"

    def test_an_unparseable_reset_does_not_discard_the_whole_window(self):
        payload = _with_window_field(CODEX_OK, "Session", "reset_at", "not-a-timestamp")
        snap = _normalize(payload)
        assert require(snap.window("Session")).used_fraction == pytest.approx(0.42)
        assert require(snap.window("Session")).reset_at is None

    def test_credentials_echoed_by_a_provider_error_are_redacted(self):
        snap = _normalize(
            {
                "ok": False,
                "provider": "p",
                "kind": "auth_failure",
                "message": "rejected Authorization: Bearer sk-ant-oat01-supersecretvalue",
            }
        )
        assert "supersecretvalue" not in snap.reason
        assert "REDACTED" in snap.reason


class TestHermesTelemetrySource:
    def _source(self, results, *, ledger=None, ttl_seconds=300.0):
        class FakeProbe:
            calls = 0

            def probe(self, providers):
                FakeProbe.calls += 1
                return {p: results[p] for p in providers if p in results}

        self.probe = FakeProbe()
        return HermesTelemetrySource(
            resources=[
                HermesResource(name="codex-sub", provider="openai-codex"),
                HermesResource(name="anthropic-sub", provider="anthropic"),
            ],
            probe=self.probe,
            ttl_seconds=ttl_seconds,
            ledger=ledger,
        )

    def test_one_probe_serves_every_configured_resource(self):
        source = self._source({"openai-codex": CODEX_OK, "anthropic": ANTHROPIC_OK})
        snapshots = TelemetryRegistry(sources=[source]).snapshot_all(at=NOW)
        assert [s.resource for s in snapshots] == ["codex-sub", "anthropic-sub"]
        assert type(self.probe).calls == 1

    def test_snapshots_are_reused_until_the_ttl_lapses(self):
        source = self._source({"openai-codex": CODEX_OK, "anthropic": ANTHROPIC_OK})
        source.snapshot("codex-sub", at=NOW)
        source.snapshot("codex-sub", at=NOW + timedelta(seconds=10))
        assert type(self.probe).calls == 1
        source.snapshot("codex-sub", at=NOW + timedelta(seconds=400))
        assert type(self.probe).calls == 2

    def test_a_provider_missing_from_the_probe_result_is_unknown(self):
        source = self._source({"openai-codex": CODEX_OK})
        snap = source.snapshot("anthropic-sub", at=NOW)
        assert snap.availability == "unknown"

    def test_a_probe_that_raises_degrades_every_resource_to_unknown(self):
        class Exploding:
            def probe(self, providers):
                raise OSError("hermes venv is gone")

        source = HermesTelemetrySource(
            resources=[HermesResource(name="codex-sub", provider="openai-codex")],
            probe=Exploding(),
        )
        snap = source.snapshot("codex-sub", at=NOW)
        assert snap.availability == "unknown"
        assert "hermes venv is gone" in snap.reason

    def test_probe_outcomes_feed_provider_health(self):
        ledger = ProviderHealthLedger()
        source = self._source(
            {"openai-codex": CODEX_AUTH_FAILURE, "anthropic": ANTHROPIC_OK}, ledger=ledger
        )
        registry = TelemetryRegistry(sources=[source], ledger=ledger)
        by_name = {s.resource: s for s in registry.snapshot_all(at=NOW)}
        assert by_name["codex-sub"].health.outcomes == {"auth_failure": 1}
        assert by_name["anthropic-sub"].health.outcomes == {"success": 1}

    def test_health_never_rewrites_the_quota_windows(self):
        ledger = ProviderHealthLedger()
        source = self._source({"anthropic": ANTHROPIC_OK}, ledger=ledger)
        for _ in range(4):
            ledger.record("anthropic-sub", "rate_limited", at=NOW)
        snap = TelemetryRegistry(sources=[source], ledger=ledger).snapshot("anthropic-sub", at=NOW)
        assert snap.availability == "available"
        assert require(snap.window("Current week")).used_fraction == pytest.approx(0.71)
        assert snap.health.outcomes["rate_limited"] == 4

    def test_duplicate_resource_names_are_rejected(self):
        with pytest.raises(TelemetryError, match="dup"):
            HermesTelemetrySource(
                resources=[
                    HermesResource(name="dup", provider="anthropic"),
                    HermesResource(name="dup", provider="openai-codex"),
                ],
                probe=_StaticProbe({}),
            )

    def test_per_resource_ttl_overrides_the_source_default(self):
        source = HermesTelemetrySource(
            resources=[HermesResource(name="codex-sub", provider="openai-codex", ttl_seconds=30.0)],
            probe=_StaticProbe({"openai-codex": CODEX_OK}),
            ttl_seconds=300.0,
        )
        assert source.snapshot("codex-sub", at=NOW).ttl_seconds == 30.0


class _StaticProbe:
    def __init__(self, results):
        self._results = results

    def probe(self, providers):
        return {p: self._results[p] for p in providers if p in self._results}


def test_bridge_script_is_syntactically_valid_python():
    # It is shipped as source text executed by a foreign interpreter, so a
    # syntax error would only surface as a confusing subprocess failure.
    compile(BRIDGE_SCRIPT, "<bridge>", "exec")


def _with_window_usage(payload, label, used_percent):
    return _with_window_field(payload, label, "used_percent", used_percent)


def _with_window_field(payload, label, key, value):
    windows = [
        {**w, key: value} if w["label"] == label else w for w in payload["snapshot"]["windows"]
    ]
    return {**payload, "snapshot": {**payload["snapshot"], "windows": windows}}
