"""Tests for the Hermes account-usage telemetry adapter.

The payloads below are recorded from a live run of the bridge script against
a real Hermes install (credentials elided), so the normalization is tested
against shapes Hermes actually emits rather than shapes we wish it emitted.
"""

from __future__ import annotations

import subprocess
import time
import types
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.config import TelemetryConfig, TelemetryResourceConfig
from ai_pr_orchestrator.v3.telemetry import (
    ProviderHealthLedger,
    TelemetryError,
    TelemetryRegistry,
)
from ai_pr_orchestrator.v3.telemetry_hermes import (
    BRIDGE_SCRIPT,
    HermesResource,
    HermesSubprocessProbe,
    HermesTelemetrySource,
    build_telemetry,
    normalize_probe_result,
    probe_outcome,
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

NO_DATA_PRIVATE = {"ok": True, "provider": "nope-provider", "fidelity": "private", "snapshot": None}

NO_DATA_PUBLIC = {"ok": True, "provider": "nope-provider", "fidelity": "public", "snapshot": None}


def _anthropic_snapshot() -> dict[str, Any]:
    snapshot = ANTHROPIC_OK["snapshot"]
    assert isinstance(snapshot, dict)
    return snapshot


def _openrouter_snapshot() -> dict[str, Any]:
    snapshot = OPENROUTER_OK["snapshot"]
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
        # No measured window and no structured balance: capacity is undetermined.
        assert snap.availability == "unknown"

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
    def test_a_spent_window_is_exhausted_not_merely_unavailable(self):
        payload = _with_window_usage(
            _with_window_usage(CODEX_OK, "Weekly", 100.0), "Session", 100.0
        )
        snap = _normalize(payload)
        assert snap.availability == "exhausted"
        assert require(snap.window("Weekly")).is_exhausted is True
        # Still recoverable, and the snapshot says exactly when.
        assert require(snap.window("Weekly")).reset_at is not None

    def test_a_spent_session_beside_a_live_week_is_still_exhausted(self):
        # Hermes maps the provider's window keys onto display labels and drops
        # the keys, so we cannot tell a window that constrains every request
        # from one that constrains a single model. A spent window of unknown
        # scope stops the resource; `spent_windows()` names which one.
        payload = _with_window_usage(CODEX_OK, "Session", 100.0)
        snap = _normalize(payload)
        assert snap.availability == "exhausted"
        assert [w.label for w in snap.spent_windows()] == ["Session"]
        assert require(snap.window("Weekly")).used_fraction == pytest.approx(0.885)

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

    def test_no_data_from_the_private_path_is_unavailable_with_an_actionable_reason(self):
        snap = _normalize(NO_DATA_PRIVATE)
        assert snap.availability == "unavailable"
        assert "no account-usage data" in snap.reason

    def test_no_data_from_the_public_fallback_is_only_unknown(self):
        # The public `fetch_account_usage` ends in a blanket
        # `except Exception: return None`, so a bare None there cannot
        # distinguish an expired credential from a network blip. Calling it
        # 'unavailable' would manufacture certainty the payload does not carry.
        snap = _normalize(NO_DATA_PUBLIC)
        assert snap.availability == "unknown"
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

    def test_unmeasured_windows_are_not_headroom(self):
        # Hermes' own `AccountUsageSnapshot.available` is `bool(windows or
        # details)` -- "is there a panel worth rendering?", not "may we
        # dispatch here?". Windows with no usage figure say nothing either way.
        payload = _with_window_field(
            _with_window_field(CODEX_OK, "Session", "used_percent", None),
            "Weekly",
            "used_percent",
            None,
        )
        snap = _normalize(payload)
        assert snap.availability == "unknown"
        assert "no measured quota window" in snap.reason

    def test_a_zero_balance_rendered_as_prose_is_never_reported_available(self):
        payload = {
            "ok": True,
            "provider": "openrouter",
            "fidelity": "private",
            "snapshot": {
                **_openrouter_snapshot(),
                "details": ["Credits balance: $0.00"],
            },
        }
        snap = _normalize(
            payload,
            resource=HermesResource(
                name="openrouter", provider="openrouter", resource_class="metered"
            ),
        )
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

    def test_two_resources_on_one_provider_are_rejected(self):
        # Hermes resolves credentials per provider from ambient machine state,
        # so it cannot select between two accounts on one provider: both rows
        # would report the same allowance under different names, and a broker
        # summing them would see twice the capacity that exists.
        with pytest.raises(TelemetryError, match="anthropic"):
            HermesTelemetrySource(
                resources=[
                    HermesResource(name="work-sub", provider="anthropic"),
                    HermesResource(name="personal-sub", provider="anthropic"),
                ],
                probe=_StaticProbe({}),
            )

    def test_provider_identity_is_canonicalized_before_deduplication(self):
        # BRIDGE_SCRIPT does strip().lower() on every provider argument and
        # keys its result by that canonical form. Left uncanonicalized here,
        # these two rows pass the uniqueness check and then both read back the
        # same probe result under different names.
        with pytest.raises(TelemetryError, match="anthropic"):
            HermesTelemetrySource(
                resources=[
                    HermesResource(name="work-sub", provider="Anthropic "),
                    HermesResource(name="personal-sub", provider="anthropic"),
                ],
                probe=_StaticProbe({}),
            )

    def test_an_oddly_spelled_provider_still_matches_its_probe_result(self):
        source = HermesTelemetrySource(
            resources=[HermesResource(name="codex-sub", provider="  OpenAI-Codex ")],
            probe=_StaticProbe({"openai-codex": CODEX_OK}),
        )
        assert source.snapshot("codex-sub", at=NOW).availability == "available"

    def test_a_broken_install_is_not_recorded_as_a_provider_failure(self):
        # An unrunnable probe says nothing about the provider. Recording it as
        # a failure would drive the broker away from a healthy account because
        # *our* install is broken.
        ledger = ProviderHealthLedger()
        source = HermesTelemetrySource(
            resources=[HermesResource(name="codex-sub", provider="openai-codex")],
            probe=_StaticProbe(
                {
                    "openai-codex": {
                        "ok": False,
                        "provider": "openai-codex",
                        "kind": "local_error",
                        "message": "Hermes interpreter /opt/hermes/venv/bin/python does not exist",
                    }
                }
            ),
            ledger=ledger,
        )
        snap = TelemetryRegistry(sources=[source], ledger=ledger).snapshot("codex-sub", at=NOW)
        assert snap.availability == "unknown"
        assert snap.health.outcomes == {}
        assert snap.health.consecutive_failures == 0

    def test_a_provider_failure_is_still_recorded(self):
        ledger = ProviderHealthLedger()
        source = HermesTelemetrySource(
            resources=[HermesResource(name="codex-sub", provider="openai-codex")],
            probe=_StaticProbe({"openai-codex": CODEX_AUTH_FAILURE}),
            ledger=ledger,
        )
        snap = TelemetryRegistry(sources=[source], ledger=ledger).snapshot("codex-sub", at=NOW)
        assert snap.health.outcomes == {"auth_failure": 1}

    def test_a_spent_windows_reset_expires_the_cache_before_the_ttl(self):
        # The provider says exactly when the allowance returns. Holding a
        # 100%-used reading past that keeps the resource out of rotation for
        # the rest of the TTL -- five minutes, by default, for a window that
        # reset in thirty seconds. Cycle 2 made this worse by giving a single
        # spent window the power to stop the whole resource.
        reset = NOW + timedelta(seconds=30)

        def payload(used):
            return {
                "ok": True,
                "provider": "anthropic",
                "fidelity": "private",
                "snapshot": {
                    **_anthropic_snapshot(),
                    "fetched_at": NOW.isoformat(),
                    "windows": [
                        {
                            "label": "Current week",
                            "used_percent": used,
                            "reset_at": reset.isoformat(),
                            "detail": None,
                        }
                    ],
                },
            }

        class Recovering:
            calls = 0

            def probe(self, providers):
                Recovering.calls += 1
                return {"anthropic": payload(100.0 if Recovering.calls == 1 else 4.0)}

        source = HermesTelemetrySource(
            resources=[HermesResource(name="anthropic-sub", provider="anthropic")],
            probe=Recovering(),
            ttl_seconds=300.0,
        )
        assert source.snapshot("anthropic-sub", at=NOW).availability == "exhausted"
        # Still cached right up to the reset: the allowance has not returned.
        assert source.snapshot("anthropic-sub", at=reset - timedelta(seconds=1)).availability == (
            "exhausted"
        )
        assert Recovering.calls == 1
        # Past it, re-probed rather than served a reading we know is stale.
        assert source.snapshot("anthropic-sub", at=reset).availability == "available"
        assert Recovering.calls == 2

    def test_a_live_windows_reset_does_not_expire_the_cache(self):
        # Only a spent window makes a cached reading wrong. Expiring on every
        # rollover would re-probe a rate-limited endpoint for no new answer.
        source = HermesTelemetrySource(
            resources=[HermesResource(name="codex-sub", provider="openai-codex")],
            probe=_StaticProbe({"openai-codex": CODEX_OK}),
            ttl_seconds=300.0,
        )
        first = require(_normalize(CODEX_OK).window("Session")).reset_at
        assert first is not None
        source.snapshot("codex-sub", at=NOW)
        snap = source.snapshot("codex-sub", at=first + timedelta(seconds=1))
        assert snap.availability == "available"

    def test_a_retry_after_deadline_survives_a_slow_probe(self):
        # Retry-After counts from when the provider answered, but `at` is read
        # before the subprocess starts. On a slow probe the whole delay could
        # elapse in transit, leaving is_throttled() false against a provider
        # that is still refusing us.
        class Slow:
            def probe(self, providers):
                time.sleep(0.3)
                return {
                    "anthropic": {
                        "ok": False,
                        "provider": "anthropic",
                        "kind": "rate_limited",
                        "message": "429",
                        "retry_after_seconds": 0.1,
                    }
                }

        ledger = ProviderHealthLedger()
        source = HermesTelemetrySource(
            resources=[HermesResource(name="anthropic-sub", provider="anthropic")],
            probe=Slow(),
            ledger=ledger,
        )
        snap = TelemetryRegistry(sources=[source], ledger=ledger).snapshot("anthropic-sub", at=NOW)
        assert snap.health.retry_after is not None
        assert snap.health.retry_after > NOW + timedelta(seconds=0.3)
        assert snap.health.is_throttled(NOW) is True

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


class TestSubprocessProbe:
    def test_it_runs_in_the_hermes_checkout_not_ours(self, tmp_path, monkeypatch):
        # `python -c` puts cwd on sys.path. Inheriting aipro's would let a
        # stray local directory named `agent` shadow Hermes' own package.
        interpreter = tmp_path / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")
        calls: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            calls.update(kwargs)
            raise OSError("not actually running it")

        monkeypatch.setattr(subprocess, "run", fake_run)
        HermesSubprocessProbe(hermes_home=tmp_path).probe(["anthropic"])
        assert calls["cwd"] == str(tmp_path)

    def test_an_unknown_checkout_leaves_cwd_unset_rather_than_guessing(self, tmp_path, monkeypatch):
        calls: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            calls.update(kwargs)
            raise OSError("not actually running it")

        monkeypatch.setattr(subprocess, "run", fake_run)
        python = tmp_path / "python"
        python.write_text("")
        HermesSubprocessProbe(python_executable=python).probe(["anthropic"])
        assert calls["cwd"] is None

    def test_a_relative_hermes_home_survives_the_cwd_switch(self, tmp_path, monkeypatch):
        # We hand `subprocess.run` an interpreter *and* a different cwd, and the
        # OS resolves a relative executable against the new cwd -- so a relative
        # path would pass the exists() preflight here and fail to launch there.
        interpreter = tmp_path / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")
        monkeypatch.chdir(tmp_path.parent)
        calls: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            calls["argv"] = argv
            calls.update(kwargs)
            raise OSError("not actually running it")

        monkeypatch.setattr(subprocess, "run", fake_run)
        HermesSubprocessProbe(hermes_home=Path(tmp_path.name)).probe(["anthropic"])
        assert calls["argv"][0] == str(interpreter)
        assert calls["cwd"] == str(tmp_path)

    @pytest.mark.parametrize("payload", ['{"results": null}', '{"results": 1}'])
    def test_a_non_iterable_results_field_is_a_local_error(self, payload, tmp_path, monkeypatch):
        # Structurally valid JSON in the wrong shape. Iterating it raised a
        # TypeError past the local-error path, and the source's blanket handler
        # then filed *our* corrupt output as a provider transport_error --
        # exactly the misattribution the local_error kind exists to prevent.
        interpreter = tmp_path / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")

        def fake_run(argv, **kwargs):
            return types.SimpleNamespace(returncode=0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = HermesSubprocessProbe(hermes_home=tmp_path).probe(["anthropic"])["anthropic"]
        assert result["kind"] == "local_error"
        assert "must be a list" in result["message"]
        assert probe_outcome(result) is None

    def test_the_bridge_gets_a_neutral_sys_path_when_no_checkout_is_known(
        self, tmp_path, monkeypatch
    ):
        # `python -c` puts the cwd on sys.path. With `hermes_python` set but no
        # `hermes_home` there is no Hermes directory to point cwd at, so the
        # subprocess inherited *ours* and any local `agent.py` shadowed Hermes'
        # real package. Verified against the live install: without this the
        # bridge imported a decoy; with it, the real agent package.
        calls: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            calls.update(kwargs)
            raise OSError("not actually running it")

        monkeypatch.setattr(subprocess, "run", fake_run)
        python = tmp_path / "python"
        python.write_text("")
        HermesSubprocessProbe(python_executable=python).probe(["anthropic"])
        assert calls["cwd"] is None
        assert calls["env"]["PYTHONSAFEPATH"] == "1"

    def test_a_known_checkout_keeps_its_cwd_on_sys_path(self, tmp_path, monkeypatch):
        # The mirror image: a checkout never installed into its venv is
        # importable *only* via cwd, so isolating there would break it.
        interpreter = tmp_path / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")
        calls: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            calls.update(kwargs)
            raise OSError("not actually running it")

        monkeypatch.setattr(subprocess, "run", fake_run)
        HermesSubprocessProbe(hermes_home=tmp_path).probe(["anthropic"])
        assert calls["cwd"] == str(tmp_path)
        assert "PYTHONSAFEPATH" not in calls["env"]

    def test_a_venv_interpreter_symlink_is_not_followed(self, tmp_path, monkeypatch):
        # Making the path absolute must stay lexical. A venv's `bin/python` is
        # a symlink to the base interpreter, and following it lands outside the
        # venv, where sys.prefix no longer finds the venv's site-packages --
        # against the real install, Hermes then failed to import httpx.
        base = tmp_path / "base" / "bin" / "python3"
        base.parent.mkdir(parents=True)
        base.write_text("")
        interpreter = tmp_path / "hermes" / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(base)
        calls: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            calls["argv"] = argv
            raise OSError("not actually running it")

        monkeypatch.setattr(subprocess, "run", fake_run)
        HermesSubprocessProbe(hermes_home=tmp_path / "hermes").probe(["anthropic"])
        assert calls["argv"][0] == str(interpreter)

    def test_a_process_timeout_blames_our_deadline_not_the_providers(self, tmp_path, monkeypatch):
        # The bridge walks providers sequentially inside one subprocess, so a
        # deadline on the whole process cannot say which provider hung. A
        # per-provider timeout would charge one to providers that already
        # answered and to providers never contacted at all.
        interpreter = tmp_path / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")

        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 5.0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        results = HermesSubprocessProbe(hermes_home=tmp_path).probe(["anthropic", "openai-codex"])
        assert {r["kind"] for r in results.values()} == {"local_error"}
        assert all(probe_outcome(r) is None for r in results.values())
        assert all("timed out" in r["message"] for r in results.values())

    @pytest.mark.parametrize(
        "reason",
        ["no Hermes interpreter configured", "does not exist"],
    )
    def test_an_unrunnable_probe_reports_a_local_error(self, reason, tmp_path):
        probes = {
            "no Hermes interpreter configured": HermesSubprocessProbe(),
            "does not exist": HermesSubprocessProbe(hermes_home=tmp_path / "absent"),
        }
        result = probes[reason].probe(["anthropic"])["anthropic"]
        assert result["kind"] == "local_error"
        assert reason in result["message"]
        assert probe_outcome(result) is None


class TestBridgeRetryAfter:
    def _retry_after(self, raw):
        namespace: dict[str, Any] = {}
        exec(BRIDGE_SCRIPT.split("\nprint(json.dumps")[0], namespace)

        class Response:
            headers: ClassVar[dict[str, str]] = {"Retry-After": raw}

        class Failure(Exception):
            response = Response()

        return namespace["_retry_after"](Failure())

    def test_delay_seconds_are_read_directly(self):
        assert self._retry_after("30") == 30.0

    def test_an_http_date_is_converted_to_seconds_from_now(self):
        # RFC 9110 permits either form. Dropping the date form would leave
        # is_throttled() false while the provider is still refusing us.
        soon = datetime.now(UTC) + timedelta(seconds=120)
        seconds = self._retry_after(format_datetime(soon, usegmt=True))
        assert seconds is not None
        assert 60 < seconds <= 120

    def test_an_http_date_in_the_past_floors_at_zero(self):
        past = datetime.now(UTC) - timedelta(hours=1)
        assert self._retry_after(format_datetime(past, usegmt=True)) == 0.0

    def test_garbage_is_dropped_rather_than_raising(self):
        assert self._retry_after("whenever") is None

    @pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "nan", "inf"])
    def test_non_finite_delays_are_dropped(self, raw):
        # float() accepts these and json.dumps emits them verbatim, so such a
        # header would survive the bridge and then raise in timedelta() on our
        # side -- turning a rate limit into a generic source failure.
        assert self._retry_after(raw) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_retry_after_from_the_bridge_is_dropped(value):
    # json.loads accepts the NaN/Infinity literals json.dumps emits, so the
    # check belongs on our side too: a bridge from a different Hermes build is
    # not ours to trust.
    assert probe_outcome(
        {"ok": False, "provider": "p", "kind": "rate_limited", "retry_after_seconds": value}
    ) == ("rate_limited", None)


def _with_window_usage(payload, label, used_percent):
    return _with_window_field(payload, label, "used_percent", used_percent)


def _with_window_field(payload, label, key, value):
    windows = [
        {**w, key: value} if w["label"] == label else w for w in payload["snapshot"]["windows"]
    ]
    return {**payload, "snapshot": {**payload["snapshot"], "windows": windows}}


class TestBuildTelemetryWiring:
    def _catalog(self) -> ModelCatalog:
        return ModelCatalog(
            entries=(
                ModelCatalogEntry(
                    ref="promo-free-generalist",
                    descriptor="vendor/promo",
                    resource_class="metered",
                    cost_class="low",
                    promotional=True,
                    promo_ends_at=NOW + timedelta(days=30),
                    input_price_per_mtok=1.0,
                    output_price_per_mtok=2.0,
                    source_updated_at=NOW - timedelta(days=90),
                ),
            )
        )

    def test_catalog_declarations_are_not_held_to_the_probe_freshness_budget(self):
        # snapshot_ttl_seconds says how long a *measurement* may be reused
        # before it is taken again. Nothing re-measures a hand-written catalog
        # entry, so handing it that budget marked every provenanced entry
        # permanently stale -- a warning no operator action could ever clear.
        registry, _ledger = build_telemetry(
            TelemetryConfig(snapshot_ttl_seconds=300.0), catalog=self._catalog()
        )
        snap = registry.snapshot("promo-free-generalist", at=NOW)
        assert snap.source == "catalog"
        assert snap.is_stale(NOW) is None
        # The age is still reported truthfully, so a reader can judge for itself.
        assert snap.age(NOW) == timedelta(days=90)

    def test_probe_snapshots_still_carry_the_configured_budget(self):
        registry, _ledger = build_telemetry(
            TelemetryConfig(
                snapshot_ttl_seconds=300.0,
                resources=[TelemetryResourceConfig(name="codex-sub", provider="codex")],
                hermes_home="/nonexistent/hermes",
            )
        )
        snap = registry.snapshot("codex-sub", at=NOW)
        assert snap.source == "hermes"
        assert snap.ttl_seconds == 300.0
