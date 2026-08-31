"""Tests for V3 model router wiring (issue #55)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from ai_pr_orchestrator.v3.broker import PolicyBroker, TaskDemand
from ai_pr_orchestrator.v3.catalog import ModelCatalogEntry
from ai_pr_orchestrator.v3.config import (
    ModelRouterConfig,
    TelemetryConfig,
    TelemetryResourceConfig,
    V3Config,
)
from ai_pr_orchestrator.v3.model_router import ModelRouterError, build_model_broker, resolve_catalog
from ai_pr_orchestrator.v3.telemetry import ProviderResourceSnapshot

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def entry(ref: str) -> ModelCatalogEntry:
    return ModelCatalogEntry(
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


def _config(*entries: ModelCatalogEntry) -> V3Config:
    return V3Config(model_router=ModelRouterConfig(catalog=list(entries)))


class _StaticTelemetry:
    """A telemetry source serving one canned snapshot per resource."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    def resources(self) -> tuple[str, ...]:
        return ("acct-main",)

    def snapshot(self, resource: str, *, at: datetime | None = None) -> ProviderResourceSnapshot:
        self.requested.append(resource)
        return ProviderResourceSnapshot(
            resource=resource,
            observed_at=at or NOW,
            availability="available",
            resource_class="subscription",
        )


def test_inline_catalog_resolves():
    config = _config(entry("alpha"), entry("beta"))
    catalog = resolve_catalog(config)
    assert sorted(catalog.refs()) == ["alpha", "beta"]


def test_catalog_path_resolves_from_file(tmp_path: Path):
    path = tmp_path / "catalog.yml"
    path.write_text(yaml.safe_dump({"models": [entry("gamma").to_dict()]}))
    config = V3Config(model_router=ModelRouterConfig(catalog_path=str(path)))
    catalog = resolve_catalog(config)
    assert catalog.refs() == ["gamma"]


def test_relative_catalog_path_resolves_against_config_dir(tmp_path: Path):
    """A relative catalog_path must resolve against the config file's directory
    (via ``base_dir``), never the process working directory."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "catalog.yml").write_text(yaml.safe_dump({"models": [entry("gamma").to_dict()]}))
    config = V3Config(model_router=ModelRouterConfig(catalog_path="catalog.yml"))

    catalog = resolve_catalog(config, base_dir=config_dir)

    assert catalog.refs() == ["gamma"]


def test_inline_and_path_together_are_rejected(tmp_path: Path):
    config = V3Config(
        model_router=ModelRouterConfig(
            catalog=[entry("alpha")], catalog_path=str(tmp_path / "catalog.yml")
        )
    )
    try:
        resolve_catalog(config)
    except ModelRouterError:
        return
    raise AssertionError("expected ModelRouterError")


def test_empty_section_yields_empty_catalog():
    assert resolve_catalog(V3Config()).refs() == []


def test_build_broker_takes_one_snapshot_per_resource():
    telemetry = _StaticTelemetry()
    config = V3Config(
        model_router=ModelRouterConfig(catalog=[entry("alpha")]),
        telemetry=TelemetryConfig(
            resources=[TelemetryResourceConfig(name="acct-main", provider="prov-main")]
        ),
    )
    broker = build_model_broker(config, telemetry_source=telemetry, at=NOW)
    assert isinstance(broker, PolicyBroker)
    assert telemetry.requested == ["acct-main"]
    decision = broker.select(TaskDemand(lane="developer", role="worker"), at=NOW)
    assert decision.assignment is not None
    assert decision.assignment.model_ref == "alpha"


def test_build_broker_without_telemetry_still_works():
    config = _config(entry("alpha"))
    broker = build_model_broker(config)
    decision = broker.select(TaskDemand(lane="developer", role="worker"), at=NOW)
    assert decision.assignment is not None


def test_empty_catalog_rejects_with_named_reason():
    broker = build_model_broker(V3Config())
    decision = broker.select(TaskDemand(lane="developer", role="worker"), at=NOW)
    assert decision.assignment is None
    assert decision.rejected == ()
    assert decision.reason


def test_snapshots_share_one_timestamp_when_at_is_omitted():
    """With ``at=None`` every resource is sampled at ONE materialized
    timestamp, not each at its own 'now' (#12)."""
    from datetime import UTC

    class RecordingTelemetry(_StaticTelemetry):
        def __init__(self) -> None:
            super().__init__()
            self.ats: list[datetime | None] = []

        def snapshot(self, resource: str, *, at: datetime | None = None):
            self.ats.append(at)
            return super().snapshot(resource, at=at)

    config = V3Config(
        model_router=ModelRouterConfig(catalog=[entry("alpha")]),
        telemetry=TelemetryConfig(
            resources=[
                TelemetryResourceConfig(name="acct-main", provider="prov-main"),
                TelemetryResourceConfig(name="acct-secondary", provider="prov-secondary"),
            ]
        ),
    )
    telemetry = RecordingTelemetry()
    build_model_broker(config, telemetry_source=telemetry)  # at omitted
    assert len(telemetry.ats) == 2
    assert telemetry.ats[0] is not None and telemetry.ats[1] is not None
    assert telemetry.ats[0] == telemetry.ats[1]
    assert telemetry.ats[0].tzinfo == UTC
