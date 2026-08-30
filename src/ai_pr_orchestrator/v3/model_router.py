"""V3 model router wiring (issue #55).

Resolves the effective model catalog from :class:`~ai_pr_orchestrator.v3.
config.ModelRouterConfig` (an inline catalog or a shared ``catalog_path``
file — never both) and builds a :class:`~ai_pr_orchestrator.v3.broker.
PolicyBroker` with telemetry snapshots taken from a
:class:`~ai_pr_orchestrator.v3.interfaces.ProviderTelemetrySource`.

This module is the *wiring*, not the ranking: it joins three vocabularies —
catalog entries (keyed by policy ref), telemetry resources (keyed by resource
name), and the config's telemetry rows (which declare the provider → resource
join) — and hands the result to the broker, which owns every economic
judgement. A missing telemetry source degrades to a broker with no snapshots:
every resource then reports as unmeasured, which is the broker's documented
behaviour for ignorance, rather than failing to build.

No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .broker import PolicyBroker
from .catalog import ModelCatalog, load_model_catalog
from .config import V3Config
from .interfaces import ProviderTelemetrySource
from .telemetry import ProviderResourceSnapshot


class ModelRouterError(ValueError):
    """Raised when the model-router configuration cannot be resolved."""


def resolve_catalog(config: V3Config) -> ModelCatalog:
    """The effective catalog for ``config``.

    An inline ``catalog`` and a ``catalog_path`` are mutually exclusive (the
    config validates this too); a section that declares neither yields an
    empty catalog, which the broker handles as "no candidate is dispatchable"
    with named rejections rather than as an error.
    """

    router = config.model_router
    if router.catalog and router.catalog_path:
        raise ModelRouterError(
            "model_router declares both an inline catalog and a catalog_path; "
            "the effective catalog would be ambiguous"
        )
    if router.catalog:
        return ModelCatalog(entries=tuple(router.catalog))
    if router.catalog_path:
        return load_model_catalog(Path(router.catalog_path))
    return ModelCatalog()


def build_model_broker(
    config: V3Config,
    *,
    telemetry_source: ProviderTelemetrySource | None = None,
    at: datetime | None = None,
) -> PolicyBroker:
    """Build a :class:`PolicyBroker` for ``config``.

    When ``telemetry_source`` is supplied, one snapshot per configured
    telemetry resource is taken at the same instant ``at`` (defaulting to
    now), so the broker evaluates every resource against one consistent
    view. The resource→provider join comes from the telemetry config rows.
    """

    catalog = resolve_catalog(config)
    snapshots: list[ProviderResourceSnapshot] = []
    resource_by_provider: dict[str, str] = {}
    if telemetry_source is not None:
        for row in config.telemetry.resources:
            snapshots.append(telemetry_source.snapshot(row.name, at=at))
            resource_by_provider[row.provider] = row.name
    return PolicyBroker(
        catalog,
        config.broker,
        snapshots=snapshots,
        resource_by_provider=resource_by_provider,
    )
