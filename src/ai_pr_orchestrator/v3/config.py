"""V3 configuration schema.

The V3 config is a pure policy schema: it describes what the engine should
do, never which vendor binary to run. Loading validates structure and
cross-section references, and preserves unknown keys as ``extras`` so configs
written by newer versions load cleanly in older readers (forward
compatibility).

Every section can be constructed either as a typed dataclass or from a plain
mapping (which is coerced in ``V3Config.__post_init__``), so configs and
tests stay terse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ._schema import SchemaError, build_dataclass, to_mapping, typed_kwargs
from ._schema import validate_declared_shapes as _validate_shapes
from .catalog import (
    MAX_QUALITY,
    MIN_QUALITY,
    VALID_RESOURCE_CLASSES,
    ModelCatalog,
    ModelCatalogEntry,
    load_model_catalog,
)
from .domain import VALID_LANE_ROLES


class V3ConfigError(SchemaError):
    """Raised when the V3 configuration is invalid."""


# --- Sections --------------------------------------------------------------


@dataclass(frozen=True)
class GitHubQueueConfig:
    """How V3 reads and writes GitHub as the authoritative workflow state."""

    enabled_label: str = "v3-work"
    done_label: str = "v3-work-done"
    error_label: str = "v3-work-error"
    #: Full lifecycle labels. ``enabled_label`` is the "ready/queued" label;
    #: the rest map to the V3 phases per :mod:`ai_pr_orchestrator.v3.queue`.
    active_label: str = "v3-work-active"
    review_label: str = "v3-work-review"
    needs_human_label: str = "v3-work-needs-human"
    state_comment_marker: str = "v3-runtime-state"
    #: How long a claim's lease stays valid without a heartbeat.
    lease_seconds: int = 900
    #: Upper bound on the serialized machine-state block, in characters.
    #: Larger payloads are compacted (see ``v3.queue`` compaction docs);
    #: if compaction cannot fit, saving raises.
    max_state_block_chars: int = 60000
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CAOControlPlaneConfig:
    """CAO control-plane attachment points.

    V3 core never constructs CAO calls itself; this section only declares the
    policy-level knobs (timeouts, polling) around the CAO session fabric.
    """

    control_dir: str = ".cao"
    session_poll_interval_seconds: int = 30
    session_timeout_seconds: int = 3600
    #: Root URL of CAO's HTTP control plane. It is the only supported way V3
    #: talks to the session fabric — no terminal scraping, no CLI shelling.
    base_url: str = "http://localhost:9889"
    #: Per-HTTP-request budget. Distinct from ``session_timeout_seconds``,
    #: which bounds how long an agent session may run: a session legitimately
    #: runs for an hour while every individual control-plane call is expected
    #: to answer in seconds.
    request_timeout_seconds: float = 30.0
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LaneProfileConfig:
    """One Hermes lane/profile template."""

    name: str
    role: str
    profile_template: str
    max_concurrent: int = 1
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HermesLanesConfig:
    """Hermes lane/profile templates."""

    lanes: list[LaneProfileConfig] = field(default_factory=list)
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRouterConfig:
    """Model catalog plus lane-to-model routing.

    The catalog is a *machine-level* artifact shared by every lane, so the
    normal deployment declares ``catalog_path`` and keeps one file per
    machine rather than repeating entries in each repo's policy config.
    Inline ``catalog`` entries remain supported for tests and single-repo
    setups. Declaring both is rejected: it would leave the effective catalog
    ambiguous, and a stale inline entry silently shadowing the shared file is
    exactly the failure this section exists to prevent.
    """

    catalog: list[ModelCatalogEntry] = field(default_factory=list)
    #: Path to the shared catalog file, resolved relative to the config file
    #: that declares it. Loaded by :func:`load_v3_config`, which is where V3
    #: config I/O lives; ``from_dict`` stays pure and never touches the disk.
    catalog_path: str | None = None
    lane_assignments: dict[str, str] = field(default_factory=dict)  # lane -> model ref
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryResourceConfig:
    """One subscription/gateway account to collect live telemetry for.

    ``name`` is the policy-level id the broker and operator use; ``provider``
    is the upstream key Hermes resolves credentials for. They are separate so
    renaming a provider upstream does not rewrite routing decisions.

    Two resources may not share a ``provider``: Hermes resolves credentials
    per provider from ambient machine state, so it cannot distinguish two
    accounts on one provider and both rows would report the same allowance.
    The Hermes telemetry source rejects that at construction.

    There is deliberately no credential field: Hermes owns credential
    resolution, so a secret never has to be written into V3 policy config.
    """

    name: str
    provider: str
    resource_class: str = "subscription"
    #: Overrides ``telemetry.snapshot_ttl_seconds`` for this resource only.
    ttl_seconds: float | None = None
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryConfig:
    """Live provider telemetry: which resources to poll, and how freshly.

    ``hermes_home``/``hermes_python`` locate the interpreter the account-usage
    bridge runs under. They are intentionally *not* required: a config must
    validate identically in CI, where no Hermes install exists. A missing
    interpreter degrades every resource to ``unknown`` at collection time,
    which is the correct answer, rather than failing to load the policy.
    """

    resources: list[TelemetryResourceConfig] = field(default_factory=list)
    #: Default freshness budget. Beyond it a snapshot is marked stale and
    #: re-probed, rather than being served as if it were current.
    snapshot_ttl_seconds: float = 300.0
    #: Root of the Hermes checkout; the bridge runs ``<home>/venv/bin/python``.
    hermes_home: str | None = None
    #: Explicit interpreter path, taking precedence over ``hermes_home``.
    hermes_python: str | None = None
    probe_timeout_seconds: float = 30.0
    #: How many recent request outcomes feed the health statistics.
    health_window_size: int = 50
    #: Also report free-tier/promotional entries from the model catalog as
    #: telemetry resources, so perishable capacity shows up in one listing
    #: alongside the subscriptions.
    include_catalog_resources: bool = True
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceReserveConfig:
    """How much of a resource's allowance the broker must not spend.

    ``fraction`` applies to every window; ``windows`` overrides it for one
    window by label. Per-window reserves exist because the windows mean
    different things: holding back a fifth of a weekly allowance is prudent,
    holding back a fifth of a five-hour session window is a much larger
    concession that a deployment may not want to make.

    Window labels are provider prose (Hermes renders them for display), so a
    per-window reserve is keyed on text that can change between provider or
    Hermes versions. An override whose label matches nothing simply does not
    bind — it cannot invent a constraint — and the broker's dry-run prints the
    labels it actually saw, which is how a drifted label is spotted.
    """

    resource: str
    fraction: float = 0.0
    windows: dict[str, float] = field(default_factory=dict)
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerConfig:
    """How the model broker ranks catalog candidates.

    Every knob here is policy, not a vendor fact: the broker must be able to
    change its mind about *what it values* without anyone editing the catalog,
    and the catalog must be able to gain a model without anyone editing the
    broker.
    """

    #: Quality tier an entry must reach for a role, over and above the floor
    #: that difficulty already implies. Adjudication-style work wants a
    #: stronger model than its nominal difficulty asks for; this is how that
    #: is expressed without inventing a role the lane registry does not have.
    min_quality_by_role: dict[str, int] = field(default_factory=dict)
    #: Quality to assume for an entry the catalog leaves unscored for a role.
    #: ``None`` (the default) makes such an entry undispatchable with an
    #: actionable reason, rather than guessing in the operator's favour.
    default_quality: int | None = None

    #: Relative importance of each score component. Raising one of these is
    #: how a deployment says "we care more about capability than spend" (or
    #: the reverse) without touching broker code.
    weight_quality: float = 1.0
    weight_cash_cost: float = 1.0
    weight_quota_pressure: float = 0.75
    weight_health: float = 0.75
    weight_perishability: float = 1.0
    #: Cost of seating a candidate whose lineage resembles the peers already
    #: in an adversarial set. Independent of ``weight_quality`` on purpose:
    #: caring more about capability must not silently change diversity policy.
    weight_diversity: float = 1.0

    #: Allowance the broker must leave unspent, per resource.
    reserves: list[ResourceReserveConfig] = field(default_factory=list)
    #: Share of a window the deployment expects its *other* work to consume
    #: per hour, keyed by telemetry resource. Subtracted from the headroom
    #: before any of it is called surplus, so allowance that is already
    #: spoken for is not pulled forward as if it were going to waste.
    projected_burn_fraction_per_hour: dict[str, float] = field(default_factory=dict)
    #: How far ahead a reset has to be before its allowance stops looking
    #: perishable. Beyond this horizon there is time for normal work to
    #: consume the window, so there is nothing to rescue.
    pull_forward_horizon_hours: float = 24.0
    #: Headroom below which an allowance is treated as scarce.
    scarcity_threshold: float = 0.25
    #: Difficulty a task must reach to spend a scarce allowance. Withholding
    #: it is a filter rather than a score penalty: an operator has to be able
    #: to read why a subscription sat idle, and a rule that only bites when
    #: the weighted arithmetic happens to land is not a policy.
    scarcity_difficulty_floor: int = 4

    #: Recent failure rate at which a resource stops being dispatchable.
    max_failure_rate: float = 0.5
    #: Requests needed before a failure rate is treated as evidence at all.
    #: Below it, one unlucky call would exclude an otherwise fine provider.
    min_health_samples: int = 5
    #: Scores for what we could not measure. Both sit below a confirmed-good
    #: reading and above a confirmed-bad one, so evidence is rewarded without
    #: freezing out a candidate that has simply never been tried.
    unknown_quota_score: float = 0.5
    unknown_health_score: float = 0.75

    #: Cash cost is scored against this reference price rather than against
    #: the other candidates, so adding an unrelated model to the catalog never
    #: changes what an existing one scores.
    reference_price_per_mtok: float = 5.0
    #: Token mix assumed for one dispatch, used to blend input and output
    #: prices into a single comparable figure.
    expected_input_mtok: float = 1.0
    expected_output_mtok: float = 0.25
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)
    #: Fallback chain is bounded so a large catalog does not become the chain.
    max_fallbacks: int = 3
    #: Require each fallback to leave the previous one's failure domain
    #: (a different provider), so a provider outage does not empty the chain.
    require_distinct_fallback_provider: bool = True


@dataclass(frozen=True)
class SafetyPolicyConfig:
    """Safety rails carried over from the V1 safety surface.

    These controls bound what the engine is allowed to touch and how much
    work one run may do; they must survive the V1-to-V3 cutover, so they are
    declared here as first-class policy rather than left implicit.

    ``disallow_forks`` restricts runs to same-repo PRs; ``disallow_workflow_
    file_changes`` rejects edits under ``.github/workflows/``; the
    ``max_*`` budgets cap iterations, commits, lane invocations, and prompt
    tokens per run; ``allowed_pr_author_associations`` whitelists PR authors.
    """

    disallow_forks: bool = True
    disallow_workflow_file_changes: bool = True
    max_total_iterations: int = 3
    max_commits_per_run: int = 1
    max_coder_invocations_per_run: int = 1
    max_reviewer_triggers_per_run: int = 3
    max_prompt_tokens: int = 100000
    allowed_pr_author_associations: list[str] = field(
        default_factory=lambda: ["OWNER", "MEMBER", "COLLABORATOR"]
    )
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewPolicyConfig:
    """Review roles and limits."""

    reviewer_lanes: list[str] = field(default_factory=list)
    max_review_rounds: int = 3
    require_coder_reply_before_resolve: bool = True
    auto_resolve_bot_threads: bool = True
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CIPolicyConfig:
    """CI/PR gating policy."""

    required_checks: list[str] = field(default_factory=list)
    ci_wait_timeout_seconds: int = 1800
    require_green_ci_before_merge: bool = True
    pr_title_prefix: str = ""
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EscalationPolicyConfig:
    """Human escalation policy."""

    max_consecutive_coder_failures: int = 3
    stagnation_rounds_threshold: int = 3
    escalation_label: str = "needs-human"
    notify_comment_marker: str = "v3-escalation"
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


# --- Section registry (used for dict coercion and deserialization) ---------

_SECTION_TYPES: dict[str, type] = {
    "github_queue": GitHubQueueConfig,
    "cao": CAOControlPlaneConfig,
    "hermes_lanes": HermesLanesConfig,
    "model_router": ModelRouterConfig,
    "telemetry": TelemetryConfig,
    "broker": BrokerConfig,
    "review_policy": ReviewPolicyConfig,
    "ci_policy": CIPolicyConfig,
    "safety": SafetyPolicyConfig,
    "escalation": EscalationPolicyConfig,
}

_NESTED_LIST_FIELDS: dict[tuple[str, str], type] = {
    ("hermes_lanes", "lanes"): LaneProfileConfig,
    ("model_router", "catalog"): ModelCatalogEntry,
    ("telemetry", "resources"): TelemetryResourceConfig,
    ("broker", "reserves"): ResourceReserveConfig,
}


# --- Root config -----------------------------------------------------------


@dataclass(frozen=True)
class V3Config:
    """Root of the V3 policy schema."""

    github_queue: GitHubQueueConfig = field(default_factory=GitHubQueueConfig)
    cao: CAOControlPlaneConfig = field(default_factory=CAOControlPlaneConfig)
    hermes_lanes: HermesLanesConfig = field(default_factory=HermesLanesConfig)
    model_router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    review_policy: ReviewPolicyConfig = field(default_factory=ReviewPolicyConfig)
    ci_policy: CIPolicyConfig = field(default_factory=CIPolicyConfig)
    safety: SafetyPolicyConfig = field(default_factory=SafetyPolicyConfig)
    escalation: EscalationPolicyConfig = field(default_factory=EscalationPolicyConfig)
    #: Unknown top-level keys, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept plain mappings for sections, coercing them into typed ones.
        # frozen dataclasses need object.__setattr__ for this normalization.
        for name, section_type in _SECTION_TYPES.items():
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, section_type())
            elif isinstance(value, dict):
                object.__setattr__(self, name, _section_from_dict(name, value))
            elif not isinstance(value, section_type):
                raise V3ConfigError(
                    f"config section {name!r} must be a mapping or "
                    f"{section_type.__name__}, got {type(value).__name__}"
                )

    def validate(self) -> None:
        """Validate cross-section invariants; raises :class:`V3ConfigError`."""
        # A hermes_lanes section that declares nothing yields the default lane
        # set at registry-build time, so references must be validated against
        # that effective set — not the raw empty list, which would make it
        # impossible for a defaults-relying config to set lane_assignments
        # or reviewer_lanes.
        from .lanes import DEFAULT_LANES

        declared_lanes = list(self.hermes_lanes.lanes)
        if declared_lanes:
            lanes = declared_lanes
        else:
            lanes = [
                LaneProfileConfig(
                    name=lane.lane,
                    role=lane.role,
                    profile_template=lane.profile_template,
                )
                for lane in DEFAULT_LANES
            ]
        lane_names = [lane.name for lane in lanes]
        if len(lane_names) != len(set(lane_names)):
            raise V3ConfigError(f"Duplicate lane names in hermes_lanes: {lane_names}")

        for lane in lanes:
            if not lane.name:
                raise V3ConfigError("lane name must be non-empty")
            if not lane.profile_template:
                raise V3ConfigError(f"lane {lane.name!r} profile_template must be non-empty")

        q = self.github_queue
        labels = (
            q.enabled_label,
            q.active_label,
            q.review_label,
            q.needs_human_label,
            q.done_label,
            q.error_label,
        )
        if len(labels) != len(set(labels)):
            raise V3ConfigError(
                "github_queue lifecycle labels must be distinct: "
                f"enabled={q.enabled_label!r} active={q.active_label!r} "
                f"review={q.review_label!r} needs_human={q.needs_human_label!r} "
                f"done={q.done_label!r} error={q.error_label!r}"
            )
        empty = [
            name
            for name, value in zip(
                ("enabled", "active", "review", "needs_human", "done", "error"),
                labels,
                strict=False,
            )
            if not value
        ]
        if empty:
            raise V3ConfigError(f"github_queue lifecycle labels must be non-empty: missing {empty}")
        if not q.state_comment_marker:
            raise V3ConfigError("github_queue.state_comment_marker must be non-empty")
        if q.lease_seconds < 1:
            raise V3ConfigError("github_queue.lease_seconds must be >= 1")
        if q.max_state_block_chars < 1:
            raise V3ConfigError("github_queue.max_state_block_chars must be >= 1")

        foremen = [lane for lane in lanes if lane.role == "foreman"]
        if len(foremen) > 1:
            raise V3ConfigError("At most one foreman lane is allowed")

        for lane in lanes:
            if lane.role not in VALID_LANE_ROLES:
                raise V3ConfigError(f"Invalid role {lane.role!r} for lane {lane.name!r}")
            if lane.max_concurrent < 1:
                raise V3ConfigError(f"lane {lane.name!r} max_concurrent must be >= 1")

        if self.model_router.catalog and self.model_router.catalog_path:
            raise V3ConfigError(
                "model_router declares both an inline catalog and a catalog_path; "
                "the effective catalog would be ambiguous. Use one or the other."
            )

        # Catalog invariants live on the catalog itself, so an inline catalog
        # and a shared catalog file are held to identical rules and report
        # identical errors. Those surface as ModelCatalogError, which shares
        # the SchemaError base with V3ConfigError.
        catalog_refs = ModelCatalog(entries=tuple(self.model_router.catalog)).refs()

        for lane_name, model_ref in self.model_router.lane_assignments.items():
            if lane_name not in lane_names:
                raise V3ConfigError(f"lane_assignments references unknown lane {lane_name!r}")
            # With a catalog_path the refs live in a file this pure validator
            # must not read. load_v3_config re-checks them once the shared
            # catalog is resolved.
            if not self.model_router.catalog_path and model_ref not in catalog_refs:
                raise V3ConfigError(
                    f"lane {lane_name!r} references unknown model ref {model_ref!r}"
                )

        self._validate_telemetry()

        for reviewer in self.review_policy.reviewer_lanes:
            lane = next((c for c in lanes if c.name == reviewer), None)
            if lane is None:
                raise V3ConfigError(f"review_policy references unknown reviewer lane {reviewer!r}")
            if lane.role != "reviewer":
                raise V3ConfigError(
                    f"reviewer lane {reviewer!r} must have role 'reviewer', got {lane.role!r}"
                )

        if self.review_policy.max_review_rounds < 1:
            raise V3ConfigError("max_review_rounds must be >= 1")

        b = self.broker
        import math

        weights = {
            "weight_quality": b.weight_quality,
            "weight_cash_cost": b.weight_cash_cost,
            "weight_quota_pressure": b.weight_quota_pressure,
            "weight_health": b.weight_health,
            "weight_perishability": b.weight_perishability,
            "weight_diversity": b.weight_diversity,
        }
        for name, value in weights.items():
            if not math.isfinite(value) or value < 0:
                raise V3ConfigError(f"broker.{name} must be finite and >= 0, got {value!r}")
        if not math.isfinite(b.pull_forward_horizon_hours) or b.pull_forward_horizon_hours <= 0:
            raise V3ConfigError(
                "broker.pull_forward_horizon_hours must be finite and > 0, "
                f"got {b.pull_forward_horizon_hours!r}"
            )
        if not math.isfinite(b.reference_price_per_mtok) or b.reference_price_per_mtok <= 0:
            raise V3ConfigError(
                "broker.reference_price_per_mtok must be finite and > 0, "
                f"got {b.reference_price_per_mtok!r}"
            )
        for name, value in (
            ("expected_input_mtok", b.expected_input_mtok),
            ("expected_output_mtok", b.expected_output_mtok),
        ):
            if not math.isfinite(value) or value < 0:
                raise V3ConfigError(f"broker.{name} must be finite and >= 0, got {value!r}")
        if b.max_fallbacks < 0:
            raise V3ConfigError(f"broker.max_fallbacks must be >= 0, got {b.max_fallbacks}")
        for role, tier in b.min_quality_by_role.items():
            if role not in VALID_LANE_ROLES:
                raise V3ConfigError(f"broker.min_quality_by_role references unknown role {role!r}")
            if not MIN_QUALITY <= tier <= MAX_QUALITY:
                raise V3ConfigError(
                    f"broker.min_quality_by_role[{role!r}] must be within "
                    f"{MIN_QUALITY}..{MAX_QUALITY}, got {tier!r}"
                )
        if b.default_quality is not None and not MIN_QUALITY <= b.default_quality <= MAX_QUALITY:
            raise V3ConfigError(
                f"broker.default_quality must be within {MIN_QUALITY}..{MAX_QUALITY}, "
                f"got {b.default_quality!r}"
            )
        seen_reserve_resources: set[str] = set()
        for reserve in b.reserves:
            if reserve.resource in seen_reserve_resources:
                raise V3ConfigError(
                    f"broker.reserves declares {reserve.resource!r} more than once; "
                    "merge the global fraction and per-window overrides into one "
                    "entry instead of letting file order pick the winner"
                )
            seen_reserve_resources.add(reserve.resource)
            if not 0.0 <= reserve.fraction <= 1.0:
                raise V3ConfigError(
                    f"broker reserve for {reserve.resource!r} fraction must be within "
                    f"0.0..1.0, got {reserve.fraction!r}"
                )
            for label, fraction in reserve.windows.items():
                if not 0.0 <= fraction <= 1.0:
                    raise V3ConfigError(
                        f"broker reserve for {reserve.resource!r} window {label!r} "
                        f"fraction must be within 0.0..1.0, got {fraction!r}"
                    )

        if self.cao.session_poll_interval_seconds < 1:
            raise V3ConfigError("cao.session_poll_interval_seconds must be >= 1")
        if self.cao.session_timeout_seconds < 1:
            raise V3ConfigError("cao.session_timeout_seconds must be >= 1")
        if not self.cao.base_url:
            raise V3ConfigError("cao.base_url must be non-empty")
        if self.cao.request_timeout_seconds <= 0:
            raise V3ConfigError("cao.request_timeout_seconds must be > 0")

        if self.escalation.max_consecutive_coder_failures < 1:
            raise V3ConfigError("max_consecutive_coder_failures must be >= 1")
        if self.escalation.stagnation_rounds_threshold < 1:
            raise V3ConfigError("stagnation_rounds_threshold must be >= 1")

        if self.ci_policy.ci_wait_timeout_seconds < 1:
            raise V3ConfigError("ci_policy.ci_wait_timeout_seconds must be >= 1")

        for budget in (
            "max_total_iterations",
            "max_commits_per_run",
            "max_coder_invocations_per_run",
            "max_reviewer_triggers_per_run",
        ):
            if getattr(self.safety, budget) < 1:
                raise V3ConfigError(f"safety.{budget} must be >= 1")
        if self.safety.max_prompt_tokens < 1:
            raise V3ConfigError("safety.max_prompt_tokens must be >= 1")

    def _validate_telemetry(self) -> None:
        """Validate the telemetry section and its overlap with the catalog."""
        from .telemetry import CatalogTelemetrySource

        telemetry = self.telemetry
        for name, value in (
            ("snapshot_ttl_seconds", telemetry.snapshot_ttl_seconds),
            ("probe_timeout_seconds", telemetry.probe_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise V3ConfigError(f"telemetry.{name} must be a finite number > 0, got {value}")
        if telemetry.health_window_size < 1:
            raise V3ConfigError(
                f"telemetry.health_window_size must be >= 1, got {telemetry.health_window_size}"
            )

        names = [resource.name for resource in telemetry.resources]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise V3ConfigError(f"Duplicate telemetry resource names: {duplicates}")

        for resource in telemetry.resources:
            if not resource.name:
                raise V3ConfigError("telemetry resource name must be non-empty")
            if not resource.provider:
                raise V3ConfigError(
                    f"telemetry resource {resource.name!r} provider must be non-empty"
                )
            if resource.resource_class not in VALID_RESOURCE_CLASSES:
                raise V3ConfigError(
                    f"telemetry resource {resource.name!r} has unknown resource_class "
                    f"{resource.resource_class!r}, must be one of "
                    f"{sorted(VALID_RESOURCE_CLASSES)}"
                )
            if resource.ttl_seconds is not None and (
                not math.isfinite(resource.ttl_seconds) or resource.ttl_seconds <= 0
            ):
                raise V3ConfigError(
                    f"telemetry resource {resource.name!r} ttl_seconds must be a finite "
                    f"number > 0, got {resource.ttl_seconds}"
                )

        # Two sources claiming one resource would make its telemetry ambiguous.
        # The registry rejects that at construction; catching it here names the
        # config key at fault instead of failing on the first collection.
        # Only checkable for an inline catalog — a catalog_path is re-checked
        # in load_v3_config, which is where the shared file is read.
        if telemetry.include_catalog_resources and self.model_router.catalog:
            catalog_owned = set(
                CatalogTelemetrySource(
                    ModelCatalog(entries=tuple(self.model_router.catalog))
                ).resources()
            )
            clashes = sorted(catalog_owned.intersection(names))
            if clashes:
                raise V3ConfigError(
                    f"telemetry resources {clashes} collide with free-tier/promotional "
                    "model catalog entries of the same name, which the catalog telemetry "
                    "source already reports. Rename them, or set "
                    "telemetry.include_catalog_resources to false."
                )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extras":
                out.update(self.extras)
                continue
            out[f.name] = to_mapping(getattr(self, f.name))
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V3Config:
        known = {f.name for f in fields(cls)} - {"extras"}
        kwargs: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
            else:
                extras[key] = value
        config = cls(**kwargs, extras=extras)
        config.validate()
        return config


# --- Loading ---------------------------------------------------------------


def load_v3_config(path: str | Path) -> V3Config:
    """Load and validate a V3 config from a YAML file path.

    When the config declares ``model_router.catalog_path``, the shared
    catalog is loaded here (config I/O belongs in this function, not in the
    pure ``from_dict`` validator) so lane assignments are checked against the
    refs that will actually be dispatched.
    """
    config_path = Path(path)
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise V3ConfigError(f"Failed to read V3 configuration file {config_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise V3ConfigError(f"Invalid YAML in V3 configuration file {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise V3ConfigError("V3 configuration must be a YAML mapping at the top level")
    config = V3Config.from_dict(raw)

    if config.model_router.catalog_path:
        from .telemetry import CatalogTelemetrySource

        catalog = resolve_model_catalog(config, base_dir=config_path.parent)
        refs = catalog.refs()
        for lane_name, model_ref in config.model_router.lane_assignments.items():
            if model_ref not in refs:
                raise V3ConfigError(
                    f"lane {lane_name!r} references unknown model ref {model_ref!r} "
                    f"(not in shared catalog {config.model_router.catalog_path})"
                )
        if config.telemetry.include_catalog_resources:
            clashes = sorted(
                set(CatalogTelemetrySource(catalog).resources()).intersection(
                    resource.name for resource in config.telemetry.resources
                )
            )
            if clashes:
                raise V3ConfigError(
                    f"telemetry resources {clashes} collide with free-tier/promotional "
                    f"entries in the shared catalog {config.model_router.catalog_path}; "
                    "rename them, or set telemetry.include_catalog_resources to false"
                )
    return config


def resolve_model_catalog(config: V3Config, *, base_dir: Path | None = None) -> ModelCatalog:
    """Return the effective model catalog for ``config``.

    Either the inline entries or the shared catalog file named by
    ``model_router.catalog_path``, which is resolved relative to ``base_dir``
    (normally the directory holding the config file) when it is not already
    absolute. The config is never mutated, and each call re-reads the file,
    so an operator can edit the shared catalog and pick the change up for
    future assignments without restarting.
    """
    catalog_path = config.model_router.catalog_path
    if not catalog_path:
        return ModelCatalog(entries=tuple(config.model_router.catalog))
    resolved = Path(catalog_path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = base_dir / resolved
    return load_model_catalog(resolved)


# --- Serialization helpers -------------------------------------------------


def _section_from_dict(section: str, value: Any) -> Any:
    if value is None:
        return _SECTION_TYPES[section]()
    section_type = _SECTION_TYPES[section]
    if not isinstance(value, dict):
        raise V3ConfigError(
            f"config section {section!r} must be a mapping, got {type(value).__name__}"
        )
    kwargs, extras = typed_kwargs(section_type, value)
    _validate_shapes(section_type, kwargs, repr(section), V3ConfigError)
    for (section_name, f_name), nested in _NESTED_LIST_FIELDS.items():
        if section_name == section and f_name in kwargs and isinstance(kwargs[f_name], list):
            kwargs[f_name] = [
                build_dataclass(nested, item, V3ConfigError) for item in kwargs[f_name]
            ]
    kwargs["extras"] = extras
    return section_type(**kwargs)
