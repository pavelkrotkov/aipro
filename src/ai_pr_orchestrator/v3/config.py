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

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ._schema import SchemaError, build_dataclass, to_mapping, typed_kwargs
from ._schema import validate_declared_shapes as _validate_shapes
from .catalog import ModelCatalog, ModelCatalogEntry, load_model_catalog
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
    state_comment_marker: str = "v3-runtime-state"
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
class SafetyPolicyConfig:
    """Safety rails carried over from the V1 safety surface.

    These controls bound what the engine is allowed to touch and how much
    work one run may do; they must survive the V1→V3 cutover, so they are
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
    "review_policy": ReviewPolicyConfig,
    "ci_policy": CIPolicyConfig,
    "safety": SafetyPolicyConfig,
    "escalation": EscalationPolicyConfig,
}

_NESTED_LIST_FIELDS: dict[tuple[str, str], type] = {
    ("hermes_lanes", "lanes"): LaneProfileConfig,
    ("model_router", "catalog"): ModelCatalogEntry,
}


# --- Root config -----------------------------------------------------------


@dataclass(frozen=True)
class V3Config:
    """Root of the V3 policy schema."""

    github_queue: GitHubQueueConfig = field(default_factory=GitHubQueueConfig)
    cao: CAOControlPlaneConfig = field(default_factory=CAOControlPlaneConfig)
    hermes_lanes: HermesLanesConfig = field(default_factory=HermesLanesConfig)
    model_router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
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
        labels = (q.enabled_label, q.done_label, q.error_label)
        if len(labels) != len(set(labels)):
            raise V3ConfigError(
                "github_queue lifecycle labels must be distinct: "
                f"enabled={q.enabled_label!r} done={q.done_label!r} error={q.error_label!r}"
            )

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
        catalog = resolve_model_catalog(config, base_dir=config_path.parent)
        refs = catalog.refs()
        for lane_name, model_ref in config.model_router.lane_assignments.items():
            if model_ref not in refs:
                raise V3ConfigError(
                    f"lane {lane_name!r} references unknown model ref {model_ref!r} "
                    f"(not in shared catalog {config.model_router.catalog_path})"
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
