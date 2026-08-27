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

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .domain import VALID_LANE_ROLES


class V3ConfigError(ValueError):
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
class ModelCatalogEntry:
    """One model catalog entry.

    ``ref`` is the policy-level key; ``descriptor`` is an opaque,
    provider-owned string that only the model broker interprets. V3 core
    never parses it.
    """

    ref: str
    descriptor: str
    max_context_tokens: int | None = None
    #: Unknown keys from a newer writer, preserved for forward compatibility.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRouterConfig:
    """Model catalog plus lane-to-model routing."""

    catalog: list[ModelCatalogEntry] = field(default_factory=list)
    lane_assignments: dict[str, str] = field(default_factory=dict)  # lane -> model ref
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
        lane_names = [lane.name for lane in self.hermes_lanes.lanes]
        if len(lane_names) != len(set(lane_names)):
            raise V3ConfigError(f"Duplicate lane names in hermes_lanes: {lane_names}")

        foremen = [lane for lane in self.hermes_lanes.lanes if lane.role == "foreman"]
        if len(foremen) > 1:
            raise V3ConfigError("At most one foreman lane is allowed")

        for lane in self.hermes_lanes.lanes:
            if lane.role not in VALID_LANE_ROLES:
                raise V3ConfigError(f"Invalid role {lane.role!r} for lane {lane.name!r}")
            if lane.max_concurrent < 1:
                raise V3ConfigError(f"lane {lane.name!r} max_concurrent must be >= 1")

        catalog_refs = [entry.ref for entry in self.model_router.catalog]
        if len(catalog_refs) != len(set(catalog_refs)):
            raise V3ConfigError(f"Duplicate model catalog refs: {catalog_refs}")

        for lane_name, model_ref in self.model_router.lane_assignments.items():
            if lane_name not in lane_names:
                raise V3ConfigError(f"lane_assignments references unknown lane {lane_name!r}")
            if model_ref not in catalog_refs:
                raise V3ConfigError(
                    f"lane {lane_name!r} references unknown model ref {model_ref!r}"
                )

        for reviewer in self.review_policy.reviewer_lanes:
            lane = next(
                (c for c in self.hermes_lanes.lanes if c.name == reviewer),
                None,
            )
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

        if self.escalation.max_consecutive_coder_failures < 1:
            raise V3ConfigError("max_consecutive_coder_failures must be >= 1")
        if self.escalation.stagnation_rounds_threshold < 1:
            raise V3ConfigError("stagnation_rounds_threshold must be >= 1")

        if self.ci_policy.ci_wait_timeout_seconds < 1:
            raise V3ConfigError("ci_policy.ci_wait_timeout_seconds must be >= 1")

        for entry in self.model_router.catalog:
            if entry.max_context_tokens is not None and entry.max_context_tokens < 1:
                raise V3ConfigError(
                    f"model catalog entry {entry.ref!r} max_context_tokens must be >= 1"
                )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extras":
                out.update(self.extras)
                continue
            out[f.name] = _section_to_dict(getattr(self, f.name))
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
    """Load and validate a V3 config from a YAML file path."""
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
    return V3Config.from_dict(raw)


# --- Serialization helpers -------------------------------------------------


def _section_to_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {
            f.name: _section_to_dict(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.name != "extras"
        }
        # Merge preserved unknown keys back into the emitted mapping.
        out.update(getattr(value, "extras", {}))
        return out
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, list):
        return [_section_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: _section_to_dict(v) for k, v in value.items()}
    return value


def _section_from_dict(section: str, value: Any) -> Any:
    if value is None:
        return _SECTION_TYPES[section]()
    section_type = _SECTION_TYPES[section]
    if not isinstance(value, dict):
        raise V3ConfigError(
            f"config section {section!r} must be a mapping, got {type(value).__name__}"
        )
    kwargs, extras = _typed_kwargs(section_type, value)
    for (section_name, f_name), nested in _NESTED_LIST_FIELDS.items():
        if section_name == section and f_name in kwargs and isinstance(kwargs[f_name], list):
            kwargs[f_name] = [_build_dataclass(nested, item) for item in kwargs[f_name]]
    kwargs["extras"] = extras
    return section_type(**kwargs)


def _build_dataclass(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise V3ConfigError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs, extras = _typed_kwargs(cls, data)
    kwargs["extras"] = extras
    return cls(**kwargs)


def _typed_kwargs(cls: type, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a raw mapping into known-field kwargs and unknown-key extras."""
    known = {f.name for f in fields(cls)} - {"extras"}
    kwargs: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in data.items():
        if key in known:
            kwargs[key] = value
        else:
            extras[key] = value
    return kwargs, extras
