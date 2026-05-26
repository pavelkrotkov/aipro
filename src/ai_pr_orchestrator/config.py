"""Configuration loading and validation."""

import types
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import yaml

VALID_PROVIDERS = {"codex_cli"}
DEFAULT_CONFIG_PATH = Path(".github/ai-review-loop.yml")


class ConfigError(ValueError):
    """Raised when the orchestrator configuration is invalid."""


@dataclass(frozen=True)
class MainCoderConfig:
    provider: str
    command: str = "codex"
    args: list[str] = field(default_factory=lambda: ["exec", "{prompt}"])
    timeout_seconds: int = 1800
    output_file: str = ".ai-orchestrator-result.json"
    env: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewerConfig:
    enabled: bool = True
    bot_logins: list[str] = field(default_factory=list)
    trigger_comment: str = ""


@dataclass(frozen=True)
class ReviewPhaseConfig:
    max_rounds: int = 1
    poll_interval_seconds: int = 30
    reviewer_timeout_seconds: int = 600
    phase_timeout_seconds: int = 900


@dataclass(frozen=True)
class ThreadPolicyConfig:
    auto_resolve_bot_threads: bool = True
    never_resolve_human_threads: bool = True
    resolve_rejected_bot_threads: bool = True
    require_reply_before_resolve: bool = True


@dataclass(frozen=True)
class GitConfig:
    base_branch: str = "main"
    commit_author_name: str = "AI PR Orchestrator"
    commit_author_email: str = "ai-pr-orchestrator@example.com"
    commit_message_prefix: str = "fix: address AI review feedback"


@dataclass(frozen=True)
class CiConfig:
    require_green_before_done: bool = True
    required_checks: list[str] = field(default_factory=list)
    ignored_checks: list[str] = field(default_factory=lambda: ["AI PR Review Loop"])
    timeout_seconds: int = 900
    relevant_failed_log_lines: int = 300


@dataclass(frozen=True)
class SafetyConfig:
    only_run_on_labeled_prs: bool = True
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


@dataclass(frozen=True)
class NotificationsConfig:
    mention_on_needs_human: list[str] = field(default_factory=list)
    mention_on_error: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    main_coder: MainCoderConfig
    enabled_label: str = "ai-loop"
    done_label: str = "ai-loop-done"
    error_label: str = "ai-loop-error"
    reviewers: dict[str, ReviewerConfig] = field(default_factory=dict)
    review_phase: ReviewPhaseConfig = field(default_factory=ReviewPhaseConfig)
    thread_policy: ThreadPolicyConfig = field(default_factory=ThreadPolicyConfig)
    git: GitConfig = field(default_factory=GitConfig)
    ci: CiConfig = field(default_factory=CiConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate an AI PR Orchestrator YAML config file."""
    config_path = Path(path)
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read configuration file {config_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration file {config_path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("config must be a YAML mapping")
    _validate_keys(
        raw,
        {
            "enabled_label",
            "done_label",
            "error_label",
            "main_coder",
            "reviewers",
            "review_phase",
            "thread_policy",
            "git",
            "ci",
            "safety",
            "notifications",
        },
        "config",
    )

    return Config(
        enabled_label=_string(raw, "enabled_label", "ai-loop"),
        done_label=_string(raw, "done_label", "ai-loop-done"),
        error_label=_string(raw, "error_label", "ai-loop-error"),
        main_coder=_main_coder(_mapping(raw, "main_coder", required=True)),
        reviewers=_reviewers(_mapping(raw, "reviewers")),
        review_phase=_dataclass_from_mapping(
            ReviewPhaseConfig, _mapping(raw, "review_phase"), "review_phase"
        ),
        thread_policy=_dataclass_from_mapping(
            ThreadPolicyConfig, _mapping(raw, "thread_policy"), "thread_policy"
        ),
        git=_dataclass_from_mapping(GitConfig, _mapping(raw, "git"), "git"),
        ci=_dataclass_from_mapping(CiConfig, _mapping(raw, "ci"), "ci"),
        safety=_dataclass_from_mapping(SafetyConfig, _mapping(raw, "safety"), "safety"),
        notifications=_dataclass_from_mapping(
            NotificationsConfig, _mapping(raw, "notifications"), "notifications"
        ),
    )


def _main_coder(raw: dict[str, Any]) -> MainCoderConfig:
    config = _dataclass_from_mapping(MainCoderConfig, raw, "main_coder")
    if config.provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"main_coder.provider must be one of {sorted(VALID_PROVIDERS)}, got {config.provider!r}"
        )
    return config


def _reviewers(raw: dict[str, Any]) -> dict[str, ReviewerConfig]:
    reviewers: dict[str, ReviewerConfig] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise ConfigError("reviewers keys must be strings")
        if not isinstance(value, dict):
            raise ConfigError(f"reviewers.{name} must be a mapping")
        reviewers[name] = _dataclass_from_mapping(ReviewerConfig, value, f"reviewers.{name}")
    return reviewers


def _dataclass_from_mapping(type_: type[Any], raw: dict[str, Any], prefix: str) -> Any:
    _validate_keys(raw, set(type_.__dataclass_fields__), prefix)
    type_hints = get_type_hints(type_)

    kwargs: dict[str, Any] = {}
    for field_name, field_info in type_.__dataclass_fields__.items():
        field_path = f"{prefix}.{field_name}"
        if (
            field_info.default is MISSING
            and field_info.default_factory is MISSING
            and field_name not in raw
        ):
            raise ConfigError(f"{field_path} is required")

        default = _default_value(field_info)
        value = raw.get(field_name)
        if value is None:
            value = default
        field_type, is_optional = _resolve_optional_type(type_hints.get(field_name))

        if value is None and is_optional:
            kwargs[field_name] = None
        elif field_type is bool:
            kwargs[field_name] = _bool_value(value, field_path)
        elif field_type is int:
            kwargs[field_name] = _int_value(value, field_path, allow_zero=_allows_zero(field_path))
        elif field_type is str:
            kwargs[field_name] = _string_value(value, field_path)
        elif field_type is list or get_origin(field_type) is list:
            kwargs[field_name] = _string_list_value(value, field_path)
        else:
            raise ConfigError(f"{field_path} has unsupported type annotation {field_type!r}")
    return type_(**kwargs)


def _resolve_optional_type(field_type: Any) -> tuple[Any, bool]:
    origin = get_origin(field_type)
    if origin not in (Union, types.UnionType):
        return field_type, False

    args = get_args(field_type)
    if type(None) not in args:
        return field_type, False

    non_none_args = [arg for arg in args if arg is not type(None)]
    if len(non_none_args) != 1:
        return field_type, True
    return non_none_args[0], True


def _allows_zero(field_path: str) -> bool:
    return field_path == "ci.relevant_failed_log_lines"


def _validate_keys(raw: dict[str, Any], allowed_keys: set[str], prefix: str) -> None:
    for key in raw:
        if key not in allowed_keys:
            raise ConfigError(f"Unknown configuration key: {prefix}.{key}")


def _default_value(field_info: Any) -> Any:
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    if field_info.default is not MISSING:
        return field_info.default
    return None


def _mapping(raw: dict[str, Any], key: str, *, required: bool = False) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{key} section is required")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _string(
    raw: dict[str, Any],
    key: str,
    default: str = "",
    *,
    field_path: str | None = None,
) -> str:
    path = field_path or key
    value = raw.get(key)
    if value is None:
        value = default
    return _string_value(value, path)


def _string_value(value: Any, field_path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_path} must be a string")
    return value


def _bool_value(value: Any, field_path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_path} must be a boolean")
    return value


def _int_value(value: Any, field_path: str, *, allow_zero: bool) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field_path} must be an integer")
    if allow_zero:
        if value < 0:
            raise ConfigError(f"{field_path} must be a non-negative integer")
    elif value <= 0:
        raise ConfigError(f"{field_path} must be a positive integer")
    return value


def _string_list_value(value: Any, field_path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field_path} must be a list of strings")
    return list(cast(list[str], value))
