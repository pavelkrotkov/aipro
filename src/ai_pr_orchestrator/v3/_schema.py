"""Shared mapping/dataclass coercion helpers for V3 declarative schemas.

Both the V3 policy config (:mod:`ai_pr_orchestrator.v3.config`) and the shared
model catalog (:mod:`ai_pr_orchestrator.v3.catalog`) are plain dataclasses
built from YAML mappings with the same three rules:

- declared field shapes are checked *before* construction, so a malformed
  value is a schema error rather than a later ``AttributeError``;
- ``None`` is rejected for any field whose annotation does not explicitly
  allow it, so a null never silently bypasses validation;
- unknown keys are preserved in an ``extras`` bucket and written back on
  serialization, so payloads from newer writers round-trip losslessly.

These helpers live here rather than in ``config`` so ``catalog`` can reuse
them without importing ``config`` (which imports ``catalog`` for its entry
type — the reverse direction would be a cycle).
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import fields
from datetime import datetime
from typing import Any, Union, get_args, get_origin, get_type_hints


class SchemaError(ValueError):
    """Raised when a declarative payload does not match its dataclass schema.

    Both :class:`~ai_pr_orchestrator.v3.config.V3ConfigError` and
    :class:`~ai_pr_orchestrator.v3.catalog.ModelCatalogError` derive from this,
    so a caller loading either artifact can catch one type.
    """


def is_optional(hint: Any) -> bool:
    """True when ``hint`` explicitly allows ``None`` (``X | None``)."""
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(hint)
    return False


def value_matches_type(hint: Any, value: Any) -> bool:
    """True when ``value`` conforms to the declared type ``hint``."""
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in get_args(hint) if arg is not type(None)]
        return any(value_matches_type(arg, value) for arg in non_none)
    if origin in (list, tuple, set, frozenset) or hint in (list, tuple, set, frozenset):
        if not isinstance(value, (list, tuple, set, frozenset)):
            return False
        args = get_args(hint)
        if not args:  # bare list/dict: container type only.
            return True
        return all(value_matches_type(args[0], item) for item in value)
    if origin is dict or hint is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(hint)
        if not args:
            return True
        key_hint, value_hint = args
        return all(
            value_matches_type(key_hint, k) and value_matches_type(value_hint, v)
            for k, v in value.items()
        )
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is str:
        return isinstance(value, str)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return isinstance(value, dict)
    return True  # Any / unhandled annotations: do not over-constrain.


def validate_declared_shapes(
    cls: type, data: dict[str, Any], where: str, error: type[SchemaError]
) -> None:
    """Check declared field shapes in a raw mapping, raising ``error``.

    Runs before dataclass construction so a malformed nested value (e.g. a
    list field given a scalar, or an int field given a string) is reported
    as a schema error instead of surfacing later as an AttributeError.
    """
    hints = get_type_hints(cls)
    for name, hint in hints.items():
        if name == "extras" or name not in data:
            continue
        if data[name] is None:
            if not is_optional(hint):
                raise error(
                    f"config {where}: field {name!r} expects {hint}, got null "
                    "(None is only valid for Optional fields)"
                )
            continue
        if not value_matches_type(hint, data[name]):
            raise error(
                f"config {where}: field {name!r} expects {hint}, "
                f"got {type(data[name]).__name__} ({data[name]!r})"
            )


def typed_kwargs(cls: type, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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


def build_dataclass(cls: type, data: Any, error: type[SchemaError]) -> Any:
    """Build ``cls`` from a raw mapping, validating shapes and keeping extras."""
    if not isinstance(data, dict):
        raise error(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs, extras = typed_kwargs(cls, data)
    validate_declared_shapes(cls, kwargs, cls.__name__, error)
    kwargs["extras"] = extras
    return cls(**kwargs)


def to_mapping(value: Any) -> Any:
    """Serialize a dataclass tree back to plain mappings, merging ``extras``.

    Datetimes are emitted in ISO 8601 form so the result is JSON-encodable,
    not just YAML-encodable.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {
            f.name: to_mapping(getattr(value, f.name))
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
        return [to_mapping(v) for v in value]
    if isinstance(value, dict):
        return {k: to_mapping(v) for k, v in value.items()}
    return value
