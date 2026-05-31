"""Structured JSON logging with secret redaction.

The orchestrator runs inside GitHub Actions where logs are world-readable to
anyone with repo access. Two properties matter:

* **Structured** — every emitted line is a single JSON object so downstream
  tooling can parse state transitions, action executions, and errors without
  scraping free-form text.
* **Redacted** — configured secret values (``GH_TOKEN`` and any env var names
  the operator lists in ``main_coder.env``) must never appear verbatim in the
  log stream, even when they leak into an exception message or a multi-line
  subprocess dump.

``setup_logging`` wires a :class:`JsonFormatter` and a
:class:`SecretRedactingFilter` (sharing one :class:`SecretRedactor`) onto the
package logger. The structured ``log_*`` helpers emit the specific event
shapes the rest of the orchestrator relies on.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TextIO

PACKAGE_LOGGER = "ai_pr_orchestrator"
REDACTION_PLACEHOLDER = "***"

# Token shapes that should always be redacted even if the operator did not list
# them explicitly: GitHub personal-access / installation / OAuth tokens. Mirrors
# the client-side redaction in ``github/client.py`` so a token that slips into a
# message we did not anticipate is still masked. The capture group keeps the
# short non-secret prefix (e.g. ``ghp_``) so a redacted line is still
# debuggable ("which kind of token leaked") while the secret body is masked.
_TOKEN_PATTERN = re.compile(r"(gh[pousr]_|github_pat_)[A-Za-z0-9_]+")

# Secrets shorter than this are not registered for redaction. A 1-3 character
# "secret" (e.g. a misconfigured env var holding "a" or "in") would otherwise
# replace those common substrings everywhere and corrupt the entire log stream.
# Real credentials (tokens, keys) are comfortably longer than this floor.
_MIN_SECRET_LENGTH = 4

# Standard ``LogRecord`` attributes. Anything *not* in this set that ends up on
# a record is treated as a caller-supplied structured ``extra`` field and is
# merged into the JSON payload.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class SecretRedactor:
    """Replaces known secret values (and token-shaped substrings) with ``***``.

    Secrets are matched longest-first so that a secret which is a substring of
    another secret does not leave a recognizable tail behind.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: list[str] = []
        for secret in secrets:
            self.add(secret)

    def add(self, value: str | None) -> None:
        """Register a literal secret value to redact.

        No-ops on empty/blank values and on values shorter than
        ``_MIN_SECRET_LENGTH`` so a misconfigured short secret can't mass-redact
        common substrings out of every log line.
        """
        if not value or not value.strip() or len(value) < _MIN_SECRET_LENGTH:
            return
        if value not in self._secrets:
            self._secrets.append(value)
            # Longest-first: a shorter secret must not pre-empt a longer one
            # that contains it, which would leave the longer secret's tail.
            self._secrets.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTION_PLACEHOLDER)
        # ``\g<1>`` preserves the token-kind prefix (e.g. ``ghp_``) while masking
        # the secret body — keeping the line debuggable, per _TOKEN_PATTERN. The
        # explicit group syntax stays unambiguous even if REDACTION_PLACEHOLDER
        # ever begins with a digit (``\1***`` is fine, ``\g<1>`` is future-proof).
        return _TOKEN_PATTERN.sub(rf"\g<1>{REDACTION_PLACEHOLDER}", text)

    def redact_recursive(self, value: object, _active: set[int] | None = None) -> object:
        """Redact secrets from a value, recursing into nested containers.

        Strings are redacted; dicts/lists/tuples/sets are rebuilt with each
        element redacted; everything else is returned unchanged. Lets the filter
        and formatter scrub secrets nested inside structured ``extra`` fields
        (e.g. ``extra={"detail": {"token": ...}}``), not just top-level strings.

        ``_active`` tracks the ids of containers on the *current* recursion path
        so a circular reference is replaced with a marker instead of recursing
        forever (which would raise ``RecursionError`` here — and leaving the
        cycle in place would later crash ``json.dumps`` with a circular-reference
        error). Ids are removed on the way back up, so a container merely *shared*
        between siblings (a DAG, not a cycle) is still fully redacted.
        """
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict | list | tuple | set):
            if _active is None:
                _active = set()
            marker = id(value)
            if marker in _active:
                return "<circular>"
            _active.add(marker)
            try:
                if isinstance(value, dict):
                    return {k: self.redact_recursive(v, _active) for k, v in value.items()}
                if isinstance(value, list):
                    return [self.redact_recursive(v, _active) for v in value]
                if isinstance(value, tuple):
                    return tuple(self.redact_recursive(v, _active) for v in value)
                return {self.redact_recursive(v, _active) for v in value}
            finally:
                _active.discard(marker)
        return value


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from a record before formatting.

    Operates on the rendered message and on structured ``extra`` fields
    (recursing into nested containers). Pairing it with a :class:`JsonFormatter`
    that shares the same redactor means even handlers without the JSON formatter
    (e.g. a test's ``caplog``) see redacted text.
    """

    def __init__(self, redactor: SecretRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        # Collapse msg+args into a single redacted string so the secret can't
        # survive inside a deferred ``%`` argument. ``getMessage()`` does
        # ``msg % args`` and can raise on a malformed log call (mismatched
        # placeholders/args); a filter that raises is NOT caught by the logging
        # machinery and would crash the caller, so swallow it and leave msg/args
        # for the handler's emit() to render under its own error handling.
        if record.args:
            try:
                record.msg = record.getMessage()
                record.args = ()
            except Exception:
                pass
        if isinstance(record.msg, str):
            record.msg = self._redactor.redact(record.msg)
        for key, value in list(record.__dict__.items()):
            if key in _RESERVED_RECORD_KEYS:
                continue
            # Recurse so secrets nested inside structured ``extra`` containers
            # (dicts/lists/...) are redacted, not just top-level strings.
            record.__dict__[key] = self._redactor.redact_recursive(value)
        return True


class JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object.

    Always present: ``ts`` (UTC ISO-8601), ``level``, ``logger``, ``message``.
    Caller-supplied ``extra`` fields (e.g. ``event``, ``pr``, ``from``, ``to``)
    are merged at the top level. When the record carries exception info a
    ``traceback`` field is added. If a ``redactor`` is supplied, every value
    (including the rendered traceback and nested ``extra`` containers) is
    redacted at format time as a backstop in case the filter was not installed.
    """

    def __init__(self, redactor: SecretRedactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key in payload:
                continue
            payload[key] = value
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        if self._redactor is not None:
            payload = {k: self._redact_value(v) for k, v in payload.items()}
        # ``sort_keys`` is intentionally omitted: arbitrary logged ``extra`` data
        # can carry a dict with mixed key types (e.g. int and str), which makes
        # sorting raise ``TypeError`` and crash the formatter. Insertion order is
        # deterministic enough for structured logs, and parsers don't rely on it.
        formatted = json.dumps(payload, default=str)
        if self._redactor is not None:
            # Final catch-all: redact the serialized string too, so secrets that
            # only surface via a custom object's ``__str__`` (rendered by
            # ``default=str`` *after* per-value redaction) are still masked.
            formatted = self._redactor.redact(formatted)
        return formatted

    def _redact_value(self, value: object) -> object:
        if self._redactor is None:
            return value
        return self._redactor.redact_recursive(value)


def collect_secret_values(env_var_names: Iterable[str]) -> list[str]:
    """Resolve env-var *names* to their current values for redaction.

    ``GH_TOKEN`` is always included (when set) so the GitHub credential is
    redacted even if the operator did not list it in ``main_coder.env``.
    Names that are unset in the environment are skipped.
    """
    names = ["GH_TOKEN", "GITHUB_TOKEN", *env_var_names]
    values: list[str] = []
    seen: set[str] = set()
    for name in names:
        value = os.environ.get(name)
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def setup_logging(
    *,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    secrets: Iterable[str] = (),
    logger_name: str = PACKAGE_LOGGER,
) -> SecretRedactor:
    """Configure structured JSON logging for the package logger.

    Returns the :class:`SecretRedactor` so callers can register additional
    secrets discovered after setup (e.g. a token fetched at runtime). Replaces
    any handlers previously installed by this function so repeated calls (tests,
    re-entrant CLI invocations) don't double-emit.
    """
    redactor = SecretRedactor(secrets)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(redactor=redactor))
    handler.addFilter(SecretRedactingFilter(redactor))

    pkg_logger = logging.getLogger(logger_name)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(_coerce_level(level))
    # Don't propagate to the root logger: that would re-emit each line through
    # any root handler (e.g. pytest's) as unstructured text.
    pkg_logger.propagate = False
    return redactor


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    # A stringified integer ("30") is a numeric level: ``getLevelName("30")``
    # would return "Level 30" (a str) and silently fall back to INFO, so parse
    # it directly first.
    try:
        return int(level)
    except ValueError:
        pass
    resolved = logging.getLevelName(level.upper())
    # ``getLevelName`` returns the string "Level X" for unknown names rather
    # than raising; fall back to INFO so a typo never silences logging.
    return resolved if isinstance(resolved, int) else logging.INFO


# ---- Structured event helpers ----
#
# These emit the specific record shapes the orchestrator's observability relies
# on. Keeping them here (rather than scattering ``extra={...}`` dicts across the
# runner) means the field names live in one place.


def log_state_transition(
    logger: logging.Logger,
    *,
    pr: int,
    from_status: str,
    to_status: str,
    head_sha: str,
    level: int = logging.INFO,
    dry_run: bool = False,
) -> None:
    """Emit a ``state_transition`` event with ``pr``/``from``/``to``/``head_sha``.

    When ``dry_run`` is set the record carries ``dry_run: true`` so downstream
    observability/auditing tools don't mistake a planned transition for one that
    actually executed.
    """
    extra: dict[str, object] = {
        "event": "state_transition",
        "pr": pr,
        "from": from_status,
        "to": to_status,
        "head_sha": head_sha,
    }
    if dry_run:
        extra["dry_run"] = True
    logger.log(level, "PR #%s: %s -> %s", pr, from_status, to_status, extra=extra)


def log_action(
    logger: logging.Logger,
    *,
    pr: int,
    action_type: str,
    level: int = logging.INFO,
    dry_run: bool = False,
) -> None:
    """Emit an ``action`` event with ``action_type`` and ``pr``.

    When ``dry_run`` is set the record carries ``dry_run: true`` so a *planned*
    action is never mistaken for one that actually ran.
    """
    extra: dict[str, object] = {"event": "action", "action_type": action_type, "pr": pr}
    if dry_run:
        extra["dry_run"] = True
    logger.log(level, "PR #%s: executing action %s", pr, action_type, extra=extra)


def log_error(
    logger: logging.Logger,
    *,
    error: object,
    pr: int | None = None,
    exc_info: bool | BaseException = True,
) -> None:
    """Emit an ``error`` event carrying the error message and a ``traceback``.

    ``exc_info`` defaults to ``True`` so the active exception's traceback is
    captured; pass an explicit exception (or ``False``) to override.
    """
    extra: dict[str, object] = {"event": "error", "error": str(error)}
    if pr is not None:
        extra["pr"] = pr
    logger.error("error: %s", error, exc_info=exc_info, extra=extra)
