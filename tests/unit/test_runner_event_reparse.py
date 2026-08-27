"""Regression tests for the payload-first event parse policy.

These exercise the *real* reparse path inside ``runner.run()`` (the CLI
forwards only the event path, and ``run()`` reparses it with the ambient
``GITHUB_EVENT_NAME``) rather than mocking ``runner.run`` away entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ai_pr_orchestrator import cli, runner


def test_cli_contradicting_event_name_still_uses_payload_pr(tmp_path: Path, monkeypatch) -> None:
    """A ``pull_request`` payload with an unrelated GITHUB_EVENT_NAME must
    resolve its PR number from the payload, never from the ambient name
    (the hint is only trusted when it agrees with what the payload implies)."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"number": 456, "head": {"sha": "abc123"}}}),
        encoding="utf-8",
    )

    calls: list[tuple[int, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append((pr_number, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path)]) == 0
    assert calls == [(456, event_path)]


def test_runner_run_reparse_preserves_head_sha_under_unrelated_event_name(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression for the downstream double-parse: an explicit ``check_run``
    payload with an unrelated GITHUB_EVENT_NAME must still yield a ParsedEvent
    carrying ``head_sha``, so the stale-CI-event guard in ci_wait is not
    silently skipped. Exercises the real ``runner.run`` reparse path."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    event_path = tmp_path / "check_run.json"
    event_path.write_text(
        json.dumps(
            {
                "check_run": {
                    "head_sha": "deadbeef1234",
                    "pull_requests": [{"number": 7}],
                }
            }
        ),
        encoding="utf-8",
    )

    captured: list[tuple[int, runner.ParsedEvent | None]] = []

    class _FakeRunner:
        def __init__(self, ctx: object) -> None:
            self._ctx = ctx

        def run(self, pr_number: int, *, event: runner.ParsedEvent | None = None) -> int:
            captured.append((pr_number, event))
            return 0

    monkeypatch.setattr(
        runner, "load_config", lambda: SimpleNamespace(main_coder=SimpleNamespace(env=[]))
    )
    monkeypatch.setattr(runner, "collect_secret_values", lambda _env: [])
    monkeypatch.setattr(runner, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_build_runtime_context",
        lambda _config, **_kwargs: SimpleNamespace(github=object()),
    )
    monkeypatch.setattr(runner, "_close_github", lambda _ctx: None)
    monkeypatch.setattr(runner, "Runner", _FakeRunner)

    assert runner.run(pr_number=7, dry_run=False, event_path=event_path) == 0
    assert len(captured) == 1
    pr_number, event = captured[0]
    assert pr_number == 7
    assert event is not None
    assert event.event_type == "check_run"
    assert event.head_sha == "deadbeef1234"
    assert event.pr_number == 7


def test_cli_review_comment_payload_with_matching_hint(tmp_path: Path, monkeypatch) -> None:
    """A ``pull_request_review_comment`` payload with the matching ambient
    GITHUB_EVENT_NAME must parse as pull_request_review_comment (not the
    coarser ``pull_request``), so the hint is not rejected as contradictory."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_review_comment")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {"number": 789, "head": {"sha": "sha-rc"}},
                "comment": {"id": 4},
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[int, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append((pr_number, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path)]) == 0
    assert calls == [(789, event_path)]


def test_runner_run_reparse_review_comment_payload(tmp_path: Path, monkeypatch) -> None:
    """The runner-level reparse must likewise accept a matching
    pull_request_review_comment hint on a review-comment payload."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_review_comment")
    event_path = tmp_path / "review_comment.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {"number": 8, "head": {"sha": "sha-rc8"}},
                "comment": {"id": 9},
            }
        ),
        encoding="utf-8",
    )

    captured: list[tuple[int, runner.ParsedEvent | None]] = []

    class _FakeRunner:
        def __init__(self, ctx: object) -> None:
            self._ctx = ctx

        def run(self, pr_number: int, *, event: runner.ParsedEvent | None = None) -> int:
            captured.append((pr_number, event))
            return 0

    monkeypatch.setattr(
        runner, "load_config", lambda: SimpleNamespace(main_coder=SimpleNamespace(env=[]))
    )
    monkeypatch.setattr(runner, "collect_secret_values", lambda _env: [])
    monkeypatch.setattr(runner, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_build_runtime_context",
        lambda _config, **_kwargs: SimpleNamespace(github=object()),
    )
    monkeypatch.setattr(runner, "_close_github", lambda _ctx: None)
    monkeypatch.setattr(runner, "Runner", _FakeRunner)

    assert runner.run(pr_number=8, dry_run=False, event_path=event_path) == 0
    assert len(captured) == 1
    pr_number, event = captured[0]
    assert pr_number == 8
    assert event is not None
    assert event.event_type == "pull_request_review_comment"
    assert event.head_sha == "sha-rc8"
    assert event.pr_number == 8


def test_runner_run_reparse_honors_hint_when_payload_is_ambiguous(
    tmp_path: Path, monkeypatch
) -> None:
    """When the payload's keys do not identify an event type, the ambient
    GITHUB_EVENT_NAME hint still applies."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "status")
    event_path = tmp_path / "status.json"
    event_path.write_text(
        json.dumps({"sha": "abc123def456", "state": "success"}),
        encoding="utf-8",
    )

    captured: list[runner.ParsedEvent | None] = []

    class _FakeRunner:
        def __init__(self, ctx: object) -> None:
            self._ctx = ctx

        def run(self, pr_number: int, *, event: runner.ParsedEvent | None = None) -> int:
            captured.append(event)
            return 0

    monkeypatch.setattr(
        runner, "load_config", lambda: SimpleNamespace(main_coder=SimpleNamespace(env=[]))
    )
    monkeypatch.setattr(runner, "collect_secret_values", lambda _env: [])
    monkeypatch.setattr(runner, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_build_runtime_context",
        lambda _config, **_kwargs: SimpleNamespace(github=object()),
    )
    monkeypatch.setattr(runner, "_close_github", lambda _ctx: None)
    monkeypatch.setattr(runner, "Runner", _FakeRunner)

    assert runner.run(pr_number=7, dry_run=False, event_path=event_path) == 0
    assert captured[0] is not None
    assert captured[0].event_type == "status"
    assert captured[0].head_sha == "abc123def456"
