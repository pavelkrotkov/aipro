from pathlib import Path

import pytest

from ai_pr_orchestrator import cli


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_run_pr_invokes_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, bool, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append(("run", pr_number, dry_run, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--pr", "123"]) == 0
    assert calls == [("run", 123, False, None)]


@pytest.mark.parametrize("command", ["run", "dry-run", "inspect"])
def test_pr_argument_must_be_positive(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([command, "--pr", "0"])

    assert exc_info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_dry_run_pr_sets_dry_run_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, bool, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append((pr_number, dry_run, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["dry-run", "--pr", "123"]) == 0
    assert calls == [(123, True, None)]


def test_inspect_pr_invokes_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_inspect(*, pr_number: int) -> int:
        calls.append(pr_number)
        return 0

    monkeypatch.setattr(cli.runner, "inspect", fake_inspect)

    assert cli.main(["inspect", "--pr", "123"]) == 0
    assert calls == [123]


def test_run_with_event_path_reads_event_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text('{"pull_request": {"number": 456}}', encoding="utf-8")
    calls: list[tuple[int, bool, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append((pr_number, dry_run, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path)]) == 0
    assert calls == [(456, False, event_path)]


def test_run_with_non_positive_event_pr_exits_cleanly(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text('{"pull_request": {"number": 0}}', encoding="utf-8")

    with pytest.raises(SystemExit, match="positive integer"):
        cli.main(["run", "--event-path", str(event_path)])


def test_run_with_missing_event_path_exits_cleanly(tmp_path: Path) -> None:
    event_path = tmp_path / "missing.json"

    with pytest.raises(SystemExit, match="Failed to read event file"):
        cli.main(["run", "--event-path", str(event_path)])


def test_run_with_invalid_event_json_exits_cleanly(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit, match="not valid JSON"):
        cli.main(["run", "--event-path", str(event_path)])


def test_run_with_invalid_event_shape_exits_cleanly(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"Could not determine"):
        cli.main(["run", "--event-path", str(event_path)])


def test_run_with_status_event_exits_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``status`` webhooks carry a SHA but no PR number. Until SHA -> PR
    lookup is wired in the CLI, surface a clear actionable error pointing
    operators at ``--pr`` instead of the generic "could not determine"
    message."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "status")
    event_path = tmp_path / "status.json"
    event_path.write_text(
        '{"sha": "abc123def456", "state": "success"}',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match=r"status events carry a commit SHA"):
        cli.main(["run", "--event-path", str(event_path)])


def test_run_status_event_with_pr_fallback_forwards_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``status`` webhook carries a SHA but no PR number. Passing ``--pr``
    alongside ``--event-path`` resolves the PR while still forwarding the event
    (and its head_sha) into Runner.run, keeping the stale-CI-event guard."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "status")
    event_path = tmp_path / "status.json"
    event_path.write_text(
        '{"sha": "abc123def456", "state": "success"}',
        encoding="utf-8",
    )

    calls: list[tuple[int, bool, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append((pr_number, dry_run, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path), "--pr", "789"]) == 0
    assert calls == [(789, False, event_path)]


def test_run_event_pr_number_wins_over_pr_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the event payload contains a PR number, it takes precedence over an
    explicitly supplied ``--pr``."""
    event_path = tmp_path / "event.json"
    event_path.write_text('{"pull_request": {"number": 456}}', encoding="utf-8")

    calls: list[int] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append(pr_number)
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path), "--pr", "789"]) == 0
    assert calls == [456]
