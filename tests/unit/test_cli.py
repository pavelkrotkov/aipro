import json
from datetime import timedelta
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
    # Pin the ambient event name so this test is deterministic regardless of
    # the CI runner context that executes it (e.g. push-to-main vs PR).
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    event_path = tmp_path / "event.json"
    event_path.write_text('{"pull_request": {"number": 456}}', encoding="utf-8")
    calls: list[tuple[int, bool, Path | None]] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append((pr_number, dry_run, event_path))
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path)]) == 0
    assert calls == [(456, False, event_path)]


def test_run_with_non_positive_event_pr_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
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
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    event_path = tmp_path / "event.json"
    event_path.write_text('{"pull_request": {"number": 456}}', encoding="utf-8")

    calls: list[int] = []

    def fake_run(*, pr_number: int, dry_run: bool, event_path: Path | None) -> int:
        calls.append(pr_number)
        return 0

    monkeypatch.setattr(cli.runner, "run", fake_run)

    assert cli.main(["run", "--event-path", str(event_path), "--pr", "789"]) == 0
    assert calls == [456]


CATALOG_YML = """
models:
  - ref: free-any-role
    descriptor: d1
    resource_class: free_tier
    cost_class: free
    capabilities: [tools, coding]
  - ref: reviewer-only-priced
    descriptor: d2
    resource_class: metered
    cost_class: low
    input_price_per_mtok: 0.5
    output_price_per_mtok: 1.5
    roles: [reviewer]
  - ref: hard-work-only
    descriptor: d3
    resource_class: subscription
    cost_class: high
    input_price_per_mtok: 3.0
    output_price_per_mtok: 15.0
    min_task_difficulty: 5
  - ref: unknown-price
    descriptor: d4
"""


def _catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.yml"
    path.write_text(CATALOG_YML, encoding="utf-8")
    return path


def test_catalog_lists_eligible_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["catalog", "--catalog", str(_catalog(tmp_path))]) == 0

    out = capsys.readouterr().out
    assert "free-any-role" in out
    assert "reviewer-only-priced" in out
    # Difficulty floor and unknown price both exclude a candidate.
    assert "hard-work-only" not in out
    assert "unknown-price" not in out


def test_catalog_filters_by_role_and_difficulty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "catalog",
                "--catalog",
                str(_catalog(tmp_path)),
                "--role",
                "worker",
                "--difficulty",
                "5",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "free-any-role" in out
    assert "hard-work-only" in out
    # Declared for reviewers only.
    assert "reviewer-only-priced" not in out


def test_catalog_json_reports_normalized_price_and_resource_class(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["catalog", "--catalog", str(_catalog(tmp_path)), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    by_ref = {row["ref"]: row for row in payload["candidates"]}
    assert by_ref["free-any-role"]["effective_input_price_per_mtok"] == 0.0
    assert by_ref["free-any-role"]["resource_class"] == "free_tier"
    assert by_ref["reviewer-only-priced"]["effective_output_price_per_mtok"] == 1.5


def test_catalog_all_shows_ineligible_entries_with_unknown_price(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["catalog", "--catalog", str(_catalog(tmp_path)), "--all", "--json"]) == 0

    by_ref = {row["ref"]: row for row in json.loads(capsys.readouterr().out)["candidates"]}
    # Unknown price must not be reported as zero.
    assert by_ref["unknown-price"]["effective_input_price_per_mtok"] is None
    assert by_ref["unknown-price"]["eligible"] is False


def test_catalog_all_text_output_marks_eligibility(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without a column the table renders undispatchable entries exactly like
    # usable ones, which reads as "available".
    assert cli.main(["catalog", "--catalog", str(_catalog(tmp_path)), "--all"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "ELIGIBLE" in lines[0]
    rows = {line.split()[0]: line.split()[-1] for line in lines[1:]}
    assert rows["free-any-role"] == "yes"
    assert rows["unknown-price"] == "no"
    assert rows["hard-work-only"] == "no"


def test_catalog_reads_the_path_declared_by_a_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _catalog(tmp_path)
    config_path = tmp_path / "v3.yml"
    config_path.write_text("model_router:\n  catalog_path: catalog.yml\n", encoding="utf-8")

    assert cli.main(["catalog", "--config", str(config_path)]) == 0
    assert "free-any-role" in capsys.readouterr().out


def test_catalog_requires_exactly_one_source() -> None:
    with pytest.raises(SystemExit):
        cli.main(["catalog"])
    with pytest.raises(SystemExit):
        cli.main(["catalog", "--catalog", "a.yml", "--config", "b.yml"])


def test_catalog_rejects_out_of_range_difficulty(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["catalog", "--catalog", str(_catalog(tmp_path)), "--difficulty", "9"])


def test_catalog_reports_a_malformed_catalog(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yml"
    path.write_text("models: [{ref: a, descriptor: d, cost_class: bogus}]", encoding="utf-8")

    with pytest.raises(SystemExit, match="cost_class"):
        cli.main(["catalog", "--catalog", str(path)])


def test_catalog_reports_when_nothing_is_eligible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "catalog.yml"
    path.write_text("models: [{ref: a, descriptor: d, enabled: false}]", encoding="utf-8")

    assert cli.main(["catalog", "--catalog", str(path)]) == 0
    assert "No eligible catalog candidates." in capsys.readouterr().out


def test_catalog_reports_an_empty_catalog_distinctly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "catalog.yml"
    path.write_text("models: []", encoding="utf-8")

    assert cli.main(["catalog", "--catalog", str(path), "--all"]) == 0
    assert "Catalog is empty." in capsys.readouterr().out


def test_catalog_reports_missing_required_fields_without_a_traceback(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yml"
    path.write_text("models: [{descriptor: d}]", encoding="utf-8")

    with pytest.raises(SystemExit, match="missing required field"):
        cli.main(["catalog", "--catalog", str(path)])


def _telemetry_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "v3.yml"
    path.write_text(body, encoding="utf-8")
    return path


HERMES_ONLY = "telemetry:\n  resources:\n    - {name: anthropic-sub, provider: anthropic}\n"


def test_telemetry_requires_a_config() -> None:
    with pytest.raises(SystemExit):
        cli.main(["telemetry"])


def test_telemetry_reports_no_resources_distinctly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["telemetry", "--config", str(_telemetry_config(tmp_path, "{}\n"))]) == 0
    assert "No telemetry resources are configured." in capsys.readouterr().out


def test_telemetry_without_a_hermes_install_reports_unknown_not_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # CI has no Hermes. The command must still succeed and say "unknown",
    # because reporting an unprobed subscription as having free capacity is
    # the exact failure this data path exists to prevent.
    config_path = _telemetry_config(tmp_path, HERMES_ONLY)

    assert cli.main(["telemetry", "--config", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "anthropic-sub" in out
    assert "unknown" in out
    assert "no Hermes interpreter configured" in out


def test_telemetry_json_carries_a_single_evaluation_timestamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _telemetry_config(tmp_path, HERMES_ONLY)

    assert cli.main(["telemetry", "--config", str(config_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluated_at"]
    rows = {row["resource"]: row for row in payload["resources"]}
    assert rows["anthropic-sub"]["availability"] == "unknown"
    assert rows["anthropic-sub"]["reason"]


def test_telemetry_includes_catalog_perishable_capacity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "catalog.yml").write_text(
        "models: [{ref: free-model, descriptor: d, cost_class: free},"
        " {ref: paid-model, descriptor: d}]\n",
        encoding="utf-8",
    )
    config_path = _telemetry_config(tmp_path, "model_router:\n  catalog_path: catalog.yml\n")

    assert cli.main(["telemetry", "--config", str(config_path), "--json"]) == 0

    refs = {row["resource"] for row in json.loads(capsys.readouterr().out)["resources"]}
    # Only perishable capacity is a telemetry resource; a paid entry is not.
    assert refs == {"free-model"}


def test_telemetry_opting_out_drops_catalog_resources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "catalog.yml").write_text(
        "models: [{ref: free-model, descriptor: d, cost_class: free}]\n", encoding="utf-8"
    )
    config_path = _telemetry_config(
        tmp_path,
        "model_router:\n  catalog_path: catalog.yml\n"
        "telemetry:\n  include_catalog_resources: false\n",
    )

    assert cli.main(["telemetry", "--config", str(config_path)]) == 0
    assert "No telemetry resources are configured." in capsys.readouterr().out


def test_telemetry_reports_a_malformed_config_without_a_traceback(tmp_path: Path) -> None:
    config_path = _telemetry_config(
        tmp_path, "telemetry:\n  resources:\n    - {name: r, provider: p, resource_class: vibes}\n"
    )

    with pytest.raises(SystemExit, match="resource_class"):
        cli.main(["telemetry", "--config", str(config_path)])


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0m"), (90, "1m"), (3600, "1h00m"), (3720, "1h02m"), (86400, "24h00m")],
)
def test_format_duration_is_readable_at_every_scale(seconds: int, expected: str) -> None:
    assert cli._format_duration(timedelta(seconds=seconds)) == expected
