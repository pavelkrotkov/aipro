"""Tests for the ``aipro reconcile`` subcommand (issue #44)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_pr_orchestrator import cli

CONFIG_BODY = """
github_queue:
  enabled_label: v3-work
  lease_seconds: 900
cao:
  base_url: http://localhost:9889
cleanup:
  session_lease_ttl_seconds: 7200
  worktree_inactivity_ttl_seconds: 86400
"""


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "v3.yml"
    config_path.write_text(CONFIG_BODY, encoding="utf-8")
    return config_path


def test_reconcile_dry_run_with_no_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path)
    exit_code = cli.main(["reconcile", "--config", str(config_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Nothing to reconcile" in out


def test_reconcile_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path)
    exit_code = cli.main(["reconcile", "--config", str(config_path), "--json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "actions" in parsed


def test_reconcile_invalid_issue_arg(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["reconcile", "--config", str(config_path), "--issue", "0"])


def test_reconcile_missing_config(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["reconcile", "--config", str(tmp_path / "missing.yml")])
