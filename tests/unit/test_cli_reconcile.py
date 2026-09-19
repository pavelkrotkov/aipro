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
  owner: test-owner
  repo: test-repo
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
    # The fake queue has no work items, so the planner emits a single
    # NOOP action with reason "No claim yet …" — that is the equivalent
    # of the old "Nothing to reconcile" textual signal.
    assert "noop" in out.lower() or "Nothing to reconcile" in out


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


def test_apply_rejects_even_authenticated_noop_before_incomplete_inventory(
    tmp_path, monkeypatch, capsys
):
    from unittest.mock import Mock

    from ai_pr_orchestrator.github.fake import FakeGitHubClient

    client_factory = Mock(return_value=(FakeGitHubClient(), False))
    monkeypatch.setattr(cli, "_build_github_client", client_factory)
    result = cli.main(
        ["reconcile", "--config", str(write_config(tmp_path)), "--apply", "--token", "test-token"]
    )
    assert result == 3
    assert "--apply is unsupported" in capsys.readouterr().out
    client_factory.assert_not_called()
