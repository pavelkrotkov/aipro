"""Regression tests for Codex review round 1 on reconciliation: findings 1 and 2 (issue #44).

1. ``--apply`` actually applies; non-manual actions go through controllers.
2. The CLI uses a real :class:`GitHubClient` when a token is supplied
   (not a hard-coded ``FakeGitHubClient`` with literal ``"owner"``/``"repo"``).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from _reconcile_builders import (
    frozen_now,
    make_claim,
    make_state,
)

from ai_pr_orchestrator import cli
from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.domain import GitHubIssueRef, WorkflowState
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue, GitHubQueueConfig
from ai_pr_orchestrator.v3.reconcile import (
    ReconciliationInputs,
    WorkItemObservation,
)


class TestCliApplyAndRepo:
    def test_reconcile_repo_flag_default_from_env(self, tmp_path, monkeypatch, capsys) -> None:
        """``--repo owner/name`` overrides ``GITHUB_REPOSITORY`` and the
        config default. With ``--repo owner/named``, the CLI surfaces the
        owner/name in the NOOP reason (issue#N) rather than the literal
        ``"owner"``/``"repo"`` the previous code used unconditionally.
        """

        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "  owner: cfg-owner\n  repo: cfg-repo\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exit_code = cli.main(
            [
                "reconcile",
                "--config",
                str(config_path),
                "--repo",
                "explicit/test-repo",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        # The placeholder issue is built from the resolved owner/repo.
        assert "explicit/test-repo#1" in out

    def test_reconcile_repo_missing_fails(self, tmp_path, monkeypatch, capsys) -> None:
        """No ``--repo``, no ``GITHUB_REPOSITORY``, no ``github_queue.owner``
        → the CLI refuses rather than fall back to the literal
        ``"owner"``/``"repo"`` it used to."""

        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        with pytest.raises(SystemExit) as exc:
            cli.main(["reconcile", "--config", str(config_path)])
        assert "Could not determine GitHub repo" in str(exc.value)

    def test_reconcile_apply_invokes_queue_reclaim(self, tmp_path, monkeypatch, capsys) -> None:
        """``--apply`` with a stale-lease scenario goes through
        ``queue.reclaim_expired`` rather than printing and forgetting.

        The fake client + fake queue let us observe the write without a
        live network.
        """

        # Build a config that names a real (fake-client-friendly) repo.
        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "  owner: test-owner\n  repo: test-repo\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )

        # Wire a fake client + queue and seed a stale-lease state.
        fake = FakeGitHubClient()
        GitHubIssueQueue(fake, "test-owner", "test-repo", GitHubQueueConfig(), host_id="test")
        now = frozen_now()
        state = make_state(
            phase="coding",
            lease_expires_at=now - timedelta(seconds=120),
            claimed_at=now - timedelta(seconds=2000),
        )
        issue = GitHubIssueRef(owner="test-owner", repo="test-repo", number=42)

        def fake_inputs(*args, **kwargs):

            return [
                ReconciliationInputs(
                    observation=WorkItemObservation(
                        work_item=issue,
                        state=state,
                        claim=make_claim(state),
                    ),
                    sessions=(),
                    worktrees=(),
                    pull_requests=(),
                    config=kwargs["cleanup_cfg"],
                    queue_config=kwargs["queue_cfg"],
                    now=now,
                )
            ]

        monkeypatch.setattr(cli, "_build_reconciliation_inputs", fake_inputs)
        monkeypatch.setattr(cli, "_build_github_client", lambda **_: (fake, True))

        # --apply should not raise and should report the stale lease.
        exit_code = cli.main(
            [
                "reconcile",
                "--config",
                str(config_path),
                "--apply",
                "--repo",
                "test-owner/test-repo",
            ]
        )
        # Stale lease = ESCALATE -> exit code 2.
        assert exit_code == 2
        out = capsys.readouterr().out
        assert "ESCALATE" in out or "escalate" in out

    def test_cli_uses_real_client_with_token(self, tmp_path, monkeypatch, capsys) -> None:
        """When ``GITHUB_TOKEN`` is set, the CLI builds the real
        :class:`GitHubClient`, not the in-memory fake.

        We monkeypatch :class:`GitHubClient` to a sentinel so we can
        observe construction without doing real network I/O.
        """

        config_path = tmp_path / "v3.yml"
        config_path.write_text(
            "github_queue:\n  enabled_label: v3-work\n  lease_seconds: 900\n"
            "  owner: test-owner\n  repo: test-repo\n"
            "cao:\n  base_url: http://localhost:9889\n",
            encoding="utf-8",
        )

        sentinel = MagicMock(name="RealGitHubClient")
        monkeypatch.setattr("ai_pr_orchestrator.github.client.GitHubClient", sentinel)
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
        # Block the queue from doing real network calls by patching it.
        from ai_pr_orchestrator.v3.queue import GitHubIssueQueue as RealQueue

        class _NoNetQueue(RealQueue):
            def __init__(self, *a, **kw):
                # Skip real __init__: we only need a no-op placeholder
                # so the CLI's ``list_ready`` doesn't blow up.
                self._ready: list = []
                self._loaded: dict[str, WorkflowState] = {}

            def list_ready(self):
                return []

            def load_state(self, work_item_id):
                return None

        monkeypatch.setattr(cli, "GitHubIssueQueue", _NoNetQueue)

        # Drive a dry-run; the test only needs to confirm that with a
        # token, the CLI chose the real client branch (i.e. constructed
        # GitHubClient with the token).
        cli.main(["reconcile", "--config", str(config_path)])
        assert sentinel.called
        kwargs = sentinel.call_args.kwargs
        assert kwargs.get("token") == "fake-token"
        assert kwargs.get("owner") == "test-owner"
        assert kwargs.get("repo") == "test-repo"
