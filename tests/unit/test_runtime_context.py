from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from ai_pr_orchestrator import cli
from ai_pr_orchestrator import runner as runner_mod
from ai_pr_orchestrator.coders.codex_cli import CodexCliCoderAdapter
from ai_pr_orchestrator.config import Config, MainCoderConfig, ReviewerConfig
from ai_pr_orchestrator.git.repo import GitRepo
from ai_pr_orchestrator.github import models as gh_models
from ai_pr_orchestrator.github.client import GitHubClient as HttpGitHubClient
from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.models import AgentRunResult, FixTask
from ai_pr_orchestrator.reviewers.gemini_github import GeminiGitHubReviewerAdapter


def make_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "main_coder": MainCoderConfig(provider="codex_cli", env=["CODEX_API_KEY"]),
        "reviewers": {
            "gemini_github": ReviewerConfig(
                enabled=True,
                bot_logins=["gemini-code-assist[bot]"],
                trigger_comment="/gemini review",
            )
        },
    }
    defaults.update(overrides)
    return Config(**defaults)


def set_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "github-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octo-org/octo-repo")


def test_build_runtime_context_returns_live_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setenv("CODEX_API_KEY", "coder-token")
    set_runtime_env(monkeypatch)
    config = make_config(
        reviewers={
            "gemini_github": ReviewerConfig(
                enabled=True,
                bot_logins=["gemini-code-assist[bot]"],
                trigger_comment="/gemini review",
            ),
            "disabled_future_reviewer": ReviewerConfig(enabled=False),
        }
    )

    ctx = runner_mod._build_runtime_context(config)

    assert isinstance(ctx.github, HttpGitHubClient)
    assert ctx.github._owner == "octo-org"
    assert ctx.github._repo == "octo-repo"
    assert ctx.github._dry_run is False
    assert isinstance(ctx.git, GitRepo)
    assert ctx.git.path == tmp_path
    assert isinstance(ctx.coder, CodexCliCoderAdapter)
    assert ctx.coder.cwd == tmp_path
    assert ctx.coder._subprocess_env() == {"CODEX_API_KEY": "coder-token"}
    assert set(ctx.reviewers) == {"gemini_github"}
    assert isinstance(ctx.reviewers["gemini_github"], GeminiGitHubReviewerAdapter)
    assert ctx.config is config
    assert ctx.dry_run is False


def test_build_runtime_context_supports_github_token_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "fallback-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octo-org/octo-repo")

    ctx = runner_mod._build_runtime_context(make_config(), dry_run=True)

    assert isinstance(ctx.github, HttpGitHubClient)


def test_build_runtime_context_strips_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", " github-token\n")
    monkeypatch.setenv("GITHUB_REPOSITORY", " octo-org / octo-repo ")

    ctx = runner_mod._build_runtime_context(make_config(), dry_run=True)

    assert isinstance(ctx.github, HttpGitHubClient)
    assert ctx.github._owner == "octo-org"
    assert ctx.github._repo == "octo-repo"


def test_build_runtime_context_requires_github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "octo-org/octo-repo")

    with pytest.raises(ValueError, match="GH_TOKEN or GITHUB_TOKEN"):
        runner_mod._build_runtime_context(make_config())


@pytest.mark.parametrize(
    "value",
    ["", "octo-org", "octo-org/octo-repo/extra", "/octo-repo", "octo-org/"],
)
def test_build_runtime_context_requires_owner_repo(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GH_TOKEN", "github-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", value)

    with pytest.raises(ValueError, match=r"GITHUB_REPOSITORY.*owner/repo"):
        runner_mod._build_runtime_context(make_config())


def test_build_runtime_context_dry_run_uses_noop_github_and_no_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_runtime_env(monkeypatch)

    ctx = runner_mod._build_runtime_context(make_config(), dry_run=True)

    assert ctx.dry_run is True
    assert ctx.git is None
    assert isinstance(ctx.github, HttpGitHubClient)
    assert ctx.github._dry_run is True


def test_build_runtime_context_can_skip_git_for_read_only_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_runtime_env(monkeypatch)

    class ExplodingGitRepo:
        def __init__(self, path: Path | str) -> None:
            raise AssertionError("git should not be initialized")

    monkeypatch.setattr(runner_mod, "GitRepo", ExplodingGitRepo)

    ctx = runner_mod._build_runtime_context(make_config(), require_git=False)

    assert ctx.git is None
    assert ctx.dry_run is False


def test_build_runtime_context_reports_git_initialization_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    set_runtime_env(monkeypatch)

    with pytest.raises(ValueError, match="Failed to initialize git repository"):
        runner_mod._build_runtime_context(make_config())


def test_build_runtime_context_can_skip_agents_for_read_only_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_runtime_env(monkeypatch)
    config = make_config(
        main_coder=MainCoderConfig(provider="unsupported"),
        reviewers={"future_reviewer": ReviewerConfig(enabled=True)},
    )

    ctx = runner_mod._build_runtime_context(
        config,
        dry_run=True,
        require_git=False,
        require_agents=False,
    )

    assert isinstance(ctx.coder, runner_mod.NoopCoderAdapter)
    assert ctx.reviewers == {}


def test_build_runtime_context_rejects_unsupported_enabled_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_runtime_env(monkeypatch)
    config = make_config(reviewers={"future_reviewer": ReviewerConfig(enabled=True)})

    with pytest.raises(ValueError, match="Unsupported enabled reviewer 'future_reviewer'"):
        runner_mod._build_runtime_context(config)


def test_run_reports_runtime_context_configuration_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner_mod, "load_config", lambda: make_config())
    monkeypatch.setattr(runner_mod, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_mod,
        "_build_runtime_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad env")),
    )

    assert runner_mod.run(pr_number=35, dry_run=False) == 1
    assert "Configuration error: bad env" in capsys.readouterr().err


def test_inspect_reports_runtime_context_configuration_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner_mod, "load_config", lambda: make_config())
    monkeypatch.setattr(
        runner_mod,
        "_build_runtime_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad env")),
    )

    assert runner_mod.inspect(pr_number=35) == 1
    assert "Configuration error: bad env" in capsys.readouterr().err


def test_inspect_does_not_require_git(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    fake_gh = FakeGitHubClient()

    class StubCoder:
        name = "stub"

        def run_fix_task(self, task: FixTask) -> AgentRunResult:
            raise AssertionError("inspect should not invoke the coder")

    def fake_build_context(config: Config, **kwargs: Any) -> runner_mod.RunnerContext:
        calls.append(kwargs)
        return runner_mod.RunnerContext(
            github=fake_gh,
            coder=StubCoder(),
            reviewers={},
            config=config,
            git=None,
        )

    monkeypatch.setattr(runner_mod, "load_config", lambda: make_config())
    monkeypatch.setattr(runner_mod, "_build_runtime_context", fake_build_context)

    assert runner_mod.inspect(pr_number=35) == 0
    assert calls == [{"dry_run": True, "require_git": False, "require_agents": False}]


def test_inspect_reports_github_api_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingGitHub(FakeGitHubClient):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def get_pr_comments(self, issue_number: int) -> list[gh_models.Comment]:
            raise RuntimeError("rate limited")

        def close(self) -> None:
            self.closed = True

    failing_gh = FailingGitHub()

    def fake_build_context(config: Config, **kwargs: Any) -> runner_mod.RunnerContext:
        return runner_mod.RunnerContext(
            github=failing_gh,
            coder=runner_mod.NoopCoderAdapter(),
            reviewers={},
            config=config,
            git=None,
            dry_run=True,
        )

    monkeypatch.setattr(runner_mod, "load_config", lambda: make_config())
    monkeypatch.setattr(runner_mod, "_build_runtime_context", fake_build_context)

    assert runner_mod.inspect(pr_number=35) == 1
    assert "GitHub API error: rate limited" in capsys.readouterr().err
    assert failing_gh.closed is True


def test_cli_run_builds_context_and_runs_loop_with_mocked_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GH_TOKEN", "github-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octo-org/octo-repo")
    fake_gh = FakeGitHubClient()
    fake_gh.seed_pr(
        gh_models.PullRequest(
            number=35,
            title="Runtime wiring",
            body="",
            state="open",
            head_sha="abc123",
            head_ref="issue-35-runtime-context",
            base_ref="main",
            author="pavel",
            labels=[],
            author_association="OWNER",
        )
    )

    class FakeGitRepo:
        def __init__(self, path: Path | str) -> None:
            self.path = Path(path)

        def is_clean(self) -> bool:
            return True

    def fake_github_client(
        token: str, owner: str, repo: str, *, dry_run: bool = False
    ) -> FakeGitHubClient:
        assert token == "github-token"
        assert owner == "octo-org"
        assert repo == "octo-repo"
        assert dry_run is False
        return fake_gh

    monkeypatch.setattr(runner_mod, "HttpGitHubClient", fake_github_client)
    monkeypatch.setattr(runner_mod, "GitRepo", FakeGitRepo)
    monkeypatch.setattr(runner_mod, "load_config", lambda: make_config())
    monkeypatch.setattr(runner_mod, "setup_logging", lambda **kwargs: None)

    assert cli.main(["run", "--pr", "35"]) == 0
