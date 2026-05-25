from pathlib import Path

import pytest

from ai_pr_orchestrator.config import ConfigError, load_config


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / ".github" / "ai-review-loop.yml"
    config_path.parent.mkdir()
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_parses_valid_config_with_all_fields(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
enabled_label: "ai-loop"
done_label: "ai-loop-done"
error_label: "ai-loop-error"
main_coder:
  provider: codex_cli
  command: "codex"
  args:
    - "exec"
    - "{prompt}"
  timeout_seconds: 1800
  output_file: ".ai-orchestrator-result.json"
  env:
    - CODEX_API_KEY
reviewers:
  gemini_github:
    enabled: true
    bot_logins:
      - "gemini-code-assist[bot]"
    trigger_comment: "/gemini review"
review_phase:
  max_rounds: 1
  poll_interval_seconds: 30
  reviewer_timeout_seconds: 600
  phase_timeout_seconds: 900
thread_policy:
  auto_resolve_bot_threads: true
  never_resolve_human_threads: true
  resolve_rejected_bot_threads: true
  require_reply_before_resolve: true
git:
  base_branch: "main"
  commit_author_name: "AI PR Orchestrator"
  commit_author_email: "ai-pr-orchestrator@example.com"
  commit_message_prefix: "fix: address AI review feedback"
ci:
  require_green_before_done: true
  required_checks:
    - "test"
  ignored_checks:
    - "AI PR Review Loop"
  relevant_failed_log_lines: 300
safety:
  only_run_on_labeled_prs: true
  disallow_forks: true
  disallow_workflow_file_changes: true
  max_total_iterations: 5
  max_commits_per_run: 1
  max_coder_invocations_per_run: 1
  max_reviewer_triggers_per_run: 3
  max_prompt_tokens: 100000
  allowed_pr_author_associations:
    - "OWNER"
    - "MEMBER"
notifications:
  mention_on_needs_human:
    - "@pavelkrotkov"
  mention_on_error:
    - "@pavelkrotkov"
""",
    )

    config = load_config(config_path)

    assert config.enabled_label == "ai-loop"
    assert config.main_coder.provider == "codex_cli"
    assert config.main_coder.args == ["exec", "{prompt}"]
    assert config.reviewers["gemini_github"].bot_logins == ["gemini-code-assist[bot]"]
    assert config.ci.required_checks == ["test"]
    assert config.safety.max_total_iterations == 5
    assert config.notifications.mention_on_error == ["@pavelkrotkov"]


def test_applies_defaults_for_optional_fields(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
main_coder:
  provider: codex_cli
""",
    )

    config = load_config(config_path)

    assert config.enabled_label == "ai-loop"
    assert config.main_coder.command == "codex"
    assert config.main_coder.timeout_seconds == 1800
    assert config.review_phase.reviewer_timeout_seconds == 600
    assert config.ci.required_checks == []
    assert config.ci.ignored_checks == ["AI PR Review Loop"]
    assert config.safety.max_total_iterations == 3


def test_rejects_config_missing_required_main_coder_provider(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, "main_coder: {}\n")

    with pytest.raises(ConfigError, match=r"main_coder\.provider"):
        load_config(config_path)


def test_rejects_config_missing_main_coder_section(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, "enabled_label: ai-loop\n")

    with pytest.raises(ConfigError, match=r"main_coder\.provider"):
        load_config(config_path)


def test_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
main_coder:
  provider: unknown
  timeout_seconds: -1
""",
    )

    with pytest.raises(ConfigError, match=r"main_coder\.provider"):
        load_config(config_path)


def test_handles_empty_config_file_gracefully(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, "")

    with pytest.raises(ConfigError, match=r"main_coder\.provider"):
        load_config(config_path)


def test_wraps_config_file_read_errors(tmp_path: Path) -> None:
    config_path = tmp_path / ".github" / "ai-review-loop.yml"

    with pytest.raises(ConfigError, match="Failed to read configuration file"):
        load_config(config_path)


def test_wraps_invalid_yaml_errors(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, "main_coder: [")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_path)


def test_rejects_unknown_nested_config_keys(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
main_coder:
  provider: codex_cli
git:
  commit_author_namee: typo
""",
    )

    with pytest.raises(ConfigError, match=r"Unknown configuration key: git\.commit_author_namee"):
        load_config(config_path)


def test_parses_safety_block(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
main_coder:
  provider: codex_cli
safety:
  max_total_iterations: 2
  allowed_pr_author_associations:
    - OWNER
    - COLLABORATOR
""",
    )

    config = load_config(config_path)

    assert config.safety.max_total_iterations == 2
    assert config.safety.allowed_pr_author_associations == ["OWNER", "COLLABORATOR"]


def test_parses_ci_block(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
main_coder:
  provider: codex_cli
ci:
  required_checks:
    - unit
  ignored_checks:
    - flaky
""",
    )

    config = load_config(config_path)

    assert config.ci.required_checks == ["unit"]
    assert config.ci.ignored_checks == ["flaky"]


def test_parses_notifications_block(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
main_coder:
  provider: codex_cli
notifications:
  mention_on_needs_human:
    - "@owner"
  mention_on_error:
    - "@maintainer"
""",
    )

    config = load_config(config_path)

    assert config.notifications.mention_on_needs_human == ["@owner"]
    assert config.notifications.mention_on_error == ["@maintainer"]
