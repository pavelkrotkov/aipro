"""Validation tests for the target-repo workflow template and sample config.

These cover the example artifacts that ship for installation in a target
repository: ``examples/target-repo-workflow.yml`` and
``examples/sample-config.yml`` (issue V1-15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_pr_orchestrator.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "examples" / "target-repo-workflow.yml"
SAMPLE_CONFIG_PATH = REPO_ROOT / "examples" / "sample-config.yml"


def _load_workflow() -> dict[Any, Any]:
    data = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(data, dict)
    return data


def _on_section(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 (PyYAML's default) parses the bare key ``on`` as the boolean
    # ``True``. GitHub Actions still reads it as the trigger block, so accept
    # either spelling here.
    section = workflow.get("on", workflow.get(True))
    assert isinstance(section, dict)
    return section


def test_workflow_is_valid_yaml() -> None:
    workflow = _load_workflow()
    assert workflow["name"] == "AI PR Review Loop"


def test_workflow_has_all_required_event_triggers() -> None:
    on = _on_section(_load_workflow())
    required = {
        "pull_request",
        "issue_comment",
        "pull_request_review",
        "pull_request_review_comment",
        "check_run",
        "check_suite",
        "workflow_dispatch",
    }
    assert required <= set(on)


def test_workflow_dispatch_accepts_pr_input() -> None:
    on = _on_section(_load_workflow())
    pr_input = on["workflow_dispatch"]["inputs"]["pr"]
    assert pr_input["required"] is True
    assert pr_input["type"] == "string"


def test_concurrency_serializes_per_pr_without_cancel() -> None:
    workflow = _load_workflow()
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    # Group keys off the PR/issue/check/dispatch number so runs serialize per PR.
    assert "github.event.pull_request.number" in concurrency["group"]


def test_permissions_are_minimal() -> None:
    workflow = _load_workflow()
    assert workflow["permissions"] == {
        "contents": "write",
        "pull-requests": "write",
        "checks": "read",
    }


def test_orchestrator_token_not_passed_to_coder_environment() -> None:
    # Token scope separation is enforced in config: the coder only receives the
    # env var names listed in main_coder.env, which must exclude the GitHub
    # (orchestrator) token.
    config = load_config(SAMPLE_CONFIG_PATH)
    coder_env = {name.upper() for name in config.main_coder.env}
    assert "GITHUB_TOKEN" not in coder_env
    assert "GH_TOKEN" not in coder_env


def test_sample_config_parses_with_config_loader() -> None:
    config = load_config(SAMPLE_CONFIG_PATH)
    assert config.main_coder.provider == "codex_cli"
    assert config.enabled_label == "ai-loop"
