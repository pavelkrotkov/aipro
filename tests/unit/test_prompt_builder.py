"""Tests for coder prompt construction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_pr_orchestrator.agents.prompt_builder import (
    PromptBuilder,
    PromptContext,
    estimate_prompt_tokens,
)
from ai_pr_orchestrator.models import Finding, FixTask

NOW = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _finding(id_: str, body: str) -> Finding:
    return Finding(
        id=id_,
        source="gemini_github",
        body=body,
        created_at=NOW,
        thread_id=f"thread-{id_}",
        path=f"src/{id_}.py",
        line=12,
    )


def _task(**overrides: Any) -> FixTask:
    defaults: dict[str, Any] = {
        "pr_number": 42,
        "head_sha": "abc123",
        "base_branch": "main",
        "findings": [
            _finding("f1", "Handle nullable values before calling strip."),
            _finding("f2", "Add coverage for the rejected branch."),
        ],
        "changed_files": ["src/app.py", "tests/test_app.py"],
        "diff_text": "diff --git a/src/app.py b/src/app.py\n+new code\n",
        "output_file": ".ai-orchestrator-result.json",
        "repo_instructions": "Follow local style and keep tests focused.",
    }
    defaults.update(overrides)
    return FixTask(**defaults)


def test_prompt_includes_all_finding_bodies_with_ids() -> None:
    prompt = PromptBuilder().build_prompt(_task())

    assert "f1" in prompt
    assert "Handle nullable values before calling strip." in prompt
    assert "f2" in prompt
    assert "Add coverage for the rejected branch." in prompt


def test_prompt_includes_diff_text() -> None:
    prompt = PromptBuilder().build_prompt(_task())

    assert "diff --git a/src/app.py b/src/app.py" in prompt
    assert "+new code" in prompt


def test_prompt_includes_repo_instructions_when_present() -> None:
    prompt = PromptBuilder().build_prompt(_task())

    assert "Follow local style and keep tests focused." in prompt


def test_prompt_omits_repo_instructions_when_none() -> None:
    prompt = PromptBuilder().build_prompt(_task(repo_instructions=None))

    assert "Repository Instructions" not in prompt


def test_prompt_includes_output_file_path() -> None:
    prompt = PromptBuilder().build_prompt(_task())

    assert ".ai-orchestrator-result.json" in prompt


def test_prompt_includes_changed_file_paths() -> None:
    prompt = PromptBuilder().build_prompt(_task())

    assert "src/app.py" in prompt
    assert "tests/test_app.py" in prompt


def test_prompt_stays_under_max_prompt_tokens() -> None:
    task = _task(diff_text="diff --git a/large.py b/large.py\n" + ("+x = 1\n" * 2_000))

    prompt = PromptBuilder(max_prompt_tokens=120).build_prompt(task)

    assert estimate_prompt_tokens(prompt) <= 120
    assert "[truncated" in prompt
    assert ".ai-orchestrator-result.json" in prompt


def test_prompt_truncates_diff_before_critical_sections() -> None:
    task = _task(
        diff_text="diff --git a/large.py b/large.py\n" + ("+x = 1\n" * 2_000),
        repo_instructions="Preserve this repository instruction.",
    )

    prompt = PromptBuilder(max_prompt_tokens=400).build_prompt(task)

    assert estimate_prompt_tokens(prompt) <= 400
    assert "[truncated to fit max_prompt_tokens]" in prompt
    assert "[diff omitted beyond prompt budget]" in prompt
    assert "Preserve this repository instruction." in prompt
    assert "Handle nullable values before calling strip." in prompt
    assert "Add coverage for the rejected branch." in prompt
    assert "Write a JSON object matching this schema." in prompt
    assert '"decisions"' in prompt
    assert '"token_usage"' in prompt


def test_adapter_can_override_prompt_format() -> None:
    class CustomFormatter:
        def format_prompt(self, context: PromptContext) -> str:
            finding_ids = ",".join(finding.id for finding in context.task.findings)
            return f"custom:{context.task.output_file}:{finding_ids}:{context.output_schema['changed']}"

    prompt = PromptBuilder(formatter=CustomFormatter()).build_prompt(_task())

    assert prompt == "custom:.ai-orchestrator-result.json:f1,f2:bool"
