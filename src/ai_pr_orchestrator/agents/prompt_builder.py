"""Build coder prompts from structured fix tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from ai_pr_orchestrator.models import FixTask

CHARS_PER_TOKEN = 4
TRUNCATION_MARKER = "\n[truncated to fit max_prompt_tokens]\n"


class PromptFormatter(Protocol):
    """Adapter hook for coder-specific prompt formatting."""

    def format_prompt(self, context: PromptContext) -> str: ...


@dataclass(frozen=True)
class PromptContext:
    """Structured prompt data shared with coder adapters."""

    task: FixTask
    output_schema: dict[str, object]
    max_prompt_tokens: int


@dataclass
class DefaultPromptFormatter:
    """Default prompt format for text-oriented coding agents."""

    def format_prompt(self, context: PromptContext) -> str:
        task = context.task
        sections = [
            "# AI PR Orchestrator Fix Task",
            "",
            f"PR: #{task.pr_number}",
            f"Base branch: {task.base_branch}",
            f"Head SHA: {task.head_sha}",
            f"Write the result JSON to: {task.output_file}",
            "",
            "## Changed Files",
            _bullet_list(task.changed_files),
            "",
        ]

        if task.repo_instructions:
            sections.extend(
                [
                    "## Repository Instructions",
                    task.repo_instructions,
                    "",
                ]
            )

        sections.extend(
            [
                "## Findings",
                _format_findings(task),
                "",
                "## Pull Request Diff",
                task.diff_text,
                "",
                "## Required Output",
                "Write a JSON object matching this schema. Include one decision for every finding.",
                _format_schema(context.output_schema),
            ]
        )
        return "\n".join(sections)


@dataclass
class PromptBuilder:
    """Build prompts and enforce the configured prompt budget."""

    max_prompt_tokens: int = 100000
    formatter: PromptFormatter = field(default_factory=DefaultPromptFormatter)

    def build_prompt(self, task: FixTask) -> str:
        context = PromptContext(
            task=task,
            output_schema=output_schema(),
            max_prompt_tokens=self.max_prompt_tokens,
        )
        prompt = self.formatter.format_prompt(context)
        return truncate_prompt(prompt, self.max_prompt_tokens)


def output_schema() -> dict[str, object]:
    return {
        "changed": "bool",
        "commit_message": "string|null",
        "summary": "string",
        "needs_human": "bool",
        "decisions": [
            {
                "finding_id": "string",
                "thread_id": "string|null",
                "verdict": "accepted|rejected|needs_human",
                "confidence": "low|medium|high",
                "reason": "string",
                "reply": "string",
                "should_resolve": "bool",
                "changed_files": ["string"],
            }
        ],
        "tests": [{"command": "string", "result": "passed|failed|not_run", "notes": "string"}],
        "token_usage": {"input_tokens": "int", "output_tokens": "int"},
    }


def estimate_prompt_tokens(prompt: str) -> int:
    """Return a conservative token estimate used for prompt budget checks."""

    if not prompt:
        return 0
    return (len(prompt) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def truncate_prompt(prompt: str, max_prompt_tokens: int) -> str:
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be greater than zero")

    max_chars = max_prompt_tokens * CHARS_PER_TOKEN
    if len(prompt) <= max_chars:
        return prompt

    if max_chars <= len(TRUNCATION_MARKER):
        return prompt[:max_chars]

    keep = max_chars - len(TRUNCATION_MARKER)
    head_chars = max(0, keep // 2)
    tail_chars = max(0, keep - head_chars)
    return prompt[:head_chars].rstrip() + TRUNCATION_MARKER + prompt[-tail_chars:].lstrip()


def _format_findings(task: FixTask) -> str:
    if not task.findings:
        return "- None"

    chunks: list[str] = []
    for finding in task.findings:
        metadata = [
            f"id={finding.id}",
            f"source={finding.source}",
        ]
        if finding.path:
            metadata.append(f"path={finding.path}")
        if finding.line is not None:
            metadata.append(f"line={finding.line}")
        if finding.thread_id:
            metadata.append(f"thread_id={finding.thread_id}")
        if finding.comment_id:
            metadata.append(f"comment_id={finding.comment_id}")
        if finding.severity:
            metadata.append(f"severity={finding.severity}")
        chunks.append(f"### Finding {finding.id}\n" + "\n".join(metadata) + f"\n\n{finding.body}")
    return "\n\n".join(chunks)


def _bullet_list(values: list[str]) -> str:
    if not values:
        return "- None"
    return "\n".join(f"- {value}" for value in values)


def _format_schema(schema: dict[str, object]) -> str:
    return json.dumps(schema, indent=2)
