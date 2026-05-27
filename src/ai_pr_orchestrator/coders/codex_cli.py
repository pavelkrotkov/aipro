"""Codex CLI coder adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ai_pr_orchestrator.config import MainCoderConfig
from ai_pr_orchestrator.models import AgentRunResult, FixTask, TestResult

BLOCKED_ENV_VARS = frozenset({"GH_TOKEN", "GITHUB_TOKEN"})


class CodexCliCoderAdapter:
    """Invoke Codex CLI as the primary coding agent."""

    name = "codex_cli"

    def __init__(self, config: MainCoderConfig, *, cwd: str | Path | None = None) -> None:
        self.config = config
        self.cwd = Path.cwd() if cwd is None else Path(cwd)

    def run_fix_task(self, task: FixTask) -> AgentRunResult:
        output_path = self._output_path(task)
        try:
            self._delete_output_file(output_path)
        except OSError as exc:
            return AgentRunResult(
                changed=False,
                summary=f"Failed to delete stale coder output file: {exc}",
                needs_human=True,
            )

        prompt = self._build_prompt(task)
        command = self._build_command(prompt)
        env = self._subprocess_env()

        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return AgentRunResult(
                changed=False,
                summary=(
                    f"Coder subprocess timed out after {self.config.timeout_seconds} seconds."
                ),
                needs_human=True,
                tests=[
                    TestResult(
                        command=" ".join(command),
                        result="not_run",
                        notes=str(exc),
                    )
                ],
            )
        except OSError as exc:
            return AgentRunResult(
                changed=False,
                summary=f"Failed to start coder subprocess: {exc}",
                needs_human=True,
            )

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            summary = f"Coder subprocess failed with exit code {completed.returncode}."
            if detail:
                summary = f"{summary} {detail}"
            return AgentRunResult(changed=False, summary=summary, needs_human=True)

        try:
            raw_result = self._read_raw_result(output_path, completed.stdout)
        except OSError as exc:
            return AgentRunResult(
                changed=False,
                summary=f"Failed to read or clean up coder result: {exc}",
                needs_human=True,
            )
        return self._parse_result(raw_result)

    def _output_path(self, task: FixTask) -> Path:
        output_file = Path(task.output_file or self.config.output_file)
        if output_file.is_absolute():
            return output_file
        return self.cwd / output_file

    def _delete_output_file(self, output_path: Path) -> None:
        output_path.unlink(missing_ok=True)

    def _build_command(self, prompt: str) -> list[str]:
        return [
            self.config.command,
            *[arg.replace("{prompt}", prompt) for arg in self.config.args],
        ]

    def _subprocess_env(self) -> dict[str, str]:
        return {
            key: os.environ[key]
            for key in self.config.env
            if key in os.environ and key not in BLOCKED_ENV_VARS
        }

    def _read_raw_result(self, output_path: Path, stdout: str) -> str:
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8")
            self._delete_output_file(output_path)
            return content
        return stdout

    def _parse_result(self, raw_result: str) -> AgentRunResult:
        try:
            data = json.loads(raw_result)
            if not isinstance(data, dict):
                raise TypeError("coder result must be a JSON object")
            return AgentRunResult.from_dict(data)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError) as exc:
            return AgentRunResult(
                changed=False,
                summary=f"Failed to parse coder output as JSON: {exc}",
                needs_human=True,
            )

    def _build_prompt(self, task: FixTask) -> str:
        output_file = task.output_file or self.config.output_file
        payload = json.dumps(task.to_dict(), indent=2, sort_keys=True)
        schema = json.dumps(_result_schema(), indent=2, sort_keys=True)
        return (
            "You are the primary coding agent for AI PR Orchestrator.\n"
            "Apply the requested fixes in the current worktree only.\n"
            f"Write the final JSON result to {output_file} using this schema:\n"
            f"{schema}\n\n"
            "Fix task:\n"
            f"{payload}"
        )


def _result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["changed", "summary"],
        "properties": {
            "changed": {"type": "boolean"},
            "summary": {"type": "string"},
            "needs_human": {"type": "boolean"},
            "commit_message": {"type": ["string", "null"]},
            "decisions": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "finding_id",
                        "verdict",
                        "confidence",
                        "reason",
                        "reply",
                        "should_resolve",
                    ],
                    "properties": {
                        "finding_id": {"type": "string"},
                        "thread_id": {"type": ["string", "null"]},
                        "verdict": {
                            "type": "string",
                            "enum": ["accepted", "rejected", "needs_human"],
                        },
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "reason": {"type": "string"},
                        "reply": {"type": "string"},
                        "should_resolve": {"type": "boolean"},
                        "changed_files": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "tests": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["command", "result"],
                    "properties": {
                        "command": {"type": "string"},
                        "result": {"type": "string", "enum": ["passed", "failed", "not_run"]},
                        "notes": {"type": "string"},
                    },
                },
            },
            "token_usage": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "input_tokens": {"type": "integer", "minimum": 0},
                    "output_tokens": {"type": "integer", "minimum": 0},
                },
            },
        },
    }
