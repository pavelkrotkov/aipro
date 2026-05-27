from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ai_pr_orchestrator.coders.codex_cli import CodexCliCoderAdapter
from ai_pr_orchestrator.config import MainCoderConfig
from ai_pr_orchestrator.models import AgentRunResult, Finding, FixTask, TokenUsage


def make_task(output_file: str = ".ai-orchestrator-result.json") -> FixTask:
    return FixTask(
        pr_number=7,
        head_sha="abc123",
        base_branch="main",
        findings=[
            Finding(
                id="finding-1",
                source="gemini_github",
                body="Fix this bug",
                created_at=datetime(2026, 5, 26, 12, 0, tzinfo=UTC),
                thread_id="thread-1",
            )
        ],
        changed_files=["src/example.py"],
        diff_text="diff --git a/src/example.py b/src/example.py",
        output_file=output_file,
        repo_instructions="Run tests before reporting success.",
    )


def result_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "changed": True,
        "summary": "Fixed the bug.",
        "needs_human": False,
        "commit_message": "fix: handle review feedback",
        "decisions": [
            {
                "finding_id": "finding-1",
                "thread_id": "thread-1",
                "verdict": "accepted",
                "confidence": "high",
                "reason": "The finding identified a real defect.",
                "reply": "Fixed by updating the implementation.",
                "should_resolve": True,
                "changed_files": ["src/example.py"],
            }
        ],
        "tests": [
            {
                "command": "uv run pytest",
                "result": "passed",
                "notes": "",
            }
        ],
        "token_usage": {
            "input_tokens": 123,
            "output_tokens": 45,
        },
    }
    payload.update(overrides)
    return payload


def test_constructs_subprocess_command_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["command"] = command
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(result_payload()), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = MainCoderConfig(provider="codex_cli", command="codex", args=["exec", "{prompt}"])

    CodexCliCoderAdapter(config, cwd=tmp_path).run_fix_task(make_task())

    assert calls["command"][0:2] == ["codex", "exec"]
    assert len(calls["command"]) == 3
    assert '"pr_number": 7' in calls["command"][2]
    assert '"finding-1"' in calls["command"][2]
    assert '"type": "object"' in calls["command"][2]
    assert '"enum": [' in calls["command"][2]
    assert calls["kwargs"]["cwd"] == tmp_path
    assert "shell" not in calls["kwargs"]


def test_passes_only_declared_env_vars_to_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(result_payload()), stderr=""
        )

    monkeypatch.setenv("CODEX_API_KEY", "codex-secret")
    monkeypatch.setenv("UNDECLARED_SECRET", "do-not-pass")
    monkeypatch.setattr(subprocess, "run", fake_run)
    config = MainCoderConfig(provider="codex_cli", env=["CODEX_API_KEY"])

    CodexCliCoderAdapter(config, cwd=tmp_path).run_fix_task(make_task())

    assert calls["env"] == {"CODEX_API_KEY": "codex-secret"}


def test_excludes_github_tokens_even_when_declared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(result_payload()), stderr=""
        )

    monkeypatch.setenv("CODEX_API_KEY", "codex-secret")
    monkeypatch.setenv("GH_TOKEN", "orchestrator-token")
    monkeypatch.setenv("GITHUB_TOKEN", "orchestrator-token")
    monkeypatch.setattr(subprocess, "run", fake_run)
    config = MainCoderConfig(
        provider="codex_cli",
        env=["CODEX_API_KEY", "GH_TOKEN", "GITHUB_TOKEN"],
    )

    CodexCliCoderAdapter(config, cwd=tmp_path).run_fix_task(make_task())

    assert calls["env"] == {"CODEX_API_KEY": "codex-secret"}


def test_reads_result_from_output_file_when_it_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_file = tmp_path / ".ai-orchestrator-result.json"

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_file.write_text(
            json.dumps(result_payload(summary="Read from file.")), encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(result_payload(summary="Read from stdout.")),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result.summary == "Read from file."
    assert not output_file.exists()


def test_falls_back_to_parsing_stdout_when_output_file_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(result_payload(summary="Read from stdout.")),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result.summary == "Read from stdout."


def test_invalid_json_output_produces_error_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="not json", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result.changed is False
    assert result.needs_human is True
    assert "Failed to parse coder output as JSON" in result.summary


def test_invalid_nested_json_output_produces_error_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(result_payload(decisions=["not a decision"])),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result.changed is False
    assert result.needs_human is True
    assert "Failed to parse coder output as JSON" in result.summary


def test_deletes_output_file_before_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_file = tmp_path / ".ai-orchestrator-result.json"
    output_file.write_text("stale", encoding="utf-8")
    observed: dict[str, bool] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["exists_during_run"] = output_file.exists()
        output_file.write_text(json.dumps(result_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    CodexCliCoderAdapter(MainCoderConfig(provider="codex_cli"), cwd=tmp_path).run_fix_task(
        make_task()
    )

    assert observed["exists_during_run"] is False


def test_output_file_cleanup_failure_before_invocation_produces_error_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_file = tmp_path / ".ai-orchestrator-result.json"
    output_file.write_text("stale", encoding="utf-8")
    observed = {"called": False}
    original_unlink = Path.unlink

    def fake_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == output_file and path.exists():
            raise PermissionError("denied")
        original_unlink(path, missing_ok=missing_ok)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["called"] = True
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(result_payload()), stderr=""
        )

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(MainCoderConfig(provider="codex_cli"), cwd=tmp_path).run_fix_task(
        make_task()
    )

    assert observed["called"] is False
    assert result.changed is False
    assert result.needs_human is True
    assert "Failed to delete stale coder output file" in result.summary


def test_output_file_cleanup_failure_after_read_produces_error_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_file = tmp_path / ".ai-orchestrator-result.json"
    original_unlink = Path.unlink

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_file.write_text(json.dumps(result_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == output_file and path.exists():
            raise PermissionError("denied")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "unlink", fake_unlink)

    result = CodexCliCoderAdapter(MainCoderConfig(provider="codex_cli"), cwd=tmp_path).run_fix_task(
        make_task()
    )

    assert result.changed is False
    assert result.needs_human is True
    assert "Failed to read or clean up coder result" in result.summary


def test_subprocess_timeout_is_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli", timeout_seconds=5),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert calls["timeout"] == 5
    assert result.changed is False
    assert result.needs_human is True
    assert "timed out" in result.summary


def test_subprocess_os_error_produces_error_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("argument list too long")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result.changed is False
    assert result.needs_human is True
    assert "Failed to start coder subprocess" in result.summary


def test_nonzero_exit_code_produces_error_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result == AgentRunResult(
        changed=False,
        summary="Coder subprocess failed with exit code 2. boom",
        needs_human=True,
    )


def test_returns_agent_run_result_with_token_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(result_payload(token_usage={"input_tokens": 9, "output_tokens": 4})),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliCoderAdapter(
        MainCoderConfig(provider="codex_cli"),
        cwd=tmp_path,
    ).run_fix_task(make_task())

    assert result.token_usage == TokenUsage(input_tokens=9, output_tokens=4)


def test_subprocess_command_is_a_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(result_payload()), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    CodexCliCoderAdapter(MainCoderConfig(provider="codex_cli"), cwd=tmp_path).run_fix_task(
        make_task()
    )

    assert isinstance(observed["command"], list)
