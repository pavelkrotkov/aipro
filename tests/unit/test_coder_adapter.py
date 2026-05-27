from ai_pr_orchestrator.coders.base import CoderAdapter
from ai_pr_orchestrator.coders.codex_cli import CodexCliCoderAdapter
from ai_pr_orchestrator.config import MainCoderConfig


def test_codex_cli_satisfies_coder_adapter_protocol(tmp_path) -> None:
    adapter = CodexCliCoderAdapter(MainCoderConfig(provider="codex_cli"), cwd=tmp_path)

    assert isinstance(adapter, CoderAdapter)
