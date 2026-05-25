import pytest

from ai_pr_orchestrator import runner


def test_run_stub_reports_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.run(pr_number=123, dry_run=False) == 1
    assert runner.NOT_IMPLEMENTED_MESSAGE in capsys.readouterr().err


def test_inspect_stub_reports_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.inspect(pr_number=123) == 1
    assert runner.NOT_IMPLEMENTED_MESSAGE in capsys.readouterr().err
