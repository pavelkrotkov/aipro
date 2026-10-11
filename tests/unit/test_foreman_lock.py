from pathlib import Path

import pytest

from ai_pr_orchestrator.cli import _coordinator_lock


def test_coordinator_lock_excludes_second_process_and_releases(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    with (
        _coordinator_lock(tmp_path),
        pytest.raises(BlockingIOError),
        _coordinator_lock(tmp_path),
    ):
        pytest.fail("second coordinator entered")
    with _coordinator_lock(tmp_path):
        pass
    (tmp_path / ".git").rename(tmp_path / "metadata")
    (tmp_path / ".git").write_text("gitdir: metadata\n")
    with pytest.raises(ValueError, match="primary repository"), _coordinator_lock(tmp_path):
        pytest.fail("linked checkout accepted")
