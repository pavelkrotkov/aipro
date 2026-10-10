"""Opt-in whole-process boundary checks against the deployed Hermes image."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.test_changed_file_authority import git
from tests.integration.test_changed_file_authority import repo as repo

IMAGE = os.environ.get("AIPRO_ISOLATION_IMAGE")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/aipro-container"
pytestmark = pytest.mark.skipif(
    not IMAGE or sys.platform != "linux" or not shutil.which("docker"),
    reason="requires Linux Docker and AIPRO_ISOLATION_IMAGE pinned by digest",
)


@pytest.mark.parametrize("linked", [True, False])
def test_container_cannot_use_coordinator_authority(repo, tmp_path, linked):
    checkout = repo
    if linked:
        checkout = tmp_path / "worker"
        git(repo, "worktree", "add", "-b", "isolated-lane", str(checkout))
    home = tmp_path / "lane-home"
    home.mkdir()
    authority = tmp_path / "authority-token"
    authority.write_text("invented authority canary")
    env = {
        **os.environ,
        "AIPRO_HERMES_IMAGE": IMAGE or "",
        "HERMES_HOME": str(home),
        "GITHUB_TOKEN": "invented environment canary",
        "SSH_AUTH_SOCK": str(tmp_path / "authority-agent.sock"),
    }
    probe = f"""
import os, pathlib, subprocess
assert not pathlib.Path({str(authority)!r}).exists()
assert not os.environ.get("GITHUB_TOKEN")
assert not os.environ.get("SSH_AUTH_SOCK")
assert not pathlib.Path("/var/run/docker.sock").exists()
assert os.geteuid() != 0
assert subprocess.run(["git", "config", "--local", "core.hooksPath", "/tmp/hooks"]).returncode
assert subprocess.run(["git", "add", "."]).returncode
subprocess.run(["hermes", "--version"], capture_output=True, check=True)
pathlib.Path("tracked").write_text("worker edit")
print("OK")
"""
    result = subprocess.run(
        [str(SCRIPT), "python3", "-c", probe],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert result.stdout.strip() == "OK"
    assert (checkout / "tracked").read_text() == "worker edit"
    assert authority.read_text() == "invented authority canary"
    assert "core.hookspath=" not in git(repo, "config", "--local", "--list")
