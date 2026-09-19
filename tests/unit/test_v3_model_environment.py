"""The broker's catalog alias reaches Hermes as actual model/provider arguments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_pr_orchestrator.v3.cao import (
    CaoAdoptionMismatchError,
    CaoSessionController,
    session_name_for,
)
from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
from ai_pr_orchestrator.v3.catalog import ModelCatalog, ModelCatalogEntry
from ai_pr_orchestrator.v3.config import CAOControlPlaneConfig
from ai_pr_orchestrator.v3.domain import ModelAssignment
from ai_pr_orchestrator.v3.interfaces import LaneExecutionContext, ModelLease
from ai_pr_orchestrator.v3.lanes import LaneRegistry
from tests.integration._fake_cao_server import FakeCAOServer

WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "aipro-hermes"


def _launch_wrapper(tmp_path: Path, env: dict[str, str]) -> dict:
    # Execute the deployed shell boundary; the harmless process reports exactly
    # what Hermes would receive, without contacting any model provider.
    hermes = tmp_path / "hermes"
    hermes.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'print(json.dumps({"argv": sys.argv[1:], "home": os.environ.get("HERMES_HOME")}))\n'
    )
    hermes.chmod(0o755)
    result = subprocess.run(
        [str(WRAPPER), "chat", "--yolo", "--source", "cao"],
        env={"PATH": f"{tmp_path}{os.pathsep}{os.defpath}", **env},
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("lane_name", ["developer", "requirements-reviewer"])
def test_leased_model_reaches_hermes_without_losing_session_env(tmp_path, lane_name):
    lanes = LaneRegistry.default()
    lane = lanes.get(lane_name)
    entry = ModelCatalogEntry(
        ref="cheap-slot", descriptor="vendor/model $(touch NEVER)", provider="openrouter"
    )
    lease = ModelLease("lease", ModelAssignment(lane_name, entry.ref))
    defaults = {
        "HERMES_HOME": "/isolated/lane",
        "AIPRO_MODEL": "wrong",
        "AIPRO_PROVIDER": "wrong",
        "KEEP": "yes",
    }
    with (
        FakeCAOServer() as cao,
        CaoSessionController(CAOControlPlaneConfig(base_url=cao.url), lanes) as controller,
    ):
        executor = CaoLaneExecutor(
            controller,
            lanes,
            catalog=ModelCatalog((entry,)),
            env=defaults,
            poll_interval_seconds=0,
        )
        cao.set_output(session_name_for("run", lane_name), "[]")
        result = executor.execute(lane, "task", str(tmp_path), LaneExecutionContext("run"), lease)
        session = cao._sessions[result.session.session_id]
        assert session.env_vars == {
            **defaults,
            "AIPRO_MODEL": entry.descriptor,
            "AIPRO_PROVIDER": entry.provider,
        }
        assert session.metadata["model_assignment"] == lease.assignment.to_dict()
        actual = _launch_wrapper(tmp_path, session.env_vars)
    assert actual == {
        "argv": [
            "chat",
            "--yolo",
            "--source",
            "cao",
            "--model",
            entry.descriptor,
            "--provider",
            entry.provider,
        ],
        "home": defaults["HERMES_HOME"],
    }
    assert defaults["AIPRO_MODEL"] == "wrong"
    assert not (tmp_path / "NEVER").exists()


@pytest.mark.parametrize(
    "entries, reason",
    [
        ((), "absent from catalog"),
        ((ModelCatalogEntry("slot", "model"),), "explicit Hermes provider"),
        (
            (
                ModelCatalogEntry(
                    "slot", "model", provider="custom", endpoint="https://example.test"
                ),
            ),
            "custom endpoint",
        ),
    ],
)
def test_unresolvable_lease_fails_before_cao_launch(tmp_path, entries, reason):
    lanes = LaneRegistry.default()
    with (
        FakeCAOServer() as cao,
        CaoSessionController(CAOControlPlaneConfig(base_url=cao.url), lanes) as controller,
    ):
        executor = CaoLaneExecutor(controller, lanes, catalog=ModelCatalog(entries))
        with pytest.raises(ValueError, match=reason):
            executor.execute(
                lanes.get("developer"),
                "task",
                str(tmp_path),
                LaneExecutionContext("run"),
                ModelLease("lease", ModelAssignment("developer", "slot")),
            )
        assert not cao._sessions


def test_adoption_cannot_reassign_running_model(tmp_path):
    lanes = LaneRegistry.default()
    entries = tuple(ModelCatalogEntry(ref, ref, provider="test") for ref in ("first", "second"))
    with (
        FakeCAOServer() as cao,
        CaoSessionController(CAOControlPlaneConfig(base_url=cao.url), lanes) as controller,
    ):
        executor = CaoLaneExecutor(
            controller, lanes, catalog=ModelCatalog(entries), poll_interval_seconds=0
        )
        lane = lanes.get("developer")
        context = LaneExecutionContext("run")
        first = ModelLease("first", ModelAssignment(lane.lane, "first"))
        result = executor.execute(lane, "first task", str(tmp_path), context, first)
        with pytest.raises(CaoAdoptionMismatchError, match="model assignment"):
            executor.execute(
                lane,
                "second task",
                str(tmp_path),
                context,
                ModelLease("second", ModelAssignment(lane.lane, "second")),
            )
        session = cao._sessions[result.session.session_id]
        assert session.env_vars["AIPRO_MODEL"] == "first"
        assert session.submitted_messages == ["first task"]
        assert not session.deleted


def test_unleased_session_preserves_base_env_and_wrapper_arguments(tmp_path):
    lanes = LaneRegistry.default()
    env = {"HERMES_HOME": "/isolated/lane", "KEEP": "yes"}
    with (
        FakeCAOServer() as cao,
        CaoSessionController(CAOControlPlaneConfig(base_url=cao.url), lanes) as controller,
    ):
        executor = CaoLaneExecutor(controller, lanes, env=env, poll_interval_seconds=0)
        result = executor.execute(
            lanes.get("developer"), "task", str(tmp_path), LaneExecutionContext("run")
        )
        assert cao._sessions[result.session.session_id].env_vars == env
    assert _launch_wrapper(tmp_path, env) == {
        "argv": ["chat", "--yolo", "--source", "cao"],
        "home": env["HERMES_HOME"],
    }
