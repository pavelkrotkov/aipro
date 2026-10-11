"""Operator views remain read-only and audit evidence survives real CAO turns."""

import json
import logging
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from ai_pr_orchestrator import cli
from ai_pr_orchestrator.logging import JsonFormatter, SecretRedactor, setup_logging
from ai_pr_orchestrator.v3.audit import workflow_snapshot
from ai_pr_orchestrator.v3.cao import CaoSessionController, CaoUnavailableError
from ai_pr_orchestrator.v3.status import status_snapshot
from tests.integration._harness import developer_output, developer_session_name, script_protocol


@pytest.mark.parametrize("corrupt_field", [None, "work_item_id", "profile_template"])
def test_status_and_step_do_not_modify_an_overdue_active_session(
    fake_cao, foreman_harness, cao_controller, monkeypatch, tmp_path, capsys, corrupt_field
):
    loop, queue, github = foreman_harness([1, 2])
    fake_cao.set_output(developer_session_name(loop.run_id), developer_output())
    execute = loop._executor.execute

    def crash(*args, **kwargs):
        execute(*args, **kwargs)
        raise KeyboardInterrupt("coordinator lost before recording result")

    monkeypatch.setattr(loop._executor, "execute", crash)
    with pytest.raises(KeyboardInterrupt):
        loop.run_pass(max_items=1)
    state = queue.load_state("owner/repo#1")
    worker = fake_cao._sessions[developer_session_name(loop.run_id)]
    worker.metadata["launched_at"] = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    worker.status_sequence = ["processing"]
    worker.status_index = 0
    if corrupt_field == "work_item_id":
        worker.metadata["work_item_id"] = "owner/repo#999"
    if corrupt_field == "profile_template":
        worker.metadata["lane"]["profile_template"] = "foreign-profile"
    metadata = deepcopy(worker.metadata)
    request = CaoSessionController._request
    calls = []

    def read(controller, method, url, **kwargs):
        assert method == "GET", (method, url)
        calls.append(url)
        return request(controller, method, url, **kwargs)

    monkeypatch.setattr(CaoSessionController, "_request", read)
    monkeypatch.setattr(
        "ai_pr_orchestrator.github.client.GitHubClient", lambda *_: nullcontext(github)
    )
    config = tmp_path / "v3.yml"
    config.write_text(yaml.safe_dump({"cao": {"base_url": fake_cao.url}}))
    assert (
        cli.main(
            [
                "step",
                "--dry-run",
                "--json",
                "--config",
                str(config),
                "--repo",
                "owner/repo",
                "--token",
                "ghp_invented_test_credential",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1 and payload["dry_run"]
    assert payload["next_step"] == {"action": "reconcile_active", "issue": "owner/repo#1"}
    active = next(
        work for work in payload["work_items"] if work["work_item_id"] == state.work_item_id
    )
    assert active["phase"] == "coding"
    runtime = active["lanes"][0]["runtime"]
    assert runtime["verified"] == (corrupt_field is None)
    assert runtime["state"] == ("running" if corrupt_field is None else None)
    assert queue.load_state(state.work_item_id) == state
    assert queue.load_state("owner/repo#2") is None
    assert worker.metadata == metadata and not worker.deleted
    assert len(worker.submitted_messages) == 1 and calls


def test_status_retains_durable_context_when_cao_is_unavailable(
    fake_cao, foreman_harness, cao_controller, monkeypatch
):
    loop, queue, _ = foreman_harness([1])
    script_protocol(fake_cao, loop)
    assert loop.run_pass()[0].final_phase == "done"
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise CaoUnavailableError("control plane unavailable")

    monkeypatch.setattr(cao_controller, "inspect_session", unavailable)
    work = status_snapshot(queue, cao_controller)[0]
    assert work["phase"] == "done" and work["pr_number"]
    assert work["head_sha"] == "sha" and work["ci_status"]["passed"]
    assert work["usage"]["coder_invocations"] == 2
    assert all(not lane["runtime"]["verified"] for lane in work["lanes"])
    assert all(lane["last_durable_state"] == "stopped" for lane in work["lanes"])
    assert len(calls) == 1
    origin = queue.load_state("owner/repo#1").extras["review_waves"]["review-1"]["lanes"]
    assert (
        work["archived"][0]["origins"][0]["model_ref"]
        == origin["requirements-reviewer"]["assignment"]["model_ref"]
    )


def test_audit_correlates_findings_to_model_head_and_disposition_without_prompts(
    fake_cao, foreman_harness, caplog, monkeypatch
):
    loop, queue, github = foreman_harness([1])
    github.seed_issue(1, ["v3-work", "v3-work-enabled"], body="private-source-canary-not-for-logs")
    script_protocol(fake_cao, loop)
    logger = logging.getLogger("ai_pr_orchestrator")
    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.INFO)
    setup_logging(logger_name="aipro_test_transport_privacy")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.INFO)
    monkeypatch.setattr(logger, "propagate", True)
    with caplog.at_level(logging.INFO):
        assert loop.run_pass()[0].final_phase == "done"
    formatter = JsonFormatter(SecretRedactor())
    events = [
        json.loads(formatter.format(record))
        for record in caplog.records
        if getattr(record, "schema_version", None) == 1
    ]
    serialized = json.dumps(events)
    assert "private-source-canary-not-for-logs" not in serialized
    assert '"prompt"' not in serialized and '"output_summary"' not in serialized
    found = next(event for event in events if event.get("work", {}).get("findings"))
    origin = found["work"]["findings"][0]["origin"]
    assert origin["role"] == "reviewer" and origin["head_sha"] == "sha"
    first = queue.load_state("owner/repo#1").extras["review_waves"]["review-1"]["lanes"]
    assert origin["model_ref"] == first["requirements-reviewer"]["assignment"]["model_ref"]
    assert "private-source-canary-not-for-logs" not in "\n".join(
        r.getMessage() for r in caplog.records
    )
    final = events[-1]["work"]
    assert final["phase"] == "done"
    assert [(d["finding_id"], d["action"]) for d in final["dispositions"]] == [
        ("guard", "rebut"),
        ("guard", "accept"),
    ]
    assert all(
        event["run_id"] == loop.run_id and event["issue"] == "owner/repo#1" for event in events
    )
    durable = workflow_snapshot(queue.load_state("owner/repo#1"))
    assert all(lane["routing"] for lane in durable["lanes"])


def test_step_preview_respects_terminal_state_and_global_capacity(
    fake_cao, foreman_harness, monkeypatch, tmp_path, capsys
):
    _, queue, github = foreman_harness([1, 2, 3])
    first = queue.list_ready()[0]
    state = queue.claim(first, "human-run")
    queue.mark_needs_human(first, state, reason="operator decision")
    github.add_label(1, "v3-work")
    monkeypatch.setattr(
        "ai_pr_orchestrator.github.client.GitHubClient", lambda *_: nullcontext(github)
    )
    config = tmp_path / "v3.yml"
    config.write_text(yaml.safe_dump({"cao": {"base_url": fake_cao.url}}))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="Status requires GITHUB_TOKEN"):
        cli.main(["status", "--config", str(config), "--repo", "owner/repo"])
    args = [
        "step",
        "--json",
        "--config",
        str(config),
        "--repo",
        "owner/repo",
        "--token",
        "ghp_invented_test_credential",
    ]
    assert cli.main(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["next_step"] == {"action": "claim", "issue": "owner/repo#2", "run_id": None}
    assert queue.load_state("owner/repo#2") is None
    queue.claim(queue.list_ready()[0], "active-run")
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["next_step"]["action"] == "reconcile_active"
    queue.claim(queue.list_ready()[0], "conflicting-run")
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["next_step"]["action"] == "halt"
    assert not fake_cao._sessions
