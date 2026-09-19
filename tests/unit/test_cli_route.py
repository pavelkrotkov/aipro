"""Read-only route diagnostics use the actual broker, not another policy path."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from ai_pr_orchestrator import cli
from ai_pr_orchestrator.v3.broker import PolicyBroker, TaskDemand
from ai_pr_orchestrator.v3.cao import CaoSessionController
from ai_pr_orchestrator.v3.config import load_v3_config
from ai_pr_orchestrator.v3.model_router import build_model_broker, resolve_catalog
from ai_pr_orchestrator.v3.queue import GitHubIssueQueue
from ai_pr_orchestrator.v3.telemetry import TelemetryRegistry
from ai_pr_orchestrator.v3.telemetry_hermes import build_telemetry

NOW = datetime(2026, 9, 19, tzinfo=UTC)
ARGS = ["route", "explain", "--role", "worker", "--difficulty", "1"]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    models = [
        {
            "ref": "primary",
            "descriptor": "d1",
            "provider": "p1",
            "cost_class": "free",
            "quality_by_role": {"worker": 3},
        },
        {
            "ref": "fallback",
            "descriptor": "d2",
            "provider": "p2",
            "cost_class": "free",
            "quality_by_role": {"worker": 3},
        },
        {"ref": "disabled", "descriptor": "d3", "enabled": False},
    ]
    (tmp_path / "catalog.yml").write_text(yaml.safe_dump({"models": models}))
    path = tmp_path / "v3.yml"
    path.write_text(
        "model_router:\n  catalog_path: catalog.yml\n"
        "telemetry:\n  resources:\n    - {name: account, provider: p1}\n"
    )
    return path


@pytest.mark.parametrize("as_json", [False, True])
def test_route_matches_broker_without_side_effects(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    config = load_v3_config(config_path)
    registry, _ = build_telemetry(
        config.telemetry, catalog=resolve_catalog(config, base_dir=config_path.parent)
    )
    expected = build_model_broker(
        config, telemetry_source=registry, at=NOW, base_dir=config_path.parent
    ).select(TaskDemand("route-explain", "worker", 1), at=NOW)
    clock = Mock()
    clock.now.return_value = NOW
    monkeypatch.setattr(cli, "datetime", clock)
    build = Mock(wraps=cli.build_telemetry)
    monkeypatch.setattr(cli, "build_telemetry", build)
    sampled = []
    original = TelemetryRegistry.snapshot

    def snapshot(self, resource, *, at=None):
        sampled.append((resource, at))
        return original(self, resource, at=at)

    monkeypatch.setattr(TelemetryRegistry, "snapshot", snapshot)
    forbidden = Mock(side_effect=AssertionError("diagnostic attempted mutation"))
    monkeypatch.setattr(PolicyBroker, "reserve", forbidden)
    monkeypatch.setattr(GitHubIssueQueue, "__init__", forbidden)
    monkeypatch.setattr(CaoSessionController, "__init__", forbidden)
    monkeypatch.setattr(cli.runner, "run", forbidden)
    argv = [*ARGS, "--config", str(config_path), *(["--json"] if as_json else [])]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    if as_json:
        assert json.loads(out) == {
            "schema_version": 1,
            "hypothetical": True,
            "decision": expected.to_dict(),
        }
        assert cli.main(argv) == 0
        assert capsys.readouterr().out == out
    else:
        assert (
            out
            == "Hypothetical current selection; no incumbent, peers, or active lane counts.\n"
            + expected.render()
            + "\n"
        )
    assert build.call_count == (2 if as_json else 1)
    assert sampled == [("account", NOW)] * build.call_count
    assert clock.now.call_count == build.call_count
    forbidden.assert_not_called()
    assert expected.assignment is not None
    assert expected.fallbacks
    assert expected.rejected[0].ref == "disabled"


@pytest.mark.parametrize(
    "secret",
    [
        "ghp_123456789abcdef",
        "github_pat_123456789abcdef",
        "sk-or-v1-123456789abcdef",
        "sk-ant-api03-123456789abcdef",
        "sk-proj-123456789abcdef",
        'CUSTOM_API_KEY="ordinary-custom-secret"',
        "Authorization: Basic dXNlcjpwYXNz",
        "https://user:password@example.test/path",
    ],
)
@pytest.mark.parametrize("as_json", [False, True])
def test_route_redacts_catalog_strings_before_json(
    config_path: Path,
    secret: str,
    as_json: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Refs occur in nested scores/rankings, assignment and fallbacks. Arbitrary
    # environment/config fields are never serialized by BrokerDecision.
    catalog_path = config_path.parent / "catalog.yml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["models"][0]["ref"] = secret
    catalog_path.write_text(yaml.safe_dump(catalog))
    assert cli.main([*ARGS, "--config", str(config_path), *(["--json"] if as_json else [])]) == 0
    out = capsys.readouterr().out
    assert secret not in out
    assert "REDACTED" in out
    if as_json:
        payload = json.loads(out)
        assert payload["decision"]["demand"]["difficulty"] == 1
        assert payload["decision"]["demand"]["lane"] == "route-explain"
        assert payload["decision"]["ranked"][0]["score"]["total"] >= 0


@pytest.mark.parametrize(
    "body, message",
    [
        ("model_router: {catalog_path: missing.yml}", "missing.yml"),
        ("model_router: {catalog: [{descriptor: missing-ref}]}", "missing required"),
        ("telemetry: {resources: [{name: a, provider: p}, {name: b, provider: p}]}", "provider"),
    ],
)
def test_route_invalid_config_exits_cleanly(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.yml"
    path.write_text(body)
    with pytest.raises(SystemExit, match=message):
        cli.main([*ARGS, "--config", str(path)])


def test_route_empty_catalog_is_an_explained_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "v3.yml"
    path.write_text("{}")
    assert cli.main([*ARGS, "--config", str(path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["decision"]
    assert result["assignment"] is None
    assert result["reason"]


@pytest.mark.parametrize(
    "args", [["route"], ["route", "explain"], [*ARGS, "--config", "missing", "--difficulty", "9"]]
)
def test_route_requires_valid_arguments(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


def test_route_redacts_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text(
        'model_router: {catalog: [{ref: "sk-proj-secret12345", descriptor: d, cost_class: invalid}]}'
    )
    with pytest.raises(SystemExit) as exc:
        cli.main([*ARGS, "--config", str(path)])
    assert "sk-proj-secret12345" not in str(exc.value)
    assert "cost_class" in str(exc.value)


def test_route_rejects_non_finite_json_without_partial_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "v3.yml"
    path.write_text(
        "broker: {weight_quality: 1.0e+308, weight_cash_cost: 1.0e+308}\n"
        "model_router:\n  catalog:\n"
        "    - {ref: free, descriptor: d, cost_class: free, quality_by_role: {worker: 5}}\n"
    )
    with pytest.raises(SystemExit, match="non-finite score"):
        cli.main([*ARGS, "--config", str(path), "--json"])
    assert capsys.readouterr().out == ""
