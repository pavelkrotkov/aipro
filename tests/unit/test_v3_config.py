"""Unit tests for the V3 configuration schema."""

from __future__ import annotations

import pytest

from ai_pr_orchestrator.v3.config import (
    HermesLanesConfig,
    LaneProfileConfig,
    ModelCatalogEntry,
    ModelRouterConfig,
    ReviewPolicyConfig,
    V3Config,
    V3ConfigError,
    load_v3_config,
)


def make_valid_config() -> V3Config:
    return V3Config(
        hermes_lanes=HermesLanesConfig(
            lanes=[
                LaneProfileConfig(name="foreman", role="foreman", profile_template="aipro-foreman"),
                LaneProfileConfig(name="coder-1", role="worker", profile_template="aipro-worker"),
                LaneProfileConfig(name="rev-1", role="reviewer", profile_template="aipro-reviewer"),
            ]
        ),
        model_router=ModelRouterConfig(
            catalog=[ModelCatalogEntry(ref="coder-main", descriptor="opaque-provider-string")],
            lane_assignments={"coder-1": "coder-main"},
        ),
        review_policy=ReviewPolicyConfig(reviewer_lanes=["rev-1"], max_review_rounds=2),
    )


class TestValidation:
    def test_valid_config_passes(self) -> None:
        make_valid_config().validate()

    def test_default_config_passes(self) -> None:
        V3Config().validate()

    def test_duplicate_lane_names_rejected(self) -> None:
        config = make_valid_config()
        config.hermes_lanes.lanes.append(
            LaneProfileConfig(name="coder-1", role="worker", profile_template="p")
        )
        with pytest.raises(V3ConfigError, match="Duplicate lane names"):
            config.validate()

    def test_two_foremen_rejected(self) -> None:
        config = make_valid_config()
        config.hermes_lanes.lanes.append(
            LaneProfileConfig(name="foreman-2", role="foreman", profile_template="p")
        )
        with pytest.raises(V3ConfigError, match="foreman"):
            config.validate()

    def test_assignment_to_unknown_lane_rejected(self) -> None:
        config = make_valid_config()
        config.model_router.lane_assignments["ghost-lane"] = "coder-main"
        with pytest.raises(V3ConfigError, match="unknown lane"):
            config.validate()

    def test_assignment_to_unknown_model_ref_rejected(self) -> None:
        config = make_valid_config()
        config.model_router.lane_assignments["coder-1"] = "nonexistent-model"
        with pytest.raises(V3ConfigError, match="unknown model ref"):
            config.validate()

    def test_reviewer_lane_must_have_reviewer_role(self) -> None:
        config = make_valid_config()
        config.review_policy.reviewer_lanes.append("coder-1")
        with pytest.raises(V3ConfigError, match="role 'reviewer'"):
            config.validate()

    def test_unknown_reviewer_lane_rejected(self) -> None:
        config = make_valid_config()
        config.review_policy.reviewer_lanes.append("ghost")
        with pytest.raises(V3ConfigError, match="unknown reviewer lane"):
            config.validate()

    def test_invalid_limits_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="max_review_rounds"):
            V3Config(review_policy={"max_review_rounds": 0}).validate()  # ty: ignore[invalid-argument-type]
        with pytest.raises(V3ConfigError, match="max_consecutive_coder_failures"):
            V3Config(escalation={"max_consecutive_coder_failures": 0}).validate()  # ty: ignore[invalid-argument-type]
        with pytest.raises(V3ConfigError, match="poll_interval"):
            V3Config(cao={"session_poll_interval_seconds": 0}).validate()  # ty: ignore[invalid-argument-type]


class TestSerializationRoundTrip:
    def test_round_trip(self) -> None:
        config = make_valid_config()
        assert V3Config.from_dict(config.to_dict()) == config

    def test_yaml_file_round_trip(self, tmp_path: object) -> None:
        import pathlib

        config = make_valid_config()
        path = pathlib.Path(str(tmp_path)) / "v3.yml"
        import yaml

        path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
        loaded = load_v3_config(path)
        assert loaded == config

    def test_load_missing_file(self, tmp_path: object) -> None:
        import pathlib

        with pytest.raises(V3ConfigError, match="Failed to read"):
            load_v3_config(pathlib.Path(str(tmp_path)) / "nope.yml")

    def test_load_non_mapping(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "v3.yml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(V3ConfigError, match="mapping"):
            load_v3_config(path)


class TestForwardCompatibility:
    def test_unknown_top_level_keys_preserved_as_extras(self) -> None:
        config = V3Config.from_dict({"some_future_section": {"x": 1}, "another_future_flag": True})
        assert config.extras["some_future_section"] == {"x": 1}
        assert config.extras["another_future_flag"] is True
        # Extras survive the round trip.
        assert V3Config.from_dict(config.to_dict()) == config

    def test_unknown_keys_inside_sections_ignored(self) -> None:
        config = V3Config.from_dict(
            {"review_policy": {"max_review_rounds": 5, "future_option": "z"}}
        )
        assert config.review_policy.max_review_rounds == 5
        assert config == V3Config(review_policy={"max_review_rounds": 5})  # ty: ignore[invalid-argument-type]

    def test_no_vendor_names_in_schema(self) -> None:
        """The catalog descriptor is opaque and never parsed by the core."""
        config = V3Config.from_dict(
            {"model_router": {"catalog": [{"ref": "coder-main", "descriptor": "anything-at-all"}]}}
        )
        assert config.model_router.catalog[0].descriptor == "anything-at-all"
