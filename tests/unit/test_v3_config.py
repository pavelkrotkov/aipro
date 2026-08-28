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

    def test_remaining_positive_limits_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="stagnation_rounds_threshold"):
            V3Config(escalation={"stagnation_rounds_threshold": 0}).validate()  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        with pytest.raises(V3ConfigError, match="session_timeout_seconds"):
            V3Config(cao={"session_timeout_seconds": 0}).validate()  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        with pytest.raises(V3ConfigError, match="ci_wait_timeout_seconds"):
            V3Config(ci_policy={"ci_wait_timeout_seconds": 0}).validate()  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        with pytest.raises(V3ConfigError, match="max_concurrent"):
            V3Config(
                hermes_lanes=HermesLanesConfig(
                    lanes=[
                        LaneProfileConfig(
                            name="l", role="worker", profile_template="p", max_concurrent=0
                        )
                    ]
                )
            ).validate()

    def test_catalog_max_context_tokens_must_be_positive(self) -> None:
        config = make_valid_config()
        config.model_router.catalog.append(
            ModelCatalogEntry(ref="cheap", descriptor="opaque", max_context_tokens=0)
        )
        with pytest.raises(V3ConfigError, match="max_context_tokens"):
            config.validate()
        # None means "unset" and is allowed.
        make_valid_config().model_router.catalog.append(
            ModelCatalogEntry(ref="unset", descriptor="opaque", max_context_tokens=None)
        )
        make_valid_config().validate()

    def test_malformed_section_shape_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="review_policy"):
            V3Config.from_dict({"review_policy": "bad"})
        with pytest.raises(V3ConfigError, match="model_router"):
            V3Config.from_dict({"model_router": ["bad"]})
        with pytest.raises(V3ConfigError, match="hermes_lanes"):
            V3Config(hermes_lanes="nope")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        # A null section loads as defaults (YAML `review_policy:` with no value).
        config = V3Config.from_dict({"review_policy": None})
        assert config.review_policy.max_review_rounds == 3

    def test_malformed_nested_item_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="ModelCatalogEntry"):
            V3Config.from_dict(
                {"model_router": {"catalog": [{"ref": "m", "descriptor": "d"}, "bad-item"]}}
            )

    def test_unknown_reviewer_lane_reports_config_error_not_stopiteration(self) -> None:
        config = make_valid_config()
        config.review_policy.reviewer_lanes.append("ghost")
        with pytest.raises(V3ConfigError, match="unknown reviewer lane"):
            config.validate()


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

    def test_unknown_keys_inside_sections_preserved(self) -> None:
        config = V3Config.from_dict(
            {"review_policy": {"max_review_rounds": 5, "future_option": "z"}}
        )
        assert config.review_policy.max_review_rounds == 5
        assert config.review_policy.extras == {"future_option": "z"}
        # Unknown section keys survive the round trip.
        assert V3Config.from_dict(config.to_dict()) == config
        assert config.to_dict()["review_policy"]["future_option"] == "z"

    def test_unknown_keys_inside_nested_items_preserved(self) -> None:
        config = V3Config.from_dict(
            {"model_router": {"catalog": [{"ref": "m", "descriptor": "d", "future_flag": True}]}}
        )
        entry = config.model_router.catalog[0]
        assert entry.extras == {"future_flag": True}
        assert V3Config.from_dict(config.to_dict()) == config

    def test_no_vendor_names_in_schema(self) -> None:
        """The catalog descriptor is opaque and never parsed by the core."""
        config = V3Config.from_dict(
            {"model_router": {"catalog": [{"ref": "coder-main", "descriptor": "anything-at-all"}]}}
        )
        assert config.model_router.catalog[0].descriptor == "anything-at-all"


class TestDeclaredShapeValidation:
    """Declared container and primitive fields are validated during loading,
    raising V3ConfigError instead of AttributeError deep in validation."""

    def test_non_list_lanes_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match=r"expects.*list"):
            V3Config.from_dict({"hermes_lanes": {"lanes": "bad"}})

    def test_non_list_required_checks_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match=r"expects.*list"):
            V3Config.from_dict({"ci_policy": {"required_checks": 7}})

    def test_non_list_reviewer_lanes_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match=r"expects.*list"):
            V3Config.from_dict({"review_policy": {"reviewer_lanes": {"a": 1}}})

    def test_string_int_field_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match="max_review_rounds"):
            V3Config.from_dict({"review_policy": {"max_review_rounds": "three"}})

    def test_bool_rejected_for_int_field(self) -> None:
        with pytest.raises(V3ConfigError, match="max_review_rounds"):
            V3Config.from_dict({"review_policy": {"max_review_rounds": True}})

    def test_bad_nested_lane_entry_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match="LaneProfileConfig"):
            V3Config.from_dict({"hermes_lanes": {"lanes": ["not-a-mapping"]}})

    def test_bad_nested_entry_field_type_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match="name"):
            V3Config.from_dict(
                {
                    "hermes_lanes": {
                        "lanes": [{"name": 7, "role": "worker", "profile_template": "p"}]
                    }
                }
            )

    def test_non_dict_lane_assignments_raises_config_error(self) -> None:
        with pytest.raises(V3ConfigError, match="lane_assignments"):
            V3Config.from_dict({"model_router": {"lane_assignments": ["a"]}})

    def test_none_still_allowed_for_optional_fields(self) -> None:
        config = V3Config.from_dict(
            {
                "model_router": {
                    "catalog": [{"ref": "m", "descriptor": "d", "max_context_tokens": None}]
                }
            }
        )
        assert config.model_router.catalog[0].max_context_tokens is None


class TestSafetyPolicySection:
    def test_safety_defaults_mirror_v1_safety_surface(self) -> None:
        config = V3Config.from_dict({})
        safety = config.safety
        assert safety.disallow_forks is True
        assert safety.disallow_workflow_file_changes is True
        assert safety.max_total_iterations == 3
        assert safety.max_commits_per_run == 1
        assert safety.max_coder_invocations_per_run == 1
        assert safety.max_reviewer_triggers_per_run == 3
        assert safety.max_prompt_tokens == 100000
        assert safety.allowed_pr_author_associations == ["OWNER", "MEMBER", "COLLABORATOR"]

    def test_safety_section_round_trips(self) -> None:
        config = V3Config.from_dict(
            {
                "safety": {
                    "max_total_iterations": 5,
                    "max_prompt_tokens": 200000,
                    "allowed_pr_author_associations": ["OWNER"],
                }
            }
        )
        assert config.safety.max_total_iterations == 5
        assert config.to_dict()["safety"]["max_prompt_tokens"] == 200000
        assert config.to_dict()["safety"]["disallow_forks"] is True

    def test_safety_budgets_must_be_positive(self) -> None:
        for field, value in (
            ("max_total_iterations", 0),
            ("max_commits_per_run", 0),
            ("max_coder_invocations_per_run", 0),
            ("max_reviewer_triggers_per_run", -1),
            ("max_prompt_tokens", 0),
        ):
            with pytest.raises(V3ConfigError, match=field):
                V3Config.from_dict({"safety": {field: value}}).validate()


class TestNoneBypassesShapeValidation:
    def test_none_for_non_optional_field_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="required_checks"):
            V3Config.from_dict({"ci_policy": {"required_checks": None}})

    def test_none_for_int_field_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="ci_wait_timeout_seconds"):
            V3Config.from_dict({"ci_policy": {"ci_wait_timeout_seconds": None}})


class TestElementTypeValidation:
    def test_int_element_in_string_list_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="required_checks"):
            V3Config.from_dict({"ci_policy": {"required_checks": [7]}})

    def test_int_value_in_str_dict_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="lane_assignments"):
            V3Config.from_dict({"model_router": {"lane_assignments": {"coder-1": 7}}})

    def test_valid_string_elements_accepted(self) -> None:
        config = V3Config.from_dict({"ci_policy": {"required_checks": ["lint", "tests"]}})
        assert config.ci_policy.required_checks == ["lint", "tests"]


class TestLaneNonEmptyValidation:
    def test_empty_lane_name_rejected_at_config_load(self) -> None:
        with pytest.raises(V3ConfigError, match="lane name"):
            V3Config.from_dict(
                {
                    "hermes_lanes": {
                        "lanes": [{"name": "", "role": "worker", "profile_template": "p"}]
                    }
                }
            )

    def test_empty_profile_template_rejected_at_config_load(self) -> None:
        with pytest.raises(V3ConfigError, match="profile_template"):
            V3Config.from_dict(
                {
                    "hermes_lanes": {
                        "lanes": [{"name": "l1", "role": "worker", "profile_template": ""}]
                    }
                }
            )


class TestQueueLabelDistinctness:
    def test_enabled_label_equal_to_done_label_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="distinct"):
            V3Config.from_dict(
                {"github_queue": {"enabled_label": "v3-work", "done_label": "v3-work"}}
            )

    def test_done_label_equal_to_error_label_rejected(self) -> None:
        with pytest.raises(V3ConfigError, match="distinct"):
            V3Config.from_dict({"github_queue": {"done_label": "v3-work-error"}})

    def test_distinct_labels_accepted(self) -> None:
        config = V3Config.from_dict({})
        config.validate()


class TestDefaultLaneValidation:
    def test_defaults_relying_configs_can_reference_the_effective_lanes(self) -> None:
        config = V3Config(
            model_router={  # ty: ignore[invalid-argument-type]
                "catalog": [{"ref": "high-capability", "descriptor": "provider:model"}],
                "lane_assignments": {"developer": "high-capability"},
            },
            review_policy={"reviewer_lanes": ["requirements-reviewer"]},  # ty: ignore[invalid-argument-type]
        )

        config.validate()

    def test_references_to_unknown_lanes_are_still_rejected(self) -> None:
        config = V3Config(model_router={"lane_assignments": {"nonexistent": "ref"}})  # ty: ignore[invalid-argument-type]

        with pytest.raises(V3ConfigError, match="nonexistent"):
            config.validate()
