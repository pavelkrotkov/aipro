"""Tests for the V3 lane registry."""

import pytest

from ai_pr_orchestrator.v3.config import HermesLanesConfig, LaneProfileConfig
from ai_pr_orchestrator.v3.domain import LaneIdentity
from ai_pr_orchestrator.v3.lanes import (
    ARCHITECTURE_REVIEWER_LANE,
    BREAKER_REVIEWER_LANE,
    DEVELOPER_LANE,
    REQUIREMENTS_REVIEWER_LANE,
    LaneRegistry,
    LaneRegistryError,
    UnknownLaneError,
)


def test_default_registry_provides_the_four_required_lanes():
    registry = LaneRegistry.default()

    assert set(registry.names()) == {
        DEVELOPER_LANE,
        REQUIREMENTS_REVIEWER_LANE,
        BREAKER_REVIEWER_LANE,
        ARCHITECTURE_REVIEWER_LANE,
    }
    assert registry.get(DEVELOPER_LANE).role == "worker"
    for reviewer in (
        REQUIREMENTS_REVIEWER_LANE,
        BREAKER_REVIEWER_LANE,
        ARCHITECTURE_REVIEWER_LANE,
    ):
        assert registry.get(reviewer).role == "reviewer"


def test_every_default_lane_owns_an_independent_profile():
    profiles = [lane.profile_template for lane in LaneRegistry.default()]

    assert len(profiles) == len(set(profiles))


def test_lanes_may_not_share_a_profile_template():
    with pytest.raises(LaneRegistryError, match="share profile template"):
        LaneRegistry(
            [
                LaneIdentity(lane="a", role="reviewer", profile_template="shared"),
                LaneIdentity(lane="b", role="reviewer", profile_template="shared"),
            ]
        )


def test_duplicate_lane_names_are_rejected():
    with pytest.raises(LaneRegistryError, match="Duplicate lane name"):
        LaneRegistry(
            [
                LaneIdentity(lane="a", role="reviewer", profile_template="one"),
                LaneIdentity(lane="a", role="reviewer", profile_template="two"),
            ]
        )


def test_unknown_lane_lookup_lists_the_registered_lanes():
    with pytest.raises(UnknownLaneError, match="Unknown lane 'nope'"):
        LaneRegistry.default().get("nope")


def test_from_config_uses_configured_lanes():
    config = HermesLanesConfig(
        lanes=[
            LaneProfileConfig(name="solo", role="worker", profile_template="solo-profile"),
        ]
    )

    registry = LaneRegistry.from_config(config)

    assert registry.names() == ("solo",)
    assert registry.get("solo").profile_template == "solo-profile"


def test_from_config_without_lanes_falls_back_to_defaults():
    registry = LaneRegistry.from_config(HermesLanesConfig())

    assert set(registry.names()) == set(LaneRegistry.default().names())
