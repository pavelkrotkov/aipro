"""V3 lane registry.

A *lane* is one agent seat: a name, a role, and the profile template that
materializes it. The registry is the single authority on the lane-to-profile
binding, so nothing downstream has to guess which profile a lane runs under.

Two invariants are enforced here rather than left to convention:

- lane names are unique, and
- **profile templates are unique** — every concurrent lane owns an independent
  profile/home. Two lanes sharing one profile would put two concurrent agent
  processes on one profile state directory, which corrupts it.

Lane names, roles, and profile templates are policy strings. No vendor or
model name appears in this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import cast

from .config import HermesLanesConfig
from .domain import AgentLaneRole, LaneIdentity, LaneName

DEVELOPER_LANE: LaneName = "developer"
REQUIREMENTS_REVIEWER_LANE: LaneName = "requirements-reviewer"
BREAKER_REVIEWER_LANE: LaneName = "breaker-reviewer"
ARCHITECTURE_REVIEWER_LANE: LaneName = "architecture-reviewer"

#: The lanes every V3 deployment is expected to provide. Profile templates are
#: named after their lane so an operator provisioning them can map one to one.
DEFAULT_LANES: tuple[LaneIdentity, ...] = (
    LaneIdentity(lane=DEVELOPER_LANE, role="worker", profile_template="aipro-developer"),
    LaneIdentity(
        lane=REQUIREMENTS_REVIEWER_LANE,
        role="reviewer",
        profile_template="aipro-requirements-reviewer",
    ),
    LaneIdentity(
        lane=BREAKER_REVIEWER_LANE,
        role="reviewer",
        profile_template="aipro-breaker-reviewer",
    ),
    LaneIdentity(
        lane=ARCHITECTURE_REVIEWER_LANE,
        role="reviewer",
        profile_template="aipro-architecture-reviewer",
    ),
)


class UnknownLaneError(LookupError):
    """Raised when a lane name is not present in the registry."""


class LaneRegistryError(ValueError):
    """Raised when a lane registry violates a uniqueness invariant."""


class LaneRegistry:
    """Immutable name-to-:class:`LaneIdentity` map with uniqueness checks."""

    def __init__(self, lanes: Iterable[LaneIdentity]) -> None:
        by_name: dict[LaneName, LaneIdentity] = {}
        by_profile: dict[str, LaneName] = {}
        for lane in lanes:
            if lane.lane in by_name:
                raise LaneRegistryError(f"Duplicate lane name {lane.lane!r}")
            owner = by_profile.get(lane.profile_template)
            if owner is not None:
                raise LaneRegistryError(
                    f"lanes {owner!r} and {lane.lane!r} share profile template "
                    f"{lane.profile_template!r}; each lane needs an independent profile"
                )
            by_name[lane.lane] = lane
            by_profile[lane.profile_template] = lane.lane
        self._lanes = by_name

    @classmethod
    def default(cls) -> LaneRegistry:
        return cls(DEFAULT_LANES)

    @classmethod
    def from_config(cls, config: HermesLanesConfig) -> LaneRegistry:
        """Build a registry from the ``hermes_lanes`` config section.

        A section that declares no lanes yields :data:`DEFAULT_LANES`, so a
        deployment only has to write the section when it wants to override the
        standard lane set.
        """
        if not config.lanes:
            return cls.default()
        return cls(
            LaneIdentity(
                lane=lane.name,
                # LaneIdentity re-validates the role against VALID_LANE_ROLES.
                role=cast(AgentLaneRole, lane.role),
                profile_template=lane.profile_template,
            )
            for lane in config.lanes
        )

    def get(self, lane: LaneName) -> LaneIdentity:
        try:
            return self._lanes[lane]
        except KeyError:
            raise UnknownLaneError(
                f"Unknown lane {lane!r}; registered lanes are {sorted(self._lanes)}"
            ) from None

    @property
    def developer_lane(self) -> LaneName:
        """The lane name that identifies the coding (developer) agent.

        The registry owns this binding: callers must look it up here rather
        than hard-coding a literal like ``"developer"`` so a deployment can
        rename the developer lane (e.g. ``implementer``) without breaking
        downstream predicates.
        """
        for lane in self._lanes.values():
            if lane.role == "worker":
                return lane.lane
        # Fall back to the historical default so legacy callers/tests that
        # build a registry with only reviewer lanes still get a usable value.
        return DEVELOPER_LANE

    def names(self) -> tuple[LaneName, ...]:
        return tuple(self._lanes)

    def __contains__(self, lane: object) -> bool:
        return lane in self._lanes

    def __iter__(self) -> Iterator[LaneIdentity]:
        return iter(self._lanes.values())

    def __len__(self) -> int:
        return len(self._lanes)
