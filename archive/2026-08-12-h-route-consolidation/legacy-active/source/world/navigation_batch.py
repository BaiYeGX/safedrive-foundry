"""Route-conditioned, runtime-safe World input contract.

World V0 remains readable.  This V1 wrapper adds the upstream navigation
mission and proves that every selectable K2 candidate carries the same
route/topology binding.  World still outputs candidate scores only; it never
outputs or edits a route manoeuvre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TrafficSignalState,
    canonical_sha256,
)

from .contracts import ActionBranchSample, WorldBatch, WorldContractError

WORLD_NAVIGATION_BATCH_SCHEMA = "safedrive.world_navigation_batch.v1"
ROUTE_ORDER = tuple(RouteManeuver)
NAVIGATION_FEATURES = 24


def navigation_features(context: RouteContextV3) -> tuple[float, ...]:
    values: list[float] = [
        1.0 if context.maneuver is maneuver else 0.0
        for maneuver in ROUTE_ORDER
    ]
    for lane in (context.left_lane, context.right_lane):
        values.extend(
            [
                float(lane.exists),
                float(lane.driving),
                float(lane.same_direction),
                float(lane.lane_change_allowed),
                float(lane.currently_clear),
                max(0.0, min(1.0, float(lane.lane_width_m) / 5.0)),
            ]
        )
    signal = {
        TrafficSignalState.UNKNOWN: 0.0,
        TrafficSignalState.GREEN: 0.25,
        TrafficSignalState.YELLOW: 0.50,
        TrafficSignalState.RED: 0.75,
        TrafficSignalState.STOP_SIGN: 1.0,
    }[context.traffic_signal_state]
    values.extend(
        [
            float(context.is_junction),
            float(context.has_crosswalk),
            signal,
            (
                1.0
                if context.stop_line_distance_m is None
                else max(
                    0.0,
                    min(1.0, float(context.stop_line_distance_m) / 40.0),
                )
            ),
        ]
    )
    if len(values) != NAVIGATION_FEATURES:
        raise AssertionError(
            f"navigation feature count {len(values)} != {NAVIGATION_FEATURES}"
        )
    return tuple(values)


@dataclass(frozen=True)
class WorldNavigationCondition:
    route_maneuver: RouteManeuver
    route_hash: str
    topology_hash: str
    feature_vector: tuple[float, ...]
    context_hash: str

    @classmethod
    def from_route_context(
        cls, context: RouteContextV3
    ) -> "WorldNavigationCondition":
        feature_vector = navigation_features(context)
        context_hash = canonical_sha256(
            {
                "route_maneuver": context.maneuver.value,
                "route_hash": context.route_hash,
                "topology_hash": context.topology_hash,
                "feature_vector": feature_vector,
            }
        )
        return cls(
            route_maneuver=context.maneuver,
            route_hash=context.route_hash,
            topology_hash=context.topology_hash,
            feature_vector=feature_vector,
            context_hash=context_hash,
        )


def v3_world_audit_fields(
    *,
    condition: WorldNavigationCondition,
    candidate_route_hashes: Sequence[str],
    candidate_topology_hashes: Sequence[str],
    guard_candidate_valid: Sequence[bool],
) -> dict[str, Any]:
    if len(candidate_route_hashes) != 2:
        raise WorldContractError("V3 World audit requires two candidate route hashes")
    if len(candidate_topology_hashes) != 2:
        raise WorldContractError(
            "V3 World audit requires two candidate topology hashes"
        )
    if len(guard_candidate_valid) != 2:
        raise WorldContractError("V3 World audit requires two Guard validity flags")
    return {
        "route_maneuver": condition.route_maneuver.value,
        "route_hash": condition.route_hash,
        "topology_hash": condition.topology_hash,
        "navigation_context_hash": condition.context_hash,
        "candidate_route_hashes": list(candidate_route_hashes),
        "candidate_topology_hashes": list(candidate_topology_hashes),
        "guard_candidate_valid": [bool(value) for value in guard_candidate_valid],
    }


def _validate_sample_binding(
    sample: ActionBranchSample,
    condition: WorldNavigationCondition,
) -> None:
    audit = dict(sample.audit)
    required = {
        "route_maneuver",
        "route_hash",
        "topology_hash",
        "navigation_context_hash",
        "candidate_route_hashes",
        "candidate_topology_hashes",
        "guard_candidate_valid",
    }
    missing = sorted(required.difference(audit))
    if missing:
        raise WorldContractError(
            "World V1 sample missing navigation binding fields: "
            + ",".join(missing)
        )
    expected = {
        "route_maneuver": condition.route_maneuver.value,
        "route_hash": condition.route_hash,
        "topology_hash": condition.topology_hash,
        "navigation_context_hash": condition.context_hash,
    }
    for key, value in expected.items():
        if str(audit[key]) != str(value):
            raise WorldContractError(f"World V1 {key} mismatch")
    route_hashes = tuple(str(value) for value in audit["candidate_route_hashes"])
    topology_hashes = tuple(
        str(value) for value in audit["candidate_topology_hashes"]
    )
    if route_hashes != (condition.route_hash, condition.route_hash):
        raise WorldContractError("World V1 candidate route binding mismatch")
    if topology_hashes != (condition.topology_hash, condition.topology_hash):
        raise WorldContractError("World V1 candidate topology binding mismatch")
    guard_valid = tuple(bool(value) for value in audit["guard_candidate_valid"])
    if len(guard_valid) != 2:
        raise WorldContractError("World V1 Guard validity length mismatch")
    for index, available in enumerate(sample.candidate_mask.tolist()):
        if bool(available) and not guard_valid[index]:
            raise WorldContractError(
                f"World V1 candidate {index} available before Guard acceptance"
            )


@dataclass(frozen=True)
class RouteBoundWorldBatch:
    """Observable World batch with immutable mission conditioning."""

    ego_history: Any
    ego_history_mask: Any
    actor_history: Any
    actor_history_mask: Any
    road: Any
    road_mask: Any
    candidates: Any
    candidate_mask: Any
    navigation: Any
    route_maneuvers: tuple[str, ...]
    route_hashes: tuple[str, ...]
    topology_hashes: tuple[str, ...]
    navigation_context_hashes: tuple[str, ...]
    sample_ids: tuple[str, ...] = ()
    schema_version: str = WORLD_NAVIGATION_BATCH_SCHEMA

    @classmethod
    def from_samples(
        cls,
        samples: list[ActionBranchSample],
        route_contexts: Sequence[RouteContextV3],
    ) -> "RouteBoundWorldBatch":
        if len(samples) != len(route_contexts):
            raise WorldContractError(
                "World V1 samples and route_contexts length mismatch"
            )
        base = WorldBatch.from_samples(samples)
        conditions = tuple(
            WorldNavigationCondition.from_route_context(context)
            for context in route_contexts
        )
        for sample, condition in zip(samples, conditions):
            _validate_sample_binding(sample, condition)
        navigation = np.asarray(
            [condition.feature_vector for condition in conditions],
            dtype=np.float32,
        )
        if navigation.shape != (len(samples), NAVIGATION_FEATURES):
            raise WorldContractError("World V1 navigation tensor shape mismatch")
        if not np.isfinite(navigation).all():
            raise WorldContractError("World V1 navigation tensor is non-finite")
        return cls(
            ego_history=base.ego_history,
            ego_history_mask=base.ego_history_mask,
            actor_history=base.actor_history,
            actor_history_mask=base.actor_history_mask,
            road=base.road,
            road_mask=base.road_mask,
            candidates=base.candidates,
            candidate_mask=base.candidate_mask,
            navigation=navigation,
            route_maneuvers=tuple(
                condition.route_maneuver.value for condition in conditions
            ),
            route_hashes=tuple(condition.route_hash for condition in conditions),
            topology_hashes=tuple(
                condition.topology_hash for condition in conditions
            ),
            navigation_context_hashes=tuple(
                condition.context_hash for condition in conditions
            ),
            sample_ids=base.sample_ids,
        )


__all__ = [
    "NAVIGATION_FEATURES",
    "RouteBoundWorldBatch",
    "WORLD_NAVIGATION_BATCH_SCHEMA",
    "WorldNavigationCondition",
    "navigation_features",
    "v3_world_audit_fields",
]
