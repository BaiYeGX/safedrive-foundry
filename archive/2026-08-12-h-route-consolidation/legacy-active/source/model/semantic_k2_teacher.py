"""Executable mixed-semantic teacher for K2 V3.

Availability means that a legal alternative can be executed.  It is not
conditioned on the alternative being the offline Oracle winner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from driving_vla.model.k2_v3_codec import build_k2_v3_bundle
from driving_vla.model.k2_v3_types import (
    AlternativeKind,
    K2PredictionBundleV3,
)
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    TargetLaneSide,
    TrafficSignalState,
)

TEACHER_SCHEMA_V3 = "safedrive.k2_semantic_teacher.v3"


@dataclass(frozen=True)
class SemanticTeacherLabelV3:
    behavior: str
    alternative_kind: AlternativeKind
    alternative_available: bool
    availability_reason: str
    target_lane_side: TargetLaneSide
    avoid_offset_m: float
    temporal_target_speed_mps: float
    departure_fraction: float = 0.08
    pass_fraction: float = 0.65
    rejoin_fraction: float = 0.92
    schema_version: str = TEACHER_SCHEMA_V3
    privileged_used_for_label_selection_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["alternative_kind"] = self.alternative_kind.value
        value["target_lane_side"] = self.target_lane_side.value
        return value


def _preferred_legal_lane(
    route_context: RouteContextV3,
    requested: TargetLaneSide,
) -> TargetLaneSide:
    if requested is not TargetLaneSide.NONE:
        return (
            requested
            if route_context.lane(requested).authorized
            else TargetLaneSide.NONE
        )
    for side in (TargetLaneSide.LEFT, TargetLaneSide.RIGHT):
        if route_context.lane(side).authorized:
            return side
    return TargetLaneSide.NONE


def temporal_alternative_saturated_v3(
    *,
    alternative_kind: AlternativeKind,
    ego_v: float,
    base_speed_mps: float,
    traffic_control_required: bool = False,
) -> bool:
    """Close a temporal branch when nominal is already the minimal action."""
    nominal_speed_threshold = 0.35 if traffic_control_required else 0.75
    return bool(
        alternative_kind is AlternativeKind.TEMPORAL_YIELD
        and float(ego_v) <= 1.0
        and float(base_speed_mps) <= nominal_speed_threshold
    )


def saturate_semantic_teacher_label_v3(
    label: SemanticTeacherLabelV3,
    *,
    ego_v: float,
    base_speed_mps: float,
) -> SemanticTeacherLabelV3:
    if not temporal_alternative_saturated_v3(
        alternative_kind=label.alternative_kind,
        ego_v=ego_v,
        base_speed_mps=base_speed_mps,
        traffic_control_required=(label.behavior == "TRAFFIC_CONTROL_STOP"),
    ):
        return label
    return replace(
        label,
        alternative_kind=AlternativeKind.NONE,
        alternative_available=False,
        availability_reason="NOMINAL_ALREADY_MINIMAL_INTERVENTION",
        target_lane_side=TargetLaneSide.NONE,
        avoid_offset_m=0.0,
        temporal_target_speed_mps=0.0,
    )


def select_semantic_teacher_v3(
    *,
    scenario_family: str,
    route_context: RouteContextV3,
    conflict_side: str = "none",
    requested_overtake_side: TargetLaneSide = TargetLaneSide.NONE,
    conflict_active: bool | None = None,
) -> SemanticTeacherLabelV3:
    family = str(scenario_family or "").lower()
    conflict = str(conflict_side or "").lower()

    if family in {"clear", "empty", "clear_no_alternative"}:
        return SemanticTeacherLabelV3(
            behavior="CLEAR",
            alternative_kind=AlternativeKind.NONE,
            alternative_available=False,
            availability_reason="CLEAR_NO_ALTERNATIVE",
            target_lane_side=TargetLaneSide.NONE,
            avoid_offset_m=0.0,
            temporal_target_speed_mps=0.0,
        )

    if route_context.traffic_signal_state in {
        TrafficSignalState.RED,
        TrafficSignalState.STOP_SIGN,
    }:
        return SemanticTeacherLabelV3(
            behavior="TRAFFIC_CONTROL_STOP",
            alternative_kind=AlternativeKind.TEMPORAL_YIELD,
            alternative_available=True,
            availability_reason="TRAFFIC_CONTROL_REQUIRES_STOP",
            target_lane_side=TargetLaneSide.NONE,
            avoid_offset_m=0.0,
            temporal_target_speed_mps=0.0,
        )

    if conflict_active is False:
        return SemanticTeacherLabelV3(
            behavior="CLEAR",
            alternative_kind=AlternativeKind.NONE,
            alternative_available=False,
            availability_reason="OBSERVABLE_CONFLICT_CLEARED",
            target_lane_side=TargetLaneSide.NONE,
            avoid_offset_m=0.0,
            temporal_target_speed_mps=0.0,
        )

    if "obstruction" in family or "narrow" in family:
        side = _preferred_legal_lane(route_context, requested_overtake_side)
        if not route_context.is_junction and side is not TargetLaneSide.NONE:
            return SemanticTeacherLabelV3(
                behavior="OVERTAKE_REJOIN",
                alternative_kind=AlternativeKind.SPATIAL_OVERTAKE,
                alternative_available=True,
                availability_reason="ADJACENT_SAME_DIRECTION_LANE_AUTHORIZED",
                target_lane_side=side,
                avoid_offset_m=0.0,
                temporal_target_speed_mps=0.0,
            )
        return SemanticTeacherLabelV3(
            behavior="FOLLOW_STOP",
            alternative_kind=AlternativeKind.TEMPORAL_YIELD,
            alternative_available=True,
            availability_reason="NO_LEGAL_OVERTAKE_WAIT_BEHIND_OBSTRUCTION",
            target_lane_side=TargetLaneSide.NONE,
            avoid_offset_m=0.0,
            temporal_target_speed_mps=0.0,
        )

    if "lead" in family or "brake" in family or "follow" in family:
        return SemanticTeacherLabelV3(
            behavior="FOLLOW_STOP",
            alternative_kind=AlternativeKind.TEMPORAL_YIELD,
            alternative_available=True,
            availability_reason="LEAD_VEHICLE_LONGITUDINAL_CONFLICT",
            target_lane_side=TargetLaneSide.NONE,
            avoid_offset_m=0.0,
            temporal_target_speed_mps=0.0,
        )

    if "cross" in family or "merge" in family or "yield" in family:
        return SemanticTeacherLabelV3(
            behavior="YIELD_WAIT",
            alternative_kind=AlternativeKind.TEMPORAL_YIELD,
            alternative_available=True,
            availability_reason="CONFLICT_ZONE_REQUIRES_TIMING_ALTERNATIVE",
            target_lane_side=TargetLaneSide.NONE,
            avoid_offset_m=0.0,
            temporal_target_speed_mps=0.0,
        )

    if "cut" in family:
        if route_context.is_junction:
            return SemanticTeacherLabelV3(
                behavior="CUT_IN_AVOID",
                alternative_kind=AlternativeKind.TEMPORAL_YIELD,
                alternative_available=True,
                availability_reason="JUNCTION_FORBIDS_SPATIAL_AVOID",
                target_lane_side=TargetLaneSide.NONE,
                avoid_offset_m=0.0,
                temporal_target_speed_mps=0.0,
            )
        side = (
            TargetLaneSide.RIGHT
            if conflict == "left"
            else TargetLaneSide.LEFT
            if conflict == "right"
            else TargetLaneSide.NONE
        )
        if side is TargetLaneSide.NONE:
            return SemanticTeacherLabelV3(
                behavior="CUT_IN_AVOID",
                alternative_kind=AlternativeKind.TEMPORAL_YIELD,
                alternative_available=True,
                availability_reason="CUT_IN_SIDE_UNKNOWN_YIELD",
                target_lane_side=TargetLaneSide.NONE,
                avoid_offset_m=0.0,
                temporal_target_speed_mps=0.0,
            )
        # A cut-in avoidance branch is only executable when the away-side
        # lane is a registered, same-direction, currently clear lane.  The
        # previous teacher emitted a spatial candidate solely from the actor
        # side and could therefore send an edge-lane fixture toward a solid
        # boundary/opposing lane.  Fail closed to the temporal branch; this
        # preserves the native route and lets the Guard/MPC yield contract
        # handle the interaction without inventing a lane permission.
        if not route_context.lane(side).authorized:
            return SemanticTeacherLabelV3(
                behavior="CUT_IN_AVOID",
                alternative_kind=AlternativeKind.TEMPORAL_YIELD,
                alternative_available=True,
                availability_reason="CUT_IN_NO_LEGAL_AWAY_SIDE_YIELD",
                target_lane_side=TargetLaneSide.NONE,
                avoid_offset_m=0.0,
                temporal_target_speed_mps=0.0,
            )
        return SemanticTeacherLabelV3(
            behavior="CUT_IN_AVOID",
            alternative_kind=AlternativeKind.SPATIAL_AVOID,
            alternative_available=True,
            availability_reason="CUT_IN_AWAY_SIDE_EXECUTABLE",
            target_lane_side=side,
            # A 2 m-wide ego in a 3.5 m lane has about 0.75 m geometric
            # centre-offset budget.  Preserve extra tracking/yaw margin while
            # remaining above the audited 0.50 m minimum intervention.
            avoid_offset_m=0.55,
            temporal_target_speed_mps=0.0,
        )

    return SemanticTeacherLabelV3(
        behavior="UNKNOWN",
        alternative_kind=AlternativeKind.NONE,
        alternative_available=False,
        availability_reason="UNSUPPORTED_FAMILY_FAIL_CLOSED",
        target_lane_side=TargetLaneSide.NONE,
        avoid_offset_m=0.0,
        temporal_target_speed_mps=0.0,
    )


def build_semantic_teacher_bundle_v3(
    *,
    native_path_xy: Sequence[Sequence[float]],
    route_context: RouteContextV3,
    label: SemanticTeacherLabelV3,
    ego_v: float,
    base_speed_mps: float,
    backbone_forward_id: str,
    observation_identity: Mapping[str, Any] | None = None,
    ego_route_error_m: float | None = None,
    overtake_actor_lon_m: float | None = None,
    overtake_phase_v3: str = "",
) -> K2PredictionBundleV3:
    effective_label = saturate_semantic_teacher_label_v3(
        label,
        ego_v=ego_v,
        base_speed_mps=base_speed_mps,
    )
    return build_k2_v3_bundle(
        native_path_xy=native_path_xy,
        route_context=route_context,
        ego_v=ego_v,
        base_speed_mps=base_speed_mps,
        alternative_kind=effective_label.alternative_kind,
        alternative_available=effective_label.alternative_available,
        alternative_reason=effective_label.availability_reason,
        target_lane_side=effective_label.target_lane_side,
        avoid_offset_m=effective_label.avoid_offset_m,
        temporal_target_speed_mps=effective_label.temporal_target_speed_mps,
        ego_route_error_m=ego_route_error_m,
        overtake_actor_lon_m=overtake_actor_lon_m,
        overtake_phase_v3=overtake_phase_v3,
        observation_identity=observation_identity,
        backbone_forward_id=backbone_forward_id,
        model_id="sdf-k2-v3@semantic-teacher",
        spatial_head_checkpoint_hash="teacher",
    )


__all__ = [
    "SemanticTeacherLabelV3",
    "TEACHER_SCHEMA_V3",
    "build_semantic_teacher_bundle_v3",
    "saturate_semantic_teacher_label_v3",
    "select_semantic_teacher_v3",
    "temporal_alternative_saturated_v3",
]
