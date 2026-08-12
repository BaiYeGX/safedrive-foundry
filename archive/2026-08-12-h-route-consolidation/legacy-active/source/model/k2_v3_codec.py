"""Deterministic topology-aware codec for K2 V3 intent.

The codec expands a learned semantic/side/continuous intent into an executable
path.  It never chooses a manoeuvre and never changes the upstream route.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from driving_vla.model.canonicalizer import cum_arclength
from driving_vla.model.frenet_codec import (
    path_frame,
    path_max_abs_curvature,
    sample_native_at_s,
    smooth_path_xy,
)
from driving_vla.model.k2_spatial_builder import _project_speed, _sample_t10_on_path
from driving_vla.model.k2_v3_types import (
    AlternativeKind,
    K2CandidateV3,
    K2PredictionBundleV3,
    K2V3Config,
    ManeuverPhase,
    candidate_intent_hash,
    load_k2_v3_config,
    stable_hash_t10,
    stable_hash_xy,
)
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    canonical_sha256,
)
from driving_vla.model.speed_convert import planner_samples_from_cruise_scalar


def _points(value: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    points = tuple((float(row[0]), float(row[1])) for row in value)
    if len(points) < 2:
        raise ValueError("candidate path requires at least two points")
    return points


def _smoothstep(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return u * u * (3.0 - 2.0 * u)


def _window(value: float, start: float, end: float) -> float:
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return _smoothstep((value - start) / max(end - start, 1.0e-6))


def _sample_polyline(
    path_xy: Sequence[Sequence[float]], normalized_s: float
) -> tuple[float, float]:
    path = _points(path_xy)
    s_values, tangents, normals = path_frame(path)
    x, y, _yaw, _nx, _ny = sample_native_at_s(
        path,
        s_values,
        tangents,
        normals,
        max(0.0, min(1.0, float(normalized_s))) * float(s_values[-1]),
    )
    return float(x), float(y)


def _project_polyline_s(
    path_xy: Sequence[Sequence[float]],
    point_xy: Sequence[float],
) -> float:
    path = _points(path_xy)
    s_values = cum_arclength(path)
    px, py = float(point_xy[0]), float(point_xy[1])
    best_distance = math.inf
    best_s = 0.0
    for index, (first, second) in enumerate(zip(path, path[1:])):
        dx = float(second[0] - first[0])
        dy = float(second[1] - first[1])
        length2 = dx * dx + dy * dy
        if length2 <= 1.0e-12:
            continue
        alpha = max(
            0.0,
            min(
                1.0,
                ((px - first[0]) * dx + (py - first[1]) * dy) / length2,
            ),
        )
        x = first[0] + alpha * dx
        y = first[1] + alpha * dy
        distance = math.hypot(px - x, py - y)
        if distance < best_distance:
            best_distance = distance
            best_s = float(s_values[index]) + alpha * math.sqrt(length2)
    return float(best_s)


def _sample_polyline_s(
    path_xy: Sequence[Sequence[float]],
    arc_s: float,
) -> tuple[float, float]:
    path = _points(path_xy)
    s_values, tangents, normals = path_frame(path)
    x, y, _yaw, _nx, _ny = sample_native_at_s(
        path,
        s_values,
        tangents,
        normals,
        max(0.0, min(float(s_values[-1]), float(arc_s))),
    )
    return float(x), float(y)


def lane_blend_path(
    origin_centerline_xy: Sequence[Sequence[float]],
    target_centerline_xy: Sequence[Sequence[float]],
    *,
    rejoin: bool,
    departure_start: float = 0.08,
    departure_end: float = 0.42,
    rejoin_start: float = 0.58,
    rejoin_end: float = 0.92,
    point_count: int | None = None,
) -> tuple[tuple[float, float], ...]:
    """Blend the target lane segment aligned to the current local path.

    Frozen topology carries a long adjacent centerline while SimLingo emits a
    short ego-local path.  Projecting both local endpoints onto the target
    prevents the entire frozen lane from being compressed into one proposal.
    """
    origin = _points(origin_centerline_xy)
    target = _points(target_centerline_xy)
    target_s0 = _project_polyline_s(target, origin[0])
    target_s1 = _project_polyline_s(target, origin[-1])
    origin_length = float(cum_arclength(origin)[-1])
    if target_s1 <= target_s0 + 0.25:
        target_s1 = target_s0 + origin_length
    count = max(20, len(origin), int(point_count or 0))
    output: list[tuple[float, float]] = []
    for index in range(count):
        u = index / max(count - 1, 1)
        alpha = _window(u, departure_start, departure_end)
        if rejoin:
            alpha *= 1.0 - _window(u, rejoin_start, rejoin_end)
        ox, oy = _sample_polyline(origin, u)
        tx, ty = _sample_polyline_s(
            target,
            target_s0 + u * (target_s1 - target_s0),
        )
        output.append(
            (
                ox + alpha * (tx - ox),
                oy + alpha * (ty - oy),
            )
        )
    return smooth_path_xy(output, passes=2, hard_max_abs_curvature=0.253)


def spatial_avoid_path(
    native_path_xy: Sequence[Sequence[float]],
    *,
    lateral_offset_m: float,
    start: float = 0.02,
    recover: float = 0.78,
) -> tuple[tuple[float, float], ...]:
    """Create a bounded out-and-back path around the native VLA proposal."""
    native = _points(native_path_xy)
    s_values, tangents, normals = path_frame(native)
    output: list[tuple[float, float]] = []
    for index, (x, y) in enumerate(native):
        u = index / max(len(native) - 1, 1)
        rise_end = min(float(recover), float(start) + 0.18)
        if u <= start:
            alpha = 0.0
        elif u < rise_end:
            alpha = _smoothstep(
                (u - float(start)) / max(rise_end - float(start), 1.0e-6)
            )
        elif u <= recover:
            alpha = 1.0
        else:
            alpha = 1.0 - _smoothstep(
                (u - float(recover)) / max(1.0 - float(recover), 1.0e-6)
            )
        nx, ny = normals[index]
        output.append(
            (
                float(x) + float(lateral_offset_m) * alpha * nx,
                float(y) + float(lateral_offset_m) * alpha * ny,
            )
        )
    return smooth_path_xy(output, passes=2, hard_max_abs_curvature=0.253)


def _trajectory_for_path(
    path_xy: Sequence[Sequence[float]],
    *,
    ego_v: float,
    target_speed_mps: float,
    config: K2V3Config,
) -> tuple[tuple[float, ...], ...]:
    """Build an executable T10 profile without changing the native XY path.

    SimLingo's sparse path can have a higher curvature at the exact T10
    segment than the path-wide curvature used by the original cruise cap.
    Guard audits the pointwise ``v²κ`` envelope, so retime against that same
    sampled curvature while preserving route geometry and semantics.
    """
    path = _points(path_xy)
    curvature = path_max_abs_curvature(path)
    curve_speed_cap = (
        0.97
        * math.sqrt(config.max_lateral_accel_mps2 / max(curvature, 1.0e-9))
        if curvature > 1.0e-9
        else math.inf
    )
    bounded_target_speed = min(
        max(0.0, float(target_speed_mps)),
        float(curve_speed_cap),
    )
    targets = planner_samples_from_cruise_scalar(
        bounded_target_speed,
        n=int(config.t_steps),
    )
    path_length = float(cum_arclength(path)[-1])

    def _fit_path_length(
        speeds_in: Sequence[float], arcs_in: Sequence[float]
    ) -> tuple[list[float], list[float], list[float]]:
        speeds_out = [float(value) for value in speeds_in]
        arcs_out = [float(value) for value in arcs_in]
        if arcs_out and arcs_out[-1] > path_length:
            scale = path_length / max(arcs_out[-1], 1.0e-6)
            speeds_out = [max(0.0, float(value) * scale) for value in speeds_out]
            arcs_out = [float(value) * scale for value in arcs_out]
            previous = max(0.0, float(ego_v) * scale)
        else:
            previous = max(0.0, float(ego_v))
        accels_out: list[float] = []
        for speed in speeds_out:
            accel = (float(speed) - previous) / max(config.dt_s, 1.0e-6)
            accels_out.append(
                max(-config.max_decel_mps2, min(config.max_accel_mps2, accel))
            )
            previous = float(speed)
        return speeds_out, arcs_out, accels_out

    # The arc-to-segment lookup changes after a cap, so iterate a fixed small
    # number of times.  This remains deterministic across devices.
    target_speeds = [float(value) for value in targets]
    rows: tuple[tuple[float, ...], ...] = tuple()
    for _ in range(6):
        speeds, arcs, accels = _project_speed(
            target_speeds,
            ego_v=float(ego_v),
            dt_s=float(config.dt_s),
            max_accel=float(config.max_accel_mps2),
            max_decel=float(config.max_decel_mps2),
        )
        speeds, arcs, accels = _fit_path_length(speeds, arcs)
        rows = _sample_t10_on_path(path, speeds, arcs, accels)
        if not rows:
            return rows
        limits: list[float] = []
        changed = False
        for row, target in zip(rows, target_speeds):
            kappa = abs(float(row[5])) if len(row) >= 6 else 0.0
            limit = (
                0.97
                * math.sqrt(
                    float(config.max_lateral_accel_mps2)
                    / max(kappa, 1.0e-9)
                )
                if kappa > 1.0e-9
                else math.inf
            )
            value = min(max(0.0, float(target)), float(limit))
            limits.append(value)
            changed = changed or value < float(target) - 1.0e-6
        # Backward braking envelope gives earlier samples enough distance to
        # reach the downstream curve cap without exceeding max deceleration.
        for index in range(len(limits) - 2, -1, -1):
            limits[index] = min(
                limits[index],
                limits[index + 1]
                + float(config.max_decel_mps2) * float(config.dt_s),
            )
        target_speeds = limits
        if not changed:
            break
    return rows


def _native_executable_path(
    native_path_xy: Sequence[Sequence[float]],
    *,
    ego_v: float,
    target_speed_mps: float,
    config: K2V3Config,
) -> tuple[tuple[float, float], ...]:
    """Continuity-smooth native geometry until its incoming-speed profile fits.

    The first T10 sample may be unable to brake from the measured incoming
    speed to a sparse near-field curve cap.  A bounded extra smoothing pass is
    a geometry-continuity operation, not a route/map substitution.  The fixed
    eight-pass ceiling makes the operation deterministic and auditable; if a
    path still cannot be made executable, Guard continues to reject it.
    """
    path = smooth_path_xy(
        _points(native_path_xy),
        passes=4,
        hard_max_abs_curvature=config.max_abs_curvature,
    )
    for _ in range(8):
        rows = _trajectory_for_path(
            path,
            ego_v=float(ego_v),
            target_speed_mps=float(target_speed_mps),
            config=config,
        )
        if rows and max(
            float(row[3]) ** 2 * abs(float(row[5])) for row in rows
        ) <= float(config.max_lateral_accel_mps2) + 1.0e-6:
            return path
        path = smooth_path_xy(
            path,
            passes=1,
            hard_max_abs_curvature=config.max_abs_curvature,
        )
    return path


def build_candidate_v3(
    *,
    candidate_id: str,
    alternative_kind: AlternativeKind,
    route_context: RouteContextV3,
    spatial_path_xy: Sequence[Sequence[float]],
    ego_v: float,
    target_speed_mps: float,
    available: bool,
    availability_reason: str,
    target_lane_side: TargetLaneSide = TargetLaneSide.NONE,
    maneuver_phase: ManeuverPhase = ManeuverPhase.PROPOSED,
    probability: float = 0.5,
    corridor_payload: Mapping[str, Any] | None = None,
    head_lineage: str = "spatial_mode_head_v3",
    feature_content_hash: str = "",
    raw_head_output_hash: str = "",
    metadata: Mapping[str, Any] | None = None,
    config: K2V3Config | None = None,
) -> K2CandidateV3:
    cfg = config or load_k2_v3_config()
    path = _points(spatial_path_xy)
    trajectory = _trajectory_for_path(
        path,
        ego_v=ego_v,
        target_speed_mps=target_speed_mps,
        config=cfg,
    )
    path_hash = stable_hash_xy(path)
    timed_hash = stable_hash_t10(trajectory)
    corridor_hash = canonical_sha256(
        corridor_payload
        or {
            "route_hash": route_context.route_hash,
            "native_path": [list(point) for point in path],
        }
    )
    intent_hash = candidate_intent_hash(
        candidate_id=candidate_id,
        alternative_kind=alternative_kind,
        route_maneuver=route_context.maneuver,
        target_lane_side=target_lane_side,
        route_hash=route_context.route_hash,
        topology_hash=route_context.topology_hash,
        corridor_hash=corridor_hash,
        spatial_path_hash=path_hash,
        timed_trajectory_hash=timed_hash,
    )
    lane = (
        None
        if target_lane_side is TargetLaneSide.NONE
        else route_context.lane(target_lane_side)
    )
    return K2CandidateV3(
        candidate_id=candidate_id,
        alternative_kind=alternative_kind,
        route_maneuver=route_context.maneuver,
        available=bool(available),
        availability_reason=str(availability_reason),
        probability=float(probability),
        target_lane_side=target_lane_side,
        maneuver_phase=maneuver_phase,
        spatial_path_xy=path,
        points_xy_yaw_v_a_kappa=trajectory,
        route_hash=route_context.route_hash,
        topology_hash=route_context.topology_hash,
        corridor_hash=corridor_hash,
        intent_hash=intent_hash,
        spatial_path_hash=path_hash,
        timed_trajectory_hash=timed_hash,
        origin_lane_signature=(
            f"{route_context.origin_road_id}:{route_context.origin_lane_id}"
        ),
        target_lane_signature=(
            ""
            if lane is None
            else f"{lane.road_id}:{lane.lane_id}"
        ),
        head_lineage=head_lineage,
        feature_content_hash=feature_content_hash,
        raw_head_output_hash=raw_head_output_hash,
        metadata=dict(metadata or {}),
    )


def build_k2_v3_bundle(
    *,
    native_path_xy: Sequence[Sequence[float]],
    route_context: RouteContextV3,
    ego_v: float,
    base_speed_mps: float,
    alternative_kind: AlternativeKind,
    alternative_available: bool,
    alternative_reason: str,
    nominal_target_speed_mps: float | None = None,
    target_lane_side: TargetLaneSide = TargetLaneSide.NONE,
    avoid_offset_m: float = 0.8,
    temporal_target_speed_mps: float = 0.0,
    departure_start: float = 0.08,
    departure_end: float = 0.42,
    rejoin_start: float = 0.58,
    rejoin_end: float = 0.92,
    ego_route_error_m: float | None = None,
    overtake_actor_lon_m: float | None = None,
    overtake_phase_v3: str = "",
    observation_identity: Mapping[str, Any] | None = None,
    backbone_forward_id: str,
    model_id: str = "sdf-k2-v3@heads",
    base_checkpoint_hash: str = "unset",
    spatial_head_checkpoint_hash: str = "unset",
    feature_content_hash: str = "",
    raw_head_output_hash: str = "",
    head_lineage: str = "spatial_mode_head_v3",
    alternative_metadata: Mapping[str, Any] | None = None,
    config: K2V3Config | None = None,
) -> K2PredictionBundleV3:
    """Build fixed nominal slot plus one route-bound semantic alternative."""
    cfg = config or load_k2_v3_config()
    native = _points(native_path_xy)
    # SimLingo remains the geometric source of truth.  The deterministic
    # continuity codec only damps vertex-level heading discontinuities before
    # PathManager/MPC; it does not consult or splice in the map route.
    # Continuity smoothing remains a native-path operation (no map centerline
    # or route replacement) and adapts only to the measured incoming speed.
    executable_native = _native_executable_path(
        native,
        ego_v=float(ego_v),
        target_speed_mps=float(base_speed_mps),
        config=cfg,
    )
    raw_native_path_hash = stable_hash_xy(native)
    executable_native_path_hash = stable_hash_xy(executable_native)
    nominal_side = TargetLaneSide.NONE
    if route_context.maneuver is RouteManeuver.ROUTE_CHANGE_LEFT:
        nominal_side = TargetLaneSide.LEFT
    elif route_context.maneuver is RouteManeuver.ROUTE_CHANGE_RIGHT:
        nominal_side = TargetLaneSide.RIGHT
    nominal_target_speed = (
        float(base_speed_mps)
        if nominal_target_speed_mps is None
        else float(nominal_target_speed_mps)
    )
    nominal = build_candidate_v3(
        candidate_id="v3_nominal_progress",
        alternative_kind=AlternativeKind.NOMINAL_PROGRESS,
        route_context=route_context,
        spatial_path_xy=executable_native,
        ego_v=ego_v,
        target_speed_mps=nominal_target_speed,
        available=True,
        availability_reason="mission_nominal",
        target_lane_side=nominal_side,
        probability=0.5,
        corridor_payload={
            "kind": "native_route_continuity_v1",
            "route_hash": route_context.route_hash,
            "raw_native_path_hash": raw_native_path_hash,
            "executable_native_path_hash": executable_native_path_hash,
            "continuity_codec": "native_xy_smooth_v1",
        },
        head_lineage="simlingo_native_anchor_continuity_v1",
        feature_content_hash=feature_content_hash,
        raw_head_output_hash=raw_head_output_hash,
        config=cfg,
    )

    alternative_path = executable_native
    alternative_speed = base_speed_mps
    phase = ManeuverPhase.PROPOSED
    corridor: dict[str, Any] = {
        "kind": alternative_kind.value,
        "route_hash": route_context.route_hash,
        "raw_native_path_hash": raw_native_path_hash,
        "executable_native_path_hash": executable_native_path_hash,
        "continuity_codec": "native_xy_smooth_v1",
    }
    invalid_spatial_side = (
        alternative_kind in {
            AlternativeKind.SPATIAL_AVOID,
            AlternativeKind.SPATIAL_OVERTAKE,
        }
        and target_lane_side is TargetLaneSide.NONE
    )
    if invalid_spatial_side:
        # Preserve the head's raw kind/side for audit, but fail closed at the
        # codec boundary.  In particular, never call lane(NONE), and never
        # synthesize a side or route rescue for an invalid spatial proposal.
        alternative_available = False
        alternative_reason = "HEAD_INVALID_SPATIAL_SIDE"
        corridor["invalid_spatial_side"] = True
    if alternative_kind is AlternativeKind.SPATIAL_AVOID:
        # path_frame's positive normal is mathematical left.  CARLA is
        # left-handed, so physical left is the negative path normal.
        sign = -1.0 if target_lane_side is TargetLaneSide.LEFT else 1.0
        effective_offset = abs(float(avoid_offset_m))
        alternative_path = spatial_avoid_path(
            executable_native,
            lateral_offset_m=sign * effective_offset,
        )
        alternative_speed = min(float(base_speed_mps), 2.0)
        phase = ManeuverPhase.DEPART
        corridor["native_path"] = [
            list(point) for point in executable_native
        ]
        corridor["desired_route_offset_m"] = sign * abs(float(avoid_offset_m))
        corridor["ego_route_error_m"] = ego_route_error_m
        corridor["effective_avoid_offset_m"] = effective_offset
    elif alternative_kind is AlternativeKind.SPATIAL_OVERTAKE:
        # An overtake without a side is retained as an unavailable diagnostic
        # candidate; it must never query the route context with NONE.
        if not invalid_spatial_side:
            lane = route_context.lane(target_lane_side)
            requested_phase = str(overtake_phase_v3 or "").upper()
            if requested_phase not in {"DEPART", "PASS", "REJOIN"}:
                requested_phase = (
                    "REJOIN"
                    if (
                        overtake_actor_lon_m is not None
                        and float(overtake_actor_lon_m) <= -2.0
                    )
                    else "DEPART"
                )
            rejoining = requested_phase == "REJOIN"
            target_path = route_context.route_xy if rejoining else lane.centerline_xy
            if target_path:
                alternative_path = lane_blend_path(
                    executable_native,
                    target_path,
                    rejoin=False,
                    departure_start=departure_start,
                    departure_end=departure_end,
                )
            alternative_speed = max(float(base_speed_mps), 3.5)
            phase = ManeuverPhase(requested_phase)
            corridor.update(
                {
                    "origin": [list(point) for point in executable_native],
                    "target": [list(point) for point in target_path],
                    "overtake_phase_v3": requested_phase,
                    "actor_lon_m": overtake_actor_lon_m,
                    "lane_width_m": lane.lane_width_m,
                    "departure_start": float(departure_start),
                    "departure_end": float(departure_end),
                    "rejoin_start": float(rejoin_start),
                    "rejoin_end": float(rejoin_end),
                }
            )
    elif alternative_kind is AlternativeKind.TEMPORAL_YIELD:
        alternative_speed = max(0.0, float(temporal_target_speed_mps))
        phase = ManeuverPhase.WAIT
        corridor["shared_path_hash"] = executable_native_path_hash
    elif alternative_kind is AlternativeKind.NONE:
        alternative_available = False
        phase = ManeuverPhase.COMPLETE
        corridor["shared_path_hash"] = executable_native_path_hash

    alternative = build_candidate_v3(
        candidate_id="v3_alternative",
        alternative_kind=alternative_kind,
        route_context=route_context,
        spatial_path_xy=alternative_path,
        ego_v=ego_v,
        target_speed_mps=alternative_speed,
        available=alternative_available,
        availability_reason=alternative_reason,
        target_lane_side=target_lane_side,
        maneuver_phase=phase,
        probability=0.5,
        corridor_payload=corridor,
        head_lineage=str(head_lineage),
        feature_content_hash=feature_content_hash,
        raw_head_output_hash=raw_head_output_hash,
        metadata=alternative_metadata,
        config=cfg,
    )
    bundle = K2PredictionBundleV3(
        route_context=route_context,
        candidates=(nominal, alternative),
        observation_identity=dict(observation_identity or {}),
        model_id=model_id,
        config_hash=cfg.config_hash(),
        base_checkpoint_hash=base_checkpoint_hash,
        spatial_head_checkpoint_hash=spatial_head_checkpoint_hash,
        backbone_forward_id=backbone_forward_id,
    )
    return replace(bundle, bundle_hash=bundle.compute_bundle_hash())


__all__ = [
    "build_candidate_v3",
    "build_k2_v3_bundle",
    "lane_blend_path",
    "spatial_avoid_path",
]
