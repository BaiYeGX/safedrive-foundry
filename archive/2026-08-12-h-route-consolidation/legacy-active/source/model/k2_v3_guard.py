"""Contract Guard for route-bound mixed-semantic K2 V3."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from driving_vla.model.frenet_codec import max_xy_separation, path_max_abs_curvature
from driving_vla.model.k2_v3_types import (
    AlternativeKind,
    K2CandidateV3,
    K2PredictionBundleV3,
    K2V3Config,
    candidate_intent_hash,
    load_k2_v3_config,
    stable_hash_t10,
    stable_hash_xy,
)
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    TrafficSignalState,
    route_heading_change_rad,
)

GUARD_OK = "OK"
GUARD_REJECT = "REJECT"


class K2V3SelectionError(RuntimeError):
    pass


def _point_to_polyline_distance(
    point: Sequence[float],
    path_xy: Sequence[Sequence[float]],
) -> float:
    px, py = float(point[0]), float(point[1])
    best = math.inf
    for first, second in zip(path_xy, path_xy[1:]):
        x0, y0 = float(first[0]), float(first[1])
        x1, y1 = float(second[0]), float(second[1])
        dx, dy = x1 - x0, y1 - y0
        denom = dx * dx + dy * dy
        if denom <= 1.0e-12:
            distance = math.hypot(px - x0, py - y0)
        else:
            alpha = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
            distance = math.hypot(px - (x0 + alpha * dx), py - (y0 + alpha * dy))
        best = min(best, distance)
    return float(best)


def _path_distance(path_xy: Sequence[Sequence[float]]) -> float:
    return sum(
        math.hypot(
            float(second[0]) - float(first[0]),
            float(second[1]) - float(first[1]),
        )
        for first, second in zip(path_xy, path_xy[1:])
    )


def _t10_progress(candidate: K2CandidateV3) -> float:
    points = candidate.points_xy_yaw_v_a_kappa
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(points, points[1:])
    )


def _mean_speed(candidate: K2CandidateV3) -> float:
    speeds = candidate.speed_samples_mps
    return sum(speeds) / len(speeds) if speeds else 0.0


def _spatial_separation(
    first: K2CandidateV3,
    second: K2CandidateV3,
) -> float:
    return max_xy_separation(first.spatial_path_xy, second.spatial_path_xy)


def _route_direction_reason(
    candidate: K2CandidateV3,
    route_context: RouteContextV3,
) -> str | None:
    expected = route_context.maneuver
    expected_delta = math.degrees(
        route_heading_change_rad(candidate.spatial_path_xy)
    )
    # A mission-level turn may still be on a straight approach. Enforce turn
    # direction only after the local mission corridor spanned by this candidate
    # has itself committed to the turn.
    local_route_delta = 0.0
    if candidate.spatial_path_xy and route_context.route_xy:
        def nearest_index(point: Sequence[float]) -> int:
            return min(
                range(len(route_context.route_xy)),
                key=lambda index: (
                    float(route_context.route_xy[index][0]) - float(point[0])
                )
                ** 2
                + (
                    float(route_context.route_xy[index][1]) - float(point[1])
                )
                ** 2,
            )

        first_index = nearest_index(candidate.spatial_path_xy[0])
        last_index = nearest_index(candidate.spatial_path_xy[-1])
        lo = max(0, min(first_index, last_index) - 1)
        hi = min(
            len(route_context.route_xy),
            max(first_index, last_index) + 2,
        )
        if hi - lo >= 2:
            local_route_delta = math.degrees(
                route_heading_change_rad(route_context.route_xy[lo:hi])
            )
    # Guard only rejects a clearly opposite local intent. Full completion is a
    # long-horizon acceptance gate, not a 2.5 s candidate-shape requirement.
    if expected in {RouteManeuver.TURN_LEFT, RouteManeuver.FOLLOW_CURVE_LEFT}:
        if local_route_delta >= 12.0 and expected_delta < -12.0:
            return "ROUTE_DIRECTION_OPPOSITE_LEFT"
    if expected in {RouteManeuver.TURN_RIGHT, RouteManeuver.FOLLOW_CURVE_RIGHT}:
        if local_route_delta <= -12.0 and expected_delta > 12.0:
            return "ROUTE_DIRECTION_OPPOSITE_RIGHT"
    if expected in {
        RouteManeuver.FOLLOW_STRAIGHT,
        RouteManeuver.JUNCTION_STRAIGHT,
    } and abs(expected_delta) > 70.0:
        return "ROUTE_DIRECTION_NOT_STRAIGHT"
    return None


def _candidate_common_reasons(
    candidate: K2CandidateV3,
    bundle: K2PredictionBundleV3,
    config: K2V3Config,
) -> list[str]:
    reasons: list[str] = []
    if candidate.route_hash != bundle.route_context.route_hash:
        reasons.append("ROUTE_HASH_MISMATCH")
    if candidate.topology_hash != bundle.route_context.topology_hash:
        reasons.append("TOPOLOGY_HASH_MISMATCH")
    if candidate.route_maneuver is not bundle.route_context.maneuver:
        reasons.append("ROUTE_MANEUVER_MISMATCH")
    if stable_hash_xy(candidate.spatial_path_xy) != candidate.spatial_path_hash:
        reasons.append("SPATIAL_PATH_HASH_MISMATCH")
    if (
        stable_hash_t10(candidate.points_xy_yaw_v_a_kappa)
        != candidate.timed_trajectory_hash
    ):
        reasons.append("TIMED_TRAJECTORY_HASH_MISMATCH")
    recomputed_intent = candidate_intent_hash(
        candidate_id=candidate.candidate_id,
        alternative_kind=candidate.alternative_kind,
        route_maneuver=candidate.route_maneuver,
        target_lane_side=candidate.target_lane_side,
        route_hash=candidate.route_hash,
        topology_hash=candidate.topology_hash,
        corridor_hash=candidate.corridor_hash,
        spatial_path_hash=candidate.spatial_path_hash,
        timed_trajectory_hash=candidate.timed_trajectory_hash,
    )
    if recomputed_intent != candidate.intent_hash:
        reasons.append("INTENT_HASH_MISMATCH")
    if len(candidate.spatial_path_xy) < 2:
        reasons.append("PATH_TOO_SHORT")
    if len(candidate.points_xy_yaw_v_a_kappa) != int(config.t_steps):
        reasons.append("T_MISMATCH")
    if candidate.spatial_path_xy:
        route_error = max(
            _point_to_polyline_distance(point, bundle.route_context.route_xy)
            for point in (
                candidate.spatial_path_xy[0],
                candidate.spatial_path_xy[-1],
            )
        )
        # Spatial overtake/route-change may legitimately use an adjacent lane
        # and is checked against the union corridor below.
        if (
            candidate.alternative_kind
            not in {AlternativeKind.SPATIAL_OVERTAKE}
            and candidate.route_maneuver
            not in {
                RouteManeuver.ROUTE_CHANGE_LEFT,
                RouteManeuver.ROUTE_CHANGE_RIGHT,
            }
            and route_error > config.route_exit_max_error_m
        ):
            reasons.append("ROUTE_CORRIDOR_EXIT_MISMATCH")
        if (
            candidate.alternative_kind
            is not AlternativeKind.SPATIAL_OVERTAKE
            and candidate.route_maneuver
            not in {
                RouteManeuver.ROUTE_CHANGE_LEFT,
                RouteManeuver.ROUTE_CHANGE_RIGHT,
            }
            # A legal junction turn necessarily leaves the approach lane
            # corridor before it reaches the registered exit.  Exit road,
            # lane, heading and MPC checks remain authoritative below and in
            # long-horizon completion; the approach-lane radius is only a
            # straight-road guard.
            and bundle.route_context.maneuver
            not in {
                RouteManeuver.JUNCTION_STRAIGHT,
                RouteManeuver.TURN_LEFT,
                RouteManeuver.TURN_RIGHT,
            }
        ):
            origin_radius = (
                0.5 * max(2.5, float(bundle.route_context.origin_lane_width_m))
                + float(config.union_corridor_margin_m)
            )
            if any(
                _point_to_polyline_distance(
                    point, bundle.route_context.route_xy
                )
                > origin_radius + 1.0e-6
                for point in candidate.spatial_path_xy
            ):
                reasons.append("ORIGIN_LANE_CORRIDOR_VIOLATION")
    direction_reason = _route_direction_reason(candidate, bundle.route_context)
    if direction_reason is not None:
        reasons.append(direction_reason)
    curvature = path_max_abs_curvature(candidate.spatial_path_xy)
    if curvature > config.max_abs_curvature + 1.0e-6:
        reasons.append("CURVATURE_ENVELOPE")
    for index, row in enumerate(candidate.points_xy_yaw_v_a_kappa):
        if len(row) < 6 or not all(math.isfinite(float(value)) for value in row[:6]):
            reasons.append(f"NON_FINITE_T10:{index}")
            break
        speed, accel, kappa = float(row[3]), float(row[4]), float(row[5])
        if speed < -1.0e-6:
            reasons.append("NEGATIVE_SPEED")
        if accel > config.max_accel_mps2 + 1.0e-6:
            reasons.append("ACCEL_ENVELOPE")
        if accel < -config.max_decel_mps2 - 1.0e-6:
            reasons.append("DECEL_ENVELOPE")
        if abs(speed * speed * kappa) > config.max_lateral_accel_mps2 + 1.0e-6:
            reasons.append("LATERAL_ACCEL_ENVELOPE")
    return reasons


def _lane_union_reasons(
    candidate: K2CandidateV3,
    bundle: K2PredictionBundleV3,
    config: K2V3Config,
) -> list[str]:
    reasons: list[str] = []
    if candidate.target_lane_side is TargetLaneSide.NONE:
        return ["TARGET_LANE_SIDE_REQUIRED"]
    lane = bundle.route_context.lane(candidate.target_lane_side)
    if not lane.exists:
        reasons.append("TARGET_LANE_MISSING")
    if not lane.driving:
        reasons.append("TARGET_LANE_NOT_DRIVING")
    if not lane.same_direction:
        reasons.append("TARGET_LANE_WRONG_DIRECTION")
    if not lane.lane_change_allowed:
        reasons.append("LANE_CHANGE_NOT_ALLOWED")
    if not lane.currently_clear:
        reasons.append("TARGET_LANE_OCCUPIED")
    if len(lane.centerline_xy) < 2:
        reasons.append("TARGET_LANE_CENTERLINE_MISSING")
        return reasons
    width = max(2.5, float(lane.lane_width_m))
    union_radius = 0.5 * width + float(config.union_corridor_margin_m)
    origin_paths = [
        bundle.route_context.route_xy,
        bundle.route_context.origin_lane_centerline_xy,
    ]
    origin_paths = [path for path in origin_paths if len(path) >= 2]
    for point in candidate.spatial_path_xy:
        distance = min(
            *(
                _point_to_polyline_distance(point, path)
                for path in origin_paths
            ),
            _point_to_polyline_distance(point, lane.centerline_xy),
        )
        if distance > union_radius + 1.0e-6:
            reasons.append("UNION_CORRIDOR_VIOLATION")
            break
    return reasons


def validate_k2_v3_bundle(
    bundle: K2PredictionBundleV3,
    config: K2V3Config | None = None,
) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    cfg = config or load_k2_v3_config()
    global_reasons: list[str] = []
    if bundle.config_hash != cfg.config_hash():
        global_reasons.append("CONFIG_HASH_MISMATCH")
    if bundle.bundle_hash != bundle.compute_bundle_hash():
        global_reasons.append("BUNDLE_HASH_MISMATCH")
    if not bundle.backbone_forward_id:
        global_reasons.append("FORWARD_ID_MISSING")

    nominal, alternative = bundle.candidates
    per_candidate: dict[str, list[str]] = {
        candidate.candidate_id: _candidate_common_reasons(
            candidate, bundle, cfg
        )
        for candidate in bundle.candidates
    }
    if nominal.alternative_kind is not AlternativeKind.NOMINAL_PROGRESS:
        per_candidate[nominal.candidate_id].append("SLOT0_NOT_NOMINAL")
    if not nominal.available:
        per_candidate[nominal.candidate_id].append("NOMINAL_UNAVAILABLE")

    if alternative.alternative_kind is AlternativeKind.NONE:
        if alternative.available:
            per_candidate[alternative.candidate_id].append(
                "NONE_MUST_BE_UNAVAILABLE"
            )
    elif not alternative.available:
        # Contract-valid unavailable proposal; it is excluded from selection.
        pass
    elif alternative.alternative_kind is AlternativeKind.TEMPORAL_YIELD:
        separation = _spatial_separation(nominal, alternative)
        speed_gap = _mean_speed(nominal) - _mean_speed(alternative)
        progress_gap = _t10_progress(nominal) - _t10_progress(alternative)
        if separation > cfg.max_temporal_shared_path_error_m + 1.0e-6:
            per_candidate[alternative.candidate_id].append(
                "TEMPORAL_YIELD_PATH_DIVERGED"
            )
        if (
            speed_gap + 1.0e-6 < cfg.min_temporal_speed_gap_mps
            and progress_gap + 1.0e-6 < cfg.min_temporal_progress_gap_m
        ):
            per_candidate[alternative.candidate_id].append(
                "TEMPORAL_YIELD_NOT_SEPARATED"
            )
    elif alternative.alternative_kind is AlternativeKind.SPATIAL_AVOID:
        if bundle.route_context.is_junction:
            per_candidate[alternative.candidate_id].append(
                "SPATIAL_AVOID_FORBIDDEN_IN_JUNCTION"
            )
        if (
            _spatial_separation(nominal, alternative) + 1.0e-6
            < cfg.min_spatial_separation_m
        ):
            per_candidate[alternative.candidate_id].append(
                "SPATIAL_ALTERNATIVE_COLLAPSE"
            )
    elif alternative.alternative_kind is AlternativeKind.SPATIAL_OVERTAKE:
        if bundle.route_context.is_junction:
            per_candidate[alternative.candidate_id].append(
                "OVERTAKE_FORBIDDEN_IN_JUNCTION"
            )
        if bundle.route_context.has_crosswalk:
            per_candidate[alternative.candidate_id].append(
                "OVERTAKE_FORBIDDEN_AT_CROSSWALK"
            )
        per_candidate[alternative.candidate_id].extend(
            _lane_union_reasons(alternative, bundle, cfg)
        )
        if (
            _spatial_separation(nominal, alternative) + 1.0e-6
            < cfg.min_spatial_separation_m
        ):
            per_candidate[alternative.candidate_id].append(
                "SPATIAL_ALTERNATIVE_COLLAPSE"
            )

    if bundle.route_context.maneuver in {
        RouteManeuver.ROUTE_CHANGE_LEFT,
        RouteManeuver.ROUTE_CHANGE_RIGHT,
    }:
        expected_side = (
            TargetLaneSide.LEFT
            if bundle.route_context.maneuver is RouteManeuver.ROUTE_CHANGE_LEFT
            else TargetLaneSide.RIGHT
        )
        if nominal.target_lane_side is not expected_side:
            per_candidate[nominal.candidate_id].append(
                "ROUTE_CHANGE_TARGET_SIDE_MISMATCH"
            )
        per_candidate[nominal.candidate_id].extend(
            _lane_union_reasons(nominal, bundle, cfg)
        )
        if bundle.route_context.is_junction:
            per_candidate[nominal.candidate_id].append(
                "ROUTE_CHANGE_FORBIDDEN_IN_JUNCTION"
            )

    signal = bundle.route_context.traffic_signal_state
    stop_distance = bundle.route_context.stop_line_distance_m
    if (
        signal in {TrafficSignalState.RED, TrafficSignalState.STOP_SIGN}
        and stop_distance is not None
    ):
        for candidate in bundle.candidates:
            if not candidate.available:
                continue
            progress = _t10_progress(candidate)
            final_speed = (
                candidate.speed_samples_mps[-1]
                if candidate.speed_samples_mps
                else math.inf
            )
            if (
                progress >= float(stop_distance) - 0.25
                and final_speed > cfg.stop_threshold_mps + 1.0e-6
            ):
                per_candidate[candidate.candidate_id].append(
                    "TRAFFIC_CONTROL_VIOLATION"
                )

    candidate_valid = {
        candidate.candidate_id: bool(
            candidate.available
            and not global_reasons
            and not per_candidate[candidate.candidate_id]
        )
        for candidate in bundle.candidates
    }
    selectable = [
        candidate.candidate_id
        for candidate in bundle.candidates
        if candidate_valid[candidate.candidate_id]
    ]
    reasons = list(global_reasons)
    for candidate_id, candidate_reasons in per_candidate.items():
        reasons.extend(f"{candidate_id}:{reason}" for reason in candidate_reasons)
    if not selectable:
        reasons.append("NO_GUARD_ACCEPTED_CANDIDATE")
    metrics = {
        "candidate_valid": candidate_valid,
        "candidate_reasons": per_candidate,
        "selectable_candidate_ids": selectable,
        "k_eff": len(selectable),
        "max_spatial_separation_m": _spatial_separation(nominal, alternative),
        "nominal_progress_m": _t10_progress(nominal),
        "alternative_progress_m": _t10_progress(alternative),
        "nominal_mean_speed_mps": _mean_speed(nominal),
        "alternative_mean_speed_mps": _mean_speed(alternative),
    }
    status = GUARD_OK if selectable else GUARD_REJECT
    return status, tuple(dict.fromkeys(reasons)), metrics


def attach_k2_v3_guard(
    bundle: K2PredictionBundleV3,
    config: K2V3Config | None = None,
) -> K2PredictionBundleV3:
    status, reasons, metrics = validate_k2_v3_bundle(bundle, config)
    return replace(
        bundle,
        guard_status=status,
        guard_reasons=reasons,
        guard_metrics=metrics,
    )


def select_k2_v3(
    bundle: K2PredictionBundleV3,
    *,
    force_index: int | None = None,
) -> K2CandidateV3:
    guarded = (
        bundle
        if bundle.guard_status in {GUARD_OK, GUARD_REJECT}
        and "candidate_valid" in bundle.guard_metrics
        else attach_k2_v3_guard(bundle)
    )
    valid = dict(guarded.guard_metrics.get("candidate_valid") or {})
    if force_index is not None:
        if int(force_index) not in (0, 1):
            raise K2V3SelectionError("force index must be 0 or 1")
        candidate = guarded.candidates[int(force_index)]
        if not bool(valid.get(candidate.candidate_id, False)):
            raise K2V3SelectionError(
                f"forced candidate rejected: {candidate.candidate_id}"
            )
        return candidate
    ordered = [guarded.top1_index, 1 - guarded.top1_index]
    for index in ordered:
        candidate = guarded.candidates[index]
        if bool(valid.get(candidate.candidate_id, False)):
            return candidate
    raise K2V3SelectionError("no Guard-accepted candidate")


__all__ = [
    "GUARD_OK",
    "GUARD_REJECT",
    "K2V3SelectionError",
    "attach_k2_v3_guard",
    "select_k2_v3",
    "validate_k2_v3_bundle",
]
