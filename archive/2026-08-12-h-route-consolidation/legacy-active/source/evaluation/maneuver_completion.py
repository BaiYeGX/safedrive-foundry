"""Long-horizon completion gates for route and Basic-1v1 manoeuvres.

The frozen 2.5 s oracle remains unchanged.  This module answers a different
question: did the selected semantic intent complete over 15–20 seconds?
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from driving_vla.model.navigation_contract import RouteContextV3, RouteManeuver


@dataclass(frozen=True)
class ManeuverCompletionReport:
    applicable: bool
    completed: bool
    reason_codes: tuple[str, ...]
    route_progress_m: float
    peak_cross_track_m: float
    final_cross_track_m: float
    final_actor_lon_m: float | None
    final_mean_speed_mps: float
    collision: bool
    offroad: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteManeuverCompletionReport:
    maneuver: str
    completed: bool
    reason_codes: tuple[str, ...]
    route_progress_m: float
    heading_change_deg: float
    final_cross_track_m: float
    cross_track_p95_m: float
    entered_junction: bool
    exited_junction: bool
    final_road_id: int | None
    final_lane_id: int | None
    target_lane_hold_m: float
    collision: bool
    offroad: bool
    illegal_lane_invasion: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InteractionCompletionReport:
    behavior: str
    applicable: bool
    completed: bool
    reason_codes: tuple[str, ...]
    route_progress_m: float
    minimum_clearance_m: float | None
    stopped: bool
    resumed: bool
    peak_signed_cross_track_m: float
    final_cross_track_m: float
    final_mean_speed_mps: float
    collision: bool
    offroad: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrafficControlCompletionReport:
    completed: bool
    reason_codes: tuple[str, ...]
    stopped_before_line: bool
    full_stop_ticks: int
    resumed_after_release: bool
    crossed_on_prohibitive_signal: bool
    route_intent_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def route_projection(
    route_xy: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> tuple[float, float]:
    """Return nearest route arc length and unsigned cross-track distance."""
    if len(route_xy) < 2:
        return 0.0, math.inf
    best_s = 0.0
    best_d = math.inf
    acc = 0.0
    for p0, p1 in zip(route_xy, route_xy[1:]):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        seg = math.sqrt(seg2)
        if seg < 1.0e-6:
            continue
        t = max(0.0, min(1.0, ((float(x) - x0) * dx + (float(y) - y0) * dy) / seg2))
        px, py = x0 + t * dx, y0 + t * dy
        distance = math.hypot(float(x) - px, float(y) - py)
        if distance < best_d:
            best_d = distance
            best_s = acc + t * seg
        acc += seg
    return float(best_s), float(best_d)


def signed_route_projection(
    route_xy: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> tuple[float, float]:
    """Return route arc length and signed cross track (+CARLA physical right).

    The path normal ``(-dy, dx)`` is mathematical left.  CARLA's map frame is
    left-handed, so that normal points to the vehicle's physical right.
    """
    if len(route_xy) < 2:
        return 0.0, math.inf
    best_s = 0.0
    best_signed = math.inf
    best_distance = math.inf
    accumulated = 0.0
    for first, second in zip(route_xy, route_xy[1:]):
        x0, y0 = float(first[0]), float(first[1])
        x1, y1 = float(second[0]), float(second[1])
        dx, dy = x1 - x0, y1 - y0
        segment2 = dx * dx + dy * dy
        segment = math.sqrt(segment2)
        if segment < 1.0e-6:
            continue
        alpha = max(
            0.0,
            min(1.0, ((float(x) - x0) * dx + (float(y) - y0) * dy) / segment2),
        )
        projected_x = x0 + alpha * dx
        projected_y = y0 + alpha * dy
        nx, ny = -dy / segment, dx / segment
        signed = (float(x) - projected_x) * nx + (float(y) - projected_y) * ny
        distance = math.hypot(float(x) - projected_x, float(y) - projected_y)
        if distance < best_distance:
            best_distance = distance
            best_signed = signed
            best_s = accumulated + alpha * segment
        accumulated += segment
    return float(best_s), float(best_signed)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(math.ceil(float(percentile) / 100.0 * len(ordered))) - 1,
        ),
    )
    return ordered[index]


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _tail_mean_speed(ticks: Sequence[Any], count: int = 10) -> float:
    tail = list(ticks[-min(int(count), len(ticks)) :])
    return (
        sum(float(_field(tick, "ego_v", 0.0)) for tick in tail) / len(tail)
        if tail
        else 0.0
    )


def _common_trace(
    *,
    route_xy: Sequence[Sequence[float]],
    ticks: Sequence[Any],
) -> dict[str, Any]:
    projections = [
        signed_route_projection(
            route_xy,
            float(_field(tick, "ego_x", 0.0)),
            float(_field(tick, "ego_y", 0.0)),
        )
        for tick in ticks
    ]
    progress = (
        float(projections[-1][0] - projections[0][0])
        if projections
        else 0.0
    )
    collision = any(bool(_field(tick, "collision", False)) for tick in ticks)
    offroad = any(
        bool(_field(tick, "offroad", False))
        and not bool(_field(tick, "authorized_lane_crossing", False))
        for tick in ticks
    )
    illegal_lane_invasion = any(
        bool(_field(tick, "lane_invasion", False))
        and not bool(_field(tick, "authorized_lane_crossing", False))
        for tick in ticks
    )
    return {
        "projections": projections,
        "progress": progress,
        "collision": collision,
        "offroad": offroad,
        "illegal_lane_invasion": illegal_lane_invasion,
        "final_speed": _tail_mean_speed(ticks),
    }


def evaluate_route_maneuver_completion(
    *,
    route_context: RouteContextV3,
    ticks: Sequence[Any],
    min_progress_m: float = 12.0,
    min_turn_heading_deg: float = 35.0,
    max_final_cross_track_m: float = 0.60,
    max_cross_track_p95_m: float = 0.85,
    min_target_lane_hold_m: float = 8.0,
) -> RouteManeuverCompletionReport:
    """Evaluate mission compliance independently of interaction response."""
    maneuver = route_context.maneuver
    if not ticks:
        return RouteManeuverCompletionReport(
            maneuver=maneuver.value,
            completed=False,
            reason_codes=("NO_TICKS",),
            route_progress_m=0.0,
            heading_change_deg=0.0,
            final_cross_track_m=math.inf,
            cross_track_p95_m=math.inf,
            entered_junction=False,
            exited_junction=False,
            final_road_id=None,
            final_lane_id=None,
            target_lane_hold_m=0.0,
            collision=False,
            offroad=False,
            illegal_lane_invasion=False,
        )
    trace = _common_trace(route_xy=route_context.route_xy, ticks=ticks)
    projections = trace["projections"]
    # Tracking error is against the selected executable candidate path when a
    # live runner provides it.  Falling back to the coarse mission route is
    # valid for nominal-only traces, but would misclassify a legal adjacent-lane
    # overtake as poor MPC tracking.
    cross_tracks = [
        abs(
            float(
                _field(
                    tick,
                    "path_tracking_error_m",
                    projections[index][1],
                )
            )
        )
        for index, tick in enumerate(ticks)
    ]
    yaws = [
        float(_field(tick, "ego_yaw_rad", _field(tick, "ego_yaw", 0.0)))
        for tick in ticks
    ]
    # CARLA positive yaw is a physical right turn.  Public completion metrics
    # use the vehicle-centric convention: left positive, right negative.
    heading_change_deg = -math.degrees(_wrap_angle(yaws[-1] - yaws[0]))
    junction_flags = [bool(_field(tick, "is_junction", False)) for tick in ticks]
    entered_junction = any(junction_flags)
    exited_junction = any(
        not current and any(junction_flags[:index])
        for index, current in enumerate(junction_flags)
    )
    final_road_id = _field(ticks[-1], "road_id")
    final_lane_id = _field(ticks[-1], "lane_id")

    target_lane_hold_m = 0.0
    target_lane_id = route_context.target_lane_id
    if target_lane_id is not None:
        matching_indices = [
            index
            for index, tick in enumerate(ticks)
            if _field(tick, "lane_id") == target_lane_id
        ]
        if matching_indices:
            last_run_start = matching_indices[-1]
            while (
                last_run_start > 0
                and _field(ticks[last_run_start - 1], "lane_id") == target_lane_id
            ):
                last_run_start -= 1
            target_lane_hold_m = max(
                0.0,
                float(projections[-1][0] - projections[last_run_start][0]),
            )

    reasons: list[str] = []
    if trace["collision"]:
        reasons.append("COLLISION")
    if trace["offroad"]:
        reasons.append("OFFROAD")
    if trace["illegal_lane_invasion"]:
        reasons.append("ILLEGAL_LANE_INVASION")
    if trace["progress"] + 1.0e-9 < float(min_progress_m):
        reasons.append("INSUFFICIENT_ROUTE_PROGRESS")
    final_cross = cross_tracks[-1]
    cross_p95 = _percentile(cross_tracks, 95.0)
    if final_cross > float(max_final_cross_track_m):
        reasons.append("FINAL_ROUTE_ERROR")
    if cross_p95 > float(max_cross_track_p95_m):
        reasons.append("ROUTE_TRACKING_P95")

    turn_left = maneuver is RouteManeuver.TURN_LEFT
    turn_right = maneuver is RouteManeuver.TURN_RIGHT
    junction_maneuver = maneuver in {
        RouteManeuver.JUNCTION_STRAIGHT,
        RouteManeuver.TURN_LEFT,
        RouteManeuver.TURN_RIGHT,
    }
    if junction_maneuver:
        if not entered_junction:
            reasons.append("JUNCTION_NOT_ENTERED")
        if not exited_junction:
            reasons.append("JUNCTION_NOT_EXITED")
    if turn_left and heading_change_deg < float(min_turn_heading_deg):
        reasons.append("LEFT_TURN_NOT_COMPLETED")
    if turn_right and heading_change_deg > -float(min_turn_heading_deg):
        reasons.append("RIGHT_TURN_NOT_COMPLETED")
    if maneuver is RouteManeuver.JUNCTION_STRAIGHT and abs(heading_change_deg) > 30.0:
        reasons.append("WRONG_JUNCTION_EXIT_DIRECTION")
    if maneuver is RouteManeuver.FOLLOW_CURVE_LEFT and heading_change_deg < 12.0:
        reasons.append("LEFT_CURVE_NOT_COMPLETED")
    if maneuver is RouteManeuver.FOLLOW_CURVE_RIGHT and heading_change_deg > -12.0:
        reasons.append("RIGHT_CURVE_NOT_COMPLETED")

    exit_index = None
    if junction_maneuver and exited_junction:
        exit_index = next(
            (
                index
                for index, current in enumerate(junction_flags)
                if not current and any(junction_flags[:index])
            ),
            None,
        )
        if exit_index is not None:
            post_exit_progress = max(
                0.0,
                float(projections[-1][0] - projections[exit_index][0]),
            )
            if post_exit_progress + 1.0e-9 < 12.0:
                reasons.append("POST_EXIT_PROGRESS_LT_12M")

    audited_road_id = (
        _field(ticks[exit_index], "road_id")
        if junction_maneuver and exit_index is not None
        else final_road_id
    )
    audited_lane_id = (
        _field(ticks[exit_index], "lane_id")
        if junction_maneuver and exit_index is not None
        else final_lane_id
    )
    if route_context.target_road_id is not None and audited_road_id is not None:
        if int(audited_road_id) != int(route_context.target_road_id):
            reasons.append("WRONG_EXIT_ROAD")
    if route_context.target_lane_id is not None and audited_lane_id is not None:
        if int(audited_lane_id) != int(route_context.target_lane_id):
            reasons.append("WRONG_EXIT_LANE")
    signature_tick = (
        ticks[exit_index]
        if junction_maneuver and exit_index is not None
        else ticks[-1]
    )
    final_exit_signature = str(
        _field(signature_tick, "exit_signature", "") or ""
    )
    if (
        route_context.exit_signature
        and final_exit_signature
        and final_exit_signature != route_context.exit_signature
    ):
        reasons.append("WRONG_EXIT_SIGNATURE")

    for tick in ticks:
        route_hash = str(_field(tick, "route_hash", "") or "")
        topology_hash = str(_field(tick, "topology_hash", "") or "")
        maneuver_value = str(_field(tick, "route_maneuver", "") or "")
        if route_hash and route_hash != route_context.route_hash:
            reasons.append("ROUTE_HASH_CHANGED_DURING_EXECUTION")
        if topology_hash and topology_hash != route_context.topology_hash:
            reasons.append("TOPOLOGY_HASH_CHANGED_DURING_EXECUTION")
        if maneuver_value and maneuver_value != route_context.maneuver.value:
            reasons.append("ROUTE_MANEUVER_CHANGED_DURING_EXECUTION")
    if maneuver in {
        RouteManeuver.ROUTE_CHANGE_LEFT,
        RouteManeuver.ROUTE_CHANGE_RIGHT,
    } and target_lane_hold_m + 1.0e-9 < float(min_target_lane_hold_m):
        reasons.append("TARGET_LANE_NOT_HELD")

    return RouteManeuverCompletionReport(
        maneuver=maneuver.value,
        completed=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        route_progress_m=float(trace["progress"]),
        heading_change_deg=float(heading_change_deg),
        final_cross_track_m=float(final_cross),
        cross_track_p95_m=float(cross_p95),
        entered_junction=entered_junction,
        exited_junction=exited_junction,
        final_road_id=None if final_road_id is None else int(final_road_id),
        final_lane_id=None if final_lane_id is None else int(final_lane_id),
        target_lane_hold_m=float(target_lane_hold_m),
        collision=bool(trace["collision"]),
        offroad=bool(trace["offroad"]),
        illegal_lane_invasion=bool(trace["illegal_lane_invasion"]),
    )


def evaluate_follow_stop_completion(
    *,
    route_xy: Sequence[Sequence[float]],
    ticks: Sequence[Any],
    minimum_gap_m: float = 2.0,
    stop_threshold_mps: float = 0.35,
    min_stop_ticks: int = 10,
    require_resume: bool = True,
    min_resume_speed_mps: float = 0.75,
) -> InteractionCompletionReport:
    trace = _common_trace(route_xy=route_xy, ticks=ticks)
    clearances = [
        float(value)
        for tick in ticks
        if (value := _field(tick, "actor_clearance_m")) is not None
    ]
    stopped_flags = [
        float(_field(tick, "ego_v", 0.0)) <= float(stop_threshold_mps)
        for tick in ticks
    ]
    longest_stop = 0
    current = 0
    for stopped in stopped_flags:
        current = current + 1 if stopped else 0
        longest_stop = max(longest_stop, current)
    stopped = longest_stop >= int(min_stop_ticks)
    resumed = stopped and _tail_mean_speed(ticks) >= float(min_resume_speed_mps)
    reasons: list[str] = []
    if trace["collision"]:
        reasons.append("COLLISION")
    if trace["offroad"]:
        reasons.append("OFFROAD")
    if clearances and min(clearances) < float(minimum_gap_m):
        reasons.append("FOLLOW_GAP_TOO_SMALL")
    if not stopped:
        reasons.append("STABLE_STOP_NOT_OBSERVED")
    if require_resume and not resumed:
        reasons.append("DID_NOT_RESUME")
    return InteractionCompletionReport(
        behavior="FOLLOW_STOP",
        applicable=True,
        completed=not reasons,
        reason_codes=tuple(reasons),
        route_progress_m=float(trace["progress"]),
        minimum_clearance_m=min(clearances) if clearances else None,
        stopped=stopped,
        resumed=resumed,
        peak_signed_cross_track_m=max(
            (abs(item[1]) for item in trace["projections"]), default=0.0
        ),
        final_cross_track_m=abs(trace["projections"][-1][1]) if ticks else math.inf,
        final_mean_speed_mps=float(trace["final_speed"]),
        collision=bool(trace["collision"]),
        offroad=bool(trace["offroad"]),
    )


def evaluate_traffic_control_completion(
    *,
    ticks: Sequence[Any],
    stop_threshold_mps: float = 0.35,
    minimum_full_stop_ticks: int = 10,
    resume_threshold_mps: float = 0.75,
) -> TrafficControlCompletionReport:
    """Require stop-line compliance while preserving the original route task."""
    if not ticks:
        return TrafficControlCompletionReport(
            completed=False,
            reason_codes=("NO_TICKS",),
            stopped_before_line=False,
            full_stop_ticks=0,
            resumed_after_release=False,
            crossed_on_prohibitive_signal=False,
            route_intent_preserved=False,
        )
    prohibitive = {"RED", "STOP_SIGN"}
    stopped_before_line = False
    longest_stop = 0
    current_stop = 0
    release_index: int | None = None
    crossed_on_prohibitive = False
    route_hashes: set[str] = set()
    topology_hashes: set[str] = set()
    maneuvers: set[str] = set()
    had_prohibitive = False
    for index, tick in enumerate(ticks):
        signal = str(_field(tick, "traffic_signal_state", "UNKNOWN") or "UNKNOWN")
        distance = _field(tick, "stop_line_distance_m")
        speed = float(_field(tick, "ego_v", 0.0))
        is_prohibitive = signal in prohibitive
        if is_prohibitive:
            had_prohibitive = True
            if distance is not None and float(distance) < -0.05:
                crossed_on_prohibitive = True
            if (
                distance is not None
                and float(distance) >= -0.05
                and speed <= float(stop_threshold_mps)
            ):
                stopped_before_line = True
                current_stop += 1
                longest_stop = max(longest_stop, current_stop)
            else:
                current_stop = 0
        else:
            current_stop = 0
            if had_prohibitive and release_index is None:
                release_index = index
        route_hash = str(_field(tick, "route_hash", "") or "")
        topology_hash = str(_field(tick, "topology_hash", "") or "")
        maneuver = str(_field(tick, "route_maneuver", "") or "")
        if route_hash:
            route_hashes.add(route_hash)
        if topology_hash:
            topology_hashes.add(topology_hash)
        if maneuver:
            maneuvers.add(maneuver)
    resumed = (
        release_index is not None
        and any(
            float(_field(tick, "ego_v", 0.0)) >= float(resume_threshold_mps)
            for tick in ticks[release_index:]
        )
    )
    route_intent_preserved = (
        len(route_hashes) <= 1
        and len(topology_hashes) <= 1
        and len(maneuvers) <= 1
        and bool(maneuvers)
    )
    reasons: list[str] = []
    if not had_prohibitive:
        reasons.append("NO_PROHIBITIVE_CONTROL_OBSERVED")
    if not stopped_before_line:
        reasons.append("DID_NOT_STOP_BEFORE_LINE")
    if longest_stop < int(minimum_full_stop_ticks):
        reasons.append("FULL_STOP_TOO_SHORT")
    if crossed_on_prohibitive:
        reasons.append("CROSSED_ON_PROHIBITIVE_SIGNAL")
    if not resumed:
        reasons.append("DID_NOT_RESUME_AFTER_RELEASE")
    if not route_intent_preserved:
        reasons.append("ROUTE_INTENT_CHANGED_AT_CONTROL")
    return TrafficControlCompletionReport(
        completed=not reasons,
        reason_codes=tuple(reasons),
        stopped_before_line=stopped_before_line,
        full_stop_ticks=longest_stop,
        resumed_after_release=resumed,
        crossed_on_prohibitive_signal=crossed_on_prohibitive,
        route_intent_preserved=route_intent_preserved,
    )


def evaluate_yield_wait_completion(
    *,
    route_xy: Sequence[Sequence[float]],
    ticks: Sequence[Any],
    conflict_point_s_m: float,
    stop_threshold_mps: float = 0.35,
    min_resume_speed_mps: float = 0.75,
) -> InteractionCompletionReport:
    trace = _common_trace(route_xy=route_xy, ticks=ticks)
    active = [bool(_field(tick, "conflict_active", False)) for tick in ticks]
    clear_index = next(
        (
            index
            for index, value in enumerate(active)
            if not value and any(active[:index])
        ),
        None,
    )
    stopped_before = any(
        active[index]
        and float(_field(tick, "ego_v", 0.0)) <= float(stop_threshold_mps)
        and float(trace["projections"][index][0]) < float(conflict_point_s_m)
        for index, tick in enumerate(ticks)
    )
    resumed = (
        clear_index is not None
        and any(
            float(_field(tick, "ego_v", 0.0)) >= float(min_resume_speed_mps)
            for tick in ticks[clear_index:]
        )
    )
    reasons: list[str] = []
    if trace["collision"]:
        reasons.append("COLLISION")
    if trace["offroad"]:
        reasons.append("OFFROAD")
    if not stopped_before:
        reasons.append("DID_NOT_WAIT_BEFORE_CONFLICT")
    if clear_index is None:
        reasons.append("CONFLICT_NEVER_CLEARED")
    if not resumed:
        reasons.append("DID_NOT_RESUME")
    return InteractionCompletionReport(
        behavior="YIELD_WAIT",
        applicable=True,
        completed=not reasons,
        reason_codes=tuple(reasons),
        route_progress_m=float(trace["progress"]),
        minimum_clearance_m=None,
        stopped=stopped_before,
        resumed=resumed,
        peak_signed_cross_track_m=max(
            (abs(item[1]) for item in trace["projections"]), default=0.0
        ),
        final_cross_track_m=abs(trace["projections"][-1][1]) if ticks else math.inf,
        final_mean_speed_mps=float(trace["final_speed"]),
        collision=bool(trace["collision"]),
        offroad=bool(trace["offroad"]),
    )


def evaluate_cut_in_completion(
    *,
    route_xy: Sequence[Sequence[float]],
    ticks: Sequence[Any],
    conflict_side: str,
    min_away_excursion_m: float = 0.50,
    max_rejoin_error_m: float = 0.60,
    min_final_speed_mps: float = 0.75,
    allow_temporal_fallback: bool = False,
) -> InteractionCompletionReport:
    trace = _common_trace(route_xy=route_xy, ticks=ticks)
    signed = [float(item[1]) for item in trace["projections"]]
    side = str(conflict_side or "").lower()
    away_peak = (
        max(signed, default=0.0)
        if side == "left"
        else -min(signed, default=0.0)
    )
    reasons: list[str] = []
    if trace["collision"]:
        reasons.append("COLLISION")
    if trace["offroad"]:
        reasons.append("OFFROAD")
    stopped_before = False
    resumed_after_clear = False
    if allow_temporal_fallback:
        stop_run = 0
        for tick in ticks:
            active = bool(_field(tick, "conflict_active", False))
            speed = float(_field(tick, "ego_v", 0.0))
            if active and speed <= 0.35:
                stop_run += 1
                if stop_run >= 10:
                    stopped_before = True
            else:
                stop_run = 0
            if stopped_before and not active and speed >= float(min_final_speed_mps):
                resumed_after_clear = True
        if not stopped_before:
            reasons.append("DID_NOT_WAIT_BEFORE_CONFLICT")
        if not resumed_after_clear:
            reasons.append("DID_NOT_RESUME")
    elif away_peak + 1.0e-9 < float(min_away_excursion_m):
        reasons.append("DID_NOT_MOVE_AWAY_FROM_CONFLICT")
    final_cross = abs(signed[-1]) if signed else math.inf
    if final_cross > float(max_rejoin_error_m):
        reasons.append("DID_NOT_REJOIN")
    if not allow_temporal_fallback and trace["final_speed"] < float(min_final_speed_mps):
        reasons.append("STOPPED_OR_CRAWLING")
    return InteractionCompletionReport(
        behavior="CUT_IN_AVOID",
        applicable=side in {"left", "right"},
        completed=side in {"left", "right"} and not reasons,
        reason_codes=tuple(reasons if side in {"left", "right"} else ["UNKNOWN_CONFLICT_SIDE"]),
        route_progress_m=float(trace["progress"]),
        minimum_clearance_m=None,
        stopped=stopped_before,
        resumed=(
            resumed_after_clear
            if allow_temporal_fallback
            else float(trace["final_speed"]) >= float(min_final_speed_mps)
        ),
        peak_signed_cross_track_m=float(away_peak),
        final_cross_track_m=float(final_cross),
        final_mean_speed_mps=float(trace["final_speed"]),
        collision=bool(trace["collision"]),
        offroad=bool(trace["offroad"]),
    )


def evaluate_overtake_completion(
    *,
    family: str,
    route_xy: Sequence[Sequence[float]],
    ticks: Sequence[Any],
    final_actor_lon_m: float | None,
    min_excursion_m: float = 0.50,
    max_rejoin_error_m: float = 0.60,
    pass_margin_m: float = 2.0,
    min_progress_m: float = 8.0,
    min_final_mean_speed_mps: float = 0.75,
) -> ManeuverCompletionReport:
    """Require obstruction avoidance to pass, move, and rejoin.

    Yield/crossing/lead-brake cases are intentionally not judged as overtakes:
    stopping can be the correct behaviour there.
    """
    fam = str(family or "").lower()
    applicable = "obstruction" in fam or "narrow" in fam
    if not ticks:
        return ManeuverCompletionReport(
            applicable=applicable,
            completed=False,
            reason_codes=("NO_TICKS",),
            route_progress_m=0.0,
            peak_cross_track_m=0.0,
            final_cross_track_m=math.inf,
            final_actor_lon_m=final_actor_lon_m,
            final_mean_speed_mps=0.0,
            collision=False,
            offroad=False,
        )
    projections = [
        route_projection(
            route_xy,
            float(_field(tick, "ego_x", 0.0)),
            float(_field(tick, "ego_y", 0.0)),
        )
        for tick in ticks
    ]
    progress = float(projections[-1][0] - projections[0][0])
    peak_cross = max(float(item[1]) for item in projections)
    final_cross = float(projections[-1][1])
    tail = list(ticks[-min(10, len(ticks)) :])
    final_mean_v = sum(float(_field(t, "ego_v", 0.0)) for t in tail) / len(tail)
    collision = any(bool(_field(t, "collision", False)) for t in ticks)
    offroad = any(
        bool(_field(t, "offroad", False))
        and not bool(_field(t, "authorized_lane_crossing", False))
        for t in ticks
    )
    reasons: list[str] = []
    if not applicable:
        reasons.append("NOT_OVERTAKE_FAMILY")
    else:
        if collision:
            reasons.append("COLLISION")
        if offroad:
            reasons.append("OFFROAD")
        if peak_cross + 1.0e-9 < float(min_excursion_m):
            reasons.append("NO_LATERAL_EXCURSION")
        if final_cross > float(max_rejoin_error_m):
            reasons.append("DID_NOT_REJOIN")
        if final_actor_lon_m is None or float(final_actor_lon_m) > -float(pass_margin_m):
            reasons.append("DID_NOT_PASS_ACTOR")
        if progress + 1.0e-9 < float(min_progress_m):
            reasons.append("INSUFFICIENT_PROGRESS")
        if final_mean_v + 1.0e-9 < float(min_final_mean_speed_mps):
            reasons.append("STOPPED_OR_CRAWLING")
    return ManeuverCompletionReport(
        applicable=applicable,
        completed=applicable and not reasons,
        reason_codes=tuple(reasons),
        route_progress_m=progress,
        peak_cross_track_m=peak_cross,
        final_cross_track_m=final_cross,
        final_actor_lon_m=final_actor_lon_m,
        final_mean_speed_mps=final_mean_v,
        collision=collision,
        offroad=offroad,
    )
