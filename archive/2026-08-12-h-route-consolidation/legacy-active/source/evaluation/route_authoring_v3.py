"""Deterministic CARLA-map route authoring for R2 V3 manoeuvres."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from driving_vla.model.navigation_contract import (
    LaneAccessV3,
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    build_route_context,
    classify_route_maneuver,
    route_heading_change_rad,
)


class RouteAuthoringError(RuntimeError):
    pass


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _signature(waypoint: Any) -> str:
    return (
        f"{int(getattr(waypoint, 'road_id', 0))}:"
        f"{int(getattr(waypoint, 'lane_id', 0))}:"
        f"{int(float(getattr(waypoint, 's', 0.0)) // 5.0)}"
    )


def _sort_key(waypoint: Any) -> tuple[int, int, float]:
    return (
        int(getattr(waypoint, "road_id", 0)),
        int(getattr(waypoint, "lane_id", 0)),
        float(getattr(waypoint, "s", 0.0)),
    )


def _same_lane(first: Any, second: Any) -> bool:
    return (
        int(getattr(first, "road_id", 0)) == int(getattr(second, "road_id", 0))
        and int(getattr(first, "lane_id", 0)) == int(getattr(second, "lane_id", 0))
    )


def _choose_successor(
    current: Any,
    options: Sequence[Any],
    maneuver: RouteManeuver,
) -> Any:
    if not options:
        raise RouteAuthoringError("route ended before required horizon")
    if len(options) == 1:
        return options[0]
    current_yaw = float(current.transform.rotation.yaw)
    scored = [
        (
            _wrap_deg(float(option.transform.rotation.yaw) - current_yaw),
            option,
        )
        for option in options
    ]
    # CARLA yaw is clockwise-positive in its left-handed XY frame.
    if maneuver in {RouteManeuver.TURN_LEFT, RouteManeuver.FOLLOW_CURVE_LEFT}:
        return min(scored, key=lambda item: (item[0], _sort_key(item[1])))[1]
    if maneuver in {RouteManeuver.TURN_RIGHT, RouteManeuver.FOLLOW_CURVE_RIGHT}:
        return max(
            scored,
            key=lambda item: (
                item[0],
                tuple(
                    -x if isinstance(x, (int, float)) else x
                    for x in _sort_key(item[1])
                ),
            ),
        )[1]
    return min(scored, key=lambda item: (abs(item[0]), _sort_key(item[1])))[1]


def _lane_access(
    origin: Any,
    side: TargetLaneSide,
    *,
    step_m: float,
    count: int,
) -> LaneAccessV3:
    target = (
        origin.get_left_lane()
        if side is TargetLaneSide.LEFT
        else origin.get_right_lane()
    )
    if target is None:
        return LaneAccessV3(side=side)
    lane_type = str(getattr(target, "lane_type", "")).lower()
    driving = "driving" in lane_type
    if not lane_type:
        driving = True
    same_direction = int(getattr(origin, "lane_id", 0)) * int(
        getattr(target, "lane_id", 0)
    ) > 0
    permission = str(getattr(origin, "lane_change", "")).lower()
    expected = "left" if side is TargetLaneSide.LEFT else "right"
    marking = (
        getattr(origin, "left_lane_marking", None)
        if side is TargetLaneSide.LEFT
        else getattr(origin, "right_lane_marking", None)
    )
    marking_type = str(getattr(marking, "type", marking) or "UNKNOWN")
    allowed = (
        expected in permission
        or "both" in permission
        or "broken" in marking_type.lower()
    )
    centerline: list[tuple[float, float]] = []
    current = target
    for _ in range(count):
        location = current.transform.location
        centerline.append((float(location.x), float(location.y)))
        options = list(current.next(step_m))
        same = [option for option in options if _same_lane(current, option)]
        # Never silently follow a junction/merge successor when the requested
        # adjacent lane ends.  That turns a route-change target into a different
        # lane and was the source of the historical right-change hold failure.
        if not same:
            break
        current = sorted(same, key=_sort_key)[0]
    return LaneAccessV3(
        side=side,
        exists=True,
        driving=driving,
        same_direction=same_direction,
        lane_change_allowed=allowed,
        currently_clear=True,
        road_id=int(getattr(target, "road_id", 0)),
        lane_id=int(getattr(target, "lane_id", 0)),
        lane_width_m=float(getattr(target, "lane_width", 0.0)),
        centerline_xy=tuple(centerline),
        marking_type=marking_type,
    )


@dataclass(frozen=True)
class AuthoredRouteV3:
    route_context: RouteContextV3
    waypoints_xyz: tuple[tuple[float, float, float], ...]
    waypoint_signatures: tuple[str, ...]
    route_length_m: float
    first_junction_entry_s_m: float | None = None
    first_junction_exit_s_m: float | None = None
    first_junction_heading_change_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        first = self.waypoints_xyz[0]
        second = self.waypoints_xyz[1]
        spawn_yaw_deg = math.degrees(
            math.atan2(second[1] - first[1], second[0] - first[0])
        )
        return {
            "route_context": self.route_context.to_dict(),
            "waypoints_xyz": [list(point) for point in self.waypoints_xyz],
            "waypoint_signatures": list(self.waypoint_signatures),
            "route_length_m": self.route_length_m,
            "first_junction_entry_s_m": self.first_junction_entry_s_m,
            "first_junction_exit_s_m": self.first_junction_exit_s_m,
            "first_junction_heading_change_deg": (
                self.first_junction_heading_change_deg
            ),
            "ego_spawn_transform": {
                "x": first[0],
                "y": first[1],
                "z": first[2] + 0.5,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": spawn_yaw_deg,
            },
        }


def navigation_waypoints_xyz(
    route: Mapping[str, Any],
) -> tuple[tuple[float, float, float], ...]:
    """Return frozen navigation XY with the source-map elevation profile.

    A route-change fixture stores both the original lane centerline and a
    continuous navigation blend in ``route_context.route_xy``.  Registry
    renderers must expose the latter to SimLingo, otherwise the task says
    "change lane" while its navigation targets remain in the original lane.
    """
    raw = tuple(
        (float(point[0]), float(point[1]), float(point[2]))
        for point in route.get("waypoints_xyz") or ()
    )
    context = dict(route.get("route_context") or {})
    navigation_xy = tuple(
        (float(point[0]), float(point[1]))
        for point in context.get("route_xy") or ()
    )
    if len(raw) < 2 or len(navigation_xy) < 2:
        raise RouteAuthoringError("route lacks navigation waypoint geometry")
    if len(raw) == len(navigation_xy):
        elevations = [point[2] for point in raw]
    else:
        elevations = []
        for index in range(len(navigation_xy)):
            alpha = index / max(1, len(navigation_xy) - 1)
            raw_index = alpha * (len(raw) - 1)
            lo = int(math.floor(raw_index))
            hi = min(len(raw) - 1, lo + 1)
            fraction = raw_index - lo
            elevations.append(
                raw[lo][2] * (1.0 - fraction) + raw[hi][2] * fraction
            )
    return tuple(
        (point[0], point[1], float(z))
        for point, z in zip(navigation_xy, elevations)
    )


def author_route_from_waypoint(
    start: Any,
    *,
    maneuver: RouteManeuver,
    step_m: float = 2.0,
    horizon_m: float = 50.0,
) -> AuthoredRouteV3:
    if bool(getattr(start, "is_junction", False)):
        raise RouteAuthoringError("route start must be before a junction")
    count = max(16, int(math.ceil(float(horizon_m) / float(step_m))) + 1)
    route = [start]
    current = start
    junction_decision_made = False
    junction_maneuver = maneuver in {
        RouteManeuver.TURN_LEFT,
        RouteManeuver.TURN_RIGHT,
        RouteManeuver.JUNCTION_STRAIGHT,
    }
    for _ in range(count - 1):
        options = list(current.next(float(step_m)))
        if not options:
            break
        same = [option for option in options if _same_lane(current, option)]
        if same and (not junction_maneuver or junction_decision_made):
            current = sorted(same, key=_sort_key)[0]
        else:
            decision = (
                RouteManeuver.FOLLOW_STRAIGHT
                if junction_decision_made
                else maneuver
            )
            current = _choose_successor(current, options, decision)
        if junction_maneuver and len(options) > 1:
            junction_decision_made = True
        route.append(current)
    if len(route) < 16:
        raise RouteAuthoringError("route contains fewer than 16 points")

    route_change_side = TargetLaneSide.NONE
    if maneuver is RouteManeuver.ROUTE_CHANGE_LEFT:
        route_change_side = TargetLaneSide.LEFT
    elif maneuver is RouteManeuver.ROUTE_CHANGE_RIGHT:
        route_change_side = TargetLaneSide.RIGHT
    left = _lane_access(
        start, TargetLaneSide.LEFT, step_m=step_m, count=len(route)
    )
    right = _lane_access(
        start, TargetLaneSide.RIGHT, step_m=step_m, count=len(route)
    )
    junction_flags = [
        bool(getattr(waypoint, "is_junction", False)) for waypoint in route
    ]
    if (
        route_change_side is not TargetLaneSide.NONE
        and any(junction_flags)
    ):
        raise RouteAuthoringError(
            f"{maneuver.value} corridor enters a junction"
        )
    first_junction_index = next(
        (index for index, flag in enumerate(junction_flags) if flag),
        None,
    )
    first_exit_index = (
        next(
            (
                index
                for index in range(int(first_junction_index) + 1, len(route))
                if not junction_flags[index]
            ),
            None,
        )
        if first_junction_index is not None
        else None
    )
    route_s = [0.0]
    for first, second in zip(route, route[1:]):
        route_s.append(
            route_s[-1]
            + math.hypot(
                float(second.transform.location.x)
                - float(first.transform.location.x),
                float(second.transform.location.y)
                - float(first.transform.location.y),
            )
        )
    origin_lane_centerline_xy = tuple(
        (
            float(waypoint.transform.location.x),
            float(waypoint.transform.location.y),
        )
        for waypoint in route
    )
    if route_change_side is not TargetLaneSide.NONE:
        route_change_heading_change_deg = math.degrees(
            route_heading_change_rad(origin_lane_centerline_xy)
        )
        if abs(route_change_heading_change_deg) > 20.0:
            raise RouteAuthoringError(
                f"{maneuver.value} corridor bends "
                f"{route_change_heading_change_deg:.1f}deg; "
                "navigation lane-change fixtures require <=20deg"
            )
    first_junction_heading_change_deg = None
    if first_junction_index is not None and first_exit_index is not None:
        local_lo = max(0, int(first_junction_index) - 2)
        local_hi = min(len(route), int(first_exit_index) + 3)
        local_route_xy = tuple(
            (
                float(waypoint.transform.location.x),
                float(waypoint.transform.location.y),
            )
            for waypoint in route[local_lo:local_hi]
        )
        first_junction_heading_change_deg = math.degrees(
            route_heading_change_rad(local_route_xy)
        )
    if (
        junction_maneuver
        and float(horizon_m) >= 80.0
        and first_junction_index is not None
    ):
        approach_m = float(route_s[first_junction_index])
        approach_min, approach_max = (
            (6.0, 10.0)
            if maneuver in {RouteManeuver.TURN_LEFT, RouteManeuver.TURN_RIGHT}
            else (12.0, 30.0)
        )
        if not approach_min <= approach_m <= approach_max:
            raise RouteAuthoringError(
                f"first junction is {approach_m:.1f}m from spawn; "
                f"{maneuver.value} requires a "
                f"{approach_min:.0f}-{approach_max:.0f}m approach"
            )
    if route_change_side is not TargetLaneSide.NONE:
        target = left if route_change_side is TargetLaneSide.LEFT else right
        if not target.authorized:
            raise RouteAuthoringError(
                f"{maneuver.value} has no authorized target lane"
            )
        if len(target.centerline_xy) < max(12, int(0.75 * len(origin_lane_centerline_xy))):
            raise RouteAuthoringError(
                f"{maneuver.value} target lane ends before the required hold corridor"
            )
        # The map route is a coarse navigation target and evaluation truth,
        # not the MPC path.  It must nevertheless be continuous: a hard
        # centerline jump gives SimLingo an impossible navigation command.
        from driving_vla.model.k2_v3_codec import lane_blend_path

        route_xy = lane_blend_path(
            origin_lane_centerline_xy,
            target.centerline_xy,
            rejoin=False,
            departure_start=0.05,
            departure_end=0.25,
            point_count=min(
                len(origin_lane_centerline_xy),
                len(target.centerline_xy),
            ),
        )
        target_road = target.road_id
        target_lane = target.lane_id
        exit_signature = f"{target_road}:{target_lane}:target"
    else:
        route_xy = tuple(
            (
                float(waypoint.transform.location.x),
                float(waypoint.transform.location.y),
            )
            for waypoint in route
        )
        # A long route can continue across several roads after the requested
        # junction.  The audited exit is the first non-junction lane after the
        # requested junction, not the route horizon endpoint.
        target_waypoint = (
            route[first_exit_index]
            if junction_maneuver and first_exit_index is not None
            else route[-1]
        )
        if junction_maneuver and first_exit_index is not None:
            # A junction exit is only a valid frozen target when the chosen
            # branch remains on that road/lane for a short post-exit corridor.
            # Without this check a route can appear to turn correctly for one
            # waypoint and then immediately merge into another road, producing
            # the historical ``JUNCTION_NOT_EXITED/WRONG_EXIT`` failures.
            exit_road = int(getattr(target_waypoint, "road_id", 0))
            exit_lane = int(getattr(target_waypoint, "lane_id", 0))
            post_exit = route[int(first_exit_index) : int(first_exit_index) + 4]
            if len(post_exit) < 3 or any(
                int(getattr(item, "road_id", 0)) != exit_road
                or int(getattr(item, "lane_id", 0)) != exit_lane
                for item in post_exit[:3]
            ):
                raise RouteAuthoringError(
                    "junction exit lacks a stable three-waypoint road/lane corridor"
                )
        target_road = int(getattr(target_waypoint, "road_id", 0))
        target_lane = int(getattr(target_waypoint, "lane_id", 0))
        exit_signature = _signature(target_waypoint)
    if junction_maneuver and first_junction_heading_change_deg is not None:
        if first_junction_heading_change_deg >= 35.0:
            classified = RouteManeuver.TURN_LEFT
        elif first_junction_heading_change_deg <= -35.0:
            classified = RouteManeuver.TURN_RIGHT
        else:
            classified = RouteManeuver.JUNCTION_STRAIGHT
    else:
        classified = classify_route_maneuver(
            route_xy,
            junction_flags=junction_flags,
            route_change_side=route_change_side,
        )
    if classified is not maneuver:
        raise RouteAuthoringError(
            f"authored route classified {classified.value}, expected {maneuver.value}"
        )
    if maneuver in {
        RouteManeuver.TURN_LEFT,
        RouteManeuver.TURN_RIGHT,
        RouteManeuver.JUNCTION_STRAIGHT,
    } and not any(junction_flags):
        raise RouteAuthoringError("junction maneuver never enters a junction")
    length = sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(route_xy, route_xy[1:])
    )
    if length < 30.0:
        raise RouteAuthoringError(f"route length {length:.1f}m is too short")
    context = build_route_context(
        route_xy,
        junction_flags=junction_flags,
        route_change_side=route_change_side,
        maneuver=maneuver,
        origin_road_id=int(getattr(start, "road_id", 0)),
        origin_lane_id=int(getattr(start, "lane_id", 0)),
        origin_lane_width_m=float(getattr(start, "lane_width", 3.5)),
        origin_lane_centerline_xy=origin_lane_centerline_xy,
        target_road_id=target_road,
        target_lane_id=target_lane,
        entry_signature=_signature(start),
        exit_signature=exit_signature,
        left_lane=left,
        right_lane=right,
        conflict_zone_id=(
            f"junction:{int(getattr(route[first_junction_index], 'junction_id', 0))}"
            if first_junction_index is not None
            else ""
        ),
    )
    waypoints_xyz = tuple(
        (
            float(route_xy[index][0]),
            float(route_xy[index][1]),
            float(waypoint.transform.location.z),
        )
        for index, waypoint in enumerate(route[: len(route_xy)])
    )
    return AuthoredRouteV3(
        route_context=context,
        waypoints_xyz=waypoints_xyz,
        waypoint_signatures=tuple(
            _signature(waypoint)
            for waypoint in route[: len(route_xy)]
        ),
        route_length_m=length,
        first_junction_entry_s_m=(
            None
            if first_junction_index is None
            else float(route_s[first_junction_index])
        ),
        first_junction_exit_s_m=(
            None
            if first_exit_index is None
            else float(route_s[first_exit_index])
        ),
        first_junction_heading_change_deg=first_junction_heading_change_deg,
    )


def find_authored_route_v3(
    world_map: Any,
    *,
    maneuver: RouteManeuver,
    used_route_hashes: Iterable[str] = (),
    horizon_m: float = 80.0,
) -> AuthoredRouteV3:
    used = set(str(value) for value in used_route_hashes)
    candidates = sorted(
        world_map.generate_waypoints(4.0),
        key=_sort_key,
    )
    failures: list[str] = []
    for waypoint in candidates:
        try:
            route = author_route_from_waypoint(
                waypoint,
                maneuver=maneuver,
                horizon_m=float(horizon_m),
            )
        except RouteAuthoringError as exc:
            if len(failures) < 5:
                failures.append(str(exc))
            continue
        if route.route_context.route_hash not in used:
            return route
    raise RouteAuthoringError(
        f"no unique {maneuver.value} route found; samples={failures}"
    )


__all__ = [
    "AuthoredRouteV3",
    "RouteAuthoringError",
    "author_route_from_waypoint",
    "find_authored_route_v3",
]
