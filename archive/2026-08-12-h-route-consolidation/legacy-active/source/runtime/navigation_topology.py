"""Current-observation-only CARLA topology adaptor for K2 V3."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable, Sequence

from driving_vla.model.navigation_contract import (
    LaneAccessV3,
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    TrafficSignalState,
    build_route_context,
)


def _location_xy(value: Any) -> tuple[float, float]:
    location = value
    if hasattr(value, "get_location"):
        location = value.get_location()
    elif hasattr(value, "location"):
        location = value.location
    return float(location.x), float(location.y)


def observe_traffic_control_v3(
    ego: Any,
) -> tuple[TrafficSignalState, float | None]:
    """Read the currently associated CARLA signal and trigger-line distance.

    This is an Observable runtime input.  It uses only the current traffic
    light association/state and trigger volume; no future phase timing is read.
    """
    try:
        light = ego.get_traffic_light()
    except Exception:  # noqa: BLE001
        light = None
    stop_line_distance: float | None = None
    if light is None:
        # CARLA associates get_traffic_light() only after the ego enters a
        # light's influence volume.  Guard must see a route-relevant red light
        # early enough to stop, so scan current signal/stop-waypoint state on
        # the ego's present road and lane (no future phase timing).
        try:
            world = ego.get_world()
            world_map = world.get_map()
            ego_location = ego.get_location()
            ego_waypoint = world_map.get_waypoint(ego_location)
            ego_transform = ego.get_transform()
            yaw = math.radians(float(ego_transform.rotation.yaw))
            fx, fy = math.cos(yaw), math.sin(yaw)
            matches: list[tuple[float, int, Any]] = []
            for candidate in world.get_actors().filter("traffic.traffic_light*"):
                for stop in candidate.get_stop_waypoints():
                    if (
                        int(stop.road_id) != int(ego_waypoint.road_id)
                        or int(stop.lane_id) != int(ego_waypoint.lane_id)
                    ):
                        continue
                    location = stop.transform.location
                    dx = float(location.x - ego_location.x)
                    dy = float(location.y - ego_location.y)
                    longitudinal = dx * fx + dy * fy
                    distance = math.hypot(dx, dy)
                    if longitudinal >= -0.5 and distance <= 40.0:
                        matches.append(
                            (float(distance), int(candidate.id), candidate)
                        )
            if matches:
                stop_line_distance, _actor_id, light = min(matches)
        except Exception:  # noqa: BLE001
            light = None
    if light is None:
        return TrafficSignalState.UNKNOWN, None
    try:
        raw_state = light.get_state()
        name = str(getattr(raw_state, "name", raw_state)).rsplit(".", 1)[-1]
        state = {
            "green": TrafficSignalState.GREEN,
            "yellow": TrafficSignalState.YELLOW,
            "red": TrafficSignalState.RED,
        }.get(name.lower(), TrafficSignalState.UNKNOWN)
    except Exception:  # noqa: BLE001
        state = TrafficSignalState.UNKNOWN
    if stop_line_distance is not None:
        return state, float(stop_line_distance)
    try:
        transform = light.get_transform()
        trigger = light.trigger_volume
        local = trigger.location
        yaw = math.radians(float(transform.rotation.yaw))
        center_x = (
            float(transform.location.x)
            + math.cos(yaw) * float(local.x)
            - math.sin(yaw) * float(local.y)
        )
        center_y = (
            float(transform.location.y)
            + math.sin(yaw) * float(local.x)
            + math.cos(yaw) * float(local.y)
        )
        ego_x, ego_y = _location_xy(ego)
        half_extent = max(
            float(getattr(trigger.extent, "x", 0.0)),
            float(getattr(trigger.extent, "y", 0.0)),
        )
        distance = max(0.0, math.hypot(center_x - ego_x, center_y - ego_y) - half_extent)
    except Exception:  # noqa: BLE001
        distance = None
    return state, distance


def _same_direction(first: Any, second: Any) -> bool:
    first_lane = int(getattr(first, "lane_id", 0))
    second_lane = int(getattr(second, "lane_id", 0))
    if first_lane and second_lane:
        return first_lane * second_lane > 0
    first_yaw = math.radians(float(first.transform.rotation.yaw))
    second_yaw = math.radians(float(second.transform.rotation.yaw))
    return math.cos(first_yaw - second_yaw) > 0.5


def _is_driving_lane(waypoint: Any) -> bool:
    lane_type = getattr(waypoint, "lane_type", None)
    if lane_type is None:
        return True
    text = str(lane_type).lower()
    if "driving" in text:
        return True
    try:
        # CARLA LaneType.Driving is bit 1 in supported releases.
        return bool(int(lane_type) & 1)
    except (TypeError, ValueError):
        return False


def _lane_change_allowed(waypoint: Any, side: TargetLaneSide) -> bool:
    permission = str(getattr(waypoint, "lane_change", "")).lower()
    wanted = "left" if side is TargetLaneSide.LEFT else "right"
    if wanted in permission or "both" in permission:
        return True
    marking = (
        getattr(waypoint, "left_lane_marking", None)
        if side is TargetLaneSide.LEFT
        else getattr(waypoint, "right_lane_marking", None)
    )
    marking_type = str(getattr(marking, "type", marking)).lower()
    return "broken" in marking_type


def _lane_marking_type(waypoint: Any, side: TargetLaneSide) -> str:
    marking = (
        getattr(waypoint, "left_lane_marking", None)
        if side is TargetLaneSide.LEFT
        else getattr(waypoint, "right_lane_marking", None)
    )
    return str(getattr(marking, "type", marking) or "UNKNOWN")


def _centerline(waypoint: Any, *, step_m: float = 2.0, count: int = 24):
    output: list[tuple[float, float]] = []
    current = waypoint
    for _ in range(max(2, int(count))):
        location = current.transform.location
        output.append((float(location.x), float(location.y)))
        if not hasattr(current, "next"):
            break
        following = list(current.next(float(step_m)))
        if not following:
            break
        # Deterministic and mission-blind: stay on the same lane when possible.
        same = [
            item
            for item in following
            if int(getattr(item, "lane_id", 0))
            == int(getattr(current, "lane_id", 0))
        ]
        current = sorted(
            same or following,
            key=lambda item: (
                int(getattr(item, "road_id", 0)),
                int(getattr(item, "lane_id", 0)),
            ),
        )[0]
    return tuple(output)


def _lane_currently_clear(
    *,
    world_map: Any,
    ego_waypoint: Any,
    target_waypoint: Any,
    actors: Iterable[Any],
    ego_actor_id: int | None,
    clear_ahead_m: float,
    clear_behind_m: float,
) -> bool:
    tx = math.cos(math.radians(float(target_waypoint.transform.rotation.yaw)))
    ty = math.sin(math.radians(float(target_waypoint.transform.rotation.yaw)))
    ex, ey = _location_xy(ego_waypoint.transform)
    for actor in actors:
        if ego_actor_id is not None and int(getattr(actor, "id", -1)) == ego_actor_id:
            continue
        try:
            actor_location = actor.get_location()
            actor_waypoint = world_map.get_waypoint(
                actor_location, project_to_road=True
            )
            if (
                int(getattr(actor_waypoint, "road_id", 0))
                != int(getattr(target_waypoint, "road_id", 0))
                or int(getattr(actor_waypoint, "lane_id", 0))
                != int(getattr(target_waypoint, "lane_id", 0))
            ):
                continue
            ax, ay = _location_xy(actor)
        except Exception:  # noqa: BLE001
            continue
        longitudinal = (ax - ex) * tx + (ay - ey) * ty
        if -float(clear_behind_m) <= longitudinal <= float(clear_ahead_m):
            return False
    return True


def _adjacent_access(
    *,
    world_map: Any,
    ego_waypoint: Any,
    side: TargetLaneSide,
    actors: Iterable[Any],
    ego_actor_id: int | None,
) -> LaneAccessV3:
    target = (
        ego_waypoint.get_left_lane()
        if side is TargetLaneSide.LEFT
        else ego_waypoint.get_right_lane()
    )
    if target is None:
        return LaneAccessV3(side=side)
    return LaneAccessV3(
        side=side,
        exists=True,
        driving=_is_driving_lane(target),
        same_direction=_same_direction(ego_waypoint, target),
        lane_change_allowed=_lane_change_allowed(ego_waypoint, side),
        currently_clear=_lane_currently_clear(
            world_map=world_map,
            ego_waypoint=ego_waypoint,
            target_waypoint=target,
            actors=actors,
            ego_actor_id=ego_actor_id,
            clear_ahead_m=30.0,
            clear_behind_m=12.0,
        ),
        road_id=int(getattr(target, "road_id", 0)),
        lane_id=int(getattr(target, "lane_id", 0)),
        lane_width_m=float(getattr(target, "lane_width", 0.0)),
        centerline_xy=_centerline(target),
        marking_type=_lane_marking_type(ego_waypoint, side),
    )


def observe_route_context_v3(
    *,
    world_map: Any,
    ego: Any,
    route_xy: Sequence[Sequence[float]],
    actors: Iterable[Any] = (),
    route_change_side: TargetLaneSide = TargetLaneSide.NONE,
    traffic_signal_state: TrafficSignalState = TrafficSignalState.UNKNOWN,
    stop_line_distance_m: float | None = None,
    has_crosswalk: bool = False,
    conflict_zone_id: str = "",
    explicit_maneuver: RouteManeuver | None = None,
) -> RouteContextV3:
    """Capture current topology without actor future, candidate, or Oracle data."""
    ego_location = ego.get_location() if hasattr(ego, "get_location") else ego.location
    ego_waypoint = world_map.get_waypoint(ego_location, project_to_road=True)
    if ego_waypoint is None:
        raise RuntimeError("ego has no driving waypoint")
    route_waypoints = [
        world_map.get_waypoint(
            type(ego_location)(
                x=float(point[0]),
                y=float(point[1]),
                z=float(getattr(ego_location, "z", 0.0)),
            ),
            project_to_road=True,
        )
        for point in route_xy
    ]
    junction_flags = [
        bool(getattr(waypoint, "is_junction", False))
        for waypoint in route_waypoints
        if waypoint is not None
    ]
    target_waypoint = next(
        (item for item in reversed(route_waypoints) if item is not None),
        ego_waypoint,
    )
    actor_list = tuple(actors)
    context = build_route_context(
        route_xy,
        junction_flags=junction_flags,
        route_change_side=route_change_side,
        maneuver=explicit_maneuver,
        origin_road_id=int(getattr(ego_waypoint, "road_id", 0)),
        origin_lane_id=int(getattr(ego_waypoint, "lane_id", 0)),
        origin_lane_width_m=float(getattr(ego_waypoint, "lane_width", 3.5)),
        origin_lane_centerline_xy=_centerline(
            ego_waypoint,
            count=max(24, len(route_xy)),
        ),
        target_road_id=int(getattr(target_waypoint, "road_id", 0)),
        target_lane_id=int(getattr(target_waypoint, "lane_id", 0)),
        entry_signature=(
            f"{int(getattr(ego_waypoint, 'road_id', 0))}:"
            f"{int(getattr(ego_waypoint, 'lane_id', 0))}"
        ),
        exit_signature=(
            f"{int(getattr(target_waypoint, 'road_id', 0))}:"
            f"{int(getattr(target_waypoint, 'lane_id', 0))}"
        ),
        has_crosswalk=bool(has_crosswalk),
        stop_line_distance_m=stop_line_distance_m,
        traffic_signal_state=traffic_signal_state,
        left_lane=_adjacent_access(
            world_map=world_map,
            ego_waypoint=ego_waypoint,
            side=TargetLaneSide.LEFT,
            actors=actor_list,
            ego_actor_id=getattr(ego, "id", None),
        ),
        right_lane=_adjacent_access(
            world_map=world_map,
            ego_waypoint=ego_waypoint,
            side=TargetLaneSide.RIGHT,
            actors=actor_list,
            ego_actor_id=getattr(ego, "id", None),
        ),
        conflict_zone_id=conflict_zone_id,
    )
    # Runtime Guard rules (no overtake/avoid/route change in a junction) need
    # the ego's current state, not "does any future point in this long mission
    # route cross a junction".
    return replace(
        context,
        is_junction=bool(getattr(ego_waypoint, "is_junction", False)),
        topology_hash="",
    )


__all__ = ["observe_route_context_v3", "observe_traffic_control_v3"]
