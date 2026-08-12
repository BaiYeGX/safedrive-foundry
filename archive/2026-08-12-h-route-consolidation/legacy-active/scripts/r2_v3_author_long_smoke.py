#!/usr/bin/env python3
"""Author the frozen 16-case K2 V3 long-smoke registry from CARLA topology."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.route_authoring_v3 import (  # noqa: E402
    AuthoredRouteV3,
    RouteAuthoringError,
    author_route_from_waypoint,
    navigation_waypoints_xyz,
)
from driving_vla.evaluation.r2_world_ready_v3 import (  # noqa: E402
    LONG_SMOKE_CASES,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    canonical_sha256,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402


SCHEMA = "safedrive.r2_v3.long_smoke_author.v1"
DEFAULT_REGISTRY_VERSION = "r2v3-long-smoke-teacher-v1"


def _quoted(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _matches(actual: str, requested: str) -> bool:
    return actual.endswith(requested) or f"/{requested}" in actual


def _xy_yaw(points: list[list[float]], index: int) -> tuple[float, float, float]:
    i = max(0, min(int(index), len(points) - 2))
    x0, y0 = float(points[i][0]), float(points[i][1])
    x1, y1 = float(points[i + 1][0]), float(points[i + 1][1])
    return x0, y0, math.degrees(math.atan2(y1 - y0, x1 - x0))


def _authored_to_mapping(route: AuthoredRouteV3) -> dict[str, Any]:
    return route.to_dict()


def _route_start_waypoint(world_map: Any, route: Mapping[str, Any]) -> Any:
    import carla

    spawn = dict(route["ego_spawn_transform"])
    waypoint = world_map.get_waypoint(
        carla.Location(
            x=float(spawn["x"]),
            y=float(spawn["y"]),
            z=float(spawn["z"]),
        ),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError("route start could not be projected to a driving waypoint")
    return waypoint


def _straight_variant(
    world_map: Any,
    routes: Mapping[str, Any],
    *,
    side: str,
    horizon_m: float,
) -> dict[str, Any]:
    del routes
    candidates = sorted(
        world_map.generate_waypoints(4.0),
        key=lambda waypoint: (
            int(waypoint.road_id),
            int(waypoint.lane_id),
            float(waypoint.s),
        ),
    )
    failures: list[str] = []
    for start in candidates:
        try:
            authored = author_route_from_waypoint(
                start,
                maneuver=RouteManeuver.FOLLOW_STRAIGHT,
                horizon_m=float(horizon_m),
            )
        except RouteAuthoringError as exc:
            if len(failures) < 5:
                failures.append(str(exc))
            continue
        lane = (
            authored.route_context.left_lane
            if side == "LEFT"
            else authored.route_context.right_lane
        )
        if lane.authorized:
            return _authored_to_mapping(authored)
    raise RuntimeError(
        f"no ordinary straight route with authorized {side} lane; samples={failures}"
    )


def _traffic_light_turn(
    world: Any,
    *,
    horizon_m: float,
    maneuver: RouteManeuver | None = None,
    used_route_hashes: Sequence[str] = (),
    used_entry_signatures: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, float]]:
    used_hashes = {str(value) for value in used_route_hashes}
    used_entries = {str(value) for value in used_entry_signatures}
    lights = sorted(
        list(world.get_actors().filter("traffic.traffic_light*")),
        key=lambda actor: (
            round(float(actor.get_transform().location.x), 3),
            round(float(actor.get_transform().location.y), 3),
            int(actor.id),
        ),
    )
    failures: list[str] = []
    authored_candidates: list[
        tuple[int, float, int, int, float, Any, Any]
    ] = []
    for light in lights:
        try:
            stop_waypoints = sorted(
                list(light.get_stop_waypoints()),
                key=lambda waypoint: (
                    int(waypoint.road_id),
                    int(waypoint.lane_id),
                    float(waypoint.s),
                ),
            )
        except Exception as exc:  # pragma: no cover - live API variance
            failures.append(f"light={light.id}:stop_waypoints:{exc}")
            continue
        for stop in stop_waypoints:
            # CARLA stop waypoints vary by map: some lie on the junction edge,
            # while others are several metres before it.  Probe a frozen,
            # deterministic set of upstream points instead of assuming that
            # exactly 8 m behind every stop line satisfies the 6--10 m turn
            # approach contract.
            approaches = [stop]
            cursor = stop
            for _ in range(6):
                previous = list(cursor.previous(2.0))
                if not previous:
                    break
                same_lane = [
                    waypoint
                    for waypoint in previous
                    if int(waypoint.road_id) == int(cursor.road_id)
                    and int(waypoint.lane_id) == int(cursor.lane_id)
                ]
                cursor = sorted(
                    same_lane or previous,
                    key=lambda waypoint: (
                        int(waypoint.road_id),
                        int(waypoint.lane_id),
                        float(waypoint.s),
                    ),
                )[0]
                approaches.append(cursor)
            for approach in approaches:
                # Prefer a turn so this smoke proves that stopping does not
                # erase the upstream turn intent.  Plain right turn is
                # independently verified by the same registry.
                maneuver_order = (
                    (maneuver,)
                    if maneuver is not None
                    else (
                        RouteManeuver.TURN_RIGHT,
                        RouteManeuver.JUNCTION_STRAIGHT,
                        RouteManeuver.TURN_LEFT,
                    )
                )
                for maneuver_priority, traffic_maneuver in enumerate(
                    maneuver_order
                ):
                    try:
                        route = author_route_from_waypoint(
                            approach,
                            maneuver=traffic_maneuver,
                            horizon_m=float(horizon_m),
                        )
                    except RouteAuthoringError as exc:
                        if len(failures) < 12:
                            failures.append(
                                f"light={light.id}:"
                                f"{traffic_maneuver.value}:{exc}"
                            )
                        continue
                    if (
                        route.route_context.route_hash in used_hashes
                        or route.route_context.entry_signature in used_entries
                    ):
                        continue
                    authored_candidates.append(
                        (
                            int(maneuver_priority),
                            float(
                                (route.first_junction_exit_s_m or math.inf)
                                - (route.first_junction_entry_s_m or 0.0)
                            ),
                            int(light.id),
                            int(stop.road_id),
                            float(stop.s),
                            route,
                            light,
                        )
                    )
                    break
    if authored_candidates:
        (
            _priority,
            _junction_span_m,
            _light_id,
            _road_id,
            _s,
            route,
            light,
        ) = min(authored_candidates, key=lambda item: item[:5])
        location = light.get_transform().location
        return (
            _authored_to_mapping(route),
            {
                "x": float(location.x),
                "y": float(location.y),
                "z": float(location.z),
            },
        )
    raise RuntimeError(f"no signalized route found; samples={failures}")


def _crossing_actor(world_map: Any, route: Mapping[str, Any]) -> dict[str, Any]:
    import carla

    points = list(navigation_waypoints_xyz(route))
    projected = [
        world_map.get_waypoint(
            carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2])),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        for point in points
    ]
    junction_indices = [
        index
        for index, waypoint in enumerate(projected)
        if waypoint is not None and bool(waypoint.is_junction)
    ]
    if not junction_indices:
        raise RuntimeError("crossing route does not enter a junction")
    # The physical conflict is close to the first junction entry.  Selecting
    # the median point of a large junction can put the crossing actor tens of
    # metres beyond the ego's actual path and trigger yield only after exit.
    index = min(junction_indices[-1], junction_indices[0] + 3)
    target = projected[index]
    assert target is not None
    tx = float(target.transform.location.x)
    ty = float(target.transform.location.y)
    # Classify crossing traffic relative to the ego's junction approach, not
    # the already-turned midpoint heading.  Using the midpoint made a parallel
    # actor look perpendicular on left-turn routes.
    _approach_x, _approach_y, tyaw = _xy_yaw(points, 0)
    candidates: list[tuple[float, float, int, int, Any]] = []
    for waypoint in world_map.generate_waypoints(2.0):
        if int(waypoint.road_id) == int(target.road_id):
            continue
        x = float(waypoint.transform.location.x)
        y = float(waypoint.transform.location.y)
        distance = math.hypot(x - tx, y - ty)
        delta = abs((float(waypoint.transform.rotation.yaw) - tyaw + 180.0) % 360.0 - 180.0)
        # Junction footprints vary substantially across Town maps.  Keep a
        # broad but still local perpendicular corridor; the registry dry-run
        # subsequently proves exact spawn validity and reproducibility.
        if 1.0 <= distance <= 12.0 and 40.0 <= delta <= 140.0:
            candidates.append(
                (
                    distance,
                    abs(delta - 90.0),
                    int(waypoint.road_id),
                    int(waypoint.lane_id),
                    waypoint,
                )
            )
    if not candidates:
        raise RuntimeError("no deterministic crossing actor waypoint")
    waypoint = min(candidates, key=lambda item: item[:4])[4]
    # Spawn upstream so the actor reaches the registered crossing point while
    # the ego is still on approach.  ``previous`` preserves the actor's travel
    # direction into the conflict.
    for _ in range(1):
        previous = list(waypoint.previous(2.0))
        if not previous:
            break
        same_lane = [
            item
            for item in previous
            if int(item.road_id) == int(waypoint.road_id)
            and int(item.lane_id) == int(waypoint.lane_id)
        ]
        waypoint = min(
            same_lane or previous,
            key=lambda item: (
                int(item.road_id),
                int(item.lane_id),
                float(item.s),
            ),
        )
    transform = waypoint.transform
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z) + 0.5,
        "yaw_deg": float(transform.rotation.yaw),
    }


def _adjacent_actor(
    route: Mapping[str, Any],
    *,
    side: str,
    index: int = 6,
) -> dict[str, Any]:
    context = dict(route["route_context"])
    lane = dict(context["left_lane" if side == "LEFT" else "right_lane"])
    if not (
        bool(lane.get("exists"))
        and bool(lane.get("driving"))
        and bool(lane.get("same_direction"))
    ):
        raise RuntimeError(f"{side} adjacent actor lane is not drivable/same-direction")
    centerline = list(lane.get("centerline_xy") or [])
    x, y, yaw = _xy_yaw(centerline, index)
    route_points = list(navigation_waypoints_xyz(route))
    z = float(route_points[min(index, len(route_points) - 1)][2]) + 0.5
    return {"x": x, "y": y, "z": z, "yaw_deg": yaw}


def _route_actor(route: Mapping[str, Any], *, index: int = 7) -> dict[str, Any]:
    points = list(navigation_waypoints_xyz(route))
    x, y, yaw = _xy_yaw(points, index)
    return {
        "x": x,
        "y": y,
        "z": float(points[min(index, len(points) - 1)][2]) + 0.5,
        "yaw_deg": yaw,
    }


def _pose_lines(prefix: str, pose: Mapping[str, Any]) -> list[str]:
    return [
        f"[{prefix}.transform]",
        f"x = {float(pose['x']):.9f}",
        f"y = {float(pose['y']):.9f}",
        f"z = {float(pose['z']):.9f}",
        f"roll_deg = {float(pose.get('roll_deg', 0.0)):.9f}",
        f"pitch_deg = {float(pose.get('pitch_deg', 0.0)):.9f}",
        f"yaw_deg = {float(pose['yaw_deg']):.9f}",
    ]


def _velocity_lines(prefix: str, yaw_deg: float, speed_mps: float) -> list[str]:
    yaw = math.radians(float(yaw_deg))
    return [
        f"[{prefix}.initial_velocity]",
        f"vx = {math.cos(yaw) * float(speed_mps):.9f}",
        f"vy = {math.sin(yaw) * float(speed_mps):.9f}",
        "vz = 0.0",
    ]


def _actor_script_lines(prefix: str, kind: str) -> list[str]:
    if kind == "follow":
        knots = (
            (0.0, 0.18, 0.0, 0.0),
            (1.5, 0.0, 0.85, 0.0),
            (5.5, 0.65, 0.0, 0.0),
            (13.0, 0.45, 0.0, 0.0),
        )
    elif kind == "cut_left":
        knots = (
            (0.0, 0.25, 0.0, 0.18),
            (2.5, 0.25, 0.0, 0.18),
            (4.5, 0.28, 0.0, 0.0),
            (11.0, 0.0, 0.4, 0.0),
        )
    elif kind == "cut_right":
        knots = (
            (0.0, 0.25, 0.0, -0.18),
            (2.5, 0.25, 0.0, -0.18),
            (4.5, 0.28, 0.0, 0.0),
            (11.0, 0.0, 0.4, 0.0),
        )
    elif kind in {"crossing", "merge_left", "merge_right"}:
        steer = {
            "crossing": 0.0,
            # Match the already verified cut-in convention: an actor in the
            # left adjacent lane steers right (+), and vice versa.
            "merge_left": 0.10,
            "merge_right": -0.10,
        }[kind]
        cruise_throttle = 0.55 if kind == "crossing" else 0.45
        straighten_s = 3.0 if kind == "crossing" else 2.0
        knots = (
            (0.0, 0.28, 0.0, steer),
            (straighten_s, cruise_throttle, 0.0, 0.0),
            (12.0, cruise_throttle, 0.0, 0.0),
        )
    elif kind == "obstruction":
        return [
            f"[{prefix}.script]",
            'script_type = "constant_brake"',
            "brake = 1.0",
        ]
    else:
        raise ValueError(f"unsupported actor script kind {kind}")
    lines = [f"[{prefix}.script]", 'script_type = "piecewise_vehicle_control"']
    for t_s, throttle, brake, steer in knots:
        lines.extend(
            [
                f"[[{prefix}.script.knots]]",
                f"t_s = {t_s:.3f}",
                f"throttle = {throttle:.3f}",
                f"brake = {brake:.3f}",
                f"steer = {steer:.3f}",
            ]
        )
    return lines


def _render_registry(
    *,
    map_name: str,
    cases: list[dict[str, Any]],
    duration_s: float,
    registry_version: str,
) -> str:
    lines = [
        "# Auto-authored before R2 V3 outcomes; exact spawn and simulation-time scripts.",
        "[registry]",
        'schema_version = "safedrive.g4a.scenario_registry.v1"',
        f"registry_version = {_quoted(registry_version)}",
        'description = "R2 V3 frozen 16-case teacher/contract long smoke"',
        "",
        "[defaults]",
        f"map_name = {_quoted(map_name)}",
        "sim_dt_s = 0.05",
        f"duration_s = {float(duration_s):.3f}",
        'vla_config_ref = "config/vla/k2_v3_semantic.toml"',
        'mpc_config_ref = "config/control/mpc_pid_baseline.toml"',
        'executor_config_ref = "g3_stable_vla_mpc_v3"',
        "expected_decision_anchor_time_s = 0.0",
        "",
        "[defaults.weather]",
        'preset = "ClearNoon"',
        "cloudiness = 10.0",
        "precipitation = 0.0",
        "precipitation_deposits = 0.0",
        "wind_intensity = 5.0",
        "sun_azimuth_angle = 0.0",
        "sun_altitude_angle = 70.0",
        "wetness = 0.0",
        "fog_density = 0.0",
        "fog_distance = 0.0",
        "",
        "[defaults.traffic_light]",
        'policy = "freeze_green"',
        'initial_state = "Green"',
        "",
        "[defaults.sensor_contract.front_rgb]",
        "width = 1024",
        "height = 512",
        "fov = 110.0",
        'attach = "ego"',
        'layout = "HWC_RGB_uint8"',
        "",
    ]
    for case in cases:
        scenario_id = str(case["scenario_id"])
        prefix = f"scenarios.{scenario_id}"
        route = dict(case["route"])
        context = dict(route["route_context"])
        lines.extend(
            [
                f"[{prefix}]",
                f"family = {_quoted(str(case['family']))}",
                f"map_name = {_quoted(str(case.get('map_name', map_name)))}",
                f"notes = {_quoted(str(case['notes']))}",
                "",
                f"[{prefix}.route]",
                f"identity = {_quoted(str(case['route_identity']))}",
                "target_speed_mps = 4.0",
                "waypoints = [",
            ]
        )
        for x, y, z in navigation_waypoints_xyz(route):
            lines.append(f"  [{float(x):.9f}, {float(y):.9f}, {float(z)+0.5:.9f}],")
        lines.extend(
            [
                "]",
                "",
                f"[{prefix}.route.navigation_context]",
                f"maneuver = {_quoted(str(context['maneuver']))}",
                f"entry_signature = {_quoted(str(context['entry_signature']))}",
                f"exit_signature = {_quoted(str(context['exit_signature']))}",
                f"route_hash = {_quoted(str(context['route_hash']))}",
                f"topology_hash = {_quoted(str(context['topology_hash']))}",
                (
                    "frozen_context_json = "
                    + _quoted(
                        json.dumps(
                            context,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                ),
                "",
            ]
        )
        if case.get("traffic_light"):
            traffic = dict(case["traffic_light"])
            lines.extend(
                [
                    f"[{prefix}.traffic_light]",
                    'policy = "scripted_red_green"',
                    'initial_state = "Red"',
                    f"green_after_s = {float(traffic['green_after_s']):.3f}",
                    f"target_x = {float(traffic['x']):.9f}",
                    f"target_y = {float(traffic['y']):.9f}",
                    f"target_z = {float(traffic['z']):.9f}",
                    "",
                ]
            )
        seed_id = str(case.get("seed_id") or "seed_a")
        ego_prefix = f"{prefix}.seeds.{seed_id}.ego"
        ego_pose = dict(route["ego_spawn_transform"])
        lines.extend(
            [
                f"[{ego_prefix}]",
                'name = "ego"',
                'role = "ego"',
                'blueprint = "vehicle.mercedes.coupe_2020"',
                "spawn_order = 0",
                "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                *_pose_lines(ego_prefix, ego_pose),
                *_velocity_lines(ego_prefix, float(ego_pose["yaw_deg"]), 2.0),
                f"[{ego_prefix}.script]",
                'script_type = "hold"',
                "",
            ]
        )
        actor = case.get("actor")
        if actor:
            actor = dict(actor)
            actor_prefix = f"{prefix}.seeds.{seed_id}.actors"
            pose = dict(actor["pose"])
            lines.extend(
                [
                    f"[[{actor_prefix}]]",
                    'name = "conflict_actor"',
                    'role = "npc"',
                    'blueprint = "vehicle.lincoln.mkz_2017"',
                    "spawn_order = 1",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(actor_prefix, pose),
                    *_velocity_lines(
                        actor_prefix,
                        float(pose["yaw_deg"]),
                        float(actor["speed_mps"]),
                    ),
                    *_actor_script_lines(actor_prefix, str(actor["script_kind"])),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _build_cases(
    world: Any,
    manifest: Mapping[str, Any],
    *,
    include_traffic: bool = True,
) -> list[dict[str, Any]]:
    routes = dict(manifest["routes"])
    horizon_m = float(manifest["horizon_m"])
    left_straight = _straight_variant(
        world.get_map(), routes, side="LEFT", horizon_m=horizon_m
    )
    right_straight = _straight_variant(
        world.get_map(), routes, side="RIGHT", horizon_m=horizon_m
    )
    traffic_route: dict[str, Any] | None = None
    traffic_light: dict[str, Any] | None = None
    if include_traffic:
        traffic_route, traffic_light = _traffic_light_turn(
            world,
            horizon_m=horizon_m,
            maneuver=RouteManeuver.TURN_LEFT,
        )
    selected_cases = [
        item
        for item in LONG_SMOKE_CASES
        if include_traffic or item[0] != "red_green_resume_route"
    ]
    case_routes: dict[str, dict[str, Any]] = {
        scenario_id: dict(routes[maneuver.value])
        for scenario_id, maneuver, _family in selected_cases
    }
    case_routes.update(
        {
            "cut_in_left": left_straight,
            "cut_in_right": right_straight,
            "overtake_left_rejoin": left_straight,
            "overtake_right_rejoin": right_straight,
        }
    )
    if include_traffic:
        assert traffic_route is not None
        case_routes["red_green_resume_route"] = traffic_route
    cases: list[dict[str, Any]] = []
    for scenario_id, maneuver, family in selected_cases:
        route = case_routes[scenario_id]
        case: dict[str, Any] = {
            "scenario_id": scenario_id,
            "family": family,
            "route": route,
            "route_identity": f"{manifest['map_name']}_{scenario_id}_{route['route_context']['route_hash'][:12]}",
            "notes": (
                f"maneuver={maneuver.value}; frozen_before_outcomes; "
                "ego_initial_speed=2.0mps"
            ),
        }
        if scenario_id == "follow_stop_resume":
            case["actor"] = {
                "pose": _route_actor(route, index=7),
                "speed_mps": 1.8,
                "script_kind": "follow",
            }
        elif scenario_id == "cut_in_left":
            case["actor"] = {
                "pose": _adjacent_actor(route, side="LEFT"),
                "speed_mps": 2.5,
                "script_kind": "cut_left",
            }
        elif scenario_id == "cut_in_right":
            case["actor"] = {
                "pose": _adjacent_actor(route, side="RIGHT"),
                "speed_mps": 2.5,
                "script_kind": "cut_right",
            }
        elif scenario_id == "left_turn_crossing_yield":
            case["actor"] = {
                "pose": _crossing_actor(world.get_map(), route),
                "speed_mps": 5.0,
                "script_kind": "crossing",
            }
        elif scenario_id == "right_turn_merge_yield":
            context = dict(route["route_context"])
            side = (
                "LEFT"
                if bool(dict(context["left_lane"]).get("driving"))
                and bool(dict(context["left_lane"]).get("same_direction"))
                else "RIGHT"
            )
            case["actor"] = {
                "pose": _adjacent_actor(route, side=side),
                "speed_mps": 3.5,
                "script_kind": f"merge_{side.lower()}",
            }
        elif scenario_id == "overtake_left_rejoin":
            case["actor"] = {
                "pose": _route_actor(route, index=7),
                "speed_mps": 0.0,
                "script_kind": "obstruction",
            }
        elif scenario_id == "overtake_right_rejoin":
            case["actor"] = {
                "pose": _route_actor(route, index=7),
                "speed_mps": 0.0,
                "script_kind": "obstruction",
            }
        elif scenario_id == "red_green_resume_route":
            assert traffic_light is not None
            case["traffic_light"] = {**traffic_light, "green_after_s": 3.0}
        cases.append(case)
    return cases


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument(
        "--registry-version",
        default=DEFAULT_REGISTRY_VERSION,
    )
    args = parser.parse_args()
    if not (15.0 <= float(args.duration_s) <= 20.0):
        parser.error("--duration-s must be within [15,20]")

    route_path = Path(args.route_fixture)
    manifest = json.loads(route_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "safedrive.r2_v3.route_fixtures.v1":
        raise RuntimeError("unsupported route fixture schema")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if canonical_sha256(body) != str(manifest.get("manifest_hash") or ""):
        raise RuntimeError("route fixture manifest hash mismatch")

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16")
    report = resolver.preflight()
    if report.status != "READY":
        raise RuntimeError(
            f"CARLA preflight must be READY, got {report.status}/{report.error_code}"
        )
    client, _ = resolver.connect(report=report)
    client.set_timeout(180.0)
    world = client.get_world()
    actual_map = str(world.get_map().name)
    map_name = str(manifest["map_name"])
    if not _matches(actual_map, map_name):
        raise RuntimeError(f"map mismatch: actual={actual_map}, fixture={map_name}")

    cases = _build_cases(world, manifest)
    text = _render_registry(
        map_name=map_name,
        cases=cases,
        duration_s=float(args.duration_s),
        registry_version=str(args.registry_version),
    )
    with tempfile.TemporaryDirectory(prefix="r2v3-long-smoke-") as tmp:
        validation_path = Path(tmp) / "scenario_registry.toml"
        validation_path.write_text(text, encoding="utf-8")
        registry = load_scenario_registry(validation_path)
        registry_hash = registry.compute_registry_sha256()
    output = Path(args.out)
    _write_exclusive(output, text)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "AUTHORED",
                "out": str(output),
                "map": map_name,
                "case_count": len(cases),
                "registry_hash": registry_hash,
                "route_fixture_hash": manifest["manifest_hash"],
                "outcome_used": False,
                "oracle_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
