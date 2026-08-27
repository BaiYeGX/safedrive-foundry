"""Deterministic physical materialization for the frozen H2 CARLA matrix.

This module contains scenario geometry and scripts only. It does not import the
offline Oracle, future labels or any H3 implementation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    import carla
except ImportError:
    carla = None

from .contracts import ScenarioKey, stable_sha256
from .live_contract import SCENARIO_ALGORITHM_VERSION, route_projection
from .matrix import MatrixEntry


WEATHER_PRESETS: dict[str, dict[str, float]] = {
    "ClearNoon": {
        "cloudiness": 5.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 75.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 0.0,
    },
    "CloudyNoon": {
        "cloudiness": 80.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 10.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 75.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 0.0,
    },
}


def transform_to_dict(transform: Any) -> dict[str, float]:
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "pitch": float(transform.rotation.pitch),
        "yaw": float(transform.rotation.yaw),
        "roll": float(transform.rotation.roll),
    }


def transform_from_dict(payload: Mapping[str, Any]) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=float(payload["x"]), y=float(payload["y"]), z=float(payload["z"])),
        carla.Rotation(
            pitch=float(payload.get("pitch", 0.0)),
            yaw=float(payload.get("yaw", 0.0)),
            roll=float(payload.get("roll", 0.0)),
        ),
    )


@dataclass(frozen=True)
class PhysicalScenario:
    scenario: ScenarioKey
    matrix_index: int
    expert_slot: int
    branch_order: tuple[str, str]
    ego_transform: Mapping[str, float]
    route: tuple[tuple[float, float], ...]
    npc_actors: tuple[Mapping[str, Any], ...]
    weather: Mapping[str, float]
    script: Mapping[str, Any]
    red_light: Mapping[str, Any] | None = None
    algorithm_version: str = SCENARIO_ALGORITHM_VERSION

    @property
    def pair_id(self) -> str:
        return self.scenario.pair_id

    @property
    def physical_sha256(self) -> str:
        return stable_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "scenario": self.scenario.to_dict(),
            "pair_id": self.pair_id,
            "matrix_index": self.matrix_index,
            "expert_slot": self.expert_slot,
            "branch_order": list(self.branch_order),
            "ego_transform": dict(self.ego_transform),
            "route": [list(point) for point in self.route],
            "npc_actors": [dict(actor) for actor in self.npc_actors],
            "weather": dict(self.weather),
            "script": dict(self.script),
            "red_light": None if self.red_light is None else dict(self.red_light),
            "algorithm_version": self.algorithm_version,
        }
        if include_hash:
            payload["physical_sha256"] = stable_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalScenario":
        scenario = payload["scenario"]
        result = cls(
            scenario=ScenarioKey(
                str(scenario["map_name"]), str(scenario["family"]), int(scenario["seed"]), str(scenario["weather"])
            ),
            matrix_index=int(payload["matrix_index"]),
            expert_slot=int(payload["expert_slot"]),
            branch_order=tuple(payload["branch_order"]),  # type: ignore[arg-type]
            ego_transform=dict(payload["ego_transform"]),
            route=tuple((float(point[0]), float(point[1])) for point in payload["route"]),
            npc_actors=tuple(dict(actor) for actor in payload.get("npc_actors", [])),
            weather=dict(payload["weather"]),
            script=dict(payload["script"]),
            red_light=None if payload.get("red_light") is None else dict(payload["red_light"]),
            algorithm_version=str(payload["algorithm_version"]),
        )
        if payload.get("physical_sha256") != result.physical_sha256:
            raise ValueError(f"physical scenario hash mismatch: {result.pair_id}")
        return result


def _next_waypoint(current: Any, distance_m: float) -> Any | None:
    options = list(current.next(distance_m))
    options.sort(
        key=lambda waypoint: (
            0 if int(waypoint.road_id) == int(current.road_id) else 1,
            0 if int(waypoint.lane_id) == int(current.lane_id) else 1,
            abs(((float(waypoint.transform.rotation.yaw) - float(current.transform.rotation.yaw) + 180.0) % 360.0) - 180.0),
            int(waypoint.id),
        )
    )
    return options[0] if options else None


def _previous_waypoint(current: Any, distance_m: float) -> Any | None:
    options = list(current.previous(distance_m))
    options.sort(key=lambda waypoint: (int(waypoint.road_id), int(waypoint.lane_id), int(waypoint.id)))
    return options[0] if options else None


def _route_from_waypoint(start: Any, *, points: int = 51, spacing_m: float = 2.0) -> tuple[Any, ...]:
    route = [start]
    seen = {int(start.id)}
    current = start
    for _ in range(points - 1):
        chosen = _next_waypoint(current, spacing_m)
        if chosen is None or int(chosen.id) in seen:
            break
        route.append(chosen)
        seen.add(int(chosen.id))
        current = chosen
    return tuple(route)


def _route_length(route: Sequence[Any]) -> float:
    return sum(
        math.hypot(
            float(route[index].transform.location.x - route[index - 1].transform.location.x),
            float(route[index].transform.location.y - route[index - 1].transform.location.y),
        )
        for index in range(1, len(route))
    )


def _raised_transform(transform: Any, z_m: float = 0.25) -> carla.Transform:
    payload = transform_to_dict(transform)
    payload["z"] += z_m
    return transform_from_dict(payload)


def _candidate_routes(carla_map: Any) -> list[tuple[Any, tuple[Any, ...]]]:
    spawns = sorted(
        carla_map.get_spawn_points(),
        key=lambda transform: (
            round(float(transform.location.x), 3), round(float(transform.location.y), 3),
            round(float(transform.rotation.yaw), 3),
        ),
    )
    output: list[tuple[Any, tuple[Any, ...]]] = []
    for spawn in spawns:
        waypoint = carla_map.get_waypoint(spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            continue
        route = _route_from_waypoint(waypoint)
        if len(route) >= 41 and _route_length(route) >= 80.0:
            output.append((_raised_transform(waypoint.transform), route))
    if not output:
        raise RuntimeError("no deterministic 80m driving routes")
    return output


def _adjacent_cutter_lane(waypoint: Any) -> tuple[Any | None, str]:
    """Find a deterministic lateral lane/shoulder from which a cutter enters.

    Town01's packaged roads have one driving lane per direction and a marked
    shoulder, rather than two same-direction driving lanes.  The H2 cut-in
    family remains physically realizable there by spawning the cutter on that
    adjacent shoulder and steering it into the ego lane.  On multi-lane towns,
    a same-direction driving lane remains preferred.
    """

    options = (("left", waypoint.get_left_lane()), ("right", waypoint.get_right_lane()))
    ranked: list[tuple[int, str, Any]] = []
    for name, lane in options:
        if lane is None or int(lane.road_id) != int(waypoint.road_id):
            continue
        lane_type = lane.lane_type
        if lane_type == carla.LaneType.Driving and int(lane.lane_id) * int(waypoint.lane_id) > 0:
            ranked.append((0, name, lane))
        elif lane_type == carla.LaneType.Shoulder:
            ranked.append((1, name, lane))
    if ranked:
        _, name, lane = sorted(ranked, key=lambda item: (item[0], item[1]))[0]
        return lane, name
    return None, "none"


def _probe_cutter_spawn(world: Any, transform: Any) -> bool:
    """Probe a cut-in transform and synchronously destroy the temporary actor.

    Materialization is deterministic and pre-capture; the probe only removes
    CARLA spawn ambiguity on maps such as Town01 and never consults outcomes.
    Test doubles without a blueprint API retain the geometry-only behavior.
    """
    library = getattr(world, "get_blueprint_library", None)
    try_spawn = getattr(world, "try_spawn_actor", None)
    if not callable(library) or not callable(try_spawn):
        return True
    blueprints = library()
    find = getattr(blueprints, "find", None)
    if not callable(find):
        return True
    blueprint = find("vehicle.lincoln.mkz_2020")
    actor = try_spawn(blueprint, transform)
    if actor is None:
        return False
    actor_id = int(getattr(actor, "id", -1))
    destroy = getattr(actor, "destroy", None)
    if callable(destroy):
        destroy()
    wait_for_tick = getattr(world, "wait_for_tick", None)
    if callable(wait_for_tick):
        wait_for_tick(seconds=1.0)
    def _remaining() -> Any | None:
        get_actors = getattr(world, "get_actors", None)
        if not callable(get_actors) or actor_id < 0:
            return None
        return next((item for item in get_actors() if int(getattr(item, "id", -1)) == actor_id), None)

    remaining = _remaining()
    for _ in range(2):
        if remaining is None:
            break
        retry_destroy = getattr(remaining, "destroy", None)
        if callable(retry_destroy):
            retry_destroy()
        if callable(wait_for_tick):
            wait_for_tick(seconds=1.0)
        remaining = _remaining()
    # Actor disappearance, rather than destroy()'s asynchronous boolean, is
    # the authoritative cleanup barrier.
    return remaining is None


def _red_route(world: Any, carla_map: Any, key: ScenarioKey) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    lights = sorted(
        (actor for actor in world.get_actors() if str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light")),
        key=lambda actor: (
            round(float(actor.get_transform().location.x), 3),
            round(float(actor.get_transform().location.y), 3),
            int(actor.id),
        ),
    )
    choices: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
    for light in lights:
        relative = light.trigger_volume.location
        trigger = carla.Location(x=float(relative.x), y=float(relative.y), z=float(relative.z))
        transformed = light.get_transform().transform(trigger)
        if transformed is not None:  # CARLA bindings differ on in-place vs return semantics.
            trigger = transformed
        stop_waypoint = carla_map.get_waypoint(trigger, project_to_road=True, lane_type=carla.LaneType.Driving)
        if stop_waypoint is None:
            continue
        start = stop_waypoint
        for _ in range(10):
            previous = _previous_waypoint(start, 2.0)
            if previous is None:
                break
            start = previous
        route = _route_from_waypoint(start)
        if len(route) < 31 or _route_length(route) < 60.0:
            continue
        xy = tuple((float(wp.transform.location.x), float(wp.transform.location.y)) for wp in route)
        stop_s, distance = route_projection(float(trigger.x), float(trigger.y), xy)
        if not (8.0 <= stop_s <= 35.0 and distance <= 8.0):
            continue
        choices.append(
            (
                _raised_transform(start.transform),
                route,
                {
                    "trigger_x": float(trigger.x),
                    "trigger_y": float(trigger.y),
                    "stop_progress_m": float(stop_s),
                },
            )
        )
    if not choices:
        raise RuntimeError("no deterministic red-light approach route")
    index = int(hashlib.sha256(key.pair_id.encode()).hexdigest(), 16) % len(choices)
    return choices[index]


def materialize_physical_scenario(world: Any, entry: MatrixEntry) -> PhysicalScenario:
    key = entry.scenario
    if str(world.get_map().name).split("/")[-1].replace("_Opt", "") != key.map_name:
        raise RuntimeError(f"materialization map mismatch for {key.pair_id}")
    carla_map = world.get_map()
    red_light: dict[str, Any] | None = None
    if key.family == "red_light_hold":
        ego_transform, waypoints, red_light = _red_route(world, carla_map, key)
    else:
        candidates = _candidate_routes(carla_map)
        ranked = sorted(candidates, key=lambda item: stable_sha256(transform_to_dict(item[0])))
        start_index = int(hashlib.sha256(key.pair_id.encode()).hexdigest(), 16) % len(ranked)
        ego_transform = None
        waypoints = ()
        selected_cutter: tuple[Any, str] | None = None
        for offset in range(len(ranked)):
            spawn, candidate_route = ranked[(start_index + offset) % len(ranked)]
            if key.family == "cut_in":
                probe_result: tuple[Any, str] | None = None
                for route_index in (8, 9, 7, 10, 6):
                    if route_index >= len(candidate_route):
                        continue
                    adjacent, side = _adjacent_cutter_lane(candidate_route[route_index])
                    if adjacent is None:
                        continue
                    if _probe_cutter_spawn(world, _raised_transform(adjacent.transform)):
                        probe_result = (adjacent, side)
                        break
                if probe_result is None:
                    continue
                selected_cutter = probe_result
            ego_transform, waypoints = spawn, candidate_route
            break
        if ego_transform is None:
            raise RuntimeError(f"no supported route for family={key.family}")

    route = tuple((float(wp.transform.location.x), float(wp.transform.location.y)) for wp in waypoints)
    npc_actors: list[dict[str, Any]] = []
    script: dict[str, Any] = {
        "family": key.family,
        "pre_roll_ticks": 20,
        "branch_ticks": 50,
        "pre_roll_target_speed_mps": 2.0,
        "pre_roll_kp": 0.30,
        "pre_roll_max_throttle": 0.35,
    }
    if key.family in {"slow_lead", "stopped_lead"}:
        target = waypoints[9 if key.family == "slow_lead" else 8]
        npc_actors.append(
            {
                "role": "lead",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(target.transform)),
            }
        )
        script["lead_control"] = (
            {"throttle": 0.18, "brake": 0.0, "steer": 0.0}
            if key.family == "slow_lead"
            else {"throttle": 0.0, "brake": 1.0, "steer": 0.0}
        )
    elif key.family == "cut_in":
        if selected_cutter is None:
            raise RuntimeError("cut-in adjacent lane unavailable_or_unspawnable")
        adjacent, side = selected_cutter
        npc_actors.append(
            {
                "role": "cutter",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(adjacent.transform)),
            }
        )
        script["cut_in_spawn_probe"] = "fixed-route-index-order-v1"
        script["cutter_control"] = {
            "throttle": 0.22,
            "brake": 0.0,
            "steer": 0.22 if side == "right" else -0.22,
            "adjacent_side": side,
        }
    elif key.family == "red_light_hold":
        script["traffic_light_state"] = "Red"

    return PhysicalScenario(
        scenario=key,
        matrix_index=entry.matrix_index,
        expert_slot=entry.expert_slot,
        branch_order=entry.branch_order,
        ego_transform=transform_to_dict(ego_transform),
        route=route,
        npc_actors=tuple(npc_actors),
        weather=WEATHER_PRESETS[key.weather],
        script=script,
        red_light=red_light,
    )


__all__ = [
    "PhysicalScenario", "WEATHER_PRESETS", "materialize_physical_scenario",
    "transform_from_dict", "transform_to_dict",
]
