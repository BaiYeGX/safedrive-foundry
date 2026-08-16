"""Deterministic physical materialization for the 96-anchor H3 CARLA Challenge Matrix.

This module materializes physical CARLA scenario setups (ego/NPC transforms, routes,
weather, and dynamic actor scripts) for interactive and adversarial scenarios:
- emergency_lead_brake: Lead vehicle cruises then suddenly applies full braking (-5 m/s²).
- aggressive_cut_in: Adjacent NPC performs sharp cut-in ahead of ego vehicle.
- red_light_dilemma: Traffic light turns red as ego approaches the stop line.
- cross_traffic_conflict: Crossing vehicle intersects the ego vehicle's path at junction.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import carla

from ..h2.contracts import ScenarioKey, stable_sha256
from ..h2.live_contract import SCENARIO_ALGORITHM_VERSION, route_projection
from ..h2.carla_scenarios import (
    WEATHER_PRESETS,
    _adjacent_cutter_lane,
    _candidate_routes,
    _probe_cutter_spawn,
    _raised_transform,
    _red_route,
    transform_from_dict,
    transform_to_dict,
)


CHALLENGE_MAPS = ("Town01", "Town03", "Town05")
CHALLENGE_FAMILIES = (
    "emergency_lead_brake",
    "aggressive_cut_in",
    "red_light_dilemma",
    "cross_traffic_conflict",
)
CHALLENGE_SEEDS = (0, 1, 2, 3)
CHALLENGE_WEATHERS = ("ClearNoon", "CloudyNoon")
CHALLENGE_MATRIX_ALGORITHM = "h3-carla-challenge-matrix-v1"


@dataclass(frozen=True)
class ChallengeMatrixEntry:
    scenario: ScenarioKey
    branch_order: tuple[str, str]
    defensive_slot: int
    matrix_index: int

    @property
    def pair_id(self) -> str:
        return self.scenario.pair_id

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pair_id"] = self.pair_id
        return result


def materialize_challenge_matrix() -> tuple[ChallengeMatrixEntry, ...]:
    keys = tuple(
        ScenarioKey(map_name, family, seed, weather)
        for map_name in CHALLENGE_MAPS
        for family in CHALLENGE_FAMILIES
        for seed in CHALLENGE_SEEDS
        for weather in CHALLENGE_WEATHERS
    )
    hash_rank = {
        key.pair_id: rank
        for rank, key in enumerate(sorted(keys, key=lambda item: stable_sha256(item.to_dict())))
    }
    rows: list[ChallengeMatrixEntry] = []
    for index, key in enumerate(keys):
        defensive_first = hash_rank[key.pair_id] % 2 == 0
        rows.append(
            ChallengeMatrixEntry(
                scenario=key,
                branch_order=("defensive", "aggressive") if defensive_first else ("aggressive", "defensive"),
                defensive_slot=0 if defensive_first else 1,
                matrix_index=index,
            )
        )
    return tuple(rows)


CHALLENGE_FIXED_MATRIX = materialize_challenge_matrix()
CHALLENGE_MATRIX_SHA256 = stable_sha256(
    {"algorithm": CHALLENGE_MATRIX_ALGORITHM, "rows": [row.to_dict() for row in CHALLENGE_FIXED_MATRIX]}
)

assert len(CHALLENGE_FIXED_MATRIX) == 96
assert sum(row.defensive_slot == 0 for row in CHALLENGE_FIXED_MATRIX) == 48


@dataclass(frozen=True)
class ChallengePhysicalScenario:
    scenario: ScenarioKey
    matrix_index: int
    defensive_slot: int
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
            "defensive_slot": self.defensive_slot,
            "branch_order": list(self.branch_order),
            "ego_transform": dict(self.ego_transform),
            "route": [list(point) for point in self.route],
            "npc_actors": [dict(actor) for actor in self.npc_actors],
            "weather": dict(self.weather),
            "script": dict(self.script),
            "red_light": dict(self.red_light) if self.red_light else None,
            "algorithm_version": self.algorithm_version,
        }
        if include_hash:
            payload["physical_sha256"] = self.physical_sha256
        return payload


def materialize_challenge_physical_scenario(world: Any, entry: ChallengeMatrixEntry) -> ChallengePhysicalScenario:
    key = entry.scenario
    if str(world.get_map().name).split("/")[-1].replace("_Opt", "") != key.map_name:
        raise RuntimeError(f"materialization map mismatch for {key.pair_id}")
    carla_map = world.get_map()
    red_light: dict[str, Any] | None = None

    if key.family == "red_light_dilemma":
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
            if key.family == "aggressive_cut_in":
                probe_result: tuple[Any, str] | None = None
                for route_index in (7, 8, 9, 6):
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
        "pre_roll_target_speed_mps": 3.0,
        "pre_roll_kp": 0.30,
        "pre_roll_max_throttle": 0.35,
    }

    if key.family == "emergency_lead_brake":
        # Lead vehicle starts 11-13m ahead and hard-brakes during branch
        target = waypoints[min(len(waypoints) - 1, 8)]
        npc_actors.append(
            {
                "role": "lead",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(target.transform)),
            }
        )
        script["lead_control"] = {"throttle": 0.0, "brake": 1.0, "steer": 0.0}
    elif key.family == "aggressive_cut_in":
        if selected_cutter is None:
            raise RuntimeError("cut-in adjacent lane unavailable")
        adjacent, side = selected_cutter
        npc_actors.append(
            {
                "role": "cutter",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(adjacent.transform)),
            }
        )
        script["cut_in_spawn_probe"] = "challenge-route-index-order-v1"
        script["cutter_control"] = {
            "throttle": 0.35,
            "brake": 0.0,
            "steer": 0.32 if side == "right" else -0.32,
            "adjacent_side": side,
        }
    elif key.family == "red_light_dilemma":
        script["traffic_light_state"] = "Red"
    elif key.family == "cross_traffic_conflict":
        target = waypoints[min(len(waypoints) - 1, 10)]
        npc_actors.append(
            {
                "role": "cross",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(target.transform)),
            }
        )
        script["cross_control"] = {"throttle": 0.30, "brake": 0.0, "steer": 0.0}

    return ChallengePhysicalScenario(
        scenario=key,
        matrix_index=entry.matrix_index,
        defensive_slot=entry.defensive_slot,
        branch_order=entry.branch_order,
        ego_transform=transform_to_dict(ego_transform),
        route=route,
        npc_actors=tuple(npc_actors),
        weather=WEATHER_PRESETS[key.weather],
        script=script,
        red_light=red_light,
    )


__all__ = [
    "CHALLENGE_FAMILIES",
    "CHALLENGE_FIXED_MATRIX",
    "CHALLENGE_MAPS",
    "CHALLENGE_MATRIX_ALGORITHM",
    "CHALLENGE_MATRIX_SHA256",
    "CHALLENGE_SEEDS",
    "CHALLENGE_WEATHERS",
    "ChallengeMatrixEntry",
    "ChallengePhysicalScenario",
    "WEATHER_PRESETS",
    "materialize_challenge_matrix",
    "materialize_challenge_physical_scenario",
]
