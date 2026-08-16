"""Frozen 96-anchor H3v2 Challenge matrix and physical scenario materialization.

This module returns H2-compatible ``PhysicalScenario`` objects so the H3
collector can reuse the proven H2 capture/branch/Safety/MPC contract verbatim.
Only physical scene geometry and NPC scripts differ from H2.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import carla

from ..h2.carla_scenarios import (
    materialize_physical_scenario as h2_materialize_physical_scenario,

    WEATHER_PRESETS,
    _adjacent_cutter_lane,
    _candidate_routes,
    _probe_cutter_spawn,
    _raised_transform,
    _red_route,
    PhysicalScenario,
    transform_from_dict,
    transform_to_dict,
)
from ..h2.contracts import ScenarioKey, stable_sha256
from dataclasses import replace as dc_replace

from ..h2.matrix import MatrixEntry

CHALLENGE_MAPS = ("Town01", "Town03", "Town05")
CHALLENGE_FAMILIES = (
    "emergency_lead_brake",
    "aggressive_cut_in",
    "red_light_dilemma",
    "cross_traffic_conflict",
)
CHALLENGE_SEEDS = (0, 1, 2, 3)
CHALLENGE_WEATHERS = ("ClearNoon", "CloudyNoon")
CHALLENGE_MATRIX_ALGORITHM = "h3-challenge-matrix-v2"
CHALLENGE_SCENARIO_ALGORITHM = "h3-challenge-scenario-v2"
H3_RED_LOOKAHEAD = 12


def _materialize_matrix() -> tuple[MatrixEntry, ...]:
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
    rows = []
    for index, key in enumerate(keys):
        expert_first = hash_rank[key.pair_id] % 2 == 0
        rows.append(
            MatrixEntry(
                scenario=key,
                branch_order=("expert", "vla") if expert_first else ("vla", "expert"),
                expert_slot=0 if expert_first else 1,
                matrix_index=index,
            )
        )
    return tuple(rows)


CHALLENGE_FIXED_MATRIX = _materialize_matrix()
CHALLENGE_PILOT_MATRIX = tuple(
    row for row in CHALLENGE_FIXED_MATRIX if row.scenario.seed == 0 and row.scenario.weather == "ClearNoon"
)
CHALLENGE_MATRIX_SHA256 = stable_sha256(
    {"algorithm": CHALLENGE_MATRIX_ALGORITHM, "rows": [row.to_dict() for row in CHALLENGE_FIXED_MATRIX]}
)

assert len(CHALLENGE_FIXED_MATRIX) == 96
assert len(CHALLENGE_PILOT_MATRIX) == 12
assert sum(row.expert_slot == 0 for row in CHALLENGE_FIXED_MATRIX) == 48


def _route_for_family(world: Any, carla_map: Any, key: ScenarioKey, family: str) -> tuple[Any, tuple[Any, ...], dict[str, Any] | None]:
    if family == "red_light_dilemma":
        return _red_route(world, carla_map, key)
    candidates = _candidate_routes(carla_map)
    ranked = sorted(candidates, key=lambda item: stable_sha256(transform_to_dict(item[0])))
    start_index = int(hashlib.sha256(key.pair_id.encode()).hexdigest(), 16) % len(ranked)
    selected_cutter: tuple[Any, str] | None = None
    for offset in range(len(ranked)):
        spawn, candidate_route = ranked[(start_index + offset) % len(ranked)]
        if family == "aggressive_cut_in":
            for route_index in (16, 17, 18, 15, 19):
                if route_index >= len(candidate_route):
                    continue
                adjacent, side = _adjacent_cutter_lane(candidate_route[route_index])
                if adjacent is not None and _probe_cutter_spawn(world, _raised_transform(adjacent.transform)):
                    selected_cutter = (adjacent, side)
                    break
            if selected_cutter is None:
                continue
        return spawn, candidate_route, {"selected_cutter": selected_cutter}
    raise RuntimeError(f"no supported route for family={family}")


def materialize_challenge_physical_scenario(world: Any, entry: MatrixEntry) -> PhysicalScenario:
    key = entry.scenario
    map_name = str(world.get_map().name).split("/")[-1].replace("_Opt", "")
    if map_name != key.map_name:
        raise RuntimeError(f"materialization map mismatch for {key.pair_id}")
    carla_map = world.get_map()
    family = key.family
    red_light: dict[str, Any] | None = None
    selected_cutter: tuple[Any, str] | None = None

    if family in {"aggressive_cut_in", "emergency_lead_brake", "cross_traffic_conflict"}:
        # Reuse the proven H2 materializations that both generators can
        # canonicalize, then overlay the H3 dynamic-event script.
        h2_family = "cut_in" if family == "aggressive_cut_in" else ("slow_lead" if family == "emergency_lead_brake" else "stopped_lead")
        h2_key = ScenarioKey(key.map_name, h2_family, key.seed, key.weather)
        h2_entry = MatrixEntry(
            scenario=h2_key, branch_order=entry.branch_order, expert_slot=entry.expert_slot, matrix_index=entry.matrix_index
        )
        base = h2_materialize_physical_scenario(world, h2_entry)
        script = dict(base.script)
        script.update({
            "family": family,
            "pre_roll_ticks": 20,
            "pre_roll_target_speed_mps": 2.0,
            "pre_roll_kp": 0.30,
            "pre_roll_max_throttle": 0.35,
        })
        if family == "emergency_lead_brake":
            script.update({"lead_pre_roll_control": {"throttle": 0.18, "brake": 0.0, "steer": 0.0},
                           "lead_control": {"throttle": 0.0, "brake": 1.0, "steer": 0.0}, "lead_brake_tick": 10})
        elif family == "cross_traffic_conflict":
            script.update({"lead_pre_roll_control": {"throttle": 0.0, "brake": 1.0, "steer": 0.0},
                           "lead_control": {"throttle": 0.18, "brake": 0.0, "steer": 0.0}, "lead_brake_tick": 8})
        elif family == "aggressive_cut_in":
            script.update({"cutter_pre_roll_control": {"throttle": 0.22, "brake": 0.0, "steer": 0.0},
                           "cutter_control": {"throttle": 0.25, "brake": 0.0, "steer": 0.45}, "cutter_cut_in_tick": 10})
        return dc_replace(base, scenario=key, algorithm_version=CHALLENGE_SCENARIO_ALGORITHM, script=script)
    if family == "red_light_dilemma":
        ego_transform, waypoints, red_light = _red_route(world, carla_map, key)
        # Start closer to the stop line so a 2.5 s branch can actually cross it.
        start_offset = 7
        if len(waypoints) > start_offset + H3_RED_LOOKAHEAD:
            old_start_progress = sum(
                math.hypot(float(waypoints[i].transform.location.x) - float(waypoints[i - 1].transform.location.x),
                           float(waypoints[i].transform.location.y) - float(waypoints[i - 1].transform.location.y))
                for i in range(1, start_offset + 1)
            )
            waypoints = waypoints[start_offset:]
            ego_transform = _raised_transform(waypoints[0].transform)
            red_light["stop_progress_m"] = max(0.0, float(red_light["stop_progress_m"]) - old_start_progress)
    else:
        result = _route_for_family(world, carla_map, key, family)
        if len(result) == 3:
            ego_transform, waypoints, extra = result
            selected_cutter = extra.get("selected_cutter") if extra else None
        else:
            ego_transform, waypoints = result[0], result[1]

    route = tuple((float(wp.transform.location.x), float(wp.transform.location.y)) for wp in waypoints)
    npc_actors: list[dict[str, Any]] = []
    script: dict[str, Any] = {
        "family": family,
        "pre_roll_ticks": 40,
        "branch_ticks": 50,
        "pre_roll_target_speed_mps": 10.0,
        "pre_roll_kp": 0.80,
        "pre_roll_max_throttle": 0.95,
    }

    if family == "emergency_lead_brake":
        target = waypoints[min(len(waypoints) - 1, 14)]
        npc_actors.append(
            {
                "role": "lead",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(target.transform)),
            }
        )
        script.update(
            {
                "lead_pre_roll_control": {"throttle": 0.80, "brake": 0.0, "steer": 0.0},
                "lead_control": {"throttle": 0.0, "brake": 1.0, "steer": 0.0},
                "lead_brake_tick": 10,
            }
        )
    elif family == "aggressive_cut_in":
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
        script.update(
            {
                "cutter_pre_roll_control": {"throttle": 0.20, "brake": 0.0, "steer": 0.0},
                "cutter_control": {"throttle": 0.25, "brake": 0.0, "steer": 0.28 if side == "right" else -0.28},
                "cutter_cut_in_tick": 12,
                "adjacent_side": side,
            }
        )
    elif family == "red_light_dilemma":
        script["traffic_light_state"] = "Red"
        script["red_at_capture"] = False
        script["red_after_tick"] = 5
        script["pre_roll_ticks"] = 10
        script["pre_roll_target_speed_mps"] = 8.0
        script["pre_roll_kp"] = 0.35
    elif family == "cross_traffic_conflict":
        target = waypoints[min(len(waypoints) - 1, 12)]
        npc_actors.append(
            {
                "role": "cross",
                "blueprint": "vehicle.lincoln.mkz_2020",
                "transform": transform_to_dict(_raised_transform(target.transform)),
            }
        )
        script.update(
            {
                "cross_pre_roll_control": {"throttle": 0.25, "brake": 0.0, "steer": 0.0},
                "cross_control": {"throttle": 0.35, "brake": 0.0, "steer": 0.0},
                "cross_conflict_tick": 8,
                "pre_roll_target_speed_mps": 7.0,
            }
        )

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
        algorithm_version=CHALLENGE_SCENARIO_ALGORITHM,
    )


def materialize_map_rows(world: Any, rows: Sequence[MatrixEntry]) -> tuple[PhysicalScenario, ...]:
    return tuple(materialize_challenge_physical_scenario(world, entry) for entry in rows)


__all__ = [
    "CHALLENGE_FAMILIES",
    "CHALLENGE_FIXED_MATRIX",
    "CHALLENGE_MAPS",
    "CHALLENGE_MATRIX_ALGORITHM",
    "CHALLENGE_MATRIX_SHA256",
    "CHALLENGE_PILOT_MATRIX",
    "CHALLENGE_SCENARIO_ALGORITHM",
    "CHALLENGE_SEEDS",
    "CHALLENGE_WEATHERS",
    "materialize_challenge_physical_scenario",
    "materialize_map_rows",
]
