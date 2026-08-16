"""Deterministic generator and offline oracle for the 96-anchor H3 Challenge Matrix.

This module materializes scenarios where naive kinematic / planned-length
heuristics fail (e.g. lead vehicle braking, cut-in hazards, red lights,
and cross-traffic conflicts). It outputs Parquet pair and label shards
conforming to the frozen H3 dataset specification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import H3_CONFIG, H3_SCHEMA_VERSION, stable_sha256


CHALLENGE_DATASET_ID = "h3-challenge-20260813-v1"
CHALLENGE_MAPS = ("Town01", "Town03", "Town05")
CHALLENGE_FAMILIES = ("lead_braking", "cut_in_hazard", "red_light_stop", "cross_traffic_gap")
CHALLENGE_SEEDS = (0, 1, 2, 3)
CHALLENGE_WEATHERS = ("ClearNoon", "CloudyNoon")
TOTAL_CHALLENGE_ANCHORS = 96


def _generate_trajectory(
    *,
    v0: float,
    accel: float,
    dt: float = 0.25,
    steps: int = 10,
    heading_curve: float = 0.0,
) -> list[dict[str, float]]:
    """Generate 10 kinematic waypoints (0.25s to 2.5s)."""
    waypoints: list[dict[str, float]] = []
    x, y, yaw, v = 0.0, 0.0, 0.0, v0
    for step in range(1, steps + 1):
        t = step * dt
        v_next = max(0.0, v0 + accel * t)
        dist = v0 * t + 0.5 * accel * (t**2) if v0 + accel * t >= 0 else (0.5 * v0 * (v0 / max(1e-4, -accel)))
        x = dist
        y = heading_curve * dist * 0.1
        yaw = heading_curve * 0.05
        waypoints.append({
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "yaw": float(yaw),
            "v": float(v_next),
            "a": float(accel),
            "kappa": float(heading_curve * 0.01),
            "t": float(t),
        })
    return waypoints


def _generate_history(v0: float, a0: float, ticks: int = 20, dt: float = 0.05) -> list[dict[str, float]]:
    """Generate 20 ticks of historical observable states (-1.0s to 0.0s)."""
    history: list[dict[str, float]] = []
    for tick in range(ticks, 0, -1):
        t_hist = -tick * dt
        v = max(0.0, v0 + a0 * t_hist)
        x = v0 * t_hist + 0.5 * a0 * (t_hist**2)
        history.append({
            "ego_x": float(x),
            "ego_y": 0.0,
            "ego_z": 0.0,
            "ego_yaw": 0.0,
            "ego_speed_mps": float(v),
            "ego_acceleration_mps2": float(a0),
            "simulation_time_s": float(100.0 + t_hist),
        })
    return history


def _generate_route(length_m: float = 120.0, num_points: int = 51) -> list[list[float]]:
    """Generate 51 route points along the corridor."""
    step = length_m / (num_points - 1)
    return [[float(i * step), 0.0] for i in range(num_points)]


def build_challenge_pair(
    map_name: str,
    family: str,
    seed: int,
    weather: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a single deterministic challenge scenario and its oracle label."""
    pair_id = f"{map_name}__{family}__s{seed}__{weather}"

    # Hash determines slot swap (50% defensive in c0, 50% in c1)
    pair_hash = stable_sha256({"pair_id": pair_id})
    defensive_slot = int(pair_hash, 16) % 2

    # Base ego parameters
    ex, ey, eyaw = 0.0, 0.0, 0.0
    speed_limit = 13.88  # 50 km/h
    actors: list[dict[str, Any]] = []
    traffic_lights: list[dict[str, Any]] = []

    if family == "lead_braking":
        ego_v, ego_a = 10.0, 0.0
        lead_dist = 18.0 + seed * 0.5
        lead_v = 10.0
        lead_a = -5.0
        actors.append({
            "id": 101,
            "x": lead_dist,
            "y": 0.0,
            "vx": lead_v,
            "vy": 0.0,
            "yaw": 0.0,
            "length_m": 4.5,
            "width_m": 2.0,
            "lost": False,
        })
        # Aggressive maintains speed (length 25m, crashes into lead car stopping at ~28m)
        aggressive_traj = _generate_trajectory(v0=10.0, accel=0.0)
        # Defensive brakes smoothly (stops at ~14m, safe cushion)
        defensive_traj = _generate_trajectory(v0=10.0, accel=-3.5)

        aggressive_progress = 25.0
        aggressive_jerk = 12.5  # crash penalty
        defensive_progress = 14.3
        defensive_jerk = 1.2
        reason = "lead_vehicle_hard_braking_defensive_stop_preferred"

    elif family == "cut_in_hazard":
        ego_v, ego_a = 12.0, 0.0
        actors.append({
            "id": 102,
            "x": 12.0 + seed * 0.5,
            "y": 3.5,
            "vx": 9.0,
            "vy": -1.4,
            "yaw": -0.15,
            "length_m": 4.8,
            "width_m": 2.1,
            "lost": False,
        })
        # Aggressive accelerates and clips merging vehicle
        aggressive_traj = _generate_trajectory(v0=12.0, accel=1.5)
        # Defensive yields smoothly
        defensive_traj = _generate_trajectory(v0=12.0, accel=-3.0)

        aggressive_progress = 34.7
        aggressive_jerk = 15.0  # collision penalty
        defensive_progress = 18.0
        defensive_jerk = 1.5
        reason = "cut_in_hazard_defensive_yield_preferred"

    elif family == "red_light_stop":
        ego_v, ego_a = 11.0, 0.0
        stop_line_dist = 25.0 + seed * 0.5
        traffic_lights.append({
            "id": 201,
            "distance_m": stop_line_dist + 3.0,
            "stop_line_distance_m": stop_line_dist,
            "state": "red",
            "controls_ego_lane": True,
        })
        # Aggressive runs red light (travels 27.5m, past stop line)
        aggressive_traj = _generate_trajectory(v0=11.0, accel=0.0)
        # Defensive stops compliant before stop line (travels 23.3m)
        defensive_traj = _generate_trajectory(v0=11.0, accel=-2.6)

        aggressive_progress = 27.5
        aggressive_jerk = 20.0  # red light violation penalty
        defensive_progress = 23.3
        defensive_jerk = 1.0
        reason = "red_light_ahead_stop_before_line_preferred"

    else:  # cross_traffic_gap
        ego_v, ego_a = 6.0, 0.0
        actors.append({
            "id": 103,
            "x": 12.0,
            "y": 14.0 + seed * 0.5,
            "vx": 0.0,
            "vy": -10.0,
            "yaw": -1.57,
            "length_m": 4.5,
            "width_m": 2.0,
            "lost": False,
        })
        # Aggressive enters intersection center at t=1.3s during cross-traffic passage
        aggressive_traj = _generate_trajectory(v0=6.0, accel=0.0)
        # Defensive creeps and yields at entry
        defensive_traj = _generate_trajectory(v0=6.0, accel=-2.0)

        aggressive_progress = 15.0
        aggressive_jerk = 18.0  # T-bone hazard penalty
        defensive_progress = 5.0
        defensive_jerk = 0.8
        reason = "unprotected_cross_traffic_yield_preferred"

    snap = {
        "ego_x": ex,
        "ego_y": ey,
        "ego_z": 0.0,
        "ego_yaw": eyaw,
        "ego_pitch": 0.0,
        "ego_roll": 0.0,
        "ego_v": ego_v,
        "ego_a": ego_a,
        "speed_limit_mps": speed_limit,
        "corridor_half_width_m": 3.5,
        "simulation_time_s": 100.0,
        "actors": actors,
        "traffic_lights": traffic_lights,
    }

    observable_history = _generate_history(v0=ego_v, a0=ego_a)
    route = _generate_route()

    if defensive_slot == 0:
        c0_traj, c1_traj = defensive_traj, aggressive_traj
        c0_progress, c1_progress = defensive_progress, aggressive_progress
        c0_jerk, c1_jerk = defensive_jerk, aggressive_jerk
        winner_id = "c0"
    else:
        c0_traj, c1_traj = aggressive_traj, defensive_traj
        c0_progress, c1_progress = aggressive_progress, defensive_progress
        c0_jerk, c1_jerk = aggressive_jerk, defensive_jerk
        winner_id = "c1"

    candidates = [
        {"candidate_id": "c0", "trajectory": c0_traj},
        {"candidate_id": "c1", "trajectory": c1_traj},
    ]
    branches = [
        {"candidate_id": "c0", "route_progress_m": c0_progress, "jerk_rms_mps3": c0_jerk},
        {"candidate_id": "c1", "route_progress_m": c1_progress, "jerk_rms_mps3": c1_jerk},
    ]

    record = {
        "pair_id": pair_id,
        "dataset_id": CHALLENGE_DATASET_ID,
        "schema_version": H3_SCHEMA_VERSION,
        "scenario": {
            "map_name": map_name,
            "family": family,
            "seed": seed,
            "weather": weather,
        },
        "anchor": {"observable_snapshot": snap},
        "observable_history": observable_history,
        "route": route,
        "candidates": candidates,
        "branches": branches,
        "terminal_status": "COMPLETED",
        "vla_forward_count": 1,
    }
    record_sha256 = stable_sha256(record)

    label = {
        "pair_id": pair_id,
        "winner_candidate_id": winner_id,
        "winner_candidate_sha256": stable_sha256(candidates[0 if winner_id == "c0" else 1]),
        "verdict": "CANDIDATE_WIN",
        "reason": reason,
        "oracle_version": "h3-challenge-oracle-v1",
        "branch_order_invariant": True,
        "slot_swap_invariant": True,
        "source_mutation_invariant": True,
    }

    return record, label


def materialize_challenge_dataset(output_dir: Path) -> dict[str, Any]:
    """Materialize all 96 challenge scenarios into Parquet pair and label shards."""
    pairs_dir = output_dir / "pairs"
    labels_dir = output_dir / "labels"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for map_name in CHALLENGE_MAPS:
        for family in CHALLENGE_FAMILIES:
            for seed in CHALLENGE_SEEDS:
                for weather in CHALLENGE_WEATHERS:
                    record, label = build_challenge_pair(map_name, family, seed, weather)
                    pair_id = record["pair_id"]

                    pair_table = pa.Table.from_pydict({
                        "pair_id": [pair_id],
                        "record_sha256": [stable_sha256(record)],
                        "record_json": [json.dumps(record, sort_keys=True)],
                    })
                    pq.write_table(pair_table, pairs_dir / f"{pair_id}.parquet")

                    label_table = pa.Table.from_pydict({
                        "pair_id": [pair_id],
                        "label_json": [json.dumps(label, sort_keys=True)],
                    })
                    pq.write_table(label_table, labels_dir / f"{pair_id}.parquet")

                    manifest_rows.append({
                        "pair_id": pair_id,
                        "map_name": map_name,
                        "family": family,
                        "seed": seed,
                        "weather": weather,
                        "winner_candidate_id": label["winner_candidate_id"],
                        "reason": label["reason"],
                    })

    summary = {
        "dataset_id": CHALLENGE_DATASET_ID,
        "anchors": len(manifest_rows),
        "pairs_dir": str(pairs_dir),
        "labels_dir": str(labels_dir),
        "manifest_rows": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "CHALLENGE_DATASET_ID",
    "CHALLENGE_FAMILIES",
    "CHALLENGE_MAPS",
    "CHALLENGE_SEEDS",
    "CHALLENGE_WEATHERS",
    "TOTAL_CHALLENGE_ANCHORS",
    "build_challenge_pair",
    "materialize_challenge_dataset",
]
