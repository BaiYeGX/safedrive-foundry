#!/usr/bin/env python3
"""Read-only CARLA topology capacity probe for R2 V3 map selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.route_authoring_v3 import _lane_access  # noqa: E402
from driving_vla.model.navigation_contract import TargetLaneSide  # noqa: E402
from runtime.carla_connection import ConnectionResolver  # noqa: E402


def _matches(actual: str, requested: str) -> bool:
    return actual.endswith(requested) or f"/{requested}" in actual


def _spawnable(lane: Any) -> bool:
    return bool(
        lane.exists
        and lane.driving
        and lane.same_direction
        and len(lane.centerline_xy) >= 2
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--spacing-m", type=float, default=8.0)
    args = parser.parse_args()

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16")
    report = resolver.preflight()
    if report.status != "READY":
        raise RuntimeError(
            f"CARLA preflight must be READY, got "
            f"{report.status}/{report.error_code}"
        )
    client, _ = resolver.connect(report=report)
    world = client.get_world()
    world_map = world.get_map()
    actual_map = str(world_map.name)
    if not _matches(actual_map, str(args.map)):
        raise RuntimeError(
            f"map mismatch: actual={actual_map}, requested={args.map}"
        )

    counts = {
        "sampled_non_junction_waypoints": 0,
        "junction_waypoints": 0,
        "spawnable_left": 0,
        "spawnable_right": 0,
        "authorized_left": 0,
        "authorized_right": 0,
    }
    corridor_keys = {key: set() for key in counts if key.endswith(("left", "right"))}
    for waypoint in world_map.generate_waypoints(float(args.spacing_m)):
        if bool(getattr(waypoint, "is_junction", False)):
            counts["junction_waypoints"] += 1
            continue
        counts["sampled_non_junction_waypoints"] += 1
        origin_key = (
            int(getattr(waypoint, "road_id", 0)),
            int(getattr(waypoint, "lane_id", 0)),
        )
        for side in (TargetLaneSide.LEFT, TargetLaneSide.RIGHT):
            lane = _lane_access(waypoint, side, step_m=2.0, count=2)
            suffix = side.value.lower()
            if _spawnable(lane):
                counts[f"spawnable_{suffix}"] += 1
                corridor_keys[f"spawnable_{suffix}"].add(origin_key)
                if lane.lane_change_allowed and lane.currently_clear:
                    counts[f"authorized_{suffix}"] += 1
                    corridor_keys[f"authorized_{suffix}"].add(origin_key)

    traffic_lights = len(world.get_actors().filter("traffic.traffic_light*"))
    required = (
        "spawnable_left",
        "spawnable_right",
        "authorized_left",
        "authorized_right",
    )
    body = {
        "map_name": str(args.map),
        "actual_map": actual_map,
        "spacing_m": float(args.spacing_m),
        "traffic_light_count": traffic_lights,
        "waypoint_counts": counts,
        "unique_origin_lane_counts": {
            key: len(value) for key, value in corridor_keys.items()
        },
        "r2_v3_side_capacity_ready": all(corridor_keys[key] for key in required),
        "traffic_control_capacity_ready": traffic_lights > 0,
    }
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
