#!/usr/bin/env python3
"""Author exact CARLA fixture registries for a planned R23 campaign.

This is geometry-only authoring.  It never executes VLA, MPC, Oracle, or branch
outcomes.  Every generated shard must still pass the normal two-rebuild dry-run
before its registry manifest exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.fixture_runtime import connect_world, restore_async  # noqa: E402
from driving_vla.evaluation.r23_collection import (  # noqa: E402
    content_hash,
    write_json_exclusive,
)


def _yaw_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _forward_chain(waypoint: Any, *, step_m: float = 8.0, count: int = 6) -> list[Any]:
    output = [waypoint]
    current = waypoint
    for _ in range(count - 1):
        following = current.next(step_m)
        if not following:
            break
        current = following[0]
        output.append(current)
    if len(output) < 5:
        raise RuntimeError("route waypoint chain shorter than five")
    return output


def _adjacent(waypoint: Any) -> Any | None:
    candidates = [waypoint.get_left_lane(), waypoint.get_right_lane()]
    for candidate in candidates:
        if candidate is None:
            continue
        same_lane_type = int(getattr(candidate, "lane_type", 0)) == int(
            getattr(waypoint, "lane_type", 0)
        )
        same_direction = int(getattr(candidate, "lane_id", 0)) * int(
            getattr(waypoint, "lane_id", 0)
        ) > 0
        if same_lane_type and same_direction:
            return candidate
    return None


def _crossing_waypoint(world_map: Any, target: Any) -> Any:
    location = target.transform.location
    yaw = float(target.transform.rotation.yaw)
    candidates = []
    for waypoint in world_map.generate_waypoints(4.0):
        if int(waypoint.road_id) == int(target.road_id):
            continue
        other = waypoint.transform.location
        distance = math.hypot(float(other.x - location.x), float(other.y - location.y))
        angle = _yaw_delta(float(waypoint.transform.rotation.yaw), yaw)
        if 4.0 <= distance <= 16.0 and 55.0 <= angle <= 125.0:
            candidates.append((distance, int(waypoint.road_id), int(waypoint.lane_id), waypoint))
    if not candidates:
        raise RuntimeError("no deterministic crossing waypoint near route")
    return min(candidates, key=lambda item: item[:3])[3]


def _pose_lines(prefix: str, waypoint: Any) -> list[str]:
    transform = waypoint.transform
    return [
        f"[{prefix}.transform]",
        f"x = {float(transform.location.x):.6f}",
        f"y = {float(transform.location.y):.6f}",
        f"z = {float(transform.location.z + 0.50):.6f}",
        f"roll_deg = {float(transform.rotation.roll):.6f}",
        f"pitch_deg = {float(transform.rotation.pitch):.6f}",
        f"yaw_deg = {float(transform.rotation.yaw):.6f}",
    ]


def _velocity_lines(prefix: str, waypoint: Any, speed: float) -> list[str]:
    yaw = math.radians(float(waypoint.transform.rotation.yaw))
    return [
        f"[{prefix}.initial_velocity]",
        f"vx = {math.cos(yaw) * speed:.6f}",
        f"vy = {math.sin(yaw) * speed:.6f}",
        "vz = 0.000000",
    ]


def _actor_script_lines(prefix: str, family: str, condition_index: int) -> list[str]:
    reactive = condition_index >= 3 and family not in {"clear", "obstruction"}
    if reactive:
        return [
            f"[{prefix}.script]",
            'script_type = "reactive_yield"',
            f"desired_speed_mps = {4.5 + 0.2 * condition_index:.3f}",
            f"yield_ttc_s = {2.2 + 0.1 * condition_index:.3f}",
            f"stop_distance_m = {4.5 + 0.2 * condition_index:.3f}",
            f"actor_has_priority = {'true' if family == 'crossing' else 'false'}",
        ]
    if family == "lead_braking":
        brake = 0.25 + 0.08 * condition_index
        return [
            f"[{prefix}.script]",
            'script_type = "piecewise_vehicle_control"',
            f"[[{prefix}.script.knots]]",
            "t_s = 0.00",
            "throttle = 0.25",
            "brake = 0.00",
            "steer = 0.00",
            f"[[{prefix}.script.knots]]",
            f"t_s = {0.55 + 0.08 * condition_index:.3f}",
            "throttle = 0.00",
            f"brake = {min(0.85, brake):.3f}",
            "steer = 0.00",
        ]
    if family == "obstruction":
        return [f"[{prefix}.script]", 'script_type = "constant_brake"', "brake = 1.0"]
    steer = 0.0
    if family in {"cut_in", "merge"}:
        steer = 0.08 + 0.01 * condition_index
    return [
        f"[{prefix}.script]",
        'script_type = "constant_throttle"',
        f"throttle = {0.25 + 0.02 * condition_index:.3f}",
        f"steer = {steer:.3f}",
    ]


def _weather_lines(prefix: str, condition_index: int) -> list[str]:
    values = (
        ("ClearNoon", 10.0, 0.0, 0.0, 70.0, 0.0),
        ("WetNoon", 45.0, 0.0, 45.0, 55.0, 45.0),
        ("HardRainSunset", 90.0, 70.0, 85.0, 18.0, 90.0),
        ("ClearNight", 15.0, 0.0, 0.0, -20.0, 0.0),
        ("WetSunset", 55.0, 0.0, 55.0, 12.0, 55.0),
        ("HardRainNight", 95.0, 75.0, 90.0, -25.0, 95.0),
    )[condition_index]
    preset, cloudiness, precipitation, wetness, altitude, deposits = values
    return [
        f"[{prefix}.weather]",
        f'preset = "{preset}"',
        f"cloudiness = {cloudiness:.3f}",
        f"precipitation = {precipitation:.3f}",
        f"precipitation_deposits = {deposits:.3f}",
        "wind_intensity = 10.000",
        "sun_azimuth_angle = 0.000",
        f"sun_altitude_angle = {altitude:.3f}",
        f"wetness = {wetness:.3f}",
        "fog_density = 0.000",
        "fog_distance = 0.000",
    ]


def _topology_signature(map_name: str, route: list[Any]) -> tuple[str, dict[str, Any]]:
    points = []
    yaw_deltas = []
    slopes = []
    for index, waypoint in enumerate(route):
        transform = waypoint.transform
        points.append(
            {
                "road_id": int(waypoint.road_id),
                "lane_id": int(waypoint.lane_id),
                "s_bucket_20m": int(float(waypoint.s) // 20.0),
                "junction": bool(getattr(waypoint, "is_junction", False)),
                "lane_width_dm": int(round(float(waypoint.lane_width) * 10.0)),
            }
        )
        if index:
            previous = route[index - 1].transform
            yaw_deltas.append(
                round(
                    _yaw_delta(
                        float(transform.rotation.yaw),
                        float(previous.rotation.yaw),
                    ),
                    1,
                )
            )
            horizontal = math.hypot(
                float(transform.location.x - previous.location.x),
                float(transform.location.y - previous.location.y),
            )
            slopes.append(
                round(
                    float(transform.location.z - previous.location.z)
                    / max(horizontal, 1e-3),
                    3,
                )
            )
    body = {
        "map_name": map_name,
        "route_points": points,
        "yaw_delta_bins_deg": yaw_deltas,
        "slope_bins": slopes,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), body


def _stop_windows_carla() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        raise RuntimeError("PowerShell unavailable; cannot cold-switch CARLA map")
    command = (
        "Get-Process -Name 'CarlaUE4*','UE4Editor*' "
        "-ErrorAction SilentlyContinue | "
        "Stop-Process -Force -ErrorAction SilentlyContinue"
    )
    subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        timeout=30.0,
    )
    time.sleep(4.0)


def _ensure_map(*, map_name: str) -> None:
    startup_timeout = 360 if map_name in {"Town11", "Town12", "Town13", "Town15"} else 180
    preflight = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sdf.py"), "sim", "preflight", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    current = {}
    try:
        current = json.loads(preflight.stdout or "{}")
    except json.JSONDecodeError:
        current = {}
    actual = str(current.get("map") or "")
    if str(current.get("status")) == "READY" and (
        actual.endswith(map_name) or f"/{map_name}" in actual
    ):
        return
    if str(current.get("process_state")) == "RUNNING" or actual:
        _stop_windows_carla()
    ensure = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sdf.py"),
            "sim",
            "ensure",
            "--map",
            map_name,
            "--rhi",
            "dx12",
            "--startup-timeout",
            str(startup_timeout),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(startup_timeout + 60),
    )
    try:
        report = json.loads(ensure.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{map_name}: invalid ensure output: {ensure.stdout[-500:]}"
        ) from exc
    if str(report.get("status")) != "READY":
        raise RuntimeError(f"{map_name}: CARLA ensure failed: {report}")
    # A READY handshake can precede the last large-map streaming/shader work by
    # a few seconds.  Do not immediately start a 60 s geometry query.
    time.sleep(8.0)


def _render_registry(shard_plan: dict[str, Any], world_map: Any, base_waypoint: Any) -> str:
    slots = sorted(shard_plan["slots"], key=lambda row: int(row["slot_index"]))
    map_name = str(slots[0]["map_name"])
    family = str(slots[0]["family"])
    lineage_id = str(slots[0]["lineage_id"])
    route_chain = _forward_chain(base_waypoint)
    lines = [
        "# Auto-authored R23 collection registry; must pass dry-run before use.",
        "[registry]",
        'schema_version = "safedrive.g4a.scenario_registry.v1"',
        f'registry_version = "r23-{slots[0]["phase"]}-{shard_plan["shard_id"]}"',
        f'description = "R23 collection {lineage_id}"',
        "",
        "[defaults]",
        f'map_name = "{map_name}"',
        "sim_dt_s = 0.05",
        "duration_s = 5.0",
        'vla_config_ref = "config/vla/k2_v2_spatial.toml"',
        'mpc_config_ref = "config/control/mpc_pid_baseline.toml"',
        'executor_config_ref = "g3_stable_vla_mpc"',
        "expected_decision_anchor_time_s = 1.0",
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
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        scenarios[str(slot["scenario_id"])].append(slot)
    ordered_scenarios = sorted(
        scenarios.items(),
        key=lambda item: min(int(row["slot_index"]) for row in item[1]),
    )
    for scenario_id, seed_slots in ordered_scenarios:
        condition_index = min(int(row["slot_index"]) for row in seed_slots) // 2
        route = route_chain
        lines.extend(
            [
                f"[scenarios.{scenario_id}]",
                f'family = "{family}"',
                f'map_name = "{map_name}"',
                (
                    f'notes = "lineage={lineage_id} condition={condition_index} '
                    'obstruction_layout=ego_lane_with_authorized_adjacent"'
                    if family == "obstruction"
                    else f'notes = "lineage={lineage_id} condition={condition_index}"'
                ),
                *_weather_lines(f"scenarios.{scenario_id}", condition_index),
                "",
                f"[scenarios.{scenario_id}.route]",
                f'identity = "{lineage_id}_route"',
                "target_speed_mps = 8.0",
                "waypoints = [",
            ]
        )
        for waypoint in route[:5]:
            location = waypoint.transform.location
            lines.append(
                f"  [{float(location.x):.6f}, {float(location.y):.6f}, "
                f"{float(location.z + 0.50):.6f}],"
            )
        lines.extend(["]", ""])
        for slot in sorted(seed_slots, key=lambda row: str(row["seed_id"])):
            seed_id = str(slot["seed_id"])
            seed_offset = 0 if seed_id == "seed_a" else 1
            ego_wp = route[0]
            ego_prefix = f"scenarios.{scenario_id}.seeds.{seed_id}.ego"
            lines.extend(
                [
                    f"[{ego_prefix}]",
                    'name = "ego"',
                    'role = "ego"',
                    'blueprint = "vehicle.tesla.model3"',
                    "spawn_order = 0",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(ego_prefix, ego_wp),
                    *_velocity_lines(ego_prefix, ego_wp, 5.5 + 0.25 * seed_offset),
                    f"[{ego_prefix}.script]",
                    'script_type = "hold"',
                    "",
                ]
            )
            if family == "clear":
                continue
            actor_wp = route[min(2 + seed_offset, len(route) - 1)]
            if family in {"cut_in", "merge"}:
                actor_wp = _adjacent(actor_wp)
                if actor_wp is None:
                    raise RuntimeError(f"{lineage_id}: no adjacent lane for {family}")
            elif family == "obstruction":
                # The obstacle is the object being overtaken, so it must block
                # the ego route.  The adjacent lane is topology authorization,
                # not the obstacle spawn lane.
                if _adjacent(actor_wp) is None:
                    raise RuntimeError(
                        f"{lineage_id}: obstruction has no adjacent "
                        "same-direction lane for legal overtake"
                    )
            elif family == "crossing":
                actor_wp = _crossing_waypoint(world_map, route[2])
            actor_prefix = f"scenarios.{scenario_id}.seeds.{seed_id}.actors"
            actor_speed = 0.0 if family == "obstruction" else 4.0 + 0.2 * seed_offset
            lines.extend(
                [
                    f"[[{actor_prefix}]]",
                    'name = "conflict_actor"',
                    'role = "npc"',
                    'blueprint = "vehicle.lincoln.mkz_2017"',
                    "spawn_order = 1",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(actor_prefix, actor_wp),
                    *_velocity_lines(actor_prefix, actor_wp, actor_speed),
                    *_actor_script_lines(actor_prefix, family, condition_index),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map-only", default="")
    args = parser.parse_args()
    campaign_root = Path(args.campaign_root)
    shard_plans = sorted(campaign_root.glob("*/shards/*/shard_plan.json"))
    if not shard_plans:
        raise SystemExit("no shard plans found; run r23_campaign.py plan first")
    plans = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in shard_plans]
    by_map: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, plan in plans:
        by_map[str(plan["slots"][0]["map_name"])].append((path, plan))
    if args.map_only:
        requested = str(args.map_only)
        if requested not in by_map:
            raise SystemExit(f"requested map is not in campaign: {requested}")
        by_map = {requested: by_map[requested]}
    used_geometry: set[str] = set()
    report = {"schema_version": "safedrive.r23_author_report.v1", "shards": []}
    for map_name, entries in sorted(by_map.items()):
        fully_authored = all(
            (path.parent / "scenario_registry.toml").is_file()
            and (path.parent / "author_manifest.json").is_file()
            for path, _plan in entries
        )
        if fully_authored:
            for plan_path, plan in sorted(
                entries, key=lambda item: item[0].as_posix()
            ):
                frozen = json.loads(
                    (plan_path.parent / "author_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if str(frozen.get("shard_plan_hash")) != content_hash(plan):
                    raise RuntimeError(
                        f"authored shard plan mismatch: {plan['shard_id']}"
                    )
                used_geometry.add(str(frozen["topology_sha256"]))
                report["shards"].append(dict(frozen["report_row"]))
            continue
        _ensure_map(map_name=map_name)
        _client, world = connect_world(
            host=args.host,
            port=args.port,
            map_name=map_name,
            sim_dt_s=0.05,
            sync=True,
            timeout_s=60.0,
            retries=3,
        )
        try:
            world_map = world.get_map()
            spawn_points = sorted(
                world_map.get_spawn_points(),
                key=lambda transform: (
                    round(float(transform.location.x), 3),
                    round(float(transform.location.y), 3),
                    round(float(transform.rotation.yaw), 2),
                ),
            )
            if not spawn_points:
                raise RuntimeError(f"{map_name}: no spawn points")
            cursor = 0
            for plan_path, plan in sorted(entries, key=lambda item: item[0].as_posix()):
                registry_path = plan_path.parent / "scenario_registry.toml"
                author_manifest_path = plan_path.parent / "author_manifest.json"
                if registry_path.exists() or author_manifest_path.exists():
                    if not registry_path.is_file() or not author_manifest_path.is_file():
                        raise RuntimeError(
                            f"partial authoring state in {plan_path.parent}"
                        )
                    frozen = json.loads(
                        author_manifest_path.read_text(encoding="utf-8")
                    )
                    if str(frozen.get("shard_plan_hash")) != content_hash(plan):
                        raise RuntimeError(
                            f"authored shard plan mismatch: {plan['shard_id']}"
                        )
                    topology_sha256 = str(frozen["topology_sha256"])
                    used_geometry.add(topology_sha256)
                    report["shards"].append(dict(frozen["report_row"]))
                    continue
                selected = None
                while cursor < len(spawn_points) * 2:
                    transform = spawn_points[cursor % len(spawn_points)]
                    cursor += 1
                    waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
                    if waypoint is None:
                        continue
                    try:
                        route = _forward_chain(waypoint)
                        family = str(plan["slots"][0]["family"])
                        if (
                            family in {"cut_in", "merge", "obstruction"}
                            and any(_adjacent(route[index]) is None for index in (2, 3))
                        ):
                            continue
                        if family == "crossing":
                            _crossing_waypoint(world_map, route[2])
                    except RuntimeError:
                        continue
                    topology_sha256, topology = _topology_signature(map_name, route)
                    if topology_sha256 in used_geometry:
                        continue
                    selected = (waypoint, topology_sha256, topology)
                    break
                if selected is None:
                    raise RuntimeError(f"{map_name}: exhausted unique valid geometry")
                waypoint, topology_sha256, topology = selected
                used_geometry.add(topology_sha256)
                registry_path.write_text(
                    _render_registry(plan, world_map, waypoint), encoding="utf-8"
                )
                report_row = {
                    "shard_id": plan["shard_id"],
                    "map_name": map_name,
                    "lineage_id": plan["slots"][0]["lineage_id"],
                    "family": plan["slots"][0]["family"],
                    "geometry_key": topology_sha256[:20],
                    "topology_sha256": topology_sha256,
                    "topology": topology,
                    "registry_path": str(registry_path),
                }
                write_json_exclusive(
                    author_manifest_path,
                    {
                        "schema_version": "safedrive.r23_author_manifest.v1",
                        "shard_plan_hash": content_hash(plan),
                        "topology_sha256": topology_sha256,
                        "report_row": report_row,
                    },
                )
                report["shards"].append(report_row)
        finally:
            restore_async(world)
    output = campaign_root / (
        f"scenario_author_report.{args.map_only}.json"
        if args.map_only
        else "scenario_author_report.json"
    )
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit(f"existing author report mismatch: {output}")
        status = "AUTHORED_REUSED"
    else:
        write_json_exclusive(output, report)
        status = "AUTHORED"
    print(json.dumps({"status": status, "n_shards": len(report["shards"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
