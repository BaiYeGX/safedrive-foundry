#!/usr/bin/env python3
"""Author R2 V4 six-condition registries from live CARLA waypoints.

This is geometry/actor authoring only.  It never loads SimLingo, runs a
candidate, observes an outcome, or consults an Oracle.  Each root lineage is
rendered as one six-scenario/two-seed ``r23-*`` registry so the existing
registry loader accepts Town04/05/06/10HD/12/13 while retaining exact-spawn
and collection-family validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.fixture_runtime import connect_world, restore_async  # noqa: E402
from driving_vla.evaluation.scenario_registry import load_scenario_registry  # noqa: E402
from driving_vla.evaluation.route_authoring_v3 import (  # noqa: E402
    RouteAuthoringError,
    author_route_from_waypoint,
)
from driving_vla.model.navigation_contract import RouteManeuver  # noqa: E402
from r23_author_scenarios import (  # noqa: E402
    _adjacent,
    _crossing_waypoint,
    _forward_chain,
    _pose_lines,
    _velocity_lines,
)
from r2_v4_campaign import CONDITION_TABLE, build_manifest  # noqa: E402


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crossing_waypoint_for_author(world_map: Any, target: Any) -> Any:
    """Find a deterministic perpendicular road actor near a route point.

    The legacy helper uses a 4--16m radius.  Town04/Town10HD have wider
    junction geometry, so keep the same road/heading constraints while using a
    bounded 30m search before declaring the fixture unavailable.
    """
    try:
        return _crossing_waypoint(world_map, target)
    except RuntimeError:
        location = target.transform.location
        yaw = float(target.transform.rotation.yaw)
        candidates = []
        for waypoint in world_map.generate_waypoints(4.0):
            if int(getattr(waypoint, "road_id", 0)) == int(getattr(target, "road_id", 0)):
                continue
            other = waypoint.transform.location
            distance = ((float(other.x) - float(location.x)) ** 2 + (float(other.y) - float(location.y)) ** 2) ** 0.5
            angle = abs((float(waypoint.transform.rotation.yaw) - yaw + 180.0) % 360.0 - 180.0)
            if 4.0 <= distance <= 30.0 and 45.0 <= angle <= 135.0:
                candidates.append((distance, int(getattr(waypoint, "road_id", 0)), int(getattr(waypoint, "lane_id", 0)), waypoint))
        if not candidates:
            raise RuntimeError("no deterministic crossing waypoint near route")
        return min(candidates, key=lambda item: item[:3])[3]


def _ensure_map(map_name: str, *, rhi: str = "dx12") -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/sdf.py"),
        "sim",
        "ensure",
        "--map",
        str(map_name),
        "--rhi",
        str(rhi),
        "--launch-mode",
        "default_engine",
        "--startup-timeout",
        "180",
        "--json",
    ]
    def run_ensure() -> dict[str, Any]:
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        try:
            # ``sdf.py --json`` emits one pretty-printed JSON object.  Parsing
            # only its final ``}`` line loses the object and falsely reports a
            # READY server as a parser failure.
            value = json.loads(completed.stdout.strip())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"CARLA ensure JSON unavailable for {map_name}: {completed.stdout[-500:]}"
            ) from exc
        return value

    value = run_ensure()
    if str(value.get("error_code")) == "MAP_MISMATCH":
        # The protocol forbids client.load_world for formal authoring.  Stop
        # only the exact CARLA server process currently owning the endpoint,
        # then let default_engine cold-start the requested map.  Ambiguous
        # process state fails closed instead of killing an unrelated process.
        query = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^CarlaUE4' } | Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        raw = query.stdout.strip()
        if not raw:
            raise RuntimeError(f"CARLA map mismatch but no listener PID was found: {value}")
        parsed = json.loads(raw)
        pids = [int(parsed)] if isinstance(parsed, int) else [int(item) for item in parsed]
        pids = sorted({pid for pid in pids if pid > 0})
        if not pids:
            raise RuntimeError(f"CARLA map mismatch has no CARLA PIDs: {pids}")
        pid_csv = ",".join(str(pid) for pid in pids)
        stop = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"$ids=@({pid_csv}); foreach ($id in $ids) {{ $p=Get-Process -Id $id -ErrorAction Stop; if ($p.ProcessName -notin @('CarlaUE4','CarlaUE4-Win64-Shipping')) {{ throw 'unexpected process' }}; Stop-Process -Id $id -Force }}",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if stop.returncode != 0:
            raise RuntimeError(f"failed to stop exact CARLA PIDs {pids}: {stop.stderr[-500:]}")
        time.sleep(5.0)
        # A Windows port-proxy can keep returning the old RPC world for a few
        # seconds after the parent/Shipping child exit.  Allow one additional
        # default-engine attempt, still without load_world or a second live
        # server, before failing the map block.
        value = run_ensure()
        if str(value.get("error_code")) == "MAP_MISMATCH":
            time.sleep(5.0)
            value = run_ensure()
    if str(value.get("status")) != "READY":
        raise RuntimeError(f"CARLA ensure failed for {map_name}: {value}")
    return value


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _condition_actor(
    *,
    family: str,
    condition: str,
    route: list[Any],
    world_map: Any,
) -> tuple[Any | None, float, str]:
    """Return actor waypoint, speed and script kind for one condition."""
    if condition == "actor_absent_green":
        return None, 0.0, "hold"
    if condition == "actor_nonconflict":
        return route[-1], 5.0, "hold"
    if family == "crossing":
        return _crossing_waypoint_for_author(world_map, route[2]), 3.5, "reactive_yield"
    if family in {"cut_in", "merge"}:
        adjacent = _adjacent(route[2])
        if adjacent is None:
            raise RuntimeError("no authorized adjacent lane")
        return adjacent, 3.0 if condition == "mild_conflict_left" else 1.5, "reactive_yield"
    if family == "obstruction":
        return route[2], 0.0, "constant_brake"
    # lead braking and the clear control use an ego-lane actor.  The red
    # signal is independently frozen in the scenario traffic_light table.
    return route[2], 4.0 if condition == "mild_conflict_left" else 0.0, (
        "piecewise_vehicle_control" if condition != "red_signal" else "constant_brake"
    )


def _script_lines(prefix: str, family: str, condition: str, speed: float) -> list[str]:
    if condition == "actor_nonconflict":
        return [f"[{prefix}.script]", 'script_type = "hold"']
    if family == "obstruction" or condition == "red_signal":
        return [f"[{prefix}.script]", 'script_type = "constant_brake"', "brake = 1.0"]
    if family == "lead_braking":
        return [
            f"[{prefix}.script]",
            'script_type = "piecewise_vehicle_control"',
            f"[[{prefix}.script.knots]]",
            "t_s = 0.00",
            "throttle = 0.25",
            "brake = 0.00",
            "steer = 0.00",
            f"[[{prefix}.script.knots]]",
            "t_s = 0.80",
            "throttle = 0.00",
            "brake = 0.45",
            "steer = 0.00",
        ]
    return [
        f"[{prefix}.script]",
        'script_type = "reactive_yield"',
        f"desired_speed_mps = {max(0.0, float(speed)):.3f}",
        "yield_ttc_s = 2.500",
        "stop_distance_m = 5.000",
        f"actor_has_priority = {'true' if family == 'crossing' else 'false'}",
    ]


def _render_registry(
    lineage: dict[str, Any],
    world_map: Any,
    route: list[Any],
    route_context: Any,
) -> str:
    map_name = str(lineage["map_name"])
    family = str(lineage["family"])
    lineage_id = str(lineage["lineage_id"])
    lines = [
        "# R2 V4 geometry-only authored registry; outcomes are not consulted.",
        "[registry]",
        'schema_version = "safedrive.g4a.scenario_registry.v1"',
        f'registry_version = "r23-r2v4-{lineage_id}"',
        f'description = "R2 V4 {lineage_id}"',
        "",
        "[defaults]",
        f'map_name = {_toml_string(map_name)}',
        "sim_dt_s = 0.05",
        "duration_s = 5.0",
        'vla_config_ref = "config/vla/k2_v2_spatial.toml"',
        'mpc_config_ref = "config/control/mpc_pid_baseline.toml"',
        'executor_config_ref = "g3_stable_vla_mpc"',
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
    route_points = [
        f"  [{float(w.transform.location.x):.6f}, {float(w.transform.location.y):.6f}, {float(w.transform.location.z + 0.50):.6f}],"
        for w in route
    ]
    frozen_context_json = json.dumps(
        route_context.to_dict(), sort_keys=True, separators=(",", ":")
    )
    for condition in CONDITION_TABLE:
        scenario_id = f"{lineage_id}__{condition}"
        lines.extend(
            [
                f"[scenarios.{scenario_id}]",
                f"family = {_toml_string(family)}",
                f"map_name = {_toml_string(map_name)}",
                f"notes = {_toml_string('condition=' + condition + ' authoring_only=' + json.dumps(CONDITION_TABLE[condition], sort_keys=True))}",
                "",
                f"[scenarios.{scenario_id}.route]",
                f"identity = {_toml_string(lineage_id + '-route')} ",
                "target_speed_mps = 8.0",
                "waypoints = [",
                *route_points,
                "]",
                "",
                f"[scenarios.{scenario_id}.route.navigation_context]",
                f"maneuver = {_toml_string(route_context.maneuver.value)}",
                f"entry_signature = {_toml_string(route_context.entry_signature)}",
                f"exit_signature = {_toml_string(route_context.exit_signature)}",
                f"route_hash = {_toml_string(route_context.route_hash)}",
                f"topology_hash = {_toml_string(route_context.topology_hash)}",
                f"frozen_context_json = {_toml_string(frozen_context_json)}",
                "",
            ]
        )
        for seed in ("seed_a", "seed_b"):
            seed_offset = 0.0 if seed == "seed_a" else 0.35
            prefix = f"scenarios.{scenario_id}.seeds.{seed}.ego"
            ego_wp = route[0]
            lines.extend(
                [
                    f"[{prefix}]",
                    'name = "ego"',
                    'role = "ego"',
                    'blueprint = "vehicle.tesla.model3"',
                    "spawn_order = 0",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(prefix, ego_wp),
                    *_velocity_lines(prefix, ego_wp, 5.5 + seed_offset),
                    f"[{prefix}.script]",
                    'script_type = "hold"',
                    "",
                ]
            )
            actor_wp, speed, _script_kind = _condition_actor(
                family=family, condition=condition, route=route, world_map=world_map
            )
            if actor_wp is None:
                continue
            actor_prefix = f"scenarios.{scenario_id}.seeds.{seed}.actors"
            lines.extend(
                [
                    f"[[{actor_prefix}]]",
                    'name = "conflict_actor"',
                    'role = "npc"',
                    'blueprint = "vehicle.lincoln.mkz_2017"',
                    "spawn_order = 1",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(actor_prefix, actor_wp),
                    *_velocity_lines(actor_prefix, actor_wp, speed),
                    *_script_lines(actor_prefix, family, condition, speed),
                    "",
                ]
            )
            if condition == "red_signal":
                lines.extend(
                    [
                        f"[scenarios.{scenario_id}.seeds.{seed}.traffic_light]",
                        'policy = "freeze_red"',
                        'initial_state = "Red"',
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def author_map(manifest: dict[str, Any], output_root: Path, map_name: str, host: str, port: int) -> dict[str, Any]:
    _ensure_map(map_name, rhi="dx12")
    preflight = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sdf.py"), "sim", "preflight", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    ready = json.loads(preflight.stdout.strip())
    resolved_host = str(ready.get("host") or host)
    client, world = connect_world(host=resolved_host, port=port, map_name=map_name, sim_dt_s=0.05, sync=True, timeout_s=90.0, retries=3)
    rows = [row for row in manifest["lineages"] if str(row["map_name"]) == map_name]
    print(
        json.dumps(
            {"author_map": map_name, "manifest_lineages": len(manifest.get("lineages", [])), "selected_lineages": len(rows)},
            sort_keys=True,
        ),
        flush=True,
    )
    map_root = output_root / map_name
    map_root.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []
    used: set[str] = set()
    try:
        world_map = world.get_map()
        spawn_points = sorted(
            world_map.get_spawn_points(),
            key=lambda t: (round(float(t.location.x), 3), round(float(t.location.y), 3), round(float(t.rotation.yaw), 2)),
        )
        # Spawn points do not cover every legal pre-junction corridor on the
        # larger maps.  Use the deterministic native waypoint inventory first
        # and keep spawn points as a fallback so all eight maneuvers can be
        # authored without sampling from an outcome.
        generated = sorted(
            list(world_map.generate_waypoints(4.0)),
            key=lambda w: (
                int(getattr(w, "road_id", 0)),
                int(getattr(w, "lane_id", 0)),
                round(float(getattr(w, "s", 0.0)), 3),
            ),
        )
        candidates = generated or [
            world_map.get_waypoint(item.location, project_to_road=True)
            for item in spawn_points
        ]
        candidates = [item for item in candidates if item is not None]
        cursor = 0
        for lineage in rows:
            lineage_id = str(lineage["lineage_id"])
            lineage_root = map_root / lineage_id
            registry_path = lineage_root / "scenario_registry.toml"
            author_path = lineage_root / "author_manifest.json"
            if registry_path.exists() and author_path.exists():
                frozen = json.loads(author_path.read_text(encoding="utf-8"))
                if str(frozen.get("source_manifest_hash")) != str(manifest["manifest_hash"]):
                    raise RuntimeError(f"existing author manifest source mismatch: {lineage_id}")
                report_rows.append(frozen["report_row"])
                used.add(str(frozen["topology_sha256"]))
                continue
            if registry_path.exists() or author_path.exists():
                raise RuntimeError(f"partial authoring state: {lineage_root}")
            selected = None
            maneuver = RouteManeuver(str(lineage["route_maneuver"]))
            while cursor < len(candidates) * 2:
                waypoint = candidates[cursor % len(candidates)]
                cursor += 1
                if waypoint is None:
                    continue
                try:
                    authored = author_route_from_waypoint(
                        waypoint,
                        maneuver=maneuver,
                        horizon_m=80.0,
                    )
                    # Rebuild the exact CARLA waypoint sequence from the
                    # frozen route context.  The authorer already selected the
                    # route with native successor decisions; this lookup only
                    # supplies actor spawn transforms for TOML rendering.
                    current = waypoint
                    route = [current]
                    for _ in range(len(authored.waypoints_xyz) - 1):
                        options = list(current.next(8.0))
                        if not options:
                            raise RouteAuthoringError("route successor disappeared")
                        target_xy = authored.waypoints_xyz[len(route)]
                        current = min(
                            options,
                            key=lambda option: (
                                (float(option.transform.location.x) - target_xy[0]) ** 2
                                + (float(option.transform.location.y) - target_xy[1]) ** 2
                            ),
                        )
                        route.append(current)
                    # The CARLA successor lookup above is the exact sequence
                    # that will be rendered into the registry.  Rebind the
                    # frozen context to those emitted XY points so the loader
                    # can prove route/context identity byte-for-byte.
                    emitted_xy = tuple(
                        (
                            float(item.transform.location.x),
                            float(item.transform.location.y),
                        )
                        for item in route
                    )
                    route_context = replace(
                        authored.route_context,
                        route_xy=emitted_xy,
                        origin_lane_centerline_xy=emitted_xy,
                        route_hash="",
                        topology_hash="",
                    )
                    if str(lineage["family"]) in {"cut_in", "merge"} and any(_adjacent(route[i]) is None for i in (2, 3)):
                        continue
                    if str(lineage["family"]) == "crossing":
                        _crossing_waypoint_for_author(world_map, route[2])
                except RuntimeError:
                    continue
                topo_body = {
                    "map": map_name,
                    "lineage_id": lineage_id,
                    "route": [(round(float(w.transform.location.x), 6), round(float(w.transform.location.y), 6)) for w in route],
                    "road_lane": [(int(w.road_id), int(w.lane_id)) for w in route],
                    "route_hash": route_context.route_hash,
                    "maneuver": route_context.maneuver.value,
                }
                topo_hash = hashlib.sha256(json.dumps(topo_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if topo_hash in used:
                    continue
                selected = topo_hash
                break
            if selected is None:
                raise RuntimeError(f"{map_name}: no unique geometry for {lineage_id}")
            used.add(selected)
            lineage_root.mkdir(parents=True, exist_ok=False)
            registry_path.write_text(
                _render_registry(lineage, world_map, route, route_context),
                encoding="utf-8",
            )
            registry = load_scenario_registry(registry_path)
            report_row = {
                "lineage_id": lineage_id,
                "map_name": map_name,
                "family": str(lineage["family"]),
                "route_maneuver": str(lineage["route_maneuver"]),
                "topology_sha256": route_context.topology_hash,
                "registry_path": str(registry_path.resolve()),
                "registry_sha256": str(registry.registry_sha256 or registry.compute_registry_sha256()),
            }
            author_path.write_text(
                json.dumps(
                    {
                        "schema_version": "safedrive.r2.v4.author_manifest.v1",
                        "source_manifest_hash": manifest["manifest_hash"],
                        "lineage_id": lineage_id,
                        "topology_sha256": route_context.topology_hash,
                        "registry_sha256": report_row["registry_sha256"],
                        "registry_file_sha256": _sha_file(registry_path),
                        "report_row": report_row,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            report_rows.append(report_row)
    finally:
        restore_async(world)
    return {"map_name": map_name, "lineages": len(report_rows), "rows": report_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map-only", default="")
    parser.add_argument("--report-name", default="author_report.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if str(manifest.get("manifest_hash") or "") == "":
        raise SystemExit("manifest hash required")
    maps = [str(args.map_only)] if args.map_only else list(manifest["maps"])
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        reports = []
        for map_name in maps:
            rows = []
            for lineage in manifest["lineages"]:
                if str(lineage["map_name"]) != map_name:
                    continue
                lineage_id = str(lineage["lineage_id"])
                author_path = output_root / map_name / lineage_id / "author_manifest.json"
                if not author_path.exists():
                    raise SystemExit(f"missing author manifest: {author_path}")
                frozen = json.loads(author_path.read_text(encoding="utf-8"))
                if str(frozen.get("source_manifest_hash")) != str(manifest["manifest_hash"]):
                    raise SystemExit(f"author manifest source mismatch: {author_path}")
                rows.append(frozen["report_row"])
            if len(rows) != sum(1 for row in manifest["lineages"] if str(row["map_name"]) == map_name):
                raise SystemExit(f"author manifest coverage mismatch for {map_name}: {len(rows)}")
            reports.append({"map_name": map_name, "lineages": len(rows), "rows": rows})
    else:
        reports = [author_map(manifest, output_root, map_name, args.host, args.port) for map_name in maps]
    report = {
        "schema_version": "safedrive.r2.v4.author_report.v1",
        "source_manifest_hash": manifest["manifest_hash"],
        "maps": reports,
        "lineages": sum(int(row["lineages"]) for row in reports),
    }
    output = output_root / str(args.report_name)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit(f"existing author report mismatch: {output}")
    else:
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "maps": len(reports), "lineages": report["lineages"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
