#!/usr/bin/env python3
"""Stable pure-VLA spatial path → constrained MPC CARLA demonstration.

The CARLA map route is used only to form two coarse navigation targets for the
VLA.  The tracked path geometry comes exclusively from SimLingo.  Inference is
serialized with CARLA rendering by default to reduce D3D/CUDA contention on a
single Windows/WSL GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simlingo-main"))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import carla  # noqa: E402

from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.neural_policy import NeuralV0Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_XYZ,
    SimLingoNeuralRuntime,
)
from driving_vla.runtime.path_manager import (  # noqa: E402
    EgoPose,
    PathManagerConfig,
    SpatialPath,
    VLAPathManager,
)
from driving_vla.runtime.vla_mpc_tracker import (  # noqa: E402
    ConstrainedVLAMPC,
    VLAMPCConfig,
)
from driving_vla.runtime.vla_speed_planner import (  # noqa: E402
    VLASpeedConfig,
    VLASpeedPlanner,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402
from run_g3_vla_v0_visual_demo import build_route, free_spawn, set_spectator_follow  # noqa: E402

# Playable full-map packages (exclude multi-tile sublevels). Selected only at process start.
DEMO_MAP_POOL: tuple[str, ...] = (
    "Town01",
    "Town02",
    "Town03",
    "Town03_Opt",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town10HD_Opt",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
)
# Random endurance runs intentionally avoid the three very large tiled towns:
# they have a much larger startup/failure surface and are opt-in by name.
# Prefer Town03_Opt over full Town03 when sampling randomly (full Town03 has
# hit startup ACCESS_VIOLATION on this host/driver combination).
DEFAULT_RANDOM_MAP_POOL: tuple[str, ...] = (
    "Town03_Opt",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town10HD_Opt",
)
CARLA_DEFAULT_ENGINE_INI = Path("/mnt/e/CARLA_0.9.16/CarlaUE4/Config/DefaultEngine.ini")
CARLA_MAPS_CONTENT = Path("/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps")
CARLA_START_TOML = ROOT / "safedrive_foundry" / "config" / "runtime" / "carla_start.toml"
# Large layered towns ship as Content/Carla/Maps/<Town>/<Town>.umap (not flat Maps/<Town>.umap).
NESTED_MAP_PACKAGE: frozenset[str] = frozenset({"Town11", "Town12", "Town13", "Town15"})
# CARLA Large Maps use tile streaming and require a hero actor near the active tile.
LARGE_MAPS: frozenset[str] = frozenset({"Town11", "Town12", "Town13"})


def _write_json(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _polyline_s(points: list[tuple[float, float]]) -> np.ndarray:
    xy = np.asarray(points, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))))


def _point_at_s(points: list[tuple[float, float]], s: np.ndarray, query: float) -> tuple[float, float]:
    query = float(np.clip(query, 0.0, float(s[-1])))
    return float(np.interp(query, s, [p[0] for p in points])), float(
        np.interp(query, s, [p[1] for p in points])
    )


def _project_route_s(
    points: list[tuple[float, float]], s: np.ndarray, x: float, y: float, hint_s: float
) -> float:
    best_s, best_d2 = hint_s, float("inf")
    lo, hi = max(0.0, hint_s - 2.0), min(float(s[-1]), hint_s + 40.0)
    for i in range(len(points) - 1):
        if s[i + 1] < lo or s[i] > hi:
            continue
        a = np.asarray(points[i], dtype=float)
        b = np.asarray(points[i + 1], dtype=float)
        p = np.asarray([x, y], dtype=float)
        ab = b - a
        denom = float(ab @ ab)
        u = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        q = a + u * ab
        d2 = float((p - q) @ (p - q))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s[i] + u * (s[i + 1] - s[i]))
    return max(hint_s, best_s)


def _navigation_targets(
    route: list[tuple[float, float]],
    route_s: np.ndarray,
    ego: EgoPose,
    progress_s: float,
) -> tuple[tuple[float, float], tuple[float, float], float, bool]:
    progress = _project_route_s(route, route_s, ego.x, ego.y, progress_s)
    p1 = _point_at_s(route, route_s, progress + 15.0)
    p2 = _point_at_s(route, route_s, progress + 30.0)
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)

    def to_ego(point: tuple[float, float]) -> tuple[float, float]:
        dx, dy = point[0] - ego.x, point[1] - ego.y
        return c * dx + s * dy, -s * dx + c * dy

    t1, t2 = to_ego(p1), to_ego(p2)
    valid = t1[0] >= 1.0 and t2[0] >= t1[0] + 2.0
    return t1, t2, progress, valid


def _build_route_segment(
    world: carla.World,
    start: carla.Transform,
    length_m: float,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    points = [(float(x), float(y)) for x, y in build_route(world, start, length_m)]
    if len(points) < 2:
        raise RuntimeError("route builder returned fewer than two points")
    cumulative = _polyline_s(points)
    if float(cumulative[-1]) < 35.0:
        raise RuntimeError(f"route segment is only {float(cumulative[-1]):.1f}m")
    return points, cumulative


def _vehicle_geometry(vehicle: carla.Vehicle) -> tuple[float, float]:
    """Read wheelbase and maximum front-wheel angle from CARLA when available."""
    fallback = (2.70, 0.60)
    try:
        wheels = list(vehicle.get_physics_control().wheels)
        if not wheels:
            return fallback
        steer_deg = max(float(w.max_steer_angle) for w in wheels)
        xs = [float(w.position.x) for w in wheels if hasattr(w, "position")]
        wheelbase = (max(xs) - min(xs)) / 100.0 if len(xs) >= 2 else fallback[0]
        if not 1.5 <= wheelbase <= 4.0:
            wheelbase = fallback[0]
        steer_rad = math.radians(steer_deg) if steer_deg > 1.0 else fallback[1]
        if not 0.20 <= steer_rad <= 1.0:
            steer_rad = fallback[1]
        return wheelbase, steer_rad
    except Exception:
        return fallback


def _ego_pose(vehicle: carla.Vehicle) -> EgoPose:
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    return EgoPose(
        x=float(tf.location.x),
        y=float(tf.location.y),
        yaw=math.radians(float(tf.rotation.yaw)),
        speed_mps=math.hypot(float(vel.x), float(vel.y)),
    )


def _draw_path(
    world: carla.World,
    raw: SpatialPath | None,
    committed: SpatialPath | None,
    *,
    life_s: float,
) -> None:
    debug = world.debug
    for path, color, z, thickness in (
        (raw, carla.Color(0, 255, 0), 0.55, 0.08),
        (committed, carla.Color(255, 220, 0), 0.70, 0.14),
    ):
        if path is None:
            continue
        stride = max(1, int(round(0.4 / max(float(np.median(np.diff(path.s))), 0.1))))
        points = path.as_xy()[::stride]
        for a, b in zip(points, points[1:]):
            debug.draw_line(
                carla.Location(a[0], a[1], z),
                carla.Location(b[0], b[1], z),
                thickness=thickness,
                color=color,
                life_time=life_s,
            )


def _spawn_ego(world: carla.World, *, large_map: bool) -> tuple[carla.Vehicle, carla.Transform]:
    spawn = free_spawn(world)
    blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
    if blueprint is None:
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    if large_map:
        # A spectator close to the chosen spawn asks the server to stream that
        # tile before the hero exists.  Without this, Town13 may spawn against
        # an unloaded tile and crash inside Unreal with address 0x8.
        spectator_tf = carla.Transform(
            carla.Location(spawn.location.x - 8.0, spawn.location.y, spawn.location.z + 8.0),
            carla.Rotation(pitch=-25.0, yaw=spawn.rotation.yaw),
        )
        world.get_spectator().set_transform(spectator_tf)
        for _ in range(30):
            world.tick()
    for transform in [spawn, *world.get_map().get_spawn_points()[:40]]:
        lifted = carla.Transform(
            carla.Location(transform.location.x, transform.location.y, transform.location.z + 0.5),
            transform.rotation,
        )
        actor = world.try_spawn_actor(blueprint, lifted)
        if actor is not None:
            return actor, lifted
    raise RuntimeError("unable to spawn ego vehicle")


def _map_content_path(map_name: str) -> str:
    """Unreal content path without the trailing asset name suffix.

    Flat towns:  /Game/Carla/Maps/Town04
    Nested:      /Game/Carla/Maps/Town13/Town13
    """
    if map_name in NESTED_MAP_PACKAGE or (CARLA_MAPS_CONTENT / map_name / f"{map_name}.umap").is_file():
        return f"/Game/Carla/Maps/{map_name}/{map_name}"
    return f"/Game/Carla/Maps/{map_name}"


def _map_asset_path(map_name: str) -> str:
    """Full DefaultEngine map asset id: /Game/.../Town13.Town13"""
    content = _map_content_path(map_name)
    return f"{content}.{map_name}"


def _write_startup_map(map_name: str) -> None:
    """Pin CARLA process-start map via DefaultEngine.ini + carla_start.toml.

    Mid-session client.load_world is intentionally avoided (D3D/shader risk).
    Command-line map tokens alone are often ignored by this install; the engine
    defaults must match the requested town.
    """
    if map_name not in DEMO_MAP_POOL:
        raise ValueError(f"unsupported map {map_name!r}; pool={DEMO_MAP_POOL}")
    content = _map_content_path(map_name)
    asset = _map_asset_path(map_name)
    umap = CARLA_MAPS_CONTENT / map_name / f"{map_name}.umap"
    if not umap.is_file():
        umap = CARLA_MAPS_CONTENT / f"{map_name}.umap"
    if not umap.is_file():
        raise FileNotFoundError(f"map package missing on disk for {map_name}: expected under {CARLA_MAPS_CONTENT}")
    if CARLA_DEFAULT_ENGINE_INI.is_file():
        text = CARLA_DEFAULT_ENGINE_INI.read_text(encoding="utf-8", errors="replace")
        for key in (
            "EditorStartupMap",
            "GameDefaultMap",
            "ServerDefaultMap",
            "TransitionMap",
        ):
            text = re.sub(
                rf"(?m)^({key})=.*$",
                rf"\1={asset}",
                text,
            )
        CARLA_DEFAULT_ENGINE_INI.write_text(text, encoding="utf-8")
    if CARLA_START_TOML.is_file():
        toml = CARLA_START_TOML.read_text(encoding="utf-8", errors="replace")
        args_line = (
            f'arguments = "{content} -windowed -ResX=800 -ResY=600 '
            f'-quality-level=Low -nosound -dx11 -carla-rpc-port=2000"'
        )
        if re.search(r'(?m)^arguments\s*=', toml):
            toml = re.sub(r'(?m)^arguments\s*=\s*".*"$', args_line, toml)
        else:
            toml = toml.rstrip() + "\n" + args_line + "\n"
        CARLA_START_TOML.write_text(toml, encoding="utf-8")
    print(f"startup map pinned: {map_name} content={content} asset={asset} umap={umap}", flush=True)


def _kill_carla_windows() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        print("WARN: powershell not found; cannot stop CarlaUE4 from WSL", flush=True)
        return
    cmd = (
        "Get-Process -Name 'CarlaUE4*','UE4Editor*' -ErrorAction SilentlyContinue "
        "| Stop-Process -Force -ErrorAction SilentlyContinue"
    )
    try:
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", cmd],
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARN: Carla stop failed: {exc}", flush=True)
    time.sleep(4.0)


def _map_matches(current_map: str, required: str) -> bool:
    return required in current_map or current_map.endswith(required)


def _resolve_requested_map(map_arg: str, *, seed: int | None) -> str:
    token = str(map_arg).strip()
    if token.lower() in {"random", "rand", "*"}:
        rng = random.Random(seed) if seed is not None else random.SystemRandom()
        choice = rng.choice(list(DEFAULT_RANDOM_MAP_POOL))
        print(
            f"random stable map selected: {choice} from {len(DEFAULT_RANDOM_MAP_POOL)} towns",
            flush=True,
        )
        return choice
    if token not in DEMO_MAP_POOL:
        raise SystemExit(
            f"unknown map {token!r}; use one of {list(DEMO_MAP_POOL)} or --map random"
        )
    return token


def _ensure_ready_map(resolver: ConnectionResolver, map_name: str, *, startup_timeout_s: float) -> Any:
    report = resolver.preflight()
    if report.status == "READY" and _map_matches(str(report.map or ""), map_name):
        print(f"preflight READY on required map={report.map}", flush=True)
        return report
    if report.status == "READY":
        print(
            f"map mismatch current={report.map} required={map_name}; cold-restart CarlaUE4",
            flush=True,
        )
    else:
        print(
            f"preflight {report.status}/{report.error_code}; cold-start map={map_name}",
            flush=True,
        )
    _write_startup_map(map_name)
    _kill_carla_windows()
    report = resolver.ensure(startup_timeout_seconds=float(startup_timeout_s))
    if report.status != "READY":
        print("ensure not READY", report.status, report.error_code, report.error_message, flush=True)
        return report
    if not _map_matches(str(report.map or ""), map_name):
        print(
            f"MAP_MISMATCH after ensure current={report.map} required={map_name}",
            flush=True,
        )
        report.status = "BLOCKED"
        return report
    print(f"ensure READY map={report.map}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable pure VLA spatial path + constrained MPC")
    parser.add_argument(
        "--map",
        default="random",
        help="Town name, or 'random' to pick from the stable non-Large-Map pool (default: random)",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--vla-period-s", type=float, default=0.50)
    parser.add_argument(
        "--v-ref",
        "--max-speed",
        dest="v_ref",
        type=float,
        default=15.0,
        help="Absolute speed cap in m/s, not a forced cruise speed (default: 15.0).",
    )
    parser.add_argument(
        "--speed-gain",
        type=float,
        default=1.50,
        help="Multiplicative calibration for the VLA speed head before the cap (default: 1.50).",
    )
    parser.add_argument(
        "--route-segment-m",
        type=float,
        default=600.0,
        help="Rolling coarse-navigation segment length; refreshed before exhaustion.",
    )
    parser.add_argument(
        "--checkpoint-period-s",
        type=float,
        default=60.0,
        help="Write progress_latest.json at this simulated-time interval.",
    )
    parser.add_argument("--cam-w", type=int, default=768)
    parser.add_argument("--cam-h", type=int, default=384)
    parser.add_argument("--steer-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --map random")
    parser.add_argument("--full-duration", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-debug-draw", action="store_true")
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=180.0,
        help="Carla cold-start handshake timeout after map pin",
    )
    parser.add_argument(
        "--no-map-restart",
        action="store_true",
        help="Do not kill/restart CARLA; fail on MAP_MISMATCH instead",
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(ROOT / "docs/architecture/evidence/g3-05/vla_mpc_stable"),
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if args.sim_dt <= 0.0:
        parser.error("--sim-dt must be positive")
    if args.vla_period_s <= 0.0:
        parser.error("--vla-period-s must be positive")
    if args.v_ref < 0.0:
        parser.error("--v-ref speed cap must be non-negative")
    if args.speed_gain < 0.0:
        parser.error("--speed-gain must be non-negative")
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    requested_map = _resolve_requested_map(args.map, seed=args.seed)
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=15.0)
    if args.no_map_restart:
        report = resolver.preflight()
        if report.status != "READY":
            print("preflight not READY", report.status, report.error_code, flush=True)
            return 75
        if not _map_matches(str(report.map or ""), requested_map):
            print(
                f"MAP_MISMATCH current={report.map} required={requested_map}",
                flush=True,
            )
            return 75
    else:
        report = _ensure_ready_map(
            resolver, requested_map, startup_timeout_s=float(args.startup_timeout_s)
        )
        if getattr(report, "status", None) != "READY":
            return 75
    client, report = resolver.connect(report=report)
    # Large layered towns need a long RPC budget; 20s is enough for Town0x/10 only.
    rpc_timeout_s = 120.0 if requested_map in NESTED_MAP_PACKAGE else 30.0
    client.set_timeout(rpc_timeout_s)
    world = client.get_world()
    current_map = str(world.get_map().name)
    if not _map_matches(current_map, requested_map):
        print(f"MAP_MISMATCH current={current_map} required={requested_map}", flush=True)
        print("Cold-restart CarlaUE4 with the required startup map.", flush=True)
        return 75

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.sim_dt)
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = max(2, int(math.ceil(float(args.sim_dt) / 0.01)))
    if requested_map in LARGE_MAPS:
        if hasattr(settings, "tile_stream_distance"):
            settings.tile_stream_distance = 2000.0
        if hasattr(settings, "actor_active_distance"):
            settings.actor_active_distance = 2000.0
        if hasattr(settings, "spectator_as_ego"):
            settings.spectator_as_ego = True
    world.apply_settings(settings)
    # Give streaming maps a few synchronous ticks before actor surgery.
    for _ in range(20 if requested_map in NESTED_MAP_PACKAGE else 5):
        world.tick()

    ego: carla.Vehicle | None = None
    camera: carla.Sensor | None = None
    collision_sensor: carla.Sensor | None = None
    lane_sensor: carla.Sensor | None = None
    try:
        for actor_filter in ("sensor.*", "vehicle.*"):
            for actor in list(world.get_actors().filter(actor_filter)):
                try:
                    actor.destroy()
                except Exception as exc:  # pragma: no cover - live only
                    print(f"WARN destroy {actor_filter} id={getattr(actor, 'id', '?')}: {exc}", flush=True)
        ego, spawn = _spawn_ego(world, large_map=requested_map in LARGE_MAPS)
        if requested_map in LARGE_MAPS and hasattr(settings, "spectator_as_ego"):
            settings.spectator_as_ego = False
            world.apply_settings(settings)
        for _ in range(12):
            ego.apply_control(carla.VehicleControl(brake=0.5))
            world.tick()

        route_xy, route_s = _build_route_segment(
            world,
            spawn,
            max(100.0, float(args.route_segment_m)),
        )
        route_progress = 0.0
        completed_route_progress = 0.0
        route_refreshes = 0
        route_refresh_failures = 0

        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.cam_w))
        cam_bp.set_attribute("image_size_y", str(args.cam_h))
        cam_bp.set_attribute("fov", str(SIMLINGO_CAMERA_FOV_DEG))
        cam_bp.set_attribute("sensor_tick", "0.10")
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(*SIMLINGO_CAMERA_XYZ)),
            attach_to=ego,
        )
        collision_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=ego,
        )
        lane_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=ego,
        )
        image_lock = threading.Lock()
        latest_image: dict[str, Any] = {"rgb": None, "frame": -1}
        event_lock = threading.Lock()
        road_events: dict[str, Any] = {
            "collisions": 0,
            "collision_impulse": 0.0,
            "lane_invasions": 0,
        }

        def on_image(image: carla.Image) -> None:
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
            with image_lock:
                latest_image["rgb"] = rgb
                latest_image["frame"] = int(image.frame)

        camera.listen(on_image)

        def on_collision(event: carla.CollisionEvent) -> None:
            impulse = event.normal_impulse
            magnitude = math.sqrt(float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2)
            with event_lock:
                road_events["collisions"] += 1
                road_events["collision_impulse"] += magnitude

        def on_lane_invasion(_event: carla.LaneInvasionEvent) -> None:
            with event_lock:
                road_events["lane_invasions"] += 1

        collision_sensor.listen(on_collision)
        lane_sensor.listen(on_lane_invasion)
        for _ in range(10):
            ego.apply_control(carla.VehicleControl(brake=0.5))
            world.tick()

        import torch

        runtime = SimLingoNeuralRuntime(device="cuda")
        load = runtime.load()
        if not load.ok:
            raise RuntimeError(f"SimLingo load failed: {load.error}")
        policy = NeuralV0Policy(runtime=runtime, keep_on_gpu=True)
        policy.ensure_loaded()
        wheelbase, max_steer_rad = _vehicle_geometry(ego)
        print(
            f"map={current_map} camera={args.cam_w}x{args.cam_h}@{SIMLINGO_CAMERA_XYZ} "
            f"wheelbase={wheelbase:.3f} max_steer={math.degrees(max_steer_rad):.1f}deg "
            f"cuda_alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB",
            flush=True,
        )

        path_manager = VLAPathManager(
            PathManagerConfig(
                # Align with unit-test tolerance: raw 20-pt polylines often show
                # κ≈0.25–0.35 from point noise even on gentle roads.
                max_abs_curvature=0.30,
                max_switch_lateral_5m=1.0,
                max_switch_heading_5m_deg=12.0,
            )
        )
        speed_cap = max(0.0, float(args.v_ref))
        speed_planner = VLASpeedPlanner(
            VLASpeedConfig(
                max_speed_mps=speed_cap,
                calibration_gain=max(0.0, float(args.speed_gain)),
                max_accel_mps2=2.50,
            )
        )
        tracker = ConstrainedVLAMPC(
            VLAMPCConfig(
                control_dt_s=float(args.sim_dt),
                prediction_dt_s=0.10,
                horizon=20,
                wheelbase_m=wheelbase,
                max_steer_rad=max_steer_rad,
                max_speed_mps=speed_cap,
                max_accel_mps2=2.50,
                max_lateral_accel_mps2=1.50,
                solver_deadline_ms=30.0,
            )
        )

        sim_start = float(world.get_snapshot().timestamp.elapsed_seconds)
        next_inference_sim_s = sim_start
        last_speed_update_sim_s = sim_start
        next_checkpoint_sim_s = sim_start + max(1.0, float(args.checkpoint_period_s))
        last_xy = (_ego_pose(ego).x, _ego_pose(ego).y)
        distance_m = 0.0
        steps = 0
        accepts = 0
        reject_reasons: Counter[str] = Counter()
        solver_modes: Counter[str] = Counter()
        solver_statuses: Counter[str] = Counter()
        steer_values: list[float] = []
        cte_values: list[float] = []
        actual_speed_values: list[float] = []
        target_speed_values: list[float] = []
        path_age_values: list[float] = []
        peak_vram_values: list[float] = []
        inference_ms: list[float] = []
        events: list[dict[str, Any]] = []
        spectator_state = {"x": last_xy[0], "y": last_xy[1], "z": 1.0, "yaw": 0.0}
        previous_steer_norm = 0.0
        previous_sign = 0
        sign_flips = 0
        offroad_steps = 0
        route_target_invalid = 0
        next_route_retry_sim_s = sim_start
        carla_map = world.get_map()

        while True:
            loop_wall = time.perf_counter()
            snapshot = world.get_snapshot()
            sim_s = float(snapshot.timestamp.elapsed_seconds)
            if sim_s - sim_start >= float(args.duration_s):
                break
            pose = _ego_pose(ego)
            actual_speed_values.append(pose.speed_mps)
            distance_m += math.hypot(pose.x - last_xy[0], pose.y - last_xy[1])
            last_xy = (pose.x, pose.y)
            route_progress = _project_route_s(route_xy, route_s, pose.x, pose.y, route_progress)
            try:
                driving_waypoint = carla_map.get_waypoint(
                    ego.get_location(),
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving,
                )
                if driving_waypoint is None:
                    offroad_steps += 1
            except RuntimeError:
                offroad_steps += 1

            if (
                float(route_s[-1]) - route_progress < 60.0
                and sim_s >= next_route_retry_sim_s
            ):
                try:
                    refreshed_xy, refreshed_s = _build_route_segment(
                        world,
                        ego.get_transform(),
                        max(100.0, float(args.route_segment_m)),
                    )
                    completed_route_progress += route_progress
                    route_xy, route_s = refreshed_xy, refreshed_s
                    route_progress = 0.0
                    route_refreshes += 1
                    print(
                        f"route refreshed n={route_refreshes} length={float(route_s[-1]):.1f}m",
                        flush=True,
                    )
                except Exception as exc:
                    route_refresh_failures += 1
                    next_route_retry_sim_s = sim_s + 5.0
                    print(f"WARN route refresh failed: {exc}", flush=True)

            if sim_s >= next_inference_sim_s:
                with image_lock:
                    image = None if latest_image["rgb"] is None else latest_image["rgb"].copy()
                    camera_frame = int(latest_image["frame"])
                if image is not None:
                    tp1, tp2, route_progress, navigation_valid = _navigation_targets(
                        route_xy, route_s, pose, route_progress
                    )
                    if not navigation_valid:
                        route_target_invalid += 1
                        print(
                            f"WARN coarse route target is behind ego; hold last VLA path "
                            f"tp1=({tp1[0]:.1f},{tp1[1]:.1f})",
                            flush=True,
                        )
                    else:
                        obs = ObservationBundle(
                            run_id="g3_vla_mpc_stable",
                            frame_id=f"carla-{camera_frame}",
                            scenario_id="pure_vla_straight_arc",
                            simulation_time_s=sim_s,
                            wall_time_s=time.time(),
                            carla_frame=camera_frame,
                            ego_x=pose.x,
                            ego_y=pose.y,
                            ego_yaw=pose.yaw,
                            ego_v=pose.speed_mps,
                            route_xy=tuple(route_xy),
                            front_rgb=image,
                            meta={"target_ego_1": tp1, "target_ego_2": tp2},
                        )
                        infer_start = time.perf_counter()
                        native = policy.predict_native(obs)
                        torch.cuda.synchronize()
                        latency_ms = (time.perf_counter() - infer_start) * 1000.0
                        inference_ms.append(latency_ms)
                        peak_vram_values.append(float(native.peak_vram_mb))
                        speed = speed_planner.update(
                            native.speed_mps,
                            dt_s=max(float(args.sim_dt), sim_s - last_speed_update_sim_s),
                        )
                        last_speed_update_sim_s = sim_s
                        update = path_manager.update(
                            native.path_map_xy,
                            ego=pose,
                            target_speed_mps=speed.target_speed_mps,
                            stamp_s=sim_s,
                            source_id=f"simlingo-{camera_frame}",
                        )
                        if update.accepted:
                            accepts += 1
                        else:
                            reject_reasons[update.reason] += 1
                        event = {
                            "sim_s": sim_s,
                            "camera_frame": camera_frame,
                            "latency_ms": latency_ms,
                            "peak_vram_mb": float(native.peak_vram_mb),
                            "target_ego_1": list(native.target_ego_1),
                            "target_ego_2": list(native.target_ego_2),
                            "vla_speed_raw_mps": speed.raw_speed_mps,
                            "vla_speed_calibrated_mps": speed.calibrated_speed_mps,
                            "desired_speed_mps": speed.target_speed_mps,
                            "vla_stop_requested": speed.stop_requested,
                            "accepted": update.accepted,
                            "reason": update.reason,
                            "quality": update.quality.__dict__,
                        }
                        events.append(event)
                        if not args.no_debug_draw:
                            _draw_path(
                                world,
                                update.raw,
                                update.committed,
                                life_s=float(args.vla_period_s) + 0.15,
                            )
                        print(
                            f"VLA n={len(events)} {update.reason} infer={latency_ms:.0f}ms "
                            f"tp1=({tp1[0]:.1f},{tp1[1]:.1f}) "
                            f"v_raw={speed.raw_speed_mps:.2f} v_cmd={speed.target_speed_mps:.2f} "
                            f"jump5={update.quality.switch_lateral_5m:.2f}m",
                            flush=True,
                        )
                next_inference_sim_s = sim_s + float(args.vla_period_s)

            committed = path_manager.committed
            if committed is None:
                ego.apply_control(carla.VehicleControl(brake=1.0))
            else:
                command = tracker.step(
                    committed,
                    pose,
                    measured_steer_rad=previous_steer_norm * max_steer_rad * float(args.steer_sign),
                    now_s=sim_s,
                )
                solver_modes[command.mode] += 1
                solver_statuses[command.solver_status] += 1
                steer_norm = float(
                    np.clip(
                        float(args.steer_sign) * command.steer_rad / max(max_steer_rad, 1e-6),
                        -1.0,
                        1.0,
                    )
                )
                accel = command.accel_mps2
                throttle = float(np.clip(accel / 2.5, 0.0, 1.0))
                brake = float(np.clip(-accel / 3.0, 0.0, 1.0))
                ego.apply_control(carla.VehicleControl(steer=steer_norm, throttle=throttle, brake=brake))
                previous_steer_norm = steer_norm
                steer_values.append(steer_norm)
                cte_values.append(abs(command.lateral_error_m))
                target_speed_values.append(command.target_speed_mps)
                path_age_values.append(command.path_age_s)
                sign = 1 if steer_norm > 0.05 else (-1 if steer_norm < -0.05 else 0)
                if sign and previous_sign and sign != previous_sign:
                    sign_flips += 1
                if sign:
                    previous_sign = sign

            world.tick()
            if steps % 4 == 0:
                set_spectator_follow(world, ego, last_good=spectator_state)
            steps += 1
            if sim_s >= next_checkpoint_sim_s:
                with event_lock:
                    checkpoint_road_events = dict(road_events)
                checkpoint = {
                    "status": "RUNNING",
                    "map": current_map,
                    "sim_elapsed_s": sim_s - sim_start,
                    "requested_duration_s": float(args.duration_s),
                    "distance_m": distance_m,
                    "route_progress_m": completed_route_progress + route_progress,
                    "route_refreshes": route_refreshes,
                    "vla_updates": len(events),
                    "vla_accepts": accepts,
                    "actual_speed_mps": pose.speed_mps,
                    "vla_speed_target_mps": speed_planner.target_speed_mps,
                    "road_events": checkpoint_road_events,
                    "offroad_fraction": offroad_steps / max(steps, 1),
                    "wall_time_s": time.time(),
                }
                _write_json(evidence_dir / "progress_latest.json", checkpoint)
                next_checkpoint_sim_s += max(1.0, float(args.checkpoint_period_s))
            remaining = float(args.sim_dt) - (time.perf_counter() - loop_wall)
            if remaining > 0.0:
                time.sleep(remaining)

        displacement = math.hypot(last_xy[0] - spawn.location.x, last_xy[1] - spawn.location.y)
        mpc_steps = int(solver_modes.get("mpc", 0))
        cte_rms_value = (
            float(math.sqrt(np.mean(np.square(cte_values)))) if cte_values else float("inf")
        )
        saturation_fraction = (
            float(np.mean(np.abs(np.asarray(steer_values)) > 0.95)) if steer_values else 1.0
        )
        distance_ratio = distance_m / max(displacement, 1e-3)
        total_route_progress = completed_route_progress + route_progress
        route_progress_efficiency = min(1.0, total_route_progress / max(distance_m, 1e-3))
        offroad_fraction = offroad_steps / max(steps, 1)
        with event_lock:
            final_road_events = dict(road_events)
        minimum_distance_m = max(20.0, min(100.0, float(args.duration_s)))
        acceptance = {
            "enough_vla_paths": accepts >= max(2, int(math.ceil(0.5 * max(len(events), 1)))),
            "mpc_fraction_ge_0_95": mpc_steps >= 0.95 * max(len(steer_values), 1),
            "cte_rms_lt_0_50m": cte_rms_value < 0.50,
            "steer_saturation_lt_0_01": saturation_fraction < 0.01,
            "steer_flip_rate_lt_0_5hz": sign_flips / max(float(args.duration_s), 1.0) < 0.5,
            "route_progress_efficiency_ge_0_65": route_progress_efficiency >= 0.65,
            "collision_free": int(final_road_events["collisions"]) == 0,
            "offroad_fraction_lt_0_02": offroad_fraction < 0.02,
            "minimum_distance": distance_m > minimum_distance_m,
        }
        demo_pass = all(acceptance.values())
        summary = {
            "demo": "pure_vla_spatial_constrained_mpc_v1",
            "map": current_map,
            "requested_map": requested_map,
            "map_arg": str(args.map),
            "speed_cap_mps": speed_cap,
            "speed_calibration_gain": float(args.speed_gain),
            "speed_semantics": "VLA speed head is primary; calibrated then capped; no positive speed floor",
            "steps": steps,
            "sim_duration_s": float(args.duration_s),
            "distance_m": distance_m,
            "displacement_m": displacement,
            "distance_over_displacement": distance_ratio,
            "route": {
                "coarse_navigation_only": True,
                "progress_m": total_route_progress,
                "progress_efficiency": route_progress_efficiency,
                "refreshes": route_refreshes,
                "refresh_failures": route_refresh_failures,
                "target_invalid_count": route_target_invalid,
                "segment_length_m": float(args.route_segment_m),
            },
            "camera": {
                "width": args.cam_w,
                "height": args.cam_h,
                "fov_deg": SIMLINGO_CAMERA_FOV_DEG,
                "mount_xyz": list(SIMLINGO_CAMERA_XYZ),
            },
            "vehicle": {"wheelbase_m": wheelbase, "max_steer_rad": max_steer_rad},
            "vla_updates": len(events),
            "vla_accepts": accepts,
            "vla_reject_reasons": dict(reject_reasons),
            "inference_ms": {
                "p50": float(np.percentile(inference_ms, 50)) if inference_ms else 0.0,
                "p95": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
            },
            "speed_mps": {
                "actual_p50": float(np.percentile(actual_speed_values, 50)) if actual_speed_values else 0.0,
                "actual_p95": float(np.percentile(actual_speed_values, 95)) if actual_speed_values else 0.0,
                "actual_max": max(actual_speed_values, default=0.0),
                "controller_target_p50": float(np.percentile(target_speed_values, 50)) if target_speed_values else 0.0,
                "controller_target_p95": float(np.percentile(target_speed_values, 95)) if target_speed_values else 0.0,
            },
            "path_age_s": {
                "p95": float(np.percentile(path_age_values, 95)) if path_age_values else 0.0,
                "max": max(path_age_values, default=0.0),
            },
            "peak_vram_mb": max(peak_vram_values, default=0.0),
            "road_events": final_road_events,
            "offroad_steps": offroad_steps,
            "offroad_fraction": offroad_fraction,
            "solver_modes": dict(solver_modes),
            "solver_statuses": dict(solver_statuses),
            # Keep evidence strict JSON even when no controller sample was produced.
            "cte_rms_m": cte_rms_value if math.isfinite(cte_rms_value) else None,
            "steer_abs_max": max((abs(v) for v in steer_values), default=0.0),
            "steer_saturation_fraction": saturation_fraction,
            "steer_sign_flips": sign_flips,
            "geometry_source": "SimLingo native path only; map route used for coarse VLA targets",
            "gpu_schedule": "serialized inference; torch.cuda.synchronize before next world.tick",
            "acceptance": acceptance,
            "result": "DEMO_PASS" if demo_pass else "DEMO_FAIL",
            "verified": False,
        }
        stamp = int(time.time() * 1000)
        _write_json(evidence_dir / f"summary_{stamp}.json", summary)
        _write_json(evidence_dir / "latest_summary.json", summary)
        _write_json(evidence_dir / f"vla_events_{stamp}.json", events)
        _write_json(
            evidence_dir / "progress_latest.json",
            {
                "status": "COMPLETE",
                "result": summary["result"],
                "map": current_map,
                "sim_elapsed_s": float(args.duration_s),
                "distance_m": distance_m,
                "route_progress_m": total_route_progress,
                "road_events": final_road_events,
                "offroad_fraction": offroad_fraction,
                "summary_file": f"summary_{stamp}.json",
            },
        )
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if demo_pass else 2
    except Exception as exc:  # pragma: no cover - requires live CARLA failure
        failure = {
            "status": "BLOCKED_EXTERNAL",
            "result": "RUNTIME_ERROR",
            "requested_map": requested_map,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_time_s": time.time(),
        }
        _write_json(evidence_dir / "failure_latest.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 75
    finally:
        for sensor in (lane_sensor, collision_sensor):
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except Exception:
                pass
        if ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
