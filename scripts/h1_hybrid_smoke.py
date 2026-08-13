#!/usr/bin/env python3
"""Bounded Town03 H1 smoke: same-frame Expert/VLA → Guard → Safety → control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
import torch  # noqa: E402

from classic_stack.control.config import config_sha256 as control_config_sha256  # noqa: E402
from classic_stack.control.controller import ControlLoop, EgoState  # noqa: E402
from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.hybrid import (  # noqa: E402
    ClassicExpertGenerator,
    H1CandidatePipeline,
    NominalVLAGenerator,
    ObservableAnchor,
    generate_hybrid_set,
    route_revision_sha256,
    simlingo_generator_hash,
)
from driving_vla.model.nominal_policy import NominalVLAPolicy  # noqa: E402
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
)
from driving_vla.runtime.safety_control_bind import (  # noqa: E402
    apply_safety_control,
)
from runtime import (  # noqa: E402
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver, READY  # noqa: E402
from safety_kernel.config import config_sha256 as safety_config_sha256  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    ObservableSnapshot,
    TrackedObject,
    TrafficLightObs,
)


def _map_basename(name: str) -> str:
    return str(name).split("/")[-1].replace("_Opt", "")


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return (result.stdout or "").strip()


def _worktree_identity() -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout
    untracked: list[dict[str, Any]] = []
    for raw_path in sorted(part for part in untracked_raw.split(b"\0") if part):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        untracked.append(
            {"path": relative, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        )
    untracked_manifest = json.dumps(
        untracked, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "commit": _git_text("rev-parse", "HEAD"),
        "branch": _git_text("branch", "--show-current"),
        "status": _git_text("status", "--short").splitlines(),
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked,
        "untracked_manifest_sha256": hashlib.sha256(untracked_manifest).hexdigest(),
    }


def _actor_xy(actor: Any) -> tuple[float, float]:
    location = actor.get_transform().location
    return float(location.x), float(location.y)


def _require_clean_scene(world: Any) -> None:
    active = [
        actor
        for actor in world.get_actors()
        if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker."))
    ]
    if active:
        ids = [int(actor.id) for actor in active]
        raise RuntimeError(f"NEEDS_CLEAN_SCENE actor_ids={ids}")


def _route_and_spawn(carla_map: Any, *, spacing_m: float = 2.0) -> tuple[Any, tuple[tuple[float, float], ...]]:
    spawns = sorted(
        carla_map.get_spawn_points(),
        key=lambda tf: (
            round(float(tf.location.x), 3),
            round(float(tf.location.y), 3),
            round(float(tf.rotation.yaw), 3),
        ),
    )
    for spawn in spawns:
        waypoint = carla_map.get_waypoint(
            spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            continue
        route = [(float(waypoint.transform.location.x), float(waypoint.transform.location.y))]
        current = waypoint
        visited: set[int] = {int(current.id)}
        for _ in range(45):
            next_waypoints = list(current.next(spacing_m))
            next_waypoints.sort(
                key=lambda wp: (
                    0 if int(wp.road_id) == int(current.road_id) else 1,
                    0 if int(wp.lane_id) == int(current.lane_id) else 1,
                    abs(
                        (float(wp.transform.rotation.yaw) - float(current.transform.rotation.yaw) + 180.0)
                        % 360.0
                        - 180.0
                    ),
                    int(wp.id),
                )
            )
            chosen = next((wp for wp in next_waypoints if int(wp.id) not in visited), None)
            if chosen is None:
                break
            current = chosen
            visited.add(int(current.id))
            route.append(
                (float(current.transform.location.x), float(current.transform.location.y))
            )
        length = sum(
            math.hypot(route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1])
            for i in range(1, len(route))
        )
        if len(route) >= 31 and length >= 60.0:
            return spawn, tuple(route)
    raise RuntimeError("no_deterministic_route_longer_than_60m")


def _image_rgb(measurement: Any) -> np.ndarray:
    width, height = int(measurement.width), int(measurement.height)
    bgra = np.frombuffer(measurement.raw_data, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])


def _ego_state(actor: Any) -> tuple[EgoState, float]:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    accel = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
    state = EgoState(
        x=float(transform.location.x),
        y=float(transform.location.y),
        yaw=math.radians(float(transform.rotation.yaw)),
        v=float(speed),
        steer=float(actor.get_control().steer) * 0.6,
    )
    return state, float(accel)


def _observable_actors(world: Any, ego: Any, simulation_time_s: float) -> tuple[TrackedObject, ...]:
    ex, ey = _actor_xy(ego)
    out: list[TrackedObject] = []
    for actor in world.get_actors():
        if int(actor.id) == int(ego.id):
            continue
        type_id = str(getattr(actor, "type_id", ""))
        if not type_id.startswith(("vehicle.", "walker.")):
            continue
        transform = actor.get_transform()
        if math.hypot(float(transform.location.x) - ex, float(transform.location.y) - ey) > 50.0:
            continue
        velocity = actor.get_velocity()
        extent = actor.bounding_box.extent
        out.append(
            TrackedObject(
                actor_id=str(actor.id),
                class_name=type_id,
                x=float(transform.location.x),
                y=float(transform.location.y),
                yaw=math.radians(float(transform.rotation.yaw)),
                vx=float(velocity.x),
                vy=float(velocity.y),
                length_m=max(0.1, 2.0 * float(extent.x)),
                width_m=max(0.1, 2.0 * float(extent.y)),
                observed_time_s=simulation_time_s,
            )
        )
    return tuple(sorted(out, key=lambda item: item.actor_id))


def _observable_lights(
    world: Any,
    ego: Any,
    simulation_time_s: float,
    route: tuple[tuple[float, float], ...],
) -> tuple[TrafficLightObs, ...]:
    ex, ey = _actor_xy(ego)
    out: list[TrafficLightObs] = []
    for actor in world.get_actors():
        if not str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light"):
            continue
        x, y = _actor_xy(actor)
        distance_m = math.hypot(x - ex, y - ey)
        if distance_m > 50.0 or min(math.hypot(x - rx, y - ry) for rx, ry in route) > 8.0:
            continue
        state = str(actor.get_state()).split(".")[-1].lower()
        out.append(
            TrafficLightObs(
                light_id=str(actor.id),
                state=state,
                distance_m=distance_m,
                observed_time_s=simulation_time_s,
            )
        )
    return tuple(sorted(out, key=lambda item: item.light_id))


def _build_anchor(runtime: ScenarioRuntime, header: Any, route: tuple[tuple[float, float], ...]) -> ObservableAnchor:
    measurement = runtime.sensor_measurement("front_camera", header.carla_frame)
    if int(measurement.frame) != int(header.carla_frame):
        raise RuntimeError("camera_frame_mismatch")
    ego = runtime._actors["ego"]
    state, acceleration = _ego_state(ego)
    observation_id = f"{runtime.identity.run_id}:frame-{header.carla_frame}"
    image = _image_rgb(measurement)
    bundle = ObservationBundle(
        run_id=runtime.identity.run_id,
        frame_id=observation_id,
        scenario_id=runtime.identity.scenario_id,
        simulation_time_s=float(header.simulation_time),
        wall_time_s=float(header.wall_time),
        carla_frame=int(header.carla_frame),
        ego_x=state.x,
        ego_y=state.y,
        ego_yaw=state.yaw,
        ego_v=state.v,
        route_xy=route,
        front_rgb=image,
        ego_history=((state.x, state.y, state.yaw, state.v),),
        meta={
            "official_contract": True,
            "image_layout": "rgb",
            "prompt_mode": "target_point",
            "camera_mount_xyz": SIMLINGO_CAMERA_XYZ,
        },
    )
    current_waypoint = runtime.world.get_map().get_waypoint(
        ego.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    half_width = 1.75 if current_waypoint is None else max(0.5, float(current_waypoint.lane_width) * 0.5)
    snapshot = ObservableSnapshot(
        run_id=bundle.run_id,
        frame_id=bundle.frame_id,
        scenario_id=bundle.scenario_id,
        simulation_time_s=bundle.simulation_time_s,
        wall_time_s=bundle.wall_time_s,
        ego_x=bundle.ego_x,
        ego_y=bundle.ego_y,
        ego_yaw=bundle.ego_yaw,
        ego_v=bundle.ego_v,
        ego_a=acceleration,
        observed_time_s=bundle.simulation_time_s,
        freshness_s=0.0,
        speed_limit_mps=max(0.0, float(ego.get_speed_limit()) / 3.6),
        actors=_observable_actors(runtime.world, ego, bundle.simulation_time_s),
        traffic_lights=_observable_lights(
            runtime.world,
            ego,
            bundle.simulation_time_s,
            route,
        ),
        corridor_centerline=route,
        corridor_half_width_m=half_width,
        coordinate_frame="map",
    )
    return ObservableAnchor(
        observation_id=observation_id,
        bundle=bundle,
        safety_snapshot=snapshot,
        route_revision=route_revision_sha256(route),
        sensor_frames={"front_camera": int(measurement.frame)},
        sensor_timestamps_s={"front_camera": float(measurement.timestamp)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h1.live_smoke.v1",
        "ok": False,
        "map_requested": args.map,
        "worktree": _worktree_identity(),
        "started_wall_time_s": time.time(),
    }
    runtime: ScenarioRuntime | None = None
    registry: RunRegistry | None = None
    run_id: str | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")
        resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=10.0)
        report = resolver.preflight()
        payload["connection"] = report.to_dict()
        if report.status != READY:
            raise RuntimeError(f"CARLA_NOT_READY:{report.error_code}:{report.error_message}")
        client, _ = resolver.connect(report=report)
        world = client.get_world()
        map_name = str(world.get_map().name)
        if _map_basename(map_name) != _map_basename(args.map):
            raise RuntimeError(f"MAP_MISMATCH:{map_name}!={args.map}")
        _require_clean_scene(world)
        spawn, route = _route_and_spawn(world.get_map())

        policy = NominalVLAPolicy(keep_on_gpu=True)
        policy.ensure_loaded()
        vla_hash = simlingo_generator_hash(policy)
        classic = ClassicExpertGenerator()
        vla = NominalVLAGenerator(policy, generator_hash=vla_hash)

        run_id = f"h1-town03-{time.time_ns()}"
        identity = RunIdentity(
            experiment_id="h1-independent-candidates",
            run_id=run_id,
            scenario_id="h1-town03-smoke",
            attempt_id=0,
            server_epoch=f"carla-{report.server_version}-{report.process_state}",
            producer_version="h1-live-smoke-v1",
        )
        profile = load_runtime_profiles(
            ROOT / "safedrive_foundry/config/runtime_profiles.toml"
        )["throughput_20hz"]
        camera_width, camera_height = SIMLINGO_CAMERA_NATIVE_SIZE
        camera = carla.Transform(
            carla.Location(
                x=float(SIMLINGO_CAMERA_XYZ[0]),
                y=float(SIMLINGO_CAMERA_XYZ[1]),
                z=float(SIMLINGO_CAMERA_XYZ[2]),
            )
        )
        spec = ScenarioSpec(
            scenario_id=identity.scenario_id,
            map_name=map_name,
            actors=(
                ActorSpec(
                    "ego", "vehicle.tesla.model3", spawn, "ego", 0, False
                ),
            ),
            sensors=(
                SensorSpec(
                    "front_camera",
                    "sensor.camera.rgb",
                    camera,
                    "ego",
                    0,
                    {
                        "image_size_x": str(camera_width),
                        "image_size_y": str(camera_height),
                        "fov": str(SIMLINGO_CAMERA_FOV_DEG),
                    },
                ),
            ),
            traffic_manager_seed=20260812,
            # The first GPU camera frame may include CARLA shader/renderer warm-up.
            # Keep the wait bounded while avoiding a false failure at the old 2 s gate.
            sensor_timeout_seconds=10.0,
        )
        registry = RunRegistry(args.evidence_dir / "run_registry.sqlite3")
        runtime = ScenarioRuntime(
            client=client,
            identity=identity,
            profile=profile,
            registry=registry,
            lease_path=ROOT / ".runtime" / f"tick-lease-{run_id}.lock",
            owner="sdf.h1.live_smoke",
        )
        runtime.start(spec)
        header = runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
        anchor = _build_anchor(runtime, header, route)
        generated = generate_hybrid_set(anchor, classic, vla)
        pipeline = H1CandidatePipeline()
        result = pipeline.decide(generated)

        if len(generated.candidates) != 2 or not all(attempt.success for attempt in generated.attempts):
            raise RuntimeError("BOTH_SOURCES_NOT_GENERATED")
        if policy.forward_count != 1:
            raise RuntimeError(f"VLA_FORWARD_COUNT:{policy.forward_count}")
        if not all(item.guard is not None for item in result.guarded_set.candidates):
            raise RuntimeError("MISSING_PER_CANDIDATE_GUARD")
        if not result.routing.pass_candidate_ids or result.routing.selected_candidate_id is None:
            raise RuntimeError("NO_GUARD_PASS_CANDIDATE")

        safety_input = result.guarded_set.to_policy_candidate_set(
            tuple(
                item.candidate
                for item in result.guarded_set.candidates
                if item.candidate.candidate_id == result.routing.selected_candidate_id
            )
        )
        ego_actor = runtime._actors["ego"]
        ego_state, _ = _ego_state(ego_actor)
        control_loop = ControlLoop()
        applied = apply_safety_control(
            result.safety.decision,
            safety_input,
            control_loop,
            ego_state,
            anchor.simulation_time_s,
        )
        if not applied.is_track_approved:
            raise RuntimeError(f"CONTROL_NOT_TRACK_APPROVED:{applied.applied_mode.value}")
        if applied.executed_id != result.safety.decision.executed_trajectory_id:
            raise RuntimeError("APPLIED_EXECUTED_ID_MISMATCH")
        execution_header = runtime.tick(
            carla.VehicleControl(
                throttle=applied.throttle,
                brake=applied.brake,
                steer=applied.steer,
            )
        )
        runtime.complete()
        runtime = None
        payload["runtime_cleanup"] = registry.record(run_id)
        if payload["runtime_cleanup"] is None or payload["runtime_cleanup"]["status"] != "COMPLETED":
            raise RuntimeError("RUNTIME_CLEANUP_NOT_COMPLETED")

        load_report = policy.runtime.load_report if policy.runtime is not None else None
        payload.update(
            {
                "ok": True,
                "run_id": run_id,
                "anchor": anchor.to_dict(),
                "hybrid_candidate_set": result.guarded_set.to_dict(),
                "routing_and_safety": result.to_dict(),
                "applied_control": applied.to_dict(),
                "execution_frame": int(execution_header.carla_frame),
                "vla_forward_count": policy.forward_count,
                "vla_native_latency_s": policy.last_latency_s,
                "vla_peak_vram_mb": policy.last_peak_vram_mb,
                "vla_generator_hash": vla_hash,
                "model_load": None if load_report is None else vars(load_report),
                "configs": {
                    "scenario_sha256": ScenarioRuntime.config_hash(spec, profile),
                    "classic_sha256": classic.generator_hash,
                    "safety_sha256": safety_config_sha256(pipeline.safety.config.raw_toml),
                    "control_sha256": control_config_sha256(control_loop.config.raw_toml),
                },
                "cuda": {
                    "device": torch.cuda.get_device_name(0),
                    "allocated_mb": torch.cuda.memory_allocated(0) / (1024.0 * 1024.0),
                    "reserved_mb": torch.cuda.memory_reserved(0) / (1024.0 * 1024.0),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}:{exc}"
        if runtime is not None:
            try:
                runtime.abort(type(exc).__name__)
                if registry is not None and run_id is not None:
                    payload["runtime_cleanup"] = registry.record(run_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                payload["cleanup_error"] = f"{type(cleanup_exc).__name__}:{cleanup_exc}"
    finally:
        payload["ended_wall_time_s"] = time.time()
        output = args.evidence_dir / "h1_smoke.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"ok": payload["ok"], "evidence": str(output), "error": payload.get("error")}, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
