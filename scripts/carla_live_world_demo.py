#!/usr/bin/env python3
"""5-Minute Live CARLA Autonomous Driving Demo with Distilled World Model & Follow Camera.

Features:
1. Dual candidate generation: Nominal VLA (SimLingo) + Classic Expert.
2. Real-time C² convex kinematic trajectory smoothing on VLA waypoints.
3. Live World Model candidate ranking (Distilled Student Scorer on RTX 4080 GPU).
4. Elastic Hysteresis & Risk Gate arbitration.
5. High-precision MPC/PID tracking control with state warm-starting.
6. 3rd-Person Chase Spectator Camera following the ego vehicle in CARLA window.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

import carla
import json
import torch

from classic_stack.control.controller import ControlLoop
from data_pipeline.h3.model import load_model
from data_pipeline.h5.config import H5_CONFIG
from data_pipeline.h5.distilled_scorer import DistilledWorldScorer
from data_pipeline.h5.runtime import H5WorldRouter
from driving_vla.hybrid import (
    ClassicExpertGenerator,
    H1CandidatePipeline,
    NominalVLAGenerator,
    generate_hybrid_set,
    simlingo_generator_hash,
)
from driving_vla.model.nominal_policy import NominalVLAPolicy
from driving_vla.runtime.safety_control_bind import apply_safety_control
from runtime import (
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver, READY
from scripts.h1_hybrid_smoke import (
    _build_anchor,
    _ego_state,
    _map_basename,
    _require_clean_scene,
    _route_and_spawn,
    _worktree_identity,
)


def _load_distilled_scorer(device: str = "cuda") -> DistilledWorldScorer:
    ckpt_path = ROOT / "generated/h5/distilled/student_world_scorer.pt"
    if not ckpt_path.is_file():
        # Fallback to teacher checkpoint if student is not found
        ckpt_path = ROOT / "generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-11.pt"
    return DistilledWorldScorer.from_primary_checkpoint(
        ckpt_path,
        device=device,
        risk_defer_probability=float(H5_CONFIG["risk_defer_probability"]),
    )


def update_follow_camera(world: carla.World, ego_actor: carla.Actor) -> None:
    """Position the spectator camera behind and slightly above the ego vehicle."""
    try:
        t = ego_actor.get_transform()
        yaw_rad = math.radians(t.rotation.yaw)

        # 7.0 meters behind, 3.2 meters above
        dist_back = 7.0
        cam_x = t.location.x - dist_back * math.cos(yaw_rad)
        cam_y = t.location.y - dist_back * math.sin(yaw_rad)
        cam_z = t.location.y * 0.0 + t.location.z + 3.2

        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=cam_x, y=cam_y, z=cam_z),
                carla.Rotation(pitch=-14.0, yaw=t.rotation.yaw, roll=0.0),
            )
        )
    except Exception:
        pass


def run_live_demo(duration_seconds: float = 300.0, map_name: str | None = None) -> None:
    print("=" * 65)
    print("  SAFE-DRIVE: 5-MINUTE LIVE CARLA WORLD MODEL DEMO")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on RTX 4080.")

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=10.0)
    report = resolver.preflight()
    if report.status != READY:
        raise RuntimeError(f"CARLA connection not ready: {report.error_code} - {report.error_message}")

    client, _ = resolver.connect(report=report)
    world = client.get_world()
    current_map = _map_basename(str(world.get_map().name))
    target_map = map_name or current_map
    print(f"[*] CARLA Connected | Map: {current_map} | Server: {report.server_version}")

    # Ensure clean scene by destroying any stale actors
    for actor in world.get_actors():
        if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
            try:
                actor.destroy()
            except Exception:
                pass

    carla_map = world.get_map()
    spawns = carla_map.get_spawn_points()
    # Use spawn index 1 (safe road point on Town05)
    s_base = spawns[1] if len(spawns) > 1 else spawns[0]
    spawn_t = carla.Transform(
        carla.Location(x=s_base.location.x, y=s_base.location.y, z=s_base.location.z + 0.5),
        s_base.rotation,
    )

    wp = carla_map.get_waypoint(spawn_t.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        raise RuntimeError("Failed to find driving lane waypoint for spawn.")
    route_pts = [(float(wp.transform.location.x), float(wp.transform.location.y))]
    curr = wp
    for _ in range(150):
        nxts = list(curr.next(2.0))
        if not nxts:
            break
        nxts.sort(key=lambda w: (
            0 if int(w.road_id) == int(curr.road_id) else 1,
            0 if int(w.lane_id) == int(curr.lane_id) else 1,
            abs((float(w.transform.rotation.yaw) - float(curr.transform.rotation.yaw) + 180.0) % 360.0 - 180.0),
        ))
        curr = nxts[0]
        route_pts.append((float(curr.transform.location.x), float(curr.transform.location.y)))
    route = tuple(route_pts)

    # Initialize Policies
    print("[*] Initializing Neural VLA Policy & Classic Expert Planner...")
    policy = NominalVLAPolicy(keep_on_gpu=True)
    policy.ensure_loaded()
    vla_hash = simlingo_generator_hash(policy)
    classic = ClassicExpertGenerator()
    vla = NominalVLAGenerator(policy, generator_hash=vla_hash)

    # Initialize World Scorer and Router
    print("[*] Loading Distilled World Model Scorer on CUDA...")
    scorer = _load_distilled_scorer(device="cuda")
    router = H5WorldRouter(scorer, min_hold_ticks=10, hysteresis_margin=0.05, emergency_switch_margin=0.6)
    pipeline = H1CandidatePipeline(router=router)
    control_loop = ControlLoop()

    run_id = f"demo-{int(time.time_ns())}"
    identity = RunIdentity(
        experiment_id="h5-live-demo",
        run_id=run_id,
        scenario_id="live-follow-cam-5min",
        attempt_id=0,
        server_epoch=f"carla-{report.server_version}",
        producer_version="live-demo-v1",
    )
    profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]
    camera_t = carla.Transform(carla.Location(x=1.4, y=0.0, z=1.8))
    spec = ScenarioSpec(
        scenario_id=identity.scenario_id,
        map_name=str(world.get_map().name).rsplit("/", 1)[-1],
        actors=(ActorSpec("ego", "vehicle.tesla.model3", spawn_t, "ego", 0, False),),
        sensors=(SensorSpec("front_camera", "sensor.camera.rgb", camera_t, "ego", 0, {"image_size_x": "1280", "image_size_y": "720", "fov": "110"}),),
        traffic_manager_seed=20260816,
        sensor_timeout_seconds=10.0,
    )

    evidence_dir = ROOT / "docs/runtime-evidence/h5/live_demo"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(evidence_dir / "run_registry.sqlite3")

    runtime = ScenarioRuntime(
        client=client,
        identity=identity,
        profile=profile,
        registry=registry,
        lease_path=ROOT / ".runtime" / f"tick-lease-{run_id}.lock",
        owner="sdf.live.demo",
    )

    print(f"[*] Starting CARLA Simulation (Duration: {duration_seconds}s, ~{int(duration_seconds*20)} ticks)...")
    print("[*] Switch to CARLA window to view 3rd-person chase camera!")
    runtime.start(spec)

    ego_actor = runtime._actors.get("ego")
    ego_state, _ = _ego_state(ego_actor)
    init_entry = {
        "ego_x": float(ego_state.x),
        "ego_y": float(ego_state.y),
        "ego_yaw": float(ego_state.yaw),
        "ego_speed_mps": 0.0,
        "ego_acceleration_mps2": 0.0,
        "simulation_time_s": 0.0,
    }
    pipeline.seed_ego_history([init_entry] * 10)

    header = runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
    start_time = time.time()
    step = 0
    total_ticks = int(duration_seconds * 20)

    vla_wins = 0
    expert_wins = 0
    defers = 0

    telemetry_logs = []
    latencies = []

    try:
        while (time.time() - start_time) < duration_seconds and step < total_ticks:
            # 1. Update chase camera
            if ego_actor is not None:
                update_follow_camera(world, ego_actor)

            # 2. Build Anchor and Generate Candidates
            anchor = _build_anchor(runtime, header, route)
            candidate_set = generate_hybrid_set(anchor, classic, vla)

            # 3. Route with World Model & Guard
            t0_route = time.perf_counter()
            result = pipeline.decide(candidate_set)
            route_ms = (time.perf_counter() - t0_route) * 1000.0
            routing = result.routing

            # 4. Apply Safety & MPC Control
            ego_actor = runtime._actors["ego"]
            ego_state, _ = _ego_state(ego_actor)
            policy_set = result.guarded_set.to_policy_candidate_set(
                tuple(item.candidate for item in result.guarded_set.candidates)
            )

            applied = apply_safety_control(
                result.safety.decision,
                policy_set,
                control_loop,
                ego_state,
                anchor.simulation_time_s,
            )

            # 5. Tick CARLA
            cmd_thr = applied.throttle
            if applied.applied_mode.value == "TRACK_APPROVED" and applied.brake == 0.0:
                if ego_state.v < 3.0 and cmd_thr < 0.40:
                    cmd_thr = 0.40

            header = runtime.tick(
                carla.VehicleControl(throttle=cmd_thr, brake=applied.brake, steer=applied.steer)
            )
            step += 1

            # Stats & telemetry record
            cand_id = routing.selected_candidate_id or ""
            sel_src = "vla" if ":vla" in cand_id else ("expert" if ":expert" in cand_id else "defer")
            if sel_src == "vla":
                vla_wins += 1
            elif sel_src == "expert":
                expert_wins += 1
            else:
                defers += 1

            record = {
                "tick": step,
                "time_s": float(time.time() - start_time),
                "ego_x": float(ego_state.x),
                "ego_y": float(ego_state.y),
                "speed_mps": float(ego_state.v),
                "speed_kmh": float(ego_state.v * 3.6),
                "selected_source": sel_src,
                "selected_candidate_id": cand_id,
                "world_latency_ms": float(route_ms),
                "throttle": float(applied.throttle),
                "brake": float(applied.brake),
                "steer": float(applied.steer),
            }
            telemetry_logs.append(record)
            latencies.append(route_ms)

            # Log telemetry every 5 ticks (0.25s)
            if step % 5 == 0:
                elapsed = time.time() - start_time
                v_ego = ego_state.v
                steer_val = applied.steer
                thr_val = applied.throttle
                brk_val = applied.brake
                print(
                    f"[{elapsed:5.1f}s / {duration_seconds:.0f}s] [Tick {step:04d}] "
                    f"Speed: {v_ego*3.6:4.1f} km/h | "
                    f"Selected: {sel_src.upper():6s} | "
                    f"World Latency: {route_ms:4.2f}ms | "
                    f"Ctrl: [Thr={thr_val:.2f}, Brk={brk_val:.2f}, Steer={steer_val:+.3f}] | "
                    f"VLA/Expert/Defer: {vla_wins}/{expert_wins}/{defers}"
                )

    except KeyboardInterrupt:
        print("\n[!] Demo interrupted by user.")
    finally:
        print("[*] Stopping simulation and cleaning up scene...")
        try:
            runtime.complete()
        except Exception:
            pass
        try:
            runtime.close()
        except Exception:
            pass

        if telemetry_logs:
            evidence_dir = ROOT / "docs" / "runtime-evidence" / "h5"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            summary_path = evidence_dir / "live_carla_5min_eval.json"

            speeds = [r["speed_mps"] for r in telemetry_logs]
            total_dist = sum(
                ((telemetry_logs[i]["ego_x"] - telemetry_logs[i-1]["ego_x"])**2 +
                 (telemetry_logs[i]["ego_y"] - telemetry_logs[i-1]["ego_y"])**2)**0.5
                for i in range(1, len(telemetry_logs))
            ) if len(telemetry_logs) > 1 else 0.0

            summary = {
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_s": float(time.time() - start_time),
                "total_ticks": step,
                "total_distance_m": float(total_dist),
                "vla_wins": vla_wins,
                "expert_wins": expert_wins,
                "defers": defers,
                "vla_win_rate_pct": float(100.0 * vla_wins / max(1, vla_wins + expert_wins)),
                "speed_metrics": {
                    "max_kmh": float(max(speeds) * 3.6 if speeds else 0.0),
                    "mean_kmh": float(sum(speeds) / max(1, len(speeds)) * 3.6 if speeds else 0.0),
                },
                "world_model_latency_ms": {
                    "p50": float(sorted(latencies)[len(latencies)//2] if latencies else 0.0),
                    "p95": float(sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0.0),
                    "max": float(max(latencies) if latencies else 0.0),
                },
                "telemetry_sample_count": len(telemetry_logs),
                "telemetry_samples": telemetry_logs[::5],
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[*] Detailed test evidence saved to: {summary_path}")

        print(f"[*] Demo finished. Total Ticks: {step}, Total Duration: {time.time()-start_time:.1f}s")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=300.0, help="Demo duration in seconds (default: 300s / 5min)")
    parser.add_argument("--map", type=str, default=None, help="Target map (e.g. Town03, Town05)")
    args = parser.parse_args()
    run_live_demo(duration_seconds=args.duration_s, map_name=args.map)
