#!/usr/bin/env python3
"""G3 live: neural VLA (+ camera) → Safety → MPC → ScenarioRuntime.

Default backend is neural SimLingo. Geometry/fingerprint cannot be used for
stage close (assert_g3_close rejects non-neural).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simlingo-main"))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402

from classic_stack.control import load_control_config  # noqa: E402
from classic_stack.control.controller import ControlLoop, EgoState  # noqa: E402
from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray, arrays_to_candidate_set  # noqa: E402
from driving_vla.baselines.route_ego import RouteEgoBaseline  # noqa: E402
from driving_vla.model.neural_policy import NeuralV0Policy, NeuralV1Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402
from driving_vla.runtime.mailbox import CandidateMailbox  # noqa: E402
from driving_vla.runtime.mode import RuntimeMode, availability_for_mode, filter_candidates_for_mode  # noqa: E402
from driving_vla.runtime.safety_control_bind import (  # noqa: E402
    AppliedMode,
    apply_safety_control,
    evaluate_episode_status,
)
from runtime import (  # noqa: E402
    ActorSpec,
    RunRegistry,
    RuntimeIdentityFactory,
    ScenarioRuntime,
    ScenarioSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402
from safety_kernel import SafetyKernel, load_safety_config  # noqa: E402
from safety_kernel.contracts.schema import SCHEMA_VERSION  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
    PolicyCandidate,
    PolicyCandidateSet,
)


def set_spectator_follow(world: carla.World, vehicle: carla.Vehicle) -> None:
    """Chase-cam so the user can watch the ego in the UE window."""
    try:
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        yaw_rad = math.radians(rotation.yaw)
        cam_loc = carla.Location(
            location.x - 8.0 * math.cos(yaw_rad),
            location.y - 8.0 * math.sin(yaw_rad),
            location.z + 4.0,
        )
        spectator.set_transform(
            carla.Transform(cam_loc, carla.Rotation(pitch=-18.0, yaw=rotation.yaw, roll=0.0))
        )
    except Exception:
        pass


def free_spawn(world: carla.World) -> carla.Transform:
    occupied = [a.get_location() for a in world.get_actors().filter("vehicle.*")]
    for tf in world.get_map().get_spawn_points():
        if any(tf.location.distance(o) < 6.0 for o in occupied):
            continue
        return tf
    pts = world.get_map().get_spawn_points()
    if not pts:
        raise RuntimeError("no spawn points")
    return pts[0]


def build_route(world: carla.World, start: carla.Transform, max_route_m: float) -> list[tuple[float, float]]:
    m = world.get_map()
    wp = m.get_waypoint(start.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    pts: list[tuple[float, float]] = []
    dist = 0.0
    while wp is not None and dist < max_route_m:
        loc = wp.transform.location
        pts.append((float(loc.x), float(loc.y)))
        nxt = wp.next(2.0)
        if not nxt:
            break
        wp = nxt[0]
        dist += 2.0
    if len(pts) < 2:
        yaw = math.radians(start.rotation.yaw)
        pts = [
            (start.location.x, start.location.y),
            (start.location.x + max_route_m * math.cos(yaw), start.location.y + max_route_m * math.sin(yaw)),
        ]
    return pts


def restamp_cset(
    cset,
    *,
    run_id,
    frame_id,
    scenario_id,
    sim_t,
    frame,
    available,
    valid_for_s: float = 1.5,
):
    cands = []
    for c in cset.candidates:
        cands.append(
            PolicyCandidate(
                candidate_id=c.candidate_id,
                source=c.source,
                generated_time_s=sim_t,
                valid_until_s=sim_t + valid_for_s,
                probability=c.probability,
                points=c.points,
                behavior=c.behavior,
                uncertainty=c.uncertainty,
                availability=available and c.availability,
                intended_action=c.intended_action,
                dynamics_meta=dict(c.dynamics_meta),
            )
        )
    return PolicyCandidateSet(
        run_id=run_id,
        frame_id=frame_id,
        scenario_id=scenario_id,
        model_id=cset.model_id,
        carla_frame=frame,
        simulation_time_s=sim_t,
        wall_time_s=time.time(),
        candidates=tuple(cands),
        schema_version=SCHEMA_VERSION,
        coordinate_frame="map",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="VLA_SAFETY", choices=["VLA_SAFETY", "HYBRID", "CLASSIC"])
    ap.add_argument("--version", default="v0", choices=["v0", "v1", "baseline"])
    ap.add_argument("--backend", default="neural", choices=["neural", "baseline"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--max-route-m", type=float, default=50.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--fault", default="", choices=["", "timeout", "stale", "nan"])
    ap.add_argument(
        "--vla-period-s",
        type=float,
        default=0.5,
        help="VLA worker period; neural forward often ~0.2–1.2s so default 0.5s",
    )
    ap.add_argument(
        "--evidence-dir",
        default=str(ROOT / "docs/runtime-evidence/g3-05/neural_live_v2"),
    )
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or [args.seed]
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRITICAL ORDER (single 4080): load VLA first, NO carla.Client yet.
    # preflight/connect AFTER warm-up — otherwise UE freezes (car visible, not moving).
    # ------------------------------------------------------------------
    shared_rt = None
    policy: Any
    policy_backend = "baseline_route"
    if args.backend == "neural" and args.version in {"v0", "v1"}:
        print("STEP1: load neural VLA on GPU (no CARLA client yet, ~20-40s)...", flush=True)
        shared_rt = SimLingoNeuralRuntime()
        load_rep = shared_rt.load()
        print(
            "neural_load",
            load_rep.ok,
            "params",
            load_rep.n_params,
            "device",
            load_rep.device,
            load_rep.error,
            flush=True,
        )
        if not load_rep.ok:
            return 1
        try:
            warm = np.zeros((256, 512, 3), dtype=np.uint8)
            warm_obs = ObservationBundle(
                run_id="warm",
                frame_id="w0",
                scenario_id="warm",
                simulation_time_s=0.0,
                ego_v=3.0,
                front_rgb=warm,
                route_xy=((0.0, 0.0), (20.0, 0.0)),
            )
            pol_tmp = NeuralV1Policy(runtime=shared_rt) if args.version == "v1" else NeuralV0Policy(runtime=shared_rt)
            pol_tmp.predict_arrays(warm_obs)
            print("STEP1b: neural warm-up forward done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("warm-up warn", exc, flush=True)
        policy = NeuralV1Policy(runtime=shared_rt) if args.version == "v1" else NeuralV0Policy(runtime=shared_rt)
        policy_backend = "neural_simlingo"
    else:
        policy = RouteEgoBaseline()
        policy_backend = "baseline_route"

    print("STEP2: preflight + connect CARLA...", flush=True)
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=15.0)
    report = resolver.preflight()
    print("preflight", report.status, report.error_code, flush=True)
    if report.status != "READY":
        (evidence_dir / "blocked_preflight.json").write_text(
            json.dumps({"status": "BLOCKED_EXTERNAL", "preflight": report.to_dict()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print("CARLA not READY — start CarlaUE4 windowed, then re-run.", flush=True)
        return 75

    client, report = resolver.connect(report=report)
    world = client.get_world()
    profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["control_50hz"]
    print("STEP3: connected map=", world.get_map().name, "starting episode...", flush=True)

    results = []
    for seed in seeds:
        results.append(
            run_episode(
                client=client,
                world=world,
                profile=profile,
                policy=policy,
                policy_backend=policy_backend,
                mode=RuntimeMode[args.mode],
                seed=seed,
                max_route_m=args.max_route_m,
                max_steps=args.max_steps,
                hold_s=args.hold_s,
                fault=args.fault,
                version=args.version,
                vla_period_s=args.vla_period_s,
                evidence_dir=evidence_dir,
            )
        )
        print("result", {k: results[-1][k] for k in ("status", "steps", "seed", "distance_m", "latency_p95_ms", "camera_frames")})

    # Honest aggregation: never hard-code sources_seen.
    sources_union: set[str] = set()
    for r in results:
        for s in r.get("sources_seen") or []:
            sources_union.add(str(s))
    classic_cf = False if args.mode == "VLA_SAFETY" else True
    summary = {
        "mode": args.mode,
        "version": args.version,
        "policy_backend": policy_backend,
        "camera": True,
        "classic_current_frame": classic_cf,
        "seeds": seeds,
        "results": results,
        "all_ok": all(r.get("status") == "COMPLETED" for r in results),
        "map": world.get_map().name,
        "sources_seen": sorted(sources_union),
        "force_throttle": False,
        "safety_bind": "apply_safety_control_v1",
        "evidence_schema": "g3_live_v2",
    }
    ts = int(time.time())
    path = evidence_dir / f"live_summary_{args.mode}_{policy_backend}_{ts}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence_dir / "latest_live_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", path, "all_ok", summary["all_ok"], "backend", policy_backend, "sources", summary["sources_seen"])
    return 0 if summary["all_ok"] else 1


def run_episode(
    *,
    client,
    world,
    profile,
    policy,
    policy_backend: str,
    mode: RuntimeMode,
    seed: int,
    max_route_m: float,
    max_steps: int,
    hold_s: float,
    fault: str,
    version: str,
    vla_period_s: float,
    evidence_dir: Path,
) -> dict[str, Any]:
    scenario_id = f"TownG3.vla_neural.seed{seed}"
    stamp = int(time.time() * 1000)
    identity = RuntimeIdentityFactory.create(
        {
            "experiment_id": f"g3-vla-neural-live-{stamp}",
            "scenario_id": scenario_id,
            "attempt_id": seed,
            "server_epoch": f"carla-0.9.16-g3n-{stamp}",
            "producer_version": "g3-neural-live-v1",
        }
    )
    lease = evidence_dir / f".runtime/tick-lease-g3n-{seed}-{stamp}.lock"
    lease.parent.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(evidence_dir / f"registry_{seed}_{stamp}.sqlite")
    runtime = ScenarioRuntime(
        client=client,
        identity=identity,
        profile=profile,
        registry=registry,
        lease_path=lease,
        owner=f"sdf.g3-neural.{seed}.{stamp}",
    )
    spawn = free_spawn(world)
    for a in list(world.get_actors().filter("vehicle.*")):
        try:
            a.destroy()
        except Exception:
            pass
    for a in list(world.get_actors().filter("sensor.*")):
        try:
            a.destroy()
        except Exception:
            pass
    route = build_route(world, spawn, max_route_m)
    spec = ScenarioSpec(
        scenario_id=scenario_id,
        map_name=world.get_map().name,
        actors=(
            ActorSpec(
                name="ego",
                blueprint="vehicle.tesla.model3",
                transform=spawn,
                role="ego",
                spawn_order=0,
            ),
        ),
        sensors=(),
        traffic_manager_seed=seed,
        sensor_timeout_seconds=2.0,
    )
    runtime.start(spec)
    ego = runtime._actors["ego"]
    try:
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(hand_brake=False, brake=0.0, throttle=0.0))
    except Exception:
        pass

    # Dedicated RGB camera (not in SensorSpec barrier — store latest for VLA worker)
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "540")
    cam_bp.set_attribute("fov", "110")
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.2))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    rgb_lock = threading.Lock()
    latest_rgb: dict[str, Any] = {"img": None, "frame": -1, "n": 0}

    def _on_image(image: carla.Image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()  # BGRA→RGB
        with rgb_lock:
            latest_rgb["img"] = arr
            latest_rgb["frame"] = int(image.frame)
            latest_rgb["n"] = int(latest_rgb["n"]) + 1

    camera.listen(_on_image)

    # Neural P50 can be ~0.1–1.2s; mailbox must not soft-stale every control tick.
    mailbox_stale_s = max(2.0, float(vla_period_s) * 6.0)
    mailbox = CandidateMailbox(soft_stale_s=mailbox_stale_s)
    kernel = SafetyKernel(load_safety_config())
    control = ControlLoop(load_control_config())
    stop = threading.Event()
    latencies: list[float] = []
    peaks: list[float] = []
    run_id = identity.run_id
    sources_seen: set[str] = set()
    cand_valid_for_s = mailbox_stale_s

    def worker() -> None:
        while not stop.is_set():
            try:
                with rgb_lock:
                    img = None if latest_rgb["img"] is None else latest_rgb["img"].copy()
                if img is None:
                    time.sleep(0.05)
                    continue
                tf = ego.get_transform()
                vel = ego.get_velocity()
                speed = math.hypot(vel.x, vel.y)
                snap = world.get_snapshot()
                sim_t = float(snap.timestamp.elapsed_seconds)
                frame = int(snap.frame)
                obs = ObservationBundle(
                    run_id=run_id,
                    frame_id=f"f{frame}",
                    scenario_id=scenario_id,
                    simulation_time_s=sim_t,
                    wall_time_s=time.time(),
                    carla_frame=frame,
                    ego_x=float(tf.location.x),
                    ego_y=float(tf.location.y),
                    ego_yaw=math.radians(tf.rotation.yaw),
                    ego_v=float(speed),
                    route_xy=tuple(route),
                    front_rgb=img,
                    meta={"image_bgr": False},
                )
                t0 = time.perf_counter()
                if fault == "timeout":
                    time.sleep(0.3)
                    mailbox.mark_degraded("timeout")
                    time.sleep(vla_period_s)
                    continue
                if version == "baseline" or policy_backend == "baseline_route":
                    arrs = policy.predict(obs)
                    mid = policy.model_id
                    src = CandidateSource.VLA_FAST if mode is RuntimeMode.VLA_SAFETY else CandidateSource.CLASSIC
                else:
                    arrs = policy.predict_arrays(obs)
                    mid = policy.model_id
                    src = CandidateSource.VLA_FAST
                    if hasattr(policy, "last_latency_s"):
                        latencies.append(policy.last_latency_s)
                    if hasattr(policy, "last_peak_vram_mb"):
                        peaks.append(policy.last_peak_vram_mb)
                    if hasattr(policy, "v0") and hasattr(policy.v0, "last_latency_s"):
                        latencies.append(policy.v0.last_latency_s)
                        peaks.append(policy.v0.last_peak_vram_mb)
                if fault == "nan" and arrs:
                    bad = list(arrs[0].points_xy_yaw_v_a_kappa)
                    r0 = bad[0]
                    bad[0] = (float("nan"), r0[1], r0[2], r0[3], r0[4], r0[5])
                    arrs = [
                        TrajectoryArray(
                            points_xy_yaw_v_a_kappa=tuple(bad),
                            probability=arrs[0].probability,
                            uncertainty=arrs[0].uncertainty,
                            candidate_id=arrs[0].candidate_id,
                        )
                    ]
                try:
                    cset = arrays_to_candidate_set(arrs, obs, model_id=mid, source=src)
                except ValueError as exc:
                    mailbox.mark_degraded(f"invalid:{exc}")
                    time.sleep(vla_period_s)
                    continue
                if fault == "stale":
                    time.sleep(0.4)
                lat = time.perf_counter() - t0
                if version == "baseline" or policy_backend == "baseline_route":
                    latencies.append(lat)
                for c in cset.candidates:
                    sources_seen.add(c.source.value)
                mailbox.publish(cset, latency_s=lat)
            except Exception as exc:  # noqa: BLE001
                mailbox.mark_degraded(str(exc))
            time.sleep(vla_period_s)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    # wait for camera + first neural publish (required before honest tracking)
    t_wait = time.time()
    while time.time() - t_wait < max(hold_s, 3.0):
        with rgb_lock:
            if latest_rgb["n"] > 0:
                break
        time.sleep(0.05)
    t_mb = time.time()
    first_cset_wait_s = max(15.0, hold_s + 10.0)
    while time.time() - t_mb < first_cset_wait_s:
        if mailbox.latest() is not None and mailbox.vla_ok():
            print("first VLA candidate ready", flush=True)
            break
        time.sleep(0.05)
    else:
        print("WARN: no VLA candidate after wait; episode will hold/brake until publish", flush=True)
    time.sleep(min(hold_s, 1.0))

    steps = 0
    decisions: list[str] = []
    applied_modes: list[str] = []
    n_track_approved = 0
    n_emergency_brake = 0
    n_no_exec = 0
    decision_counts: dict[str, int] = {}
    moved = 0.0
    last_xy = (spawn.location.x, spawn.location.y)
    try:
        for _ in range(max_steps):
            snap = world.get_snapshot()
            sim_t = float(snap.timestamp.elapsed_seconds)
            frame = int(snap.frame)
            tf = ego.get_transform()
            vel = ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            moved += math.hypot(tf.location.x - last_xy[0], tf.location.y - last_xy[1])
            last_xy = (tf.location.x, tf.location.y)
            frame_id = f"f{frame}"
            obs_snap = ObservableSnapshot(
                run_id=run_id,
                frame_id=frame_id,
                scenario_id=scenario_id,
                simulation_time_s=sim_t,
                wall_time_s=time.time(),
                ego_x=float(tf.location.x),
                ego_y=float(tf.location.y),
                ego_yaw=math.radians(tf.rotation.yaw),
                ego_v=float(speed),
                observed_time_s=sim_t,
                freshness_s=0.0,
                corridor_centerline=tuple(route[:40]),
                corridor_half_width_m=2.5,
                privilege=ObservationPrivilege.OBSERVABLE,
            )
            entry = mailbox.latest()
            cset = None
            if entry is not None:
                cset = restamp_cset(
                    entry.candidate_set,
                    run_id=run_id,
                    frame_id=frame_id,
                    scenario_id=scenario_id,
                    sim_t=sim_t,
                    frame=frame,
                    available=entry.ok,
                    valid_for_s=cand_valid_for_s,
                )
                cset = filter_candidates_for_mode(cset, mode)

            if mode is RuntimeMode.HYBRID:
                bl = RouteEgoBaseline()
                o = ObservationBundle(
                    run_id=run_id,
                    frame_id=frame_id,
                    scenario_id=scenario_id,
                    simulation_time_s=sim_t,
                    ego_x=float(tf.location.x),
                    ego_y=float(tf.location.y),
                    ego_yaw=math.radians(tf.rotation.yaw),
                    ego_v=float(speed),
                    route_xy=tuple(route),
                )
                classic_set = arrays_to_candidate_set(
                    bl.predict(o), o, model_id=bl.model_id, source=CandidateSource.CLASSIC
                )
                sources_seen.add("classic")
                if cset is None:
                    cset = classic_set
                else:
                    cset = PolicyCandidateSet(
                        run_id=run_id,
                        frame_id=frame_id,
                        scenario_id=scenario_id,
                        model_id=cset.model_id,
                        carla_frame=frame,
                        simulation_time_s=sim_t,
                        wall_time_s=time.time(),
                        candidates=tuple(list(cset.candidates) + list(classic_set.candidates)),
                        schema_version=SCHEMA_VERSION,
                        coordinate_frame="map",
                    )

            avail = availability_for_mode(mode, vla_ok=mailbox.vla_ok())
            result = kernel.tick(obs_snap, cset, now_s=sim_t, availability=avail)
            kind_s = str(getattr(result.decision.decision_kind, "value", result.decision.decision_kind))
            decisions.append(kind_s)
            decision_counts[kind_s] = decision_counts.get(kind_s, 0) + 1

            ego_state = EgoState(
                x=float(tf.location.x),
                y=float(tf.location.y),
                yaw=math.radians(tf.rotation.yaw),
                v=float(speed),
                steer=float(ego.get_control().steer) * 0.6,
            )
            # Honest bind: no force throttle, no first-available, no open-loop route steer.
            applied = apply_safety_control(
                result.decision,
                cset,
                control,
                ego_state,
                sim_t,
            )
            applied_modes.append(applied.applied_mode.value)
            if applied.applied_mode is AppliedMode.TRACK_APPROVED:
                n_track_approved += 1
            elif applied.applied_mode is AppliedMode.EMERGENCY_BRAKE:
                n_emergency_brake += 1
            else:
                n_no_exec += 1

            thr, brk, ste = applied.throttle, applied.brake, applied.steer
            vc = carla.VehicleControl(
                throttle=max(0.0, min(1.0, thr)),
                steer=max(-1.0, min(1.0, ste)),
                brake=max(0.0, min(1.0, brk)),
                hand_brake=False,
                manual_gear_shift=False,
            )
            runtime.tick(vc)
            if steps % 2 == 0:
                set_spectator_follow(world, ego)
            steps += 1
            if steps < 15 or steps % 25 == 0:
                print(
                    f"step={steps} speed={speed:.2f} thr={vc.throttle:.2f} brk={vc.brake:.2f} ste={vc.steer:.2f} "
                    f"mode={applied.applied_mode.value} exec={applied.executed_id} "
                    f"mailbox={mailbox.vla_ok()} dec={kind_s}",
                    flush=True,
                )
    finally:
        stop.set()
        th.join(timeout=5.0)
        try:
            camera.stop()
            camera.destroy()
        except Exception:
            pass
        try:
            runtime.complete()
        except Exception as exc:  # noqa: BLE001
            print("cleanup warn", exc)

    p95 = None
    if latencies:
        p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000.0
    with rgb_lock:
        cam_n = int(latest_rgb["n"])
    n_emergency = sum(1 for d in decisions if d == "EMERGENCY")
    status, fail_reasons = evaluate_episode_status(
        steps=steps,
        max_steps=max_steps,
        min_steps=min(80, max_steps),
        distance_m=moved,
        decisions=decisions,
        n_track_approved=n_track_approved,
        n_emergency=n_emergency,
        sources_seen=sources_seen,
        camera_frames=cam_n,
        fault=fault or "",
        mode=mode.value,
        classic_current_frame=False if mode is RuntimeMode.VLA_SAFETY else True,
    )
    return {
        "status": status,
        "fail_reasons": fail_reasons,
        "steps": steps,
        "seed": seed,
        "mode": mode.value,
        "version": version,
        "policy_backend": policy_backend,
        "fault": fault,
        "distance_m": moved,
        "latency_n": len(latencies),
        "latency_p95_ms": p95,
        "peak_vram_mb": max(peaks) if peaks else None,
        "decision_tail": decisions[-15:],
        "decision_counts": decision_counts,
        "applied_mode_tail": applied_modes[-15:],
        "n_track_approved": n_track_approved,
        "n_emergency": n_emergency,
        "n_emergency_brake": n_emergency_brake,
        "n_no_exec": n_no_exec,
        "run_id": run_id,
        "camera_frames": cam_n,
        "sources_seen": sorted(sources_seen),
        "force_throttle": False,
        "safety_bind": "apply_safety_control_v1",
    }


if __name__ == "__main__":
    raise SystemExit(main())
