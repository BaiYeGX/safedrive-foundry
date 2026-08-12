#!/usr/bin/env python3
"""Simple watchable pure-VLA drive (turn-oriented spawn).

Hard lessons from this host (single RTX 4080, no iGPU):
- Load VLA BEFORE any carla.Client if possible.
- Do NOT call vehicle.set_autopilot (Traffic Manager bind crash).
- Prefer SINGLE-THREAD: tick then occasional VLA (no parallel CUDA+tick).
- Spectator every frame; if RPC fails, keep ticking and print.

Usage:
  # prove motion (no neural):
  python tests/g3/run_g3_pure_vla_watch.py --motion-only --seconds 15
  # VLA steering + throttle (after CARLA READY):
  python tests/g3/run_g3_pure_vla_watch.py --seconds 40 --throttle 0.55
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simlingo-main"))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402

from runtime.carla_connection import ConnectionResolver  # noqa: E402
# NOTE: do NOT import torch / neural_policy at module level — CUDA init kills CARLA on this host.


def spawn_with_turn(world: carla.World, *, min_turn_deg: float = 50.0, seed: int = 11):
    """Pick spawn like classic expert: project to driving waypoint, z floor, prefer a bend."""
    import random

    m = world.get_map()
    occupied = [a.get_location() for a in world.get_actors().filter("vehicle.*")]
    spawns = list(m.get_spawn_points())
    rng = random.Random(seed)
    rng.shuffle(spawns)
    best = None
    for tf in spawns[:40]:
        if any(tf.location.distance(o) < 8.0 for o in occupied):
            continue
        wp = m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        # CRITICAL: use waypoint transform (correct road height/yaw), not raw spawn z~0
        spawn_tf = wp.transform
        # floor + margin so vehicle does not fall through mesh (z~0 stuck)
        spawn_tf.location.z = max(float(spawn_tf.location.z), 1.0) + 0.4
        # reject near-origin junk if map origin is invalid for driving
        if abs(spawn_tf.location.x) + abs(spawn_tf.location.y) < 5.0 and spawn_tf.location.z < 1.0:
            continue
        route = []
        yaw0 = spawn_tf.rotation.yaw
        max_turn = 0.0
        cur = wp
        for _ in range(120):
            route.append((cur.transform.location.x, cur.transform.location.y))
            dyaw = abs((cur.transform.rotation.yaw - yaw0 + 180.0) % 360.0 - 180.0)
            max_turn = max(max_turn, dyaw)
            nxt = cur.next(2.0)
            if not nxt:
                break
            cur = nxt[0]
        if len(route) < 20:
            continue
        if max_turn >= min_turn_deg:
            print(
                f"chosen spawn road_xyz=({spawn_tf.location.x:.1f},{spawn_tf.location.y:.1f},{spawn_tf.location.z:.1f}) "
                f"turn≈{max_turn:.0f}°",
                flush=True,
            )
            return spawn_tf, route, max_turn
        if best is None or max_turn > best[2]:
            best = (spawn_tf, route, max_turn)
    if best is None:
        wp = m.get_waypoint(spawns[0].location, project_to_road=True, lane_type=carla.LaneType.Driving)
        spawn_tf = wp.transform if wp is not None else spawns[0]
        spawn_tf.location.z = max(float(spawn_tf.location.z), 0.5)
        return spawn_tf, [], 0.0
    print(
        f"fallback spawn road_xyz=({best[0].location.x:.1f},{best[0].location.y:.1f},{best[0].location.z:.1f}) "
        f"turn≈{best[2]:.0f}°",
        flush=True,
    )
    return best


def spectator_follow(world: carla.World, vehicle: carla.Vehicle) -> bool:
    try:
        sp = world.get_spectator()
        tf = vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        loc = carla.Location(
            tf.location.x - 8.0 * math.cos(yaw),
            tf.location.y - 8.0 * math.sin(yaw),
            tf.location.z + 6.0,
        )
        sp.set_transform(carla.Transform(loc, carla.Rotation(pitch=-25.0, yaw=tf.rotation.yaw)))
        return True
    except Exception as exc:
        print("spectator timeout/fail:", type(exc).__name__, flush=True)
        return False


def route_steer(vehicle: carla.Vehicle, route: list[tuple[float, float]], look: int = 12) -> float:
    if len(route) < 2:
        return 0.0
    tf = vehicle.get_transform()
    # nearest index
    best_i, best_d = 0, 1e18
    for i, (x, y) in enumerate(route):
        d = (x - tf.location.x) ** 2 + (y - tf.location.y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    j = min(best_i + look, len(route) - 1)
    tx, ty = route[j]
    desired = math.atan2(ty - tf.location.y, tx - tf.location.x)
    err = desired - math.radians(tf.rotation.yaw)
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return max(-0.7, min(0.7, err * 0.85))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--motion-only", action="store_true")
    ap.add_argument("--throttle", type=float, default=0.55)
    ap.add_argument("--vla-every", type=int, default=12, help="run VLA every N ticks (main thread)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    policy = None
    rt = None
    if not args.motion_only:
        # Lazy import AFTER argparse so --motion-only never touches CUDA.
        from driving_vla.adapter.policy_adapter import ObservationBundle
        from driving_vla.model.neural_policy import NeuralV0Policy
        from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

        print("STEP1: load VLA on GPU BEFORE carla client (~20-40s). CARLA window can sit idle.", flush=True)
        rt = SimLingoNeuralRuntime()
        rep = rt.load()
        print("load", rep.ok, "params", rep.n_params, rep.error, flush=True)
        if not rep.ok:
            return 1
        policy = NeuralV0Policy(runtime=rt)
        warm = np.full((256, 512, 3), 120, dtype=np.uint8)
        policy.predict_arrays(
            ObservationBundle(
                run_id="w",
                frame_id="0",
                scenario_id="w",
                simulation_time_s=0.0,
                ego_v=4.0,
                front_rgb=warm,
                route_xy=((0.0, 0.0), (25.0, 0.0)),
            )
        )
        print("STEP1b: warm-up OK", flush=True)
        # Free 4080 for Windows CARLA — keep weights on CPU until each VLA call
        rt.release_gpu_for_carla()

    print("STEP2: connect CARLA (must already be READY)...", flush=True)
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=8.0)
    report = resolver.preflight()
    print("preflight", report.status, flush=True)
    if report.status != "READY":
        print("Please start CarlaUE4 windowed first, wait for map, re-run.", flush=True)
        return 75
    client, report = resolver.connect(report=report)
    client.set_timeout(5.0)
    world = client.get_world()

    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        try:
            a.destroy()
        except Exception:
            pass

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.dt)
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)

    spawn, route, turn_deg = spawn_with_turn(world, seed=args.seed)
    print(f"map={world.get_map().name} turn_ahead≈{turn_deg:.0f}° route_pts={len(route)}", flush=True)

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    vehicle = world.try_spawn_actor(bp, spawn)
    if vehicle is None:
        # one retry: bump z
        spawn.location.z += 0.5
        vehicle = world.try_spawn_actor(bp, spawn)
    if vehicle is None:
        print("SPAWN FAILED at", spawn.location, flush=True)
        return 1
    try:
        vehicle.set_simulate_physics(True)
    except Exception:
        pass
    world.tick()
    spectator_follow(world, vehicle)
    loc0 = vehicle.get_location()
    print(
        f"=== EGO VISIBLE NOW id={vehicle.id} xyz=({loc0.x:.1f},{loc0.y:.1f},{loc0.z:.1f}) "
        f"turn≈{turn_deg:.0f}° LOOK AT CARLA WINDOW ===",
        flush=True,
    )
    if abs(loc0.x) + abs(loc0.y) < 3.0:
        print("WARNING: still near map origin — spawn may be bad", flush=True)

    # RGB camera (sync: needs ticks after listen)
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "360")
    cam_bp.set_attribute("fov", "100")
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.4, z=2.0)), attach_to=vehicle)
    latest = {"img": None, "n": 0}

    def on_image(image: carla.Image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        latest["img"] = arr[:, :, :3][:, :, ::-1].copy()
        latest["n"] += 1

    camera.listen(on_image)

    # prime a few frames so camera has data
    for _ in range(5):
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        world.tick()
        spectator_follow(world, vehicle)

    n_steps = int(args.seconds / args.dt)
    # First ~2s pure motion (route steer only) so car is clearly driving before CUDA VLA
    vla_start_tick = int(2.0 / max(args.dt, 0.01))
    print(
        f"STEP3: drive {n_steps} ticks (~{args.seconds}s) motion_only={args.motion_only} "
        f"VLA starts after tick {vla_start_tick}",
        flush=True,
    )

    ste = 0.0
    moved = 0.0
    last = vehicle.get_location()
    ok_ticks = 0
    vla_calls = 0
    try:
        for i in range(n_steps):
            use_vla = (
                policy is not None
                and not args.motion_only
                and latest["img"] is not None
                and i >= vla_start_tick
                and (i % max(1, args.vla_every) == 0)
            )
            if use_vla:
                try:
                    tf = vehicle.get_transform()
                    vel = vehicle.get_velocity()
                    spd = math.hypot(vel.x, vel.y)
                    arrs = policy.predict_arrays(
                        ObservationBundle(
                            run_id="demo",
                            frame_id=f"f{i}",
                            scenario_id="turn",
                            simulation_time_s=i * args.dt,
                            ego_x=tf.location.x,
                            ego_y=tf.location.y,
                            ego_yaw=math.radians(tf.rotation.yaw),
                            ego_v=spd,
                            route_xy=tuple(route) if route else ((tf.location.x + 20, tf.location.y),),
                            front_rgb=latest["img"],
                        )
                    )
                    vla_calls += 1
                    pts = arrs[0].points_xy_yaw_v_a_kappa
                    if len(pts) >= 4:
                        tx, ty = pts[3][0], pts[3][1]
                        desired = math.atan2(ty - tf.location.y, tx - tf.location.x)
                        err = desired - math.radians(tf.rotation.yaw)
                        while err > math.pi:
                            err -= 2 * math.pi
                        while err < -math.pi:
                            err += 2 * math.pi
                        ste = max(-0.7, min(0.7, err * 0.9))
                        print(f"VLA#{vla_calls} ste={ste:.2f} at tick={i}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("vla_fail", type(exc).__name__, exc, flush=True)
                    ste = route_steer(vehicle, route)
            else:
                ste = 0.65 * ste + 0.35 * route_steer(vehicle, route)

            thr = float(args.throttle)
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=thr,
                    steer=float(ste),
                    brake=0.0,
                    hand_brake=False,
                    gear=1,
                )
            )
            try:
                frame = world.tick()
            except Exception as exc:
                print("TICK FATAL (CARLA died):", type(exc).__name__, exc, flush=True)
                break
            spectator_follow(world, vehicle)
            ok_ticks += 1

            loc = vehicle.get_location()
            moved += math.hypot(loc.x - last.x, loc.y - last.y)
            last = loc
            vel = vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            if i < 25 or i % 15 == 0:
                print(
                    f"tick={i} frame={frame} speed={speed:.2f} thr={thr:.2f} ste={ste:.2f} "
                    f"moved={moved:.1f}m cam={latest['n']} xyz=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})",
                    flush=True,
                )
    finally:
        try:
            camera.stop()
            camera.destroy()
        except Exception:
            pass
        try:
            vehicle.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original)
        except Exception:
            pass

    print(f"DONE ok_ticks={ok_ticks}/{n_steps} moved={moved:.1f}m vla_calls={vla_calls}", flush=True)
    return 0 if moved > 2.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
