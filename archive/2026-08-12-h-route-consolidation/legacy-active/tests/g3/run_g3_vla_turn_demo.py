#!/usr/bin/env python3
"""Two-process pure-VLA turn demo (watchable).

Process A (this file): CARLA only — NO torch/CUDA.
Process B (vla_worker_ipc.py): VLA on CPU — NO carla, NO CUDA.

Serialized IPC: never run VLA concurrent with world.tick() (same 4080 / host
contention freezes UE even across processes when both are busy).

Usage (WSL, CARLA already READY):
  python tests/g3/run_g3_vla_turn_demo.py --seconds 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
from runtime.carla_connection import ConnectionResolver  # noqa: E402

IPC = Path("/tmp/sdf_vla_ipc")


def spawn_turn(world: carla.World, seed: int = 11):
    import random

    m = world.get_map()
    spawns = list(m.get_spawn_points())
    random.Random(seed).shuffle(spawns)
    best = None
    for tf in spawns[:50]:
        wp = m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        st = wp.transform
        st.location.z = max(float(st.location.z), 1.0) + 0.3
        if abs(st.location.x) + abs(st.location.y) < 25:
            continue
        route = []
        yaw0 = st.rotation.yaw
        max_turn = 0.0
        cur = wp
        for _ in range(100):
            route.append((cur.transform.location.x, cur.transform.location.y))
            max_turn = max(max_turn, abs((cur.transform.rotation.yaw - yaw0 + 180) % 360 - 180))
            nxt = cur.next(2.0)
            if not nxt:
                break
            cur = nxt[0]
        if len(route) < 25:
            continue
        if max_turn >= 50:
            return st, route, max_turn
        if best is None or max_turn > best[2]:
            best = (st, route, max_turn)
    return best if best else (spawns[0], [], 0.0)


def route_steer(vehicle, route, look=12) -> float:
    if len(route) < 2:
        return 0.0
    tf = vehicle.get_transform()
    best_i, best_d = 0, 1e18
    for i, (x, y) in enumerate(route):
        d = (x - tf.location.x) ** 2 + (y - tf.location.y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    j = min(best_i + look, len(route) - 1)
    tx, ty = route[j]
    err = math.atan2(ty - tf.location.y, tx - tf.location.x) - math.radians(tf.rotation.yaw)
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return max(-0.7, min(0.7, err * 0.85))


def spectator_follow(world, vehicle) -> None:
    try:
        sp = world.get_spectator()
        tf = vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        loc = carla.Location(
            tf.location.x - 8 * math.cos(yaw),
            tf.location.y - 8 * math.sin(yaw),
            tf.location.z + 5.5,
        )
        sp.set_transform(carla.Transform(loc, carla.Rotation(pitch=-22, yaw=tf.rotation.yaw)))
    except Exception:
        pass


def wait_vla_cmd(frame_id: int, timeout_s: float = 180.0) -> dict | None:
    """Block until worker writes cmd for this frame (no CARLA ticks during wait)."""
    t0 = time.time()
    cmd_path = IPC / "cmd.json"
    while time.time() - t0 < timeout_s:
        if cmd_path.is_file():
            try:
                cmd = json.loads(cmd_path.read_text(encoding="utf-8"))
                if int(cmd.get("frame", -1)) == frame_id and cmd.get("ok"):
                    return cmd
            except Exception:
                pass
        time.sleep(0.05)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--throttle", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--vla-every", type=int, default=20, help="ticks between serialized VLA calls")
    ap.add_argument("--max-vla", type=int, default=8, help="max VLA inferences this run")
    args = ap.parse_args()

    IPC.mkdir(parents=True, exist_ok=True)
    for name in ("stop", "worker.ready", "frame.npz", "cmd.json"):
        p = IPC / name
        if p.exists():
            p.unlink()

    py = "/home/sdf/.venvs/sdf/bin/python"
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = f"{ROOT}/simlingo-main:{ROOT}/safedrive_foundry"
    # CUDA + serialize: worker borrows GPU only while driver is paused (no tick).
    env["SDF_VLA_DEVICE"] = os.environ.get("SDF_VLA_DEVICE", "cuda")
    worker = subprocess.Popen(
        [py, str(ROOT / "tests/g3/vla_worker_ipc.py")],
        cwd=str(ROOT),
        env=env,
    )
    print(f"waiting VLA worker ready (device={env['SDF_VLA_DEVICE']})...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if (IPC / "worker.ready").exists():
            break
        if worker.poll() is not None:
            print("worker died early", worker.returncode, flush=True)
            return 1
        time.sleep(0.5)
    if not (IPC / "worker.ready").exists():
        print("worker timeout", flush=True)
        worker.kill()
        return 1
    print("VLA worker ready — connect CARLA", flush=True)

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=8.0)
    report = resolver.preflight()
    print("preflight", report.status, flush=True)
    if report.status != "READY":
        (IPC / "stop").write_text("1")
        worker.wait(timeout=10)
        return 75
    client, report = resolver.connect(report=report)
    client.set_timeout(30.0)
    world = client.get_world()
    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        try:
            a.destroy()
        except Exception:
            pass

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = args.dt
    s.substepping = True
    s.max_substep_delta_time = 0.01
    s.max_substeps = 10
    world.apply_settings(s)

    spawn, route, turn = spawn_turn(world, args.seed)
    print(f"spawn ({spawn.location.x:.1f},{spawn.location.y:.1f},{spawn.location.z:.1f}) turn≈{turn:.0f}°", flush=True)
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    vehicle = world.try_spawn_actor(bp, spawn)
    if vehicle is None:
        print("spawn failed", flush=True)
        (IPC / "stop").write_text("1")
        return 1

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "180")
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.4, z=2.0)), attach_to=vehicle)
    latest = {"img": None, "n": 0}

    def on_img(image: carla.Image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        latest["img"] = arr[:, :, :3][:, :, ::-1].copy()
        latest["n"] += 1

    camera.listen(on_img)

    for _ in range(5):
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        world.tick()
        spectator_follow(world, vehicle)

    loc0 = vehicle.get_location()
    print(f"=== EGO VISIBLE id={vehicle.id} xyz=({loc0.x:.1f},{loc0.y:.1f},{loc0.z:.1f}) WATCH WINDOW ===", flush=True)

    n = int(args.seconds / args.dt)
    ste = 0.0
    moved = 0.0
    last = vehicle.get_location()
    vla_used = 0
    vla_calls = 0
    try:
        for i in range(n):
            # Serialized VLA: pause ticking, wait for result, then resume.
            do_vla = (
                latest["img"] is not None
                and vla_calls < args.max_vla
                and i > 0
                and i % max(1, args.vla_every) == 0
            )
            if do_vla:
                tf = vehicle.get_transform()
                vel = vehicle.get_velocity()
                # remove old cmd so we wait for this frame
                cmd_path = IPC / "cmd.json"
                if cmd_path.exists():
                    cmd_path.unlink()
                np.savez(
                    IPC / "frame.npz",
                    rgb=latest["img"],
                    ego_x=tf.location.x,
                    ego_y=tf.location.y,
                    ego_yaw=math.radians(tf.rotation.yaw),
                    ego_v=math.hypot(vel.x, vel.y),
                    route=np.array(route, dtype=np.float32) if route else np.zeros((0, 2)),
                    frame=i,
                    t=i * args.dt,
                )
                print(f"--- pause sim for VLA frame={i} ---", flush=True)
                cmd = wait_vla_cmd(i, timeout_s=200.0)
                if cmd is not None:
                    ste = float(cmd["steer"])
                    vla_used += 1
                    vla_calls += 1
                    print(
                        f"VLA ok ste={ste:.2f} lat_ms={float(cmd.get('latency_s', 0))*1000:.0f} "
                        f"calls={vla_calls}",
                        flush=True,
                    )
                else:
                    print(f"VLA timeout frame={i}, keep route ste", flush=True)
                    ste = route_steer(vehicle, route)
                    vla_calls += 1  # don't retry forever
            else:
                # between VLA calls: soft blend route so car keeps moving
                if vla_used == 0:
                    ste = 0.5 * ste + 0.5 * route_steer(vehicle, route)

            vehicle.apply_control(
                carla.VehicleControl(throttle=float(args.throttle), steer=float(ste), brake=0.0, hand_brake=False)
            )
            world.tick()
            spectator_follow(world, vehicle)
            loc = vehicle.get_location()
            moved += math.hypot(loc.x - last.x, loc.y - last.y)
            last = loc
            speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            if i < 15 or i % 20 == 0:
                print(
                    f"tick={i} speed={speed:.2f} ste={ste:.2f} moved={moved:.1f}m "
                    f"cam={latest['n']} vla={vla_used}",
                    flush=True,
                )
    finally:
        (IPC / "stop").write_text("1")
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
        try:
            worker.wait(timeout=15)
        except Exception:
            worker.kill()

    print(f"DONE moved={moved:.1f}m vla_msgs={vla_used}", flush=True)
    return 0 if moved > 5.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
