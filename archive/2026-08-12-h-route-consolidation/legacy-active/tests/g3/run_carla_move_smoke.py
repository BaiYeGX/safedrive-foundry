#!/usr/bin/env python3
"""No-torch CARLA motion smoke — prove spawn+tick+throttle without CUDA.

If this moves the car, CARLA is fine; VLA must not init CUDA before/while ticking
without releasing the GPU carefully.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
from runtime.carla_connection import ConnectionResolver  # noqa: E402


def main() -> int:
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=8.0)
    report = resolver.preflight()
    print("preflight", report.status, flush=True)
    if report.status != "READY":
        return 75
    client, report = resolver.connect(report=report)
    client.set_timeout(8.0)
    world = client.get_world()
    for a in list(world.get_actors().filter("vehicle.*")):
        try:
            a.destroy()
        except Exception:
            pass

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    s.substepping = True
    s.max_substep_delta_time = 0.01
    s.max_substeps = 10
    world.apply_settings(s)

    # Use classic-style road spawn (NOT near 0,0 junk)
    m = world.get_map()
    spawn = None
    for tf in m.get_spawn_points():
        wp = m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        st = wp.transform
        st.location.z = max(st.location.z, 1.0) + 0.3
        if abs(st.location.x) + abs(st.location.y) < 20:
            continue
        spawn = st
        break
    if spawn is None:
        spawn = m.get_spawn_points()[5]
        spawn.location.z = max(spawn.location.z, 1.2)

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    vehicle = world.try_spawn_actor(bp, spawn)
    if vehicle is None:
        print("spawn fail", spawn.location)
        return 1
    print(f"spawned id={vehicle.id} at ({spawn.location.x:.1f},{spawn.location.y:.1f},{spawn.location.z:.1f})", flush=True)

    moved = 0.0
    last = vehicle.get_location()
    try:
        for i in range(200):
            vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0, brake=0.0, hand_brake=False))
            world.tick()
            try:
                sp = world.get_spectator()
                tf = vehicle.get_transform()
                yaw = math.radians(tf.rotation.yaw)
                sp.set_transform(
                    carla.Transform(
                        carla.Location(
                            tf.location.x - 8 * math.cos(yaw),
                            tf.location.y - 8 * math.sin(yaw),
                            tf.location.z + 5,
                        ),
                        carla.Rotation(pitch=-20, yaw=tf.rotation.yaw),
                    )
                )
            except Exception:
                pass
            loc = vehicle.get_location()
            moved += math.hypot(loc.x - last.x, loc.y - last.y)
            last = loc
            vel = vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            if i < 15 or i % 20 == 0:
                print(f"i={i} speed={speed:.2f} moved={moved:.1f} xyz=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})", flush=True)
    finally:
        try:
            vehicle.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original)
        except Exception:
            pass
    print(f"DONE moved={moved:.1f}m", flush=True)
    return 0 if moved > 3.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
