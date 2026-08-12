#!/usr/bin/env python3
"""Diagnose camera vs no-camera tick stability."""
from __future__ import annotations

import math
import queue
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
from runtime.carla_connection import ConnectionResolver  # noqa: E402


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "queue"  # none|listen|queue
    r = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=8.0)
    rep = r.preflight()
    print("preflight", rep.status, "host", getattr(rep, "host", "?"), flush=True)
    if rep.status != "READY":
        return 75
    client, rep = r.connect(report=rep)
    client.set_timeout(20.0)
    world = client.get_world()
    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        try:
            a.destroy()
        except Exception:
            pass
    orig = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    s.substepping = True
    s.max_substep_delta_time = 0.01
    s.max_substeps = 10
    world.apply_settings(s)
    m = world.get_map()
    sp = None
    for tf in m.get_spawn_points():
        wp = m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        st = wp.transform
        st.location.z = max(float(st.location.z), 1.0) + 0.5
        if abs(st.location.x) + abs(st.location.y) > 25:
            sp = st
            break
    print("mode", mode, "spawn", f"({sp.location.x:.1f},{sp.location.y:.1f},{sp.location.z:.1f})", flush=True)
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    veh = world.try_spawn_actor(bp, sp)
    if veh is None:
        print("spawn failed")
        return 1

    cam = None
    q: queue.Queue = queue.Queue(maxsize=4)
    n = {"c": 0}
    if mode != "none":
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "160")
        cam_bp.set_attribute("image_size_y", "90")
        cam_bp.set_attribute("fov", "90")
        try:
            cam_bp.set_attribute("enable_postprocess_effects", "False")
        except Exception:
            pass
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.4, z=2.0)), attach_to=veh)
        if mode == "listen":

            def on_img(_img: carla.Image) -> None:
                n["c"] += 1

            cam.listen(on_img)
        else:

            def on_img(img: carla.Image) -> None:
                n["c"] += 1
                try:
                    q.put_nowait(img.frame)
                except queue.Full:
                    pass

            cam.listen(on_img)

    # settle without throttle
    for i in range(10):
        veh.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        world.tick()
        print(f"settle={i} z={veh.get_location().z:.2f} cam={n['c']}", flush=True)

    moved = 0.0
    last = veh.get_location()
    try:
        for i in range(120):
            veh.apply_control(carla.VehicleControl(throttle=0.5, steer=0.02, brake=0.0))
            world.tick()
            loc = veh.get_location()
            moved += math.hypot(loc.x - last.x, loc.y - last.y)
            last = loc
            if i < 8 or i % 20 == 0:
                sp_d = math.hypot(veh.get_velocity().x, veh.get_velocity().y)
                print(f"tick={i} speed={sp_d:.2f} moved={moved:.1f} cam={n['c']}", flush=True)
    finally:
        if cam is not None:
            try:
                cam.stop()
                cam.destroy()
            except Exception as exc:
                print("cam destroy", exc, flush=True)
        try:
            veh.destroy()
        except Exception as exc:
            print("veh destroy", exc, flush=True)
        try:
            world.apply_settings(orig)
        except Exception as exc:
            print("settings", exc, flush=True)
    print("DONE mode", mode, "moved", moved, flush=True)
    return 0 if moved > 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
