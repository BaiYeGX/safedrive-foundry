#!/usr/bin/env python3
"""Windows CARLA camera diagnostics: sync/async/no_render variants."""
from __future__ import annotations

import math
import sys
import time

import carla


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "async_cam"  # async_cam|sync_cam|sync_none
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(25.0)
    world = client.get_world()
    print("mode", mode, "map", world.get_map().name, flush=True)
    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        try:
            a.destroy()
        except Exception:
            pass

    orig = world.get_settings()
    s = world.get_settings()
    if mode.startswith("sync"):
        s.synchronous_mode = True
        s.fixed_delta_seconds = 0.05
        s.substepping = True
        s.max_substep_delta_time = 0.01
        s.max_substeps = 10
    else:
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
    if "norender" in mode:
        s.no_rendering_mode = True
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
    print(f"spawn ({sp.location.x:.1f},{sp.location.y:.1f},{sp.location.z:.1f})", flush=True)
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    veh = world.try_spawn_actor(bp, sp)
    assert veh is not None

    cam = None
    n = {"c": 0}
    if "cam" in mode:
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "256")
        cam_bp.set_attribute("image_size_y", "144")
        cam_bp.set_attribute("sensor_tick", "0.1")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.4, z=2.0)), attach_to=veh)

        def on_img(_img: carla.Image) -> None:
            n["c"] += 1

        cam.listen(on_img)

    # spectator follow once
    try:
        spc = world.get_spectator()
        tf = veh.get_transform()
        spc.set_transform(
            carla.Transform(
                tf.location + carla.Location(x=-8, z=6),
                carla.Rotation(pitch=-20, yaw=tf.rotation.yaw),
            )
        )
    except Exception:
        pass

    moved = 0.0
    last = veh.get_location()
    t_end = time.time() + 12.0
    i = 0
    try:
        while time.time() < t_end:
            veh.apply_control(carla.VehicleControl(throttle=0.5, steer=0.02, brake=0.0))
            if mode.startswith("sync"):
                world.tick()
            else:
                time.sleep(0.05)
                world.wait_for_tick(seconds=2.0)
            loc = veh.get_location()
            moved += math.hypot(loc.x - last.x, loc.y - last.y)
            last = loc
            if i < 12 or i % 20 == 0:
                sp_d = math.hypot(veh.get_velocity().x, veh.get_velocity().y)
                print(f"i={i} speed={sp_d:.2f} moved={moved:.1f} cam={n['c']}", flush=True)
            i += 1
    finally:
        if cam is not None:
            try:
                cam.stop()
                cam.destroy()
            except Exception as exc:
                print("cam", exc)
        try:
            veh.destroy()
        except Exception as exc:
            print("veh", exc)
        try:
            world.apply_settings(orig)
        except Exception as exc:
            print("settings", exc)
    print("DONE", mode, "moved", moved, "cam", n["c"], "ticks", i, flush=True)
    return 0 if moved > 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
