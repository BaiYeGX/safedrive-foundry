#!/usr/bin/env python3
"""VLA worker process — never imports carla.

Default: CUDA (fast). Driver MUST serialize: no world.tick() during inference.
Override: SDF_VLA_DEVICE=cpu if CUDA still freezes the Windows UE process.

IPC files under /tmp/sdf_vla_ipc:
  frame.npz  : rgb, speed, ego pose, route, frame id
  cmd.json   : {steer, ts, ok, frame, latency_s}
  worker.ready
  stop
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simlingo-main"))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

IPC = Path("/tmp/sdf_vla_ipc")
IPC.mkdir(parents=True, exist_ok=True)


def _route_target(obs) -> tuple[float, float]:
    if obs.route_xy and len(obs.route_xy) >= 2:
        rx, ry = obs.route_xy[min(10, len(obs.route_xy) - 1)]
        dx = rx - obs.ego_x
        dy = ry - obs.ego_y
        c, s = math.cos(-obs.ego_yaw), math.sin(-obs.ego_yaw)
        return (float(c * dx - s * dy), float(s * dx + c * dy))
    return (12.0, 0.0)


def main() -> int:
    device = os.environ.get("SDF_VLA_DEVICE", "cuda").strip().lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        use_cuda = False
    else:
        use_cuda = True

    from driving_vla.adapter.policy_adapter import ObservationBundle
    from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

    print(f"VLA worker: loading model device={device}...", flush=True)
    rt = SimLingoNeuralRuntime(device="cuda" if use_cuda else "cpu")
    rep = rt.load()
    print("VLA worker load", rep.ok, rep.n_params, rep.error, flush=True)
    if not rep.ok:
        return 1

    warm = np.full((180, 320, 3), 100, dtype=np.uint8)
    print("VLA warm start...", flush=True)
    try:
        result = rt.forward_numpy(
            warm, speed_mps=3.0, target_point_xy=(12.0, 0.0), borrow_gpu=use_cuda
        )
        print(f"VLA warm lat_ms={result.latency_s*1000:.0f} device={device}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("warm fail", type(exc).__name__, exc, flush=True)
        return 1

    # After warm, release GPU so CARLA can start cleanly; re-borrow per forward if cuda.
    if use_cuda:
        try:
            rt.release_gpu_for_carla()
        except Exception as exc:  # noqa: BLE001
            print("release warn", exc, flush=True)

    print("VLA worker READY", flush=True)
    (IPC / "worker.ready").write_text("1", encoding="utf-8")

    last_seen = 0.0
    while not (IPC / "stop").exists():
        frame_path = IPC / "frame.npz"
        if not frame_path.is_file():
            time.sleep(0.05)
            continue
        try:
            mtime = frame_path.stat().st_mtime
            if mtime <= last_seen:
                time.sleep(0.02)
                continue
            data = np.load(frame_path, allow_pickle=True)
            last_seen = mtime
            rgb = data["rgb"]
            route = [tuple(x) for x in data["route"].tolist()] if "route" in data else []
            obs = ObservationBundle(
                run_id="ipc",
                frame_id=str(int(data.get("frame", 0))),
                scenario_id="ipc",
                simulation_time_s=float(data.get("t", 0.0)),
                ego_x=float(data["ego_x"]),
                ego_y=float(data["ego_y"]),
                ego_yaw=float(data["ego_yaw"]),
                ego_v=float(data["ego_v"]),
                route_xy=tuple(route) if route else ((float(data["ego_x"]) + 20, float(data["ego_y"])),),
                front_rgb=rgb,
            )
            result = rt.forward_numpy(
                rgb,
                speed_mps=float(obs.ego_v if obs.ego_v > 0.05 else 3.0),
                target_point_xy=_route_target(obs),
                borrow_gpu=use_cuda,
            )
            c, s = math.cos(obs.ego_yaw), math.sin(obs.ego_yaw)
            path_map = []
            for x, y in result.route_xy:
                path_map.append(
                    (obs.ego_x + c * float(x) - s * float(y), obs.ego_y + s * float(x) + c * float(y))
                )
            ste = 0.0
            if len(path_map) >= 4:
                tx, ty = path_map[3][0], path_map[3][1]
                desired = math.atan2(ty - obs.ego_y, tx - obs.ego_x)
                err = desired - obs.ego_yaw
                while err > math.pi:
                    err -= 2 * math.pi
                while err < -math.pi:
                    err += 2 * math.pi
                ste = max(-0.7, min(0.7, err * 0.9))
            cmd = {
                "steer": ste,
                "ts": time.time(),
                "ok": True,
                "latency_s": result.latency_s,
                "device": device,
                "frame": int(data.get("frame", -1)),
            }
            (IPC / "cmd.json").write_text(json.dumps(cmd), encoding="utf-8")
            print(
                f"VLA frame={cmd['frame']} ste={ste:.2f} lat_ms={result.latency_s*1000:.0f} {device}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print("worker err", type(exc).__name__, exc, flush=True)
            time.sleep(0.1)
    print("VLA worker stop", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
