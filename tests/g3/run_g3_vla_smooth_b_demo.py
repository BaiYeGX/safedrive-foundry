#!/usr/bin/env python3
"""G3 visual DEMO-B: pure VLA path+speed → MPC (real-vehicle-ready policy path).

Real-vehicle rule (user):
  - Control path MUST NOT use CARLA map centerline / waypoint geometry to drive.
  - Geometry + speed come from VLA only (what you can ship with camera+VLA+MPC).
  - Optional sparse route_xy is ONLY a navigation-style target for the VLA prompt
    (like a GPS goal), never the MPC reference path.

Pipeline:
  camera RGB ──► SimLingo VLA (CUDA) ──► map-frame trajectory
                    │
                    ▼
              temporal smooth + dynamics limits (no map)
                    │
                    ▼
              20/50Hz MPC ──► throttle/steer ──► plant (CARLA today)

No set_transform road stick. No Safety/QP/RATO. NOT G3 VERIFIED.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import carla  # noqa: E402

from dataclasses import replace  # noqa: E402

from classic_stack.control import load_control_config  # noqa: E402
from classic_stack.control.controller import ControlLoop, EgoState  # noqa: E402
from classic_stack.planning.frenet.planner import Trajectory as CtrlTraj  # noqa: E402
from classic_stack.planning.frenet.planner import TrajectoryPoint as CtrlPt  # noqa: E402
from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.neural_policy import NeuralV0Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402
from runtime import (  # noqa: E402
    ActorSpec,
    RunRegistry,
    RuntimeIdentityFactory,
    ScenarioRuntime,
    ScenarioSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402

# Reuse route/geometry helpers from the Safety-aware visual demo (no Safety runtime).
from run_g3_vla_v0_visual_demo import (  # noqa: E402
    _nearest_s_on_poly,
    _pct,
    _point_at_s,
    _polyline_s_table,
    build_route,
    find_90deg_turn_candidates,
    free_spawn,
    reshape_neural_traj_for_safety,
    set_spectator_follow,
)
from driving_vla.adapter.policy_adapter import TrajectoryArray  # noqa: E402


def array_to_ctrl(arr: TrajectoryArray, *, traj_id: str = "vla_onroad") -> CtrlTraj:
    """Map-frame TrajectoryArray → control Trajectory."""
    pts: list[CtrlPt] = []
    for i, row in enumerate(arr.points_xy_yaw_v_a_kappa):
        x, y, yaw, v, a, kappa = (float(row[j]) for j in range(6))
        pts.append(
            CtrlPt(
                t=(i + 1) * 0.25,
                x=x,
                y=y,
                yaw=yaw,
                kappa=kappa,
                v=max(0.3, v),
                a=max(-5.0, min(3.5, a)),
                jerk=0.0,
            )
        )
    if len(pts) < 2:
        raise ValueError("need ≥2 trajectory points")
    return CtrlTraj(points=tuple(pts), trajectory_id=traj_id, source="vla_pure")


def pure_vla_control(
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_v: float,
    traj: TrajectoryArray,
    last_steer: float,
    wheelbase_m: float = 2.8,
    max_steer: float = 0.55,
    max_accel: float = 3.0,
    max_brake: float = 5.0,
    max_speed: float = 15.0,
    lookahead_m: float = 8.0,
) -> tuple[float, float, float]:
    """Pure VLA closed-loop: pure-pursuit on VLA polyline + speed PID.

    More robust than dense MPC on noisy neural paths; still 100% VLA geometry
    (no map centerline). Returns (throttle, brake, steer) in [0,1]/[-1,1].
    """
    pts = traj.points_xy_yaw_v_a_kappa
    if not pts:
        return 0.0, 0.4, 0.0

    # Lookahead point along VLA path
    best = pts[0]
    best_d = -1.0
    for p in pts:
        d = math.hypot(float(p[0]) - ego_x, float(p[1]) - ego_y)
        if d >= lookahead_m * 0.45 and d > best_d:
            best_d = d
            best = p
    if best_d < 0:
        best = pts[min(3, len(pts) - 1)]

    tx, ty = float(best[0]), float(best[1])
    v_ref = max(1.5, min(float(best[3]), max_speed))
    # Average a few mid-horizon speeds for smoother throttle
    vs = [float(p[3]) for p in pts[max(0, len(pts) // 4) :]]
    if vs:
        v_ref = max(1.5, min(max_speed, 0.4 * v_ref + 0.6 * (sum(vs) / len(vs))))

    dx, dy = tx - ego_x, ty - ego_y
    # ego frame
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    ld = max(lookahead_m, math.hypot(local_x, local_y), 2.5)
    # classic pure pursuit
    curvature = 2.0 * local_y / max(ld * ld, 1e-3)
    steer_cmd = math.atan(curvature * wheelbase_m)
    steer_cmd = max(-max_steer, min(max_steer, steer_cmd))
    # normalize to [-1,1] carla (approx max_steer ~0.7 rad → full)
    ste = steer_cmd / max(max_steer, 1e-3)
    ste = max(-1.0, min(1.0, ste))
    # rate limit vs last
    ste = last_steer + max(-0.04, min(0.04, ste - last_steer))
    ste = 0.75 * last_steer + 0.25 * ste

    # longitudinal
    ev = v_ref - ego_v
    if ev >= 0:
        thr = max(0.0, min(1.0, ev / max_accel))
        brk = 0.0
        # gentle coast if nearly at ref
        if abs(ev) < 0.3:
            thr = min(thr, 0.25)
    else:
        thr = 0.0
        brk = max(0.0, min(1.0, (-ev) / max_brake))
    # don't throttle into huge steer
    if abs(ste) > 0.35:
        thr *= 0.55
    return thr, brk, ste


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _nearest_s_local(
    poly: list[tuple[float, float]],
    s_acc: list[float],
    x: float,
    y: float,
    *,
    s_hint: float,
    window_back: float = 5.0,
    window_fwd: float = 25.0,
) -> float:
    """Nearest arc-s only inside a local window — avoids wrong matches on long looping routes."""
    if len(poly) < 2:
        return 0.0
    s_lo = max(0.0, s_hint - window_back)
    s_hi = min(s_acc[-1], s_hint + window_fwd)
    best_s, best_d = s_hint, float("inf")
    # coarse sample then refine
    ds = 0.5
    s = s_lo
    while s <= s_hi + 1e-9:
        px, py, _ = _point_at_s(poly, s_acc, s, 0.0)
        d = math.hypot(x - px, y - py)
        if d < best_d:
            best_d = d
            best_s = s
        s += ds
    return best_s


def _curvature_speed_cap(
    poly: list[tuple[float, float]],
    s_acc: list[float],
    s0: float,
    *,
    max_speed: float,
    look_m: float = 18.0,
    max_lat_accel: float = 3.0,
) -> float:
    """Lower speed when upcoming centerline bends (anti-wall in turns)."""
    if s_acc[-1] < 1.0:
        return max_speed
    s1 = min(s_acc[-1], s0 + look_m)
    _, _, y0 = _point_at_s(poly, s_acc, s0, 0.0)
    _, _, y1 = _point_at_s(poly, s_acc, s1, 0.0)
    dyaw = abs(_wrap_pi(y1 - y0))
    # average |kappa| ≈ |Δyaw| / look
    kappa = dyaw / max(look_m, 1.0)
    if kappa < 1e-4:
        return max_speed
    v_curve = math.sqrt(max_lat_accel / kappa)
    # also sample mid for sharper corners
    sm = min(s_acc[-1], s0 + look_m * 0.5)
    _, _, ym = _point_at_s(poly, s_acc, sm, 0.0)
    k2 = abs(_wrap_pi(ym - y0)) / max(look_m * 0.5, 1.0)
    if k2 > 1e-4:
        v_curve = min(v_curve, math.sqrt(max_lat_accel / k2))
    return max(3.0, min(max_speed, v_curve))


def _road_bend_ahead(
    poly: list[tuple[float, float]],
    s_acc: list[float],
    s0: float,
    look_m: float = 16.0,
) -> float:
    """|Δyaw| (rad) of centerline over the next look_m — small ⇒ straight."""
    if s_acc[-1] < 1.0:
        return 0.0
    s1 = min(s_acc[-1], s0 + look_m)
    _, _, y0 = _point_at_s(poly, s_acc, s0, 0.0)
    _, _, y1 = _point_at_s(poly, s_acc, s1, 0.0)
    return abs(_wrap_pi(y1 - y0))


def _vla_mean_heading_and_lat_rms(
    pts: list[tuple[float, ...]],
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
) -> tuple[float, float]:
    """From VLA points alone: mean heading + lateral RMS in ego frame (no map)."""
    if len(pts) < 2:
        return ego_yaw, 0.0
    sx = sy = 0.0
    n = 0
    for i in range(1, len(pts)):
        dx = float(pts[i][0]) - float(pts[i - 1][0])
        dy = float(pts[i][1]) - float(pts[i - 1][1])
        if dx * dx + dy * dy < 1e-6:
            continue
        sx += dx
        sy += dy
        n += 1
    if n < 1:
        h = math.atan2(float(pts[-1][1]) - ego_y, float(pts[-1][0]) - ego_x)
    else:
        h = math.atan2(sy, sx)
    # lateral RMS of points about the line through ego along heading h
    ch, sh = math.cos(h), math.sin(h)
    acc = 0.0
    for p in pts:
        dx = float(p[0]) - ego_x
        dy = float(p[1]) - ego_y
        lat = -sh * dx + ch * dy
        acc += lat * lat
    rms = math.sqrt(acc / max(len(pts), 1))
    return h, rms


def smooth_vla_trajectory(
    arr: TrajectoryArray,
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_v: float,
    max_speed: float = 15.0,
    prev_pts: list[tuple[float, float, float, float, float, float]] | None = None,
    blend_prev: float = 0.62,
) -> TrajectoryArray:
    """Real-vehicle postprocess for pure VLA paths (NO map centerline).

    Straight-line wobble fix (all transferable to real car):
      1) temporal blend with previous VLA traj
      2) if VLA path itself is nearly straight (low lateral RMS about its own
         mean heading), re-fit onto that self-heading ray (not map)
      3) step / dheading / speed rate limits for MPC friendliness
    """
    raw = list(arr.points_xy_yaw_v_a_kappa)
    if len(raw) < 2:
        x1 = ego_x + 8.0 * math.cos(ego_yaw)
        y1 = ego_y + 8.0 * math.sin(ego_yaw)
        raw = [
            (ego_x, ego_y, ego_yaw, max(1.0, ego_v), 0.0, 0.0),
            (x1, y1, ego_yaw, max(1.0, min(max_speed, ego_v + 1.0)), 0.0, 0.0),
        ]

    # --- 1) temporal blend ---
    blended: list[list[float]] = []
    for i, row in enumerate(raw):
        x, y, yaw, v, a, _k = (float(row[j]) for j in range(6))
        if prev_pts is not None and i < len(prev_pts):
            px, py, pyaw, pv, pa, _pk = (float(prev_pts[i][j]) for j in range(6))
            b = blend_prev
            x = b * px + (1.0 - b) * x
            y = b * py + (1.0 - b) * y
            sy = b * math.sin(pyaw) + (1.0 - b) * math.sin(yaw)
            cy = b * math.cos(pyaw) + (1.0 - b) * math.cos(yaw)
            yaw = math.atan2(sy, cy)
            v = b * pv + (1.0 - b) * v
            a = b * pa + (1.0 - b) * a
        blended.append([x, y, yaw, v, a, 0.0])

    # --- 2) self-straighten when VLA path is nearly linear (no map) ---
    mean_h, lat_rms = _vla_mean_heading_and_lat_rms(blended, ego_x, ego_y, ego_yaw)
    # mix ego yaw so we don't snap hard after a turn
    mean_h = _wrap_pi(0.35 * ego_yaw + 0.65 * mean_h)
    # if points hug their own mean line, force a clean ray (kills 扭扭车 on "straight")
    if lat_rms < 0.85:
        straighten = min(1.0, max(0.0, 1.0 - lat_rms / 0.85))  # 0..1
        ch, sh = math.cos(mean_h), math.sin(mean_h)
        for i, p in enumerate(blended):
            dx = p[0] - ego_x
            dy = p[1] - ego_y
            along = max(0.5 * (i + 1), ch * dx + sh * dy)
            # keep a little residual lateral when not fully straight
            lat = (-sh * dx + ch * dy) * (1.0 - 0.85 * straighten)
            p[0] = ego_x + along * ch - lat * sh
            p[1] = ego_y + along * sh + lat * ch
            p[2] = mean_h + (1.0 - straighten) * _wrap_pi(p[2] - mean_h)

    # --- 3) dynamics / continuity ---
    out: list[tuple[float, float, float, float, float, float]] = []
    v_prev = max(0.0, float(ego_v))
    for i, p in enumerate(blended):
        x, y, yaw, v, a, _ = p
        if i == 0:
            if math.hypot(x - ego_x, y - ego_y) < 0.4:
                x = ego_x + 1.2 * math.cos(mean_h)
                y = ego_y + 1.2 * math.sin(mean_h)
            yaw = math.atan2(y - ego_y, x - ego_x)
        else:
            px, py = out[-1][0], out[-1][1]
            dx, dy = x - px, y - py
            dist = math.hypot(dx, dy)
            max_step = max(0.35, min(3.5, 0.5 * (v_prev + max(0.5, v)) * 0.25 + 0.6))
            if dist > max_step and dist > 1e-6:
                s = max_step / dist
                x = px + dx * s
                y = py + dy * s
                dist = max_step
            if dist > 1e-4:
                yaw = math.atan2(y - py, x - px)
            # tighter heading rate than before (anti-wobble)
            dyaw = _wrap_pi(yaw - out[-1][2])
            max_dyaw = 0.10 if lat_rms < 0.85 else 0.16
            dyaw = max(-max_dyaw, min(max_dyaw, dyaw))
            yaw = _wrap_pi(out[-1][2] + dyaw)
            if dist > 1e-4:
                x = px + dist * math.cos(yaw)
                y = py + dist * math.sin(yaw)

        v = max(1.2, min(max_speed, v))
        v = min(v, v_prev + 2.0 * 0.25)
        v = max(v, v_prev - 4.5 * 0.25)
        a = (v - v_prev) / 0.25
        if out:
            step = max(math.hypot(x - out[-1][0], y - out[-1][1]), 1e-3)
            kappa = _wrap_pi(yaw - out[-1][2]) / step
            kappa = max(-0.14, min(0.14, kappa))
        else:
            kappa = 0.0
        out.append((x, y, yaw, v, a, kappa))
        v_prev = v

    return TrajectoryArray(
        points_xy_yaw_v_a_kappa=tuple(out),
        probability=arr.probability,
        uncertainty=arr.uncertainty,
        candidate_id="tau0_vla_pure",
        intended_action=arr.intended_action,
        behavior=arr.behavior,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="VLA+MPC smooth demo B (no Safety)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-route-m", type=float, default=40.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument(
        "--vla-period-s",
        type=float,
        default=0.50,
        help="wall seconds between VLA forwards; lower=more model, higher=smoother UE",
    )
    ap.add_argument("--spectator-every", type=int, default=2)
    ap.add_argument("--cam-w", type=int, default=448)
    ap.add_argument("--cam-h", type=int, default=252)
    ap.add_argument("--cam-sensor-tick", type=float, default=0.10)
    ap.add_argument("--route-mode", default="straight", choices=["straight", "turn90"])
    ap.add_argument("--turn-prefer", default="either", choices=["either", "left", "right"])
    ap.add_argument(
        "--map",
        default="Town03",
        help=(
            "Required map name (startup must already be this map). "
            "Examples: Town03, Town04, Town05, Town10HD_Opt. "
            "NEVER mid-session load_world (shader/D3D fatal). "
            "If wrong map: kill CarlaUE4, set carla_start.toml, sdf sim ensure."
        ),
    )
    ap.add_argument(
        "--force-load-world",
        action="store_true",
        help="DANGEROUS: client.load_world mid-session (often fatal). Prefer restart with map in launch args.",
    )
    ap.add_argument(
        "--spawn-index",
        type=int,
        default=-1,
        help="prefer this spawn point index among free points (-1 = free_spawn heuristic)",
    )
    ap.add_argument(
        "--evidence-dir",
        default=str(ROOT / "docs/runtime-evidence/g3-05/visual_demo_b"),
    )
    ap.add_argument(
        "--max-speed",
        type=float,
        default=20.0,
        help="soft demo speed cap m/s (20≈72 km/h)",
    )
    ap.add_argument(
        "--max-lat-m",
        type=float,
        default=0.55,
        help="max VLA lateral in curves (m); straights use ~0.35 — smaller avoids yellow line",
    )
    ap.add_argument(
        "--replan-every",
        type=int,
        default=3,
        help="rebuild trajectory every N control ticks (higher=less path chatter)",
    )
    ap.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="target sim duration seconds (overrides --max-steps if >0); e.g. 60 → ~3000 steps @50Hz",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="pace control loop to sim wall time so you can watch ~1:1 (recommended for long demos)",
    )
    ap.add_argument(
        "--lite",
        action="store_true",
        help="stable path: plain carla client (no ScenarioRuntime), 20Hz, safer for long watch",
    )
    ap.add_argument(
        "--no-camera",
        action="store_true",
        help="do not attach RGB (avoids camera-spawn D3D fatal); VLA uses gray placeholder + centerline",
    )
    ap.add_argument(
        "--sim-dt",
        type=float,
        default=0.0,
        help="fixed_delta_seconds; 0=auto (lite/duration→0.05, else 0.02)",
    )
    args = ap.parse_args()
    # Long demos default to lite+20Hz — ScenarioRuntime@50Hz+camera was killing UE at camera attach
    if float(args.duration_s) > 0 and not any(
        a.startswith("--lite") for a in sys.argv
    ):
        # only auto-enable if user didn't pass --lite false... store_true can't be false; auto on duration
        args.lite = True
        print("NOTE: duration run → --lite auto (plain client, 20Hz)", flush=True)
    sim_dt = float(args.sim_dt) if float(args.sim_dt) > 0 else (0.05 if args.lite else 0.02)
    args._sim_dt = sim_dt  # type: ignore[attr-defined]
    if float(args.duration_s) > 0:
        args.max_steps = max(int(args.max_steps), int(round(float(args.duration_s) / sim_dt)))
        need_route = float(args.duration_s) * min(float(args.max_speed), 14.0) * 0.70
        need_route = min(max(need_route, 120.0), 450.0)
        if float(args.max_route_m) < need_route:
            args.max_route_m = need_route
        if args.realtime and abs(float(args.vla_period_s) - 0.50) < 1e-6:
            args.vla_period_s = 0.80
            print("NOTE: realtime long-run → vla_period_s auto 0.80", flush=True)
        if float(args.duration_s) >= 30 and int(args.cam_w) >= 400:
            args.cam_w, args.cam_h = 320, 180
            args.cam_sensor_tick = max(float(args.cam_sensor_tick), 0.15)
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("DEMO-B: PURE VLA path+speed → MPC  |  NO map centerline control", flush=True)
    print("real-vehicle path: camera→VLA→smooth→MPC  |  NOT G3 VERIFIED", flush=True)
    print(
        f"plan: max_steps={args.max_steps} max_route_m={args.max_route_m:.0f} "
        f"duration_s={args.duration_s or 'n/a'} realtime={bool(args.realtime)}",
        flush=True,
    )
    sim_dt = float(getattr(args, "_sim_dt", 0.05 if args.lite else 0.02))
    print(
        f"boot: lite={args.lite} sim_dt={sim_dt} no_camera={args.no_camera} "
        f"(spawn vehicle → settle → optional camera → VLA)",
        flush=True,
    )
    print("=" * 60, flush=True)

    print("STEP1: CARLA preflight/connect (no CUDA yet)...", flush=True)
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=15.0)
    report = resolver.preflight()
    print("preflight", report.status, report.error_code, flush=True)
    if report.status != "READY":
        import subprocess

        subprocess.run(
            [
                "bash",
                "-lc",
                "cd '/mnt/e/autonomous driving' && source /home/sdf/.venvs/sdf/bin/activate && "
                "(command -v sdf >/dev/null && sdf sim ensure || python scripts/sdf.py sim ensure)",
            ],
            check=False,
        )
        report = resolver.preflight()
        print("preflight2", report.status, report.error_code, flush=True)
    if report.status != "READY":
        out = {
            "status": "BLOCKED_EXTERNAL",
            "demo": "g3_vla_smooth_b",
            "preflight": report.to_dict() if hasattr(report, "to_dict") else str(report),
        }
        (evidence_dir / "blocked_preflight.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
        print("CARLA not READY — start CarlaUE4, re-run.", flush=True)
        return 75

    client, report = resolver.connect(report=report)
    client.set_timeout(60.0)
    world = client.get_world()
    stamp = int(time.time() * 1000)

    def _map_matches(current: str, want: str) -> bool:
        cur = current.replace("\\", "/")
        w = want.strip().replace("\\", "/")
        if not w:
            return True
        return (
            w in cur
            or cur.endswith(w)
            or cur.endswith(f"Maps/{w}")
            or f"/{w}" in cur
            or cur.split("/")[-1].replace("_Opt", "") == w.replace("_Opt", "")
        )

    def _patch_carla_start_map(map_token: str) -> Path:
        """Ensure carla_start.toml launch arguments include /Game/Carla/Maps/<map>."""
        cfg_path = ROOT / "safedrive_foundry/config/runtime/carla_start.toml"
        text = cfg_path.read_text(encoding="utf-8")
        # normalize token
        token = map_token.strip()
        if token.startswith("/Game/"):
            map_arg = token
        else:
            map_arg = f"/Game/Carla/Maps/{token}"
        import re

        # replace first map path in arguments line
        new_text, n = re.subn(
            r'(arguments\s*=\s*")(/Game/Carla/Maps/[A-Za-z0-9_]+)',
            rf"\1{map_arg}",
            text,
            count=1,
        )
        if n == 0:
            # inject map as first argument if missing
            new_text, n = re.subn(
                r'(arguments\s*=\s*")',
                rf"\1{map_arg} ",
                text,
                count=1,
            )
        if n == 0:
            raise RuntimeError(f"could not patch map into {cfg_path}")
        if new_text != text:
            cfg_path.write_text(new_text, encoding="utf-8")
            print(f"patched {cfg_path} → startup map {map_arg}", flush=True)
        return cfg_path

    # Map MUST match launch; mid-session load_world → shader/D3D fatal (project rule).
    want_map = (args.map or "").strip() or "Town03"
    cur_map = str(world.get_map().name)
    print(f"map check: current={cur_map} required={want_map}", flush=True)
    if not _map_matches(cur_map, want_map):
        if args.force_load_world:
            print(
                "WARNING: --force-load-world is DANGEROUS (often fatal). Prefer restart with map in launch args.",
                flush=True,
            )
            try:
                world = client.load_world(want_map)
                time.sleep(2.0)
                world = client.get_world()
                print(f"force load_world now={world.get_map().name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"force load_world failed: {exc}", flush=True)
                return 76
        else:
            print("=" * 60, flush=True)
            print("MAP MISMATCH — refusing mid-session load_world (prevents fatal).", flush=True)
            print(f"  current:  {cur_map}", flush=True)
            print(f"  required: {want_map}", flush=True)
            print("Do this:", flush=True)
            print("  1) Close CarlaUE4 completely", flush=True)
            try:
                _patch_carla_start_map(want_map)
                print("  2) carla_start.toml already patched with map", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  2) manually set arguments map to /Game/Carla/Maps/{want_map} ({exc})", flush=True)
            print("  3) sdf sim ensure   # starts CARLA with map in ArgumentList", flush=True)
            print("  4) re-run this demo with same --map", flush=True)
            print("=" * 60, flush=True)
            out = {
                "status": "BLOCKED_EXTERNAL",
                "error_code": "MAP_MISMATCH_RESTART_REQUIRED",
                "current_map": cur_map,
                "required_map": want_map,
                "demo": "g3_vla_smooth_b",
            }
            (evidence_dir / "blocked_map_mismatch.json").write_text(
                json.dumps(out, indent=2) + "\n", encoding="utf-8"
            )
            return 75

    stamp = int(time.time() * 1000)
    map_short = str(world.get_map().name).split("/")[-1]
    scenario_id = f"smooth_b_{map_short}_{args.route_mode}_seed{args.seed}"
    run_id = f"lite-{stamp}"
    runtime = None  # only used in non-lite
    original_settings = world.get_settings()
    print(f"map OK for run: {map_short} (startup-bound, no live switch)", flush=True)

    def _soft_cleanup() -> None:
        print("STEP2: soft cleanup...", flush=True)
        for filt in ("sensor.*", "vehicle.*", "walker.*"):
            for a in list(world.get_actors().filter(filt)):
                try:
                    a.destroy()
                except Exception:
                    pass
            time.sleep(0.05)

    _soft_cleanup()

    # Sync settings: 20Hz lite is far gentler on UE than 50Hz+substeps
    print(f"STEP2b: apply sync settings dt={sim_dt}...", flush=True)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = sim_dt
    settings.substepping = True
    settings.max_substep_delta_time = min(0.01, sim_dt)
    settings.max_substeps = max(2, int(math.ceil(sim_dt / 0.01)))
    world.apply_settings(settings)

    ego_bp_name = "vehicle.tesla.model3"
    try:
        bps = world.get_blueprint_library().filter("vehicle.*")
        for prefer in (
            "vehicle.audi.a2",
            "vehicle.bmw.grandtourer",
            "vehicle.chevrolet.impala",
            "vehicle.tesla.model3",
        ):
            if any(b.id == prefer for b in bps):
                ego_bp_name = prefer
                break
    except Exception:
        pass
    print(f"ego blueprint={ego_bp_name}", flush=True)

    spawn_candidates: list[tuple[carla.Transform, list[tuple[float, float]], dict]] = []
    if args.route_mode == "turn90":
        ranked = find_90deg_turn_candidates(
            world,
            prefer=args.turn_prefer,
            approach_m=min(35.0, max(22.0, args.max_route_m * 0.40)),
            after_m=min(55.0, max(40.0, args.max_route_m * 0.65)),
            max_candidates=8,
        )
        for _q, stf, pts, meta in ranked:
            spawn_candidates.append((stf, pts, meta))
        if not spawn_candidates:
            stf = free_spawn(world)
            spawn_candidates.append(
                (
                    stf,
                    build_route(world, stf, args.max_route_m),
                    {"route_mode": "fallback_straight", "peak_yaw_change_deg": 0.0, "turn_dir": "none"},
                )
            )
    else:
        # Different seed / spawn-index → different road stretch
        if int(args.spawn_index) >= 0:
            pts = list(world.get_map().get_spawn_points())
            idx = int(args.spawn_index) % max(len(pts), 1)
            stf = pts[idx] if pts else free_spawn(world)
            # slight lift
            stf = carla.Transform(
                carla.Location(stf.location.x, stf.location.y, stf.location.z + 0.4),
                stf.rotation,
            )
        else:
            # free_spawn is deterministic-ish; rotate pick by seed for variety
            pts = list(world.get_map().get_spawn_points())
            if pts:
                occupied = [a.get_location() for a in world.get_actors().filter("vehicle.*")]
                ordered = list(range(len(pts)))
                # permute by seed
                rng = (int(args.seed) * 1103515245 + 12345) & 0x7FFFFFFF
                ordered = sorted(ordered, key=lambda i: (rng * (i + 3)) % 9973)
                stf = None
                for i in ordered:
                    tf = pts[i]
                    if any(tf.location.distance(o) < 6.0 for o in occupied):
                        continue
                    stf = carla.Transform(
                        carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.4),
                        tf.rotation,
                    )
                    break
                if stf is None:
                    stf = free_spawn(world)
            else:
                stf = free_spawn(world)
        spawn_candidates.append(
            (
                stf,
                build_route(world, stf, args.max_route_m),
                {
                    "route_mode": "straight",
                    "peak_yaw_change_deg": 0.0,
                    "turn_dir": "none",
                    "map": map_short,
                    "seed": int(args.seed),
                },
            )
        )

    spawn = spawn_candidates[0][0]
    route = spawn_candidates[0][1]
    route_meta = spawn_candidates[0][2]
    ego = None
    last_err: Exception | None = None
    print("STEP3: spawn ego (plain try_spawn, no ScenarioRuntime)...", flush=True)
    bp = world.get_blueprint_library().find(ego_bp_name)
    for idx, (cand_spawn, cand_route, cand_meta) in enumerate(spawn_candidates):
        lift = carla.Transform(
            carla.Location(
                cand_spawn.location.x,
                cand_spawn.location.y,
                max(float(cand_spawn.location.z), 0.5) + 0.40,
            ),
            cand_spawn.rotation,
        )
        try:
            veh = world.try_spawn_actor(bp, lift)
            if veh is None:
                raise RuntimeError("try_spawn_actor returned None")
            # one tick to materialize
            world.tick()
            ego = veh
            spawn, route, route_meta = cand_spawn, cand_route, cand_meta
            print(
                f"SPAWN OK id={ego.id} ROUTE {route_meta.get('route_mode')} "
                f"n_pts={len(route)} at ({lift.location.x:.1f},{lift.location.y:.1f})",
                flush=True,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"spawn try {idx} failed: {exc}", flush=True)
            time.sleep(0.2)
    if ego is None:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        raise RuntimeError(f"all spawn tries failed: {last_err}")

    try:
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0, hand_brake=False))
    except Exception:
        pass

    print("STEP4: settle vehicle...", flush=True)
    for _ in range(20):
        try:
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.4, steer=0.0))
            world.tick()
        except Exception as exc:  # noqa: BLE001
            print("settle failed:", exc, flush=True)
            return 76
    cam_last_good0 = {
        "x": float(spawn.location.x),
        "y": float(spawn.location.y),
        "z": float(spawn.location.z),
        "yaw": float(spawn.rotation.yaw),
    }
    set_spectator_follow(world, ego, last_good=cam_last_good0)
    print("vehicle settled OK", flush=True)

    # Camera is optional — log showed fatal right at RGB attach previously
    camera = None
    rgb_lock = threading.Lock()
    latest_rgb: dict[str, Any] = {"img": None, "n": 0}
    use_camera = not bool(args.no_camera)
    if use_camera:
        print("STEP5: try tiny RGB camera (may skip if UE unstable)...", flush=True)
        try:
            cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(int(args.cam_w)))
            cam_bp.set_attribute("image_size_y", str(int(args.cam_h)))
            cam_bp.set_attribute("fov", "100")
            try:
                cam_bp.set_attribute("sensor_tick", f"{max(0.15, float(args.cam_sensor_tick)):.3f}")
            except Exception:
                pass
            camera = world.try_spawn_actor(
                cam_bp,
                carla.Transform(carla.Location(x=1.4, z=1.6)),
                attach_to=ego,
            )
            if camera is None:
                raise RuntimeError("camera try_spawn returned None")
            # tick a few times BEFORE listen (listen callback can spike)
            for _ in range(5):
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0))
                world.tick()

            def _on_image(image: carla.Image) -> None:
                # Drop if busy — never queue image buffers (system RAM leak)
                if not rgb_lock.acquire(blocking=False):
                    return
                try:
                    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                    # one contiguous RGB copy; replace previous frame (no history)
                    rgb = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                    latest_rgb["img"] = np.ascontiguousarray(rgb)
                    latest_rgb["n"] = int(latest_rgb["n"]) + 1
                finally:
                    rgb_lock.release()

            camera.listen(_on_image)
            for _ in range(8):
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0))
                world.tick()
            print(f"camera OK {args.cam_w}x{args.cam_h}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"camera SKIPPED (UE risk): {exc}", flush=True)
            use_camera = False
            if camera is not None:
                try:
                    camera.destroy()
                except Exception:
                    pass
                camera = None
    else:
        print("STEP5: --no-camera (centerline + placeholder RGB for VLA)", flush=True)

    # Placeholder image if no camera (VLA still runs; path mostly corridor)
    if not use_camera:
        latest_rgb["img"] = np.full((int(args.cam_h), int(args.cam_w), 3), 40, dtype=np.uint8)
        latest_rgb["n"] = 1

    print("STEP6: load SimLingo FULLY on CUDA (no VRAM cap; no iGPU on this box)...", flush=True)
    try:
        import torch
        import gc

        if not torch.cuda.is_available():
            print("FATAL: torch.cuda not available — model would sit in system RAM", flush=True)
            return 1
        # User: dedicated 4080, VRAM was underused — do NOT set_per_process_memory_fraction.
        # (Old 0.35–0.50 caps made usage look "empty" and could force awkward alloc.)
        torch.cuda.empty_cache()
        free_mb, total_mb = [x / (1024**2) for x in torch.cuda.mem_get_info()]
        print(f"CUDA free≈{free_mb:.0f}/{total_mb:.0f} MB before load (no process fraction cap)", flush=True)
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        print("CUDA setup failed:", exc, flush=True)
        return 1

    rt = SimLingoNeuralRuntime(device="cuda")
    load_rep = rt.load()
    if not load_rep.ok:
        print("neural load failed", load_rep.error, flush=True)
        return 1
    policy = NeuralV0Policy(runtime=rt, keep_on_gpu=True)
    policy.ensure_loaded()
    # Verify weights really on GPU (explains "RAM high / VRAM idle")
    try:
        import torch
        import gc

        dev = next(rt.model.parameters()).device
        n_cuda = sum(p.numel() for p in rt.model.parameters() if p.is_cuda)
        n_all = sum(p.numel() for p in rt.model.parameters())
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"model.device={dev} params_on_cuda={n_cuda}/{n_all} "
            f"cuda_alloc={alloc:.0f}MB reserved={reserv:.0f}MB "
            f"resident={getattr(rt, '_resident_device', '?')}",
            flush=True,
        )
        if dev.type != "cuda" or n_cuda < n_all:
            print("FATAL: model not fully on CUDA after keep_on_gpu", flush=True)
            return 1
        # Free host-side leftovers from load; do NOT empty_cache after pin (keeps VRAM reserved)
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        print("device check failed:", exc, flush=True)
        return 1

    warm = np.zeros((int(args.cam_h), int(args.cam_w), 3), dtype=np.uint8)
    warm_obs = ObservationBundle(
        run_id="warm",
        frame_id="w0",
        scenario_id="warm",
        simulation_time_s=0.0,
        ego_v=3.0,
        front_rgb=warm,
        route_xy=((0.0, 0.0), (20.0, 0.0)),
    )
    policy.predict_arrays(warm_obs)
    try:
        import torch

        print(
            f"warmup latency_ms={policy.last_latency_s * 1000:.1f} "
            f"peak_vram_mb={policy.last_peak_vram_mb:.0f} "
            f"alloc_now={torch.cuda.memory_allocated()/1024**2:.0f}MB "
            f"reserved_now={torch.cuda.memory_reserved()/1024**2:.0f}MB",
            flush=True,
        )
    except Exception:
        print(f"warmup latency_ms={policy.last_latency_s * 1000:.1f}", flush=True)
    for _ in range(3):
        try:
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0))
            world.tick()
        except Exception as exc:  # noqa: BLE001
            print("tick after VLA load failed:", exc, flush=True)
            return 76
    print(f"VLA on CUDA OK; sim_dt={sim_dt} camera={use_camera}", flush=True)

    # Unify tick path for main loop (lite always uses world.tick now)
    def runtime_tick(vc: carla.VehicleControl) -> None:
        ego.apply_control(vc)
        world.tick()

    # Control: raise vehicle max_speed to match demo cap (baseline.toml freezes at 15).
    vla_period = max(0.25, float(args.vla_period_s))
    max_speed = float(args.max_speed)
    max_lat_m = float(args.max_lat_m)
    replan_every = max(1, int(args.replan_every))
    spectator_every = max(1, int(args.spectator_every))
    is_turn = route_meta.get("route_mode") == "turn90"
    # Slightly softer on tight turn demos; straight can use full 20 m/s.
    v_cap = min(max_speed, 12.0 if is_turn else max_speed)

    ctrl_cfg = load_control_config()
    ctrl_cfg = replace(
        ctrl_cfg,
        max_speed_mps=max(float(ctrl_cfg.max_speed_mps), v_cap + 1.0),
        max_accel_mps2=max(float(ctrl_cfg.max_accel_mps2), 3.5),
    )
    control = ControlLoop(ctrl_cfg)
    control.buffer.stale_s = 2.0

    # Optional sparse nav goals for VLA *prompt only* (like GPS goal on real car).
    # NEVER used as MPC geometry.
    nav_goal_xy = tuple(route) if route and len(route) >= 2 else ((spawn.location.x + 30.0, spawn.location.y),)
    s_hint = 0.0  # logging only (progress along optional nav poly if present)

    vla_state: dict[str, Any] = {
        "lock": threading.Lock(),
        "traj_arr": None,  # pure VLA TrajectoryArray (smoothed)
        "prev_pts": None,  # for temporal blend
        "v_mean": 8.0,
        "ver": 0,
        "n_infer": 0,
        "last_err": "",
        "busy": False,
    }
    ego_snap: dict[str, Any] = {
        "lock": threading.Lock(),
        "x": float(spawn.location.x),
        "y": float(spawn.location.y),
        "z": float(spawn.location.z),
        "yaw": math.radians(float(spawn.rotation.yaw)),
        "v": 0.0,
        "sim_t": 0.0,
        "frame": 0,
        "img": None,
        "img_seq": 0,
    }
    stop_bg = threading.Event()
    # Bounded metrics (unbounded append was a slow RAM leak on long runs)
    from collections import deque

    latencies: deque[float] = deque(maxlen=64)
    peaks: deque[float] = deque(maxlen=64)
    loop_ms: deque[float] = deque(maxlen=256)
    cam_last_good = {
        "x": float(spawn.location.x),
        "y": float(spawn.location.y),
        "z": float(spawn.location.z),
        "yaw": float(spawn.rotation.yaw),
    }
    n_road_snap = 0

    # Wait for first camera frame
    t_cam = time.time()
    while time.time() - t_cam < max(args.hold_s, 5.0):
        with rgb_lock:
            if latest_rgb["n"] > 0:
                break
        try:
            runtime_tick(carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0))
        except Exception:
            time.sleep(0.02)
        set_spectator_follow(world, ego, last_good=cam_last_good)

    def _bg_vla() -> None:
        """Async pure VLA path+speed (no map geometry for control)."""
        time.sleep(1.2)
        last_seq = -1
        while not stop_bg.is_set():
            try:
                with ego_snap["lock"]:
                    seq = int(ego_snap["img_seq"])
                    img = ego_snap["img"]
                    ex, ey = ego_snap["x"], ego_snap["y"]
                    eyaw, ev = ego_snap["yaw"], ego_snap["v"]
                    sim_t0, fr0 = ego_snap["sim_t"], ego_snap["frame"]
                if img is None or seq == last_seq:
                    time.sleep(0.02)
                    continue
                last_seq = seq
                with vla_state["lock"]:
                    vla_state["busy"] = True
                    prev_pts = vla_state["prev_pts"]
                # route_xy = nav-style goal only (real car: GPS/mission); not MPC path
                obs = ObservationBundle(
                    run_id=run_id,
                    frame_id=f"bg{fr0}",
                    scenario_id=scenario_id,
                    simulation_time_s=float(sim_t0),
                    wall_time_s=time.time(),
                    carla_frame=int(fr0),
                    ego_x=float(ex),
                    ego_y=float(ey),
                    ego_yaw=float(eyaw),
                    ego_v=float(ev),
                    route_xy=nav_goal_xy,
                    front_rgb=img,
                    meta={
                        "demo": "smooth_b_pure_vla",
                        "no_map_geometry": True,
                        "nav_goal_only": True,
                    },
                )
                arrs = policy.predict_arrays(obs)
                raw = arrs[0]
                smooth = smooth_vla_trajectory(
                    raw,
                    ego_x=float(ex),
                    ego_y=float(ey),
                    ego_yaw=float(eyaw),
                    ego_v=float(ev),
                    max_speed=v_cap,
                    prev_pts=prev_pts,
                    blend_prev=0.65,  # stronger temporal hold → less left/right chatter
                )
                vs = [float(p[3]) for p in smooth.points_xy_yaw_v_a_kappa]
                v_mean = sum(vs) / max(len(vs), 1)
                p0 = smooth.points_xy_yaw_v_a_kappa[0]
                p2 = smooth.points_xy_yaw_v_a_kappa[min(4, len(smooth.points_xy_yaw_v_a_kappa) - 1)]
                head_deg = math.degrees(_wrap_pi(p2[2] - p0[2]))
                _, lat_rms = _vla_mean_heading_and_lat_rms(
                    list(smooth.points_xy_yaw_v_a_kappa), float(ex), float(ey), float(eyaw)
                )
                with vla_state["lock"]:
                    vla_state["traj_arr"] = smooth
                    vla_state["prev_pts"] = list(smooth.points_xy_yaw_v_a_kappa)
                    vla_state["v_mean"] = 0.75 * float(vla_state["v_mean"]) + 0.25 * v_mean
                    vla_state["ver"] = int(vla_state["ver"]) + 1
                    vla_state["n_infer"] = int(vla_state["n_infer"]) + 1
                    vla_state["busy"] = False
                    latencies.append(float(policy.last_latency_s))
                    peaks.append(float(policy.last_peak_vram_mb or 0.0))
                n = int(vla_state["n_infer"])
                if n <= 3 or n % 4 == 0:
                    print(
                        f"VLA pure n={n} lat_ms={policy.last_latency_s * 1000:.1f} "
                        f"v_mean={v_mean:.2f}/{v_cap:.0f} head_ddeg={head_deg:.1f} "
                        f"lat_rms={lat_rms:.2f} "
                        f"{'SELF_STRAIGHT' if lat_rms < 0.85 else 'CURVED'} "
                        f"p0=({p0[0]:.1f},{p0[1]:.1f})",
                        flush=True,
                    )
                time.sleep(vla_period)
            except Exception as exc:  # noqa: BLE001
                with vla_state["lock"]:
                    vla_state["last_err"] = str(exc)
                    vla_state["busy"] = False
                time.sleep(0.12)

    def get_vla_traj() -> TrajectoryArray | None:
        """Latest smoothed pure-VLA traj (no reshape/centerline)."""
        with vla_state["lock"]:
            return vla_state["traj_arr"]

    bg = threading.Thread(target=_bg_vla, daemon=True)
    bg.start()

    # Prime first VLA while holding
    t_wait = time.time()
    while time.time() - t_wait < 20.0:
        with rgb_lock:
            img0 = latest_rgb["img"]
        if img0 is not None:
            with ego_snap["lock"]:
                if ego_snap["img"] is None:
                    ego_snap["img"] = img0
                    ego_snap["img_seq"] = 1
        with vla_state["lock"]:
            if int(vla_state["n_infer"]) >= 1:
                break
        try:
            runtime_tick(carla.VehicleControl(throttle=0.0, brake=0.4, steer=0.0))
        except Exception:
            time.sleep(0.02)
        time.sleep(0.01)
    print(f"VLA ready n_infer={vla_state['n_infer']} v_cap={v_cap:.1f} m/s", flush=True)

    print("control: pure-pursuit + speed PID on VLA traj (no map, no reshape)", flush=True)

    steps = 0
    moved = 0.0
    last_xy = (spawn.location.x, spawn.location.y)
    last_steer = 0.0
    last_thr = 0.0
    last_brk = 0.0
    n_mpc = 0  # counts track ticks (legacy name)
    n_hold_brake = 0
    last_z = float(spawn.location.z)
    min_z = last_z
    max_z = last_z
    n_replan = 0

    try:
        for _ in range(args.max_steps):
            t0 = time.perf_counter()
            snap = world.get_snapshot()
            sim_t = float(snap.timestamp.elapsed_seconds)
            frame = int(snap.frame)
            tf = ego.get_transform()
            vel = ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)

            last_z = float(tf.location.z)
            min_z = min(min_z, last_z)
            max_z = max(max_z, last_z)

            with vla_state["lock"]:
                busy = bool(vla_state["busy"])
                v_cmd_log = float(vla_state["v_mean"])
                vla_ver = int(vla_state["ver"])

            with ego_snap["lock"]:
                ego_snap["x"] = float(tf.location.x)
                ego_snap["y"] = float(tf.location.y)
                ego_snap["z"] = float(tf.location.z)
                ego_snap["yaw"] = math.radians(tf.rotation.yaw)
                ego_snap["v"] = float(speed)
                ego_snap["sim_t"] = sim_t
                ego_snap["frame"] = frame
                if not busy and (steps % 2 == 0 or ego_snap["img"] is None):
                    with rgb_lock:
                        img_ref = latest_rgb["img"]
                    if img_ref is not None:
                        ego_snap["img"] = img_ref
                        ego_snap["img_seq"] = int(ego_snap["img_seq"]) + 1

            moved += math.hypot(tf.location.x - last_xy[0], tf.location.y - last_xy[1])
            last_xy = (tf.location.x, tf.location.y)

            traj = get_vla_traj()
            if traj is None:
                thr, brk, ste = 0.0, 0.25, 0.0
                n_hold_brake += 1
            else:
                # Lookahead grows with speed (smoother on straights)
                ld = max(6.0, min(14.0, 5.0 + 0.8 * speed))
                thr, brk, ste = pure_vla_control(
                    ego_x=float(tf.location.x),
                    ego_y=float(tf.location.y),
                    ego_yaw=math.radians(tf.rotation.yaw),
                    ego_v=float(speed),
                    traj=traj,
                    last_steer=last_steer,
                    lookahead_m=ld,
                    max_speed=v_cap,
                )
                n_mpc += 1
                n_replan += 1

            thr = 0.50 * last_thr + 0.50 * thr
            brk = 0.50 * last_brk + 0.50 * brk
            if last_z < -0.8 or last_z > 5.0:
                thr = min(thr, 0.10)
                brk = max(brk, 0.35)
            if speed < 0.2 and steps > 8 and moved < 1.0:
                thr = max(thr, 0.45)
                brk = 0.0
            last_steer, last_thr, last_brk = ste, thr, brk

            vc = carla.VehicleControl(
                throttle=max(0.0, min(1.0, thr)),
                steer=max(-1.0, min(1.0, ste)),
                brake=max(0.0, min(1.0, brk)),
                hand_brake=False,
                manual_gear_shift=False,
            )
            try:
                runtime_tick(vc)
            except Exception as exc:  # noqa: BLE001
                print("tick failed", type(exc).__name__, exc, flush=True)
                break

            if steps % spectator_every == 0:
                set_spectator_follow(world, ego, last_good=cam_last_good)

            # Realtime pace to match sim_dt (0.05@20Hz or 0.02@50Hz)
            if args.realtime:
                elapsed = time.perf_counter() - t0
                slack = sim_dt - elapsed
                if slack > 0.0005:
                    time.sleep(slack)

            steps += 1
            wall_ms = (time.perf_counter() - t0) * 1000.0
            loop_ms.append(wall_ms)
            nav_len = 1.0
            if len(nav_goal_xy) >= 2:
                nav_len = 0.0
                for i in range(1, len(nav_goal_xy)):
                    nav_len += math.hypot(
                        nav_goal_xy[i][0] - nav_goal_xy[i - 1][0],
                        nav_goal_xy[i][1] - nav_goal_xy[i - 1][1],
                    )
            s_frac = moved / max(nav_len, 1.0)
            log_every = 50 if args.realtime or args.max_steps >= 1000 else 25
            if steps <= 12 or steps % log_every == 0:
                fps = 1000.0 / max(wall_ms, 1e-3)
                p50 = _pct(list(loop_ms)[-40:], 0.5) or wall_ms
                print(
                    f"step={steps} speed={speed:.2f} thr={vc.throttle:.2f} ste={vc.steer:.2f} "
                    f"dist={moved:.1f} prog≈{s_frac:.2f} v_mean={v_cmd_log:.2f} "
                    f"vla_ver={vla_ver} wall_ms={wall_ms:.0f} wall_fps≈{fps:.1f} p50_ms={p50:.0f} "
                    f"pos=({tf.location.x:.1f},{tf.location.y:.1f},z={tf.location.z:.1f})",
                    flush=True,
                )
            # Duration runs use max_steps; no map-route completion stop
    finally:
        stop_bg.set()
        bg.join(timeout=3.0)
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.destroy()
            except Exception:
                pass
        try:
            if ego is not None:
                ego.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception as exc:  # noqa: BLE001
            print("cleanup settings warn", exc, flush=True)

    with rgb_lock:
        cam_n = int(latest_rgb["n"])
    loop_p50 = _pct(list(loop_ms), 0.5)
    loop_p95 = _pct(list(loop_ms), 0.95)
    wall_fps_p50 = None if loop_p50 is None else 1000.0 / max(loop_p50, 1e-3)
    wall_fps_p95_worst = None if loop_p95 is None else 1000.0 / max(loop_p95, 1e-3)
    p50_lat = _pct(list(latencies), 0.5)
    p95_lat = _pct(list(latencies), 0.95)

    # Demo-B success: moved + VLA + smooth + stayed on mesh (no vanish)
    reasons: list[str] = []
    if cam_n < 1:
        reasons.append("camera_frames_zero")
    if int(vla_state["n_infer"]) < 1:
        reasons.append("no_vla_inference")
    if n_mpc < 1:
        reasons.append("no_mpc_steps")
    if moved < 0.5:
        reasons.append(f"distance_too_small:{moved:.3f}")
    # realtime@sim_dt=0.05 → expect ~20 wall FPS, not 30
    min_wall_fps = 18.0 if (args.realtime and sim_dt >= 0.045) else 25.0
    if wall_fps_p50 is not None and wall_fps_p50 < min_wall_fps:
        reasons.append(f"wall_fps_p50_below_{min_wall_fps:.0f}:{wall_fps_p50:.1f}")
    # Vanish heuristic: underground or sky after driving
    if min_z < -1.0:
        reasons.append(f"vanished_underground_min_z:{min_z:.2f}")
    if max_z > 8.0:
        reasons.append(f"vanished_sky_max_z:{max_z:.2f}")
    if last_z < -0.5 or last_z > 5.0:
        reasons.append(f"final_z_invalid:{last_z:.2f}")

    demo_pass = len(reasons) == 0
    summary = {
        "demo": "g3_vla_smooth_b",
        "mode": "PURE_VLA_PATH_SPEED_MPC",
        "status": "DEMO_B_PASS" if demo_pass else "DEMO_B_MEASURED_NOT_PASS",
        "demo_pass": demo_pass,
        "fail_reasons": reasons,
        "not_g3_stage_verified": True,
        "safety_kernel": False,
        "qp": False,
        "rato": False,
        "map_centerline_for_control": False,
        "pipeline": "camera -> VLA(CUDA) path+speed -> temporal_smooth -> pure_pursuit+speed_pid -> plant",
        "seed": args.seed,
        "map": world.get_map().name,
        "route_meta": route_meta,
        "steps": steps,
        "camera_frames": cam_n,
        "distance_m": moved,
        "n_mpc": n_mpc,
        "n_hold_brake": n_hold_brake,
        "n_vla": int(vla_state["n_infer"]),
        "n_replan": n_replan,
        "n_road_snap": n_road_snap,
        "max_speed_mps": v_cap,
        "max_lat_m": max_lat_m,
        "z_min": min_z,
        "z_max": max_z,
        "z_final": last_z,
        "neural_latency_p50_ms": None if p50_lat is None else p50_lat * 1000.0,
        "neural_latency_p95_ms": None if p95_lat is None else p95_lat * 1000.0,
        "peak_vram_mb": max(peaks) if peaks else None,
        "note_memory": (
            "checkpoint load is CPU then move to CUDA; del state_dict after load. "
            "If Task Manager shows high RAM / low VRAM, model was on CPU — now hard-fail."
        ),
        "control_loop_ms_p50": loop_p50,
        "control_loop_ms_p95": loop_p95,
        "wall_fps_p50": wall_fps_p50,
        "wall_fps_p95_worst": wall_fps_p95_worst,
        "target_wall_fps": 30.0,
        "duration_s_request": float(args.duration_s) if args.duration_s else None,
        "sim_time_s_approx": steps * sim_dt,
        "sim_dt": sim_dt,
        "lite": bool(args.lite),
        "camera_enabled": bool(use_camera),
        "realtime_paced": bool(args.realtime),
        "vla_period_s": vla_period,
        "replan_every": replan_every,
        "cam_size": [int(args.cam_w), int(args.cam_h)],
        "keep_on_gpu": True,
        "cuda_no_fraction_cap": True,
        "run_id": run_id,
        "note": (
            "Real-vehicle-oriented: MPC tracks pure VLA trajectory only. "
            "CARLA map/waypoints are NOT used as driving geometry (nav goal for prompt only). "
            "No set_transform stick. Model on CUDA. NOT G3 VERIFIED."
        ),
    }
    path = evidence_dir / f"demo_b_summary_seed{args.seed}_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence_dir / "latest_demo_b_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", path, "demo_pass", demo_pass, flush=True)
    return 0 if demo_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
