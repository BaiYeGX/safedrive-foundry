#!/usr/bin/env python3
"""Authoring helper: snap R2 registry fixtures to live Town03 driving waypoints.

Run only before outcome observation. Does not compute candidate outcomes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import carla

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "safedrive_foundry" / "config" / "g4a" / "scenario_registry_v1.toml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--blind-v2",
        action="store_true",
        help="author the unobserved Spatial R2-X blind registry on new Town03 corridors",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = (
        Path(args.out)
        if args.out
        else (
            ROOT / "safedrive_foundry/config/g4a/scenario_registry_v2_blind.toml"
            if args.blind_v2
            else OUT
        )
    )
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    m = world.get_map()

    def wp_at(x: float, y: float, z: float = 0.5):
        return m.get_waypoint(
            carla.Location(x, y, z),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def along(wp, dist: float):
        seq = wp.next(dist)
        return seq[0] if seq else None

    def pose(wp, z_off: float = 0.5) -> dict:
        t = wp.transform
        return {
            "x": float(t.location.x),
            "y": float(t.location.y),
            "z": float(t.location.z + z_off),
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": float(t.rotation.yaw),
        }

    def vel_forward(wp, speed: float) -> dict:
        yaw = math.radians(wp.transform.rotation.yaw)
        return {
            "vx": speed * math.cos(yaw),
            "vy": speed * math.sin(yaw),
            "vz": 0.0,
        }

    def route_wps(wp, length: float = 100.0, step: float = 25.0) -> list[dict]:
        pts = [pose(wp)]
        cur = wp
        d = step
        while d <= length:
            n = along(cur, step)
            if n is None:
                break
            pts.append(pose(n))
            cur = n
            d += step
        return pts

    def side_lane(wp):
        right = wp.get_right_lane()
        left = wp.get_left_lane()
        if right is not None and right.lane_type == carla.LaneType.Driving:
            return right
        if left is not None and left.lane_type == carla.LaneType.Driving:
            return left
        return None

    # Blind v2 uses disjoint Town03 corridors selected from map topology only.
    # No VLA/candidate/oracle is loaded by this authoring script.
    lead_anchor = (128.56, -190.40) if args.blind_v2 else (17.1, 193.5)
    cut_anchor = (-74.39, 99.72) if args.blind_v2 else (241.09, 42.23)
    cross_anchor = (81.0, -145.0) if args.blind_v2 else (-88.39, -10.4)

    # Lead braking
    ego_lb = wp_at(*lead_anchor)
    if ego_lb is None:
        raise SystemExit("ego_lb waypoint missing")
    ego_lb2 = along(ego_lb, 40.0) or ego_lb
    lead_mod_a = along(ego_lb, 18.0)
    lead_mod_b = along(ego_lb, 15.0)
    lead_hard_a = along(ego_lb2, 12.0)
    lead_hard_b = along(ego_lb2, 10.0)

    # Cut-in multi-lane corridor
    ego_ci = wp_at(*cut_anchor)
    if ego_ci is None:
        raise SystemExit("ego_ci waypoint missing")
    side = side_lane(ego_ci)
    if side is None:
        raise SystemExit("no adjacent lane for cut-in")
    cutter_early_a = along(side, 12.0) or side
    cutter_early_b = along(side, 10.0) or side
    ego_ci2 = along(ego_ci, 30.0) or ego_ci
    side2 = side_lane(ego_ci2) or side
    cutter_late_a = along(side2, 20.0) or side2
    cutter_late_b = along(side2, 24.0) or side2

    # Crossing
    ego_cr = wp_at(*cross_anchor)
    if ego_cr is None:
        raise SystemExit("ego_cr waypoint missing")
    ego_cr = along(ego_cr, 10.0) or ego_cr
    ego_cr2 = along(ego_cr, 15.0) or ego_cr
    ahead = along(ego_cr, 25.0) or ego_cr
    yaw = math.radians(ego_cr.transform.rotation.yaw)
    lx = ahead.transform.location.x + 20 * math.cos(yaw + math.pi / 2)
    ly = ahead.transform.location.y + 20 * math.sin(yaw + math.pi / 2)
    cross_wp = wp_at(lx, ly)
    if cross_wp is None:
        raise SystemExit("cross waypoint missing")
    prev8 = cross_wp.previous(8.0)
    prev3 = cross_wp.previous(3.0)
    prev1 = cross_wp.previous(1.0)
    cross_clear_a = cross_wp
    cross_clear_b = prev8[0] if prev8 else cross_wp
    cross_tight_a = prev3[0] if prev3 else cross_wp
    cross_tight_b = prev1[0] if prev1 else cross_wp

    fixtures = {
        "lead_brake_moderate": {
            "family": "lead_braking",
            "notes": "Lead moderate brake on Town03 snapped road",
            "route": route_wps(ego_lb),
            "npc_bp": "vehicle.audi.a2",
            "npc_name": "lead",
            "seeds": {
                "seed_a": {
                    "ego": ego_lb,
                    "npc": lead_mod_a,
                    "ego_v": 8.0,
                    "npc_v": 7.0,
                    "knots": [(0.0, 0.35, 0.0, 0.0), (0.80, 0.0, 0.35, 0.0), (2.50, 0.0, 0.20, 0.0)],
                },
                "seed_b": {
                    "ego": ego_lb,
                    "npc": lead_mod_b,
                    "ego_v": 8.0,
                    "npc_v": 7.0,
                    "knots": [(0.0, 0.35, 0.0, 0.0), (0.60, 0.0, 0.40, 0.0), (2.50, 0.0, 0.25, 0.0)],
                },
            },
        },
        "lead_brake_hard": {
            "family": "lead_braking",
            "notes": "Lead hard brake shorter headway",
            "route": route_wps(ego_lb2),
            "npc_bp": "vehicle.lincoln.mkz_2017",
            "npc_name": "lead",
            "seeds": {
                "seed_a": {
                    "ego": ego_lb2,
                    "npc": lead_hard_a,
                    "ego_v": 8.0,
                    "npc_v": 7.5,
                    "knots": [(0.0, 0.30, 0.0, 0.0), (0.40, 0.0, 0.70, 0.0), (2.50, 0.0, 0.50, 0.0)],
                },
                "seed_b": {
                    "ego": ego_lb2,
                    "npc": lead_hard_b,
                    "ego_v": 8.0,
                    "npc_v": 7.5,
                    "knots": [(0.0, 0.30, 0.0, 0.0), (0.30, 0.0, 0.80, 0.0), (2.50, 0.0, 0.55, 0.0)],
                },
            },
        },
        "cut_in_early": {
            "family": "cut_in",
            "notes": "Adjacent lane merge early",
            "route": route_wps(ego_ci),
            "npc_bp": "vehicle.toyota.prius",
            "npc_name": "cutter",
            "seeds": {
                "seed_a": {
                    "ego": ego_ci,
                    "npc": cutter_early_a,
                    "ego_v": 8.0,
                    "npc_v": 7.5,
                    "knots": [(0.0, 0.40, 0.0, -0.15), (0.70, 0.35, 0.0, 0.0), (2.50, 0.30, 0.0, 0.0)],
                },
                "seed_b": {
                    "ego": ego_ci,
                    "npc": cutter_early_b,
                    "ego_v": 8.0,
                    "npc_v": 7.5,
                    "knots": [(0.0, 0.42, 0.0, -0.18), (0.55, 0.35, 0.0, 0.0), (2.50, 0.30, 0.0, 0.0)],
                },
            },
        },
        "cut_in_late": {
            "family": "cut_in",
            "notes": "Later merge narrower window",
            "route": route_wps(ego_ci2),
            "npc_bp": "vehicle.mini.cooper_s",
            "npc_name": "cutter",
            "seeds": {
                "seed_a": {
                    "ego": ego_ci2,
                    "npc": cutter_late_a,
                    "ego_v": 8.0,
                    "npc_v": 8.0,
                    "knots": [
                        (0.0, 0.40, 0.0, 0.0),
                        (1.20, 0.35, 0.0, -0.20),
                        (1.80, 0.35, 0.0, 0.0),
                        (2.50, 0.30, 0.0, 0.0),
                    ],
                },
                "seed_b": {
                    "ego": ego_ci2,
                    "npc": cutter_late_b,
                    "ego_v": 8.0,
                    "npc_v": 8.0,
                    "knots": [
                        (0.0, 0.40, 0.0, 0.0),
                        (1.40, 0.35, 0.0, -0.22),
                        (2.00, 0.35, 0.0, 0.0),
                        (2.50, 0.30, 0.0, 0.0),
                    ],
                },
            },
        },
        "cross_vehicle_clear": {
            "family": "crossing",
            "notes": "Crossing with larger margin",
            "route": route_wps(ego_cr),
            "npc_bp": "vehicle.nissan.patrol",
            "npc_name": "crosser",
            "seeds": {
                "seed_a": {
                    "ego": ego_cr,
                    "npc": cross_clear_a,
                    "ego_v": 7.0,
                    "npc_v": 6.0,
                    "knots": [(0.0, 0.35, 0.0, 0.0), (2.50, 0.35, 0.0, 0.0)],
                },
                "seed_b": {
                    "ego": ego_cr,
                    "npc": cross_clear_b,
                    "ego_v": 7.0,
                    "npc_v": 6.0,
                    "knots": [(0.0, 0.38, 0.0, 0.0), (2.50, 0.38, 0.0, 0.0)],
                },
            },
        },
        "cross_vehicle_tight": {
            "family": "crossing",
            "notes": "Tight crossing timing",
            "route": route_wps(ego_cr2),
            "npc_bp": "vehicle.chevrolet.impala",
            "npc_name": "crosser",
            "seeds": {
                "seed_a": {
                    "ego": ego_cr2,
                    "npc": cross_tight_a,
                    "ego_v": 7.0,
                    "npc_v": 8.0,
                    "knots": [(0.0, 0.45, 0.0, 0.0), (2.50, 0.45, 0.0, 0.0)],
                },
                "seed_b": {
                    "ego": ego_cr2,
                    "npc": cross_tight_b,
                    "ego_v": 7.0,
                    "npc_v": 8.5,
                    "knots": [(0.0, 0.48, 0.0, 0.0), (2.50, 0.48, 0.0, 0.0)],
                },
            },
        },
    }

    registry_version = "v2-blind-20260727" if args.blind_v2 else "v1"
    if args.blind_v2:
        rename = {
            "lead_brake_moderate": "blind_south_lead_gap",
            "lead_brake_hard": "blind_south_lead_stop",
            "cut_in_early": "blind_west_cut_in_near",
            "cut_in_late": "blind_west_cut_in_delayed",
            "cross_vehicle_clear": "blind_south_cross_clear",
            "cross_vehicle_tight": "blind_south_cross_tight",
        }
        fixtures = {rename[key]: value for key, value in fixtures.items()}

    for sid, fx in fixtures.items():
        for seed, body in fx["seeds"].items():
            if body["npc"] is None or body["ego"] is None:
                raise SystemExit(f"missing actor for {sid}/{seed}")
            pe, po = pose(body["ego"]), pose(body["npc"])
            d = math.hypot(pe["x"] - po["x"], pe["y"] - po["y"])
            print(f"{sid}/{seed} dist={d:.2f} ego=({pe['x']:.2f},{pe['y']:.2f}) npc=({po['x']:.2f},{po['y']:.2f})")

    def fmt_pose(p: dict) -> str:
        return (
            f"x = {p['x']:.6f}\n"
            f"y = {p['y']:.6f}\n"
            f"z = {p['z']:.6f}\n"
            f"roll_deg = {p['roll_deg']:.6f}\n"
            f"pitch_deg = {p['pitch_deg']:.6f}\n"
            f"yaw_deg = {p['yaw_deg']:.6f}"
        )

    def fmt_vel(v: dict) -> str:
        return f"vx = {v['vx']:.6f}\nvy = {v['vy']:.6f}\nvz = {v['vz']:.6f}"

    lines: list[str] = []
    lines.append(
        f"""# R2-X Spatial K2 Blind Scenario Registry
# Snapped to Town03 driving waypoints BEFORE outcome observation.
# Exact spawn only; simulation-time actor scripts; no free/fallback spawn.

[registry]
schema_version = "safedrive.g4a.scenario_registry.v1"
registry_version = "{registry_version}"
description = "R2-X blind pilot: disjoint Town03 corridors; no candidate outcome at authoring"

[defaults]
map_name = "Town03"
sim_dt_s = 0.05
duration_s = 5.0
vla_config_ref = "config/vla/k2_v2_spatial.toml"
mpc_config_ref = "config/control/mpc_pid_baseline.toml"
executor_config_ref = "g3_stable_vla_mpc"
expected_decision_anchor_time_s = 0.0

[defaults.weather]
preset = "ClearNoon"
cloudiness = 10.0
precipitation = 0.0
precipitation_deposits = 0.0
wind_intensity = 5.0
sun_azimuth_angle = 0.0
sun_altitude_angle = 70.0
wetness = 0.0
fog_density = 0.0
fog_distance = 0.0

[defaults.traffic_light]
policy = "freeze_green"
initial_state = "Green"

[defaults.sensor_contract.front_rgb]
width = 1024
height = 512
fov = 110.0
attach = "ego"
layout = "HWC_RGB_uint8"
"""
    )

    order = list(fixtures)
    for sid in order:
        fx = fixtures[sid]
        lines.append(f"\n[scenarios.{sid}]")
        lines.append(f'family = "{fx["family"]}"')
        lines.append('map_name = "Town03"')
        lines.append(f'notes = "{fx["notes"]}"')
        lines.append(f"\n[scenarios.{sid}.route]")
        lines.append(f'identity = "town03_{sid}"')
        lines.append("target_speed_mps = 8.0")
        wp_str = ",\n  ".join(
            f'[{p["x"]:.6f}, {p["y"]:.6f}, {p["z"]:.6f}]' for p in fx["route"]
        )
        lines.append(f"waypoints = [\n  {wp_str}\n]")
        for seed in ("seed_a", "seed_b"):
            body = fx["seeds"][seed]
            pe = pose(body["ego"])
            po = pose(body["npc"])
            ve = vel_forward(body["ego"], body["ego_v"])
            vo = vel_forward(body["npc"], body["npc_v"])
            base = f"scenarios.{sid}.seeds.{seed}"
            lines.append(f"\n[{base}.ego]")
            lines.append('name = "ego"')
            lines.append('role = "ego"')
            lines.append('blueprint = "vehicle.tesla.model3"')
            lines.append("spawn_order = 0")
            lines.append("bounding_box_extent_m = [2.3, 1.0, 0.75]")
            lines.append(f"\n[{base}.ego.transform]")
            lines.append(fmt_pose(pe))
            lines.append(f"\n[{base}.ego.initial_velocity]")
            lines.append(fmt_vel(ve))
            lines.append(f"\n[{base}.ego.script]")
            lines.append('script_type = "hold"')
            lines.append(f"\n[[{base}.actors]]")
            lines.append(f'name = "{fx["npc_name"]}"')
            lines.append('role = "npc"')
            lines.append(f'blueprint = "{fx["npc_bp"]}"')
            lines.append("spawn_order = 1")
            lines.append("bounding_box_extent_m = [2.2, 1.0, 0.75]")
            lines.append(f"\n[{base}.actors.transform]")
            lines.append(fmt_pose(po))
            lines.append(f"\n[{base}.actors.initial_velocity]")
            lines.append(fmt_vel(vo))
            lines.append(f"\n[{base}.actors.script]")
            lines.append('script_type = "piecewise_vehicle_control"')
            for t, th, br, st in body["knots"]:
                lines.append(f"\n[[{base}.actors.script.knots]]")
                lines.append(f"t_s = {t:.2f}")
                lines.append(f"throttle = {th:.2f}")
                lines.append(f"brake = {br:.2f}")
                lines.append(f"steer = {st:.2f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path} bytes={out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
