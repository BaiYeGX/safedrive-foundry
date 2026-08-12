#!/usr/bin/env python3
"""Author one map's frozen R2 V3 calibration lineage registries.

The command reads only CARLA topology, traffic controls, the frozen campaign
manifest and route fixtures.  It does not execute the VLA, MPC or Oracle and
therefore cannot select fixtures from outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.evaluation.route_authoring_v3 import (  # noqa: E402
    navigation_waypoints_xyz,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    canonical_sha256,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402
from scripts.r2_v3_author_long_smoke import (  # noqa: E402
    _actor_script_lines,
    _adjacent_actor,
    _build_cases,
    _crossing_actor,
    _pose_lines,
    _quoted,
    _route_actor,
    _traffic_light_turn,
    _velocity_lines,
)


CONDITION_WEATHER = {
    "mild": ("ClearNoon", 10.0, 0.0, 0.0, 70.0),
    "medium": ("WetNoon", 45.0, 0.0, 45.0, 50.0),
    "hard": ("HardRainSunset", 90.0, 70.0, 85.0, 15.0),
}


def _matches(actual: str, requested: str) -> bool:
    return actual.endswith(requested) or f"/{requested}" in actual


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _weather_lines(prefix: str, condition: str) -> list[str]:
    preset, cloudiness, precipitation, wetness, altitude = (
        CONDITION_WEATHER[condition]
    )
    return [
        f"[{prefix}.weather]",
        f"preset = {_quoted(preset)}",
        f"cloudiness = {cloudiness:.3f}",
        f"precipitation = {precipitation:.3f}",
        f"precipitation_deposits = {wetness:.3f}",
        "wind_intensity = 8.000",
        "sun_azimuth_angle = 0.000",
        f"sun_altitude_angle = {altitude:.3f}",
        f"wetness = {wetness:.3f}",
        "fog_density = 0.000",
        "fog_distance = 0.000",
    ]


def _offset_pose(
    pose: Mapping[str, Any],
    *,
    forward_m: float,
) -> dict[str, float]:
    yaw = math.radians(float(pose["yaw_deg"]))
    return {
        **{key: float(value) for key, value in pose.items()},
        "x": float(pose["x"]) + math.cos(yaw) * float(forward_m),
        "y": float(pose["y"]) + math.sin(yaw) * float(forward_m),
    }


def _case_for_slot(
    *,
    world: Any,
    route_manifest: Mapping[str, Any],
    prototype_by_id: Mapping[str, Mapping[str, Any]],
    template_id: str,
    maneuver: RouteManeuver,
    route_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    maneuver = RouteManeuver(
        str(getattr(maneuver, "value", maneuver))
    )
    routes = dict(route_manifest.get("routes") or {})
    if route_override is not None:
        route = dict(route_override)
        prototype: dict[str, Any] = {
            "family": {
                "straight_curve": "clear",
                "junction_straight": "clear",
                "turn_left_right": "clear",
                "route_change_left_right": "clear",
                "follow_stop": "lead_braking",
                "cut_in_left": "cut_in",
                "cut_in_right": "cut_in",
                "merge_yield": "merge",
                "crossing_turn_yield": "crossing",
                "overtake_left": "obstruction",
                "overtake_right": "obstruction",
                "traffic_control": "traffic_control",
            }[template_id],
            "route": route,
        }
        if template_id == "follow_stop":
            prototype["actor"] = {
                "pose": _route_actor(route, index=7),
                "speed_mps": 1.8,
                "script_kind": "follow",
            }
        elif template_id in {"cut_in_left", "cut_in_right"}:
            side = "LEFT" if template_id.endswith("left") else "RIGHT"
            prototype["actor"] = {
                "pose": _adjacent_actor(route, side=side),
                "speed_mps": 2.5,
                "script_kind": f"cut_{side.lower()}",
            }
        elif template_id == "crossing_turn_yield":
            prototype["actor"] = {
                "pose": _crossing_actor(world.get_map(), route),
                "speed_mps": 3.5,
                "script_kind": "crossing",
            }
        elif template_id == "merge_yield":
            context = dict(route["route_context"])
            if maneuver is RouteManeuver.FOLLOW_STRAIGHT:
                side = "RIGHT"
            else:
                side = next(
                    (
                        candidate
                        for candidate in ("LEFT", "RIGHT")
                        if bool(
                            dict(
                                context[f"{candidate.lower()}_lane"]
                            ).get("authorized")
                        )
                    ),
                    "",
                )
            if not side:
                raise RuntimeError("merge route has no authorized adjacent lane")
            prototype["actor"] = {
                "pose": _adjacent_actor(route, side=side),
                "speed_mps": 2.5,
                "script_kind": f"merge_{side.lower()}",
            }
        elif template_id in {"overtake_left", "overtake_right"}:
            prototype["actor"] = {
                "pose": _route_actor(route, index=7),
                "speed_mps": 0.0,
                "script_kind": "obstruction",
            }
        elif template_id == "traffic_control":
            traffic = route.get("traffic_light")
            if not isinstance(traffic, Mapping):
                raise RuntimeError("traffic-control route lacks frozen light")
            prototype["traffic_light"] = {
                **dict(traffic),
                "green_after_s": 3.0,
            }
        actual = RouteManeuver(
            str(prototype["route"]["route_context"]["maneuver"])
        )
        if actual is not maneuver:
            raise RuntimeError(
                f"{template_id}: route-bank maneuver {actual.value}, "
                f"expected {maneuver.value}"
            )
        return prototype

    direct_ids = {
        "junction_straight": "junction_straight",
        "follow_stop": "follow_stop_resume",
        "cut_in_left": "cut_in_left",
        "cut_in_right": "cut_in_right",
        "overtake_left": "overtake_left_rejoin",
        "overtake_right": "overtake_right_rejoin",
    }
    if template_id in direct_ids:
        prototype = dict(prototype_by_id[direct_ids[template_id]])
    elif template_id == "straight_curve":
        scenario_id = {
            RouteManeuver.FOLLOW_STRAIGHT: "straight_clear",
            RouteManeuver.FOLLOW_CURVE_LEFT: "curve_left_clear",
            RouteManeuver.FOLLOW_CURVE_RIGHT: "curve_right_clear",
        }[maneuver]
        prototype = dict(prototype_by_id[scenario_id])
    elif template_id == "turn_left_right":
        prototype = dict(
            prototype_by_id[
                "turn_left"
                if maneuver is RouteManeuver.TURN_LEFT
                else "turn_right"
            ]
        )
    elif template_id == "route_change_left_right":
        prototype = dict(
            prototype_by_id[
                "route_change_left"
                if maneuver is RouteManeuver.ROUTE_CHANGE_LEFT
                else "route_change_right"
            ]
        )
    elif template_id == "crossing_turn_yield":
        route = dict(routes[maneuver.value])
        prototype = {
            "family": "crossing",
            "route": route,
            "actor": {
                "pose": _crossing_actor(world.get_map(), route),
                "speed_mps": 3.5,
                "script_kind": "crossing",
            },
        }
    elif template_id == "merge_yield":
        if maneuver is RouteManeuver.TURN_RIGHT:
            prototype = dict(prototype_by_id["right_turn_merge_yield"])
        elif maneuver is RouteManeuver.FOLLOW_STRAIGHT:
            # Use a route whose right adjacent lane is frozen as legal and
            # same-direction; the merge remains a temporal choice while both
            # candidates keep the requested straight maneuver.
            prototype = dict(prototype_by_id["cut_in_right"])
            route = dict(prototype["route"])
            prototype.update(
                {
                    "family": "merge",
                    "actor": {
                        "pose": _adjacent_actor(route, side="RIGHT"),
                        "speed_mps": 2.5,
                        "script_kind": "merge_right",
                    },
                }
            )
        else:
            raise ValueError(
                "merge_yield only supports FOLLOW_STRAIGHT or TURN_RIGHT"
            )
    elif template_id == "traffic_control":
        route, light = _traffic_light_turn(
            world,
            horizon_m=float(route_manifest["horizon_m"]),
            maneuver=maneuver,
        )
        prototype = {
            "family": "traffic_control",
            "route": route,
            "traffic_light": {**light, "green_after_s": 3.0},
        }
    else:
        raise ValueError(f"unsupported R2 V3 template {template_id}")
    actual = RouteManeuver(
        str(prototype["route"]["route_context"]["maneuver"])
    )
    if actual is not maneuver:
        raise RuntimeError(
            f"{template_id}: authored maneuver {actual.value}, "
            f"expected {maneuver.value}"
        )
    return prototype


def _render_registry(
    *,
    map_name: str,
    lineage_id: str,
    slots: list[Mapping[str, Any]],
    prototype: Mapping[str, Any],
) -> str:
    route = dict(prototype["route"])
    context = dict(route["route_context"])
    family = str(slots[0]["family"])
    template_id = str(slots[0]["template_id"])
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for slot in slots:
        by_condition[str(slot["condition"])].append(slot)
    if set(by_condition) != set(CONDITION_WEATHER):
        raise RuntimeError(f"{lineage_id}: condition matrix mismatch")
    lines = [
        "# Geometry-only R2 V3 campaign lineage; outcomes were not read.",
        "[registry]",
        'schema_version = "safedrive.g4a.scenario_registry.v1"',
        f"registry_version = {_quoted('r2v3-campaign-' + lineage_id)}",
        f"description = {_quoted('R2 V3 calibration ' + lineage_id)}",
        "",
        "[defaults]",
        f"map_name = {_quoted(map_name)}",
        "sim_dt_s = 0.05",
        "duration_s = 5.0",
        'vla_config_ref = "config/vla/k2_v3_semantic.toml"',
        'mpc_config_ref = "config/control/mpc_pid_baseline.toml"',
        'executor_config_ref = "g3_stable_vla_mpc_v3"',
        "expected_decision_anchor_time_s = 0.0",
        "",
        "[defaults.traffic_light]",
        'policy = "freeze_green"',
        'initial_state = "Green"',
        "",
        "[defaults.sensor_contract.front_rgb]",
        "width = 1024",
        "height = 512",
        "fov = 110.0",
        'attach = "ego"',
        'layout = "HWC_RGB_uint8"',
        "",
    ]
    for condition_index, condition in enumerate(
        ("mild", "medium", "hard")
    ):
        scenario_id = f"{template_id}_{condition}"
        prefix = f"scenarios.{scenario_id}"
        lines.extend(
            [
                f"[{prefix}]",
                f"family = {_quoted(family)}",
                f"map_name = {_quoted(map_name)}",
                (
                    "notes = "
                    + _quoted(
                        f"lineage={lineage_id};template={template_id};"
                        f"condition={condition};frozen_before_outcomes"
                    )
                ),
                *_weather_lines(prefix, condition),
                "",
                f"[{prefix}.route]",
                f"identity = {_quoted(str(slots[0]['route_fixture_id']))}",
                "target_speed_mps = 4.0",
                "waypoints = [",
            ]
        )
        for x, y, z in navigation_waypoints_xyz(route):
            lines.append(
                f"  [{float(x):.9f}, {float(y):.9f}, "
                f"{float(z) + 0.5:.9f}],"
            )
        lines.extend(
            [
                "]",
                "",
                f"[{prefix}.route.navigation_context]",
                f"maneuver = {_quoted(str(context['maneuver']))}",
                f"entry_signature = {_quoted(str(context['entry_signature']))}",
                f"exit_signature = {_quoted(str(context['exit_signature']))}",
                f"route_hash = {_quoted(str(context['route_hash']))}",
                f"topology_hash = {_quoted(str(context['topology_hash']))}",
                (
                    "frozen_context_json = "
                    + _quoted(
                        json.dumps(
                            context,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                ),
                "",
            ]
        )
        traffic = prototype.get("traffic_light")
        if traffic:
            traffic = dict(traffic)
            lines.extend(
                [
                    f"[{prefix}.traffic_light]",
                    'policy = "scripted_red_green"',
                    'initial_state = "Red"',
                    f"green_after_s = {2.5 + 0.5 * condition_index:.3f}",
                    f"target_x = {float(traffic['x']):.9f}",
                    f"target_y = {float(traffic['y']):.9f}",
                    f"target_z = {float(traffic['z']):.9f}",
                    "",
                ]
            )
        for seed_index, seed_id in enumerate(("seed_a", "seed_b")):
            ego_prefix = f"{prefix}.seeds.{seed_id}.ego"
            ego_pose = dict(route["ego_spawn_transform"])
            lines.extend(
                [
                    f"[{ego_prefix}]",
                    'name = "ego"',
                    'role = "ego"',
                    'blueprint = "vehicle.mercedes.coupe_2020"',
                    "spawn_order = 0",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(ego_prefix, ego_pose),
                    *_velocity_lines(
                        ego_prefix,
                        float(ego_pose["yaw_deg"]),
                        2.0 + 0.1 * seed_index,
                    ),
                    f"[{ego_prefix}.script]",
                    'script_type = "hold"',
                    "",
                ]
            )
            actor = prototype.get("actor")
            if not actor:
                continue
            actor = dict(actor)
            actor_prefix = f"{prefix}.seeds.{seed_id}.actors"
            script_kind = str(actor["script_kind"])
            direction = (
                1.0
                if script_kind
                in {"crossing", "merge_left", "merge_right"}
                else -1.0
            )
            pose = _offset_pose(
                dict(actor["pose"]),
                forward_m=direction
                * (0.45 * condition_index + 0.35 * seed_index),
            )
            base_actor_speed = float(actor["speed_mps"])
            speed_delta = {
                "follow": -0.20,
                "cut_left": 0.20,
                "cut_right": 0.20,
                "crossing": 0.35,
                "merge_left": 0.25,
                "merge_right": 0.25,
                "obstruction": 0.0,
            }[script_kind]
            actor_speed = max(
                0.0,
                base_actor_speed
                + speed_delta * condition_index
                + 0.10 * seed_index,
            )
            lines.extend(
                [
                    f"[[{actor_prefix}]]",
                    'name = "conflict_actor"',
                    'role = "npc"',
                    'blueprint = "vehicle.lincoln.mkz_2017"',
                    "spawn_order = 1",
                    "bounding_box_extent_m = [2.3, 1.0, 0.75]",
                    *_pose_lines(actor_prefix, pose),
                    *_velocity_lines(
                        actor_prefix,
                        float(pose["yaw_deg"]),
                        actor_speed,
                    ),
                    *_actor_script_lines(actor_prefix, script_kind),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--route-fixture", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    route_manifest = json.loads(
        Path(args.route_fixture).read_text(encoding="utf-8")
    )
    for value, name in (
        (manifest, "campaign"),
        (route_manifest, "route fixture"),
    ):
        body = dict(value)
        stored = str(body.pop("manifest_hash", ""))
        if stored != canonical_sha256(body):
            raise ValueError(f"{name} manifest hash mismatch")
    map_name = str(args.map)
    if str(route_manifest["map_name"]) != map_name:
        raise ValueError("route fixture map mismatch")
    slots = [
        slot
        for slot in manifest["slots"]
        if str(slot["map_name"]) == map_name
    ]
    if not slots:
        raise ValueError(f"manifest contains no slots for {map_name}")

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16")
    report = resolver.preflight()
    if report.status != "READY":
        raise RuntimeError(
            f"CARLA preflight must be READY, got "
            f"{report.status}/{report.error_code}"
        )
    client, _ = resolver.connect(report=report)
    world = client.get_world()
    actual_map = str(world.get_map().name)
    if not _matches(actual_map, map_name):
        raise RuntimeError(
            f"map mismatch: actual={actual_map}, requested={map_name}"
        )
    route_bank = (
        dict(route_manifest.get("routes_by_fixture_id") or {})
        if str(route_manifest.get("schema_version") or "")
        == "safedrive.r2_v3.route_bank.v2"
        else {}
    )
    prototypes = (
        {}
        if route_bank
        else {
            str(case["scenario_id"]): case
            for case in _build_cases(
                world,
                route_manifest,
                include_traffic=False,
            )
        }
    )
    by_lineage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for slot in slots:
        by_lineage[str(slot["lineage_id"])].append(slot)
    authored: list[dict[str, str]] = []
    for lineage_id in sorted(by_lineage):
        lineage_slots = by_lineage[lineage_id]
        maneuver = RouteManeuver(str(lineage_slots[0]["maneuver"]))
        template_id = str(lineage_slots[0]["template_id"])
        route_fixture_id = str(lineage_slots[0]["route_fixture_id"])
        if route_bank and route_fixture_id not in route_bank:
            raise RuntimeError(
                f"{lineage_id}: route bank missing {route_fixture_id}"
            )
        prototype = _case_for_slot(
            world=world,
            route_manifest=route_manifest,
            prototype_by_id=prototypes,
            template_id=template_id,
            maneuver=maneuver,
            route_override=(
                route_bank[route_fixture_id]
                if route_bank
                else None
            ),
        )
        output = Path(args.out_root) / map_name / lineage_id / (
            "scenario_registry.toml"
        )
        text = _render_registry(
            map_name=map_name,
            lineage_id=lineage_id,
            slots=lineage_slots,
            prototype=prototype,
        )
        # Cold schema validation before the file becomes a campaign input.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="r2v3-campaign-author-") as tmp:
            probe = Path(tmp) / "scenario_registry.toml"
            probe.write_text(text, encoding="utf-8")
            registry = load_scenario_registry(probe)
            registry_hash = registry.compute_registry_sha256()
        _write_exclusive(output, text)
        authored.append(
            {
                "lineage_id": lineage_id,
                "registry": str(output.as_posix()),
                "registry_hash": registry_hash,
            }
        )
    index_body = {
        "schema_version": "safedrive.r2_v3.campaign_authoring.v1",
        "map_name": map_name,
        "campaign_manifest_hash": str(manifest["manifest_hash"]),
        "route_fixture_hash": str(route_manifest["manifest_hash"]),
        "outcome_used": False,
        "oracle_used": False,
        "lineages": authored,
    }
    index = {**index_body, "manifest_hash": canonical_sha256(index_body)}
    index_path = Path(args.out_root) / map_name / "authoring_index.json"
    _write_exclusive(
        index_path,
        json.dumps(index, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
