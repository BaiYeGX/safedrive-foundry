#!/usr/bin/env python3
"""Author unique, split-safe R2 V3 route fixtures for all frozen gates.

Only CARLA topology and already-frozen manifests are read.  Model outputs,
candidate outcomes and Oracle results are deliberately unavailable here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.route_authoring_v3 import (  # noqa: E402
    RouteAuthoringError,
    author_route_from_waypoint,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    canonical_sha256,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402
from scripts.r2_v3_author_long_smoke import (  # noqa: E402
    _traffic_light_turn,
)

SCHEMA = "safedrive.r2_v3.route_bank.v2"
PLAN_FILES = (
    "calibration_360.json",
    "core_blind_12.json",
    "world_ready_audit_84.json",
    "unseen_long_audit_16.json",
)


def _matches(actual: str, requested: str) -> bool:
    return actual.endswith(requested) or f"/{requested}" in actual


def _read_frozen(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if stored != canonical_sha256(body):
        raise ValueError(f"{path}: manifest hash mismatch")
    return value


def _requirement_from_row(
    row: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, str]:
    route_fixture_id = str(row.get("route_fixture_id") or "")
    if not route_fixture_id:
        raise ValueError(f"{phase}: route_fixture_id missing")
    return {
        "route_fixture_id": route_fixture_id,
        "phase": phase,
        "split": str(row.get("split") or phase),
        "map_name": str(row["map_name"]),
        "template_id": str(row["template_id"]),
        "family": str(row["family"]),
        "maneuver": RouteManeuver(str(row["maneuver"])).value,
    }


def build_route_requirements(
    plan_dir: Path,
    *,
    map_name: str,
) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for name in PLAN_FILES:
        manifest = _read_frozen(plan_dir / name)
        rows = (
            list(manifest.get("slots") or ())
            if name == "calibration_360.json"
            else list(manifest.get("cases") or ())
        )
        for row in rows:
            if str(row.get("map_name") or "") != str(map_name):
                continue
            requirement = _requirement_from_row(
                row,
                phase=str(manifest["phase"]),
            )
            route_id = requirement["route_fixture_id"]
            previous = by_id.get(route_id)
            if previous is not None and previous != requirement:
                raise ValueError(
                    f"{route_id}: conflicting frozen route requirements"
                )
            by_id[route_id] = requirement
    if not by_id:
        raise ValueError(f"no frozen route requirements for {map_name}")
    return list(by_id.values())


def _route_contexts_from_json(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    routes: list[Mapping[str, Any]] = []
    for key in ("routes", "routes_by_fixture_id"):
        raw = value.get(key)
        if isinstance(raw, Mapping):
            for route in raw.values():
                if isinstance(route, Mapping):
                    context = route.get("route_context")
                    if isinstance(context, Mapping):
                        routes.append(context)
    return routes


def _load_exclusions(
    *,
    route_fixtures: list[str],
    scenario_registries: list[str],
) -> tuple[set[str], set[str]]:
    hashes: set[str] = set()
    entries: set[str] = set()
    for raw_path in route_fixtures:
        value = _read_frozen(Path(raw_path))
        for context in _route_contexts_from_json(value):
            hashes.add(str(context["route_hash"]))
            entries.add(str(context["entry_signature"]))
    for raw_path in scenario_registries:
        registry = load_scenario_registry(raw_path)
        for fixture in registry.fixtures:
            navigation = dict(fixture.route.navigation_context or {})
            frozen = navigation.get("frozen_context_json")
            if not frozen:
                continue
            context = json.loads(str(frozen))
            hashes.add(str(context["route_hash"]))
            entries.add(str(context["entry_signature"]))
    return hashes, entries


def _required_adjacent_side(requirement: Mapping[str, str]) -> str:
    template = str(requirement["template_id"])
    maneuver = RouteManeuver(str(requirement["maneuver"]))
    if template == "cut_in_left":
        return "SPAWNABLE_LEFT"
    if template == "cut_in_right":
        return "SPAWNABLE_RIGHT"
    if template == "overtake_left":
        return "AUTHORIZED_LEFT"
    if template == "overtake_right":
        return "AUTHORIZED_RIGHT"
    if (
        template == "merge_yield"
        and maneuver is RouteManeuver.FOLLOW_STRAIGHT
    ):
        return "SPAWNABLE_RIGHT"
    if template == "merge_yield":
        return "SPAWNABLE_ANY"
    return "NONE"


def _adjacent_ok(route: Mapping[str, Any], required_side: str) -> bool:
    if required_side == "NONE":
        return True
    context = dict(route["route_context"])

    mode, _, requested_side = required_side.partition("_")

    def lane_ok(side: str) -> bool:
        lane = dict(context[f"{side.lower()}_lane"])
        spawnable = bool(
            lane.get("exists")
            and lane.get("driving")
            and lane.get("same_direction")
            and len(lane.get("centerline_xy") or ()) >= 2
        )
        if mode == "SPAWNABLE":
            return spawnable
        if mode == "AUTHORIZED":
            return bool(
                spawnable
                and lane.get("lane_change_allowed")
                and lane.get("currently_clear")
            )
        raise ValueError(f"unsupported adjacent requirement {required_side}")

    if requested_side == "ANY":
        return lane_ok("LEFT") or lane_ok("RIGHT")
    return lane_ok(requested_side)


def _author_one(
    *,
    world: Any,
    requirement: Mapping[str, str],
    horizon_m: float,
    used_hashes: set[str],
    used_entries: set[str],
    candidate_pool: Sequence[Mapping[str, Any]] = (),
    reusable_routes: Sequence[
        tuple[Mapping[str, str], Mapping[str, Any]]
    ] = (),
) -> tuple[dict[str, Any], str]:
    maneuver = RouteManeuver(str(requirement["maneuver"]))
    if str(requirement["template_id"]) == "traffic_control":
        route, traffic = _traffic_light_turn(
            world,
            horizon_m=horizon_m,
            maneuver=maneuver,
            used_route_hashes=tuple(used_hashes),
            used_entry_signatures=tuple(used_entries),
        )
        return {**route, "traffic_light": traffic}, ""

    required_side = _required_adjacent_side(requirement)
    for raw_route in candidate_pool:
        route = dict(raw_route)
        context = dict(route["route_context"])
        route_hash = str(context["route_hash"])
        if (
            route_hash not in used_hashes
            and
            str(context["entry_signature"]) not in used_entries
            and _adjacent_ok(route, required_side)
        ):
            return route, ""
    # A small map may not expose one unique road corridor per interaction
    # template.  Reuse is allowed only inside the same held-out formal phase;
    # calibration and every cross-phase boundary remain route-disjoint.
    if str(requirement["phase"]) != "calibration":
        for prior_requirement, prior_route in reusable_routes:
            if (
                str(prior_requirement["phase"])
                == str(requirement["phase"])
                and str(prior_requirement["maneuver"])
                == maneuver.value
                and _adjacent_ok(prior_route, required_side)
            ):
                return (
                    dict(prior_route),
                    str(prior_requirement["route_fixture_id"]),
                )
    raise RouteAuthoringError(
        f"no split-safe route for {requirement['route_fixture_id']}"
    )


def _candidate_pool(
    *,
    world_map: Any,
    maneuver: RouteManeuver,
    horizon_m: float,
    requirements: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Scan one maneuver once and retain only relevant unique entries."""
    modes = {
        _required_adjacent_side(requirement)
        for requirement in requirements
    }
    # A corridor can satisfy several modes (for example SPAWNABLE_LEFT and
    # AUTHORIZED_LEFT), but it may only be assigned to one calibration
    # fixture.  Reserve enough independent corridors for the entire maneuver
    # group per mode; counting only the rows requesting that exact mode can
    # strand a later fixture after overlapping corridors were consumed.
    targets = {mode: len(requirements) + 2 for mode in modes}
    counts = {mode: 0 for mode in modes}
    routes: list[dict[str, Any]] = []
    hashes: set[str] = set()
    entries: set[str] = set()
    starts = sorted(
        world_map.generate_waypoints(4.0),
        key=lambda waypoint: (
            int(waypoint.road_id),
            int(waypoint.lane_id),
            float(waypoint.s),
        ),
    )
    for start in starts:
        try:
            route = author_route_from_waypoint(
                start,
                maneuver=maneuver,
                horizon_m=float(horizon_m),
            ).to_dict()
        except RouteAuthoringError:
            continue
        context = dict(route["route_context"])
        route_hash = str(context["route_hash"])
        entry = str(context["entry_signature"])
        if route_hash in hashes or entry in entries:
            continue
        matching = {
            mode for mode in modes if _adjacent_ok(route, mode)
        }
        # Once a mode has enough corridors, routes which only satisfy that
        # already-filled mode are useless.  Keeping them made a map with no
        # left adjacent lane accumulate O(100k) ordinary straight routes
        # while it searched to the end of the topology.
        unmet_matching = {
            mode
            for mode in matching
            if counts[mode] < targets[mode]
        }
        if not unmet_matching:
            continue
        routes.append(route)
        hashes.add(route_hash)
        entries.add(entry)
        for mode in unmet_matching:
            counts[mode] += 1
        if all(counts[mode] >= targets[mode] for mode in modes):
            break
    return routes


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-m", type=float, default=140.0)
    parser.add_argument("--exclude-route-fixture", action="append", default=[])
    parser.add_argument(
        "--exclude-scenario-registry",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    if float(args.horizon_m) < 80.0:
        parser.error("--horizon-m must be at least 80")

    requirements = build_route_requirements(
        Path(args.plan_dir),
        map_name=str(args.map),
    )
    used_hashes, used_entries = _load_exclusions(
        route_fixtures=list(args.exclude_route_fixture),
        scenario_registries=list(args.exclude_scenario_registry),
    )
    excluded_hashes = sorted(used_hashes)
    excluded_entries = sorted(used_entries)

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
    if not _matches(actual_map, str(args.map)):
        raise RuntimeError(
            f"map mismatch: actual={actual_map}, requested={args.map}"
        )

    # Author the scarcest corridors first so ordinary straight roads cannot
    # consume a signalized/adjacent-lane entry signature.
    requirements.sort(
        key=lambda item: (
            0 if item["template_id"] == "traffic_control" else 1,
            0 if _required_adjacent_side(item) != "NONE" else 1,
            item["route_fixture_id"],
        )
    )
    routes: dict[str, Any] = {}
    frozen_requirements: dict[str, Any] = {}
    authored_routes: list[
        tuple[Mapping[str, str], Mapping[str, Any]]
    ] = []
    route_hash_phase = {
        route_hash: "__excluded__" for route_hash in used_hashes
    }
    entry_phase = {entry: "__excluded__" for entry in used_entries}
    reuse_by_fixture_id: dict[str, str] = {}
    pools: dict[RouteManeuver, list[dict[str, Any]]] = {}
    ordinary_by_maneuver = {
        maneuver: [
            requirement
            for requirement in requirements
            if requirement["template_id"] != "traffic_control"
            and RouteManeuver(str(requirement["maneuver"])) is maneuver
        ]
        for maneuver in RouteManeuver
    }
    for requirement_index, requirement in enumerate(requirements, 1):
        route_id = str(requirement["route_fixture_id"])
        maneuver = RouteManeuver(str(requirement["maneuver"]))
        if (
            requirement["template_id"] != "traffic_control"
            and maneuver not in pools
        ):
            print(
                f"[route-bank] scanning {maneuver.value} candidate pool",
                flush=True,
            )
            pools[maneuver] = _candidate_pool(
                world_map=world.get_map(),
                maneuver=maneuver,
                horizon_m=float(args.horizon_m),
                requirements=ordinary_by_maneuver[maneuver],
            )
            print(
                f"[route-bank] {maneuver.value} candidates="
                f"{len(pools[maneuver])}",
                flush=True,
            )
        print(
            f"[route-bank] {requirement_index}/{len(requirements)} "
            f"{route_id}",
            flush=True,
        )
        route, reused_from = _author_one(
            world=world,
            requirement=requirement,
            horizon_m=float(args.horizon_m),
            used_hashes=used_hashes,
            used_entries=used_entries,
            candidate_pool=pools.get(maneuver, ()),
            reusable_routes=authored_routes,
        )
        context = dict(route["route_context"])
        route_hash = str(context["route_hash"])
        entry = str(context["entry_signature"])
        phase = str(requirement["phase"])
        prior_hash_phase = route_hash_phase.get(route_hash)
        prior_entry_phase = entry_phase.get(entry)
        if (
            prior_hash_phase is not None
            and (phase == "calibration" or prior_hash_phase != phase)
        ):
            raise AssertionError("cross-phase route hash reuse invariant violated")
        if (
            prior_entry_phase is not None
            and (phase == "calibration" or prior_entry_phase != phase)
        ):
            raise AssertionError("cross-phase entry reuse invariant violated")
        used_hashes.add(route_hash)
        used_entries.add(entry)
        route_hash_phase[route_hash] = phase
        entry_phase[entry] = phase
        routes[route_id] = route
        frozen_requirements[route_id] = dict(requirement)
        authored_routes.append((dict(requirement), route))
        if reused_from:
            reuse_by_fixture_id[route_id] = reused_from

    body = {
        "schema_version": SCHEMA,
        "map_name": str(args.map),
        "actual_map": actual_map,
        "horizon_m": float(args.horizon_m),
        "route_count": len(routes),
        "routes_by_fixture_id": routes,
        "requirements_by_fixture_id": frozen_requirements,
        "reuse_by_fixture_id": reuse_by_fixture_id,
        "excluded_route_hashes": excluded_hashes,
        "excluded_entry_signatures": excluded_entries,
        "authoring_constraints": {
            "map_topology_only": True,
            "unique_route_hash": not bool(reuse_by_fixture_id),
            "unique_entry_signature": not bool(reuse_by_fixture_id),
            "calibration_route_hash_unique": True,
            "cross_phase_route_hash_unique": True,
            "cross_phase_entry_signature_unique": True,
            "within_formal_phase_reuse_only_on_capacity_exhaustion": True,
            "model_output_used": False,
            "candidate_outcome_used": False,
            "oracle_used": False,
        },
    }
    payload = {**body, "manifest_hash": canonical_sha256(body)}
    _write_exclusive(Path(args.out), payload)
    print(
        json.dumps(
            {
                "status": "AUTHORED",
                "out": str(args.out),
                "map": str(args.map),
                "route_count": len(routes),
                "manifest_hash": payload["manifest_hash"],
                "outcome_used": False,
                "oracle_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
