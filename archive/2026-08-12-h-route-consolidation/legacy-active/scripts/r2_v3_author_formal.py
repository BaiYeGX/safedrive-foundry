#!/usr/bin/env python3
"""Author frozen single-fixture registries for R2 V3 blind/audit gates."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    canonical_sha256,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402
from scripts.r2_v3_author_campaign import _case_for_slot  # noqa: E402
from scripts.r2_v3_author_long_smoke import _render_registry  # noqa: E402


def _read_frozen(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if stored != canonical_sha256(body):
        raise ValueError(f"{path}: manifest hash mismatch")
    return value


def _matches(actual: str, requested: str) -> bool:
    return actual.endswith(requested) or f"/{requested}" in actual


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--route-bank", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    args = parser.parse_args()
    if float(args.duration_s) < 15.0:
        parser.error("--duration-s must be at least 15")

    manifest = _read_frozen(Path(args.manifest))
    route_bank = _read_frozen(Path(args.route_bank))
    if (
        route_bank.get("schema_version")
        != "safedrive.r2_v3.route_bank.v2"
    ):
        raise ValueError("formal author requires route-bank v2")
    map_name = str(args.map)
    if str(route_bank["map_name"]) != map_name:
        raise ValueError("route-bank map mismatch")
    rows = [
        dict(row)
        for row in manifest.get("cases") or ()
        if str(row.get("map_name") or "") == map_name
    ]
    if not rows:
        raise ValueError(f"manifest contains no formal cases for {map_name}")

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16")
    preflight = resolver.preflight()
    if preflight.status != "READY":
        raise RuntimeError(
            f"CARLA preflight must be READY, got "
            f"{preflight.status}/{preflight.error_code}"
        )
    client, _ = resolver.connect(report=preflight)
    world = client.get_world()
    actual_map = str(world.get_map().name)
    if not _matches(actual_map, map_name):
        raise RuntimeError(
            f"map mismatch: actual={actual_map}, requested={map_name}"
        )

    authored = []
    routes = dict(route_bank["routes_by_fixture_id"])
    for row in rows:
        fixture_id = str(row["fixture_id"])
        route_fixture_id = str(row["route_fixture_id"])
        if route_fixture_id not in routes:
            raise ValueError(
                f"{fixture_id}: route bank missing {route_fixture_id}"
            )
        maneuver = RouteManeuver(str(row["maneuver"]))
        prototype = _case_for_slot(
            world=world,
            route_manifest={
                "routes": {},
                "horizon_m": route_bank["horizon_m"],
            },
            prototype_by_id={},
            template_id=str(row["template_id"]),
            maneuver=maneuver,
            route_override=routes[route_fixture_id],
        )
        route = dict(prototype["route"])
        case = {
            "scenario_id": fixture_id,
            "family": str(row["family"]),
            "route": route,
            "route_identity": route_fixture_id,
            "seed_id": str(row.get("seed_id") or "seed_a"),
            "notes": (
                f"phase={manifest['phase']};fixture={fixture_id};"
                "frozen_before_outcomes"
            ),
        }
        if prototype.get("actor"):
            case["actor"] = dict(prototype["actor"])
        if prototype.get("traffic_light"):
            case["traffic_light"] = dict(prototype["traffic_light"])
        registry_version = (
            f"r2v3-formal-{manifest['phase']}-{fixture_id}"
        )
        text = _render_registry(
            map_name=map_name,
            cases=[case],
            duration_s=float(args.duration_s),
            registry_version=registry_version,
        )
        with tempfile.TemporaryDirectory(
            prefix="r2v3-formal-author-"
        ) as tmp:
            probe = Path(tmp) / "scenario_registry.toml"
            probe.write_text(text, encoding="utf-8")
            registry = load_scenario_registry(probe)
            registry_hash = registry.compute_registry_sha256()
            loaded = registry.get(fixture_id, case["seed_id"])
            navigation = dict(loaded.route.navigation_context)
            if str(navigation.get("maneuver")) != maneuver.value:
                raise RuntimeError("formal registry maneuver mismatch")
        output = (
            Path(args.out_root)
            / str(manifest["phase"])
            / map_name
            / fixture_id
            / "scenario_registry.toml"
        )
        _write_exclusive(output, text)
        authored.append(
            {
                "fixture_id": fixture_id,
                "seed_id": case["seed_id"],
                "route_fixture_id": route_fixture_id,
                "route_hash": str(
                    route["route_context"]["route_hash"]
                ),
                "registry": str(output.as_posix()),
                "registry_hash": registry_hash,
            }
        )
    body = {
        "schema_version": "safedrive.r2_v3.formal_authoring.v1",
        "phase": str(manifest["phase"]),
        "map_name": map_name,
        "source_manifest_hash": str(manifest["manifest_hash"]),
        "route_bank_hash": str(route_bank["manifest_hash"]),
        "outcome_used": False,
        "oracle_used": False,
        "fixtures": authored,
    }
    index = {**body, "manifest_hash": canonical_sha256(body)}
    index_path = (
        Path(args.out_root)
        / str(manifest["phase"])
        / map_name
        / "authoring_index.json"
    )
    _write_exclusive(
        index_path,
        json.dumps(index, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
