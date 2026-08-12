#!/usr/bin/env python3
"""Author and freeze exact CARLA route fixtures for K2 V3.

This command only reads map topology.  It does not spawn actors, run the VLA,
execute candidates, or inspect outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.route_authoring_v3 import (  # noqa: E402
    find_authored_route_v3,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    canonical_sha256,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402


SCHEMA = "safedrive.r2_v3.route_fixtures.v1"


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze eight exact route-manoeuvre fixtures from CARLA topology"
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-m", type=float, default=80.0)
    args = parser.parse_args()
    if args.horizon_m < 45.0:
        parser.error("--horizon-m must be at least 45 m")

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16")
    report = resolver.preflight()
    if report.status != "READY":
        raise RuntimeError(
            f"CARLA preflight must be READY, got {report.status}/{report.error_code}"
        )
    client, _ = resolver.connect(report=report)
    world_map = client.get_world().get_map()
    actual_map = str(world_map.name)
    if not (actual_map.endswith(args.map) or f"/{args.map}" in actual_map):
        raise RuntimeError(
            f"map mismatch: server={actual_map!r}, requested={args.map!r}"
        )

    used: list[str] = []
    routes: dict[str, Any] = {}
    for maneuver in RouteManeuver:
        authored = find_authored_route_v3(
            world_map,
            maneuver=maneuver,
            used_route_hashes=used,
            horizon_m=float(args.horizon_m),
        )
        used.append(authored.route_context.route_hash)
        routes[maneuver.value] = authored.to_dict()

    body = {
        "schema_version": SCHEMA,
        "map_name": str(args.map),
        "actual_map": actual_map,
        "horizon_m": float(args.horizon_m),
        "route_count": len(routes),
        "routes": routes,
        "authoring_constraints": {
            "map_topology_only": True,
            "model_output_used": False,
            "candidate_outcome_used": False,
            "oracle_used": False,
        },
    }
    payload = {**body, "manifest_hash": canonical_sha256(body)}
    out = Path(args.out)
    _write_json_exclusive(out, payload)
    print(
        json.dumps(
            {
                "out": str(out),
                "map": args.map,
                "route_count": len(routes),
                "manifest_hash": payload["manifest_hash"],
                "route_hashes": {
                    key: value["route_context"]["route_hash"]
                    for key, value in routes.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
