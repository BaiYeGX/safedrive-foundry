#!/usr/bin/env python3
"""Explicitly reload the CARLA authoring map without changing engine config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from runtime.carla_connection import ConnectionResolver  # noqa: E402


def _matches(actual: str, requested: str) -> bool:
    return actual.endswith(requested) or f"/{requested}" in actual


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reload one CARLA map for route/scenario authoring. "
            "Does not tick, spawn actors, or modify DefaultEngine.ini."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Reload even when the requested map is already active.",
    )
    args = parser.parse_args()
    if args.timeout < 10.0:
        parser.error("--timeout must be at least 10 seconds")

    resolver = ConnectionResolver(ROOT, expected_version="0.9.16")
    report = resolver.preflight()
    if report.status != "READY":
        raise RuntimeError(
            f"CARLA preflight must be READY, got {report.status}/{report.error_code}"
        )
    client, _ = resolver.connect(report=report)
    client.set_timeout(float(args.timeout))
    before = str(client.get_world().get_map().name)
    if _matches(before, args.map) and not args.force_reload:
        after = before
        reloaded = False
    else:
        world = client.load_world(str(args.map), reset_settings=True)
        after = str(world.get_map().name)
        reloaded = True
    if not _matches(after, args.map):
        raise RuntimeError(
            f"map reload mismatch: requested={args.map!r}, actual={after!r}"
        )
    print(
        json.dumps(
            {
                "status": "READY",
                "requested_map": str(args.map),
                "before_map": before,
                "actual_map": after,
                "reloaded": reloaded,
                "tick_used": False,
                "actors_spawned": False,
                "engine_config_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
