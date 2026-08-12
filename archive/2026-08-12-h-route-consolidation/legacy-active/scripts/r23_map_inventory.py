#!/usr/bin/env python3
"""Read-only CARLA map inventory for an R23 campaign."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r23_collection import (  # noqa: E402
    content_hash,
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--required-map", action="append", default=[])
    args = parser.parse_args()

    import carla

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(15.0)
    available = sorted(
        {
            str(item).rsplit("/", 1)[-1]
            for item in client.get_available_maps()
            if str(item).rsplit("/", 1)[-1]
        }
    )
    required = [str(item) for item in args.required_map]
    missing = sorted(set(required).difference(available))
    body = {
        "schema_version": "safedrive.r23_map_inventory.v1",
        "server_version": str(client.get_server_version()),
        "client_version": str(client.get_client_version()),
        "host": str(args.host),
        "port": int(args.port),
        "available_maps": available,
        "required_maps": required,
        "missing_required_maps": missing,
        "status": "READY" if not missing else "MISSING_REQUIRED_MAPS",
    }
    report = {**body, "inventory_content_hash": content_hash(body)}
    write_json_exclusive(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
