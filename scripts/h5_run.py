#!/usr/bin/env python3
"""H5 entry point.

This script is intentionally a preflight/entry helper.  Actual closed-loop
on/off execution requires a separate H5 task with a frozen scenario matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_readiness() -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "h5_readiness.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"ready": False, "error": result.stdout.strip() or result.stderr.strip()}
    return json.loads(result.stdout)


def _run_carla_preflight() -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sdf.py"), "sim", "preflight", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"ready": False, "error": result.stdout.strip() or result.stderr.strip()}
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", default=True)
    args = parser.parse_args()

    readiness = _run_readiness()
    carla = _run_carla_preflight()
    ready = bool(readiness.get("ready")) and carla.get("status") == "READY"
    print(json.dumps({
        "ready": ready,
        "h5_readiness": readiness,
        "carla_preflight": {
            "status": carla.get("status"),
            "map": carla.get("map"),
            "server_version": carla.get("server_version"),
            "tick_owner": carla.get("tick_owner"),
        },
        "next_step": "Authorize and run closed-loop on/off experiment with H5WorldRouter.",
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
