#!/usr/bin/env python3
"""H5 orchestrator: one-shot pilot/full closed-loop execution and acceptance.

Run:
    python scripts/h5_run.py --dataset-id h5-<UTC> --pilot --full --accept
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
MAPS = ("Town01", "Town03", "Town05")


def _run(arguments: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )
    output = (completed.stdout or "").strip().splitlines()
    payload: dict[str, Any] = {
        "command": [sys.executable, *arguments],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if output:
        try:
            payload["result"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            try:
                payload["result"] = json.loads(output[-1])
            except json.JSONDecodeError:
                pass
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return payload


def _restart(map_name: str) -> dict[str, Any]:
    return _run(
        [
            "scripts/sdf.py", "sim", "restart", "--map", map_name, "--rhi", "dx12",
            "--startup-timeout", "180", "--shutdown-timeout", "30", "--json",
        ]
    )


def _collect(dataset_id: str, map_name: str, scope: str) -> dict[str, Any]:
    return _run(
        ["scripts/h5_collect.py", "--dataset-id", dataset_id, "--map", map_name, "--scope", scope]
    )


def _accept(dataset_id: str, *, integrity_only: bool = False, scope: str = "full") -> dict[str, Any]:
    args = ["scripts/h5_acceptance.py", "--dataset-id", dataset_id, "--scope", scope]
    if integrity_only:
        args.append("--integrity-only")
    return _run(args)


def _readiness() -> dict[str, Any]:
    return _run(["scripts/h5_readiness.py"])


def _carla_preflight() -> dict[str, Any]:
    return _run(["scripts/sdf.py", "sim", "preflight", "--json"])


def _map_ready(map_name: str) -> bool:
    try:
        pre = _carla_preflight()
    except Exception:
        return False
    result = pre.get("result") or {}
    return result.get("status") == "READY" and str(result.get("map", "")).endswith(map_name)


def _set_world_async() -> None:
    """Make CARLA world asynchronous so `sim restart` can proceed."""
    try:
        import carla  # noqa: F401
        from runtime.carla_connection import ConnectionResolver, READY

        resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=10.0)
        report = resolver.preflight()
        if report.status != READY:
            return
        client, _ = resolver.connect(report=report)
        world = client.get_world()
        settings = world.get_settings()
        if settings.synchronous_mode:
            settings.synchronous_mode = False
            world.apply_settings(settings)
    except Exception:
        pass


def _ensure_map(map_name: str) -> dict[str, Any]:
    if _map_ready(map_name):
        return {"status": "SKIP_RESTART_ALREADY_READY", "map": map_name}
    # Ensure the world is asynchronous before asking for a cold restart.
    _set_world_async()
    try:
        return _restart(map_name)
    except RuntimeError as exc:
        # A restart may time out in the wrapper while CARLA later becomes READY.
        if _map_ready(map_name):
            return {"status": "RESTART_TIMEOUT_RECOVERED", "map": map_name, "error": str(exc)}
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    dataset_id = args.dataset_id or time.strftime("h5-%Y%m%dT%H%M%SZ", time.gmtime())
    trace: dict[str, Any] = {
        "dataset_id": dataset_id,
        "started_wall_time_s": time.time(),
        "steps": [],
    }
    try:
        trace["steps"].append(_readiness())
        preflight = _carla_preflight()
        trace["steps"].append(preflight)
        if args.preflight:
            ready = bool((preflight.get("result") or {}).get("status") == "READY")
            print(json.dumps({"ok": ready, "dataset_id": dataset_id, "preflight": preflight.get("result")}, sort_keys=True))
            return 0 if ready else 1
        if args.pilot:
            for map_name in MAPS:
                trace["steps"].append(_ensure_map(map_name))
                trace["steps"].append(_collect(dataset_id, map_name, "pilot"))
            trace["steps"].append(_accept(dataset_id, integrity_only=True, scope="pilot"))
            pilot_result = trace["steps"][-1]
            if not bool((pilot_result.get("result") or {}).get("gate_passed")):
                raise RuntimeError("PILOT_INTEGRITY_GATE_FAILED")
        if args.full:
            for map_name in MAPS:
                trace["steps"].append(_ensure_map(map_name))
                trace["steps"].append(_collect(dataset_id, map_name, "full"))
        if args.accept or args.full:
            trace["steps"].append(_accept(dataset_id))
        trace["ok"] = True
        trace["gate_passed"] = bool((trace["steps"][-1].get("result") or {}).get("gate_passed"))
    except BaseException as exc:
        trace["ok"] = False
        trace["error"] = f"{type(exc).__name__}:{exc}"
    trace["ended_wall_time_s"] = time.time()
    evidence = ROOT / "docs" / "runtime-evidence" / "h5" / dataset_id / "orchestrator.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(trace, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": trace["ok"], "dataset_id": dataset_id, "evidence": str(evidence), "gate_passed": trace.get("gate_passed"), "error": trace.get("error")}, sort_keys=True))
    return 0 if trace["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
