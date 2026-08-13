#!/usr/bin/env python3
"""Run the fixed H2 sequence: restart/materialize → pilot gate → full matrix/audit."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args()
    dataset_id = args.dataset_id or time.strftime("h2-%Y%m%dT%H%M%SZ", time.gmtime())
    trace: dict[str, Any] = {"dataset_id": dataset_id, "started_wall_time_s": time.time(), "steps": []}
    try:
        for map_name in MAPS:
            trace["steps"].append(_restart(map_name))
            trace["steps"].append(
                _run(["scripts/h2_collect.py", "--dataset-id", dataset_id, "restart-smoke", "--map", map_name])
            )
            trace["steps"].append(
                _run(["scripts/h2_collect.py", "--dataset-id", dataset_id, "materialize-map", "--map", map_name])
            )
        trace["steps"].append(_run(["scripts/h2_collect.py", "--dataset-id", dataset_id, "freeze-manifest"]))
        for map_name in MAPS:
            trace["steps"].append(_restart(map_name))
            trace["steps"].append(
                _run(
                    [
                        "scripts/h2_collect.py", "--dataset-id", dataset_id,
                        "collect-map", "--map", map_name, "--scope", "pilot",
                    ]
                )
            )
        pilot = _run(["scripts/h2_label_audit.py", "--dataset-id", dataset_id, "--scope", "pilot"])
        trace["steps"].append(pilot)
        if not bool((pilot.get("result") or {}).get("gate_passed")):
            raise RuntimeError("PILOT_GATE_FAILED_FIXED_MATRIX_NOT_EXPANDED")
        for map_name in MAPS:
            trace["steps"].append(_restart(map_name))
            trace["steps"].append(
                _run(
                    [
                        "scripts/h2_collect.py", "--dataset-id", dataset_id,
                        "collect-map", "--map", map_name, "--scope", "full",
                    ]
                )
            )
        full = _run(["scripts/h2_label_audit.py", "--dataset-id", dataset_id, "--scope", "full"])
        trace["steps"].append(full)
        trace["ok"] = True
        trace["gate_passed"] = bool((full.get("result") or {}).get("gate_passed"))
    except BaseException as exc:
        trace["ok"] = False
        trace["error"] = f"{type(exc).__name__}:{exc}"
    trace["ended_wall_time_s"] = time.time()
    evidence = ROOT / "docs" / "runtime-evidence" / "h2" / dataset_id / "orchestrator.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(trace, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": trace["ok"], "dataset_id": dataset_id, "evidence": str(evidence), "error": trace.get("error")}, sort_keys=True))
    return 0 if trace["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
