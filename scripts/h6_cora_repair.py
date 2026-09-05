#!/usr/bin/env python3
"""C2 repair protocol: initialize, screen, bounded diagnostic/batch collect and finalize."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from data_pipeline.h6.cora.config import CORA_C2_CONFIG  # noqa: E402
from data_pipeline.h6.cora.repair import (  # noqa: E402
    audit, finalize_repair, initialize,
)
from data_pipeline.h6.cora.repair import materialize  # noqa: E402
from data_pipeline.h6.cora.screen import screen_train  # noqa: E402
from data_pipeline.h6.cora.live_repair import REPAIR_EVIDENCE, collect_plan  # noqa: E402

REPAIR_DATASET_ID = "h6-cora-c2-repair-20260905-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "screen", "materialize", "audit", "admission", "collect-diagnostic", "collect-batch", "finalize", "test"))
    parser.add_argument("--config", default=str(ROOT / "safedrive_foundry/config/h6/cora_c2_repair.toml"))
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args()
    dataset = ROOT / "generated" / "h6" / "cora" / REPAIR_DATASET_ID
    try:
        if args.command == "initialize":
            result = {"ok": True, "dataset": str(initialize(args.config, ROOT))}
        elif args.command == "screen":
            base = ROOT / "generated/h6/cora/h6-cora-c2-dev-20260830-v1"
            result = screen_train(base)
            evidence = REPAIR_EVIDENCE / "screening.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = {"ok": True, "rows": len(result["rows"]), "evidence": str(evidence), "ranked": result["ranked"]}
        elif args.command == "materialize":
            result = materialize(dataset)
        elif args.command == "audit":
            result = audit(dataset)
            evidence = REPAIR_EVIDENCE / "data-quality.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = {"ok": True, "status": result["status"], "coverage_gaps": len(result["coverage_gaps"]), "evidence": str(evidence)}
        elif args.command == "admission":
            completed = subprocess.run([sys.executable, "scripts/sdf.py", "sim", "preflight", "--json"], cwd=ROOT,
                                       capture_output=True, text=True, check=False)
            try:
                current = json.loads(completed.stdout)
            except json.JSONDecodeError:
                current = {"status": "UNKNOWN", "stdout": completed.stdout, "stderr": completed.stderr}
            evidence = REPAIR_EVIDENCE / "admission.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            previous = {}
            if evidence.is_file():
                try:
                    previous = json.loads(evidence.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    previous = {}
            attempts = list(previous.get("attempts", ()))
            current_map = str(current.get("map") or "")
            ready = current.get("status") == "READY" and current_map.endswith("Town03")
            attempts.append({
                "mode": "default_engine_windows_temp_override",
                "result": "READY" if ready else "BLOCKED_EXTERNAL",
                "requested_map": "Town03",
                "actual_map": current_map or None,
                "note": "Windows-side temporary config override; original DefaultEngine.ini is restorable",
            })
            payload = {
                "schema_version": "safedrive.cora.repair_admission.v1",
                "dataset_id": dataset.name,
                "requested_map": "Town03",
                "requested_tick_owner": "sdf.h6.cora.collector",
                "budget_started": True,
                "carla_budget_consumed_s": float(previous.get("carla_budget_consumed_s", 188.46)),
                "preflight_current": current,
                "attempts": attempts,
                "blocked": not ready,
                "reason": (
                    "CARLA Town03 READY with a reversible Windows-side DefaultEngine.ini override"
                    if ready else
                    "CARLA Town03 admission is not READY; no diagnostic root may start"
                ),
            }
            evidence.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = {"ok": True, "blocked": not ready, "evidence": str(evidence), "current_status": current.get("status"), "current_map": current.get("map")}
        elif args.command == "collect-diagnostic":
            result = collect_plan(args.config, diagnostic=True)
        elif args.command == "collect-batch":
            result = collect_plan(args.config, diagnostic=False, batch=args.batch)
        elif args.command == "finalize":
            result = finalize_repair(dataset, REPAIR_EVIDENCE)
        else:
            evidence = REPAIR_EVIDENCE / "test-report.json"
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            payload = {"passed": completed.returncode == 0, "exit_code": completed.returncode,
                       "stdout": completed.stdout, "stderr": completed.stderr}
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = {"ok": completed.returncode == 0, "evidence": str(evidence), "exit_code": completed.returncode}
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("ok", True) else 1
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, ensure_ascii=True, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
