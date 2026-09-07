#!/usr/bin/env python3
"""Prepare, audit, train and finalize the existing-data C2 World baseline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h6.cora.devbaseline import finalize_devbaseline, prepare_release, train_baselines  # noqa: E402

CONFIG = ROOT / "safedrive_foundry/config/h6/cora_c2_devbaseline_v1.toml"
DATASET = ROOT / "generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1"
EVIDENCE = ROOT / "docs/runtime-evidence/h6/h6-cora-c2-devbaseline-20260907-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "audit", "baseline", "finalize", "test"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_release(args.config, ROOT), indent=2))
    elif args.command == "audit":
        print(json.dumps(prepare_release(args.config, ROOT), indent=2))
    elif args.command == "baseline":
        print(json.dumps(train_baselines(args.config, ROOT), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize_devbaseline(args.config, ROOT), indent=2))
    else:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        payload = {"schema_version": "safedrive.cora.dev_test_report.v1", "passed": result.returncode == 0, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        (EVIDENCE / "test-report.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": result.returncode == 0, "evidence": str(EVIDENCE / "test-report.json"), "exit_code": result.returncode}))
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
