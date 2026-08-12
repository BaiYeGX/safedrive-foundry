#!/usr/bin/env python3
"""R2-H0 data availability audit for Spatial K2 dual labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "runtime-evidence" / "r2x-training"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = {
        "simlingo_code": (ROOT / "simlingo-main").is_dir(),
        "simlingo_ckpt_dir": (ROOT / "models" / "simlingo").is_dir()
        or any((ROOT / "models").glob("**/pytorch_model*")),
        "internvl": (ROOT / "models" / "InternVL2-1B").is_dir(),
        "r2_pilot_readonly": (
            ROOT / "docs/runtime-evidence/r2-g4a-paired-pilot/run_set_report.json"
        ).is_file(),
        "carla_hint": (ROOT / "docs/ENVIRONMENT.md").is_file(),
    }
    # Decision: we can always GENERATE_TEACHER offline (Frenet lattice on native paths)
    # using frozen pilot anchors as geometry-only seeds without leaking outcomes into train
    # by regenerating labels only from path geometry (not oracle winners).
    decision = "GENERATE_TEACHER"
    reasons = [
        "No licensed dual-label Action Dreaming corpus wired into safedrive_foundry yet.",
        "Offline Frenet defensive teacher from native geometry is allowed as bootstrap",
        "and does not copy R2 pair outcomes into train labels.",
        "Old R2 12-pair remains holdout/regression only.",
    ]
    if not assets["simlingo_code"]:
        decision = "BLOCKED_DATA"
        reasons.append("simlingo-main missing")
    report = {
        "schema_version": "safedrive.r2x.data_decision.v1",
        "R2H_DATA_DECISION": decision,
        "assets": assets,
        "reasons": reasons,
        "train_policy": {
            "forbid_r2_pilot_in_train": True,
            "forbid_regression_in_train": True,
            "teacher": "offline_frenet_lattice_from_native_path",
            "runtime_lineage": "spatial_mode_head",
        },
    }
    path = OUT / "R2H_DATA_DECISION.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if decision != "BLOCKED_DATA" else 2


if __name__ == "__main__":
    raise SystemExit(main())
