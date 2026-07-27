#!/usr/bin/env python3
"""Run leakage-safe Spatial K2 feature support/predictability probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.feature_predictability import (  # noqa: E402
    build_feature_predictability_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(
            ROOT
            / "docs/runtime-evidence/r2x-training/dataset-v6-formal/samples.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(
            ROOT
            / "docs/runtime-evidence/r2x-training/feature_probe_v6.json"
        ),
    )
    parser.add_argument("--min-train-per-class", type=int, default=5)
    parser.add_argument("--min-val-per-class", type=int, default=3)
    args = parser.parse_args()

    data = Path(args.data)
    rows = [
        json.loads(line)
        for line in data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = build_feature_predictability_report(
        rows,
        min_train_per_class=args.min_train_per_class,
        min_val_per_class=args.min_val_per_class,
    )
    report["data_path"] = str(data.as_posix())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "FEATURE_PROBE_PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
