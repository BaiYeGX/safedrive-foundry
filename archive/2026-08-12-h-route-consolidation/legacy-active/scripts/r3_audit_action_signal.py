#!/usr/bin/env python3
"""Audit candidate-conditioned actor response against an A/A noise floor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.action_signal import action_signal_report  # noqa: E402
from driving_vla.world.dataset import ActionBranchDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--aa-p99", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-reactive-fraction", type=float, default=0.25)
    parser.add_argument("--min-reactive-decisive-fraction", type=float, default=0.50)
    args = parser.parse_args()
    dataset = ActionBranchDataset(Path(args.dataset))
    samples = [dataset[index].sample for index in range(len(dataset))]
    report = action_signal_report(
        samples,
        aa_p99=float(args.aa_p99),
        min_reactive_fraction=float(args.min_reactive_fraction),
        min_reactive_decisive_fraction=float(args.min_reactive_decisive_fraction),
    )
    report["dataset"] = str(Path(args.dataset))
    report["dataset_manifest"] = dataset.manifest
    report["passed"] = bool(
        report["gate_has_observable_signal"]
        and report["gate_reactive_fraction"]
        and report["gate_reactive_decisive_fraction"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
