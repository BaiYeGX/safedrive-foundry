#!/usr/bin/env python3
"""Validate the immutable R3 final-head dataset against every hard gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.dataset import ActionBranchDatasetV1  # noqa: E402
from driving_vla.world.final_quality import validate_final_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--r2-checkpoint-sha256", required=True)
    parser.add_argument("--namespace", default="r3_final_head_formal")
    parser.add_argument(
        "--aa-p99",
        type=float,
        default=None,
        help="simulator A/A P99 actor-future distance; required for final action-response gates",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = ActionBranchDatasetV1(Path(args.dataset), verify_hashes=True)
    report = validate_final_dataset(
        dataset,
        checkpoint_sha256=str(args.r2_checkpoint_sha256),
        namespace=str(args.namespace),
        aa_p99=args.aa_p99,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_hard_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
