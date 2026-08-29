#!/usr/bin/env python3
"""Create an immutable H6 VLA75 formal run lock.

The lock is created only after development readiness has passed.  It records
the selected lineage/matrix, three VLA75 checkpoints, calibration, scoped
runtime hashes and the complete dirty-worktree identity; documents updated
after a run are intentionally not part of the scoped code hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h6.matrix import load_h6_vla75_matrix  # noqa: E402
from data_pipeline.h6.run_lock import (  # noqa: E402
    build_h6_vla75_run_lock,
    verify_run_lock,
    write_run_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        choices=("vla75-v2",),
        default="vla75-v2",
        help="Formal lock schema; v1 locks are intentionally unsupported.",
    )
    parser.add_argument("--formal-lineage", choices=("a", "b", "c"), required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training-root", action="append", type=Path, default=[])
    parser.add_argument("--versions", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset_id.startswith("h6-vla75-"):
        raise SystemExit("h6_vla75_dataset_id_required")
    calibration_input = json.loads(args.calibration.read_text(encoding="utf-8"))
    # Accept either the complete v75 training summary or an already extracted
    # calibration object, but persist one canonical lock shape.  Formal
    # collection can then consume the exact deployment thresholds, temperatures
    # and temporal router values that readiness audited.
    if isinstance(calibration_input, dict) and "calibration" in calibration_input:
        calibration = {
            "deployment": dict(calibration_input.get("calibration") or {}),
            "router": dict(calibration_input.get("router_calibration") or {}),
            "temperatures": dict(calibration_input.get("temperature_calibration") or {}),
        }
        # C1 summary bindings travel inside the deployment payload so the
        # existing run-lock shape remains self-hashed and backward readable.
        calibration["deployment"].setdefault(
            "c1_bindings",
            {
                "evaluator_sha256": [
                    str(item.get("sha256"))
                    for item in calibration_input.get("evaluators", ())
                ],
                "validation_lineage_sha256": calibration_input.get(
                    "validation_lineage_sha256"
                ),
                "training_input_sha256": calibration_input.get(
                    "train_lineage_sha256"
                ),
            },
        )
    else:
        calibration = dict(calibration_input)
    versions = (
        {}
        if args.versions is None
        else json.loads(args.versions.read_text(encoding="utf-8"))
    )
    rows = load_h6_vla75_matrix(
        args.formal_lineage,
        full=args.scope == "full",
    )
    checkpoints = tuple(
        path if path.is_absolute() else ROOT / path for path in args.checkpoint
    )
    lock = build_h6_vla75_run_lock(
        ROOT,
        lineage_id=args.formal_lineage,
        dataset_id=args.dataset_id,
        matrix_rows=rows,
        checkpoint_paths=checkpoints,
        calibration=calibration,
        training_roots=args.training_root,
        versions=versions,
    )
    verification = verify_run_lock(lock, root=ROOT)
    if not verification["valid"]:
        raise SystemExit(
            "h6_vla75_run_lock_invalid:"
            + ",".join(str(item) for item in verification["failures"])
        )
    digest = write_run_lock(args.output, lock)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "lineage_id": args.formal_lineage,
                "scope": args.scope,
                "matrix_pairs": len(rows),
                "lock_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
