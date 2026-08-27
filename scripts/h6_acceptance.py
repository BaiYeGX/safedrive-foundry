#!/usr/bin/env python3
"""Evaluate v1/v2 actual-execution closed-loop acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h5.store import H5Store  # noqa: E402
from data_pipeline.h6.acceptance import evaluate_vla75_gate, evaluate_vla90_gate  # noqa: E402
from data_pipeline.h6.matrix import load_h6_matrix, load_h6_vla75_matrix  # noqa: E402
from data_pipeline.h6.run_lock import verify_run_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-arm", default="on")
    parser.add_argument("--baseline-arm", default="off")
    parser.add_argument("--scope", choices=("pilot", "full"))
    parser.add_argument("--contract", choices=("vla90", "vla75-v2"), default="vla90")
    parser.add_argument("--formal-lineage", choices=("a", "b", "c"), default=None)
    parser.add_argument("--run-lock", type=Path, default=None)
    args = parser.parse_args()
    if args.contract == "vla75-v2":
        if not args.dataset_id.startswith("h6-vla75-"):
            raise ValueError("h6_vla75_dataset_id_required")
        if not args.formal_lineage:
            raise ValueError("h6_vla75_formal_lineage_required")
        if args.run_lock is None:
            raise ValueError("h6_vla75_run_lock_required")
    run_lock = None
    if args.run_lock is not None:
        run_lock = json.loads(args.run_lock.read_text(encoding="utf-8"))
        verification = verify_run_lock(run_lock, root=ROOT)
        if not verification["valid"]:
            raise ValueError(f"run_lock_invalid:{verification['failures']}")
        if args.contract == "vla75-v2":
            if str(run_lock.get("dataset_id")) != str(args.dataset_id):
                raise ValueError("h6_vla75_run_lock_dataset_mismatch")
            if str(run_lock.get("lineage_id")) != str(args.formal_lineage):
                raise ValueError("h6_vla75_run_lock_lineage_mismatch")
            if args.scope is not None:
                expected_pairs = 108 if args.scope == "full" else 12
                if int(run_lock.get("matrix_pairs", 0)) != expected_pairs:
                    raise ValueError("h6_vla75_run_lock_matrix_scope_mismatch")
    runs = H5Store(ROOT / "generated" / "h5", args.dataset_id).list_runs()
    if args.contract == "vla75-v2":
        result = evaluate_vla75_gate(
            runs,
            target_arm=args.target_arm,
            baseline_arm=args.baseline_arm,
            lineage_id=args.formal_lineage,
            run_lock=run_lock,
            run_lock_root=ROOT,
        )
    else:
        result = evaluate_vla90_gate(
            runs, target_arm=args.target_arm, baseline_arm=args.baseline_arm
        )
    if args.scope is not None:
        expected = {
            row.pair_id
            for row in (
                load_h6_vla75_matrix(args.formal_lineage, full=args.scope == "full")
                if args.contract == "vla75-v2"
                else load_h6_matrix(full=args.scope == "full")
            )
        }
        observed_target = {
            str(run["pair_id"])
            for run in runs
            if run.get("arm") == args.target_arm
        }
        observed_baseline = {
            str(run["pair_id"])
            for run in runs
            if run.get("arm") == args.baseline_arm
        }
        missing_target = sorted(expected - observed_target)
        missing_baseline = sorted(expected - observed_baseline)
        unexpected = sorted((observed_target | observed_baseline) - expected)
        if missing_target or missing_baseline or unexpected:
            result["failures"] = sorted(set(result["failures"]) | {"matrix_coverage"})
            result["passed"] = False
        result["matrix"] = {
            "scope": args.scope,
            "expected_pairs": len(expected),
            "observed_target_pairs": len(observed_target),
            "observed_baseline_pairs": len(observed_baseline),
            "missing_target": missing_target,
            "missing_baseline": missing_baseline,
            "unexpected": unexpected,
        }
    result["dataset_id"] = args.dataset_id
    if run_lock is not None:
        result["run_lock_sha256"] = run_lock.get("lock_sha256")
    result["evidence_sha256"] = stable_sha256(result)
    output = ROOT / "docs" / "runtime-evidence" / "h6" / args.dataset_id / "final-delivery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": result["passed"],
                "evidence": str(output),
                "evidence_sha256": result["evidence_sha256"],
                "failures": result["failures"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
