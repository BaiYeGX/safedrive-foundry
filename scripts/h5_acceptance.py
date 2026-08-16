#!/usr/bin/env python3
"""H5 final acceptance: protocol integrity, safety, progress, chattering, resource."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h5.config import H5_CONFIG, H5_CONFIG_SHA256  # noqa: E402
from data_pipeline.h5.matrix import h5_matrix_sha256, load_h5_matrix  # noqa: E402
from data_pipeline.h5.metrics import evaluate_gate, summarize_run  # noqa: E402
from data_pipeline.h5.store import H5Store  # noqa: E402

DATA_ROOT = ROOT / "generated" / "h5"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h5"


def _integrity_checks(runs: list[dict]) -> list[str]:
    failures = []
    by_pair: dict[str, dict[str, dict]] = {}
    for run in runs:
        by_pair.setdefault(run["pair_id"], {})[run["arm"]] = run
    for pair_id in sorted(by_pair):
        arms = by_pair[pair_id]
        for arm, run in arms.items():
            if run.get("config_sha256") != H5_CONFIG_SHA256:
                failures.append(f"config_sha_mismatch:{pair_id}:{arm}")
        if set(arms) != {"off", "on", "defer"}:
            failures.append(f"missing_arms:{pair_id}:{sorted(arms)}")
            continue
        off = arms["off"]
        on = arms["on"]
        defer = arms["defer"]
        if not (off["ok"] and on["ok"] and defer["ok"]):
            failures.append(f"run_not_ok:{pair_id}")
        if not (off["cleanup_complete"] and on["cleanup_complete"] and defer["cleanup_complete"]):
            failures.append(f"cleanup_incomplete:{pair_id}")
        # Defer arm must never rank with World; it always falls back.
        for decision in defer.get("decisions", []):
            if decision.get("routing", {}).get("world") == "RANKED":
                failures.append(f"defer_ranked_violation:{pair_id}")
                break
        # Reset comparisons: exactly one reference arm may lack comparison;
        # all other arms must be comparable to that reference.
        reference_arms = [arm for arm in ("off", "on", "defer") if arms[arm].get("reset_comparison") is None]
        if len(reference_arms) != 1:
            failures.append(f"reset_reference_count:{pair_id}:{len(reference_arms)}")
        for arm in ("off", "on", "defer"):
            rc = arms[arm].get("reset_comparison")
            if rc is not None and not rc.get("comparable", False):
                failures.append(f"reset_not_comparable:{pair_id}:{arm}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    parser.add_argument("--map", default=None)
    args = parser.parse_args()
    dataset_id = args.dataset_id
    store = H5Store(DATA_ROOT, dataset_id)
    runs = store.list_runs()
    if args.map:
        runs = [r for r in runs if r.get("scenario", {}).get("map_name") == args.map]
    expected_all = load_h5_matrix(ROOT, full=(args.scope == "full"))
    if args.map:
        expected = [s for s in expected_all if s.scenario.map_name == args.map]
    else:
        expected = list(expected_all)
    expected_combos = {(s.pair_id, arm) for s in expected for arm in s.arm_order}
    actual_combos = {(r["pair_id"], r["arm"]) for r in runs}
    missing = sorted(expected_combos - actual_combos)
    extra = sorted(actual_combos - expected_combos)
    integrity_failures = _integrity_checks(runs)
    if missing:
        integrity_failures.append(f"missing_runs:{missing[:20]}")
    if extra:
        integrity_failures.append(f"extra_runs:{extra[:20]}")
    gate = evaluate_gate(runs)
    if args.integrity_only:
        checks = {
            "integrity": not integrity_failures,
            "protocol_all_ok": gate["checks"]["protocol_all_ok"],
        }
        failures = integrity_failures + ([] if checks["protocol_all_ok"] else ["protocol_not_all_ok"])
    else:
        checks = {
            "integrity": not integrity_failures,
            "protocol_all_ok": gate["checks"]["protocol_all_ok"],
            "safety_noninferior": gate["checks"]["safety_noninferior"],
            "progress_net_benefit": gate["checks"]["progress_net_benefit"],
            "chattering_noninferior": gate["checks"]["chattering_noninferior"],
            "resource": gate["checks"]["resource"],
        }
        failures = integrity_failures + gate["failures"]
    passed = not failures
    payload = {
        "schema_version": "safedrive.h5.acceptance.v1",
        "dataset_id": dataset_id,
        "h5_config_sha256": H5_CONFIG_SHA256,
        "expected_scenarios": len(expected),
        "expected_runs": len(expected_combos),
        "actual_runs": len(runs),
        "checks": checks,
        "failures": failures,
        "gate": {
            "passed": passed,
            "failures": failures,
        },
        "progress": gate["progress"],
        "safety": gate["safety"],
        "chattering": gate["chattering"],
        "resource": gate["resource"],
        "matrix_sha256": h5_matrix_sha256(expected),
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    out = EVIDENCE_ROOT / dataset_id / "final-delivery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": passed, "gate_passed": passed, "evidence": str(out), "evidence_sha256": payload["evidence_sha256"], "failures": failures}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
