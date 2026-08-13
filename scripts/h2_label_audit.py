#!/usr/bin/env python3
"""Offline-only H2 Oracle labeling, permutation audit and fixed gate report."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import (  # noqa: E402
    OracleLabel,
    OracleVerdict,
    branch_outcome_from_dict,
    stable_sha256,
)
from data_pipeline.h2.config import H2_CONFIG_SHA256, config_identity  # noqa: E402
from data_pipeline.h2.oracle import ORACLE_VERSION, label_pair  # noqa: E402
from data_pipeline.h2.quality import audit_h2_gate  # noqa: E402
from data_pipeline.h2.store import PairedOutcomeStore  # noqa: E402


DATA_ROOT = ROOT / "generated" / "h2" / "paired-outcomes"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h2"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _offline_label(record: Mapping[str, Any]) -> tuple[OracleLabel, bool]:
    branches = [branch_outcome_from_dict(item) for item in record.get("branches", ())]
    if len(branches) != 2:
        return OracleLabel(
            OracleVerdict.INVALID_PAIR,
            None,
            None,
            "not_two_executed_branches",
            ORACLE_VERSION,
        ), True
    normal = label_pair(branches[0], branches[1])
    swapped = label_pair(branches[1], branches[0])
    invariant = normal == swapped
    return normal, invariant


def label_and_audit(dataset_id: str, scope: str) -> dict[str, Any]:
    store = PairedOutcomeStore(DATA_ROOT, dataset_id)
    records = list(store.iter_pair_dicts())
    if scope == "pilot":
        records = [
            row for row in records
            if int(row["scenario"]["seed"]) == 0 and row["scenario"]["weather"] == "ClearNoon"
        ]
    permutation_failures: list[str] = []
    for record in records:
        label, invariant = _offline_label(record)
        slot_mutated = copy.deepcopy(record)
        for candidate in slot_mutated.get("candidates", ()):
            candidate["slot"] = 1 - int(candidate["slot"])
        slot_label, _ = _offline_label(slot_mutated)
        source_mutated = copy.deepcopy(record)
        for candidate in source_mutated.get("candidates", ()):
            candidate["source"] = "vla" if candidate.get("source") == "expert" else "expert"
        source_label, _ = _offline_label(source_mutated)
        slot_invariant = label == slot_label
        source_invariant = label == source_label
        if not (invariant and slot_invariant and source_invariant):
            permutation_failures.append(str(record["pair_id"]))
        store.write_label(
            str(record["pair_id"]),
            {
                **label.to_dict(),
                "pair_id": record["pair_id"],
                "matrix_sha256": record["matrix_sha256"],
                "slot_swap_invariant": slot_invariant,
                "branch_order_invariant": invariant,
                "source_mutation_invariant": source_invariant,
            },
            update_manifest=False,
        )
    store.write_manifest()
    manifest_valid, manifest_reasons = store.verify_manifest()
    selected_ids = {str(row["pair_id"]) for row in records}
    labeled = [row for row in store.iter_labeled_pair_dicts() if str(row["pair_id"]) in selected_ids]
    whole_gpu_peak_gb = max(
        (float(branch.get("whole_gpu_peak_gb", 0.0)) for row in labeled for branch in row.get("branches", ())),
        default=0.0,
    )
    dataset_bytes = sum(path.stat().st_size for path in store.root.rglob("*") if path.is_file())
    gate = audit_h2_gate(
        labeled,
        scope=scope,
        manifest_valid=manifest_valid and not permutation_failures,
        dataset_bytes=dataset_bytes,
        whole_gpu_peak_gb=whole_gpu_peak_gb,
    )
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h2.offline_label_audit.v1",
        "dataset_id": dataset_id,
        "scope": scope,
        "oracle_version": ORACLE_VERSION,
        "config": config_identity(),
        "config_sha256": H2_CONFIG_SHA256,
        "records": len(records),
        "permutation_failures": permutation_failures,
        "manifest_valid": manifest_valid,
        "manifest_reasons": list(manifest_reasons),
        "gate": gate.to_dict(),
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    evidence = EVIDENCE_ROOT / dataset_id / f"offline-label-audit-{scope}.json"
    _atomic_json(evidence, payload)
    return {
        "ok": manifest_valid and not permutation_failures,
        "gate_passed": gate.passed,
        "failures": list(gate.failures),
        "evidence": str(evidence),
        "evidence_sha256": payload["evidence_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scope", required=True, choices=("pilot", "full"))
    args = parser.parse_args()
    try:
        result = label_and_audit(args.dataset_id, args.scope)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
