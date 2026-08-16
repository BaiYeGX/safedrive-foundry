#!/usr/bin/env python3
"""Offline-label and audit H3v2 Challenge pairs, then run the data-quality gate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import OracleVerdict, branch_outcome_from_dict, stable_sha256  # noqa: E402
from data_pipeline.h2.oracle import ORACLE_VERSION, label_pair  # noqa: E402
from data_pipeline.h2.store import PairedOutcomeStore  # noqa: E402
from data_pipeline.h3.quality import audit_challenge_dataset, audit_offline_labels  # noqa: E402


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _offline_label(record: Mapping[str, Any]) -> tuple[Any, bool]:
    branches = [branch_outcome_from_dict(item) for item in record.get("branches", ())]
    if len(branches) != 2:
        return None, True
    normal = label_pair(branches[0], branches[1])
    swapped = label_pair(branches[1], branches[0])
    return normal, normal == swapped


def label_dataset(dataset_id: str, root: Path) -> tuple[int, list[str]]:
    store = PairedOutcomeStore(root.parent, root.name)
    permutation_failures = []
    for record in store.iter_pair_dicts():
        label, invariant = _offline_label(record)
        if label is None:
            continue
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
            permutation_failures.append(record["pair_id"])
        store.write_label(
            record["pair_id"],
            {
                **label.to_dict(),
                "pair_id": record["pair_id"],
                "matrix_sha256": record.get("matrix_sha256", ""),
                "slot_swap_invariant": slot_invariant,
                "branch_order_invariant": invariant,
                "source_mutation_invariant": source_invariant,
            },
            update_manifest=False,
        )
    store.write_manifest()
    return len(permutation_failures), permutation_failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--skip-label", action="store_true")
    args = parser.parse_args()
    root = Path(args.root) if args.root else (ROOT / "generated" / "h3" / "carla-challenge-v2" / args.dataset_id)
    if not args.skip_label:
        failures, rows = label_dataset(args.dataset_id, root)
    else:
        failures, rows = 0, []
    quality = audit_challenge_dataset(args.dataset_id, root=root)
    labels = audit_offline_labels(args.dataset_id, root=root)
    if failures:
        labels["failures"] = sorted(set(labels["failures"]) | set(rows))
        labels["passed"] = not labels["failures"]
    payload = {
        "dataset_id": args.dataset_id,
        "permutation_failures": rows,
        "quality": quality,
        "offline_labels": labels,
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    evidence = ROOT / "docs" / "runtime-evidence" / "h3" / args.dataset_id / "challenge-audit.json"
    _write(evidence, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if quality["passed"] and labels["passed"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
