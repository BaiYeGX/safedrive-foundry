#!/usr/bin/env python3
"""Apply the 12-pair R2 V4 core blind gate to frozen pair reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate(manifest: Path, evidence_roots: list[Path]) -> dict[str, Any]:
    frozen = _read(manifest)
    expected = {(str(row["scenario_id"]), str(row["seed_id"])): row for row in frozen.get("pairs", [])}
    reports = []
    for root in evidence_roots:
        reports.extend(_read(path) for path in root.rglob("pair_report.json"))
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        if str(report.get("namespace") or "") not in {"r2v4_core_blind", "r2v4_blind_audit", "r2_v4_core"}:
            raise ValueError("unexpected core report namespace")
        key = (str(report.get("scenario_id") or ""), str(report.get("seed_id") or ""))
        if key in keyed:
            raise ValueError(f"duplicate core pair: {key}")
        keyed[key] = report
    if set(keyed) != set(expected):
        raise ValueError("core pair coverage mismatch")
    checkpoint_hashes = {str(row.get("checkpoint_sha256") or "") for row in keyed.values()}
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("core report must bind one checkpoint")
    manifest_hashes = {str(row.get("source_manifest_hash") or "") for row in keyed.values()}
    if manifest_hashes != {str(frozen.get("manifest_hash") or "")}:
        raise ValueError("core report source manifest binding mismatch")
    comparable = sum(bool(row.get("comparable")) for row in keyed.values())
    decisive = sum(bool(row.get("decisive")) for row in keyed.values())
    fatal = sum(bool(row.get("fatal")) for row in keyed.values())
    wins = Counter(str(row.get("winner")) for row in keyed.values() if row.get("winner") in {0, 1, "0", "1"})
    families = {str(row["family"]) for row in expected.values()}
    candidate1_families = {
        str(expected[key]["family"])
        for key, row in keyed.items()
        if str(row.get("winner")) == "1"
    }
    gates = {
        "comparable_min": comparable >= 10,
        "decisive_min": decisive >= 4,
        "wins_each_min": wins["0"] >= 2 and wins["1"] >= 2,
        "candidate1_cross_family": len(candidate1_families) >= 2,
        "fatal_zero": fatal == 0,
    }
    return {
        "schema_version": "safedrive.r2_v4.core_blind_gate.v1",
        "source_manifest_hash": str(frozen.get("manifest_hash") or ""),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "pairs": len(keyed),
        "comparable": comparable,
        "decisive": decisive,
        "wins": {"0": wins["0"], "1": wins["1"]},
        "candidate1_families": sorted(candidate1_families),
        "families": sorted(families),
        "fatal": fatal,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(Path(args.manifest), [Path(value) for value in args.evidence_root])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
