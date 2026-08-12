#!/usr/bin/env python3
"""Apply the pre-frozen 512-slot R3 train/val development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "safedrive.r3.development_gate.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reports(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        for path in sorted(root.rglob("pair_report.json")):
            rows.append((path, _read(path)))
    return rows


def validate(campaign_path: Path, evidence_roots: list[Path]) -> dict[str, Any]:
    campaign = _read(campaign_path)
    slots = list(campaign.get("development_slots") or [])
    if len(slots) != 512:
        raise ValueError(f"campaign development_slots must contain 512 slots, got {len(slots)}")
    expected = {str(row["scenario_id"]): row for row in slots}
    if any(str(row.get("split")) == "test" for row in slots):
        raise ValueError("R3 development gate cannot contain Town13/test slots")
    reports = _reports(evidence_roots)
    keyed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, report in reports:
        if str(report.get("namespace") or "") != "r3_final_head_formal":
            continue
        scenario_id = str(report.get("scenario_id") or "")
        if scenario_id not in expected:
            continue
        if scenario_id in keyed:
            raise ValueError(f"duplicate R3 development slot: {scenario_id}")
        if str(report.get("source_manifest_hash") or "") not in {
            str(campaign.get("manifest_hash") or "")
        }:
            raise ValueError(f"development report manifest binding mismatch: {scenario_id}")
        keyed[scenario_id] = (path, report)
    if set(keyed) != set(expected):
        raise ValueError(
            f"development coverage mismatch missing={len(set(expected)-set(keyed))} "
            f"extra={len(set(keyed)-set(expected))}"
        )
    fatal = sum(bool(report.get("fatal")) for _path, report in keyed.values())
    comparable = sum(bool(report.get("comparable")) for _path, report in keyed.values())
    decisive = sum(bool(report.get("decisive")) for _path, report in keyed.values())
    wins = Counter(
        str(report.get("winner"))
        for _path, report in keyed.values()
        if report.get("winner") in {0, 1, "0", "1"}
    )
    status = "R3_DEVELOPMENT_GATE_PASS" if (
        len(keyed) == 512
        and fatal == 0
        and comparable >= 256
        and decisive >= 64
        and wins["0"] >= 16
        and wins["1"] >= 16
    ) else "R3_DATA_LIMITED"
    gates = {
        "completed_exact_512": len(keyed) == 512,
        "fatal_zero": fatal == 0,
        "comparable_min_256": comparable >= 256,
        "decisive_min_64": decisive >= 64,
        "candidate_0_wins_min_16": wins["0"] >= 16,
        "candidate_1_wins_min_16": wins["1"] >= 16,
    }
    return {
        "schema_version": SCHEMA,
        "status": status,
        "campaign_manifest": str(campaign_path),
        "campaign_manifest_sha256": _sha(campaign_path),
        "source_manifest_hash": str(campaign.get("manifest_hash") or ""),
        "completed": len(keyed),
        "fatal": fatal,
        "comparable": comparable,
        "decisive": decisive,
        "wins": {"0": wins["0"], "1": wins["1"]},
        "gates": gates,
        "passed": all(gates.values()),
        "evidence_report_sha256": {
            scenario_id: _sha(path) for scenario_id, (path, _report) in sorted(keyed.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--evidence-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate(
        Path(args.campaign_manifest).resolve(),
        [Path(value).resolve() for value in args.evidence_root],
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

