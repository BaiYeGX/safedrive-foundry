#!/usr/bin/env python3
"""Apply the exact learned-only R2 V4 smoke gates to immutable reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate(paths: list[Path], *, expected_cases: int) -> dict[str, Any]:
    if len(paths) != 1:
        raise ValueError("provide one consolidated learned-only smoke report")
    report = _read(paths[0])
    checkpoint = str(report.get("checkpoint_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
        raise ValueError("learned-only smoke must bind a 64-hex checkpoint SHA256")
    completed = int(report.get("completed", report.get("cases_passed", 0)))
    rows = list(report.get("rows") or [])
    collision = int(report.get("collision", report.get("collisions", 0)))
    offroad = int(report.get("offroad", report.get("offroad_count", 0)))
    wrong_exit = int(report.get("wrong_exit", report.get("wrong_exit_count", 0)))
    stall = int(report.get("permanent_stall", report.get("stall_count", 0)))
    if rows:
        identities = [
            (str(row.get("scenario_id") or ""), str(row.get("seed_id") or ""))
            for row in rows
        ]
        if len(identities) != expected_cases or len(set(identities)) != len(identities):
            raise ValueError("learned-only smoke rows have duplicate or incomplete identities")
        row_hashes = {str(row.get("checkpoint_sha256") or checkpoint).lower() for row in rows}
        if row_hashes != {checkpoint}:
            raise ValueError("learned-only smoke rows do not bind one checkpoint")
        completed = sum(bool(row.get("completed", row.get("passed", False))) for row in rows)
        collision = sum(bool(row.get("collision")) for row in rows)
        offroad = sum(bool(row.get("offroad")) for row in rows)
        wrong_exit = sum(bool(row.get("wrong_exit")) for row in rows)
        stall = sum(bool(row.get("permanent_stall")) for row in rows)
    gates = {
        "case_count": completed == expected_cases,
        "collision_zero": collision == 0,
        "offroad_zero": offroad == 0,
        "wrong_exit_zero": wrong_exit == 0,
        "permanent_stall_zero": stall == 0,
    }
    return {
        "schema_version": "safedrive.r2_v4.learned_smoke_gate.v1",
        "source_report": str(paths[0]),
        "expected_cases": expected_cases,
        "completed": completed,
        "collision": collision,
        "offroad": offroad,
        "wrong_exit": wrong_exit,
        "permanent_stall": stall,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-cases", type=int, choices=(16, 32), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate([Path(args.report)], expected_cases=int(args.expected_cases))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
