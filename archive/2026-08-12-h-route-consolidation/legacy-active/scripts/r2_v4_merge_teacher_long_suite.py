#!/usr/bin/env python3
"""Merge independently rerun teacher cases without hiding failed attempts.

The native gate consumes one 16-case report, while CARLA recovery may produce
separate one-case reports.  This utility chooses a clean measured row for each
case, keeps every superseded attempt and its hash in ``attempt_history``, and
fails closed on missing/duplicate case identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or row.get("scenario_id") or "").strip()


def _clean(row: Mapping[str, Any]) -> bool:
    reasons = row.get("reason_codes") or []
    route = row.get("route_completion") or {}
    interaction = row.get("interaction_completion") or {}
    return bool(
        row.get("completed", False)
        and not reasons
        and not route.get("collision", False)
        and not route.get("offroad", False)
        and not interaction.get("collision", False)
        and not interaction.get("offroad", False)
    )


def _rows(report: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: rows must be a list of objects")
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--replacement", action="append", default=[], help="CASE_ID=campaign_report.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_report).resolve()
    base = _read(base_path)
    base_rows = _rows(base, base_path)
    if len(base_rows) != 16:
        raise ValueError(f"base teacher suite must contain 16 rows, got {len(base_rows)}")
    primary: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for row in base_rows:
        case = _case_id(row)
        if not case or case in primary:
            raise ValueError(f"base report has duplicate/empty case identity: {case!r}")
        primary[case] = row
    history.append({"path": str(base_path), "sha256": _sha(base_path), "rows": base_rows})

    replacements: dict[str, Path] = {}
    for raw in args.replacement:
        if "=" not in raw:
            raise ValueError(f"replacement must be CASE_ID=PATH: {raw}")
        case, raw_path = raw.split("=", 1)
        case = case.strip()
        path = Path(raw_path).resolve()
        if not case or case in replacements:
            raise ValueError(f"duplicate replacement case: {case!r}")
        report = _read(path)
        rows = _rows(report, path)
        if len(rows) != 1 or _case_id(rows[0]) != case:
            raise ValueError(f"{path}: expected one row for {case!r}")
        if not _clean(rows[0]):
            raise ValueError(f"{path}: replacement {case!r} is not clean")
        replacements[case] = path
        history.append({"path": str(path), "sha256": _sha(path), "rows": rows})
        if case not in primary:
            raise ValueError(f"replacement case not present in base suite: {case!r}")
        primary[case] = dict(rows[0])

    ordered = [primary[case] for case in sorted(primary)]
    completed = sum(1 for row in ordered if _clean(row))
    failed = len(ordered) - completed
    report = {
        "schema_version": "safedrive.r2.v4.teacher_long_suite.v1",
        "campaign_id": "r2-v4-native-repair-teacher-long-suite-v1",
        "expected_cases": 16,
        "completed": completed,
        "failed": failed,
        "passed": bool(len(ordered) == 16 and completed == 16 and failed == 0),
        "rows": ordered,
        "replaced_cases": sorted(replacements),
        "attempt_history": history,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("expected_cases", "completed", "failed", "passed", "replaced_cases")}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
