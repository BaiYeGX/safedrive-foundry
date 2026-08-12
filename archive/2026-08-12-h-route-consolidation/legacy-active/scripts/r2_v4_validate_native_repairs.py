#!/usr/bin/env python3
"""Validate the pre-collection native maneuver repair gate for R2 V4.

The V4 campaign is not allowed to start merely because a route-change or
junction case passed once.  This validator binds two distinct repair reports
to the exact maneuver names, requires three consecutive clean passes for each,
and requires the independent teacher long suite to be 16/16.  It never runs
CARLA and it never rewrites a failed report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_CASES = {
    "ROUTE_CHANGE_RIGHT": "route_change_right",
    "LEFT_TURN_CROSSING": "left_turn_crossing_yield",
}
SCHEMA = "safedrive.r2.v4.native_repair_gate.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_name(value: Mapping[str, Any]) -> str:
    raw = str(value.get("case_id") or value.get("scenario_id") or value.get("maneuver") or "")
    normalized = raw.upper().replace("-", "_").replace(" ", "_")
    if "ROUTE_CHANGE_RIGHT" in normalized:
        return "ROUTE_CHANGE_RIGHT"
    if "LEFT_TURN_CROSSING" in normalized or "CROSSING_YIELD" in normalized:
        return "LEFT_TURN_CROSSING"
    return normalized


def _run_pass(row: Mapping[str, Any]) -> bool:
    """Accept only an explicitly clean run; unknown reports fail closed."""
    if "passed" in row:
        passed = bool(row["passed"])
    elif "completed" in row:
        passed = bool(row["completed"])
    else:
        passed = False
    if not passed:
        return False
    fatal = bool(row.get("fatal", False))
    if int(row.get("collision", row.get("collision_count", row.get("collision_episodes", 0))) or 0) != 0:
        fatal = True
    if float(row.get("offroad_fraction", 0.0) or 0.0) > 0.02:
        fatal = True
    failure = str(row.get("failure_code") or row.get("failure") or "").upper()
    if any(token in failure for token in ("WRONG_EXIT", "PERMANENT_STALL", "COLLISION", "OFFROAD")):
        fatal = True
    return not fatal


def _consecutive_passes(report: Mapping[str, Any]) -> tuple[int, list[bool]]:
    raw_runs = report.get("runs") or report.get("attempts") or report.get("repairs") or ()
    if isinstance(raw_runs, Mapping):
        raw_runs = list(raw_runs.values())
    statuses = [_run_pass(row) for row in raw_runs if isinstance(row, Mapping)]
    declared = report.get("consecutive_passes")
    if declared is not None:
        try:
            count = int(declared)
        except (TypeError, ValueError):
            count = 0
    else:
        count = 0
        for value in reversed(statuses):
            if not value:
                break
            count += 1
    return count, statuses


def validate(repair_reports: Sequence[Path], teacher_report: Path) -> dict[str, Any]:
    if len(repair_reports) != 2:
        raise ValueError("native repair gate requires exactly two repair reports")
    reports = [_read(path) for path in repair_reports]
    names = [_case_name(report) for report in reports]
    expected = set(EXPECTED_CASES)
    if set(names) != expected:
        raise ValueError(f"repair reports must cover {sorted(expected)}, got {sorted(names)}")
    if len(set(names)) != len(names):
        raise ValueError("repair reports must be for distinct maneuvers")
    cases: dict[str, Any] = {}
    for path, report, name in zip(repair_reports, reports, names):
        count, statuses = _consecutive_passes(report)
        cases[name] = {
            "report": str(path),
            "report_sha256": _sha(path),
            "expected_scenario_id": EXPECTED_CASES[name],
            "consecutive_passes": count,
            "run_statuses": statuses,
            "gate": count >= 3 and all(statuses[-3:]) if len(statuses) >= 3 else False,
        }
    teacher = _read(teacher_report)
    teacher_expected = int(teacher.get("expected_cases", teacher.get("cases_expected", 16)) or 0)
    teacher_completed = int(teacher.get("completed", teacher.get("cases_passed", 0)) or 0)
    teacher_failed = int(teacher.get("failed", teacher.get("cases_failed", 0)) or 0)
    teacher_gate = bool(
        teacher_expected == 16
        and teacher_completed == 16
        and teacher_failed == 0
        and bool(teacher.get("passed", teacher.get("all_pass", True)))
    )
    gates = {
        "route_change_right_three_consecutive": bool(cases["ROUTE_CHANGE_RIGHT"]["gate"]),
        "left_turn_crossing_three_consecutive": bool(cases["LEFT_TURN_CROSSING"]["gate"]),
        "teacher_long_suite_16_of_16": teacher_gate,
    }
    return {
        "schema_version": SCHEMA,
        "repair_reports": [str(path) for path in repair_reports],
        "teacher_report": str(teacher_report),
        "teacher_report_sha256": _sha(teacher_report),
        "cases": cases,
        "teacher": {
            "expected": teacher_expected,
            "completed": teacher_completed,
            "failed": teacher_failed,
            "gate": teacher_gate,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-report", action="append", required=True)
    parser.add_argument("--teacher-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate(
        [Path(value).resolve() for value in args.repair_report],
        Path(args.teacher_report).resolve(),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

