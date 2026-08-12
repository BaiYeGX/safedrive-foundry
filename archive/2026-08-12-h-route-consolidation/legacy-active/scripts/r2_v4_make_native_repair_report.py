#!/usr/bin/env python3
"""Bind independent long-smoke campaign reports into a native-repair report.

The report is deliberately lossless: each campaign row is copied into the
ordered run list and the three-pass count is computed from the tail.  Failed
attempts remain visible, so a later clean tail cannot be mistaken for a
single cherry-picked success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{path}: expected exactly one campaign row")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--campaign-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for raw in args.campaign_report:
        path = Path(raw).resolve()
        campaign = _read(path)
        row = dict(campaign["rows"][0])
        row["campaign_report"] = str(path)
        runs.append(row)
        sources.append({"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()})

    tail = 0
    for row in reversed(runs):
        if not bool(row.get("completed", False)):
            break
        reasons = row.get("reason_codes") or []
        if reasons:
            break
        route = row.get("route_completion") or {}
        if bool(route.get("collision")) or bool(route.get("offroad")):
            break
        tail += 1

    report = {
        "schema_version": "safedrive.r2.v4.native_repair_report.v1",
        "case_id": str(args.case),
        "runs": runs,
        "source_campaign_reports": sources,
        "consecutive_passes": tail,
        "passed": bool(tail >= 3),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
