#!/usr/bin/env python3
"""Merge immutable V3 pair/long evidence and apply the frozen formal gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r2_world_ready_v3 import (  # noqa: E402
    evaluate_core_blind_v3,
    evaluate_world_ready_audit_v3,
)
from driving_vla.model.navigation_contract import canonical_sha256  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _read_frozen(path: Path) -> dict[str, Any]:
    value = _read(path)
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if stored != canonical_sha256(body):
        raise ValueError(f"{path}: manifest hash mismatch")
    return value


def _index_unique(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        fixture_id = str(row["fixture_id"])
        if fixture_id in result:
            raise ValueError(f"{label}: duplicate fixture {fixture_id}")
        result[fixture_id] = row
    return result


def _collect(
    paths: list[str],
    report_name: str,
    *,
    phase: str,
    source_manifest_hash: str,
) -> list[dict[str, Any]]:
    rows = []
    for raw_path in paths:
        report = _read(Path(raw_path))
        if (
            str(report.get("phase") or "") != phase
            or str(report.get("source_manifest_hash") or "")
            != source_manifest_hash
        ):
            raise ValueError(
                f"{raw_path}: {report_name} frozen manifest binding mismatch"
            )
        source_rows = report.get("rows")
        if not isinstance(source_rows, list):
            raise ValueError(f"{raw_path}: {report_name} rows missing")
        rows.extend(dict(row) for row in source_rows)
    return rows


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair-report", action="append", default=[])
    parser.add_argument("--long-report", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = _read_frozen(Path(args.manifest))
    phase = str(manifest["phase"])
    expected = {
        str(case["fixture_id"]): dict(case)
        for case in manifest.get("cases") or ()
    }
    long_rows = _index_unique(
        _collect(
            args.long_report,
            "long",
            phase=phase,
            source_manifest_hash=str(manifest["manifest_hash"]),
        ),
        label="long",
    )
    if set(long_rows) != set(expected):
        raise ValueError(
            "long fixture coverage mismatch: "
            f"missing={sorted(set(expected) - set(long_rows))} "
            f"extra={sorted(set(long_rows) - set(expected))}"
        )
    checkpoint_hashes = {
        str(row.get("checkpoint_sha256") or "")
        for row in long_rows.values()
    }
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("long reports do not bind one checkpoint")
    checkpoint_hash = next(iter(checkpoint_hashes))

    if phase == "unseen_long_audit":
        fatal = sum(
            bool(row.get(key))
            for row in long_rows.values()
            for key in ("collision", "offroad", "wrong_exit")
        )
        completed = sum(
            bool(row.get("completed")) for row in long_rows.values()
        )
        report = {
            "schema_version": "safedrive.r2_v3.formal_gate.v1",
            "phase": phase,
            "source_manifest_hash": manifest["manifest_hash"],
            "checkpoint_sha256": checkpoint_hash,
            "completed": completed,
            "expected": len(expected),
            "fatal_events": fatal,
            "passed": completed == len(expected) and fatal == 0,
            "rows": [
                long_rows[str(case["fixture_id"])]
                for case in manifest["cases"]
            ],
        }
        _write_exclusive(Path(args.out), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2

    if not args.pair_report:
        raise ValueError(f"{phase} requires --pair-report")
    pair_rows = _index_unique(
        _collect(
            args.pair_report,
            "pair",
            phase=phase,
            source_manifest_hash=str(manifest["manifest_hash"]),
        ),
        label="pair",
    )
    if set(pair_rows) != set(expected):
        raise ValueError("pair fixture coverage mismatch")
    pair_checkpoint_hashes = {
        str(row.get("checkpoint_sha256") or "")
        for row in pair_rows.values()
    }
    if pair_checkpoint_hashes != {checkpoint_hash}:
        raise ValueError("pair/long checkpoint binding mismatch")

    records = []
    for case in manifest["cases"]:
        fixture_id = str(case["fixture_id"])
        pair = pair_rows[fixture_id]
        long = long_rows[fixture_id]
        route_completion = dict(long.get("route_completion") or {})
        records.append(
            {
                **case,
                "comparable": bool(pair.get("comparable")),
                "decisive": bool(pair.get("decisive")),
                "winner": pair.get("winner"),
                "pair_label": str(pair.get("pair_label") or ""),
                "candidate1_available": bool(
                    pair.get("candidate1_available")
                ),
                "safe_candidate_exists": bool(
                    pair.get("safe_candidate_exists")
                ),
                "both_bad": bool(pair.get("both_bad")),
                "guard_mpc_failure": bool(
                    pair.get("guard_mpc_failure")
                    or pair.get("status") == "FAILED"
                ),
                "route_completed": bool(
                    route_completion.get("completed")
                ),
                "long_completed": bool(long.get("completed")),
                "collision": bool(long.get("collision")),
                "offroad": bool(long.get("offroad")),
                "wrong_exit": bool(long.get("wrong_exit")),
                "pair_evidence_dir": pair.get("attempt_dir")
                or pair.get("evidence_dir"),
                "long_evidence_dir": long.get("evidence_dir"),
            }
        )
    if phase == "core_blind":
        gate = evaluate_core_blind_v3(records)
    elif phase == "world_ready_audit":
        gate = evaluate_world_ready_audit_v3(records)
    else:
        raise ValueError(f"unsupported formal phase {phase}")
    # Pair rankings alone are insufficient.  Every selected learned policy
    # must complete both its route and its registered interaction.
    gate["gates"]["long_completion"] = all(
        bool(record["long_completed"]) for record in records
    )
    gate["passed"] = all(gate["gates"].values())
    report = {
        **gate,
        "schema_version": "safedrive.r2_v3.formal_gate.v1",
        "phase": phase,
        "source_manifest_hash": manifest["manifest_hash"],
        "checkpoint_sha256": checkpoint_hash,
        "records": records,
    }
    _write_exclusive(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
