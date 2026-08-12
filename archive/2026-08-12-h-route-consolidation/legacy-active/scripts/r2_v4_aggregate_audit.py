#!/usr/bin/env python3
"""Aggregate the frozen 252-pair R2 V4 blind audit and apply its gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _walk(root: Path) -> list[dict[str, Any]]:
    return [_read(path) for path in sorted(root.rglob("pair_report.json"))]


def _fatal(report: dict[str, Any]) -> bool:
    if bool(report.get("fatal")):
        return True
    for branch in (report.get("branches") or {}).values():
        metrics = dict(branch.get("metrics") or {})
        if (
            int(metrics.get("collision_episode_count", branch.get("collision_episodes", 0))) > 0
            or float(metrics.get("offroad_fraction", 0.0)) >= 0.02
            or not bool(metrics.get("completed_primary_horizon", False))
        ):
            return True
    return False


def _guard_mpc_failure(report: dict[str, Any]) -> bool:
    """Count every executable Guard/MPC failure for map/family/maneuver caps."""
    if bool(report.get("guard_mpc_failure")):
        return True
    if bool(report.get("candidate1_available")) and not bool(report.get("comparable")):
        return True
    for branch in (report.get("branches") or {}).values():
        metrics = dict(branch.get("metrics") or {})
        if int(metrics.get("mpc_timeout_count", 0) or 0) > 0:
            return True
        if int(metrics.get("mpc_fallback_count", 0) or 0) > 0:
            return True
        if str(branch.get("guard_status") or "OK").upper() not in {"OK", ""}:
            return True
    return False


def aggregate(manifest: Path, evidence_roots: list[Path], *, expected_namespace: str = "r2v4_blind_audit") -> dict[str, Any]:
    frozen = _read(manifest)
    stored = str(frozen.get("manifest_hash") or "")
    body = dict(frozen)
    body.pop("manifest_hash", None)
    from hashlib import sha256

    actual = sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if stored != actual:
        raise ValueError("audit manifest hash mismatch")
    expected = {
        (str(row["scenario_id"]), str(row["seed_id"])): row
        for row in frozen.get("pairs", [])
    }
    reports = [report for root in evidence_roots for report in _walk(root)]
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        if str(report.get("namespace") or "") != expected_namespace:
            raise ValueError(f"unexpected audit namespace: {report.get('namespace')}")
        key = (str(report.get("scenario_id") or ""), str(report.get("seed_id") or ""))
        if key in keyed:
            raise ValueError(f"duplicate audit pair: {key}")
        keyed[key] = report
    if set(keyed) != set(expected):
        raise ValueError(
            f"audit coverage mismatch missing={len(set(expected)-set(keyed))} extra={len(set(keyed)-set(expected))}"
        )
    checkpoint_hashes = {str(row.get("checkpoint_sha256") or "") for row in keyed.values()}
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("audit does not bind one checkpoint")
    manifest_hashes = {str(row.get("source_manifest_hash") or "") for row in keyed.values()}
    if manifest_hashes != {stored}:
        raise ValueError("audit report source manifest binding mismatch")

    rows = list(keyed.values())
    family_counts = Counter(str(row.get("family") or expected[key]["family"]) for key, row in keyed.items())
    family_success = Counter(
        str(row.get("family") or expected[key]["family"])
        for key, row in keyed.items()
        if bool(row.get("strict_success"))
    )
    wins = Counter(str(row.get("winner")) for row in rows if row.get("winner") in {0, 1, "0", "1"})
    map_failure = Counter()
    family_failure = Counter()
    maneuver_failure = Counter()
    for key, row in keyed.items():
        if _guard_mpc_failure(row):
            expected_row = expected[key]
            map_failure[str(row.get("map_name") or expected_row["map_name"])] += 1
            family_failure[str(row.get("family") or expected_row["family"])] += 1
            maneuver_failure[str(expected_row["route_maneuver"])] += 1
    safe = sum(bool(row.get("safe_candidate_exists")) for row in rows)
    both_bad = sum(bool(row.get("both_bad")) for row in rows)
    comparable = sum(bool(row.get("comparable")) for row in rows)
    decisive = sum(bool(row.get("decisive")) for row in rows)
    fatal = sum(_fatal(row) for row in rows)
    candidate1_available = sum(bool(row.get("candidate1_available")) for row in rows)
    strict_success = sum(bool(row.get("strict_success")) for row in rows)
    rescue = sum(int(row.get("semantic_rescue_count", 0)) for row in rows)
    max_failure = max(
        [
            *(value / max(sum(1 for key in keyed if (expected[key]["map_name"] == name)), 1) for name, value in map_failure.items()),
            *(value / max(family_counts[name], 1) for name, value in family_failure.items()),
            *(value / max(sum(1 for key in keyed if expected[key]["route_maneuver"] == name), 1) for name, value in maneuver_failure.items()),
        ],
        default=0.0,
    )
    gates = {
        "fatal_zero": fatal == 0,
        "strict_success_min": strict_success >= 237,
        "family_min": all(family_success[name] >= 39 for name in family_counts),
        "candidate1_available_min": candidate1_available >= 150,
        "comparable_min": comparable >= 144,
        "decisive_min": decisive >= 54,
        "wins_balanced": wins["0"] >= 18 and wins["1"] >= 18,
        "safe_candidate_min": safe >= math.ceil(252 * 0.95),
        "both_bad_max": both_bad < 252 * 0.10,
        "guard_mpc_failure_max": max_failure <= 0.20,
        "semantic_rescue_zero": rescue == 0,
    }
    report = {
        "schema_version": "safedrive.r2_v4.blind_audit_report.v1",
        "namespace": expected_namespace,
        "source_manifest_hash": stored,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "pairs": len(rows),
        "fatal": fatal,
        "strict_success": strict_success,
        "candidate1_available": candidate1_available,
        "comparable": comparable,
        "decisive": decisive,
        "wins": {"0": wins["0"], "1": wins["1"]},
        "safe_candidate_exists": safe,
        "both_bad": both_bad,
        "semantic_rescue_count": rescue,
        "family_success": dict(family_success),
        "family_counts": dict(family_counts),
        "guard_mpc_failure_rates": {
            "map": dict(map_failure),
            "family": dict(family_failure),
            "maneuver": dict(maneuver_failure),
            "max_rate": max_failure,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", default="r2v4_blind_audit")
    args = parser.parse_args()
    report = aggregate(Path(args.manifest), [Path(value) for value in args.evidence_root], expected_namespace=args.namespace)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
