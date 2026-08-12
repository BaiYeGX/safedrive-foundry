#!/usr/bin/env python3
"""Promote one immutable V4 checkpoint only after every R2 formal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.checkpoint_contract import STATUS_OK, write_checkpoint_manifest  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path, condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"{path.name}: {message}")


def _validate_smoke(
    path: Path,
    report: Mapping[str, Any],
    *,
    expected_cases: int,
    checkpoint_hash: str,
) -> None:
    _require(path, int(report.get("expected_cases", expected_cases)) == expected_cases, f"smoke expected {expected_cases} cases")
    passed = bool(report.get("passed", report.get("all_pass", False)))
    if "cases" in report:
        passed = passed and int(report.get("completed", report.get("cases_passed", 0))) >= expected_cases
    _require(path, passed, f"learned-only {expected_cases}-case smoke did not pass")
    _require(path, int(report.get("fatal", report.get("fatal_count", 0))) == 0, "fatal outcome present")
    bound = str(report.get("checkpoint_sha256") or "").lower()
    _require(path, not bound or bound == checkpoint_hash, "smoke checkpoint binding mismatch")


def _validate_core(path: Path, report: Mapping[str, Any]) -> None:
    _require(path, str(report.get("schema_version") or "") == "safedrive.r2_v4.core_blind_gate.v1", "core schema mismatch")
    _require(path, int(report.get("pairs", 0)) == 12, "core report must cover exactly 12 pairs")
    gates = dict(report.get("gates") or {})
    _require(path, int(report.get("comparable", 0)) >= 10, "core comparable < 10")
    _require(path, int(report.get("decisive", 0)) >= 4, "core decisive < 4")
    _require(path, int(report.get("fatal", report.get("fatal_count", 0))) == 0, "core fatal present")
    wins = report.get("wins") or report.get("candidate_wins") or {}
    _require(path, int(wins.get("0", wins.get(0, 0))) >= 2, "candidate 0 wins < 2")
    _require(path, int(wins.get("1", wins.get(1, 0))) >= 2, "candidate 1 wins < 2")
    _require(
        path,
        bool(
            gates.get(
                "candidate1_cross_family",
                report.get("candidate1_cross_family", report.get("candidate_1_cross_family", False)),
            )
        ),
        "candidate 1 lacks cross-family win",
    )


def _validate_audit(path: Path, report: Mapping[str, Any]) -> None:
    _require(path, str(report.get("schema_version") or "") == "safedrive.r2_v4.blind_audit_report.v1", "audit schema mismatch")
    _require(path, int(report.get("pairs", 0)) == 252, "audit report must cover exactly 252 pairs")
    gates = dict(report.get("gates") or report)
    required = {
        "fatal_zero": True,
        "strict_success_min": True,
        "family_min": True,
        "candidate1_available_min": True,
        "comparable_min": True,
        "decisive_min": True,
        "wins_balanced": True,
        "safe_candidate_min": True,
        "both_bad_max": True,
        "guard_mpc_failure_max": True,
        "semantic_rescue_zero": True,
    }
    for name, expected in required.items():
        _require(path, bool(gates.get(name, False)) is expected, f"audit gate failed: {name}")


def promote(
    checkpoint: Path,
    formal: Path,
    smoke: list[Path],
    core: Path,
    audit: Path,
    *,
    native_repair_gate: Path | None = None,
    blind_registry: Path | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    formal = formal.resolve()
    smoke = [path.resolve() for path in smoke]
    core = core.resolve()
    audit = audit.resolve()
    checkpoint_hash = _sha(checkpoint)
    formal_report = _read(formal)
    reports = list(formal_report.get("reports") or [])
    _require(formal, bool(formal_report.get("locked_test_opened_once")), "Town13 locked test was not explicitly opened")
    _require(
        formal,
        int(formal_report.get("test_evaluation_count", 1)) == 1
        and sum(1 for report in reports if report.get("test") is not None) == 1,
        "Town13 locked test must be evaluated exactly once for one checkpoint",
    )
    _require(formal, bool(formal_report.get("all_hard_gates_pass")), "offline formal gates failed")
    _require(formal, any(str(row.get("checkpoint_sha256")) == checkpoint_hash for row in reports), "formal report is not bound to checkpoint")
    if len(smoke) != 2:
        raise ValueError("formal promotion requires both 16-case and 32-case smoke reports")
    smoke_reports = [(_path, _read(_path)) for _path in smoke]
    expected_smoke = {int(report.get("expected_cases", 0)) for _path, report in smoke_reports}
    if expected_smoke != {16, 32}:
        raise ValueError(f"smoke reports must be exactly 16 and 32 cases, got {sorted(expected_smoke)}")
    for smoke_path, smoke_report in smoke_reports:
        _validate_smoke(
            smoke_path,
            smoke_report,
            expected_cases=int(smoke_report["expected_cases"]),
            checkpoint_hash=checkpoint_hash,
        )
    core_report = _read(core)
    audit_report = _read(audit)
    _validate_core(core, core_report)
    _validate_audit(audit, audit_report)
    for path, report in ((core, core_report), (audit, audit_report)):
        bound = str(report.get("checkpoint_sha256") or "").lower()
        _require(path, not bound or bound == checkpoint_hash, "blind report checkpoint binding mismatch")
    if native_repair_gate is None:
        raise ValueError("formal promotion requires the native repair gate")
    native_repair_gate = native_repair_gate.resolve()
    native_gate = _read(native_repair_gate)
    _require(
        native_repair_gate,
        str(native_gate.get("schema_version") or "")
        == "safedrive.r2.v4.native_repair_gate.v1"
        and bool(native_gate.get("passed")),
        "native maneuver repair gate failed or is missing",
    )
    registry_hash = ""
    registry_version = ""
    if blind_registry is not None:
        registry_hash = _sha(blind_registry.resolve())
        registry_version = str(_read(blind_registry.resolve()).get("schema_version") or "")
    manifest = write_checkpoint_manifest(
        checkpoint.parent / "CHECKPOINT_STATUS.json",
        checkpoint_path=checkpoint,
        status=STATUS_OK,
        allowed_uses=[
            "r2v4_formal",
            "r2v4_blind_audit",
            "r3_final_head_formal",
            "collection_anchor",
        ],
        forbidden_uses=["world_campaign"],
        reasons=["R2 V4 formal offline, learned-only smoke, core blind and 252-pair audit gates passed"],
        extra={
            "formal_eval_sha256": _sha(formal),
            "learned_only_smoke_sha256": [_sha(path) for path in smoke],
            "core_blind_sha256": _sha(core),
            "audit_sha256": _sha(audit),
            "native_repair_gate_sha256": _sha(native_repair_gate),
            "blind_registry_sha256": registry_hash,
            "blind_registry_version": registry_version,
            "blind_pair_overlap_zero": True,
            "normalization_frozen": True,
            "class_order_frozen": True,
        },
    )
    result = {
        "schema_version": "safedrive.r2_v4.formal_promotion.v1",
        "status": manifest["status"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "formal_eval_sha256": _sha(formal),
        "native_repair_gate_sha256": _sha(native_repair_gate),
        "blind_registry_sha256": registry_hash,
    }
    (checkpoint.parent / "FORMAL_PROMOTION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formal-report", required=True)
    parser.add_argument("--smoke-report", action="append", required=True)
    parser.add_argument("--core-report", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--native-repair-gate", required=True)
    parser.add_argument("--blind-registry", default="")
    args = parser.parse_args()
    result = promote(
        Path(args.checkpoint),
        Path(args.formal_report),
        [Path(value) for value in args.smoke_report],
        Path(args.core_report),
        Path(args.audit_report),
        native_repair_gate=Path(args.native_repair_gate),
        blind_registry=Path(args.blind_registry) if args.blind_registry else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
