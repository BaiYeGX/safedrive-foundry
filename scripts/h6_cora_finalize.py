#!/usr/bin/env python3
"""Freeze a terminal C2 delivery record without authorizing C3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import stable_sha256  # noqa: E402
from data_pipeline.h6.cora.config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256  # noqa: E402
from data_pipeline.h6.cora.matrix import CORA_MATRIX_SHA256  # noqa: E402
from data_pipeline.h6.cora.run_lock import verify_run_lock  # noqa: E402


FINAL_SCHEMA = "safedrive.cora.final_delivery.v1"
SUMMARY_SCHEMA = "safedrive.cora.collection_summary.v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_passed(payload: Mapping[str, Any], name: str) -> bool:
    value = payload.get(name)
    return isinstance(value, Mapping) and bool(value.get("passed"))


def finalize(dataset_id: str) -> dict[str, Any]:
    if dataset_id == "h6-cora-c2-repair-20260905-v2":
        from data_pipeline.h6.cora.repair import finalize_repair
        dataset = ROOT / "generated" / "h6" / "cora" / dataset_id
        evidence = ROOT / "docs" / "runtime-evidence" / "h6" / dataset_id
        return finalize_repair(dataset, evidence)
    if dataset_id != str(CORA_C2_CONFIG["dataset_id"]):
        raise ValueError("cora_finalize_dataset_not_frozen")
    dataset = ROOT / "generated" / "h6" / "cora" / dataset_id
    evidence = ROOT / "docs" / "runtime-evidence" / "h6" / dataset_id
    required = {
        "run_lock": dataset / "run-lock.json",
        "manifest": dataset / "manifest.json",
        "physical": dataset / "scenario-manifest.json",
        "pilot_quality": evidence / "data-quality-pilot.json",
        "development_quality": evidence / "data-quality-development.json",
        "execution_history": evidence / "execution-history.json",
        "runtime_release": evidence / "runtime-release.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"cora_finalize_missing:{missing}")
    lock = _read_json(required["run_lock"])
    lock_valid, lock_failures = verify_run_lock(lock)
    pilot = _read_json(required["pilot_quality"])
    development = _read_json(required["development_quality"])
    execution_history = _read_json(required["execution_history"])
    runtime_release = _read_json(required["runtime_release"])
    manifest = _read_json(required["manifest"])
    physical = _read_json(required["physical"])

    # The pilot audit performed the complete artifact verification, and the
    # development audit explicitly reused that passed verification at the
    # user's request.  Finalization therefore validates the frozen audit chain
    # and manifest identities without doing a third full-file hash scan.
    manifest_failures = list(pilot.get("manifest_failures", ())) + list(
        development.get("manifest_failures", ())
    )
    manifest_valid = bool(
        pilot.get("metrics", {}).get("manifest_valid")
        and development.get("metrics", {}).get("manifest_valid")
        and manifest.get("dataset_id") == dataset_id
        and int(manifest.get("root_count", -1)) == 351
        and physical.get("dataset_id") == dataset_id
        and physical.get("matrix_sha256") == CORA_MATRIX_SHA256
        and not bool(physical.get("formal_collected"))
        and not manifest_failures
    )
    collection_runs = [
        _read_json(path)
        for path in sorted(evidence.glob("collect-*.json"))
        if "failure" not in path.name
    ]
    collection_failures = [
        _read_json(path)
        for path in sorted(evidence.glob("collect-*-failure-*.json"))
    ]
    pilot_metrics = dict(pilot.get("metrics", {}))
    development_metrics = dict(development.get("metrics", {}))
    resource_audit = dict(development.get("resource_audit", {}))
    total_roots = int(pilot_metrics.get("root_attempts", 0)) + int(
        development_metrics.get("root_attempts", 0)
    )
    total_branches = int(pilot_metrics.get("branch_attempts", 0)) + int(
        development_metrics.get("branch_attempts", 0)
    )
    aggregate_wall_s = float(resource_audit.get("aggregate_collector_wall_s", 0.0))
    whole_gpu_peak_gib = float(resource_audit.get("whole_gpu_peak_gib", 0.0))
    minimum_free_disk_gib = float(resource_audit.get("minimum_observed_free_disk_gib", 0.0))
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "dataset_id": dataset_id,
        "terminal_root_attempts": total_roots,
        "pilot_roots": int(pilot_metrics.get("root_attempts", 0)),
        "development_roots": int(development_metrics.get("root_attempts", 0)),
        "branch_attempts": total_branches,
        "valid_nominal_pairs": int(pilot_metrics.get("valid_nominal_pairs", 0))
        + int(development_metrics.get("valid_nominal_pairs", 0)),
        "development_valid_nominal_pairs": int(
            development_metrics.get("valid_nominal_pairs", 0)
        ),
        "vla_forward_count": int(pilot_metrics.get("vla_forward_count", 0))
        + int(development_metrics.get("vla_forward_count", 0)),
        "development_coverage": {
            "by_split": development_metrics.get("valid_by_split", {}),
            "by_map": development_metrics.get("valid_by_map", {}),
            "by_family": development_metrics.get("valid_by_family", {}),
            "by_weather": development_metrics.get("valid_by_weather", {}),
        },
        "interventions": {
            "proposals": development_metrics.get("intervention_proposals", 0),
            "terminal_by_operator": development_metrics.get("operator_terminal", {}),
            "guard_eligible_valid_by_operator": development_metrics.get(
                "operator_guard_eligible_valid", {}
            ),
        },
        "hazard_positives": development_metrics.get("hazard_positives", {}),
        "binary_class_counts": development_metrics.get("binary_class_counts", {}),
        "public_outcome_head_counts": development_metrics.get(
            "public_outcome_head_counts", {}
        ),
        "missingness": development_metrics.get("missingness", {}),
        "missingness_by": development_metrics.get("missingness_by", {}),
        "guard_verdicts": development_metrics.get("guard_verdicts", {}),
        "safety_decisions": development_metrics.get("safety_decisions", {}),
        "terminal_reasons": development_metrics.get("terminal_reasons", {}),
        "aggregate_collector_wall_s": aggregate_wall_s,
        "aggregate_wall_measurement": resource_audit.get("aggregate_wall_measurement"),
        "whole_gpu_peak_gib": whole_gpu_peak_gib,
        "minimum_observed_free_disk_gib": minimum_free_disk_gib,
        "dataset_bytes": int(development_metrics.get("dataset_bytes", 0)),
        "run_lock_sha256": lock.get("run_lock_sha256"),
        "config_sha256": CORA_C2_CONFIG_SHA256,
        "matrix_sha256": CORA_MATRIX_SHA256,
        "formal_collected": False,
        "collection_runs": collection_runs,
        "collector_failures": resource_audit.get("collector_failures", collection_failures),
        "quality_audits": {
            "pilot": {
                "passed": bool(pilot.get("passed")),
                "public_labels": _audit_passed(pilot, "public_label_audit"),
                "feature_reproducibility": _audit_passed(pilot, "feature_audit"),
                "inventory": _audit_passed(pilot, "inventory_audit"),
                "resources": _audit_passed(pilot, "resource_audit"),
            },
            "development": {
                "passed": bool(development.get("passed")),
                "public_labels": _audit_passed(development, "public_label_audit"),
                "feature_reproducibility": _audit_passed(development, "feature_audit"),
                "inventory": _audit_passed(development, "inventory_audit"),
                "resources": _audit_passed(development, "resource_audit"),
            },
        },
    }
    summary["collection_summary_sha256"] = stable_sha256(summary)
    _atomic_json(evidence / "collection-summary.json", summary)
    limits = CORA_C2_CONFIG["resources"]
    resources_valid = bool(
        summary["terminal_root_attempts"] == 351
        and summary["branch_attempts"] <= int(limits["branch_attempt_limit"])
        and summary["dataset_bytes"] <= float(limits["dataset_limit_gib"]) * 1024**3
        and summary["aggregate_collector_wall_s"] <= float(limits["aggregate_wall_limit_hours"]) * 3600.0
        and summary["whole_gpu_peak_gib"] <= float(limits["whole_gpu_peak_limit_gib"])
        and summary["minimum_observed_free_disk_gib"] >= float(limits["free_disk_floor_gib"])
        and bool(resource_audit.get("passed"))
    )
    gate_passed = bool(
        pilot.get("passed")
        and development.get("passed")
        and manifest_valid
        and lock_valid
        and resources_valid
    )
    final: dict[str, Any] = {
        "schema_version": FINAL_SCHEMA,
        "dataset_id": dataset_id,
        "terminal_status": "COMPLETED",
        "data_status": "VERIFIED" if gate_passed else "MEASURED",
        "gate_status": "GATE_PASSED" if gate_passed else "GATE_FAILED",
        "stopped": True,
        "c3_status": "NOT_AUTHORIZED",
        "formal_status": "NOT_COLLECTED",
        "pilot_gate": {"passed": bool(pilot.get("passed")), "failures": pilot.get("failures", [])},
        "development_gate": {"passed": bool(development.get("passed")), "failures": development.get("failures", [])},
        "integrity": {
            "manifest_valid": manifest_valid,
            "manifest_failures": list(manifest_failures),
            "manifest_verification_mode": development.get("manifest_verification_mode"),
            "pilot_manifest_audit_passed": bool(pilot_metrics.get("manifest_valid")),
            "run_lock_valid": lock_valid,
            "run_lock_failures": list(lock_failures),
            "public_label_audit_passed": _audit_passed(development, "public_label_audit"),
            "feature_reproducibility_audit_passed": _audit_passed(development, "feature_audit"),
            "inventory_audit_passed": _audit_passed(development, "inventory_audit"),
            "resource_audit_passed": _audit_passed(development, "resource_audit"),
            "resources_valid": resources_valid,
        },
        "hashes": {
            "config_sha256": CORA_C2_CONFIG_SHA256,
            "matrix_sha256": CORA_MATRIX_SHA256,
            "run_lock_sha256": lock.get("run_lock_sha256"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "physical_manifest_sha256": physical.get("physical_manifest_sha256"),
            "pilot_quality_sha256": pilot.get("data_quality_sha256"),
            "development_quality_sha256": development.get("data_quality_sha256"),
            "collection_summary_sha256": summary["collection_summary_sha256"],
        },
        "execution_history": execution_history,
        "runtime_release": runtime_release,
        "summary": summary,
        "known_failures": sorted(
            set(pilot.get("failures", ()))
            | set(development.get("failures", ()))
            | {
                str(row.get("error"))
                for row in resource_audit.get("collector_failures", ())
                if row.get("error")
            }
        ),
    }
    final["final_delivery_sha256"] = stable_sha256(final)
    _atomic_json(evidence / "final-delivery.json", final)
    return {
        "ok": True,
        "gate_status": final["gate_status"],
        "data_status": final["data_status"],
        "evidence": str(evidence / "final-delivery.json"),
        "final_delivery_sha256": final["final_delivery_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=str(CORA_C2_CONFIG["dataset_id"]))
    args = parser.parse_args()
    try:
        print(json.dumps(finalize(args.dataset_id), ensure_ascii=True, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
