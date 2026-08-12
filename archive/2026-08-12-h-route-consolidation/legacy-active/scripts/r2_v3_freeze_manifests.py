#!/usr/bin/env python3
"""Exclusive-freeze R2 V3 plans and thresholds before live outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r2_world_ready_v3 import (  # noqa: E402
    CALIBRATION_MAPS,
    build_calibration_manifest_v3,
    build_core_blind_manifest_v3,
    build_long_smoke_manifest_v3,
    build_unseen_long_audit_manifest_v3,
    build_world_ready_audit_manifest_v3,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import canonical_sha256  # noqa: E402


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--route-fixture", required=True)
    parser.add_argument("--scenario-registry", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing non-empty plan directory: {out}")
    route = json.loads(Path(args.route_fixture).read_text(encoding="utf-8"))
    registry = load_scenario_registry(args.scenario_registry)
    calibration = build_calibration_manifest_v3()
    pilot_slots = [
        slot
        for slot in calibration["slots"]
        if str(slot["map_name"]) in CALIBRATION_MAPS[:2]
    ]
    pilot_body = {
        "schema_version": calibration["schema_version"],
        "phase": "calibration_pilot",
        "source_calibration_manifest_hash": calibration["manifest_hash"],
        "maps": list(CALIBRATION_MAPS[:2]),
        "slot_count": len(pilot_slots),
        "slots": pilot_slots,
    }
    pilot = {**pilot_body, "manifest_hash": canonical_sha256(pilot_body)}
    threshold_body = {
        "schema_version": "safedrive.r2_v3.frozen_thresholds.v1",
        "route_fixture_hash": str(route["manifest_hash"]),
        "long_smoke_registry_hash": registry.compute_registry_sha256(),
        "oracle_version": "oracle_v2_clearance_saturated",
        "route_maneuver_consistency": 1.0,
        "semantic_accuracy_min": 0.90,
        "direction_accuracy_min": 0.95,
        "availability_recall_min": 0.80,
        "availability_specificity_min": 0.80,
        "none_close_rate": 1.0,
        "legal_lane_exit_rate": 1.0,
        "guard_mpc_acceptance_min": 0.90,
        "turn_final_lane_center_error_max_m": 0.60,
        "path_tracking_p95_max_m": 0.85,
        "spatial_separation_min_m": 0.50,
        "lineage_route_overlap_allowed": False,
        "post_blind_checkpoint_selection_allowed": False,
    }
    thresholds = {
        **threshold_body,
        "manifest_hash": canonical_sha256(threshold_body),
    }
    runtime_body = {
        "schema_version": "safedrive.r2_v3.long_smoke_runtime.v3",
        "mode": "teacher_contract",
        "map": str(route["map_name"]),
        "scenario_registry_hash": registry.compute_registry_sha256(),
        "duration_s": 20.0,
        "sim_dt_s": 0.05,
        "vla_period_s": 0.75,
        "speed_cap_mps": 6.0,
        "speed_gain": 1.0,
        "official_simlingo_contract": True,
        "coarse_route_spacing_m": 10.0,
        "contract_auto_select": True,
        "continue_policy": "continue_all",
        "retry_policy": "one_initial_plus_two_substantive_repairs",
        "outcome_used": False,
        "implementation_hashes": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in (
                ROOT
                / "safedrive_foundry/driving_vla/model/navigation_contract.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/simlingo_contract.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/k2_v3_codec.py",
                ROOT / "safedrive_foundry/driving_vla/model/k2_v3_guard.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/k2_spatial_builder.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/semantic_k2_teacher.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/semantic_mode_heads.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/neural_policy.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/checkpoint_contract.py",
                ROOT
                / "safedrive_foundry/driving_vla/model/k2_v3_types.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/route_authoring_v3.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/maneuver_completion.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/long_horizon_observer.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/scenario_registry.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/fixture_runtime.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/comparability.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/executability_metrics.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/k2_v3_artifact.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/outcome_metrics.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/oracle_v2.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/paired_live.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/paired_live_v3.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/r2_v3_dataset.py",
                ROOT
                / "safedrive_foundry/driving_vla/evaluation/r2_world_ready_v3.py",
                ROOT
                / "safedrive_foundry/driving_vla/runtime/basic1v1_observable.py",
                ROOT
                / "safedrive_foundry/driving_vla/runtime/navigation_topology.py",
                ROOT
                / "safedrive_foundry/driving_vla/runtime/k2_execution.py",
                ROOT / "tests/g3/run_g3_vla_mpc_stable.py",
                ROOT / "scripts/r2_v3_aggregate_formal.py",
                ROOT / "scripts/r2_v3_author_campaign.py",
                ROOT / "scripts/r2_v3_author_formal.py",
                ROOT / "scripts/r2_v3_author_long_smoke.py",
                ROOT / "scripts/r2_v3_author_route_bank.py",
                ROOT / "scripts/r2_v3_build_dataset.py",
                ROOT / "scripts/r2_v3_collect_campaign.py",
                ROOT / "scripts/r2_v3_freeze_manifests.py",
                ROOT / "scripts/r2_v3_offline_eval.py",
                ROOT / "scripts/r2_v3_promote_formal.py",
                ROOT / "scripts/r2_v3_run_formal_long.py",
                ROOT / "scripts/r2_v3_run_formal_pairs.py",
                ROOT / "scripts/r2_v3_run_long_smokes.py",
                ROOT / "scripts/r2_v3_train_heads.py",
            )
        },
    }
    runtime_contract = {
        **runtime_body,
        "manifest_hash": canonical_sha256(runtime_body),
    }
    artifacts = {
        "long_smoke_teacher.json": build_long_smoke_manifest_v3(learned=False),
        "long_smoke_learned.json": build_long_smoke_manifest_v3(learned=True),
        "unseen_long_audit_16.json": build_unseen_long_audit_manifest_v3(),
        "calibration_360.json": calibration,
        "calibration_pilot_144.json": pilot,
        "core_blind_12.json": build_core_blind_manifest_v3(),
        "world_ready_audit_84.json": build_world_ready_audit_manifest_v3(),
        "thresholds.json": thresholds,
        "runtime_contract.json": runtime_contract,
    }
    for name, value in artifacts.items():
        _write_exclusive(out / name, value)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "out_dir": str(out),
                "files": sorted(artifacts),
                "pilot_slots": len(pilot_slots),
                "calibration_slots": len(calibration["slots"]),
                "route_fixture_hash": route["manifest_hash"],
                "registry_hash": registry.compute_registry_sha256(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
