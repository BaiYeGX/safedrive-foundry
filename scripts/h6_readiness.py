#!/usr/bin/env python3
"""Readiness gate for the VLA-primary World-v3 closed-loop experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import torch  # noqa: E402

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h6.matrix import (  # noqa: E402
    H6_SEEDS,
    H6_TRAIN_SEEDS,
    load_h6_matrix,
    load_h6_training_matrix,
    load_h6_vla75_matrix,
)
from data_pipeline.h6.config import H6_VLA75_FORMAL_LINEAGES, h6_vla75_config_sha256  # noqa: E402
from data_pipeline.h6.run_lock import (  # noqa: E402
    verify_run_lock,
    verify_summary_checkpoints_against_lock,
)


def evaluate_vla75_readiness(
    summary: dict,
    *,
    scope: str,
    lineage_id: str,
    root: Path = ROOT,
) -> dict:
    """Validate v2 development readiness without touching formal data."""

    lineage = str(lineage_id).lower()
    failures: list[str] = []
    if lineage not in H6_VLA75_FORMAL_LINEAGES:
        failures.append("unknown_formal_lineage")
    if summary.get("schema_version") not in {
        "safedrive.world.vla75.training_summary.v1",
        "safedrive.world.vla75.training_summary.v2",
    }:
        failures.append("vla75_training_summary_schema_missing")
    if not bool(summary.get("calibration", {}).get("passed")):
        failures.append("world_vla75_dev_calibration_failed")
    calibration = dict(summary.get("calibration") or {})
    if float(calibration.get("vla_coverage", 0.0)) < 0.90:
        failures.append("dev_raw_vla_coverage_below_90")
    if float(calibration.get("unsafe_delta", 1.0)) > 0.01:
        failures.append("dev_unsafe_delta_above_1pp")
    if float(calibration.get("risk_ceiling", 1.0)) > 0.20 + 1e-12:
        failures.append("vla75_risk_ceiling_above_frozen_limit")
    if float(calibration.get("mean_progress_delta_m", -1.0)) < 0.0:
        failures.append("dev_progress_delta_negative")
    actual = summary.get("h6_calibration_actual_vla_coverage")
    if actual is None or float(actual) < 0.75:
        failures.append("actual_dev_vla_execution_below_75")
    purity = summary.get(
        "training_outcome_attribution_purity",
        summary.get("h6_training_outcome_attribution_purity"),
    )
    if purity is None or float(purity) < 0.90:
        failures.append("training_outcome_attribution_below_90")
    if summary.get("calibration_label_scope") not in {
        "h6_tickwise_scores_plus_paired_policy_outcomes",
        "h6_vla75_tickwise_scores_plus_paired_policy_outcomes",
    }:
        failures.append("h6_policy_calibration_scope_missing")
    router_calibration = summary.get("router_calibration")
    if not isinstance(router_calibration, dict) or not bool(router_calibration.get("passed")):
        failures.append("vla75_router_calibration_missing_or_failed")
    else:
        try:
            router_alpha = float(router_calibration.get("ema_alpha", -1.0))
        except (TypeError, ValueError):
            router_alpha = -1.0
        try:
            router_hold = int(router_calibration.get("hold_ticks", -1))
        except (TypeError, ValueError):
            router_hold = -1
        try:
            router_hysteresis = float(router_calibration.get("hysteresis", -1.0))
        except (TypeError, ValueError):
            router_hysteresis = -1.0
        try:
            router_switches = int(router_calibration.get("switches", -1))
        except (TypeError, ValueError):
            router_switches = -1
        if router_alpha not in {0.25, 0.50, 0.75}:
            failures.append("vla75_router_ema_not_frozen_grid")
        if router_hold not in {6, 10, 14}:
            failures.append("vla75_router_hold_not_frozen_grid")
        if router_hysteresis not in {0.05, 0.10, 0.20}:
            failures.append("vla75_router_hysteresis_not_frozen_grid")
        if router_switches < 0:
            failures.append("vla75_router_switch_count_invalid")
        try:
            router_rows = int(router_calibration.get("rows", 0))
        except (TypeError, ValueError):
            router_rows = 0
        try:
            router_sequences = int(router_calibration.get("sequences", 0))
        except (TypeError, ValueError):
            router_sequences = 0
        if router_rows <= 0 or router_sequences <= 0:
            failures.append("vla75_router_tickwise_rows_missing")
        if bool(router_calibration.get("ping_pong", False)):
            failures.append("vla75_router_ping_pong")
        try:
            router_coverage = float(router_calibration.get("vla_coverage", -1.0))
        except (TypeError, ValueError):
            router_coverage = -1.0
        try:
            router_delta = float(router_calibration.get("unsafe_delta", float("inf")))
        except (TypeError, ValueError):
            router_delta = float("inf")
        if router_coverage < 0.75:
            failures.append("vla75_router_coverage_below_75")
        if router_delta > 0.01:
            failures.append("vla75_router_unsafe_delta_above_1pp")
    if not summary.get("h6_roots"):
        failures.append("fresh_h6_training_outcomes_missing")
    if bool(summary.get("h6_acceptance_seeds_loaded")):
        failures.append("acceptance_seed_training_leakage")
    formal_seeds = set(H6_VLA75_FORMAL_LINEAGES[lineage])
    loaded = {int(item) for item in summary.get("seeds_loaded", ())}
    if loaded & formal_seeds or 101 in loaded:
        failures.append("vla75_formal_or_consumed_seed_loaded")
    for item in summary.get("models", ()):
        path = Path(item["checkpoint_path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            failures.append(f"checkpoint_missing:{path}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item.get("checkpoint_sha256"):
            failures.append(f"checkpoint_hash:{path}")
        if item.get("schema_version") != "safedrive.world.vla75.pair_exec.v1":
            failures.append(f"checkpoint_schema:{path}")
        if int(item.get("seed", -1)) not in {11, 23, 37}:
            failures.append(f"checkpoint_seed:{path}")
    if len(summary.get("models", ())) != 3:
        failures.append("vla75_ensemble_requires_three_models")
    if sorted(int(item.get("seed", -1)) for item in summary.get("models", ())) != [11, 23, 37]:
        failures.append("vla75_model_seed_set")
    if not torch.cuda.is_available():
        failures.append("cuda_unavailable")
    full = load_h6_vla75_matrix(lineage, full=True)
    pilot = load_h6_vla75_matrix(lineage, full=False)
    training_full = load_h6_training_matrix(full=True)
    training_pilot = load_h6_training_matrix(full=False)
    if len(full) != 108 or len(pilot) != 12:
        failures.append("vla75_matrix_cardinality")
    if len(training_full) != 108 or len(training_pilot) != 24:
        failures.append("h6_training_matrix_cardinality")
    expected_matrix = training_full if scope == "full" else training_pilot
    expected_train = {
        row.pair_id for row in expected_matrix if row.scenario.seed == H6_TRAIN_SEEDS[0]
    }
    expected_calibration = {
        row.pair_id for row in expected_matrix if row.scenario.seed == H6_TRAIN_SEEDS[1]
    }
    observed_train = set(summary.get("h6_train_pair_ids") or ())
    observed_calibration = set(summary.get("h6_calibration_pair_ids") or ())
    if observed_train != expected_train:
        failures.append(f"h6_{scope}_training_seed_coverage")
    if observed_calibration != expected_calibration:
        failures.append(f"h6_{scope}_calibration_seed_coverage")
    if summary.get("contract") not in (None, "vla75-v2"):
        failures.append("vla75_contract_mismatch")
    return {
        "ready": not failures,
        "failures": sorted(set(failures)),
        "contract": "vla75-v2",
        "lineage_id": lineage,
        "config_sha256": h6_vla75_config_sha256(lineage),
        "scope": scope,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "calibration": calibration,
        "actual_dev_vla_execution": actual,
        "training_outcome_attribution_purity": purity,
        "h6_full_scenarios": len(full),
        "h6_pilot_scenarios": len(pilot),
        "h6_formal_seeds": sorted(formal_seeds),
        "h6_training_full_scenarios": len(training_full),
        "h6_training_pilot_scenarios": len(training_pilot),
        "h6_expected_train_pairs": len(expected_train),
        "h6_observed_train_pairs": len(observed_train),
        "h6_expected_calibration_pairs": len(expected_calibration),
        "h6_observed_calibration_pairs": len(observed_calibration),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-v3-summary", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("pilot", "full"),
        default="full",
        help="Require coverage for the requested collection scope.",
    )
    parser.add_argument("--contract", choices=("vla90", "vla75-v2"), default="vla90")
    parser.add_argument("--formal-lineage", choices=("a", "b", "c"), default=None)
    parser.add_argument("--run-lock", type=Path, default=None)
    parser.add_argument(
        "--require-run-lock",
        action="store_true",
        help="Require and verify an immutable lock (formal H6 only).",
    )
    args = parser.parse_args()
    if args.contract == "vla75-v2":
        if not args.formal_lineage:
            raise ValueError("h6_vla75_formal_lineage_required")
        summary = json.loads(args.world_v3_summary.read_text(encoding="utf-8"))
        payload = evaluate_vla75_readiness(
            summary,
            scope=args.scope,
            lineage_id=args.formal_lineage,
        )
        if args.run_lock is None and args.require_run_lock:
            payload["failures"] = sorted(set(payload["failures"]) | {"run_lock_missing"})
            payload["ready"] = False
        elif args.run_lock is not None:
            lock = json.loads(args.run_lock.read_text(encoding="utf-8"))
            verification = verify_run_lock(lock, root=ROOT)
            if not verification["valid"]:
                payload["failures"] = sorted(
                    set(payload["failures"]) | {f"run_lock:{item}" for item in verification["failures"]}
                )
                payload["ready"] = False
            elif lock.get("config_sha256") != payload.get("config_sha256"):
                payload["failures"] = sorted(set(payload["failures"]) | {"run_lock_config_mismatch"})
                payload["ready"] = False
            else:
                binding = verify_summary_checkpoints_against_lock(
                    summary.get("models"), lock, root=ROOT
                )
                if not binding["valid"]:
                    payload["failures"] = sorted(
                        set(payload["failures"])
                        | {f"run_lock_summary:{item}" for item in binding["failures"]}
                    )
                    payload["ready"] = False
                locked_calibration = lock.get("calibration")
                if not isinstance(locked_calibration, dict):
                    payload["failures"] = sorted(
                        set(payload["failures"]) | {"run_lock_calibration_missing"}
                    )
                    payload["ready"] = False
                else:
                    locked_deployment = locked_calibration.get("deployment")
                    if not isinstance(locked_deployment, dict) or stable_sha256(
                        dict(summary.get("calibration") or {})
                    ) != stable_sha256(locked_deployment):
                        payload["failures"] = sorted(
                            set(payload["failures"]) | {"run_lock_calibration_mismatch"}
                        )
                        payload["ready"] = False
                    locked_router = locked_calibration.get("router")
                    if not isinstance(locked_router, dict) or stable_sha256(
                        dict(summary.get("router_calibration") or {})
                    ) != stable_sha256(locked_router):
                        payload["failures"] = sorted(
                            set(payload["failures"]) | {"run_lock_router_calibration_mismatch"}
                        )
                        payload["ready"] = False
                    locked_temperatures = locked_calibration.get("temperatures")
                    if not isinstance(locked_temperatures, dict):
                        payload["failures"] = sorted(
                            set(payload["failures"])
                            | {"run_lock_temperature_calibration_missing"}
                        )
                        payload["ready"] = False
                    else:
                        observed_temperatures = dict(
                            summary.get("temperature_calibration") or {}
                        )
                        if stable_sha256(observed_temperatures) != stable_sha256(
                            locked_temperatures
                        ):
                            payload["failures"] = sorted(
                                set(payload["failures"])
                                | {"run_lock_temperature_calibration_mismatch"}
                            )
                            payload["ready"] = False
            payload["run_lock_sha256"] = lock.get("lock_sha256")
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["ready"] else 2
    failures = []
    summary = json.loads(args.world_v3_summary.read_text(encoding="utf-8"))
    calibration = dict(summary.get("calibration") or {})
    if not calibration.get("passed"):
        failures.append("world_v3_dev_calibration_failed")
    if float(calibration.get("vla_coverage", 0.0)) < 0.90:
        failures.append("dev_vla_coverage_below_90")
    if float(calibration.get("unsafe_delta", 1.0)) > 0.01:
        failures.append("dev_unsafe_delta_above_1pp")
    if float(calibration.get("trust_threshold", 0.0)) < 0.50:
        failures.append("trust_threshold_not_high")
    if float(calibration.get("mean_progress_delta_m", -1.0)) < 0.0:
        failures.append("dev_progress_delta_negative")
    actual_dev_vla = summary.get("h6_calibration_actual_vla_coverage")
    if actual_dev_vla is None or float(actual_dev_vla) < 0.90:
        failures.append("actual_dev_vla_execution_below_90")
    if (
        summary.get("calibration_label_scope")
        != "h6_tickwise_scores_plus_paired_policy_outcomes"
    ):
        failures.append("h6_policy_calibration_scope_missing")
    if not summary.get("h6_roots"):
        failures.append("fresh_h6_training_outcomes_missing")
    if bool(summary.get("h6_acceptance_seeds_loaded")):
        failures.append("acceptance_seed_training_leakage")
    for item in summary.get("models", ()):
        path = Path(item["checkpoint_path"])
        if not path.is_file():
            failures.append(f"checkpoint_missing:{path}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item.get("checkpoint_sha256"):
            failures.append(f"checkpoint_hash:{path}")
    if not torch.cuda.is_available():
        failures.append("cuda_unavailable")

    old_split = json.loads(
        (ROOT / "generated/h3/h3-v2-20260815d-final/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    old_seeds = {int(item["seed"]) for item in old_split.get("rows", ())}
    if old_seeds & (set(H6_SEEDS) | set(H6_TRAIN_SEEDS)):
        failures.append("h6_seed_overlap")
    if set(H6_SEEDS) & set(H6_TRAIN_SEEDS):
        failures.append("h6_train_accept_seed_overlap")
    full = load_h6_matrix(full=True)
    pilot = load_h6_matrix(full=False)
    training_full = load_h6_training_matrix(full=True)
    training_pilot = load_h6_training_matrix(full=False)
    if len(full) != 108 or len(pilot) != 12:
        failures.append("h6_matrix_cardinality")
    if len(training_full) != 108 or len(training_pilot) != 24:
        failures.append("h6_training_matrix_cardinality")
    expected_matrix = training_full if args.scope == "full" else training_pilot
    expected_train = {
        row.pair_id
        for row in expected_matrix
        if row.scenario.seed == H6_TRAIN_SEEDS[0]
    }
    expected_calibration = {
        row.pair_id
        for row in expected_matrix
        if row.scenario.seed == H6_TRAIN_SEEDS[1]
    }
    observed_train = set(summary.get("h6_train_pair_ids") or ())
    observed_calibration = set(summary.get("h6_calibration_pair_ids") or ())
    if observed_train != expected_train:
        failures.append(f"h6_{args.scope}_training_seed_coverage")
    if observed_calibration != expected_calibration:
        failures.append(f"h6_{args.scope}_calibration_seed_coverage")
    payload = {
        "ready": not failures,
        "failures": failures,
        "scope": args.scope,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "world_v3_summary": str(args.world_v3_summary),
        "calibration": calibration,
        "actual_dev_vla_execution": actual_dev_vla,
        "h6_full_scenarios": len(full),
        "h6_pilot_scenarios": len(pilot),
        "h6_seeds": list(H6_SEEDS),
        "h6_training_seeds": list(H6_TRAIN_SEEDS),
        "h6_training_full_scenarios": len(training_full),
        "h6_training_pilot_scenarios": len(training_pilot),
        "h6_expected_train_pairs": len(expected_train),
        "h6_observed_train_pairs": len(observed_train),
        "h6_expected_calibration_pairs": len(expected_calibration),
        "h6_observed_calibration_pairs": len(observed_calibration),
        "old_seeds": sorted(old_seeds),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
