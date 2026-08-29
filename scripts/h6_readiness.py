#!/usr/bin/env python3
"""Readiness gate for the VLA-primary World-v3 closed-loop experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

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
from data_pipeline.h6.evaluator import (  # noqa: E402
    EVALUATOR_SCHEMA,
    MEASURED,
    SUMMARY_SCHEMA,
    file_sha256,
    verify_vla75_evaluator,
)
from data_pipeline.h6.run_lock import (  # noqa: E402
    verify_run_lock,
    verify_summary_checkpoints_against_lock,
)
from data_pipeline.h6.model import (  # noqa: E402
    WORLD_V3_HEAD_WEIGHTS,
    WORLD_VLA75_EXTRA_HEAD_WEIGHTS,
    vla75_checkpoint_selection_key,
)


def _positive_finite(value: Any, *, allow_zero: bool = False) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and (numeric >= 0.0 if allow_zero else numeric > 0.0)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def evaluate_vla75_readiness(
    summary: dict,
    *,
    scope: str,
    lineage_id: str,
    root: Path = ROOT,
) -> dict:
    """Validate C1 artifacts without making an algorithm-quality claim."""

    lineage = str(lineage_id).lower()
    failures: list[str] = []
    schema = summary.get("schema_version")
    if schema == "safedrive.world.vla75.training_summary.v1":
        failures.append("legacy_summary_not_c1_ready")
    elif schema != SUMMARY_SCHEMA:
        failures.append("vla75_training_summary_schema_missing")
    summary_payload = {
        key: value for key, value in summary.items() if key != "summary_sha256"
    }
    if schema == SUMMARY_SCHEMA and summary.get("summary_sha256") != stable_sha256(
        summary_payload
    ):
        failures.append("summary_hash")
    if summary.get("evidence_state") != MEASURED:
        failures.append("summary_not_measured")
    if summary.get("artifact_verification") != "VERIFIED":
        failures.append("summary_artifact_not_verified")
    if summary.get("cora_algorithm_state") != "NOT_VERIFIED":
        failures.append("cora_algorithm_state_invalid")
    if lineage not in H6_VLA75_FORMAL_LINEAGES:
        failures.append("unknown_formal_lineage")
        formal_seeds: set[int] = set()
    else:
        formal_seeds = set(H6_VLA75_FORMAL_LINEAGES[lineage])

    train_lineage = summary.get("train_lineage_sha256")
    validation_lineage = summary.get("validation_lineage_sha256")
    for name, value in (
        ("train_lineage", train_lineage),
        ("validation_lineage", validation_lineage),
        ("training_config", summary.get("training_config_sha256")),
        ("code", summary.get("code_sha256")),
        ("worktree", summary.get("worktree_sha256")),
    ):
        if not isinstance(value, str) or len(value) != 64:
            failures.append(f"summary_{name}_hash_missing")

    raw_models = summary.get("models")
    raw_evaluators = summary.get("evaluators")
    if (
        not isinstance(raw_models, list)
        or len(raw_models) != 3
        or not all(isinstance(item, Mapping) for item in raw_models)
    ):
        failures.append("vla75_ensemble_requires_three_models")
        models = []
    else:
        models = [dict(item) for item in raw_models]
    if (
        not isinstance(raw_evaluators, list)
        or len(raw_evaluators) != 3
        or not all(isinstance(item, Mapping) for item in raw_evaluators)
    ):
        failures.append("vla75_ensemble_requires_three_evaluators")
        evaluators = []
    else:
        evaluators = [dict(item) for item in raw_evaluators]
    expected_seeds = [11, 23, 37]
    if [_safe_int(item.get("seed")) for item in models] != expected_seeds:
        failures.append("vla75_model_seed_order")
    if [_safe_int(item.get("seed")) for item in evaluators] != expected_seeds:
        failures.append("vla75_evaluator_seed_order")

    for index, item in enumerate(models):
        raw = Path(str(item.get("checkpoint_path", "")))
        path = raw if raw.is_absolute() else root / raw
        if not path.is_file():
            failures.append(f"checkpoint_missing:{index}")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != item.get("checkpoint_sha256"):
            failures.append(f"checkpoint_hash:{index}")
        if item.get("schema_version") != "safedrive.world.vla75.pair_exec.v1":
            failures.append(f"checkpoint_schema:{index}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            metadata = dict(checkpoint.get("metadata") or {})
        except Exception:
            failures.append(f"checkpoint_load:{index}")
            continue
        if _safe_int(metadata.get("seed")) != expected_seeds[index]:
            failures.append(f"checkpoint_seed:{index}")
        if metadata.get("train_lineage_sha256") != train_lineage:
            failures.append(f"checkpoint_train_lineage:{index}")
        if metadata.get("validation_lineage_sha256") != validation_lineage:
            failures.append(f"checkpoint_validation_lineage:{index}")
        selection = metadata.get("selection_metrics")
        if not isinstance(selection, Mapping) or stable_sha256(dict(selection)) != metadata.get(
            "selection_metrics_sha256"
        ):
            failures.append(f"checkpoint_selection_hash:{index}")
        else:
            try:
                vla75_checkpoint_selection_key(selection)
            except (TypeError, ValueError):
                failures.append(f"checkpoint_selection_invalid:{index}")
            if selection.get("evaluator_lineage_sha256") != validation_lineage:
                failures.append(f"checkpoint_selection_lineage:{index}")

    for index, item in enumerate(evaluators):
        raw = Path(str(item.get("path", "")))
        path = raw if raw.is_absolute() else root / raw
        if not path.is_file():
            failures.append(f"evaluator_missing:{index}")
            continue
        if file_sha256(path) != item.get("file_sha256"):
            failures.append(f"evaluator_file_hash:{index}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            failures.append(f"evaluator_parse:{index}")
            continue
        if artifact.get("schema_version") != EVALUATOR_SCHEMA:
            failures.append(f"evaluator_schema:{index}")
        if artifact.get("evaluator_sha256") != item.get("sha256"):
            failures.append(f"evaluator_summary_hash:{index}")
        verification = verify_vla75_evaluator(artifact, root=root)
        failures.extend(f"evaluator_{index}:{failure}" for failure in verification["failures"])
        checkpoint = dict(artifact.get("checkpoint") or {})
        inputs = dict(artifact.get("inputs") or {})
        if index < len(models):
            model_item = models[index]
            if model_item.get("evaluator_sha256") != item.get("sha256"):
                failures.append(f"model_evaluator_hash:{index}")
            if model_item.get("checkpoint_sha256") != checkpoint.get("sha256"):
                failures.append(f"evaluator_checkpoint_hash_binding:{index}")
        if _safe_int(checkpoint.get("seed")) != expected_seeds[index]:
            failures.append(f"evaluator_seed:{index}")
        if inputs.get("training_lineage_sha256") != train_lineage:
            failures.append(f"evaluator_train_lineage:{index}")
        if inputs.get("validation_lineage_sha256") != validation_lineage:
            failures.append(f"evaluator_validation_lineage:{index}")
        expected_input = stable_sha256(
            {
                "training": inputs.get("training_lineage_sha256"),
                "validation": inputs.get("validation_lineage_sha256"),
                "config": inputs.get("config_sha256"),
                "code": inputs.get("code_sha256"),
                "worktree": inputs.get("worktree_sha256"),
            }
        )
        if inputs.get("input_sha256") != expected_input:
            failures.append(f"evaluator_input_hash:{index}")
        for field, summary_field in (
            ("config_sha256", "training_config_sha256"),
            ("code_sha256", "code_sha256"),
            ("worktree_sha256", "worktree_sha256"),
        ):
            if inputs.get(field) != summary.get(summary_field):
                failures.append(f"evaluator_{field}:{index}")

        validation = dict(artifact.get("validation") or {})
        loss = dict(validation.get("loss") or {})
        if loss.get("status") != MEASURED or _safe_int(loss.get("count"), 0) <= 0:
            failures.append(f"evaluator_validation_not_measured:{index}")
        elif not _finite_number(loss.get("value")):
            failures.append(f"evaluator_validation_nonfinite:{index}")
        heads = validation.get("heads")
        if not isinstance(heads, Mapping) or not heads:
            failures.append(f"evaluator_heads_missing:{index}")
        else:
            required_heads = set(WORLD_V3_HEAD_WEIGHTS) | set(
                WORLD_VLA75_EXTRA_HEAD_WEIGHTS
            )
            for name in sorted(required_heads - set(heads)):
                failures.append(f"evaluator_head_missing:{index}:{name}")
            for name, head in heads.items():
                if not isinstance(head, Mapping) or head.get("status") != MEASURED:
                    failures.append(f"evaluator_head_not_measured:{index}:{name}")
                    continue
                if _safe_int(head.get("count"), 0) <= 0 or not _finite_number(head.get("loss")):
                    failures.append(f"evaluator_head_invalid:{index}:{name}")
                if name in {"collision", "red_light", "offroad"} and _safe_int(
                    head.get("positive_count"), 0
                ) <= 0:
                    failures.append(f"evaluator_hazard_positive_zero:{index}:{name}")
        pair = dict(validation.get("pair") or {})
        if pair.get("status") != MEASURED or _safe_int(pair.get("count"), 0) <= 0:
            failures.append(f"evaluator_pair_not_measured:{index}")
        for field in ("accuracy", "regret"):
            if not _positive_finite(pair.get(field), allow_zero=True):
                failures.append(f"evaluator_pair_{field}:{index}")
        groups = validation.get("groups")
        if not isinstance(groups, Mapping) or not groups:
            failures.append(f"evaluator_groups_missing:{index}")
        else:
            for name, group in groups.items():
                if not isinstance(group, Mapping) or group.get("status") != MEASURED:
                    failures.append(f"evaluator_group_not_measured:{index}:{name}")
                elif _safe_int(group.get("count"), 0) <= 0 or not _finite_number(
                    group.get("loss")
                ) or not _positive_finite(group.get("weight")):
                    failures.append(f"evaluator_group_invalid:{index}:{name}")

        probes = dict(artifact.get("probes") or {})
        for name in ("candidate_swap", "source_metadata_swap", "action_mask", "context_history_mask"):
            probe = dict(probes.get(name) or {})
            if probe.get("status") != MEASURED or _safe_int(probe.get("count"), 0) <= 0:
                failures.append(f"evaluator_probe_not_measured:{index}:{name}")
                continue
            value = probe.get("value")
            if not _positive_finite(value, allow_zero=True):
                failures.append(f"evaluator_probe_nonfinite:{index}:{name}")
            elif name in {"candidate_swap", "source_metadata_swap"} and float(value) > 1e-6:
                failures.append(f"evaluator_probe_invariant:{index}:{name}")
            elif name in {"action_mask", "context_history_mask"} and float(value) <= 0.0:
                failures.append(f"evaluator_probe_no_sensitivity:{index}:{name}")
        latency = dict(artifact.get("latency") or {})
        if latency.get("status") != MEASURED or _safe_int(latency.get("iterations"), 0) <= 0:
            failures.append(f"evaluator_latency_not_measured:{index}")
        for field in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
            if not _positive_finite(latency.get(field)):
                failures.append(f"evaluator_latency_{field}:{index}")
        gpu = dict(artifact.get("gpu_peak") or {})
        if gpu.get("status") != MEASURED or not _positive_finite(
            gpu.get("incremental_peak_gib")
        ):
            failures.append(f"evaluator_gpu_not_measured:{index}")

    if not summary.get("h6_roots"):
        failures.append("fresh_h6_training_outcomes_missing")
    if bool(summary.get("h6_acceptance_seeds_loaded")):
        failures.append("acceptance_seed_training_leakage")
    raw_loaded = summary.get("seeds_loaded", ())
    if not isinstance(raw_loaded, (list, tuple)):
        failures.append("seeds_loaded_invalid")
        raw_loaded = ()
    loaded_values = [_safe_int(item) for item in raw_loaded]
    if any(value < 0 for value in loaded_values):
        failures.append("seeds_loaded_invalid")
    loaded = {value for value in loaded_values if value >= 0}
    if loaded & formal_seeds or 101 in loaded:
        failures.append("vla75_formal_or_consumed_seed_loaded")
    if not torch.cuda.is_available():
        failures.append("cuda_unavailable")
    full = load_h6_vla75_matrix(lineage, full=True) if formal_seeds else []
    pilot = load_h6_vla75_matrix(lineage, full=False) if formal_seeds else []
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
    result = {
        "ready": not failures,
        "failures": sorted(set(failures)),
        "contract": "vla75-v2",
        "lineage_id": lineage,
        "config_sha256": h6_vla75_config_sha256(lineage),
        "scope": scope,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifact_verification_state": "VERIFIED" if not failures else "FAILED",
        "cora_algorithm_state": "NOT_VERIFIED",
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
    result["readiness_sha256"] = stable_sha256(result)
    return result


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
        payload["readiness_sha256"] = stable_sha256(
            {key: value for key, value in payload.items() if key != "readiness_sha256"}
        )
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
