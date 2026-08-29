#!/usr/bin/env python3
"""Train and dev-calibrate the outcome/trust World v3 ensemble."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h6.calibration import (  # noqa: E402
    CalibrationRow,
    calibrate_temperature_heads,
    calibrate_vla_deployment,
    select_vla75_router_config,
)
from data_pipeline.h6.dataset import (  # noqa: E402
    load_h6_closed_loop_examples,
    load_h6_policy_calibration_examples,
    load_outcome_examples,
    outcome_examples_lineage_sha256,
)
from data_pipeline.h6.evaluator import (  # noqa: E402
    build_vla75_evaluator,
    file_sha256,
    finalize_training_summary_v2,
)
from data_pipeline.h6.matrix import H6_TRAIN_SEEDS  # noqa: E402
from data_pipeline.h6.config import H6_VLA75_FORMAL_LINEAGES  # noqa: E402
from data_pipeline.h6.model import (  # noqa: E402
    WorldV3TrainConfig,
    WorldVLA75TrainConfig,
    train_world_v3,
    train_world_vla75,
)
from data_pipeline.h6.runtime import WorldV3Scorer, WorldVLA75Scorer  # noqa: E402
from data_pipeline.h6.run_lock import scoped_file_hashes, worktree_identity  # noqa: E402


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _pair_payload(candidate):
    return candidate.candidate_key, candidate.context, candidate.candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", type=Path, default=[])
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--h6-root", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--contract", choices=("vla90", "vla75-v2"), default="vla90")
    parser.add_argument("--formal-lineage", choices=("a", "b", "c"), default=None)
    args = parser.parse_args()

    requested_seeds = tuple(
        int(item.strip()) for item in args.seeds.split(",") if item.strip()
    )
    if args.contract == "vla75-v2" and requested_seeds != (11, 23, 37):
        raise SystemExit("vla75_model_seeds_must_be_11_23_37")

    if bool(args.data_root) != bool(args.split_manifest):
        raise SystemExit("old data roots and --split-manifest must be supplied together")
    if not args.data_root and not args.h6_root:
        raise SystemExit("at least one --data-root or --h6-root is required")
    manifest = (
        json.loads(args.split_manifest.read_text(encoding="utf-8"))
        if args.split_manifest is not None
        else {"rows": []}
    )
    train = (
        load_outcome_examples(
            args.data_root, manifest, splits=("dev_fold_2", "dev_fold_3")
        )
        if args.data_root
        else []
    )
    val = (
        load_outcome_examples(args.data_root, manifest, splits=("dev_fold_1",))
        if args.data_root
        else []
    )
    h6_train = []
    h6_val = []
    h6_policy_train = []
    h6_policy_calibration = []
    if args.h6_root:
        h6_train = load_h6_closed_loop_examples(
            args.h6_root,
            seeds=(H6_TRAIN_SEEDS[0],),
            split="h6_train",
        )
        h6_val = load_h6_closed_loop_examples(
            args.h6_root,
            seeds=(H6_TRAIN_SEEDS[1],),
            split="h6_calibration",
        )
        h6_policy_train = load_h6_policy_calibration_examples(
            args.h6_root,
            seeds=(H6_TRAIN_SEEDS[0],),
        )
        h6_policy_calibration = load_h6_policy_calibration_examples(
            args.h6_root,
            seeds=(H6_TRAIN_SEEDS[1],),
        )
        train.extend(h6_train)
        train.extend(
            item.as_outcome_pair(split="h6_train_policy")
            for item in h6_policy_train
        )
        val.extend(h6_val)
        val.extend(
            item.as_outcome_pair(split="h6_calibration_policy")
            for item in h6_policy_calibration
        )
        if args.contract == "vla75-v2":
            # Seed 101 is permanently consumed by the historical H6/v1
            # evidence.  Keep the isolation check at the training entrypoint
            # as well as in the loaders so a mixed or hand-built artifact can
            # never silently enter v75 training/calibration.
            loaded_h6_seeds = {
                int(pair.seed)
                for pair in (h6_train + h6_val)
            }
            loaded_h6_seeds.update(
                int(item.seed)
                for item in (h6_policy_train + h6_policy_calibration)
            )
            if 101 in loaded_h6_seeds:
                raise SystemExit("h6_seed_101_is_consumed_and_training_forbidden")
    if not train or not val:
        raise SystemExit("world_v3_empty_train_or_val")
    if args.contract == "vla75-v2":
        # Formal/consumed H6 seeds are never a training or calibration input,
        # including when a caller supplies legacy ``--data-root`` artifacts
        # alongside the fresh h6-root.  Keep this check at the final merged
        # dataset boundary so no alternate loader can bypass isolation.
        forbidden_h6_seeds = {101}
        for lineage_seeds in H6_VLA75_FORMAL_LINEAGES.values():
            forbidden_h6_seeds.update(int(seed) for seed in lineage_seeds)
        loaded_seeds = {
            int(pair.seed)
            for pair in tuple(train) + tuple(val)
            if getattr(pair, "seed", None) is not None
        }
        overlap = sorted(loaded_seeds & forbidden_h6_seeds)
        if overlap:
            raise SystemExit(
                f"h6_vla75_formal_or_consumed_seed_training_forbidden:{overlap}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        WorldVLA75TrainConfig() if args.contract == "vla75-v2" else WorldV3TrainConfig(),
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    train_lineage_sha256 = outcome_examples_lineage_sha256(train)
    validation_lineage_sha256 = outcome_examples_lineage_sha256(val)
    c1_code_paths = (
        "safedrive_foundry/data_pipeline/h6/dataset.py",
        "safedrive_foundry/data_pipeline/h6/evaluator.py",
        "safedrive_foundry/data_pipeline/h6/model.py",
        "scripts/train_world_v3.py",
    )
    code_sha256 = stable_sha256(scoped_file_hashes(ROOT, c1_code_paths))
    worktree_sha256 = worktree_identity(ROOT)["full_worktree_hash"]
    training_config_sha256 = stable_sha256(asdict(cfg))
    results = []
    evaluator_rows = []
    checkpoint_paths = []
    for seed in requested_seeds:
        path = args.output_dir / "checkpoints" / f"seed-{seed}.pt"
        result = (
            train_world_vla75
            if args.contract == "vla75-v2"
            else train_world_v3
        )(
            train,
            val,
            seed=seed,
            checkpoint_path=path,
            device=args.device,
            config=cfg,
        )
        result_payload = asdict(result)
        if args.contract == "vla75-v2":
            result_payload["schema_version"] = "safedrive.world.vla75.pair_exec.v1"
            evaluator = build_vla75_evaluator(
                path,
                val,
                device=args.device,
                training_input_sha256=train_lineage_sha256,
                config_sha256=training_config_sha256,
                code_sha256=code_sha256,
                worktree_sha256=worktree_sha256,
            )
            evaluator_path = args.output_dir / "evaluators" / f"seed-{seed}.json"
            evaluator_path.parent.mkdir(parents=True, exist_ok=True)
            evaluator_path.write_text(
                json.dumps(evaluator, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            evaluator_row = {
                "seed": seed,
                "path": str(evaluator_path),
                "sha256": evaluator["evaluator_sha256"],
                "file_sha256": file_sha256(evaluator_path),
            }
            evaluator_rows.append(evaluator_row)
            result_payload.update(
                evaluator_path=str(evaluator_path),
                evaluator_sha256=evaluator["evaluator_sha256"],
                train_lineage_sha256=train_lineage_sha256,
                validation_lineage_sha256=validation_lineage_sha256,
            )
        results.append(result_payload)
        checkpoint_paths.append(path)

    scorer_type = WorldVLA75Scorer if args.contract == "vla75-v2" else WorldV3Scorer
    scorer = scorer_type.from_checkpoints(
        checkpoint_paths,
        device=args.device,
        calibration={"trust_threshold": 0.0, "risk_ceiling": 1.0},
    )
    calibration_rows = []
    latencies = []
    swap_errors = []
    hazard_correct = {"collision": 0, "red_light": 0, "offroad": 0, "trust": 0}
    hazard_total = 0
    h6_calibration_tick_ids = {
        id(pair) for pair in h6_val if pair.arm == "on"
    }
    for pair in val:
        first, second = pair.candidates
        started = time.perf_counter()
        score = scorer.score_pair(_pair_payload(first), _pair_payload(second))
        latencies.append((time.perf_counter() - started) * 1000.0)
        reverse = scorer.score_pair(_pair_payload(second), _pair_payload(first))
        by_id = {item.candidate_key: item for item in score.predictions}
        reverse_by_id = {item.candidate_key: item for item in reverse.predictions}
        swap_errors.append(
            max(
                abs(item.deployment_score - reverse_by_id[key].deployment_score)
                for key, item in by_id.items()
            )
        )
        for candidate in pair.candidates:
            if not candidate.safety_observed:
                continue
            prediction = by_id[candidate.candidate_key]
            hazard_correct["collision"] += int(
                (prediction.collision_probability >= 0.5) == candidate.collision
            )
            hazard_correct["red_light"] += int(
                (prediction.red_light_probability >= 0.5) == candidate.red_light_violation
            )
            hazard_correct["offroad"] += int(
                (prediction.offroad_probability >= 0.5) == candidate.offroad
            )
            hazard_correct["trust"] += int(
                (prediction.trust_probability >= 0.5) == candidate.trust
            )
            hazard_total += 1
        if not h6_policy_calibration or id(pair) in h6_calibration_tick_ids:
            vla = next(item for item in pair.candidates if item.source == "vla")
            expert = next(item for item in pair.candidates if item.source == "expert")
            calibration_rows.append(
                CalibrationRow(
                    pair_id=f"{pair.pair_id}:{pair.arm}:{pair.tick}",
                    vla_prediction=by_id[vla.candidate_key],
                    expert_prediction=by_id[expert.candidate_key],
                    vla_unsafe=vla.hard_unsafe,
                    expert_unsafe=expert.hard_unsafe,
                    vla_progress_m=vla.progress_m,
                    expert_progress_m=expert.progress_m,
                    raw_preference=(
                        getattr(by_id[vla.candidate_key], "preference_utility", 0.0)
                        >= getattr(by_id[expert.candidate_key], "preference_utility", 0.0)
                        if args.contract == "vla75-v2"
                        else None
                    ),
                    executable=getattr(vla, "executable", None),
                    applied_source=pair.executed_source,
                    phase=vla.phase,
                    group_key=vla.group_key,
                    sequence_id=(
                        f"{pair.pair_id}:{pair.arm}"
                        if pair.arm is not None
                        else None
                    ),
                    tick=pair.tick,
                )
            )
    policy_calibration_rows = []
    for pair in h6_policy_calibration:
        first, second = pair.candidates
        score = scorer.score_pair(_pair_payload(first), _pair_payload(second))
        by_id = {item.candidate_key: item for item in score.predictions}
        vla = next(item for item in pair.candidates if item.source == "vla")
        expert = next(item for item in pair.candidates if item.source == "expert")
        policy_calibration_rows.append(
            CalibrationRow(
                pair_id=pair.pair_id,
                vla_prediction=by_id[vla.candidate_key],
                expert_prediction=by_id[expert.candidate_key],
                vla_unsafe=vla.hard_unsafe,
                expert_unsafe=expert.hard_unsafe,
                vla_progress_m=vla.progress_m,
                expert_progress_m=expert.progress_m,
                raw_preference=(
                    getattr(by_id[vla.candidate_key], "preference_utility", 0.0)
                    >= getattr(by_id[expert.candidate_key], "preference_utility", 0.0)
                    if args.contract == "vla75-v2"
                    else None
                ),
                executable=getattr(vla, "executable", None),
                applied_source=(
                    "vla" if pair.actual_vla_coverage >= 0.75 else "expert"
                ),
                phase=vla.phase,
                group_key=vla.group_key,
                sequence_id=None,
                tick=None,
            )
        )
    temperature_rows = []
    for pair in val:
        # ``val`` already contains only development/calibration rows.  Each
        # hazard head is calibrated independently; unobserved repair labels
        # are omitted rather than turned into synthetic negatives.
        first, second = pair.candidates
        score = scorer.score_pair(_pair_payload(first), _pair_payload(second))
        predictions = {item.candidate_key: item for item in score.predictions}
        for candidate in pair.candidates:
            prediction = predictions[candidate.candidate_key]
            row = {
                "collision_logit": prediction.collision_logit,
                "collision": candidate.collision,
                "red_logit": prediction.red_light_logit,
                "red_light_violation": candidate.red_light_violation,
                "offroad_logit": prediction.offroad_logit,
                "offroad": candidate.offroad,
                "trust_logit": prediction.trust_logit,
                "trust": candidate.trust,
            }
            if candidate.repair_success is not None:
                row.update(
                    repair_logit=prediction.repair_success_logit,
                    repair_success=candidate.repair_success,
                )
            temperature_rows.append(row)

    # Deployment threshold calibration is deliberately computed before the
    # serialized payload is assembled.  It uses only the development rows
    # (plus the separate paired-policy calibration rows), never a formal
    # matrix or seed 101.
    calibration = calibrate_vla_deployment(
        calibration_rows,
        policy_rows=policy_calibration_rows or None,
        # Outcome attribution purity remains the original 90% requirement;
        # VLA75's 75% target is enforced on applied closed-loop evidence.
        target_vla_coverage=0.90,
        max_risk_ceiling=(0.20 if args.contract == "vla75-v2" else 1.0),
    )
    temperature_calibration = (
        calibrate_temperature_heads(temperature_rows)
        if args.contract == "vla75-v2"
        else None
    )
    router_calibration = None
    if args.contract == "vla75-v2":
        # Router calibration must replay the actual on-arm tick sequence.
        # Whole-policy rows have one aggregate coverage number and therefore
        # cannot identify EMA/hold/hysteresis behavior; using them here would
        # make every grid point report the same zero-switch result.
        router_rows = [
            row
            for row in calibration_rows
            if row.sequence_id is not None and row.tick is not None
        ]
        router_calibration = (
            select_vla75_router_config(router_rows)
            if router_rows
            else None
        )
    calibration_payload = calibration.to_dict()
    if temperature_calibration is not None:
        # Runtime scorers consume the same bounded temperatures that readiness
        # and the run-lock record, so calibration cannot silently diverge from
        # the model used during formal collection.
        calibration_payload["temperatures"] = dict(
            temperature_calibration.temperatures
        )
    actual_calibration_vla_coverage = (
        sum(item.actual_vla_coverage for item in h6_policy_calibration)
        / len(h6_policy_calibration)
        if h6_policy_calibration
        else None
    )
    # Attribution purity is about whether an observed label is attached to
    # the source that actually controlled that tick, not about how many
    # counterfactual candidates are intentionally masked.  The latter is
    # expected to be roughly half of a paired, source-blind batch and must not
    # make an honest dataset fail its 90% purity contract.
    purity_source_pairs = (
        list(h6_train)
        + [item.as_outcome_pair(split="h6_train_policy") for item in h6_policy_train]
        if args.contract == "vla75-v2"
        else list(train)
    )
    observed_candidates = [
        (pair, candidate)
        for pair in purity_source_pairs
        for candidate in pair.candidates
        if candidate.outcome_observed
    ]
    attributed_outcomes = sum(
        int(
            candidate.source in {"expert", "vla"}
            and (
                candidate.executable is True
                or (
                    args.contract != "vla75-v2"
                    and candidate.executable is None
                )
            )
        )
        for _pair, candidate in observed_candidates
    )
    observed_outcomes = len(observed_candidates)
    attribution_purity = attributed_outcomes / max(1, observed_outcomes)
    legacy_label_fraction = len(
        [candidate for pair in train for candidate in pair.candidates if candidate.outcome_observed]
    ) / max(1, 2 * len(train))
    payload = {
        "schema_version": (
            "safedrive.world.vla75.training_summary.v2"
            if args.contract == "vla75-v2"
            else "safedrive.world.v3.training_summary.v1"
        ),
        "contract": args.contract,
        "formal_lineage": args.formal_lineage,
        "train_pairs": len(train),
        "val_pairs": len(val),
        "train_splits": sorted({pair.split for pair in train}),
        "calibration_split": sorted({pair.split for pair in val}),
        "locked_test_loaded": False,
        "data_roots": [str(path) for path in args.data_root],
        "h6_roots": [str(path) for path in args.h6_root],
        "h6_train_seeds": [H6_TRAIN_SEEDS[0]],
        "h6_calibration_seeds": [H6_TRAIN_SEEDS[1]],
        "h6_train_pairs": len({pair.pair_id for pair in h6_train}),
        "h6_calibration_pairs": len({pair.pair_id for pair in h6_val}),
        "h6_train_tick_rows": len(h6_train),
        "h6_calibration_tick_rows": len(h6_val),
        "h6_train_policy_rows": len(h6_policy_train),
        "h6_calibration_policy_rows": len(h6_policy_calibration),
        "h6_train_pair_ids": sorted({pair.pair_id for pair in h6_train}),
        "h6_calibration_pair_ids": sorted(
            {pair.pair_id for pair in h6_policy_calibration}
        ),
        "h6_calibration_actual_vla_coverage": actual_calibration_vla_coverage,
        "training_outcome_attribution_purity": (
            attribution_purity
            if args.contract == "vla75-v2"
            else legacy_label_fraction
        ),
        "calibration_label_scope": (
            "h6_tickwise_scores_plus_paired_policy_outcomes"
            if h6_policy_calibration
            else "fully_observed_legacy_pairs"
        ),
        "h6_acceptance_seeds_loaded": False,
        "split_manifest": None if args.split_manifest is None else str(args.split_manifest),
        "split_manifest_sha256": (
            None if args.split_manifest is None else stable_sha256(manifest)
        ),
        "source_identity_is_model_input": False,
        "train_lineage_sha256": train_lineage_sha256,
        "validation_lineage_sha256": validation_lineage_sha256,
        "training_config_sha256": training_config_sha256,
        "code_sha256": code_sha256,
        "worktree_sha256": worktree_sha256,
        "models": results,
        "calibration": calibration_payload,
        "validation": {
            "hazard_accuracy": {
                key: value / max(1, hazard_total) for key, value in hazard_correct.items()
            },
            "swap_max_error": max(swap_errors, default=0.0),
            "latency_ms": {
                "p50": statistics.median(latencies),
                "p99": _percentile(latencies, 0.99),
                "max": max(latencies, default=0.0),
            },
        },
    }
    if args.contract == "vla75-v2":
        payload["seeds_loaded"] = list(H6_TRAIN_SEEDS)
        payload["model_seeds"] = list(requested_seeds)
        payload["h6_training_outcome_attribution_purity"] = attribution_purity
        payload["observed_outcome_labels"] = observed_outcomes
        payload["temperature_calibration"] = (
            None if temperature_calibration is None else temperature_calibration.to_dict()
        )
        payload["router_calibration"] = (
            None if router_calibration is None else router_calibration.to_dict()
        )
        payload["evaluators"] = evaluator_rows
        payload["calibration"]["c1_bindings"] = {
            "evaluator_sha256": [item["sha256"] for item in evaluator_rows],
            "validation_lineage_sha256": validation_lineage_sha256,
            "training_input_sha256": train_lineage_sha256,
        }
        payload = finalize_training_summary_v2(payload)
    else:
        payload["evidence_sha256"] = stable_sha256(payload)
    summary = args.output_dir / "training-summary.json"
    summary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True if args.contract == "vla75-v2" else calibration.passed,
                "summary": str(summary),
                "train_pairs": len(train),
                "val_pairs": len(val),
                "calibration": calibration.to_dict(),
                "calibration_is_diagnostic": args.contract == "vla75-v2",
            },
            sort_keys=True,
        )
    )
    return 0 if args.contract == "vla75-v2" or calibration.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
