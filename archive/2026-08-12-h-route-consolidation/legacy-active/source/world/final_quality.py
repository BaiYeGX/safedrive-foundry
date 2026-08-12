"""Hard acceptance gates for the R3 final-head ActionBranchDatasetV1."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ActionBranchSample, WorldContractError, content_hash
from .dataset import ActionBranchDataset, quality_report
from .action_signal import action_signal_report
from .baselines import observable_rule_scores, predict_actor_future

FINAL_NAMESPACE = "r3_final_head_formal"
FINAL_FAMILIES = frozenset(
    {"lead_braking", "cut_in", "crossing", "merge", "obstruction", "clear"}
)


def _split_map(dataset: ActionBranchDataset) -> dict[str, str]:
    return {
        str(record["identity"]["sample_id"]): str(record["split"])
        for record in dataset._records  # noqa: SLF001 - validated immutable source index
    }


def validate_final_dataset(
    dataset: ActionBranchDataset,
    *,
    checkpoint_sha256: str,
    namespace: str = FINAL_NAMESPACE,
    aa_p99: float | None = None,
) -> dict[str, Any]:
    expected_hash = str(checkpoint_sha256).lower()
    if len(expected_hash) != 64:
        raise WorldContractError("final R3 checkpoint hash must be 64 hex characters")
    manifest_namespace = str(dataset.manifest.get("namespace") or "")
    manifest_hash = str(dataset.manifest.get("r2_checkpoint_sha256") or "").lower()
    samples = [dataset[index].sample for index in range(len(dataset))]
    split_by_sample = _split_map(dataset)
    base = quality_report(
        samples,
        split_by_sample,
        min_comparable=1000,
        min_decisive=250,
        min_wins_per_slot=100,
        min_future_coverage=0.95,
        min_test_samples=288,
        min_test_dual=168,
        min_test_decisive=60,
        min_test_wins_per_slot=16,
    )
    records = list(dataset._records)  # noqa: SLF001
    sample_ids = [sample.identity.sample_id for sample in samples]
    groups: dict[str, set[str]] = {}
    for sample in samples:
        groups.setdefault(sample.identity.group_key, set()).add(
            split_by_sample[sample.identity.sample_id]
        )
    overlap = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    decisive = [sample for sample in samples if sample.comparable and sample.rank_mask and not sample.tie_target]
    wins = {
        "candidate_0": sum(sample.rank_target > 0 for sample in decisive),
        "candidate_1": sum(sample.rank_target < 0 for sample in decisive),
    }
    split_quality: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        subset = [sample for sample in samples if split_by_sample[sample.identity.sample_id] == split]
        subset_decisive = [
            sample for sample in subset if sample.comparable and sample.rank_mask and not sample.tie_target
        ]
        split_quality[split] = {
            "completed": len(subset),
            "dual": sum(int(sample.candidate_mask.all()) for sample in subset),
            "comparable": sum(bool(sample.comparable) for sample in subset),
            "decisive": len(subset_decisive),
            "winner_candidate_0": sum(sample.rank_target > 0 for sample in subset_decisive),
            "winner_candidate_1": sum(sample.rank_target < 0 for sample in subset_decisive),
        }
    controller_counts = Counter(
        str(sample.audit.get("actor_controller_kind") or "unknown") for sample in samples
    )
    family_coverage: dict[str, dict[str, bool]] = {}
    for split in ("train", "val", "test"):
        subset = [sample for sample in samples if split_by_sample[sample.identity.sample_id] == split]
        family_coverage[split] = {
            family: bool(
                any(sample.identity.family == family for sample in subset)
                and any(sample.identity.family == family and sample.comparable for sample in subset)
                and any(
                    sample.identity.family == family
                    and sample.comparable
                    and sample.rank_mask
                    and not sample.tie_target
                    for sample in subset
                )
                and any(
                    sample.identity.family == family
                    and sample.comparable
                    and sample.rank_target > 0
                    for sample in subset
                )
                and any(
                    sample.identity.family == family
                    and sample.comparable
                    and sample.rank_target < 0
                    for sample in subset
                )
            )
            for family in sorted(FINAL_FAMILIES)
        }
    binding_ok = all(
        sample.identity.model_hash.lower() == expected_hash
        and str(sample.audit.get("namespace") or "") == str(namespace)
        and str(sample.audit.get("r2_checkpoint_sha256") or "").lower() == expected_hash
        and str(sample.audit.get("artifact_schema") or "")
        in {"safedrive.k2_anchor.v3", "safedrive.k2_anchor.v4"}
        for sample in samples
    )
    repeat_aa_binding_ok = all(
        bool(str(sample.audit.get("repeat_group") or ""))
        and str(sample.audit.get("aa_noise_identity") or "").lower()
        == content_hash(
            {
                "namespace": "r3_aa_noise_probe",
                "repeat_group": str(sample.audit.get("repeat_group") or ""),
                "candidate_id": "v3_nominal_progress",
            }
        ).lower()
        for sample in samples
    )
    def _branch_hash_binding(sample: ActionBranchSample) -> bool:
        expected_keys = {str(index) for index, available in enumerate(sample.candidate_mask) if bool(available)}
        summary_hashes = dict(sample.audit.get("branch_summary_sha256") or {})
        future_hashes = dict(sample.audit.get("branch_future_trace_sha256") or {})
        scene_hashes = dict(sample.audit.get("observable_scene_sha256") or {})
        return (
            len(str(sample.audit.get("artifact_file_sha256") or "")) == 64
            and expected_keys.issubset(summary_hashes)
            and expected_keys.issubset(future_hashes)
            and expected_keys.issubset(scene_hashes)
            and all(len(str(value)) == 64 for value in (*summary_hashes.values(), *future_hashes.values(), *scene_hashes.values()))
        )

    artifact_branch_binding_ok = all(_branch_hash_binding(sample) for sample in samples)
    finite_ok = all(sample.validate() is None for sample in samples)
    runtime_leakage = sum(
        int(sample.audit.get("scenario_family_runtime_use", 0))
        + int(sample.audit.get("semantic_rescue_count", 0))
        for sample in samples
    )
    fixed_ratio = controller_counts["fixed"] / max(len(samples), 1)
    reactive_ratio = controller_counts["reactive"] / max(len(samples), 1)
    aa_values = {
        float(sample.audit["aa_p99"])
        for sample in samples
        if sample.audit.get("aa_p99") is not None
    }
    if aa_p99 is None and len(aa_values) == 1:
        aa_p99 = next(iter(aa_values))
    action_signal = (
        action_signal_report(samples, aa_p99=float(aa_p99))
        if aa_p99 is not None
        else {
            "gate_reactive_fraction": False,
            "gate_reactive_decisive_fraction": False,
            "gate_has_observable_signal": False,
            "aa_p99": None,
        }
    )
    val_samples = [
        sample
        for sample in samples
        if split_by_sample[sample.identity.sample_id] == "val"
        and sample.rank_mask
        and not sample.tie_target
    ]
    # Rebuild every deterministic baseline from the frozen Observable tensors.
    # Keep both future ADE and the candidate-conditioned selection report so a
    # reviewer can reproduce persistence/CV/CTRV, no-action and action-swap
    # numbers without consulting Oracle labels at runtime.
    baseline_suite: dict[str, dict[str, Any]] = {}
    for mode in ("persistence", "cv", "ctrv"):
        distance_sum = 0.0
        valid_points = 0
        rule_correct = 0
        rule_count = 0
        reward_margin_sum = 0.0
        for sample in samples:
            prediction, _ = predict_actor_future(sample, mode=mode)
            deltas = np.linalg.norm(
                prediction[..., :2] - sample.actor_future[..., :2], axis=-1
            )
            valid = sample.actor_future_mask
            distance_sum += float(deltas[valid].sum())
            valid_points += int(valid.sum())
            if (
                split_by_sample[sample.identity.sample_id] == "val"
                and sample.rank_mask
                and not sample.tie_target
            ):
                scores = observable_rule_scores(sample, prediction)
                selected = int(np.argmax(scores))
                target = 0 if sample.rank_target > 0 else 1
                rule_correct += int(selected == target)
                reward_margin_sum += float(abs(scores[0] - scores[1]))
                rule_count += 1
        baseline_suite[mode] = {
            "actor_ADE_m": distance_sum / max(valid_points, 1),
            "actor_valid_points": valid_points,
            "val_decisive": rule_count,
            "observable_reward_accuracy": rule_correct / max(rule_count, 1),
            "reward_margin_mean": reward_margin_sum / max(rule_count, 1),
        }

    baseline_correct = 0
    baseline_no_action = 0
    baseline_swap = 0
    for sample in val_samples:
        prediction, _ = predict_actor_future(sample, mode="cv")
        scores = observable_rule_scores(sample, prediction)
        selected = int(np.argmax(scores))
        target = 0 if sample.rank_target > 0 else 1
        baseline_correct += int(selected == target)
        baseline_no_action += int(target == 0)
        baseline_swap += int((1 - selected) == target)
    val_decisive = len(val_samples)
    baseline_report = {
        "val_decisive": val_decisive,
        "observable_reward_accuracy": baseline_correct / max(1, val_decisive),
        "no_action_accuracy": baseline_no_action / max(1, val_decisive),
        "action_swap_accuracy": baseline_swap / max(1, val_decisive),
        "action_swap_drop": (baseline_correct - baseline_swap) / max(1, val_decisive),
        "reconstructed_baselines": baseline_suite,
    }
    hard = {
        "namespace": manifest_namespace == str(namespace),
        "checkpoint_binding": manifest_hash == expected_hash,
        "completed_minimum": len(samples) >= 2000,
        "comparable_minimum": base["comparable_count"] >= 1000,
        "decisive_minimum": base["decisive_count"] >= 250,
        "candidate_wins_each_minimum": min(wins.values()) >= 100,
        "test_completed_exact": split_quality["test"]["completed"] == 288,
        "test_dual_minimum": split_quality["test"]["dual"] >= 168,
        "test_comparable_minimum": split_quality["test"]["comparable"] >= 144,
        "test_decisive_minimum": split_quality["test"]["decisive"] >= 60,
        "test_wins_each_minimum": min(
            split_quality["test"]["winner_candidate_0"],
            split_quality["test"]["winner_candidate_1"],
        )
        >= 16,
        "schema_hash_artifact_binding": binding_ok,
        "repeat_group_aa_identity_binding": repeat_aa_binding_ok,
        "artifact_observation_branch_hash_binding": artifact_branch_binding_ok,
        "finite_on_valid_mask": finite_ok,
        "duplicate_sample_identity": len(sample_ids) == len(set(sample_ids)),
        "split_root_lineage_overlap": not overlap,
        "runtime_oracle_namespace_leakage": runtime_leakage == 0,
        "fixed_reactive_balance": 0.45 <= fixed_ratio <= 0.55
        and 0.45 <= reactive_ratio <= 0.55,
        "family_coverage": all(all(values.values()) for values in family_coverage.values()),
        "actor_future_coverage": float(base["future_coverage"]) >= 0.95,
        "reactive_action_response_fraction": bool(action_signal["gate_reactive_fraction"]),
        "reactive_decisive_action_response_fraction": bool(
            action_signal["gate_reactive_decisive_fraction"]
        ),
        "candidate_conditioned_observable_accuracy": baseline_report[
            "observable_reward_accuracy"
        ]
        >= 0.60,
        "candidate_conditioned_beats_no_action": baseline_report[
            "observable_reward_accuracy"
        ]
        - baseline_report["no_action_accuracy"]
        >= 0.05,
        "candidate_shuffle_drop": baseline_report["action_swap_drop"] >= 0.05,
    }
    return {
        "schema_version": "safedrive.r3.final_quality.v1",
        "namespace": str(namespace),
        "checkpoint_sha256": expected_hash,
        "sample_count_completed": len(samples),
        "comparable_count": base["comparable_count"],
        "decisive_count": base["decisive_count"],
        "winner_candidate_0": wins["candidate_0"],
        "winner_candidate_1": wins["candidate_1"],
        "controller_counts": dict(controller_counts),
        "split_quality": split_quality,
        "family_coverage": family_coverage,
        "group_overlap": overlap,
        "base_quality": base,
        "action_signal": action_signal,
        "baseline_report": baseline_report,
        "baseline_suite": baseline_suite,
        "hard_gates": hard,
        "all_hard_gates_pass": all(bool(value) for value in hard.values()),
    }


__all__ = ["FINAL_FAMILIES", "FINAL_NAMESPACE", "validate_final_dataset"]
