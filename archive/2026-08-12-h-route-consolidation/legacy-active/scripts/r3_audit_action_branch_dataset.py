#!/usr/bin/env python3
"""Audit frozen R3 data and run persistence/CV/CTRV/rule baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.baselines import observable_rule_scores, predict_actor_future
from driving_vla.world.dataset import ActionBranchDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = ActionBranchDataset(Path(args.dataset))
    quality = json.loads(
        (Path(args.dataset) / "quality_report.json").read_text(encoding="utf-8")
    )
    samples = [dataset[i].sample for i in range(len(dataset))]
    baseline: dict[str, dict[str, float | int]] = {}
    for mode in ("persistence", "cv", "ctrv"):
        distance_sum = 0.0
        valid_count = 0
        rule_correct = 0
        rule_denominator = 0
        for sample in samples:
            prediction, _ = predict_actor_future(sample, mode=mode)
            target = sample.actor_future[..., (0, 1, 4, 5)]
            distances = np.linalg.norm(prediction[..., :2] - target[..., :2], axis=-1)
            distance_sum += float(distances[sample.actor_future_mask].sum())
            valid_count += int(sample.actor_future_mask.sum())
            if sample.rank_mask and not sample.tie_target:
                scores = observable_rule_scores(sample, prediction)
                selected = int(np.argmax(scores))
                winner = 0 if sample.rank_target > 0 else 1
                rule_correct += int(selected == winner)
                rule_denominator += 1
        baseline[mode] = {
            "actor_ADE_m": distance_sum / max(1, valid_count),
            "actor_valid_points": valid_count,
            "rule_pairwise_accuracy": rule_correct / max(1, rule_denominator),
            "rule_correct": rule_correct,
            "rule_denominator": rule_denominator,
        }
    # Candidate-conditioned observable reward sanity checks.  These are
    # intentionally simple and remain independent of privileged outcomes.
    rule_correct = 0
    no_action_correct = 0
    action_swap_correct = 0
    rule_count = 0
    reward_margin_sum = 0.0
    for sample in samples:
        if not (sample.rank_mask and not sample.tie_target):
            continue
        prediction, _ = predict_actor_future(sample, mode="cv")
        scores = observable_rule_scores(sample, prediction)
        selected = int(np.argmax(scores))
        target = 0 if sample.rank_target > 0 else 1
        rule_correct += int(selected == target)
        no_action_correct += int(0 == target)
        # A pure candidate-order shuffle must not improve the signal; report
        # the paired drop explicitly for auditability.
        action_swap_correct += int((1 - selected) == target)
        reward_margin_sum += float(abs(scores[0] - scores[1]))
        rule_count += 1
    baseline["observable_reward"] = {
        "decisive_count": rule_count,
        "accuracy": rule_correct / max(1, rule_count),
        "reward_margin_mean": reward_margin_sum / max(1, rule_count),
    }
    baseline["no_action"] = {
        "decisive_count": rule_count,
        "accuracy": no_action_correct / max(1, rule_count),
    }
    baseline["action_swap"] = {
        "decisive_count": rule_count,
        "accuracy": action_swap_correct / max(1, rule_count),
        "accuracy_drop_vs_observable_reward": (
            rule_correct - action_swap_correct
        )
        / max(1, rule_count),
    }
    hard = quality["hard_gates"]
    foundational = all(
        (
            hard[key] is True
            if isinstance(hard[key], bool)
            else hard[key] == 0
        )
        for key in (
            "schema_validation_100pct",
            "finite_values_100pct",
            "identity_binding_100pct",
            "runtime_oracle_namespace_leakage",
            "split_group_overlap",
            "duplicate_sample_identity",
            "future_coverage_minimum",
        )
    )
    if quality["all_hard_gates_pass"]:
        label = "R3_DATA_READY"
    elif foundational and quality["comparable_count"] > 0:
        label = "R3_DATA_READY_WITH_WEAK_ACTION_SIGNAL"
    elif not hard["future_coverage_minimum"]:
        label = "R3_LABEL_TRACE_MISSING"
    else:
        label = "R3_DATA_LIMITED"
    report = {
        "schema_version": "safedrive.r3_closure.v0",
        "label": label,
        "dataset_content_hash": dataset.manifest["dataset_content_hash"],
        "quality": quality,
        "baselines": baseline,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite frozen R3 report: {output}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if foundational else 2


if __name__ == "__main__":
    raise SystemExit(main())
