#!/usr/bin/env python3
"""One-shot frozen evaluation and closure report for World-V0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.checkpoint import load_checkpoint
from driving_vla.world.contracts import WorldBatch
from driving_vla.world.dataset import ActionBranchDataset
from driving_vla.world.metrics import (
    action_sensitivity,
    candidate_swap_error,
    paired_ranking_bootstrap,
)
from driving_vla.world.training import predict_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--no-action-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--baseline-report", default="")
    args = parser.parse_args()
    dataset = ActionBranchDataset(Path(args.dataset))
    samples = list(dataset.iter_split("test"))
    if not samples:
        raise SystemExit("frozen test split is empty")
    model, manifest = load_checkpoint(Path(args.checkpoint), device=args.device)
    if manifest["data_hash"] != dataset.manifest["dataset_content_hash"]:
        raise SystemExit("checkpoint/dataset hash mismatch")
    model.eval()
    metrics, predictions = predict_samples(model, samples, batch_size=args.batch_size)
    probe = WorldBatch.from_samples(samples[: min(8, len(samples))])
    swap = candidate_swap_error(model, probe)
    sensitivity = None
    no_action_metrics = None
    bootstrap = None
    if args.no_action_checkpoint:
        no_action, no_action_manifest = load_checkpoint(
            Path(args.no_action_checkpoint), device=args.device
        )
        if no_action_manifest["data_hash"] != dataset.manifest["dataset_content_hash"]:
            raise SystemExit("no-action checkpoint/dataset hash mismatch")
        no_action.eval()
        conditioned_prediction = model(probe)
        no_action_prediction = no_action(probe)
        sensitivity = action_sensitivity(conditioned_prediction, no_action_prediction)
        no_action_metrics, no_action_predictions = predict_samples(
            no_action, samples, batch_size=args.batch_size
        )
        bootstrap = paired_ranking_bootstrap(
            predictions, no_action_predictions, samples
        )
    contracts_ok = (
        swap["score_max_abs_error"] <= 1e-5
        and swap["future_max_abs_error"] <= 1e-5
        and int(manifest["parameter_count"]) >= 4_000_000
        and int(manifest["parameter_count"]) <= 8_000_000
    )
    rule_accuracy = 0.0
    if args.baseline_report:
        baseline_report = json.loads(
            Path(args.baseline_report).read_text(encoding="utf-8")
        )
        rule_accuracy = float(
            baseline_report.get("baselines", {})
            .get("cv", {})
            .get("rule_pairwise_accuracy", 0.0)
        )
    action_signal = (
        sensitivity is not None
        and sensitivity["dual_candidate_batches"] > 0
        and sensitivity["conditioned_slot_future_delta_mean"] > 1e-4
        and sensitivity["conditioned_vs_no_action_future_delta_mean"] > 1e-4
        and bootstrap is not None
        and int(bootstrap["decisive_count"]) > 0
        and float(bootstrap["improvement"]) >= 0.05
        and float(bootstrap["ci95_low"]) >= 0.0
        and float(bootstrap["conditioned_accuracy"]) >= rule_accuracy + 0.05
        and no_action_metrics is not None
        and float(metrics["actor_ADE_m"])
        <= 1.05 * float(no_action_metrics["actor_ADE_m"])
    )
    if not contracts_ok:
        label = "R4_WORLD_TRAINING_INVALID"
    elif not action_signal:
        label = "R4_WORLD_ACTION_SIGNAL_NOT_PROVEN"
    elif int(metrics.get("decisive_count", 0)) == 0:
        label = "R4_WORLD_V0_READY_WITH_LIMITS"
    else:
        label = "R4_WORLD_V0_READY"
    report = {
        "schema_version": "safedrive.r4_closure.v0",
        "label": label,
        "dataset_content_hash": dataset.manifest["dataset_content_hash"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "parameter_count": manifest["parameter_count"],
        "test_metrics": metrics,
        "no_action_metrics": no_action_metrics,
        "candidate_swap": swap,
        "action_sensitivity": sensitivity,
        "paired_ranking_bootstrap": bootstrap,
        "rule_baseline_accuracy": rule_accuracy,
        "contracts_ok": contracts_ok,
        "action_signal_proven": action_signal,
        "denominators": {
            "test_samples": len(samples),
            "decisive": int(metrics.get("decisive_count", 0)),
            "ties": int(metrics.get("tie_count", 0)),
            "outcomes": int(metrics.get("outcome_count", 0)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite frozen R4 report: {output}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if contracts_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
