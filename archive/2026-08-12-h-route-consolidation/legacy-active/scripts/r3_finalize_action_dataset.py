#!/usr/bin/env python3
"""Freeze the final-head R3 ActionBranchDatasetV1 and all reproducible audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.baselines import observable_rule_scores, predict_actor_future  # noqa: E402
from driving_vla.world.dataset import (  # noqa: E402
    sample_from_attempt,
    write_dataset,
    ActionBranchDatasetV1,
)
from driving_vla.world.final_quality import validate_final_dataset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attempts(roots: list[Path]) -> list[Path]:
    values: set[Path] = set()
    for root in roots:
        if "r3_r2_blind_holdout" in str(root) or "r2v4_blind_audit" in str(root):
            raise ValueError(f"holdout namespace cannot enter final R3 dataset: {root}")
        for manifest in root.rglob("pair_manifest.json"):
            report_path = manifest.parent / "pair_report.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if str(report.get("namespace") or "") != "r3_final_head_formal":
                continue
            if str(report.get("status", "COMPLETED")).upper() == "FAILED":
                continue
            if not (manifest.parent / "branch-0/oracle/actor_future_trace.jsonl").is_file():
                continue
            values.add(manifest.parent)
    return sorted(values)


def _baseline_report(dataset: ActionBranchDatasetV1) -> dict[str, Any]:
    baseline: dict[str, Any] = {}
    samples = [dataset[index].sample for index in range(len(dataset))]
    for mode in ("persistence", "cv", "ctrv"):
        distance = 0.0
        valid = 0
        for sample in samples:
            prediction, _ = predict_actor_future(sample, mode=mode)
            deltas = np.linalg.norm(
                prediction[..., :2] - sample.actor_future[..., :2], axis=-1
            )
            distance += float(deltas[sample.actor_future_mask].sum())
            valid += int(sample.actor_future_mask.sum())
        baseline[mode] = {"actor_ADE_m": distance / max(valid, 1), "valid_points": valid}
    decisive = [sample for sample in samples if sample.rank_mask and not sample.tie_target]
    correct = no_action = swap = 0
    for sample in decisive:
        prediction, _ = predict_actor_future(sample, mode="cv")
        selected = int(np.argmax(observable_rule_scores(sample, prediction)))
        target = 0 if sample.rank_target > 0 else 1
        correct += int(selected == target)
        no_action += int(target == 0)
        swap += int((1 - selected) == target)
    baseline["observable_reward"] = {"accuracy": correct / max(1, len(decisive)), "decisive": len(decisive)}
    baseline["no_action"] = {"accuracy": no_action / max(1, len(decisive)), "decisive": len(decisive)}
    baseline["action_swap"] = {
        "accuracy": swap / max(1, len(decisive)),
        "accuracy_drop": (correct - swap) / max(1, len(decisive)),
        "decisive": len(decisive),
    }
    return baseline


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    campaign_path = Path(args.campaign_manifest).resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    development_gate_path = Path(args.development_gate).resolve()
    development_gate = json.loads(development_gate_path.read_text(encoding="utf-8"))
    if (
        str(development_gate.get("schema_version") or "")
        != "safedrive.r3.development_gate.v1"
        or not bool(development_gate.get("passed"))
    ):
        raise ValueError("R3 finalization requires a passing pre-frozen 512-slot development gate")
    if str(development_gate.get("source_manifest_hash") or "") != str(campaign.get("manifest_hash") or ""):
        raise ValueError("R3 development gate is bound to a different campaign manifest")
    aa_report_path = Path(args.aa_report).resolve()
    aa_report = json.loads(aa_report_path.read_text(encoding="utf-8"))
    if str(aa_report.get("schema_version") or "") != "safedrive.r3.aa_noise_report.v1":
        raise ValueError("R3 finalization requires the frozen A/A noise report")
    if int(aa_report.get("repeat_count", 0)) != 168:
        raise ValueError("R3 A/A noise report must contain all 168 repeats")
    aa_value = float(aa_report.get("aa_p99"))
    if not np.isfinite(aa_value) or abs(aa_value - float(args.aa_p99)) > 1.0e-9:
        raise ValueError("--aa-p99 does not match the frozen A/A noise report")
    checkpoint = str(args.r2_checkpoint_sha256).lower()
    if len(checkpoint) != 64 or any(char not in "0123456789abcdef" for char in checkpoint):
        raise ValueError("final R3 requires a 64-hex formal R2 checkpoint hash")
    attempts = _attempts([Path(value).resolve() for value in args.evidence_root])
    if not attempts:
        raise ValueError("no completed final-head attempts with actor futures")
    campaign_hash = str(campaign.get("manifest_hash") or "")
    if len(campaign_hash) != 64:
        raise ValueError("R3 campaign manifest hash is missing")
    # A matching scenario_id is not sufficient for a resumable formal sample:
    # the pair report itself must prove that this exact pre-frozen campaign and
    # final R2 checkpoint produced it.
    for attempt in attempts:
        report = json.loads((attempt / "pair_report.json").read_text(encoding="utf-8"))
        if str(report.get("source_manifest_hash") or "") != campaign_hash:
            raise ValueError(f"final R3 attempt is bound to a different campaign: {attempt}")
        if str(report.get("checkpoint_sha256") or "").lower() != checkpoint:
            raise ValueError(f"final R3 attempt is bound to a different R2 checkpoint: {attempt}")
        if not bool(report.get("actor_future_sidecar")):
            raise ValueError(f"final R3 attempt lacks the actor-future sidecar: {attempt}")
    samples = [sample_from_attempt(path) for path in attempts]
    if any(str(sample.audit.get("namespace")) != "r3_final_head_formal" for sample in samples):
        raise ValueError("final dataset contains a non-final namespace")
    scenario_ids = [sample.identity.scenario_id for sample in samples]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("final R3 dataset contains duplicate scenario slot attempts")
    aa_checkpoint = str(aa_report.get("checkpoint_sha256") or "").lower()
    if aa_checkpoint and aa_checkpoint != checkpoint:
        raise ValueError("A/A noise report is bound to a different R2 checkpoint")
    aa_groups = {str(value) for value in aa_report.get("repeat_groups") or ()}
    if aa_groups and any(str(sample.audit.get("repeat_group") or "") not in aa_groups for sample in samples):
        raise ValueError("final dataset contains a repeat_group without a frozen A/A probe")
    for sample in samples:
        sample.audit = {**dict(sample.audit), "aa_p99": float(args.aa_p99)}
    campaign_rows = list(campaign.get("slots", [])) + list(campaign.get("reserve_slots", []))
    scenario_splits = {str(row["scenario_id"]): str(row["split"]) for row in campaign_rows}
    expected_test_ids = {
        str(row["scenario_id"])
        for row in campaign.get("slots", [])
        if str(row.get("split")) == "test"
    }
    development_ids = {str(row["scenario_id"]) for row in campaign.get("development_slots", [])}
    if len(samples) > 2520:
        raise ValueError("R3 final hard cap exceeded: more than 2520 completed slots")
    missing = [sample.identity.scenario_id for sample in samples if sample.identity.scenario_id not in scenario_splits]
    if missing:
        raise ValueError(f"campaign split missing scenarios: {missing[:3]}")
    collected_ids = {sample.identity.scenario_id for sample in samples}
    collected_test_ids = {
        scenario_id
        for scenario_id in collected_ids
        if scenario_splits.get(scenario_id) == "test"
    }
    if expected_test_ids != collected_test_ids:
        raise ValueError(
            "R3 final dataset must execute every pre-frozen Town13 test slot: "
            f"missing={len(expected_test_ids - collected_test_ids)} "
            f"extra={len(collected_test_ids - expected_test_ids)}"
        )
    missing_development = sorted(development_ids - collected_ids)
    if missing_development:
        raise ValueError(
            "final R3 dataset is missing pre-frozen development slots: "
            + ", ".join(missing_development[:3])
        )
    split_by_sample = {
        sample.identity.sample_id: scenario_splits[sample.identity.scenario_id]
        for sample in samples
    }
    output = Path(args.output).resolve()
    manifest = write_dataset(
        samples,
        output,
        split_by_sample=split_by_sample,
        shard_size=int(args.shard_size),
        namespace="r3_final_head_formal",
        r2_checkpoint_sha256=checkpoint,
        quality_thresholds={
            "min_comparable": 1000,
            "min_decisive": 250,
            "min_wins_per_slot": 100,
            "min_future_coverage": 0.95,
            "min_test_samples": 288,
            "min_test_dual": 168,
            "min_test_decisive": 60,
            "min_test_wins_per_slot": 16,
        },
    )
    dataset = ActionBranchDatasetV1(output, verify_hashes=True)
    quality = validate_final_dataset(
        dataset,
        checkpoint_sha256=checkpoint,
        aa_p99=args.aa_p99,
    )
    baseline = _baseline_report(dataset)
    final = {
        "schema_version": "safedrive.r3.final_head_closure.v1",
        "status": "R3_ACTION_BRANCH_DATA_READY" if quality["all_hard_gates_pass"] else "R3_DATA_LIMITED",
        "dataset": str(output),
        "dataset_manifest_sha256": _sha(output / "dataset_manifest.json"),
        "campaign_manifest_sha256": _sha(campaign_path),
        "development_gate_sha256": _sha(development_gate_path),
        "aa_noise_report_sha256": _sha(aa_report_path),
        "r2_checkpoint_sha256": checkpoint,
        "quality": quality,
        "baselines": baseline,
        "r4_training": "NOT_STARTED",
    }
    (output / "final_quality_report.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--development-gate", required=True)
    parser.add_argument("--evidence-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--r2-checkpoint-sha256", required=True)
    parser.add_argument("--aa-p99", type=float, required=True)
    parser.add_argument("--aa-report", required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()
    report = finalize(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["quality"]["all_hard_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
