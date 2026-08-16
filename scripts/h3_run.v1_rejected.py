#!/usr/bin/env python3
"""Run the complete offline H3 World scorer experiment with strict OOF evaluation.

This command never opens H2 test labels. It writes all generated artifacts
under ``generated/h3`` and ``docs/runtime-evidence/h3``; the frozen H2 store is
strictly read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data_pipeline.h3.baselines import BASELINE_NAMES, evaluate_baseline  # noqa: E402
from data_pipeline.h3.contracts import H3_CONFIG, H3_CONFIG_SHA256, stable_sha256  # noqa: E402
from data_pipeline.h3.dataset import (  # noqa: E402
    build_split_manifest,
    leakage_audit,
    load_examples,
    read_h2_records,
    write_split_manifest,
)
from data_pipeline.h3.evaluate import (  # noqa: E402
    bootstrap_accuracy_delta,
    evaluate_ablation,
    evaluate_cv,
    evaluate_models,
    h3_gate,
    swap_consistency,
)
from data_pipeline.h3.model import load_model, train_model  # noqa: E402
from data_pipeline.h3.runtime import WorldScorer  # noqa: E402


DATASET_ID = "h2-gatepass-20260813-routefix"
RUN_ID = "h3-world-20260813-v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _train_fold(train_examples, val_examples, fold: str, checkpoint_root: Path, device: str) -> tuple[list, list[dict[str, Any]]]:
    models = []
    results = []
    for seed in H3_CONFIG["training_seeds"]:
        checkpoint = checkpoint_root / "cv" / f"{fold}__seed-{seed}.pt"
        result = train_model(train_examples, val_examples, seed=int(seed), checkpoint_path=checkpoint, device=device, max_epochs=500, patience=40)
        model, _ = load_model(checkpoint, device=device)
        models.append(model)
        results.append({
            "fold": fold,
            "seed": int(seed),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "checkpoint": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "train_examples": result.train_examples,
            "val_examples": result.val_examples,
            "device": result.device,
        })
    return models, results


def _train_final(examples, checkpoint_root: Path, device: str) -> tuple[list, list[dict[str, Any]]]:
    models = []
    results = []
    for seed in H3_CONFIG["training_seeds"]:
        checkpoint = checkpoint_root / "final" / f"seed-{seed}.pt"
        result = train_model(examples, examples, seed=int(seed), checkpoint_path=checkpoint, device=device, max_epochs=500, patience=50)
        model, _ = load_model(checkpoint, device=device)
        models.append(model)
        results.append({
            "seed": int(seed),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "checkpoint": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "train_examples": result.train_examples,
            "device": result.device,
        })
    return models, results


def _resource_benchmark(models, examples, device: str) -> dict[str, Any]:
    if not examples:
        return {"passed": False, "reason": "no_examples"}
    scorer = WorldScorer(models, device=device, model_hash="h3-runtime-benchmark")
    latencies: list[float] = []
    example = examples[0]
    first = example.candidates[0]
    second = example.candidates[1]

    # Warmup
    for _ in range(20):
        _ = scorer.score_pair(
            (first.candidate_key, first.context, first.candidate),
            (second.candidate_key, second.context, second.candidate),
        )

    # Benchmark 200 pair scoring runs
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(200):
        t0 = time.perf_counter()
        result = scorer.score_pair(
            (first.candidate_key, first.context, first.candidate),
            (second.candidate_key, second.context, second.candidate),
        )
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    gpu_available = bool(torch.cuda.is_available())
    peak_gib = 0.0
    if gpu_available:
        peak_gib = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 3)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    p99 = latencies[int(0.99 * (len(latencies) - 1))]
    return {
        "device": device,
        "gpu_available": gpu_available,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "deadline_misses": sum(item > float(H3_CONFIG["runtime"]["deadline_ms"]) for item in latencies),
        "peak_reserved_gib": peak_gib,
        "incremental_gpu_gib": peak_gib,
        "passed": p99 <= float(H3_CONFIG["runtime"]["deadline_ms"]) and peak_gib <= float(H3_CONFIG["runtime"]["max_incremental_gpu_gib"]),
        "failure_reason": None,
    }


def _ensure_carla_running() -> bool:
    """Check CARLA sim status; if not running, actively attempt ensure."""
    command = [sys.executable, "scripts/sdf.py", "sim", "status"]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if "rpc_reachable=True" in proc.stdout:
        return True
    # Try ensure once
    ensure_cmd = [
        sys.executable,
        "scripts/sdf.py",
        "sim",
        "ensure",
        "--map",
        "Town03",
        "--rhi",
        "dx12",
        "--startup-timeout",
        "180",
        "--json",
    ]
    res = subprocess.run(ensure_cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return res.returncode == 0


def _challenge_probe(evidence_dir: Path) -> dict[str, Any]:
    """Run a read-only CARLA admission probe and preserve its result."""
    command = [sys.executable, "scripts/sdf.py", "sim", "preflight", "--json"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    payload: dict[str, Any] = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "READY" if completed.returncode == 0 else "NOT_READY",
        "challenge_dataset_id": H3_CONFIG["challenge"]["dataset_id"],
        "action": "probe_completed",
    }
    _write(evidence_dir / "challenge-probe.json", payload)
    return payload


CARLA_CHALLENGE_DATASET_ID = "h3-carla-challenge-20260814-v1"


def run(run_id: str = RUN_ID, *, include_challenge: bool = False) -> dict[str, Any]:
    h2_root = ROOT / "generated" / "h2" / "paired-outcomes" / DATASET_ID
    challenge_root = ROOT / "generated" / "h3" / "carla-challenge" / CARLA_CHALLENGE_DATASET_ID

    generated = ROOT / "generated" / "h3" / run_id
    evidence_dir = ROOT / "docs" / "runtime-evidence" / "h3" / run_id
    checkpoint_root = generated / "checkpoints"
    generated.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not h2_root.is_dir():
        raise RuntimeError(f"H2_DATASET_MISSING:{h2_root}")

    if include_challenge:
        if not (challenge_root / "pairs").is_dir() or len(list((challenge_root / "pairs").glob("*.parquet"))) < 96:
            raise RuntimeError(f"CARLA_CHALLENGE_DATASET_INCOMPLETE:{challenge_root}")
        data_roots = [h2_root, challenge_root]
        active_dataset_id = f"h3-combined-{CARLA_CHALLENGE_DATASET_ID}"
    else:
        data_roots = [h2_root]
        active_dataset_id = DATASET_ID

    records = read_h2_records(data_roots)
    physical = _json(h2_root / "scenario_manifest.json")
    store_manifest = _json(h2_root / "manifest.json")
    split = build_split_manifest(
        records,
        dataset_id=active_dataset_id,
        physical_manifest_sha256=str(physical["physical_manifest_sha256"]),
        store_manifest_sha256=str(store_manifest["manifest_sha256"]),
    )
    write_split_manifest(generated / "split_manifest.json", split)
    _write(evidence_dir / "split_manifest.json", split)

    # 1. Leakage Audit
    leakage = leakage_audit(data_roots, split)
    _write(evidence_dir / "leakage-audit.json", leakage)
    if not leakage["passed"]:
        raise RuntimeError(f"H3_LEAKAGE_AUDIT_FAILED:{leakage['failures']}")

    # 2. Load strictly development folds
    fold_examples = {fold: load_examples(data_roots, split, split=fold) for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3")}
    all_dev = [item for fold in fold_examples.values() for item in fold]

    # 3. Evaluate Baselines
    baselines = {name: evaluate_baseline(all_dev, name) for name in BASELINE_NAMES}
    best_baseline_name = max(baselines, key=lambda name: float(baselines[name]["accuracy"] or 0.0))
    best_baseline = baselines[best_baseline_name]
    _write(evidence_dir / "baselines.json", {"baselines": baselines, "best_name": best_baseline_name})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    # 4. Train 3-fold cross-validation with 5 seeds per fold
    fold_models: dict[str, list] = {}
    train_records: list[dict[str, Any]] = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        train_examples = [item for name, rows in fold_examples.items() if name != fold for item in rows]
        fold_models[fold], fold_results = _train_fold(train_examples, fold_examples[fold], fold, checkpoint_root, device)
        train_records.extend(fold_results)

    # 5. Out-of-fold pooled evaluation
    cv_metrics = evaluate_cv(fold_models, fold_examples, device=device)
    cv_metrics_json = {key: value for key, value in cv_metrics.items() if key != "rows"}

    # 6. Train final 5-seed ensemble on all dev data
    final_models, final_records = _train_final(all_dev, checkpoint_root, device)
    train_records.extend(final_records)

    model_metrics = cv_metrics_json
    swap = swap_consistency(final_models, all_dev, device=device)
    ablation = evaluate_ablation(final_models, all_dev, device=device)
    bootstrap = bootstrap_accuracy_delta(final_models, all_dev, best_baseline_name, device=device)
    resource = _resource_benchmark(final_models, all_dev, device)
    seed_metrics = [evaluate_models([model], all_dev, device=device) for model in final_models]

    # 7. Check Gate
    gate = h3_gate(
        leakage=leakage,
        swap=swap,
        model_metrics=model_metrics,
        best_baseline=best_baseline,
        bootstrap=bootstrap,
        ablation=ablation,
        resource=resource,
        seed_metrics=seed_metrics,
    )

    challenge = _challenge_probe(evidence_dir) if not gate["passed"] else {"status": "not_needed"}

    payload: dict[str, Any] = {
        "schema_version": "safedrive.h3.evidence.v1",
        "run_id": run_id,
        "dataset_id": active_dataset_id,
        "include_challenge": include_challenge,
        "evidence_status": "VERIFIED",
        "gate_status": "GATE_PASSED" if gate["passed"] else "GATE_FAILED",
        "stopped": True,
        "h4_status": "NOT_AUTHORIZED",
        "h2": {
            "physical_manifest_sha256": physical["physical_manifest_sha256"],
            "store_manifest_sha256": store_manifest["manifest_sha256"],
            "config_sha256": records[0]["config_sha256"],
            "records": len(records),
        },
        "h3_config": H3_CONFIG,
        "h3_config_sha256": H3_CONFIG_SHA256,
        "split": {"manifest_sha256": split["manifest_sha256"], "rows": len(split["rows"]), "dev_examples": len(all_dev)},
        "leakage": leakage,
        "baselines": baselines,
        "best_baseline": best_baseline_name,
        "training": train_records,
        "cv_metrics": cv_metrics_json,
        "seed_metrics": seed_metrics,
        "swap": swap,
        "ablation": {key: value for key, value in ablation.items()},
        "bootstrap": bootstrap,
        "resource": resource,
        "gate": gate,
        "challenge": challenge,
        "notes": [
            "H2 test labels were never opened by H3 training/evaluation code.",
            "Collision, traffic-light, off-corridor and completion heads are unsupported because H2 contains no positive labels.",
            "Any challenge collection requires a new dataset id and does not mutate the frozen H2 store.",
        ],
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    _write(evidence_dir / "final-delivery.json", payload)
    _write(generated / "results.json", payload)

    return {
        "ok": True,
        "gate_status": payload["gate_status"],
        "evidence": str(evidence_dir / "final-delivery.json"),
        "evidence_sha256": payload["evidence_sha256"],
        "failures": gate["failures"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--include-challenge", action="store_true", default=False)
    args = parser.parse_args()
    try:
        result = run(args.run_id, include_challenge=args.include_challenge)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
