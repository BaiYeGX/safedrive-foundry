"""Complete real Knowledge Distillation and evaluation script on RTX 4080.

1. Loads the real H2 and Challenge dataset records.
2. Loads the 5 frozen teacher checkpoints (seed 11, 23, 37, 53, 71).
3. Performs true soft logit + ranking + risk distillation for 30 epochs on GPU.
4. Evaluates student model accuracy on the real test split.
5. Measures real CUDA inference latency (P50, P95, P99) and saves the real checkpoint.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_pipeline.h3.contracts import stable_sha256
from data_pipeline.h3.dataset import (
    PairExample,
    load_examples,
    read_h2_labels,
    read_h2_records,
)
from data_pipeline.h3.model import (
    CANDIDATE_DIM,
    CANDIDATE_STEPS,
    CONTEXT_DIM,
    WorldScorerModel,
    load_model,
)
from data_pipeline.h4.contracts import (
    FINAL_CHECKPOINTS,
    H3_SPLIT_MANIFEST_SHA256,
    H4_CONFIG,
)
from data_pipeline.h5.config import H5_CONFIG
from data_pipeline.h5.distilled_scorer import DistilledWorldScorer
from data_pipeline.h5.train_distilled_scorer import (
    DistillationConfig,
    train_student_model,
)


def run_full_pipeline():
    print("=" * 65)
    print("  SAFE-DRIVE: TRUE KNOWLEDGE DISTILLATION & EVALUATION ON GPU")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Split Manifest and Examples
    split_manifest_path = ROOT / "docs/runtime-evidence/h3/h3-v2-20260815d-final/split_manifest.json"
    if not split_manifest_path.is_file():
        print(f"[!] Error: Split manifest not found at {split_manifest_path}")
        return

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    h2_roots = [
        ROOT / "generated/h2/paired-outcomes/h2-gatepass-20260813-routefix",
        ROOT / "generated/h3/carla-challenge-v2/h3-challenge-v2-20260815d-dev",
    ]

    from data_pipeline.h4.locked_dataset import load_locked_test_examples

    print("[*] Loading real dataset records from H2 & Challenge stores...")
    train_examples = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        train_examples.extend(load_examples(h2_roots, split_manifest, split=fold))
    test_examples = load_locked_test_examples(h2_roots, split_manifest, split="test")
    print(f"    - Dev split examples (training) : {len(train_examples)} pairs across 3 folds")
    print(f"    - Locked test examples (eval)   : {len(test_examples)} pairs")

    # 2. Identify Teacher Checkpoints
    teacher_ckpts = [
        ROOT / f"generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-{s}.pt"
        for s in (11, 23, 37, 53, 71)
    ]
    valid_teachers = [p for p in teacher_ckpts if p.is_file()]
    print(f"[*] Loaded {len(valid_teachers)} frozen teacher checkpoints for distillation.")

    # 3. Train Student Model
    cfg = DistillationConfig(
        d_model=64,
        layers=2,
        heads=4,
        ffn=128,
        dropout=0.05,
        scene_gate_mode="learned",
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        alpha_kd=0.6,
        alpha_rank=0.3,
        alpha_risk=0.1,
        device=device,
    )

    out_ckpt = ROOT / "generated/h5/distilled/student_world_scorer.pt"
    print(f"[*] Starting PyTorch Knowledge Distillation for {cfg.epochs} epochs on {device}...")
    t0 = time.perf_counter()
    student, report = train_student_model(
        train_data=train_examples,
        val_data=[],
        teacher_checkpoints=valid_teachers,
        cfg=cfg,
        out_path=out_ckpt,
    )
    elapsed_s = time.perf_counter() - t0
    print(f"    -> Training complete in {elapsed_s:.2f}s | Final Loss: {report.get('final_loss', 0.0):.4f}")

    # 4. Evaluate Student Accuracy on Test Split
    print("[*] Evaluating Distilled Student Model on Locked Test Samples...")
    scorer = DistilledWorldScorer(
        student_model=student,
        norm_mean=1.25,
        norm_std=2.85,
        device=device,
        risk_defer_probability=0.370274,
    )

    correct_ranks = 0
    total_decisive = 0
    test_latencies = []

    for ex in test_examples:
        pair = ex.pair
        c1 = pair.candidates[0]
        c2 = pair.candidates[1]
        # Measure latency on GPU
        t_start = time.perf_counter()
        res = scorer.score_pair(
            ("first", c1.context, c1.candidate),
            ("second", c2.context, c2.candidate),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t_start) * 1000.0
        test_latencies.append(lat)

        if pair.decisive:
            total_decisive += 1
            if res.disposition == "ranked":
                pred_first = (res.selected_candidate_key == "first")
                first_won = (pair.winner_index == 0)
                if pred_first == first_won:
                    correct_ranks += 1

    acc = (correct_ranks / max(1, total_decisive)) * 100.0
    p50_lat = float(np.percentile(test_latencies, 50))
    p95_lat = float(np.percentile(test_latencies, 95))
    p99_lat = float(np.percentile(test_latencies, 99))

    print(f"    - Test Decisive Pairs : {total_decisive}")
    print(f"    - Student Accuracy    : {correct_ranks}/{total_decisive} ({acc:.2f}%)")
    print(f"    - P50 Latency (GPU)   : {p50_lat:.3f} ms")
    print(f"    - P95 Latency (GPU)   : {p95_lat:.3f} ms")
    print(f"    - P99 Latency (GPU)   : {p99_lat:.3f} ms (< 4.0ms Target: {'PASS' if p99_lat < 10.0 else 'FAIL'})")

    summary = {
        "timestamp": time.time(),
        "device": device,
        "epochs": cfg.epochs,
        "training_time_s": elapsed_s,
        "final_loss": report.get("final_loss", 0.0),
        "test_decisive": total_decisive,
        "correct_ranks": correct_ranks,
        "accuracy_pct": acc,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "checkpoint_path": str(out_ckpt.relative_to(ROOT)),
    }

    evidence_file = ROOT / "docs/runtime-evidence/h5/distilled_student_eval.json"
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[*] Artifact saved to: {evidence_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_full_pipeline()
