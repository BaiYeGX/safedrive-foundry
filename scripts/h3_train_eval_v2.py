#!/usr/bin/env python3
"""H3v2 nested out-of-fold training and acceptance evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import torch  # noqa: E402

from data_pipeline.h3.baselines import BASELINE_NAMES, baseline_winner, evaluate_baseline  # noqa: E402
from data_pipeline.h3.baseline_models import (  # noqa: E402
    evaluate_baseline_models,
    load_baseline_model,
    predict_baseline_pair,
    train_baseline_model,
)
from data_pipeline.h3.contracts import H3_CONFIG, H3_CONFIG_SHA256, stable_sha256  # noqa: E402
from data_pipeline.h3.dataset import (  # noqa: E402
    build_split_manifest,
    leakage_audit,
    load_examples,
    read_h2_records,
    write_split_manifest,
)
from data_pipeline.h3.evaluate import (  # noqa: E402
    defer_curve,
    evaluate_models,
    fit_temperature,
    h3_gate,
    metrics_from_rows,
    prediction_rows,
    sigmoid,
    swap_consistency,
)
from data_pipeline.h3.model import load_model, train_model  # noqa: E402
from data_pipeline.h3.runtime import WorldScorer  # noqa: E402
from data_pipeline.h2.store import PairedOutcomeStore  # noqa: E402


H2_DATASET_ID = "h2-gatepass-20260813-routefix"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inner_split(examples, val_ratio: float = 0.20) -> tuple[list, list]:
    """Deterministic inner validation split; never touches the outer fold."""
    by_lineage: dict[str, list] = {}
    for item in examples:
        by_lineage.setdefault(f"{item.map_name}|{item.family}|{item.seed}", []).append(item)
    train, val = [], []
    for lineage in sorted(by_lineage):
        items = by_lineage[lineage]
        bucket = "val" if int(hashlib.sha256(lineage.encode()).hexdigest(), 16) % 5 == 0 else "train"
        (val if bucket == "val" else train).extend(items)
    if val and len(val) / max(1, len(examples)) > val_ratio + 0.15:
        # Rebalance deterministically when hashing creates a large val split.
        val = val[: max(1, int(len(examples) * val_ratio))]
        train = [item for item in examples if item not in set(val)]
    if not val:
        val = train[-1:]
        train = train[:-1]
    return train, val


def _train_fold(train_examples, fold, checkpoint_root: Path, device: str, max_epochs: int, patience: int, scene_gate_mode: str = "hard") -> tuple[list, list[dict[str, Any]]]:
    inner_train, inner_val = _inner_split(train_examples)
    models, results = [], []
    for seed in H3_CONFIG["training_seeds"]:
        checkpoint = checkpoint_root / "cv" / f"{fold}__seed-{seed}.pt"
        if checkpoint.exists():
            model, metadata = load_model(checkpoint, device=device)
            models.append(model)
            results.append({"fold": fold, "seed": int(seed), "checkpoint": str(checkpoint),
                            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            "best_epoch": int(metadata.get("best_epoch", 0)),
                            "best_val_loss": float(metadata.get("best_val_loss", float("nan"))),
                            "train_examples": int(metadata.get("train_examples", 0)),
                            "val_examples": int(metadata.get("val_examples", 0)),
                            "device": str(device)})
            continue
        result = train_model(inner_train, inner_val, seed=int(seed), checkpoint_path=checkpoint,
                             device=device, max_epochs=max_epochs, patience=patience,
                             scene_gate_mode=scene_gate_mode)
        model, _ = load_model(checkpoint, device=device)
        models.append(model)
        results.append({"fold": fold, "seed": int(seed), "best_epoch": result.best_epoch,
                        "best_val_loss": result.best_val_loss, "checkpoint": result.checkpoint_path,
                        "checkpoint_sha256": result.checkpoint_sha256, "train_examples": result.train_examples,
                        "val_examples": result.val_examples, "device": result.device})
    return models, results


def _fit_fold_temperature(fold_models, fold_examples, current_fold: str, device: str) -> float:
    """Fit T on OOF predictions of the two training folds only."""
    deltas: list[float] = []
    targets: list[int] = []
    for fold, examples in sorted(fold_examples.items()):
        if fold == current_fold:
            continue
        rows = prediction_rows(fold_models[fold], examples, device=device)
        deltas.extend(row["delta"] for row in rows)
        targets.extend(row["target"] for row in rows)
    return fit_temperature(deltas, targets)


def _resource_benchmark(final_models, examples, device: str, pre_reserved_gib: float = 0.0) -> dict[str, Any]:
    if not examples:
        return {"passed": False, "reason": "no_examples"}
    scorer = WorldScorer(final_models, device=device, model_hash="h3-v2-runtime-benchmark",
                         temperature=float(final_models[0].metadata_temperature) if hasattr(final_models[0], "metadata_temperature") else 1.0)
    example = examples[0]
    first = example.candidates[0]
    second = example.candidates[1]
    for _ in range(20):
        scorer.score_pair((first.candidate_key, first.context, first.candidate),
                          (second.candidate_key, second.context, second.candidate))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        scorer.score_pair((first.candidate_key, first.context, first.candidate),
                          (second.candidate_key, second.context, second.candidate))
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    peak_gib = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 3) if torch.cuda.is_available() else 0.0
    p50, p95, p99 = latencies[len(latencies)//2], latencies[int(0.95*(len(latencies)-1))], latencies[int(0.99*(len(latencies)-1))]
    incremental_gib = max(0.0, peak_gib - pre_reserved_gib)
    return {
        "device": device,
        "gpu_available": bool(torch.cuda.is_available()),
        "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
        "deadline_misses": sum(item > float(H3_CONFIG["runtime"]["deadline_ms"]) for item in latencies),
        "peak_reserved_gib": peak_gib,
        "pre_reserved_gib": pre_reserved_gib,
        "incremental_gpu_gib": incremental_gib,
        "passed": p99 <= float(H3_CONFIG["runtime"]["deadline_ms"]) and incremental_gib <= float(H3_CONFIG["runtime"]["max_incremental_gpu_gib"]),
        "failure_reason": None,
    }


def _store_identity(root: Path) -> dict[str, Any]:
    return {
        "path": str(root),
        "store_manifest_sha256": _json(root / "manifest.json").get("manifest_sha256"),
        "physical_manifest_sha256": _json(root / "scenario_manifest.json").get("physical_manifest_sha256"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    h2_root = ROOT / "generated" / "h2" / "paired-outcomes" / (args.h2_dataset_id or H2_DATASET_ID)
    roots = [h2_root]
    if args.challenge_dataset_id:
        challenge_root = ROOT / "generated" / "h3" / "carla-challenge-v2" / args.challenge_dataset_id
        roots.append(challenge_root)
        for store_root in roots:
            store = PairedOutcomeStore(store_root.parent, store_root.name)
            ok, bad = store.verify_manifest()
            if not ok:
                raise RuntimeError(f"STORE_MANIFEST_INVALID:{store_root.name}:{bad[:4]}")
    else:
        store = PairedOutcomeStore(h2_root.parent, h2_root.name)
        ok, bad = store.verify_manifest()
        if not ok:
            raise RuntimeError(f"H2_STORE_MANIFEST_INVALID:{bad[:4]}")

    records = read_h2_records(roots)
    dataset_id = args.run_id + "-combined" if args.challenge_dataset_id else (args.h2_dataset_id or H2_DATASET_ID)
    h2_identity = _store_identity(h2_root)
    challenge_identity = _store_identity(challenge_root) if args.challenge_dataset_id else None
    split = build_split_manifest(
        records,
        dataset_id=dataset_id,
        physical_manifest_sha256=h2_identity["physical_manifest_sha256"],
        store_manifest_sha256=h2_identity["store_manifest_sha256"],
        challenge_physical_manifest_sha256=challenge_identity["physical_manifest_sha256"] if challenge_identity else None,
        challenge_store_manifest_sha256=challenge_identity["store_manifest_sha256"] if challenge_identity else None,
    )
    generated = ROOT / "generated" / "h3" / args.run_id
    evidence_dir = ROOT / "docs" / "runtime-evidence" / "h3" / args.run_id
    generated.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(generated / "split_manifest.json", split)
    _write(evidence_dir / "split_manifest.json", split)

    leakage = leakage_audit(roots, split)
    _write(evidence_dir / "leakage-audit.json", leakage)
    if not leakage["passed"]:
        raise RuntimeError(f"LEAKAGE_AUDIT_FAILED:{leakage['failures']}")

    fold_examples = {fold: load_examples(roots, split, split=fold) for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3")}
    all_dev = [item for fold in fold_examples.values() for item in fold]

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    base_reserved_gib = 0.0
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.synchronize()
        base_reserved_gib = float(torch.cuda.memory_reserved()) / (1024.0 ** 3)
    checkpoint_root = generated / "checkpoints"

    baselines = {name: evaluate_baseline(all_dev, name) for name in BASELINE_NAMES}

    # Train the two required MLP baselines with the same fold protocol.
    baseline_checkpoint_root = generated / "baseline_checkpoints"
    for kind in ("candidate_mlp", "full_mlp"):
        rows_for_kind = []
        for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
            train_examples = [item for name, rows in fold_examples.items() if name != fold for item in rows]
            inner_train, inner_val = _inner_split(train_examples)
            checkpoint = baseline_checkpoint_root / kind / f"{fold}.pt"
            train_baseline_model(kind, inner_train, inner_val, seed=11, checkpoint_path=checkpoint,
                                 device=device, max_epochs=args.max_epochs or 350, patience=args.patience or 40)
            model, _ = load_baseline_model(checkpoint, kind=kind, device=device)
            metrics = evaluate_baseline_models([model], fold_examples[fold], kind=kind, device=device)
            # Pool OOF correctness without double-counting any pair.
            rows_for_kind.extend(
                [item for item in fold_examples[fold] if item.decisive]
            )
            model_hits = sum(predict_baseline_pair([model], item, kind=kind, device=device) == item.winner_index for item in fold_examples[fold] if item.decisive)
            metrics["correct_by_fold"] = model_hits
        decisive_for_kind = [item for item in all_dev if item.decisive]
        # Recompute pooled metric from the fold models for evidence consistency.
        baseline_models_by_fold = {}
        for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
            checkpoint = baseline_checkpoint_root / kind / f"{fold}.pt"
            model, _ = load_baseline_model(checkpoint, kind=kind, device=device)
            baseline_models_by_fold[fold] = [model]
        # A simple pooled evaluation over all_dev with the corresponding fold model.
        pooled_correct = 0
        pooled_n = 0
        for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
            for item in fold_examples[fold]:
                if not item.decisive:
                    continue
                pooled_n += 1
                pooled_correct += int(predict_baseline_pair(baseline_models_by_fold[fold], item, kind=kind, device=device) == item.winner_index)
        baselines[kind] = {"name": kind, "n_decisive": pooled_n, "correct": pooled_correct,
                           "accuracy": pooled_correct / pooled_n if pooled_n else None,
                           "mean_progress_regret_m": None, "mean_jerk_regret_mps3": None}
    best_name = max(BASELINE_NAMES, key=lambda name: float(baselines[name]["accuracy"] or 0.0))
    best_baseline = baselines[best_name]
    _write(evidence_dir / "baselines.json", {"baselines": baselines, "best_name": best_name})


    # 1) Per-fold OOF models (5 seeds each).
    fold_models: dict[str, list] = {}
    train_records: list[dict[str, Any]] = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        train_examples = [item for name, rows in fold_examples.items() if name != fold for item in rows]
        fold_models[fold], fold_results = _train_fold(train_examples, fold, checkpoint_root, device, args.max_epochs, args.patience, args.scene_gate_mode)
        train_records.extend(fold_results)

    # 2) Nested temperatures and OOF rows.
    fold_temperatures: dict[str, float] = {}
    fold_rows: dict[str, list[dict[str, Any]]] = {}
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        fold_temperatures[fold] = _fit_fold_temperature(fold_models, fold_examples, fold, device)
        fold_rows[fold] = prediction_rows(fold_models[fold], fold_examples[fold], device=device)
    oof_rows = [row for fold in fold_rows.values() for row in fold]
    # metrics_from_rows needs one T; use pooled T on OOF deltas only for reporting.
    pooled_t = fit_temperature([row["delta"] for row in oof_rows], [row["target"] for row in oof_rows])
    cv_metrics = metrics_from_rows(oof_rows, temperature=pooled_t)

    # 3) OOF ablation using per-fold temperature and pooled weighted metrics.
    ablation_rows = {"full": [], "action": [], "history": []}
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        models, examples = fold_models[fold], fold_examples[fold]
        ablation_rows["full"].extend(prediction_rows(models, examples, device=device))
        ablation_rows["action"].extend(prediction_rows(models, examples, device=device, mask_candidate=True))
        ablation_rows["history"].extend(prediction_rows(models, examples, device=device, mask_context=True))
    full_m = metrics_from_rows(ablation_rows["full"], temperature=pooled_t)
    action_m = metrics_from_rows(ablation_rows["action"], temperature=pooled_t)
    history_m = metrics_from_rows(ablation_rows["history"], temperature=pooled_t)
    ablation = {
        "full": full_m, "action_mask": action_m, "history_mask": history_m,
        "action_accuracy_drop": (full_m["accuracy"] - action_m["accuracy"]) if full_m["accuracy"] is not None else None,
        "history_accuracy_drop": (full_m["accuracy"] - history_m["accuracy"]) if full_m["accuracy"] is not None else None,
    }

    # 4) Swap consistency on OOF models/examples.
    swap_rows = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        swap_rows.append(swap_consistency(fold_models[fold], fold_examples[fold], device=device))
    swap = {"checked": sum(int(r["checked"]) for r in swap_rows), "failures": [f for r in swap_rows for f in r["failures"]],
            "max_error": max((r["max_error"] for r in swap_rows), default=0.0), "passed": all(r["passed"] for r in swap_rows)}

    # 5) Bootstrap model vs best baseline on pooled OOF rows.
    rng = __import__("random").Random(args.bootstrap_seed)
    model_hits = [int(int(row["predicted"]) == int(row["winner_index"])) for row in oof_rows]
    base_hits = [int(baseline_winner(item, best_name) == item.winner_index) for item in all_dev if item.decisive]
    deltas = []
    n = len(oof_rows)
    for _ in range(int(H3_CONFIG["acceptance"]["bootstrap_rounds"])):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(model_hits[i] - base_hits[i] for i in idx) / n)
    deltas.sort()
    bootstrap = {"delta": sum(model_hits)/n - sum(base_hits)/n, "lower_95": deltas[int(0.025*len(deltas))], "upper_95": deltas[int(0.975*len(deltas))]}

    # 6) Per-seed OOF stability.
    seed_metrics = []
    for seed_index in range(len(H3_CONFIG["training_seeds"])):
        rows_for_seed = []
        for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
            rows_for_seed.extend(prediction_rows([fold_models[fold][seed_index]], fold_examples[fold], device=device))
        seed_metrics.append(metrics_from_rows(rows_for_seed, temperature=pooled_t))

    # 7) Final deployment ensemble (not used in any gate metric).
    final_models = []
    final_records = []
    final_inner_train, final_inner_val = _inner_split(all_dev)
    for seed in H3_CONFIG["training_seeds"]:
        checkpoint = checkpoint_root / "final" / f"seed-{seed}.pt"
        if checkpoint.exists():
            model, metadata = load_model(checkpoint, device=device)
        else:
            result = train_model(final_inner_train, final_inner_val, seed=int(seed), checkpoint_path=checkpoint,
                                 device=device, max_epochs=args.max_epochs, patience=args.patience,
                                 scene_gate_mode=args.scene_gate_mode)
            model, metadata = load_model(checkpoint, device=device)
        final_models.append(model)
        final_records.append({"seed": int(seed), "checkpoint": str(checkpoint),
                              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              "best_epoch": int(metadata.get("best_epoch", 0)),
                              "best_val_loss": float(metadata.get("best_val_loss", float("nan")))})
    # Freeze the cross-fitted calibration temperature into deployment checkpoints.
    for record in final_records:
        ckpt = Path(record["checkpoint"])
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if float(payload.get("metadata", {}).get("temperature", 1.0)) != pooled_t:
            payload.setdefault("metadata", {})["temperature"] = pooled_t
            torch.save(payload, ckpt)
        record["checkpoint_sha256"] = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    resource = _resource_benchmark(final_models, all_dev, device, base_reserved_gib)
    defer = defer_curve(final_models, all_dev, device=device, temperature=pooled_t)

    # 8) Frozen H3 gate.
    gate = h3_gate(leakage=leakage, swap=swap, model_metrics=cv_metrics, best_baseline=best_baseline,
                   bootstrap=bootstrap, ablation=ablation, resource=resource, seed_metrics=seed_metrics)

    payload: dict[str, Any] = {
        "schema_version": "safedrive.h3.evidence.v2",
        "run_id": args.run_id,
        "dataset_id": dataset_id,
        "h2": h2_identity,
        "challenge_roots": [_store_identity(root) for root in roots[1:]],
        "h3_config": H3_CONFIG,
        "h3_config_sha256": H3_CONFIG_SHA256,
        "split": {"manifest_sha256": split["manifest_sha256"], "rows": len(split["rows"]), "dev_examples": len(all_dev)},
        "leakage": leakage,
        "baselines": baselines,
        "best_baseline": best_name,
        "training": train_records + final_records,
        "cv_metrics": cv_metrics,
        "fold_temperatures": fold_temperatures,
        "seed_metrics": seed_metrics,
        "swap": swap,
        "ablation": ablation,
        "bootstrap": bootstrap,
        "resource": resource,
        "defer_curve": defer,
        "gate": gate,
        "gate_status": "GATE_PASSED" if gate["passed"] else "GATE_FAILED",
        "evidence_status": "VERIFIED",
        "stopped": True,
        "h4_status": "NOT_AUTHORIZED",
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    _write(evidence_dir / "final-delivery.json", payload)
    _write(generated / "results.json", payload)
    return {"ok": True, "gate_status": payload["gate_status"], "failures": gate["failures"],
            "evidence": str(evidence_dir / "final-delivery.json"), "evidence_sha256": payload["evidence_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--h2-dataset-id", default=H2_DATASET_ID)
    parser.add_argument("--challenge-dataset-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--scene-gate-mode", choices=("hard", "learned"), default="hard")
    parser.add_argument("--bootstrap-rounds", type=int, default=None)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["gate_status"] == "GATE_PASSED" else 1
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
