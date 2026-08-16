#!/usr/bin/env python3
"""H4 locked evaluation: frozen inputs, dev-only calibration, single blind run."""

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

from data_pipeline.h2.store import PairedOutcomeStore  # noqa: E402
from data_pipeline.h3.baselines import BASELINE_NAMES, baseline_winner, evaluate_baseline  # noqa: E402
from data_pipeline.h3.contracts import H3_CONFIG_SHA256, stable_sha256  # noqa: E402
from data_pipeline.h3.dataset import load_examples  # noqa: E402
from data_pipeline.h3.model import load_model  # noqa: E402
from data_pipeline.h4.contracts import FINAL_CHECKPOINTS, H4_CONFIG, H4_CONFIG_SHA256  # noqa: E402
from data_pipeline.h4.locked_dataset import audit_test_isolation, load_locked_test_examples  # noqa: E402
from data_pipeline.h4.metrics import (  # noqa: E402
    bootstrap_accuracy_delta,
    defer_metrics,
    end_to_end_selector_metrics,
    metrics_from_rows,
    normalized_prediction_rows,
    source_wins,
    swap_consistency,
)
from data_pipeline.h4.runtime import NormalizedWorldScorer  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _roots() -> list[Path]:
    h2 = ROOT / "generated" / "h2" / "paired-outcomes" / H4_CONFIG["h2_dataset_id"]
    challenge = ROOT / "generated" / "h3" / "carla-challenge-v2" / H4_CONFIG["challenge_dataset_id"]
    return [h2, challenge]


def _verify_store(root: Path) -> None:
    store = PairedOutcomeStore(root.parent, root.name)
    ok, bad = store.verify_manifest()
    if not ok:
        raise RuntimeError(f"STORE_MANIFEST_INVALID:{root.name}:{bad[:4]}")


def verify_frozen_inputs() -> dict[str, Any]:
    evidence_path = ROOT / "docs" / "runtime-evidence" / "h3" / H4_CONFIG["h3_run_id"] / "final-delivery.json"
    split_path = ROOT / "docs" / "runtime-evidence" / "h3" / H4_CONFIG["h3_run_id"] / "split_manifest.json"
    if not evidence_path.exists():
        raise RuntimeError(f"H3_EVIDENCE_MISSING:{evidence_path}")
    if not split_path.exists():
        raise RuntimeError(f"H3_SPLIT_MISSING:{split_path}")

    evidence = _json(evidence_path)
    split = _json(split_path)
    actual_evidence_sha = stable_sha256({k: v for k, v in evidence.items() if k != "evidence_sha256"})
    if actual_evidence_sha != H4_CONFIG["h3_evidence_sha256"]:
        raise RuntimeError(f"H3_EVIDENCE_SHA_MISMATCH:{actual_evidence_sha}")
    if split.get("manifest_sha256") != H4_CONFIG["h3_split_manifest_sha256"]:
        raise RuntimeError(f"H3_SPLIT_SHA_MISMATCH:{split.get('manifest_sha256')}")
    if H4_CONFIG["h3_config_sha256"] != H3_CONFIG_SHA256:
        raise RuntimeError(f"H3_CONFIG_SHA_MISMATCH:{H3_CONFIG_SHA256}")

    roots = _roots()
    for root in roots:
        _verify_store(root)

    checkpoint_errors = []
    for seed, info in FINAL_CHECKPOINTS.items():
        path = ROOT / info["path"]
        if not path.exists():
            checkpoint_errors.append(f"missing:{path}")
            continue
        actual = _sha256_file(path)
        if actual != info["sha256"]:
            checkpoint_errors.append(f"sha_mismatch:{path}:{actual}")
    if checkpoint_errors:
        raise RuntimeError(f"CHECKPOINT_VERIFY_FAILED:{checkpoint_errors}")

    return {
        "h3_evidence_sha256": actual_evidence_sha,
        "h3_split_manifest_sha256": split["manifest_sha256"],
        "h3_config_sha256": H3_CONFIG_SHA256,
        "h2_dataset_id": H4_CONFIG["h2_dataset_id"],
        "challenge_dataset_id": H4_CONFIG["challenge_dataset_id"],
        "checkpoints_verified": True,
    }


def load_models(device: str) -> tuple[list, dict[str, str]]:
    models = []
    digests = {}
    for seed, info in FINAL_CHECKPOINTS.items():
        path = ROOT / info["path"]
        model, metadata = load_model(path, device=device)
        models.append(model)
        digests[seed] = _sha256_file(path)
        if float(metadata.get("temperature", 1.0)) != float(H4_CONFIG["temperature"]):
            raise RuntimeError(f"CHECKPOINT_TEMPERATURE_MISMATCH:{seed}")
    return models, digests


def compute_normalization_stats(models: Sequence, dev_examples: Sequence, device: str) -> dict[str, Any]:
    """Compute per-model utility mean/std from dev only (no test labels)."""
    from data_pipeline.h3.model import predict_model

    stats: dict[str, Any] = {"seeds": list(FINAL_CHECKPOINTS.keys()), "items": []}
    for seed, model in zip(FINAL_CHECKPOINTS.keys(), models):
        values = []
        for example in dev_examples:
            p0, p1 = predict_model(model, example, device=device)
            values.extend([p0.utility, p1.utility])
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        stats["items"].append({"seed": seed, "mean": float(mean), "std": float(std)})
    return stats


def _stats_list(stats: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(item["mean"]), float(item["std"])) for item in stats["items"]]


def _load_dev_examples(split: Mapping[str, Any]) -> list:
    roots = _roots()
    examples = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        examples.extend(load_examples(roots, split, split=fold))
    return examples


def _resource_benchmark(
    models: Sequence,
    stats_list: Sequence[tuple[float, float]],
    examples: Sequence,
    device: str,
    pre_reserved_gib: float,
) -> dict[str, Any]:
    if not examples:
        return {"passed": False, "reason": "no_examples"}
    scorer = NormalizedWorldScorer(models, stats_list, device=device, temperature=float(H4_CONFIG["temperature"]))
    # Warmup.
    example = examples[0]
    pair = example.pair if hasattr(example, "pair") else example
    first = pair.candidates[0]
    second = pair.candidates[1]
    for _ in range(20):
        scorer.score_pair(
            (first.candidate_key, first.context, first.candidate),
            (second.candidate_key, second.context, second.candidate),
        )
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        scorer.score_pair(
            (first.candidate_key, first.context, first.candidate),
            (second.candidate_key, second.context, second.candidate),
        )
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    peak_reserved_gib = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 3) if torch.cuda.is_available() else 0.0
    p50, p95, p99 = latencies[len(latencies) // 2], latencies[int(0.95 * (len(latencies) - 1))], latencies[int(0.99 * (len(latencies) - 1))]
    incremental_gib = max(0.0, peak_reserved_gib - pre_reserved_gib)
    return {
        "device": device,
        "gpu_available": bool(torch.cuda.is_available()),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "deadline_misses": sum(item > float(H4_CONFIG["runtime"]["deadline_ms"]) for item in latencies),
        "peak_reserved_gib": peak_reserved_gib,
        "pre_reserved_gib": pre_reserved_gib,
        "incremental_gpu_gib": incremental_gib,
        "passed": p99 <= float(H4_CONFIG["runtime"]["deadline_ms"]) and incremental_gib <= float(H4_CONFIG["runtime"]["max_incremental_gpu_gib"]),
        "failure_reason": None,
    }


def _gate(
    *,
    isolation: dict,
    metrics: dict,
    best_baseline: dict,
    resource: dict,
    n_decisive: int,
) -> dict[str, Any]:
    checks = {
        "isolation": bool(isolation.get("passed")),
        "sufficient_test_power": n_decisive >= int(H4_CONFIG["min_decisive_for_claim"]),
        "ranking_superior_to_best_simple_baseline": bool(
            metrics.get("accuracy") is not None
            and best_baseline.get("accuracy") is not None
            and float(metrics["accuracy"]) > float(best_baseline["accuracy"])
        ),
        "resource": bool(resource.get("passed")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {"passed": not failures, "checks": checks, "failures": failures, "best_baseline": best_baseline}


def run_evaluate(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    verified = verify_frozen_inputs()
    split = _json(ROOT / "docs" / "runtime-evidence" / "h3" / H4_CONFIG["h3_run_id"] / "split_manifest.json")
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    if mode == "verify":
        return {"ok": True, "mode": mode, "verified": verified, "h4_config_sha256": H4_CONFIG_SHA256}

    # Initialize CUDA context before model load so incremental VRAM is measured
    # relative to an empty-but-initialized context.
    pre_reserved_gib = 0.0
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize()
        pre_reserved_gib = float(torch.cuda.memory_reserved()) / (1024.0 ** 3)

    models, model_digests = load_models(device)
    dev_examples = _load_dev_examples(split)
    stats = compute_normalization_stats(models, dev_examples, device)
    stats_list = _stats_list(stats)

    if mode == "dev-smoke":
        rows = normalized_prediction_rows(models, stats_list, dev_examples, device=device)
        metrics = metrics_from_rows(rows, temperature=float(H4_CONFIG["temperature"]))
        defer = defer_metrics(rows, temperature=float(H4_CONFIG["temperature"]))
        resource = _resource_benchmark(models, stats_list, dev_examples, device, pre_reserved_gib)
        return {
            "ok": True,
            "mode": mode,
            "verified": verified,
            "dev_examples": len(dev_examples),
            "dev_metrics": metrics,
            "dev_defer": defer,
            "resource": resource,
            "normalization_stats": stats,
            "h4_config_sha256": H4_CONFIG_SHA256,
        }

    if mode != "evaluate":
        raise ValueError(f"unknown_mode:{mode}")

    test_examples = load_locked_test_examples(_roots(), split, split="test")
    isolation = audit_test_isolation(_roots(), split, dev_examples, test_examples)

    rows = normalized_prediction_rows(models, stats_list, test_examples, device=device)
    metrics = metrics_from_rows(rows, temperature=float(H4_CONFIG["temperature"]))
    baselines = {name: evaluate_baseline([example.pair for example in test_examples], name) for name in BASELINE_NAMES}
    best_name = max(BASELINE_NAMES, key=lambda name: float(baselines[name]["accuracy"] or 0.0))
    best_baseline = baselines[best_name]

    # Deterministic tie-break: if multiple baselines share the best accuracy,
    # the lexicographically first name is the frozen choice.
    best_accuracy = float(best_baseline["accuracy"] or 0.0)
    tied = [name for name in BASELINE_NAMES if float(baselines[name]["accuracy"] or 0.0) == best_accuracy]
    if tied and best_name != min(tied):
        best_name = min(tied)
        best_baseline = baselines[best_name]

    defer = defer_metrics(rows, temperature=float(H4_CONFIG["temperature"]))
    e2e = end_to_end_selector_metrics(rows, test_examples, fallback_baseline="h1_soft_selector")
    wins = source_wins(rows)
    bootstrap = bootstrap_accuracy_delta(
        rows,
        test_examples,
        best_name,
        seed=int(H4_CONFIG["bootstrap"]["seed"]),
        rounds=int(H4_CONFIG["bootstrap"]["rounds"]),
    )
    swap = swap_consistency(models, stats_list, test_examples, device=device)
    resource = _resource_benchmark(models, stats_list, test_examples, device, pre_reserved_gib)
    gate = _gate(
        isolation=isolation,
        metrics=metrics,
        best_baseline=best_baseline,
        resource=resource,
        n_decisive=int(metrics.get("n_decisive", 0)),
    )

    payload: dict[str, Any] = {
        "schema_version": H4_CONFIG["schema_version"],
        "run_id": args.run_id,
        "mode": mode,
        "h4_config": H4_CONFIG,
        "h4_config_sha256": H4_CONFIG_SHA256,
        "verified": verified,
        "git_commit": _git_head(),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "split_manifest_sha256": split["manifest_sha256"],
        "normalization_stats": stats,
        "test_counts": {
            "test_valid_rows": len([r for r in split["rows"] if r["split"] == "test" and r["valid_pair"]]),
            "test_examples": len(test_examples),
            "decisive": int(metrics.get("n_decisive", 0)),
            "ties": len(test_examples) - int(metrics.get("n_decisive", 0)),
        },
        "isolation": isolation,
        "baselines": baselines,
        "best_baseline_name": best_name,
        "model_metrics": metrics,
        "defer": defer,
        "end_to_end": e2e,
        "source_wins": wins,
        "bootstrap": bootstrap,
        "swap": swap,
        "resource": resource,
        "gate": gate,
        "gate_status": "GATE_PASSED" if gate["passed"] else "GATE_FAILED",
        "h5_authorized": bool(gate["passed"] and defer["ranked_n"] > 0 and e2e["e2e_accuracy"] is not None and e2e["e2e_accuracy"] > float(e2e["fallback_accuracy"] or 0.0)),
        "h4_status": "COMPLETED / VERIFIED / STOPPED",
        "test_predictions": rows,
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    evidence_dir = ROOT / "docs" / "runtime-evidence" / "h4" / args.run_id
    generated_dir = ROOT / "generated" / "h4" / args.run_id
    lock_path = evidence_dir / "run.lock.json"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        raise RuntimeError(f"H4_RUN_LOCKED:{lock_path}")
    _write(evidence_dir / "final-delivery.json", payload)
    _write(generated_dir / "results.json", payload)
    _write(generated_dir / "normalization_stats.json", stats)
    _write(lock_path, {
        "run_id": args.run_id,
        "mode": mode,
        "script_sha256": payload["script_sha256"],
        "evidence_sha256": payload["evidence_sha256"],
        "locked": True,
    })
    return {"ok": True, "gate_status": payload["gate_status"], "failures": gate["failures"], "evidence": str(evidence_dir / "final-delivery.json"), "evidence_sha256": payload["evidence_sha256"]}


def _git_head() -> str:
    try:
        return subprocess_check(["git", "rev-parse", "HEAD"])
    except Exception:
        return "unknown"


def subprocess_check(args: Sequence[str]) -> str:
    import subprocess
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="h4-locked-20260816-final")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("verify", "dev-smoke", "evaluate"), default="evaluate")
    args = parser.parse_args()
    try:
        result = run_evaluate(args, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        error_payload = {"ok": False, "run_id": args.run_id, "mode": args.mode,
                         "error": f"{type(exc).__name__}:{exc}",
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if args.mode == "evaluate":
            failure_dir = ROOT / "docs" / "runtime-evidence" / "h4" / "failed" / f"{args.run_id}-{int(time.time())}"
            try:
                _write(failure_dir / "error.json", error_payload)
                error_payload["failure_evidence"] = str(failure_dir / "error.json")
            except Exception:
                pass
        print(json.dumps(error_payload, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
