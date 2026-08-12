#!/usr/bin/env python3
"""Evaluate frozen R2 V4 checkpoints without opening the locked test early."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.semantic_mode_heads_v4 import SpatialSemanticHeadV4  # noqa: E402
from driving_vla.model.v4_token_features import split_ordered_tokens  # noqa: E402
from r2_v4_train_heads import (  # noqa: E402
    AUX_DIM,
    KIND_ORDER,
    V4Dataset,
    _metrics,
    _normalization,
    load_rows,
)


def _checkpoint(
    path: Path, device: torch.device
) -> tuple[SpatialSemanticHeadV4, np.ndarray, np.ndarray, str, float, dict[str, float]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    kwargs = dict(payload.get("model_kwargs") or {})
    model = SpatialSemanticHeadV4(**kwargs)
    if payload.get("lora"):
        from driving_vla.model.v4_lora import V4LoRAConfig, apply_v4_lora_qv

        apply_v4_lora_qv(model, V4LoRAConfig(**dict(payload["lora"].get("config") or payload["lora"])))
    model = model.to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    normalization = payload.get("normalization") or {}
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape != (AUX_DIM,) or std.shape != (AUX_DIM,):
        raise ValueError(f"{path}: train-only normalization missing or malformed")
    token_mode = str(payload.get("token_mode", "token-aware+history"))
    availability_threshold = float(payload.get("availability_threshold", 0.5))
    if not 0.0 < availability_threshold < 1.0:
        raise ValueError(f"{path}: frozen availability threshold is malformed")
    distillation = dict(
        (payload.get("lora") or {}).get("route_speed_distillation") or {}
    )
    drift = {
        "route_p95_drift_m": float(distillation.get("route_p95_drift_m", 0.0)),
        "speed_p95_drift_mps": float(distillation.get("speed_p95_drift_mps", 0.0)),
    }
    return model, mean, np.maximum(std, 1.0e-4), token_mode, availability_threshold, drift


def _bootstrap_lower(
    rows,
    model,
    mean,
    std,
    device,
    *,
    token_mode: str = "token-aware+history",
    availability_threshold: float = 0.5,
    seed: int = 3407,
    draws: int = 1000,
) -> float:
    groups: dict[str, list[Any]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    if len(groups) < 2:
        return 0.0
    rng = random.Random(seed)
    values: list[float] = []
    group_values = list(groups.values())
    for _ in range(int(draws)):
        sampled = [row for _ in range(len(group_values)) for row in rng.choice(group_values)]
        metrics = _metrics(
            model,
            DataLoader(
                V4Dataset(sampled, mean, std, token_mode=token_mode),
                batch_size=64,
            ),
            device=device,
            availability_threshold=availability_threshold,
        )
        values.append(float(metrics.get("macro_f1", 0.0)))
    return float(np.percentile(np.asarray(values), 2.5))


def _latency_ms(
    model,
    rows,
    mean,
    std,
    device,
    *,
    token_mode: str = "token-aware+history",
    repeats: int = 128,
) -> dict[str, float | int]:
    if not rows:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "deadline_miss": 0}
    ds = V4Dataset(rows[:1], mean, std, token_mode=token_mode)
    tokens, aux, *_ = ds[0]
    tokens = tokens[None].to(device)
    aux = aux[None].to(device)
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(int(repeats)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(tokens, aux)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "deadline_miss": sum(value > 5.0 for value in times),
    }


def _contract_metrics(rows) -> dict[str, float | int]:
    executable = [row for row in rows if bool(getattr(row, "available", False))]
    executable_pass = sum(bool(getattr(row, "guard_mpc_executable", False)) for row in executable)
    legal_pass = sum(bool(getattr(row, "legal_route_target", False)) for row in rows)
    return {
        "guard_mpc_executable_rate": executable_pass / max(len(executable), 1),
        "guard_mpc_executable_count": executable_pass,
        "guard_mpc_executable_denominator": len(executable),
        "legal_route_target_rate": legal_pass / max(len(rows), 1),
        "legal_route_target_count": legal_pass,
        "legal_route_target_denominator": len(rows),
    }


def evaluate(
    data: Path,
    checkpoints: list[Path],
    device_name: str,
    *,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    # Town13 is a locked test split.  It may only be opened after the
    # train/val selection has produced one immutable checkpoint; evaluating
    # several candidates here would leak test outcomes back into selection.
    if evaluate_test and len(checkpoints) != 1:
        raise ValueError(
            "locked Town13 test may be opened exactly once for one selected checkpoint"
        )
    rows = load_rows(data)
    device = torch.device(device_name)
    split_rows = {split: [row for row in rows if row.split == split] for split in ("train", "val", "test")}
    if evaluate_test and not split_rows["test"]:
        raise ValueError("locked Town13 test split is missing")
    reports: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        model, mean, std, token_mode, availability_threshold, drift = _checkpoint(checkpoint, device)
        val = _metrics(
            model,
            DataLoader(
                V4Dataset(split_rows["val"], mean, std, token_mode=token_mode),
                batch_size=64,
            ),
            device=device,
            availability_threshold=availability_threshold,
        )
        test = None
        if evaluate_test:
            test = _metrics(
                model,
                DataLoader(
                    V4Dataset(split_rows["test"], mean, std, token_mode=token_mode),
                    batch_size=64,
                ),
                device=device,
                availability_threshold=availability_threshold,
            )
        runtime_leak = sum(
            int(row_value.get("scenario_family_runtime_use", 0))
            + int(row_value.get("semantic_rescue_count", 0))
            for row in _jsonl_rows(data)
        )
        report: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
            "token_mode": token_mode,
            "availability_threshold": availability_threshold,
            "val": val,
            "test": test,
            "runtime_leakage": runtime_leak,
            "route_speed_distillation": drift,
        }
        if evaluate_test:
            report["root_lineage_bootstrap_macro_f1_lower95"] = _bootstrap_lower(
                split_rows["test"],
                model,
                mean,
                std,
                device,
                token_mode=token_mode,
                availability_threshold=availability_threshold,
            )
        else:
            report["root_lineage_bootstrap_macro_f1_lower95"] = None
        report["latency_ms"] = _latency_ms(
            model,
            split_rows["val"],
            mean,
            std,
            device,
            token_mode=token_mode,
        )
        report["contract"] = _contract_metrics(rows)
        reports.append(report)
    macro_std = statistics.pstdev(
        [float((report["test"] or {}).get("macro_f1", 0.0)) for report in reports]
    ) if len(reports) > 1 else 0.0
    for report in reports:
        test = report["test"] or {}
        recalls = test.get("kind_recall", {})
        gates = {
            "semantic_accuracy": float(test.get("accuracy", 0.0)) >= 0.90,
            "macro_f1": float(test.get("macro_f1", 0.0)) >= 0.85,
            "kind_recall": all(float(recalls.get(kind, 0.0)) >= 0.80 for kind in KIND_ORDER),
            "spatial_side_accuracy": float(test.get("side_accuracy_spatial", 0.0)) >= 0.95,
            "availability_precision": float(test.get("availability_precision", 0.0)) >= 0.90,
            "availability_recall": float(test.get("availability_recall", 0.0)) >= 0.90,
            "availability_specificity": float(test.get("availability_specificity", 0.0)) >= 0.90,
            "clear_none_specificity": float(test.get("clear_none_specificity", 0.0)) >= 1.0,
            "guard_mpc_executable_rate": float(report["contract"]["guard_mpc_executable_rate"]) >= 0.95,
            "legal_route_target": float(report["contract"]["legal_route_target_rate"]) >= 1.0,
            "bootstrap_lower95": (
                report["root_lineage_bootstrap_macro_f1_lower95"] is not None
                and float(report["root_lineage_bootstrap_macro_f1_lower95"]) >= 0.80
            ),
            "seed_std": macro_std <= 0.03,
            "scenario_family_runtime_use": report["runtime_leakage"] == 0,
            "head_latency_p95": float(report["latency_ms"]["p95"]) <= 5.0,
            "route_p95_drift": float(report["route_speed_distillation"]["route_p95_drift_m"]) <= 0.10,
            "speed_p95_drift": float(report["route_speed_distillation"]["speed_p95_drift_mps"]) <= 0.10,
        }
        report["hard_gates"] = gates
        report["all_hard_gates_pass"] = all(gates.values())
    # Select the sole formal checkpoint from train-only/val evidence before
    # Town13 is opened: all val hard gates first, then macro-F1, ECE, P95.
    val_eligible = []
    for report in reports:
        val = report.get("val") or {}
        recalls = val.get("kind_recall", {})
        contract = report.get("contract") or {}
        val_gates = {
            "semantic_accuracy": float(val.get("accuracy", 0.0)) >= 0.90,
            "macro_f1": float(val.get("macro_f1", 0.0)) >= 0.85,
            "kind_recall": all(float(recalls.get(kind, 0.0)) >= 0.80 for kind in KIND_ORDER),
            "spatial_side_accuracy": float(val.get("side_accuracy_spatial", 0.0)) >= 0.95,
            "availability_precision": float(val.get("availability_precision", 0.0)) >= 0.90,
            "availability_recall": float(val.get("availability_recall", 0.0)) >= 0.90,
            "availability_specificity": float(val.get("availability_specificity", 0.0)) >= 0.90,
            "clear_none_specificity": float(val.get("clear_none_specificity", 0.0)) >= 1.0,
            "guard_mpc_executable_rate": float(contract.get("guard_mpc_executable_rate", 0.0)) >= 0.95,
            "legal_route_target": float(contract.get("legal_route_target_rate", 0.0)) >= 1.0,
            "head_latency_p95": float(report.get("latency_ms", {}).get("p95", float("inf"))) <= 5.0,
        }
        report["val_hard_gates"] = val_gates
        if all(val_gates.values()):
            val_eligible.append(report)
    val_eligible.sort(
        key=lambda item: (
            -float((item.get("val") or {}).get("macro_f1", 0.0)),
            float((item.get("val") or {}).get("ece", float("inf"))),
            float((item.get("latency_ms") or {}).get("p95", float("inf"))),
            str(item.get("checkpoint_sha256") or ""),
        )
    )
    selection = {
        "eligible_count": len(val_eligible),
        "selected_checkpoint": (
            val_eligible[0].get("checkpoint") if val_eligible else None
        ),
        "selected_checkpoint_sha256": (
            val_eligible[0].get("checkpoint_sha256") if val_eligible else None
        ),
        "order": ["all_val_hard_gates", "macro_f1_desc", "ece_asc", "latency_p95_asc"],
    }
    return {
        "schema_version": "safedrive.r2_v4.formal_eval.v1",
        "data": str(data),
        "data_sha256": __import__("hashlib").sha256(data.read_bytes()).hexdigest(),
        "locked_test_opened_once": bool(evaluate_test),
        "test_evaluation_count": len(reports) if evaluate_test else 0,
        "locked_test_checkpoint_sha256": (
            reports[0].get("checkpoint_sha256")
            if evaluate_test and reports
            else None
        ),
        "seed_test_macro_f1_std": macro_std,
        "reports": reports,
        "selection": selection,
        "all_hard_gates_pass": bool(reports) and all(report["all_hard_gates_pass"] for report in reports),
    }


def _jsonl_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="explicitly open the sealed Town13 test split exactly once",
    )
    args = parser.parse_args()
    report = evaluate(
        Path(args.data),
        [Path(value) for value in args.checkpoint],
        args.device,
        evaluate_test=bool(args.evaluate_test),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_hard_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
