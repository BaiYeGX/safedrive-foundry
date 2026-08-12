#!/usr/bin/env python3
"""Train candidate-conditioned World-V0 or its no-action control."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.checkpoint import save_checkpoint
from driving_vla.world.dataset import ActionBranchDataset, file_sha256
from driving_vla.world.losses import WorldLossWeights
from driving_vla.world.model_v0 import WorldV0, WorldV0Config
from driving_vla.world.training import predict_samples, train


def _code_hash() -> str:
    h = hashlib.sha256()
    for path in sorted((ROOT / "safedrive_foundry/driving_vla/world").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "safedrive_foundry/config/world/world_v0.toml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--integrity-overfit", type=int, default=0)
    args = parser.parse_args()
    cfg_raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    model_values = dict(cfg_raw["model"])
    model_values.pop("schema_version", None)
    model_values["no_action"] = bool(args.no_action)
    model_config = WorldV0Config(**model_values)
    loss_weights = WorldLossWeights(**cfg_raw["loss"])
    train_cfg = cfg_raw["training"]
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu explicitly")
    device = torch.device(requested_device)
    dataset = ActionBranchDataset(Path(args.dataset))
    train_samples = list(dataset.iter_split("train"))
    val_samples = list(dataset.iter_split("val"))
    if args.integrity_overfit:
        train_samples = train_samples[: args.integrity_overfit]
        val_samples = train_samples
    model = WorldV0(model_config).to(device)
    gates = cfg_raw["gates"]
    if not int(gates["min_parameters"]) <= model.parameter_count <= int(gates["max_parameters"]):
        raise SystemExit(f"parameter budget violation: {model.parameter_count}")
    result, optimizer, scaler = train(
        model,
        train_samples,
        val_samples,
        epochs=args.epochs or int(train_cfg["epochs"]),
        batch_size=args.batch_size or int(train_cfg["batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
        patience=int(train_cfg["patience"]),
        seed=int(train_cfg["seed"]),
        weights=loss_weights,
        mixed_precision=str(train_cfg["precision"]) == "mixed",
    )
    val_metrics, _ = predict_samples(
        model,
        val_samples or train_samples,
        batch_size=args.batch_size or int(train_cfg["batch_size"]),
    )
    times = result.step_times_ms
    sorted_times = sorted(times)
    timing = {
        "step_time_p50_ms": statistics.median(times) if times else 0.0,
        "step_time_p95_ms": (
            sorted_times[min(len(sorted_times) - 1, int(0.95 * len(sorted_times)))]
            if sorted_times
            else 0.0
        ),
        "peak_vram_mb": result.peak_vram_mb,
    }
    manifest = save_checkpoint(
        Path(args.output),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=result.best_epoch,
        global_step=result.global_step,
        best_metric=result.best_val_loss,
        data_hash=dataset.manifest["dataset_content_hash"],
        split_hash=file_sha256(Path(args.dataset) / "split_manifest.json"),
        code_hash=_code_hash(),
        precision="fp16_amp" if device.type == "cuda" else "fp32",
        extra={
            "no_action": args.no_action,
            "integrity_overfit_n": args.integrity_overfit,
            "history": result.history,
            "validation": val_metrics,
            "timing": timing,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
