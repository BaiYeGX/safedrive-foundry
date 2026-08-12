"""Reusable deterministic training/evaluation helpers for World-V0."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch

from .contracts import ActionBranchSample, WorldBatch
from .losses import WorldLossWeights, stack_labels, world_v0_loss
from .metrics import evaluate_world_predictions
from .model_v0 import WorldV0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(
    samples: Sequence[ActionBranchSample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[list[ActionBranchSample]]:
    indices = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start : start + batch_size]]


@dataclass
class TrainResult:
    best_state: dict
    best_epoch: int
    best_val_loss: float
    global_step: int
    history: list[dict[str, float]]
    step_times_ms: list[float]
    peak_vram_mb: float


def evaluate_loss(
    model: WorldV0,
    samples: Sequence[ActionBranchSample],
    *,
    batch_size: int,
    weights: WorldLossWeights,
) -> float:
    if not samples:
        return float("nan")
    device = next(model.parameters()).device
    model.eval()
    totals: list[tuple[float, int]] = []
    with torch.no_grad():
        for chunk in batches(samples, batch_size=batch_size, shuffle=False, seed=0):
            prediction = model(WorldBatch.from_samples(chunk))
            loss = world_v0_loss(
                prediction,
                stack_labels(chunk, device=device),
                weights=weights,
            )["total"]
            totals.append((float(loss.detach().cpu()), len(chunk)))
    return sum(value * count for value, count in totals) / max(
        1, sum(count for _, count in totals)
    )


def train(
    model: WorldV0,
    train_samples: Sequence[ActionBranchSample],
    val_samples: Sequence[ActionBranchSample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    patience: int,
    seed: int,
    weights: WorldLossWeights,
    mixed_precision: bool,
) -> tuple[TrainResult, torch.optim.Optimizer, object | None]:
    if not train_samples:
        raise ValueError("training split is empty")
    seed_everything(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    global_step = 0
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(int(epochs)):
        model.train()
        epoch_total = 0.0
        epoch_count = 0
        for chunk in batches(
            train_samples,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(WorldBatch.from_samples(chunk))
                loss_map = world_v0_loss(
                    prediction,
                    stack_labels(chunk, device=device),
                    weights=weights,
                )
                loss = loss_map["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append((time.perf_counter() - started) * 1000.0)
            epoch_total += float(loss.detach().cpu()) * len(chunk)
            epoch_count += len(chunk)
            global_step += 1
        train_loss = epoch_total / max(1, epoch_count)
        val_loss = evaluate_loss(
            model,
            val_samples or train_samples,
            batch_size=batch_size,
            weights=weights,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return (
        TrainResult(
            best_state=best_state,
            best_epoch=best_epoch,
            best_val_loss=best_val,
            global_step=global_step,
            history=history,
            step_times_ms=step_times,
            peak_vram_mb=peak_vram,
        ),
        optimizer,
        scaler,
    )


@torch.no_grad()
def predict_samples(
    model: WorldV0,
    samples: Sequence[ActionBranchSample],
    *,
    batch_size: int,
) -> tuple[dict[str, float | int | dict], list]:
    predictions = []
    metrics_accumulator = []
    model.eval()
    for chunk in batches(samples, batch_size=batch_size, shuffle=False, seed=0):
        prediction = model(WorldBatch.from_samples(chunk))
        prediction.validate_finite()
        predictions.append(prediction)
        metrics_accumulator.append(evaluate_world_predictions(prediction, chunk))
    if not metrics_accumulator:
        return {}, predictions
    weighted_keys = (
        "actor_ADE_m",
        "actor_FDE_m",
        "decisive_pairwise_accuracy",
        "tie_false_decision_rate",
        "collision_brier",
        "offroad_brier",
    )
    report: dict[str, float | int | dict] = {}
    for key in weighted_keys:
        report[key] = float(np.mean([float(row[key]) for row in metrics_accumulator]))
    for key in (
        "actor_valid_points",
        "decisive_correct",
        "decisive_count",
        "tie_false_decisions",
        "tie_count",
        "outcome_count",
    ):
        report[key] = int(sum(int(row[key]) for row in metrics_accumulator))
    return report, predictions
