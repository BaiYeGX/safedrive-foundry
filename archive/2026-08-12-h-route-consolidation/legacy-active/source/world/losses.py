"""Masked multi-task objectives for World-V0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import ActionBranchSample, WorldPrediction


@dataclass(frozen=True)
class WorldLossWeights:
    trajectory: float = 1.0
    collision: float = 1.0
    offroad: float = 0.5
    ttc: float = 0.5
    rank: float = 1.0
    tie: float = 0.25
    consistency: float = 0.1
    tie_margin: float = 0.10


def _tensor(
    values: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(values, (list, tuple)) and values and hasattr(values[0], "shape"):
        values = np.stack(values)
    return torch.as_tensor(values, device=device, dtype=dtype)


def stack_labels(
    samples: Sequence[ActionBranchSample],
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("stack_labels requires samples")
    for sample in samples:
        sample.validate()
    return {
        "future": _tensor(
            [sample.actor_future for sample in samples],
            device=device,
            dtype=torch.float32,
        ),
        "future_mask": _tensor(
            [sample.actor_future_mask for sample in samples],
            device=device,
            dtype=torch.bool,
        ),
        "outcomes": _tensor(
            [sample.outcomes for sample in samples],
            device=device,
            dtype=torch.float32,
        ),
        "outcome_mask": _tensor(
            [sample.outcome_mask for sample in samples],
            device=device,
            dtype=torch.bool,
        ),
        "rank_target": _tensor(
            [sample.rank_target for sample in samples],
            device=device,
            dtype=torch.float32,
        ),
        "rank_mask": _tensor(
            [sample.rank_mask for sample in samples],
            device=device,
            dtype=torch.bool,
        ),
        "rank_weight": _tensor(
            [sample.rank_weight for sample in samples],
            device=device,
            dtype=torch.float32,
        ),
        "tie_target": _tensor(
            [sample.tie_target for sample in samples],
            device=device,
            dtype=torch.bool,
        ),
    }


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _balanced_bce(logit: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid_target = target[mask]
    if not valid_target.numel():
        return logit.sum() * 0.0
    positives = valid_target.sum()
    negatives = valid_target.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(0.5, 20.0)
    loss = F.binary_cross_entropy_with_logits(
        logit,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    return _masked_mean(loss, mask)


def world_v0_loss(
    prediction: WorldPrediction,
    labels: dict[str, Tensor],
    *,
    weights: WorldLossWeights | None = None,
) -> dict[str, Tensor]:
    weights = weights or WorldLossWeights()
    future_target = labels["future"][..., (0, 1, 4, 5)]
    future_mask = labels["future_mask"].unsqueeze(-1).expand_as(future_target)
    trajectory_raw = F.smooth_l1_loss(
        prediction.actor_future_mean,
        future_target,
        reduction="none",
        beta=1.0,
    )
    trajectory = _masked_mean(trajectory_raw, future_mask)

    outcomes = labels["outcomes"]
    outcome_mask = labels["outcome_mask"]
    collision = _balanced_bce(
        prediction.collision_logit,
        outcomes[..., 0],
        outcome_mask,
    )
    offroad = _balanced_bce(
        prediction.offroad_logit,
        (outcomes[..., 1] > 0.02).to(outcomes.dtype),
        outcome_mask,
    )
    ttc_censored_target = (outcomes[..., 2] >= 2.5 - 1e-5).to(outcomes.dtype)
    ttc_finite_mask = outcome_mask & ~ttc_censored_target.bool()
    ttc_regression = _masked_mean(
        F.smooth_l1_loss(
            prediction.ttc_value,
            outcomes[..., 2],
            reduction="none",
            beta=0.5,
        ),
        ttc_finite_mask,
    )
    ttc_censored = _balanced_bce(
        prediction.ttc_censored_logit,
        ttc_censored_target,
        outcome_mask,
    )
    ttc = ttc_regression + ttc_censored

    score_delta = prediction.utility_score[:, 0] - prediction.utility_score[:, 1]
    decisive_mask = labels["rank_mask"] & ~labels["tie_target"]
    rank_binary = (labels["rank_target"] > 0).to(score_delta.dtype)
    rank_raw = F.binary_cross_entropy_with_logits(
        score_delta, rank_binary, reduction="none"
    )
    rank_weights = labels["rank_weight"] * decisive_mask.to(score_delta.dtype)
    rank = (rank_raw * rank_weights).sum() / rank_weights.sum().clamp_min(1.0)
    tie_mask = labels["rank_mask"] & labels["tie_target"]
    tie = _masked_mean(
        torch.relu(score_delta.abs() - weights.tie_margin),
        tie_mask,
    )

    risk_proxy = (
        torch.sigmoid(prediction.collision_logit)
        + 0.5 * torch.sigmoid(prediction.offroad_logit)
        - 0.1 * prediction.ttc_value.clamp(max=5.0)
    )
    consistency = _masked_mean(
        F.smooth_l1_loss(
            prediction.utility_score,
            -risk_proxy.detach(),
            reduction="none",
        ),
        outcome_mask,
    )
    total = (
        weights.trajectory * trajectory
        + weights.collision * collision
        + weights.offroad * offroad
        + weights.ttc * ttc
        + weights.rank * rank
        + weights.tie * tie
        + weights.consistency * consistency
    )
    return {
        "total": total,
        "trajectory": trajectory,
        "collision": collision,
        "offroad": offroad,
        "ttc": ttc,
        "rank": rank,
        "tie": tie,
        "consistency": consistency,
    }
