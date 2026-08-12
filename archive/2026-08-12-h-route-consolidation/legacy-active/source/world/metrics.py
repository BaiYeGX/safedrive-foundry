"""World-V0 predictive, ranking and action-contract metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .contracts import ActionBranchSample, WorldPrediction


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate_world_predictions(
    prediction: WorldPrediction,
    samples: Sequence[ActionBranchSample],
    *,
    tie_margin: float = 0.10,
) -> dict[str, Any]:
    mean = _numpy(prediction.actor_future_mean)
    scores = _numpy(prediction.utility_score)
    collision_probability = 1.0 / (1.0 + np.exp(-_numpy(prediction.collision_logit)))
    offroad_probability = 1.0 / (1.0 + np.exp(-_numpy(prediction.offroad_logit)))
    total_distance = 0.0
    total_points = 0
    fde_sum = 0.0
    fde_count = 0
    decisive_correct = 0
    decisive_count = 0
    decisive_slot = {0: [0, 0], 1: [0, 0]}
    tie_false = 0
    tie_count = 0
    collision_sq = 0.0
    offroad_sq = 0.0
    outcome_count = 0
    for bi, sample in enumerate(samples):
        target = sample.actor_future[..., (0, 1, 4, 5)]
        mask = sample.actor_future_mask
        distance = np.linalg.norm(mean[bi, ..., :2] - target[..., :2], axis=-1)
        total_distance += float(distance[mask].sum())
        total_points += int(mask.sum())
        for ki in range(2):
            for ai in range(mask.shape[1]):
                valid = np.flatnonzero(mask[ki, ai])
                if len(valid):
                    fde_sum += float(distance[ki, ai, int(valid[-1])])
                    fde_count += 1
        if sample.rank_mask and not sample.tie_target:
            predicted = 0 if scores[bi, 0] >= scores[bi, 1] else 1
            target_slot = 0 if sample.rank_target > 0 else 1
            correct = predicted == target_slot
            decisive_correct += int(correct)
            decisive_count += 1
            decisive_slot[target_slot][0] += int(correct)
            decisive_slot[target_slot][1] += 1
        elif sample.rank_mask and sample.tie_target:
            tie_false += int(abs(float(scores[bi, 0] - scores[bi, 1])) > tie_margin)
            tie_count += 1
        for ki in range(2):
            if sample.outcome_mask[ki]:
                collision_sq += (
                    float(collision_probability[bi, ki]) - float(sample.outcomes[ki, 0])
                ) ** 2
                offroad_target = float(sample.outcomes[ki, 1] > 0.02)
                offroad_sq += (
                    float(offroad_probability[bi, ki]) - offroad_target
                ) ** 2
                outcome_count += 1
    slot_accuracy = {
        str(slot): correct / max(1, count)
        for slot, (correct, count) in decisive_slot.items()
    }
    return {
        "actor_ADE_m": total_distance / max(1, total_points),
        "actor_FDE_m": fde_sum / max(1, fde_count),
        "actor_valid_points": total_points,
        "decisive_pairwise_accuracy": decisive_correct / max(1, decisive_count),
        "decisive_correct": decisive_correct,
        "decisive_count": decisive_count,
        "winner_slot_accuracy": slot_accuracy,
        "tie_false_decision_rate": tie_false / max(1, tie_count),
        "tie_false_decisions": tie_false,
        "tie_count": tie_count,
        "collision_brier": collision_sq / max(1, outcome_count),
        "offroad_brier": offroad_sq / max(1, outcome_count),
        "outcome_count": outcome_count,
    }


@torch.no_grad()
def candidate_swap_error(model: Any, batch: Any) -> dict[str, float]:
    original = model(batch)
    swapped_batch = type(batch)(
        ego_history=batch.ego_history,
        ego_history_mask=batch.ego_history_mask,
        actor_history=batch.actor_history,
        actor_history_mask=batch.actor_history_mask,
        road=batch.road,
        road_mask=batch.road_mask,
        candidates=np.asarray(batch.candidates)[:, ::-1].copy(),
        candidate_mask=np.asarray(batch.candidate_mask)[:, ::-1].copy(),
        sample_ids=batch.sample_ids,
    )
    swapped = model(swapped_batch)
    score_error = np.max(
        np.abs(
            _numpy(original.utility_score)[:, ::-1]
            - _numpy(swapped.utility_score)
        )
    )
    future_error = np.max(
        np.abs(
            _numpy(original.actor_future_mean)[:, ::-1]
            - _numpy(swapped.actor_future_mean)
        )
    )
    return {
        "score_max_abs_error": float(score_error),
        "future_max_abs_error": float(future_error),
    }


def action_sensitivity(
    conditioned: WorldPrediction,
    no_action: WorldPrediction,
) -> dict[str, float]:
    c_future = _numpy(conditioned.actor_future_mean)
    n_future = _numpy(no_action.actor_future_mean)
    c_score = _numpy(conditioned.utility_score)
    n_score = _numpy(no_action.utility_score)
    candidate_mask = _numpy(conditioned.candidate_mask).astype(bool)
    dual_mask = candidate_mask.all(axis=1)
    if dual_mask.any():
        conditioned_slot_future = float(
            np.mean(np.abs(c_future[dual_mask, 0] - c_future[dual_mask, 1]))
        )
        no_action_slot_future = float(
            np.mean(np.abs(n_future[dual_mask, 0] - n_future[dual_mask, 1]))
        )
        conditioned_slot_score = float(
            np.mean(np.abs(c_score[dual_mask, 0] - c_score[dual_mask, 1]))
        )
    else:
        conditioned_slot_future = 0.0
        no_action_slot_future = 0.0
        conditioned_slot_score = 0.0
    available_future_mask = candidate_mask[:, :, None, None, None]
    available_score_mask = candidate_mask
    return {
        "dual_candidate_batches": int(dual_mask.sum()),
        "conditioned_slot_future_delta_mean": conditioned_slot_future,
        "no_action_slot_future_delta_mean": no_action_slot_future,
        "conditioned_slot_score_delta_mean": conditioned_slot_score,
        "conditioned_vs_no_action_future_delta_mean": float(
            np.mean(np.abs(c_future - n_future)[available_future_mask.repeat(
                c_future.shape[2], axis=2
            ).repeat(c_future.shape[3], axis=3).repeat(c_future.shape[4], axis=4)])
        ),
        "conditioned_vs_no_action_score_delta_mean": float(
            np.mean(np.abs(c_score - n_score)[available_score_mask])
        ),
    }


def paired_ranking_bootstrap(
    conditioned: WorldPrediction | Sequence[WorldPrediction],
    no_action: WorldPrediction | Sequence[WorldPrediction],
    samples: Sequence[ActionBranchSample],
    *,
    n_resamples: int = 2000,
    seed: int = 3407,
) -> dict[str, float | int]:
    """Paired bootstrap of conditioned minus no-action decisive accuracy."""
    def scores(value: WorldPrediction | Sequence[WorldPrediction]) -> np.ndarray:
        if isinstance(value, Sequence):
            return np.concatenate(
                [_numpy(prediction.utility_score) for prediction in value], axis=0
            )
        return _numpy(value.utility_score)

    conditioned_scores = scores(conditioned)
    no_action_scores = scores(no_action)
    conditioned_correct: list[float] = []
    no_action_correct: list[float] = []
    for index, sample in enumerate(samples):
        if not sample.rank_mask or sample.tie_target:
            continue
        target = 0 if sample.rank_target > 0 else 1
        conditioned_slot = (
            0 if conditioned_scores[index, 0] >= conditioned_scores[index, 1] else 1
        )
        no_action_slot = (
            0 if no_action_scores[index, 0] >= no_action_scores[index, 1] else 1
        )
        conditioned_correct.append(float(conditioned_slot == target))
        no_action_correct.append(float(no_action_slot == target))
    count = len(conditioned_correct)
    if not count:
        return {
            "decisive_count": 0,
            "conditioned_accuracy": 0.0,
            "no_action_accuracy": 0.0,
            "improvement": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    conditioned_array = np.asarray(conditioned_correct)
    no_action_array = np.asarray(no_action_correct)
    delta = conditioned_array - no_action_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(int(n_resamples), count))
    boot = delta[indices].mean(axis=1)
    return {
        "decisive_count": count,
        "conditioned_accuracy": float(conditioned_array.mean()),
        "no_action_accuracy": float(no_action_array.mean()),
        "improvement": float(delta.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }
