"""Persistence/CV/CTRV and observable rule baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .contracts import K, MAX_ACTORS, T, ActionBranchSample


def _last_valid(history: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int | None]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return np.zeros(history.shape[-1], dtype=np.float32), None
    index = int(indices[-1])
    return history[index], index


def predict_actor_future(
    sample: ActionBranchSample,
    *,
    mode: Literal["persistence", "cv", "ctrv"],
) -> tuple[np.ndarray, np.ndarray]:
    sample.validate()
    output = np.zeros((K, MAX_ACTORS, T, 4), dtype=np.float32)
    mask = np.zeros((K, MAX_ACTORS, T), dtype=bool)
    for actor_index in range(MAX_ACTORS):
        current, last_index = _last_valid(
            sample.actor_history[actor_index],
            sample.actor_history_mask[actor_index],
        )
        if last_index is None:
            continue
        x0, y0 = float(current[0]), float(current[1])
        vx, vy = float(current[4]), float(current[5])
        speed = math.hypot(vx, vy)
        yaw0 = math.atan2(float(current[2]), float(current[3]))
        yaw_rate = 0.0
        previous_indices = np.flatnonzero(sample.actor_history_mask[actor_index, :last_index])
        if len(previous_indices):
            previous = sample.actor_history[actor_index, int(previous_indices[-1])]
            previous_yaw = math.atan2(float(previous[2]), float(previous[3]))
            yaw_rate = math.atan2(
                math.sin(yaw0 - previous_yaw), math.cos(yaw0 - previous_yaw)
            ) / 0.25
        for ti in range(T):
            dt = (ti + 1) * 0.25
            if mode == "persistence":
                x, y, fx, fy = x0, y0, 0.0, 0.0
            elif mode == "cv" or abs(yaw_rate) < 1e-4:
                x, y, fx, fy = x0 + vx * dt, y0 + vy * dt, vx, vy
            else:
                yaw = yaw0 + yaw_rate * dt
                x = x0 + speed / yaw_rate * (math.sin(yaw) - math.sin(yaw0))
                y = y0 - speed / yaw_rate * (math.cos(yaw) - math.cos(yaw0))
                fx, fy = speed * math.cos(yaw), speed * math.sin(yaw)
            output[:, actor_index, ti] = (x, y, fx, fy)
            mask[:, actor_index, ti] = sample.candidate_mask
    return output, mask


@dataclass(frozen=True)
class ObservableRuleWeights:
    clearance: float = 3.0
    ttc: float = 2.0
    progress: float = 0.25
    curvature: float = 1.0
    comfort: float = 0.2


def observable_rule_scores(
    sample: ActionBranchSample,
    actor_prediction: np.ndarray,
    *,
    weights: ObservableRuleWeights | None = None,
) -> np.ndarray:
    """Score candidates using predicted geometry only; higher is better."""
    sample.validate()
    weights = weights or ObservableRuleWeights()
    score = np.full(K, -np.inf, dtype=np.float32)
    actor_present = sample.actor_history_mask[:, -1]
    for candidate_index in range(K):
        if not sample.candidate_mask[candidate_index]:
            continue
        candidate = sample.candidates[candidate_index]
        ego_xy = candidate[:, :2]
        actor_xy = actor_prediction[candidate_index, :, :, :2]
        if actor_present.any():
            distances = np.linalg.norm(
                actor_xy[actor_present] - ego_xy[None, :, :],
                axis=-1,
            )
            clearance = float(np.min(distances))
            ego_v = np.maximum(candidate[:, 4], 0.0)
            relative_range = np.maximum(distances, 0.01)
            ttc = float(np.min(relative_range / np.maximum(ego_v[None, :], 0.1)))
        else:
            clearance, ttc = 100.0, 10.0
        progress = float(ego_xy[-1, 0] - ego_xy[0, 0])
        curvature = float(np.mean(np.abs(candidate[:, 6])))
        comfort = float(np.mean(np.abs(candidate[:, 5])))
        score[candidate_index] = (
            weights.clearance * min(clearance, 20.0) / 20.0
            + weights.ttc * min(ttc, 5.0) / 5.0
            + weights.progress * progress
            - weights.curvature * curvature
            - weights.comfort * comfort
        )
    return score
