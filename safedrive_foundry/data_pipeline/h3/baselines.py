"""Source-neutral H3 ranking baselines.

``h1_soft_selector`` is computed from the real Safety soft scorer at dataset
build time; it is no longer an alias of the hand reward.
"""

from __future__ import annotations

import math
from typing import Sequence

from .dataset import CandidateExample, PairExample


def _planned_length(candidate: CandidateExample) -> float:
    points = candidate.candidate
    return sum(
        math.hypot((points[i][0] - points[i - 1][0]) * 50.0, (points[i][1] - points[i - 1][1]) * 10.0)
        for i in range(1, len(points))
    )


def _final_speed(candidate: CandidateExample) -> float:
    return float(candidate.candidate[-1][4]) * 10.0


def _planned_jerk(candidate: CandidateExample) -> float:
    points = candidate.candidate
    if len(points) < 2:
        return 0.0
    jerks = []
    for i in range(1, len(points)):
        a_curr = float(points[i][5]) * 10.0
        a_prev = float(points[i - 1][5]) * 10.0
        t_curr = float(points[i][7]) * 2.5
        t_prev = float(points[i - 1][7]) * 2.5
        dt = max(0.01, t_curr - t_prev)
        jerks.append((a_curr - a_prev) / dt)
    return math.sqrt(sum(j * j for j in jerks) / len(jerks)) if jerks else 0.0


def _planned_curvature_rms(candidate: CandidateExample) -> float:
    kappas = [float(p[6]) * 0.5 for p in candidate.candidate]
    return math.sqrt(sum(k * k for k in kappas) / len(kappas)) if kappas else 0.0


def _comfort_score(candidate: CandidateExample) -> float:
    return -1.0 * _planned_jerk(candidate)


def _hand_reward(candidate: CandidateExample) -> float:
    return _planned_length(candidate) - 0.35 * _planned_jerk(candidate)


def _candidate_only_score(candidate: CandidateExample) -> float:
    return _planned_length(candidate) + 0.5 * _final_speed(candidate) - 0.25 * _planned_jerk(candidate) - 2.0 * _planned_curvature_rms(candidate)


def _cv_ctrv_score(candidate: CandidateExample) -> float:
    context = candidate.context
    points = candidate.candidate
    actors = []
    for idx in range(8):
        base = 140 + 204 + idx * 12
        if base + 11 >= len(context):
            break
        dx, dy = context[base] * 50.0, context[base + 1] * 20.0
        vx, vy = context[base + 2] * 10.0, context[base + 3] * 10.0
        length_m, lost = context[base + 6] * 10.0, context[base + 8] > 0.5
        if length_m > 0.1 and not lost:
            actors.append((dx, dy, vx, vy))

    min_clearance = 50.0
    for pt in points:
        t = float(pt[7]) * 2.5
        ego_x, ego_y = float(pt[0]) * 50.0, float(pt[1]) * 10.0
        for ax0, ay0, avx, avy in actors:
            dist = math.hypot(ego_x - (ax0 + avx * t), ego_y - (ay0 + avy * t))
            min_clearance = min(min_clearance, dist)

    red_light_penalty = 0.0
    for idx in range(6):
        base = 140 + 204 + 8 * 12 + idx * 9
        if base + 8 >= len(context):
            break
        stop_dist = context[base + 1] * 100.0
        is_red = context[base + 2] > 0.5
        controls_lane = context[base + 6] > 0.5
        if is_red and controls_lane and stop_dist > 0:
            final_x = float(points[-1][0]) * 50.0
            if final_x > stop_dist:
                red_light_penalty += 50.0

    clearance_bonus = min(20.0, min_clearance)
    collision_penalty = 100.0 if min_clearance < 2.5 else (20.0 if min_clearance < 4.0 else 0.0)
    return _planned_length(candidate) + 0.5 * clearance_bonus - collision_penalty - red_light_penalty - 0.2 * _planned_jerk(candidate)


def baseline_winner(example: PairExample, name: str) -> int:
    if name == "no_action":
        return 0
    if name == "planned_length":
        values = [_planned_length(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "final_speed":
        values = [_final_speed(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "planned_jerk":
        values = [-_planned_jerk(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "comfort":
        values = [_comfort_score(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "hand_reward":
        values = [_hand_reward(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "candidate_only":
        values = [_candidate_only_score(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "cv_ctrv":
        values = [_cv_ctrv_score(item) for item in example.candidates]
        return int(values[1] > values[0])
    if name == "h1_soft_selector":
        values = [item.h1_soft_score for item in example.candidates]
        return int(values[1] > values[0])
    raise ValueError(f"unknown_baseline:{name}")


def evaluate_baseline(examples: Sequence[PairExample], name: str) -> dict:
    decisive = [item for item in examples if item.decisive]
    correct = sum(baseline_winner(item, name) == item.winner_index for item in decisive)
    regrets = [
        max(0.0, item.candidates[item.winner_index].progress_m - item.candidates[baseline_winner(item, name)].progress_m)
        for item in decisive
    ]
    jerk_regrets = [
        max(0.0, item.candidates[baseline_winner(item, name)].jerk_rms_mps3 - item.candidates[item.winner_index].jerk_rms_mps3)
        for item in decisive
    ]
    return {
        "name": name,
        "n_decisive": len(decisive),
        "correct": correct,
        "accuracy": correct / len(decisive) if decisive else None,
        "mean_progress_regret_m": sum(regrets) / len(regrets) if regrets else None,
        "mean_jerk_regret_mps3": sum(jerk_regrets) / len(jerk_regrets) if jerk_regrets else None,
    }


BASELINE_NAMES = (
    "no_action",
    "candidate_only",
    "planned_length",
    "final_speed",
    "planned_jerk",
    "cv_ctrv",
    "h1_soft_selector",
    "hand_reward",
    "comfort",
)

__all__ = ["BASELINE_NAMES", "baseline_winner", "evaluate_baseline"]
