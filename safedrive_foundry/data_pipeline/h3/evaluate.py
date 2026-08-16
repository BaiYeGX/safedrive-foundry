"""Out-of-fold metrics, defer curves and the frozen H3 gate for H3v2."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np
import scipy.optimize

from .baselines import baseline_winner
from .contracts import H3_CONFIG
from .dataset import PairExample
from .model import WorldScorerModel, ensemble_predict


def _ece(probabilities: Sequence[float], targets: Sequence[int], bins: int = 10) -> float:
    if not probabilities:
        return float("nan")
    total = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        selected = [
            i for i, probability in enumerate(probabilities)
            if lo <= probability < hi or (index == bins - 1 and probability <= hi)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[i] for i in selected) / len(selected)
        accuracy = sum(targets[i] for i in selected) / len(selected)
        total += len(selected) / len(probabilities) * abs(confidence - accuracy)
    return total


def _brier(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, targets)) / len(probabilities) if probabilities else float("nan")


def fit_temperature(deltas: Sequence[float], targets: Sequence[int], *, bounds: Sequence[float] | None = None) -> float:
    """Fit T* by NLL on provided (training-only) predictions."""
    if not deltas or not targets or len(deltas) != len(targets):
        return 1.0
    arr_deltas = np.array(deltas, dtype=np.float64)
    arr_targets = np.array(targets, dtype=np.float64)
    lo, hi = (float(x) for x in (bounds or H3_CONFIG["runtime"]["temperature_bounds"]))

    def nll(temp: float) -> float:
        t = max(lo, min(hi, float(temp)))
        logits = np.clip(arr_deltas / t, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        return float(-np.mean(arr_targets * np.log(probs + eps) + (1.0 - arr_targets) * np.log(1.0 - probs + eps)))

    result = scipy.optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    return float(result.x) if result.success else 1.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def prediction_rows(
    models: Sequence[WorldScorerModel],
    examples: Sequence[PairExample],
    *,
    device: str = "cpu",
    mask_context: bool = False,
    mask_candidate: bool = False,
    context_mask_mode: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        if not example.decisive:
            continue
        first, second, uncertainty = ensemble_predict(
            models, example, device=device, mask_context=mask_context, mask_candidate=mask_candidate,
            context_mask_mode=context_mask_mode,
        )
        delta = first.utility - second.utility
        predicted = 0 if delta >= 0.0 else 1
        rows.append(
            {
                "pair_id": example.pair_id,
                "delta": float(delta),
                "predicted": int(predicted),
                "target": 1 if example.winner_index == 0 else 0,
                "winner_index": int(example.winner_index),
                "uncertainty": float(uncertainty),
                "progress_regret": max(0.0, example.candidates[example.winner_index].progress_m - example.candidates[predicted].progress_m),
                "jerk_regret": max(0.0, example.candidates[predicted].jerk_rms_mps3 - example.candidates[example.winner_index].jerk_rms_mps3),
            }
        )
    return rows


def metrics_from_rows(rows: Sequence[dict[str, Any]], *, temperature: float) -> dict[str, Any]:
    decisive = list(rows)
    if not decisive:
        return {"n_decisive": 0, "correct": 0, "accuracy": None, "ece": float("nan"), "brier": float("nan")}
    correct = sum(int(row["predicted"]) == int(row["winner_index"]) for row in decisive)
    targets = [row["target"] for row in decisive]
    probabilities = [sigmoid(row["delta"] / temperature) for row in decisive]
    return {
        "n_decisive": len(decisive),
        "correct": correct,
        "accuracy": correct / len(decisive),
        "mean_progress_regret_m": sum(row["progress_regret"] for row in decisive) / len(decisive),
        "mean_jerk_regret_mps3": sum(row["jerk_regret"] for row in decisive) / len(decisive),
        "brier": _brier(probabilities, targets),
        "ece": _ece(probabilities, targets),
        "fitted_temperature": temperature,
        "mean_uncertainty": sum(row["uncertainty"] for row in decisive) / len(decisive),
    }


def evaluate_models(
    models: Sequence[WorldScorerModel],
    examples: Sequence[PairExample],
    *,
    device: str = "cpu",
    mask_context: bool = False,
    mask_candidate: bool = False,
    temperature: float | None = None,
) -> dict[str, Any]:
    rows = prediction_rows(models, examples, device=device, mask_context=mask_context, mask_candidate=mask_candidate)
    t_eff = temperature if temperature is not None else fit_temperature([row["delta"] for row in rows], [row["target"] for row in rows])
    return metrics_from_rows(rows, temperature=t_eff)


def evaluate_cv(
    fold_models: dict[str, Sequence[WorldScorerModel]],
    fold_examples: dict[str, Sequence[PairExample]],
    *,
    device: str = "cpu",
    temperature: float | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold, examples in sorted(fold_examples.items()):
        models = fold_models.get(fold, ())
        if not models:
            continue
        rows.extend(prediction_rows(models, examples, device=device))
    t_eff = temperature if temperature is not None else fit_temperature([row["delta"] for row in rows], [row["target"] for row in rows])
    metrics = metrics_from_rows(rows, temperature=t_eff)
    metrics["rows"] = rows
    return metrics


def bootstrap_accuracy_delta(
    models: Sequence[WorldScorerModel],
    examples: Sequence[PairExample],
    baseline_name: str,
    *,
    seed: int = 7,
    rounds: int | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    rounds = rounds or int(H3_CONFIG["acceptance"]["bootstrap_rounds"])
    rows = prediction_rows(models, examples, device=device)
    decisive = [item for item in examples if item.decisive]
    if not rows or not decisive:
        return {"delta": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    model_hits = [int(int(row["predicted"]) == int(row["winner_index"])) for row in rows]
    baseline_hits = [int(baseline_winner(item, baseline_name) == item.winner_index) for item in decisive]
    rng = random.Random(seed)
    deltas: list[float] = []
    n = len(rows)
    for _ in range(rounds):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(model_hits[index] - baseline_hits[index] for index in indices) / n)
    deltas.sort()
    delta = sum(model_hits) / n - sum(baseline_hits) / n
    return {"delta": delta, "lower_95": deltas[int(0.025 * len(deltas))], "upper_95": deltas[int(0.975 * len(deltas))]}


def swap_consistency(models: Sequence[WorldScorerModel], examples: Sequence[PairExample], *, device: str = "cpu") -> dict[str, Any]:
    failures = []
    max_error = 0.0
    for example in examples:
        first, second, _ = ensemble_predict(models, example, device=device)
        swapped = PairExample(
            example.pair_id, example.map_name, example.family, example.seed, example.weather, example.split,
            (example.candidates[1], example.candidates[0]),
            None if example.winner_index is None else 1 - example.winner_index,
            example.tie,
        )
        second_first, first_second, _ = ensemble_predict(models, swapped, device=device)
        error = max(abs(first.utility - first_second.utility), abs(second.utility - second_first.utility))
        max_error = max(max_error, error)
        if error > 1e-6:
            failures.append(example.pair_id)
    return {"checked": len(examples), "failures": failures, "max_error": max_error, "passed": not failures}


def evaluate_ablation(models: Sequence[WorldScorerModel], examples: Sequence[PairExample], *, device: str = "cpu", temperature: float) -> dict[str, Any]:
    full = evaluate_models(models, examples, device=device, temperature=temperature)
    action_mask = evaluate_models(models, examples, device=device, mask_candidate=True, temperature=temperature)
    history_mask = evaluate_models(models, examples, device=device, mask_context=True, temperature=temperature)
    return {
        "full": full,
        "action_mask": action_mask,
        "history_mask": history_mask,
        "action_accuracy_drop": (full["accuracy"] - action_mask["accuracy"]) if full["accuracy"] is not None and action_mask["accuracy"] is not None else None,
        "history_accuracy_drop": (full["accuracy"] - history_mask["accuracy"]) if full["accuracy"] is not None and history_mask["accuracy"] is not None else None,
    }


def defer_curve(models: Sequence[WorldScorerModel], examples: Sequence[PairExample], *, device: str = "cpu", temperature: float) -> list[dict[str, Any]]:
    rows = prediction_rows(models, examples, device=device)
    rows_sorted = sorted(rows, key=lambda row: (row["uncertainty"], -abs(row["delta"])))
    curve: list[dict[str, Any]] = []
    defer_margin = float(H3_CONFIG["runtime"]["defer_margin"])
    for cutoff in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.90):
        # Mirror the runtime scorer: cap uncertainty at 1.0 and require the
        # frozen score margin.
        kept = [
            row for row in rows_sorted
            if min(1.0, row["uncertainty"]) <= cutoff and abs(row["delta"]) >= defer_margin
        ]
        if not kept:
            curve.append({"coverage": 0.0, "cutoff": cutoff, "accuracy": None, "n": 0})
            continue
        correct = sum(int(row["predicted"]) == int(row["winner_index"]) for row in kept)
        curve.append({"coverage": len(kept) / len(rows), "cutoff": cutoff, "accuracy": correct / len(kept), "n": len(kept)})
    return curve


def h3_gate(
    *,
    leakage: dict,
    swap: dict,
    model_metrics: dict,
    best_baseline: dict,
    bootstrap: dict,
    ablation: dict,
    resource: dict,
    seed_metrics: Sequence[dict],
) -> dict[str, Any]:
    acc = H3_CONFIG["acceptance"]
    checks = {
        "leakage": bool(leakage.get("passed")),
        "swap": bool(swap.get("passed")) and float(swap.get("max_error", 1.0)) <= 1e-6,
        "accuracy_plus_2pp": float(model_metrics.get("accuracy", 0.0)) >= float(best_baseline.get("accuracy", 0.0)) + float(acc["accuracy_delta_pp"]) / 100.0,
        "progress_regret": float(model_metrics.get("mean_progress_regret_m", float("inf"))) <= float(best_baseline.get("mean_progress_regret_m", float("inf")) + 1e-9),
        "jerk_regret": float(model_metrics.get("mean_jerk_regret_mps3", float("inf"))) <= float(best_baseline.get("mean_jerk_regret_mps3", float("inf")) + 1e-9),
        "bootstrap_lower_nonnegative": float(bootstrap.get("lower_95", -1.0)) >= 0.0,
        "action_sensitivity": float(ablation.get("action_accuracy_drop", 0.0)) >= float(acc["action_accuracy_drop_pp"]) / 100.0,
        "history_sensitivity": float(ablation.get("history_accuracy_drop", 0.0)) >= float(acc["history_accuracy_drop_pp"]) / 100.0,
        "ece": float(model_metrics.get("ece", float("inf"))) <= float(acc["max_ece"]),
        "inference_resource": bool(resource.get("passed", False)),
        "seed_stability": sum(float(item.get("accuracy", 0.0)) >= float(best_baseline.get("accuracy", 0.0)) for item in seed_metrics) >= 4,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "best_baseline": best_baseline,
        "model": model_metrics,
        "bootstrap": bootstrap,
        "ablation": ablation,
        "resource": resource,
    }


__all__ = [
    "bootstrap_accuracy_delta",
    "defer_curve",
    "evaluate_ablation",
    "evaluate_cv",
    "evaluate_models",
    "fit_temperature",
    "h3_gate",
    "metrics_from_rows",
    "prediction_rows",
    "sigmoid",
    "swap_consistency",
]
