"""H4 locked-evaluation metrics.

All probability metrics use the frozen H4 temperature and the normalized
5-seed ensemble.  No function in this module reads test labels itself; callers
pass already-loaded examples/rows.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from data_pipeline.h3.baselines import baseline_winner
from data_pipeline.h3.contracts import WorldPrediction
from data_pipeline.h3.dataset import PairExample
from data_pipeline.h3.model import WorldScorerModel, predict_model
from data_pipeline.h4.contracts import H4_CONFIG
from data_pipeline.h4.locked_dataset import LockedPairExample


def _auc(deltas: Sequence[float], targets: Sequence[int]) -> float | None:
    """Mann-Whitney U AUC for candidate-0-wins scores."""
    if not deltas or len(deltas) != len(targets):
        return None
    classes = set(int(t) for t in targets)
    if classes != {0, 1}:
        return None
    pos = [float(d) for d, t in zip(deltas, targets) if int(t) == 1]
    neg = [float(d) for d, t in zip(deltas, targets) if int(t) == 0]
    combined = sorted([(d, 1) for d in pos] + [(d, 0) for d in neg])
    # Average ranks for ties.
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, combined) if label == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _brier(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, targets)) / len(probabilities) if probabilities else float("nan")


def _ece(probabilities: Sequence[float], targets: Sequence[int], bins: int = 10) -> float:
    if not probabilities:
        return float("nan")
    total = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        selected = [
            i for i, p in enumerate(probabilities)
            if lo <= p < hi or (index == bins - 1 and p <= hi)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[i] for i in selected) / len(selected)
        accuracy = sum(targets[i] for i in selected) / len(selected)
        total += len(selected) / len(probabilities) * abs(confidence - accuracy)
    return total


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def normalized_prediction_rows(
    models: Sequence[WorldScorerModel],
    stats: Sequence[tuple[float, float]],
    examples: Sequence[LockedPairExample | PairExample],
    *,
    device: str = "cpu",
    mask_context: bool = False,
    mask_candidate: bool = False,
    context_mask_mode: str | None = None,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    """Produce prediction rows using the normalized H4 ensemble.

    ``stats`` must be aligned with ``models``.
    """
    temp = float(temperature if temperature is not None else H4_CONFIG["temperature"])
    rows: list[dict[str, Any]] = []
    for example in examples:
        pair = example.pair if isinstance(example, LockedPairExample) else example
        if not pair.decisive:
            continue
        utilities: list[list[float]] = [[], []]
        all_predictions: list[tuple[Any, Any]] = []
        for model_index, model in enumerate(models):
            p0, p1 = predict_model(model, pair, device=device, mask_context=mask_context, mask_candidate=mask_candidate, context_mask_mode=context_mask_mode)
            all_predictions.append((p0, p1))
            mean, std = stats[model_index]
            utilities[0].append((p0.utility - mean) / max(1e-9, std))
            utilities[1].append((p1.utility - mean) / max(1e-9, std))
        avg0 = sum(utilities[0]) / len(utilities[0])
        avg1 = sum(utilities[1]) / len(utilities[1])
        delta = avg0 - avg1
        predicted = 0 if delta >= 0.0 else 1
        variance = 0.5 * (
            sum((u - avg0) ** 2 for u in utilities[0]) / len(utilities[0])
            + sum((u - avg1) ** 2 for u in utilities[1]) / len(utilities[1])
        )
        uncertainty = min(1.0, math.sqrt(max(0.0, variance)))
        # Use normalized utility only for ranking; other heads remain raw mean.
        p0_avg = WorldPrediction(
            all_predictions[0][0].candidate_key,
            avg0,
            sum(p.progress_mean_m for p, _ in all_predictions) / len(all_predictions),
            sum(p.progress_logvar for p, _ in all_predictions) / len(all_predictions),
            sum(p.jerk_mean_log1p for p, _ in all_predictions) / len(all_predictions),
            sum(p.jerk_logvar for p, _ in all_predictions) / len(all_predictions),
            sum(p.risk_logit for p, _ in all_predictions) / len(all_predictions),
        )
        p1_avg = WorldPrediction(
            all_predictions[0][1].candidate_key,
            avg1,
            sum(p.progress_mean_m for _, p in all_predictions) / len(all_predictions),
            sum(p.progress_logvar for _, p in all_predictions) / len(all_predictions),
            sum(p.jerk_mean_log1p for _, p in all_predictions) / len(all_predictions),
            sum(p.jerk_logvar for _, p in all_predictions) / len(all_predictions),
            sum(p.risk_logit for _, p in all_predictions) / len(all_predictions),
        )
        row: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "map": pair.map_name,
            "family": pair.family,
            "seed": pair.seed,
            "delta": float(delta),
            "predicted": int(predicted),
            "target": 1 if pair.winner_index == 0 else 0,
            "winner_index": int(pair.winner_index),
            "uncertainty": float(uncertainty),
            "progress_regret": max(0.0, pair.candidates[pair.winner_index].progress_m - pair.candidates[predicted].progress_m),
            "jerk_regret": max(0.0, pair.candidates[predicted].jerk_rms_mps3 - pair.candidates[pair.winner_index].jerk_rms_mps3),
            "winner_source": example.sources[pair.winner_index] if isinstance(example, LockedPairExample) else None,
            "candidate_sources": list(example.sources) if isinstance(example, LockedPairExample) else None,
            "probability_first_wins": sigmoid(delta / temp),
        }
        rows.append(row)
    return rows


def metrics_from_rows(rows: Sequence[dict[str, Any]], *, temperature: float) -> dict[str, Any]:
    decisive = list(rows)
    if not decisive:
        return {"n_decisive": 0, "correct": 0, "accuracy": None, "auroc": None,
                "ece": float("nan"), "brier": float("nan"), "nll": float("nan")}
    correct = sum(int(row["predicted"]) == int(row["winner_index"]) for row in decisive)
    probabilities = [row["probability_first_wins"] for row in decisive]
    targets = [row["target"] for row in decisive]
    nll = -sum(
        t * math.log(max(1e-12, p)) + (1 - t) * math.log(max(1e-12, 1 - p))
        for p, t in zip(probabilities, targets)
    ) / len(decisive)
    return {
        "n_decisive": len(decisive),
        "correct": correct,
        "accuracy": correct / len(decisive),
        "auroc": _auc([row["delta"] for row in decisive], targets),
        "mean_progress_regret_m": sum(row["progress_regret"] for row in decisive) / len(decisive),
        "mean_jerk_regret_mps3": sum(row["jerk_regret"] for row in decisive) / len(decisive),
        "brier": _brier(probabilities, targets),
        "ece": _ece(probabilities, targets, bins=int(H4_CONFIG["ece_bins"])),
        "nll": nll,
        "fitted_temperature": temperature,
        "mean_uncertainty": sum(row["uncertainty"] for row in decisive) / len(decisive),
    }


def defer_metrics(rows: Sequence[dict[str, Any]], *, temperature: float) -> dict[str, Any]:
    """Runtime-faithful defer coverage using the frozen H4 thresholds."""
    decisive = list(rows)
    if not decisive:
        return {"coverage": 0.0, "ranked_n": 0, "decisive_n": 0,
                "accuracy_ranked": None, "mean_uncertainty_ranked": None}
    max_uncertainty = float(H4_CONFIG["runtime"]["max_uncertainty"])
    defer_margin = float(H4_CONFIG["runtime"]["defer_margin"])
    ranked = []
    for row in decisive:
        margin = abs(row["delta"])
        if row["uncertainty"] <= max_uncertainty and margin >= defer_margin:
            ranked.append(row)
    accuracy = sum(int(row["predicted"]) == int(row["winner_index"]) for row in ranked) / len(ranked) if ranked else None
    return {
        "coverage": len(ranked) / len(decisive),
        "ranked_n": len(ranked),
        "decisive_n": len(decisive),
        "accuracy_ranked": accuracy,
        "mean_uncertainty_ranked": sum(row["uncertainty"] for row in ranked) / len(ranked) if ranked else None,
    }


def end_to_end_selector_metrics(
    rows: Sequence[dict[str, Any]],
    examples: Sequence[LockedPairExample],
    *,
    fallback_baseline: str = "h1_soft_selector",
) -> dict[str, Any]:
    """Simulate the online rule: World when ranked, frozen fallback otherwise."""
    by_id = {row["pair_id"]: row for row in rows}
    fallback_correct = 0
    fallback_n = 0
    e2e_correct = 0
    e2e_n = 0
    fallback_hits: list[int] = []
    for example in examples:
        pair = example.pair
        if not pair.decisive:
            continue
        row = by_id.get(pair.pair_id)
        if row is None:
            continue
        fallback_pred = baseline_winner(pair, fallback_baseline)
        fallback_hit = int(fallback_pred == pair.winner_index)
        fallback_correct += fallback_hit
        fallback_n += 1
        margin = abs(row["delta"])
        ranked = row["uncertainty"] <= float(H4_CONFIG["runtime"]["max_uncertainty"]) and margin >= float(H4_CONFIG["runtime"]["defer_margin"])
        e2e_pred = row["predicted"] if ranked else fallback_pred
        e2e_correct += int(e2e_pred == pair.winner_index)
        e2e_n += 1
        fallback_hits.append(fallback_hit)
    return {
        "fallback_baseline": fallback_baseline,
        "fallback_accuracy": fallback_correct / fallback_n if fallback_n else None,
        "e2e_accuracy": e2e_correct / e2e_n if e2e_n else None,
        "n_decisive": e2e_n,
    }


def source_wins(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    decisive = [row for row in rows if row.get("winner_source") is not None]
    expert = sum(1 for row in decisive if row["winner_source"] == "expert")
    vla = sum(1 for row in decisive if row["winner_source"] == "vla")
    unknown = sum(1 for row in decisive if row["winner_source"] not in {"expert", "vla"})
    return {"expert_wins": expert, "vla_wins": vla, "unknown_source_wins": unknown, "n_decisive": len(decisive)}


def bootstrap_accuracy_delta(
    rows: Sequence[dict[str, Any]],
    examples: Sequence[LockedPairExample],
    baseline_name: str,
    *,
    seed: int = 7,
    rounds: int = 10000,
) -> dict[str, float]:
    rows = list(rows)
    decisive = [example for example in examples if example.pair.decisive]
    if not rows or not decisive:
        return {"delta": float("nan"), "lower_95": float("nan"), "upper_95": float("nan"), "n": 0}
    model_hits = [int(row["predicted"] == row["winner_index"]) for row in rows]
    baseline_hits = [int(baseline_winner(example.pair, baseline_name) == example.pair.winner_index) for example in decisive]
    if len(model_hits) != len(baseline_hits):
        raise ValueError("bootstrap_row_count_mismatch")
    rng = random.Random(seed)
    n = len(rows)
    deltas = []
    for _ in range(rounds):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(model_hits[i] - baseline_hits[i] for i in indices) / n)
    deltas.sort()
    return {
        "delta": sum(model_hits) / n - sum(baseline_hits) / n,
        "lower_95": deltas[int(0.025 * len(deltas))],
        "upper_95": deltas[int(0.975 * len(deltas))],
        "n": n,
    }


def swap_consistency(
    models: Sequence[WorldScorerModel],
    stats: Sequence[tuple[float, float]],
    examples: Sequence[LockedPairExample],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    failures: list[str] = []
    max_error = 0.0
    for example in examples:
        pair = example.pair
        first, second, _ = _normalized_pair_predict(models, stats, pair, device=device)
        swapped_pair = PairExample(
            pair.pair_id, pair.map_name, pair.family, pair.seed, pair.weather, pair.split,
            (pair.candidates[1], pair.candidates[0]),
            None if pair.winner_index is None else 1 - pair.winner_index,
            pair.tie,
        )
        second_first, first_second, _ = _normalized_pair_predict(models, stats, swapped_pair, device=device)
        error = max(abs(first.utility - first_second.utility), abs(second.utility - second_first.utility))
        max_error = max(max_error, error)
        if error > 1e-6:
            failures.append(pair.pair_id)
    return {"checked": len(examples), "failures": failures, "max_error": float(max_error), "passed": not failures}


def _normalized_pair_predict(
    models: Sequence[WorldScorerModel],
    stats: Sequence[tuple[float, float]],
    pair: PairExample,
    *,
    device: str,
    context_mask_mode: str | None = None,
) -> tuple[Any, Any, float]:
    utilities: list[list[float]] = [[], []]
    all_predictions = []
    for model_index, model in enumerate(models):
        p0, p1 = predict_model(model, pair, device=device, context_mask_mode=context_mask_mode)
        all_predictions.append((p0, p1))
        mean, std = stats[model_index]
        utilities[0].append((p0.utility - mean) / max(1e-9, std))
        utilities[1].append((p1.utility - mean) / max(1e-9, std))
    avg0 = sum(utilities[0]) / len(utilities[0])
    avg1 = sum(utilities[1]) / len(utilities[1])
    variance = 0.5 * (
        sum((u - avg0) ** 2 for u in utilities[0]) / len(utilities[0])
        + sum((u - avg1) ** 2 for u in utilities[1]) / len(utilities[1])
    )
    uncertainty = math.sqrt(max(0.0, variance))
    first = WorldPrediction(
        all_predictions[0][0].candidate_key,
        avg0,
        sum(p.progress_mean_m for p, _ in all_predictions) / len(all_predictions),
        sum(p.progress_logvar for p, _ in all_predictions) / len(all_predictions),
        sum(p.jerk_mean_log1p for p, _ in all_predictions) / len(all_predictions),
        sum(p.jerk_logvar for p, _ in all_predictions) / len(all_predictions),
        sum(p.risk_logit for p, _ in all_predictions) / len(all_predictions),
    )
    second = WorldPrediction(
        all_predictions[0][1].candidate_key,
        avg1,
        sum(p.progress_mean_m for _, p in all_predictions) / len(all_predictions),
        sum(p.progress_logvar for _, p in all_predictions) / len(all_predictions),
        sum(p.jerk_mean_log1p for _, p in all_predictions) / len(all_predictions),
        sum(p.jerk_logvar for _, p in all_predictions) / len(all_predictions),
        sum(p.risk_logit for _, p in all_predictions) / len(all_predictions),
    )
    return first, second, uncertainty


__all__ = [
    "bootstrap_accuracy_delta",
    "defer_metrics",
    "end_to_end_selector_metrics",
    "metrics_from_rows",
    "normalized_prediction_rows",
    "source_wins",
    "swap_consistency",
]
