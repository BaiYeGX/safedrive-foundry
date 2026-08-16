"""Offline-trained H3 scorer and safe rank/defer adapter.

The runtime scorer loads the frozen temperature from checkpoint metadata and
uses it for probability/defer calculations.  Source, slot, Guard, Oracle and
outcome fields do not exist in its public API.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

try:
    from driving_vla.hybrid.contracts import HybridCandidateSet, RoutingResult, WorldDisposition
    from driving_vla.hybrid.router import FrozenH1Router
except ImportError:
    from safedrive_foundry.driving_vla.hybrid.contracts import HybridCandidateSet, RoutingResult, WorldDisposition
    from safedrive_foundry.driving_vla.hybrid.router import FrozenH1Router

from .contracts import H3_CONFIG, H3_SCHEMA_VERSION, WorldPrediction, WorldScoreResult
from .model import CANDIDATE_DIM, CANDIDATE_STEPS, CONTEXT_DIM, WorldScorerModel, load_model
from .evaluate import sigmoid


class WorldScorer:
    """Shared candidate-conditioned scorer with a frozen calibration temperature."""

    def __init__(self, models: Sequence[WorldScorerModel], *, device: str = "cpu", model_hash: str = "", temperature: float | None = None) -> None:
        if not models:
            raise ValueError("world_scorer_requires_model")
        self.models = tuple(models)
        self.device = torch.device(device)
        self.temperature = float(temperature if temperature is not None else H3_CONFIG["runtime"]["temperature_bounds"][0])
        self.model_hash = model_hash or hashlib.sha256(str(len(models)).encode()).hexdigest()

    @classmethod
    def from_checkpoints(cls, checkpoints: Sequence[Path], *, device: str = "cpu") -> "WorldScorer":
        models = []
        digests = []
        temperatures = []
        for checkpoint in checkpoints:
            model, metadata = load_model(checkpoint, device=device)
            models.append(model)
            digests.append(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            temperatures.append(float(metadata.get("temperature", H3_CONFIG["loss"]["train_temperature"])))
        if len(set(temperatures)) != 1:
            raise ValueError(f"checkpoint_temperature_mismatch:{temperatures}")
        return cls(models, device=device, model_hash=hashlib.sha256("|".join(digests).encode()).hexdigest(), temperature=temperatures[0])

    def _validate(self, context: Sequence[float], candidate: Sequence[Sequence[float]]) -> None:
        if len(context) != CONTEXT_DIM:
            raise ValueError(f"world_context_dimension:{len(context)}")
        if len(candidate) != CANDIDATE_STEPS or any(len(row) != CANDIDATE_DIM for row in candidate):
            raise ValueError("world_candidate_shape")
        if not all(math.isfinite(float(value)) for value in context):
            raise ValueError("world_context_nonfinite")
        if not all(math.isfinite(float(value)) for row in candidate for value in row):
            raise ValueError("world_candidate_nonfinite")

    def score(self, context: Sequence[float], candidate: Sequence[Sequence[float]], candidate_key: str = "candidate") -> WorldPrediction:
        self._validate(context, candidate)
        context_tensor = torch.tensor([list(context)], dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor([list(candidate)], dtype=torch.float32, device=self.device)
        rows = []
        with torch.no_grad():
            for model in self.models:
                rows.append(model(context_tensor, candidate_tensor)[0].detach().cpu().tolist())
        values = [sum(row[index] for row in rows) / len(rows) for index in range(6)]
        return WorldPrediction(candidate_key, values[0], values[1], values[2], values[3], values[4], values[5])

    def score_pair(self, first: tuple[str, Sequence[float], Sequence[Sequence[float]]], second: tuple[str, Sequence[float], Sequence[Sequence[float]]]) -> WorldScoreResult:
        started = time.perf_counter()
        self._validate(first[1], first[2])
        self._validate(second[1], second[2])

        context_tensor = torch.tensor([list(first[1]), list(second[1])], dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor([list(first[2]), list(second[2])], dtype=torch.float32, device=self.device)
        model_outputs = []
        with torch.no_grad():
            for model in self.models:
                model_outputs.append(model(context_tensor, candidate_tensor).detach().cpu().numpy())

        avg_outputs = [
            [sum(m[cand_idx][head_idx] for m in model_outputs) / len(model_outputs) for head_idx in range(6)]
            for cand_idx in range(2)
        ]
        first_prediction = WorldPrediction(first[0], *[avg_outputs[0][i] for i in range(6)])
        second_prediction = WorldPrediction(second[0], *[avg_outputs[1][i] for i in range(6)])

        delta = first_prediction.utility - second_prediction.utility
        probability = sigmoid(delta / max(0.05, self.temperature))

        variance = 0.0
        if len(self.models) > 1:
            first_utils = [m[0][0] for m in model_outputs]
            second_utils = [m[1][0] for m in model_outputs]
            mean_first = sum(first_utils) / len(first_utils)
            mean_second = sum(second_utils) / len(second_utils)
            variance = 0.5 * (
                sum((u - mean_first) ** 2 for u in first_utils) / len(first_utils)
                + sum((u - mean_second) ** 2 for u in second_utils) / len(second_utils)
            )
        uncertainty = min(1.0, math.sqrt(max(0.0, float(variance))))
        margin = abs(delta)
        max_uncertainty = float(H3_CONFIG["runtime"]["max_uncertainty"])
        defer_margin = float(H3_CONFIG["runtime"]["defer_margin"])
        latency_ms = (time.perf_counter() - started) * 1000.0

        if uncertainty > max_uncertainty:
            disposition, selected, reason = "defer_low_confidence", None, "uncertainty_over_threshold"
        elif margin < defer_margin:
            disposition, selected, reason = "defer_low_confidence", None, "score_margin_below_threshold"
        else:
            disposition, selected, reason = "ranked", (first[0] if delta >= 0 else second[0]), None

        return WorldScoreResult(
            disposition=disposition,
            selected_candidate_key=selected,
            predictions=(first_prediction, second_prediction),
            probability_first_wins=probability,
            uncertainty=uncertainty,
            defer_reason=reason,
            latency_ms=latency_ms,
            model_hash=self.model_hash,
            feature_schema=H3_SCHEMA_VERSION,
            temperature=self.temperature,
        )


class H3WorldRouter:
    """World rank/defer wrapper that never bypasses Guard or Safety."""

    selector_name = "h3_world_v2"

    def __init__(self, scorer: WorldScorer | None, fallback: FrozenH1Router | None = None) -> None:
        self.scorer = scorer
        self.fallback = fallback or FrozenH1Router()

    def _defer(self, candidate_set: HybridCandidateSet, reason: str) -> RoutingResult:
        baseline = self.fallback.route(candidate_set)
        return replace(baseline, world=WorldDisposition.DEFERRED_LOW_CONFIDENCE, selector=baseline.selector, reason=f"h3_defer:{reason}")

    def route(self, candidate_set: HybridCandidateSet, features: Mapping[str, tuple[Sequence[float], Sequence[Sequence[float]]]] | None = None) -> RoutingResult:
        if self.scorer is None:
            return self._defer(candidate_set, "model_unavailable")
        passed = [item for item in candidate_set.candidates if item.guard is not None and item.guard.passed]
        if len(passed) < 2:
            result = self.fallback.route(candidate_set)
            return replace(result, world=WorldDisposition.DEFERRED_NOT_APPLICABLE, reason="h3_not_applicable_single_or_zero_pass")
        baseline = self.fallback.route(candidate_set)
        if baseline.selection_space.value == "NO_SELECTION_SPACE":
            return self._defer(candidate_set, "no_selection_space")
        if features is None:
            return self._defer(candidate_set, "feature_missing")
        ordered = sorted(passed, key=lambda item: item.candidate.candidate_id)
        payloads = []
        for item in ordered:
            key = item.candidate.candidate_id
            if key not in features:
                return self._defer(candidate_set, "feature_missing")
            context, candidate = features[key]
            payloads.append((key, context, candidate))
        try:
            score = self.scorer.score_pair(payloads[0], payloads[1])
        except (TypeError, ValueError, RuntimeError):
            return self._defer(candidate_set, "invalid_input")
        if score.latency_ms > float(H3_CONFIG["runtime"]["deadline_ms"]):
            return self._defer(candidate_set, "deadline")
        if score.disposition != "ranked" or score.selected_candidate_key is None:
            return self._defer(candidate_set, score.defer_reason or "low_confidence")
        return RoutingResult(
            pass_candidate_ids=baseline.pass_candidate_ids,
            rejected_candidate_ids=baseline.rejected_candidate_ids,
            selected_candidate_id=score.selected_candidate_key,
            selection_space=baseline.selection_space,
            world=WorldDisposition.RANKED,
            selector=self.selector_name,
            reason="h3_world_ranked",
            difference=baseline.difference,
            scores={item.candidate_key: item.utility for item in score.predictions},
        )


__all__ = ["H3WorldRouter", "WorldScorer"]
