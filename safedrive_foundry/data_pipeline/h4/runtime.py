"""Normalized H4 runtime scorer.

H3's raw 5-seed ensemble has different per-seed utility scales, which made the
frozen uncertainty threshold defer every development pair.  H4 fixes this by
normalizing each seed's utility with dev-only mean/std before ensembling.  This
normalization is itself part of the H4 freeze and is computed without reading
test labels.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Sequence

import torch

from data_pipeline.h3.contracts import WorldPrediction, WorldScoreResult
from data_pipeline.h3.model import CONTEXT_DIM, CANDIDATE_DIM, CANDIDATE_STEPS, WorldScorerModel, load_model
from data_pipeline.h3.evaluate import sigmoid
from data_pipeline.h4.contracts import H4_CONFIG


class NormalizedWorldScorer:
    """Shared candidate-conditioned scorer with per-model utility normalization."""

    def __init__(
        self,
        models: Sequence[WorldScorerModel],
        stats: Sequence[tuple[float, float]],
        *,
        device: str = "cpu",
        model_hash: str = "",
        temperature: float | None = None,
        risk_defer_probability: float = 0.5,
    ) -> None:
        if not models:
            raise ValueError("world_scorer_requires_model")
        if len(models) != len(stats):
            raise ValueError("normalization_stats_model_count_mismatch")
        self.models = tuple(models)
        self.stats = [(float(mean), float(std)) for mean, std in stats]
        self.device = torch.device(device)
        self.temperature = float(temperature if temperature is not None else H4_CONFIG["temperature"])
        self.risk_defer_probability = float(risk_defer_probability)
        self.model_hash = model_hash or hashlib.sha256(str(len(models)).encode()).hexdigest()

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Sequence[tuple[Path, str]],
        stats: Sequence[tuple[float, float]],
        *,
        device: str = "cpu",
    ) -> "NormalizedWorldScorer":
        models = []
        digests = []
        temperatures = []
        for path, expected_sha in checkpoints:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != expected_sha:
                raise ValueError(f"checkpoint_sha_mismatch:{path}:{actual}")
            model, metadata = load_model(path, device=device)
            models.append(model)
            digests.append(actual)
            temperatures.append(float(metadata.get("temperature", H4_CONFIG["temperature"])))
        if len(set(temperatures)) != 1:
            raise ValueError(f"checkpoint_temperature_mismatch:{temperatures}")
        return cls(
            models,
            stats,
            device=device,
            model_hash=hashlib.sha256("|".join(digests).encode()).hexdigest(),
            temperature=temperatures[0],
        )

    def _validate(self, context: Sequence[float], candidate: Sequence[Sequence[float]]) -> None:
        if len(context) != CONTEXT_DIM:
            raise ValueError(f"world_context_dimension:{len(context)}")
        if len(candidate) != CANDIDATE_STEPS or any(len(row) != CANDIDATE_DIM for row in candidate):
            raise ValueError("world_candidate_shape")
        if not all(math.isfinite(float(value)) for value in context):
            raise ValueError("world_context_nonfinite")
        if not all(math.isfinite(float(value)) for row in candidate for value in row):
            raise ValueError("world_candidate_nonfinite")

    def score_pair(
        self,
        first: tuple[str, Sequence[float], Sequence[Sequence[float]]],
        second: tuple[str, Sequence[float], Sequence[Sequence[float]]],
    ) -> WorldScoreResult:
        started = time.perf_counter()
        self._validate(first[1], first[2])
        self._validate(second[1], second[2])

        context_tensor = torch.tensor([list(first[1]), list(second[1])], dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor([list(first[2]), list(second[2])], dtype=torch.float32, device=self.device)
        model_outputs = []
        with torch.no_grad():
            for model in self.models:
                model_outputs.append(model(context_tensor, candidate_tensor).detach().cpu().numpy())

        # Normalize only the utility head (index 0); keep other heads raw.
        normalized_outputs = []
        for model_index, outputs in enumerate(model_outputs):
            mean, std = self.stats[model_index]
            normalized = outputs.copy()
            for cand_index in range(2):
                normalized[cand_index, 0] = (outputs[cand_index, 0] - mean) / max(1e-9, std)
            normalized_outputs.append(normalized)

        avg_outputs = [
            [sum(normalized_outputs[m][cand_idx][head_idx] for m in range(len(normalized_outputs))) / len(normalized_outputs) for head_idx in range(6)]
            for cand_idx in range(2)
        ]
        first_prediction = WorldPrediction(first[0], *[avg_outputs[0][i] for i in range(6)])
        second_prediction = WorldPrediction(second[0], *[avg_outputs[1][i] for i in range(6)])

        delta = first_prediction.utility - second_prediction.utility
        probability = sigmoid(delta / max(0.05, self.temperature))

        variance = 0.0
        if len(self.models) > 1:
            first_utils = [normalized_outputs[m][0][0] for m in range(len(normalized_outputs))]
            second_utils = [normalized_outputs[m][1][0] for m in range(len(normalized_outputs))]
            mean_first = sum(first_utils) / len(first_utils)
            mean_second = sum(second_utils) / len(second_utils)
            variance = 0.5 * (
                sum((u - mean_first) ** 2 for u in first_utils) / len(first_utils)
                + sum((u - mean_second) ** 2 for u in second_utils) / len(second_utils)
            )
        uncertainty = min(1.0, math.sqrt(max(0.0, float(variance))))
        margin = abs(delta)
        max_uncertainty = float(H4_CONFIG["runtime"]["max_uncertainty"])
        defer_margin = float(H4_CONFIG["runtime"]["defer_margin"])
        latency_ms = (time.perf_counter() - started) * 1000.0

        # The structured risk head must be part of the online decision, not a
        # disconnected auxiliary output.  If either candidate is predicted to
        # be hard-unsafe above the frozen probability threshold, World defers
        # to the non-learning fallback instead of selecting a risky trajectory.
        risk_first = sigmoid(first_prediction.risk_logit)
        risk_second = sigmoid(second_prediction.risk_logit)
        if max(risk_first, risk_second) > self.risk_defer_probability:
            disposition, selected, reason = "defer_low_confidence", None, "predicted_hard_risk_over_threshold"
        elif uncertainty > max_uncertainty:
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
            feature_schema=H4_CONFIG["schema_version"],
            temperature=self.temperature,
        )


__all__ = ["NormalizedWorldScorer"]
