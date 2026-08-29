"""Low-latency ensemble runtime for World v3."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from data_pipeline.h3.contracts import (
    H3_CANDIDATE_DIM,
    H3_CANDIDATE_STEPS,
    H3_CONTEXT_DIM,
    H3_SCHEMA_VERSION,
)
from data_pipeline.h6.contracts import (
    WORLD_V3_SCHEMA_VERSION,
    WorldV3Prediction,
    WorldV3ScoreResult,
    WORLD_VLA75_SCHEMA_VERSION,
    WorldVLA75Prediction,
    WorldVLA75ScoreResult,
)
from data_pipeline.h6.model import load_world_v3, load_world_vla75
from data_pipeline.h6.temporal import (
    TemporalSelectorConfig as TemporalStabilizerConfig,
    TemporalSelectorCore,
    normalize_source,
)


class TemporalPreferenceStabilizer:
    """Compatibility facade over the shared source-scoped selector."""

    def __init__(self, config: TemporalStabilizerConfig | None = None) -> None:
        self.config = config or TemporalStabilizerConfig()
        self._core = TemporalSelectorCore(self.config)

    def reset(self) -> None:
        self._core.reset()

    def metrics(self) -> dict[str, object]:
        metrics = self._core.metrics()
        return {
            "switches": metrics["switches"],
            "hold_ticks": metrics["hold_age"],
            "history": metrics["history"],
            "ping_pong": metrics["ping_pong"],
            "trace": metrics["trace"],
        }

    def update(
        self,
        scores: Mapping[str, float],
        *,
        raw_preferred_candidate_id: str,
        raw_preferred_source: str,
        candidate_sources: Mapping[str, str],
        event_break: bool = False,
        risk_breach: bool = False,
        eligible_changed: bool = False,
    ) -> tuple[str, str, dict[str, object]]:
        source_ids = {
            normalize_source(source): str(candidate_id)
            for candidate_id, source in candidate_sources.items()
        }
        source_scores = {
            normalize_source(candidate_sources[candidate_id]): float(value)
            for candidate_id, value in scores.items()
            if candidate_id in candidate_sources
        }
        raw_source = normalize_source(raw_preferred_source)
        decision = self._core.step(
            scope_key="temporal-preference-stabilizer",
            source_scores=source_scores,
            fresh_candidate_ids=source_ids,
            eligible_sources=set(source_ids),
            raw_preferred_source=raw_source,
            event_break=bool(event_break or eligible_changed),
            unsafe_sources=({raw_source} if risk_breach else set()),
        )
        if decision.selected_candidate_id is None or decision.selected_source is None:
            raise RuntimeError("temporal_stabilizer_selected_mrm")
        trace = decision.to_dict()
        trace.update(
            raw_preferred_candidate_id=str(raw_preferred_candidate_id),
            stabilized_preferred_candidate_id=decision.selected_candidate_id,
            stabilized_preferred_source=decision.selected_source,
            hold_ticks=decision.hold_age,
            switches=decision.switch_count,
        )
        return decision.selected_candidate_id, decision.selected_source, trace


class WorldV3Scorer:
    """Scores candidates without source identity; router applies VLA policy."""

    def __init__(
        self,
        models,
        model_hashes: Sequence[str],
        *,
        device: str | torch.device,
        vla_trust_threshold: float,
        vla_risk_ceiling: float,
        max_uncertainty: float = 2.0,
        objective_weight: float = 1.0,
        trust_weight: float = 2.5,
        risk_weight: float = 8.0,
        completion_weight: float = 1.0,
        repair_weight: float = 4.0,
        temperature_scales: Mapping[str, float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("world_v3_models_required")
        if len(models) != len(model_hashes):
            raise ValueError("world_v3_model_hash_count")
        self.models = tuple(model.eval() for model in models)
        self.device = torch.device(device)
        self.vla_trust_threshold = float(vla_trust_threshold)
        self.vla_risk_ceiling = float(vla_risk_ceiling)
        self.max_uncertainty = float(max_uncertainty)
        self.objective_weight = float(objective_weight)
        self.trust_weight = float(trust_weight)
        self.risk_weight = float(risk_weight)
        self.completion_weight = float(completion_weight)
        self.repair_weight = float(repair_weight)
        self.temperature_scales = {
            str(key): float(value) for key, value in (temperature_scales or {}).items()
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.temperature_scales.values()
        ):
            raise ValueError("world_v3_temperature_scales_invalid")
        self.model_hash = hashlib.sha256("|".join(model_hashes).encode("ascii")).hexdigest()
        self._cuda_baseline_allocated_bytes: float | None = None
        self.last_incremental_gpu_gib: float | None = None

    def _record_incremental_gpu(self) -> float | None:
        """Record scorer allocation growth relative to the loaded model.

        ``nvidia-smi`` remains the authoritative whole-card sampler.  This
        process-local value is the World-only delta used by the v2 admission
        gate; it is deliberately ``None`` on CPU so a formal run cannot
        silently turn an unmeasured resource into a passing zero.
        """

        if self.device.type != "cuda" or not torch.cuda.is_available():
            self.last_incremental_gpu_gib = None
            return None
        allocated = float(torch.cuda.memory_allocated(self.device))
        if self._cuda_baseline_allocated_bytes is None:
            self._cuda_baseline_allocated_bytes = allocated
        peak = float(torch.cuda.max_memory_allocated(self.device))
        self.last_incremental_gpu_gib = max(
            0.0, peak - self._cuda_baseline_allocated_bytes
        ) / (1024.0 ** 3)
        return self.last_incremental_gpu_gib

    def _scaled_logit(self, head: str, value: float) -> float:
        """Apply a development-fitted temperature to one hazard head."""

        temperature = self.temperature_scales.get(
            head,
            self.temperature_scales.get(f"{head}_logit", 1.0),
        )
        return float(value) / float(temperature)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Sequence[Path],
        *,
        device: str | torch.device,
        calibration: Mapping[str, float | bool],
    ) -> "WorldV3Scorer":
        models = []
        hashes = []
        for path in checkpoint_paths:
            models.append(load_world_v3(path, device=device)[0])
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return cls(
            models,
            hashes,
            device=device,
            vla_trust_threshold=float(calibration["trust_threshold"]),
            vla_risk_ceiling=float(calibration["risk_ceiling"]),
            temperature_scales=calibration.get("temperatures"),
        )

    @staticmethod
    def _validate(context, candidate) -> None:
        context_array = np.asarray(context, dtype=np.float32)
        candidate_array = np.asarray(candidate, dtype=np.float32)
        if context_array.shape != (H3_CONTEXT_DIM,):
            raise ValueError(f"world_v3_context_shape:{context_array.shape}")
        if candidate_array.shape != (H3_CANDIDATE_STEPS, H3_CANDIDATE_DIM):
            raise ValueError(f"world_v3_candidate_shape:{candidate_array.shape}")
        if not np.isfinite(context_array).all() or not np.isfinite(candidate_array).all():
            raise ValueError("world_v3_non_finite_input")

    def score_pair(self, first, second) -> WorldV3ScoreResult:
        started = time.perf_counter()
        payloads = (first, second)
        for _key, context, candidate in payloads:
            self._validate(context, candidate)
        context_tensor = torch.tensor(
            [list(first[1]), list(second[1])], dtype=torch.float32, device=self.device
        )
        candidate_tensor = torch.tensor(
            [list(first[2]), list(second[2])], dtype=torch.float32, device=self.device
        )
        outputs = []
        with torch.inference_mode():
            for model in self.models:
                outputs.append(model(context_tensor, candidate_tensor).detach().cpu().numpy())
        incremental_gpu_gib = self._record_incremental_gpu()
        stacked = np.stack(outputs, axis=0)
        mean = stacked.mean(axis=0)
        objective_std = stacked[:, :, 0].std(axis=0)
        predictions = []
        for index, payload in enumerate(payloads):
            row = mean[index]
            provisional = WorldV3Prediction(
                candidate_key=str(payload[0]),
                objective_utility=float(row[0]),
                progress_mean_m=float(row[1]),
                progress_logvar=float(np.clip(row[2], -6.0, 5.0)),
                completion_logit=float(row[3]),
                collision_logit=self._scaled_logit("collision", row[4]),
                red_light_logit=self._scaled_logit("red", row[5]),
                offroad_logit=self._scaled_logit("offroad", row[6]),
                jerk_mean_log1p=float(row[7]),
                acceleration_mean_mps2=float(row[8]),
                lateral_acceleration_mean_mps2=float(row[9]),
                repair_success_logit=self._scaled_logit("repair", row[10]),
                trust_logit=self._scaled_logit("trust", row[11]),
                ensemble_std=float(objective_std[index]),
            )
            score = (
                self.objective_weight * provisional.objective_utility
                + self.trust_weight * provisional.trust_probability
                - self.risk_weight * provisional.unsafe_probability
                + self.completion_weight * provisional.completion_probability
                + self.repair_weight
                * provisional.unsafe_probability
                * provisional.repair_success_probability
            )
            predictions.append(
                WorldV3Prediction(
                    **{
                        **provisional.__dict__,
                        "deployment_score": float(score),
                    }
                )
            )
        uncertainty = max(
            prediction.ensemble_std
            + math.sqrt(max(0.0, math.exp(prediction.progress_logvar))) / 20.0
            for prediction in predictions
        )
        if uncertainty > self.max_uncertainty:
            selected = None
            disposition = "defer_low_confidence"
            defer_reason = "world_v3_uncertainty"
        else:
            selected_prediction = max(
                predictions, key=lambda item: (item.deployment_score, item.candidate_key)
            )
            selected = selected_prediction.candidate_key
            disposition = "ranked"
            defer_reason = None
        delta = predictions[0].deployment_score - predictions[1].deployment_score
        delta = max(-40.0, min(40.0, delta))
        probability = 1.0 / (1.0 + math.exp(-delta))
        return WorldV3ScoreResult(
            disposition=disposition,
            selected_candidate_key=selected,
            predictions=tuple(predictions),
            probability_first_wins=probability,
            uncertainty=float(uncertainty),
            defer_reason=defer_reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_hash=self.model_hash,
            feature_schema=H3_SCHEMA_VERSION,
            trust_threshold=self.vla_trust_threshold,
            risk_ceiling=self.vla_risk_ceiling,
        )


class WorldVLA75Scorer(WorldV3Scorer):
    """VLA75 scorer with explicit preference and executable heads.

    It intentionally does not accept a source feature.  Candidate source is
    attached only by the offline collector/router when interpreting the pair.
    ``score_pair`` emits both ``predictions`` and ``raw_predictions`` so the
    formal gate can prove that stabilization never changed the raw 90% metric.
    """

    schema_version = WORLD_VLA75_SCHEMA_VERSION

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Sequence[Path],
        *,
        device: str | torch.device,
        calibration: Mapping[str, float | bool],
    ) -> "WorldVLA75Scorer":
        models = []
        hashes = []
        for path in checkpoint_paths:
            models.append(load_world_vla75(path, device=device)[0])
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return cls(
            models,
            hashes,
            device=device,
            vla_trust_threshold=float(calibration["trust_threshold"]),
            vla_risk_ceiling=float(calibration["risk_ceiling"]),
            temperature_scales=calibration.get("temperatures"),
        )

    def score_pair(self, first, second) -> WorldVLA75ScoreResult:
        started = time.perf_counter()
        payloads = (first, second)
        for _key, context, candidate in payloads:
            self._validate(context, candidate)
        context_tensor = torch.tensor(
            [list(first[1]), list(second[1])], dtype=torch.float32, device=self.device
        )
        candidate_tensor = torch.tensor(
            [list(first[2]), list(second[2])], dtype=torch.float32, device=self.device
        )
        outputs = []
        with torch.inference_mode():
            for model in self.models:
                result = model(context_tensor, candidate_tensor).detach().cpu().numpy()
                if result.shape[-1] != 14:
                    raise ValueError("world_vla75_checkpoint_output_dim")
                outputs.append(result)
        incremental_gpu_gib = self._record_incremental_gpu()
        stacked = np.stack(outputs, axis=0)
        mean = stacked.mean(axis=0)
        objective_std = stacked[:, :, 0].std(axis=0)
        predictions: list[WorldVLA75Prediction] = []
        for index, payload in enumerate(payloads):
            row = mean[index]
            provisional = WorldVLA75Prediction(
                candidate_key=str(payload[0]),
                objective_utility=float(row[0]),
                progress_mean_m=float(row[1]),
                progress_logvar=float(np.clip(row[2], -6.0, 5.0)),
                completion_logit=float(row[3]),
                collision_logit=self._scaled_logit("collision", row[4]),
                red_light_logit=self._scaled_logit("red", row[5]),
                offroad_logit=self._scaled_logit("offroad", row[6]),
                jerk_mean_log1p=float(row[7]),
                acceleration_mean_mps2=float(row[8]),
                lateral_acceleration_mean_mps2=float(row[9]),
                repair_success_logit=self._scaled_logit("repair", row[10]),
                trust_logit=self._scaled_logit("trust", row[11]),
                preference_utility=float(row[12]),
                executable_logit=float(row[13]),
                ensemble_std=float(objective_std[index]),
            )
            score = (
                self.objective_weight * provisional.objective_utility
                + self.trust_weight * provisional.trust_probability
                - self.risk_weight * provisional.unsafe_probability
                + self.completion_weight * provisional.completion_probability
                + self.repair_weight
                * provisional.unsafe_probability
                * provisional.repair_success_probability
            )
            predictions.append(
                WorldVLA75Prediction(
                    **{**provisional.__dict__, "deployment_score": float(score)}
                )
            )
        uncertainty = max(
            prediction.ensemble_std
            + math.sqrt(max(0.0, math.exp(prediction.progress_logvar))) / 20.0
            for prediction in predictions
        )
        raw_order = tuple(
            item.candidate_key
            for item in sorted(
                predictions,
                key=lambda item: (item.deployment_score, item.preference_utility, item.candidate_key),
                reverse=True,
            )
        )
        if uncertainty > self.max_uncertainty:
            selected = None
            disposition = "defer_low_confidence"
            defer_reason = "world_vla75_uncertainty"
        else:
            selected = raw_order[0]
            disposition = "ranked"
            defer_reason = None
        delta = predictions[0].deployment_score - predictions[1].deployment_score
        delta = max(-40.0, min(40.0, delta))
        probability = 1.0 / (1.0 + math.exp(-delta))
        return WorldVLA75ScoreResult(
            disposition=disposition,
            selected_candidate_key=selected,
            predictions=tuple(predictions),
            probability_first_wins=probability,
            uncertainty=float(uncertainty),
            defer_reason=defer_reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_hash=self.model_hash,
            feature_schema=H3_SCHEMA_VERSION,
            trust_threshold=self.vla_trust_threshold,
            risk_ceiling=self.vla_risk_ceiling,
            raw_preference_order=raw_order,
            world_incremental_gpu_gib=incremental_gpu_gib,
        )


__all__ = [
    "WorldV3Scorer",
    "WorldVLA75Scorer",
    "TemporalStabilizerConfig",
    "TemporalPreferenceStabilizer",
    "WORLD_V3_SCHEMA_VERSION",
    "WORLD_VLA75_SCHEMA_VERSION",
]
