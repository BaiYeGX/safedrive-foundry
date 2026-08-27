"""Low-latency ensemble runtime for World v3."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TemporalStabilizerConfig:
    """Frozen development-selected temporal routing parameters."""

    ema_alpha: float = 0.50
    hold_ticks: int = 10
    hysteresis: float = 0.10
    emergency_switch_margin: float = 1.5
    ping_pong_window_ticks: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha_must_be_in_(0,1]")
        if int(self.hold_ticks) < 0:
            raise ValueError("hold_ticks_must_be_nonnegative")
        if float(self.hysteresis) < 0.0:
            raise ValueError("hysteresis_must_be_nonnegative")
        if float(self.emergency_switch_margin) < float(self.hysteresis):
            raise ValueError("emergency_margin_must_be_ge_hysteresis")


class TemporalPreferenceStabilizer:
    """EMA/hold/hysteresis over raw pair preference, with event breaks.

    This helper is deliberately independent of Safety.  It can only preserve
    a source that the caller has already marked eligible; it cannot create a
    candidate or bypass a Safety fallback.
    """

    def __init__(self, config: TemporalStabilizerConfig | None = None) -> None:
        self.config = config or TemporalStabilizerConfig()
        self.reset()

    def reset(self) -> None:
        self._ema: dict[str, float] = {}
        self._source: str | None = None
        self._candidate: str | None = None
        self._hold = 0
        self._history: list[str] = []
        self._switches = 0

    def metrics(self) -> dict[str, object]:
        return {
            "switches": self._switches,
            "hold_ticks": self._hold,
            "history": list(self._history),
            "ping_pong": self._ping_pong(),
        }

    def _ping_pong(self) -> bool:
        history = self._history
        window = int(self.config.ping_pong_window_ticks)
        for start, source in enumerate(history):
            if source not in {"vla", "expert"}:
                continue
            opposite = False
            for value in history[start + 1 : start + window + 1]:
                if value in {"vla", "expert"} and value != source:
                    opposite = True
                if opposite and value == source:
                    return True
        return False

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
        if not scores:
            raise ValueError("temporal_scores_required")
        values = {str(key): float(value) for key, value in scores.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("temporal_scores_must_be_finite")
        # Candidate ids are frame-scoped.  Retaining an old frame's id would
        # let a stale candidate win the max() call and would turn temporal
        # smoothing into an implicit trajectory generator.
        previous_ema = self._ema
        self._ema = {}
        for key, value in values.items():
            previous = previous_ema.get(key, value)
            self._ema[key] = self.config.ema_alpha * value + (1.0 - self.config.ema_alpha) * previous
        raw_id = str(raw_preferred_candidate_id)
        if raw_id not in values:
            raise ValueError("raw_preferred_candidate_missing")
        raw_source = str(raw_preferred_source)
        if raw_source not in {"vla", "expert", "mrm"}:
            raise ValueError("raw_preferred_source_invalid")
        proposed_id = max(self._ema, key=lambda key: (self._ema[key], key))
        proposed_source = str(candidate_sources.get(proposed_id, raw_source)).lower()
        if proposed_source in {"vla_fast", "vla_slow"}:
            proposed_source = "vla"
        elif proposed_source in {"classic", "classic_expert"}:
            proposed_source = "expert"
        if proposed_source not in {"vla", "expert", "mrm"}:
            proposed_source = raw_source
        margin = 0.0
        if len(self._ema) >= 2:
            ordered = sorted(self._ema.values(), reverse=True)
            margin = float(ordered[0] - ordered[1])
        break_now = bool(event_break or risk_breach or eligible_changed)
        previous_source = self._source
        keep = (
            not break_now
            and previous_source in {"vla", "expert"}
            and proposed_source != previous_source
            and self._hold < self.config.hold_ticks
            and (
                margin < self.config.hysteresis
                or margin < self.config.emergency_switch_margin
            )
        )
        if keep:
            selected_source = previous_source
            def _normalized_source(value: object) -> str:
                text = str(value).lower()
                if text in {"vla_fast", "vla_slow"}:
                    return "vla"
                if text in {"classic", "classic_expert"}:
                    return "expert"
                return text
            selected_id = next(
                (
                    key
                    for key, source in candidate_sources.items()
                    if _normalized_source(source) == previous_source and key in values
                ),
                proposed_id,
            )
            self._hold += 1
            reason = "hold_hysteresis"
        else:
            selected_source = proposed_source
            selected_id = proposed_id
            if previous_source is not None and selected_source != previous_source:
                self._switches += 1
            self._hold = 1
            reason = "event_break" if break_now else "ema_rank"
        self._source, self._candidate = selected_source, selected_id
        self._history.append(selected_source)
        return selected_id, selected_source, {
            "raw_preferred_candidate_id": raw_id,
            "raw_preferred_source": raw_source,
            "stabilized_preferred_candidate_id": selected_id,
            "stabilized_preferred_source": selected_source,
            "reason": reason,
            "margin": margin,
            "hold_ticks": self._hold,
            "switches": self._switches,
            "ping_pong": self._ping_pong(),
        }


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
