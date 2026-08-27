"""Contracts for the outcome-aware, VLA-primary World v3 scorer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


WORLD_V3_SCHEMA_VERSION = "safedrive.world.v3.outcome_trust.v1"
WORLD_V3_OUTPUT_DIM = 12
WORLD_VLA75_SCHEMA_VERSION = "safedrive.world.vla75.pair_exec.v1"
WORLD_VLA75_OUTPUT_DIM = WORLD_V3_OUTPUT_DIM + 2


@dataclass(frozen=True)
class WorldV3Prediction:
    candidate_key: str
    objective_utility: float
    progress_mean_m: float
    progress_logvar: float
    completion_logit: float
    collision_logit: float
    red_light_logit: float
    offroad_logit: float
    jerk_mean_log1p: float
    acceleration_mean_mps2: float
    lateral_acceleration_mean_mps2: float
    repair_success_logit: float
    trust_logit: float
    ensemble_std: float = 0.0
    deployment_score: float = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    @property
    def utility(self) -> float:
        """Compatibility alias consumed by the online router."""
        return float(self.deployment_score)

    @property
    def collision_probability(self) -> float:
        return self._sigmoid(self.collision_logit)

    @property
    def red_light_probability(self) -> float:
        return self._sigmoid(self.red_light_logit)

    @property
    def offroad_probability(self) -> float:
        return self._sigmoid(self.offroad_logit)

    @property
    def completion_probability(self) -> float:
        return self._sigmoid(self.completion_logit)

    @property
    def repair_success_probability(self) -> float:
        return self._sigmoid(self.repair_success_logit)

    @property
    def trust_probability(self) -> float:
        return self._sigmoid(self.trust_logit)

    @property
    def unsafe_probability(self) -> float:
        # Union approximation for three separately calibrated hazards.
        safe = (
            (1.0 - self.collision_probability)
            * (1.0 - self.red_light_probability)
            * (1.0 - self.offroad_probability)
        )
        return 1.0 - safe

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "utility": self.utility,
                "completion_probability": self.completion_probability,
                "collision_probability": self.collision_probability,
                "red_light_probability": self.red_light_probability,
                "offroad_probability": self.offroad_probability,
                "repair_success_probability": self.repair_success_probability,
                "trust_probability": self.trust_probability,
                "unsafe_probability": self.unsafe_probability,
            }
        )
        return payload


@dataclass(frozen=True)
class WorldV3ScoreResult:
    disposition: str
    selected_candidate_key: str | None
    predictions: tuple[WorldV3Prediction, ...]
    probability_first_wins: float | None
    uncertainty: float | None
    defer_reason: str | None
    latency_ms: float
    model_hash: str
    feature_schema: str
    trust_threshold: float
    risk_ceiling: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "selected_candidate_key": self.selected_candidate_key,
            "predictions": [item.to_dict() for item in self.predictions],
            "probability_first_wins": self.probability_first_wins,
            "uncertainty": self.uncertainty,
            "defer_reason": self.defer_reason,
            "latency_ms": self.latency_ms,
            "model_hash": self.model_hash,
            "feature_schema": self.feature_schema,
            "trust_threshold": self.trust_threshold,
            "risk_ceiling": self.risk_ceiling,
        }


@dataclass(frozen=True)
class WorldVLA75Prediction:
    """VLA75 candidate prediction with preference and executability heads.

    The encoder remains source-blind.  ``preference_utility`` is a per
    candidate quantity; a pair is preferred by comparing the VLA candidate's
    value with the Expert candidate's value.  Consequently swapping the two
    candidate tensors swaps the values and flips the pair margin without ever
    exposing a source bit to the model.
    """

    candidate_key: str
    objective_utility: float
    progress_mean_m: float
    progress_logvar: float
    completion_logit: float
    collision_logit: float
    red_light_logit: float
    offroad_logit: float
    jerk_mean_log1p: float
    acceleration_mean_mps2: float
    lateral_acceleration_mean_mps2: float
    repair_success_logit: float
    trust_logit: float
    preference_utility: float
    executable_logit: float
    ensemble_std: float = 0.0
    deployment_score: float = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    @property
    def utility(self) -> float:
        return float(self.deployment_score)

    @property
    def collision_probability(self) -> float:
        return self._sigmoid(self.collision_logit)

    @property
    def red_light_probability(self) -> float:
        return self._sigmoid(self.red_light_logit)

    @property
    def offroad_probability(self) -> float:
        return self._sigmoid(self.offroad_logit)

    @property
    def completion_probability(self) -> float:
        return self._sigmoid(self.completion_logit)

    @property
    def repair_success_probability(self) -> float:
        return self._sigmoid(self.repair_success_logit)

    @property
    def trust_probability(self) -> float:
        return self._sigmoid(self.trust_logit)

    @property
    def executable_probability(self) -> float:
        return self._sigmoid(self.executable_logit)

    @property
    def unsafe_probability(self) -> float:
        safe = (
            (1.0 - self.collision_probability)
            * (1.0 - self.red_light_probability)
            * (1.0 - self.offroad_probability)
        )
        return 1.0 - safe

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "utility": self.utility,
                "completion_probability": self.completion_probability,
                "collision_probability": self.collision_probability,
                "red_light_probability": self.red_light_probability,
                "offroad_probability": self.offroad_probability,
                "repair_success_probability": self.repair_success_probability,
                "trust_probability": self.trust_probability,
                "executable_probability": self.executable_probability,
                "unsafe_probability": self.unsafe_probability,
            }
        )
        return payload

    @classmethod
    def from_v3(
        cls, prediction: WorldV3Prediction, *, preference_utility: float | None = None,
        executable_logit: float = 0.0,
    ) -> "WorldVLA75Prediction":
        return cls(
            candidate_key=prediction.candidate_key,
            objective_utility=prediction.objective_utility,
            progress_mean_m=prediction.progress_mean_m,
            progress_logvar=prediction.progress_logvar,
            completion_logit=prediction.completion_logit,
            collision_logit=prediction.collision_logit,
            red_light_logit=prediction.red_light_logit,
            offroad_logit=prediction.offroad_logit,
            jerk_mean_log1p=prediction.jerk_mean_log1p,
            acceleration_mean_mps2=prediction.acceleration_mean_mps2,
            lateral_acceleration_mean_mps2=prediction.lateral_acceleration_mean_mps2,
            repair_success_logit=prediction.repair_success_logit,
            trust_logit=prediction.trust_logit,
            preference_utility=(
                prediction.objective_utility
                if preference_utility is None
                else float(preference_utility)
            ),
            executable_logit=executable_logit,
            ensemble_std=prediction.ensemble_std,
            deployment_score=prediction.deployment_score,
        )


@dataclass(frozen=True)
class WorldVLA75ScoreResult:
    """Pair execution score emitted by the VLA75 runtime."""

    disposition: str
    selected_candidate_key: str | None
    predictions: tuple[WorldVLA75Prediction, ...]
    probability_first_wins: float | None
    uncertainty: float | None
    defer_reason: str | None
    latency_ms: float
    model_hash: str
    feature_schema: str
    trust_threshold: float
    risk_ceiling: float
    raw_preference_order: tuple[str, ...] = ()
    world_incremental_gpu_gib: float | None = None

    @property
    def preference_margin(self) -> float | None:
        """Raw ``q_first - q_second`` pair margin.

        The scorer never uses source identity to construct this value.  When
        callers swap the two candidate tensors, the same per-candidate heads
        appear in the opposite order and this margin therefore changes sign;
        the property is useful to audit that invariant without relying on the
        stabilized router choice.
        """

        if len(self.predictions) != 2:
            return None
        return float(
            self.predictions[0].preference_utility
            - self.predictions[1].preference_utility
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORLD_VLA75_SCHEMA_VERSION,
            "disposition": self.disposition,
            "selected_candidate_key": self.selected_candidate_key,
            "predictions": [item.to_dict() for item in self.predictions],
            "raw_predictions": [item.to_dict() for item in self.predictions],
            "probability_first_wins": self.probability_first_wins,
            "uncertainty": self.uncertainty,
            "defer_reason": self.defer_reason,
            "latency_ms": self.latency_ms,
            "model_hash": self.model_hash,
            "feature_schema": self.feature_schema,
            "trust_threshold": self.trust_threshold,
            "risk_ceiling": self.risk_ceiling,
            "raw_preference_order": list(self.raw_preference_order),
            "preference_margin": self.preference_margin,
            "pair_preference_margin": self.preference_margin,
            "world_incremental_gpu_gib": self.world_incremental_gpu_gib,
        }


__all__ = [
    "WORLD_V3_OUTPUT_DIM",
    "WORLD_V3_SCHEMA_VERSION",
    "WORLD_VLA75_OUTPUT_DIM",
    "WORLD_VLA75_SCHEMA_VERSION",
    "WorldV3Prediction",
    "WorldV3ScoreResult",
    "WorldVLA75Prediction",
    "WorldVLA75ScoreResult",
]
