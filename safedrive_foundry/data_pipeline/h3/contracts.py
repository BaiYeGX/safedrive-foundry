"""Frozen H3v2 configuration and small serializable contracts.

H3v2 fixes the original H3 data/evaluation boundary violations:
* no hand-written candidates, no fake Guard/Safety,
* all acceptance metrics are out-of-fold,
* runtime temperature is frozen in the checkpoint and used online,
* richer observable context than v1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

H3_SCHEMA_VERSION = "safedrive.h3.world_scorer.v2"
H3_SPLIT_VERSION = "safedrive.h3.lineage_split.v2"
H3_EVIDENCE_VERSION = "safedrive.h3.evidence.v2"

# Feature tensor contract.  The model, dataset, runtime and tests all import
# these constants; no component may hard-code its own shape.
H3_HISTORY_TICKS = 20
H3_ROUTE_POINTS = 51
H3_ACTOR_SLOTS = 8
H3_LIGHT_SLOTS = 6
H3_CANDIDATE_STEPS = 10
H3_CANDIDATE_DIM = 8
# ego history 20*7 + route 51*4 + actors 8*12 + lights 6*9 + ego current 5
H3_CONTEXT_DIM = 140 + 204 + 96 + 54 + 5

H3_CONFIG: dict[str, Any] = {
    "schema_version": H3_SCHEMA_VERSION,
    "split_version": H3_SPLIT_VERSION,
    "evidence_version": H3_EVIDENCE_VERSION,
    "split_rule": "rank0_test_rank1_3_dev_by_map_family_lineage_hash",
    "lineage": ["map", "family", "seed"],
    "weather_together": True,
    "history_ticks": H3_HISTORY_TICKS,
    "route_points": H3_ROUTE_POINTS,
    "actor_slots": H3_ACTOR_SLOTS,
    "light_slots": H3_LIGHT_SLOTS,
    "candidate_steps": H3_CANDIDATE_STEPS,
    "candidate_dim": H3_CANDIDATE_DIM,
    "context_dim": H3_CONTEXT_DIM,
    "candidate_dt_s": 0.25,
    "model": {
        "d_model": 128,
        "layers": 2,
        "heads": 4,
        "ffn": 256,
        "dropout": 0.1,
    },
    "loss": {
        "pairwise_bce": 1.0,
        "progress_nll": 0.5,
        "jerk_nll": 0.25,
        "risk_bce": 1.0,
        "tie_margin": 0.25,
        "tie_margin_value": 0.1,
        "train_temperature": 1.0,
    },
    "optimizer": {
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "gradient_clip": 1.0,
        "max_epochs": 350,
        "patience": 40,
    },
    "training_seeds": [11, 23, 37, 53, 71],
    "runtime": {
        "deadline_ms": 50.0,
        "max_incremental_gpu_gib": 1.5,
        "defer_margin": 0.05,
        "max_uncertainty": 0.35,
        "temperature_bounds": [0.05, 10.0],
    },
    "targets": {
        "risk_supported": True,
        "risk_reason": "H3v2 challenge data is required to contain hard-unsafe positives",
    },
    "challenge": {
        "dataset_id_prefix": "h3-carla-challenge-v2",
        "anchors": 96,
        "pilot_anchors": 12,
        "maps": ["Town01", "Town03", "Town05"],
        "families": [
            "emergency_lead_brake",
            "aggressive_cut_in",
            "red_light_dilemma",
            "cross_traffic_conflict",
        ],
        "seeds": [0, 1, 2, 3],
        "weathers": ["ClearNoon", "CloudyNoon"],
    },
    "acceptance": {
        "accuracy_delta_pp": 2.0,
        "action_accuracy_drop_pp": 5.0,
        "history_accuracy_drop_pp": 2.0,
        "max_ece": 0.10,
        "bootstrap_rounds": 10000,
        "min_hard_unsafe_branches": 4,
        "max_source_only_baseline": 0.80,
        "min_valid_pairs": 72,
        "min_per_map_valid": 20,
        "min_per_family_valid": 12,
        "min_per_weather_valid": 30,
        "min_decisive": 48,
        "min_source_wins": 5,
    },
}


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


H3_CONFIG_SHA256 = stable_sha256(H3_CONFIG)


@dataclass(frozen=True)
class SplitRow:
    pair_id: str
    map_name: str
    family: str
    seed: int
    weather: str
    lineage: str
    lineage_rank: int
    split: str
    valid_pair: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "map": self.map_name,
            "family": self.family,
            "seed": self.seed,
            "weather": self.weather,
            "lineage": self.lineage,
            "lineage_rank": self.lineage_rank,
            "split": self.split,
            "valid_pair": self.valid_pair,
        }


@dataclass(frozen=True)
class WorldPrediction:
    """One candidate prediction; source/slot/Guard fields are never accepted."""

    candidate_key: str
    utility: float
    progress_mean_m: float
    progress_logvar: float
    jerk_mean_log1p: float
    jerk_logvar: float
    risk_logit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "utility": float(self.utility),
            "progress_mean_m": float(self.progress_mean_m),
            "progress_logvar": float(self.progress_logvar),
            "jerk_mean_log1p": float(self.jerk_mean_log1p),
            "jerk_logvar": float(self.jerk_logvar),
            "risk_logit": float(self.risk_logit),
        }


@dataclass(frozen=True)
class WorldScoreResult:
    disposition: str
    selected_candidate_key: str | None
    predictions: tuple[WorldPrediction, ...]
    probability_first_wins: float | None
    uncertainty: float | None
    defer_reason: str | None
    latency_ms: float
    model_hash: str
    feature_schema: str
    temperature: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "selected_candidate_key": self.selected_candidate_key,
            "predictions": [item.to_dict() for item in self.predictions],
            "probability_first_wins": self.probability_first_wins,
            "uncertainty": self.uncertainty,
            "defer_reason": self.defer_reason,
            "latency_ms": float(self.latency_ms),
            "model_hash": self.model_hash,
            "feature_schema": self.feature_schema,
            "temperature": self.temperature,
        }


__all__ = [
    "H3_ACTOR_SLOTS",
    "H3_CANDIDATE_DIM",
    "H3_CANDIDATE_STEPS",
    "H3_CONFIG",
    "H3_CONFIG_SHA256",
    "H3_CONTEXT_DIM",
    "H3_EVIDENCE_VERSION",
    "H3_HISTORY_TICKS",
    "H3_LIGHT_SLOTS",
    "H3_ROUTE_POINTS",
    "H3_SCHEMA_VERSION",
    "H3_SPLIT_VERSION",
    "SplitRow",
    "WorldPrediction",
    "WorldScoreResult",
    "stable_sha256",
]
