"""Mixed-semantic K2 V3 candidate contract.

V1/V2 artifacts remain readable and unchanged.  V3 adds an immutable mission
binding and explicit alternative semantics so World can rank only candidates
that preserve the upstream route.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    canonical_sha256,
)

SCHEMA_V3 = "safedrive.k2.mixed_semantic.v3"
BRANCH_TYPE_V3 = "learned_route_bound_mixed_semantic"
DEFAULT_K2_V3_TOML = (
    Path(__file__).resolve().parents[2] / "config" / "vla" / "k2_v3_semantic.toml"
)


class AlternativeKind(str, Enum):
    NOMINAL_PROGRESS = "NOMINAL_PROGRESS"
    SPATIAL_AVOID = "SPATIAL_AVOID"
    SPATIAL_OVERTAKE = "SPATIAL_OVERTAKE"
    TEMPORAL_YIELD = "TEMPORAL_YIELD"
    NONE = "NONE"


class ManeuverPhase(str, Enum):
    PROPOSED = "PROPOSED"
    DEPART = "DEPART"
    PASS = "PASS"
    REJOIN = "REJOIN"
    HOLD_TARGET = "HOLD_TARGET"
    WAIT = "WAIT"
    RESUME = "RESUME"
    COMPLETE = "COMPLETE"


def _enum_value(value: Enum | str) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def stable_hash_xy(path_xy: Sequence[Sequence[float]]) -> str:
    return canonical_sha256(
        [[round(float(point[0]), 6), round(float(point[1]), 6)] for point in path_xy]
    )


def stable_hash_t10(points: Sequence[Sequence[float]]) -> str:
    return canonical_sha256(
        [[round(float(value), 6) for value in row[:6]] for row in points]
    )


@dataclass(frozen=True)
class K2V3Config:
    k: int = 2
    t_steps: int = 10
    dt_s: float = 0.25
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 3.0
    max_abs_curvature: float = 0.253
    max_lateral_accel_mps2: float = 1.5
    min_spatial_separation_m: float = 0.50
    max_temporal_shared_path_error_m: float = 0.25
    min_temporal_speed_gap_mps: float = 0.50
    min_temporal_progress_gap_m: float = 0.50
    route_exit_max_error_m: float = 3.0
    union_corridor_margin_m: float = 0.20
    vehicle_half_width_m: float = 1.0
    stop_threshold_mps: float = 0.35
    schema_version: str = SCHEMA_V3
    branch_type: str = BRANCH_TYPE_V3

    def config_hash(self) -> str:
        return canonical_sha256(asdict(self))


def load_k2_v3_config(path: Path | str | None = None) -> K2V3Config:
    config_path = Path(path) if path is not None else DEFAULT_K2_V3_TOML
    if not config_path.is_file():
        return K2V3Config()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    contract = raw.get("contract") or {}
    dynamics = raw.get("dynamics") or {}
    guard = raw.get("guard") or {}
    return K2V3Config(
        k=int(contract.get("k", 2)),
        t_steps=int(contract.get("t_steps", 10)),
        dt_s=float(contract.get("dt_s", 0.25)),
        max_accel_mps2=float(dynamics.get("max_accel_mps2", 2.5)),
        max_decel_mps2=float(dynamics.get("max_decel_mps2", 3.0)),
        max_abs_curvature=float(dynamics.get("max_abs_curvature", 0.253)),
        max_lateral_accel_mps2=float(
            dynamics.get("max_lateral_accel_mps2", 1.5)
        ),
        min_spatial_separation_m=float(
            guard.get("min_spatial_separation_m", 0.50)
        ),
        max_temporal_shared_path_error_m=float(
            guard.get("max_temporal_shared_path_error_m", 0.25)
        ),
        min_temporal_speed_gap_mps=float(
            guard.get("min_temporal_speed_gap_mps", 0.50)
        ),
        min_temporal_progress_gap_m=float(
            guard.get("min_temporal_progress_gap_m", 0.50)
        ),
        route_exit_max_error_m=float(guard.get("route_exit_max_error_m", 3.0)),
        union_corridor_margin_m=float(
            guard.get("union_corridor_margin_m", 0.20)
        ),
        vehicle_half_width_m=float(guard.get("vehicle_half_width_m", 1.0)),
        stop_threshold_mps=float(dynamics.get("stop_threshold_mps", 0.35)),
    )


@dataclass(frozen=True)
class K2CandidateV3:
    candidate_id: str
    alternative_kind: AlternativeKind
    route_maneuver: RouteManeuver
    available: bool
    availability_reason: str
    probability: float
    target_lane_side: TargetLaneSide
    maneuver_phase: ManeuverPhase
    spatial_path_xy: tuple[tuple[float, float], ...]
    points_xy_yaw_v_a_kappa: tuple[tuple[float, ...], ...]
    route_hash: str
    topology_hash: str
    corridor_hash: str
    intent_hash: str
    spatial_path_hash: str
    timed_trajectory_hash: str
    origin_lane_signature: str = ""
    target_lane_signature: str = ""
    head_lineage: str = "spatial_mode_head_v3"
    feature_content_hash: str = ""
    raw_head_output_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id required")
        if not math.isfinite(float(self.probability)) or self.probability < 0.0:
            raise ValueError("candidate probability must be finite and non-negative")
        path_hash = stable_hash_xy(self.spatial_path_xy)
        timed_hash = stable_hash_t10(self.points_xy_yaw_v_a_kappa)
        if self.spatial_path_hash and self.spatial_path_hash != path_hash:
            raise ValueError("spatial_path_hash mismatch")
        if self.timed_trajectory_hash and self.timed_trajectory_hash != timed_hash:
            raise ValueError("timed_trajectory_hash mismatch")
        object.__setattr__(self, "spatial_path_hash", path_hash)
        object.__setattr__(self, "timed_trajectory_hash", timed_hash)

    @property
    def speed_samples_mps(self) -> tuple[float, ...]:
        return tuple(float(row[3]) for row in self.points_xy_yaw_v_a_kappa)

    def binding_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "alternative_kind": self.alternative_kind.value,
            "route_maneuver": self.route_maneuver.value,
            "available": self.available,
            "availability_reason": self.availability_reason,
            "target_lane_side": self.target_lane_side.value,
            "maneuver_phase": self.maneuver_phase.value,
            "route_hash": self.route_hash,
            "topology_hash": self.topology_hash,
            "corridor_hash": self.corridor_hash,
            "spatial_path_hash": self.spatial_path_hash,
            "timed_trajectory_hash": self.timed_trajectory_hash,
            "origin_lane_signature": self.origin_lane_signature,
            "target_lane_signature": self.target_lane_signature,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.binding_payload(),
            "probability": float(self.probability),
            "intent_hash": self.intent_hash,
            "spatial_path_xy": [list(point) for point in self.spatial_path_xy],
            "points_xy_yaw_v_a_kappa": [
                list(row) for row in self.points_xy_yaw_v_a_kappa
            ],
            "head_lineage": self.head_lineage,
            "feature_content_hash": self.feature_content_hash,
            "raw_head_output_hash": self.raw_head_output_hash,
            "metadata": dict(self.metadata),
        }


def candidate_intent_hash(
    *,
    candidate_id: str,
    alternative_kind: AlternativeKind | str,
    route_maneuver: RouteManeuver | str,
    target_lane_side: TargetLaneSide | str,
    route_hash: str,
    topology_hash: str,
    corridor_hash: str,
    spatial_path_hash: str,
    timed_trajectory_hash: str,
) -> str:
    return canonical_sha256(
        {
            "candidate_id": candidate_id,
            "alternative_kind": _enum_value(alternative_kind),
            "route_maneuver": _enum_value(route_maneuver),
            "target_lane_side": _enum_value(target_lane_side),
            "route_hash": route_hash,
            "topology_hash": topology_hash,
            "corridor_hash": corridor_hash,
            "spatial_path_hash": spatial_path_hash,
            "timed_trajectory_hash": timed_trajectory_hash,
        }
    )


@dataclass(frozen=True)
class K2PredictionBundleV3:
    route_context: RouteContextV3
    candidates: tuple[K2CandidateV3, K2CandidateV3]
    observation_identity: Mapping[str, Any]
    model_id: str
    config_hash: str
    base_checkpoint_hash: str
    spatial_head_checkpoint_hash: str
    backbone_forward_id: str
    top1_index: int = 0
    probability_source: str = "fixed_prior_uncalibrated"
    schema_version: str = SCHEMA_V3
    branch_type: str = BRANCH_TYPE_V3
    guard_status: str = "UNVALIDATED"
    guard_reasons: tuple[str, ...] = ()
    guard_metrics: Mapping[str, Any] = field(default_factory=dict)
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_V3:
            raise ValueError(f"unsupported K2 V3 schema: {self.schema_version}")
        if len(self.candidates) != 2:
            raise ValueError("K2 V3 requires exactly two fixed slots")
        if self.top1_index not in (0, 1):
            raise ValueError("top1_index must be 0 or 1")
        if len({candidate.candidate_id for candidate in self.candidates}) != 2:
            raise ValueError("candidate ids must be unique")
        computed = self.compute_bundle_hash()
        if self.bundle_hash and self.bundle_hash != computed:
            raise ValueError("bundle_hash mismatch")
        object.__setattr__(self, "bundle_hash", computed)

    def compute_bundle_hash(self) -> str:
        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "branch_type": self.branch_type,
                "route_context": self.route_context.to_dict(),
                "candidate_bindings": [
                    candidate.binding_payload() | {"intent_hash": candidate.intent_hash}
                    for candidate in self.candidates
                ],
                "observation_identity": dict(self.observation_identity),
                "model_id": self.model_id,
                "config_hash": self.config_hash,
                "base_checkpoint_hash": self.base_checkpoint_hash,
                "spatial_head_checkpoint_hash": self.spatial_head_checkpoint_hash,
                "backbone_forward_id": self.backbone_forward_id,
                "top1_index": self.top1_index,
                "probability_source": self.probability_source,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "branch_type": self.branch_type,
            "route_context": self.route_context.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "observation_identity": dict(self.observation_identity),
            "model_id": self.model_id,
            "config_hash": self.config_hash,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "spatial_head_checkpoint_hash": self.spatial_head_checkpoint_hash,
            "backbone_forward_id": self.backbone_forward_id,
            "top1_index": self.top1_index,
            "probability_source": self.probability_source,
            "guard_status": self.guard_status,
            "guard_reasons": list(self.guard_reasons),
            "guard_metrics": dict(self.guard_metrics),
            "bundle_hash": self.bundle_hash,
        }


__all__ = [
    "AlternativeKind",
    "BRANCH_TYPE_V3",
    "K2CandidateV3",
    "K2PredictionBundleV3",
    "K2V3Config",
    "ManeuverPhase",
    "SCHEMA_V3",
    "candidate_intent_hash",
    "load_k2_v3_config",
    "stable_hash_t10",
    "stable_hash_xy",
]
