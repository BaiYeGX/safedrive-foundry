"""Public H1 contracts for independent Expert/VLA candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from driving_vla.adapter.policy_adapter import ObservationBundle
from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.contracts.serialize import candidate_to_dict
from safety_kernel.contracts.types import (
    ConstraintMargin,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
)


class HybridContractError(ValueError):
    """Raised when an H1 object is not bound to one observable anchor."""


class HybridSource(str, Enum):
    EXPERT = "expert"
    VLA = "vla"


class GuardVerdict(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"


class SelectionSpace(str, Enum):
    ZERO_PASS = "ZERO_PASS"
    SINGLE_PASS = "SINGLE_PASS"
    DISTINCT = "DISTINCT"
    NO_SELECTION_SPACE = "NO_SELECTION_SPACE"


class WorldDisposition(str, Enum):
    DEFERRED_NOT_IMPLEMENTED = "DEFERRED_NOT_IMPLEMENTED"


@dataclass(frozen=True)
class ObservableAnchor:
    """The immutable identity shared by both H1 generators."""

    observation_id: str
    bundle: ObservationBundle
    safety_snapshot: ObservableSnapshot
    route_revision: str
    sensor_frames: Mapping[str, int]
    sensor_timestamps_s: Mapping[str, float]

    def __post_init__(self) -> None:
        bundle, safety = self.bundle, self.safety_snapshot
        if not self.observation_id:
            raise HybridContractError("missing_observation_id")
        if bundle.frame_id != self.observation_id or safety.frame_id != self.observation_id:
            raise HybridContractError("observation_id_frame_mismatch")
        if bundle.run_id != safety.run_id or bundle.scenario_id != safety.scenario_id:
            raise HybridContractError("anchor_identity_mismatch")
        if abs(bundle.simulation_time_s - safety.simulation_time_s) > 1e-9:
            raise HybridContractError("anchor_simulation_time_mismatch")
        if abs(bundle.wall_time_s - safety.wall_time_s) > 1e-6:
            raise HybridContractError("anchor_wall_time_mismatch")
        ego_values = (
            (bundle.ego_x, safety.ego_x),
            (bundle.ego_y, safety.ego_y),
            (bundle.ego_yaw, safety.ego_yaw),
            (bundle.ego_v, safety.ego_v),
        )
        if any(abs(float(left) - float(right)) > 1e-9 for left, right in ego_values):
            raise HybridContractError("anchor_ego_mismatch")
        if safety.coordinate_frame != "map":
            raise HybridContractError("anchor_coordinate_frame_not_map")
        if safety.privilege.value != "observable" or safety.oracle_fields:
            raise HybridContractError("anchor_contains_oracle")
        if abs(safety.observed_time_s - bundle.simulation_time_s) > 1e-6:
            raise HybridContractError("anchor_observed_time_mismatch")
        if len(bundle.route_xy) < 2 or not self.route_revision:
            raise HybridContractError("anchor_route_missing")
        if not self.sensor_frames or set(self.sensor_frames) != set(self.sensor_timestamps_s):
            raise HybridContractError("anchor_sensor_manifest_mismatch")
        for name, frame in self.sensor_frames.items():
            if int(frame) != int(bundle.carla_frame):
                raise HybridContractError(f"anchor_sensor_frame_mismatch:{name}")
            stamp = float(self.sensor_timestamps_s[name])
            if abs(stamp - bundle.simulation_time_s) > 0.05 + 1e-9:
                raise HybridContractError(f"anchor_sensor_time_mismatch:{name}")

    @property
    def simulation_time_s(self) -> float:
        return float(self.bundle.simulation_time_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "run_id": self.bundle.run_id,
            "frame_id": self.bundle.frame_id,
            "scenario_id": self.bundle.scenario_id,
            "carla_frame": self.bundle.carla_frame,
            "simulation_time_s": self.bundle.simulation_time_s,
            "wall_time_s": self.bundle.wall_time_s,
            "coordinate_frame": "map",
            "ego": {
                "x": self.bundle.ego_x,
                "y": self.bundle.ego_y,
                "yaw": self.bundle.ego_yaw,
                "v": self.bundle.ego_v,
                "a": self.safety_snapshot.ego_a,
            },
            "route_revision": self.route_revision,
            "route_points": len(self.bundle.route_xy),
            "actor_count": len(self.safety_snapshot.actors),
            "traffic_light_count": len(self.safety_snapshot.traffic_lights),
            "sensor_frames": dict(self.sensor_frames),
            "sensor_timestamps_s": dict(self.sensor_timestamps_s),
        }


@dataclass(frozen=True)
class CandidateProvenance:
    source: HybridSource
    candidate_id: str
    observation_id: str
    frame_id: str
    carla_frame: int
    simulation_time_s: float
    route_revision: str
    generator_id: str
    generator_hash: str
    raw_sha256: str
    canonical_sha256: str
    canonicalizer_version: str
    canonicalization_error_m: float
    coverage_shortfall_m: float
    generation_latency_s: float
    generated_wall_time_s: float
    freshness_s: float
    coordinate_frame: str = "map"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        return data


@dataclass(frozen=True)
class GuardCheck:
    stage: str
    name: str
    passed: bool
    message: str
    margin: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardResult:
    candidate_id: str
    verdict: GuardVerdict
    checks: tuple[GuardCheck, ...]
    reject_reasons: tuple[str, ...]
    latency_ms: float
    margins: tuple[ConstraintMargin, ...] = ()
    controller_mode: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict is GuardVerdict.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "verdict": self.verdict.value,
            "checks": [check.to_dict() for check in self.checks],
            "reject_reasons": list(self.reject_reasons),
            "latency_ms": self.latency_ms,
            "controller_mode": self.controller_mode,
            "margins": [
                {
                    "name": margin.name,
                    "margin": margin.margin,
                    "hard": margin.hard,
                    "message": margin.message,
                    "first_violation_time_s": margin.first_violation_time_s,
                    "actor_id": margin.actor_id,
                    "rule_id": margin.rule_id,
                }
                for margin in self.margins
            ],
        }


@dataclass(frozen=True)
class HybridCandidate:
    candidate: PolicyCandidate
    provenance: CandidateProvenance
    guard: GuardResult | None = None

    def with_guard(self, guard: GuardResult) -> "HybridCandidate":
        if guard.candidate_id != self.candidate.candidate_id:
            raise HybridContractError("guard_candidate_id_mismatch")
        return replace(self, guard=guard)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": candidate_to_dict(self.candidate),
            "provenance": self.provenance.to_dict(),
            "guard": None if self.guard is None else self.guard.to_dict(),
        }


@dataclass(frozen=True)
class GenerationAttempt:
    source: HybridSource
    success: bool
    generation_latency_s: float
    candidate_id: str | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "success": self.success,
            "generation_latency_s": self.generation_latency_s,
            "candidate_id": self.candidate_id,
            "error": self.error,
        }


@dataclass(frozen=True)
class CandidateDifference:
    max_position_delta_m: float
    rms_speed_delta_mps: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HybridCandidateSet:
    anchor: ObservableAnchor
    candidates: tuple[HybridCandidate, ...]
    attempts: tuple[GenerationAttempt, ...] = ()
    schema_version: str = "safedrive.h1.hybrid_candidates.v1"

    def __post_init__(self) -> None:
        ids = [item.candidate.candidate_id for item in self.candidates]
        sources = [item.provenance.source for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise HybridContractError("duplicate_candidate_id")
        if len(sources) != len(set(sources)):
            raise HybridContractError("duplicate_candidate_source")
        if len(self.candidates) > 2:
            raise HybridContractError("too_many_h1_candidates")
        for item in self.candidates:
            p = item.provenance
            if p.candidate_id != item.candidate.candidate_id:
                raise HybridContractError("candidate_provenance_id_mismatch")
            if p.observation_id != self.anchor.observation_id:
                raise HybridContractError("candidate_observation_mismatch")

    def to_policy_candidate_set(
        self, candidates: tuple[PolicyCandidate, ...] | None = None
    ) -> PolicyCandidateSet:
        selected = tuple(item.candidate for item in self.candidates) if candidates is None else candidates
        bundle = self.anchor.bundle
        return PolicyCandidateSet(
            run_id=bundle.run_id,
            frame_id=bundle.frame_id,
            scenario_id=bundle.scenario_id,
            model_id="h1-hybrid@1.0.0",
            carla_frame=bundle.carla_frame,
            simulation_time_s=bundle.simulation_time_s,
            wall_time_s=bundle.wall_time_s,
            candidates=selected,
            schema_version=SCHEMA_VERSION,
            coordinate_frame="map",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "anchor": self.anchor.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class RoutingResult:
    pass_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    selection_space: SelectionSpace
    world: WorldDisposition
    selector: str
    reason: str
    difference: CandidateDifference | None
    scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_candidate_ids": list(self.pass_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "selection_space": self.selection_space.value,
            "world": self.world.value,
            "selector": self.selector,
            "reason": self.reason,
            "difference": None if self.difference is None else self.difference.to_dict(),
            "scores": dict(self.scores),
        }


__all__ = [
    "CandidateDifference",
    "CandidateProvenance",
    "GenerationAttempt",
    "GuardCheck",
    "GuardResult",
    "GuardVerdict",
    "HybridCandidate",
    "HybridCandidateSet",
    "HybridContractError",
    "HybridSource",
    "ObservableAnchor",
    "RoutingResult",
    "SelectionSpace",
    "WorldDisposition",
]
