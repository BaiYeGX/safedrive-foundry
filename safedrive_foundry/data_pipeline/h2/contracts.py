"""Frozen H2 data contracts and the physically isolated H3 feature view."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


H2_SCHEMA_VERSION = "safedrive.h2.paired_outcomes.v1"
H3_FEATURE_SCHEMA_VERSION = "safedrive.h3.observable_candidate.v1"


class H2ContractError(ValueError):
    """Raised when a paired-outcome payload violates a frozen H2 boundary."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise H2ContractError(f"non_finite:{name}")


class PairTerminalStatus(str, Enum):
    INELIGIBLE = "INELIGIBLE"
    INVALID_PAIR = "INVALID_PAIR"
    VALID_PAIR = "VALID_PAIR"
    CAPTURE_FAILED = "CAPTURE_FAILED"


class OracleVerdict(str, Enum):
    CANDIDATE_WIN = "CANDIDATE_WIN"
    TIE = "TIE"
    UNRESOLVED = "UNRESOLVED"
    INVALID_PAIR = "INVALID_PAIR"


@dataclass(frozen=True)
class ScenarioKey:
    map_name: str
    family: str
    seed: int
    weather: str

    @property
    def pair_id(self) -> str:
        return f"{self.map_name}__{self.family}__s{self.seed}__{self.weather}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActorInitialState:
    role: str
    x: float
    y: float
    yaw_deg: float
    speed_mps: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw_deg", "speed_mps"):
            _require_finite(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResetSignature:
    actors: tuple[ActorInitialState, ...]
    route_sha256: str
    weather_sha256: str
    light_sha256: str
    script_sha256: str

    def __post_init__(self) -> None:
        roles = [actor.role for actor in self.actors]
        if not roles or len(roles) != len(set(roles)):
            raise H2ContractError("reset_actor_roles_missing_or_duplicate")
        for name in ("route_sha256", "weather_sha256", "light_sha256", "script_sha256"):
            value = getattr(self, name)
            if len(value) != 64:
                raise H2ContractError(f"invalid_hash:{name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actors": [actor.to_dict() for actor in self.actors],
            "route_sha256": self.route_sha256,
            "weather_sha256": self.weather_sha256,
            "light_sha256": self.light_sha256,
            "script_sha256": self.script_sha256,
        }


@dataclass(frozen=True)
class ResetComparison:
    comparable: bool
    reasons: tuple[str, ...]
    max_position_delta_m: float
    max_yaw_delta_deg: float
    max_speed_delta_mps: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_reset_signatures(
    captured: ResetSignature,
    branch: ResetSignature,
    *,
    position_limit_m: float = 0.05,
    yaw_limit_deg: float = 0.5,
    speed_limit_mps: float = 0.10,
) -> ResetComparison:
    reasons: list[str] = []
    for field_name in ("route_sha256", "weather_sha256", "light_sha256", "script_sha256"):
        if getattr(captured, field_name) != getattr(branch, field_name):
            reasons.append(f"{field_name}_mismatch")
    left = {actor.role: actor for actor in captured.actors}
    right = {actor.role: actor for actor in branch.actors}
    if set(left) != set(right):
        reasons.append("actor_roles_mismatch")
    position_deltas: list[float] = []
    yaw_deltas: list[float] = []
    speed_deltas: list[float] = []
    for role in sorted(set(left) & set(right)):
        a, b = left[role], right[role]
        position_deltas.append(math.hypot(a.x - b.x, a.y - b.y))
        yaw_deltas.append(abs(((a.yaw_deg - b.yaw_deg + 180.0) % 360.0) - 180.0))
        speed_deltas.append(abs(a.speed_mps - b.speed_mps))
    max_position = max(position_deltas, default=math.inf)
    max_yaw = max(yaw_deltas, default=math.inf)
    max_speed = max(speed_deltas, default=math.inf)
    if max_position > position_limit_m + 1e-12:
        reasons.append("position_delta_exceeded")
    if max_yaw > yaw_limit_deg + 1e-12:
        reasons.append("yaw_delta_exceeded")
    if max_speed > speed_limit_mps + 1e-12:
        reasons.append("speed_delta_exceeded")
    return ResetComparison(not reasons, tuple(reasons), max_position, max_yaw, max_speed)


def reset_comparison_from_dict(payload: Mapping[str, Any]) -> ResetComparison:
    return ResetComparison(
        comparable=bool(payload["comparable"]),
        reasons=tuple(str(item) for item in payload.get("reasons", ())),
        max_position_delta_m=float(payload["max_position_delta_m"]),
        max_yaw_delta_deg=float(payload["max_yaw_delta_deg"]),
        max_speed_delta_mps=float(payload["max_speed_delta_mps"]),
    )


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_id: str
    canonical_sha256: str
    source: str
    slot: int
    trajectory: tuple[Mapping[str, Any], ...]
    guard: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.candidate_id or len(self.canonical_sha256) != 64:
            raise H2ContractError("candidate_identity_invalid")
        if self.slot not in (0, 1):
            raise H2ContractError("candidate_slot_invalid")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class BranchOutcome:
    candidate_id: str
    candidate_sha256: str
    reset: ResetComparison
    safety_executed: bool
    safety_input_id: str | None
    safety_final_id: str | None
    safety_executed_id: str | None
    applied_id: str | None
    pre_binding_trajectory_sha256: str
    post_binding_trajectory_sha256: str
    ticks_executed: int
    cleanup_complete: bool
    collision_count: int
    red_light_violation: bool
    off_corridor_duration_s: float
    route_completed: bool
    route_progress_m: float
    jerk_rms_mps3: float
    acceleration_rms_mps2: float = 0.0
    lateral_acceleration_rms_mps2: float = 0.0
    deadline_misses: int = 0
    timeline_path: str = ""
    actor_future_path: str = ""
    event_path: str = ""
    branch_latency_s: float = 0.0
    whole_gpu_peak_gb: float = 0.0
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id or len(self.candidate_sha256) != 64:
            raise H2ContractError("branch_candidate_identity_invalid")
        for name in (
            "off_corridor_duration_s", "route_progress_m", "jerk_rms_mps3",
            "acceleration_rms_mps2", "lateral_acceleration_rms_mps2", "branch_latency_s",
            "whole_gpu_peak_gb",
        ):
            _require_finite(name, getattr(self, name))
        if self.ticks_executed < 0 or self.collision_count < 0 or self.deadline_misses < 0:
            raise H2ContractError("negative_branch_counter")

    @property
    def execution_binding_valid(self) -> bool:
        return bool(
            self.safety_executed
            and self.safety_input_id == self.candidate_id
            and self.safety_executed_id
            and self.applied_id == self.safety_executed_id
            and self.pre_binding_trajectory_sha256 == self.post_binding_trajectory_sha256
        )

    @property
    def complete(self) -> bool:
        return self.reset.comparable and self.execution_binding_valid and self.ticks_executed == 50 and self.cleanup_complete

    @property
    def hard_unsafe(self) -> bool:
        return self.collision_count > 0 or self.red_light_violation or self.off_corridor_duration_s > 0.25

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def branch_outcome_from_dict(payload: Mapping[str, Any]) -> BranchOutcome:
    values = dict(payload)
    values["reset"] = reset_comparison_from_dict(values["reset"])
    values["errors"] = tuple(values.get("errors", ()))
    return BranchOutcome(**values)


@dataclass(frozen=True)
class OracleLabel:
    verdict: OracleVerdict
    winner_candidate_id: str | None
    winner_candidate_sha256: str | None
    reason: str
    oracle_version: str = "h2-offline-oracle-v1"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PairRecord:
    dataset_id: str
    scenario: ScenarioKey
    matrix_sha256: str
    anchor: Mapping[str, Any]
    observable_history: tuple[Mapping[str, Any], ...]
    route: tuple[tuple[float, float], ...]
    candidates: tuple[CandidateSnapshot, ...]
    terminal_status: PairTerminalStatus
    branch_order: tuple[str, ...] = ()
    branches: tuple[BranchOutcome, ...] = ()
    vla_forward_count: int = 0
    capture_reset: ResetSignature | None = None
    label: OracleLabel | None = None
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    # Filled from the canonical TOML when omitted for source compatibility
    # with early H2 fixtures.  Live records always serialize the explicit hash.
    config_sha256: str = ""
    schema_version: str = H2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id or len(self.matrix_sha256) != 64:
            raise H2ContractError("pair_dataset_or_matrix_invalid")
        if not self.config_sha256:
            from .config import H2_CONFIG_SHA256

            object.__setattr__(self, "config_sha256", H2_CONFIG_SHA256)
        if len(self.config_sha256) != 64:
            raise H2ContractError("pair_config_hash_invalid")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise H2ContractError("duplicate_candidate_id")
        if self.vla_forward_count < 0:
            raise H2ContractError("negative_vla_forward_count")
        if self.branches and set(branch.candidate_id for branch in self.branches) - set(candidate_ids):
            raise H2ContractError("branch_candidate_not_captured")
        if self.branch_order and set(self.branch_order) != set(candidate_ids):
            raise H2ContractError("branch_order_candidate_mismatch")

    @property
    def pair_id(self) -> str:
        return self.scenario.pair_id

    @property
    def content_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("content_sha256", None)
        return stable_sha256(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "pair_id": self.pair_id,
            "scenario": self.scenario.to_dict(),
            "matrix_sha256": self.matrix_sha256,
            "config_sha256": self.config_sha256,
            "anchor": _jsonable(self.anchor),
            "observable_history": _jsonable(self.observable_history),
            "route": _jsonable(self.route),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "terminal_status": self.terminal_status.value,
            "branch_order": list(self.branch_order),
            "branches": [branch.to_dict() for branch in self.branches],
            "vla_forward_count": self.vla_forward_count,
            "capture_reset": None if self.capture_reset is None else self.capture_reset.to_dict(),
            "label": None if self.label is None else self.label.to_dict(),
            "artifact_hashes": dict(self.artifact_hashes),
            "errors": list(self.errors),
        }
        payload["content_sha256"] = stable_sha256(payload)
        return payload


_H3_TOP_LEVEL = frozenset({"schema_version", "sample_id", "anchor", "observable_history", "route", "candidate"})
_FORBIDDEN_H3_TOKENS = frozenset(
    {"source", "slot", "branch_order", "actor_future", "outcome", "oracle", "regression", "label", "winner"}
)


def h3_feature_view(pair: PairRecord, candidate_id: str) -> dict[str, Any]:
    """Return the only H2-derived payload permitted to cross into H3.

    Source, slot, execution ordering, actor future and all outcome/Oracle fields are
    reconstructed nowhere: they are absent rather than redacted after serialization.
    """

    candidate = next((item for item in pair.candidates if item.candidate_id == candidate_id), None)
    if candidate is None:
        raise H2ContractError("unknown_h3_candidate")
    anchor = pair.anchor
    observable = anchor.get("observable_snapshot", {})
    safe_anchor = {
        key: anchor[key]
        for key in (
            "observation_id", "carla_frame", "simulation_time_s", "wall_time_s",
            "coordinate_frame", "ego", "sensor_frames", "sensor_timestamps_s",
        )
        if key in anchor
    }
    if isinstance(observable, Mapping):
        safe_anchor["observable_snapshot"] = {
            key: _jsonable(observable[key])
            for key in (
                "simulation_time_s", "ego_x", "ego_y", "ego_yaw", "ego_v", "ego_a",
                "observed_time_s", "freshness_s", "speed_limit_mps", "actors",
                "traffic_lights", "corridor_centerline", "corridor_half_width_m",
                "coordinate_frame",
            )
            if key in observable
        }
        actors = safe_anchor["observable_snapshot"].get("actors", [])
        safe_anchor["observable_snapshot"]["actors"] = [
            {key: value for key, value in dict(actor).items() if key != "source"}
            for actor in actors
        ]
    view = {
        "schema_version": H3_FEATURE_SCHEMA_VERSION,
        "sample_id": stable_sha256({"pair_id": pair.pair_id, "candidate_sha256": candidate.canonical_sha256}),
        "anchor": safe_anchor,
        "observable_history": [dict(item) for item in pair.observable_history],
        "route": [[float(x), float(y)] for x, y in pair.route],
        "candidate": {
            "canonical_sha256": candidate.canonical_sha256,
            "trajectory": [dict(point) for point in candidate.trajectory],
        },
    }
    if set(view) != _H3_TOP_LEVEL:
        raise AssertionError("H3 feature allowlist drift")
    serialized_keys = {str(key).lower() for key in _walk_keys(view)}
    leaked = sorted(serialized_keys & _FORBIDDEN_H3_TOKENS)
    if leaked:
        raise H2ContractError(f"h3_forbidden_fields:{','.join(leaked)}")
    return view


def _walk_keys(value: Any) -> Sequence[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


__all__ = [
    "ActorInitialState", "BranchOutcome", "CandidateSnapshot", "H2ContractError",
    "H2_SCHEMA_VERSION", "OracleLabel", "OracleVerdict", "PairRecord",
    "PairTerminalStatus", "ResetComparison", "ResetSignature", "ScenarioKey",
    "branch_outcome_from_dict", "compare_reset_signatures", "h3_feature_view",
    "reset_comparison_from_dict", "stable_json_bytes", "stable_sha256",
]
