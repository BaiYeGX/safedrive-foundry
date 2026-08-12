"""R2 paired-outcome contracts: identities, anchor artifact, canonical hashes.

Offline-only. Branch runners load serialized artifacts; they must not rebuild
K2 by re-forwarding VLA.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "safedrive.g4a.paired_contract.v1"
K2_ANCHOR_SCHEMA = "safedrive.g4a.k2_anchor_artifact.v1"
MEASURED_STATE_SCHEMA = "safedrive.g4a.measured_initial_state.v1"

# Quantization for cross-run measured-state hashing (R2 §6.1).
Q_POSITION_M = 0.01
Q_ROTATION_DEG = 0.1
Q_VELOCITY = 0.01
Q_SIM_TIME_S = 0.001

# Primary outcome horizon (matches K2 horizon).
PRIMARY_HORIZON_S = 2.50
CONTROL_DT_S = 0.05
EXPECTED_PRIMARY_TICKS = 50

ORACLE_TRACE_FLAGS = {
    "oracle_only": True,
    "consumed_by_control": False,
}

# Control-namespace names that must never receive oracle fields.
CONTROL_NAMESPACE_DENYLIST = frozenset(
    {
        "oracle_trace",
        "oracle_only",
        "privileged_future",
        "true_actor_future",
        "first_conflict_time",
        "minimum_actor_clearance_oracle",
        "minimum_ttc_oracle",
    }
)


class ContractError(ValueError):
    """Invalid paired-contract payload or identity input."""


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _require_finite(value: Any, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(x):
        raise ContractError(f"{name} must be finite")
    return x


def quantize(value: float, step: float) -> float:
    """Round to nearest multiple of ``step`` for stable hashing only."""
    if step <= 0:
        raise ContractError("quantize step must be > 0")
    return round(float(value) / step) * step


def quantize_deg(yaw_deg: float, step: float = Q_ROTATION_DEG) -> float:
    """Quantize degrees into (-180, 180] then to step."""
    y = float(yaw_deg) % 360.0
    if y > 180.0:
        y -= 360.0
    return quantize(y, step)


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding for content hashes."""
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ContractError("non-finite float in canonical JSON")
        # Normalize -0.0
        if obj == 0.0:
            return 0.0
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonicalize(asdict(obj))
    if hasattr(obj, "tolist"):
        return _canonicalize(obj.tolist())
    raise ContractError(f"unsupported type for canonical JSON: {type(obj)!r}")


def content_hash(obj: Any, *, nibble: int | None = 64) -> str:
    digest = hashlib.sha256(canonical_json_bytes(obj)).hexdigest()
    if nibble is None:
        return digest
    return digest[: int(nibble)]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_pair_id(
    *,
    scenario_registry_hash: str,
    scenario_id: str,
    seed_id: str,
    model_checkpoint_config_retimer_hash: str,
    executor_config_hash: str,
) -> str:
    """Stable pair identity (R2 §3.1), first 20 hex chars of sha256."""
    payload = {
        "scenario_registry_hash": _require_str(scenario_registry_hash, "scenario_registry_hash"),
        "scenario_id": _require_str(scenario_id, "scenario_id"),
        "seed_id": _require_str(seed_id, "seed_id"),
        "model_checkpoint_config_retimer_hash": _require_str(
            model_checkpoint_config_retimer_hash, "model_checkpoint_config_retimer_hash"
        ),
        "executor_config_hash": _require_str(executor_config_hash, "executor_config_hash"),
    }
    return content_hash(payload, nibble=20)


def compute_run_id(
    *,
    pair_id: str,
    role: str,
    attempt_id: int = 0,
) -> str:
    """Derive anchor/branch run ids; attempt_id isolates retries without overwrite."""
    role_n = _require_str(role, "role")
    if role_n not in {"anchor", "branch_0", "branch_1"}:
        raise ContractError(f"unsupported run role: {role_n}")
    if isinstance(attempt_id, bool) or int(attempt_id) < 0:
        raise ContractError("attempt_id must be a non-negative integer")
    payload = {
        "pair_id": _require_str(pair_id, "pair_id"),
        "role": role_n,
        "attempt_id": int(attempt_id),
    }
    return f"{role_n}-" + content_hash(payload, nibble=16)


@dataclass(frozen=True)
class TransformPose:
    x: float
    y: float
    z: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def quantized(self) -> dict[str, float]:
        return {
            "x": quantize(self.x, Q_POSITION_M),
            "y": quantize(self.y, Q_POSITION_M),
            "z": quantize(self.z, Q_POSITION_M),
            "roll_deg": quantize_deg(self.roll_deg),
            "pitch_deg": quantize_deg(self.pitch_deg),
            "yaw_deg": quantize_deg(self.yaw_deg),
        }

    def raw_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "roll_deg": float(self.roll_deg),
            "pitch_deg": float(self.pitch_deg),
            "yaw_deg": float(self.yaw_deg),
        }


@dataclass(frozen=True)
class VelocityState:
    vx: float
    vy: float
    vz: float
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def quantized(self) -> dict[str, float]:
        return {
            "vx": quantize(self.vx, Q_VELOCITY),
            "vy": quantize(self.vy, Q_VELOCITY),
            "vz": quantize(self.vz, Q_VELOCITY),
            "wx": quantize(self.wx, Q_VELOCITY),
            "wy": quantize(self.wy, Q_VELOCITY),
            "wz": quantize(self.wz, Q_VELOCITY),
        }

    def raw_dict(self) -> dict[str, float]:
        return {
            "vx": float(self.vx),
            "vy": float(self.vy),
            "vz": float(self.vz),
            "wx": float(self.wx),
            "wy": float(self.wy),
            "wz": float(self.wz),
        }

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class ActorSnapshot:
    name: str
    role: str
    blueprint: str
    transform: TransformPose
    velocity: VelocityState
    control: Mapping[str, float] = field(default_factory=dict)
    script_phase: str = "init"
    bounding_box_extent_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def quantized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "blueprint": self.blueprint,
            "transform": self.transform.quantized(),
            "velocity": self.velocity.quantized(),
            "control": {str(k): quantize(float(v), 1e-3) for k, v in sorted(self.control.items())},
            "script_phase": self.script_phase,
            "bounding_box_extent_m": [
                quantize(float(x), Q_POSITION_M) for x in self.bounding_box_extent_m
            ],
        }

    def raw_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "blueprint": self.blueprint,
            "transform": self.transform.raw_dict(),
            "velocity": self.velocity.raw_dict(),
            "control": {str(k): float(v) for k, v in self.control.items()},
            "script_phase": self.script_phase,
            "bounding_box_extent_m": [float(x) for x in self.bounding_box_extent_m],
        }


@dataclass(frozen=True)
class MeasuredInitialState:
    """Actual world state at decision anchor (raw + hash of quantized form)."""

    schema_version: str
    map_name: str
    open_drive_identity: str
    world_settings: Mapping[str, Any]
    weather: Mapping[str, Any]
    actors: tuple[ActorSnapshot, ...]
    traffic_light_state: Mapping[str, Any]
    route_anchor: Mapping[str, Any]
    sensor_calibration: Mapping[str, Any]
    carla_server_epoch: str
    carla_version: str
    simulation_frame: int
    simulation_time_s: float
    actor_script_phase: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != MEASURED_STATE_SCHEMA:
            raise ContractError(
                f"measured state schema must be {MEASURED_STATE_SCHEMA}, got {self.schema_version}"
            )

    def quantized_payload(self) -> dict[str, Any]:
        actors_q = [a.quantized() for a in sorted(self.actors, key=lambda x: x.name)]
        return {
            "schema_version": self.schema_version,
            "map_name": self.map_name,
            "open_drive_identity": self.open_drive_identity,
            "world_settings": dict(self.world_settings),
            "weather": {
                k: quantize(float(v), 0.01) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in sorted(self.weather.items())
            },
            "actors": actors_q,
            "traffic_light_state": dict(self.traffic_light_state),
            "route_anchor": dict(self.route_anchor),
            "sensor_calibration": dict(self.sensor_calibration),
            "carla_server_epoch": self.carla_server_epoch,
            "carla_version": self.carla_version,
            "simulation_frame": int(self.simulation_frame),
            "simulation_time_s": quantize(self.simulation_time_s, Q_SIM_TIME_S),
            "actor_script_phase": dict(sorted(self.actor_script_phase.items())),
        }

    def measured_hash(self) -> str:
        return content_hash(self.quantized_payload(), nibble=64)

    def raw_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "map_name": self.map_name,
            "open_drive_identity": self.open_drive_identity,
            "world_settings": dict(self.world_settings),
            "weather": dict(self.weather),
            "actors": [a.raw_dict() for a in self.actors],
            "traffic_light_state": dict(self.traffic_light_state),
            "route_anchor": dict(self.route_anchor),
            "sensor_calibration": dict(self.sensor_calibration),
            "carla_server_epoch": self.carla_server_epoch,
            "carla_version": self.carla_version,
            "simulation_frame": int(self.simulation_frame),
            "simulation_time_s": float(self.simulation_time_s),
            "actor_script_phase": dict(self.actor_script_phase),
            "measured_initial_state_hash": self.measured_hash(),
        }

    def ego(self) -> ActorSnapshot:
        for a in self.actors:
            if a.role == "ego":
                return a
        raise ContractError("measured state missing ego actor")


@dataclass(frozen=True)
class ObservationFingerprint:
    front_rgb_sha256: str
    image_height: int
    image_width: int
    image_channels: int
    image_layout: str
    ego_observable: Mapping[str, Any]
    route_targets: Sequence[Any]
    camera_frame: Mapping[str, Any]
    k2_bundle_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_rgb_sha256": self.front_rgb_sha256,
            "image_height": int(self.image_height),
            "image_width": int(self.image_width),
            "image_channels": int(self.image_channels),
            "image_layout": self.image_layout,
            "ego_observable": dict(self.ego_observable),
            "route_targets": list(self.route_targets),
            "camera_frame": dict(self.camera_frame),
            "k2_bundle_hash": self.k2_bundle_hash,
        }


@dataclass(frozen=True)
class SerializedCandidate:
    candidate_id: str
    candidate_index: int
    probability: float
    points_xy_yaw_v_a_kappa: tuple[tuple[float, float, float, float, float, float], ...]
    spatial_path_xy: tuple[tuple[float, float], ...]
    speed_samples_mps: tuple[float, ...]
    timed_trajectory_hash: str
    native_path_hash: str
    branch_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": int(self.candidate_index),
            "probability": float(self.probability),
            "points_xy_yaw_v_a_kappa": [list(p) for p in self.points_xy_yaw_v_a_kappa],
            "spatial_path_xy": [list(p) for p in self.spatial_path_xy],
            "speed_samples_mps": list(self.speed_samples_mps),
            "timed_trajectory_hash": self.timed_trajectory_hash,
            "native_path_hash": self.native_path_hash,
            "branch_type": self.branch_type,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SerializedCandidate":
        pts = tuple(tuple(float(x) for x in row) for row in d["points_xy_yaw_v_a_kappa"])
        path = tuple(tuple(float(x) for x in row) for row in d["spatial_path_xy"])
        speeds = tuple(float(x) for x in d["speed_samples_mps"])
        return SerializedCandidate(
            candidate_id=str(d["candidate_id"]),
            candidate_index=int(d["candidate_index"]),
            probability=float(d["probability"]),
            points_xy_yaw_v_a_kappa=pts,  # type: ignore[arg-type]
            spatial_path_xy=path,  # type: ignore[arg-type]
            speed_samples_mps=speeds,
            timed_trajectory_hash=str(d["timed_trajectory_hash"]),
            native_path_hash=str(d["native_path_hash"]),
            branch_type=str(d["branch_type"]),
        )


@dataclass(frozen=True)
class K2AnchorArtifactV1:
    """One frozen K2 from a single real forward; shared by branch-0 and branch-1."""

    schema_version: str
    pair_id: str
    scenario_id: str
    seed_id: str
    anchor_run_id: str
    anchor_carla_frame: int
    anchor_simulation_time_s: float
    requested_initial_state_hash: str
    measured_initial_state_hash: str
    observation_fingerprint: ObservationFingerprint
    model_id: str
    model_checkpoint_hash: str
    config_hash: str
    retimer_version: str
    retimer_hash: str
    executor_config_hash: str
    native_path_xy: tuple[tuple[float, float], ...]
    native_path_hash: str
    candidates: tuple[SerializedCandidate, ...]
    top1_index: int
    guard_status: str
    guard_reasons: tuple[str, ...]
    guard_metrics: Mapping[str, Any] = field(default_factory=dict)
    probability_source: str = "fixed_equal_prior_unscaled"
    branch_type: str = "longitudinal_temporal"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    k: int = 2
    t_steps: int = 10
    dt_s: float = 0.25
    horizon_s: float = 2.5

    def __post_init__(self) -> None:
        if self.schema_version != K2_ANCHOR_SCHEMA:
            raise ContractError(
                f"anchor schema must be {K2_ANCHOR_SCHEMA}, got {self.schema_version}"
            )
        if self.k != 2 or self.t_steps != 10:
            raise ContractError("R2 freezes K=2 T=10")
        if abs(self.dt_s - 0.25) > 1e-9 or abs(self.horizon_s - 2.5) > 1e-9:
            raise ContractError("R2 freezes dt=0.25 horizon=2.5")
        if len(self.candidates) != 2:
            raise ContractError(f"expected 2 candidates, got {len(self.candidates)}")
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != 2:
            raise ContractError(f"duplicate candidate ids: {ids}")
        if self.top1_index not in (0, 1):
            raise ContractError("top1_index must be 0 or 1")

    def payload_for_hash(self) -> dict[str, Any]:
        """Full artifact payload excluding the content hash field itself."""
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "seed_id": self.seed_id,
            "anchor_run_id": self.anchor_run_id,
            "anchor_carla_frame": int(self.anchor_carla_frame),
            "anchor_simulation_time_s": float(self.anchor_simulation_time_s),
            "requested_initial_state_hash": self.requested_initial_state_hash,
            "measured_initial_state_hash": self.measured_initial_state_hash,
            "observation_fingerprint": self.observation_fingerprint.to_dict(),
            "model_id": self.model_id,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "config_hash": self.config_hash,
            "retimer_version": self.retimer_version,
            "retimer_hash": self.retimer_hash,
            "executor_config_hash": self.executor_config_hash,
            "native_path_xy": [list(p) for p in self.native_path_xy],
            "native_path_hash": self.native_path_hash,
            "candidates": [c.to_dict() for c in self.candidates],
            "top1_index": int(self.top1_index),
            "guard_status": self.guard_status,
            "guard_reasons": list(self.guard_reasons),
            "guard_metrics": dict(self.guard_metrics),
            "probability_source": self.probability_source,
            "branch_type": self.branch_type,
            "diagnostics": dict(self.diagnostics),
            "k": int(self.k),
            "t_steps": int(self.t_steps),
            "dt_s": float(self.dt_s),
            "horizon_s": float(self.horizon_s),
        }

    def artifact_content_hash(self) -> str:
        return content_hash(self.payload_for_hash(), nibble=64)

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_for_hash()
        payload["artifact_content_hash"] = self.artifact_content_hash()
        return payload

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "K2AnchorArtifactV1":
        obs_raw = d["observation_fingerprint"]
        obs = ObservationFingerprint(
            front_rgb_sha256=str(obs_raw["front_rgb_sha256"]),
            image_height=int(obs_raw["image_height"]),
            image_width=int(obs_raw["image_width"]),
            image_channels=int(obs_raw["image_channels"]),
            image_layout=str(obs_raw["image_layout"]),
            ego_observable=dict(obs_raw["ego_observable"]),
            route_targets=list(obs_raw["route_targets"]),
            camera_frame=dict(obs_raw["camera_frame"]),
            k2_bundle_hash=str(obs_raw["k2_bundle_hash"]),
        )
        cands = tuple(SerializedCandidate.from_dict(c) for c in d["candidates"])
        path = tuple(tuple(float(x) for x in row) for row in d["native_path_xy"])
        art = K2AnchorArtifactV1(
            schema_version=str(d["schema_version"]),
            pair_id=str(d["pair_id"]),
            scenario_id=str(d["scenario_id"]),
            seed_id=str(d["seed_id"]),
            anchor_run_id=str(d["anchor_run_id"]),
            anchor_carla_frame=int(d["anchor_carla_frame"]),
            anchor_simulation_time_s=float(d["anchor_simulation_time_s"]),
            requested_initial_state_hash=str(d["requested_initial_state_hash"]),
            measured_initial_state_hash=str(d["measured_initial_state_hash"]),
            observation_fingerprint=obs,
            model_id=str(d["model_id"]),
            model_checkpoint_hash=str(d["model_checkpoint_hash"]),
            config_hash=str(d["config_hash"]),
            retimer_version=str(d["retimer_version"]),
            retimer_hash=str(d["retimer_hash"]),
            executor_config_hash=str(d["executor_config_hash"]),
            native_path_xy=path,  # type: ignore[arg-type]
            native_path_hash=str(d["native_path_hash"]),
            candidates=cands,
            top1_index=int(d["top1_index"]),
            guard_status=str(d["guard_status"]),
            guard_reasons=tuple(str(x) for x in d.get("guard_reasons", ())),
            guard_metrics=dict(d.get("guard_metrics", {})),
            probability_source=str(d.get("probability_source", "fixed_equal_prior_unscaled")),
            branch_type=str(d.get("branch_type", "longitudinal_temporal")),
            diagnostics=dict(d.get("diagnostics", {})),
            k=int(d.get("k", 2)),
            t_steps=int(d.get("t_steps", 10)),
            dt_s=float(d.get("dt_s", 0.25)),
            horizon_s=float(d.get("horizon_s", 2.5)),
        )
        expected = d.get("artifact_content_hash")
        if expected is not None and str(expected) != art.artifact_content_hash():
            raise ContractError(
                "artifact_content_hash mismatch on deserialize "
                f"(stored={expected}, recomputed={art.artifact_content_hash()})"
            )
        return art

    @staticmethod
    def from_json_bytes(data: bytes) -> "K2AnchorArtifactV1":
        return K2AnchorArtifactV1.from_dict(json.loads(data.decode("utf-8")))


# ---------------------------------------------------------------------------
# Spatial K2 V2 anchor artifact (R2-X). Independent of longitudinal V1 schema.
# ---------------------------------------------------------------------------

K2_ANCHOR_SCHEMA_V2 = "safedrive.g4a.k2_anchor_artifact.v2"


@dataclass(frozen=True)
class SerializedCandidateV2:
    candidate_id: str
    candidate_index: int
    mode_id: str
    available: bool
    availability_reason: str
    probability: float
    points_xy_yaw_v_a_kappa: tuple[tuple[float, ...], ...]
    frenet_s: tuple[float, ...]
    frenet_d: tuple[float, ...]
    spatial_path_xy: tuple[tuple[float, float], ...]
    speed_samples_mps: tuple[float, ...]
    proposal_path_hash: str
    timed_trajectory_hash: str
    native_anchor_hash: str
    head_lineage: str
    branch_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": int(self.candidate_index),
            "mode_id": self.mode_id,
            "available": bool(self.available),
            "availability_reason": self.availability_reason,
            "probability": float(self.probability),
            "points_xy_yaw_v_a_kappa": [list(p) for p in self.points_xy_yaw_v_a_kappa],
            "frenet_s": list(self.frenet_s),
            "frenet_d": list(self.frenet_d),
            "spatial_path_xy": [list(p) for p in self.spatial_path_xy],
            "speed_samples_mps": list(self.speed_samples_mps),
            "proposal_path_hash": self.proposal_path_hash,
            "timed_trajectory_hash": self.timed_trajectory_hash,
            "native_anchor_hash": self.native_anchor_hash,
            "head_lineage": self.head_lineage,
            "branch_type": self.branch_type,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SerializedCandidateV2":
        pts = tuple(tuple(float(x) for x in row) for row in d["points_xy_yaw_v_a_kappa"])
        path = tuple(tuple(float(x) for x in row) for row in d["spatial_path_xy"])
        return SerializedCandidateV2(
            candidate_id=str(d["candidate_id"]),
            candidate_index=int(d["candidate_index"]),
            mode_id=str(d.get("mode_id", "")),
            available=bool(d.get("available", True)),
            availability_reason=str(d.get("availability_reason", "")),
            probability=float(d["probability"]),
            points_xy_yaw_v_a_kappa=pts,
            frenet_s=tuple(float(x) for x in d.get("frenet_s", ())),
            frenet_d=tuple(float(x) for x in d.get("frenet_d", ())),
            spatial_path_xy=path,  # type: ignore[arg-type]
            speed_samples_mps=tuple(float(x) for x in d["speed_samples_mps"]),
            proposal_path_hash=str(d["proposal_path_hash"]),
            timed_trajectory_hash=str(d["timed_trajectory_hash"]),
            native_anchor_hash=str(d["native_anchor_hash"]),
            head_lineage=str(d.get("head_lineage", "")),
            branch_type=str(d.get("branch_type", "learned_spatial_semantic")),
        )


@dataclass(frozen=True)
class K2AnchorArtifactV2:
    """Frozen Spatial K2 V2 from a single backbone forward; branch force uses this only."""

    schema_version: str
    pair_id: str
    scenario_id: str
    seed_id: str
    anchor_run_id: str
    anchor_carla_frame: int
    anchor_simulation_time_s: float
    requested_initial_state_hash: str
    measured_initial_state_hash: str
    observation_fingerprint: ObservationFingerprint
    model_id: str
    model_checkpoint_hash: str
    spatial_head_checkpoint_hash: str
    config_hash: str
    backbone_forward_id: str
    executor_config_hash: str
    native_path_xy: tuple[tuple[float, float], ...]
    native_path_hash: str
    candidates: tuple[SerializedCandidateV2, ...]
    top1_index: int
    guard_status: str
    guard_reasons: tuple[str, ...]
    guard_metrics: Mapping[str, Any] = field(default_factory=dict)
    probability_source: str = "fixed_prior_uncalibrated"
    branch_type: str = "learned_spatial_semantic"
    set_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    evidence_lineage: str = "spatial_mode_head"  # or contract_probe
    k: int = 2
    t_steps: int = 10
    dt_s: float = 0.25
    horizon_s: float = 2.5

    def __post_init__(self) -> None:
        if self.schema_version != K2_ANCHOR_SCHEMA_V2:
            raise ContractError(
                f"anchor V2 schema must be {K2_ANCHOR_SCHEMA_V2}, got {self.schema_version}"
            )
        if self.k != 2 or self.t_steps != 10:
            raise ContractError("R2-X freezes K=2 T=10")
        if abs(self.dt_s - 0.25) > 1e-9 or abs(self.horizon_s - 2.5) > 1e-9:
            raise ContractError("R2-X freezes dt=0.25 horizon=2.5")
        if len(self.candidates) != 2:
            raise ContractError(f"expected 2 candidates, got {len(self.candidates)}")
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != 2:
            raise ContractError(f"duplicate candidate ids: {ids}")
        if self.top1_index not in (0, 1):
            raise ContractError("top1_index must be 0 or 1")
        if self.evidence_lineage not in {"spatial_mode_head", "contract_probe", "teacher_label"}:
            raise ContractError(f"unsupported evidence_lineage: {self.evidence_lineage}")

    def payload_for_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "seed_id": self.seed_id,
            "anchor_run_id": self.anchor_run_id,
            "anchor_carla_frame": int(self.anchor_carla_frame),
            "anchor_simulation_time_s": float(self.anchor_simulation_time_s),
            "requested_initial_state_hash": self.requested_initial_state_hash,
            "measured_initial_state_hash": self.measured_initial_state_hash,
            "observation_fingerprint": self.observation_fingerprint.to_dict(),
            "model_id": self.model_id,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "spatial_head_checkpoint_hash": self.spatial_head_checkpoint_hash,
            "config_hash": self.config_hash,
            "backbone_forward_id": self.backbone_forward_id,
            "executor_config_hash": self.executor_config_hash,
            "native_path_xy": [list(p) for p in self.native_path_xy],
            "native_path_hash": self.native_path_hash,
            "candidates": [c.to_dict() for c in self.candidates],
            "top1_index": int(self.top1_index),
            "guard_status": self.guard_status,
            "guard_reasons": list(self.guard_reasons),
            "guard_metrics": dict(self.guard_metrics),
            "probability_source": self.probability_source,
            "branch_type": self.branch_type,
            "set_diagnostics": dict(self.set_diagnostics),
            "evidence_lineage": self.evidence_lineage,
            "k": int(self.k),
            "t_steps": int(self.t_steps),
            "dt_s": float(self.dt_s),
            "horizon_s": float(self.horizon_s),
        }

    def artifact_content_hash(self) -> str:
        return content_hash(self.payload_for_hash(), nibble=64)

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_for_hash()
        payload["artifact_content_hash"] = self.artifact_content_hash()
        return payload

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "K2AnchorArtifactV2":
        obs_raw = d["observation_fingerprint"]
        obs = ObservationFingerprint(
            front_rgb_sha256=str(obs_raw["front_rgb_sha256"]),
            image_height=int(obs_raw["image_height"]),
            image_width=int(obs_raw["image_width"]),
            image_channels=int(obs_raw["image_channels"]),
            image_layout=str(obs_raw["image_layout"]),
            ego_observable=dict(obs_raw["ego_observable"]),
            route_targets=list(obs_raw["route_targets"]),
            camera_frame=dict(obs_raw["camera_frame"]),
            k2_bundle_hash=str(obs_raw["k2_bundle_hash"]),
        )
        cands = tuple(SerializedCandidateV2.from_dict(c) for c in d["candidates"])
        path = tuple(tuple(float(x) for x in row) for row in d["native_path_xy"])
        art = K2AnchorArtifactV2(
            schema_version=str(d["schema_version"]),
            pair_id=str(d["pair_id"]),
            scenario_id=str(d["scenario_id"]),
            seed_id=str(d["seed_id"]),
            anchor_run_id=str(d["anchor_run_id"]),
            anchor_carla_frame=int(d["anchor_carla_frame"]),
            anchor_simulation_time_s=float(d["anchor_simulation_time_s"]),
            requested_initial_state_hash=str(d["requested_initial_state_hash"]),
            measured_initial_state_hash=str(d["measured_initial_state_hash"]),
            observation_fingerprint=obs,
            model_id=str(d["model_id"]),
            model_checkpoint_hash=str(d["model_checkpoint_hash"]),
            spatial_head_checkpoint_hash=str(d.get("spatial_head_checkpoint_hash", "unset")),
            config_hash=str(d["config_hash"]),
            backbone_forward_id=str(d["backbone_forward_id"]),
            executor_config_hash=str(d["executor_config_hash"]),
            native_path_xy=path,  # type: ignore[arg-type]
            native_path_hash=str(d["native_path_hash"]),
            candidates=cands,
            top1_index=int(d["top1_index"]),
            guard_status=str(d["guard_status"]),
            guard_reasons=tuple(str(x) for x in d.get("guard_reasons", ())),
            guard_metrics=dict(d.get("guard_metrics", {})),
            probability_source=str(d.get("probability_source", "fixed_prior_uncalibrated")),
            branch_type=str(d.get("branch_type", "learned_spatial_semantic")),
            set_diagnostics=dict(d.get("set_diagnostics", {})),
            evidence_lineage=str(d.get("evidence_lineage", "spatial_mode_head")),
            k=int(d.get("k", 2)),
            t_steps=int(d.get("t_steps", 10)),
            dt_s=float(d.get("dt_s", 0.25)),
            horizon_s=float(d.get("horizon_s", 2.5)),
        )
        expected = d.get("artifact_content_hash")
        if expected is not None and str(expected) != art.artifact_content_hash():
            raise ContractError(
                "artifact_content_hash mismatch on deserialize "
                f"(stored={expected}, recomputed={art.artifact_content_hash()})"
            )
        return art

    @staticmethod
    def from_json_bytes(data: bytes) -> "K2AnchorArtifactV2":
        return K2AnchorArtifactV2.from_dict(json.loads(data.decode("utf-8")))


def assert_no_oracle_in_control_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if oracle-only fields leak into a control-facing dict."""
    stack: list[Any] = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Mapping):
            for k, v in cur.items():
                key = str(k)
                if key in CONTROL_NAMESPACE_DENYLIST:
                    raise ContractError(f"oracle field leaked into control payload: {key}")
                if key.startswith("oracle_") and key not in {"oracle_decision_level"}:
                    # Allow nothing under control path; offline pair oracle files are separate.
                    raise ContractError(f"oracle-prefixed field in control payload: {key}")
                stack.append(v)
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)


def build_model_retimer_hash(
    *,
    model_id: str,
    model_checkpoint_hash: str,
    config_hash: str,
    retimer_version: str,
    retimer_hash: str,
) -> str:
    return content_hash(
        {
            "model_id": model_id,
            "model_checkpoint_hash": model_checkpoint_hash,
            "config_hash": config_hash,
            "retimer_version": retimer_version,
            "retimer_hash": retimer_hash,
        },
        nibble=32,
    )
