"""R2-X Spatial K2 V2 types (schema independent of longitudinal V1)."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_V2 = "safedrive.k2.spatial.v2"
BRANCH_TYPE_V2 = "learned_spatial_semantic"
DEFAULT_K2_V2_TOML = (
    Path(__file__).resolve().parents[2] / "config" / "vla" / "k2_v2_spatial.toml"
)

MODE_NOMINAL = "nominal_progress"
MODE_DEFENSIVE = "defensive_alternative"
ID_NOMINAL = "v2_nominal_progress"
ID_DEFENSIVE = "v2_defensive_alternative"

GUARD_OK = "OK"
GUARD_REJECT = "REJECT"

# Fail-closed reason codes (R2X §15.4)
FORWARD_ID_MISMATCH = "FORWARD_ID_MISMATCH"
HEAD_LINEAGE_INVALID = "HEAD_LINEAGE_INVALID"
PROPOSAL_PATH_HASH_MISMATCH = "PROPOSAL_PATH_HASH_MISMATCH"
TIMED_PATH_BINDING_MISMATCH = "TIMED_PATH_BINDING_MISMATCH"
FRENET_PROGRESS_NEGATIVE = "FRENET_PROGRESS_NEGATIVE"
FRENET_LATERAL_ENVELOPE = "FRENET_LATERAL_ENVELOPE"
NEAR_FIELD_DISCONTINUITY = "NEAR_FIELD_DISCONTINUITY"
SPATIAL_COLLAPSE_ELIGIBLE = "SPATIAL_COLLAPSE_ELIGIBLE"
CURVATURE_ENVELOPE = "CURVATURE_ENVELOPE"
ALTERNATIVE_UNAVAILABLE = "ALTERNATIVE_UNAVAILABLE"
SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
SET_IDENTITY = "SET_IDENTITY"
PROBABILITY_INVALID = "PROBABILITY_INVALID"
TOP1_INVALID = "TOP1_INVALID"
ACCEL_ENVELOPE = "ACCEL_ENVELOPE"
POSITION_INTEGRAL_RESIDUAL = "POSITION_INTEGRAL_RESIDUAL"
YAW_RECOMPUTE_MISMATCH = "YAW_RECOMPUTE_MISMATCH"
KAPPA_RECOMPUTE_MISMATCH = "KAPPA_RECOMPUTE_MISMATCH"
NATIVE_CORRIDOR = "NATIVE_CORRIDOR"
TIMED_XTRACK = "TIMED_XTRACK"
PATH_TOO_SHORT = "PATH_TOO_SHORT"
SELF_INTERSECTION = "SELF_INTERSECTION"
FORWARD_RATIO = "FORWARD_RATIO"
END_POINT_COPY_POS_SPEED = "END_POINT_COPY_POS_SPEED"
T_MISMATCH = "T_MISMATCH"
UNAVAILABLE_FORCE = "UNAVAILABLE_FORCE"
EXEC_SPEC_HASH_MISMATCH = "EXEC_SPEC_HASH_MISMATCH"


def stable_hash_xy(path_xy: Sequence[tuple[float, float]]) -> str:
    """Short hash for logs/metrics (not formal lineage)."""
    payload = ";".join(f"{float(x):.6f},{float(y):.6f}" for x, y in path_xy)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_hash_traj(points: Sequence[Sequence[float]]) -> str:
    """Short hash for logs/metrics (not formal lineage)."""
    payload = ";".join(
        ",".join(f"{float(v):.6f}" for v in row[:6]) for row in points
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def canonical_sha256(obj: Any) -> str:
    """Full SHA256 of canonical JSON (formal lineage)."""
    import json

    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class K2SpatialConfig:
    k: int = 2
    t_steps: int = 10
    dt_s: float = 0.25
    horizon_s: float = 2.5
    top1_index: int = 0
    candidate_ids: tuple[str, str] = (ID_NOMINAL, ID_DEFENSIVE)
    mode_ids: tuple[str, str] = (MODE_NOMINAL, MODE_DEFENSIVE)
    max_lateral_residual_m: float = 1.0
    first_step_position_residual_m: float = 0.05
    near_field_horizon_s: float = 0.5
    near_field_max_inter_candidate_lat_m: float = 0.20
    ambiguity_min_spatial_sep_m: float = 0.50
    envelope_ramp_points: int = 3
    max_delta_s_per_step_m: float = 3.0
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 3.0
    stop_threshold_mps: float = 0.35
    require_shared_backbone_forward_id: bool = True
    reject_fixed_bias_lineage: bool = True
    reject_noise_lineage: bool = True
    native_corridor_cross_track_max_m: float = 1.25
    timed_to_proposal_xtrack_max_m: float = 0.15
    proposal_path_min_points: int = 2
    signed_progress_min_m: float = -1.0e-6
    hard_max_abs_curvature: float = 1.0
    max_switch_lateral_5m: float = 1.0
    min_nominal_speed_mps: float = 0.75
    min_final_progress_m: float = 1.5
    availability_semantics: str = "executability_only_v1"
    learned_confidence_non_blocking: bool = True
    nominal_anchor_exact: bool = True
    # recompute tolerances
    position_integral_tol_m: float = 0.35
    position_integral_gross_reject_m: float = 3.0
    yaw_recompute_tol_rad: float = 0.75
    kappa_recompute_tol: float = 1.5
    forward_ratio_min: float = 0.55
    branch_type: str = BRANCH_TYPE_V2
    schema_version: str = SCHEMA_V2

    def config_hash(self) -> str:
        """Full SHA256 of frozen contract thresholds (formal lineage)."""
        payload = {
            "schema_version": self.schema_version,
            "branch_type": self.branch_type,
            "k": self.k,
            "t_steps": self.t_steps,
            "dt_s": self.dt_s,
            "horizon_s": self.horizon_s,
            "top1_index": self.top1_index,
            "candidate_ids": list(self.candidate_ids),
            "mode_ids": list(self.mode_ids),
            "max_lateral_residual_m": self.max_lateral_residual_m,
            "first_step_position_residual_m": self.first_step_position_residual_m,
            "near_field_horizon_s": self.near_field_horizon_s,
            "near_field_max_inter_candidate_lat_m": self.near_field_max_inter_candidate_lat_m,
            "ambiguity_min_spatial_sep_m": self.ambiguity_min_spatial_sep_m,
            "envelope_ramp_points": self.envelope_ramp_points,
            "max_delta_s_per_step_m": self.max_delta_s_per_step_m,
            "max_accel_mps2": self.max_accel_mps2,
            "max_decel_mps2": self.max_decel_mps2,
            "stop_threshold_mps": self.stop_threshold_mps,
            "require_shared_backbone_forward_id": self.require_shared_backbone_forward_id,
            "reject_fixed_bias_lineage": self.reject_fixed_bias_lineage,
            "reject_noise_lineage": self.reject_noise_lineage,
            "native_corridor_cross_track_max_m": self.native_corridor_cross_track_max_m,
            "timed_to_proposal_xtrack_max_m": self.timed_to_proposal_xtrack_max_m,
            "proposal_path_min_points": self.proposal_path_min_points,
            "signed_progress_min_m": self.signed_progress_min_m,
            "hard_max_abs_curvature": self.hard_max_abs_curvature,
            "max_switch_lateral_5m": self.max_switch_lateral_5m,
            "min_nominal_speed_mps": self.min_nominal_speed_mps,
            "min_final_progress_m": self.min_final_progress_m,
            "availability_semantics": self.availability_semantics,
            "learned_confidence_non_blocking": self.learned_confidence_non_blocking,
            "nominal_anchor_exact": self.nominal_anchor_exact,
            "position_integral_tol_m": self.position_integral_tol_m,
            "position_integral_gross_reject_m": self.position_integral_gross_reject_m,
            "yaw_recompute_tol_rad": self.yaw_recompute_tol_rad,
            "kappa_recompute_tol": self.kappa_recompute_tol,
            "forward_ratio_min": self.forward_ratio_min,
        }
        return canonical_sha256(payload)


def load_k2_spatial_config(path: Path | str | None = None) -> K2SpatialConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_K2_V2_TOML
    if not cfg_path.is_file():
        return K2SpatialConfig()
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)
    c = raw.get("contract", {})
    f = raw.get("frenet", {})
    d = raw.get("dynamics", {})
    g = raw.get("guard", {})
    e = raw.get("eligibility", {})
    p = raw.get("path_manager_align", {})
    proposal = raw.get("proposal", {})
    ids = c.get("candidate_ids") or [ID_NOMINAL, ID_DEFENSIVE]
    modes = c.get("mode_ids") or [MODE_NOMINAL, MODE_DEFENSIVE]
    ramp_s = float(f.get("envelope_ramp_s", 0.5))
    dt = float(c.get("dt_s", 0.25))
    ramp_pts = max(1, int(round(ramp_s / max(dt, 1e-6))))
    return K2SpatialConfig(
        k=int(c.get("k", 2)),
        t_steps=int(c.get("t_steps", 10)),
        dt_s=dt,
        horizon_s=float(c.get("horizon_s", 2.5)),
        top1_index=int(c.get("top1_index", 0)),
        candidate_ids=(str(ids[0]), str(ids[1])),
        mode_ids=(str(modes[0]), str(modes[1])),
        max_lateral_residual_m=float(f.get("max_lateral_residual_m", 1.0)),
        first_step_position_residual_m=float(f.get("first_step_position_residual_m", 0.05)),
        near_field_horizon_s=float(f.get("near_field_horizon_s", 0.5)),
        near_field_max_inter_candidate_lat_m=float(
            f.get("near_field_max_inter_candidate_lat_m", 0.20)
        ),
        ambiguity_min_spatial_sep_m=float(f.get("ambiguity_min_spatial_sep_m", 0.50)),
        envelope_ramp_points=ramp_pts,
        max_delta_s_per_step_m=float(f.get("max_delta_s_per_step_m", 3.0)),
        max_accel_mps2=float(d.get("max_accel_mps2", 2.5)),
        max_decel_mps2=float(d.get("max_decel_mps2", 3.0)),
        stop_threshold_mps=float(d.get("stop_threshold_mps", 0.35)),
        require_shared_backbone_forward_id=bool(
            g.get("require_shared_backbone_forward_id", True)
        ),
        reject_fixed_bias_lineage=bool(g.get("reject_fixed_bias_lineage", True)),
        reject_noise_lineage=bool(g.get("reject_noise_lineage", True)),
        native_corridor_cross_track_max_m=float(
            g.get("native_corridor_cross_track_max_m", 1.25)
        ),
        timed_to_proposal_xtrack_max_m=float(g.get("timed_to_proposal_xtrack_max_m", 0.15)),
        proposal_path_min_points=int(g.get("proposal_path_min_points", 2)),
        signed_progress_min_m=float(g.get("signed_progress_min_m", -1.0e-6)),
        hard_max_abs_curvature=float(p.get("hard_max_abs_curvature", 1.0)),
        max_switch_lateral_5m=float(p.get("max_switch_lateral_5m", 1.0)),
        min_nominal_speed_mps=float(e.get("min_nominal_speed_mps", 0.75)),
        min_final_progress_m=float(e.get("min_final_progress_m", 1.5)),
        availability_semantics=str(
            proposal.get("availability_semantics", "executability_only_v1")
        ),
        learned_confidence_non_blocking=bool(
            proposal.get("learned_confidence_non_blocking", True)
        ),
        nominal_anchor_exact=bool(proposal.get("nominal_anchor_exact", True)),
        position_integral_tol_m=float(g.get("position_integral_tol_m", 0.35)),
        position_integral_gross_reject_m=float(
            g.get("position_integral_gross_reject_m", 3.0)
        ),
        yaw_recompute_tol_rad=float(g.get("yaw_recompute_tol_rad", 0.75)),
        kappa_recompute_tol=float(g.get("kappa_recompute_tol", 1.5)),
        forward_ratio_min=float(g.get("forward_ratio_min", 0.55)),
    )


@dataclass(frozen=True)
class K2ExecutionSpecV2:
    candidate_id: str
    spatial_path_xy: tuple[tuple[float, float], ...]
    speed_samples_mps: tuple[float, ...]
    spatial_path_hash: str
    timed_trajectory_hash: str
    native_anchor_hash: str
    branch_type: str = BRANCH_TYPE_V2
    mode_id: str = MODE_NOMINAL
    available: bool = True
    head_lineage: str = "spatial_mode_head"
    raw_head_output_hash: str = ""
    feature_content_hash: str = ""
    codec_config_hash: str = ""
    postprocess_hash: str = ""


@dataclass(frozen=True)
class K2CandidateV2:
    candidate_id: str
    mode_id: str
    available: bool
    availability_reason: str
    probability: float
    points_xy_yaw_v_a_kappa: tuple[tuple[float, ...], ...]
    frenet_s: tuple[float, ...]
    frenet_d: tuple[float, ...]
    proposal_path_hash: str
    timed_trajectory_hash: str
    native_anchor_hash: str
    head_lineage: str
    spatial_path_xy: tuple[tuple[float, float], ...]
    raw_head_output_hash: str = ""
    feature_content_hash: str = ""
    codec_config_hash: str = ""
    postprocess_hash: str = ""


@dataclass(frozen=True)
class K2PredictionBundleV2:
    schema_version: str
    observation_identity: Mapping[str, Any]
    model_id: str
    config_hash: str
    base_checkpoint_hash: str
    spatial_head_checkpoint_hash: str
    backbone_forward_id: str
    native_path_xy: tuple[tuple[float, float], ...]
    native_path_hash: str
    candidates: tuple[K2CandidateV2, K2CandidateV2]
    execution_specs: Mapping[str, K2ExecutionSpecV2]
    top1_index: int
    probability_source: str
    branch_type: str
    guard_status: str = GUARD_OK
    guard_reasons: tuple[str, ...] = ()
    build_error: str | None = None
    set_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    raw_head_output_hash: str = ""
    feature_content_hash: str = ""
    codec_config_hash: str = ""
    postprocess_hash: str = ""

    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(c.candidate_id for c in self.candidates)
