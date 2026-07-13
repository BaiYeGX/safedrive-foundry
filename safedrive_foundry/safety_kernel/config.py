"""Versioned Safety Kernel configuration loader."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _ROOT / "config" / "safety_kernel" / "baseline.toml"


def config_sha256(raw_toml: str) -> str:
    return hashlib.sha256(raw_toml.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QpRepairConfig:
    """Frozen longitudinal QP parameters (not tunable by learning modules)."""

    enabled: bool
    solver: str
    deadline_ms: float
    max_iter: int
    abs_tol: float
    rel_tol: float
    warm_start: bool
    w_v_ref: float
    w_a: float
    w_jerk: float
    w_slack: float
    w_progress: float
    slack_stop_max_m: float
    slack_lead_max_m: float
    slack_speed_max_mps: float
    min_gap_m: float
    stop_line_buffer_m: float
    time_headway_s: float
    min_progress_ratio: float
    repair_on_hard_reject: bool


@dataclass(frozen=True)
class RatoScpConfig:
    """Frozen restricted 2D RATO-SCP parameters (secondary repair only)."""

    enabled: bool
    deadline_ms: float
    max_scp_iters: int
    trust_radius_m: float
    max_lateral_step_m: float
    warm_start: bool
    w_path: float
    w_smooth: float
    w_slack: float
    w_progress: float
    slack_corridor_max_m: float
    slack_collision_max_m: float
    min_lateral_clearance_m: float
    min_qp_progress_to_skip: float
    oscillation_eps_m: float
    min_progress_ratio: float
    repair_on_hard_reject: bool


@dataclass(frozen=True)
class ArbitrationConfig:
    """Frozen arbitration / shadow / degradation (G2-04). Soft scores cannot override hard safety."""

    enabled: bool
    deadline_ms: float
    w_progress: float
    w_comfort: float
    w_margin: float
    w_probability: float
    w_uncertainty: float
    classic_source_bonus: float
    vla_source_bonus: float
    world_ranked_bonus: float
    max_final_candidates: int
    shadow_enabled: bool
    overconfident_prob_min: float
    overconfident_uncertainty_max: float
    ood_uncertainty_min: float
    soft_stale_age_s: float


@dataclass(frozen=True)
class SafetyKernelConfig:
    schema_version: str
    name: str
    raw_toml: str
    control_period_s: float
    state_check_deadline_ms: float
    candidate_check_deadline_ms: float
    default_horizon_s: float
    min_horizon_s: float
    max_horizon_s: float
    max_candidate_age_s: float
    min_points: int
    wheelbase_m: float
    max_speed_mps: float
    max_accel_mps2: float
    max_decel_mps2: float
    max_jerk_mps3: float
    max_curvature_per_m: float
    max_lateral_accel_mps2: float
    width_m: float
    length_m: float
    collision_inflate_m: float
    require_drivable: bool
    max_offroad_m: float
    lane_half_width_m: float
    enforce_speed_limit: bool
    speed_limit_margin_mps: float
    enforce_red_light_stop: bool
    red_light_stop_distance_m: float
    red_light_max_approach_speed_mps: float
    escalate_debounce_frames: int
    recover_debounce_frames: int
    min_dwell_s: float
    recover_clear_frames: int
    emit_safety_events: bool
    record_failure_samples: bool
    max_failure_samples: int
    qp: QpRepairConfig
    rato: RatoScpConfig
    arbitration: ArbitrationConfig

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "control_period_s": self.control_period_s,
            "state_check_deadline_ms": self.state_check_deadline_ms,
            "candidate_check_deadline_ms": self.candidate_check_deadline_ms,
            "max_candidate_age_s": self.max_candidate_age_s,
            "max_speed_mps": self.max_speed_mps,
            "max_curvature_per_m": self.max_curvature_per_m,
            "escalate_debounce_frames": self.escalate_debounce_frames,
            "min_dwell_s": self.min_dwell_s,
            "recover_clear_frames": self.recover_clear_frames,
            "qp_enabled": self.qp.enabled,
            "qp_deadline_ms": self.qp.deadline_ms,
            "qp_slack_stop_max_m": self.qp.slack_stop_max_m,
            "qp_slack_lead_max_m": self.qp.slack_lead_max_m,
            "qp_min_progress_ratio": self.qp.min_progress_ratio,
            "rato_enabled": self.rato.enabled,
            "rato_deadline_ms": self.rato.deadline_ms,
            "rato_max_scp_iters": self.rato.max_scp_iters,
            "rato_min_qp_progress_to_skip": self.rato.min_qp_progress_to_skip,
            "arbitration_enabled": self.arbitration.enabled,
            "shadow_enabled": self.arbitration.shadow_enabled,
        }


def load_safety_config(path: Path | str | None = None) -> SafetyKernelConfig:
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    raw = cfg_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    return _parse(data, raw)


def _f(mapping: Mapping[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    return float(value)


def _i(mapping: Mapping[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    return int(value)


def _b(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    return bool(value)


def _parse_qp(data: Mapping[str, Any]) -> QpRepairConfig:
    qp = data.get("qp", {})
    return QpRepairConfig(
        enabled=_b(qp, "enabled", True),
        solver=str(qp.get("solver", "osqp_auto")),
        deadline_ms=_f(qp, "deadline_ms", 50.0),
        max_iter=_i(qp, "max_iter", 8000),
        abs_tol=_f(qp, "abs_tol", 1.0e-4),
        rel_tol=_f(qp, "rel_tol", 1.0e-4),
        warm_start=_b(qp, "warm_start", True),
        w_v_ref=_f(qp, "w_v_ref", 1.0),
        w_a=_f(qp, "w_a", 0.15),
        w_jerk=_f(qp, "w_jerk", 0.05),
        w_slack=_f(qp, "w_slack", 80.0),
        w_progress=_f(qp, "w_progress", 2.0),
        slack_stop_max_m=_f(qp, "slack_stop_max_m", 0.35),
        slack_lead_max_m=_f(qp, "slack_lead_max_m", 0.50),
        slack_speed_max_mps=_f(qp, "slack_speed_max_mps", 0.25),
        min_gap_m=_f(qp, "min_gap_m", 2.5),
        stop_line_buffer_m=_f(qp, "stop_line_buffer_m", 1.0),
        time_headway_s=_f(qp, "time_headway_s", 0.8),
        min_progress_ratio=_f(qp, "min_progress_ratio", 0.35),
        repair_on_hard_reject=_b(qp, "repair_on_hard_reject", True),
    )


def _parse_rato(data: Mapping[str, Any]) -> RatoScpConfig:
    rato = data.get("rato", {})
    return RatoScpConfig(
        enabled=_b(rato, "enabled", True),
        deadline_ms=_f(rato, "deadline_ms", 80.0),
        max_scp_iters=_i(rato, "max_scp_iters", 4),
        trust_radius_m=_f(rato, "trust_radius_m", 1.5),
        max_lateral_step_m=_f(rato, "max_lateral_step_m", 0.85),
        warm_start=_b(rato, "warm_start", True),
        w_path=_f(rato, "w_path", 1.0),
        w_smooth=_f(rato, "w_smooth", 2.5),
        w_slack=_f(rato, "w_slack", 100.0),
        w_progress=_f(rato, "w_progress", 1.5),
        slack_corridor_max_m=_f(rato, "slack_corridor_max_m", 0.25),
        slack_collision_max_m=_f(rato, "slack_collision_max_m", 0.40),
        min_lateral_clearance_m=_f(rato, "min_lateral_clearance_m", 0.55),
        min_qp_progress_to_skip=_f(rato, "min_qp_progress_to_skip", 0.55),
        oscillation_eps_m=_f(rato, "oscillation_eps_m", 0.03),
        min_progress_ratio=_f(rato, "min_progress_ratio", 0.20),
        repair_on_hard_reject=_b(rato, "repair_on_hard_reject", True),
    )


def _parse_arbitration(data: Mapping[str, Any]) -> ArbitrationConfig:
    arb = data.get("arbitration", {})
    return ArbitrationConfig(
        enabled=_b(arb, "enabled", True),
        deadline_ms=_f(arb, "deadline_ms", 10.0),
        w_progress=_f(arb, "w_progress", 1.0),
        w_comfort=_f(arb, "w_comfort", 0.35),
        w_margin=_f(arb, "w_margin", 1.5),
        w_probability=_f(arb, "w_probability", 0.5),
        w_uncertainty=_f(arb, "w_uncertainty", 0.4),
        classic_source_bonus=_f(arb, "classic_source_bonus", 0.15),
        vla_source_bonus=_f(arb, "vla_source_bonus", 0.05),
        world_ranked_bonus=_f(arb, "world_ranked_bonus", 0.02),
        max_final_candidates=_i(arb, "max_final_candidates", 4),
        shadow_enabled=_b(arb, "shadow_enabled", True),
        overconfident_prob_min=_f(arb, "overconfident_prob_min", 0.95),
        overconfident_uncertainty_max=_f(arb, "overconfident_uncertainty_max", 0.05),
        ood_uncertainty_min=_f(arb, "ood_uncertainty_min", 0.85),
        soft_stale_age_s=_f(arb, "soft_stale_age_s", 0.20),
    )


def _parse(data: Mapping[str, Any], raw: str) -> SafetyKernelConfig:
    timing = data.get("timing", {})
    vehicle = data.get("vehicle", {})
    road = data.get("road", {})
    rules = data.get("rules", {})
    sm = data.get("state_machine", {})
    logging = data.get("logging", {})
    return SafetyKernelConfig(
        schema_version=str(data.get("schema_version", "safedrive.safety.config.v1")),
        name=str(data.get("name", "safety_kernel_baseline")),
        raw_toml=raw,
        control_period_s=_f(timing, "control_period_s", 0.02),
        state_check_deadline_ms=_f(timing, "state_check_deadline_ms", 5.0),
        candidate_check_deadline_ms=_f(timing, "candidate_check_deadline_ms", 15.0),
        default_horizon_s=_f(timing, "default_horizon_s", 4.0),
        min_horizon_s=_f(timing, "min_horizon_s", 2.5),
        max_horizon_s=_f(timing, "max_horizon_s", 5.5),
        max_candidate_age_s=_f(timing, "max_candidate_age_s", 0.25),
        min_points=_i(timing, "min_points", 3),
        wheelbase_m=_f(vehicle, "wheelbase_m", 2.8),
        max_speed_mps=_f(vehicle, "max_speed_mps", 15.0),
        max_accel_mps2=_f(vehicle, "max_accel_mps2", 2.5),
        max_decel_mps2=_f(vehicle, "max_decel_mps2", 5.0),
        max_jerk_mps3=_f(vehicle, "max_jerk_mps3", 6.0),
        max_curvature_per_m=_f(vehicle, "max_curvature_per_m", 0.30),
        max_lateral_accel_mps2=_f(vehicle, "max_lateral_accel_mps2", 3.0),
        width_m=_f(vehicle, "width_m", 1.9),
        length_m=_f(vehicle, "length_m", 4.5),
        collision_inflate_m=_f(vehicle, "collision_inflate_m", 0.35),
        require_drivable=_b(road, "require_drivable", True),
        max_offroad_m=_f(road, "max_offroad_m", 0.5),
        lane_half_width_m=_f(road, "lane_half_width_m", 1.75),
        enforce_speed_limit=_b(rules, "enforce_speed_limit", True),
        speed_limit_margin_mps=_f(rules, "speed_limit_margin_mps", 0.5),
        enforce_red_light_stop=_b(rules, "enforce_red_light_stop", True),
        red_light_stop_distance_m=_f(rules, "red_light_stop_distance_m", 8.0),
        red_light_max_approach_speed_mps=_f(rules, "red_light_max_approach_speed_mps", 1.0),
        escalate_debounce_frames=_i(sm, "escalate_debounce_frames", 2),
        recover_debounce_frames=_i(sm, "recover_debounce_frames", 5),
        min_dwell_s=_f(sm, "min_dwell_s", 0.10),
        recover_clear_frames=_i(sm, "recover_clear_frames", 8),
        emit_safety_events=_b(logging, "emit_safety_events", True),
        record_failure_samples=_b(logging, "record_failure_samples", True),
        max_failure_samples=_i(logging, "max_failure_samples", 32),
        qp=_parse_qp(data),
        rato=_parse_rato(data),
        arbitration=_parse_arbitration(data),
    )
