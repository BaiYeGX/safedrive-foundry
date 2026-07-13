"""Map Safety contracts to existing sdf_interfaces message-shaped dictionaries.

ROS SafetyStatus.msg:
  UNKNOWN=0, CLEAR=1, CAUTION=2, INTERVENE=3, EMERGENCY=4

Task SafetyMode:
  NORMAL, DEGRADED, MINIMAL_RISK, EMERGENCY

These adapters do not create ROS nodes or import rclpy.
"""

from __future__ import annotations

from typing import Any, Mapping

from safety_kernel.contracts.serialize import decision_to_dict, point_to_dict
from safety_kernel.contracts.types import (
    DecisionKind,
    PolicyCandidate,
    SafetyDecision,
    SafetyMode,
)

# Mirrors sdf_interfaces/msg/SafetyStatus.msg integer levels.
ROS_UNKNOWN = 0
ROS_CLEAR = 1
ROS_CAUTION = 2
ROS_INTERVENE = 3
ROS_EMERGENCY = 4

# Mirrors sdf_interfaces/msg/PolicyDecision.msg policy_kind.
POLICY_UNKNOWN = 0
POLICY_CLASSIC_EXPERT = 1
POLICY_VLA = 2
POLICY_SAFETY_FALLBACK = 3

_MODE_TO_LEVEL = {
    SafetyMode.NORMAL: ROS_CLEAR,
    SafetyMode.DEGRADED: ROS_CAUTION,
    SafetyMode.MINIMAL_RISK: ROS_INTERVENE,
    SafetyMode.EMERGENCY: ROS_EMERGENCY,
}

_MODE_TO_NAME = {
    SafetyMode.NORMAL: "CLEAR",
    SafetyMode.DEGRADED: "CAUTION",
    SafetyMode.MINIMAL_RISK: "INTERVENE",
    SafetyMode.EMERGENCY: "EMERGENCY",
}


def safety_mode_to_ros_level(mode: SafetyMode) -> int:
    return _MODE_TO_LEVEL[mode]


def safety_mode_to_ros_name(mode: SafetyMode) -> str:
    return _MODE_TO_NAME[mode]


def decision_to_safety_status_dict(
    decision: SafetyDecision,
    *,
    frame: Mapping[str, Any] | None = None,
    mode: SafetyMode | None = None,
) -> dict[str, Any]:
    """Project SafetyDecision into SafetyStatus.msg-shaped dict."""
    effective_mode = mode or decision.state_after
    hard_margins = [m for m in decision.constraint_margins if m.hard]
    min_ttc = 0.0
    # Approximate TTC proxy from collision margin if present.
    for m in hard_margins:
        if m.name == "collision" and m.margin >= 0:
            min_ttc = max(min_ttc, float(m.margin))
    slack = float(decision.slack)
    reason = decision.fallback_request.reason_code if decision.fallback_request else (
        decision.reject_reasons[0] if decision.reject_reasons else decision.decision_kind.value
    )
    return {
        "frame": dict(frame or {}),
        "level": safety_mode_to_ros_level(effective_mode),
        "level_name": safety_mode_to_ros_name(effective_mode),
        "oracle_observation": False,
        "min_ttc_s": float(min_ttc),
        "max_constraint_slack": slack,
        "risk_upper_bound": max(0.0, -slack) if slack < 0 else 0.0,
        "reason_code": str(reason)[:128],
        "decision_kind": decision.decision_kind.value,
        "decision_id": decision.decision_id,
    }


def decision_to_policy_decision_dict(
    decision: SafetyDecision,
    *,
    frame: Mapping[str, Any] | None = None,
    policy_id: str = "safety_kernel",
) -> dict[str, Any]:
    """Project into PolicyDecision.msg-shaped dict.

    Confidence reflects executable decisions: ACCEPT=1.0, QP=0.9, RATO=0.85.
    Non-executable rejects/fallbacks/emergency stay at 0.0.
    """
    kind = POLICY_SAFETY_FALLBACK
    confidence = 0.0
    has_exec = bool(decision.executed_trajectory_id)
    if decision.decision_kind is DecisionKind.ACCEPT and has_exec:
        confidence = 1.0
        if decision.accepted_candidate is not None:
            src = decision.accepted_candidate.source.value
            if src.startswith("vla"):
                kind = POLICY_VLA
            elif src == "classic":
                kind = POLICY_CLASSIC_EXPERT
            else:
                kind = POLICY_SAFETY_FALLBACK
    elif decision.decision_kind is DecisionKind.QP and has_exec:
        confidence = 0.9
        kind = POLICY_SAFETY_FALLBACK
    elif decision.decision_kind is DecisionKind.RATO and has_exec:
        confidence = 0.85
        kind = POLICY_SAFETY_FALLBACK
    return {
        "frame": dict(frame or {}),
        "policy_kind": kind,
        "policy_id": policy_id,
        "candidate_trajectory_id": decision.executed_trajectory_id or "",
        "confidence": confidence,
        "uncertainty": float(decision.accepted_candidate.uncertainty) if decision.accepted_candidate else 1.0,
        "generated_frame": 0,
        "planned_frame": 0,
        "decision_kind": decision.decision_kind.value,
    }


def candidate_to_trajectory_dict(candidate: PolicyCandidate, *, frame: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project PolicyCandidate into Trajectory.msg-shaped dict (points simplified)."""
    return {
        "frame": dict(frame or {}),
        "trajectory_id": candidate.candidate_id,
        "generated_frame": 0,
        "planned_frame": 0,
        "intended_execution_frame": 0,
        "points": [
            {
                "time_from_start_s": p.t,
                "pose": {
                    "position": {"x": p.x, "y": p.y, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "speed_mps": p.v,
                "acceleration_mps2": p.a,
                "curvature_per_m": p.kappa,
                "yaw": p.yaw,
                "jerk": p.jerk,
            }
            for p in candidate.points
        ],
        "risk_cost": float(candidate.dynamics_meta.get("risk_cost", candidate.uncertainty)),
        "tracking_cost": float(candidate.dynamics_meta.get("tracking_cost", 0.0)),
        "source": candidate.source.value,
    }


def candidate_points_native(candidate: PolicyCandidate) -> list[dict[str, float]]:
    return [point_to_dict(p) for p in candidate.points]
