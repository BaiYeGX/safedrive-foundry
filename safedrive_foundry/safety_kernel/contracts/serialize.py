"""Schema validation and dict serialization for Safety contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.contracts.types import (
    CandidateSource,
    ConstraintMargin,
    DecisionKind,
    EventPhase,
    FallbackRequest,
    FallbackTarget,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyEvent,
    SafetyMode,
    TrajectoryPoint,
)


class ContractSchemaError(ValueError):
    """Raised when a payload fails Safety contract schema checks."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractSchemaError(f"{field} must be an object")
    return value


def _require_str(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ContractSchemaError(f"{field} must be a non-empty string")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractSchemaError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractSchemaError(f"{field} must be finite")
    return result


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractSchemaError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractSchemaError(f"{field} must be >= {minimum}")
    return value


def _point_from_dict(raw: Mapping[str, Any], prefix: str) -> TrajectoryPoint:
    required = ("t", "x", "y", "yaw", "kappa", "v", "a")
    for key in required:
        if key not in raw:
            raise ContractSchemaError(f"{prefix}.{key} is required")
    return TrajectoryPoint(
        t=_require_number(raw["t"], f"{prefix}.t"),
        x=_require_number(raw["x"], f"{prefix}.x"),
        y=_require_number(raw["y"], f"{prefix}.y"),
        yaw=_require_number(raw["yaw"], f"{prefix}.yaw"),
        kappa=_require_number(raw["kappa"], f"{prefix}.kappa"),
        v=_require_number(raw["v"], f"{prefix}.v"),
        a=_require_number(raw["a"], f"{prefix}.a"),
        jerk=_require_number(raw.get("jerk", 0.0), f"{prefix}.jerk"),
    )


def _candidate_from_dict(raw: Mapping[str, Any], prefix: str) -> PolicyCandidate:
    points_raw = raw.get("points")
    if not isinstance(points_raw, Sequence) or isinstance(points_raw, (str, bytes)):
        raise ContractSchemaError(f"{prefix}.points must be an array")
    points = tuple(_point_from_dict(_require_mapping(p, f"{prefix}.points[i]"), f"{prefix}.points[i]") for i, p in enumerate(points_raw))
    source_raw = raw.get("source", "unknown")
    try:
        source = CandidateSource(str(source_raw))
    except ValueError:
        source = CandidateSource.UNKNOWN
    dynamics = raw.get("dynamics_meta", {})
    if dynamics is None:
        dynamics = {}
    if not isinstance(dynamics, Mapping):
        raise ContractSchemaError(f"{prefix}.dynamics_meta must be an object")
    return PolicyCandidate(
        candidate_id=_require_str(raw.get("candidate_id"), f"{prefix}.candidate_id"),
        source=source,
        generated_time_s=_require_number(raw.get("generated_time_s"), f"{prefix}.generated_time_s"),
        valid_until_s=_require_number(raw.get("valid_until_s"), f"{prefix}.valid_until_s"),
        probability=_require_number(raw.get("probability", 1.0), f"{prefix}.probability"),
        points=points,
        behavior=str(raw.get("behavior", "") or ""),
        critical_actor=raw.get("critical_actor"),
        conflict_type=raw.get("conflict_type"),
        risk_horizon_s=_require_number(raw.get("risk_horizon_s", 0.0), f"{prefix}.risk_horizon_s"),
        intended_action=str(raw.get("intended_action", "") or ""),
        uncertainty=_require_number(raw.get("uncertainty", 0.0), f"{prefix}.uncertainty"),
        availability=bool(raw.get("availability", True)),
        dynamics_meta=dict(dynamics),
    )


def validate_candidate_set_schema(payload: Mapping[str, Any]) -> list[str]:
    """Return list of schema issues; empty means schema-ok (not dynamics-ok)."""
    issues: list[str] = []
    try:
        candidate_set_from_dict(payload)
    except ContractSchemaError as exc:
        issues.append(str(exc))
    return issues


def candidate_set_from_dict(payload: Mapping[str, Any]) -> PolicyCandidateSet:
    data = _require_mapping(payload, "PolicyCandidateSet")
    schema = data.get("schema_version", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise ContractSchemaError(f"unsupported schema_version: {schema!r}")
    candidates_raw = data.get("candidates")
    if not isinstance(candidates_raw, Sequence) or isinstance(candidates_raw, (str, bytes)):
        raise ContractSchemaError("candidates must be an array")
    candidates = tuple(
        _candidate_from_dict(_require_mapping(item, f"candidates[{i}]"), f"candidates[{i}]")
        for i, item in enumerate(candidates_raw)
    )
    return PolicyCandidateSet(
        run_id=_require_str(data.get("run_id"), "run_id"),
        frame_id=_require_str(data.get("frame_id"), "frame_id"),
        scenario_id=_require_str(data.get("scenario_id"), "scenario_id"),
        model_id=_require_str(data.get("model_id"), "model_id", allow_empty=True),
        carla_frame=_require_int(data.get("carla_frame", 0), "carla_frame", minimum=0),
        simulation_time_s=_require_number(data.get("simulation_time_s"), "simulation_time_s"),
        wall_time_s=_require_number(data.get("wall_time_s"), "wall_time_s"),
        candidates=candidates,
        schema_version=str(schema),
        coordinate_frame=str(data.get("coordinate_frame", "map") or "map"),
    )


def point_to_dict(point: TrajectoryPoint) -> dict[str, float]:
    return {
        "t": point.t,
        "x": point.x,
        "y": point.y,
        "yaw": point.yaw,
        "kappa": point.kappa,
        "v": point.v,
        "a": point.a,
        "jerk": point.jerk,
    }


def candidate_to_dict(candidate: PolicyCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source.value,
        "generated_time_s": candidate.generated_time_s,
        "valid_until_s": candidate.valid_until_s,
        "probability": candidate.probability,
        "points": [point_to_dict(p) for p in candidate.points],
        "behavior": candidate.behavior,
        "critical_actor": candidate.critical_actor,
        "conflict_type": candidate.conflict_type,
        "risk_horizon_s": candidate.risk_horizon_s,
        "intended_action": candidate.intended_action,
        "uncertainty": candidate.uncertainty,
        "availability": candidate.availability,
        "dynamics_meta": dict(candidate.dynamics_meta),
    }


def candidate_set_to_dict(candidate_set: PolicyCandidateSet) -> dict[str, Any]:
    return {
        "schema_version": candidate_set.schema_version,
        "run_id": candidate_set.run_id,
        "frame_id": candidate_set.frame_id,
        "scenario_id": candidate_set.scenario_id,
        "model_id": candidate_set.model_id,
        "carla_frame": candidate_set.carla_frame,
        "simulation_time_s": candidate_set.simulation_time_s,
        "wall_time_s": candidate_set.wall_time_s,
        "coordinate_frame": candidate_set.coordinate_frame,
        "candidates": [candidate_to_dict(c) for c in candidate_set.candidates],
    }


def decision_to_dict(decision: SafetyDecision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "run_id": decision.run_id,
        "frame_id": decision.frame_id,
        "prefilter_candidate_ids": list(decision.prefilter_candidate_ids),
        "final_candidate_id": decision.final_candidate_id,
        "pre_repair_trajectory_id": decision.pre_repair_trajectory_id,
        "post_repair_trajectory_id": decision.post_repair_trajectory_id,
        "executed_trajectory_id": decision.executed_trajectory_id,
        "constraint_margins": [
            {
                "name": m.name,
                "margin": m.margin,
                "hard": m.hard,
                "first_violation_time_s": m.first_violation_time_s,
                "actor_id": m.actor_id,
                "rule_id": m.rule_id,
                "message": m.message,
            }
            for m in decision.constraint_margins
        ],
        "decision_kind": decision.decision_kind.value,
        "modification_norm": decision.modification_norm,
        "slack": decision.slack,
        "progress_loss": decision.progress_loss,
        "solver_status": decision.solver_status,
        "latency_ms": decision.latency_ms,
        "state_before": decision.state_before.value,
        "state_after": decision.state_after.value,
        "recovery_conditions": list(decision.recovery_conditions),
        "fallback_request": decision.fallback_request.to_dict() if decision.fallback_request else None,
        "reject_reasons": list(decision.reject_reasons),
        "learning_modules_required": decision.learning_modules_required,
    }


def event_to_dict(event: SafetyEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "frame_id": event.frame_id,
        "phase": event.phase.value,
        "decision": decision_to_dict(event.decision) if event.decision else None,
        "availability": event.availability.to_dict(),
        "privilege": event.privilege.value,
        "message": event.message,
        "simulation_time_s": event.simulation_time_s,
        "wall_time_s": event.wall_time_s,
    }


def margin_ok(margins: Sequence[ConstraintMargin]) -> bool:
    return all((not m.hard) or m.margin >= 0.0 for m in margins)


# Re-export enums used by serializers for type checkers.
__all_enums__ = (DecisionKind, EventPhase, FallbackTarget, SafetyMode)
