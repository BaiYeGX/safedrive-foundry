"""Contract schema identity and deterministic hash."""

from __future__ import annotations

import hashlib
import json

# Frozen schema identity for PolicyCandidateSet / SafetyDecision / SafetyEvent / FallbackRequest.
SCHEMA_VERSION = "safedrive.safety.contracts.v1"

# Field inventory used for hash stability (must update when contracts change).
_CONTRACT_FIELD_INVENTORY = {
    "schema_version": SCHEMA_VERSION,
    "PolicyCandidateSet": [
        "run_id",
        "frame_id",
        "scenario_id",
        "model_id",
        "carla_frame",
        "simulation_time_s",
        "wall_time_s",
        "schema_version",
        "candidates",
        "coordinate_frame",
    ],
    "PolicyCandidate": [
        "candidate_id",
        "source",
        "generated_time_s",
        "valid_until_s",
        "probability",
        "points",
        "behavior",
        "critical_actor",
        "conflict_type",
        "risk_horizon_s",
        "intended_action",
        "uncertainty",
        "availability",
        "dynamics_meta",
    ],
    "TrajectoryPoint": ["t", "x", "y", "yaw", "kappa", "v", "a", "jerk"],
    "SafetyDecision": [
        "decision_id",
        "run_id",
        "frame_id",
        "prefilter_candidate_ids",
        "final_candidate_id",
        "pre_repair_trajectory_id",
        "post_repair_trajectory_id",
        "executed_trajectory_id",
        "constraint_margins",
        "decision_kind",
        "modification_norm",
        "slack",
        "progress_loss",
        "solver_status",
        "latency_ms",
        "state_before",
        "state_after",
        "recovery_conditions",
        "fallback_request",
        "reject_reasons",
        "learning_modules_required",
    ],
    "SafetyEvent": [
        "event_id",
        "run_id",
        "frame_id",
        "phase",
        "decision",
        "availability",
        "privilege",
        "message",
        "simulation_time_s",
        "wall_time_s",
    ],
    "FallbackRequest": [
        "reason_code",
        "target",
        "from_state",
        "to_state",
        "urgency",
        "source_candidate_id",
    ],
    "ComponentAvailability": ["classic", "vla", "world", "safety", "detail"],
    "SafetyMode": ["NORMAL", "DEGRADED", "MINIMAL_RISK", "EMERGENCY"],
    "ObservableSnapshot": [
        "run_id",
        "frame_id",
        "scenario_id",
        "simulation_time_s",
        "wall_time_s",
        "ego_x",
        "ego_y",
        "ego_yaw",
        "ego_v",
        "ego_a",
        "observed_time_s",
        "actors",
        "privilege",
        "schema_version",
        "coordinate_frame",
    ],
    "TrackedObject": [
        "actor_id",
        "class_name",
        "x",
        "y",
        "yaw",
        "vx",
        "vy",
        "length_m",
        "width_m",
        "observed_time_s",
        "lost",
    ],
}


def contracts_schema_hash() -> str:
    payload = json.dumps(_CONTRACT_FIELD_INVENTORY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
