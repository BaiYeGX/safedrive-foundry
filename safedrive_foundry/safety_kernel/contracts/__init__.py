"""Frozen Safety contracts for G2+ (schema versioned, hashable)."""

from __future__ import annotations

from safety_kernel.contracts.schema import SCHEMA_VERSION, contracts_schema_hash
from safety_kernel.contracts.types import (
    CandidateSource,
    ComponentAvailability,
    ConstraintMargin,
    DecisionKind,
    EventPhase,
    FallbackRequest,
    FallbackTarget,
    ObservableSnapshot,
    ObservationPrivilege,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyEvent,
    SafetyMode,
    TrackedObject,
    TrafficLightObs,
    TrajectoryPoint,
)
from safety_kernel.contracts.serialize import (
    candidate_set_from_dict,
    candidate_set_to_dict,
    decision_to_dict,
    event_to_dict,
    validate_candidate_set_schema,
)

__all__ = [
    "SCHEMA_VERSION",
    "CandidateSource",
    "ComponentAvailability",
    "ConstraintMargin",
    "DecisionKind",
    "EventPhase",
    "FallbackRequest",
    "FallbackTarget",
    "ObservableSnapshot",
    "ObservationPrivilege",
    "PolicyCandidate",
    "PolicyCandidateSet",
    "SafetyDecision",
    "SafetyEvent",
    "SafetyMode",
    "TrackedObject",
    "TrafficLightObs",
    "TrajectoryPoint",
    "candidate_set_from_dict",
    "candidate_set_to_dict",
    "contracts_schema_hash",
    "decision_to_dict",
    "event_to_dict",
    "validate_candidate_set_schema",
]
