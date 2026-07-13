"""Independent Safety Kernel (G2): contracts, validator, QP/RATO repair, state machine.

Learning modules cannot override hard constraints, slack caps, fallback authority,
or emergency actions. Runtime validation uses Observable inputs only.
"""

from __future__ import annotations

from safety_kernel.arbitration import (
    ArbitrationPipeline,
    ArbitrationRecord,
    DegradationReason,
    ShadowResult,
    SoftScore,
)
from safety_kernel.config import (
    ArbitrationConfig,
    SafetyKernelConfig,
    QpRepairConfig,
    RatoScpConfig,
    config_sha256,
    load_safety_config,
)
from safety_kernel.contracts import (
    SCHEMA_VERSION,
    ComponentAvailability,
    FallbackRequest,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyEvent,
    SafetyMode,
    TrajectoryPoint,
    contracts_schema_hash,
)
from safety_kernel.kernel import KernelTickResult, SafetyKernel
from safety_kernel.repair import (
    RepairInterface,
    RepairMode,
    RepairResult,
    RestrictedRatoScpRepair,
    SolverStatus,
    has_legal_lateral_corridor,
    is_longitudinally_repairable,
    is_rato_eligible_hints,
    osqp_available,
    osqp_local_site,
)
from safety_kernel.state_machine import SafetyStateMachine
from safety_kernel.validator import TrajectoryValidator, ValidationStage

__all__ = [
    "SCHEMA_VERSION",
    "ArbitrationConfig",
    "ArbitrationPipeline",
    "ArbitrationRecord",
    "ComponentAvailability",
    "DegradationReason",
    "FallbackRequest",
    "KernelTickResult",
    "ObservableSnapshot",
    "PolicyCandidate",
    "PolicyCandidateSet",
    "QpRepairConfig",
    "RatoScpConfig",
    "RepairInterface",
    "RepairMode",
    "RepairResult",
    "RestrictedRatoScpRepair",
    "SafetyDecision",
    "SafetyEvent",
    "SafetyKernel",
    "SafetyKernelConfig",
    "SafetyMode",
    "SafetyStateMachine",
    "ShadowResult",
    "SoftScore",
    "SolverStatus",
    "TrajectoryPoint",
    "TrajectoryValidator",
    "ValidationStage",
    "config_sha256",
    "contracts_schema_hash",
    "has_legal_lateral_corridor",
    "is_longitudinally_repairable",
    "is_rato_eligible_hints",
    "load_safety_config",
    "osqp_available",
    "osqp_local_site",
]
