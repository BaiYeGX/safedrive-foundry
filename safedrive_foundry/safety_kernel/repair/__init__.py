"""G2 trajectory repair: longitudinal QP, restricted RATO-SCP, Raw/Rule/HardReject."""

from __future__ import annotations

from safety_kernel.repair.corridor import has_legal_lateral_corridor, is_rato_eligible_hints
from safety_kernel.repair.interface import RepairInterface
from safety_kernel.repair.longitudinal_qp import LongitudinalQPRepair, is_longitudinally_repairable
from safety_kernel.repair.qp_solver import LongitudinalQPSolver, osqp_available, osqp_local_site
from safety_kernel.repair.rato_scp import RestrictedRatoScpRepair
from safety_kernel.repair.types import (
    RepairMetrics,
    RepairMode,
    RepairResult,
    SolverStatus,
    SolverTrace,
)

__all__ = [
    "LongitudinalQPRepair",
    "LongitudinalQPSolver",
    "RepairInterface",
    "RepairMetrics",
    "RepairMode",
    "RepairResult",
    "RestrictedRatoScpRepair",
    "SolverStatus",
    "SolverTrace",
    "has_legal_lateral_corridor",
    "is_longitudinally_repairable",
    "is_rato_eligible_hints",
    "osqp_available",
    "osqp_local_site",
]

