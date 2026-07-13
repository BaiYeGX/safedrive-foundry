"""G2-05 offline fault injection helpers."""

from __future__ import annotations

from safety_kernel.faults.matrix import (
    DEFAULT_MATRIX,
    FaultId,
    FaultSpec,
    apply_fault_to_candidate,
    apply_fault_to_obs,
    apply_fault_to_set,
    expected_action_holds,
)

__all__ = [
    "DEFAULT_MATRIX",
    "FaultId",
    "FaultSpec",
    "apply_fault_to_candidate",
    "apply_fault_to_obs",
    "apply_fault_to_set",
    "expected_action_holds",
]
