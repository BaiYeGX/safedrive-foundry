"""Oracle-only offline evaluation helpers.

Runtime Safety must never call these paths for control decisions.
Oracle results may label/diagnose offline trajectories only.
"""

from __future__ import annotations

from dataclasses import dataclass

from safety_kernel.config import SafetyKernelConfig, load_safety_config
from safety_kernel.contracts.types import (
    ConstraintMargin,
    ObservableSnapshot,
    ObservationPrivilege,
    PolicyCandidate,
)
from safety_kernel.validator.checks import hard_violations, run_full_checks


class OracleUsedAtRuntimeError(RuntimeError):
    """Raised if Oracle evaluation is incorrectly wired into a runtime path."""


@dataclass(frozen=True)
class OracleEvalResult:
    privilege: ObservationPrivilege
    margins: tuple[ConstraintMargin, ...]
    hard_reject: bool
    reject_reasons: tuple[str, ...]
    note: str = "offline_oracle_upper_bound_only"


def evaluate_with_oracle(
    candidate: PolicyCandidate,
    oracle_obs: ObservableSnapshot,
    cfg: SafetyKernelConfig | None = None,
    *,
    now_s: float | None = None,
    allow_oracle: bool = True,
) -> OracleEvalResult:
    """Evaluate a candidate against an Oracle-privileged observation.

    Callers must keep this offline. Setting allow_oracle=False simulates a
    runtime guard that forbids privileged inputs.
    """
    if not allow_oracle:
        raise OracleUsedAtRuntimeError("oracle evaluation forbidden on runtime path")
    if oracle_obs.privilege is not ObservationPrivilege.ORACLE:
        # Still allow evaluation if oracle_fields present, but require explicit privilege.
        if not oracle_obs.oracle_fields:
            raise ValueError("oracle evaluation requires ObservationPrivilege.ORACLE or oracle_fields")

    config = cfg or load_safety_config()
    now = oracle_obs.simulation_time_s if now_s is None else now_s
    # Oracle offline path intentionally bypasses runtime privilege hard-reject by
    # evaluating dynamics/collision/etc. on a privilege-stripped view for geometry
    # and a separate privilege tag in the result.
    stripped = ObservableSnapshot(
        run_id=oracle_obs.run_id,
        frame_id=oracle_obs.frame_id,
        scenario_id=oracle_obs.scenario_id,
        simulation_time_s=oracle_obs.simulation_time_s,
        wall_time_s=oracle_obs.wall_time_s,
        ego_x=oracle_obs.ego_x,
        ego_y=oracle_obs.ego_y,
        ego_yaw=oracle_obs.ego_yaw,
        ego_v=oracle_obs.ego_v,
        ego_a=oracle_obs.ego_a,
        observed_time_s=oracle_obs.observed_time_s,
        freshness_s=oracle_obs.freshness_s,
        speed_limit_mps=oracle_obs.speed_limit_mps,
        actors=oracle_obs.actors,
        traffic_lights=oracle_obs.traffic_lights,
        corridor_centerline=oracle_obs.corridor_centerline,
        corridor_half_width_m=oracle_obs.corridor_half_width_m,
        privilege=ObservationPrivilege.OBSERVABLE,
        oracle_fields={},
    )
    margins = run_full_checks(candidate, stripped, config, now_s=now)
    viol = hard_violations(margins)
    return OracleEvalResult(
        privilege=ObservationPrivilege.ORACLE,
        margins=tuple(margins),
        hard_reject=bool(viol),
        reject_reasons=tuple(f"{v.name}:{v.message}" for v in viol),
    )
