"""Policy adapter: H-route trajectory arrays to Safety candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.contracts.types import (
    CandidateSource,
    PolicyCandidate,
    PolicyCandidateSet,
    TrajectoryPoint,
)

from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS


@dataclass
class ObservationBundle:
    """Minimal observable bundle for baselines / VLA (no oracle)."""

    run_id: str
    frame_id: str
    scenario_id: str
    simulation_time_s: float
    wall_time_s: float = 0.0
    carla_frame: int = 0
    ego_x: float = 0.0
    ego_y: float = 0.0
    ego_yaw: float = 0.0
    ego_v: float = 0.0
    # Route as polyline in map frame [(x,y), ...]
    route_xy: tuple[tuple[float, float], ...] = ()
    # Optional front image bytes or path; baselines may ignore.
    front_rgb: Any = None
    # Low-dim history: list of (x,y,yaw,v) oldest to newest.
    ego_history: tuple[tuple[float, float, float, float], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryArray:
    """One candidate: shape (T, 6) columns x,y,yaw,v,a,kappa; times implicit."""

    points_xy_yaw_v_a_kappa: tuple[tuple[float, float, float, float, float, float], ...]
    probability: float = 1.0
    uncertainty: float = 0.1
    candidate_id: str = "cand0"
    behavior: str = "follow"
    intended_action: str = "nominal"

    @property
    def t_steps(self) -> int:
        return len(self.points_xy_yaw_v_a_kappa)


def validate_trajectory_array(arr: TrajectoryArray, *, require_t: int = T_STEPS) -> None:
    if arr.t_steps != require_t:
        raise ValueError(f"expected T={require_t}, got {arr.t_steps}")
    if not (0.0 <= arr.probability <= 1.0):
        raise ValueError("probability out of [0,1]")
    for i, row in enumerate(arr.points_xy_yaw_v_a_kappa):
        if len(row) != 6:
            raise ValueError(f"point {i} must have 6 fields")
        for j, v in enumerate(row):
            if not math.isfinite(float(v)):
                raise ValueError(f"non-finite value at point {i} field {j}")


def _to_points(arr: TrajectoryArray, *, t0: float = 0.0, dt: float = DT_S) -> tuple[TrajectoryPoint, ...]:
    """Stamp times as t0+(i+1)*dt → 0.25..2.5 for default contract (matches canonicalizer)."""
    pts: list[TrajectoryPoint] = []
    previous_a: float | None = None
    for i, (x, y, yaw, v, a, kappa) in enumerate(arr.points_xy_yaw_v_a_kappa):
        a_value = float(a)
        jerk = 0.0 if previous_a is None else (a_value - previous_a) / dt
        pts.append(
            TrajectoryPoint(
                t=t0 + (i + 1) * dt,
                x=float(x),
                y=float(y),
                yaw=float(yaw),
                kappa=float(kappa),
                v=float(v),
                a=a_value,
                jerk=jerk,
            )
        )
        previous_a = a_value
    return tuple(pts)


def arrays_to_candidate_set(
    arrays: Sequence[TrajectoryArray],
    obs: ObservationBundle,
    *,
    model_id: str,
    source: CandidateSource = CandidateSource.VLA_FAST,
    now_s: float | None = None,
    valid_for_s: float = 0.40,
    coordinate_frame: str = "map",
    dynamics_meta: Mapping[str, Any] | None = None,
) -> PolicyCandidateSet:
    """Convert K trajectory arrays into a Safety-ready PolicyCandidateSet."""
    if not arrays:
        raise ValueError("arrays must be non-empty")
    now = obs.simulation_time_s if now_s is None else now_s
    meta_base = dict(dynamics_meta or {})
    candidates: list[PolicyCandidate] = []
    for arr in arrays:
        validate_trajectory_array(arr)
        points = _to_points(arr, t0=0.0)
        if not points:
            raise ValueError("empty trajectory points")
        # Contract: last absolute time is HORIZON_S (2.5s) for T=10, dt=0.25.
        last_t = float(points[-1].t)
        if abs(last_t - HORIZON_S) > 1e-6 and len(points) == T_STEPS:
            raise ValueError(
                f"horizon contract violation: last_t={last_t} expected {HORIZON_S} "
                f"(use t=(i+1)*dt for T={T_STEPS}, dt={DT_S})"
            )
        candidates.append(
            PolicyCandidate(
                candidate_id=arr.candidate_id,
                source=source,
                generated_time_s=now,
                valid_until_s=now + valid_for_s,
                probability=float(arr.probability),
                points=points,
                behavior=arr.behavior,
                intended_action=arr.intended_action,
                uncertainty=float(arr.uncertainty),
                availability=True,
                # Span for Safety schema (min_horizon_s=2.0); absolute end matches HORIZON_S.
                risk_horizon_s=float(points[-1].t - points[0].t),
                dynamics_meta={
                    **meta_base,
                    "model_id": model_id,
                    "dt_s": DT_S,
                    "t_steps": T_STEPS,
                    "absolute_horizon_s": last_t,
                },
            )
        )
    return PolicyCandidateSet(
        run_id=obs.run_id,
        frame_id=obs.frame_id,
        scenario_id=obs.scenario_id,
        model_id=model_id,
        carla_frame=obs.carla_frame,
        simulation_time_s=obs.simulation_time_s,
        wall_time_s=obs.wall_time_s,
        candidates=tuple(candidates),
        schema_version=SCHEMA_VERSION,
        coordinate_frame=coordinate_frame,
    )
