"""Trajectory shape contract shared by the H-route candidate sources."""

from __future__ import annotations

from dataclasses import dataclass


DT_S = 0.25
T_STEPS = 10
HORIZON_S = 2.5  # T_STEPS * DT_S
NOMINAL_K = 1
HYBRID_K = 2


@dataclass(frozen=True)
class TrajectoryContract:
    k: int
    t_steps: int = T_STEPS
    dt_s: float = DT_S
    horizon_s: float = HORIZON_S

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.t_steps != T_STEPS or abs(self.dt_s - DT_S) > 1e-9:
            raise ValueError("first-version contract freezes T=10, dt=0.25")
        if abs(self.horizon_s - HORIZON_S) > 1e-9:
            raise ValueError("horizon must be 2.5s; no 2.5→3.0 extrapolation")


NOMINAL_CONTRACT = TrajectoryContract(k=NOMINAL_K)
HYBRID_CONTRACT = TrajectoryContract(k=HYBRID_K)
