"""Frozen K/T/dt/horizon contract for V0/V1."""

from __future__ import annotations

from dataclasses import dataclass


DT_S = 0.25
T_STEPS = 10
HORIZON_S = 2.5  # T_STEPS * DT_S
V0_K = 1
V1_K = 2


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


V0_CONTRACT = TrajectoryContract(k=V0_K)
V1_CONTRACT = TrajectoryContract(k=V1_K)
