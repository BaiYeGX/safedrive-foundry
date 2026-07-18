"""Route/Ego constant-velocity along route baseline (G3-02)."""

from __future__ import annotations

import math
from typing import Sequence

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.schema.trajectory_contract import DT_S, T_STEPS


def _polyline_advance(
    route: Sequence[tuple[float, float]],
    x0: float,
    y0: float,
    dist: float,
) -> tuple[float, float, float]:
    """Walk `dist` meters along route from nearest point to (x0,y0). Returns x,y,yaw."""
    if len(route) < 2:
        return x0 + dist, y0, 0.0
    # Find nearest segment start
    best_i = 0
    best_d = float("inf")
    for i, (x, y) in enumerate(route):
        d = (x - x0) ** 2 + (y - y0) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    # Advance from best_i
    remaining = dist
    i = best_i
    cx, cy = route[i]
    yaw = 0.0
    while remaining > 1e-6 and i + 1 < len(route):
        nx, ny = route[i + 1]
        seg = math.hypot(nx - cx, ny - cy)
        if seg < 1e-9:
            i += 1
            cx, cy = nx, ny
            continue
        yaw = math.atan2(ny - cy, nx - cx)
        if remaining <= seg:
            r = remaining / seg
            return cx + r * (nx - cx), cy + r * (ny - cy), yaw
        remaining -= seg
        cx, cy = nx, ny
        i += 1
    return cx, cy, yaw


class RouteEgoBaseline:
    """Non-language K=1 baseline: follow route at current speed (clamped)."""

    model_id = "baseline_route_ego_v0"
    k = 1

    def __init__(self, *, v_min: float = 0.5, v_max: float = 12.0) -> None:
        self.v_min = v_min
        self.v_max = v_max

    def predict(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        v = max(self.v_min, min(self.v_max, float(obs.ego_v) if obs.ego_v > 0.1 else 3.0))
        route = obs.route_xy
        if len(route) < 2:
            # Straight in ego yaw
            yaw = obs.ego_yaw
            pts = []
            for i in range(T_STEPS):
                t = (i + 1) * DT_S  # 0.25 .. 2.5
                pts.append(
                    (
                        obs.ego_x + v * t * math.cos(yaw),
                        obs.ego_y + v * t * math.sin(yaw),
                        yaw,
                        v,
                        0.0,
                        0.0,
                    )
                )
            return [
                TrajectoryArray(
                    points_xy_yaw_v_a_kappa=tuple(pts),
                    probability=1.0,
                    uncertainty=0.2,
                    candidate_id="route_ego_0",
                    intended_action="nominal",
                )
            ]

        pts = []
        for i in range(T_STEPS):
            dist = v * ((i + 1) * DT_S)
            x, y, yaw = _polyline_advance(route, obs.ego_x, obs.ego_y, dist)
            pts.append((x, y, yaw, v, 0.0, 0.0))
        return [
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=tuple(pts),
                probability=1.0,
                uncertainty=0.15,
                candidate_id="route_ego_0",
                intended_action="nominal",
            )
        ]
