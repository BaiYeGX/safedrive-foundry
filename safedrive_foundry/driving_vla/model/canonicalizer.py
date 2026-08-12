"""Deterministic canonicalizer: path plus speed to a fixed H-route trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from driving_vla.adapter.policy_adapter import TrajectoryArray
from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS

CANONICALIZER_VERSION = "safedrive.trajectory_canonicalizer.v1"


@dataclass(frozen=True)
class UpstreamPathSpeed:
    """Native upstream outputs (SimLingo-style)."""

    # (N, 2) path points in ego or map frame (~1m spacing typically N=20)
    path_xy: tuple[tuple[float, float], ...]
    # (M,) speeds at 0.25s (M=10) or (M,1)
    speed_mps: tuple[float, ...]
    frame: str = "ego"  # ego|map


def cum_arclength(path: Sequence[tuple[float, float]]) -> list[float]:
    """Cumulative arc-length table for a polyline."""
    s = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        s.append(s[-1] + math.hypot(dx, dy))
    return s


# Backward-compatible private aliases
_cum_arclength = cum_arclength


def interp_xy(
    path: Sequence[tuple[float, float]], s_list: Sequence[float], s_query: float
) -> tuple[float, float, float]:
    """Return x,y,yaw at arc-length s_query (clamped to path ends)."""
    if len(path) == 0:
        return 0.0, 0.0, 0.0
    if len(path) == 1:
        return path[0][0], path[0][1], 0.0
    total = s_list[-1]
    if total < 1e-9:
        return path[0][0], path[0][1], 0.0
    sq = max(0.0, min(s_query, total))
    for i in range(1, len(s_list)):
        if s_list[i] >= sq - 1e-12:
            s0, s1 = s_list[i - 1], s_list[i]
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            seg = max(s1 - s0, 1e-12)
            r = (sq - s0) / seg
            x = x0 + r * (x1 - x0)
            y = y0 + r * (y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0)
            return x, y, yaw
    x0, y0 = path[-2]
    x1, y1 = path[-1]
    return x1, y1, math.atan2(y1 - y0, x1 - x0)


_interp_xy = interp_xy


def _speed_at(speeds: Sequence[float], t: float, dt: float = DT_S) -> float:
    """Interpolate non-negative speed samples using ``t / dt`` with clamping."""
    if not speeds:
        return 0.0
    # speeds[i] at time (i+1)*dt or i*dt — use i*dt for index i
    idx = t / dt
    i0 = int(math.floor(idx))
    i1 = min(i0 + 1, len(speeds) - 1)
    i0 = max(0, min(i0, len(speeds) - 1))
    if i0 == i1:
        return max(0.0, float(speeds[i0]))
    r = idx - i0
    return max(0.0, float(speeds[i0]) * (1 - r) + float(speeds[i1]) * r)


class TrajectoryCanonicalizer:
    """Arc-length interpolate path + integrate speed → fixed T/dt/horizon."""

    version = CANONICALIZER_VERSION

    def __init__(self, *, t_steps: int = T_STEPS, dt_s: float = DT_S, horizon_s: float = HORIZON_S) -> None:
        self.t_steps = t_steps
        self.dt_s = dt_s
        self.horizon_s = horizon_s
        if abs(t_steps * dt_s - horizon_s) > 1e-6 and abs((t_steps - 1) * dt_s - horizon_s) > 1e-6:
            # Allow either T*dt or (T-1)*dt conventions; we use times t=(i+1)*dt → last=horizon
            pass

    def canonicalize(
        self,
        upstream: UpstreamPathSpeed,
        *,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        origin_yaw: float = 0.0,
        to_map: bool = True,
    ) -> TrajectoryArray:
        path = list(upstream.path_xy)
        if len(path) < 2:
            # Degenerate: straight at mean speed
            v_mean = sum(upstream.speed_mps) / max(len(upstream.speed_mps), 1) if upstream.speed_mps else 3.0
            pts = []
            for i in range(self.t_steps):
                t = (i + 1) * self.dt_s
                # ego frame then map
                xe, ye = v_mean * t, 0.0
                x, y, yaw = self._ego_to_map(xe, ye, 0.0, origin_xy, origin_yaw) if to_map else (xe, ye, 0.0)
                pts.append((x, y, yaw, v_mean, 0.0, 0.0))
            return TrajectoryArray(points_xy_yaw_v_a_kappa=tuple(pts), candidate_id="tau0", probability=1.0)

        # Work in ego frame if path is ego; else treat path as map and skip transform
        s_list = _cum_arclength(path)
        pts: list[tuple[float, float, float, float, float, float]] = []
        s_pos = 0.0
        prev_v = _speed_at(upstream.speed_mps, self.dt_s)
        for i in range(self.t_steps):
            t = (i + 1) * self.dt_s  # 0.25 .. 2.5
            v = _speed_at(upstream.speed_mps, t)
            # integrate distance over this step
            s_pos += v * self.dt_s
            x_p, y_p, yaw_p = _interp_xy(path, s_list, s_pos)
            if upstream.frame == "ego" and to_map:
                x, y, yaw = self._ego_to_map(x_p, y_p, yaw_p, origin_xy, origin_yaw)
            else:
                x, y, yaw = x_p, y_p, yaw_p
            a = (v - prev_v) / self.dt_s
            # curvature approx from yaw change
            if i == 0:
                kappa = 0.0
            else:
                dyaw = yaw - pts[-1][2]
                while dyaw > math.pi:
                    dyaw -= 2 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2 * math.pi
                ds = max(v * self.dt_s, 1e-3)
                kappa = dyaw / ds
            pts.append((x, y, yaw, v, a, kappa))
            prev_v = v

        return TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(pts),
            probability=1.0,
            uncertainty=0.1,
            candidate_id="tau0",
            intended_action="nominal",
        )

    @staticmethod
    def _ego_to_map(
        xe: float,
        ye: float,
        yaw_e: float,
        origin_xy: tuple[float, float],
        origin_yaw: float,
    ) -> tuple[float, float, float]:
        c, s = math.cos(origin_yaw), math.sin(origin_yaw)
        x = origin_xy[0] + c * xe - s * ye
        y = origin_xy[1] + s * xe + c * ye
        yaw = origin_yaw + yaw_e
        return x, y, yaw
