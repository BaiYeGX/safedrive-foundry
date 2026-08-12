"""Convert SimLingo 2D speed waypoints to scalar m/s (official + series helpers)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def official_desired_speed_mps(
    speed_wps_xy: Sequence[Sequence[float]],
    *,
    carla_fps: float = 20.0,
    wp_dilation: float = 1.0,
    data_save_freq: float = 5.0,
) -> float:
    """Match agent_simlingo.control_pid desired_speed from pred_speed_wps.

    desired = ||wp[half_second-2] - wp[one_second-2]|| * 2
    with one_second = carla_fps // (wp_dilation * data_save_freq).
    For defaults (20, 1, 5): one_second=4, half_second=2 → ||wp[0]-wp[2]||*2.
    """
    pts = np.asarray(list(speed_wps_xy), dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    one_second = int(float(carla_fps) // max(float(wp_dilation) * float(data_save_freq), 1e-6))
    one_second = max(2, one_second)
    half_second = max(1, one_second // 2)
    i0 = max(0, half_second - 2)
    i1 = max(0, one_second - 2)
    i1 = min(i1, pts.shape[0] - 1)
    i0 = min(i0, i1)
    delta = pts[i1] - pts[i0]
    return float(max(0.0, math.hypot(float(delta[0]), float(delta[1])) * 2.0))


def speed_wps_2d_to_mps(
    speed_wps_xy: Sequence[Sequence[float]],
    *,
    dt_s: float = 0.25,
    n_out: int = 10,
) -> tuple[float, ...]:
    """Per-interval finite-difference speeds (legacy series for planners).

    Prefer :func:`official_desired_speed_mps` as the primary VLA cruise command.
    """
    pts = [(float(p[0]), float(p[1])) for p in speed_wps_xy]
    if len(pts) < 2:
        return tuple(0.0 for _ in range(n_out))
    speeds: list[float] = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        speeds.append(max(0.0, math.hypot(dx, dy) / max(dt_s, 1e-6)))
    if not speeds:
        speeds = [0.0]
    while len(speeds) < n_out:
        speeds.append(speeds[-1])
    return tuple(speeds[:n_out])


def speed_wps_to_planner_samples(
    speed_wps_xy: Sequence[Sequence[float]],
    *,
    use_official_scalar: bool = True,
) -> tuple[float, ...]:
    """Samples fed into VLASpeedPlanner.

    Official contract: a single scalar matching agent_simlingo desired_speed,
    repeated so the planner's median of first elements is that value.
    """
    if use_official_scalar:
        v = official_desired_speed_mps(speed_wps_xy)
        return (v, v, v, v, v)
    return speed_wps_2d_to_mps(speed_wps_xy, n_out=10)
