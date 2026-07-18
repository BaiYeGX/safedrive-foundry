"""Convert SimLingo 2D speed waypoints to scalar m/s series (dt=0.25s)."""

from __future__ import annotations

import math
from typing import Sequence


def speed_wps_2d_to_mps(
    speed_wps_xy: Sequence[Sequence[float]],
    *,
    dt_s: float = 0.25,
    n_out: int = 10,
) -> tuple[float, ...]:
    """speed_wps are cumsum ego-frame positions; finite differences / dt → speed."""
    pts = [(float(p[0]), float(p[1])) for p in speed_wps_xy]
    if len(pts) < 2:
        return tuple(0.0 for _ in range(n_out))
    speeds: list[float] = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        speeds.append(max(0.0, math.hypot(dx, dy) / max(dt_s, 1e-6)))
    # pad/truncate to n_out
    if not speeds:
        speeds = [0.0]
    while len(speeds) < n_out:
        speeds.append(speeds[-1])
    return tuple(speeds[:n_out])
