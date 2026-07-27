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


K2_SPEED_NORMALIZE_VERSION = "safedrive.k2_speed_normalize.v1"


def _representative_nonneg_scalar(speed_mps: Sequence[float]) -> float:
    """Pick a single non-negative cruise scalar from planner samples or series."""
    vals = [max(0.0, float(v)) for v in speed_mps if math.isfinite(float(v))]
    if not vals:
        return 0.0
    # Official path: five identical planner samples → all equal.
    if len(vals) >= 2 and max(vals) - min(vals) <= 1e-9:
        return float(vals[0])
    # Legacy / mixed series: robust near-field median (same spirit as planner).
    near = sorted(vals[: min(5, len(vals))])
    mid = len(near) // 2
    if len(near) % 2:
        return float(near[mid])
    return float(0.5 * (near[mid - 1] + near[mid]))


def normalize_k2_target_speed_profile(
    speed_mps: Sequence[float],
    *,
    t_steps: int = 10,
    mode: str = "official",
    version: str = K2_SPEED_NORMALIZE_VERSION,
) -> tuple[float, ...]:
    """Versioned T10 target profile for R1 K2 (not K1 planner samples).

    - official: expand a single desired-speed scalar to length ``t_steps``.
    - legacy: if ``len(speed_mps) >= t_steps`` finite samples, pad/truncate
      explicitly without skipping index 0; otherwise fall back to scalar expand.

    Does not change :func:`speed_wps_to_planner_samples` K1 semantics.
    """
    if version != K2_SPEED_NORMALIZE_VERSION:
        raise ValueError(f"unsupported k2 speed normalize version: {version}")
    if t_steps < 1:
        raise ValueError("t_steps must be >= 1")
    mode_l = str(mode).strip().lower()
    if mode_l not in {"official", "legacy", "auto"}:
        raise ValueError(f"unsupported k2 speed mode: {mode}")

    samples = [float(v) for v in speed_mps]
    if mode_l == "auto":
        if len(samples) >= t_steps and max(samples) - min(samples) > 1e-9:
            mode_l = "legacy"
        else:
            mode_l = "official"

    if mode_l == "legacy" and len(samples) >= t_steps:
        out = [max(0.0, float(v)) if math.isfinite(float(v)) else 0.0 for v in samples[:t_steps]]
        return tuple(out)

    scalar = _representative_nonneg_scalar(samples)
    return tuple(float(scalar) for _ in range(t_steps))


def planner_samples_from_cruise_scalar(
    cruise_mps: float,
    *,
    n: int = 5,
) -> tuple[float, ...]:
    """Repeat a cruise scalar for VLASpeedPlanner (K1-compatible sample shape)."""
    v = max(0.0, float(cruise_mps))
    return tuple(v for _ in range(max(1, int(n))))
