"""Shared curvature → longitudinal speed limits (tracker-faithful).

Used by ConstrainedVLAMPC and offline executability prefilters so they cannot
diverge on the definition of curve_limit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def curve_speed_limit_from_kappa_window(
    kappa_ref: np.ndarray | list[float],
    *,
    max_lat_accel_mps2: float = 1.0,
    max_speed_mps: float = 2.5,
    curve_limit_quantile: float = 0.90,
) -> tuple[float, float]:
    """Return (curve_limit_mps, kappa_quantile) matching ConstrainedVLAMPC.

    ``curve_limit = sqrt(a_lat / kappa_q)`` when kappa_q > 1e-4, else max_speed.
    """
    arr = np.asarray(kappa_ref, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(max_speed_mps), 0.0
    k_q = float(np.quantile(np.abs(arr), np.clip(curve_limit_quantile, 0.0, 1.0)))
    if k_q > 1e-4:
        return float(min(max_speed_mps, math.sqrt(max(float(max_lat_accel_mps2), 0.0) / k_q))), k_q
    return float(max_speed_mps), k_q


def prediction_horizon_distances(
    *,
    linearization_speed_mps: float,
    horizon: int = 20,
    prediction_dt_s: float = 0.10,
    min_linearization_speed_mps: float = 0.60,
) -> np.ndarray:
    v = max(float(linearization_speed_mps), float(min_linearization_speed_mps))
    return np.arange(1, int(horizon) + 1, dtype=float) * v * float(prediction_dt_s)


def sample_prediction_kappa_from_spatial_path(
    path: Any,
    *,
    progress_s: float,
    linearization_speed_mps: float,
    horizon: int = 20,
    prediction_dt_s: float = 0.10,
    min_linearization_speed_mps: float = 0.60,
) -> np.ndarray:
    """Sample densified path κ over the same look-ahead as ConstrainedVLAMPC._reference."""
    distances = prediction_horizon_distances(
        linearization_speed_mps=linearization_speed_mps,
        horizon=horizon,
        prediction_dt_s=prediction_dt_s,
        min_linearization_speed_mps=min_linearization_speed_mps,
    )
    _x, _y, _yaw, kappa = path.sample(float(progress_s) + distances)
    return np.asarray(kappa, dtype=float)


def path_end_horizon_speed_limit(
    *,
    path_length_m: float,
    progress_s: float,
    path_end_margin_m: float = 2.0,
    max_brake_mps2: float = 3.0,
) -> float:
    remaining = max(0.0, float(path_length_m) - float(progress_s) - float(path_end_margin_m))
    return float(math.sqrt(2.0 * max(float(max_brake_mps2), 0.0) * remaining))


def tracker_longitudinal_target_cap(
    *,
    path_target_speed_mps: float,
    curve_limit_mps: float,
    horizon_limit_mps: float,
    max_speed_mps: float = 2.5,
    freshness_limit_mps: float | None = None,
) -> float:
    """Unified min() used in ConstrainedVLAMPC._longitudinal (fresh path)."""
    caps = [
        float(path_target_speed_mps),
        float(max_speed_mps),
        float(curve_limit_mps),
        float(horizon_limit_mps),
    ]
    if freshness_limit_mps is not None:
        caps.append(float(freshness_limit_mps))
    return max(0.0, min(caps))
