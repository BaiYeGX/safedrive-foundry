"""Joint executability checks for Spatial K2 proposals (offline + smoke).

Semantics (do NOT conflate):

- ``hard_max_abs_curvature=1.0``: PathManager/Guard **anomaly** hard reject.
  Not vehicle trackable κ.

- Densified PM Q90/local-max: from ``SpatialPath.kappa`` after densify.

- MPC steer κ: ``tan(δ_max)/L ≈ 0.253``.

- Lateral a_y layers:
  - ``conservative_static``: max_v² × max_κ (over-pessimistic diagnostic)
  - ``arc_aligned``: max_i(v_i²|κ_i|) with T10 XY **projected** onto committed path
  - ``ideal_pointwise_cap``: per-point sqrt(a/κ) — **tautological diagnostic only**
  - ``tracker_horizon_capped``: single curve_limit from prediction-window κ Q90
    (shared with ConstrainedVLAMPC via ``runtime.curve_limits``)

Prefilter status names:
  - ``PM_STEER_PREFILTER_PASS``: densified PM + steer-κ + PM accepted (not CARLA)
  - never claim tracker-faithful a_y from ideal pointwise caps
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from driving_vla.runtime.curve_limits import (
    curve_speed_limit_from_kappa_window,
    path_end_horizon_speed_limit,
    sample_prediction_kappa_from_spatial_path,
    tracker_longitudinal_target_cap,
)

DEFAULT_WHEELBASE_M = 2.70
DEFAULT_MAX_STEER_RAD = 0.60
DEFAULT_MAX_LAT_ACCEL_MPS2 = 1.00
DEFAULT_PM_SOFT_KAPPA = 0.20
DEFAULT_PM_HARD_KAPPA = 1.00
STABLE_LIVE_PM_SOFT_KAPPA = 0.30
DEFAULT_MPC_HORIZON = 20
DEFAULT_PRED_DT_S = 0.10
DEFAULT_MIN_LIN_V = 0.60
DEFAULT_MAX_SPEED_MPS = 2.50
DEFAULT_MAX_BRAKE = 3.00
DEFAULT_PATH_END_MARGIN = 2.00
DEFAULT_CURVE_Q = 0.90


def mpc_kinematic_kappa_max(
    *,
    wheelbase_m: float = DEFAULT_WHEELBASE_M,
    max_steer_rad: float = DEFAULT_MAX_STEER_RAD,
) -> float:
    L = max(float(wheelbase_m), 1e-6)
    return float(math.tan(float(max_steer_rad)) / L)


def path_manager_committed_local_max_cap(soft_kappa: float) -> float:
    s = float(soft_kappa)
    return max(s * 2.5, s + 0.15)


def path_manager_q90_cap(soft_kappa: float) -> float:
    return float(soft_kappa) * 1.25


def lateral_accel_from_speed_kappa(speed_mps: float, kappa: float) -> float:
    v = max(0.0, float(speed_mps))
    return float(v * v * abs(float(kappa)))


def mpc_curve_speed_limit(
    kappa: float,
    *,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> float:
    """Ideal **pointwise** cap (diagnostic). Prefer tracker horizon Q90 API."""
    k = abs(float(kappa))
    if k <= 1e-4:
        return float(max_speed_mps)
    return float(min(max_speed_mps, math.sqrt(max(float(max_lat_accel_mps2), 0.0) / k)))


def densified_kappa_profile(
    s: np.ndarray,
    kappa: np.ndarray,
    *,
    margin_m: float = 0.20,
) -> dict[str, float]:
    s = np.asarray(s, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    if s.size == 0 or kappa.size == 0 or s.size != kappa.size:
        return {
            "n_points": 0.0,
            "length_m": 0.0,
            "max_abs_curvature": float("inf"),
            "curvature_q90": float("inf"),
            "mean_abs_curvature": 0.0,
            "source": "densified_empty",
        }
    length = float(s[-1]) if s.size else 0.0
    margin = max(0.05, float(margin_m))
    core = np.abs(kappa[(s >= margin) & (s <= max(margin, length - margin))])
    if core.size == 0:
        core = np.abs(kappa)
    return {
        "n_points": float(s.size),
        "length_m": length,
        "max_abs_curvature": float(np.max(core)) if core.size else 0.0,
        "curvature_q90": float(np.quantile(core, 0.90)) if core.size else 0.0,
        "mean_abs_curvature": float(np.mean(core)) if core.size else 0.0,
        "source": "densified_spatial_path",
    }


def profile_from_spatial_path(path: Any, *, margin_m: float = 0.20) -> dict[str, float]:
    if path is None:
        return densified_kappa_profile(np.zeros(0), np.zeros(0), margin_m=margin_m)
    s = getattr(path, "s", None)
    k = getattr(path, "kappa", None)
    if s is None or k is None:
        return densified_kappa_profile(np.zeros(0), np.zeros(0), margin_m=margin_m)
    return densified_kappa_profile(np.asarray(s), np.asarray(k), margin_m=margin_m)


def polyline_arclength_and_kappa(
    path_xy: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray([(float(x), float(y)) for x, y in path_xy], dtype=float)
    if pts.shape[0] < 2:
        return np.zeros(0), np.zeros(0)
    dxy = np.diff(pts, axis=0)
    ds = np.hypot(dxy[:, 0], dxy[:, 1])
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if pts.shape[0] < 3:
        return s, np.zeros(pts.shape[0])
    yaw = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
    k_seg = np.zeros(len(yaw))
    for i in range(1, len(yaw)):
        dpsi = float(yaw[i] - yaw[i - 1])
        dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi
        k_seg[i] = abs(dpsi) / max(float(ds[i - 1]), 1e-6)
    k_v = np.zeros(pts.shape[0])
    k_v[0] = k_seg[0] if len(k_seg) else 0.0
    k_v[-1] = k_seg[-1] if len(k_seg) else 0.0
    for i in range(1, pts.shape[0] - 1):
        k_v[i] = 0.5 * (k_seg[i - 1] + k_seg[i]) if i < len(k_seg) else k_seg[i - 1]
    return s, k_v


def sample_speed_along_arclength(
    s_query: np.ndarray,
    *,
    speed_samples_mps: Sequence[float],
    path_length_m: float,
    ego_v: float,
) -> np.ndarray:
    """Legacy uniform index→arc map (synthetic). Prefer T10 projection API."""
    if s_query.size == 0:
        return np.zeros(0)
    speeds = [max(0.0, float(v)) for v in speed_samples_mps] if speed_samples_mps else [float(ego_v)]
    if len(speeds) == 1:
        return np.full_like(s_query, speeds[0], dtype=float)
    L = max(float(path_length_m), 1e-6)
    s_spd = np.linspace(0.0, L, len(speeds))
    return np.interp(np.clip(s_query, 0.0, L), s_spd, np.asarray(speeds, dtype=float))


def project_t10_speeds_onto_spatial_path(
    spatial_path: Any,
    points_xy_yaw_v: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project each T10 (x,y,v) onto densified path → (s_i, v_i, kappa(s_i)).

    This is the **real** spatial alignment (not uniform arclength stretch of T10).
    """
    if spatial_path is None or not points_xy_yaw_v:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    s_list: list[float] = []
    v_list: list[float] = []
    k_list: list[float] = []
    hint = 0.0
    for row in points_xy_yaw_v:
        if len(row) < 4:
            continue
        x, y, _yaw, v = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        si = float(spatial_path.project_s(x, y, hint_s=hint))
        hint = si
        _x, _y, _yaw, kk = spatial_path.sample(si)
        s_list.append(si)
        v_list.append(max(0.0, v))
        k_list.append(abs(float(kk[0])))
    return (
        np.asarray(s_list, dtype=float),
        np.asarray(v_list, dtype=float),
        np.asarray(k_list, dtype=float),
    )


def max_arc_aligned_lat_accel(
    s: np.ndarray,
    kappa: np.ndarray,
    speed: np.ndarray,
) -> float:
    if s.size == 0 or kappa.size == 0 or speed.size == 0:
        return 0.0
    n = min(s.size, kappa.size, speed.size)
    v = np.maximum(speed[:n], 0.0)
    k = np.abs(kappa[:n])
    return float(np.max(v * v * k)) if n else 0.0


def ideal_pointwise_capped_ay(
    kappa: np.ndarray,
    speed: np.ndarray,
    *,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> float:
    """Per-point min(v, sqrt(a/κ)) then max a_y — **tautological** if cap applied fully.

    Kept only as IDEAL_POINTWISE_CAP_DIAGNOSTIC. Do not use as live gate.
    """
    if kappa.size == 0 or speed.size == 0:
        return 0.0
    n = min(kappa.size, speed.size)
    ays = []
    for i in range(n):
        k = abs(float(kappa[i]))
        v = min(max(0.0, float(speed[i])), mpc_curve_speed_limit(k, max_lat_accel_mps2=max_lat_accel_mps2, max_speed_mps=max_speed_mps))
        ays.append(v * v * k)
    return float(max(ays)) if ays else 0.0


def tracker_horizon_curve_limit_on_path(
    spatial_path: Any,
    *,
    ego_v: float,
    progress_s: float = 0.0,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    curve_limit_quantile: float = DEFAULT_CURVE_Q,
    horizon: int = DEFAULT_MPC_HORIZON,
    prediction_dt_s: float = DEFAULT_PRED_DT_S,
    min_linearization_speed_mps: float = DEFAULT_MIN_LIN_V,
) -> tuple[float, float, np.ndarray]:
    """Return (curve_limit, kappa_q90, kappa_window) as ConstrainedVLAMPC does."""
    if spatial_path is None:
        return float(max_speed_mps), 0.0, np.zeros(0)
    # Linearization speed: ego, floored like tracker
    lin_v = max(float(ego_v), float(min_linearization_speed_mps))
    # Also allow looking slightly above measured speed (tracker slack not always applied
    # to longitudinal; use ego for faithful default).
    kappa_win = sample_prediction_kappa_from_spatial_path(
        spatial_path,
        progress_s=float(progress_s),
        linearization_speed_mps=lin_v,
        horizon=horizon,
        prediction_dt_s=prediction_dt_s,
        min_linearization_speed_mps=min_linearization_speed_mps,
    )
    curve_lim, k_q = curve_speed_limit_from_kappa_window(
        kappa_win,
        max_lat_accel_mps2=max_lat_accel_mps2,
        max_speed_mps=max_speed_mps,
        curve_limit_quantile=curve_limit_quantile,
    )
    return curve_lim, k_q, kappa_win


def tracker_faithful_capped_speeds_and_ay(
    spatial_path: Any,
    s_t10: np.ndarray,
    v_t10: np.ndarray,
    kappa_at_t10: np.ndarray,
    *,
    ego_v: float,
    path_target_speed_mps: float,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    max_brake_mps2: float = DEFAULT_MAX_BRAKE,
    path_end_margin_m: float = DEFAULT_PATH_END_MARGIN,
    curve_limit_quantile: float = DEFAULT_CURVE_Q,
    horizon: int = DEFAULT_MPC_HORIZON,
    prediction_dt_s: float = DEFAULT_PRED_DT_S,
    min_linearization_speed_mps: float = DEFAULT_MIN_LIN_V,
) -> dict[str, float | bool]:
    """Apply **one** tracker curve_limit (horizon Q90) + path-end limit to T10 speeds.

    Then compute max_i(v_cap_i² |κ_i|) at projected T10 samples.
    This is NOT tautological: a single limit from window Q90 can leave high-κ
    points with a_y > a_max if those points are outside the Q90 mass.
    """
    if spatial_path is None or s_t10.size == 0:
        return {
            "curve_limit_mps": float(max_speed_mps),
            "horizon_limit_mps": float(max_speed_mps),
            "tracker_target_cap_mps": float(max_speed_mps),
            "kappa_q90_horizon": 0.0,
            "tracker_capped_ay": 0.0,
            "pass_tracker_capped_ay": True,
        }
    progress0 = float(s_t10[0]) if s_t10.size else 0.0
    curve_lim, k_q, _win = tracker_horizon_curve_limit_on_path(
        spatial_path,
        ego_v=ego_v,
        progress_s=progress0,
        max_lat_accel_mps2=max_lat_accel_mps2,
        max_speed_mps=max_speed_mps,
        curve_limit_quantile=curve_limit_quantile,
        horizon=horizon,
        prediction_dt_s=prediction_dt_s,
        min_linearization_speed_mps=min_linearization_speed_mps,
    )
    length = float(getattr(spatial_path, "length_m", 0.0) or 0.0)
    hor_lim = path_end_horizon_speed_limit(
        path_length_m=length,
        progress_s=progress0,
        path_end_margin_m=path_end_margin_m,
        max_brake_mps2=max_brake_mps2,
    )
    unified_cap = tracker_longitudinal_target_cap(
        path_target_speed_mps=path_target_speed_mps,
        curve_limit_mps=curve_lim,
        horizon_limit_mps=hor_lim,
        max_speed_mps=max_speed_mps,
        freshness_limit_mps=None,  # offline smoke assumes fresh path
    )
    v_cap = np.minimum(np.maximum(v_t10, 0.0), unified_cap)
    ay = max_arc_aligned_lat_accel(s_t10, kappa_at_t10, v_cap)
    return {
        "curve_limit_mps": float(curve_lim),
        "horizon_limit_mps": float(hor_lim),
        "tracker_target_cap_mps": float(unified_cap),
        "kappa_q90_horizon": float(k_q),
        "tracker_capped_ay": float(ay),
        "pass_tracker_capped_ay": bool(ay <= max_lat_accel_mps2 + 1e-9),
    }


@dataclass(frozen=True)
class LayeredExecutabilityReport:
    densified_source: str
    densified_max_abs_curvature: float
    densified_curvature_q90: float
    densified_length_m: float
    pm_soft_kappa: float
    pm_hard_kappa: float
    pm_q90_cap: float
    pm_local_max_cap: float
    pass_anomaly_hard: bool
    pass_pm_q90_densified: bool
    pass_pm_local_max_densified: bool
    pass_pm_accepted: bool
    mpc_kappa_max: float
    pass_mpc_steer_kappa: bool
    ref_speed_max_mps: float
    # a_y layers
    conservative_static_ay: float
    arc_aligned_ay: float
    ideal_pointwise_cap_ay: float
    tracker_horizon_capped_ay: float
    max_lat_accel_mps2: float
    pass_conservative_static_ay: bool
    pass_arc_aligned_ay: bool
    # ideal pointwise is diagnostic only — always near-true
    ideal_pointwise_cap_diagnostic: bool
    pass_tracker_horizon_capped_ay: bool
    tracker_curve_limit_mps: float
    tracker_kappa_q90_horizon: float
    # aggregates
    pass_proposal_speed_feasibility: bool  # arc-aligned + steer
    pass_tracker_longitudinal_feasibility: bool  # tracker horizon cap + densified PM + steer
    pass_committed_densified_feasibility: bool
    conservative_static_live_prep: bool
    # Honest name: PM + steer (+ accepted); NOT full tracker a_y claim alone
    pm_steer_prefilter: bool
    # Full offline prefilter including tracker-faithful a_y
    live_prefilter: bool
    t10_projection_used: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # aliases for smoke report clarity
        d["mpc_capped_label"] = "IDEAL_POINTWISE_CAP_DIAGNOSTIC"
        d["mpc_capped_ay_alias_ideal_pointwise"] = self.ideal_pointwise_cap_ay
        return d


def evaluate_branch_executability(
    *,
    path_xy: Sequence[tuple[float, float]],
    speed_samples_mps: Sequence[float],
    ego_v: float,
    path_manager_accepted: bool,
    raw_spatial_path: Any | None = None,
    committed_spatial_path: Any | None = None,
    t10_points_xy_yaw_v: Sequence[Sequence[float]] | None = None,
    pm_soft_kappa: float = DEFAULT_PM_SOFT_KAPPA,
    pm_hard_kappa: float = DEFAULT_PM_HARD_KAPPA,
    wheelbase_m: float = DEFAULT_WHEELBASE_M,
    max_steer_rad: float = DEFAULT_MAX_STEER_RAD,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    max_brake_mps2: float = DEFAULT_MAX_BRAKE,
    path_end_margin_m: float = DEFAULT_PATH_END_MARGIN,
    curve_limit_quantile: float = DEFAULT_CURVE_Q,
    horizon: int = DEFAULT_MPC_HORIZON,
    prediction_dt_s: float = DEFAULT_PRED_DT_S,
    min_linearization_speed_mps: float = DEFAULT_MIN_LIN_V,
) -> LayeredExecutabilityReport:
    dens = profile_from_spatial_path(committed_spatial_path)
    if dens["n_points"] <= 0:
        dens = profile_from_spatial_path(raw_spatial_path)
    dens_source = str(dens.get("source", "unknown"))
    sp = committed_spatial_path if committed_spatial_path is not None else raw_spatial_path

    if dens["n_points"] <= 0:
        s_poly, k_poly = polyline_arclength_and_kappa(path_xy)
        dens = {
            "n_points": float(len(s_poly)),
            "length_m": float(s_poly[-1]) if s_poly.size else 0.0,
            "max_abs_curvature": float(np.max(np.abs(k_poly))) if k_poly.size else 0.0,
            "curvature_q90": float(np.quantile(np.abs(k_poly), 0.90)) if k_poly.size else 0.0,
            "mean_abs_curvature": float(np.mean(np.abs(k_poly))) if k_poly.size else 0.0,
            "source": "pre_densify_polyline_proxy",
        }
        dens_source = "pre_densify_polyline_proxy"
        sp = None

    k_max = float(dens["max_abs_curvature"])
    k_q90 = float(dens["curvature_q90"])
    length = float(dens["length_m"])
    q_cap = path_manager_q90_cap(pm_soft_kappa)
    local_cap = path_manager_committed_local_max_cap(pm_soft_kappa)
    k_mpc = mpc_kinematic_kappa_max(wheelbase_m=wheelbase_m, max_steer_rad=max_steer_rad)

    speeds = [max(0.0, float(v)) for v in (speed_samples_mps or [ego_v])]
    v_max = max([float(ego_v)] + speeds)
    path_target = float(getattr(sp, "target_speed_mps", v_max) if sp is not None else v_max)

    # --- T10 projection arc alignment (preferred) ---
    t10_used = False
    if sp is not None and t10_points_xy_yaw_v:
        s_t, v_t, k_t = project_t10_speeds_onto_spatial_path(sp, t10_points_xy_yaw_v)
        if s_t.size > 0:
            t10_used = True
            arc_ay = max_arc_aligned_lat_accel(s_t, k_t, v_t)
            ideal_ay = ideal_pointwise_capped_ay(
                k_t, v_t, max_lat_accel_mps2=max_lat_accel_mps2, max_speed_mps=max_speed_mps
            )
            tr = tracker_faithful_capped_speeds_and_ay(
                sp,
                s_t,
                v_t,
                k_t,
                ego_v=float(ego_v),
                path_target_speed_mps=path_target,
                max_lat_accel_mps2=max_lat_accel_mps2,
                max_speed_mps=max_speed_mps,
                max_brake_mps2=max_brake_mps2,
                path_end_margin_m=path_end_margin_m,
                curve_limit_quantile=curve_limit_quantile,
                horizon=horizon,
                prediction_dt_s=prediction_dt_s,
                min_linearization_speed_mps=min_linearization_speed_mps,
            )
        else:
            t10_used = False
            arc_ay = 0.0
            ideal_ay = 0.0
            tr = {
                "curve_limit_mps": max_speed_mps,
                "tracker_capped_ay": 0.0,
                "pass_tracker_capped_ay": True,
                "kappa_q90_horizon": 0.0,
            }
    else:
        # Fallback: densified s with uniform planner samples (legacy, flagged)
        if sp is not None:
            s_for = np.asarray(sp.s, dtype=float)
            k_for = np.abs(np.asarray(sp.kappa, dtype=float))
            v_along = sample_speed_along_arclength(
                s_for, speed_samples_mps=speeds, path_length_m=max(length, 1e-6), ego_v=float(ego_v)
            )
            arc_ay = max_arc_aligned_lat_accel(s_for, k_for, v_along)
            ideal_ay = ideal_pointwise_capped_ay(
                k_for, v_along, max_lat_accel_mps2=max_lat_accel_mps2, max_speed_mps=max_speed_mps
            )
            # synthesize pseudo T10 from densified samples for tracker layer
            n = min(10, s_for.size)
            idx = np.linspace(0, max(s_for.size - 1, 0), n).astype(int) if s_for.size else np.zeros(0, dtype=int)
            s_t = s_for[idx] if s_for.size else np.zeros(0)
            v_t = v_along[idx] if v_along.size else np.zeros(0)
            k_t = k_for[idx] if k_for.size else np.zeros(0)
            tr = tracker_faithful_capped_speeds_and_ay(
                sp,
                s_t,
                v_t,
                k_t,
                ego_v=float(ego_v),
                path_target_speed_mps=path_target,
                max_lat_accel_mps2=max_lat_accel_mps2,
                max_speed_mps=max_speed_mps,
                max_brake_mps2=max_brake_mps2,
                path_end_margin_m=path_end_margin_m,
                curve_limit_quantile=curve_limit_quantile,
                horizon=horizon,
                prediction_dt_s=prediction_dt_s,
                min_linearization_speed_mps=min_linearization_speed_mps,
            )
        else:
            arc_ay = lateral_accel_from_speed_kappa(v_max, k_max)
            ideal_ay = 0.0
            tr = {
                "curve_limit_mps": max_speed_mps,
                "tracker_capped_ay": arc_ay,
                "pass_tracker_capped_ay": arc_ay <= max_lat_accel_mps2 + 1e-9,
                "kappa_q90_horizon": k_q90,
            }

    cons_ay = lateral_accel_from_speed_kappa(v_max, k_max)
    pass_hard = k_max <= pm_hard_kappa + 1e-9
    pass_q90 = k_q90 <= q_cap + 1e-9
    pass_local = k_max <= local_cap + 1e-9
    pass_mpc_k = k_max <= k_mpc + 1e-9
    pass_cons = cons_ay <= max_lat_accel_mps2 + 1e-9
    pass_arc = arc_ay <= max_lat_accel_mps2 + 1e-9
    pass_tracker_ay = bool(tr["pass_tracker_capped_ay"])
    accepted = bool(path_manager_accepted)

    pass_proposal = pass_mpc_k and pass_arc
    pass_tracker_long = (
        pass_hard and pass_q90 and pass_local and pass_mpc_k and pass_tracker_ay
    )
    pass_committed = accepted and pass_q90 and pass_local and pass_hard and pass_mpc_k
    cons_live = pass_hard and pass_q90 and pass_local and pass_mpc_k and pass_cons
    # Honest: densified PM + steer + accepted (does NOT require a_y layers)
    pm_steer = accepted and pass_hard and pass_q90 and pass_local and pass_mpc_k
    # Full offline prefilter includes tracker-faithful a_y
    live_pf = pm_steer and pass_tracker_ay

    return LayeredExecutabilityReport(
        densified_source=dens_source,
        densified_max_abs_curvature=k_max,
        densified_curvature_q90=k_q90,
        densified_length_m=length,
        pm_soft_kappa=float(pm_soft_kappa),
        pm_hard_kappa=float(pm_hard_kappa),
        pm_q90_cap=q_cap,
        pm_local_max_cap=local_cap,
        pass_anomaly_hard=pass_hard,
        pass_pm_q90_densified=pass_q90,
        pass_pm_local_max_densified=pass_local,
        pass_pm_accepted=accepted,
        mpc_kappa_max=k_mpc,
        pass_mpc_steer_kappa=pass_mpc_k,
        ref_speed_max_mps=v_max,
        conservative_static_ay=cons_ay,
        arc_aligned_ay=float(arc_ay),
        ideal_pointwise_cap_ay=float(ideal_ay),
        tracker_horizon_capped_ay=float(tr["tracker_capped_ay"]),
        max_lat_accel_mps2=float(max_lat_accel_mps2),
        pass_conservative_static_ay=pass_cons,
        pass_arc_aligned_ay=pass_arc,
        ideal_pointwise_cap_diagnostic=True,  # always diagnostic; not a gate
        pass_tracker_horizon_capped_ay=pass_tracker_ay,
        tracker_curve_limit_mps=float(tr["curve_limit_mps"]),
        tracker_kappa_q90_horizon=float(tr.get("kappa_q90_horizon", 0.0)),
        pass_proposal_speed_feasibility=pass_proposal,
        pass_tracker_longitudinal_feasibility=pass_tracker_long,
        pass_committed_densified_feasibility=pass_committed,
        conservative_static_live_prep=cons_live,
        pm_steer_prefilter=pm_steer,
        live_prefilter=live_pf,
        t10_projection_used=t10_used,
    )


# Back-compat thin wrapper
@dataclass(frozen=True)
class ExecutabilityReport:
    max_abs_curvature: float
    curvature_q90: float
    length_m: float
    pm_soft_kappa: float
    pm_hard_kappa: float
    pm_q90_cap: float
    pm_local_max_cap: float
    mpc_kappa_max: float
    max_lat_accel_mps2: float
    ref_speed_mps: float
    max_lat_accel_implied: float
    pass_anomaly_hard: bool
    pass_pm_q90: bool
    pass_pm_local_max: bool
    pass_mpc_steer_kappa: bool
    pass_lat_accel: bool
    pass_live_prep: bool
    note: str = "legacy_wrapper"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_path_executability(
    path_xy: Sequence[tuple[float, float]],
    *,
    ref_speed_mps: float,
    pm_soft_kappa: float = DEFAULT_PM_SOFT_KAPPA,
    pm_hard_kappa: float = DEFAULT_PM_HARD_KAPPA,
    wheelbase_m: float = DEFAULT_WHEELBASE_M,
    max_steer_rad: float = DEFAULT_MAX_STEER_RAD,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
) -> ExecutabilityReport:
    layered = evaluate_branch_executability(
        path_xy=path_xy,
        speed_samples_mps=[ref_speed_mps],
        ego_v=ref_speed_mps,
        path_manager_accepted=True,
    )
    return ExecutabilityReport(
        max_abs_curvature=layered.densified_max_abs_curvature,
        curvature_q90=layered.densified_curvature_q90,
        length_m=layered.densified_length_m,
        pm_soft_kappa=layered.pm_soft_kappa,
        pm_hard_kappa=layered.pm_hard_kappa,
        pm_q90_cap=layered.pm_q90_cap,
        pm_local_max_cap=layered.pm_local_max_cap,
        mpc_kappa_max=layered.mpc_kappa_max,
        max_lat_accel_mps2=layered.max_lat_accel_mps2,
        ref_speed_mps=layered.ref_speed_max_mps,
        max_lat_accel_implied=layered.conservative_static_ay,
        pass_anomaly_hard=layered.pass_anomaly_hard,
        pass_pm_q90=layered.pass_pm_q90_densified,
        pass_pm_local_max=layered.pass_pm_local_max_densified,
        pass_mpc_steer_kappa=layered.pass_mpc_steer_kappa,
        pass_lat_accel=layered.pass_conservative_static_ay,
        pass_live_prep=layered.conservative_static_live_prep,
        note="legacy_conservative_static_on_pre_densify_proxy",
    )


# Meaningful speed for rollout coverage: below this, lat-accel gate is not informative.
DEFAULT_MEANINGFUL_SPEED_MPS = 0.75


def evaluate_mpc_rollout_executability(
    tick_records: Sequence[Mapping[str, Any]],
    *,
    max_lat_accel_mps2: float = DEFAULT_MAX_LAT_ACCEL_MPS2,
    meaningful_speed_mps: float = DEFAULT_MEANINGFUL_SPEED_MPS,
) -> dict[str, Any]:
    """Evaluate a real ConstrainedVLAMPC closed-loop tick trace (tracker-faithful).

    Each tick should provide (from VLAMPCCommand + ego state before integrate):
      speed_mps, reference_curvature, curve_speed_limit_mps, target_speed_mps,
      mode, solver_status

    Computes max_t (v_t² |κ_ref,t|) — NOT a static path_target near-zero trap.
    """
    if not tick_records:
        return {
            "n_ticks": 0,
            "max_rollout_ay": 0.0,
            "pass_rollout_ay": False,
            "n_mpc_solved": 0,
            "n_fallback": 0,
            "n_timeout": 0,
            "pass_all_mpc_solved": False,
            "pass_no_fallback": False,
            "max_speed_mps": 0.0,
            "mean_speed_mps": 0.0,
            "n_ticks_meaningful_speed": 0,
            "meaningful_speed_coverage": False,
            "pass_tracker_rollout": False,
            "note": "empty_rollout",
        }
    ays: list[float] = []
    speeds: list[float] = []
    n_solved = 0
    n_fallback = 0
    n_timeout = 0
    n_meaningful = 0
    for t in tick_records:
        v = max(0.0, float(t.get("speed_mps", 0.0)))
        k = abs(float(t.get("reference_curvature", 0.0)))
        ays.append(v * v * k)
        speeds.append(v)
        if v >= meaningful_speed_mps:
            n_meaningful += 1
        mode = str(t.get("mode", "")).lower()
        status = str(t.get("solver_status", "")).lower()
        if "fallback" in mode or mode == "bounded_fallback":
            n_fallback += 1
        if "solved" in status:
            n_solved += 1
        if "timeout" in status or "deadline" in status:
            n_timeout += 1
    max_ay = float(max(ays)) if ays else 0.0
    max_v = float(max(speeds)) if speeds else 0.0
    mean_v = float(sum(speeds) / len(speeds)) if speeds else 0.0
    n = len(tick_records)
    pass_ay = max_ay <= float(max_lat_accel_mps2) + 1e-9
    pass_solved = n_solved == n and n_timeout == 0
    pass_no_fb = n_fallback == 0
    coverage = n_meaningful >= max(1, int(math.ceil(0.25 * n)))
    # Full tracker rollout gate requires solved + no fallback + a_y + coverage
    pass_full = pass_ay and pass_solved and pass_no_fb and coverage
    return {
        "n_ticks": n,
        "max_rollout_ay": max_ay,
        "pass_rollout_ay": pass_ay,
        "n_mpc_solved": n_solved,
        "n_fallback": n_fallback,
        "n_timeout": n_timeout,
        "pass_all_mpc_solved": pass_solved,
        "pass_no_fallback": pass_no_fb,
        "max_speed_mps": max_v,
        "mean_speed_mps": mean_v,
        "n_ticks_meaningful_speed": n_meaningful,
        "meaningful_speed_mps_threshold": float(meaningful_speed_mps),
        "meaningful_speed_coverage": coverage,
        "pass_tracker_rollout": pass_full,
        "note": (
            "max_t v_t^2|kappa_ref,t| from live MPC ticks; "
            "not static path_target near-zero cap"
        ),
    }


def semantics_dict() -> dict[str, Any]:
    k_mpc = mpc_kinematic_kappa_max()
    return {
        "hard_max_abs_curvature_1_0": {
            "meaning": "PathManager/Guard anomaly hard reject",
            "is_vehicle_trackable_limit": False,
            "value_m_inv": DEFAULT_PM_HARD_KAPPA,
        },
        "path_manager_default": {
            "soft_max_abs_curvature": DEFAULT_PM_SOFT_KAPPA,
            "q90_cap": path_manager_q90_cap(DEFAULT_PM_SOFT_KAPPA),
            "committed_local_max_cap": path_manager_committed_local_max_cap(
                DEFAULT_PM_SOFT_KAPPA
            ),
            "hard": DEFAULT_PM_HARD_KAPPA,
            "kappa_source": "densified SpatialPath.kappa (raw/committed)",
        },
        "mpc_bicycle": {
            "wheelbase_m": DEFAULT_WHEELBASE_M,
            "max_steer_rad": DEFAULT_MAX_STEER_RAD,
            "kappa_max_m_inv": k_mpc,
            "formula": "kappa = tan(delta)/L",
            "curve_limit_shared_module": "driving_vla.runtime.curve_limits",
            "curve_limit_formula": "sqrt(a_lat / kappa_q90_prediction_window)",
        },
        "lateral_accel_layers": {
            "conservative_static": "max_v^2 * max_kappa",
            "arc_aligned": "T10 (x,y,v) project_s onto committed path → max_i v_i^2|kappa_i|",
            "ideal_pointwise_cap": "IDEAL_POINTWISE_CAP_DIAGNOSTIC (near-tautological)",
            "tracker_horizon_capped": "single curve_limit from horizon Q90 + path-end limit",
        },
        "status_names": {
            "PM_STEER_PREFILTER_PASS": "densified PM + steer-kappa + accepted",
            "LIVE_PREFILTER_PASS": (
                "PM_STEER + tracker 30-tick rollout max_t(v^2|kappa_ref|) "
                "+ all MPC solved + no fallback + meaningful speed coverage"
            ),
            "tracker_horizon_static_path_target": "DEPRECATED_vacuous_when_target_near_zero",
        },
        "live_prefilter_only": True,
        "does_not_cover": [
            "steering rate",
            "steering acceleration",
            "initial lateral error",
            "solver status",
            "CARLA closed-loop outcome",
        ],
    }
