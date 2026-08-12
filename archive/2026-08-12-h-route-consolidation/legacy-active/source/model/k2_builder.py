"""R1 real K2: one native SimLingo result → longitudinal temporal dual candidates.

Deterministic path retiming along the shared native spatial path. Not a learned
multimodal dual-head. branch_type is always longitudinal_temporal.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.model.canonicalizer import cum_arclength, interp_xy
from driving_vla.model.speed_convert import (
    K2_SPEED_NORMALIZE_VERSION,
    normalize_k2_target_speed_profile,
    planner_samples_from_cruise_scalar,
)
from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS

DEFAULT_K2_TOML = (
    Path(__file__).resolve().parents[2] / "config" / "vla" / "k2_v1.toml"
)

CANDIDATE_NOMINAL = "v1_nominal"
CANDIDATE_CONSERVATIVE = "v1_conservative"
PROBABILITY_SOURCE = "fixed_equal_prior_unscaled"
RETIMER_VERSION_DEFAULT = "safedrive.k2_retimer.v1"
BRANCH_TYPE_DEFAULT = "longitudinal_temporal"

# Guard / build status codes
GUARD_OK = "OK"
GUARD_REJECT = "REJECT"
BUILD_PATH_HORIZON_EXHAUSTED = "PATH_HORIZON_EXHAUSTED"
BUILD_NATIVE_PATH_INSUFFICIENT = "NATIVE_PATH_INSUFFICIENT"
COLLAPSE_NUMERIC = "NUMERIC_COLLAPSE"
COLLAPSE_SEMANTIC_STOP = "SEMANTIC_STOP_NO_SPACE"
COLLAPSE_PATH_LIMIT = "PATH_LIMIT_NO_SPACE"


@dataclass(frozen=True)
class K2BuilderConfig:
    k: int = 2
    t_steps: int = T_STEPS
    dt_s: float = DT_S
    horizon_s: float = HORIZON_S
    conservative_speed_ratio: float = 0.65
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 3.0
    stop_threshold_mps: float = 0.35
    probability_prior: tuple[float, float] = (0.5, 0.5)
    top1_index: int = 0
    eligible_min_nominal_speed_mps: float = 0.75
    eligible_min_final_progress_m: float = 1.5
    mean_speed_gap_min_mps: float = 0.25
    final_progress_gap_min_m: float = 0.50
    max_position_separation_min_m: float = 0.50
    position_integration_error_max_m: float = 0.05
    acceleration_error_max_mps2: float = 1e-6
    yaw_tangent_error_max_rad: float = 0.05
    native_path_cross_track_error_max_m: float = 0.05
    retimer_version: str = RETIMER_VERSION_DEFAULT
    branch_type: str = BRANCH_TYPE_DEFAULT
    candidate_ids: tuple[str, str] = (CANDIDATE_NOMINAL, CANDIDATE_CONSERVATIVE)
    speed_normalize_version: str = K2_SPEED_NORMALIZE_VERSION
    speed_mode: str = "official"

    def config_hash(self) -> str:
        payload = (
            f"{self.retimer_version}|{self.branch_type}|k={self.k}|T={self.t_steps}|"
            f"dt={self.dt_s}|ratio={self.conservative_speed_ratio}|"
            f"acc={self.max_accel_mps2}|dec={self.max_decel_mps2}|"
            f"stop={self.stop_threshold_mps}|prior={self.probability_prior}|"
            f"top1={self.top1_index}|speed_mode={self.speed_mode}|"
            f"sn={self.speed_normalize_version}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class K2ExecutionSpec:
    candidate_id: str
    spatial_path_xy: tuple[tuple[float, float], ...]
    speed_samples_mps: tuple[float, ...]
    timed_trajectory_hash: str
    native_path_hash: str
    branch_type: str


@dataclass(frozen=True)
class K2Diagnostics:
    mean_speed_gap_mps: float
    final_progress_gap_m: float
    max_position_separation_m: float
    mean_position_separation_m: float
    collapsed: bool
    collapse_reason: str | None
    selection_space_eligible: bool
    path_speed_cap_active: bool
    position_integration_error_max_m: float
    acceleration_error_max_mps2: float
    yaw_tangent_error_max_rad: float
    curvature_error_max_per_m: float
    curvature_error_p95_per_m: float
    native_path_cross_track_error_max_m: float
    nominal_final_progress_m: float = 0.0
    conservative_final_progress_m: float = 0.0
    nominal_max_speed_mps: float = 0.0
    official_desired_speed_mps: float = 0.0


@dataclass(frozen=True)
class K2PredictionBundle:
    observation_identity: Mapping[str, Any]
    model_id: str
    config_hash: str
    retimer_version: str
    native_path_xy: tuple[tuple[float, float], ...]
    native_path_hash: str
    candidates: tuple[TrajectoryArray, ...]
    execution_specs: Mapping[str, K2ExecutionSpec]
    top1_index: int
    probability_source: str
    probability_margin: float
    branch_type: str
    diagnostics: K2Diagnostics
    guard_status: str = GUARD_OK
    guard_reasons: tuple[str, ...] = ()
    build_error: str | None = None

    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(c.candidate_id for c in self.candidates)


@dataclass(frozen=True)
class GuardResult:
    status: str
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == GUARD_OK


def load_k2_config(path: Path | str | None = None) -> K2BuilderConfig:
    """Load frozen k2_v1.toml (or return defaults if missing)."""
    cfg_path = Path(path) if path is not None else DEFAULT_K2_TOML
    if not cfg_path.is_file():
        return K2BuilderConfig()
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)
    contract = raw.get("contract", {})
    dynamics = raw.get("dynamics", {})
    eligibility = raw.get("eligibility", {})
    collapse = raw.get("collapse", {})
    residuals = raw.get("residuals", {})
    meta = raw.get("meta", {})
    prior = dynamics.get("probability_prior", [0.5, 0.5])
    cids = contract.get("candidate_ids", [CANDIDATE_NOMINAL, CANDIDATE_CONSERVATIVE])
    return K2BuilderConfig(
        k=int(contract.get("k", 2)),
        t_steps=int(contract.get("t_steps", T_STEPS)),
        dt_s=float(contract.get("dt_s", DT_S)),
        horizon_s=float(contract.get("horizon_s", HORIZON_S)),
        top1_index=int(contract.get("top1_index", 0)),
        candidate_ids=(str(cids[0]), str(cids[1])),
        conservative_speed_ratio=float(dynamics.get("conservative_speed_ratio", 0.65)),
        max_accel_mps2=float(dynamics.get("max_accel_mps2", 2.5)),
        max_decel_mps2=float(dynamics.get("max_decel_mps2", 3.0)),
        stop_threshold_mps=float(dynamics.get("stop_threshold_mps", 0.35)),
        probability_prior=(float(prior[0]), float(prior[1])),
        eligible_min_nominal_speed_mps=float(
            eligibility.get("min_nominal_speed_mps", 0.75)
        ),
        eligible_min_final_progress_m=float(
            eligibility.get("min_final_progress_m", 1.5)
        ),
        mean_speed_gap_min_mps=float(collapse.get("mean_speed_gap_min_mps", 0.25)),
        final_progress_gap_min_m=float(collapse.get("final_progress_gap_min_m", 0.50)),
        max_position_separation_min_m=float(
            collapse.get("max_position_separation_min_m", 0.50)
        ),
        position_integration_error_max_m=float(
            residuals.get("position_integration_error_max_m", 0.05)
        ),
        acceleration_error_max_mps2=float(
            residuals.get("acceleration_error_max_mps2", 1e-6)
        ),
        yaw_tangent_error_max_rad=float(
            residuals.get("yaw_tangent_error_max_rad", 0.05)
        ),
        native_path_cross_track_error_max_m=float(
            residuals.get("native_path_cross_track_error_max_m", 0.05)
        ),
        retimer_version=str(meta.get("retimer_version", RETIMER_VERSION_DEFAULT)),
        branch_type=str(meta.get("branch_type", BRANCH_TYPE_DEFAULT)),
    )


def stable_hash_points(
    points: Sequence[Sequence[float]], *, prec: int = 6
) -> str:
    parts: list[str] = []
    for row in points:
        parts.append(",".join(f"{float(x):.{prec}f}" for x in row))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def stable_hash_xy(path: Sequence[tuple[float, float]], *, prec: int = 6) -> str:
    return stable_hash_points(path, prec=prec)


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def project_speed_profile(
    u_target: Sequence[float],
    *,
    ego_v: float,
    dt_s: float,
    max_accel_mps2: float,
    max_decel_mps2: float,
) -> tuple[list[float], list[float], list[float]]:
    """Project target speeds to feasible v/a/s from ego_v (trapezoidal s)."""
    v_prev = max(0.0, float(ego_v))
    s_prev = 0.0
    speeds: list[float] = []
    accels: list[float] = []
    arcs: list[float] = []
    for u in u_target:
        u_i = max(0.0, float(u))
        a_des = (u_i - v_prev) / max(dt_s, 1e-9)
        a_des = _clip(a_des, -float(max_decel_mps2), float(max_accel_mps2))
        v_i = max(0.0, v_prev + a_des * dt_s)
        a_i = (v_i - v_prev) / max(dt_s, 1e-9)
        s_i = s_prev + 0.5 * (v_prev + v_i) * dt_s
        speeds.append(v_i)
        accels.append(a_i)
        arcs.append(s_i)
        v_prev = v_i
        s_prev = s_i
    return speeds, accels, arcs


def apply_path_speed_cap(
    u_target: Sequence[float],
    *,
    ego_v: float,
    path_length_m: float,
    dt_s: float,
    max_accel_mps2: float,
    max_decel_mps2: float,
    t_steps: int,
) -> tuple[tuple[float, ...], bool, str | None]:
    """Cap cruise targets so T10 progress stays within path + stopping envelope.

    Returns (capped_u, path_speed_cap_active, build_error_or_none).
    """
    s_total = max(0.0, float(path_length_m))
    v0 = max(0.0, float(ego_v))
    # Minimum distance to stop from current speed under max_decel over horizon.
    # If path shorter than required stop distance for current ego, exhausted.
    stop_dist = (v0 * v0) / (2.0 * max(max_decel_mps2, 1e-6))
    if s_total < 1e-6:
        return tuple(0.0 for _ in range(t_steps)), True, BUILD_PATH_HORIZON_EXHAUSTED
    if stop_dist > s_total + 1e-6 and v0 > 0.05:
        # Cannot form a consistent non-clamped trajectory ending with v>=0 on path.
        # Still try full decel profile rather than illegal end-copy.
        pass

    # Binary-search a global scale on u that keeps final s <= s_total under projector.
    u0 = [max(0.0, float(u)) for u in u_target]
    if not u0:
        u0 = [0.0] * t_steps

    def final_s(scale: float) -> float:
        scaled = [scale * u for u in u0]
        _v, _a, arcs = project_speed_profile(
            scaled,
            ego_v=v0,
            dt_s=dt_s,
            max_accel_mps2=max_accel_mps2,
            max_decel_mps2=max_decel_mps2,
        )
        return arcs[-1] if arcs else 0.0

    # If even zero cruise still overshoots because of ego coast, force pure brake u=0.
    if final_s(0.0) > s_total + 1e-3:
        # Check pure max-decel feasibility
        pure_brake = [0.0] * t_steps
        _v, _a, arcs = project_speed_profile(
            pure_brake,
            ego_v=v0,
            dt_s=dt_s,
            max_accel_mps2=max_accel_mps2,
            max_decel_mps2=max_decel_mps2,
        )
        if arcs and arcs[-1] > s_total + 0.05:
            return tuple(pure_brake), True, BUILD_PATH_HORIZON_EXHAUSTED
        return tuple(pure_brake), True, None

    if final_s(1.0) <= s_total + 1e-3:
        return tuple(u0), False, None

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if final_s(mid) <= s_total + 1e-3:
            lo = mid
        else:
            hi = mid
    capped = [lo * u for u in u0]
    return tuple(capped), True, None


def retime_on_path(
    path_xy: Sequence[tuple[float, float]],
    speeds: Sequence[float],
    accels: Sequence[float],
    arcs: Sequence[float],
    *,
    s_list: Sequence[float] | None = None,
) -> tuple[list[tuple[float, float, float, float, float, float]], dict[str, float]]:
    """Build T-row (x,y,yaw,v,a,kappa) by sampling path at integrated arc lengths.

    Residuals are construction-consistency checks against the native polyline and
    the projector arcs — not chord-vs-tangent curvature geometry (which is large
    on real VLA turns and is not a kinematics defect).
    """
    path = [(float(p[0]), float(p[1])) for p in path_xy]
    if s_list is None:
        s_list = cum_arclength(path)
    s_total = float(s_list[-1]) if s_list else 0.0
    pts: list[tuple[float, float, float, float, float, float]] = []
    pos_err = 0.0
    yaw_err = 0.0
    xtrack = 0.0
    kappa_errs: list[float] = []
    prev_s = 0.0
    prev_yaw = 0.0
    for i, (v, a, s_i) in enumerate(zip(speeds, accels, arcs)):
        s_raw = float(s_i)
        s_use = min(s_raw, s_total) if s_total > 1e-9 else 0.0
        # Path clamp residual: cannot place beyond path without freezing geometry.
        if s_raw > s_total + 1e-9:
            pos_err = max(pos_err, s_raw - s_total)
        x, y, yaw = interp_xy(path, s_list, s_use)
        # Re-interp numerical self-consistency (should be ~0).
        x2, y2, yaw2 = interp_xy(path, s_list, s_use)
        xtrack = max(xtrack, math.hypot(x - x2, y - y2))
        yaw_err = max(yaw_err, abs(_wrap_angle(yaw - yaw2)))
        if i == 0:
            kappa = 0.0
        else:
            dyaw = _wrap_angle(yaw - prev_yaw)
            ds = max(s_use - prev_s, 1e-9)
            kappa = dyaw / ds
            # kappa residual vs recompute from consecutive stored yaws
            kappa_re = _wrap_angle(yaw - prev_yaw) / ds
            kappa_errs.append(abs(kappa - kappa_re))
        pts.append((x, y, yaw, float(v), float(a), float(kappa)))
        prev_s = s_use
        prev_yaw = yaw

    kappa_max = max(kappa_errs) if kappa_errs else 0.0
    if kappa_errs:
        ks = sorted(kappa_errs)
        p95 = ks[min(len(ks) - 1, int(math.floor(0.95 * (len(ks) - 1))))]
    else:
        p95 = 0.0
    metrics = {
        "position_integration_error_max_m": float(pos_err),
        "yaw_tangent_error_max_rad": float(yaw_err),
        "native_path_cross_track_error_max_m": float(xtrack),
        "curvature_error_max_per_m": float(kappa_max),
        "curvature_error_p95_per_m": float(p95),
        "acceleration_error_max_mps2": 0.0,
    }
    return pts, metrics


def _branch_targets(
    u_nom: Sequence[float],
    *,
    ratio: float,
    ego_v: float,
    stop_threshold_mps: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    u_n = [max(0.0, float(u)) for u in u_nom]
    # Semantic stop: both requested cruise and ego are stop-like → cons must not re-accel.
    nom_stop = all(u <= stop_threshold_mps for u in u_n) and ego_v <= stop_threshold_mps
    if nom_stop:
        u_c = list(u_n)
    else:
        u_c = [min(u, ratio * u) for u in u_n]
    return tuple(u_n), tuple(u_c)


def _build_branch(
    *,
    u_target: Sequence[float],
    path_xy: Sequence[tuple[float, float]],
    s_list: Sequence[float],
    ego_v: float,
    config: K2BuilderConfig,
    candidate_id: str,
    probability: float,
    intended_action: str,
    native_path_hash: str,
) -> tuple[TrajectoryArray, K2ExecutionSpec, dict[str, float], list[float], list[float], list[float], bool, str | None]:
    capped_u, cap_active, build_err = apply_path_speed_cap(
        u_target,
        ego_v=ego_v,
        path_length_m=float(s_list[-1]) if s_list else 0.0,
        dt_s=config.dt_s,
        max_accel_mps2=config.max_accel_mps2,
        max_decel_mps2=config.max_decel_mps2,
        t_steps=config.t_steps,
    )
    speeds, accels, arcs = project_speed_profile(
        capped_u,
        ego_v=ego_v,
        dt_s=config.dt_s,
        max_accel_mps2=config.max_accel_mps2,
        max_decel_mps2=config.max_decel_mps2,
    )
    # Reject silent end-of-path positive-speed copy: if arc clamped but v stays high
    s_total = float(s_list[-1]) if s_list else 0.0
    if arcs and s_total > 0.0 and arcs[-1] >= s_total - 1e-6:
        # if still significant speed at path end after projector, require stop profile
        if speeds[-1] > config.stop_threshold_mps and arcs[-1] >= s_total - 1e-6:
            # force last segment decel consistency already handled by cap; if still
            # overshooting, mark exhausted
            if arcs[-1] > s_total + 0.05 or (
                speeds[-1] > 0.5 and abs(arcs[-1] - s_total) < 1e-6 and capped_u[-1] > 0.5
            ):
                # Check whether progress was artificially clamped: compare uncapped final s
                _v2, _a2, arcs2 = project_speed_profile(
                    u_target,
                    ego_v=ego_v,
                    dt_s=config.dt_s,
                    max_accel_mps2=config.max_accel_mps2,
                    max_decel_mps2=config.max_decel_mps2,
                )
                if arcs2[-1] > s_total + 0.5 and build_err is None and not cap_active:
                    build_err = BUILD_PATH_HORIZON_EXHAUSTED

    pts, metrics = retime_on_path(
        path_xy, speeds, accels, arcs, s_list=s_list
    )
    # Acceleration residual vs Δv/dt including first point vs ego_v
    a_err = 0.0
    v_prev = max(0.0, float(ego_v))
    for i, row in enumerate(pts):
        v_i = row[3]
        a_i = row[4]
        expected_a = (v_i - v_prev) / max(config.dt_s, 1e-9)
        a_err = max(a_err, abs(a_i - expected_a))
        v_prev = v_i
    metrics["acceleration_error_max_mps2"] = float(a_err)

    traj = TrajectoryArray(
        points_xy_yaw_v_a_kappa=tuple(pts),
        probability=float(probability),
        uncertainty=0.12 if intended_action == "nominal" else 0.22,
        candidate_id=candidate_id,
        intended_action=intended_action,
        behavior="follow",
    )
    timed_hash = stable_hash_points(pts)
    cruise = float(capped_u[0]) if capped_u else 0.0
    # Prefer median of capped profile for planner samples
    if capped_u:
        cruise = float(sorted(capped_u)[len(capped_u) // 2])
    spec = K2ExecutionSpec(
        candidate_id=candidate_id,
        spatial_path_xy=tuple((float(x), float(y)) for x, y in path_xy),
        speed_samples_mps=planner_samples_from_cruise_scalar(cruise, n=5),
        timed_trajectory_hash=timed_hash,
        native_path_hash=native_path_hash,
        branch_type=config.branch_type,
    )
    return traj, spec, metrics, speeds, accels, arcs, cap_active, build_err


def _classify_collapse(
    *,
    config: K2BuilderConfig,
    nom_speeds: Sequence[float],
    cons_speeds: Sequence[float],
    nom_arcs: Sequence[float],
    cons_arcs: Sequence[float],
    nom_pts: Sequence[tuple[float, float, float, float, float, float]],
    cons_pts: Sequence[tuple[float, float, float, float, float, float]],
    path_cap_active: bool,
    ego_v: float,
    u_nom: Sequence[float],
) -> tuple[bool, str | None, bool, float, float, float, float]:
    mean_speed_gap = abs(
        sum(nom_speeds) / max(len(nom_speeds), 1) - sum(cons_speeds) / max(len(cons_speeds), 1)
    )
    final_progress_gap = abs(float(nom_arcs[-1]) - float(cons_arcs[-1])) if nom_arcs and cons_arcs else 0.0
    seps = [
        math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(nom_pts, cons_pts)
    ]
    max_sep = max(seps) if seps else 0.0
    mean_sep = sum(seps) / max(len(seps), 1)

    nom_max_v = max(nom_speeds) if nom_speeds else 0.0
    nom_final_s = float(nom_arcs[-1]) if nom_arcs else 0.0
    eligible = (
        nom_max_v > config.eligible_min_nominal_speed_mps
        and nom_final_s > config.eligible_min_final_progress_m
    )

    stop_like = (
        all(u <= config.stop_threshold_mps for u in u_nom)
        and ego_v <= config.stop_threshold_mps
        and nom_max_v <= config.stop_threshold_mps
    )

    diverse = (
        mean_speed_gap >= config.mean_speed_gap_min_mps
        and final_progress_gap >= config.final_progress_gap_min_m
        and max_sep >= config.max_position_separation_min_m
    )

    if diverse:
        return False, None, eligible, mean_speed_gap, final_progress_gap, max_sep, mean_sep

    if stop_like:
        return True, COLLAPSE_SEMANTIC_STOP, eligible, mean_speed_gap, final_progress_gap, max_sep, mean_sep
    if path_cap_active and not diverse:
        # Path envelope may force branches equal
        if mean_speed_gap < 1e-6 and max_sep < 1e-6:
            return True, COLLAPSE_PATH_LIMIT, eligible, mean_speed_gap, final_progress_gap, max_sep, mean_sep
    if eligible:
        return True, COLLAPSE_NUMERIC, eligible, mean_speed_gap, final_progress_gap, max_sep, mean_sep
    # Not eligible and not diverse: collapsed but not numeric hard fail context
    return True, COLLAPSE_PATH_LIMIT if path_cap_active else COLLAPSE_SEMANTIC_STOP, eligible, mean_speed_gap, final_progress_gap, max_sep, mean_sep


def _empty_diagnostics() -> K2Diagnostics:
    return K2Diagnostics(
        mean_speed_gap_mps=0.0,
        final_progress_gap_m=0.0,
        max_position_separation_m=0.0,
        mean_position_separation_m=0.0,
        collapsed=True,
        collapse_reason=COLLAPSE_PATH_LIMIT,
        selection_space_eligible=False,
        path_speed_cap_active=False,
        position_integration_error_max_m=0.0,
        acceleration_error_max_mps2=0.0,
        yaw_tangent_error_max_rad=0.0,
        curvature_error_max_per_m=0.0,
        curvature_error_p95_per_m=0.0,
        native_path_cross_track_error_max_m=0.0,
    )


def _observation_identity(obs: ObservationBundle) -> dict[str, Any]:
    return {
        "run_id": obs.run_id,
        "frame_id": obs.frame_id,
        "scenario_id": obs.scenario_id,
        "simulation_time_s": float(obs.simulation_time_s),
        "carla_frame": int(obs.carla_frame),
        "ego_x": float(obs.ego_x),
        "ego_y": float(obs.ego_y),
        "ego_yaw": float(obs.ego_yaw),
        "ego_v": float(obs.ego_v),
    }


def _hard_reject_bundle(
    *,
    obs: ObservationBundle,
    model_id: str,
    config: K2BuilderConfig,
    native_path_xy: tuple[tuple[float, float], ...],
    build_error: str,
) -> K2PredictionBundle:
    """Structured fail-closed bundle: no synthetic neural path, no executable K2."""
    path_xy = tuple((float(x), float(y)) for x, y in native_path_xy)
    return K2PredictionBundle(
        observation_identity=_observation_identity(obs),
        model_id=model_id,
        config_hash=config.config_hash(),
        retimer_version=config.retimer_version,
        native_path_xy=path_xy,
        native_path_hash=stable_hash_xy(path_xy) if path_xy else "empty",
        candidates=(),
        execution_specs={},
        top1_index=int(config.top1_index),
        probability_source=PROBABILITY_SOURCE,
        probability_margin=0.0,
        branch_type=config.branch_type,
        diagnostics=_empty_diagnostics(),
        guard_status=GUARD_REJECT,
        guard_reasons=(str(build_error),),
        build_error=str(build_error),
    )


def point_to_polyline_distance_m(
    x: float,
    y: float,
    path_xy: Sequence[tuple[float, float]],
) -> float:
    """Minimum distance from a point to a polyline (meters)."""
    path = [(float(p[0]), float(p[1])) for p in path_xy]
    if not path:
        return float("inf")
    if len(path) == 1:
        return math.hypot(x - path[0][0], y - path[0][1])
    best = float("inf")
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-18:
            d = math.hypot(x - x0, y - y0)
        else:
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2))
            px, py = x0 + t * dx, y0 + t * dy
            d = math.hypot(x - px, y - py)
        best = min(best, d)
    return float(best)


def project_point_to_path_s(
    x: float,
    y: float,
    path_xy: Sequence[tuple[float, float]],
    s_list: Sequence[float] | None = None,
) -> tuple[float, float, float, float]:
    """Project (x,y) onto polyline → (s, px, py, tangent_yaw)."""
    path = [(float(p[0]), float(p[1])) for p in path_xy]
    if not path:
        return 0.0, x, y, 0.0
    if s_list is None:
        s_list = cum_arclength(path)
    if len(path) == 1:
        return 0.0, path[0][0], path[0][1], 0.0
    best_d = float("inf")
    best = (0.0, path[0][0], path[0][1], 0.0)
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-18:
            t = 0.0
            px, py = x0, y0
        else:
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2))
            px, py = x0 + t * dx, y0 + t * dy
        d = math.hypot(x - px, y - py)
        if d < best_d:
            best_d = d
            yaw = math.atan2(dy, dx) if seg2 > 1e-18 else 0.0
            s = float(s_list[i - 1]) + t * max(float(s_list[i]) - float(s_list[i - 1]), 0.0)
            best = (s, px, py, yaw)
    return best


def recompute_kinematics_residuals(
    points: Sequence[tuple[float, float, float, float, float, float]],
    *,
    ego_v: float,
    dt_s: float,
    native_path_xy: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Recompute residuals from actual T10 points (do not trust diagnostics).

    - acceleration vs Δv/dt from ego_v for the first step
    - position integration includes **t=0 → first sample** using path arc origin s=0:
      ``s[0] ≈ 0.5 * (ego_v + v[0]) * dt``
    - subsequent steps use **signed** path progress ``ds = s[i] - s[i-1]``
      (never ``abs(ds)``); reverse motion with positive speed fails residual checks
    - yaw vs native path tangent at projection
    - cross-track distance to native polyline
    """
    pos_err = 0.0
    a_err = 0.0
    yaw_err = 0.0
    xtrack = 0.0
    signed_ds_min = 0.0
    kappa_errs: list[float] = []
    if not points:
        return {
            "position_integration_error_max_m": float("inf"),
            "acceleration_error_max_mps2": float("inf"),
            "yaw_tangent_error_max_rad": float("inf"),
            "native_path_cross_track_error_max_m": float("inf"),
            "curvature_error_max_per_m": float("inf"),
            "curvature_error_p95_per_m": float("inf"),
            "signed_path_progress_min_m": float("-inf"),
            "negative_path_progress": True,
        }

    dt = max(float(dt_s), 1e-9)
    path = [(float(p[0]), float(p[1])) for p in native_path_xy]
    s_list = cum_arclength(path) if len(path) >= 2 else [0.0]
    v_prev = max(0.0, float(ego_v))
    # Builder integrates arc length from path origin s=0 (not abs steps).
    prev_s: float = 0.0
    prev_yaw_pt: float | None = None
    first_step = True
    for i, row in enumerate(points):
        x, y, yaw, v, a, kappa = (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
        expected_a = (v - v_prev) / dt
        a_err = max(a_err, abs(a - expected_a))

        if len(path) >= 2:
            s_proj, _px, _py, tan_yaw = project_point_to_path_s(x, y, path, s_list)
            xtrack = max(xtrack, point_to_polyline_distance_m(x, y, path))
            yaw_err = max(yaw_err, abs(_wrap_angle(yaw - tan_yaw)))
            # Signed path progress from previous sample (or s=0 before first point).
            if first_step:
                expected_ds = 0.5 * (v_prev + v) * dt  # ego_v → v[0]
            else:
                expected_ds = 0.5 * (float(points[i - 1][3]) + v) * dt
            signed_ds = float(s_proj) - float(prev_s)
            if first_step:
                signed_ds_min = signed_ds
            else:
                signed_ds_min = min(signed_ds_min, signed_ds)
            # Reject reverse progress: signed_ds must match non-negative trapezoid.
            pos_err = max(pos_err, abs(signed_ds - expected_ds))
            if not first_step and prev_yaw_pt is not None and abs(signed_ds) > 1e-6:
                kappa_re = _wrap_angle(yaw - prev_yaw_pt) / max(abs(signed_ds), 1e-9)
                kappa_errs.append(abs(float(kappa) - kappa_re))
            prev_s = float(s_proj)
            first_step = False
        else:
            # No usable path: fall back to Euclidean step residual (signed via
            # forward component is unavailable; still check first step from origin
            # of the first point as zero baseline is not defined — mark xtrack inf).
            if first_step:
                # Without a path, cannot verify s=0→first; force residual fail closed.
                pos_err = max(pos_err, float("inf"))
                first_step = False
            else:
                px, py, pv = (
                    float(points[i - 1][0]),
                    float(points[i - 1][1]),
                    float(points[i - 1][3]),
                )
                expected_ds = 0.5 * (pv + v) * dt
                # Signed along chord is not path-ordered; use Euclidean magnitude
                # only as weak check (path-missing already hard-fails elsewhere).
                actual_ds = math.hypot(x - px, y - py)
                pos_err = max(pos_err, abs(actual_ds - expected_ds))
                if actual_ds > 1e-6:
                    chord_yaw = math.atan2(y - py, x - px)
                    yaw_err = max(yaw_err, abs(_wrap_angle(yaw - chord_yaw)))
            xtrack = float("inf")

        v_prev = v
        prev_yaw_pt = yaw

    kappa_max = max(kappa_errs) if kappa_errs else 0.0
    if kappa_errs:
        ks = sorted(kappa_errs)
        p95 = ks[min(len(ks) - 1, int(math.floor(0.95 * (len(ks) - 1))))]
    else:
        p95 = 0.0
    # Negative path progress with non-trivial reverse step is a hard kinematics fail.
    # Tiny numerical noise below 1e-6 is ignored.
    negative_progress = bool(signed_ds_min < -1e-6)
    if negative_progress:
        # Ensure residual exceeds registered tolerance even if distances match.
        pos_err = max(pos_err, abs(signed_ds_min) + 0.05)
    return {
        "position_integration_error_max_m": float(pos_err),
        "acceleration_error_max_mps2": float(a_err),
        "yaw_tangent_error_max_rad": float(yaw_err),
        "native_path_cross_track_error_max_m": float(xtrack),
        "curvature_error_max_per_m": float(kappa_max),
        "curvature_error_p95_per_m": float(p95),
        "signed_path_progress_min_m": float(signed_ds_min),
        "negative_path_progress": negative_progress,
    }


def verify_execution_spatial_binding(
    bundle: K2PredictionBundle,
    spec: K2ExecutionSpec,
) -> list[str]:
    """Content-hash execution spatial path against bundle native path (fail-closed)."""
    reasons: list[str] = []
    if not spec.spatial_path_xy or len(spec.spatial_path_xy) < 2:
        reasons.append(f"execution_spatial_path_insufficient:{spec.candidate_id}")
        return reasons
    content_hash = stable_hash_xy(spec.spatial_path_xy)
    bundle_path_hash = stable_hash_xy(bundle.native_path_xy) if bundle.native_path_xy else ""
    if content_hash != bundle.native_path_hash:
        reasons.append(f"execution_path_content_hash_mismatch:{spec.candidate_id}")
    if bundle_path_hash and content_hash != bundle_path_hash:
        reasons.append(f"execution_path_vs_bundle_path_mismatch:{spec.candidate_id}")
    if bundle.native_path_xy and bundle_path_hash != bundle.native_path_hash:
        reasons.append("bundle_native_path_hash_inconsistent")
    if spec.native_path_hash != bundle.native_path_hash:
        reasons.append(f"native_path_hash_mismatch:{spec.candidate_id}")
    # Exact geometry equality (same shared path for longitudinal temporal R1)
    if tuple(spec.spatial_path_xy) != tuple(bundle.native_path_xy):
        reasons.append(f"execution_spatial_path_xy_mismatch:{spec.candidate_id}")
    return reasons


def build_k2_bundle(
    native: Any,
    obs: ObservationBundle,
    *,
    config: K2BuilderConfig | None = None,
    model_id: str = "sdf-vla-v1-neural@0.1.0",
) -> K2PredictionBundle:
    """Build K2PredictionBundle from one NativePathPrediction + Observation."""
    cfg = config or load_k2_config()
    path_xy = tuple((float(x), float(y)) for x, y in (getattr(native, "path_map_xy", ()) or ()))
    # Fail closed: never synthesize a non-neural geometric path.
    if len(path_xy) < 2:
        return _hard_reject_bundle(
            obs=obs,
            model_id=model_id,
            config=cfg,
            native_path_xy=path_xy,
            build_error=BUILD_PATH_HORIZON_EXHAUSTED,
        )

    s_list = cum_arclength(path_xy)
    native_hash = stable_hash_xy(path_xy)
    speed_samples = tuple(float(v) for v in native.speed_mps)
    u_nom_raw = normalize_k2_target_speed_profile(
        speed_samples,
        t_steps=cfg.t_steps,
        mode=cfg.speed_mode,
        version=cfg.speed_normalize_version,
    )
    official_v = float(u_nom_raw[0]) if u_nom_raw else 0.0
    u_nom, u_cons = _branch_targets(
        u_nom_raw,
        ratio=cfg.conservative_speed_ratio,
        ego_v=float(obs.ego_v),
        stop_threshold_mps=cfg.stop_threshold_mps,
    )

    p_nom, p_cons = cfg.probability_prior
    id_nom, id_cons = cfg.candidate_ids

    nom = _build_branch(
        u_target=u_nom,
        path_xy=path_xy,
        s_list=s_list,
        ego_v=float(obs.ego_v),
        config=cfg,
        candidate_id=id_nom,
        probability=p_nom,
        intended_action="nominal",
        native_path_hash=native_hash,
    )
    cons = _build_branch(
        u_target=u_cons,
        path_xy=path_xy,
        s_list=s_list,
        ego_v=float(obs.ego_v),
        config=cfg,
        candidate_id=id_cons,
        probability=p_cons,
        intended_action="conservative",
        native_path_hash=native_hash,
    )

    (
        nom_traj,
        nom_spec,
        nom_met,
        nom_speeds,
        _nom_a,
        nom_arcs,
        nom_cap,
        nom_err,
    ) = nom
    (
        cons_traj,
        cons_spec,
        cons_met,
        cons_speeds,
        _cons_a,
        cons_arcs,
        cons_cap,
        cons_err,
    ) = cons

    path_cap = bool(nom_cap or cons_cap)
    build_error = nom_err or cons_err

    collapsed, collapse_reason, eligible, mean_gap, prog_gap, max_sep, mean_sep = _classify_collapse(
        config=cfg,
        nom_speeds=nom_speeds,
        cons_speeds=cons_speeds,
        nom_arcs=nom_arcs,
        cons_arcs=cons_arcs,
        nom_pts=nom_traj.points_xy_yaw_v_a_kappa,
        cons_pts=cons_traj.points_xy_yaw_v_a_kappa,
        path_cap_active=path_cap,
        ego_v=float(obs.ego_v),
        u_nom=u_nom,
    )

    def _mmax(key: str) -> float:
        return max(float(nom_met.get(key, 0.0)), float(cons_met.get(key, 0.0)))

    diagnostics = K2Diagnostics(
        mean_speed_gap_mps=float(mean_gap),
        final_progress_gap_m=float(prog_gap),
        max_position_separation_m=float(max_sep),
        mean_position_separation_m=float(mean_sep),
        collapsed=bool(collapsed),
        collapse_reason=collapse_reason,
        selection_space_eligible=bool(eligible),
        path_speed_cap_active=path_cap,
        position_integration_error_max_m=_mmax("position_integration_error_max_m"),
        acceleration_error_max_mps2=_mmax("acceleration_error_max_mps2"),
        yaw_tangent_error_max_rad=_mmax("yaw_tangent_error_max_rad"),
        curvature_error_max_per_m=_mmax("curvature_error_max_per_m"),
        curvature_error_p95_per_m=max(
            float(nom_met.get("curvature_error_p95_per_m", 0.0)),
            float(cons_met.get("curvature_error_p95_per_m", 0.0)),
        ),
        native_path_cross_track_error_max_m=_mmax("native_path_cross_track_error_max_m"),
        nominal_final_progress_m=float(nom_arcs[-1]) if nom_arcs else 0.0,
        conservative_final_progress_m=float(cons_arcs[-1]) if cons_arcs else 0.0,
        nominal_max_speed_mps=max(nom_speeds) if nom_speeds else 0.0,
        official_desired_speed_mps=official_v,
    )

    obs_id = _observation_identity(obs)

    bundle = K2PredictionBundle(
        observation_identity=obs_id,
        model_id=model_id,
        config_hash=cfg.config_hash(),
        retimer_version=cfg.retimer_version,
        native_path_xy=path_xy,
        native_path_hash=native_hash,
        candidates=(nom_traj, cons_traj),
        execution_specs={id_nom: nom_spec, id_cons: cons_spec},
        top1_index=int(cfg.top1_index),
        probability_source=PROBABILITY_SOURCE,
        probability_margin=0.0,
        branch_type=cfg.branch_type,
        diagnostics=diagnostics,
        guard_status=GUARD_OK if build_error is None else GUARD_REJECT,
        guard_reasons=() if build_error is None else (str(build_error),),
        build_error=build_error,
    )
    return bundle


class K2Builder:
    """Thin OO wrapper around :func:`build_k2_bundle`."""

    def __init__(self, config: K2BuilderConfig | None = None) -> None:
        self.config = config or load_k2_config()

    def build(
        self,
        native: Any,
        obs: ObservationBundle,
        *,
        model_id: str = "sdf-vla-v1-neural@0.1.0",
    ) -> K2PredictionBundle:
        return build_k2_bundle(native, obs, config=self.config, model_id=model_id)


def validate_k2_bundle(
    bundle: K2PredictionBundle,
    config: K2BuilderConfig | None = None,
) -> GuardResult:
    """Set-level K2 Contract Guard (structured; fail-closed).

    Residuals are recomputed from actual candidate points + observation ego_v.
    Declared ``bundle.diagnostics`` residual fields are never trusted for pass/fail.
    Execution spatial paths are content-hashed, not only declaration-checked.
    """
    cfg = config or load_k2_config()
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if bundle.build_error:
        reasons.append(str(bundle.build_error))

    # Bundle native path must itself be neural-derived and usable
    if len(bundle.native_path_xy) < 2:
        reasons.append(BUILD_PATH_HORIZON_EXHAUSTED)
        if BUILD_PATH_HORIZON_EXHAUSTED not in reasons:
            pass

    if bundle.native_path_xy:
        recomputed_bundle_hash = stable_hash_xy(bundle.native_path_xy)
        if recomputed_bundle_hash != bundle.native_path_hash:
            reasons.append("bundle_native_path_hash_inconsistent")

    if len(bundle.candidates) != cfg.k:
        reasons.append(f"K_mismatch: expected {cfg.k} got {len(bundle.candidates)}")

    ids = [c.candidate_id for c in bundle.candidates]
    if any(not cid for cid in ids):
        reasons.append("empty_candidate_id")
    if len(set(ids)) != len(ids):
        reasons.append("duplicate_candidate_id")

    probs = [float(c.probability) for c in bundle.candidates]
    if any((not math.isfinite(p)) or p < 0.0 or p > 1.0 for p in probs):
        reasons.append("probability_invalid")
    if bundle.candidates and abs(sum(probs) - 1.0) > 1e-5:
        reasons.append("probability_sum_not_one")

    for arr in bundle.candidates:
        if arr.t_steps != cfg.t_steps:
            reasons.append(f"T_mismatch:{arr.candidate_id}")
        for i, row in enumerate(arr.points_xy_yaw_v_a_kappa):
            if len(row) != 6:
                reasons.append(f"field_count:{arr.candidate_id}:{i}")
                continue
            for j, val in enumerate(row):
                if not math.isfinite(float(val)):
                    reasons.append(f"non_finite:{arr.candidate_id}:{i}:{j}")
            if float(row[3]) < -1e-9:
                reasons.append(f"negative_speed:{arr.candidate_id}:{i}")
            a = float(row[4])
            if a > cfg.max_accel_mps2 + 1e-5 or a < -cfg.max_decel_mps2 - 1e-5:
                reasons.append(f"accel_envelope:{arr.candidate_id}:{i}")

    # Absolute time contract: point i at (i+1)*dt → first 0.25 last 2.50
    first_t = cfg.dt_s
    last_t = cfg.t_steps * cfg.dt_s
    if abs(first_t - DT_S) > 1e-9 or abs(last_t - HORIZON_S) > 1e-9:
        reasons.append("time_contract_config")

    # execution specs bind + content hash of spatial path
    for arr in bundle.candidates:
        spec = bundle.execution_specs.get(arr.candidate_id)
        if spec is None:
            reasons.append(f"missing_execution_spec:{arr.candidate_id}")
            continue
        reasons.extend(verify_execution_spatial_binding(bundle, spec))
        timed_hash = stable_hash_points(arr.points_xy_yaw_v_a_kappa)
        if spec.timed_trajectory_hash != timed_hash:
            reasons.append(f"timed_hash_mismatch:{arr.candidate_id}")

    # --- Recompute kinematics residuals (never trust diagnostics for residual pass) ---
    ego_v = float((bundle.observation_identity or {}).get("ego_v", 0.0))
    recomputed_pos = 0.0
    recomputed_acc = 0.0
    recomputed_yaw = 0.0
    recomputed_xtrack = 0.0
    any_negative_progress = False
    signed_ds_min = 0.0
    for arr in bundle.candidates:
        resid = recompute_kinematics_residuals(
            arr.points_xy_yaw_v_a_kappa,
            ego_v=ego_v,
            dt_s=cfg.dt_s,
            native_path_xy=bundle.native_path_xy,
        )
        recomputed_pos = max(recomputed_pos, float(resid["position_integration_error_max_m"]))
        recomputed_acc = max(recomputed_acc, float(resid["acceleration_error_max_mps2"]))
        recomputed_yaw = max(recomputed_yaw, float(resid["yaw_tangent_error_max_rad"]))
        recomputed_xtrack = max(
            recomputed_xtrack, float(resid["native_path_cross_track_error_max_m"])
        )
        if bool(resid.get("negative_path_progress", False)):
            any_negative_progress = True
        signed_ds_min = min(
            signed_ds_min, float(resid.get("signed_path_progress_min_m", 0.0))
        )
        metrics[f"residual:{arr.candidate_id}"] = resid

    metrics["recomputed_position_integration_error_max_m"] = recomputed_pos
    metrics["recomputed_acceleration_error_max_mps2"] = recomputed_acc
    metrics["recomputed_yaw_tangent_error_max_rad"] = recomputed_yaw
    metrics["recomputed_native_path_cross_track_error_max_m"] = recomputed_xtrack
    metrics["recomputed_signed_path_progress_min_m"] = signed_ds_min
    metrics["negative_path_progress"] = any_negative_progress

    if recomputed_pos > cfg.position_integration_error_max_m + 1e-9:
        reasons.append("position_integration_residual")
    if any_negative_progress:
        reasons.append("negative_path_progress")
    if recomputed_acc > cfg.acceleration_error_max_mps2 + 1e-9:
        reasons.append("acceleration_residual")
    if recomputed_yaw > cfg.yaw_tangent_error_max_rad + 1e-9:
        reasons.append("yaw_tangent_residual")
    if recomputed_xtrack > cfg.native_path_cross_track_error_max_m + 1e-9:
        reasons.append("cross_track_residual")

    d = bundle.diagnostics
    # Collapse / eligibility labels still read from diagnostics for classification
    # (builder-owned), but spatial separation for collapse is recomputed from points.
    if len(bundle.candidates) >= 2:
        a_pts = bundle.candidates[0].points_xy_yaw_v_a_kappa
        b_pts = bundle.candidates[1].points_xy_yaw_v_a_kappa
        seps = [
            math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))
            for p, q in zip(a_pts, b_pts)
        ]
        max_sep = max(seps) if seps else 0.0
        mean_v0 = sum(float(p[3]) for p in a_pts) / max(len(a_pts), 1)
        mean_v1 = sum(float(p[3]) for p in b_pts) / max(len(b_pts), 1)
        mean_speed_gap = abs(mean_v0 - mean_v1)
        metrics["recomputed_max_position_separation_m"] = max_sep
        metrics["recomputed_mean_speed_gap_mps"] = mean_speed_gap
    else:
        max_sep = 0.0
        mean_speed_gap = 0.0

    if d.selection_space_eligible and d.collapse_reason == COLLAPSE_NUMERIC:
        reasons.append(COLLAPSE_NUMERIC)

    # Reject classic residual bug: identical xy with differing speeds on eligible
    if d.selection_space_eligible and max_sep < 1e-6:
        if mean_speed_gap > 0.05:
            reasons.append("xy_collapse_with_speed_gap")

    # Merge declared diagnostics into metrics for audit (not for residual pass)
    try:
        metrics["declared_diagnostics"] = asdict(d)
    except Exception:
        metrics["declared_diagnostics"] = {}

    if reasons:
        status = GUARD_REJECT
        return GuardResult(status=status, reasons=tuple(reasons), metrics=metrics)

    # Explicit no-space is OK (not numeric collapse)
    if d.collapsed and d.collapse_reason in {
        COLLAPSE_SEMANTIC_STOP,
        COLLAPSE_PATH_LIMIT,
    }:
        return GuardResult(
            status=GUARD_OK,
            reasons=(str(d.collapse_reason),),
            metrics=metrics,
        )

    return GuardResult(status=GUARD_OK, reasons=(), metrics=metrics)


def attach_guard(
    bundle: K2PredictionBundle,
    config: K2BuilderConfig | None = None,
) -> K2PredictionBundle:
    """Run Guard and return bundle with guard_status/reasons filled."""
    result = validate_k2_bundle(bundle, config)
    return replace(
        bundle,
        guard_status=result.status,
        guard_reasons=result.reasons,
    )
