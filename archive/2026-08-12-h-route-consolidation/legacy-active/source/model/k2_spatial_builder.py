"""Build Spatial K2 V2 bundles from native path + residual heads (or synthetic residuals)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from driving_vla.model.frenet_codec import (
    decode_frenet_residual_path,
    first_point_error,
    max_xy_separation,
    smooth_path_xy,
    smooth_scalar_series,
)
from driving_vla.model.k2_spatial_types import (
    BRANCH_TYPE_V2,
    ID_DEFENSIVE,
    ID_NOMINAL,
    MODE_DEFENSIVE,
    MODE_NOMINAL,
    SCHEMA_V2,
    K2CandidateV2,
    K2ExecutionSpecV2,
    K2PredictionBundleV2,
    K2SpatialConfig,
    load_k2_spatial_config,
    stable_hash_traj,
    stable_hash_xy,
)
from driving_vla.model.speed_convert import planner_samples_from_cruise_scalar


def _project_speed(
    u_target: Sequence[float],
    *,
    ego_v: float,
    dt_s: float,
    max_accel: float,
    max_decel: float,
) -> tuple[list[float], list[float], list[float]]:
    """Return speeds, arcs, accels along integrated s."""
    v = max(0.0, float(ego_v))
    s = 0.0
    speeds: list[float] = []
    arcs: list[float] = []
    accels: list[float] = []
    for u in u_target:
        u = max(0.0, float(u))
        a = (u - v) / max(dt_s, 1e-9)
        a = max(-max_decel, min(max_accel, a))
        v_next = max(0.0, v + a * dt_s)
        s = s + 0.5 * (v + v_next) * dt_s
        speeds.append(v_next)
        arcs.append(s)
        accels.append(a)
        v = v_next
    return speeds, arcs, accels


def _sample_t10_on_path(
    path_xy: Sequence[tuple[float, float]],
    speeds: Sequence[float],
    arcs: Sequence[float],
    accels: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Sample T10 (x,y,yaw,v,a,kappa) by arc length on candidate path."""
    from driving_vla.model.canonicalizer import cum_arclength

    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 2:
        return tuple()
    s_list = cum_arclength(pts)
    s_max = float(s_list[-1])
    out: list[tuple[float, ...]] = []
    prev_xy: tuple[float, float] | None = None
    for i, (v, s, a) in enumerate(zip(speeds, arcs, accels)):
        s_q = min(max(0.0, float(s)), s_max)
        # If integrated arc exceeds path length, hold at end with stop speed
        # to avoid END_POINT_COPY_POS_SPEED (duplicate xy with v>stop_threshold).
        at_end = float(s) >= s_max - 1e-6 or s_max < 0.05
        # locate segment
        j = 0
        for k in range(len(s_list) - 1):
            if s_list[k + 1] + 1e-12 >= s_q:
                j = k
                break
            j = k
        s0, s1 = float(s_list[j]), float(s_list[j + 1])
        u = (s_q - s0) / max(s1 - s0, 1e-9)
        x = pts[j][0] + u * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + u * (pts[j + 1][1] - pts[j][1])
        # Nudge only moving samples that numerically collapse.  A stopped
        # trajectory must hold position; nudging zero-speed samples to the
        # next vertex creates an alternating forward/backward fake path.
        if (
            prev_xy is not None
            and float(v) > 0.05
            and math.hypot(x - prev_xy[0], y - prev_xy[1]) < 1e-4
        ):
            if j + 1 < len(pts):
                x = pts[min(j + 1, len(pts) - 1)][0]
                y = pts[min(j + 1, len(pts) - 1)][1]
            v = 0.0
            a = min(float(a), -0.5)
        if at_end and prev_xy is not None and math.hypot(x - prev_xy[0], y - prev_xy[1]) < 1e-3:
            v = 0.0
            a = min(float(a), -1.0)
        tx = pts[j + 1][0] - pts[j][0]
        ty = pts[j + 1][1] - pts[j][1]
        yaw = math.atan2(ty, tx)
        # curvature proxy
        if j + 2 < len(pts):
            tx2 = pts[j + 2][0] - pts[j + 1][0]
            ty2 = pts[j + 2][1] - pts[j + 1][1]
            ang1 = math.atan2(ty, tx)
            ang2 = math.atan2(ty2, tx2)
            dpsi = (ang2 - ang1 + math.pi) % (2 * math.pi) - math.pi
            ds = max(math.hypot(tx, ty), 1e-6)
            kappa = dpsi / ds
        else:
            kappa = 0.0
        out.append((x, y, yaw, float(v), float(a), float(kappa)))
        prev_xy = (x, y)
    return tuple(out)


def build_candidate_from_residual(
    *,
    native_path_xy: Sequence[tuple[float, float]],
    raw_delta_s: Sequence[float],
    raw_d: Sequence[float],
    speed_scale: float,
    base_speed_mps: float,
    ego_v: float,
    candidate_id: str,
    mode_id: str,
    available: bool,
    availability_reason: str,
    probability: float,
    head_lineage: str,
    config: K2SpatialConfig,
    raw_head_output_hash: str = "",
    feature_content_hash: str = "",
    exact_native_anchor: bool = False,
) -> tuple[K2CandidateV2, K2ExecutionSpecV2]:
    from driving_vla.model.canonicalizer import cum_arclength
    from driving_vla.model.frenet_codec import project_path_to_frenet
    from driving_vla.model.k2_spatial_types import canonical_sha256

    if exact_native_anchor:
        # Fixed semantic slot: candidate 0 is the upstream native proposal,
        # not a second learned residual that can drift with candidate 1.
        path = tuple((float(x), float(y)) for x, y in native_path_xy)
        fr_s = tuple(float(v) for v in cum_arclength(path))
        fr_d = tuple(0.0 for _ in path)
    else:
        # Smooth residual lateral before decode to reduce densify spikes.
        # hard_max_abs_curvature=1.0 is the anomaly reject, not trackable κ.
        raw_d_s = smooth_scalar_series([float(x) for x in raw_d], passes=2)
        path, fr_s, fr_d = decode_frenet_residual_path(
            native_path_xy,
            raw_delta_s,
            raw_d_s,
            max_lateral_m=config.max_lateral_residual_m,
            max_delta_s_per_step_m=config.max_delta_s_per_step_m,
            ramp_points=config.envelope_ramp_points,
        )
        path = smooth_path_xy(
            path,
            passes=2,
            hard_max_abs_curvature=0.25,
        )
        # Segment project_s recompute Frenet after XY smooth.
        fr_s, fr_d = project_path_to_frenet(native_path_xy, path)

    # speed_scale must be finite formal value (no silent 1.0/0.85 fallback)
    try:
        sp_f = float(speed_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_speed_scale:{speed_scale!r}") from exc
    if not math.isfinite(sp_f) or sp_f <= 0.0:
        raise ValueError(f"non_finite_or_non_positive_speed_scale:{sp_f}")
    cruise = max(0.0, float(base_speed_mps) * sp_f)
    u = planner_samples_from_cruise_scalar(cruise, n=int(config.t_steps))
    speeds, arcs, accels = _project_speed(
        u,
        ego_v=ego_v,
        dt_s=config.dt_s,
        max_accel=config.max_accel_mps2,
        max_decel=config.max_decel_mps2,
    )
    # Fit temporal arc profile into actual path length (avoids end-point pile-up)
    path_s_max = float(cum_arclength([(float(x), float(y)) for x, y in path])[-1])
    if arcs and float(arcs[-1]) > max(path_s_max, 1e-3):
        scale = path_s_max / max(float(arcs[-1]), 1e-6)
        speeds = [max(0.0, float(v) * scale) for v in speeds]
        arcs = [float(a) * scale for a in arcs]
        accels2: list[float] = []
        v_prev = max(0.0, float(ego_v) * scale)
        for v in speeds:
            a = (float(v) - v_prev) / max(config.dt_s, 1e-9)
            a = max(-config.max_decel_mps2, min(config.max_accel_mps2, a))
            accels2.append(a)
            v_prev = float(v)
        accels = accels2
    points = _sample_t10_on_path(path, speeds, arcs, accels)
    # Execution speed samples MUST match final projected T10 speeds
    final_speeds = tuple(float(row[3]) for row in points)
    if len(final_speeds) != int(config.t_steps):
        fs = list(final_speeds)
        while len(fs) < int(config.t_steps):
            fs.append(fs[-1] if fs else 0.0)
        final_speeds = tuple(fs[: int(config.t_steps)])
    native_h = stable_hash_xy(native_path_xy)
    path_h = stable_hash_xy(path)
    traj_h = stable_hash_traj(points)
    codec_h = canonical_sha256(
        {
            "max_lateral_m": config.max_lateral_residual_m,
            "max_delta_s": config.max_delta_s_per_step_m,
            "ramp_points": config.envelope_ramp_points,
            "codec": "frenet_decode_v2",
            "exact_native_anchor": bool(exact_native_anchor),
        }
    )
    post_h = canonical_sha256(
        {
            "smooth_passes": 2,
            "hard_max_abs_curvature": 0.25,
            "exact_native_anchor": bool(exact_native_anchor),
            "path": [[float(x), float(y)] for x, y in path],
            "frenet_s": list(fr_s),
            "frenet_d": list(fr_d),
            "t10": [list(map(float, row[:6])) for row in points],
            "speeds": list(final_speeds),
        }
    )
    raw_head_h = str(raw_head_output_hash or "")
    feat_h = str(feature_content_hash or "")
    cand = K2CandidateV2(
        candidate_id=candidate_id,
        mode_id=mode_id,
        available=bool(available),
        availability_reason=str(availability_reason),
        probability=float(probability),
        points_xy_yaw_v_a_kappa=points,
        frenet_s=fr_s[: len(path)] if fr_s else tuple(),
        frenet_d=fr_d[: len(path)] if fr_d else tuple(),
        proposal_path_hash=path_h,
        timed_trajectory_hash=traj_h,
        native_anchor_hash=native_h,
        head_lineage=head_lineage,
        spatial_path_xy=path,
        raw_head_output_hash=raw_head_h,
        feature_content_hash=feat_h,
        codec_config_hash=codec_h,
        postprocess_hash=post_h,
    )
    spec = K2ExecutionSpecV2(
        candidate_id=candidate_id,
        spatial_path_xy=path,
        speed_samples_mps=final_speeds,
        spatial_path_hash=path_h,
        timed_trajectory_hash=traj_h,
        native_anchor_hash=native_h,
        branch_type=BRANCH_TYPE_V2,
        mode_id=mode_id,
        available=bool(available),
        head_lineage=head_lineage,
        raw_head_output_hash=raw_head_h,
        feature_content_hash=feat_h,
        codec_config_hash=codec_h,
        postprocess_hash=post_h,
    )
    return cand, spec


def build_spatial_k2_bundle_from_residuals(
    *,
    native_path_xy: Sequence[tuple[float, float]],
    ego_xy: tuple[float, float],
    ego_v: float,
    base_speed_mps: float,
    residual_nominal: Mapping[str, Sequence[float]],
    residual_defensive: Mapping[str, Sequence[float]],
    observation_identity: Mapping[str, Any],
    backbone_forward_id: str,
    model_id: str = "sdf-spatial-k2-v2@heads",
    base_checkpoint_hash: str = "unset",
    spatial_head_checkpoint_hash: str = "unset",
    defensive_available: bool = True,
    defensive_reason: str = "ok",
    nominal_probability: float = 0.5,
    defensive_probability: float = 0.5,
    config: K2SpatialConfig | None = None,
    probability_source: str = "fixed_prior_uncalibrated",
) -> K2PredictionBundleV2:
    """Construct V2 bundle from explicit residual sequences (heads or synthetic)."""
    cfg = config or load_k2_spatial_config()
    n_path = max(2, len(tuple(native_path_xy)))

    def _pad(seq: Sequence[float], n: int, fill: float = 0.0) -> list[float]:
        xs = [float(x) for x in seq]
        if len(xs) >= n:
            return xs[:n]
        return xs + [fill] * (n - len(xs))

    id0, id1 = cfg.candidate_ids
    m0, m1 = cfg.mode_ids
    nom_ds = _pad(residual_nominal.get("delta_s") or residual_nominal.get("raw_delta_s") or [], n_path, 0.5)
    nom_d = _pad(residual_nominal.get("d") or residual_nominal.get("raw_d") or [], n_path, 0.0)
    def_ds = _pad(residual_defensive.get("delta_s") or residual_defensive.get("raw_delta_s") or [], n_path, 0.5)
    def_d = _pad(residual_defensive.get("d") or residual_defensive.get("raw_d") or [], n_path, 0.0)

    if "speed_scale" not in residual_nominal or "speed_scale" not in residual_defensive:
        raise ValueError("residual_missing_speed_scale")
    sp0 = residual_nominal.get("speed_scale")
    sp1 = residual_defensive.get("speed_scale")
    try:
        sp0f = float(sp0)  # type: ignore[arg-type]
        sp1f = float(sp1)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_residual_speed_scale:{sp0!r}:{sp1!r}") from exc
    if not math.isfinite(sp0f) or not math.isfinite(sp1f) or sp0f <= 0 or sp1f <= 0:
        raise ValueError(f"non_finite_speed_scale:{sp0f}:{sp1f}")

    from driving_vla.model.k2_spatial_types import canonical_sha256

    raw_head_hash = str(
        residual_nominal.get("raw_head_hash")
        or residual_defensive.get("raw_head_hash")
        or canonical_sha256(
            {
                "nom_ds": nom_ds,
                "nom_d": nom_d,
                "def_ds": def_ds,
                "def_d": def_d,
                "sp0": sp0f,
                "sp1": sp1f,
            }
        )
    )
    feature_hash = str(
        observation_identity.get("feature_hash")
        or observation_identity.get("feature_content_hash")
        or ""
    )

    c0, s0 = build_candidate_from_residual(
        native_path_xy=native_path_xy,
        raw_delta_s=nom_ds,
        raw_d=nom_d,
        speed_scale=sp0f,
        base_speed_mps=base_speed_mps,
        ego_v=ego_v,
        candidate_id=id0,
        mode_id=m0,
        available=True,
        availability_reason="always",
        probability=float(nominal_probability),
        head_lineage=str(residual_nominal.get("head_lineage", "spatial_mode_head")),
        config=cfg,
        raw_head_output_hash=raw_head_hash,
        feature_content_hash=feature_hash,
        exact_native_anchor=bool(cfg.nominal_anchor_exact),
    )
    c1, s1 = build_candidate_from_residual(
        native_path_xy=native_path_xy,
        raw_delta_s=def_ds,
        raw_d=def_d,
        speed_scale=sp1f,
        base_speed_mps=base_speed_mps,
        ego_v=ego_v,
        candidate_id=id1,
        mode_id=m1,
        available=bool(defensive_available),
        availability_reason=str(defensive_reason),
        probability=float(defensive_probability),
        head_lineage=str(residual_defensive.get("head_lineage", "spatial_mode_head")),
        config=cfg,
        raw_head_output_hash=raw_head_hash,
        feature_content_hash=feature_hash,
        exact_native_anchor=False,
    )

    sep = max_xy_separation(c0.spatial_path_xy, c1.spatial_path_xy)
    near_n = max(1, int(round(cfg.near_field_horizon_s / max(cfg.dt_s, 1e-6))))
    near_sep = 0.0
    for i in range(min(near_n, len(c0.spatial_path_xy), len(c1.spatial_path_xy))):
        near_sep = max(
            near_sep,
            math.hypot(
                c0.spatial_path_xy[i][0] - c1.spatial_path_xy[i][0],
                c0.spatial_path_xy[i][1] - c1.spatial_path_xy[i][1],
            ),
        )
    first_err = max(
        first_point_error(c0.spatial_path_xy, ego_xy),
        first_point_error(c1.spatial_path_xy, ego_xy),
    )
    diag = {
        "max_spatial_separation_m": float(sep),
        "near_field_separation_m": float(near_sep),
        "first_point_error_max_m": float(first_err),
        "defensive_available": bool(defensive_available),
        "eligible_for_diversity": bool(defensive_available and sep >= cfg.ambiguity_min_spatial_sep_m * 0.5),
        "availability_semantics": cfg.availability_semantics,
        "learned_confidence_non_blocking": cfg.learned_confidence_non_blocking,
    }
    return K2PredictionBundleV2(
        schema_version=SCHEMA_V2,
        observation_identity=dict(observation_identity),
        model_id=model_id,
        config_hash=cfg.config_hash(),
        base_checkpoint_hash=base_checkpoint_hash,
        spatial_head_checkpoint_hash=spatial_head_checkpoint_hash,
        backbone_forward_id=str(backbone_forward_id),
        native_path_xy=tuple((float(x), float(y)) for x, y in native_path_xy),
        native_path_hash=stable_hash_xy(native_path_xy),
        candidates=(c0, c1),
        execution_specs={id0: s0, id1: s1},
        top1_index=int(cfg.top1_index),
        probability_source=probability_source,
        branch_type=BRANCH_TYPE_V2,
        set_diagnostics=diag,
        raw_head_output_hash=raw_head_hash,
        feature_content_hash=feature_hash,
        codec_config_hash=c0.codec_config_hash,
        postprocess_hash=canonical_sha256(
            {"c0": c0.postprocess_hash, "c1": c1.postprocess_hash}
        ),
    )


def synthetic_diverse_residuals(
    n: int = 20,
    *,
    lateral_sign: float = 1.0,
    lineage: str = "contract_probe",
) -> tuple[dict, dict]:
    """Contract-probe residuals: nominal ~on path; defensive smooth lateral.

    Default lineage is ``contract_probe`` (executor sensitivity / Guard plumbing only).
    Never use this lineage as learned live Evidence; pass ``lineage='spatial_mode_head'``
    only for unit tests that intentionally claim head-shaped labels.
    """
    nom = {
        "raw_delta_s": [0.8] * n,
        "raw_d": [0.0] * n,
        "speed_scale": 1.0,
        "head_lineage": lineage,
    }
    raw_d = [0.0] * n
    for i in range(n):
        raw_d[i] = float(lateral_sign) * min(2.0, 0.15 * i)
    alt = {
        "raw_delta_s": [0.7] * n,
        "raw_d": raw_d,
        "speed_scale": 0.85,
        "head_lineage": lineage,
    }
    return nom, alt
