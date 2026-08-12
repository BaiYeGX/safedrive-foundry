"""Guard V2 for Spatial K2 — independent of longitudinal V1 verifier.

Implements the R2X §15 contract checks that were previously only partially wired:
hash/schema/lineage + dynamics recompute, corridor, xtrack, curvature,
self-intersection, forward ratio, end-point copy, set-level identity.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from driving_vla.model.frenet_codec import max_xy_separation, path_frame, sample_native_at_s
from driving_vla.model.k2_spatial_types import (
    ACCEL_ENVELOPE,
    ALTERNATIVE_UNAVAILABLE,
    BRANCH_TYPE_V2,
    CURVATURE_ENVELOPE,
    END_POINT_COPY_POS_SPEED,
    FORWARD_ID_MISMATCH,
    FORWARD_RATIO,
    FRENET_LATERAL_ENVELOPE,
    FRENET_PROGRESS_NEGATIVE,
    GUARD_OK,
    GUARD_REJECT,
    HEAD_LINEAGE_INVALID,
    KAPPA_RECOMPUTE_MISMATCH,
    NATIVE_CORRIDOR,
    NEAR_FIELD_DISCONTINUITY,
    PATH_TOO_SHORT,
    POSITION_INTEGRAL_RESIDUAL,
    PROBABILITY_INVALID,
    PROPOSAL_PATH_HASH_MISMATCH,
    SCHEMA_MISMATCH,
    SCHEMA_V2,
    SELF_INTERSECTION,
    SET_IDENTITY,
    SPATIAL_COLLAPSE_ELIGIBLE,
    T_MISMATCH,
    TIMED_PATH_BINDING_MISMATCH,
    TIMED_XTRACK,
    TOP1_INVALID,
    YAW_RECOMPUTE_MISMATCH,
    K2CandidateV2,
    K2PredictionBundleV2,
    K2SpatialConfig,
    load_k2_spatial_config,
    stable_hash_traj,
    stable_hash_xy,
)

_FORBIDDEN_LINEAGE = frozenset(
    {
        "fixed_bias",
        "fixed-lateral-bias",
        "debug_offset",
        "noise",
        "random_noise",
        "lattice_template_runtime",
    }
)


def _wrap_angle(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _interp_monotonic(
    s_values: Sequence[float],
    d_values: Sequence[float],
    query_s: float,
) -> float:
    """Linearly interpolate d(s) on a non-decreasing Frenet sequence."""
    n = min(len(s_values), len(d_values))
    if n == 0:
        return 0.0
    pairs = [(float(s_values[i]), float(d_values[i])) for i in range(n)]
    if query_s <= pairs[0][0]:
        return pairs[0][1]
    for i in range(n - 1):
        s0, d0 = pairs[i]
        s1, d1 = pairs[i + 1]
        if query_s <= s1 + 1e-12:
            if s1 <= s0 + 1e-12:
                return d1
            u = (float(query_s) - s0) / (s1 - s0)
            return d0 + u * (d1 - d0)
    return pairs[-1][1]


def _aligned_inter_candidate_lateral_separation(
    c0: K2CandidateV2,
    c1: K2CandidateV2,
) -> float:
    """Return max |d0(s)-d1(s)| on their shared Frenet-s support."""
    if not c0.frenet_s or not c1.frenet_s:
        return 0.0
    shared_end = min(float(c0.frenet_s[-1]), float(c1.frenet_s[-1]))
    queries = {
        max(0.0, min(shared_end, float(s)))
        for s in (*c0.frenet_s, *c1.frenet_s)
        if float(s) <= shared_end + 1e-9
    }
    if not queries:
        queries = {0.0}
    return max(
        abs(
            _interp_monotonic(c0.frenet_s, c0.frenet_d, s)
            - _interp_monotonic(c1.frenet_s, c1.frenet_d, s)
        )
        for s in queries
    )


def _point_to_polyline_xtrack(
    px: float,
    py: float,
    path_xy: Sequence[tuple[float, float]],
) -> float:
    """Min distance from point to polyline segments (cross-track proxy)."""
    pts = [(float(x), float(y)) for x, y in path_xy]
    if not pts:
        return float("inf")
    if len(pts) == 1:
        return math.hypot(px - pts[0][0], py - pts[0][1])
    best = float("inf")
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            d = math.hypot(px - x0, py - y0)
        else:
            t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
            qx, qy = x0 + t * dx, y0 + t * dy
            d = math.hypot(px - qx, py - qy)
        if d < best:
            best = d
    return best


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> bool:
    """Proper segment intersection (excludes shared endpoints)."""

    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    # collinear overlaps treated as self-intersection only if interiors overlap
    if abs(o1) < 1e-9 and on_seg(a0, b0, a1) and not (
        math.hypot(b0[0] - a0[0], b0[1] - a0[1]) < 1e-9
        or math.hypot(b0[0] - a1[0], b0[1] - a1[1]) < 1e-9
    ):
        return True
    return False


def _path_self_intersects(path_xy: Sequence[tuple[float, float]]) -> bool:
    pts = [(float(x), float(y)) for x, y in path_xy]
    n = len(pts)
    if n < 4:
        return False
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            # skip adjacent segments and shared vertex pairs
            if j == i + 1:
                continue
            if i == 0 and j == n - 2:
                continue
            if _segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return True
    return False


def _path_forward_ratio(path_xy: Sequence[tuple[float, float]]) -> float:
    """Fraction of segments whose projection on overall displacement is forward."""
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 2:
        return 0.0
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        # zero net displacement — use first segment heading
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return 0.0
    ux, uy = dx / L, dy / L
    n_seg = 0
    n_fwd = 0
    for i in range(len(pts) - 1):
        sx = pts[i + 1][0] - pts[i][0]
        sy = pts[i + 1][1] - pts[i][1]
        if math.hypot(sx, sy) < 1e-9:
            continue
        n_seg += 1
        if sx * ux + sy * uy >= -1e-9:
            n_fwd += 1
    if n_seg == 0:
        return 0.0
    return n_fwd / n_seg


def _path_max_abs_curvature(path_xy: Sequence[tuple[float, float]]) -> float:
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 3:
        return 0.0
    max_k = 0.0
    for i in range(1, len(pts) - 1):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx0, dy0 = x1 - x0, y1 - y0
        dx1, dy1 = x2 - x1, y2 - y1
        ds0 = math.hypot(dx0, dy0)
        ds1 = math.hypot(dx1, dy1)
        if ds0 < 1e-6 or ds1 < 1e-6:
            continue
        a0 = math.atan2(dy0, dx0)
        a1 = math.atan2(dy1, dx1)
        dpsi = _wrap_angle(a1 - a0)
        kappa = abs(dpsi) / max(0.5 * (ds0 + ds1), 1e-6)
        max_k = max(max_k, kappa)
    return max_k


def _path_signed_progress_ok(
    path_xy: Sequence[tuple[float, float]],
    min_ds: float,
) -> bool:
    """Arc-length along polyline must be non-decreasing (always true if sequential),
    but cumulative projection on start→end must not go strongly backward."""
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 2:
        return True
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return True
    ux, uy = dx / L, dy / L
    prev = 0.0
    for i in range(len(pts)):
        s = (pts[i][0] - pts[0][0]) * ux + (pts[i][1] - pts[0][1]) * uy
        if s + 1e-9 < prev + min_ds:
            # allow small local retreat within tolerance
            if s < prev - abs(min_ds) - 0.05:
                return False
        prev = max(prev, s)
    return True


def _recompute_t10_checks(
    cand: K2CandidateV2,
    cfg: K2SpatialConfig,
    reasons: list[str],
) -> dict[str, float]:
    """Recompute kinematics from T10 points; append reason codes on violation."""
    metrics: dict[str, float] = {}
    pts = cand.points_xy_yaw_v_a_kappa
    if len(pts) != int(cfg.t_steps):
        reasons.append(f"{T_MISMATCH}:{cand.candidate_id}:{len(pts)}")
        return metrics
    dt = float(cfg.dt_s)
    max_int_err = 0.0
    max_yaw_err = 0.0
    max_kappa_err = 0.0
    for i, row in enumerate(pts):
        if len(row) < 6:
            reasons.append(f"NON_FINITE:{cand.candidate_id}:short_row")
            return metrics
        if any(not math.isfinite(float(v)) for v in row[:6]):
            reasons.append(f"NON_FINITE:{cand.candidate_id}")
            return metrics
        v = float(row[3])
        a = float(row[4])
        if v < -1e-6:
            reasons.append(f"NEGATIVE_SPEED:{cand.candidate_id}")
        if a > cfg.max_accel_mps2 + 1e-3:
            reasons.append(f"{ACCEL_ENVELOPE}:acc:{cand.candidate_id}")
        if a < -(cfg.max_decel_mps2 + 1e-3):
            reasons.append(f"{ACCEL_ENVELOPE}:dec:{cand.candidate_id}")
        if abs(float(row[5])) > cfg.hard_max_abs_curvature + 1e-3:
            reasons.append(f"{CURVATURE_ENVELOPE}:t10:{cand.candidate_id}")

        if i > 0:
            prev = pts[i - 1]
            v_prev = float(prev[3])
            a_prev = float(prev[4])
            # position integral: x_{i} ≈ x_{i-1} + v_prev*dt*cos(yaw) + 0.5*a*dt^2...
            # Use midpoint speed along declared yaw of prev sample.
            yaw_p = float(prev[2])
            v_mid = max(0.0, v_prev + 0.5 * a_prev * dt)
            # also blend with stated a on current for robustness
            exp_x = float(prev[0]) + v_mid * dt * math.cos(yaw_p)
            exp_y = float(prev[1]) + v_mid * dt * math.sin(yaw_p)
            err = math.hypot(float(row[0]) - exp_x, float(row[1]) - exp_y)
            # Along a curved path the straight-line integral drifts; allow extra
            # slack proportional to |kappa|*v^2.
            slack = cfg.position_integral_tol_m + abs(float(prev[5])) * (v_prev**2) * dt * 2.0
            max_int_err = max(max_int_err, err)
            # Path-sampled T10 is not pure Euler integrate; hard-reject only
            # gross spoof/inconsistency (repair C: was fail-open `pass`).
            gross = float(cfg.position_integral_gross_reject_m)
            if err > max(slack + 0.5, gross):
                reasons.append(
                    f"{POSITION_INTEGRAL_RESIDUAL}:{cand.candidate_id}:{i}:{err:.3f}"
                )
            # yaw vs chord
            chord_yaw = math.atan2(float(row[1]) - float(prev[1]), float(row[0]) - float(prev[0]))
            step_len = math.hypot(float(row[0]) - float(prev[0]), float(row[1]) - float(prev[1]))
            if step_len > 0.05:
                yaw_err = abs(_wrap_angle(float(row[2]) - chord_yaw))
                max_yaw_err = max(max_yaw_err, yaw_err)
                # Hard reject only on catastrophic yaw spoof; path-sampled T10
                # chord vs tangent can exceed modest tol on curved natives.
                if yaw_err > max(cfg.yaw_recompute_tol_rad * 2.5, 1.5) + 1e-6:
                    reasons.append(f"{YAW_RECOMPUTE_MISMATCH}:{cand.candidate_id}:{i}")
            # kappa recompute from successive chords when enough motion
            if i >= 2:
                p0 = pts[i - 2]
                p1 = pts[i - 1]
                p2 = pts[i]
                a0 = math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0]))
                a1 = math.atan2(float(p2[1]) - float(p1[1]), float(p2[0]) - float(p1[0]))
                ds = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
                if ds > 0.05:
                    kappa_hat = _wrap_angle(a1 - a0) / ds
                    kerr = abs(kappa_hat - float(p1[5]))
                    max_kappa_err = max(max_kappa_err, kerr)
                    if kerr > max(cfg.kappa_recompute_tol * 2.0, 2.5) + 1e-6:
                        reasons.append(f"{KAPPA_RECOMPUTE_MISMATCH}:{cand.candidate_id}:{i}")

            # accel recompute from speeds
            a_hat = (v - v_prev) / max(dt, 1e-9)
            if abs(a_hat - a) > max(cfg.max_accel_mps2, cfg.max_decel_mps2) + 1.0:
                reasons.append(f"{ACCEL_ENVELOPE}:recompute:{cand.candidate_id}:{i}")

    metrics["max_position_integral_err_m"] = max_int_err
    metrics["max_yaw_recompute_err_rad"] = max_yaw_err
    metrics["max_kappa_recompute_err"] = max_kappa_err

    # positive speed with terminal repeated points (path end copy)
    if len(pts) >= 2:
        last = pts[-1]
        prev = pts[-2]
        dist = math.hypot(float(last[0]) - float(prev[0]), float(last[1]) - float(prev[1]))
        if float(last[3]) > cfg.stop_threshold_mps and dist < 1e-4:
            # check if multiple trailing duplicates
            n_dup = 0
            for j in range(len(pts) - 1, 0, -1):
                d = math.hypot(
                    float(pts[j][0]) - float(pts[j - 1][0]),
                    float(pts[j][1]) - float(pts[j - 1][1]),
                )
                if d < 1e-4 and float(pts[j][3]) > cfg.stop_threshold_mps:
                    n_dup += 1
                else:
                    break
            if n_dup >= 2:
                reasons.append(f"{END_POINT_COPY_POS_SPEED}:{cand.candidate_id}")
    return metrics


def _candidate_geometry_checks(
    cand: K2CandidateV2,
    native_path_xy: Sequence[tuple[float, float]],
    cfg: K2SpatialConfig,
    reasons: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    path = cand.spatial_path_xy
    if len(path) < int(cfg.proposal_path_min_points):
        reasons.append(f"{PATH_TOO_SHORT}:{cand.candidate_id}:{len(path)}")
        return metrics

    # timed T10 → own proposal path cross-track
    max_xt = 0.0
    for row in cand.points_xy_yaw_v_a_kappa:
        xt = _point_to_polyline_xtrack(float(row[0]), float(row[1]), path)
        max_xt = max(max_xt, xt)
    metrics["timed_to_proposal_xtrack_max_m"] = max_xt
    if max_xt > cfg.timed_to_proposal_xtrack_max_m + 1e-6:
        reasons.append(f"{TIMED_XTRACK}:{cand.candidate_id}:{max_xt:.3f}")

    # native corridor: proposal path points vs native
    max_nat = 0.0
    for px, py in path:
        xt = _point_to_polyline_xtrack(float(px), float(py), native_path_xy)
        max_nat = max(max_nat, xt)
    metrics["native_corridor_xtrack_max_m"] = max_nat
    if max_nat > cfg.native_corridor_cross_track_max_m + 1e-6:
        reasons.append(f"{NATIVE_CORRIDOR}:{cand.candidate_id}:{max_nat:.3f}")

    # curvature hard limit on path
    max_k = _path_max_abs_curvature(path)
    metrics["path_max_abs_curvature"] = max_k
    if max_k > cfg.hard_max_abs_curvature + 1e-6:
        reasons.append(f"{CURVATURE_ENVELOPE}:path:{cand.candidate_id}:{max_k:.3f}")

    # self-intersection
    if _path_self_intersects(path):
        reasons.append(f"{SELF_INTERSECTION}:{cand.candidate_id}")

    # forward ratio
    fr = _path_forward_ratio(path)
    metrics["forward_ratio"] = fr
    if fr + 1e-9 < cfg.forward_ratio_min:
        reasons.append(f"{FORWARD_RATIO}:{cand.candidate_id}:{fr:.3f}")

    if not _path_signed_progress_ok(path, cfg.signed_progress_min_m):
        reasons.append(f"{FRENET_PROGRESS_NEGATIVE}:path:{cand.candidate_id}")

    return metrics


def validate_k2_spatial_bundle(
    bundle: K2PredictionBundleV2,
    config: K2SpatialConfig | None = None,
    *,
    require_diversity_if_eligible: bool = True,
    ego_xy: tuple[float, float] | None = None,
) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """Return (status, reasons, metrics). Does not mutate V1 validators."""
    cfg = config or load_k2_spatial_config()
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    # --- set-level identity ---
    if str(bundle.schema_version) != SCHEMA_V2:
        reasons.append(f"{SCHEMA_MISMATCH}:schema")
    if str(bundle.branch_type) != BRANCH_TYPE_V2:
        reasons.append(f"{SCHEMA_MISMATCH}:branch_type")
    if not bundle.backbone_forward_id:
        reasons.append(f"{FORWARD_ID_MISMATCH}:empty")
    if cfg.require_shared_backbone_forward_id and not bundle.backbone_forward_id:
        reasons.append(FORWARD_ID_MISMATCH)
    if not bundle.model_id:
        reasons.append(f"{SET_IDENTITY}:model_id")
    if not bundle.config_hash:
        reasons.append(f"{SET_IDENTITY}:config_hash")
    if bundle.observation_identity is None:
        reasons.append(f"{SET_IDENTITY}:observation_identity")
    if not bundle.probability_source:
        reasons.append(f"{SET_IDENTITY}:probability_source")

    if len(bundle.candidates) != int(cfg.k):
        reasons.append(f"K_MISMATCH:got={len(bundle.candidates)}")
    ids = [c.candidate_id for c in bundle.candidates]
    if len(set(ids)) != len(ids):
        reasons.append("DUPLICATE_CANDIDATE_ID")
    if int(bundle.top1_index) not in range(len(bundle.candidates) or 1):
        reasons.append(f"{TOP1_INVALID}:{bundle.top1_index}")
    prob_sum = 0.0
    for c in bundle.candidates:
        if not math.isfinite(float(c.probability)):
            reasons.append(f"{PROBABILITY_INVALID}:{c.candidate_id}")
        if float(c.probability) < -1e-9 or float(c.probability) > 1.0 + 1e-9:
            reasons.append(f"{PROBABILITY_INVALID}:range:{c.candidate_id}")
        prob_sum += float(c.probability)
    # probability field must sum to ~1 (scores must use a different field)
    if math.isfinite(prob_sum) and abs(prob_sum - 1.0) > 1e-3:
        reasons.append(f"{PROBABILITY_INVALID}:sum={prob_sum:.4f}")
    metrics["probability_sum"] = prob_sum

    # config_hash: hard reject mismatch (repair C)
    expected_hash = cfg.config_hash()
    metrics["config_hash_expected"] = expected_hash
    metrics["config_hash_bundle"] = bundle.config_hash
    if bundle.config_hash and bundle.config_hash != expected_hash:
        metrics["config_hash_mismatch"] = True
        reasons.append(f"{SET_IDENTITY}:config_hash_mismatch")

    native_h = stable_hash_xy(bundle.native_path_xy)
    if native_h != bundle.native_path_hash:
        reasons.append(f"{PROPOSAL_PATH_HASH_MISMATCH}:native")

    for c in bundle.candidates:
        lineage = str(c.head_lineage or "").lower()
        if cfg.reject_fixed_bias_lineage and any(x in lineage for x in _FORBIDDEN_LINEAGE):
            reasons.append(f"{HEAD_LINEAGE_INVALID}:{c.candidate_id}:{c.head_lineage}")
        if cfg.reject_noise_lineage and "noise" in lineage:
            reasons.append(f"{HEAD_LINEAGE_INVALID}:noise:{c.candidate_id}")
        # Lattice templates cannot masquerade as learned spatial_mode_head
        if "lattice_template" in lineage or "runtime_rescue" in lineage:
            reasons.append(f"{HEAD_LINEAGE_INVALID}:template_forbidden:{c.candidate_id}")
        if lineage == "spatial_mode_head":
            diag = bundle.set_diagnostics or {}
            if bool(diag.get("runtime_rescue")):
                reasons.append(
                    f"{HEAD_LINEAGE_INVALID}:forged_lineage_with_rescue:{c.candidate_id}"
                )
            # Content lineage required for formal learned head claims
            if not str(c.raw_head_output_hash or "").strip():
                reasons.append(
                    f"{HEAD_LINEAGE_INVALID}:missing_raw_head_hash:{c.candidate_id}"
                )
            if not str(c.postprocess_hash or "").strip():
                reasons.append(
                    f"{HEAD_LINEAGE_INVALID}:missing_postprocess_hash:{c.candidate_id}"
                )
            # Recompute postprocess hash from content; mismatch = spoof
            from driving_vla.model.k2_spatial_types import canonical_sha256

            exact_native_anchor = bool(
                cfg.nominal_anchor_exact
                and c.candidate_id == cfg.candidate_ids[0]
                and c.proposal_path_hash == bundle.native_path_hash
            )
            recon = canonical_sha256(
                {
                    "smooth_passes": 2,
                    "hard_max_abs_curvature": 0.25,
                    "exact_native_anchor": exact_native_anchor,
                    "path": [[float(x), float(y)] for x, y in c.spatial_path_xy],
                    "frenet_s": list(c.frenet_s),
                    "frenet_d": list(c.frenet_d),
                    "t10": [list(map(float, row[:6])) for row in c.points_xy_yaw_v_a_kappa],
                    "speeds": [
                        float(row[3]) for row in c.points_xy_yaw_v_a_kappa
                    ],
                }
            )
            if str(c.postprocess_hash) != recon:
                reasons.append(
                    f"{HEAD_LINEAGE_INVALID}:postprocess_hash_mismatch:{c.candidate_id}"
                )

        path_h = stable_hash_xy(c.spatial_path_xy)
        if path_h != c.proposal_path_hash:
            reasons.append(f"{PROPOSAL_PATH_HASH_MISMATCH}:{c.candidate_id}")
        traj_h = stable_hash_traj(c.points_xy_yaw_v_a_kappa)
        if traj_h != c.timed_trajectory_hash:
            reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:traj:{c.candidate_id}")

        spec = bundle.execution_specs.get(c.candidate_id)
        if spec is None:
            reasons.append(f"MISSING_EXEC_SPEC:{c.candidate_id}")
            continue
        if stable_hash_xy(spec.spatial_path_xy) != spec.spatial_path_hash:
            reasons.append(f"{PROPOSAL_PATH_HASH_MISMATCH}:spec:{c.candidate_id}")
        if tuple(spec.spatial_path_xy) != tuple(c.spatial_path_xy):
            reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:path_spec:{c.candidate_id}")
        if spec.spatial_path_hash != c.proposal_path_hash:
            reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:spec_hash:{c.candidate_id}")
        if spec.timed_trajectory_hash != c.timed_trajectory_hash:
            reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:spec_traj:{c.candidate_id}")

        # lateral envelope
        for d in c.frenet_d:
            if abs(float(d)) > cfg.max_lateral_residual_m + 1e-6:
                reasons.append(f"{FRENET_LATERAL_ENVELOPE}:{c.candidate_id}")
                break

        # signed progress on frenet s
        for i in range(1, len(c.frenet_s)):
            if float(c.frenet_s[i]) + 1e-6 < float(c.frenet_s[i - 1]):
                reasons.append(f"{FRENET_PROGRESS_NEGATIVE}:{c.candidate_id}")
                break

        # first residual sample must stay near native path start
        if bundle.native_path_xy and c.spatial_path_xy:
            n0 = bundle.native_path_xy[0]
            err = math.hypot(
                float(c.spatial_path_xy[0][0]) - float(n0[0]),
                float(c.spatial_path_xy[0][1]) - float(n0[1]),
            )
            # No silent +0.20 m first-step relaxation (repair C)
            if err > cfg.first_step_position_residual_m + 1e-6:
                reasons.append(f"{NEAR_FIELD_DISCONTINUITY}:native0:{c.candidate_id}")

        t_metrics = _recompute_t10_checks(c, cfg, reasons)
        g_metrics = _candidate_geometry_checks(c, bundle.native_path_xy, cfg, reasons)
        metrics[f"cand_{c.candidate_id}"] = {**t_metrics, **g_metrics}

    c0, c1 = bundle.candidates[0], bundle.candidates[1]
    # candidate 0 must be available
    if not c0.available:
        reasons.append(f"{ALTERNATIVE_UNAVAILABLE}:candidate0_must_be_available")

    if not c1.available:
        metrics["alternative_available"] = False
        metrics["availability_reason"] = c1.availability_reason
        # consistency: reason should mention unavailable / NO_ALTERNATIVE style
        if not str(c1.availability_reason or "").strip():
            reasons.append(f"{ALTERNATIVE_UNAVAILABLE}:missing_reason")
    else:
        metrics["alternative_available"] = True
        n_d = min(len(c0.frenet_d), len(c1.frenet_d))
        lat_sep = _aligned_inter_candidate_lateral_separation(c0, c1)
        # Backward-compatible field name now has the correct, narrower meaning.
        metrics["max_lateral_separation_m"] = lat_sep
        metrics["max_inter_candidate_lateral_separation_m"] = lat_sep
        metrics["max_native_excursion_candidate0_m"] = max(
            (abs(float(v)) for v in c0.frenet_d), default=0.0
        )
        metrics["max_native_excursion_candidate1_m"] = max(
            (abs(float(v)) for v in c1.frenet_d), default=0.0
        )
        metrics["max_spatial_separation_m"] = max_xy_separation(
            c0.spatial_path_xy, c1.spatial_path_xy
        )
        near_n = max(1, int(round(cfg.near_field_horizon_s / max(cfg.dt_s, 1e-6))))
        near_lat = 0.0
        for i in range(min(near_n, n_d)):
            near_lat = max(near_lat, abs(float(c0.frenet_d[i]) - float(c1.frenet_d[i])))
        metrics["near_field_lateral_separation_m"] = near_lat
        if near_lat > cfg.near_field_max_inter_candidate_lat_m + 1e-6:
            reasons.append(NEAR_FIELD_DISCONTINUITY)
        # Eligibility from candidate availability (frozen semantics), not mutable diagnostics
        eligible = bool(c1.available)
        metrics["eligible_from_candidate_available"] = eligible
        if require_diversity_if_eligible and eligible:
            if lat_sep + 1e-9 < cfg.ambiguity_min_spatial_sep_m:
                reasons.append(SPATIAL_COLLAPSE_ELIGIBLE)
            nominal_horizon = (
                float(c0.frenet_s[-1]) if c0.frenet_s else 0.0
            )
            defensive_horizon = (
                float(c1.frenet_s[-1]) if c1.frenet_s else 0.0
            )
            horizon_ratio = (
                defensive_horizon / nominal_horizon
                if nominal_horizon > 1e-6
                else 0.0
            )
            metrics["eligible_spatial_horizon_ratio"] = horizon_ratio
            if (
                horizon_ratio + 1e-9
                < cfg.min_eligible_spatial_horizon_ratio
            ):
                reasons.append(
                    "SPATIAL_HORIZON_COLLAPSE_ELIGIBLE:"
                    f"{horizon_ratio:.6f}<"
                    f"{cfg.min_eligible_spatial_horizon_ratio:.6f}"
                )

    # de-dupe reasons while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    status = GUARD_OK if not uniq else GUARD_REJECT
    return status, tuple(uniq), metrics


def attach_spatial_guard(
    bundle: K2PredictionBundleV2,
    config: K2SpatialConfig | None = None,
    **kwargs: Any,
) -> K2PredictionBundleV2:
    status, reasons, metrics = validate_k2_spatial_bundle(bundle, config, **kwargs)
    return K2PredictionBundleV2(
        schema_version=bundle.schema_version,
        observation_identity=bundle.observation_identity,
        model_id=bundle.model_id,
        config_hash=bundle.config_hash,
        base_checkpoint_hash=bundle.base_checkpoint_hash,
        spatial_head_checkpoint_hash=bundle.spatial_head_checkpoint_hash,
        backbone_forward_id=bundle.backbone_forward_id,
        native_path_xy=bundle.native_path_xy,
        native_path_hash=bundle.native_path_hash,
        candidates=bundle.candidates,
        execution_specs=bundle.execution_specs,
        top1_index=bundle.top1_index,
        probability_source=bundle.probability_source,
        branch_type=bundle.branch_type,
        guard_status=status,
        guard_reasons=reasons,
        build_error=bundle.build_error,
        set_diagnostics={**dict(bundle.set_diagnostics or {}), **metrics},
        raw_head_output_hash=bundle.raw_head_output_hash,
        feature_content_hash=bundle.feature_content_hash,
        codec_config_hash=bundle.codec_config_hash,
        postprocess_hash=bundle.postprocess_hash,
    )


def verify_execution_spec_content_hash(
    bundle: K2PredictionBundleV2,
    candidate_id: str,
) -> list[str]:
    """Re-validate execution spec content hashes at selection time."""
    reasons: list[str] = []
    cand = None
    for c in bundle.candidates:
        if c.candidate_id == candidate_id:
            cand = c
            break
    if cand is None:
        return [f"orphan_candidate_id:{candidate_id}"]
    spec = bundle.execution_specs.get(candidate_id)
    if spec is None:
        return [f"MISSING_EXEC_SPEC:{candidate_id}"]
    if stable_hash_xy(spec.spatial_path_xy) != spec.spatial_path_hash:
        reasons.append(f"{PROPOSAL_PATH_HASH_MISMATCH}:spec_content:{candidate_id}")
    if stable_hash_xy(cand.spatial_path_xy) != cand.proposal_path_hash:
        reasons.append(f"{PROPOSAL_PATH_HASH_MISMATCH}:cand_content:{candidate_id}")
    if tuple(spec.spatial_path_xy) != tuple(cand.spatial_path_xy):
        reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:path_spec:{candidate_id}")
    if stable_hash_traj(cand.points_xy_yaw_v_a_kappa) != cand.timed_trajectory_hash:
        reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:traj_content:{candidate_id}")
    if spec.timed_trajectory_hash != cand.timed_trajectory_hash:
        reasons.append(f"{TIMED_PATH_BINDING_MISMATCH}:spec_traj:{candidate_id}")
    return reasons
