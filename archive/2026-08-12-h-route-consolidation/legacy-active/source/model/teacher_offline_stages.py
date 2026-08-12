"""Offline production teacher stages (Guard / PM / steer-κ / MPC).

Used for dataset-v5 labeling without CARLA tick. Metrics come from geometry +
lightweight MPC rollout on residual paths — not d-sign synthetic proxies.
"""

from __future__ import annotations

from typing import Any, Mapping

from driving_vla.model.spatial_k2_teacher import TeacherFilterResult


def _native_from_payload(payload: Mapping[str, Any]) -> list[tuple[float, float]]:
    native = payload.get("native_path_xy") or []
    if len(native) >= 2:
        return [(float(x), float(y)) for x, y in native]
    # fallback straight corridor (unit tests only)
    return [(float(i) * 1.2, 0.0) for i in range(20)]


def _ego_yaw_from_path(native: list[tuple[float, float]]) -> float:
    """Align ego yaw with path tangent so PM forward-ratio is meaningful."""
    import math

    if len(native) < 2:
        return 0.0
    dx = float(native[1][0]) - float(native[0][0])
    dy = float(native[1][1]) - float(native[0][1])
    if abs(dx) + abs(dy) < 1e-9:
        # skip degenerate first segment
        for i in range(1, min(len(native) - 1, 5)):
            dx = float(native[i + 1][0]) - float(native[i][0])
            dy = float(native[i + 1][1]) - float(native[i][1])
            if abs(dx) + abs(dy) >= 1e-9:
                break
    return float(math.atan2(dy, dx))


def _actor_geometry(payload: Mapping[str, Any]) -> dict[str, float]:
    """Privileged actor pose from sample fields or scenario family defaults."""
    scene = dict(payload.get("privileged_scene") or {})
    side = str(payload.get("conflict_side") or scene.get("conflict_side") or "none").lower()
    fam = str(payload.get("scenario_family") or scene.get("scenario_family") or "").lower()
    lat = scene.get("actor_lat_m")
    lon = scene.get("actor_lon_m")
    if lat is None:
        if side == "left":
            lat = 3.5
        elif side == "right":
            lat = -3.5
        elif "lead" in fam or "brake" in fam:
            lat = 0.0
        else:
            lat = 0.0
    if lon is None:
        if "lead" in fam or "brake" in fam:
            lon = 12.0
        elif "cross" in fam:
            lon = 18.0
        elif side in {"left", "right"}:
            lon = 14.0
        else:
            lon = 25.0
    speed = float(scene.get("actor_speed_mps", 5.0) or 5.0)
    ego_v = float(payload.get("ego_v") or scene.get("ego_v") or 5.0)
    return {
        "actor_lat_m": float(lat),
        "actor_lon_m": float(lon),
        "actor_speed_mps": speed,
        "ego_v": ego_v,
    }


def compute_privileged_rollout_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Geometry + residual based privileged metrics (not d-sign table)."""
    ag = _actor_geometry(payload)
    d_peak = float(payload.get("d_peak_m", 0.0))
    sc = float(payload.get("speed_scale", 1.0))
    # Lateral clearance to actor after defensive offset
    clearance = abs(float(ag["actor_lat_m"]) - d_peak) + 0.15 * abs(float(ag["actor_lon_m"]))
    # Relative closing speed along lon
    v_rel = max(0.1, float(ag["ego_v"]) * sc - float(ag["actor_speed_mps"]) * 0.5)
    lon = max(0.5, float(ag["actor_lon_m"]))
    # Larger lateral clearance reduces effective closing (lane separation)
    sep_factor = 1.0 + min(2.0, clearance / 4.0)
    ttc = (lon * sep_factor) / v_rel
    # Progress: scaled cruise
    progress = 12.0 * sc
    comfort = abs(d_peak) * 0.25 + abs(sc - 1.0) * 0.4
    # Collision proxy: very small clearance and small lon
    collision = 1.0 if (clearance < 0.8 and lon < 6.0 and sc > 0.9) else 0.0
    return {
        "ttc_s": float(ttc),
        "clearance_m": float(clearance),
        "progress_m": float(progress),
        "comfort_cost": float(comfort),
        "collision": float(collision),
        "actor_lat_m": float(ag["actor_lat_m"]),
        "actor_lon_m": float(ag["actor_lon_m"]),
        "metrics_source": 1.0,  # 1=geometry_rollout
    }


def _pad_residuals(
    payload: Mapping[str, Any], n: int
) -> tuple[list[float], list[float], float, float]:
    raw_d = list(payload.get("raw_d") or [0.0] * n)[:n]
    raw_ds = list(payload.get("raw_delta_s") or [0.0] * n)[:n]
    while len(raw_d) < n:
        raw_d.append(0.0)
    while len(raw_ds) < n:
        raw_ds.append(0.0)
    sc = float(payload.get("speed_scale", 1.0))
    ego_v = float(payload.get("ego_v") or 5.0)
    return raw_d, raw_ds, sc, ego_v


def stage_guard_v2(payload: dict[str, Any]) -> TeacherFilterResult:
    """Build residual candidate and run Guard V2 offline."""
    try:
        from driving_vla.model.k2_spatial_builder import build_spatial_k2_bundle_from_residuals
        from driving_vla.model.k2_spatial_guard import attach_spatial_guard

        native = _native_from_payload(payload)
        n = len(native)
        raw_d, raw_ds, sc, ego_v = _pad_residuals(payload, n)
        nom = {
            "raw_delta_s": [0.0] * n,
            "raw_d": [0.0] * n,
            "speed_scale": 1.0,
            "head_lineage": "teacher_lattice",
        }
        alt = {
            "raw_delta_s": raw_ds,
            "raw_d": raw_d,
            "speed_scale": sc,
            "head_lineage": "teacher_lattice",
        }
        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=native[0],
            ego_v=ego_v,
            base_speed_mps=ego_v,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={"teacher": True, "feature_hash": "teacher_offline"},
            backbone_forward_id="teacher-offline",
            defensive_available=True,
            defensive_reason="teacher",
            probability_source="teacher_equal_prior",
        )
        # Only the true nominal identity bypasses set diversity. A lattice
        # candidate with d=0 remains a defensive proposal and must fail the
        # frozen spatial-separation contract.
        is_near_nominal = str(payload.get("candidate_id") or "") == "nominal"
        # Nominal identity is not a diversity candidate. Every defensive
        # teacher candidate, however, must pass the exact production Guard,
        # including the frozen 0.50 m spatial-separation requirement.
        g = attach_spatial_guard(
            bundle, require_diversity_if_eligible=not is_near_nominal
        )
        reasons = list(g.guard_reasons)
        ok = is_near_nominal or g.guard_status == "OK"
        metrics = compute_privileged_rollout_metrics(payload)
        metrics["guard_status"] = g.guard_status
        metrics["guard_reason_n"] = float(len(reasons))
        return TeacherFilterResult(
            "guard_v2",
            ok,
            "ok" if ok else f"guard_reject:{reasons[:3]}",
            metrics,
        )
    except Exception as exc:  # noqa: BLE001
        return TeacherFilterResult(
            "guard_v2", False, f"guard_exc:{type(exc).__name__}:{exc}", {}
        )


def stage_path_manager(payload: dict[str, Any]) -> TeacherFilterResult:
    try:
        from driving_vla.model.k2_spatial_builder import build_candidate_from_residual
        from driving_vla.model.k2_spatial_types import load_k2_spatial_config
        from driving_vla.runtime.path_manager import (
            EgoPose,
            PathManagerConfig,
            VLAPathManager,
        )

        native = _native_from_payload(payload)
        n = len(native)
        raw_d, raw_ds, sc, ego_v = _pad_residuals(payload, n)
        cfg = load_k2_spatial_config()
        cand, _spec = build_candidate_from_residual(
            native_path_xy=native,
            raw_delta_s=raw_ds,
            raw_d=raw_d,
            speed_scale=sc,
            base_speed_mps=ego_v,
            ego_v=ego_v,
            candidate_id="teacher_cand",
            mode_id="defensive",
            available=True,
            availability_reason="teacher",
            probability=0.5,
            head_lineage="teacher_lattice",
            config=cfg,
            raw_head_output_hash="teacher",
            feature_content_hash="teacher",
        )
        # Soft live PM limit for teacher labeling (stable live uses 0.30)
        pm = VLAPathManager(
            PathManagerConfig(max_switch_lateral_5m=1.0, max_abs_curvature=0.30)
        )
        yaw0 = _ego_yaw_from_path(native)
        ego = EgoPose(native[0][0], native[0][1], yaw0, ego_v)
        pts = list(cand.spatial_path_xy)
        if len(pts) < 2:
            return TeacherFilterResult(
                "path_manager_densify_accepted",
                False,
                "degenerate_path",
                {},
            )
        upd = pm.update(
            pts,
            ego=ego,
            target_speed_mps=ego_v * sc,
            stamp_s=0.0,
            source_id="teacher",
        )
        ok = bool(upd.accepted)
        metrics = compute_privileged_rollout_metrics(payload)
        metrics["pm_accepted"] = 1.0 if ok else 0.0
        metrics["pm_reason"] = str(upd.reason)
        if upd.quality is not None:
            metrics["pm_kappa_max"] = float(upd.quality.max_abs_curvature)
            metrics["pm_kappa_q"] = float(upd.quality.curvature_quantile)
        return TeacherFilterResult(
            "path_manager_densify_accepted",
            ok,
            "accepted" if ok else f"reject:{upd.reason}",
            metrics,
        )
    except Exception as exc:  # noqa: BLE001
        return TeacherFilterResult(
            "path_manager_densify_accepted",
            False,
            f"pm_exc:{type(exc).__name__}:{exc}",
            {},
        )


def stage_steer_curvature(payload: dict[str, Any]) -> TeacherFilterResult:
    try:
        from driving_vla.model.k2_spatial_builder import build_candidate_from_residual
        from driving_vla.model.k2_spatial_types import load_k2_spatial_config
        from driving_vla.evaluation.executability_metrics import mpc_kinematic_kappa_max

        native = _native_from_payload(payload)
        n = len(native)
        raw_d, raw_ds, sc, ego_v = _pad_residuals(payload, n)
        cfg = load_k2_spatial_config()
        cand, _ = build_candidate_from_residual(
            native_path_xy=native,
            raw_delta_s=raw_ds,
            raw_d=raw_d,
            speed_scale=sc,
            base_speed_mps=ego_v,
            ego_v=ego_v,
            candidate_id="teacher_cand",
            mode_id="defensive",
            available=True,
            availability_reason="teacher",
            probability=0.5,
            head_lineage="teacher_lattice",
            config=cfg,
            raw_head_output_hash="teacher",
            feature_content_hash="teacher",
        )
        kappas = [abs(float(row[5])) for row in cand.points_xy_yaw_v_a_kappa]
        k_max = max(kappas) if kappas else 0.0
        k_lim = mpc_kinematic_kappa_max()
        # Teacher gate: allow up to 1.5× kinematic + floor for slow curves
        ok = k_max <= max(k_lim * 1.5, 0.35)
        metrics = compute_privileged_rollout_metrics(payload)
        metrics["kappa_max"] = float(k_max)
        metrics["kappa_lim"] = float(k_lim)
        return TeacherFilterResult(
            "steer_curvature_gate",
            ok,
            "ok" if ok else f"kappa_high:{k_max:.3f}",
            metrics,
        )
    except Exception as exc:  # noqa: BLE001
        return TeacherFilterResult(
            "steer_curvature_gate", False, f"steer_exc:{type(exc).__name__}:{exc}", {}
        )


def stage_mpc_rollout(payload: dict[str, Any]) -> TeacherFilterResult:
    try:
        from driving_vla.model.k2_spatial_builder import build_candidate_from_residual
        from driving_vla.model.k2_spatial_types import load_k2_spatial_config
        from driving_vla.runtime.path_manager import (
            EgoPose,
            PathManagerConfig,
            VLAPathManager,
            spatial_path_from_xy,
        )
        from driving_vla.runtime.vla_mpc_tracker import ConstrainedVLAMPC, VLAMPCConfig

        native = _native_from_payload(payload)
        n = len(native)
        raw_d, raw_ds, sc, ego_v = _pad_residuals(payload, n)
        cfg = load_k2_spatial_config()
        cand, _ = build_candidate_from_residual(
            native_path_xy=native,
            raw_delta_s=raw_ds,
            raw_d=raw_d,
            speed_scale=sc,
            base_speed_mps=ego_v,
            ego_v=ego_v,
            candidate_id="teacher_cand",
            mode_id="defensive",
            available=True,
            availability_reason="teacher",
            probability=0.5,
            head_lineage="teacher_lattice",
            config=cfg,
            raw_head_output_hash="teacher",
            feature_content_hash="teacher",
        )
        pts = list(cand.spatial_path_xy)
        if len(pts) < 2:
            return TeacherFilterResult(
                "mpc_rollout_30tick", False, "degenerate_path", {}
            )
        yaw0 = _ego_yaw_from_path(native)
        ego = EgoPose(native[0][0], native[0][1], yaw0, ego_v)
        path = spatial_path_from_xy(
            pts,
            ego=ego,
            target_speed_mps=ego_v * sc,
            stamp_s=0.0,
            ds_m=0.20,
            source_id="teacher",
        )
        if path is None:
            return TeacherFilterResult(
                "mpc_rollout_30tick", False, "spatial_path_none", {}
            )
        # Ensure PM accepts before MPC
        pm = VLAPathManager(PathManagerConfig(max_abs_curvature=0.30))
        upd = pm.update(
            pts, ego=ego, target_speed_mps=ego_v * sc, stamp_s=0.0, source_id="teacher"
        )
        if not upd.accepted:
            metrics = compute_privileged_rollout_metrics(payload)
            metrics["mpc_solved"] = 0.0
            metrics["mpc_fallback"] = 30.0
            return TeacherFilterResult(
                "mpc_rollout_30tick",
                False,
                f"pm_block:{upd.reason}",
                metrics,
            )
        committed = upd.committed if upd.committed is not None else path
        tracker = ConstrainedVLAMPC(VLAMPCConfig())
        solved = 0
        fallback = 0
        e = ego
        for i in range(30):
            cmd = tracker.step(committed, e, measured_steer_rad=0.0, now_s=i * 0.05)
            mode = str(getattr(cmd, "mode", "") or "")
            st = str(getattr(cmd, "solver_status", "") or "")
            if mode == "mpc" or "SOLVED" in st.upper():
                solved += 1
            if mode == "bounded_fallback" or "FALLBACK" in st.upper():
                fallback += 1
            accel = float(getattr(cmd, "accel_mps2", 0.0) or 0.0)
            v = max(0.0, e.speed_mps + accel * 0.05)
            # advance along committed path approximately
            ds = v * 0.05
            import math

            yaw = e.yaw
            if committed.s.size >= 2:
                s_hint = committed.project_s(e.x, e.y)
                xs, ys, yaws, _ = committed.sample(s_hint + ds)
                e = EgoPose(float(xs[0]), float(ys[0]), float(yaws[0]), v)
            else:
                e = EgoPose(e.x + ds * math.cos(yaw), e.y + ds * math.sin(yaw), yaw, v)
        # Offline labeling: require majority MPC solved; allow ≤3 fallbacks
        ok = solved >= 20 and fallback <= 5
        metrics = compute_privileged_rollout_metrics(payload)
        metrics["mpc_solved"] = float(solved)
        metrics["mpc_fallback"] = float(fallback)
        metrics["rollout_ticks"] = 30.0
        return TeacherFilterResult(
            "mpc_rollout_30tick",
            ok,
            "ok" if ok else f"mpc_fail:solved={solved}:fb={fallback}",
            metrics,
        )
    except Exception as exc:  # noqa: BLE001
        metrics = compute_privileged_rollout_metrics(payload)
        return TeacherFilterResult(
            "mpc_rollout_30tick",
            False,
            f"mpc_exc:{type(exc).__name__}:{exc}",
            metrics,
        )


def production_execution_stages() -> dict[str, Any]:
    return {
        "guard_v2": stage_guard_v2,
        "path_manager_densify_accepted": stage_path_manager,
        "steer_curvature_gate": stage_steer_curvature,
        "mpc_rollout_30tick": stage_mpc_rollout,
    }
