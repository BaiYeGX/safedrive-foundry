#!/usr/bin/env python3
"""R2-X learned force smoke (offline, no CARLA) with layered executability.

Denominators (strict):
  label_eligible → pred_available → guard_ok → attempted (both PM accepted)
  FN and Guard rejects are first-class, never mixed into attempted.

Executability layers (prefilter only, not CARLA substitute):
  - conservative_static_live_prep: max_v²×max_κ (diagnostic; over-pessimistic)
  - arc_aligned a_y: max_i(v_i²|κ_i|)
  - mpc_capped a_y: curve speed limit as ConstrainedVLAMPC
  - densified PM Q90/local-max from path_update.raw/committed.kappa
  - live_prefilter: accepted + densified PM + steer-κ + mpc_capped a_y

Bootstrap scene_proxy heads are not R2-K authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.executability_metrics import (  # noqa: E402
    DEFAULT_MAX_LAT_ACCEL_MPS2,
    DEFAULT_MEANINGFUL_SPEED_MPS,
    DEFAULT_PM_HARD_KAPPA,
    DEFAULT_PM_SOFT_KAPPA,
    evaluate_branch_executability,
    evaluate_mpc_rollout_executability,
    mpc_kinematic_kappa_max,
    semantics_dict,
)
from driving_vla.evaluation.k2_spatial_artifact import (  # noqa: E402
    artifact_from_bundle_v2,
    bundle_from_artifact_v2,
    make_dummy_observation_fingerprint,
)
from driving_vla.evaluation.paired_contract import content_hash  # noqa: E402
from driving_vla.model.k2_spatial_builder import build_spatial_k2_bundle_from_residuals  # noqa: E402
from driving_vla.model.k2_spatial_guard import attach_spatial_guard  # noqa: E402
from driving_vla.model.spatial_mode_heads import SpatialK2HeadRuntime  # noqa: E402
from driving_vla.runtime.k2_execution import apply_k2_to_executors, select_k2_spatial  # noqa: E402
from driving_vla.runtime.path_manager import EgoPose, PathManagerConfig, VLAPathManager  # noqa: E402
from driving_vla.runtime.vla_mpc_tracker import ConstrainedVLAMPC, VLAMPCConfig  # noqa: E402
from driving_vla.runtime.vla_speed_planner import VLASpeedPlanner  # noqa: E402

DATA_V4 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real/samples.jsonl"
DATA_V2 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v2/samples.jsonl"
DATA = DATA_V4 if DATA_V4.is_file() else DATA_V2
CKPT = ROOT / "docs/runtime-evidence/r2x-training/checkpoints/spatial_heads_last.pt"
OUT = ROOT / "docs/runtime-evidence/r2x-learned-force-smoke"

STEER_MAE_MEANINGFUL = 1e-3


def _path_hash(spatial) -> str:
    if spatial is None:
        return "none"
    arr = np.stack([spatial.x, spatial.y], axis=-1).astype(np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _run(bundle, force: int, ego_xy, ego_v: float, n_ticks: int = 30):
    sel = select_k2_spatial(bundle, mode="force", force_index=force)
    pm = VLAPathManager(
        PathManagerConfig(
            max_switch_lateral_5m=1.0,
            hard_max_abs_curvature=DEFAULT_PM_HARD_KAPPA,
            max_abs_curvature=DEFAULT_PM_SOFT_KAPPA,
        )
    )
    sp = VLASpeedPlanner()
    mpc_cfg = VLAMPCConfig()
    mpc = ConstrainedVLAMPC(mpc_cfg)
    path = sel.execution_spec.spatial_path_xy
    yaw0 = 0.0
    if len(path) >= 2:
        yaw0 = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    ego = EgoPose(float(ego_xy[0]), float(ego_xy[1]), yaw0, float(ego_v))
    applied = apply_k2_to_executors(
        sel,
        speed_planner=sp,
        path_manager=pm,
        ego=ego,
        stamp_s=0.0,
        frame_id=f"learned-f{force}",
        dt_s=0.05,
        ego_speed_mps=ego_v,
        nav_target_map_xy=(float(path[-1][0]), float(path[-1][1])) if path else None,
    )
    controls = []
    rollout_ticks = []
    x, y, yaw, v = ego.x, ego.y, ego.yaw, ego.speed_mps
    steer = 0.0
    for i in range(n_ticks):
        pose = EgoPose(x, y, yaw, v)
        if pm.committed is None:
            break
        # Optional: re-nudge path target with later speed samples so planner can climb
        # (still uses real MPC command fields for the gate, not static path_target alone).
        cmd = mpc.step(pm.committed, pose, measured_steer_rad=steer, now_s=i * 0.05)
        steer = float(cmd.steer_rad)
        accel = float(cmd.accel_mps2)
        tick = {
            "t": float(i) * 0.05,
            "speed_mps": float(v),
            "steer_rad": steer,
            "accel_mps2": accel,
            "reference_curvature": float(cmd.reference_curvature),
            "curve_speed_limit_mps": float(cmd.curve_speed_limit_mps),
            "target_speed_mps": float(cmd.target_speed_mps),
            "horizon_speed_limit_mps": float(cmd.horizon_speed_limit_mps),
            "mode": str(cmd.mode),
            "solver_status": str(cmd.solver_status),
            "lateral_error_m": float(cmd.lateral_error_m),
        }
        rollout_ticks.append(tick)
        controls.append({"steer_rad": steer, "accel_mps2": accel, "speed_mps": v})
        x = x + v * math.cos(yaw) * 0.05
        y = y + v * math.sin(yaw) * 0.05
        yaw = yaw + v / 2.7 * math.tan(steer) * 0.05
        v = max(0.0, v + accel * 0.05)

    # T10 projection layers (arc-aligned diagnostic) + densified PM
    t10 = None
    for c in bundle.candidates:
        if c.candidate_id == sel.candidate_id:
            t10 = c.points_xy_yaw_v_a_kappa
            break
    exec_rep = evaluate_branch_executability(
        path_xy=path,
        speed_samples_mps=list(sel.execution_spec.speed_samples_mps),
        ego_v=float(ego_v),
        path_manager_accepted=bool(applied.path_update.accepted),
        raw_spatial_path=applied.path_update.raw,
        committed_spatial_path=applied.path_update.committed,
        t10_points_xy_yaw_v=t10,
        pm_soft_kappa=DEFAULT_PM_SOFT_KAPPA,
        pm_hard_kappa=DEFAULT_PM_HARD_KAPPA,
        max_lat_accel_mps2=float(mpc_cfg.max_lateral_accel_mps2),
        max_speed_mps=float(mpc_cfg.max_speed_mps),
        max_brake_mps2=float(mpc_cfg.max_brake_mps2),
        path_end_margin_m=float(mpc_cfg.path_end_margin_m),
        curve_limit_quantile=float(mpc_cfg.curve_limit_quantile),
        horizon=int(mpc_cfg.horizon),
        prediction_dt_s=float(mpc_cfg.prediction_dt_s),
        min_linearization_speed_mps=float(mpc_cfg.min_linearization_speed_mps),
    )
    # Tracker-faithful gate: real 30-tick MPC rollout (not near-zero path_target)
    rollout_rep = evaluate_mpc_rollout_executability(
        rollout_ticks,
        max_lat_accel_mps2=float(mpc_cfg.max_lateral_accel_mps2),
        meaningful_speed_mps=DEFAULT_MEANINGFUL_SPEED_MPS,
    )
    return {
        "candidate_id": sel.candidate_id,
        "proposal_hash": sel.execution_spec.spatial_path_hash,
        "committed_hash": _path_hash(applied.path_update.committed),
        "raw_hash": _path_hash(applied.path_update.raw),
        "accepted": bool(applied.path_update.accepted),
        "path_reason": str(applied.path_update.reason),
        "controls": controls,
        "rollout_ticks": rollout_ticks,
        "source_id": applied.source_id,
        "executability": exec_rep.to_dict(),
        "tracker_rollout": rollout_rep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=0, help="0=all label-eligible val")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.is_file() or not CKPT.is_file():
        print("missing data or checkpoint")
        return 2

    runtime = SpatialK2HeadRuntime(device="cpu", checkpoint_path=str(CKPT))
    rows = []
    with DATA.open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s.get("split_id") == "val" and s.get("alternative_available"):
                rows.append(s)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    n_label_eligible = len(rows)
    n_pred_available = 0
    n_false_negative = 0
    n_guard_ok = 0
    n_guard_reject = 0
    n_attempted = 0
    n_path_div = 0
    n_ctrl_div = 0
    n_cons_static_both = 0
    n_arc_aligned_both = 0
    n_ideal_pointwise_diag = 0
    n_static_horizon_deprecated = 0
    n_tracker_rollout_both = 0
    n_pm_steer_both = 0
    n_live_prefilter_both = 0
    n_t10_proj = 0
    n_rollout_coverage_fail = 0
    results = []

    for s in rows:
        native = [tuple(p) for p in s["native_path_xy"]]
        o0, o1 = runtime.predict_modes(
            native,
            ego_v=float(s["ego_v"]),
            base_speed_mps=float(s["base_speed_mps"]),
            sample=s,
        )
        if not o1.available:
            n_false_negative += 1
            results.append(
                {
                    "sample_id": s["sample_id"],
                    "label_eligible": True,
                    "pred_available": False,
                    "skipped": "pred_unavailable_false_negative",
                }
            )
            continue
        n_pred_available += 1
        b = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=tuple(native[0]),
            ego_v=float(s["ego_v"]),
            base_speed_mps=float(s["base_speed_mps"]),
            residual_nominal={
                "raw_delta_s": o0.raw_delta_s,
                "raw_d": o0.raw_d,
                "speed_scale": o0.speed_scale,
                "head_lineage": "spatial_mode_head",
            },
            residual_defensive={
                "raw_delta_s": o1.raw_delta_s,
                "raw_d": o1.raw_d,
                "speed_scale": o1.speed_scale,
                "head_lineage": "spatial_mode_head",
            },
            observation_identity={"sample_id": s["sample_id"]},
            backbone_forward_id=f"learned-{s['sample_id']}",
            spatial_head_checkpoint_hash=runtime.spatial_head_checkpoint_hash,
            defensive_available=True,
        )
        b = replace(b, set_diagnostics={"eligible_for_diversity": True})
        g = attach_spatial_guard(b, require_diversity_if_eligible=True)
        if g.guard_status != "OK":
            n_guard_reject += 1
            results.append(
                {
                    "sample_id": s["sample_id"],
                    "label_eligible": True,
                    "pred_available": True,
                    "skipped": "guard_reject",
                    "reasons": list(g.guard_reasons),
                }
            )
            continue
        n_guard_ok += 1
        fp = make_dummy_observation_fingerprint(
            k2_bundle_hash=content_hash({"s": s["sample_id"]}, nibble=16)
        )
        art = artifact_from_bundle_v2(
            g,
            pair_id=f"learned-{s['sample_id']}",
            scenario_id=str(s.get("scenario_family") or "val"),
            seed_id="seed_a",
            anchor_run_id="learned-anchor",
            anchor_carla_frame=0,
            anchor_simulation_time_s=0.0,
            requested_initial_state_hash="0" * 64,
            measured_initial_state_hash="0" * 64,
            observation_fingerprint=fp,
            model_checkpoint_hash=runtime.spatial_head_checkpoint_hash,
            executor_config_hash="offline_learned_smoke_layered_v4",
            evidence_lineage="spatial_mode_head",
        )
        cold = bundle_from_artifact_v2(art)
        cold = replace(
            cold,
            guard_status=art.guard_status,
            guard_reasons=art.guard_reasons,
            set_diagnostics=dict(art.set_diagnostics),
        )
        try:
            r0 = _run(cold, 0, native[0], float(s["ego_v"]))
            r1 = _run(cold, 1, native[0], float(s["ego_v"]))
        except Exception as exc:
            results.append(
                {
                    "sample_id": s["sample_id"],
                    "label_eligible": True,
                    "pred_available": True,
                    "guard_ok": True,
                    "skipped": f"exec_error:{type(exc).__name__}",
                    "error": str(exc)[:200],
                }
            )
            continue

        both_acc = bool(r0["accepted"] and r1["accepted"])
        if not both_acc:
            results.append(
                {
                    "sample_id": s["sample_id"],
                    "label_eligible": True,
                    "pred_available": True,
                    "guard_ok": True,
                    "skipped": "path_manager_reject",
                    "branch_0_reason": r0["path_reason"],
                    "branch_1_reason": r1["path_reason"],
                    "executability_0": r0["executability"],
                    "executability_1": r1["executability"],
                }
            )
            continue

        # attempted := guard_ok AND both PathManager accepted
        n_attempted += 1
        n = min(len(r0["controls"]), len(r1["controls"]))
        steer_mae = (
            sum(abs(r0["controls"][i]["steer_rad"] - r1["controls"][i]["steer_rad"]) for i in range(n))
            / max(n, 1)
        )
        proposal_differ = r0["proposal_hash"] != r1["proposal_hash"]
        committed_differ = (
            r0["committed_hash"] != r1["committed_hash"]
            and r0["committed_hash"] != "none"
            and r1["committed_hash"] != "none"
        )
        path_diverge = proposal_differ and committed_differ
        control_diverge = path_diverge and steer_mae >= STEER_MAE_MEANINGFUL
        e0, e1 = r0["executability"], r1["executability"]
        tr0, tr1 = r0["tracker_rollout"], r1["tracker_rollout"]
        cons_both = bool(e0["conservative_static_live_prep"] and e1["conservative_static_live_prep"])
        arc_both = bool(
            e0["pass_arc_aligned_ay"]
            and e1["pass_arc_aligned_ay"]
            and e0["pass_mpc_steer_kappa"]
            and e1["pass_mpc_steer_kappa"]
        )
        # Deprecated static horizon (path_target near-zero) — record only, never gate
        static_horizon_deprecated_both = bool(
            e0.get("pass_tracker_longitudinal_feasibility")
            and e1.get("pass_tracker_longitudinal_feasibility")
        )
        # Real tracker gate: 30-tick MPC rollout
        tracker_rollout_both = bool(
            tr0.get("pass_tracker_rollout") and tr1.get("pass_tracker_rollout")
        )
        pm_steer_both = bool(e0["pm_steer_prefilter"] and e1["pm_steer_prefilter"])
        # LIVE_PREFILTER requires PM+steer AND real rollout gate (not static path_target)
        live_pf_both = bool(pm_steer_both and tracker_rollout_both)
        if e0.get("t10_projection_used") and e1.get("t10_projection_used"):
            n_t10_proj += 1
        if path_diverge:
            n_path_div += 1
        if control_diverge:
            n_ctrl_div += 1
        if cons_both:
            n_cons_static_both += 1
        if arc_both:
            n_arc_aligned_both += 1
        n_ideal_pointwise_diag += 1
        if static_horizon_deprecated_both:
            n_static_horizon_deprecated += 1
        if tracker_rollout_both:
            n_tracker_rollout_both += 1
        else:
            # classify coverage shortfall for diagnostics
            if not (
                tr0.get("meaningful_speed_coverage") and tr1.get("meaningful_speed_coverage")
            ):
                n_rollout_coverage_fail += 1
        if pm_steer_both:
            n_pm_steer_both += 1
        if live_pf_both:
            n_live_prefilter_both += 1

        results.append(
            {
                "sample_id": s["sample_id"],
                "label_eligible": True,
                "pred_available": True,
                "guard_ok": True,
                "attempted": True,
                "both_accepted": True,
                "proposal_differ": proposal_differ,
                "committed_differ": committed_differ,
                "steer_mae": steer_mae,
                "path_diverge": path_diverge,
                "control_diverge": control_diverge,
                "conservative_static_live_prep_both": cons_both,
                "arc_aligned_feasibility_both": arc_both,
                "ideal_pointwise_cap_diagnostic": "IDEAL_POINTWISE_CAP_DIAGNOSTIC",
                "static_tracker_horizon_deprecated_both": static_horizon_deprecated_both,
                "tracker_rollout_feasibility_both": tracker_rollout_both,
                "pm_steer_prefilter_both": pm_steer_both,
                "live_prefilter_both": live_pf_both,
                "t10_projection_used": bool(
                    e0.get("t10_projection_used") and e1.get("t10_projection_used")
                ),
                "executability_0": e0,
                "executability_1": e1,
                "tracker_rollout_0": tr0,
                "tracker_rollout_1": tr1,
                "branch_0": {
                    k: v
                    for k, v in r0.items()
                    if k not in {"controls", "executability", "rollout_ticks", "tracker_rollout"}
                },
                "branch_1": {
                    k: v
                    for k, v in r1.items()
                    if k not in {"controls", "executability", "rollout_ticks", "tracker_rollout"}
                },
                "evidence_lineage": "spatial_mode_head",
            }
        )

    def rate(num: int, den: int) -> float:
        return num / max(den, 1)

    path_rate = rate(n_path_div, n_attempted)
    ctrl_rate = rate(n_ctrl_div, n_attempted)
    pipeline_ok = (
        n_attempted >= max(1, int(math.ceil(0.5 * n_label_eligible)))
        and n_attempted > 0
        and path_rate >= 1.0 - 1e-12
        and ctrl_rate >= 1.0 - 1e-12
    )
    pm_steer_ok = pipeline_ok and rate(n_pm_steer_both, n_attempted) >= 1.0 - 1e-12
    live_pf_ok = pipeline_ok and rate(n_live_prefilter_both, n_attempted) >= 1.0 - 1e-12
    # Honest hierarchy: LIVE only if real 30-tick rollout gate passes with speed coverage
    if live_pf_ok:
        status = "LIVE_PREFILTER_PASS"
    elif pm_steer_ok:
        status = "PM_STEER_PREFILTER_PASS"
    elif pipeline_ok:
        status = "PIPELINE_DIVERGE_ONLY"
    else:
        status = "FAIL"

    report = {
        "schema_version": "safedrive.r2x.learned_force_smoke.v6",
        "status": status,
        "r2x_status_hint": (
            "REAL_FEATURE_OFFLINE"
            if "v4" in str(DATA.as_posix())
            else "BOOTSTRAP_OFFLINE_PASS / X5_REPAIR_REQUIRED / REAL_FEATURE_AND_LIVE_PENDING"
        ),
        "data_path": str(DATA.as_posix()),
        "curvature_semantics": semantics_dict(),
        "denominators": {
            "n_label_eligible": n_label_eligible,
            "n_pred_available": n_pred_available,
            "n_false_negative": n_false_negative,
            "n_guard_ok": n_guard_ok,
            "n_guard_reject": n_guard_reject,
            "n_attempted": n_attempted,
            "n_path_diverge": n_path_div,
            "n_control_diverge": n_ctrl_div,
            "n_conservative_static_live_prep_both": n_cons_static_both,
            "n_arc_aligned_feasibility_both": n_arc_aligned_both,
            "n_ideal_pointwise_cap_diagnostic_attempted": n_ideal_pointwise_diag,
            "n_static_tracker_horizon_deprecated_both": n_static_horizon_deprecated,
            "n_tracker_rollout_feasibility_both": n_tracker_rollout_both,
            "n_rollout_coverage_fail": n_rollout_coverage_fail,
            "n_pm_steer_prefilter_both": n_pm_steer_both,
            "n_live_prefilter_both": n_live_prefilter_both,
            "n_t10_projection_both": n_t10_proj,
        },
        "rates": {
            "pred_available_on_eligible": rate(n_pred_available, n_label_eligible),
            "path_diverge_on_attempted": path_rate,
            "control_diverge_on_attempted": ctrl_rate,
            "conservative_static_live_prep_on_attempted": rate(
                n_cons_static_both, n_attempted
            ),
            "arc_aligned_feasibility_on_attempted": rate(n_arc_aligned_both, n_attempted),
            "static_tracker_horizon_deprecated_on_attempted": rate(
                n_static_horizon_deprecated, n_attempted
            ),
            "tracker_rollout_feasibility_on_attempted": rate(
                n_tracker_rollout_both, n_attempted
            ),
            "pm_steer_prefilter_on_attempted": rate(n_pm_steer_both, n_attempted),
            "live_prefilter_on_attempted": rate(n_live_prefilter_both, n_attempted),
            "t10_projection_on_attempted": rate(n_t10_proj, n_attempted),
            "path_diverge_on_eligible": rate(n_path_div, n_label_eligible),
            "control_diverge_on_eligible": rate(n_ctrl_div, n_label_eligible),
        },
        "gates": {
            "path_manager_soft_kappa": DEFAULT_PM_SOFT_KAPPA,
            "path_manager_hard_kappa_anomaly_only": DEFAULT_PM_HARD_KAPPA,
            "mpc_kappa_max": mpc_kinematic_kappa_max(),
            "max_lat_accel_mps2": DEFAULT_MAX_LAT_ACCEL_MPS2,
            "meaningful_speed_mps": DEFAULT_MEANINGFUL_SPEED_MPS,
            "steer_mae_meaningful": STEER_MAE_MEANINGFUL,
            "attempted_definition": "pred_available AND guard_ok AND both PathManager accepted",
            "pipeline_requires_all_attempted_path_and_control_diverge": True,
            "arc_aligned_uses_t10_xy_projection": True,
            "ideal_pointwise_cap": "IDEAL_POINTWISE_CAP_DIAGNOSTIC_not_a_gate",
            "static_tracker_horizon_path_target": "DEPRECATED_vacuous_near_zero_startup",
            "tracker_rollout_gate": (
                "30-tick MPC: max_t v^2|kappa_ref| + all solved + no fallback "
                "+ meaningful speed coverage"
            ),
            "live_prefilter_requires_tracker_rollout": True,
            "conservative_static_is_not_live_exec_rate": True,
            "pm_steer_prefilter_is_not_carla": True,
            "live_prefilter_is_not_carla_smoke": True,
            "hard_1_0_is_not_vehicle_trackable_limit": True,
            "r2k_authorized": False,
        },
        "checkpoint": str(CKPT.as_posix()),
        "checkpoint_hash": runtime.spatial_head_checkpoint_hash,
        "feature_note": (
            "dataset-v4-real + real SimLingo mean64 heads"
            if "v4" in str(DATA.as_posix())
            else "Bootstrap head on scene_proxy_v1 — NOT real SimLingo driving features."
        )
        + " PM_STEER_PREFILTER_PASS is not CARLA; LIVE_PREFILTER needs 30-tick rollout gate.",
        "carla": False,
        "results": results,
    }
    (OUT / "learned_force_smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "curvature_semantics.json").write_text(
        json.dumps(semantics_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "results"}, indent=2))
    # Non-zero unless full live_prefilter on bootstrap is still only prefilter —
    # exit 0 only for LIVE_PREFILTER_PASS (still not R2-K).
    if status == "LIVE_PREFILTER_PASS":
        return 0
    if status == "PM_STEER_PREFILTER_PASS":
        return 5
    if status == "PIPELINE_DIVERGE_ONLY":
        return 4
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
