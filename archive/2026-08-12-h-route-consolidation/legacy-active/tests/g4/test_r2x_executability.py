"""Joint executability: T10 projection arc-align + tracker-faithful curve limit."""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.executability_metrics import (  # noqa: E402
    evaluate_branch_executability,
    evaluate_path_executability,
    ideal_pointwise_capped_ay,
    lateral_accel_from_speed_kappa,
    max_arc_aligned_lat_accel,
    mpc_curve_speed_limit,
    mpc_kinematic_kappa_max,
    path_manager_committed_local_max_cap,
    path_manager_q90_cap,
    project_t10_speeds_onto_spatial_path,
    semantics_dict,
)
from driving_vla.evaluation.paired_live import (  # noqa: E402
    load_anchor_artifact_any,
    select_from_anchor_artifact,
)
from driving_vla.evaluation.k2_spatial_artifact import (  # noqa: E402
    artifact_from_bundle_v2,
    make_dummy_observation_fingerprint,
)
from driving_vla.model.k2_spatial_builder import (  # noqa: E402
    build_spatial_k2_bundle_from_residuals,
    synthetic_diverse_residuals,
)
from driving_vla.model.k2_spatial_guard import attach_spatial_guard  # noqa: E402
from driving_vla.evaluation.paired_contract import content_hash  # noqa: E402
from driving_vla.runtime.curve_limits import (  # noqa: E402
    curve_speed_limit_from_kappa_window,
)
from driving_vla.runtime.k2_execution import apply_k2_to_executors, select_k2_spatial  # noqa: E402
from driving_vla.runtime.path_manager import EgoPose, PathManagerConfig, VLAPathManager  # noqa: E402
from driving_vla.runtime.vla_speed_planner import (  # noqa: E402
    VLASpeedConfig,
    VLASpeedPlanner,
)


class CurvatureSemanticsTest(unittest.TestCase):
    def test_mpc_kappa_max_approx_0_253(self) -> None:
        k = mpc_kinematic_kappa_max(wheelbase_m=2.70, max_steer_rad=0.60)
        self.assertAlmostEqual(k, math.tan(0.60) / 2.70, places=6)

    def test_hard_1_is_not_vehicle_limit(self) -> None:
        sem = semantics_dict()
        self.assertFalse(sem["hard_max_abs_curvature_1_0"]["is_vehicle_trackable_limit"])

    def test_ay_at_5mps_kappa1_is_25(self) -> None:
        self.assertAlmostEqual(lateral_accel_from_speed_kappa(5.0, 1.0), 25.0, places=6)

    def test_pm_caps(self) -> None:
        self.assertAlmostEqual(path_manager_q90_cap(0.30), 0.375, places=6)
        self.assertAlmostEqual(path_manager_committed_local_max_cap(0.30), 0.75, places=6)

    def test_arc_aligned_not_same_as_maxv_maxk(self) -> None:
        s = np.array([0.0, 1.0, 2.0, 3.0])
        kappa = np.array([0.5, 0.5, 0.01, 0.01])
        speed = np.array([1.0, 1.0, 5.0, 5.0])
        arc = max_arc_aligned_lat_accel(s, kappa, speed)
        cons = lateral_accel_from_speed_kappa(5.0, 0.5)
        self.assertAlmostEqual(arc, 0.5, places=6)
        self.assertGreater(cons, arc)

    def test_ideal_pointwise_near_tautological(self) -> None:
        # After per-point cap, a_y should be <= a_max (within float noise)
        kappa = np.array([0.1, 0.5, 1.0, 0.2])
        speed = np.array([5.0, 5.0, 5.0, 5.0])
        ay = ideal_pointwise_capped_ay(kappa, speed, max_lat_accel_mps2=1.0)
        self.assertLessEqual(ay, 1.0 + 1e-6)

    def test_shared_curve_limit_matches_formula(self) -> None:
        # kappa all 0.25 → q90=0.25 → v=2 when a=1
        lim, kq = curve_speed_limit_from_kappa_window(
            np.array([0.25] * 20), max_lat_accel_mps2=1.0, max_speed_mps=10.0, curve_limit_quantile=0.90
        )
        self.assertAlmostEqual(kq, 0.25, places=6)
        self.assertAlmostEqual(lim, 2.0, places=6)

    def test_mpc_curve_speed_limit_pointwise(self) -> None:
        self.assertAlmostEqual(mpc_curve_speed_limit(0.25, max_lat_accel_mps2=1.0), 2.0, places=6)

    def test_straight_path_pm_steer(self) -> None:
        path = [(float(i), 0.0) for i in range(20)]
        rep = evaluate_branch_executability(
            path_xy=path,
            speed_samples_mps=[2.0] * 5,
            ego_v=2.0,
            path_manager_accepted=True,
        )
        self.assertTrue(rep.pm_steer_prefilter)
        self.assertTrue(rep.pass_mpc_steer_kappa)

    def test_legacy_wrapper(self) -> None:
        path = [(float(i), 0.0) for i in range(20)]
        rep = evaluate_path_executability(path, ref_speed_mps=2.0)
        self.assertIn("conservative", rep.note)


class MpcRolloutGateTest(unittest.TestCase):
    def test_cold_branch_speed_planner_starts_from_measured_ego_speed(self) -> None:
        cfg = VLASpeedConfig(max_speed_mps=8.0, max_accel_mps2=2.5)
        uninitialized = VLASpeedPlanner(cfg)
        cold = uninitialized.update(
            [5.0] * 10, dt_s=0.05, ego_speed_mps=7.0
        )
        initialized = VLASpeedPlanner(cfg)
        initialized.reset(target_speed_mps=7.0)
        aligned = initialized.update(
            [5.0] * 10, dt_s=0.05, ego_speed_mps=7.0
        )
        self.assertLess(cold.target_speed_mps, 0.2)
        self.assertAlmostEqual(aligned.target_speed_mps, 5.0, places=6)

    def test_rollout_ay_from_ticks(self) -> None:
        from driving_vla.evaluation.executability_metrics import (
            evaluate_mpc_rollout_executability,
        )

        ticks = [
            {
                "speed_mps": 2.0,
                "reference_curvature": 0.1,
                "curve_speed_limit_mps": 2.5,
                "target_speed_mps": 2.0,
                "mode": "mpc",
                "solver_status": "solved",
            },
            {
                "speed_mps": 2.0,
                "reference_curvature": 0.4,
                "curve_speed_limit_mps": 1.5,
                "target_speed_mps": 1.5,
                "mode": "mpc",
                "solver_status": "solved",
            },
        ]
        # max ay = max(4*0.1, 4*0.4) = 1.6
        rep = evaluate_mpc_rollout_executability(ticks, max_lat_accel_mps2=1.0)
        self.assertAlmostEqual(rep["max_rollout_ay"], 1.6, places=6)
        self.assertFalse(rep["pass_rollout_ay"])
        self.assertTrue(rep["pass_all_mpc_solved"])
        self.assertTrue(rep["meaningful_speed_coverage"])
        self.assertFalse(rep["pass_tracker_rollout"])

    def test_near_zero_speed_fails_coverage(self) -> None:
        from driving_vla.evaluation.executability_metrics import (
            evaluate_mpc_rollout_executability,
        )

        ticks = [
            {
                "speed_mps": 0.05,
                "reference_curvature": 0.5,
                "curve_speed_limit_mps": 2.5,
                "target_speed_mps": 0.1,
                "mode": "mpc",
                "solver_status": "solved",
            }
        ] * 30
        rep = evaluate_mpc_rollout_executability(ticks, max_lat_accel_mps2=1.0)
        self.assertTrue(rep["pass_rollout_ay"])  # tiny v → tiny a_y
        self.assertFalse(rep["meaningful_speed_coverage"])
        self.assertFalse(rep["pass_tracker_rollout"])


class T10ProjectionTest(unittest.TestCase):
    def test_project_t10_onto_committed(self) -> None:
        native = tuple((float(i) * 1.2, 0.0) for i in range(20))
        nom, alt = synthetic_diverse_residuals(20, lateral_sign=1.0, lineage="contract_probe")
        alt["raw_d"] = [min(2.0, 0.25 * i) for i in range(20)]
        b = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=native[0],
            ego_v=2.0,
            base_speed_mps=2.5,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={},
            backbone_forward_id="fwd",
            defensive_available=True,
        )
        g = attach_spatial_guard(
            replace(b, set_diagnostics={"eligible_for_diversity": True}),
            require_diversity_if_eligible=True,
        )
        self.assertEqual(g.guard_status, "OK", msg=g.guard_reasons)
        sel = select_k2_spatial(g, mode="force", force_index=1)
        pm = VLAPathManager(PathManagerConfig(max_switch_lateral_5m=1.0))
        sp = VLASpeedPlanner()
        path = sel.execution_spec.spatial_path_xy
        yaw0 = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        ego = EgoPose(float(native[0][0]), float(native[0][1]), yaw0, 2.0)
        applied = apply_k2_to_executors(
            sel,
            speed_planner=sp,
            path_manager=pm,
            ego=ego,
            stamp_s=0.0,
            frame_id="t",
            dt_s=0.05,
            nav_target_map_xy=path[-1],
        )
        cand = g.candidates[1]
        s_t, v_t, k_t = project_t10_speeds_onto_spatial_path(
            applied.path_update.committed or applied.path_update.raw,
            cand.points_xy_yaw_v_a_kappa,
        )
        self.assertGreater(s_t.size, 0)
        # s should be non-decreasing along trajectory samples
        self.assertTrue(np.all(np.diff(s_t) >= -1e-6) or s_t.size < 2)

        rep = evaluate_branch_executability(
            path_xy=path,
            speed_samples_mps=list(sel.execution_spec.speed_samples_mps),
            ego_v=2.0,
            path_manager_accepted=bool(applied.path_update.accepted),
            raw_spatial_path=applied.path_update.raw,
            committed_spatial_path=applied.path_update.committed,
            t10_points_xy_yaw_v=cand.points_xy_yaw_v_a_kappa,
        )
        self.assertTrue(rep.t10_projection_used)
        self.assertIn("densified", rep.densified_source)
        self.assertEqual(rep.mpc_capped_label if hasattr(rep, "mpc_capped_label") else "IDEAL_POINTWISE_CAP_DIAGNOSTIC", rep.to_dict()["mpc_capped_label"])


class V2PairedLiveSelectTest(unittest.TestCase):
    def test_select_from_v2_artifact_roundtrip(self) -> None:
        native = tuple((float(i) * 1.2, 0.0) for i in range(20))
        nom, alt = synthetic_diverse_residuals(20, lateral_sign=1.0, lineage="contract_probe")
        alt["raw_d"] = [min(2.0, 0.35 * i) for i in range(20)]
        b = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=native[0],
            ego_v=5.0,
            base_speed_mps=6.0,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={"t": "x"},
            backbone_forward_id="fwd",
            defensive_available=True,
        )
        g = attach_spatial_guard(
            replace(b, set_diagnostics={"eligible_for_diversity": True}),
            require_diversity_if_eligible=True,
        )
        self.assertEqual(g.guard_status, "OK", msg=g.guard_reasons)
        art = artifact_from_bundle_v2(
            g,
            pair_id="p",
            scenario_id="s",
            seed_id="a",
            anchor_run_id="r",
            anchor_carla_frame=0,
            anchor_simulation_time_s=0.0,
            requested_initial_state_hash="0" * 64,
            measured_initial_state_hash="0" * 64,
            observation_fingerprint=make_dummy_observation_fingerprint(
                k2_bundle_hash=content_hash({"a": 1}, nibble=16)
            ),
            model_checkpoint_hash="m",
            executor_config_hash="e",
            evidence_lineage="contract_probe",
        )
        loaded = load_anchor_artifact_any(art.to_json_bytes())
        s0 = select_from_anchor_artifact(loaded, force_index=0)
        s1 = select_from_anchor_artifact(loaded, force_index=1)
        self.assertNotEqual(
            s0.execution_spec.spatial_path_hash, s1.execution_spec.spatial_path_hash
        )


if __name__ == "__main__":
    unittest.main()
