"""R1 offline contract: retiming K2, Guard, single forward, collapse taxonomy."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from dataclasses import replace  # noqa: E402

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray  # noqa: E402
from driving_vla.model.canonicalizer import cum_arclength, interp_xy  # noqa: E402
from driving_vla.model.k2_builder import (  # noqa: E402
    BUILD_PATH_HORIZON_EXHAUSTED,
    COLLAPSE_NUMERIC,
    COLLAPSE_SEMANTIC_STOP,
    GUARD_OK,
    GUARD_REJECT,
    K2BuilderConfig,
    K2Diagnostics,
    K2ExecutionSpec,
    attach_guard,
    build_k2_bundle,
    load_k2_config,
    project_point_to_path_s,
    project_speed_profile,
    recompute_kinematics_residuals,
    stable_hash_points,
    stable_hash_xy,
    validate_k2_bundle,
)
from driving_vla.model.neural_policy import NeuralV1Policy  # noqa: E402
from driving_vla.model.speed_convert import (  # noqa: E402
    normalize_k2_target_speed_profile,
    speed_wps_to_planner_samples,
)
from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS  # noqa: E402


class _CountingFakeRuntime:
    def __init__(self, *, speed: float = 5.0, path_len: int = 20, path_step: float = 1.0) -> None:
        self.load_report = SimpleNamespace(ok=True, error="")
        self.forward_count = 0
        self.speed = float(speed)
        self.path_len = int(path_len)
        self.path_step = float(path_step)

    def forward_numpy(self, _image, **kwargs):
        self.forward_count += 1
        route = np.column_stack(
            (
                np.arange(self.path_step, self.path_step * (self.path_len + 0.1), self.path_step)[
                    : self.path_len
                ],
                np.zeros(self.path_len),
            )
        )
        return SimpleNamespace(
            route_xy=route,
            speed_mps=(self.speed,) * 5,
            speed_wps_xy=np.zeros((10, 2)),
            latency_s=0.01,
            peak_vram_mb=100.0,
        )


def _moving_obs(**kwargs) -> ObservationBundle:
    base = dict(
        run_id="r1",
        frame_id="f1",
        scenario_id="s1",
        simulation_time_s=1.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=5.0,
        front_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        meta={"official_contract": True, "target_ego_1": (15.0, 0.0), "target_ego_2": (30.0, 0.0)},
    )
    base.update(kwargs)
    return ObservationBundle(**base)


class R1ContractTest(unittest.TestCase):
    def test_load_k2_config(self) -> None:
        cfg = load_k2_config()
        self.assertEqual(cfg.k, 2)
        self.assertEqual(cfg.t_steps, 10)
        self.assertAlmostEqual(cfg.conservative_speed_ratio, 0.65)
        self.assertEqual(cfg.branch_type, "longitudinal_temporal")

    def test_normalize_official_five_to_t10(self) -> None:
        samples = (4.0, 4.0, 4.0, 4.0, 4.0)
        prof = normalize_k2_target_speed_profile(samples, t_steps=10, mode="official")
        self.assertEqual(len(prof), 10)
        self.assertTrue(all(abs(v - 4.0) < 1e-9 for v in prof))
        # K1 planner samples unchanged
        k1 = speed_wps_to_planner_samples(
            [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0)],
            use_official_scalar=True,
        )
        self.assertEqual(len(k1), 5)

    def test_shape_time_identity_and_single_forward(self) -> None:
        rt = _CountingFakeRuntime(speed=5.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = _moving_obs()
        bundle = policy.predict_bundle(obs)
        self.assertEqual(rt.forward_count, 1)
        self.assertEqual(policy.last_forward_count, 1)
        self.assertEqual(len(bundle.candidates), 2)
        for c in bundle.candidates:
            self.assertEqual(c.t_steps, T_STEPS)
        self.assertEqual(bundle.candidates[0].candidate_id, "v1_nominal")
        self.assertEqual(bundle.candidates[1].candidate_id, "v1_conservative")
        self.assertEqual(bundle.top1_index, 0)
        self.assertAlmostEqual(
            bundle.candidates[0].probability + bundle.candidates[1].probability, 1.0
        )
        self.assertAlmostEqual(bundle.candidates[0].probability, 0.5)
        self.assertAlmostEqual(bundle.candidates[1].probability, 0.5)
        self.assertEqual(bundle.probability_source, "fixed_equal_prior_unscaled")
        self.assertAlmostEqual(bundle.probability_margin, 0.0)
        self.assertEqual(bundle.branch_type, "longitudinal_temporal")
        # implicit times 0.25..2.50
        self.assertAlmostEqual(DT_S * T_STEPS, HORIZON_S)

    def test_determinism(self) -> None:
        rt = _CountingFakeRuntime(speed=5.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = _moving_obs()
        b1 = policy.predict_bundle(obs)
        b2 = policy.predict_bundle(obs)
        self.assertEqual(
            b1.candidates[0].points_xy_yaw_v_a_kappa,
            b2.candidates[0].points_xy_yaw_v_a_kappa,
        )
        self.assertEqual(b1.native_path_hash, b2.native_path_hash)

    def test_moving_diversity_and_retiming(self) -> None:
        rt = _CountingFakeRuntime(speed=6.0, path_len=25, path_step=1.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = _moving_obs(ego_v=5.0)
        bundle = attach_guard(policy.predict_bundle(obs))
        d = bundle.diagnostics
        self.assertGreater(d.max_position_separation_m, 0.5)
        self.assertGreaterEqual(d.mean_speed_gap_mps, 0.25)
        self.assertGreaterEqual(d.final_progress_gap_m, 0.50)
        self.assertTrue(d.selection_space_eligible)
        self.assertNotEqual(d.collapse_reason, COLLAPSE_NUMERIC)
        # xy differ; not speed-only collapse
        nom = bundle.candidates[0].points_xy_yaw_v_a_kappa
        cons = bundle.candidates[1].points_xy_yaw_v_a_kappa
        self.assertNotEqual(nom, cons)
        # kinematics a ≈ Δv/dt from ego_v
        v_prev = obs.ego_v
        for row in nom:
            expected_a = (row[3] - v_prev) / DT_S
            self.assertAlmostEqual(row[4], expected_a, places=5)
            v_prev = row[3]
        self.assertEqual(bundle.guard_status, GUARD_OK)

    def test_semantic_stop_no_space(self) -> None:
        rt = _CountingFakeRuntime(speed=0.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = _moving_obs(ego_v=0.0)
        bundle = policy.predict_bundle(obs)
        self.assertTrue(bundle.diagnostics.collapsed)
        self.assertEqual(bundle.diagnostics.collapse_reason, COLLAPSE_SEMANTIC_STOP)
        # Guard allows explicit semantic stop
        self.assertEqual(bundle.guard_status, GUARD_OK)

    def test_residuals_pass_on_builder_output(self) -> None:
        rt = _CountingFakeRuntime(speed=5.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        bundle = policy.predict_bundle(_moving_obs())
        d = bundle.diagnostics
        cfg = load_k2_config()
        self.assertLessEqual(
            d.position_integration_error_max_m, cfg.position_integration_error_max_m + 1e-6
        )
        self.assertLessEqual(
            d.acceleration_error_max_mps2, cfg.acceleration_error_max_mps2 + 1e-9
        )
        self.assertLessEqual(
            d.native_path_cross_track_error_max_m,
            cfg.native_path_cross_track_error_max_m + 1e-9,
        )

    def test_path_exhaustion_rejects_end_positive_speed_copy(self) -> None:
        # Very short path, high ego speed → horizon exhausted
        rt = _CountingFakeRuntime(speed=8.0, path_len=3, path_step=0.5)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = _moving_obs(ego_v=8.0)
        bundle = policy.predict_bundle(obs)
        # Either exhausted or path-capped without illegal free-running end
        if bundle.build_error == BUILD_PATH_HORIZON_EXHAUSTED:
            self.assertEqual(bundle.guard_status, GUARD_REJECT)
        else:
            # capped: final speeds should not claim free progress past path
            for c in bundle.candidates:
                pts = c.points_xy_yaw_v_a_kappa
                # positions stay near short path extent
                xs = [p[0] for p in pts]
                self.assertLess(max(xs), 5.0)

    def test_guard_rejects_nan_and_bad_t(self) -> None:
        rt = _CountingFakeRuntime(speed=5.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        bundle = policy.predict_bundle(_moving_obs())
        # Corrupt a candidate
        bad_pts = list(bundle.candidates[0].points_xy_yaw_v_a_kappa)
        x, y, yaw, v, a, k = bad_pts[0]
        bad_pts[0] = (float("nan"), y, yaw, v, a, k)
        from driving_vla.adapter.policy_adapter import TrajectoryArray
        from dataclasses import replace

        bad0 = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(bad_pts),
            probability=0.5,
            candidate_id="v1_nominal",
        )
        corrupted = replace(bundle, candidates=(bad0, bundle.candidates[1]))
        result = validate_k2_bundle(corrupted)
        self.assertEqual(result.status, GUARD_REJECT)
        self.assertTrue(any("non_finite" in r for r in result.reasons))

    def test_guard_rejects_speed_only_xy_collapse(self) -> None:
        """Classic residual bug: same xy, different v on eligible frame."""
        rt = _CountingFakeRuntime(speed=6.0, path_len=25)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        good = policy.predict_bundle(_moving_obs(ego_v=5.0))
        nom = good.candidates[0]
        # Build fake cons with same xy as nom but scaled speeds
        pts = []
        for row in nom.points_xy_yaw_v_a_kappa:
            x, y, yaw, v, a, k = row
            pts.append((x, y, yaw, v * 0.65, a * 0.65, k))
        from driving_vla.adapter.policy_adapter import TrajectoryArray
        from dataclasses import replace

        cons = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(pts),
            probability=0.5,
            candidate_id="v1_conservative",
            intended_action="conservative",
        )
        from driving_vla.model.k2_builder import K2Diagnostics

        d = K2Diagnostics(
            mean_speed_gap_mps=1.0,
            final_progress_gap_m=1.0,
            max_position_separation_m=0.0,
            mean_position_separation_m=0.0,
            collapsed=True,
            collapse_reason=COLLAPSE_NUMERIC,
            selection_space_eligible=True,
            path_speed_cap_active=False,
            position_integration_error_max_m=0.0,
            acceleration_error_max_mps2=0.0,
            yaw_tangent_error_max_rad=0.0,
            curvature_error_max_per_m=0.0,
            curvature_error_p95_per_m=0.0,
            native_path_cross_track_error_max_m=0.0,
        )
        corrupted = replace(
            good,
            candidates=(nom, cons),
            diagnostics=d,
            # keep specs but timed hashes will mismatch → also reject
        )
        result = validate_k2_bundle(corrupted)
        self.assertEqual(result.status, GUARD_REJECT)

    def test_projector_from_ego_v(self) -> None:
        speeds, accels, arcs = project_speed_profile(
            [5.0] * 10,
            ego_v=3.0,
            dt_s=0.25,
            max_accel_mps2=2.5,
            max_decel_mps2=3.0,
        )
        self.assertEqual(len(speeds), 10)
        # first step accel relative to ego_v
        self.assertAlmostEqual(accels[0], (speeds[0] - 3.0) / 0.25, places=6)
        self.assertGreater(arcs[-1], arcs[0])

    def test_short_native_path_hard_reject_no_synthetic_geometry(self) -> None:
        """Empty/single-point native path must not synthesize a fake 20m path."""
        for path in ((), ((0.0, 0.0),)):
            native = SimpleNamespace(
                path_map_xy=path,
                speed_mps=(5.0, 5.0, 5.0, 5.0, 5.0),
                latency_s=0.0,
                peak_vram_mb=0.0,
            )
            obs = _moving_obs(ego_v=4.0)
            bundle = build_k2_bundle(native, obs)
            self.assertEqual(bundle.build_error, BUILD_PATH_HORIZON_EXHAUSTED)
            self.assertEqual(bundle.guard_status, GUARD_REJECT)
            self.assertLess(len(bundle.native_path_xy), 2)
            # Must not invent a 21-point / ~20m neural-looking path
            self.assertNotEqual(len(bundle.native_path_xy), 21)
            if bundle.native_path_xy:
                total = 0.0
                pts = list(bundle.native_path_xy)
                for i in range(1, len(pts)):
                    total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                self.assertLess(total, 1.0)
            self.assertEqual(len(bundle.candidates), 0)
            result = validate_k2_bundle(bundle)
            self.assertEqual(result.status, GUARD_REJECT)
            self.assertTrue(
                any(
                    BUILD_PATH_HORIZON_EXHAUSTED in r or "K_mismatch" in r
                    for r in result.reasons
                )
            )

    def test_guard_recomputes_residuals_not_declared_diagnostics(self) -> None:
        """Tampered T10 with zeroed diagnostics + matching timed hash still fails."""
        rt = _CountingFakeRuntime(speed=6.0, path_len=25)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        good = policy.predict_bundle(_moving_obs(ego_v=5.0))
        self.assertEqual(good.guard_status, GUARD_OK)

        # Break kinematics: keep xy, scale v, leave a stale → inconsistent Δv/dt
        bad_pts = []
        for row in good.candidates[0].points_xy_yaw_v_a_kappa:
            x, y, yaw, v, a, k = row
            bad_pts.append((x, y, yaw, float(v) * 2.0, a, k))
        bad0 = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(bad_pts),
            probability=0.5,
            candidate_id="v1_nominal",
            intended_action="nominal",
        )
        # Update timed hash so only residual recompute can catch the attack
        new_hash = stable_hash_points(bad_pts)
        spec0 = good.execution_specs["v1_nominal"]
        specs = dict(good.execution_specs)
        specs["v1_nominal"] = replace(spec0, timed_trajectory_hash=new_hash)
        # Claim residuals are perfect (what a trusting Guard would accept)
        fake_diag = K2Diagnostics(
            mean_speed_gap_mps=good.diagnostics.mean_speed_gap_mps,
            final_progress_gap_m=good.diagnostics.final_progress_gap_m,
            max_position_separation_m=good.diagnostics.max_position_separation_m,
            mean_position_separation_m=good.diagnostics.mean_position_separation_m,
            collapsed=False,
            collapse_reason=None,
            selection_space_eligible=True,
            path_speed_cap_active=False,
            position_integration_error_max_m=0.0,
            acceleration_error_max_mps2=0.0,
            yaw_tangent_error_max_rad=0.0,
            curvature_error_max_per_m=0.0,
            curvature_error_p95_per_m=0.0,
            native_path_cross_track_error_max_m=0.0,
        )
        tampered = replace(
            good,
            candidates=(bad0, good.candidates[1]),
            execution_specs=specs,
            diagnostics=fake_diag,
            guard_status=GUARD_OK,
            guard_reasons=(),
            build_error=None,
        )
        result = validate_k2_bundle(tampered)
        self.assertEqual(result.status, GUARD_REJECT)
        self.assertTrue(
            any(
                r in result.reasons
                for r in (
                    "position_integration_residual",
                    "acceleration_residual",
                    "yaw_tangent_residual",
                    "cross_track_residual",
                )
            ),
            msg=f"expected residual reject, got {result.reasons}",
        )

    def _shift_points_along_path(
        self,
        points: tuple,
        path_xy: tuple,
        delta_s: float,
    ) -> tuple:
        path = list(path_xy)
        s_list = cum_arclength(path)
        out = []
        for row in points:
            x, y, _yaw, v, a, k = row
            s, _, _, _ = project_point_to_path_s(float(x), float(y), path, s_list)
            s2 = max(0.0, float(s) + float(delta_s))
            nx, ny, nyaw = interp_xy(path, s_list, s2)
            out.append((nx, ny, nyaw, float(v), float(a), float(k)))
        return tuple(out)

    def test_guard_rejects_constant_path_offset_first_step(self) -> None:
        """Whole-trajectory +1m shift along path must fail t=0→0.25s integration."""
        rt = _CountingFakeRuntime(speed=6.0, path_len=30, path_step=1.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        good = policy.predict_bundle(_moving_obs(ego_v=5.0))
        self.assertEqual(good.guard_status, GUARD_OK)

        shifted = []
        for cand in good.candidates:
            pts = self._shift_points_along_path(
                cand.points_xy_yaw_v_a_kappa, good.native_path_xy, 1.0
            )
            shifted.append(
                TrajectoryArray(
                    points_xy_yaw_v_a_kappa=pts,
                    probability=cand.probability,
                    candidate_id=cand.candidate_id,
                    intended_action=cand.intended_action,
                    uncertainty=cand.uncertainty,
                    behavior=cand.behavior,
                )
            )
        specs = dict(good.execution_specs)
        for arr in shifted:
            sp = specs[arr.candidate_id]
            specs[arr.candidate_id] = replace(
                sp, timed_trajectory_hash=stable_hash_points(arr.points_xy_yaw_v_a_kappa)
            )
        fake_diag = replace(
            good.diagnostics,
            position_integration_error_max_m=0.0,
            acceleration_error_max_mps2=0.0,
            yaw_tangent_error_max_rad=0.0,
            native_path_cross_track_error_max_m=0.0,
        )
        tampered = replace(
            good,
            candidates=tuple(shifted),
            execution_specs=specs,
            diagnostics=fake_diag,
            guard_status=GUARD_OK,
            guard_reasons=(),
            build_error=None,
        )
        result = validate_k2_bundle(tampered)
        self.assertEqual(result.status, GUARD_REJECT)
        self.assertIn("position_integration_residual", result.reasons)
        # Explicit first-step residual must exceed 0.05 m (1 m offset)
        self.assertGreater(
            float(result.metrics["recomputed_position_integration_error_max_m"]), 0.5
        )

    def test_guard_rejects_reverse_path_progress(self) -> None:
        """Positive speeds walking backward along the path must fail closed."""
        rt = _CountingFakeRuntime(speed=5.0, path_len=30, path_step=1.0)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        good = policy.predict_bundle(_moving_obs(ego_v=4.0))
        self.assertEqual(good.guard_status, GUARD_OK)

        path = list(good.native_path_xy)
        s_list = cum_arclength(path)
        # Place samples going backward from s≈12m with positive v matching |ds|/dt
        rev_pts = []
        s_cur = 12.0
        v_const = 4.0
        dt = 0.25
        v_prev = 4.0  # ego_v match for a consistency of first step if s matches
        # First point still at s matching forward integral so only reverse *between*
        # points would be soft; force all steps reverse after a valid first sample.
        s0 = 0.5 * (4.0 + v_const) * dt
        s_cur = s0
        for i in range(10):
            if i == 0:
                s_cur = s0
            else:
                s_cur = s_cur - v_const * dt  # reverse along path
            s_cur = max(0.0, s_cur)
            x, y, yaw = interp_xy(path, s_list, s_cur)
            # Keep positive speed and a≈0 for cruise
            a = 0.0 if i > 0 else (v_const - 4.0) / dt
            rev_pts.append((x, y, yaw, v_const, a, 0.0))

        bad0 = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(rev_pts),
            probability=0.5,
            candidate_id="v1_nominal",
            intended_action="nominal",
        )
        # Keep conservative good so K=2 structure holds
        specs = dict(good.execution_specs)
        specs["v1_nominal"] = replace(
            specs["v1_nominal"],
            timed_trajectory_hash=stable_hash_points(rev_pts),
        )
        fake_diag = replace(
            good.diagnostics,
            position_integration_error_max_m=0.0,
            acceleration_error_max_mps2=0.0,
            yaw_tangent_error_max_rad=0.0,
            native_path_cross_track_error_max_m=0.0,
            collapsed=False,
            collapse_reason=None,
        )
        tampered = replace(
            good,
            candidates=(bad0, good.candidates[1]),
            execution_specs=specs,
            diagnostics=fake_diag,
            guard_status=GUARD_OK,
            guard_reasons=(),
            build_error=None,
        )
        result = validate_k2_bundle(tampered)
        self.assertEqual(result.status, GUARD_REJECT)
        self.assertTrue(
            "negative_path_progress" in result.reasons
            or "position_integration_residual" in result.reasons,
            msg=f"expected reverse reject, got {result.reasons}",
        )
        # Direct residual helper must report negative signed progress
        resid = recompute_kinematics_residuals(
            rev_pts,
            ego_v=4.0,
            dt_s=0.25,
            native_path_xy=good.native_path_xy,
        )
        self.assertTrue(resid["negative_path_progress"])
        self.assertLess(resid["signed_path_progress_min_m"], -0.1)

    def test_guard_rejects_execution_path_content_hash_spoof(self) -> None:
        """Replacing spatial_path_xy while keeping declared native_path_hash must fail."""
        rt = _CountingFakeRuntime(speed=5.5, path_len=25)
        policy = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        good = policy.predict_bundle(_moving_obs(ego_v=4.0))
        self.assertEqual(good.guard_status, GUARD_OK)

        # Different geometry, same declared hash field
        spoof_path = tuple((float(i), 1.0) for i in range(0, 25))
        specs = dict(good.execution_specs)
        for cid, spec in list(specs.items()):
            specs[cid] = K2ExecutionSpec(
                candidate_id=spec.candidate_id,
                spatial_path_xy=spoof_path,
                speed_samples_mps=spec.speed_samples_mps,
                timed_trajectory_hash=spec.timed_trajectory_hash,
                native_path_hash=good.native_path_hash,  # spoof: keep old declaration
                branch_type=spec.branch_type,
            )
        spoofed = replace(good, execution_specs=specs)
        result = validate_k2_bundle(spoofed)
        self.assertEqual(result.status, GUARD_REJECT)
        self.assertTrue(
            any(
                "execution_path" in r or "spatial_path" in r or "content_hash" in r
                for r in result.reasons
            ),
            msg=f"expected content-hash reject, got {result.reasons}",
        )


if __name__ == "__main__":
    unittest.main()
