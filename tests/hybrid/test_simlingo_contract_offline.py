"""Offline SimLingo official contract checks (no CARLA required for most tests).

GPU tests for 20× forward variance are skipped when the model/CUDA is unavailable.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "simlingo-main"))

from driving_vla.model.nominal_policy import NominalVLAPolicy  # noqa: E402
from driving_vla.model.simlingo_contract import (  # noqa: E402
    RoutePlannerXY,
    SimLingoContractConfig,
    build_official_prompt,
    densify_polyline_xy,
    inverse_conversion_2d,
    legacy_contract_config,
    navigation_targets_legacy,
    navigation_targets_official,
    preprocess_camera_legacy_rgb,
    preprocess_camera_official_bgr,
    resolve_navigation_prompt_conditioning,
    crop_bottom_official,
)
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
    NeuralForwardResult,
)
from driving_vla.model.speed_convert import speed_wps_2d_to_mps  # noqa: E402
from driving_vla.runtime.path_manager import (  # noqa: E402
    EgoPose,
    PathManagerConfig,
    VLAPathManager,
)


def official_inverse_conversion_2d(point, translation, yaw):
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=float
    )
    return rotation_matrix.T @ (np.asarray(point, dtype=float) - np.asarray(translation, dtype=float))


def _reference_route_planner_targets(
    route_xy: list[tuple[float, float]],
    ego_xy: tuple[float, float],
    *,
    min_d: float = 7.5,
    max_d: float = 50.0,
    ds: float = 1.0,
    progress_s: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Independent reimplementation: densify, forward suffix, RoutePlanner [1],[2]."""
    dense = densify_polyline_xy(route_xy, ds_m=ds)
    s = [0.0]
    for i in range(1, len(dense)):
        s.append(s[-1] + math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]))
    if progress_s is None:
        # nearest s
        best_s, best_d = 0.0, float("inf")
        for p, sp in zip(dense, s):
            d = math.hypot(p[0] - ego_xy[0], p[1] - ego_xy[1])
            if d < best_d:
                best_d, best_s = d, sp
        progress_s = best_s
    start_s = max(0.0, float(progress_s) - 2.0)
    start_idx = 0
    for i, sp in enumerate(s):
        if sp >= start_s:
            start_idx = i
            break
    forward = dense[start_idx:] or dense
    planner = RoutePlannerXY(min_d, max_d)
    planner.set_route(forward)
    rem = planner.run_step(ego_xy)
    if len(rem) > 2:
        return rem[1], rem[2]
    if len(rem) > 1:
        return rem[1], rem[1]
    return rem[0], rem[0]


class SimLingoContractOfflineTest(unittest.TestCase):
    def test_official_prompt_keeps_target_point_and_command_modes_disjoint(self) -> None:
        target_prompt = build_official_prompt(
            speed_mps=3.04,
            eval_route_as="target_point",
        )
        self.assertEqual(target_prompt.count("<TARGET_POINT>"), 2)
        self.assertNotIn("Command:", target_prompt)

        command_text = (
            "Command: do a lane change to the left in 8 meter "
            "then follow the road."
        )
        command_prompt = build_official_prompt(
            speed_mps=3.04,
            command_text=command_text,
            eval_route_as="command",
        )
        self.assertIn(command_text, command_prompt)
        self.assertNotIn("<TARGET_POINT>", command_prompt)
        with self.assertRaisesRegex(ValueError, "requires command_text"):
            build_official_prompt(speed_mps=3.0, eval_route_as="command")

    def test_route_change_uses_native_hlc_until_registered_lane_is_reached(self) -> None:
        left = resolve_navigation_prompt_conditioning(
            maneuver="ROUTE_CHANGE_LEFT",
            target_distance_m=8.9,
            current_road_id=7,
            current_lane_id=-1,
            target_road_id=7,
            target_lane_id=-2,
        )
        self.assertEqual(left.eval_route_as, "command")
        self.assertEqual(
            left.command_text,
            "Command: do a lane change to the left in 8 meter then follow the road.",
        )
        self.assertEqual(left.reason, "ROUTE_CHANGE_LEFT_HLC")
        self.assertFalse(left.target_lane_reached)

        right = resolve_navigation_prompt_conditioning(
            maneuver="ROUTE_CHANGE_RIGHT",
            target_distance_m=12.0,
            current_road_id=7,
            current_lane_id=-2,
            target_road_id=7,
            target_lane_id=-1,
        )
        self.assertEqual(right.eval_route_as, "command")
        self.assertIn("lane change to the right in 12 meter", right.command_text or "")

        reached = resolve_navigation_prompt_conditioning(
            maneuver="ROUTE_CHANGE_LEFT",
            target_distance_m=3.0,
            current_road_id=7,
            current_lane_id=-2,
            target_road_id=7,
            target_lane_id=-2,
        )
        self.assertEqual(reached.eval_route_as, "target_point")
        self.assertIsNone(reached.command_text)
        self.assertTrue(reached.target_lane_reached)
        self.assertEqual(reached.reason, "TARGET_LANE_REACHED")

        wrong_road = resolve_navigation_prompt_conditioning(
            maneuver="ROUTE_CHANGE_LEFT",
            target_distance_m=3.0,
            current_road_id=None,
            current_lane_id=-2,
            target_road_id=7,
            target_lane_id=-2,
        )
        self.assertEqual(wrong_road.eval_route_as, "command")
        self.assertFalse(wrong_road.target_lane_reached)

    def test_non_lane_change_maneuvers_keep_target_point_contract(self) -> None:
        for maneuver in (
            "FOLLOW_STRAIGHT",
            "FOLLOW_CURVE_LEFT",
            "FOLLOW_CURVE_RIGHT",
            "JUNCTION_STRAIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
        ):
            resolved = resolve_navigation_prompt_conditioning(
                maneuver=maneuver,
                target_distance_m=9.0,
            )
            self.assertEqual(resolved.eval_route_as, "target_point", maneuver)
            self.assertIsNone(resolved.command_text, maneuver)

    def test_camera_constants_match_official_mount_fov(self) -> None:
        self.assertEqual(SIMLINGO_CAMERA_XYZ, (-1.5, 0.0, 2.0))
        self.assertEqual(SIMLINGO_CAMERA_FOV_DEG, 110)
        self.assertEqual(SIMLINGO_CAMERA_NATIVE_SIZE, (1024, 512))
        cfg = SimLingoContractConfig()
        self.assertTrue(cfg.official_contract)
        self.assertEqual(cfg.camera_width, 1024)
        self.assertEqual(cfg.camera_height, 512)
        self.assertFalse(cfg.lateral_mode_flip_enabled)

    def test_map_to_ego_matches_official_inverse_conversion(self) -> None:
        cases = [
            (0.0, (1.0, 2.0), (10.0, 0.0)),
            (math.pi / 2, (1.0, 2.0), (0.0, 5.0)),
            (-math.pi / 2, (3.0, 1.0), (2.0, -4.0)),
            (math.pi / 4, (0.0, 0.0), (3.0, 4.0)),
        ]
        for yaw, ego_xy, world_pt in cases:
            ours = inverse_conversion_2d(world_pt, ego_xy, yaw)
            official = official_inverse_conversion_2d(world_pt, ego_xy, yaw)
            np.testing.assert_allclose(ours, official, atol=1e-9)

    def test_ego_map_roundtrip_at_cardinal_yaw(self) -> None:
        pol = NominalVLAPolicy.__new__(NominalVLAPolicy)

        class Obs:
            pass

        for yaw_deg in (0.0, 90.0, -90.0, 45.0):
            yaw = math.radians(yaw_deg)
            obs = Obs()
            obs.ego_x, obs.ego_y, obs.ego_yaw = 10.0, -3.0, yaw
            path = np.array([[1.0, 0.0], [2.0, 0.5], [5.0, -1.0]], dtype=float)
            world = pol._ego_path_to_map(path, obs)
            for (ex, ey), (mx, my) in zip(path, world, strict=True):
                back = pol._map_to_ego(mx, my, obs)
                self.assertAlmostEqual(back[0], float(ex), places=9)
                self.assertAlmostEqual(back[1], float(ey), places=9)

    def test_plus_x_is_forward_plus_y_is_carla_right_at_yaw0(self) -> None:
        pol = NominalVLAPolicy.__new__(NominalVLAPolicy)

        class Obs:
            ego_x = 0.0
            ego_y = 0.0
            ego_yaw = 0.0

        obs = Obs()
        w = pol._ego_path_to_map(np.array([[5.0, 0.0]]), obs)[0]
        self.assertAlmostEqual(w[0], 5.0, places=9)
        self.assertAlmostEqual(w[1], 0.0, places=9)
        w2 = pol._ego_path_to_map(np.array([[0.0, 5.0]]), obs)[0]
        self.assertAlmostEqual(w2[0], 0.0, places=9)
        self.assertAlmostEqual(w2[1], 5.0, places=9)

    def test_route_planner_targets_match_reference_straight_arc_s(self) -> None:
        """Official densify + RoutePlanner[1]/[2] matches independent reference."""
        cases = {
            "straight": [(float(x), 0.0) for x in range(0, 81)],
            "arc": [
                (15.0 * math.sin(t), 15.0 * (1.0 - math.cos(t)))
                for t in np.linspace(0.0, 2.0, 81)
            ],
            "s_curve": [(float(x), 3.0 * math.sin(x / 8.0)) for x in range(0, 81)],
        }
        for name, route in cases.items():
            ego = (10.0, float(route[10][1]) if name != "straight" else 0.0)
            # project ego onto route-ish for S
            if name == "s_curve":
                ego = (10.0, 3.0 * math.sin(10.0 / 8.0))
            if name == "arc":
                # at t corresponding to ~10m arc
                ego = route[20]
            ref1, ref2 = _reference_route_planner_targets(route, ego)
            got = navigation_targets_official(
                route,
                ego_x=ego[0],
                ego_y=ego[1],
                ego_yaw=0.0 if name == "straight" else math.atan2(
                    route[min(21, len(route) - 1)][1] - ego[1],
                    route[min(21, len(route) - 1)][0] - ego[0],
                ),
            )
            # Compare map targets (independent of yaw)
            self.assertAlmostEqual(got.target_map_1[0], ref1[0], places=5, msg=name)
            self.assertAlmostEqual(got.target_map_1[1], ref1[1], places=5, msg=name)
            self.assertAlmostEqual(got.target_map_2[0], ref2[0], places=5, msg=name)
            self.assertAlmostEqual(got.target_map_2[1], ref2[1], places=5, msg=name)
            # Official targets should be much closer than legacy 15/30 on straight
            if name == "straight":
                self.assertLess(got.target1_distance_m, 12.0)
                leg = navigation_targets_legacy(
                    route, ego_x=10.0, ego_y=0.0, ego_yaw=0.0
                )
                self.assertAlmostEqual(leg.target1_distance_m, 15.0, places=3)

    def test_sparse_official_plan_gives_second_target_junction_lookahead(self) -> None:
        """Dense authored routes must emulate the sparse Leaderboard plan."""
        route = [(float(x), 0.0) for x in range(0, 81)]
        cfg = SimLingoContractConfig(densify_ds_m=10.0)
        got = navigation_targets_official(
            route,
            ego_x=32.0,
            ego_y=0.0,
            ego_yaw=0.0,
            progress_hint_s=32.0,
            config=cfg,
        )
        self.assertTrue(got.valid)
        self.assertLess(got.target1_distance_m, 12.0)
        self.assertGreaterEqual(got.target2_distance_m, 17.0)
        self.assertGreaterEqual(got.target_separation_m, 9.5)

    def test_bottom_crop_and_official_jpeg_pipeline_shape(self) -> None:
        h, w = 512, 1024
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[..., 0] = 10  # B
        bgr[..., 1] = 20  # G
        bgr[..., 2] = 200  # R channel in BGR layout → high "red" in BGR means blue-ish RGB after convert
        # Pattern: make BGR so after BGR2RGB we get distinctive channels
        bgr[:, :, 0] = 30
        bgr[:, :, 1] = 60
        bgr[:, :, 2] = 180
        out = preprocess_camera_official_bgr(bgr)
        expected_h = int(h - (h * 4.8) // 16)
        self.assertEqual(out.shape, (expected_h, w, 3))
        self.assertEqual(out.dtype, np.uint8)
        # RGB: R high from original BGR[...,2]
        self.assertGreater(float(out[:, :, 0].mean()), float(out[:, :, 2].mean()))
        # Legacy RGB path still crops the same way
        rgb = bgr[:, :, ::-1].copy()
        leg = preprocess_camera_legacy_rgb(rgb)
        self.assertEqual(leg.shape, out.shape)

    def test_crop_formula_matches_official_4p8_over_16(self) -> None:
        def official_cut(h: int) -> int:
            return int(h - (h * 4.8) // 16)

        for h in (512, 320, 256, 480):
            cropped = crop_bottom_official(np.zeros((h, 64, 3), dtype=np.uint8))
            self.assertEqual(cropped.shape[0], official_cut(h))

    def test_heads_are_separate_fields_on_result(self) -> None:
        fields = NeuralForwardResult.__dataclass_fields__
        self.assertIn("route_xy", fields)
        self.assertIn("speed_wps_xy", fields)
        self.assertIn("speed_mps", fields)
        # speed converter never returns path-length 20
        speed_wps = [(1.0 * i, 0.0) for i in range(10)]
        speeds = speed_wps_2d_to_mps(speed_wps, dt_s=0.25, n_out=10)
        self.assertEqual(len(speeds), 10)
        self.assertTrue(all(abs(s - 4.0) < 1e-6 for s in speeds))

    def test_s_curve_does_not_trigger_default_lateral_mode_flip(self) -> None:
        mgr = VLAPathManager(PathManagerConfig())  # defaults: flip OFF
        self.assertFalse(mgr.config.enable_lateral_mode_flip)
        left = [(float(x), 0.12 * x) for x in range(1, 21)]
        right = [(float(x), -0.10 * x) for x in range(1, 21)]
        self.assertTrue(
            mgr.update(left, ego=EgoPose(0, 0, 0, 3), target_speed_mps=3.0, stamp_s=1.0).accepted
        )
        nxt = mgr.update(
            right, ego=EgoPose(0, 0, 0, 3), target_speed_mps=3.0, stamp_s=1.5
        )
        if not nxt.accepted:
            self.assertNotEqual(nxt.reason, "lateral_mode_flip")

    def test_legacy_config_switch_exists(self) -> None:
        leg = legacy_contract_config()
        self.assertFalse(leg.official_contract)
        self.assertEqual(leg.camera_width, 640)
        self.assertTrue(leg.lateral_mode_flip_enabled)

    def test_official_desired_speed_matches_agent_formula(self) -> None:
        from driving_vla.model.speed_convert import official_desired_speed_mps

        # Cumulative wps: 0, 1.25, 2.5 m along x → ||wp0-wp2||*2 = 5.0
        wps = [(0.0, 0.0), (1.25, 0.0), (2.5, 0.0), (3.75, 0.0)]
        self.assertAlmostEqual(official_desired_speed_mps(wps), 5.0, places=5)

    def test_inter_forward_variance_formula_is_max_abs_delta(self) -> None:
        """Regression: np.var(stack) confounds spatial path variance with run-to-run."""
        # Two identical routes with large along-track spread.
        a = np.stack([np.linspace(0, 20, 20), np.zeros(20)], axis=1)
        stack = np.stack([a, a], axis=0)
        bogus = float(np.var(stack))
        max_delta = float(np.max(np.abs(stack - stack[0:1])))
        self.assertGreater(bogus, 1.0)
        self.assertEqual(max_delta, 0.0)

    @unittest.skipUnless(
        os.environ.get("SDF_CONTRACT_LIVE_FORWARD", "").strip() in {"1", "true", "TRUE"},
        "set SDF_CONTRACT_LIVE_FORWARD=1 to run 20× model variance (needs GPU+ckpt)",
    )
    def test_fixed_input_route_variance_20_forwards(self) -> None:
        from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

        rt = SimLingoNeuralRuntime()
        rep = rt.load()
        self.assertTrue(rep.ok, rep.error)
        rt.keep_model_on_gpu()
        bgr = np.random.default_rng(0).integers(0, 255, (512, 1024, 3), dtype=np.uint8)
        routes = []
        for _ in range(20):
            out = rt.forward_numpy(
                bgr,
                speed_mps=3.0,
                target_point_xy=(8.0, 0.2),
                target_point2_xy=(10.0, 0.4),
                keep_on_gpu=True,
                image_layout="bgr",
                official_contract=True,
            )
            routes.append(out.route_xy.copy())
        stack = np.stack(routes, axis=0)
        # Correct metric: max abs delta vs first forward (not np.var of geometry).
        max_delta = float(np.max(np.abs(stack - stack[0:1])))
        self.assertLess(max_delta, 1e-5, f"inter-forward max|Δ| too high: {max_delta}")


if __name__ == "__main__":
    unittest.main()
