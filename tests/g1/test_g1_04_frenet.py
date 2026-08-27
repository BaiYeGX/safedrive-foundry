"""G1-04 Frenet lattice + S-T speed acceptance tests."""

from __future__ import annotations

import json
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.planning.frenet import (  # noqa: E402
    CenterlineConstantSpeedPlanner,
    FrenetPlanner,
    config_sha256,
    load_frenet_st_config,
)
from classic_stack.planning.frenet.scenarios import SCENARIO_KINDS, make_scenario  # noqa: E402


class G104FrenetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_frenet_st_config()
        cls.config_hash = config_sha256(cls.config.raw_toml)
        cls.planner = FrenetPlanner(cls.config)
        cls.baseline = CenterlineConstantSpeedPlanner(cls.config)

    def test_config_is_versioned_and_hashed(self) -> None:
        self.assertTrue(self.config.name)
        self.assertIn("frenet_st", self.config.schema_version)
        self.assertEqual(len(self.config_hash), 64)
        self.assertEqual(self.planner.config_hash, self.config_hash)
        self.assertEqual(self.config.prediction_model, "cv_ctrv_idm")

    def test_five_scenarios_solvable_or_diagnosable(self) -> None:
        for kind in SCENARIO_KINDS:
            with self.subTest(kind=kind):
                req = make_scenario(kind, seed=1)
                result = self.planner.plan(req)
                if result.ok:
                    self.assertIsNotNone(result.trajectory)
                    assert result.trajectory is not None
                    pts = result.trajectory.points
                    self.assertGreaterEqual(len(pts), 2)
                    for p in pts:
                        for field in ("t", "x", "y", "yaw", "kappa", "v", "a", "jerk"):
                            self.assertTrue(hasattr(p, field))
                    # hard constraint soft checks
                    self.assertTrue(all(p.v >= -1e-3 for p in pts))
                    self.assertTrue(all(abs(p.kappa) <= self.config.vehicle.max_curvature_per_m + 1e-3 for p in pts))
                else:
                    self.assertIsNotNone(result.failure_code)
                    self.assertIsInstance(result.reject_reasons, dict)

    def test_stop_reaches_low_speed(self) -> None:
        result = self.planner.plan(make_scenario("stop", seed=2))
        self.assertTrue(result.ok, result.failure_code)
        assert result.trajectory is not None
        self.assertLess(result.trajectory.points[-1].v, 1.25)
        # Braking distance consistency: final s should be near stop region
        pts = result.trajectory.points
        self.assertGreater(len(pts), 3)
        # Speed should generally decrease toward stop (not force-appended single zero)
        self.assertLess(pts[-1].v, pts[0].v + 0.5)

    def test_st_dp_unit_kinematics_and_block(self) -> None:
        from classic_stack.geometry import VehicleParams
        from classic_stack.planning.speed.st_dp import (
            STGrid,
            solve_st_dp,
            validate_profile_kinematics,
            smooth_jerk,
        )

        vehicle = VehicleParams()
        # Empty free grid
        s_bins, t_bins = 40, 25
        free = [[False] * s_bins for _ in range(t_bins)]
        grid = STGrid(ds=1.0, dt=0.2, s_bins=s_bins, t_bins=t_bins, occupied=free)
        profile, err = solve_st_dp(grid, v0=6.0, v_target=6.0, vehicle=vehicle, stop_at_s=None)
        self.assertIsNone(err, err)
        self.assertGreaterEqual(len(profile), 2)
        self.assertAlmostEqual(profile[-1][0], (t_bins - 1) * grid.dt)
        self.assertIsNone(validate_profile_kinematics(profile, vehicle=vehicle))
        # Wall of occupancy should block progress
        wall = [[False] * s_bins for _ in range(t_bins)]
        for ti in range(t_bins):
            for si in range(8, 12):
                wall[ti][si] = True
        grid_b = STGrid(ds=1.0, dt=0.2, s_bins=s_bins, t_bins=t_bins, occupied=wall)
        prof_b, err_b = solve_st_dp(grid_b, v0=8.0, v_target=8.0, vehicle=vehicle)
        if prof_b:
            # must not deeply enter wall region
            self.assertTrue(all(s < 12.5 for _, s, _, _ in prof_b))
        # Impossible stop: stop behind immediate solid wall with no room to brake from high speed
        wall2 = [[False] * s_bins for _ in range(t_bins)]
        for ti in range(t_bins):
            for si in range(3, s_bins):
                wall2[ti][si] = True
        grid_s = STGrid(ds=1.0, dt=0.2, s_bins=s_bins, t_bins=t_bins, occupied=wall2)
        _, err_s = solve_st_dp(grid_s, v0=10.0, v_target=0.0, vehicle=vehicle, stop_at_s=20.0)
        self.assertIsNotNone(err_s)
        self.assertIn(err_s, ("ST_INFEASIBLE_STOP", "ST_NO_PATH", "ST_BLOCKED", "ST_START_OCCUPIED"))
        # A distant stop beyond this finite horizon returns a safe approach
        # prefix, not a fabricated terminal stop or an unconditional reject.
        prefix, prefix_err = solve_st_dp(
            grid, v0=6.0, v_target=6.0, vehicle=vehicle, stop_at_s=40.0
        )
        self.assertIsNone(prefix_err, prefix_err)
        self.assertGreaterEqual(len(prefix), 2)
        self.assertLessEqual(prefix[-1][1], 40.0 + 1e-6)
        remaining = max(0.0, 40.0 - prefix[-1][1])
        self.assertLessEqual(prefix[-1][2], (2.0 * vehicle.max_decel_mps2 * remaining) ** 0.5 + 0.5)
        # Feasible stop on free grid (horizon must cover braking + approach)
        grid_long = STGrid(ds=1.0, dt=0.2, s_bins=50, t_bins=40, occupied=[[False] * 50 for _ in range(40)])
        prof_stop, err_stop = solve_st_dp(
            grid_long, v0=6.0, v_target=0.0, vehicle=vehicle, stop_at_s=18.0
        )
        self.assertIsNone(err_stop, err_stop)
        assert prof_stop is not None
        self.assertLess(prof_stop[-1][2], 0.95)
        self.assertLess(abs(prof_stop[-1][1] - 18.0), 5.0)
        # No forced terminal-only zero: last few samples should be low speed
        if len(prof_stop) >= 3:
            self.assertLess(prof_stop[-2][2], 2.5)
        sm = smooth_jerk(prof_stop, window=3, max_jerk=vehicle.max_jerk_mps3, vehicle=vehicle)
        self.assertGreaterEqual(len(sm), 2)
        for i in range(1, len(sm)):
            self.assertLessEqual(abs(sm[i][4]), vehicle.max_jerk_mps3 + 1e-3)

    def test_trajectory_adjacent_kinematics(self) -> None:
        result = self.planner.plan(make_scenario("follow", seed=1))
        self.assertTrue(result.ok, result.failure_code)
        assert result.trajectory is not None
        pts = result.trajectory.points
        for i in range(1, len(pts)):
            dt = pts[i].t - pts[i - 1].t
            self.assertGreater(dt, 0.0)
            # rough consistency: speed non-negative; accel within loose bounds
            self.assertGreaterEqual(pts[i].v, -1e-3)
            self.assertLessEqual(pts[i].a, self.config.vehicle.max_accel_mps2 + 0.5)
            self.assertGreaterEqual(pts[i].a, -self.config.vehicle.max_decel_mps2 - 0.5)
            self.assertLessEqual(abs(pts[i].jerk), self.config.vehicle.max_jerk_mps3 + 0.5)

    def test_lane_change_prefers_offset(self) -> None:
        result = self.planner.plan(make_scenario("lane_change", seed=3))
        self.assertTrue(result.ok, msg=f"{result.failure_code} {result.reject_reasons}")
        assert result.trajectory is not None
        # path should leave centerline: final y magnitude larger than start
        y0 = result.trajectory.points[0].y
        y1 = result.trajectory.points[-1].y
        self.assertGreater(abs(y1 - y0), 0.3)

    def test_constraint_rejects_extreme_offset(self) -> None:
        from classic_stack.geometry import ReferencePath
        from classic_stack.planning.frenet.planner import PlanRequest

        ref = ReferencePath.from_xy([0, 10, 20, 30], [0, 0, 0, 0])
        req = PlanRequest(reference=ref, preferred_offset_m=10.0, scenario_kind="cruise", v0=5.0)
        result = self.planner.plan(req)
        # Extreme preferred offset is sampled then rejected at road boundary.
        # Other legal offsets may still succeed; the reject histogram must record it.
        self.assertGreaterEqual(
            result.reject_reasons.get("road_boundary", 0),
            1,
            msg=result.reject_reasons,
        )

    def test_multi_seed_stable_success_flags(self) -> None:
        for kind in SCENARIO_KINDS:
            flags = []
            for seed in (1, 2, 3):
                flags.append(self.planner.plan(make_scenario(kind, seed=seed)).ok)
            # fixed synthetic world: seeds should not flip feasibility for fixed sampling
            self.assertEqual(len(set(flags)), 1, msg=f"{kind} seed instability {flags}")

    def test_baseline_comparison_table(self) -> None:
        rows = []
        times_frenet: list[float] = []
        for kind in SCENARIO_KINDS:
            req = make_scenario(kind, seed=7)
            fr = self.planner.plan(req)
            bl = self.baseline.plan(req)
            times_frenet.append(fr.wall_time_ms)
            rows.append(
                {
                    "scenario": kind,
                    "frenet_ok": fr.ok,
                    "baseline_ok": bl.ok,
                    "frenet_ms": fr.wall_time_ms,
                    "baseline_ms": bl.wall_time_ms,
                    "frenet_candidates": fr.candidates,
                    "frenet_fail": fr.failure_code,
                    "baseline_fail": bl.failure_code,
                }
            )
        self.assertEqual(len(rows), 5)
        # frenet should solve at least the core set
        ok_count = sum(1 for r in rows if r["frenet_ok"])
        # Core follow/stop/cut_in should solve; avoid/lane_change may fail diagnostically.
        self.assertGreaterEqual(ok_count, 3, msg=rows)
        by = {r["scenario"]: r for r in rows}
        self.assertTrue(by["follow"]["frenet_ok"], by["follow"])
        self.assertTrue(by["stop"]["frenet_ok"], by["stop"])
        p50 = statistics.median(times_frenet)
        times_sorted = sorted(times_frenet)
        p95 = times_sorted[max(0, int(0.95 * (len(times_sorted) - 1)))]
        p99 = times_sorted[-1]
        self.assertGreaterEqual(p99, p95)
        self.assertGreaterEqual(p95, p50)
        # attach for evidence writer convenience
        self.__class__.comparison_rows = rows
        self.__class__.timings = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "samples": times_frenet}

    def test_reject_reasons_recorded(self) -> None:
        result = self.planner.plan(make_scenario("avoid", seed=1))
        # success or failure both should populate candidates and reject histogram type
        self.assertGreaterEqual(result.candidates, 0)
        self.assertIsInstance(result.reject_reasons, dict)

    def test_g1_01_to_g1_03_not_broken(self) -> None:
        from classic_stack.map import load_map_fixture
        from classic_stack.route import RoutePlanner, RouteRequest

        graph = load_map_fixture("Town01")
        plan = RoutePlanner(graph).plan(
            RouteRequest(start_node_id="R1:S0:L-1", goal_node_ids=("R2:S0:L-1",), seed=1)
        )
        self.assertTrue(plan.ok)


if __name__ == "__main__":
    unittest.main()
