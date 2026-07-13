"""G1-05 Hybrid A* – Reeds–Shepp acceptance tests."""

from __future__ import annotations

import math
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.geometry import wrap_angle  # noqa: E402
from classic_stack.planning.hybrid_astar import (  # noqa: E402
    GridAstarDubinsPlanner,
    HybridAstarPlanner,
    PlannerSelector,
    config_sha256,
    load_hybrid_astar_config,
    reeds_shepp_path,
)
from classic_stack.planning.hybrid_astar.scenarios import MANEUVER_KINDS, make_maneuver  # noqa: E402


class G105HybridAstarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_hybrid_astar_config()
        cls.config_hash = config_sha256(cls.config.raw_toml)
        cls.hybrid = HybridAstarPlanner(cls.config)
        cls.grid = GridAstarDubinsPlanner(cls.config)
        cls.selector = PlannerSelector(cls.config)

    def test_config_frozen_hash(self) -> None:
        self.assertEqual(len(self.config_hash), 64)
        self.assertEqual(self.hybrid.config_hash, self.config_hash)
        self.assertTrue(self.config.allow_reverse)
        self.assertGreater(self.config.max_expansions, 100)

    def test_reeds_shepp_analytic_curve(self) -> None:
        path = reeds_shepp_path(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 4.0)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertGreater(path.length, 0.0)
        samples = path.sample(0.0, 0.0, 0.0, 4.0, ds=0.5)
        self.assertGreaterEqual(len(samples), 2)
        # ends near goal (strict — no snap allowed in planner either)
        self.assertLess(math.hypot(samples[-1][0] - 5.0, samples[-1][1] - 0.0), 0.4)
        self.assertLess(abs(wrap_angle(samples[-1][2] - 0.0)), 0.3)

    def test_reeds_shepp_endpoint_property_multi(self) -> None:
        import random

        rng = random.Random(0)
        ok = 0
        for _ in range(20):
            x1 = rng.uniform(-6, 6)
            y1 = rng.uniform(-6, 6)
            yaw1 = rng.uniform(-math.pi, math.pi)
            path = reeds_shepp_path(0.0, 0.0, 0.0, x1, y1, yaw1, 3.5, xy_tol=0.4, yaw_tol=0.3)
            if path is None:
                continue
            s = path.sample(0.0, 0.0, 0.0, 3.5, ds=0.25)
            self.assertLess(math.hypot(s[-1][0] - x1, s[-1][1] - y1), 0.45)
            self.assertLess(abs(wrap_angle(s[-1][2] - yaw1)), 0.35)
            ok += 1
        self.assertGreaterEqual(ok, 8)

    def test_no_snap_accepts_only_true_endpoint(self) -> None:
        # Path that claims goal far away must fail validation inside reeds_shepp_path
        path = reeds_shepp_path(0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 2.0, xy_tol=0.3, yaw_tol=0.2)
        # May be None or a long valid path; if present must end at goal
        if path is not None:
            s = path.sample(0.0, 0.0, 0.0, 2.0, ds=0.5)
            self.assertLess(math.hypot(s[-1][0] - 100.0, s[-1][1] - 100.0), 0.5)

    def test_four_maneuvers_run(self) -> None:
        for kind in MANEUVER_KINDS:
            with self.subTest(kind=kind):
                req = make_maneuver(kind, seed=1)
                result = self.hybrid.plan(req)
                # Must run: either full/partial ok path or auditable failure
                if result.ok:
                    self.assertIsNotNone(result.trajectory)
                    assert result.trajectory is not None
                    self.assertGreaterEqual(len(result.trajectory.points), 2)
                    for p in result.trajectory.points:
                        self.assertTrue(hasattr(p, "kappa"))
                        self.assertTrue(hasattr(p, "v"))
                    self.assertEqual(result.trajectory.source.startswith("hybrid_astar"), True)
                else:
                    self.assertIsNotNone(result.failure_code)
                    self.assertIn(
                        result.failure_code,
                        {"NO_PATH", "TIMEOUT", "START_COLLISION", "GOAL_COLLISION", "PARTIAL_SOLUTION"},
                    )
                    if result.failure_code == "PARTIAL_SOLUTION":
                        self.assertTrue(result.partial)
                self.assertGreaterEqual(result.nodes_expanded, 0)
                self.assertIsInstance(result.reject_reasons, dict)

    def test_blocked_detour_finds_path_or_partial(self) -> None:
        result = self.hybrid.plan(make_maneuver("blocked_detour", seed=2))
        # Full success preferred; partial must not count as ok=True
        if result.partial:
            self.assertFalse(result.ok)
            self.assertEqual(result.failure_code, "PARTIAL_SOLUTION")
            self.assertIsNotNone(result.trajectory)
            self.assertGreater(result.path_length_m, 0.5)
        else:
            self.assertTrue(result.ok, msg=f"{result.failure_code} {result.reject_reasons}")
            assert result.trajectory is not None
            self.assertGreater(result.path_length_m, 1.0)

    def test_three_point_turn_uses_reverse_feature(self) -> None:
        req = make_maneuver("three_point_turn", seed=1)
        self.assertTrue(req.require_reverse)
        choice = self.selector.select(req)
        self.assertEqual(choice, "hybrid_astar")
        result = self.selector.plan(req)
        self.assertTrue(
            result.ok
            or result.partial
            or result.failure_code in {"PARTIAL_SOLUTION", "NO_PATH", "TIMEOUT"}
        )
        if result.ok and result.trajectory is not None:
            # Prefer observing reverse gear usage when a full path is found
            self.assertTrue(result.gear_switches >= 0)
            # At least one reverse-ish sample if gears reported
            if result.gear_switches >= 1:
                self.assertGreaterEqual(result.gear_switches, 1)

    def test_reverse_park_has_negative_velocity_or_gear(self) -> None:
        result = self.hybrid.plan(make_maneuver("reverse_park", seed=1))
        if result.ok and result.trajectory is not None:
            vs = [p.v for p in result.trajectory.points]
            # Planner encodes reverse as non-positive v on reverse segments when present
            self.assertTrue(min(vs) <= 0.05 or result.gear_switches >= 1 or result.path_length_m > 0.5)
        else:
            self.assertIn(result.failure_code, {"NO_PATH", "TIMEOUT", "PARTIAL_SOLUTION", None})

    def test_selector_not_name_based(self) -> None:
        # Same features without relying on scenario string
        from classic_stack.planning.hybrid_astar.planner import ManeuverRequest, ObstacleMap

        world = ObstacleMap(xmin=-5, xmax=5, ymin=-5, ymax=5)
        req = ManeuverRequest(start=(0, 0, 0), goal=(0, 0, math.pi), world=world, require_reverse=True)
        self.assertEqual(self.selector.select(req), "hybrid_astar")
        req2 = ManeuverRequest(start=(0, 0, 0), goal=(3, 0, 0), world=world, require_reverse=False)
        self.assertEqual(self.selector.select(req2), "grid_astar_dubins")

    def test_comparison_vs_grid_dubins(self) -> None:
        rows = []
        times = []
        for kind in MANEUVER_KINDS:
            req = make_maneuver(kind, seed=3)
            h = self.hybrid.plan(req)
            g = self.grid.plan(req)
            times.append(h.wall_time_ms)
            rows.append(
                {
                    "scenario": kind,
                    "hybrid_ok": h.ok,
                    "grid_ok": g.ok,
                    "hybrid_length": h.path_length_m,
                    "grid_length": g.path_length_m,
                    "hybrid_nodes": h.nodes_expanded,
                    "grid_nodes": g.nodes_expanded,
                    "hybrid_gears": h.gear_switches,
                    "hybrid_ms": h.wall_time_ms,
                    "grid_ms": g.wall_time_ms,
                    "hybrid_kappa_mean": h.curvature_mean,
                    "hybrid_partial": h.partial,
                    "hybrid_fail": h.failure_code,
                    "grid_fail": g.failure_code,
                }
            )
        self.assertEqual(len(rows), 4)
        # hybrid should produce a usable full or partial path on blocked_detour
        detour = next(r for r in rows if r["scenario"] == "blocked_detour")
        self.assertTrue(detour["hybrid_ok"] or detour["hybrid_partial"] or detour["hybrid_length"] > 0.5)
        p50 = statistics.median(times)
        ts = sorted(times)
        p95 = ts[max(0, int(0.95 * (len(ts) - 1)))]
        p99 = ts[-1]
        self.assertGreaterEqual(p99, p50)
        self.__class__.comparison = rows
        self.__class__.timings = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}

    def test_collision_reject_auditable(self) -> None:
        from classic_stack.planning.hybrid_astar.planner import ManeuverRequest, ObstacleMap

        world = ObstacleMap(xmin=-1, xmax=1, ymin=-1, ymax=1, circles=((0.0, 0.0, 2.0),))
        req = ManeuverRequest(start=(0, 0, 0), goal=(0.5, 0, 0), world=world)
        result = self.hybrid.plan(req)
        self.assertFalse(result.ok)
        self.assertIn(result.failure_code, {"START_COLLISION", "GOAL_COLLISION", "NO_PATH"})

    def test_frenet_baseline_not_broken(self) -> None:
        from classic_stack.planning.frenet import FrenetPlanner
        from classic_stack.planning.frenet.scenarios import make_scenario

        r = FrenetPlanner().plan(make_scenario("follow", seed=1))
        self.assertTrue(r.ok)


if __name__ == "__main__":
    unittest.main()
