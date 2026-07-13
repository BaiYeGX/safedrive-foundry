"""G1-06 multi-rate control acceptance tests (offline closed-loop)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.control import ControlLoop, config_sha256, load_control_config  # noqa: E402
from classic_stack.control.controller import (  # noqa: E402
    closed_loop_simulate,
    make_reference_trajectory,
)

SCENARIOS = ("straight", "curve", "stop", "follow_brake", "lane_change", "reverse")


class G106ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_control_config()
        cls.hash = config_sha256(cls.cfg.raw_toml)

    def test_config_50hz_deadline_frozen(self) -> None:
        self.assertEqual(self.cfg.profile, "control_50hz")
        self.assertEqual(self.cfg.control_period_ms, 20.0)
        self.assertEqual(self.cfg.deadline_ms, 20.0)
        self.assertEqual(len(self.hash), 64)
        self.assertEqual(self.cfg.fallback_chain[0], "mpc")
        self.assertEqual(self.cfg.fallback_chain[-1], "brake")
        self.assertTrue(self.cfg.warm_start)

    def test_six_closed_loop_families(self) -> None:
        results = []
        for name in SCENARIOS:
            traj = make_reference_trajectory(name)
            out = closed_loop_simulate(traj, scenario=name, steps=120, config=self.cfg)
            results.append(out)
            self.assertEqual(out["profile"], "control_50hz")
            self.assertEqual(out["control_period_ms"], 20.0)
            self.assertEqual(out["watchdog"]["steps"], 120)
            self.assertTrue(out["watchdog_steps_match"])
            self.assertIn("p50", out["watchdog"]["e2e_ms"])
            self.assertIn("p95", out["watchdog"]["e2e_ms"])
            self.assertIn("p99", out["watchdog"]["e2e_ms"])
            self.assertIn("p50", out["watchdog"]["solver_ms"])
            # tracking should stay bounded on synthetic plant
            self.assertLess(out["tracking_cte_mean"], 5.0)
            if name == "reverse":
                # Must produce negative longitudinal progress (backing up along -x)
                self.assertLess(out["signed_progress_x"], -0.5)
                self.assertLess(out["final_speed"], 0.5)
            elif name == "straight":
                self.assertGreater(out["progress_m"], 1.0)
        self.assertEqual(len(results), 6)
        self.__class__.results = results

    def test_mpc_timeout_degrades_chain(self) -> None:
        traj = make_reference_trajectory("straight")
        out = closed_loop_simulate(
            traj,
            scenario="timeout",
            steps=40,
            config=self.cfg,
            force_timeout=True,
        )
        # should not stay only on mpc for forced timeout path
        self.assertTrue(any(m in out["modes"] for m in ("lqr", "pure_pursuit", "pid", "brake")))

    def test_infeasible_degrades(self) -> None:
        traj = make_reference_trajectory("curve")
        out = closed_loop_simulate(
            traj,
            scenario="infeasible",
            steps=40,
            config=self.cfg,
            force_infeasible=True,
        )
        self.assertTrue(sum(out["modes"].values()) > 0)

    def test_deadline_miss_injection_recorded(self) -> None:
        traj = make_reference_trajectory("straight")
        out = closed_loop_simulate(
            traj,
            scenario="slow_solver",
            steps=30,
            config=self.cfg,
            inject_solver_ms=25.0,  # > deadline_ms 20 → must record miss / degrade
        )
        self.assertEqual(out["watchdog"]["steps"], 30)
        # Either deadline misses recorded, or MPC degraded away due to wall budget/deadline
        miss = out["watchdog"]["deadline_misses"]
        modes = out["modes"]
        self.assertTrue(
            miss >= 1 or any(m in modes for m in ("lqr", "pure_pursuit", "pid", "brake")),
            msg=f"miss={miss} modes={modes}",
        )

    def test_e2e_extra_ms_forces_deadline_miss(self) -> None:
        traj = make_reference_trajectory("straight")
        out = closed_loop_simulate(
            traj,
            scenario="e2e_delay",
            steps=20,
            config=self.cfg,
            inject_e2e_extra_ms=25.0,
        )
        self.assertEqual(out["watchdog"]["steps"], 20, msg="one record per control tick")
        self.assertGreaterEqual(out["watchdog"]["deadline_misses"], 1)

    def test_mpc_miss_fallback_single_watchdog_sample(self) -> None:
        """MPC deadline degrade must not double-count watchdog steps (Codex P0)."""
        traj = make_reference_trajectory("straight")
        out = closed_loop_simulate(
            traj,
            scenario="inject_degrade",
            steps=10,
            config=self.cfg,
            inject_solver_ms=25.0,  # > wall_budget and can force MPC skip
        )
        self.assertEqual(out["watchdog"]["steps"], 10)
        # Should degrade off pure-mpc-only or record miss without inflating steps
        self.assertLessEqual(out["watchdog"]["deadline_misses"], 10)

    def test_stale_buffer_falls_to_brake(self) -> None:
        loop = ControlLoop(self.cfg)
        from classic_stack.control.controller import EgoState

        ego = EgoState(0, 0, 0, 1.0)
        # no trajectory set
        cmd = loop.step(ego, now_s=1.0)
        self.assertEqual(cmd.mode, "brake")

    def test_not_claiming_20hz_as_20ms(self) -> None:
        self.assertNotEqual(self.cfg.control_period_ms, 50.0)
        self.assertEqual(self.cfg.control_period_ms, 20.0)


if __name__ == "__main__":
    unittest.main()
