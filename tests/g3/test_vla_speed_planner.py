from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.vla_speed_planner import VLASpeedConfig, VLASpeedPlanner  # noqa: E402


class VLASpeedPlannerTest(unittest.TestCase):
    def test_max_speed_is_a_cap_not_a_floor(self) -> None:
        planner = VLASpeedPlanner(
            VLASpeedConfig(max_speed_mps=15.0, calibration_gain=1.0, max_accel_mps2=100.0)
        )
        decision = planner.update([4.0] * 5, dt_s=0.5)
        self.assertAlmostEqual(decision.raw_speed_mps, 4.0)
        self.assertAlmostEqual(decision.target_speed_mps, 4.0)
        self.assertLess(decision.target_speed_mps, 15.0)

    def test_default_calibration_gain_is_one(self) -> None:
        self.assertAlmostEqual(VLASpeedConfig().calibration_gain, 1.0)

    def test_acceleration_is_slew_limited_but_braking_is_immediate(self) -> None:
        planner = VLASpeedPlanner(
            VLASpeedConfig(max_speed_mps=15.0, calibration_gain=1.0, max_accel_mps2=2.0)
        )
        accelerated = planner.update([12.0] * 5, dt_s=0.5)
        self.assertAlmostEqual(accelerated.target_speed_mps, 1.0)
        planner.reset(target_speed_mps=10.0)
        braked = planner.update([3.0] * 5, dt_s=0.5)
        self.assertAlmostEqual(braked.target_speed_mps, 3.0)

    def test_stop_and_invalid_output_fail_to_zero(self) -> None:
        planner = VLASpeedPlanner(VLASpeedConfig(max_speed_mps=15.0))
        planner.reset(target_speed_mps=8.0)
        stopped = planner.update([0.1, 0.0, 0.2], dt_s=0.5)
        self.assertTrue(stopped.stop_requested)
        self.assertEqual(stopped.target_speed_mps, 0.0)
        self.assertEqual(stopped.stop_source, "vla_head")
        planner.reset(target_speed_mps=8.0)
        invalid = planner.update([math.nan], dt_s=0.5)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.target_speed_mps, 0.0)

    def test_launch_hysteresis_ignores_brief_blip_above_stop_threshold(self) -> None:
        """0.34↔0.36 chatter must not release stop without sustained confirm."""
        cfg = VLASpeedConfig(
            max_speed_mps=15.0,
            stop_threshold_mps=0.35,
            launch_threshold_mps=0.50,
            launch_confirm_frames=4,
            max_accel_mps2=100.0,
        )
        planner = VLASpeedPlanner(cfg)
        planner.reset(target_speed_mps=0.0)
        # Enter stop.
        d0 = planner.update([0.2] * 5, dt_s=0.05, ego_speed_mps=0.0)
        self.assertTrue(d0.stop_requested)
        # Two frames at 0.37 — below launch_threshold 0.50 and < confirm.
        for _ in range(2):
            d = planner.update([0.37] * 5, dt_s=0.05, ego_speed_mps=0.0)
            self.assertTrue(d.stop_requested)
            self.assertEqual(d.stop_source, "launch_hysteresis")
        # Sustained above launch threshold → release.
        for _ in range(4):
            d = planner.update([0.55] * 5, dt_s=0.05, ego_speed_mps=0.0)
        self.assertFalse(d.stop_requested)
        self.assertGreater(d.target_speed_mps, 0.5)

    def test_execution_stale_recovery_allows_launch_from_subthreshold_vla(self) -> None:
        """After execution stale + fresh path, raw≈0.30 can restart via MPC floor."""
        cfg = VLASpeedConfig(
            max_speed_mps=15.0,
            stop_threshold_mps=0.35,
            launch_threshold_mps=0.50,
            launch_confirm_frames=4,
            recovery_raw_min_mps=0.20,
            recovery_launch_floor_mps=1.00,
            recovery_confirm_frames=3,
            recovery_window_s=4.0,
            max_accel_mps2=100.0,
        )
        planner = VLASpeedPlanner(cfg)
        planner.reset(target_speed_mps=0.0)
        planner.notify_execution_stale_stop()
        self.assertTrue(planner.execution_stale_latched)
        planner.notify_path_accepted(reanchor=True, path_age_s=0.0)
        self.assertFalse(planner.execution_stale_latched)
        d = None
        for _ in range(3):
            d = planner.update([0.30] * 5, dt_s=0.05, ego_speed_mps=0.05)
        assert d is not None
        self.assertFalse(d.stop_requested)
        self.assertGreaterEqual(d.target_speed_mps, 0.99)
        self.assertTrue(d.recovery_active or d.recovery_timer_s >= 0.0)

    def test_path_accept_alone_never_overrides_semantic_vla_stop(self) -> None:
        """A red-light-like VLA stop is not recovery-authorized by geometry."""
        planner = VLASpeedPlanner(
            VLASpeedConfig(
                max_accel_mps2=100.0,
                recovery_confirm_frames=2,
            )
        )
        planner.reset(target_speed_mps=0.0)
        stopped = planner.update([0.25] * 5, dt_s=0.05, ego_speed_mps=0.0)
        self.assertTrue(stopped.stop_requested)
        self.assertEqual(stopped.stop_source, "launch_hysteresis")

        # Even the legacy reanchor marker has no authority without a previously
        # observed MPC freshness stop.
        planner.notify_path_accepted(reanchor=True, path_age_s=0.0)
        for _ in range(4):
            stopped = planner.update([0.30] * 5, dt_s=0.05, ego_speed_mps=0.0)
        self.assertTrue(stopped.stop_requested)
        self.assertFalse(stopped.recovery_active)
        self.assertEqual(stopped.target_speed_mps, 0.0)


if __name__ == "__main__":
    unittest.main()
