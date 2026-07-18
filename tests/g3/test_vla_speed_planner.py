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
            VLASpeedConfig(max_speed_mps=15.0, calibration_gain=1.5, max_accel_mps2=100.0)
        )
        decision = planner.update([4.0] * 5, dt_s=0.5)
        self.assertAlmostEqual(decision.raw_speed_mps, 4.0)
        self.assertAlmostEqual(decision.target_speed_mps, 6.0)
        self.assertLess(decision.target_speed_mps, 15.0)

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
        planner.reset(target_speed_mps=8.0)
        invalid = planner.update([math.nan], dt_s=0.5)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.target_speed_mps, 0.0)


if __name__ == "__main__":
    unittest.main()
