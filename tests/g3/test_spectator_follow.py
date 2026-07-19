"""Offline tests for chase-cam height (no CARLA)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "g3"))

from run_g3_vla_v0_visual_demo import (  # noqa: E402
    SPECTATOR_BACK_M,
    SPECTATOR_UP_M,
    _spectator_pose_is_fallen,
    compute_chase_cam_world,
)


class SpectatorChaseCamTest(unittest.TestCase):
    def test_high_elevation_town15_camera_stays_above_vehicle(self) -> None:
        # Town15-like absolute Z; old z>8 gate would pin camera near z=7 underground.
        z_vehicle = 116.4
        cx, cy, cz, yaw = compute_chase_cam_world(
            x=394.0, y=163.0, z=z_vehicle, yaw_deg=40.0
        )
        self.assertAlmostEqual(cz, z_vehicle + SPECTATOR_UP_M, places=5)
        self.assertGreater(cz, 120.0)
        self.assertNotAlmostEqual(cz, 7.0, places=0)
        # Behind the car along yaw.
        yaw_rad = math.radians(40.0)
        self.assertAlmostEqual(cx, 394.0 - SPECTATOR_BACK_M * math.cos(yaw_rad), places=5)
        self.assertAlmostEqual(cy, 163.0 - SPECTATOR_BACK_M * math.sin(yaw_rad), places=5)
        self.assertAlmostEqual(yaw, 40.0)

    def test_low_town_also_uses_relative_height(self) -> None:
        cx, cy, cz, _ = compute_chase_cam_world(x=0.0, y=0.0, z=0.4, yaw_deg=0.0)
        self.assertAlmostEqual(cz, 0.4 + SPECTATOR_UP_M, places=5)
        self.assertAlmostEqual(cx, -SPECTATOR_BACK_M, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_fallen_is_relative_drop_not_absolute_altitude(self) -> None:
        last = {"x": 0.0, "y": 0.0, "z": 116.0, "yaw": 0.0}
        self.assertFalse(_spectator_pose_is_fallen(116.0, last_good=last))
        self.assertFalse(_spectator_pose_is_fallen(100.0, last_good=last))  # mild change
        self.assertTrue(_spectator_pose_is_fallen(90.0, last_good=last))  # -26m drop
        self.assertTrue(_spectator_pose_is_fallen(float("nan"), last_good=last))
        # High absolute Z alone is NOT fallen when no last_good history.
        self.assertFalse(_spectator_pose_is_fallen(200.0, last_good=None))


if __name__ == "__main__":
    unittest.main()
