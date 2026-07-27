from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_live import (  # noqa: E402
    compute_chase_cam_world,
)


class SpectatorFollowTest(unittest.TestCase):
    def test_chase_camera_is_behind_vehicle_at_zero_yaw(self) -> None:
        x, y, z, yaw = compute_chase_cam_world(
            x=10.0,
            y=20.0,
            z=1.0,
            yaw_deg=0.0,
        )
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 20.0)
        self.assertAlmostEqual(z, 6.5)
        self.assertAlmostEqual(yaw, 0.0)

    def test_chase_camera_rotates_with_vehicle_yaw(self) -> None:
        x, y, z, yaw = compute_chase_cam_world(
            x=10.0,
            y=20.0,
            z=2.0,
            yaw_deg=90.0,
        )
        self.assertAlmostEqual(x, 10.0, places=6)
        self.assertAlmostEqual(y, 12.0, places=6)
        self.assertAlmostEqual(z, 7.5)
        self.assertAlmostEqual(yaw, 90.0)


if __name__ == "__main__":
    unittest.main()
