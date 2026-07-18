from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.path_manager import (  # noqa: E402
    EgoPose,
    PathManagerConfig,
    VLAPathManager,
)


class VLAPathManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ego = EgoPose(0.0, 0.0, 0.0, 2.0)
        self.manager = VLAPathManager(
            PathManagerConfig(
                max_abs_curvature=0.30,
                max_switch_lateral_5m=1.0,
                max_switch_heading_5m_deg=12.0,
            )
        )

    def test_straight_path_is_dense_and_accepted(self) -> None:
        points = [(float(x), 0.0) for x in range(1, 21)]
        update = self.manager.update(points, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0)
        self.assertTrue(update.accepted, update.reason)
        self.assertIsNotNone(update.committed)
        assert update.committed is not None
        self.assertGreater(update.committed.length_m, 18.0)
        self.assertLess(float(np.max(np.abs(update.committed.y))), 1e-8)
        self.assertLess(float(np.max(np.abs(update.committed.kappa))), 1e-6)

    def test_near_prefix_is_committed_while_tail_blends(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0).accepted
        )
        gentle = [(float(x), 0.04 * x) for x in range(1, 21)]
        update = self.manager.update(gentle, ego=self.ego, target_speed_mps=2.0, stamp_s=1.5)
        self.assertTrue(update.accepted, update.reason)
        assert update.committed is not None
        _x, y, _yaw, _k = update.committed.sample(np.array([1.0, 8.0]))
        self.assertLess(abs(float(y[0])), 0.02)
        self.assertGreater(float(y[1]), 0.05)
        self.assertLess(float(y[1]), 0.32)

    def test_large_lateral_jump_holds_last_committed_path(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        first = self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0)
        jumped = [(float(x), 0.4 * x) for x in range(1, 21)]
        update = self.manager.update(jumped, ego=self.ego, target_speed_mps=2.0, stamp_s=1.5)
        self.assertFalse(update.accepted)
        self.assertIn(update.reason, {"lateral_switch", "heading_switch", "curvature_limit"})
        self.assertIs(update.committed, first.committed)

    def test_rejected_geometry_can_still_reduce_vla_speed(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        first = self.manager.update(straight, ego=self.ego, target_speed_mps=8.0, stamp_s=1.0)
        assert first.committed is not None
        jumped = [(float(x), 0.4 * x) for x in range(1, 21)]
        update = self.manager.update(jumped, ego=self.ego, target_speed_mps=0.0, stamp_s=1.5)
        self.assertFalse(update.accepted)
        assert update.committed is not None
        self.assertEqual(update.committed.target_speed_mps, 0.0)
        self.assertEqual(update.committed.stamp_s, first.committed.stamp_s)
        np.testing.assert_allclose(update.committed.x, first.committed.x)

    def test_accepted_geometry_does_not_temporally_average_a_vla_stop(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.manager.update(straight, ego=self.ego, target_speed_mps=8.0, stamp_s=1.0)
        gentle = [(float(x), 0.02 * x) for x in range(1, 21)]
        stopped = self.manager.update(gentle, ego=self.ego, target_speed_mps=0.0, stamp_s=1.5)
        self.assertTrue(stopped.accepted, stopped.reason)
        assert stopped.committed is not None
        self.assertEqual(stopped.committed.target_speed_mps, 0.0)

    def test_self_intersection_is_rejected(self) -> None:
        loop = [(1.0, 0.0), (3.0, 2.0), (1.0, 2.0), (3.0, 0.0), (5.0, 0.0)]
        update = self.manager.update(loop, ego=self.ego, target_speed_mps=1.0, stamp_s=1.0)
        self.assertFalse(update.accepted)
        self.assertEqual(update.reason, "self_intersection")


if __name__ == "__main__":
    unittest.main()
