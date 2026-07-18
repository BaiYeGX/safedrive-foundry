"""G3-04: V1 K2, oracle best-of-K, collapse detection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.backbone_loader import SimLingoCheckpointHandle  # noqa: E402
from driving_vla.model.v1_policy import (  # noqa: E402
    V1Policy,
    detect_collapse,
    max_position_separation,
    oracle_best_of_k,
)


class TestV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ckpt = ROOT / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
        if not ckpt.is_file():
            raise unittest.SkipTest("no ckpt")
        cls.handle = SimLingoCheckpointHandle(ckpt)
        if not cls.handle.load().ok:
            raise unittest.SkipTest("ckpt load failed")

    def test_k2_shape(self) -> None:
        p = V1Policy(self.handle)
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=6.0,
            route_xy=tuple((float(i), 0.0) for i in range(40)),
            ego_history=((0.0, 0.0, 0.0, 5.0), (0.1, 0.0, 0.0, 5.5), (0.2, 0.0, 0.0, 6.0), (0.3, 0.0, 0.0, 6.0)),
        )
        arrs = p.predict_arrays(obs)
        self.assertEqual(len(arrs), 2)
        self.assertEqual(arrs[0].t_steps, 10)
        self.assertEqual(arrs[1].t_steps, 10)
        self.assertAlmostEqual(arrs[0].probability + arrs[1].probability, 1.0, places=5)
        # Spatial fork required for World/oracle selection space
        self.assertFalse(detect_collapse(arrs, eps=0.05))
        self.assertGreater(max_position_separation(arrs), 0.15)
        bi, score = oracle_best_of_k(arrs)
        self.assertIn(bi, (0, 1))
        self.assertGreater(score, 0.0)

    def test_oracle_with_expert(self) -> None:
        p = V1Policy(self.handle)
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=5.0,
            route_xy=tuple((float(i), 0.0) for i in range(40)),
        )
        arrs = p.predict_arrays(obs)
        bi, ade = oracle_best_of_k(arrs, expert=arrs[0])
        self.assertEqual(bi, 0)
        self.assertAlmostEqual(ade, 0.0, places=5)
        # Prefer slower/offset conservative when expert matches it
        bi2, ade2 = oracle_best_of_k(arrs, expert=arrs[1])
        self.assertEqual(bi2, 1)
        self.assertAlmostEqual(ade2, 0.0, places=5)

    def test_kinematics_reintegrated(self) -> None:
        p = V1Policy(self.handle)
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=5.0,
            route_xy=tuple((float(i), 0.0) for i in range(40)),
        )
        arrs = p.predict_arrays(obs)
        for arr in arrs:
            pts = arr.points_xy_yaw_v_a_kappa
            for i in range(1, len(pts)):
                v_prev, v = pts[i - 1][3], pts[i][3]
                a = pts[i][4]
                # a should roughly match Δv/dt (reintegrated, not stale base a)
                expected = (v - v_prev) / 0.25
                self.assertAlmostEqual(a, expected, places=4)


if __name__ == "__main__":
    unittest.main()
