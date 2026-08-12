"""G3-04 / R1: real NeuralV1 K2 (formal) + fingerprint V1 debug (non-acceptance)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.backbone_loader import SimLingoCheckpointHandle  # noqa: E402
from driving_vla.model.neural_policy import NeuralV1Policy  # noqa: E402
from driving_vla.model.v1_policy import (  # noqa: E402
    V1Policy,
    detect_collapse,
    max_position_separation,
    oracle_best_of_k,
)


class _FakeRuntime:
    def __init__(self, speed: float = 5.5) -> None:
        self.load_report = SimpleNamespace(ok=True, error="")
        self.forward_count = 0
        self.speed = speed

    def forward_numpy(self, _image, **kwargs):
        self.forward_count += 1
        route = np.column_stack((np.arange(1.0, 26.0), np.zeros(25)))
        return SimpleNamespace(
            route_xy=route,
            speed_mps=(self.speed,) * 5,
            latency_s=0.012,
            peak_vram_mb=80.0,
        )


class TestNeuralV1RealK2(unittest.TestCase):
    """Formal R1 acceptance path: NeuralV1Policy + fake neural runtime."""

    def test_k2_shape_and_single_forward(self) -> None:
        rt = _FakeRuntime()
        p = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=5.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            front_rgb=np.zeros((48, 48, 3), dtype=np.uint8),
            meta={
                "official_contract": True,
                "target_ego_1": (15.0, 0.0),
                "target_ego_2": (30.0, 0.0),
            },
        )
        arrs = p.predict_arrays(obs)
        self.assertEqual(rt.forward_count, 1)
        self.assertEqual(len(arrs), 2)
        self.assertEqual(arrs[0].t_steps, 10)
        self.assertEqual(arrs[1].t_steps, 10)
        self.assertAlmostEqual(arrs[0].probability + arrs[1].probability, 1.0, places=5)
        self.assertAlmostEqual(arrs[0].probability, 0.5)
        self.assertFalse(detect_collapse(arrs, eps=0.05))
        self.assertGreater(max_position_separation(arrs), 0.50)
        bundle = p.last_bundle
        assert bundle is not None
        self.assertEqual(bundle.top1_index, 0)
        self.assertEqual(bundle.branch_type, "longitudinal_temporal")

    def test_kinematics_reintegrated_from_ego(self) -> None:
        rt = _FakeRuntime(speed=5.0)
        p = NeuralV1Policy(runtime=rt)  # type: ignore[arg-type]
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=4.0,
            front_rgb=np.zeros((48, 48, 3), dtype=np.uint8),
            meta={"official_contract": True, "target_ego_1": (12.0, 0.0), "target_ego_2": (24.0, 0.0)},
        )
        arrs = p.predict_arrays(obs)
        for arr in arrs:
            pts = arr.points_xy_yaw_v_a_kappa
            v_prev = obs.ego_v
            for i in range(len(pts)):
                v = pts[i][3]
                a = pts[i][4]
                expected = (v - v_prev) / 0.25
                self.assertAlmostEqual(a, expected, places=4)
                v_prev = v


class TestFingerprintV1DebugOnly(unittest.TestCase):
    """Legacy fingerprint V1Policy — NOT R1 formal acceptance."""

    @classmethod
    def setUpClass(cls) -> None:
        ckpt = ROOT / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
        if not ckpt.is_file():
            raise unittest.SkipTest("no ckpt")
        cls.handle = SimLingoCheckpointHandle(ckpt)
        if not cls.handle.load().ok:
            raise unittest.SkipTest("ckpt load failed")

    def test_debug_fingerprint_still_produces_k2(self) -> None:
        p = V1Policy(self.handle)
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=6.0,
            route_xy=tuple((float(i), 0.0) for i in range(40)),
            ego_history=(
                (0.0, 0.0, 0.0, 5.0),
                (0.1, 0.0, 0.0, 5.5),
                (0.2, 0.0, 0.0, 6.0),
                (0.3, 0.0, 0.0, 6.0),
            ),
        )
        arrs = p.predict_arrays(obs)
        self.assertEqual(len(arrs), 2)
        # Lateral-bias fork is debug-only spatial diversity
        self.assertGreater(max_position_separation(arrs), 0.15)
        bi, score = oracle_best_of_k(arrs)
        self.assertIn(bi, (0, 1))
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
