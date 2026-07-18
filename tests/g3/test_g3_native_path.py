from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.neural_policy import NeuralV0Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.load_report = SimpleNamespace(ok=True, error="")
        self.kwargs = {}

    def forward_numpy(self, _image, **kwargs):
        self.kwargs = kwargs
        route = np.column_stack((np.arange(1.0, 21.0), np.zeros(20)))
        return SimpleNamespace(
            route_xy=route,
            speed_mps=np.full(10, 2.0),
            latency_s=0.01,
            peak_vram_mb=123.0,
        )


class NativePathContractTest(unittest.TestCase):
    def test_upstream_camera_contract_is_explicit(self) -> None:
        self.assertEqual(SIMLINGO_CAMERA_XYZ, (-1.5, 0.0, 2.0))
        self.assertEqual(SIMLINGO_CAMERA_NATIVE_SIZE, (1024, 512))
        self.assertEqual(SIMLINGO_CAMERA_FOV_DEG, 110.0)

    def test_native_path_preserves_twenty_spatial_points(self) -> None:
        runtime = _FakeRuntime()
        policy = NeuralV0Policy(runtime=runtime)
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_x=10.0,
            ego_y=20.0,
            ego_yaw=math.pi / 2.0,
            ego_v=2.0,
            front_rgb=np.zeros((384, 768, 3), dtype=np.uint8),
            meta={"target_ego_1": (15.0, 1.0), "target_ego_2": (30.0, 2.0)},
        )
        native = policy.predict_native(obs)
        self.assertEqual(len(native.path_map_xy), 20)
        self.assertAlmostEqual(native.path_map_xy[0][0], 10.0, places=6)
        self.assertAlmostEqual(native.path_map_xy[0][1], 21.0, places=6)
        self.assertEqual(native.target_ego_1, (15.0, 1.0))
        self.assertEqual(native.target_ego_2, (30.0, 2.0))
        self.assertEqual(runtime.kwargs["target_point2_xy"], (30.0, 2.0))

        canonical = policy.predict_arrays(obs)[0]
        self.assertEqual(canonical.t_steps, 10)


if __name__ == "__main__":
    unittest.main()
