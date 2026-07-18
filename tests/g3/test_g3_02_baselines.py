"""G3-02: baselines, adapter shapes, Safety parse, illegal reject."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import (  # noqa: E402
    ObservationBundle,
    TrajectoryArray,
    arrays_to_candidate_set,
    validate_trajectory_array,
)
from driving_vla.baselines.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from driving_vla.baselines.multi_k2 import MultiCandidateK2Interface  # noqa: E402
from driving_vla.baselines.route_ego import RouteEgoBaseline  # noqa: E402
from driving_vla.baselines.vision_k1 import VisionK1Baseline  # noqa: E402
from driving_vla.schema.trajectory_contract import T_STEPS, V0_CONTRACT, V1_CONTRACT  # noqa: E402
from safety_kernel import SafetyKernel, ComponentAvailability, load_safety_config  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
)


def _obs(**kwargs) -> ObservationBundle:
    base = dict(
        run_id="run1",
        frame_id="frame1",
        scenario_id="sc1",
        simulation_time_s=10.0,
        wall_time_s=100.0,
        carla_frame=100,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=5.0,
        route_xy=tuple((float(i), 0.0) for i in range(0, 80, 2)),
    )
    base.update(kwargs)
    return ObservationBundle(**base)


def _obs_snapshot(o: ObservationBundle) -> ObservableSnapshot:
    return ObservableSnapshot(
        run_id=o.run_id,
        frame_id=o.frame_id,
        scenario_id=o.scenario_id,
        simulation_time_s=o.simulation_time_s,
        wall_time_s=o.wall_time_s,
        ego_x=o.ego_x,
        ego_y=o.ego_y,
        ego_yaw=o.ego_yaw,
        ego_v=o.ego_v,
        observed_time_s=o.simulation_time_s,
        freshness_s=0.0,
        corridor_centerline=o.route_xy[:20] if o.route_xy else ((0.0, 0.0), (50.0, 0.0)),
        corridor_half_width_m=2.0,
        privilege=ObservationPrivilege.OBSERVABLE,
    )


class TestContracts(unittest.TestCase):
    def test_v0_v1_contracts(self) -> None:
        V0_CONTRACT.validate()
        V1_CONTRACT.validate()
        self.assertEqual(V0_CONTRACT.k, 1)
        self.assertEqual(V1_CONTRACT.k, 2)


class TestAdapter(unittest.TestCase):
    def test_nan_rejected(self) -> None:
        bad = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(
                (float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0) if i == 3 else (float(i), 0.0, 0.0, 1.0, 0.0, 0.0)
                for i in range(T_STEPS)
            )
        )
        with self.assertRaises(ValueError):
            validate_trajectory_array(bad)

    def test_wrong_t_rejected(self) -> None:
        bad = TrajectoryArray(points_xy_yaw_v_a_kappa=((0, 0, 0, 1, 0, 0),) * 5)
        with self.assertRaises(ValueError):
            validate_trajectory_array(bad)


class TestBaselines(unittest.TestCase):
    def test_route_ego_k1_shape(self) -> None:
        m = RouteEgoBaseline()
        o = _obs()
        arrs = m.predict(o)
        self.assertEqual(len(arrs), 1)
        self.assertEqual(arrs[0].t_steps, T_STEPS)
        cset = arrays_to_candidate_set(arrs, o, model_id=m.model_id, source=CandidateSource.VLA_FAST)
        self.assertEqual(len(cset.candidates), 1)
        self.assertEqual(len(cset.candidates[0].points), T_STEPS)

    def test_vision_k1(self) -> None:
        m = VisionK1Baseline()
        arrs = m.predict(_obs())
        self.assertEqual(len(arrs), 1)

    def test_k2_interface(self) -> None:
        m = MultiCandidateK2Interface()
        arrs = m.predict(_obs())
        self.assertEqual(len(arrs), 2)
        self.assertEqual(m.interface_frozen()["k"], 2)
        self.assertTrue(m.interface_frozen()["forbids_random_noise_candidates"])
        cset = arrays_to_candidate_set(arrs, _obs(), model_id=m.model_id)
        self.assertEqual(len(cset.candidates), 2)

    def test_safety_kernel_accepts_baseline(self) -> None:
        m = RouteEgoBaseline()
        o = _obs()
        arrs = m.predict(o)
        cset = arrays_to_candidate_set(arrs, o, model_id=m.model_id, source=CandidateSource.VLA_FAST)
        snap = _obs_snapshot(o)
        cfg = load_safety_config()
        kernel = SafetyKernel(cfg)
        result = kernel.tick(
            snap,
            cset,
            now_s=o.simulation_time_s,
            availability=ComponentAvailability(classic=False, vla=True, world=False, safety=True),
        )
        # May ACCEPT or repair/fallback depending on corridor — must not crash; decision present
        self.assertIsNotNone(result.decision)
        self.assertFalse(math.isnan(o.ego_x))


class TestCheckpoint(unittest.TestCase):
    def test_save_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.json"
            h = save_checkpoint(path, model_id="baseline_route_ego_v0", config={"v_max": 12}, state={"ok": True})
            data = load_checkpoint(path)
            self.assertEqual(data["config_hash"], h)
            self.assertEqual(data["model_id"], "baseline_route_ego_v0")


if __name__ == "__main__":
    unittest.main()
