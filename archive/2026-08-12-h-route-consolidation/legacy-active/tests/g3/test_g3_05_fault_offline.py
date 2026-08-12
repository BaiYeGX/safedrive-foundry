"""G3-05 offline fault inject: timeout/stale/nan degrade without crashing control path."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray, arrays_to_candidate_set  # noqa: E402
from driving_vla.runtime.mailbox import CandidateMailbox  # noqa: E402
from driving_vla.runtime.mode import RuntimeMode, availability_for_mode, filter_candidates_for_mode  # noqa: E402
from driving_vla.schema.trajectory_contract import T_STEPS  # noqa: E402
from safety_kernel import SafetyKernel, load_safety_config  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
)


def _good_arr() -> TrajectoryArray:
    pts = tuple((float(i) * 0.5, 0.0, 0.0, 4.0, 0.0, 0.0) for i in range(T_STEPS))
    return TrajectoryArray(points_xy_yaw_v_a_kappa=pts, probability=1.0, candidate_id="v0")


class TestG305Faults(unittest.TestCase):
    def test_nan_rejected_by_adapter(self) -> None:
        bad = list(_good_arr().points_xy_yaw_v_a_kappa)
        bad[2] = (float("nan"), 0.0, 0.0, 4.0, 0.0, 0.0)
        arr = TrajectoryArray(points_xy_yaw_v_a_kappa=tuple(bad), candidate_id="bad")
        obs = ObservationBundle(run_id="r", frame_id="f", scenario_id="s", simulation_time_s=1.0)
        with self.assertRaises(ValueError):
            arrays_to_candidate_set([arr], obs, model_id="m", source=CandidateSource.VLA_FAST)

    def test_timeout_marks_mailbox(self) -> None:
        mb = CandidateMailbox()
        obs = ObservationBundle(run_id="r", frame_id="f", scenario_id="s", simulation_time_s=1.0)
        cset = arrays_to_candidate_set([_good_arr()], obs, model_id="m", source=CandidateSource.VLA_FAST)
        mb.publish(cset, latency_s=0.01)
        mb.mark_degraded("timeout")
        self.assertFalse(mb.vla_ok())
        self.assertEqual(mb.last_error(), "timeout")

    def test_vla_safety_kernel_with_unavailable(self) -> None:
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=5.0,
            ego_v=3.0,
            route_xy=tuple((float(i), 0.0) for i in range(30)),
        )
        cset = arrays_to_candidate_set([_good_arr()], obs, model_id="m", source=CandidateSource.VLA_FAST)
        cset = filter_candidates_for_mode(cset, RuntimeMode.VLA_SAFETY)
        snap = ObservableSnapshot(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=5.0,
            wall_time_s=0.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=3.0,
            observed_time_s=5.0,
            corridor_centerline=tuple((float(i), 0.0) for i in range(30)),
            corridor_half_width_m=2.5,
            privilege=ObservationPrivilege.OBSERVABLE,
        )
        kernel = SafetyKernel(load_safety_config())
        # VLA unavailable → should not crash
        res = kernel.tick(
            snap,
            cset,
            now_s=5.0,
            availability=availability_for_mode(RuntimeMode.VLA_SAFETY, vla_ok=False),
        )
        self.assertIsNotNone(res.decision)


if __name__ == "__main__":
    unittest.main()
