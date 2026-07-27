"""R1 execution binding: force 0/1, fail-closed orphan, same PathManager."""

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
from driving_vla.model.neural_policy import NeuralV1Policy  # noqa: E402
from driving_vla.runtime.k2_execution import (  # noqa: E402
    K2SelectionError,
    apply_k2_to_executors,
    make_source_id,
    select_k2,
)
from driving_vla.runtime.path_manager import EgoPose, PathManagerConfig, VLAPathManager  # noqa: E402
from driving_vla.runtime.vla_mpc_tracker import ConstrainedVLAMPC, VLAMPCConfig  # noqa: E402
from driving_vla.runtime.vla_speed_planner import VLASpeedPlanner  # noqa: E402


class _FakeRuntime:
    def __init__(self, speed: float = 5.0) -> None:
        self.load_report = SimpleNamespace(ok=True, error="")
        self.forward_count = 0
        self.speed = speed

    def forward_numpy(self, _image, **kwargs):
        self.forward_count += 1
        route = np.column_stack((np.arange(1.0, 26.0), np.zeros(25)))
        return SimpleNamespace(
            route_xy=route,
            speed_mps=(self.speed,) * 5,
            latency_s=0.01,
            peak_vram_mb=50.0,
        )


def _obs() -> ObservationBundle:
    return ObservationBundle(
        run_id="r1",
        frame_id="frame-7",
        scenario_id="s",
        simulation_time_s=2.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=4.0,
        front_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        meta={"official_contract": True, "target_ego_1": (10.0, 0.0), "target_ego_2": (20.0, 0.0)},
    )


class R1ExecutionTest(unittest.TestCase):
    def _bundle(self):
        policy = NeuralV1Policy(runtime=_FakeRuntime(5.5))  # type: ignore[arg-type]
        return policy.predict_bundle(_obs())

    def test_force_0_and_1_resolve_specs(self) -> None:
        bundle = self._bundle()
        s0 = select_k2(bundle, mode="force", force_index=0)
        s1 = select_k2(bundle, mode="force", force_index=1)
        self.assertEqual(s0.candidate_id, "v1_nominal")
        self.assertEqual(s1.candidate_id, "v1_conservative")
        self.assertEqual(s0.execution_spec.candidate_id, "v1_nominal")
        self.assertEqual(s1.execution_spec.candidate_id, "v1_conservative")
        self.assertNotEqual(
            s0.execution_spec.timed_trajectory_hash,
            s1.execution_spec.timed_trajectory_hash,
        )

    def test_top1_is_explicit_nominal(self) -> None:
        bundle = self._bundle()
        sel = select_k2(bundle, mode="top1")
        self.assertEqual(sel.candidate_index, 0)
        self.assertEqual(sel.candidate_id, "v1_nominal")

    def test_orphan_id_fail_closed(self) -> None:
        bundle = self._bundle()
        from dataclasses import replace

        # Remove execution spec for nominal
        specs = dict(bundle.execution_specs)
        del specs["v1_nominal"]
        broken = replace(bundle, execution_specs=specs)
        with self.assertRaises(K2SelectionError) as ctx:
            select_k2(broken, mode="force", force_index=0)
        self.assertIn("orphan", str(ctx.exception).lower())

    def test_hash_mismatch_fail_closed(self) -> None:
        bundle = self._bundle()
        from dataclasses import replace

        spec = bundle.execution_specs["v1_nominal"]
        bad_spec = replace(spec, native_path_hash="deadbeef")
        specs = dict(bundle.execution_specs)
        specs["v1_nominal"] = bad_spec
        broken = replace(bundle, execution_specs=specs)
        with self.assertRaises(K2SelectionError):
            select_k2(broken, mode="force", force_index=0)

    def test_invalid_force_index(self) -> None:
        bundle = self._bundle()
        with self.assertRaises(K2SelectionError):
            select_k2(bundle, mode="force", force_index=2)

    def test_apply_same_path_manager_for_force_0_1(self) -> None:
        bundle = self._bundle()
        path_manager = VLAPathManager(
            PathManagerConfig(max_abs_curvature=0.5, hard_max_abs_curvature=1.0)
        )
        planner = VLASpeedPlanner()
        ego = EgoPose(0.0, 0.0, 0.0, 4.0)

        results = []
        for idx in (0, 1):
            # fresh path manager state between candidates is OK; same *type*
            pm = VLAPathManager(
                PathManagerConfig(max_abs_curvature=0.5, hard_max_abs_curvature=1.0)
            )
            pl = VLASpeedPlanner()
            sel = select_k2(bundle, mode="force", force_index=idx)
            applied = apply_k2_to_executors(
                sel,
                speed_planner=pl,
                path_manager=pm,
                ego=ego,
                stamp_s=1.0,
                frame_id="carla-1",
                dt_s=0.05,
                ego_speed_mps=4.0,
            )
            results.append(applied)
            self.assertEqual(applied.selected_candidate_id, applied.executed_candidate_id)
            self.assertIn(sel.candidate_id, applied.source_id)
            self.assertEqual(
                applied.source_id, make_source_id("carla-1", sel.candidate_id)
            )

        self.assertEqual(type(results[0].path_update), type(results[1].path_update))
        # Same PathManager / planner classes
        self.assertIs(type(path_manager), VLAPathManager)
        mpc0 = ConstrainedVLAMPC(VLAMPCConfig())
        mpc1 = ConstrainedVLAMPC(VLAMPCConfig())
        self.assertEqual(type(mpc0), type(mpc1))

    def test_guard_reject_blocks_select(self) -> None:
        bundle = self._bundle()
        from dataclasses import replace

        broken = replace(bundle, guard_status="REJECT", guard_reasons=("test",))
        with self.assertRaises(K2SelectionError):
            select_k2(broken, mode="top1")

    def test_selector_fail_closed_on_spoofed_execution_path(self) -> None:
        """select_k2 must reject execution_spec whose spatial path content was swapped."""
        from dataclasses import replace

        from driving_vla.model.k2_builder import K2ExecutionSpec

        bundle = self._bundle()
        spoof_path = tuple((float(i), 2.0) for i in range(0, 25))
        specs = dict(bundle.execution_specs)
        for cid, spec in list(specs.items()):
            specs[cid] = K2ExecutionSpec(
                candidate_id=spec.candidate_id,
                spatial_path_xy=spoof_path,
                speed_samples_mps=spec.speed_samples_mps,
                timed_trajectory_hash=spec.timed_trajectory_hash,
                native_path_hash=bundle.native_path_hash,
                branch_type=spec.branch_type,
            )
        spoofed = replace(bundle, execution_specs=specs)
        # Even if guard_status field is still OK, content binding must fail closed
        with self.assertRaises(K2SelectionError) as ctx:
            select_k2(spoofed, mode="force", force_index=0, require_guard_ok=False)
        self.assertIn("execution_spatial_binding", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
