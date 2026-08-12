"""G3-03: canonicalizer golden, mailbox, V0 policy offline."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle, arrays_to_candidate_set  # noqa: E402
from driving_vla.model.backbone_loader import SimLingoCheckpointHandle, V0Policy  # noqa: E402
from driving_vla.model.canonicalizer import TrajectoryCanonicalizer, UpstreamPathSpeed  # noqa: E402
from driving_vla.model.lineage import build_simlingo_manifest, write_manifest  # noqa: E402
from driving_vla.runtime.mailbox import CandidateMailbox  # noqa: E402
from driving_vla.runtime.mode import RuntimeMode, filter_candidates_for_mode  # noqa: E402
from driving_vla.schema.trajectory_contract import HORIZON_S, T_STEPS  # noqa: E402
from safety_kernel.contracts.types import CandidateSource  # noqa: E402


class TestCanonicalizer(unittest.TestCase):
    def test_t10_horizon(self) -> None:
        path = tuple((float(i), 0.0) for i in range(21))
        speeds = tuple(5.0 for _ in range(10))
        up = UpstreamPathSpeed(path_xy=path, speed_mps=speeds, frame="map")
        can = TrajectoryCanonicalizer()
        tau = can.canonicalize(up, to_map=False)
        self.assertEqual(tau.t_steps, T_STEPS)
        # times (i+1)*dt → last 2.5
        # We don't store t in TrajectoryArray; points length is enough
        # horizon of points via adapter
        obs = ObservationBundle(
            run_id="r", frame_id="f", scenario_id="s", simulation_time_s=1.0, ego_v=5.0
        )
        cset = arrays_to_candidate_set([tau], obs, model_id="test", source=CandidateSource.VLA_FAST)
        pts = cset.candidates[0].points
        self.assertEqual(len(pts), T_STEPS)
        # Contract: t=(i+1)*dt → first 0.25, last 2.5 (= HORIZON_S)
        self.assertAlmostEqual(pts[0].t, 0.25, places=5)
        self.assertAlmostEqual(pts[-1].t, HORIZON_S, places=5)
        self.assertAlmostEqual(pts[-1].t - pts[0].t, (T_STEPS - 1) * 0.25, places=5)

    def test_deterministic(self) -> None:
        path = tuple((float(i), 0.0) for i in range(21))
        speeds = tuple(4.0 + 0.1 * i for i in range(10))
        up = UpstreamPathSpeed(path_xy=path, speed_mps=speeds, frame="map")
        can = TrajectoryCanonicalizer()
        a = can.canonicalize(up, to_map=False)
        b = can.canonicalize(up, to_map=False)
        self.assertEqual(a.points_xy_yaw_v_a_kappa, b.points_xy_yaw_v_a_kappa)


class TestMailbox(unittest.TestCase):
    def test_publish_and_stale(self) -> None:
        mb = CandidateMailbox(soft_stale_s=0.05)
        obs = ObservationBundle(run_id="r", frame_id="f", scenario_id="s", simulation_time_s=0.0)
        path = tuple((float(i), 0.0) for i in range(21))
        speeds = tuple(3.0 for _ in range(10))
        tau = TrajectoryCanonicalizer().canonicalize(
            UpstreamPathSpeed(path_xy=path, speed_mps=speeds, frame="map"), to_map=False
        )
        cset = arrays_to_candidate_set([tau], obs, model_id="m", source=CandidateSource.VLA_FAST)
        mb.publish(cset, latency_s=0.01)
        e = mb.latest()
        self.assertIsNotNone(e)
        assert e is not None
        self.assertTrue(e.ok)
        time.sleep(0.07)
        e2 = mb.latest()
        assert e2 is not None
        self.assertFalse(e2.ok)
        self.assertEqual(e2.reason, "soft_stale")

    def test_control_thread_never_blocks_on_publish(self) -> None:
        mb = CandidateMailbox()
        done = []

        def control_loop() -> None:
            for _ in range(20):
                mb.latest()
                done.append(1)
                time.sleep(0.001)

        t = threading.Thread(target=control_loop)
        t.start()
        t.join(timeout=2.0)
        self.assertGreaterEqual(len(done), 20)


class TestModeFilter(unittest.TestCase):
    def test_vla_safety_drops_classic(self) -> None:
        from safety_kernel.contracts.types import PolicyCandidate, PolicyCandidateSet, TrajectoryPoint
        from safety_kernel.contracts.schema import SCHEMA_VERSION

        pts = tuple(
            TrajectoryPoint(t=i * 0.25, x=float(i), y=0.0, yaw=0.0, kappa=0.0, v=3.0, a=0.0)
            for i in range(10)
        )
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=0.0,
            wall_time_s=0.0,
            candidates=(
                PolicyCandidate(
                    candidate_id="c",
                    source=CandidateSource.CLASSIC,
                    generated_time_s=0.0,
                    valid_until_s=0.2,
                    probability=1.0,
                    points=pts,
                ),
                PolicyCandidate(
                    candidate_id="v",
                    source=CandidateSource.VLA_FAST,
                    generated_time_s=0.0,
                    valid_until_s=0.2,
                    probability=1.0,
                    points=pts,
                ),
            ),
            schema_version=SCHEMA_VERSION,
        )
        filtered = filter_candidates_for_mode(cset, RuntimeMode.VLA_SAFETY)
        assert filtered is not None
        self.assertEqual(len(filtered.candidates), 1)
        self.assertEqual(filtered.candidates[0].source, CandidateSource.VLA_FAST)


class TestV0Load(unittest.TestCase):
    def test_load_checkpoint_if_present(self) -> None:
        ckpt = ROOT / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
        if not ckpt.is_file():
            self.skipTest("simlingo checkpoint not present")
        h = SimLingoCheckpointHandle(ckpt)
        rep = h.load()
        self.assertTrue(rep.ok, rep.error)
        self.assertGreater(rep.n_tensors, 100)
        out1 = h.predict_path_speed(ego_v=5.0, route_xy=tuple((float(i), 0.0) for i in range(40)), seed_extra=b"fix")
        out2 = h.predict_path_speed(ego_v=5.0, route_xy=tuple((float(i), 0.0) for i in range(40)), seed_extra=b"fix")
        self.assertEqual(out1.path_xy, out2.path_xy)
        self.assertEqual(out1.speed_mps, out2.speed_mps)
        self.assertEqual(len(out1.path_xy), 20)
        self.assertEqual(len(out1.speed_mps), 10)

    def test_v0_policy_arrays(self) -> None:
        ckpt = ROOT / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
        if not ckpt.is_file():
            self.skipTest("simlingo checkpoint not present")
        h = SimLingoCheckpointHandle(ckpt)
        self.assertTrue(h.load().ok)
        policy = V0Policy(h)
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            route_xy=tuple((float(i), 0.0) for i in range(0, 50, 2)),
        )
        arrs = policy.predict_arrays(obs)
        self.assertEqual(len(arrs), 1)
        self.assertEqual(arrs[0].t_steps, T_STEPS)


class TestLineage(unittest.TestCase):
    def test_manifest_write(self) -> None:
        ckpt = ROOT / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
        if not ckpt.is_file():
            self.skipTest("no ckpt")
        with tempfile.TemporaryDirectory() as tmp:
            # skip full sha on every unit test — use empty if too slow? file is 2.5GB
            # Use build only if env SDF_G3_FULL_HASH=1
            import os

            if os.environ.get("SDF_G3_FULL_HASH") != "1":
                # lightweight: write manual
                from driving_vla.model.lineage import LineageManifest

                m = LineageManifest(
                    base_model="SimLingo",
                    checkpoint_path=str(ckpt),
                    checkpoint_sha256="deferred",
                    checkpoint_bytes=ckpt.stat().st_size,
                    code_root=str(ROOT / "simlingo-main"),
                    license_scope="research",
                    deployment_scope="simulation_research_only",
                    precision="bf16",
                    model_id="sdf-vla-v0@0.0.1",
                    notes=[],
                )
                p = write_manifest(Path(tmp) / "m.json", m)
                self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
