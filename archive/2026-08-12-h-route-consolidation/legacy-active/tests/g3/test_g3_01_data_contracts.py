"""G3-01: identity, hash stability, split, leakage rejection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.vla.datacard import write_datacard  # noqa: E402
from data_pipeline.vla.leakage import LeakageAuditor, LeakageError  # noqa: E402
from data_pipeline.vla.schema import (  # noqa: E402
    FrameIdentity,
    LayerBundle,
    SampleRecord,
    content_hash,
    sample_from_dict,
    sample_to_dict,
)
from data_pipeline.vla.split import SplitAssigner, SplitName, SplitSpec  # noqa: E402
from data_pipeline.vla.store import ShardStore  # noqa: E402


def _sample(
    *,
    run_id: str = "r1",
    frame_id: str = "f1",
    scenario_id: str = "s1",
    town: str = "Town10HD",
    route_id: str = "route_a",
    family: str = "urban_follow",
    weather: str = "clear",
    ego_v: float = 5.0,
    extra_label: dict | None = None,
) -> SampleRecord:
    layers = LayerBundle(
        policy_input={
            "front_rgb_uri": f"mem://{run_id}/{frame_id}.png",
            "ego_state": {"v": ego_v, "x": 0.0, "y": 0.0, "yaw": 0.0},
            "route": {"route_id": route_id},
        },
        privileged_label=dict(extra_label or {"expert_trajectory": [[0, 0], [1, 0]]}),
        evaluation_only={},
        regression_frozen={},
    )
    rec = SampleRecord(
        identity=FrameIdentity(
            run_id=run_id,
            frame_id=frame_id,
            scenario_id=scenario_id,
            town=town,
            route_id=route_id,
            scenario_family=family,
            weather=weather,
        ),
        layers=layers,
    )
    rec.recompute_parameter_hash()
    rec.content_hash = content_hash(rec)
    return rec


class TestG301Schema(unittest.TestCase):
    def test_hash_stable_fixed_seed_payload(self) -> None:
        a = _sample()
        b = _sample()
        self.assertEqual(content_hash(a), content_hash(b))
        d = sample_to_dict(a)
        c = sample_from_dict(d)
        self.assertEqual(content_hash(c), a.content_hash)

    def test_hash_changes_with_content(self) -> None:
        a = _sample(ego_v=5.0)
        b = _sample(ego_v=6.0)
        self.assertNotEqual(content_hash(a), content_hash(b))


class TestG301Split(unittest.TestCase):
    def test_regression_family_pinned(self) -> None:
        assigner = SplitAssigner(SplitSpec())
        s = _sample(family="regression_core")
        self.assertEqual(assigner.assign_sample(s), SplitName.REGRESSION)

    def test_deterministic_assignment(self) -> None:
        assigner = SplitAssigner(SplitSpec(seed=11))
        s = _sample(scenario_id="fixed_scen")
        a = assigner.assign_sample(s)
        b = assigner.assign_sample(s)
        self.assertEqual(a, b)

    def test_failure_cluster_affects_bucket(self) -> None:
        assigner = SplitAssigner(SplitSpec(seed=11))
        a = _sample(scenario_id="sc_fc")
        b = _sample(scenario_id="sc_fc")
        # mutate failure_cluster only
        from data_pipeline.vla.schema import FrameIdentity

        b.identity = FrameIdentity(
            run_id=b.identity.run_id,
            frame_id=b.identity.frame_id,
            scenario_id=b.identity.scenario_id,
            town=b.identity.town,
            route_id=b.identity.route_id,
            scenario_family=b.identity.scenario_family,
            weather=b.identity.weather,
            failure_cluster="cluster_B",
        )
        a.identity = FrameIdentity(
            run_id=a.identity.run_id,
            frame_id=a.identity.frame_id,
            scenario_id=a.identity.scenario_id,
            town=a.identity.town,
            route_id=a.identity.route_id,
            scenario_family=a.identity.scenario_family,
            weather=a.identity.weather,
            failure_cluster="cluster_A",
        )
        # Not always different splits, but bucket material must include cluster:
        # force many seeds and ensure at least the hash material differs.
        ha = assigner._bucket(a.identity)
        hb = assigner._bucket(b.identity)
        self.assertNotEqual(ha, hb)


class TestG301Leakage(unittest.TestCase):
    def test_cross_split_content_rejected(self) -> None:
        assigner = SplitAssigner()
        auditor = LeakageAuditor(assigner)
        s1 = _sample(run_id="r1", frame_id="f1")
        # Force identical payload fields (URI without identity coupling).
        s1.layers.policy_input["front_rgb_uri"] = "mem://shared.png"
        s1.content_hash = content_hash(s1)
        auditor.admit(s1)
        s2 = _sample(run_id="r2", frame_id="f2")
        s2.layers.policy_input["front_rgb_uri"] = "mem://shared.png"
        s2.content_hash = content_hash(s2)
        natural = assigner.assign_sample(s2)
        other = SplitName.VAL if natural != SplitName.VAL else SplitName.TEST
        with self.assertRaises(LeakageError) as ctx:
            auditor.check_admit(s2, target_split=other)
        msg = str(ctx.exception)
        self.assertTrue(
            "cross_split_content" in msg or "near_duplicate" in msg,
            msg,
        )

    def test_cross_split_identity_rejected(self) -> None:
        assigner = SplitAssigner()
        auditor = LeakageAuditor(assigner)
        s1 = _sample(run_id="r1", frame_id="f1")
        split = auditor.admit(s1)
        s_same_id = _sample(run_id="r1", frame_id="f1", ego_v=9.0)
        other = SplitName.TEST if split != SplitName.TEST else SplitName.VAL
        with self.assertRaises(LeakageError) as ctx:
            auditor.check_admit(s_same_id, target_split=other)
        self.assertIn("cross_split_identity", str(ctx.exception))

    def test_near_duplicate_rejected(self) -> None:
        assigner = SplitAssigner()
        auditor = LeakageAuditor(assigner)
        s1 = _sample(run_id="r1", frame_id="f1")
        auditor.admit(s1)
        s_dup = _sample(run_id="r1", frame_id="f1")
        with self.assertRaises(LeakageError) as ctx:
            auditor.check_admit(s_dup)
        self.assertIn("near_duplicate", str(ctx.exception))

    def test_near_duplicate_prefix_not_only_exact_payload(self) -> None:
        """Different identity / tiny label change but same image+ego → near-dup via fingerprint."""
        assigner = SplitAssigner()
        auditor = LeakageAuditor(assigner, near_dup_hash_prefix=16)
        s1 = _sample(run_id="r1", frame_id="f1", ego_v=5.0)
        s1.layers.policy_input["front_rgb_uri"] = "mem://same.png"
        auditor.admit(s1)
        s2 = _sample(run_id="r2", frame_id="f2", ego_v=5.0)
        s2.layers.policy_input["front_rgb_uri"] = "mem://same.png"
        s2.layers.privileged_label = {"expert_trajectory": [[0, 0], [2, 0]]}  # payload differs
        with self.assertRaises(LeakageError) as ctx:
            auditor.check_admit(s2)
        self.assertIn("near_duplicate", str(ctx.exception))
        self.assertIn("fingerprint prefix", str(ctx.exception))

    def test_regression_frozen_not_in_train(self) -> None:
        assigner = SplitAssigner()
        auditor = LeakageAuditor(assigner)
        s = _sample(family="urban_follow")
        s.layers.regression_frozen = {"frozen_metric": 1.0}
        # May assign to train depending on hash — force train target
        with self.assertRaises(LeakageError) as ctx:
            auditor.check_admit(s, target_split=SplitName.TRAIN)
        self.assertIn("regression_write", str(ctx.exception))

    def test_regression_escape_rejected(self) -> None:
        assigner = SplitAssigner()
        auditor = LeakageAuditor(assigner)
        s = _sample(family="holdout_gate")
        with self.assertRaises(LeakageError) as ctx:
            auditor.check_admit(s, target_split=SplitName.TRAIN)
        self.assertIn("regression_escape", str(ctx.exception))


class TestG301Store(unittest.TestCase):
    def test_resume_no_duplicate_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ShardStore(root)
            s = _sample(run_id="r9", frame_id="f9", scenario_id="sc9")
            store.write_sample(s)
            store2 = ShardStore(root)
            with self.assertRaises(LeakageError):
                store2.write_sample(_sample(run_id="r9", frame_id="f9", scenario_id="sc9"))
            # New sample ok (different ego + uri neighborhood)
            store2.write_sample(
                _sample(run_id="r10", frame_id="f10", scenario_id="sc10", ego_v=8.5, route_id="route_b")
            )
            self.assertTrue((root / "manifest.json").is_file())
            # Authoritative format is parquet
            import json

            man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man.get("format"), "parquet")
            # At least one parquet file written
            self.assertTrue(any(root.glob("*.parquet")))


class TestG301Datacard(unittest.TestCase):
    def test_write_datacard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_datacard(Path(tmp) / "datacard.json")
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("simulation_research_only", text)


if __name__ == "__main__":
    unittest.main()
