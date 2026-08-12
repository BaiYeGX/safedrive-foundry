from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from driving_vla.evaluation.actor_future_collector import (
    ActorFutureCollector,
    load_actor_future_trace,
    stable_actor_key,
)


class FakeActor:
    def __init__(self, x: float) -> None:
        self.x = x
        self.bounding_box = SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0))

    def get_transform(self):
        return SimpleNamespace(
            location=SimpleNamespace(x=self.x, y=1.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )

    def get_velocity(self):
        return SimpleNamespace(x=1.0, y=0.0, z=0.0)


class WorldCollectorTest(unittest.TestCase):
    def test_stable_key_ignores_carla_actor_id(self) -> None:
        kwargs = dict(
            scenario_id="s",
            seed_id="a",
            name="npc",
            role="vehicle",
            blueprint="vehicle.tesla.model3",
        )
        self.assertEqual(stable_actor_key(**kwargs), stable_actor_key(**kwargs))

    def test_trace_is_oracle_only_and_resampled(self) -> None:
        actor = SimpleNamespace(
            name="npc",
            role="vehicle",
            blueprint="vehicle.tesla.model3",
            actor=FakeActor(0.0),
        )
        collector = ActorFutureCollector(
            scenario_id="s",
            seed_id="a",
            pair_id="p",
            attempt_id=0,
            branch_index=0,
            anchor_artifact_hash="1" * 64,
            registry_hash="2" * 64,
            model_hash="3" * 64,
            guard_hash="4" * 64,
            executor_hash="5" * 64,
        )
        for i in range(1, 51):
            actor.actor.x = i * 0.05
            collector.record(time_s=i * 0.05, frame=i, actors=[actor])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = collector.finalize(Path(tmp))
            self.assertEqual(manifest["target_coverage"], 1.0)
            trace = Path(tmp) / "oracle" / "actor_future_trace.jsonl"
            rows = load_actor_future_trace(trace)
            self.assertEqual(len(rows), 50)
            payload = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(payload["oracle_only"])
            self.assertFalse(payload["consumed_by_control"])

    def test_no_overwrite(self) -> None:
        collector = ActorFutureCollector(
            scenario_id="s",
            seed_id="a",
            pair_id="p",
            attempt_id=0,
            branch_index=0,
            anchor_artifact_hash="1" * 64,
            registry_hash="2" * 64,
            model_hash="3" * 64,
            guard_hash="4" * 64,
            executor_hash="5" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            collector.finalize(Path(tmp))
            with self.assertRaisesRegex(Exception, "overwrite"):
                collector.finalize(Path(tmp))
