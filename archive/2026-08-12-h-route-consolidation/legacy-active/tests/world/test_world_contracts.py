from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from driving_vla.world.contracts import WorldBatch, WorldContractError
from driving_vla.world.observable_builder import assert_observable_only, world_to_ego

from .helpers import make_sample


class WorldContractTest(unittest.TestCase):
    def test_sample_and_runtime_batch_roundtrip(self) -> None:
        sample = make_sample()
        sample.validate()
        batch = WorldBatch.from_samples([sample])
        self.assertEqual(np.asarray(batch.candidates).shape, (1, 2, 10, 8))
        self.assertFalse(hasattr(batch, "actor_future"))
        self.assertFalse(hasattr(batch, "outcomes"))

    def test_unavailable_candidate_requires_reason(self) -> None:
        sample = make_sample(candidate1_available=False)
        sample.unavailable_reasons = (None, None)
        with self.assertRaisesRegex(WorldContractError, "without exact reason"):
            sample.validate()

    def test_rank_requires_two_available_candidates(self) -> None:
        sample = make_sample(candidate1_available=False)
        sample.rank_mask = True
        with self.assertRaisesRegex(WorldContractError, "rank supervision"):
            sample.validate()

    def test_observable_rejects_nested_oracle_key(self) -> None:
        with self.assertRaisesRegex(WorldContractError, "oracle"):
            assert_observable_only({"actors": [{"oracle_winner": 1}]})

    def test_coordinate_transform(self) -> None:
        x, y = world_to_ego(10, 12, ego_x=10, ego_y=10, ego_yaw_rad=np.pi / 2)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_runtime_world_namespace_does_not_import_oracle_collectors(self) -> None:
        world_root = (
            Path(__file__).resolve().parents[2]
            / "safedrive_foundry"
            / "driving_vla"
            / "world"
        )
        runtime_modules = (
            "__init__.py",
            "contracts.py",
            "model_v0.py",
            "observable_builder.py",
        )
        source = "\n".join(
            (world_root / name).read_text(encoding="utf-8")
            for name in runtime_modules
        )
        self.assertNotIn("driving_vla.evaluation.oracle", source)
        self.assertNotIn("actor_future_collector", source)
