from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h6.cora.devbaseline import (  # noqa: E402
    FEATURE_DIM,
    TARGETS,
    DevWorldMLP,
    _PairArrays,
    _swap_check,
)
from data_pipeline.h6.cora.loader import load_cora_roots  # noqa: E402


class CoraDevBaselineTests(unittest.TestCase):
    def test_quality_profile_enforces_training_and_evaluation_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            sample = {"pair_id": "r1", "split": "train", "candidates": []}
            (path / "release-index.json").write_text(json.dumps({
                "quality_profile": "c2_dev_baseline_v1",
                "status": "DEV_DATA_READY",
                "samples": [sample],
            }), encoding="utf-8")
            rows = load_cora_roots(path, splits=("train",), purpose="training", quality_profile="c2_dev_baseline_v1")
            self.assertEqual(rows[0]["pair_id"], "r1")
            with self.assertRaises(ValueError):
                load_cora_roots(path, splits=("validation",), purpose="training", quality_profile="c2_dev_baseline_v1")

    def test_quality_profile_rejects_old_failed_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "release-index.json").write_text(json.dumps({
                "quality_profile": "c2_dev_baseline_v1",
                "status": "GATE_FAILED",
                "samples": [{"pair_id": "r1", "split": "train"}],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cora_roots(path, splits=("train",), purpose="training", quality_profile="c2_dev_baseline_v1")

    def test_shared_model_follows_candidate_swap_without_source_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            model = DevWorldMLP()
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            arrays = _PairArrays(
                pair_ids=["r1", "r2"],
                groups=[("Town01", "free_flow", "ClearNoon")] * 2,
                x=np.random.default_rng(3).normal(size=(2, 2, FEATURE_DIM)).astype(np.float32),
                y=np.zeros((2, 2, len(TARGETS)), dtype=np.float32),
                mask=np.ones((2, 2, len(TARGETS)), dtype=bool),
                source=[("expert", "vla"), ("expert", "vla")],
            )
            result = _swap_check({"checkpoint": str(checkpoint)}, arrays, {"mean": [0.0] * 4, "std": [1.0] * 4})
            self.assertTrue(result["passed"])
            self.assertFalse(result["source_metadata_in_input"])


if __name__ == "__main__":
    unittest.main()
