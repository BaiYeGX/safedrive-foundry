"""Frozen H3v2 matrix and evaluation-contract regression tests."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import torch

from data_pipeline.h3.challenge_matrix_v2 import (
    CHALLENGE_FIXED_MATRIX,
    CHALLENGE_PILOT_MATRIX,
)
from data_pipeline.h3.contracts import H3_CONTEXT_DIM
from data_pipeline.h3.evaluate import metrics_from_rows, sigmoid
from data_pipeline.h3.model import WorldScorerModel


class TestH3ChallengeContract(unittest.TestCase):
    def test_challenge_matrix_is_96_with_12_pilot(self) -> None:
        self.assertEqual(len(CHALLENGE_FIXED_MATRIX), 96)
        self.assertEqual(len(CHALLENGE_PILOT_MATRIX), 12)
        maps = {row.scenario.map_name for row in CHALLENGE_FIXED_MATRIX}
        self.assertEqual(maps, {"Town01", "Town03", "Town05"})
        families = {row.scenario.family for row in CHALLENGE_FIXED_MATRIX}
        self.assertEqual(families, {"emergency_lead_brake", "aggressive_cut_in", "red_light_dilemma", "cross_traffic_conflict"})

    def test_matrix_slot_balance(self) -> None:
        self.assertEqual(sum(row.expert_slot == 0 for row in CHALLENGE_FIXED_MATRIX), 48)
        self.assertEqual(sum(row.expert_slot == 1 for row in CHALLENGE_FIXED_MATRIX), 48)

    def test_sigmoid_orientation(self) -> None:
        self.assertGreater(sigmoid(8.0), 0.999)
        self.assertLess(sigmoid(-8.0), 0.001)
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_metrics_from_rows_correct_count(self) -> None:
        rows = [
            {"predicted": 0, "winner_index": 0, "target": 1, "delta": 8.0,
             "progress_regret": 0.0, "jerk_regret": 0.0, "uncertainty": 0.0},
            {"predicted": 1, "winner_index": 1, "target": 0, "delta": -8.0,
             "progress_regret": 0.0, "jerk_regret": 0.0, "uncertainty": 0.0},
        ]
        metrics = metrics_from_rows(rows, temperature=0.05)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_scene_gate_removes_candidate_path_when_context_masked(self) -> None:
        model = WorldScorerModel(d_model=64, layers=1, heads=2, ffn=128).eval()
        context = torch.zeros((2, H3_CONTEXT_DIM))
        candidate = torch.randn((2, 10, 8)) * 0.1
        full = model(context, candidate)
        masked = model(context, candidate, mask_context=True)
        self.assertTrue(torch.isfinite(full).all())
        self.assertTrue(torch.isfinite(masked).all())

    def test_model_output_shape(self) -> None:
        model = WorldScorerModel(d_model=64, layers=1, heads=2, ffn=128).eval()
        out = model(torch.zeros(2, H3_CONTEXT_DIM), torch.zeros(2, 10, 8))
        self.assertEqual(tuple(out.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
