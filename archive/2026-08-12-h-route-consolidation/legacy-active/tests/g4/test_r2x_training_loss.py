"""Tests for training/runtime alignment of Spatial K2 diversity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.spatial_mode_heads import (  # noqa: E402
    decoded_diversity_floor_loss,
    decoded_lateral_smoothness_loss,
    decoded_lateral_separation,
)


class DecodedDiversityLossTest(unittest.TestCase):
    def test_diversity_floor_is_applied_per_sample_not_batch_mean(self) -> None:
        mode0 = torch.zeros((2, 20), dtype=torch.float32)
        mode1 = torch.zeros((2, 20), dtype=torch.float32)
        mode1[0, 8:] = 3.0  # first sample easily clears the floor
        mode1[1, :] = 0.0   # second sample is fully collapsed
        loss = decoded_diversity_floor_loss(mode0, mode1, target_m=0.65)
        self.assertGreater(float(loss.item()), 0.30)

    def test_decoded_smoothness_penalizes_spike_not_smooth_residual(self) -> None:
        smooth = torch.full((1, 20), 0.8, dtype=torch.float32)
        spike = smooth.clone()
        spike[:, 10] = -2.0
        smooth_loss = decoded_lateral_smoothness_loss(smooth)
        spike_loss = decoded_lateral_smoothness_loss(spike)
        self.assertGreater(float(spike_loss.item()), float(smooth_loss.item()) + 0.05)

    def test_decoded_smoothness_has_gradient(self) -> None:
        raw = torch.zeros((2, 20), dtype=torch.float32, requires_grad=True)
        raw.data[:, 8] = 1.0
        loss = decoded_lateral_smoothness_loss(raw)
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertGreater(float(raw.grad.abs().sum().item()), 0.0)

    def test_near_field_envelope_matches_runtime_shape(self) -> None:
        mode0 = torch.zeros((1, 10), dtype=torch.float32)
        mode1 = torch.zeros((1, 10), dtype=torch.float32)
        mode1[0, 0] = 10.0  # first point is suppressed by the runtime envelope
        self.assertAlmostEqual(
            float(decoded_lateral_separation(mode0, mode1).item()), 0.0, places=6
        )
        mode1[0, 6] = 1.0
        expected = float(torch.tanh(torch.tensor(1.0)).item())
        self.assertAlmostEqual(
            float(decoded_lateral_separation(mode0, mode1).item()),
            expected,
            places=6,
        )

    def test_loss_is_differentiable(self) -> None:
        mode0 = torch.zeros((1, 10), dtype=torch.float32, requires_grad=True)
        mode1 = torch.zeros((1, 10), dtype=torch.float32, requires_grad=True)
        mode1.data[0, 8] = 0.2
        separation = decoded_lateral_separation(mode0, mode1).mean()
        torch.relu(torch.tensor(0.65) - separation).backward()
        self.assertIsNotNone(mode1.grad)
        self.assertGreater(float(mode1.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
