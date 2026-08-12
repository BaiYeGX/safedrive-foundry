from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from driving_vla.world.baselines import predict_actor_future
from driving_vla.world.checkpoint import load_checkpoint, save_checkpoint
from driving_vla.world.contracts import WorldBatch
from driving_vla.world.losses import stack_labels, world_v0_loss
from driving_vla.world.metrics import action_sensitivity, candidate_swap_error
from driving_vla.world.model_v0 import WorldV0, WorldV0Config

from .helpers import make_sample


def small_config(*, no_action: bool = False) -> WorldV0Config:
    return WorldV0Config(
        d_model=32,
        n_heads=4,
        dim_feedforward=64,
        scene_layers=1,
        temporal_layers=1,
        candidate_layers=1,
        decoder_layers=1,
        dropout=0.0,
        no_action=no_action,
    )


class WorldModelV0Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.samples = [make_sample(0, winner=0), make_sample(1, winner=1)]
        self.batch = WorldBatch.from_samples(self.samples)

    def test_production_parameter_budget(self) -> None:
        model = WorldV0()
        self.assertGreaterEqual(model.parameter_count, 4_000_000)
        self.assertLessEqual(model.parameter_count, 8_000_000)

    def test_forward_masks_and_loss(self) -> None:
        model = WorldV0(small_config())
        prediction = model(self.batch)
        prediction.validate_finite()
        self.assertEqual(tuple(prediction.actor_future_mean.shape), (2, 2, 8, 10, 4))
        labels = stack_labels(self.samples, device=torch.device("cpu"))
        losses = world_v0_loss(prediction, labels)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_zero_initialized_forecast_equals_cv(self) -> None:
        model = WorldV0(small_config()).eval()
        sample = self.samples[0]
        prediction = model(WorldBatch.from_samples([sample]))
        cv, _ = predict_actor_future(sample, mode="cv")
        np.testing.assert_allclose(
            prediction.actor_future_mean.detach().numpy()[0],
            cv,
            atol=1e-6,
        )

    def test_candidate_swap_permutation_equivariance(self) -> None:
        model = WorldV0(small_config()).eval()
        error = candidate_swap_error(model, self.batch)
        self.assertLess(error["score_max_abs_error"], 1e-5)
        self.assertLess(error["future_max_abs_error"], 1e-5)

    def test_no_action_has_identical_slot_outputs(self) -> None:
        model = WorldV0(small_config(no_action=True)).eval()
        prediction = model(self.batch)
        np.testing.assert_allclose(
            prediction.utility_score.detach().numpy()[:, 0],
            prediction.utility_score.detach().numpy()[:, 1],
            atol=1e-6,
        )

    def test_singleton_and_invalid_fallback(self) -> None:
        sample = make_sample(candidate1_available=False)
        model = WorldV0(small_config()).eval()
        singleton_batch = WorldBatch.from_samples([sample])
        prediction = model(singleton_batch)
        selected, reason = model.select_candidate(prediction)
        self.assertEqual((selected, reason), (0, "NO_RANKING_NEEDED"))
        no_action = WorldV0(small_config(no_action=True)).eval()(singleton_batch)
        sensitivity = action_sensitivity(prediction, no_action)
        self.assertEqual(sensitivity["dual_candidate_batches"], 0)
        self.assertTrue(
            np.isfinite(sensitivity["conditioned_vs_no_action_score_delta_mean"])
        )

    def test_checkpoint_roundtrip_is_exact(self) -> None:
        model = WorldV0(small_config()).eval()
        expected = model(self.batch).utility_score.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "checkpoint"
            manifest = save_checkpoint(
                directory,
                model=model,
                optimizer=None,
                scaler=None,
                epoch=1,
                global_step=2,
                best_metric=0.5,
                data_hash="a" * 64,
                split_hash="b" * 64,
                code_hash="c" * 64,
                precision="fp32",
            )
            loaded, _ = load_checkpoint(directory)
            loaded.eval()
            actual = loaded(self.batch).utility_score.detach()
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(manifest["parameter_count"], model.parameter_count)
