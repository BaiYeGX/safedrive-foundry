from __future__ import annotations

import unittest

import numpy as np

from driving_vla.world.contracts import (
    K,
    MAX_ACTORS,
    T,
    WorldPrediction,
)
from driving_vla.world.metrics import paired_ranking_bootstrap

from .helpers import make_sample


def prediction(scores: np.ndarray) -> WorldPrediction:
    batch = scores.shape[0]
    future = np.zeros((batch, K, MAX_ACTORS, T, 4), dtype=np.float32)
    scalar = np.zeros((batch, K), dtype=np.float32)
    return WorldPrediction(
        actor_future_mean=future,
        actor_future_log_scale=future,
        collision_logit=scalar,
        offroad_logit=scalar,
        ttc_value=scalar,
        ttc_censored_logit=scalar,
        utility_score=scores.astype(np.float32),
        candidate_mask=np.ones((batch, K), dtype=bool),
    )


class WorldActionGateTest(unittest.TestCase):
    def test_paired_bootstrap_uses_same_decisive_samples(self) -> None:
        samples = [make_sample(index, winner=index % 2) for index in range(20)]
        conditioned_scores = np.asarray(
            [[2.0, 1.0] if index % 2 == 0 else [1.0, 2.0] for index in range(20)]
        )
        no_action_scores = np.asarray([[2.0, 1.0] for _ in range(20)])
        report = paired_ranking_bootstrap(
            prediction(conditioned_scores),
            prediction(no_action_scores),
            samples,
            n_resamples=500,
        )
        self.assertEqual(report["decisive_count"], 20)
        self.assertEqual(report["conditioned_accuracy"], 1.0)
        self.assertEqual(report["no_action_accuracy"], 0.5)
        self.assertGreaterEqual(report["ci95_low"], 0.0)
