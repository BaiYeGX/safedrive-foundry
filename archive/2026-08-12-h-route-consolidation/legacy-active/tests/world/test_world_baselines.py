from __future__ import annotations

import unittest

import numpy as np

from driving_vla.world.baselines import observable_rule_scores, predict_actor_future

from .helpers import make_sample


class WorldBaselineTest(unittest.TestCase):
    def test_persistence_cv_ctrv_shapes(self) -> None:
        sample = make_sample()
        persistence, mask = predict_actor_future(sample, mode="persistence")
        cv, _ = predict_actor_future(sample, mode="cv")
        ctrv, _ = predict_actor_future(sample, mode="ctrv")
        self.assertEqual(persistence.shape, (2, 8, 10, 4))
        self.assertEqual(mask.shape, (2, 8, 10))
        self.assertFalse(np.allclose(persistence, cv))
        self.assertTrue(np.isfinite(ctrv).all())

    def test_rule_masks_unavailable_candidate(self) -> None:
        sample = make_sample(candidate1_available=False)
        cv, _ = predict_actor_future(sample, mode="cv")
        score = observable_rule_scores(sample, cv)
        self.assertTrue(np.isfinite(score[0]))
        self.assertEqual(score[1], -np.inf)
