"""Tests for H4 locked-evaluation metrics and loader contracts.

These tests use synthetic data only and never open real test labels.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from data_pipeline.h4.contracts import H4_CONFIG_SHA256, H4_SCHEMA_VERSION
from data_pipeline.h4.locked_dataset import load_locked_test_examples
from data_pipeline.h4.metrics import (
    _auc,
    bootstrap_accuracy_delta,
    defer_metrics,
    metrics_from_rows,
    source_wins,
)


class H4AucTest(unittest.TestCase):
    def test_auc_perfect_positive(self) -> None:
        self.assertAlmostEqual(_auc([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]), 1.0)

    def test_auc_perfect_negative(self) -> None:
        self.assertAlmostEqual(_auc([4.0, 3.0, 2.0, 1.0], [0, 0, 1, 1]), 0.0)

    def test_auc_single_class_is_none(self) -> None:
        self.assertIsNone(_auc([1.0, 2.0], [1, 1]))


class H4MetricsTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        return [
            {
                "pair_id": "p1", "delta": 1.0, "predicted": 0, "target": 1,
                "winner_index": 0, "uncertainty": 0.1,
                "progress_regret": 0.0, "jerk_regret": 0.0,
                "winner_source": "expert", "probability_first_wins": 0.9,
            },
            {
                "pair_id": "p2", "delta": -1.0, "predicted": 1, "target": 0,
                "winner_index": 1, "uncertainty": 0.2,
                "progress_regret": 0.0, "jerk_regret": 0.0,
                "winner_source": "vla", "probability_first_wins": 0.1,
            },
        ]

    def test_metrics_from_rows(self) -> None:
        m = metrics_from_rows(self._rows(), temperature=0.05)
        self.assertEqual(m["n_decisive"], 2)
        self.assertEqual(m["correct"], 2)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["auroc"], 1.0)
        self.assertLessEqual(m["ece"], 0.5)

    def test_defer_metrics_ranked_all(self) -> None:
        d = defer_metrics(self._rows(), temperature=0.05)
        self.assertEqual(d["ranked_n"], 2)
        self.assertEqual(d["coverage"], 1.0)
        self.assertEqual(d["accuracy_ranked"], 1.0)

    def test_defer_metrics_high_uncertainty_defers(self) -> None:
        rows = self._rows()
        rows[0]["uncertainty"] = 0.9
        d = defer_metrics(rows, temperature=0.05)
        self.assertEqual(d["ranked_n"], 1)
        self.assertAlmostEqual(d["coverage"], 0.5)

    def test_defer_metrics_low_margin_defers(self) -> None:
        rows = self._rows()
        rows[0]["delta"] = 0.01
        d = defer_metrics(rows, temperature=0.05)
        self.assertEqual(d["ranked_n"], 1)
        self.assertAlmostEqual(d["coverage"], 0.5)

    def test_source_wins(self) -> None:
        w = source_wins(self._rows())
        self.assertEqual(w["expert_wins"], 1)
        self.assertEqual(w["vla_wins"], 1)
        self.assertEqual(w["unknown_source_wins"], 0)


class H4IsolationAuditTest(unittest.TestCase):
    def _candidate(self, key: str = "c"):
        from data_pipeline.h3.dataset import CandidateExample
        return CandidateExample(
            candidate_key=key,
            context=tuple([0.0] * 499),
            candidate=tuple(tuple([0.0] * 8 for _ in range(10))),
            progress_m=0.0,
            jerk_rms_mps3=0.0,
            risk=False,
            h1_soft_score=0.0,
        )

    def _pair(self, pair_id: str, split: str):
        from data_pipeline.h3.dataset import PairExample
        c = self._candidate(pair_id)
        return PairExample(pair_id, "Town01", "family", 0, "ClearNoon", split, (c, c), 0, False)

    def _locked(self, pair_id: str):
        from data_pipeline.h4.locked_dataset import LockedPairExample
        return LockedPairExample(pair=self._pair(pair_id, "test"), sources=("expert", "vla"))

    def test_cross_split_duplicate_fails(self):
        from data_pipeline.h4.locked_dataset import audit_test_isolation
        split = {"rows": [
            {"pair_id": "dev1", "lineage": "Town01|family|0", "split": "dev_fold_1", "valid_pair": True},
            {"pair_id": "test1", "lineage": "Town01|family|1", "split": "test", "valid_pair": True},
        ]}
        dev = [self._pair("dev1", "dev_fold_1")]
        test = [self._locked("test1")]
        # The feature payloads are identical; this must be reported as a leak.
        audit = audit_test_isolation([], split, dev, test)
        self.assertFalse(audit["passed"])
        self.assertTrue(any("cross_split_duplicate_payload" in f for f in audit["failures"]))

    def test_duplicate_within_test_passes(self):
        from data_pipeline.h4.locked_dataset import audit_test_isolation
        split = {"rows": [
            {"pair_id": "test1", "lineage": "Town01|family|1", "split": "test", "valid_pair": True},
            {"pair_id": "test2", "lineage": "Town01|family|2", "split": "test", "valid_pair": True},
        ]}
        test = [self._locked("test1"), self._locked("test2")]
        audit = audit_test_isolation([], split, [], test)
        self.assertTrue(audit["passed"])


class FakeRiskModel:
    """Minimal callable with the same batched output shape as WorldScorerModel."""
    def __init__(self, risk_logit: float = 0.0, utility_first: float = 1.0, utility_second: float = -1.0):
        self.risk_logit = risk_logit
        self.utility_first = utility_first
        self.utility_second = utility_second

    def __call__(self, context, candidate):
        import torch
        batch = context.shape[0]
        out = torch.zeros(batch, 6)
        out[0, 0] = self.utility_first
        if batch > 1:
            out[1, 0] = self.utility_second
        out[:, 5] = self.risk_logit
        return out


class H4RuntimeRiskGateTest(unittest.TestCase):
    def _payloads(self):
        context = tuple([0.5] * 499)
        candidate = tuple(tuple([0.0] * 8 for _ in range(10)))
        return ("a", context, candidate), ("b", context, candidate)

    def test_high_predicted_risk_defers(self):
        from data_pipeline.h4.runtime import NormalizedWorldScorer
        scorer = NormalizedWorldScorer([FakeRiskModel(risk_logit=5.0)], [(0.0, 1.0)], device="cpu")
        first, second = self._payloads()
        result = scorer.score_pair(first, second)
        self.assertEqual(result.disposition, "defer_low_confidence")
        self.assertEqual(result.defer_reason, "predicted_hard_risk_over_threshold")

    def test_zero_context_defers(self):
        from data_pipeline.h4.runtime import NormalizedWorldScorer
        scorer = NormalizedWorldScorer(
            [FakeRiskModel(risk_logit=-5.0, utility_first=2.0, utility_second=-2.0)],
            [(0.0, 1.0)], device="cpu",
        )
        context = tuple([0.0] * 499)
        candidate = tuple(tuple([0.0] * 8 for _ in range(10)))
        result = scorer.score_pair(("a", context, candidate), ("b", context, candidate))
        self.assertEqual(result.disposition, "defer_low_confidence")
        self.assertEqual(result.defer_reason, "context_masked_or_empty")

    def test_probability_temperature_floor_prevents_saturation(self):
        from data_pipeline.h4.runtime import NormalizedWorldScorer
        scorer = NormalizedWorldScorer(
            [FakeRiskModel(risk_logit=-5.0, utility_first=0.2, utility_second=-0.2)],
            [(0.0, 1.0)],
            device="cpu",
            temperature=0.05,
            probability_temperature_floor=0.5,
        )
        result = scorer.score_pair(*self._payloads())
        self.assertGreater(result.probability_first_wins, 0.5)
        self.assertLess(result.probability_first_wins, 0.8)

    def test_low_predicted_risk_ranks(self):
        from data_pipeline.h4.runtime import NormalizedWorldScorer
        scorer = NormalizedWorldScorer([FakeRiskModel(risk_logit=-5.0, utility_first=2.0, utility_second=-2.0)], [(0.0, 1.0)], device="cpu")
        first, second = self._payloads()
        result = scorer.score_pair(first, second)
        self.assertEqual(result.disposition, "ranked")


class H4LoaderContractTest(unittest.TestCase):
    def test_locked_loader_rejects_non_test_split(self) -> None:
        with self.assertRaises(Exception):
            load_locked_test_examples([], {"rows": []}, split="dev")


class H4ConfigTest(unittest.TestCase):
    def test_schema_version(self) -> None:
        self.assertEqual(H4_SCHEMA_VERSION, "safedrive.h4.locked_eval.v1")
        self.assertEqual(len(H4_CONFIG_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
