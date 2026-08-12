"""Offline tests for leakage-safe Spatial K2 feature probes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.feature_predictability import (  # noqa: E402
    build_feature_predictability_report,
)


def _row(split: str, side: str, available: bool, direction: float, value: float):
    raw_d = [0.0, direction, direction * 2.0]
    return {
        "split_id": split,
        "conflict_side": side,
        "alternative_available": available,
        "driving_feature": [value, value * value, float(available), direction],
        "defensive": {"raw_d": raw_d},
    }


class FeaturePredictabilityTest(unittest.TestCase):
    def test_realistic_missing_support_blocks_without_scoring_exam(self) -> None:
        rows = []
        for split, n in (("train", 6), ("val", 3)):
            rows.extend(_row(split, "right", True, 1.0, 2.0 + i) for i in range(n))
            rows.extend(_row(split, "none", False, 0.0, -2.0 - i) for i in range(n))
        rows.append(_row("holdout_exam", "left", True, -1.0, 99.0))
        report = build_feature_predictability_report(rows)
        self.assertEqual(report["status"], "DATA_SUPPORT_BLOCKED")
        conflict = next(t for t in report["tasks"] if t["task"] == "conflict_side")
        self.assertFalse(conflict["probe_ran"])
        self.assertTrue(any("train[left]" in r for r in conflict["support_reasons"]))
        self.assertIn("never fit", report["holdout_exam_policy"])

    def test_supported_separable_data_runs_all_probes(self) -> None:
        rows = []
        specs = [
            ("left", True, -1.0, -4.0),
            ("right", True, 1.0, 4.0),
            ("center", False, 0.0, 1.0),
            ("none", False, 0.0, -1.0),
        ]
        for split, n in (("train", 7), ("val", 4)):
            for side, available, direction, value in specs:
                for i in range(n):
                    rows.append(
                        _row(
                            split,
                            side,
                            available,
                            direction,
                            value + i * 0.01,
                        )
                    )
        report = build_feature_predictability_report(rows)
        self.assertEqual(report["status"], "FEATURE_PROBE_PASS")
        self.assertTrue(all(task["probe_ran"] for task in report["tasks"]))
        self.assertTrue(
            all(
                task["metrics"]["holdout_exam_used"] is False
                for task in report["tasks"]
            )
        )


if __name__ == "__main__":
    unittest.main()
