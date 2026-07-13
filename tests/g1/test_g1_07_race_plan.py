from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.planning.race import run_race_plan_ablation  # noqa: E402
from classic_stack.risk import evaluate_risk_field, monotonicity_ok  # noqa: E402


class G107RacePlanTests(unittest.TestCase):
    def test_risk_field_and_monotonicity(self) -> None:
        field = evaluate_risk_field(ego_v=8.0, actors=[{"s": 15.0, "v": 5.0}], uncertainty_scale=1.0)
        self.assertTrue(field.samples)
        self.assertIsNotNone(field.observable_score)
        self.assertGreaterEqual(field.observable_score, 0.0)
        self.assertLess(field.observable_score, 1e6)
        # past actor must not explode
        past = evaluate_risk_field(ego_v=8.0, actors=[{"s": -1.0, "v": 0.0}], uncertainty_scale=1.0)
        self.assertGreaterEqual(past.observable_score, 0.0)
        self.assertLess(past.observable_score, 1e6)
        self.assertTrue(monotonicity_ok([{"s": 15.0, "v": 5.0}]))

    def test_ablation_matrix(self) -> None:
        out = run_race_plan_ablation(ROOT)
        self.assertEqual(set(out["matrix"]), {"basic", "p1", "p2", "full"})
        self.assertTrue(out["risk_monotonicity_ok"])
        self.assertEqual(len(out["frenet_baseline_hash"]), 64)
        self.assertEqual(len(out["hybrid_baseline_hash"]), 64)
        self.assertIn("promote_full_to_default", out["default_admission"])
        self.assertIn("candidates_raw_sum", out["matrix"]["basic"])
        self.assertIn("nodes_raw_sum", out["matrix"]["basic"])
        self.assertIn("honesty", out)
        # Net-benefit: full must not promote if it only adds work (current data)
        adm = out["default_admission"]
        work_ratio = adm.get("work_ratio_full_over_basic", 1.0)
        if work_ratio > 1.0 + 1e-9:
            self.assertFalse(
                adm["promote_full_to_default"],
                msg=f"full work_ratio={work_ratio} must not promote",
            )
            self.assertIn(adm.get("recommended_default"), ("basic", "p1"))
        # baseline hashes must match frozen files
        import hashlib

        fh = hashlib.sha256(
            (ROOT / "safedrive_foundry/config/classic_stack/frenet_st_baseline.toml").read_bytes()
        ).hexdigest()
        self.assertEqual(out["frenet_baseline_hash"], fh)


if __name__ == "__main__":
    unittest.main()
