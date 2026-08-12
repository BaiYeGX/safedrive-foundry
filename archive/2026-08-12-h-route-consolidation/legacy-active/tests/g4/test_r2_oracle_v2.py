from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.oracle import (  # noqa: E402
    LABEL_CANDIDATE_1_BEST,
    LABEL_TOP1_BEST,
    LEVEL_CLEARANCE,
    LEVEL_PROGRESS,
)
from driving_vla.evaluation.oracle_v2 import (  # noqa: E402
    evaluate_pair_oracle_v2,
    minimum_intervention_cost,
)
from driving_vla.evaluation.outcome_metrics import (  # noqa: E402
    aggregate_branch_outcome,
    make_synthetic_ticks,
)


def _metrics(candidate_id: str, index: int, *, clearance: float, speed: float):
    return aggregate_branch_outcome(
        make_synthetic_ticks(
            n=50,
            candidate_id=candidate_id,
            speed_mps=speed,
            clearance_m=clearance,
            ttc_s=2.0,
            jerk=0.5,
        ),
        candidate_id=candidate_id,
        candidate_index=index,
    )


class OracleV2Test(unittest.TestCase):
    def test_safe_clearance_saturates_then_progress_wins(self) -> None:
        nominal = _metrics(
            "v3_nominal_progress", 0, clearance=5.0, speed=6.0
        )
        detour = _metrics("v3_alternative", 1, clearance=10.0, speed=3.0)
        result = evaluate_pair_oracle_v2(
            pair_id="pair",
            scenario_id="cut_in",
            seed_id="seed_a",
            family="cut_in",
            comparable=True,
            top1_index=0,
            metrics0=nominal,
            metrics1=detour,
        )
        self.assertEqual(result.oracle_candidate_index, 0)
        self.assertEqual(result.oracle_decision_level, LEVEL_PROGRESS)
        self.assertEqual(result.pair_label, LABEL_TOP1_BEST)
        self.assertEqual(
            result.outcome_delta["saturated_clearance_m"], [2.0, 2.0]
        )

    def test_below_safe_clearance_remains_decisive(self) -> None:
        unsafe = _metrics("v3_nominal_progress", 0, clearance=0.8, speed=6.0)
        safe = _metrics("v3_alternative", 1, clearance=2.5, speed=3.0)
        result = evaluate_pair_oracle_v2(
            pair_id="pair",
            scenario_id="cut_in",
            seed_id="seed_a",
            family="cut_in",
            comparable=True,
            top1_index=0,
            metrics0=unsafe,
            metrics1=safe,
        )
        self.assertEqual(result.oracle_candidate_index, 1)
        self.assertEqual(result.oracle_decision_level, LEVEL_CLEARANCE)
        self.assertEqual(result.pair_label, LABEL_CANDIDATE_1_BEST)

    def test_minimum_intervention_prefers_progress_when_both_safe(self) -> None:
        progress = minimum_intervention_cost(
            {
                "collision": False,
                "ttc_s": 3.0,
                "clearance_m": 5.0,
                "progress_m": 10.0,
                "jerk_p95": 1.0,
            }
        )
        detour = minimum_intervention_cost(
            {
                "collision": False,
                "ttc_s": 3.0,
                "clearance_m": 10.0,
                "progress_m": 6.0,
                "jerk_p95": 1.0,
            }
        )
        self.assertLess(progress, detour)


if __name__ == "__main__":
    unittest.main()
