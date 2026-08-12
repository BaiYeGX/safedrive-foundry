"""R2 offline: lexicographic oracle, tie, both-bad, pilot labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.oracle import (  # noqa: E402
    LABEL_BOTH_BAD,
    LABEL_CANDIDATE_1_BEST,
    LABEL_INCOMPARABLE,
    LABEL_TIE,
    LABEL_TOP1_BEST,
    LEVEL_CLEARANCE,
    LEVEL_COLLISION,
    LEVEL_COMFORT,
    LEVEL_NEAR_CONFLICT,
    LEVEL_OFFROAD,
    LEVEL_PROGRESS,
    LEVEL_TIE_TOP1,
    PILOT_ENTER_WORLD,
    PILOT_IMPROVE_VLA,
    PILOT_INCONCLUSIVE,
    PILOT_NONE,
    PILOT_WEAK,
    assign_pilot_label,
    evaluate_pair_oracle,
)
from driving_vla.evaluation.outcome_metrics import (  # noqa: E402
    aggregate_branch_outcome,
    make_synthetic_ticks,
)


def _metrics(
    cid: str,
    idx: int,
    *,
    collision_at: float | None = None,
    offroad_fraction: float = 0.0,
    clearance_m: float | None = 5.0,
    ttc_s: float | None = 2.0,
    speed_mps: float = 5.0,
    jerk: float = 0.5,
    ticks: int = 50,
) -> object:
    ticks_list = make_synthetic_ticks(
        n=ticks,
        candidate_id=cid,
        speed_mps=speed_mps,
        collision_at=collision_at,
        offroad_fraction=offroad_fraction,
        clearance_m=clearance_m,
        ttc_s=ttc_s,
        jerk=jerk,
    )
    return aggregate_branch_outcome(ticks_list, candidate_id=cid, candidate_index=idx)


class G4AOracleTest(unittest.TestCase):
    def test_collision_beats_progress(self) -> None:
        m0 = _metrics("v1_nominal", 0, collision_at=1.0, speed_mps=8.0)
        m1 = _metrics("v1_conservative", 1, collision_at=None, speed_mps=3.0)
        r = evaluate_pair_oracle(
            pair_id="p1",
            scenario_id="lead_brake_hard",
            seed_id="seed_a",
            family="lead_braking",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.oracle_decision_level, LEVEL_COLLISION)
        self.assertEqual(r.pair_label, LABEL_CANDIDATE_1_BEST)

    def test_offroad_beats_progress(self) -> None:
        m0 = _metrics("v1_nominal", 0, offroad_fraction=0.20, speed_mps=8.0)
        m1 = _metrics("v1_conservative", 1, offroad_fraction=0.0, speed_mps=3.0)
        r = evaluate_pair_oracle(
            pair_id="p2",
            scenario_id="s",
            seed_id="seed_a",
            family="cut_in",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.oracle_decision_level, LEVEL_OFFROAD)

    def test_ttc_bucket_tiebreak(self) -> None:
        m0 = _metrics("v1_nominal", 0, ttc_s=0.3, clearance_m=5.0)
        m1 = _metrics("v1_conservative", 1, ttc_s=1.8, clearance_m=5.0)
        r = evaluate_pair_oracle(
            pair_id="p3",
            scenario_id="s",
            seed_id="seed_a",
            family="crossing",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.oracle_decision_level, LEVEL_NEAR_CONFLICT)

    def test_clearance_tiebreak(self) -> None:
        m0 = _metrics("v1_nominal", 0, ttc_s=2.0, clearance_m=1.0)
        m1 = _metrics("v1_conservative", 1, ttc_s=2.0, clearance_m=2.0)
        r = evaluate_pair_oracle(
            pair_id="p4",
            scenario_id="s",
            seed_id="seed_a",
            family="lead_braking",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.oracle_decision_level, LEVEL_CLEARANCE)

    def test_progress_tiebreak(self) -> None:
        m0 = _metrics("v1_nominal", 0, ttc_s=2.0, clearance_m=5.0, speed_mps=3.0)
        m1 = _metrics("v1_conservative", 1, ttc_s=2.0, clearance_m=5.0, speed_mps=6.0)
        r = evaluate_pair_oracle(
            pair_id="p5",
            scenario_id="s",
            seed_id="seed_a",
            family="lead_braking",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        # higher progress wins → candidate 1
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.oracle_decision_level, LEVEL_PROGRESS)

    def test_comfort_tiebreak(self) -> None:
        m0 = _metrics("v1_nominal", 0, ttc_s=2.0, clearance_m=5.0, speed_mps=5.0, jerk=3.0)
        m1 = _metrics("v1_conservative", 1, ttc_s=2.0, clearance_m=5.0, speed_mps=5.0, jerk=0.2)
        r = evaluate_pair_oracle(
            pair_id="p6",
            scenario_id="s",
            seed_id="seed_a",
            family="cut_in",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.oracle_decision_level, LEVEL_COMFORT)

    def test_exact_tie_returns_top1_label_tie(self) -> None:
        m0 = _metrics("v1_nominal", 0, ttc_s=2.0, clearance_m=5.0, speed_mps=5.0, jerk=0.5)
        m1 = _metrics("v1_conservative", 1, ttc_s=2.0, clearance_m=5.0, speed_mps=5.0, jerk=0.5)
        r = evaluate_pair_oracle(
            pair_id="p7",
            scenario_id="s",
            seed_id="seed_a",
            family="crossing",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.pair_label, LABEL_TIE)
        self.assertEqual(r.oracle_candidate_index, 0)
        self.assertEqual(r.oracle_decision_level, LEVEL_TIE_TOP1)

    def test_both_bad_keeps_relative_winner(self) -> None:
        m0 = _metrics("v1_nominal", 0, collision_at=0.5)
        m1 = _metrics("v1_conservative", 1, collision_at=1.5)
        r = evaluate_pair_oracle(
            pair_id="p8",
            scenario_id="s",
            seed_id="seed_a",
            family="lead_braking",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertTrue(r.both_bad)
        self.assertEqual(r.pair_label, LABEL_BOTH_BAD)
        # later collision better → index 1
        self.assertEqual(r.oracle_candidate_index, 1)
        self.assertEqual(r.relative_winner_if_both_bad, "v1_conservative")

    def test_incomparable(self) -> None:
        r = evaluate_pair_oracle(
            pair_id="p9",
            scenario_id="s",
            seed_id="seed_a",
            family="cut_in",
            comparable=False,
            top1_index=0,
            metrics0=None,
            metrics1=None,
            incomparable_reasons=("INITIAL_STATE_MISMATCH",),
        )
        self.assertEqual(r.pair_label, LABEL_INCOMPARABLE)
        self.assertIsNone(r.oracle_candidate_id)

    def test_top1_best(self) -> None:
        m0 = _metrics("v1_nominal", 0, collision_at=None)
        m1 = _metrics("v1_conservative", 1, collision_at=1.0)
        r = evaluate_pair_oracle(
            pair_id="p10",
            scenario_id="s",
            seed_id="seed_a",
            family="crossing",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertEqual(r.pair_label, LABEL_TOP1_BEST)
        self.assertEqual(r.oracle_candidate_index, 0)

    def test_does_not_use_probability(self) -> None:
        # Metrics only — probability field is not even on BranchOutcomeMetrics
        m0 = _metrics("v1_nominal", 0, clearance_m=1.0)
        m1 = _metrics("v1_conservative", 1, clearance_m=3.0)
        r = evaluate_pair_oracle(
            pair_id="p11",
            scenario_id="s",
            seed_id="seed_a",
            family="lead_braking",
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )
        self.assertNotIn("probability", r.to_dict())
        self.assertEqual(r.oracle_candidate_index, 1)

    def _pair(
        self,
        pair_id: str,
        family: str,
        label_kind: str,
        *,
        comparable: bool = True,
    ):
        if not comparable:
            return evaluate_pair_oracle(
                pair_id=pair_id,
                scenario_id=family + "_s",
                seed_id="seed_a",
                family=family,
                comparable=False,
                top1_index=0,
                metrics0=None,
                metrics1=None,
                incomparable_reasons=("x",),
            )
        if label_kind == "c1":
            m0 = _metrics("v1_nominal", 0, collision_at=1.0)
            m1 = _metrics("v1_conservative", 1, collision_at=None)
        elif label_kind == "top1":
            m0 = _metrics("v1_nominal", 0, collision_at=None)
            m1 = _metrics("v1_conservative", 1, collision_at=1.0)
        elif label_kind == "tie":
            m0 = _metrics("v1_nominal", 0)
            m1 = _metrics("v1_conservative", 1)
        elif label_kind == "both":
            m0 = _metrics("v1_nominal", 0, collision_at=0.5)
            m1 = _metrics("v1_conservative", 1, collision_at=0.5)
        else:
            raise ValueError(label_kind)
        return evaluate_pair_oracle(
            pair_id=pair_id,
            scenario_id=family + "_s",
            seed_id="seed_a",
            family=family,
            comparable=True,
            top1_index=0,
            metrics0=m0,
            metrics1=m1,
        )

    def test_pilot_improve_vla_both_bad(self) -> None:
        rows = [self._pair(f"b{i}", "lead_braking", "both") for i in range(6)]
        rows += [self._pair(f"t{i}", "cut_in", "top1") for i in range(2)]
        summary = assign_pilot_label(rows)
        self.assertEqual(summary.label, PILOT_IMPROVE_VLA)

    def test_pilot_improve_vla_low_comparable(self) -> None:
        rows = [self._pair(f"i{i}", "lead_braking", "top1", comparable=False) for i in range(8)]
        rows += [self._pair(f"t{i}", "cut_in", "top1") for i in range(2)]
        summary = assign_pilot_label(rows)
        self.assertEqual(summary.label, PILOT_IMPROVE_VLA)

    def test_pilot_enter_world(self) -> None:
        rows = [
            self._pair("c1", "lead_braking", "c1"),
            self._pair("c2", "cut_in", "c1"),
            self._pair("t1", "crossing", "top1"),
            self._pair("t2", "lead_braking", "top1"),
            self._pair("t3", "cut_in", "top1"),
            self._pair("t4", "crossing", "top1"),
        ]
        summary = assign_pilot_label(rows)
        self.assertEqual(summary.label, PILOT_ENTER_WORLD)

    def test_pilot_weak(self) -> None:
        rows = [
            self._pair("c1", "lead_braking", "c1"),
            self._pair("t1", "lead_braking", "top1"),
            self._pair("t2", "lead_braking", "top1"),
            self._pair("t3", "lead_braking", "top1"),
        ]
        summary = assign_pilot_label(rows)
        self.assertEqual(summary.label, PILOT_WEAK)

    def test_pilot_no_selection_space(self) -> None:
        rows = [self._pair(f"t{i}", "lead_braking", "top1") for i in range(6)]
        rows += [self._pair(f"e{i}", "cut_in", "tie") for i in range(4)]
        summary = assign_pilot_label(rows)
        self.assertEqual(summary.label, PILOT_NONE)

    def test_pilot_inconclusive_fallback(self) -> None:
        # Comparable rate medium, no c1 wins, but top1+tie share < 80% of comparable
        # because many both_bad without triggering 50%? Use mixed incomparable + few ties.
        rows = [self._pair(f"i{i}", "lead_braking", "top1", comparable=False) for i in range(3)]
        rows += [self._pair(f"b{i}", "cut_in", "both") for i in range(2)]
        rows += [self._pair(f"t{i}", "crossing", "tie") for i in range(2)]
        summary = assign_pilot_label(rows)
        # both_bad 2/4 = 50% of comparable → IMPROVE_VLA actually
        # Adjust: 1 both_bad only
        rows = [self._pair(f"i{i}", "lead_braking", "top1", comparable=False) for i in range(3)]
        rows += [self._pair("b0", "cut_in", "both")]
        rows += [self._pair(f"e{i}", "crossing", "tie") for i in range(3)]
        summary = assign_pilot_label(rows)
        # comparable=4, both=1 (25%), rate=4/7≈0.57, c1=0, top1+tie = 3/4=0.75 < 0.80
        # → INCONCLUSIVE
        self.assertEqual(summary.label, PILOT_INCONCLUSIVE)


if __name__ == "__main__":
    unittest.main()
