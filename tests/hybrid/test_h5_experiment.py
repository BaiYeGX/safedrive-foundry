"""Tests for H5 closed-loop matrix, router hardening, and metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from data_pipeline.h5.matrix import load_h5_matrix  # noqa: E402
from data_pipeline.h5.metrics import evaluate_gate, paired_progress_ci  # noqa: E402
from data_pipeline.h5.runtime import H5WorldRouter  # noqa: E402
from driving_vla.hybrid.contracts import (  # noqa: E402
    CandidateDifference,
    RoutingResult,
    SelectionSpace,
    WorldDisposition,
)


class _Guard:
    def __init__(self, passed: bool = True):
        self.passed = passed


class _Candidate:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id


class _Provenance:
    def __init__(self, candidate_id: str):
        self.source = type("Src", (), {"value": candidate_id.split(":")[-1]})()
        self.candidate_id = candidate_id


class _Item:
    def __init__(self, candidate_id: str, passed: bool = True):
        self.guard = _Guard(passed)
        self.candidate = _Candidate(candidate_id)
        self.provenance = _Provenance(candidate_id)


class _CandidateSet:
    def __init__(self, ids):
        self.candidates = []
        for spec in ids:
            if isinstance(spec, tuple):
                candidate_id, passed = spec
                self.candidates.append(_Item(candidate_id, passed))
            else:
                self.candidates.append(_Item(spec))


class _Score:
    def __init__(self, selected, utility_first=1.0, utility_second=-1.0, disposition="ranked", reason=None, latency_ms=1.0):
        self.disposition = disposition
        self.selected_candidate_key = selected
        self.defer_reason = reason
        self.latency_ms = latency_ms
        self.predictions = [
            SimpleNamespace(candidate_key="a", utility=utility_first),
            SimpleNamespace(candidate_key="b", utility=utility_second),
        ]


class _Scorer:
    def __init__(self, selected="a", utility_first=1.0, utility_second=-1.0, disposition="ranked", reason=None, latency_ms=1.0):
        self.selected = selected
        self.utility_first = utility_first
        self.utility_second = utility_second
        self.disposition = disposition
        self.reason = reason
        self.latency_ms = latency_ms

    def score_pair(self, first, second):
        return _Score(
            self.selected, self.utility_first, self.utility_second,
            self.disposition, self.reason, self.latency_ms,
        )


class _Fallback:
    def route(self, candidate_set):
        return RoutingResult(
            pass_candidate_ids=tuple(i.candidate.candidate_id for i in candidate_set.candidates),
            rejected_candidate_ids=(),
            selected_candidate_id=candidate_set.candidates[0].candidate.candidate_id,
            selection_space=SelectionSpace.DISTINCT,
            world=WorldDisposition.DEFERRED_LOW_CONFIDENCE,
            selector="fallback",
            reason="fallback",
            difference=CandidateDifference(max_position_delta_m=1.0, rms_speed_delta_mps=0.6),
        )


def _features(ids=("a", "b")):
    return {i: ([0.0] * 499, tuple(tuple([0.0] * 8 for _ in range(10)))) for i in ids}


class H5RouterHardeningTest(unittest.TestCase):
    def test_force_defer_still_scores_and_falls_back(self):
        router = H5WorldRouter(_Scorer("b"), _Fallback(), force_defer=True)
        cs = _CandidateSet(["a", "b"])
        r = router.route(cs, _features())
        self.assertEqual(r.selector, "fallback")
        self.assertEqual(r.world, WorldDisposition.DEFERRED_LOW_CONFIDENCE)
        self.assertEqual(router.metrics()["defer_count"], 1)
        self.assertIsNotNone(router.last_score)

    def test_scorer_deadline_defers(self):
        router = H5WorldRouter(_Scorer("b", latency_ms=60.0), _Fallback(), scorer_deadline_ms=50.0)
        r = router.route(_CandidateSet(["a", "b"]), _features())
        self.assertEqual(r.selector, "fallback")
        self.assertIn("deadline", r.reason)

    def test_single_pass_glitch_preserves_cumulative_counters(self):
        router = H5WorldRouter(_Scorer("b"), _Fallback(), min_hold_ticks=2, single_pass_grace_ticks=2)
        cs = _CandidateSet(["a", "b"])
        router.route(cs, _features())
        # Three single-pass ticks should trigger clear_hold twice but keep counters.
        for _ in range(3):
            router.route(_CandidateSet([("a", True), ("b", False)]), _features(("a", "b")))
        self.assertEqual(router.metrics()["defer_count"], 0)
        self.assertEqual(router.metrics()["switch_count"], 0)


class H5MatrixTest(unittest.TestCase):
    def test_full_matrix_is_locked_test_valid_rows(self):
        scenarios = load_h5_matrix(ROOT, full=True)
        self.assertEqual(len(scenarios), 74)
        self.assertTrue(all(s.scenario.map_name in {"Town01", "Town03", "Town05"} for s in scenarios))
        self.assertTrue(all(len(s.arm_order) == 3 for s in scenarios))
        self.assertTrue(all(set(s.arm_order) == {"off", "on", "defer"} for s in scenarios))

    def test_pilot_is_subset(self):
        pilot = load_h5_matrix(ROOT, full=False)
        full = load_h5_matrix(ROOT, full=True)
        self.assertEqual(len(pilot), 12)
        pilot_ids = {s.pair_id for s in pilot}
        full_ids = {s.pair_id for s in full}
        self.assertTrue(pilot_ids <= full_ids)


def _run(pair_id, arm, progress=10.0, switch=0, collision=0, ok=True, cleanup=True, scorer=None):
    return {
        "pair_id": pair_id,
        "arm": arm,
        "route_progress_m": progress,
        "route_completed": progress > 50.0,
        "collision_count": collision,
        "red_light_violation": False,
        "off_corridor_duration_s": 0.0,
        "switch_count": switch,
        "defer_count": 0,
        "fallback_count": 0,
        "safety_fallback_count": 0,
        "scorer_deadline_misses": 0,
        "deadline_misses": 0,
        "p50_scorer_ms": 10.0,
        "p95_scorer_ms": 15.0,
        "p99_scorer_ms": 20.0,
        "whole_gpu_peak_gb": 3.0,
        "ok": ok,
        "cleanup_complete": cleanup,
        "ticks_executed": 50,
        "vla_forward_count": 50,
        "decisions": [],
    }


class H5MetricsTest(unittest.TestCase):
    def test_paired_progress_ci_positive(self):
        runs = [_run(f"p{i}", "off", progress=10.0) for i in range(10)]
        runs += [_run(f"p{i}", "on", progress=12.0) for i in range(10)]
        ci = paired_progress_ci(runs, rounds=200)
        self.assertEqual(ci["n"], 10)
        self.assertGreater(ci["mean_delta"], 0.0)
        self.assertGreater(ci["lower_95"], 0.0)

    def test_gate_reports_failure_when_on_unsafe(self):
        runs = [_run("p1", "off", progress=10.0), _run("p1", "on", progress=9.0, collision=1)]
        gate = evaluate_gate(runs)
        self.assertFalse(gate["passed"])
        self.assertIn("safety_noninferior", gate["failures"])


if __name__ == "__main__":
    unittest.main()
