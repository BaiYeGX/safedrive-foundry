"""Tests for the H5 World-on/off hysteresis router."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from driving_vla.hybrid.contracts import (
    CandidateDifference,
    RoutingResult,
    SelectionSpace,
    WorldDisposition,
)
from data_pipeline.h5.runtime import H5WorldRouter


class _Guard:
    def __init__(self, passed: bool = True):
        self.passed = passed


class _Candidate:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id


class _Item:
    def __init__(self, candidate_id: str, passed: bool = True):
        self.guard = _Guard(passed)
        self.candidate = _Candidate(candidate_id)


class _CandidateSet:
    def __init__(self, ids):
        self.candidates = [_Item(i) for i in ids]


class _Score:
    def __init__(self, selected, utility_first=1.0, utility_second=-1.0, disposition="ranked", reason=None):
        self.disposition = disposition
        self.selected_candidate_key = selected
        self.defer_reason = reason
        self.predictions = [
            SimpleNamespace(candidate_key="a", utility=utility_first),
            SimpleNamespace(candidate_key="b", utility=utility_second),
        ]


class _Scorer:
    def __init__(self, selected="a", utility_first=1.0, utility_second=-1.0, disposition="ranked", reason=None):
        self.selected = selected
        self.utility_first = utility_first
        self.utility_second = utility_second
        self.disposition = disposition
        self.reason = reason

    def score_pair(self, first, second):
        return _Score(self.selected, self.utility_first, self.utility_second, self.disposition, self.reason)


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


class H5HysteresisTest(unittest.TestCase):
    def _features(self, ids=("a", "b")):
        return {i: ([0.0] * 499, tuple(tuple([0.0] * 8 for _ in range(10)))) for i in ids}

    def test_holds_selection_until_min_ticks(self):
        router = H5WorldRouter(_Scorer("b"), _Fallback(), min_hold_ticks=3, hysteresis_margin=0.0)
        cs = _CandidateSet(["a", "b"])
        features = self._features()
        # First call selects b and starts hold.
        r1 = router.route(cs, features)
        self.assertEqual(r1.selected_candidate_id, "b")
        self.assertEqual(r1.reason, "h5_world_ranked")
        # Even though scorer still says b, no switch issue; to test hold, we
        # change scorer to propose a after first call.
        router.scorer = _Scorer("a", 0.2, -0.2)
        r2 = router.route(cs, features)
        self.assertEqual(r2.selected_candidate_id, "b")
        self.assertEqual(r2.reason, "h5_world_hold_hysteresis")
        # After hold ticks and large margin, switch is allowed.
        router._hold_count = router.min_hold_ticks
        r3 = router.route(cs, features)
        self.assertEqual(r3.selected_candidate_id, "a")
        self.assertEqual(r3.reason, "h5_world_ranked")

    def test_emergency_margin_breaks_hold(self):
        router = H5WorldRouter(
            _Scorer("b"), _Fallback(),
            min_hold_ticks=10, hysteresis_margin=0.05, emergency_switch_margin=1.5,
        )
        cs = _CandidateSet(["a", "b"])
        features = self._features()
        r1 = router.route(cs, features)
        self.assertEqual(r1.selected_candidate_id, "b")
        # Huge new-candidate advantage must switch before the hold expires.
        router.scorer = _Scorer("a", 3.0, -3.0)
        r2 = router.route(cs, features)
        self.assertEqual(r2.selected_candidate_id, "a")
        self.assertEqual(r2.reason, "h5_world_ranked")

    def test_metrics_tracks_switch_and_defer(self):
        router = H5WorldRouter(_Scorer("b"), _Fallback(), min_hold_ticks=3, hysteresis_margin=0.0)
        cs = _CandidateSet(["a", "b"]); features = self._features()
        router.route(cs, features)
        router.scorer = _Scorer("a", 3.0, -3.0)
        router.route(cs, features)
        router.scorer = _Scorer(selected=None, disposition="defer_low_confidence", reason="risk")
        router.route(cs, features)
        m = router.metrics()
        self.assertEqual(m["switch_count"], 1)
        self.assertEqual(m["defer_count"], 1)

    def test_defer_resets_and_falls_back(self):
        scorer = _Scorer(selected=None, disposition="defer_low_confidence", reason="risk")
        router = H5WorldRouter(scorer, _Fallback(), min_hold_ticks=3)
        cs = _CandidateSet(["a", "b"])
        r = router.route(cs, self._features())
        self.assertEqual(r.selector, "fallback")
        self.assertEqual(r.world, WorldDisposition.DEFERRED_LOW_CONFIDENCE)
        self.assertEqual(router._last_selected_id, None)


if __name__ == "__main__":
    unittest.main()
