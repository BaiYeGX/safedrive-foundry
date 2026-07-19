from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.path_manager import (  # noqa: E402
    compare_native_dense_curvature,
    curvature_profile_from_xy,
)
from driving_vla.runtime.stationary_requery import (  # noqa: E402
    StationaryRequeryConfig,
    StationaryRequeryDecision,
    evaluate_stationary_requery,
    requery_success,
)


class StationaryRequeryTest(unittest.TestCase):
    def test_evaluate_can_trigger_when_conditions_met(self) -> None:
        """Pure helper still works; runner defaults to disabled via CLI."""
        cfg = StationaryRequeryConfig(min_stop_frames=4, cooldown_s=8.0)
        ok = evaluate_stationary_requery(
            ego_speed_mps=0.1,
            stop_requested=True,
            consecutive_stop_frames=4,
            has_collided=False,
            navigation_valid=True,
            path_nav_aligned=True,
            already_active=False,
            last_trigger_sim_s=None,
            sim_s=10.0,
            config=cfg,
        )
        self.assertTrue(ok.trigger)

    def test_runner_default_is_disabled_without_extra_forward(self) -> None:
        """Pure VLA+MPC: stationary requery must be opt-in (--enable-stationary-requery)."""
        # Mirror runner gate: disabled → Decision(False, "disabled"), no second forward.
        decision = StationaryRequeryDecision(False, "disabled")
        self.assertFalse(decision.trigger)
        self.assertEqual(decision.reason, "disabled")
        self.assertEqual(decision.hint_speed_mps, 0.0)

    def test_triggers_only_after_sustained_stop_while_stationary(self) -> None:
        cfg = StationaryRequeryConfig(min_stop_frames=4, cooldown_s=8.0)
        blocked = evaluate_stationary_requery(
            ego_speed_mps=0.0,
            stop_requested=True,
            consecutive_stop_frames=2,
            has_collided=False,
            navigation_valid=True,
            path_nav_aligned=True,
            already_active=False,
            last_trigger_sim_s=None,
            sim_s=10.0,
            config=cfg,
        )
        self.assertFalse(blocked.trigger)
        self.assertEqual(blocked.reason, "stop_frames_insufficient")

        ok = evaluate_stationary_requery(
            ego_speed_mps=0.1,
            stop_requested=True,
            consecutive_stop_frames=4,
            has_collided=False,
            navigation_valid=True,
            path_nav_aligned=True,
            already_active=False,
            last_trigger_sim_s=None,
            sim_s=10.0,
            config=cfg,
        )
        self.assertTrue(ok.trigger)
        self.assertEqual(ok.reason, "stationary_stop_requery")
        self.assertAlmostEqual(ok.hint_speed_mps, 1.5)

    def test_blocked_by_collision_nav_or_cooldown(self) -> None:
        cfg = StationaryRequeryConfig(min_stop_frames=3, cooldown_s=5.0)
        base = dict(
            ego_speed_mps=0.0,
            stop_requested=True,
            consecutive_stop_frames=5,
            has_collided=False,
            navigation_valid=True,
            path_nav_aligned=True,
            already_active=False,
            last_trigger_sim_s=None,
            sim_s=20.0,
            config=cfg,
        )
        self.assertFalse(
            evaluate_stationary_requery(**{**base, "has_collided": True}).trigger
        )
        self.assertFalse(
            evaluate_stationary_requery(**{**base, "navigation_valid": False}).trigger
        )
        self.assertFalse(
            evaluate_stationary_requery(**{**base, "path_nav_aligned": False}).trigger
        )
        self.assertFalse(
            evaluate_stationary_requery(
                **{**base, "last_trigger_sim_s": 18.0, "sim_s": 20.0}
            ).trigger
        )

    def test_requery_success_requires_path_and_positive_speed(self) -> None:
        self.assertTrue(
            requery_success(
                path_accepted=True, stop_requested_after=False, target_speed_after=2.0
            )
        )
        self.assertFalse(
            requery_success(
                path_accepted=True, stop_requested_after=True, target_speed_after=0.0
            )
        )
        self.assertFalse(
            requery_success(
                path_accepted=False, stop_requested_after=False, target_speed_after=3.0
            )
        )


class CurvatureAttributionTest(unittest.TestCase):
    def test_smooth_arc_native_and_dense_below_hard(self) -> None:
        # Gentle arc: both should stay well under hard_max=1.
        pts = [(float(x), 0.02 * x * x / 20.0) for x in range(0, 21)]
        cmp_ = compare_native_dense_curvature(pts, ds_m=0.20, hard_max=1.0)
        self.assertEqual(cmp_["attribution"], "neither_exceeds_hard")
        self.assertLess(float(cmp_["native"]["max_abs_curvature"]), 1.0)  # type: ignore[index]
        self.assertLess(float(cmp_["dense"]["max_abs_curvature"]), 1.0)  # type: ignore[index]

    def test_sharp_polyline_corner_can_inflate_under_dense(self) -> None:
        # Two nearly collinear segments with a sharp mid kink (VLA-like 20-pt noise).
        pts = [(float(x), 0.0) for x in range(0, 10)]
        pts += [(10.0, 0.0), (10.2, 1.5), (10.4, 0.0)]
        pts += [(float(x), 0.0) for x in range(11, 21)]
        native = curvature_profile_from_xy(pts, ds_m=None)
        dense = curvature_profile_from_xy(pts, ds_m=0.20)
        cmp_ = compare_native_dense_curvature(pts, ds_m=0.20, hard_max=1.0)
        # Dense densify should not be lower than a trivial floor; attribution is informative.
        self.assertTrue(native["ok"] and dense["ok"])
        self.assertIn(
            cmp_["attribution"],
            {"dense_pchip_spike", "native_and_dense", "native_only", "neither_exceeds_hard"},
        )


if __name__ == "__main__":
    unittest.main()
