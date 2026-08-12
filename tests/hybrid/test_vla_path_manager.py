from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.path_manager import (  # noqa: E402
    EgoPose,
    PathManagerConfig,
    VLAPathManager,
)


class VLAPathManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ego = EgoPose(0.0, 0.0, 0.0, 2.0)
        self.manager = VLAPathManager(
            PathManagerConfig(
                max_abs_curvature=0.30,
                max_switch_lateral_5m=1.0,
                max_switch_heading_5m_deg=12.0,
                # Default live config disables this gate; unit tests that
                # exercise it opt in explicitly.
                enable_lateral_mode_flip=False,
            )
        )

    def test_straight_path_is_dense_and_accepted(self) -> None:
        points = [(float(x), 0.0) for x in range(1, 21)]
        update = self.manager.update(points, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0)
        self.assertTrue(update.accepted, update.reason)
        self.assertIsNotNone(update.committed)
        assert update.committed is not None
        self.assertGreater(update.committed.length_m, 18.0)
        self.assertLess(float(np.max(np.abs(update.committed.y))), 1e-8)
        self.assertLess(float(np.max(np.abs(update.committed.kappa))), 1e-6)

    def test_native_commitment_can_refresh_execution_metadata_without_geometry(self) -> None:
        points = [(float(x), 0.0) for x in range(1, 21)]
        update = self.manager.update(points, ego=self.ego, target_speed_mps=0.1, stamp_s=1.0)
        self.assertTrue(update.accepted, update.reason)
        assert update.committed is not None
        x_before = update.committed.x.copy()
        y_before = update.committed.y.copy()
        source_before = update.committed.source_id
        self.assertTrue(
            self.manager.update_committed_execution(target_speed_mps=2.5, stamp_s=2.0)
        )
        refreshed = self.manager.committed
        assert refreshed is not None
        self.assertAlmostEqual(refreshed.target_speed_mps, 2.5)
        self.assertAlmostEqual(refreshed.stamp_s, 2.0)
        self.assertEqual(refreshed.source_id, source_before)
        np.testing.assert_allclose(refreshed.x, x_before)
        np.testing.assert_allclose(refreshed.y, y_before)

    def test_native_commitment_refresh_without_path_is_noop(self) -> None:
        self.assertFalse(
            self.manager.update_committed_execution(target_speed_mps=2.0, stamp_s=1.0)
        )

    def test_near_prefix_is_committed_while_tail_blends(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0).accepted
        )
        gentle = [(float(x), 0.04 * x) for x in range(1, 21)]
        update = self.manager.update(gentle, ego=self.ego, target_speed_mps=2.0, stamp_s=1.5)
        self.assertTrue(update.accepted, update.reason)
        assert update.committed is not None
        _x, y, _yaw, _k = update.committed.sample(np.array([1.0, 8.0]))
        # ego_snap keeps near field on latest; allow small residual from densify.
        self.assertLess(abs(float(y[0])), 0.05)
        self.assertGreater(float(y[1]), 0.05)
        self.assertLess(float(y[1]), 0.40)

    def test_large_lateral_jump_is_soft_committed_without_stale_age(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        first = self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0)
        jumped = [(float(x), 0.4 * x) for x in range(1, 21)]
        update = self.manager.update(jumped, ego=self.ego, target_speed_mps=2.0, stamp_s=1.5)
        self.assertTrue(update.accepted, update.reason)
        self.assertTrue(update.reason.startswith("accepted_soft_"), update.reason)
        assert update.committed is not None and first.committed is not None
        self.assertEqual(update.committed.stamp_s, 1.5)
        self.assertIsNot(update.committed, first.committed)
        self.assertLessEqual(
            float(np.max(np.abs(update.committed.kappa))),
            self.manager.config.hard_max_abs_curvature + 1e-6,
        )

    def test_hard_rejected_geometry_can_still_reduce_vla_speed(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        first = self.manager.update(straight, ego=self.ego, target_speed_mps=8.0, stamp_s=1.0)
        assert first.committed is not None
        invalid = [(1.0, 0.0), (3.0, 2.0), (1.0, 2.0), (3.0, 0.0), (5.0, 0.0)]
        update = self.manager.update(invalid, ego=self.ego, target_speed_mps=0.0, stamp_s=1.5)
        self.assertFalse(update.accepted)
        self.assertEqual(update.reason, "self_intersection")
        assert update.committed is not None
        self.assertEqual(update.committed.target_speed_mps, 0.0)
        self.assertEqual(update.committed.stamp_s, first.committed.stamp_s)
        np.testing.assert_allclose(update.committed.x, first.committed.x)

    def test_accepted_geometry_does_not_temporally_average_a_vla_stop(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.manager.update(straight, ego=self.ego, target_speed_mps=8.0, stamp_s=1.0)
        gentle = [(float(x), 0.02 * x) for x in range(1, 21)]
        stopped = self.manager.update(gentle, ego=self.ego, target_speed_mps=0.0, stamp_s=1.5)
        self.assertTrue(stopped.accepted, stopped.reason)
        assert stopped.committed is not None
        self.assertEqual(stopped.committed.target_speed_mps, 0.0)

    def test_self_intersection_is_rejected(self) -> None:
        loop = [(1.0, 0.0), (3.0, 2.0), (1.0, 2.0), (3.0, 0.0), (5.0, 0.0)]
        update = self.manager.update(loop, ego=self.ego, target_speed_mps=1.0, stamp_s=1.0)
        self.assertFalse(update.accepted)
        self.assertEqual(update.reason, "self_intersection")

    def test_single_interpolation_curvature_spike_uses_robust_gate(self) -> None:
        points = [(float(x), 0.2 if x == 10 else 0.0) for x in range(1, 21)]
        update = self.manager.update(points, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0)
        self.assertTrue(update.accepted, update.reason)
        self.assertGreater(update.quality.max_abs_curvature, 0.30)
        self.assertLess(update.quality.curvature_quantile, 0.30)

    @staticmethod
    def _lateral_jump_path(offset_y: float = 2.0, ramp_m: float = 8.0) -> list[tuple[float, float]]:
        """Smooth lateral shift: low kappa, but >1 m offset at 5 m (lateral_switch)."""
        pts: list[tuple[float, float]] = []
        for x in range(1, 21):
            xf = float(x)
            y = float(offset_y) * min(1.0, xf / max(ramp_m, 1e-3))
            pts.append((xf, y))
        return pts

    def test_legitimate_turn_refreshes_reference_before_vehicle_stops(self) -> None:
        """A switch diagnostic must not create the stale→stop→reanchor loop."""
        straight = [(float(x), 0.0) for x in range(1, 21)]
        turned = self._lateral_jump_path(2.0)
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0).accepted
        )
        update = self.manager.update(
            turned,
            ego=EgoPose(0.0, 0.0, 0.0, 1.0),
            target_speed_mps=2.0,
            stamp_s=4.0,
            nav_target_map_xy=(20.0, 2.0),
        )
        self.assertTrue(update.accepted, update.reason)
        self.assertTrue(update.reason.startswith("accepted_soft_"), update.reason)
        self.assertEqual(update.reanchor_pending_count, 0)
        assert update.committed is not None
        self.assertEqual(update.committed.stamp_s, 4.0)

    def test_town15_style_turn_sequence_never_ages_on_switch_diagnostics(self) -> None:
        """Regression for n=147–151: valid turning frames must remain fresh."""
        manager = VLAPathManager(
            PathManagerConfig(
                max_abs_curvature=0.30,
                max_switch_lateral_5m=0.25,
                max_switch_heading_5m_deg=5.0,
            )
        )
        straight = [(float(x), 0.0) for x in range(1, 21)]
        first = manager.update(
            straight,
            ego=self.ego,
            target_speed_mps=5.0,
            stamp_s=10.0,
            nav_target_map_xy=(25.0, 0.0),
        )
        self.assertTrue(first.accepted)

        reasons: list[str] = []
        for frame, offset in enumerate((1.5, 2.0, 2.5, 3.0), start=1):
            update = manager.update(
                self._lateral_jump_path(offset, ramp_m=8.0),
                ego=EgoPose(0.0, 0.0, 0.0, 2.5),
                target_speed_mps=4.5,
                stamp_s=10.0 + 0.75 * frame,
                nav_target_map_xy=(20.0, offset),
            )
            self.assertTrue(update.accepted, update.reason)
            self.assertEqual(update.reanchor_pending_count, 0)
            assert update.committed is not None
            self.assertEqual(update.committed.stamp_s, 10.0 + 0.75 * frame)
            self.assertLessEqual(
                float(np.max(np.abs(update.committed.kappa))),
                manager.config.hard_max_abs_curvature + 1e-6,
            )
            reasons.append(update.reason)
        self.assertTrue(any(r.startswith("accepted_soft_") for r in reasons), reasons)

    def test_projected_prefix_does_not_create_near_endpoint_kink(self) -> None:
        """A small tracking offset must not be reconnected by a sharp ego segment."""
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(
                straight, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0
            ).accepted
        )
        moved = EgoPose(1.0, 0.12, 0.05, 2.0)
        latest = [(1.0 + float(x), 0.12 + 0.03 * x) for x in range(1, 21)]
        update = self.manager.update(
            latest,
            ego=moved,
            target_speed_mps=3.0,
            stamp_s=1.75,
            nav_target_map_xy=(25.0, 1.0),
        )
        self.assertTrue(update.accepted, update.reason)
        assert update.committed is not None
        near = np.abs(update.committed.kappa[update.committed.s <= 0.5])
        self.assertGreater(near.size, 0)
        self.assertLess(float(np.max(near)), 0.75)

    def test_reverse_candidate_is_hard_rejected(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0).accepted
        )
        # Path going backward relative to coarse nav in +x.
        reverse = [(-float(x), 0.0) for x in range(1, 21)]
        stopped = EgoPose(0.0, 0.0, 0.0, 0.0)
        update = self.manager.update(
            reverse,
            ego=stopped,
            target_speed_mps=2.0,
            stamp_s=5.0,
            nav_target_map_xy=(30.0, 0.0),
        )
        self.assertFalse(update.accepted)
        self.assertIn(update.reason, {"nav_reverse", "not_forward"})
        self.assertEqual(update.reanchor_pending_count, 0)
        # Committed still the original straight path.
        assert update.committed is not None
        self.assertGreater(float(update.committed.x[-1]), 5.0)

    def test_single_frame_divergent_candidate_uses_bounded_transition(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        left = self._lateral_jump_path(2.0)
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0).accepted
        )
        first = self.manager.update(
            left,
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=2.0,
            stamp_s=5.0,
            nav_target_map_xy=(20.0, 2.0),
        )
        self.assertTrue(first.accepted, first.reason)
        self.assertTrue(first.reason.startswith("accepted_soft_"), first.reason)
        self.assertEqual(first.reanchor_pending_count, 0)
        self.assertIsNotNone(first.committed)
        assert first.committed is not None
        self.assertEqual(first.committed.stamp_s, 5.0)
        self.assertLessEqual(
            float(np.max(np.abs(first.committed.kappa))),
            self.manager.config.hard_max_abs_curvature + 1e-6,
        )

    def test_consistent_turn_frames_remain_fresh_without_reanchor(self) -> None:
        straight = [(float(x), 0.0) for x in range(1, 21)]
        offset = self._lateral_jump_path(2.0)
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=2.0, stamp_s=1.0).accepted
        )
        stopped = EgoPose(0.0, 0.0, 0.0, 0.0)
        nav = (25.0, 2.0)
        last = None
        for i in range(3):
            last = self.manager.update(
                offset,
                ego=stopped,
                target_speed_mps=2.0,
                stamp_s=4.0 + i,
                nav_target_map_xy=nav,
            )
        assert last is not None
        self.assertTrue(last.accepted, last.reason)
        self.assertNotEqual(last.reason, "accepted_reanchor")
        self.assertEqual(last.reanchor_pending_count, 0)
        assert last.committed is not None
        self.assertEqual(last.committed.stamp_s, 6.0)

    def test_legal_gentle_arc_is_not_rejected(self) -> None:
        """Ordinary gentle arcs while moving must still pass switch gates."""
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(straight, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0).accepted
        )
        gentle = [(float(x), 0.03 * x) for x in range(1, 21)]
        update = self.manager.update(
            gentle,
            ego=EgoPose(0.0, 0.0, 0.0, 3.0),
            target_speed_mps=3.0,
            stamp_s=1.5,
            nav_target_map_xy=(30.0, 0.0),
        )
        self.assertTrue(update.accepted, update.reason)
        self.assertEqual(update.reason, "accepted")

    def test_lateral_mode_flip_is_diagnostic_not_a_hard_reject(self) -> None:
        """Ego-frame lateral sign is ambiguous on S-curves, even when enabled."""
        mgr = VLAPathManager(
            PathManagerConfig(
                max_abs_curvature=0.30,
                max_switch_lateral_5m=10.0,
                max_switch_heading_5m_deg=180.0,
                enable_lateral_mode_flip=True,
            )
        )
        left = [(float(x), 0.15 * x) for x in range(1, 21)]
        right = [(float(x), -0.25 * x) for x in range(1, 21)]
        self.assertTrue(
            mgr.update(
                left, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0, nav_target_map_xy=(25.0, 2.0)
            ).accepted
        )
        flipped = mgr.update(
            right,
            ego=EgoPose(0.0, 0.0, 0.0, 3.0),
            target_speed_mps=3.0,
            stamp_s=1.5,
            nav_target_map_xy=(25.0, 2.0),
        )
        self.assertTrue(flipped.accepted, flipped.reason)
        self.assertEqual(flipped.reason, "accepted_soft_lateral_mode_flip")

    def test_s_curve_does_not_trip_disabled_lateral_mode_flip(self) -> None:
        """Default-off gate: legal S-curve lateral sign change is not rejected as mode flip."""
        left = [(float(x), 0.12 * x) for x in range(1, 21)]
        # Gentle right bias after left commit — if only mode_flip were active it would fire;
        # with gate off and jump under switch limits this should accept.
        right = [(float(x), -0.08 * x) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(
                left, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0, nav_target_map_xy=(25.0, 1.0)
            ).accepted
        )
        # Ensure default config has flip disabled
        self.assertFalse(self.manager.config.enable_lateral_mode_flip)
        nxt = self.manager.update(
            right,
            ego=EgoPose(0.0, 0.0, 0.0, 3.0),
            target_speed_mps=3.0,
            stamp_s=1.5,
            nav_target_map_xy=(25.0, -1.0),
        )
        # May still fail lateral/heading switch if jump large; must NOT be mode_flip
        if not nxt.accepted:
            self.assertNotEqual(nxt.reason, "lateral_mode_flip")

    def test_early_lane_change_heuristic_is_diagnostic_only(self) -> None:
        """A straight coarse target cannot be a hard lane oracle.

        Path is gentle near the ego (so lateral/heading switch and hard kappa
        do not fire first) but still has large lateral at the ~10 m mode probe.
        """
        straight = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(
            self.manager.update(
                straight, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0, nav_target_map_xy=(30.0, 0.0)
            ).accepted
        )
        # Near: ~0.25 m at 5 m; far: ~2.7 m at 10 m while nav stays ~0.3 m lateral.
        side = [
            (float(x), 0.05 * x if x <= 5 else 0.25 + 0.55 * (x - 5)) for x in range(1, 21)
        ]
        update = self.manager.update(
            side,
            ego=EgoPose(0.0, 0.0, 0.0, 3.0),
            target_speed_mps=3.0,
            stamp_s=1.5,
            nav_target_map_xy=(30.0, 0.3),
        )
        self.assertTrue(update.accepted, update.reason)
        self.assertEqual(update.reason, "accepted_soft_early_lane_change")

    def test_near_horizon_is_frozen_from_committed(self) -> None:
        """RTC-style: mid-near may hold; ego snap + far use latest; κ stays sane."""
        straight = [(float(x), 0.0) for x in range(1, 21)]
        first = self.manager.update(
            straight, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0, nav_target_map_xy=(30.0, 0.0)
        )
        self.assertTrue(first.accepted)
        assert first.committed is not None
        gentle = [(float(x), 0.04 * x) for x in range(1, 21)]
        second = self.manager.update(
            gentle,
            ego=EgoPose(0.0, 0.0, 0.0, 3.0),
            target_speed_mps=3.0,
            stamp_s=1.5,
            nav_target_map_xy=(30.0, 0.5),
        )
        self.assertTrue(second.accepted, second.reason)
        assert second.committed is not None
        # Ego-near snap: first 0.5 m follows latest (may move), far still updates.
        _xf0, yf0, _, _ = first.committed.sample(np.array([12.0]))
        _xf1, yf1, _, _ = second.committed.sample(np.array([12.0]))
        self.assertGreater(abs(float(yf1[0]) - float(yf0[0])), 0.05)
        # Committed densify must not invent hard-limit spikes on gentle updates.
        self.assertLess(float(np.max(np.abs(second.committed.kappa))), 1.0)

    def test_blend_falls_back_when_committed_kappa_spikes(self) -> None:
        """If freeze blend densify would hard-spike κ, use latest-only path."""
        mgr = VLAPathManager(
            PathManagerConfig(
                max_abs_curvature=0.30,
                hard_max_abs_curvature=1.0,
                near_freeze_min_m=2.5,
                far_blend_m=1.0,
                ego_snap_m=0.2,
                recheck_committed_curvature=True,
            )
        )
        a = [(float(x), 0.0) for x in range(1, 21)]
        self.assertTrue(mgr.update(a, ego=self.ego, target_speed_mps=3.0, stamp_s=1.0).accepted)
        # Large lateral jump that would kink if hard-frozen then re-PCHIP'd.
        b = [(float(x), 2.5 if x > 3 else 0.0) for x in range(1, 21)]
        u = mgr.update(b, ego=EgoPose(0, 0, 0, 3), target_speed_mps=3.0, stamp_s=1.5)
        # Either reject at switch gates or accept with non-spiking committed.
        if u.accepted and u.committed is not None:
            self.assertLessEqual(
                float(np.max(np.abs(u.committed.kappa))),
                mgr.config.hard_max_abs_curvature + 1e-6,
            )


if __name__ == "__main__":
    unittest.main()
