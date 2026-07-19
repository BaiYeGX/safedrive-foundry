from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.path_manager import EgoPose, SpatialPath, spatial_path_from_xy  # noqa: E402
from driving_vla.runtime.vla_mpc_tracker import ConstrainedVLAMPC, VLAMPCConfig  # noqa: E402


def _plant(ego: EgoPose, steer: float, accel: float, *, dt: float, wheelbase: float) -> EgoPose:
    speed = max(0.0, ego.speed_mps + accel * dt)
    yaw = ego.yaw + speed / wheelbase * math.tan(steer) * dt
    return EgoPose(
        x=ego.x + speed * math.cos(ego.yaw) * dt,
        y=ego.y + speed * math.sin(ego.yaw) * dt,
        yaw=yaw,
        speed_mps=speed,
    )


class ConstrainedVLAMPCTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = VLAMPCConfig(
            control_dt_s=0.05,
            prediction_dt_s=0.10,
            horizon=20,
            wheelbase_m=2.7,
            max_steer_rad=0.55,
            max_steer_rate_rps=0.35,
            max_speed_mps=2.0,
            solver_deadline_ms=100.0,
        )

    def test_straight_path_converges_without_steer_jump(self) -> None:
        initial = EgoPose(0.0, 0.60, 0.08, 1.0)
        path = spatial_path_from_xy(
            [(float(x), 0.0) for x in range(0, 41)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=2.0,
            stamp_s=1.0,
        )
        assert path is not None
        tracker = ConstrainedVLAMPC(self.cfg)
        ego = initial
        previous = 0.0
        max_step = 0.0
        modes: set[str] = set()
        for _ in range(140):
            cmd = tracker.step(path, ego)
            modes.add(cmd.mode)
            max_step = max(max_step, abs(cmd.steer_rad - previous))
            previous = cmd.steer_rad
            ego = _plant(
                ego,
                cmd.steer_rad,
                cmd.accel_mps2,
                dt=self.cfg.control_dt_s,
                wheelbase=self.cfg.wheelbase_m,
            )
        self.assertIn("mpc", modes)
        self.assertLess(abs(ego.y), 0.18)
        self.assertLess(max_step, self.cfg.max_steer_rate_rps * self.cfg.control_dt_s + 1e-6)

    def test_freshness_thresholds_must_be_ordered(self) -> None:
        bad = VLAMPCConfig(
            path_stale_soft_s=2.0,
            path_stale_hard_s=1.0,
            path_stale_zero_s=5.0,
        )
        with self.assertRaisesRegex(ValueError, "soft <= hard <= zero"):
            ConstrainedVLAMPC(bad)

    def test_large_radius_arc_tracks_without_saturation(self) -> None:
        radius = 30.0
        theta = np.linspace(0.0, 1.2, 121)
        points = [(radius * math.sin(t), radius * (1.0 - math.cos(t))) for t in theta]
        path = spatial_path_from_xy(
            points,
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=2.0,
            stamp_s=1.0,
        )
        assert path is not None
        tracker = ConstrainedVLAMPC(self.cfg)
        ego = EgoPose(0.0, 0.0, 0.0, 1.0)
        max_steer = 0.0
        for _ in range(180):
            cmd = tracker.step(path, ego)
            max_steer = max(max_steer, abs(cmd.steer_rad))
            ego = _plant(
                ego,
                cmd.steer_rad,
                cmd.accel_mps2,
                dt=self.cfg.control_dt_s,
                wheelbase=self.cfg.wheelbase_m,
            )
        cte = abs(math.hypot(ego.x, ego.y - radius) - radius)
        self.assertLess(cte, 0.45)
        self.assertLess(max_steer, 0.30)

    def test_short_path_limits_speed_to_stopping_distance(self) -> None:
        path = spatial_path_from_xy(
            [(0.0, 0.0), (6.0, 0.0)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=15.0,
            stamp_s=1.0,
        )
        assert path is not None
        tracker = ConstrainedVLAMPC(self.cfg)
        cmd = tracker.step(path, EgoPose(0.0, 0.0, 0.0, 1.0), now_s=1.0)
        expected = math.sqrt(2.0 * self.cfg.max_brake_mps2 * 4.0)
        self.assertLessEqual(cmd.target_speed_mps, expected + 1e-6)
        self.assertAlmostEqual(cmd.horizon_speed_limit_mps, expected, places=5)

    def test_linearization_speed_stays_near_actual_not_cruise_cap(self) -> None:
        """Car at ~1.5 m/s must not linearize at path target 6 m/s."""
        path = spatial_path_from_xy(
            [(float(x), 0.15 * math.sin(x / 4.0)) for x in range(0, 41)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=6.0,
            stamp_s=1.0,
        )
        assert path is not None
        cfg = VLAMPCConfig(
            control_dt_s=0.05,
            prediction_dt_s=0.10,
            horizon=20,
            wheelbase_m=2.7,
            max_steer_rad=0.55,
            max_steer_rate_rps=0.35,
            max_speed_mps=15.0,
            min_linearization_speed_mps=0.60,
            linearization_speed_slack_mps=0.50,
            solver_deadline_ms=100.0,
        )
        tracker = ConstrainedVLAMPC(cfg)
        # Access linearization via reference curvature path: run step and ensure
        # feedforward steer magnitude stays moderate (over-linearization blows it up).
        ego = EgoPose(0.0, 0.0, 0.0, 1.5)
        steers = []
        for i in range(30):
            # Re-stamp path each few steps to mimic continuous replan.
            if i % 5 == 0:
                path = spatial_path_from_xy(
                    [(float(x), 0.15 * math.sin((x + 0.2 * i) / 4.0)) for x in range(0, 41)],
                    ego=EgoPose(ego.x, ego.y, ego.yaw, ego.speed_mps),
                    target_speed_mps=6.0,
                    stamp_s=1.0 + 0.05 * i,
                )
                assert path is not None
            cmd = tracker.step(path, ego, now_s=1.0 + 0.05 * i)
            steers.append(cmd.steer_rad)
            ego = _plant(
                ego,
                cmd.steer_rad,
                cmd.accel_mps2,
                dt=cfg.control_dt_s,
                wheelbase=cfg.wheelbase_m,
            )
            ego = EgoPose(ego.x, ego.y, ego.yaw, min(ego.speed_mps, 1.8))
        flips = sum(
            1
            for a, b in zip(steers, steers[1:])
            if a * b < 0 and abs(a) > 0.05 and abs(b) > 0.05
        )
        self.assertLess(flips, 12, f"too many steer flips under replan: {flips}")
        self.assertLess(max(abs(s) for s in steers), 0.45)

    def test_stale_path_crawls_at_hard_then_zero_at_zero_s(self) -> None:
        path = spatial_path_from_xy(
            [(float(x), 0.0) for x in range(0, 41)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=2.0,
            stamp_s=1.0,
        )
        assert path is not None
        tracker = ConstrainedVLAMPC(self.cfg)
        # At hard threshold: crawl, not hard park (intersection reject storm).
        cmd_hard = tracker.step(
            path,
            EgoPose(0.0, 0.0, 0.0, 1.0),
            now_s=1.0 + self.cfg.path_stale_hard_s,
        )
        self.assertGreater(cmd_hard.freshness_speed_limit_mps, 0.5)
        self.assertIn(cmd_hard.freshness_regime, {"crawl", "soft_ramp"})
        self.assertGreater(cmd_hard.target_speed_mps, 0.0)
        # Only at zero_s: full stop.
        cmd_zero = tracker.step(
            path,
            EgoPose(0.0, 0.0, 0.0, 1.0),
            now_s=1.0 + self.cfg.path_stale_zero_s,
        )
        self.assertEqual(cmd_zero.freshness_speed_limit_mps, 0.0)
        self.assertEqual(cmd_zero.freshness_regime, "hard_stop")
        self.assertEqual(cmd_zero.target_speed_mps, 0.0)
        self.assertLess(cmd_zero.accel_mps2, 0.0)

    def test_fresh_path_stamp_clears_stale_limit(self) -> None:
        """Reanchor-equivalent: new stamp restores full freshness immediately."""
        path_old = spatial_path_from_xy(
            [(float(x), 0.0) for x in range(0, 41)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=5.0,
            stamp_s=1.0,
        )
        path_new = spatial_path_from_xy(
            [(float(x), 0.0) for x in range(0, 41)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=5.0,
            stamp_s=10.0,
        )
        assert path_old is not None and path_new is not None
        tracker = ConstrainedVLAMPC(self.cfg)
        stale = tracker.step(
            path_old, EgoPose(0.0, 0.0, 0.0, 1.0), now_s=1.0 + self.cfg.path_stale_zero_s
        )
        self.assertEqual(stale.freshness_regime, "hard_stop")
        fresh = tracker.step(path_new, EgoPose(0.0, 0.0, 0.0, 0.1), now_s=10.0)
        self.assertEqual(fresh.freshness_regime, "fresh")
        # Cap is cfg.max_speed_mps (2.0 in setUp), not path target alone.
        self.assertGreaterEqual(fresh.freshness_speed_limit_mps, self.cfg.max_speed_mps - 1e-6)
        self.assertGreater(fresh.target_speed_mps, 1.5)

    def test_single_curvature_spike_does_not_collapse_straight_speed(self) -> None:
        s = np.linspace(0.0, 40.0, 201)
        kappa = np.zeros_like(s)
        kappa[50] = 0.30
        path = SpatialPath(
            s=s,
            x=s.copy(),
            y=np.zeros_like(s),
            yaw=np.zeros_like(s),
            kappa=kappa,
            target_speed_mps=10.0,
            stamp_s=1.0,
        )
        cfg = VLAMPCConfig(
            **{
                **self.cfg.__dict__,
                "max_speed_mps": 10.0,
                "max_lateral_accel_mps2": 1.5,
            }
        )
        cmd = ConstrainedVLAMPC(cfg).step(path, EgoPose(0.0, 0.0, 0.0, 1.0), now_s=1.0)
        self.assertGreater(cmd.curve_speed_limit_mps, 9.9)


if __name__ == "__main__":
    unittest.main()
