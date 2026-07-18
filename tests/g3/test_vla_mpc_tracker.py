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

    def test_stale_path_brakes_to_zero(self) -> None:
        path = spatial_path_from_xy(
            [(float(x), 0.0) for x in range(0, 41)],
            ego=EgoPose(0.0, 0.0, 0.0, 0.0),
            target_speed_mps=2.0,
            stamp_s=1.0,
        )
        assert path is not None
        tracker = ConstrainedVLAMPC(self.cfg)
        cmd = tracker.step(
            path,
            EgoPose(0.0, 0.0, 0.0, 1.0),
            now_s=1.0 + self.cfg.path_stale_hard_s,
        )
        self.assertEqual(cmd.target_speed_mps, 0.0)
        self.assertLess(cmd.accel_mps2, 0.0)

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
