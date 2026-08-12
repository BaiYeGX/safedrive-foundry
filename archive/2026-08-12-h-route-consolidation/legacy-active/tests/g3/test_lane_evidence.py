"""Lane invasion episodes + oracle isolation tests (no CARLA)."""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.lane_evidence import (  # noqa: E402
    LaneInvasionEpisodeBook,
    lane_center_error_stats,
    multi_deadband_sign_flips,
    observe_lane_oracle,
    steer_derivative_metrics,
)


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Rot:
    def __init__(self, yaw: float) -> None:
        self.yaw = yaw


class _Tf:
    def __init__(self, x: float, y: float, yaw_deg: float) -> None:
        self.location = _Loc(x, y)
        self.rotation = _Rot(yaw_deg)


class _Wp:
    def __init__(
        self,
        *,
        road_id: int,
        lane_id: int,
        is_junction: bool,
        lane_width: float,
        x: float,
        y: float,
        yaw_deg: float,
    ) -> None:
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self.lane_width = lane_width
        self.transform = _Tf(x, y, yaw_deg)


class _Map:
    def __init__(self, wp: _Wp | None) -> None:
        self._wp = wp

    def get_waypoint(self, location, project_to_road=True, lane_type=None):  # noqa: ANN001
        return self._wp


class LaneOracleTest(unittest.TestCase):
    def test_oracle_signed_lateral_and_oracle_only_flag(self) -> None:
        # Lane center at origin, heading +x (yaw=0). Ego at y=+0.4 → left positive.
        wp = _Wp(
            road_id=12,
            lane_id=-1,
            is_junction=False,
            lane_width=3.5,
            x=0.0,
            y=0.0,
            yaw_deg=0.0,
        )
        carla_mod = types.SimpleNamespace(LaneType=types.SimpleNamespace(Driving=2))
        obs = observe_lane_oracle(
            _Map(wp), _Loc(1.0, 0.4), ego_yaw=0.0, carla_module=carla_mod
        )
        self.assertTrue(obs["oracle_only"])
        self.assertTrue(obs["ok"])
        self.assertEqual(obs["road_id"], 12)
        self.assertEqual(obs["lane_id"], -1)
        self.assertAlmostEqual(obs["lane_center_error_m"], 0.4, places=5)
        self.assertFalse(obs["is_junction"])

    def test_oracle_not_consumed_by_control_namespace(self) -> None:
        """Guard: oracle keys must not be part of control decision dicts."""
        # Simulates a control payload builder that only takes explicit keys.
        control_keys = {"steer", "throttle", "brake", "target_speed_mps", "mode"}
        wp = _Wp(
            road_id=1,
            lane_id=1,
            is_junction=True,
            lane_width=3.0,
            x=0.0,
            y=0.0,
            yaw_deg=90.0,
        )
        carla_mod = types.SimpleNamespace(LaneType=types.SimpleNamespace(Driving=2))
        oracle = observe_lane_oracle(
            _Map(wp), _Loc(0.2, 0.0), ego_yaw=math.pi / 2, carla_module=carla_mod
        )
        leaked = control_keys.intersection(oracle.keys())
        self.assertEqual(leaked, set())
        # Control builder must not pull oracle fields by design of this API.
        control = {
            "steer": 0.1,
            "throttle": 0.2,
            "brake": 0.0,
            "target_speed_mps": 5.0,
            "mode": "mpc",
        }
        for k in oracle:
            self.assertNotIn(k, control)
        self.assertTrue(oracle["oracle_only"])

    def test_lane_center_error_stats_split_junction(self) -> None:
        samples = [
            {"ok": True, "is_junction": False, "lane_center_error_m": 0.2},
            {"ok": True, "is_junction": False, "lane_center_error_m": -0.4},
            {"ok": True, "is_junction": True, "lane_center_error_m": 1.0},
        ]
        st = lane_center_error_stats(samples)
        self.assertTrue(st["oracle_only"])
        self.assertEqual(st["all"]["n"], 3)
        self.assertEqual(st["junction"]["n"], 1)
        self.assertEqual(st["non_junction"]["n"], 2)
        self.assertAlmostEqual(st["junction"]["max"], 1.0)


class LaneInvasionEpisodeTest(unittest.TestCase):
    def test_merges_within_gap_and_splits_after(self) -> None:
        book = LaneInvasionEpisodeBook(gap_s=0.5)
        book.set_sim_s(1.0)
        book.update_control_snapshot({"steer": 0.1, "throttle": 0.3, "brake": 0.0})
        book.on_invasion(
            sim_s=1.0,
            carla_frame=10,
            ego_pose={"x": 1.0, "y": 2.0, "yaw": 0.1},
            speed_mps=4.0,
            road_id=3,
            lane_id=-1,
            committed_source_id="vla_committed",
        )
        book.on_invasion(sim_s=1.2, carla_frame=14, road_id=3, lane_id=-1)
        book.on_invasion(
            sim_s=3.0,
            carla_frame=50,
            ego_pose={"x": 5.0, "y": 2.0, "yaw": 0.0},
            speed_mps=3.0,
            road_id=4,
            lane_id=1,
        )
        summary = book.summary()
        self.assertEqual(summary["raw_event_count"], 3)
        self.assertEqual(summary["episode_count"], 2)
        first = summary["episodes"][0]
        self.assertEqual(first["raw_event_count"], 2)
        self.assertEqual(first["road_id"], 3)
        self.assertEqual(first["committed_source_id"], "vla_committed")
        self.assertEqual(first["pre_invasion_control"]["throttle"], 0.3)


class ControlMetricsTest(unittest.TestCase):
    def test_multi_deadband_sign_flips(self) -> None:
        # Clear flips at high amplitude; small noise within 0.05 should not count.
        vals = [0.1, -0.1, 0.1, 0.02, -0.02, 0.2, -0.2]
        flips = multi_deadband_sign_flips(vals, deadbands=(0.01, 0.03, 0.05))
        self.assertGreaterEqual(flips["deadband_0.01"], flips["deadband_0.05"])
        self.assertGreater(flips["deadband_0.01"], 0)

    def test_steer_derivative_metrics(self) -> None:
        # Linear ramp: constant rate, zero accel.
        steers = [0.0, 0.1, 0.2, 0.3, 0.4]
        m = steer_derivative_metrics(steers, dt_s=0.05)
        self.assertAlmostEqual(m["delta_steer_abs"]["max"], 0.1, places=5)
        self.assertAlmostEqual(m["steer_rate_abs"]["max"], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
