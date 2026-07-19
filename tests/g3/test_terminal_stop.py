"""Terminal-stop classification unit tests (no CARLA)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.terminal_stop import (  # noqa: E402
    TERMINAL_EXPECTED_RED,
    TERMINAL_GREEN_STUCK,
    TERMINAL_MOVING,
    TERMINAL_UNEXPLAINED,
    classify_terminal_stop,
)


def _const_speeds(n: int, v: float) -> list[float]:
    return [float(v)] * n


def _tl(state: str, *, is_at: bool = True, light_id: int = 7) -> dict:
    return {
        "oracle_only": True,
        "is_at_traffic_light": is_at,
        "traffic_light_state": state,
        "traffic_light_id": light_id,
        "ok": True,
    }


class TerminalStopClassificationTest(unittest.TestCase):
    def test_sustained_motion_is_moving(self) -> None:
        # 20s @ 0.05 → 400 samples, all moving.
        speeds = _const_speeds(400, 3.0)
        out = classify_terminal_stop(
            speed_samples_mps=speeds,
            traffic_samples=[_tl("Green")] * 400,
            sim_dt_s=0.05,
            window_s=20.0,
        )
        self.assertEqual(out["terminal_stop_classification"], TERMINAL_MOVING)
        self.assertTrue(out["tail_acceptance_ok"])
        self.assertGreaterEqual(out["moving_fraction"], 0.50)

    def test_stationary_at_red_is_expected_legal_stop(self) -> None:
        speeds = _const_speeds(400, 0.0)
        tls = [_tl("Red")] * 400
        out = classify_terminal_stop(
            speed_samples_mps=speeds,
            traffic_samples=tls,
            sim_dt_s=0.05,
            window_s=20.0,
        )
        self.assertEqual(out["terminal_stop_classification"], TERMINAL_EXPECTED_RED)
        self.assertTrue(out["tail_acceptance_ok"])
        self.assertLess(out["moving_fraction"], 0.50)

    def test_stationary_without_traffic_light_is_unexplained(self) -> None:
        speeds = _const_speeds(400, 0.05)
        tls = [_tl("Unknown", is_at=False)] * 400
        out = classify_terminal_stop(
            speed_samples_mps=speeds,
            traffic_samples=tls,
            sim_dt_s=0.05,
            window_s=20.0,
        )
        self.assertEqual(out["terminal_stop_classification"], TERMINAL_UNEXPLAINED)
        self.assertFalse(out["tail_acceptance_ok"])

    def test_green_beyond_grace_while_stopped_is_green_stuck(self) -> None:
        # 20s stationary, all green at light → >5s grace → stuck.
        speeds = _const_speeds(400, 0.0)
        tls = [_tl("Green")] * 400
        out = classify_terminal_stop(
            speed_samples_mps=speeds,
            traffic_samples=tls,
            sim_dt_s=0.05,
            window_s=20.0,
            green_grace_s=5.0,
        )
        self.assertEqual(out["terminal_stop_classification"], TERMINAL_GREEN_STUCK)
        self.assertFalse(out["tail_acceptance_ok"])
        self.assertGreater(out.get("green_while_stopped_s", 0.0), 5.0)

    def test_short_green_after_red_not_yet_green_stuck(self) -> None:
        # Most of tail is red stop; last 2s green (within grace).
        dt = 0.05
        n = 400
        speeds = _const_speeds(n, 0.0)
        tls = [_tl("Red")] * n
        green_n = int(2.0 / dt)
        for i in range(green_n):
            tls[-(i + 1)] = _tl("Green")
        out = classify_terminal_stop(
            speed_samples_mps=speeds,
            traffic_samples=tls,
            sim_dt_s=dt,
            window_s=20.0,
            green_grace_s=5.0,
        )
        # Green streak 2s < 5s grace → not green_stuck; terminal is Green so
        # not expected_red either → unexplained (or still not pass on green).
        self.assertEqual(out["terminal_stop_classification"], TERMINAL_UNEXPLAINED)
        self.assertFalse(out["tail_acceptance_ok"])

    def test_does_not_feed_control_keys(self) -> None:
        out = classify_terminal_stop(
            speed_samples_mps=[0.0] * 10,
            traffic_samples=[_tl("Red")] * 10,
            sim_dt_s=0.05,
            window_s=0.5,
        )
        self.assertTrue(out.get("oracle_only"))
        for banned in ("steer", "throttle", "brake", "target_speed_mps"):
            self.assertNotIn(banned, out)


if __name__ == "__main__":
    unittest.main()
