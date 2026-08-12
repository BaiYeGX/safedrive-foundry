from __future__ import annotations

import math
import unittest

from safedrive_foundry.driving_vla.evaluation.long_horizon_observer import (
    InteractionBehavior,
    LongHorizonObserver,
)
from safedrive_foundry.driving_vla.model.navigation_contract import (
    RouteManeuver,
    build_route_context,
)


def _base_ticks(context, *, count: int = 321):
    ticks = []
    for index in range(count):
        ticks.append(
            {
                "simulation_time_s": index * 0.05,
                "ego_x": index * 0.05,
                "ego_y": 0.0,
                "ego_yaw_rad": 0.0,
                "ego_v": 3.0,
                "road_id": 1,
                "lane_id": 1,
                "route_hash": context.route_hash,
                "topology_hash": context.topology_hash,
                "route_maneuver": context.maneuver.value,
                "path_tracking_error_m": 0.05,
                "replan_id": f"replan-{index // 40}",
                "mpc_status": "solved",
                "candidate1_available": False,
            }
        )
    return ticks


class LongHorizonObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.route = [(float(index), 0.0) for index in range(50)]
        self.context = build_route_context(
            self.route, maneuver=RouteManeuver.FOLLOW_STRAIGHT
        )

    def _run(self, behavior, ticks, **kwargs):
        observer = LongHorizonObserver(
            case_id=f"case-{behavior}",
            route_context=self.context,
            behavior=behavior,
            **kwargs,
        )
        for tick in ticks:
            observer.observe(tick)
        return observer.finalize(
            final_actor_lon_m=ticks[-1].get("actor_lon_m")
        )

    def test_clear_complete_without_second_candidate(self) -> None:
        report = self._run(InteractionBehavior.CLEAR, _base_ticks(self.context))
        self.assertTrue(report.completed, report.reason_codes)

    def test_clear_uses_executable_path_tracking_not_map_centerline_offset(self) -> None:
        ticks = _base_ticks(self.context)
        for tick in ticks:
            tick["ego_y"] = 0.75
            tick["path_tracking_error_m"] = 0.05
        report = self._run(InteractionBehavior.CLEAR, ticks)
        self.assertTrue(report.completed, report.reason_codes)

    def test_clear_rejects_repeated_executable_path_tracking_flips(self) -> None:
        ticks = _base_ticks(self.context)
        for index, tick in enumerate(ticks):
            tick["path_tracking_error_m"] = 0.35 if index % 2 else -0.35
        report = self._run(InteractionBehavior.CLEAR, ticks)
        self.assertFalse(report.completed)
        self.assertIn(
            "INTERACTION:CLEAR_SNAKE_OR_LATERAL_EXCURSION",
            report.reason_codes,
        )

    def test_follow_stop_resume_state_machine(self) -> None:
        ticks = _base_ticks(self.context)
        for index, tick in enumerate(ticks):
            if 100 <= index < 140:
                tick["ego_v"] = 0.0
            tick["actor_clearance_m"] = 4.0
        report = self._run(InteractionBehavior.FOLLOW_STOP, ticks)
        self.assertTrue(report.completed, report.reason_codes)
        self.assertIn("STOPPED", report.phase_history)

    def test_cut_in_away_and_rejoin_state_machine(self) -> None:
        ticks = _base_ticks(self.context)
        for index, tick in enumerate(ticks):
            tick["ego_y"] = 0.9 * math.sin(math.pi * index / (len(ticks) - 1)) ** 2
        report = self._run(
            InteractionBehavior.CUT_IN_AVOID,
            ticks,
            conflict_side="left",
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertIn("AVOID", report.phase_history)
        self.assertIn("REJOIN", report.phase_history)

    def test_cut_in_temporal_fallback_requires_executed_temporal_candidate(self) -> None:
        ticks = _base_ticks(self.context)
        for index, tick in enumerate(ticks):
            tick["alternative_kind"] = "TEMPORAL_YIELD"
            tick["conflict_active"] = index < 60
            if index < 60:
                tick["ego_v"] = 0.0
        report = self._run(
            InteractionBehavior.CUT_IN_AVOID,
            ticks,
            conflict_side="left",
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertTrue(report.interaction_completion["stopped"])
        self.assertTrue(report.interaction_completion["resumed"])

    def test_yield_wait_resume_state_machine(self) -> None:
        ticks = _base_ticks(self.context)
        for index, tick in enumerate(ticks):
            tick["conflict_active"] = index < 150
            if 80 <= index < 155:
                tick["ego_v"] = 0.0
                tick["ego_x"] = min(tick["ego_x"], 5.0)
        report = self._run(
            InteractionBehavior.YIELD_WAIT,
            ticks,
            conflict_point_s_m=7.0,
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertIn("WAIT", report.phase_history)

    def test_overtake_pass_rejoin_state_machine(self) -> None:
        ticks = _base_ticks(self.context)
        for index, tick in enumerate(ticks):
            tick["ego_y"] = 3.5 * math.sin(math.pi * index / (len(ticks) - 1)) ** 2
            tick["actor_lon_m"] = 6.0 - index * 0.04
            tick["candidate1_available"] = True
        report = self._run(InteractionBehavior.OVERTAKE_REJOIN, ticks)
        self.assertTrue(report.completed, report.reason_codes)
        self.assertIn("PASS", report.phase_history)
        self.assertIn("REJOIN", report.phase_history)


if __name__ == "__main__":
    unittest.main()
