from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.maneuver_completion import (
    evaluate_cut_in_completion,
    evaluate_follow_stop_completion,
    evaluate_overtake_completion,
    evaluate_route_maneuver_completion,
    evaluate_traffic_control_completion,
    evaluate_yield_wait_completion,
)
from driving_vla.model.k2_v3_codec import lane_blend_path
from driving_vla.model.navigation_contract import (
    LaneAccessV3,
    RouteManeuver,
    TargetLaneSide,
    build_route_context,
)


def _ticks(*, stop: bool = False, collision: bool = False):
    values = []
    for index in range(61):
        x = index * 0.2
        # Smooth excursion and return.
        y = 0.9 * __import__("math").sin(__import__("math").pi * index / 60.0) ** 2
        values.append(
            {
                "ego_x": x,
                "ego_y": y,
                "ego_v": 0.0 if stop and index >= 50 else 4.0,
                "collision": collision,
                "offroad": False,
            }
        )
    return values


class ManeuverCompletionTest(unittest.TestCase):
    def test_pass_rejoin_and_continue_is_complete(self) -> None:
        report = evaluate_overtake_completion(
            family="obstruction",
            route_xy=[(0.0, 0.0), (20.0, 0.0)],
            ticks=_ticks(),
            final_actor_lon_m=-3.0,
        )
        self.assertTrue(report.completed)
        self.assertGreater(report.peak_cross_track_m, 0.5)
        self.assertLess(report.final_cross_track_m, 0.1)

    def test_stop_behind_actor_is_not_success(self) -> None:
        report = evaluate_overtake_completion(
            family="obstruction",
            route_xy=[(0.0, 0.0), (20.0, 0.0)],
            ticks=_ticks(stop=True),
            final_actor_lon_m=3.0,
        )
        self.assertFalse(report.completed)
        self.assertIn("DID_NOT_PASS_ACTOR", report.reason_codes)
        self.assertIn("STOPPED_OR_CRAWLING", report.reason_codes)

    def test_collision_is_never_success(self) -> None:
        report = evaluate_overtake_completion(
            family="narrow_corridor",
            route_xy=[(0.0, 0.0), (20.0, 0.0)],
            ticks=_ticks(collision=True),
            final_actor_lon_m=-3.0,
        )
        self.assertFalse(report.completed)
        self.assertIn("COLLISION", report.reason_codes)

    def test_authorized_lane_transition_is_not_offroad(self) -> None:
        ticks = _ticks()
        ticks[20]["offroad"] = True
        ticks[20]["authorized_lane_crossing"] = True
        report = evaluate_overtake_completion(
            family="obstruction",
            route_xy=[(0.0, 0.0), (20.0, 0.0)],
            ticks=ticks,
            final_actor_lon_m=-3.0,
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertFalse(report.offroad)

    def test_yield_family_is_not_forced_to_overtake(self) -> None:
        report = evaluate_overtake_completion(
            family="crossing",
            route_xy=[(0.0, 0.0), (20.0, 0.0)],
            ticks=_ticks(stop=True),
            final_actor_lon_m=3.0,
        )
        self.assertFalse(report.applicable)
        self.assertEqual(report.reason_codes, ("NOT_OVERTAKE_FAMILY",))

    def test_left_turn_enters_and_exits_expected_junction(self) -> None:
        arc = [
            (
                10.0 * __import__("math").sin(index * __import__("math").pi / 36.0),
                -10.0 * (1.0 - __import__("math").cos(index * __import__("math").pi / 36.0)),
            )
            for index in range(19)
        ]
        route = arc + [(arc[-1][0], arc[-1][1] - index) for index in range(1, 16)]
        context = build_route_context(
            route,
            maneuver=RouteManeuver.TURN_LEFT,
            origin_road_id=1,
            origin_lane_id=1,
            target_road_id=2,
            target_lane_id=1,
            junction_flags=[False] * 3 + [True] * 12 + [False] * 4,
        )
        ticks = []
        for index, (x, y) in enumerate(route):
            ticks.append(
                {
                    "ego_x": x,
                    "ego_y": y,
                    "ego_yaw_rad": (
                        -min(index, 18) * __import__("math").pi / 36.0
                    ),
                    "ego_v": 3.0,
                    "is_junction": 3 <= index < 15,
                    "road_id": 1 if index < 15 else 2,
                    "lane_id": 1,
                    "route_hash": context.route_hash,
                    "topology_hash": context.topology_hash,
                    "route_maneuver": context.maneuver.value,
                }
            )
        report = evaluate_route_maneuver_completion(
            route_context=context,
            ticks=ticks,
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertGreater(report.heading_change_deg, 80.0)

    def test_right_turn_wrong_exit_fails(self) -> None:
        route = [
            (
                10.0 * __import__("math").sin(index * __import__("math").pi / 36.0),
                10.0 * (1.0 - __import__("math").cos(index * __import__("math").pi / 36.0)),
            )
            for index in range(19)
        ]
        context = build_route_context(
            route,
            maneuver=RouteManeuver.TURN_RIGHT,
            target_road_id=3,
            target_lane_id=-1,
            junction_flags=[False] * 3 + [True] * 12 + [False] * 4,
        )
        ticks = [
            {
                "ego_x": x,
                "ego_y": y,
                "ego_yaw_rad": index * __import__("math").pi / 36.0,
                "ego_v": 3.0,
                "is_junction": 3 <= index < 15,
                "road_id": 99 if index >= 15 else 1,
                "lane_id": -1,
            }
            for index, (x, y) in enumerate(route)
        ]
        report = evaluate_route_maneuver_completion(
            route_context=context,
            ticks=ticks,
        )
        self.assertFalse(report.completed)
        self.assertIn("WRONG_EXIT_ROAD", report.reason_codes)

    def test_route_change_holds_target_lane_and_authorized_crossing(self) -> None:
        origin = [(float(index), 0.0) for index in range(40)]
        target = [(float(index), 3.5) for index in range(40)]
        route = lane_blend_path(origin, target, rejoin=False)
        access = LaneAccessV3(
            side=TargetLaneSide.LEFT,
            exists=True,
            driving=True,
            same_direction=True,
            lane_change_allowed=True,
            currently_clear=True,
            road_id=1,
            lane_id=2,
            lane_width_m=3.5,
            centerline_xy=tuple(target),
        )
        context = build_route_context(
            route,
            maneuver=RouteManeuver.ROUTE_CHANGE_LEFT,
            target_road_id=1,
            target_lane_id=2,
            left_lane=access,
        )
        ticks = []
        for index, (x, y) in enumerate(route):
            ticks.append(
                {
                    "ego_x": x,
                    "ego_y": y,
                    "ego_yaw_rad": 0.0,
                    "ego_v": 3.0,
                    "road_id": 1,
                    "lane_id": 2 if index >= 20 else 1,
                    "lane_invasion": index == 18,
                    "authorized_lane_crossing": index == 18,
                }
            )
        report = evaluate_route_maneuver_completion(
            route_context=context,
            ticks=ticks,
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertGreaterEqual(report.target_lane_hold_m, 8.0)

    def test_follow_stop_and_resume(self) -> None:
        ticks = []
        for index in range(80):
            speed = 4.0
            if 25 <= index < 50:
                speed = 0.0
            ticks.append(
                {
                    "ego_x": index * 0.15,
                    "ego_y": 0.0,
                    "ego_v": speed,
                    "actor_clearance_m": 4.0,
                }
            )
        report = evaluate_follow_stop_completion(
            route_xy=[(0.0, 0.0), (30.0, 0.0)],
            ticks=ticks,
        )
        self.assertTrue(report.completed, report.reason_codes)
        self.assertTrue(report.stopped)
        self.assertTrue(report.resumed)

    def test_yield_wait_then_resume(self) -> None:
        ticks = []
        for index in range(80):
            conflict = index < 45
            speed = 0.0 if 20 <= index < 48 else 3.0
            ticks.append(
                {
                    "ego_x": min(index * 0.15, 5.5) if conflict else 5.5 + (index - 45) * 0.15,
                    "ego_y": 0.0,
                    "ego_v": speed,
                    "conflict_active": conflict,
                }
            )
        report = evaluate_yield_wait_completion(
            route_xy=[(0.0, 0.0), (30.0, 0.0)],
            ticks=ticks,
            conflict_point_s_m=7.0,
        )
        self.assertTrue(report.completed, report.reason_codes)

    def test_cut_in_moves_away_and_rejoins(self) -> None:
        ticks = []
        for index in range(61):
            # Route points toward +X.  In CARLA, +Y is physical right, hence
            # this is an avoidance excursion away from a left-side conflict.
            y = 0.9 * __import__("math").sin(
                __import__("math").pi * index / 60.0
            ) ** 2
            ticks.append(
                {
                    "ego_x": index * 0.2,
                    "ego_y": y,
                    "ego_v": 4.0,
                }
            )
        report = evaluate_cut_in_completion(
            route_xy=[(0.0, 0.0), (20.0, 0.0)],
            ticks=ticks,
            conflict_side="left",
        )
        self.assertTrue(report.completed, report.reason_codes)

    def test_red_stop_green_resume_keeps_left_turn_intent(self) -> None:
        context = build_route_context(
            [(0.0, 0.0), (30.0, 0.0)],
            maneuver=RouteManeuver.TURN_LEFT,
        )
        ticks = []
        for index in range(40):
            red = index < 25
            stopped = 10 <= index < 25
            ticks.append(
                {
                    "traffic_signal_state": "RED" if red else "GREEN",
                    "stop_line_distance_m": max(0.5, 5.0 - index * 0.2),
                    "ego_v": 0.0 if stopped else (2.0 if not red else 1.0),
                    "route_hash": context.route_hash,
                    "topology_hash": context.topology_hash,
                    "route_maneuver": context.maneuver.value,
                }
            )
        report = evaluate_traffic_control_completion(ticks=ticks)
        self.assertTrue(report.completed, report.reason_codes)
        self.assertTrue(report.route_intent_preserved)


if __name__ == "__main__":
    unittest.main()
