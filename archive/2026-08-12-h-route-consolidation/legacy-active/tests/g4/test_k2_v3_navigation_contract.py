from __future__ import annotations

import math
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.k2_v3_codec import (  # noqa: E402
    build_k2_v3_bundle,
    lane_blend_path,
)
from driving_vla.model.k2_v3_codec import (  # noqa: E402
    _native_executable_path,
    _trajectory_for_path,
)
from driving_vla.model.frenet_codec import path_max_abs_curvature  # noqa: E402
from driving_vla.model.k2_v3_guard import (  # noqa: E402
    GUARD_OK,
    K2V3SelectionError,
    _route_direction_reason,
    attach_k2_v3_guard,
    select_k2_v3,
)
from driving_vla.model.k2_v3_types import (  # noqa: E402
    AlternativeKind,
    K2V3Config,
    ManeuverPhase,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    LaneAccessV3,
    RouteManeuver,
    TargetLaneSide,
    TrafficSignalState,
    build_route_context,
    classify_route_maneuver,
    route_heading_change_rad,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    RegistryError,
    RouteSpec,
    _route_from_mapping,
)
from driving_vla.evaluation.route_authoring_v3 import (  # noqa: E402
    author_route_from_waypoint,
)
from driving_vla.runtime.navigation_topology import (  # noqa: E402
    observe_route_context_v3,
    observe_traffic_control_v3,
)
from driving_vla.runtime.k2_execution import (  # noqa: E402
    select_k2_semantic_v3,
    selection_event_fields,
)
from scripts.r2_v3_load_map import _matches as authoring_map_matches  # noqa: E402


def _line(y: float = 0.0, n: int = 40):
    return tuple((float(index), float(y)) for index in range(n))


def _lane(side: TargetLaneSide, y: float) -> LaneAccessV3:
    return LaneAccessV3(
        side=side,
        exists=True,
        driving=True,
        same_direction=True,
        lane_change_allowed=True,
        currently_clear=True,
        road_id=1,
        lane_id=2 if side is TargetLaneSide.LEFT else -2,
        lane_width_m=3.5,
        centerline_xy=_line(y),
        marking_type="Broken",
    )


class _MockWaypoint:
    def __init__(self, x, y, yaw, index, *, junction=False):
        self.road_id = 1 if index < 5 else 2
        self.lane_id = 1
        self.s = float(index * 2)
        self.is_junction = junction
        self.junction_id = 7 if junction else 0
        self.lane_type = "Driving"
        self.lane_width = 3.5
        self.lane_change = "None"
        self.left_lane_marking = SimpleNamespace(type="Solid")
        self.right_lane_marking = SimpleNamespace(type="Solid")
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
            rotation=SimpleNamespace(yaw=float(yaw)),
        )
        self._next = []

    def next(self, _step):
        return list(self._next)

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None


def _mock_left_junction():
    approach = [_MockWaypoint(index * 2, 0, 0, index) for index in range(5)]
    for first, second in zip(approach, approach[1:]):
        first._next = [second]
    left = [
        _MockWaypoint(8, -(index + 1) * 2, -90, index + 5, junction=index < 3)
        for index in range(24)
    ]
    straight = _MockWaypoint(10, 0, 0, 100, junction=True)
    approach[-1]._next = [straight, left[0]]
    for first, second in zip(left, left[1:]):
        first._next = [second]
    return approach[0]


class NavigationContractV3Test(unittest.TestCase):
    def test_native_curve_near_field_is_lateral_accel_executable(self) -> None:
        # Regression fixture from the first Town03 V4 pilot rejection: retain
        # native XY while continuity smoothing and retiming make exact T10
        # samples executable under the Guard's pointwise v²*kappa check.
        path = [
            (-145.463043, -86.765541),
            (-145.414612, -94.687172),
            (-144.413788, -102.312836),
            (-142.128708, -109.656609),
            (-138.626144, -116.503838),
            (-134.008484, -122.654419),
            (-128.407440, -127.907219),
            (-121.960564, -132.070084),
            (-114.871719, -135.009476),
            (-107.370705, -136.630112),
            (-99.636345, -136.906052),
            (-92.683647, -135.924286),
            (-89.012276, -130.633057),
            (-88.700684, -123.084892),
            (-88.678864, -115.084923),
            (-88.657043, -107.084953),
            (-88.635231, -99.084984),
            (-88.613411, -91.085014),
            (-88.591591, -83.085045),
            (-88.569778, -75.085068),
        ]
        config = K2V3Config()
        executable = _native_executable_path(
            path,
            ego_v=5.85,
            target_speed_mps=8.0,
            config=config,
        )
        rows = _trajectory_for_path(
            executable,
            ego_v=5.85,
            target_speed_mps=8.0,
            config=config,
        )
        self.assertTrue(rows)
        self.assertLessEqual(
            max(float(row[3]) ** 2 * abs(float(row[5])) for row in rows),
            config.max_lateral_accel_mps2 + 1.0e-6,
        )

    def test_authoring_map_reload_match_is_suffix_safe(self) -> None:
        self.assertTrue(authoring_map_matches("Carla/Maps/Town03", "Town03"))
        self.assertTrue(authoring_map_matches("Carla/Maps/Town12/Town12", "Town12"))
        self.assertFalse(authoring_map_matches("Carla/Maps/Town03", "Town3"))

    def test_current_traffic_light_state_and_trigger_distance_are_observable(self) -> None:
        light = SimpleNamespace(
            get_state=lambda: SimpleNamespace(name="Red"),
            get_transform=lambda: SimpleNamespace(
                location=SimpleNamespace(x=10.0, y=0.0),
                rotation=SimpleNamespace(yaw=0.0),
            ),
            trigger_volume=SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0),
                extent=SimpleNamespace(x=2.0, y=1.0),
            ),
        )
        ego = SimpleNamespace(
            get_traffic_light=lambda: light,
            get_location=lambda: SimpleNamespace(x=3.0, y=0.0),
        )
        state, distance = observe_traffic_control_v3(ego)
        self.assertEqual(state, TrafficSignalState.RED)
        self.assertAlmostEqual(distance, 5.0)

    def test_route_relevant_red_light_is_visible_before_influence_volume(self) -> None:
        stop = SimpleNamespace(
            road_id=4,
            lane_id=-1,
            transform=SimpleNamespace(
                location=SimpleNamespace(x=12.0, y=0.0)
            ),
        )
        light = SimpleNamespace(
            id=9,
            get_state=lambda: SimpleNamespace(name="Red"),
            get_stop_waypoints=lambda: [stop],
        )
        actors = SimpleNamespace(filter=lambda _pattern: [light])
        world_map = SimpleNamespace(
            get_waypoint=lambda _location: SimpleNamespace(
                road_id=4,
                lane_id=-1,
            )
        )
        world = SimpleNamespace(
            get_map=lambda: world_map,
            get_actors=lambda: actors,
        )
        ego = SimpleNamespace(
            get_traffic_light=lambda: None,
            get_world=lambda: world,
            get_location=lambda: SimpleNamespace(x=2.0, y=0.0),
            get_transform=lambda: SimpleNamespace(
                rotation=SimpleNamespace(yaw=0.0)
            ),
        )
        state, distance = observe_traffic_control_v3(ego)
        self.assertEqual(state, TrafficSignalState.RED)
        self.assertAlmostEqual(distance, 10.0)

    def test_runtime_junction_state_is_current_ego_not_future_route(self) -> None:
        def waypoint(*, road_id: int, junction: bool):
            return SimpleNamespace(
                road_id=road_id,
                lane_id=1,
                lane_width=3.5,
                is_junction=junction,
                transform=SimpleNamespace(
                    location=SimpleNamespace(x=float(road_id), y=0.0, z=0.0),
                    rotation=SimpleNamespace(yaw=0.0),
                ),
                get_left_lane=lambda: None,
                get_right_lane=lambda: None,
            )

        current = waypoint(road_id=1, junction=False)
        future = waypoint(road_id=2, junction=True)
        world_map = SimpleNamespace(
            get_waypoint=lambda location, project_to_road=True: (
                current if float(location.x) < 5.0 else future
            )
        )
        ego = SimpleNamespace(
            id=7,
            get_location=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )
        context = observe_route_context_v3(
            world_map=world_map,
            ego=ego,
            route_xy=[(0.0, 0.0), (10.0, 0.0)],
            explicit_maneuver=RouteManeuver.FOLLOW_STRAIGHT,
        )
        self.assertFalse(context.is_junction)

    def test_classifies_route_maneuvers(self) -> None:
        straight = _line()
        left_turn = tuple(
            (
                10.0 * math.sin(alpha),
                -10.0 * (1.0 - math.cos(alpha)),
            )
            for alpha in [index * math.pi / 36.0 for index in range(19)]
        )
        right_turn = tuple((x, -y) for x, y in left_turn)
        self.assertEqual(
            classify_route_maneuver(straight),
            RouteManeuver.FOLLOW_STRAIGHT,
        )
        self.assertEqual(
            classify_route_maneuver(left_turn, junction_flags=[True] * len(left_turn)),
            RouteManeuver.TURN_LEFT,
        )
        self.assertEqual(
            classify_route_maneuver(right_turn, junction_flags=[True] * len(right_turn)),
            RouteManeuver.TURN_RIGHT,
        )
        self.assertEqual(
            classify_route_maneuver(
                straight, route_change_side=TargetLaneSide.LEFT
            ),
            RouteManeuver.ROUTE_CHANGE_LEFT,
        )

    def test_context_hash_binds_lane_authorization(self) -> None:
        first = build_route_context(
            _line(),
            origin_road_id=1,
            origin_lane_id=1,
            left_lane=_lane(TargetLaneSide.LEFT, 3.5),
        )
        second = replace(
            first,
            left_lane=replace(first.left_lane, currently_clear=False),
            route_hash="",
            topology_hash="",
        )
        self.assertEqual(first.route_hash, second.route_hash)
        self.assertNotEqual(first.topology_hash, second.topology_hash)
        narrower = replace(
            first,
            origin_lane_width_m=3.0,
            route_hash="",
            topology_hash="",
        )
        self.assertEqual(first.route_hash, narrower.route_hash)
        self.assertNotEqual(first.topology_hash, narrower.topology_hash)

    def test_temporal_yield_uses_shared_path_not_fake_swerve(self) -> None:
        context = build_route_context(_line())
        bundle = build_k2_v3_bundle(
            native_path_xy=_line(),
            route_context=context,
            ego_v=5.0,
            base_speed_mps=6.0,
            alternative_kind=AlternativeKind.TEMPORAL_YIELD,
            alternative_available=True,
            alternative_reason="conflict_zone_ahead",
            temporal_target_speed_mps=0.0,
            backbone_forward_id="forward-1",
        )
        guarded = attach_k2_v3_guard(bundle)
        self.assertEqual(guarded.guard_status, GUARD_OK, guarded.guard_reasons)
        self.assertEqual(guarded.guard_metrics["k_eff"], 2)
        self.assertLessEqual(
            guarded.guard_metrics["max_spatial_separation_m"], 1.0e-6
        )
        self.assertEqual(
            select_k2_v3(guarded, force_index=1).alternative_kind,
            AlternativeKind.TEMPORAL_YIELD,
        )

    def test_zero_speed_t10_holds_position_without_vertex_oscillation(self) -> None:
        bundle = build_k2_v3_bundle(
            native_path_xy=_line(),
            route_context=build_route_context(_line()),
            ego_v=0.0,
            base_speed_mps=0.0,
            alternative_kind=AlternativeKind.TEMPORAL_YIELD,
            alternative_available=True,
            alternative_reason="zero_speed_sampling_regression",
            temporal_target_speed_mps=0.0,
            backbone_forward_id="stopped-forward",
        )
        for candidate in bundle.candidates:
            positions = [
                (round(float(row[0]), 6), round(float(row[1]), 6))
                for row in candidate.points_xy_yaw_v_a_kappa
            ]
            self.assertEqual(len(set(positions)), 1)

    def test_clear_is_native_singleton(self) -> None:
        context = build_route_context(_line())
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=_line(),
                route_context=context,
                ego_v=5.0,
                base_speed_mps=6.0,
                alternative_kind=AlternativeKind.NONE,
                alternative_available=False,
                alternative_reason="CLEAR_NO_ALTERNATIVE",
                backbone_forward_id="forward-clear",
            )
        )
        self.assertEqual(guarded.guard_status, GUARD_OK)
        self.assertEqual(guarded.guard_metrics["k_eff"], 1)
        with self.assertRaises(K2V3SelectionError):
            select_k2_v3(guarded, force_index=1)

    def test_selection_event_exposes_unselected_v3_teacher_slot(self) -> None:
        bundle = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=_line(),
                route_context=build_route_context(_line()),
                ego_v=3.0,
                base_speed_mps=4.0,
                alternative_kind=AlternativeKind.SPATIAL_AVOID,
                alternative_available=True,
                alternative_reason="dataset_anchor",
                target_lane_side=TargetLaneSide.RIGHT,
                avoid_offset_m=0.6,
                backbone_forward_id="forward-dataset-anchor",
            )
        )
        selection = select_k2_semantic_v3(
            bundle,
            mode="force",
            force_index=0,
        )
        event = selection_event_fields(selection)
        self.assertEqual(
            event["alternative_slot_kind"],
            AlternativeKind.SPATIAL_AVOID.value,
        )
        self.assertEqual(
            event["alternative_slot_target_lane_side"],
            TargetLaneSide.RIGHT.value,
        )
        self.assertTrue(event["alternative_slot_available"])
        self.assertGreater(
            len(event["alternative_slot_spatial_path_xy"]), 2
        )

    def test_turn_direction_guard_waits_for_local_route_commitment(self) -> None:
        approach_then_left = tuple(
            [(float(index), 0.0) for index in range(20)]
            + [
                (
                    20.0 + 8.0 * math.sin(alpha),
                    -8.0 * (1.0 - math.cos(alpha)),
                )
                for alpha in [index * math.pi / 36.0 for index in range(1, 19)]
            ]
        )
        context = build_route_context(
            approach_then_left,
            maneuver=RouteManeuver.TURN_LEFT,
            junction_flags=[False] * 20 + [True] * 18,
        )
        bundle = build_k2_v3_bundle(
            native_path_xy=[
                (float(index), 0.1 * float(index)) for index in range(10)
            ],
            route_context=context,
            ego_v=3.0,
            base_speed_mps=3.0,
            alternative_kind=AlternativeKind.NONE,
            alternative_available=False,
            alternative_reason="CLEAR_NO_ALTERNATIVE",
            backbone_forward_id="straight-approach",
        )
        self.assertIsNone(
            _route_direction_reason(bundle.candidates[0], context)
        )

    def test_native_vertex_noise_is_smoothed_without_map_route_replacement(self) -> None:
        native = tuple(
            (float(index), 0.22 if index % 2 else -0.22)
            for index in range(40)
        )
        self.assertGreater(path_max_abs_curvature(native), 0.253)
        context = build_route_context(_line())
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=native,
                route_context=context,
                ego_v=4.0,
                base_speed_mps=5.0,
                alternative_kind=AlternativeKind.NONE,
                alternative_available=False,
                alternative_reason="CLEAR_NO_ALTERNATIVE",
                backbone_forward_id="forward-noisy-native",
            )
        )
        nominal = guarded.candidates[0]
        self.assertLessEqual(
            path_max_abs_curvature(nominal.spatial_path_xy),
            0.253 + 1.0e-6,
        )
        self.assertEqual(
            nominal.head_lineage,
            "simlingo_native_anchor_continuity_v1",
        )
        self.assertNotEqual(nominal.spatial_path_xy, context.route_xy)
        self.assertEqual(guarded.guard_status, GUARD_OK, guarded.guard_reasons)

    def test_authorized_overtake_uses_full_adjacent_lane(self) -> None:
        context = build_route_context(
            _line(),
            origin_road_id=1,
            origin_lane_id=1,
            left_lane=_lane(TargetLaneSide.LEFT, 3.5),
        )
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=_line(),
                route_context=context,
                ego_v=4.0,
                base_speed_mps=5.0,
                alternative_kind=AlternativeKind.SPATIAL_OVERTAKE,
                alternative_available=True,
                alternative_reason="same_lane_obstruction_adjacent_clear",
                target_lane_side=TargetLaneSide.LEFT,
                backbone_forward_id="forward-overtake",
            )
        )
        self.assertEqual(guarded.guard_status, GUARD_OK, guarded.guard_reasons)
        self.assertGreater(
            guarded.guard_metrics["max_spatial_separation_m"], 3.0
        )
        alternative = guarded.candidates[1]
        self.assertGreater(alternative.spatial_path_xy[-1][1], 3.0)

    def test_overtake_rejected_when_lane_not_clear(self) -> None:
        context = build_route_context(
            _line(),
            left_lane=replace(
                _lane(TargetLaneSide.LEFT, 3.5),
                currently_clear=False,
            ),
        )
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=_line(),
                route_context=context,
                ego_v=4.0,
                base_speed_mps=5.0,
                alternative_kind=AlternativeKind.SPATIAL_OVERTAKE,
                alternative_available=True,
                alternative_reason="bad_authoring",
                target_lane_side=TargetLaneSide.LEFT,
                backbone_forward_id="forward-bad-lane",
            )
        )
        self.assertEqual(guarded.guard_metrics["k_eff"], 1)
        self.assertTrue(
            any("TARGET_LANE_OCCUPIED" in reason for reason in guarded.guard_reasons)
        )

    def test_spatial_avoid_cannot_leave_origin_lane_corridor(self) -> None:
        context = build_route_context(_line(), origin_lane_width_m=2.5)
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=_line(),
                route_context=context,
                ego_v=4.0,
                base_speed_mps=5.0,
                alternative_kind=AlternativeKind.SPATIAL_AVOID,
                alternative_available=True,
                alternative_reason="bad_large_swerve",
                target_lane_side=TargetLaneSide.LEFT,
                avoid_offset_m=2.4,
                backbone_forward_id="forward-corridor",
            )
        )
        self.assertFalse(
            guarded.guard_metrics["candidate_valid"]["v3_alternative"]
        )
        self.assertTrue(
            any(
                "ORIGIN_LANE_CORRIDOR_VIOLATION" in reason
                for reason in guarded.guard_reasons
            )
        )

    def test_route_change_is_not_overtake_and_holds_target_lane(self) -> None:
        target = _lane(TargetLaneSide.LEFT, 3.5)
        context = build_route_context(
            _line(),
            route_change_side=TargetLaneSide.LEFT,
            origin_road_id=1,
            origin_lane_id=1,
            target_road_id=1,
            target_lane_id=2,
            left_lane=target,
        )
        native_route_change = lane_blend_path(
            _line(), target.centerline_xy, rejoin=False
        )
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=native_route_change,
                route_context=context,
                ego_v=4.0,
                base_speed_mps=5.0,
                alternative_kind=AlternativeKind.NONE,
                alternative_available=False,
                alternative_reason="route_change_clear_singleton",
                backbone_forward_id="forward-route-change",
            )
        )
        self.assertEqual(guarded.guard_status, GUARD_OK, guarded.guard_reasons)
        self.assertGreater(guarded.candidates[0].spatial_path_xy[-1][1], 3.0)

    def test_route_change_guard_union_includes_frozen_origin_lane(self) -> None:
        origin = _line(0.0)
        target = _line(3.5)
        context = build_route_context(
            lane_blend_path(origin, target, rejoin=False),
            maneuver=RouteManeuver.ROUTE_CHANGE_LEFT,
            origin_lane_centerline_xy=origin,
            target_road_id=1,
            target_lane_id=2,
            left_lane=_lane(TargetLaneSide.LEFT, 3.5),
        )
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=origin,
                route_context=context,
                ego_v=3.0,
                base_speed_mps=4.0,
                alternative_kind=AlternativeKind.NONE,
                alternative_available=False,
                alternative_reason="route_change_transition",
                backbone_forward_id="forward-route-change-origin-union",
            )
        )
        self.assertEqual(guarded.guard_status, GUARD_OK, guarded.guard_reasons)

    def test_spatial_avoid_holds_registered_route_offset(self) -> None:
        bundle = build_k2_v3_bundle(
            native_path_xy=_line(),
            route_context=build_route_context(_line()),
            ego_v=3.0,
            base_speed_mps=4.0,
            alternative_kind=AlternativeKind.SPATIAL_AVOID,
            alternative_available=True,
            alternative_reason="right_side_cut_in",
            target_lane_side=TargetLaneSide.RIGHT,
            avoid_offset_m=0.6,
            ego_route_error_m=0.4,
            backbone_forward_id="forward-residual-avoid",
        )
        offset = max(point[1] for point in bundle.candidates[1].spatial_path_xy)
        self.assertGreater(offset, 0.55)
        self.assertLess(offset, 0.65)

    def test_passed_overtake_actor_switches_to_route_rejoin(self) -> None:
        context = build_route_context(
            _line(),
            origin_lane_centerline_xy=_line(),
            left_lane=_lane(TargetLaneSide.LEFT, 3.5),
        )
        bundle = build_k2_v3_bundle(
            native_path_xy=_line(3.5),
            route_context=context,
            ego_v=3.0,
            base_speed_mps=4.0,
            alternative_kind=AlternativeKind.SPATIAL_OVERTAKE,
            alternative_available=True,
            alternative_reason="actor_passed_rejoin",
            target_lane_side=TargetLaneSide.LEFT,
            overtake_actor_lon_m=-2.1,
            backbone_forward_id="forward-overtake-rejoin",
        )
        alternative = bundle.candidates[1]
        self.assertEqual(alternative.maneuver_phase, ManeuverPhase.REJOIN)
        self.assertLess(alternative.spatial_path_xy[-1][1], 0.25)

    def test_overtake_depart_is_one_way_to_target_lane(self) -> None:
        context = build_route_context(
            _line(),
            origin_lane_centerline_xy=_line(),
            left_lane=_lane(TargetLaneSide.LEFT, 3.5),
        )
        bundle = build_k2_v3_bundle(
            native_path_xy=_line(),
            route_context=context,
            ego_v=2.0,
            base_speed_mps=0.5,
            alternative_kind=AlternativeKind.SPATIAL_OVERTAKE,
            alternative_available=True,
            alternative_reason="persistent_depart",
            target_lane_side=TargetLaneSide.LEFT,
            overtake_phase_v3="DEPART",
            backbone_forward_id="forward-overtake-depart",
        )
        alternative = bundle.candidates[1]
        self.assertEqual(alternative.maneuver_phase, ManeuverPhase.DEPART)
        self.assertGreater(alternative.spatial_path_xy[-1][1], 3.0)
        self.assertGreaterEqual(min(alternative.speed_samples_mps), 2.0)

    def test_red_light_nominal_is_filtered_before_world(self) -> None:
        context = build_route_context(
            _line(),
            stop_line_distance_m=3.0,
            traffic_signal_state=TrafficSignalState.RED,
        )
        guarded = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=_line(),
                route_context=context,
                ego_v=5.0,
                base_speed_mps=6.0,
                alternative_kind=AlternativeKind.TEMPORAL_YIELD,
                alternative_available=True,
                alternative_reason="red_light_stop",
                temporal_target_speed_mps=0.0,
                backbone_forward_id="forward-red",
            )
        )
        validity = guarded.guard_metrics["candidate_valid"]
        self.assertFalse(validity["v3_nominal_progress"])
        self.assertTrue(validity["v3_alternative"])
        self.assertEqual(select_k2_v3(guarded).candidate_id, "v3_alternative")

    def test_legacy_route_serialization_stays_unchanged(self) -> None:
        route = RouteSpec(
            identity="legacy",
            waypoints=((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
            target_speed_mps=8.0,
        )
        self.assertEqual(
            route.to_dict(),
            {
                "identity": "legacy",
                "waypoints": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "target_speed_mps": 8.0,
            },
        )

    def test_registry_navigation_extension_is_fail_closed(self) -> None:
        context = build_route_context(_line())
        route = _route_from_mapping(
            {
                "identity": "v3",
                "waypoints": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "navigation_context": {
                    "maneuver": context.maneuver.value,
                    "entry_signature": "1:1:entry",
                    "exit_signature": "1:1:exit",
                    "route_hash": context.route_hash,
                    "topology_hash": context.topology_hash,
                },
            }
        )
        self.assertEqual(
            route.navigation_context["route_hash"], context.route_hash
        )
        frozen = _route_from_mapping(
            {
                "identity": "v3-frozen",
                "waypoints": [[x, y, 0.0] for x, y in context.route_xy],
                "navigation_context": {
                    "maneuver": context.maneuver.value,
                    "entry_signature": context.entry_signature,
                    "exit_signature": context.exit_signature,
                    "route_hash": context.route_hash,
                    "topology_hash": context.topology_hash,
                    "frozen_context_json": json.dumps(context.to_dict()),
                },
            }
        )
        self.assertIn(
            "frozen_context_json", frozen.navigation_context
        )
        with self.assertRaises(RegistryError):
            _route_from_mapping(
                {
                    "identity": "bad",
                    "waypoints": [[0.0, 0.0], [20.0, 0.0]],
                    "navigation_context": {"maneuver": "TURN_LEFT"},
                }
            )

    def test_map_authoring_chooses_registered_left_exit(self) -> None:
        authored = author_route_from_waypoint(
            _mock_left_junction(),
            maneuver=RouteManeuver.TURN_LEFT,
            horizon_m=45.0,
        )
        self.assertEqual(
            authored.route_context.maneuver, RouteManeuver.TURN_LEFT
        )
        self.assertTrue(authored.route_context.is_junction)
        self.assertEqual(authored.route_context.target_road_id, 2)
        self.assertTrue(authored.route_context.exit_signature.startswith("2:1:"))
        self.assertGreater(
            float(authored.first_junction_heading_change_deg), 35.0
        )
        self.assertGreater(authored.route_length_m, 30.0)
        self.assertGreater(
            math.degrees(route_heading_change_rad(authored.route_context.route_xy)),
            35.0,
        )


if __name__ == "__main__":
    unittest.main()
