from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from safedrive_foundry.driving_vla.evaluation.scenario_registry import (
    load_scenario_registry,
)
from safedrive_foundry.driving_vla.evaluation.route_authoring_v3 import (
    navigation_waypoints_xyz,
)
from safedrive_foundry.driving_vla.model.navigation_contract import (
    LaneAccessV3,
    RouteManeuver,
    TargetLaneSide,
    build_route_context,
)
from scripts.r2_v3_author_campaign import _case_for_slot, _render_registry
from scripts.r2_v3_author_long_smoke import _build_cases


class R2V3CampaignRuntimeTest(unittest.TestCase):
    def test_route_change_renderer_uses_frozen_navigation_blend(self) -> None:
        raw = [[float(index), 0.0, 1.0] for index in range(40)]
        navigation = tuple((float(index), 3.5) for index in range(40))
        context = build_route_context(
            navigation,
            maneuver=RouteManeuver.ROUTE_CHANGE_LEFT,
            route_change_side=TargetLaneSide.LEFT,
        )
        rendered = navigation_waypoints_xyz(
            {"route_context": context.to_dict(), "waypoints_xyz": raw}
        )
        self.assertEqual(rendered[10], (10.0, 3.5, 1.0))

    @staticmethod
    def _straight_route() -> dict:
        route_xy = tuple((float(index), 0.0) for index in range(40))
        adjacent_xy = tuple((float(index), -3.5) for index in range(40))
        context = build_route_context(
            route_xy,
            origin_road_id=1,
            origin_lane_id=-1,
            target_road_id=1,
            target_lane_id=-1,
            entry_signature="1:-1:0",
            exit_signature="1:-1:8",
            right_lane=LaneAccessV3(
                side=TargetLaneSide.RIGHT,
                exists=True,
                driving=True,
                same_direction=True,
                lane_change_allowed=True,
                currently_clear=True,
                centerline_xy=adjacent_xy,
                lane_id=-2,
                road_id=1,
            ),
        )
        return {
            "route_context": context.to_dict(),
            "waypoints_xyz": [
                [float(index), 0.0, 0.0] for index in range(40)
            ],
            "ego_spawn_transform": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.5,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
            },
        }

    def test_straight_merge_keeps_maneuver_and_uses_right_actor(self) -> None:
        route = self._straight_route()
        case = _case_for_slot(
            world=object(),
            route_manifest={
                "horizon_m": 80.0,
                "routes": {"FOLLOW_STRAIGHT": route},
            },
            prototype_by_id={
                "cut_in_right": {
                    "route": route,
                    "family": "cut_in",
                    "actor": {},
                }
            },
            template_id="merge_yield",
            maneuver=RouteManeuver.FOLLOW_STRAIGHT,
        )
        self.assertEqual(
            case["route"]["route_context"]["maneuver"],
            "FOLLOW_STRAIGHT",
        )
        self.assertEqual(case["family"], "merge")
        self.assertEqual(case["actor"]["script_kind"], "merge_right")
        self.assertAlmostEqual(case["actor"]["pose"]["y"], -3.5)

    def test_campaign_prototypes_do_not_require_signalized_turn(self) -> None:
        route = self._straight_route()
        routes = {
            maneuver.value: route
            for maneuver in RouteManeuver
        }
        manifest = {
            "map_name": "Town03",
            "horizon_m": 80.0,
            "routes": routes,
        }

        class _World:
            @staticmethod
            def get_map() -> object:
                return object()

        with (
            patch(
                "scripts.r2_v3_author_long_smoke._straight_variant",
                return_value=route,
            ),
            patch(
                "scripts.r2_v3_author_long_smoke._traffic_light_turn",
                side_effect=AssertionError("traffic author must not run"),
            ),
            patch(
                "scripts.r2_v3_author_long_smoke._route_actor",
                return_value={"x": 1.0, "y": 0.0, "z": 0.5, "yaw_deg": 0.0},
            ),
            patch(
                "scripts.r2_v3_author_long_smoke._adjacent_actor",
                return_value={"x": 1.0, "y": 3.5, "z": 0.5, "yaw_deg": 0.0},
            ),
            patch(
                "scripts.r2_v3_author_long_smoke._crossing_actor",
                return_value={"x": 1.0, "y": 4.0, "z": 0.5, "yaw_deg": 90.0},
            ),
        ):
            cases = _build_cases(
                _World(),
                manifest,
                include_traffic=False,
            )
        self.assertEqual(len(cases), 15)
        self.assertNotIn(
            "red_green_resume_route",
            {case["scenario_id"] for case in cases},
        )

    def test_lineage_renderer_roundtrips_three_conditions_two_seeds(self) -> None:
        route = self._straight_route()
        slots = [
            {
                "slot_id": f"slot-{condition}-{seed}",
                "lineage_id": "lineage-1",
                "split": "train",
                "map_name": "Town03",
                "template_id": "straight_curve",
                "family": "clear",
                "maneuver": "FOLLOW_STRAIGHT",
                "condition": condition,
                "seed_id": seed,
                "route_fixture_id": "route-1",
                "actor_script_id": f"actors-{condition}",
            }
            for condition in ("mild", "medium", "hard")
            for seed in ("seed_a", "seed_b")
        ]
        text = _render_registry(
            map_name="Town03",
            lineage_id="lineage-1",
            slots=slots,
            prototype={"route": route, "family": "clear"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.toml"
            path.write_text(text, encoding="utf-8")
            registry = load_scenario_registry(path)
        self.assertEqual(len(registry.fixtures), 6)
        self.assertEqual(
            {fixture.seed_id for fixture in registry.fixtures},
            {"seed_a", "seed_b"},
        )
        self.assertTrue(
            registry.registry_version.startswith("r2v3-campaign-")
        )


if __name__ == "__main__":
    unittest.main()
