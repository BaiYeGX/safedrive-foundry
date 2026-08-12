from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from safedrive_foundry.driving_vla.evaluation.r2_world_ready_v3 import (
    TEMPLATES,
    build_calibration_manifest_v3,
    build_core_blind_manifest_v3,
    build_long_smoke_manifest_v3,
    build_unseen_long_audit_manifest_v3,
    build_world_campaign_manifest_v3,
    build_world_ready_audit_manifest_v3,
    evaluate_core_blind_v3,
    evaluate_world_ready_audit_v3,
)
from scripts.r2_v3_author_route_bank import (
    _adjacent_ok,
    _author_one,
    _candidate_pool,
    _required_adjacent_side,
    build_route_requirements,
)
from driving_vla.evaluation.route_authoring_v3 import RouteAuthoringError
from driving_vla.model.navigation_contract import RouteManeuver, build_route_context


class R2WorldReadyManifestTest(unittest.TestCase):
    def test_candidate_pool_does_not_retain_filled_generic_mode(self) -> None:
        starts = [
            SimpleNamespace(road_id=index, lane_id=1, s=0.0)
            for index in range(100)
        ]

        class WorldMap:
            def generate_waypoints(self, _spacing):
                return starts

        class Authored:
            def __init__(self, index: int):
                self.index = index

            def to_dict(self):
                lane = {
                    "exists": False,
                    "driving": False,
                    "same_direction": False,
                    "lane_change_allowed": False,
                    "currently_clear": False,
                    "centerline_xy": [],
                }
                return {
                    "route_context": {
                        "route_hash": f"hash-{self.index}",
                        "entry_signature": f"entry-{self.index}",
                        "left_lane": lane,
                        "right_lane": lane,
                    }
                }

        requirements = [
            {
                "template_id": "follow_stop",
                "maneuver": "FOLLOW_STRAIGHT",
            },
            {
                "template_id": "cut_in_left",
                "maneuver": "FOLLOW_STRAIGHT",
            },
        ]
        with patch(
            "scripts.r2_v3_author_route_bank.author_route_from_waypoint",
            side_effect=lambda start, **_kwargs: Authored(start.road_id),
        ):
            routes = _candidate_pool(
                world_map=WorldMap(),
                maneuver=RouteManeuver.FOLLOW_STRAIGHT,
                horizon_m=80.0,
                requirements=requirements,
            )
        self.assertEqual(len(routes), 4)

    def test_cut_in_needs_spawnable_lane_but_overtake_needs_authorized_lane(self) -> None:
        context = build_route_context(
            [(float(index), 0.0) for index in range(40)]
        ).to_dict()
        context["left_lane"] = {
            "side": "LEFT",
            "exists": True,
            "driving": True,
            "same_direction": True,
            "lane_change_allowed": False,
            "currently_clear": False,
            "centerline_xy": [[0.0, -3.5], [20.0, -3.5]],
        }
        route = {"route_context": context}
        cut = {
            "template_id": "cut_in_left",
            "maneuver": "FOLLOW_STRAIGHT",
        }
        overtake = {
            "template_id": "overtake_left",
            "maneuver": "FOLLOW_STRAIGHT",
        }
        self.assertTrue(
            _adjacent_ok(route, _required_adjacent_side(cut))
        )
        self.assertFalse(
            _adjacent_ok(route, _required_adjacent_side(overtake))
        )

    def test_route_reuse_is_formal_phase_local_and_never_calibration(self) -> None:
        class World:
            class Map:
                pass

            def get_map(self):
                return self.Map()

        route = {
            "route_context": build_route_context(
                [(float(index), 0.0) for index in range(40)]
            ).to_dict()
        }
        prior = {
            "route_fixture_id": "prior",
            "phase": "world_ready_audit",
            "template_id": "follow_stop",
            "maneuver": "FOLLOW_STRAIGHT",
        }
        request = {
            **prior,
            "route_fixture_id": "next",
            "template_id": "follow_stop",
        }
        reused, source = _author_one(
            world=World(),
            requirement=request,
            horizon_m=80.0,
            used_hashes={route["route_context"]["route_hash"]},
            used_entries={
                route["route_context"]["entry_signature"]
            },
            reusable_routes=[(prior, route)],
        )
        self.assertEqual(source, "prior")
        self.assertEqual(reused, route)
        with self.assertRaises(RouteAuthoringError):
            _author_one(
                world=World(),
                requirement={**request, "phase": "calibration"},
                horizon_m=80.0,
                used_hashes=set(),
                used_entries=set(),
                reusable_routes=[(prior, route)],
            )

    def test_traffic_control_straight_is_a_junction_maneuver(self) -> None:
        traffic = next(
            template
            for template in TEMPLATES
            if template.template_id == "traffic_control"
        )
        self.assertNotIn(
            "FOLLOW_STRAIGHT",
            {maneuver.value for maneuver in traffic.maneuver_cycle},
        )
        self.assertIn(
            "JUNCTION_STRAIGHT",
            {maneuver.value for maneuver in traffic.maneuver_cycle},
        )

    def test_calibration_is_exact_360_and_pilot_144(self) -> None:
        manifest = build_calibration_manifest_v3()
        self.assertEqual(len(manifest["slots"]), 360)
        self.assertEqual(manifest["split_slots"], {"train": 252, "dev": 60, "test": 48})
        self.assertEqual(manifest["pilot_slots"], 144)

    def test_smoke_core_audit_and_world_counts(self) -> None:
        self.assertEqual(len(build_long_smoke_manifest_v3(learned=False)["cases"]), 16)
        self.assertEqual(len(build_long_smoke_manifest_v3(learned=True)["cases"]), 16)
        unseen = build_unseen_long_audit_manifest_v3()
        self.assertEqual(len(unseen["cases"]), 16)
        self.assertEqual(
            len({case["route_fixture_id"] for case in unseen["cases"]}),
            16,
        )
        core = build_core_blind_manifest_v3()
        self.assertEqual(len(core["cases"]), 12)
        town13_straight = [
            case
            for case in core["cases"]
            if case["map_name"] == "Town13"
            and case["maneuver"] == "FOLLOW_STRAIGHT"
        ]
        self.assertEqual(town13_straight, [])
        self.assertGreaterEqual(
            sum(bool(case["dual_candidate_expected"]) for case in core["cases"]),
            10,
        )
        audit = build_world_ready_audit_manifest_v3()
        self.assertEqual(len(audit["cases"]), 84)
        audit_maps = {case["map_name"] for case in audit["cases"]}
        self.assertNotIn("Town07", audit_maps)
        self.assertIn("Town13", audit_maps)
        template_counts = {
            template_id: sum(
                case["template_id"] == template_id
                for case in audit["cases"]
            )
            for template_id in (
                "cut_in_left",
                "cut_in_right",
                "overtake_left",
                "overtake_right",
            )
        }
        self.assertEqual(template_counts["cut_in_left"], template_counts["cut_in_right"])
        self.assertEqual(
            template_counts["overtake_left"],
            template_counts["overtake_right"],
        )
        world = build_world_campaign_manifest_v3(formal_checkpoint_hash="a" * 64)
        self.assertEqual(len(world["slots"]), 1008)

    def test_route_bank_requirements_are_unique_and_frozen(self) -> None:
        manifests = {
            "calibration_360.json": build_calibration_manifest_v3(),
            "core_blind_12.json": build_core_blind_manifest_v3(),
            "world_ready_audit_84.json": (
                build_world_ready_audit_manifest_v3()
            ),
            "unseen_long_audit_16.json": (
                build_unseen_long_audit_manifest_v3()
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in manifests.items():
                (root / name).write_text(
                    json.dumps(value),
                    encoding="utf-8",
                )
            requirements = build_route_requirements(
                root,
                map_name="Town03",
            )
        route_ids = [
            requirement["route_fixture_id"]
            for requirement in requirements
        ]
        self.assertEqual(len(route_ids), len(set(route_ids)))
        self.assertGreaterEqual(len(route_ids), 24)
        self.assertEqual(
            {requirement["map_name"] for requirement in requirements},
            {"Town03"},
        )

    def test_core_gate(self) -> None:
        records = []
        families = ("cut_in", "obstruction", "crossing")
        for index in range(12):
            records.append(
                {
                    "comparable": index < 10,
                    "decisive": index < 6,
                    "winner": index % 2 if index < 6 else None,
                    "family": families[index % len(families)],
                    "pair_label": "C0" if index % 2 == 0 else "C1",
                    "collision": False,
                    "offroad": False,
                    "wrong_exit": False,
                }
            )
        self.assertTrue(evaluate_core_blind_v3(records)["passed"])

    def test_world_ready_gate_pass_and_fatal_fail(self) -> None:
        audit = build_world_ready_audit_manifest_v3()
        records = []
        for index, case in enumerate(audit["cases"]):
            records.append(
                {
                    **case,
                    "route_completed": True,
                    "candidate1_available": index < 60,
                    "comparable": index < 60,
                    "decisive": index < 24,
                    "winner": index % 2 if index < 24 else None,
                    "safe_candidate_exists": True,
                    "both_bad": False,
                    "guard_mpc_failure": False,
                    "collision": False,
                    "offroad": False,
                    "wrong_exit": False,
                }
            )
        self.assertTrue(evaluate_world_ready_audit_v3(records)["passed"])
        records[0]["wrong_exit"] = True
        self.assertFalse(evaluate_world_ready_audit_v3(records)["passed"])


if __name__ == "__main__":
    unittest.main()
