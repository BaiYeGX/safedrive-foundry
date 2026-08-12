from __future__ import annotations

import unittest

from safedrive_foundry.driving_vla.evaluation.r2_v3_dataset import (
    R2V3DatasetError,
    build_v3_dataset_row,
    evaluate_offline_prediction_records_v3,
    validate_v3_dataset_manifest,
    validate_v3_dataset_rows,
)
from safedrive_foundry.driving_vla.model.navigation_contract import (
    TrafficSignalState,
    build_route_context,
)


def _row(index: int, split: str):
    context = build_route_context(
        [(float(point), float(index) * 5.0) for point in range(40)]
    )
    return {
        "sample_id": f"sample-{index}",
        "lineage_id": f"lineage-{index}",
        "split": split,
        "route_fixture_id": f"route-{index}",
        "route_context": context.to_dict(),
        "native_path_xy": [[0.0, 0.0], [20.0, 0.0]],
        "ego_v": 4.0,
        "base_speed_mps": 5.0,
        "alternative_kind": "NONE",
        "target_lane_side": "NONE",
        "alternative_available": False,
    }


class R2V3DatasetTest(unittest.TestCase):
    def test_teacher_event_builds_same_forward_dataset_row(self) -> None:
        context = build_route_context(
            [(float(index), 0.0) for index in range(40)]
        )
        slot = {
            "slot_id": "slot-1",
            "lineage_id": "lineage-1",
            "split": "train",
            "map_name": "Town03",
            "template_id": "cut_in_left",
            "family": "cut_in",
            "condition": "mild",
            "seed_id": "seed_a",
            "route_fixture_id": "route-1",
            "actor_script_id": "actor-1",
            "maneuver": "FOLLOW_STRAIGHT",
            "alternative_kind": "SPATIAL_AVOID",
        }
        event = {
            "vla_version": "v3",
            "route_maneuver": "FOLLOW_STRAIGHT",
            "route_hash": context.route_hash,
            "alternative_slot_kind": "SPATIAL_AVOID",
            "alternative_slot_available": True,
            "alternative_slot_target_lane_side": "RIGHT",
            "alternative_slot_availability_reason": "CUT_IN",
            "accepted": True,
            "candidate_valid": {
                "v3_nominal_progress": True,
                "v3_alternative": True,
            },
            "driving_feature": [0.1] * 64,
            "driving_feature_hash": "feature",
            "driving_feature_raw_hash": "raw",
            "raw_path_map_xy": [
                [float(index), 0.0] for index in range(20)
            ],
            "vla_speed_samples_mps": [3.0] * 10,
            "resolved_vla_input_speed_mps": 2.0,
            "observable_scene_v1": {
                "actor_present": True,
                "actor_lat_m": 2.0,
            },
            "camera_frame": 10,
        }
        row = build_v3_dataset_row(
            slot=slot,
            route_context=context.to_dict(),
            event=event,
        )
        self.assertEqual(row["target_lane_side"], "RIGHT")
        self.assertEqual(row["alternative_kind"], "SPATIAL_AVOID")
        self.assertEqual(len(row["driving_feature"]), 64)
        self.assertTrue(row["teacher_guard_accepted"])

    def test_teacher_signal_observation_refreshes_dynamic_context(self) -> None:
        context = build_route_context(
            [(float(index), 0.0) for index in range(40)],
            traffic_signal_state=TrafficSignalState.UNKNOWN,
        )
        slot = {
            "slot_id": "slot-signal",
            "lineage_id": "lineage-signal",
            "split": "train",
            "map_name": "Town05",
            "template_id": "traffic_control",
            "family": "traffic_control",
            "condition": "mild",
            "seed_id": "seed_a",
            "route_fixture_id": "route-signal",
            "actor_script_id": "actor-signal",
            "maneuver": "FOLLOW_STRAIGHT",
            "alternative_kind": "TEMPORAL_YIELD",
        }
        event = {
            "vla_version": "v3",
            "route_maneuver": "FOLLOW_STRAIGHT",
            "route_hash": context.route_hash,
            "alternative_slot_kind": "TEMPORAL_YIELD",
            "alternative_slot_available": True,
            "alternative_slot_target_lane_side": "NONE",
            "alternative_slot_availability_reason": "TRAFFIC_CONTROL_REQUIRES_STOP",
            "traffic_signal_state": "RED",
            "stop_line_distance_m": 4.0,
            "accepted": True,
            "candidate_valid": {
                "v3_nominal_progress": True,
                "v3_alternative": True,
            },
            "driving_feature": [0.1] * 64,
            "raw_path_map_xy": [[float(index), 0.0] for index in range(20)],
            "vla_speed_samples_mps": [3.0] * 10,
            "resolved_vla_input_speed_mps": 2.0,
            "camera_frame": 10,
        }
        row = build_v3_dataset_row(
            slot=slot,
            route_context=context.to_dict(),
            event=event,
        )
        self.assertEqual(
            row["route_context"]["traffic_signal_state"],
            "RED",
        )
        self.assertAlmostEqual(
            float(row["route_context"]["stop_line_distance_m"]),
            4.0,
        )

    def test_split_isolation(self) -> None:
        rows = [_row(0, "train"), _row(1, "dev"), _row(2, "test")]
        audit = validate_v3_dataset_rows(rows)
        self.assertEqual(audit["lineage_overlap"], 0)
        rows[1]["lineage_id"] = rows[0]["lineage_id"]
        with self.assertRaises(R2V3DatasetError):
            validate_v3_dataset_rows(rows)

    def test_actual_route_hash_isolation(self) -> None:
        rows = [_row(0, "train"), _row(1, "dev"), _row(2, "test")]
        rows[1]["route_context"] = dict(rows[0]["route_context"])
        with self.assertRaisesRegex(
            R2V3DatasetError,
            "actual route hash overlap",
        ):
            validate_v3_dataset_rows(rows)

    def test_dataset_manifest_rejects_incomplete_full_campaign(self) -> None:
        with self.assertRaisesRegex(R2V3DatasetError, "requires 360"):
            validate_v3_dataset_manifest(
                {"phase": "calibration", "slots": []}
            )

    def test_offline_gates(self) -> None:
        records = []
        for index in range(20):
            available = index < 16
            side = "LEFT" if index % 2 == 0 else "RIGHT"
            kind = "SPATIAL_AVOID" if available else "NONE"
            records.append(
                {
                    "target_kind": kind,
                    "predicted_kind": kind,
                    "target_side": side if available else "NONE",
                    "predicted_side": side if available else "NONE",
                    "target_available": available,
                    "predicted_available": available,
                    "input_route_maneuver": "FOLLOW_STRAIGHT",
                    "output_route_maneuver": "FOLLOW_STRAIGHT",
                    "legal_route_target": True,
                    "guard_accepted": True,
                    "mpc_accepted": True,
                }
            )
        report = evaluate_offline_prediction_records_v3(records)
        self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
