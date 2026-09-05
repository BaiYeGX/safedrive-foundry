"""C2 correction regression tests; no historical file hashing."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h6.cora.repair import diagnostic_quality, root_counts, coverage_gaps
from data_pipeline.h6.cora.repair_labels import route_labels, safety_trace


class RootCoverageTests(unittest.TestCase):
    def test_branch_duplication_and_opposite_classes(self):
        def branch(value):
            return dict(outcome_valid=True, guard_verdict="REVIEW", heads={
                "executable": dict(valid=True, value=value),
                "off_corridor_duration_s": dict(valid=True, value=0.2)})
        root = dict(root_id="a", root_cluster_id="capture-a", split="train",
                    branches=[branch(True), branch(True), branch(False)])
        duplicate = copy.deepcopy(root); duplicate["root_id"] = "retry"
        counts = root_counts([root, duplicate])
        self.assertEqual(counts["train"]["offroad"]["positive"], 1)
        self.assertEqual(counts["train"]["executable"], dict(positive=1, negative=1, missing=0))
        duplicate["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split_leakage"):
            root_counts([root, duplicate])

    def test_auxiliary_and_diagnostic_excluded(self):
        b = dict(outcome_valid=True, auxiliary_only=True, guard_verdict="REVIEW",
                 heads={"executable": dict(valid=True, value=False)})
        r = dict(root_id="a", split="train", branches=[b])
        self.assertEqual(root_counts([r])["train"]["executable"]["negative"], 0)
        b["auxiliary_only"] = False; b["guard_verdict"] = "REJECT"
        self.assertEqual(root_counts([r])["train"]["executable"]["negative"], 0)
        b["guard_verdict"] = "PASS"; r["diagnostic"] = True
        self.assertEqual(root_counts([r])["train"]["executable"]["negative"], 0)

    def test_thresholds_unchanged(self):
        gaps = coverage_gaps(root_counts([]))
        self.assertEqual(len(gaps), 28)
        self.assertEqual(next(x for x in gaps if x["split"] == "train" and x["head"] == "repair_success")["required"], 12)

    def test_diagnostic_gate_counts_valid_failures_by_root(self):
        def branch(attempted, success, offroad=0.0):
            return {
                "outcome_valid": True,
                "guard_verdict": "REVIEW",
                "heads": {
                    "repair_attempted": {"valid": True, "value": attempted},
                    "repair_success": {"valid": True, "value": success},
                    "off_corridor_duration_s": {"valid": True, "value": offroad},
                },
            }
        records = [
            dict(root_id="a", root_cluster_id="a", diagnostic=True,
                 branches=[branch(True, False)]),
            dict(root_id="a-retry", root_cluster_id="a", diagnostic=True,
                 branches=[branch(True, False, 1.0)]),
            dict(root_id="b", root_cluster_id="b", diagnostic=True,
                 branches=[branch(False, None, 1.0)]),
        ]
        result = diagnostic_quality(records, {
            "min_diagnostic_repair_failure_roots": 2,
            "min_diagnostic_offroad_roots": 1,
        })
        self.assertEqual(result["repair_failure_roots"], 1)
        self.assertEqual(result["offroad_roots"], 2)
        self.assertFalse(result["passed"])


class RouteLabelTests(unittest.TestCase):
    def test_absolute_completion_from_offset_start(self):
        heads = route_labels([dict(x=19, y=0, tick=0)], {"route": [[0,0],[20,0]]}, [10,0])
        self.assertTrue(heads["route_completed"]["value"])

    def test_crossing_uses_crossing_state_not_final_state(self):
        scene = dict(route=[[0,0],[30,0]], red_light=dict(stop_progress_m=10))
        rows = [dict(x=12,y=0,tick=0,traffic_light_state="Green"),
                dict(x=13,y=0,tick=1,traffic_light_state="Red")]
        self.assertFalse(route_labels(rows,scene,[8,0])["red_light_violation"]["value"])
        rows[0]["traffic_light_state"] = "Red"
        self.assertTrue(route_labels(rows,scene,[8,0])["red_light_violation"]["value"])

    def test_missing_light_independent_mask(self):
        scene = dict(route=[[0,0],[30,0]], red_light=dict(stop_progress_m=10))
        heads = route_labels([dict(x=12,y=0,tick=0)],scene,[8,0])
        self.assertFalse(heads["red_light_violation"]["valid"])
        self.assertTrue(heads["route_completed"]["valid"])


class RepairTraceTests(unittest.TestCase):
    def test_all_attempts_failed_keeps_parent_and_false_success(self):
        decision = SimpleNamespace(
            decision_kind=SimpleNamespace(value="HARD_REJECT"),
            accepted_candidate=None, post_repair_trajectory_id=None,
        )
        result = SimpleNamespace(
            decision=decision,
            repair_result=None,
        )
        failed = {"mode": "longitudinal", "pre_repair_id": "proposal-1",
                  "post_repair_id": "proposal-1:qp", "success": False,
                  "solver_trace": {"status": "infeasible"}}
        with patch("data_pipeline.h6.cora.repair_labels.decision_to_dict", return_value={"decision_kind": "HARD_REJECT"}):
            trace = safety_trace(result, [failed], "proposal-1")
        self.assertTrue(trace["repair_attempted"])
        self.assertFalse(trace["repair_success"])
        self.assertEqual(trace["attempts"][0]["pre_repair_id"], "proposal-1")

    def test_unattempted_has_unknown_success(self):
        decision = SimpleNamespace(
            decision_kind=SimpleNamespace(value="EMERGENCY"),
            accepted_candidate=None, post_repair_trajectory_id=None,
        )
        result = SimpleNamespace(decision=decision, repair_result=None)
        with patch("data_pipeline.h6.cora.repair_labels.decision_to_dict", return_value={"decision_kind": "EMERGENCY"}):
            trace = safety_trace(result, [], "proposal-1")
        self.assertFalse(trace["repair_attempted"])
        self.assertIsNone(trace["repair_success"])
