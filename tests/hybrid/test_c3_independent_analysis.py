"""Regressions for C3 reporting errors, separate from model training."""
import copy
import tempfile
import unittest
from pathlib import Path

from scripts import h6_cora_independent_analysis as analysis


def fixture(root="r"):
    sample = {"root_id": root, "route_target": [[float(i), 0.0] for i in range(20)],
              "speed_target": [[0.25*(i+1), 0.0] for i in range(10)],
              "route_mask": [True]*12 + [False]*8, "speed_mask": [True]*10}
    row = {"root_id": root, "route": copy.deepcopy(sample["route_target"]),
           "speed": copy.deepcopy(sample["speed_target"])}
    return sample, row


class IndependentAnalysisTests(unittest.TestCase):
    def test_partial_truth_does_not_invalidate_real_canonical_output(self):
        sample, row = fixture()
        score = analysis._score(row, sample)
        self.assertTrue(score["prediction_valid"])
        self.assertFalse(score["route_full_valid"])
        self.assertIsNone(score["route_fde_m"])
        self.assertEqual(score["route_ade_m"], 0.0)

    def test_finite_degenerate_route_is_rejected_by_actual_converter(self):
        sample, row = fixture()
        row["route"] = [[0.0, 0.0]]*20
        score = analysis._score(row, sample)
        self.assertTrue(score["native_finite"])
        self.assertFalse(score["prediction_valid"])
        self.assertTrue(any("degenerate_path" in r for r in score["failure_reasons"]))

    def test_nonfinite_outside_truth_support_remains_prediction_failure(self):
        sample, row = fixture()
        row["route"][-1][0] = float("nan")
        score = analysis._score(row, sample)
        self.assertEqual(score["route_ade_m"], 0.0)
        self.assertFalse(score["prediction_valid"])

    def test_bad_shape_and_nonnumeric_predictions_do_not_crash_or_remove_row(self):
        for bad in ([], [[1.0, 2.0]], [["bad", 0]]*20):
            with self.subTest(bad=bad):
                sample, row = fixture()
                row["route"] = bad
                score = analysis._score(row, sample)
                self.assertFalse(score["prediction_valid"])
                self.assertIsNone(score["route_ade_m"])

    def test_prediction_failure_keeps_full_denominator(self):
        s1, r1 = fixture("a")
        s2, r2 = fixture("b")
        r2["speed"] = []
        scores = [analysis._score(r1, s1), analysis._score(r2, s2)]
        result = analysis._paired_summary({"m0": scores, "m1": scores})
        self.assertEqual(result["root_count"], {"m0": 2, "m1": 2})
        self.assertEqual(result["prediction_fail_rate_percent"]["m1"], 50.0)
        self.assertIsNone(result["speed_wp_ade_m"]["m1"])

    def test_invalid_truth_is_an_error_not_a_prediction_failure(self):
        sample, row = fixture()
        sample["route_target"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid_frozen_truth"):
            analysis._score(row, sample)

    def test_duplicate_root_cannot_change_weights(self):
        sample, row = fixture()
        score = analysis._score(row, sample)
        with self.assertRaisesRegex(ValueError, "duplicate_or_empty_root"):
            analysis._paired_summary({"m0": [score, score], "m1": [score]})

    def test_navigation_comparison_uses_identical_support_and_recomputes_errors(self):
        sample, nav = fixture()
        nav["geometry_route_mask"] = [True]*4+[False]*16
        nav["route_ade_m"] = 999.0  # Saved scalar must not be trusted.
        pred = copy.deepcopy(nav)
        pred["route"][6][1] = 10.0  # Outside navigation's support.
        diag = {"rows": {"navigation_geometry": [nav]}}
        _, compared = analysis._recompute_diagnostics(diag, {"r": sample}, {"m0": {"r": pred}, "m1": {"r": pred}})
        for row in compared.values():
            self.assertEqual(row["route_valid_points"], 4)
            self.assertEqual(row["route_ade_root_equal_m"], 0.0)
            self.assertIn("speed_wp_ade_root_equal_m", row)

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"report.json"
            output.write_text("original")
            with self.assertRaises(FileExistsError):
                analysis.main(["--run-dir", tmp, "--output", str(output)])
            self.assertEqual(output.read_text(), "original")

    def test_baseline_failure_is_not_silently_removed_from_mean(self):
        sample, row = fixture("a")
        failed_sample, failed_row = fixture("b")
        failed_row["speed"] = []
        summary = analysis._simple_baseline([analysis._score(row, sample), analysis._score(failed_row, failed_sample)])
        self.assertEqual(summary["root_count"], 2)
        self.assertIsNone(summary["speed_wp_ade_root_equal_m"])


if __name__ == "__main__":
    unittest.main()
