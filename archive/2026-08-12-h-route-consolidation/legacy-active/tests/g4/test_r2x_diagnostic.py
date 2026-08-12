"""R2-F diagnostic offline tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from scripts.r2x_selection_space_diagnostic import (  # noqa: E402
    _classify,
    run_diagnostic,
)


class R2XDiagnosticTest(unittest.TestCase):
    def test_classify_temporal_only(self) -> None:
        b = _classify(
            same_path=True,
            sep_max=1.5,
            speed_mean=1.2,
            ctrl={"n": 50, "steer_mae": 0.01, "throttle_mae": 0.05},
            outcome_delta=None,
            incomplete=False,
        )
        self.assertEqual(b, "PROPOSAL_TEMPORAL_ONLY")

    def test_classify_incomplete(self) -> None:
        b = _classify(
            same_path=None,
            sep_max=None,
            speed_mean=None,
            ctrl={"n": 0},
            outcome_delta=None,
            incomplete=True,
        )
        self.assertEqual(b, "INCOMPLETE_EVIDENCE")

    def test_run_on_frozen_pilot_if_present(self) -> None:
        pilot = ROOT / "docs" / "runtime-evidence" / "r2-g4a-paired-pilot"
        if not (pilot / "run_set_report.json").is_file():
            self.skipTest("pilot evidence missing")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "diag"
            result = run_diagnostic(pilot_dir=pilot, out_dir=out)
            self.assertEqual(result["n"], 11)
            self.assertTrue((out / "pair_diagnostics.jsonl").is_file())
            self.assertTrue((out / "aggregate.json").is_file())
            self.assertTrue((out / "report.md").is_file())
            agg = result["aggregate"]
            self.assertEqual(agg["n_comparable_diagnosed"], 11)
            self.assertEqual(agg["same_spatial_path_count"], 11)
            self.assertIn("PROPOSAL_TEMPORAL_ONLY", agg["bottleneck_counts"])
            lines = (out / "pair_diagnostics.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 11)
            row = json.loads(lines[0])
            self.assertIn(row["primary_bottleneck"], {
                "PROPOSAL_TEMPORAL_ONLY",
                "PROPOSAL_SAME_PATH",
                "SPEED_PLANNER_COMPRESSION",
                "MPC_CONTROL_COMPRESSION",
                "ORACLE_DEADBAND_ONLY",
                "SCENE_INSENSITIVE",
                "INCOMPLETE_EVIDENCE",
            })


if __name__ == "__main__":
    unittest.main()
