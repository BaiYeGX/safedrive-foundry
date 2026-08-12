from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from r2_v4_validate_native_repairs import validate  # noqa: E402


class NativeRepairGateTest(unittest.TestCase):
    def _write(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_requires_both_three_pass_repairs_and_teacher_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route = self._write(
                root,
                "route.json",
                {
                    "case_id": "route_change_right",
                    "runs": [{"passed": True}, {"passed": True}, {"passed": True}],
                },
            )
            crossing = self._write(
                root,
                "crossing.json",
                {
                    "scenario_id": "left_turn_crossing_yield",
                    "runs": [{"completed": True}, {"completed": True}, {"completed": True}],
                },
            )
            teacher = self._write(
                root,
                "teacher.json",
                {"expected_cases": 16, "completed": 16, "failed": 0, "passed": True},
            )
            report = validate([route, crossing], teacher)
            self.assertTrue(report["passed"])
            self.assertTrue(report["gates"]["teacher_long_suite_16_of_16"])

    def test_any_failed_tail_blocks_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route = self._write(
                root,
                "route.json",
                {
                    "case_id": "ROUTE_CHANGE_RIGHT",
                    "runs": [{"passed": True}, {"passed": True}, {"passed": False}, {"passed": True}],
                },
            )
            crossing = self._write(
                root,
                "crossing.json",
                {
                    "case_id": "LEFT_TURN_CROSSING",
                    "consecutive_passes": 3,
                    "runs": [{"passed": True}, {"passed": True}, {"passed": True}],
                },
            )
            teacher = self._write(
                root,
                "teacher.json",
                {"expected_cases": 16, "completed": 16, "failed": 0},
            )
            report = validate([route, crossing], teacher)
            self.assertFalse(report["passed"])
            self.assertFalse(report["gates"]["route_change_right_three_consecutive"])


if __name__ == "__main__":
    unittest.main()

