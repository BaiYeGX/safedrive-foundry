"""Counts-first formal X5H classification."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "r2x_live_force_smoke_v2",
    ROOT / "scripts" / "r2x_live_force_smoke_v2.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _dual() -> dict:
    return {
        "status": "PASS",
        "defensive_available": True,
        "forces_run": [0, 1],
        "path_diverge": True,
    }


class X5HGateTest(unittest.TestCase):
    def test_mpc_timeout_cannot_pass_branch_gate(self) -> None:
        self.assertTrue(
            MODULE.branch_mpc_ok(
                {"mpc_solved": 50, "mpc_timeout": 0, "mpc_fallback": 0}
            )
        )
        self.assertFalse(
            MODULE.branch_mpc_ok(
                {"mpc_solved": 49, "mpc_timeout": 1, "mpc_fallback": 0}
            )
        )

    def test_two_dual_plus_fail_closed_collapse_passes_with_limits(self) -> None:
        status, allowed = MODULE.classify_x5h_results(
            [
                _dual(),
                _dual(),
                {
                    "status": "FAILED",
                    "error": "guard_not_ok_v2: SPATIAL_COLLAPSE_ELIGIBLE",
                },
            ],
            formal=True,
        )
        self.assertEqual(status, "X5H_PASS_WITH_LIMITS")
        self.assertTrue(allowed)

    def test_rpc_or_execution_failure_never_authorizes(self) -> None:
        status, allowed = MODULE.classify_x5h_results(
            [_dual(), _dual(), {"status": "FAILED", "error": "RPC timeout"}],
            formal=True,
        )
        self.assertEqual(status, "SMOKE_PARTIAL")
        self.assertFalse(allowed)

    def test_development_smoke_never_authorizes_r2k(self) -> None:
        status, allowed = MODULE.classify_x5h_results(
            [_dual(), _dual(), _dual()],
            formal=False,
        )
        self.assertEqual(status, "SMOKE_ALL_PAIRS_RAN_WITH_SOME_DUAL")
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
