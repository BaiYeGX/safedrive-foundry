from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.control.adaptation import run_race_control_ablation, stable_seed  # noqa: E402
from classic_stack.control import load_control_config, config_sha256  # noqa: E402


class G108RaceControlTests(unittest.TestCase):
    def test_stable_seed_process_independent(self) -> None:
        a = stable_seed("full", "noise", "curve", base=2)
        b = stable_seed("full", "noise", "curve", base=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, stable_seed("fixed", "noise", "curve", base=2))

    def test_ablation_four_variants(self) -> None:
        out = run_race_control_ablation()
        self.assertEqual(set(out["variants"]), {"fixed", "warm", "adaptive", "full"})
        self.assertEqual(out["baseline_control_hash"], config_sha256(load_control_config().raw_toml))
        for var in out["variants"]:
            self.assertIn("straight", out["matrix"][var]["scenarios"])
            scen = out["matrix"][var]["scenarios"]["straight"]
            self.assertEqual(scen["variant"], var)
            self.assertIn("lateral_err", scen)
            self.assertIn("deadline_miss_rate", scen)
            self.assertIn("p99", scen["watchdog"]["e2e_ms"])
            # Measured fields must be finite non-negative rates
            self.assertGreaterEqual(scen["lateral_err"], 0.0)
            self.assertGreaterEqual(scen["deadline_miss_rate"], 0.0)
        self.assertTrue(out["disturbances"])
        for row in out["disturbances"]:
            self.assertIn("lateral_err", row)
            self.assertIn("raw_runs", row)
        self.assertIn("promote_full_to_default", out["default_admission"])
        self.assertIn("honesty", out)


if __name__ == "__main__":
    unittest.main()
