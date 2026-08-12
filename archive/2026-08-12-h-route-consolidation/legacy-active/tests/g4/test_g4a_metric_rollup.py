"""Regression: close-r2 metric rollup maps real BranchOutcomeMetrics field names."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

# Load CLI module (not a package)
_path = ROOT / "tests" / "g4" / "run_g4a_paired.py"
_spec = importlib.util.spec_from_file_location("run_g4a_paired_metric_mod", _path)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class MetricFieldMapTest(unittest.TestCase):
    def test_extract_real_branch_outcome_fields(self) -> None:
        metrics = {
            "collision_episode_count": 0,
            "offroad_fraction": 0.0,
            "minimum_ttc_s": 12.5,
            "minimum_actor_clearance_m": 3.2,
            "route_progress_delta_m": 4.1,
            "jerk_abs_p95": 1.5,
        }
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "collision_episode_count"), 0.0
        )
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "offroad_fraction"), 0.0
        )
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "minimum_ttc_s"), 12.5
        )
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "minimum_actor_clearance_m"), 3.2
        )
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "route_progress_delta_m"), 4.1
        )
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "jerk_abs_p95"), 1.5
        )

    def test_legacy_aliases_still_work(self) -> None:
        metrics = {
            "had_collision": 1.0,
            "min_ttc_s": 2.0,
            "min_clearance_m": 1.0,
            "progress_m": 5.0,
            "jerk_p95": 0.5,
            "offroad_fraction": 0.1,
        }
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "collision_episode_count"), 1.0
        )
        self.assertEqual(mod.extract_branch_metric_value(metrics, "minimum_ttc_s"), 2.0)
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "minimum_actor_clearance_m"), 1.0
        )
        self.assertEqual(
            mod.extract_branch_metric_value(metrics, "route_progress_delta_m"), 5.0
        )
        self.assertEqual(mod.extract_branch_metric_value(metrics, "jerk_abs_p95"), 0.5)

    def test_classify_carla_timeout_hyphenated(self) -> None:
        failures = [
            {
                "error": (
                    "time-out of 5000ms while waiting for the simulator, "
                    "make sure the simulator is ready and connected to 127.0.0.1:2000"
                )
            }
        ]
        self.assertEqual(mod.classify_dominant_failure_class(failures), "carla")

    def test_rollup_from_synthetic_branch_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs = Path(td) / "pairs"
            pid = "abc123"
            adir = pairs / pid / "attempt_0"
            for br, ttc, progress in (
                ("branch-0", 10.0, 2.0),
                ("branch-1", 8.0, 2.5),
            ):
                bdir = adir / br
                bdir.mkdir(parents=True)
                (bdir / "branch_summary.json").write_text(
                    json.dumps(
                        {
                            "mpc_solved": 50,
                            "mpc_timeout": 0,
                            "mpc_fallback": 0,
                            "cleanup_ok": True,
                            "metrics": {
                                "collision_episode_count": 0,
                                "offroad_fraction": 0.0,
                                "minimum_ttc_s": ttc,
                                "minimum_actor_clearance_m": 4.0,
                                "route_progress_delta_m": progress,
                                "jerk_abs_p95": 1.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            (adir / "anchor").mkdir(parents=True)
            (adir / "anchor" / "run_config.json").write_text(
                json.dumps({"latency_ms": 100.0, "peak_vram_mb": 2000.0}),
                encoding="utf-8",
            )
            slots = [{"pair_id": pid, "planned_attempt_id": 0}]
            results = [
                {
                    "pair_id": pid,
                    "attempt_id": 0,
                    "status": "COMPLETED",
                    "comparable": True,
                }
            ]
            roll = mod._rollup_d_metrics(pairs, slots, results)
            samples = roll["outcome_metric_samples"]
            self.assertEqual(samples["collision_episode_count"]["n"], 2)
            self.assertEqual(samples["offroad_fraction"]["n"], 2)
            self.assertEqual(samples["minimum_ttc_s"]["n"], 2)
            self.assertEqual(samples["minimum_ttc_s"]["min"], 8.0)
            self.assertEqual(samples["minimum_ttc_s"]["max"], 10.0)
            self.assertEqual(samples["minimum_actor_clearance_m"]["n"], 2)
            self.assertEqual(samples["route_progress_delta_m"]["n"], 2)
            self.assertEqual(samples["jerk_abs_p95"]["n"], 2)
            self.assertEqual(roll["n_branch_summaries"], 2)
            self.assertEqual(roll["mpc"]["solved_ticks"], 100)
            self.assertIsNone(roll["dominant_failure_class"])
            self.assertEqual(roll["forward_latency_ms"]["n"], 1)

    def test_frozen_evidence_branch_metrics_nonempty_when_present(self) -> None:
        """If frozen pilot Evidence exists, rollup must not report n=0 for real fields."""
        evidence = (
            ROOT
            / "docs"
            / "runtime-evidence"
            / "r2-g4a-paired-pilot"
        )
        man_path = evidence / "run_set_manifest.json"
        rep_path = evidence / "run_set_report.json"
        if not man_path.is_file() or not rep_path.is_file():
            self.skipTest("frozen R2 pilot Evidence not present")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        slots = man.get("pairs") or []
        results = rep.get("pair_results") or []
        if len(slots) != 12 or len(results) != 12:
            self.skipTest("run-set not length 12")
        roll = mod._rollup_d_metrics(evidence / "pairs", slots, results)
        samples = roll["outcome_metric_samples"]
        # 11 comparable × 2 branches = 22 summaries when both branches finished
        self.assertEqual(roll["n_branch_summaries"], 22)
        n_ttc = samples["minimum_ttc_s"]["n"]
        n_clear = samples["minimum_actor_clearance_m"]["n"]
        n_prog = samples["route_progress_delta_m"]["n"]
        n_jerk = samples["jerk_abs_p95"]["n"]
        n_off = samples["offroad_fraction"]["n"]
        n_coll = samples["collision_episode_count"]["n"]
        # Always-present runtime fields must cover all completed branches
        self.assertEqual(n_clear, 22, msg=f"clearance n={n_clear}")
        self.assertEqual(n_prog, 22, msg=f"progress n={n_prog}")
        self.assertEqual(n_jerk, 22, msg=f"jerk n={n_jerk}")
        self.assertEqual(n_off, 22, msg=f"offroad n={n_off}")
        self.assertEqual(n_coll, 22, msg=f"collision n={n_coll}")
        # TTC may be null when no privileged near-actor sample; still must map when present
        self.assertGreater(n_ttc, 0, msg=f"minimum_ttc_s n={n_ttc}")
        self.assertLessEqual(n_ttc, 22)
        if roll["n_failed_or_incomparable_rows"] > 0:
            self.assertEqual(roll["dominant_failure_class"], "carla")


if __name__ == "__main__":
    unittest.main()
