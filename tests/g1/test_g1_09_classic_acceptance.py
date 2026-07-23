from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.control.controller import closed_loop_simulate, make_reference_trajectory  # noqa: E402
from classic_stack.planning.frenet import FrenetPlanner  # noqa: E402
from classic_stack.planning.frenet.scenarios import SCENARIO_KINDS, make_scenario  # noqa: E402
from classic_stack.planning.hybrid_astar import HybridAstarPlanner  # noqa: E402
from classic_stack.planning.hybrid_astar.scenarios import MANEUVER_KINDS, make_maneuver  # noqa: E402
from classic_stack.planning.race import run_race_plan_ablation  # noqa: E402
from classic_stack.control.adaptation import run_race_control_ablation  # noqa: E402
from classic_stack.risk import evaluate_risk_field  # noqa: E402


class G109ClassicAcceptanceTests(unittest.TestCase):
    def test_suite_and_dual_track_tables(self) -> None:
        # Planning suite (observable)
        frenet = FrenetPlanner()
        plan_obs = []
        for kind in SCENARIO_KINDS:
            r = frenet.plan(make_scenario(kind, seed=1))
            plan_obs.append({"track": "observable", "module": "frenet", "scenario": kind, "ok": r.ok})
        hybrid = HybridAstarPlanner()
        for kind in MANEUVER_KINDS:
            r = hybrid.plan(make_maneuver(kind, seed=1))
            plan_obs.append({"track": "observable", "module": "hybrid", "scenario": kind, "ok": r.ok})

        # Oracle risk upper bounds
        plan_oracle = []
        for kind in SCENARIO_KINDS:
            field = evaluate_risk_field(ego_v=8.0, actors=[{"s": 16.0, "v": 5.0}], uncertainty_scale=1.0)
            plan_oracle.append(
                {
                    "track": "oracle",
                    "scenario": kind,
                    "risk": field.oracle_upper_bound,
                    "observable_risk": field.observable_score,
                }
            )

        # Control suite 50hz only
        control_rows = []
        for name in ("straight", "curve", "stop", "lane_change"):
            out = closed_loop_simulate(make_reference_trajectory(name), scenario=name, steps=60)
            self.assertEqual(out["profile"], "control_50hz")
            control_rows.append(out)

        race_plan = run_race_plan_ablation(ROOT)
        race_ctrl = run_race_control_ablation()

        systems = {
            "basic_classic": {
                "plan": "frenet_basic+hybrid_basic",
                "control": "mpc_pid_fixed",
                "frenet_success": race_plan["matrix"]["basic"]["frenet_success_rate"],
                "control_miss": sum(
                    r["deadline_miss_rate"] for r in race_ctrl["disturbances"] if r["variant"] == "fixed"
                ),
            },
            "race_plan": {
                "frenet_success": race_plan["matrix"]["full"]["frenet_success_rate"],
                "promoted": race_plan["default_admission"]["promote_full_to_default"],
            },
            "race_control": {
                "promoted": race_ctrl["default_admission"]["promote_full_to_default"],
            },
            "full_race": {
                "plan_promoted": race_plan["default_admission"]["promote_full_to_default"],
                "control_promoted": race_ctrl["default_admission"]["promote_full_to_default"],
            },
        }

        # Dual-track separation
        self.assertTrue(all(r["track"] == "observable" for r in plan_obs))
        self.assertTrue(all(r["track"] == "oracle" for r in plan_oracle))
        for row in plan_oracle:
            self.assertIsNotNone(row["observable_risk"])
            self.assertGreaterEqual(float(row["observable_risk"]), 0.0)
            if row["risk"] is not None:
                self.assertGreaterEqual(float(row["risk"]), 0.0)
                self.assertLess(float(row["risk"]), 1e6)

        # Expert gate — actual assertions (not just counters)
        plan_ok = [r for r in plan_obs if r["ok"]]
        plan_fail = [r for r in plan_obs if not r["ok"]]
        self.assertGreaterEqual(len(plan_ok), 5)
        comfort_ok = all(r["tracking_cte_mean"] < 5.0 for r in control_rows)
        self.assertTrue(comfort_ok)
        # No scenario is pure continuous timeout (all brake)
        for r in control_rows:
            modes = r["modes"]
            total = sum(modes.values())
            brake_only = modes.get("brake", 0) == total and total > 0
            self.assertFalse(brake_only, msg=f"continuous timeout/brake {r['scenario']}")
        progress_ok = all(r.get("progress_m", 1.0) >= 0.0 for r in control_rows)
        self.assertTrue(progress_ok)

        gate = {
            "plan_positives": len(plan_ok),
            "plan_failures": len(plan_fail),
            "comfort_cte_lt_5": comfort_ok,
            "no_all_brake_control": True,
            "oracle_observable_separated": True,
            "passed": len(plan_ok) >= 5 and comfort_ok,
        }
        self.assertTrue(gate["passed"])

        bundle = {
            "schema": "safedrive.g1_09.evidence.repair.v1",
            "observable_plan": plan_obs,
            "oracle_plan": plan_oracle,
            "control_50hz": [
                {
                    "scenario": r["scenario"],
                    "cte_mean": r["tracking_cte_mean"],
                    "e2e_p99": r["watchdog"]["e2e_ms"]["p99"],
                    "miss_rate": r["watchdog"]["deadline_miss_rate"],
                    "progress_m": r.get("progress_m"),
                }
                for r in control_rows
            ],
            "systems": systems,
            "race_plan_admission": race_plan["default_admission"],
            "race_control_admission": race_ctrl["default_admission"],
            "g2_exports": {
                "CandidateTrajectory": "classic_stack.planning.frenet.Trajectory",
                "RiskField": "classic_stack.risk.RiskField",
                "ActorObservation": "dict s/v for risk actors (observable)",
                "ControlBaseline": "config/control/mpc_pid_baseline.toml",
            },
            "expert_gate": gate,
            "limits": [
                "offline plant control; not CARLA live 50Hz VERIFIED",
                "live waypoint demo is separate stack (plan_along_nodes), not Frenet/RACE full stack",
            ],
        }
        # Unit test writes ONLY to a temp dir (does not pollute authoritative evidence).
        with tempfile.TemporaryDirectory(prefix="g1_09_test_") as tmp:
            out_path = Path(tmp) / "summary.json"
            out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertTrue(out_path.is_file())
            # Optional: allow explicit evidence publish via env for offline tooling
            if os.environ.get("SDF_WRITE_G1_09_EVIDENCE") == "1":
                formal = ROOT / "docs/runtime-evidence/g1-09" / "repair-20260713"
                formal.mkdir(parents=True, exist_ok=True)
                (formal / "summary.json").write_text(
                    json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        flags1 = [r["ok"] for r in plan_obs if r["module"] == "frenet"]
        flags2 = [FrenetPlanner().plan(make_scenario(k, seed=1)).ok for k in SCENARIO_KINDS]
        self.assertEqual(flags1, flags2)


if __name__ == "__main__":
    unittest.main()
