"""Offline tests for R2-E repeat audit plan freeze (no CARLA / no outcomes)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.runner_contract import (  # noqa: E402
    REPEAT_AUDIT_SEED,
    RunnerContractError,
    build_repeat_audit_plan,
    classify_r2_closure,
    compare_repeat_label_consistency,
    ensure_repeat_audit_plan,
    require_frozen_registry,
    select_repeat_pair_indices,
)

DEFAULT_REGISTRY_PATH = (
    ROOT / "safedrive_foundry" / "config" / "g4a" / "scenario_registry_v1.toml"
)
FROZEN_MANIFEST = (
    ROOT
    / "docs"
    / "runtime-evidence"
    / "r2-g4a-paired-pilot"
    / "registry"
    / "registry_manifest.json"
)

MODEL_H = "model_retimer_test"
EXEC_H = "executor_test"


class RepeatAuditPlanTest(unittest.TestCase):
    def test_selection_deterministic_and_prefers_families(self) -> None:
        slots = [
            {
                "index": i,
                "scenario_id": f"s{i}",
                "seed_id": "seed_a" if i % 2 == 0 else "seed_b",
                "family": ["lead_braking", "cut_in", "crossing"][i % 3],
            }
            for i in range(12)
        ]
        a = select_repeat_pair_indices(
            slots, registry_sha256="abc", audit_seed=REPEAT_AUDIT_SEED
        )
        b = select_repeat_pair_indices(
            slots, registry_sha256="abc", audit_seed=REPEAT_AUDIT_SEED
        )
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)
        fams = {slots[i]["family"] for i in a}
        # Prefer distinct families when possible (12 slots span 3 families)
        self.assertEqual(len(fams), 2)

    def test_build_and_exclusive_reuse_no_reselect(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            reg, audit, _ = require_frozen_registry(
                DEFAULT_REGISTRY_PATH,
                manifest_path=FROZEN_MANIFEST,
                repo_root=ROOT,
            )
            plan = build_repeat_audit_plan(
                registry=reg,
                freeze_audit=audit,
                pairs_root=pairs_root,
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
                model_checkpoint_hash="ckpt",
                retimer_hash="rh",
            )
            self.assertTrue(plan["immutable"])
            self.assertTrue(plan["frozen_before_outcomes"])
            self.assertEqual(len(plan["pairs"]), 2)
            self.assertEqual(plan["audit_seed"], REPEAT_AUDIT_SEED)
            for p in plan["pairs"]:
                self.assertFalse(p["in_d_denominator"])
                self.assertGreater(p["repeat_attempt_id"], p["d_planned_attempt_id"])
                self.assertEqual(len(p["branch_order"]), 2)

            path = Path(td) / "repeat_audit_plan.json"
            m1, mode1 = ensure_repeat_audit_plan(path, plan)
            self.assertEqual(mode1, "created")
            # Second call with identity only — no reselect
            identity = {
                "registry_sha256": plan["registry_sha256"],
                "model_retimer_hash": MODEL_H,
                "model_checkpoint_hash": "ckpt",
                "retimer_hash": "rh",
                "executor_config_hash": EXEC_H,
                "audit_seed": REPEAT_AUDIT_SEED,
            }
            m2, mode2 = ensure_repeat_audit_plan(path, identity)
            self.assertEqual(mode2, "reused")
            self.assertEqual(m1["plan_content_hash"], m2["plan_content_hash"])
            self.assertEqual(m1["selected_indices"], m2["selected_indices"])
            # build_fn must not run
            def boom():
                raise AssertionError("must not rebuild")

            m3, mode3 = ensure_repeat_audit_plan(path, identity, build_fn=boom)
            self.assertEqual(mode3, "reused")
            # identity mismatch fail-closed
            bad = dict(identity)
            bad["model_retimer_hash"] = "OTHER"
            with self.assertRaises(RunnerContractError):
                ensure_repeat_audit_plan(path, bad)

    def test_compare_and_closure_rules(self) -> None:
        cmp_ok = compare_repeat_label_consistency(
            original={"pair_label": "TIE", "comparable": True},
            repeat={"pair_label": "TIE", "comparable": True},
        )
        self.assertTrue(cmp_ok["label_consistent"])
        cmp_bad = compare_repeat_label_consistency(
            original={"pair_label": "TIE", "comparable": True},
            repeat={"pair_label": "TOP1_BEST", "comparable": True},
        )
        self.assertFalse(cmp_bad["label_consistent"])

        c = classify_r2_closure(
            n_comparable=12,
            min_comparable=10,
            repeat_labels_all_consistent=True,
            n_repeat_done=2,
            pilot_label="NO_SELECTION_SPACE",
        )
        self.assertEqual(c["r2_status"], "COMPLETED_WITH_LIMITS")

        c2 = classify_r2_closure(
            n_comparable=8,
            min_comparable=10,
            repeat_labels_all_consistent=None,
            n_repeat_done=0,
            pilot_label="PILOT_INCONCLUSIVE",
            dominant_failure_class="carla",
        )
        self.assertEqual(c2["r2_status"], "BLOCKED_EXTERNAL")

        c3 = classify_r2_closure(
            n_comparable=11,
            min_comparable=10,
            repeat_labels_all_consistent=False,
            n_repeat_done=2,
            pilot_label="ENTER_WORLD",
        )
        self.assertEqual(c3["r2_status"], "PILOT_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
