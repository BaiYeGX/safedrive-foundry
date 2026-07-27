"""Offline regressions for R2 runner contracts (no CARLA / no VLA forward)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.comparability import (  # noqa: E402
    STATUS_INCOMPARABLE,
    evaluate_pair_comparability,
)
from driving_vla.evaluation.runner_contract import (  # noqa: E402
    PAIR_STATUS_COMPLETED,
    PAIR_STATUS_FAILED,
    ExpectedPairHashes,
    RunnerContractError,
    append_ledger_if_new,
    build_completed_manifest,
    finalize_branch_failure_codes,
    ledger_has_entry,
    ledger_path_for_evidence_root,
    next_attempt_id,
    plan_pair_attempt,
    require_frozen_registry,
    validate_frozen_registry_manifest,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    REGISTRY_SCHEMA,
    freeze_registry,
    load_scenario_registry,
)
from tests.g4.test_g4a_comparability import (  # noqa: E402
    _anchor,
    _branch,
    _measured,
)


FROZEN_MANIFEST = (
    ROOT
    / "docs"
    / "runtime-evidence"
    / "r2-g4a-paired-pilot"
    / "registry"
    / "registry_manifest.json"
)


class FrozenRegistryGateTest(unittest.TestCase):
    def test_require_frozen_ok(self) -> None:
        self.assertTrue(DEFAULT_REGISTRY_PATH.is_file())
        self.assertTrue(FROZEN_MANIFEST.is_file())
        reg, audit, path = require_frozen_registry(
            DEFAULT_REGISTRY_PATH,
            manifest_path=FROZEN_MANIFEST,
            repo_root=ROOT,
        )
        self.assertEqual(len(reg.fixtures), 12)
        self.assertTrue(audit["ok"])
        self.assertTrue(audit["frozen"])
        self.assertEqual(audit["n_pairs"], 12)
        self.assertEqual(path, FROZEN_MANIFEST)

    def test_hash_mismatch_fail_closed(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        man["registry_sha256"] = "0" * 64
        with self.assertRaises(RunnerContractError) as ctx:
            validate_frozen_registry_manifest(reg, man)
        self.assertIn("registry_sha256 mismatch", str(ctx.exception))

    def test_not_frozen_fail_closed(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        man = freeze_registry(reg).freeze_manifest()
        man["frozen"] = False
        with self.assertRaises(RunnerContractError) as ctx:
            validate_frozen_registry_manifest(reg, man)
        self.assertIn("frozen must be true", str(ctx.exception))

    def test_pair_count_mismatch_fail_closed(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        man["n_pairs"] = 11
        man["pairs"] = man["pairs"][:11]
        with self.assertRaises(RunnerContractError) as ctx:
            validate_frozen_registry_manifest(reg, man)
        self.assertIn("n_pairs", str(ctx.exception))

    def test_schema_mismatch_fail_closed(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        man["schema_version"] = "wrong.schema"
        with self.assertRaises(RunnerContractError) as ctx:
            validate_frozen_registry_manifest(reg, man)
        self.assertIn("schema_version", str(ctx.exception))
        self.assertEqual(REGISTRY_SCHEMA, reg.schema_version)

    def test_post_v1_frozen_registry_uses_manifest_pair_set(self) -> None:
        registry_path = (
            ROOT
            / "safedrive_foundry"
            / "config"
            / "g4a"
            / "scenario_registry_v2_blind.toml"
        )
        manifest_path = (
            ROOT
            / "docs"
            / "runtime-evidence"
            / "r2-spatial-k2-pilot-v5-blind"
            / "registry"
            / "registry_manifest.json"
        )
        reg, audit, _ = require_frozen_registry(
            registry_path,
            manifest_path=manifest_path,
            repo_root=ROOT,
        )
        self.assertNotEqual(reg.registry_version, "v1")
        self.assertEqual(len(reg.fixtures), 12)
        self.assertTrue(audit["ok"])


class AttemptIdempotencyTest(unittest.TestCase):
    def _expected(self, pair_id: str = "pair_abc") -> ExpectedPairHashes:
        return ExpectedPairHashes(
            pair_id=pair_id,
            scenario_id="lead_brake_moderate",
            seed_id="seed_a",
            registry_sha256="reghash",
            model_retimer_hash="modelhash",
            executor_config_hash="exhash",
            artifact_content_hash="arthash",
        )

    def _write_completed(self, attempt_dir: Path, expected: ExpectedPairHashes) -> dict:
        man = build_completed_manifest(
            pair_id=expected.pair_id,
            scenario_id=expected.scenario_id,
            seed_id=expected.seed_id,
            family="lead_braking",
            registry_sha256=expected.registry_sha256,
            model_retimer_hash=expected.model_retimer_hash,
            executor_config_hash=expected.executor_config_hash,
            artifact_content_hash=expected.artifact_content_hash or "arthash",
            attempt_id=0,
            branch_order=(0, 1),
            forward_count_total=1,
            comparable=True,
            comparability={"status": "COMPARABLE"},
            oracle={"pair_label": "TIE"},
            anchor={},
            branch_0={},
            branch_1={},
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "pair_manifest.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8"
        )
        return man

    def test_idempotent_read_when_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = self._expected()
            self._write_completed(root / exp.pair_id / "attempt_0", exp)
            plan = plan_pair_attempt(root, exp)
            self.assertEqual(plan.mode, "idempotent_read")
            self.assertEqual(plan.attempt_id, 0)
            self.assertIsNotNone(plan.existing_manifest)
            self.assertEqual(plan.existing_manifest["status"], PAIR_STATUS_COMPLETED)

    def test_new_attempt_when_hash_differs_keeps_old(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = self._expected()
            old_dir = root / exp.pair_id / "attempt_0"
            self._write_completed(old_dir, exp)
            old_man = json.loads((old_dir / "pair_manifest.json").read_text(encoding="utf-8"))
            exp2 = ExpectedPairHashes(
                pair_id=exp.pair_id,
                scenario_id=exp.scenario_id,
                seed_id=exp.seed_id,
                registry_sha256="OTHER_REG",
                model_retimer_hash=exp.model_retimer_hash,
                executor_config_hash=exp.executor_config_hash,
                artifact_content_hash=exp.artifact_content_hash,
            )
            plan = plan_pair_attempt(root, exp2)
            self.assertEqual(plan.mode, "new_run")
            self.assertEqual(plan.attempt_id, 1)
            self.assertEqual(plan.attempt_dir.name, "attempt_1")
            # old evidence untouched
            self.assertTrue((old_dir / "pair_manifest.json").is_file())
            self.assertEqual(
                json.loads((old_dir / "pair_manifest.json").read_text(encoding="utf-8")),
                old_man,
            )

    def test_failed_attempt_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = self._expected()
            fail_dir = root / exp.pair_id / "attempt_0"
            fail_dir.mkdir(parents=True)
            fail_man = {
                "status": PAIR_STATUS_FAILED,
                "pair_id": exp.pair_id,
                "scenario_id": exp.scenario_id,
                "seed_id": exp.seed_id,
                "registry_sha256": exp.registry_sha256,
                "model_retimer_hash": exp.model_retimer_hash,
                "executor_config_hash": exp.executor_config_hash,
                "attempt_id": 0,
                "error": "boom",
            }
            (fail_dir / "pair_manifest.json").write_text(
                json.dumps(fail_man), encoding="utf-8"
            )
            plan = plan_pair_attempt(root, exp)
            self.assertEqual(plan.mode, "new_run")
            self.assertEqual(plan.attempt_id, 1)
            # failed attempt retained
            self.assertEqual(
                json.loads((fail_dir / "pair_manifest.json").read_text(encoding="utf-8"))[
                    "status"
                ],
                PAIR_STATUS_FAILED,
            )

    def test_next_attempt_id_first_unoccupied(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pair_x"
            (root / "attempt_0").mkdir(parents=True)
            (root / "attempt_2").mkdir(parents=True)
            # first free id is 1 (gap), not max+1 — freeze planner semantics
            self.assertEqual(next_attempt_id(root), 1)
            (root / "attempt_1").mkdir()
            self.assertEqual(next_attempt_id(root), 3)

    def test_ledger_no_duplicate_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pairs"
            root.mkdir()
            ledger = ledger_path_for_evidence_root(root)
            self.assertEqual(ledger.name, "paired_outcomes.jsonl")
            row = {
                "pair_id": "p1",
                "attempt_id": 0,
                "artifact_content_hash": "a1",
                "pair_label": "TIE",
            }
            self.assertTrue(append_ledger_if_new(ledger, row))
            self.assertFalse(append_ledger_if_new(ledger, row))
            self.assertTrue(
                ledger_has_entry(
                    ledger, pair_id="p1", attempt_id=0, artifact_content_hash="a1"
                )
            )
            lines = [ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)
            # different attempt may append
            row2 = dict(row)
            row2["attempt_id"] = 1
            self.assertTrue(append_ledger_if_new(ledger, row2))
            lines = [ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)


class RunSetPlanTest(unittest.TestCase):
    def test_counterbalance_and_plan_length(self) -> None:
        from driving_vla.evaluation.runner_contract import (
            counterbalance_branch_order,
            plan_run_set_pairs,
            run_set_exit_code,
            summarize_run_set_results,
        )

        self.assertEqual(counterbalance_branch_order("seed_a"), (0, 1))
        self.assertEqual(counterbalance_branch_order("seed_b"), (1, 0))
        with self.assertRaises(RunnerContractError):
            counterbalance_branch_order("seed_z")

        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        plan = plan_run_set_pairs(reg)
        self.assertEqual(len(plan), 12)
        # registry order: each scenario seed_a then seed_b
        self.assertEqual(plan[0]["seed_id"], "seed_a")
        self.assertEqual(plan[0]["branch_order"], [0, 1])
        self.assertEqual(plan[1]["seed_id"], "seed_b")
        self.assertEqual(plan[1]["branch_order"], [1, 0])
        # all indices unique 0..11
        self.assertEqual([p["index"] for p in plan], list(range(12)))
        seeds_a = [p for p in plan if p["seed_id"] == "seed_a"]
        seeds_b = [p for p in plan if p["seed_id"] == "seed_b"]
        self.assertEqual(len(seeds_a), 6)
        self.assertEqual(len(seeds_b), 6)
        self.assertTrue(all(p["branch_order"] == [0, 1] for p in seeds_a))
        self.assertTrue(all(p["branch_order"] == [1, 0] for p in seeds_b))

        summary = summarize_run_set_results(
            [
                {"status": "OK", "comparable": True},
                {"status": "OK", "comparable": True, "idempotent_read": True},
                {"status": "OK", "comparable": False},
                {"status": "FAILED", "error": "x"},
            ],
            n_planned=12,
        )
        self.assertEqual(summary["n_ok"], 3)
        self.assertEqual(summary["n_fail"], 1)
        self.assertEqual(summary["n_comparable"], 2)
        self.assertEqual(summary["n_incomparable"], 1)
        self.assertEqual(summary["n_idempotent"], 1)
        self.assertEqual(run_set_exit_code(summary, min_comparable=10), 1)
        ok_summary = {
            "n_planned": 12,
            "n_ok": 12,
            "n_fail": 0,
            "n_comparable": 11,
        }
        self.assertEqual(run_set_exit_code(ok_summary, min_comparable=10), 0)
        gate = {"n_planned": 12, "n_ok": 12, "n_fail": 0, "n_comparable": 9}
        self.assertEqual(run_set_exit_code(gate, min_comparable=10), 4)

    def test_cmd_run_set_plan_only_no_carla(self) -> None:
        import importlib.util

        path = ROOT / "tests" / "g4" / "run_g4a_paired.py"
        spec = importlib.util.spec_from_file_location("run_g4a_paired_mod", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td)
            ns = type(
                "A",
                (),
                {
                    "registry": str(DEFAULT_REGISTRY_PATH),
                    "registry_manifest": str(FROZEN_MANIFEST),
                    "evidence_dir": str(evidence),
                    "plan_only": True,
                    "stop_on_fail": False,
                    "continue_policy": "",
                    "device": "cpu",
                    "host": "127.0.0.1",
                    "port": 2000,
                },
            )()
            code = mod.cmd_run_set(ns)
            self.assertEqual(code, 0)
            man_path = evidence / "run_set_manifest.json"
            self.assertTrue(man_path.is_file())
            plan_path = evidence / "run_set_plan.json"
            self.assertTrue(plan_path.is_file())
            man = json.loads(man_path.read_text(encoding="utf-8"))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertTrue(man["immutable"])
            self.assertEqual(man["n_pairs"], 12)
            self.assertEqual(len(man["pairs"]), 12)
            self.assertEqual(man["pairs"][0]["branch_order"], [0, 1])
            self.assertEqual(man["pairs"][1]["branch_order"], [1, 0])
            self.assertEqual(plan["mode"], "plan_only")
            self.assertEqual(len(plan["pairs"]), 12)
            # plan_only should not create attempt dirs
            self.assertFalse((evidence / "pairs").exists() and any((evidence / "pairs").iterdir()))


class CleanupBeforeReportTest(unittest.TestCase):
    def test_finalize_codes_include_cleanup_failure(self) -> None:
        codes = finalize_branch_failure_codes(cleanup_ok=False)
        self.assertIn("CLEANUP_FAILURE", codes)
        codes2 = finalize_branch_failure_codes(
            cleanup_ok=False, extra_codes=("SPAWN_FAILED",)
        )
        self.assertEqual(codes2[0], "SPAWN_FAILED")
        self.assertIn("CLEANUP_FAILURE", codes2)
        self.assertEqual(finalize_branch_failure_codes(cleanup_ok=True), ())

    def test_cleanup_failure_makes_incomparable(self) -> None:
        art = _anchor()
        ah = art.artifact_content_hash()
        b0 = _branch(
            0,
            anchor_hash=ah,
            cleanup_ok=False,
            failure_codes=("CLEANUP_FAILURE",),
        )
        b1 = _branch(1, anchor_hash=ah)
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=b0,
            branch1=b1,
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_INCOMPARABLE)
        self.assertIn("CLEANUP_FAILURE", r.failure_codes)


if __name__ == "__main__":
    unittest.main()
