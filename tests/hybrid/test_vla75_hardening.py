"""Fail-closed and training-contract regressions for H6 VLA75."""

from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import H3_CANDIDATE_DIM, H3_CANDIDATE_STEPS, H3_CONTEXT_DIM, stable_sha256  # noqa: E402
from data_pipeline.h6.acceptance import evaluate_vla75_gate  # noqa: E402
from data_pipeline.h6.matrix import load_h6_vla75_matrix  # noqa: E402
from data_pipeline.h6.model import (  # noqa: E402
    WorldVLA75Model,
    event_aware_preference_consistency_loss,
    group_dro_weights,
)
from data_pipeline.h6.calibration import select_vla75_router_config  # noqa: E402
from data_pipeline.h6.run_lock import (  # noqa: E402
    build_h6_vla75_run_lock,
    verify_run_lock,
    verify_summary_checkpoints_against_lock,
)
from data_pipeline.h6.lineage import (  # noqa: E402
    all_formal_lineages_failed,
    assert_formal_lineage_available,
    formal_lineage_state_path,
    frozen_run_lock_identity,
    read_formal_lineage_state,
    record_formal_lineage_result,
)
from data_pipeline.h6.runtime import TemporalPreferenceStabilizer  # noqa: E402
from tests.hybrid.test_vla75_contract import _runs  # noqa: E402


class VLA75FailClosedTest(unittest.TestCase):
    def test_formal_lineage_baseline_audit_is_bound_before_hash_checks(self):
        """A first formal pair must not read an uninitialized/stale baseline list."""

        result = evaluate_vla75_gate(_runs(), lineage_id="a")
        self.assertFalse(result["passed"])
        self.assertIn("provenance", result["failures"])
        self.assertIn("run_lock_missing", result["provenance_failures"])

    def test_missing_head_is_in_denominator_and_provenance_failure(self):
        rows = _runs()
        del rows[0]["decisions"][0]["world_score"]["predictions"][0]["preference_utility"]
        result = evaluate_vla75_gate(rows)
        self.assertFalse(result["passed"])
        self.assertIn("provenance", result["failures"])
        self.assertEqual(result["coverage"]["all_decision_ticks"], 600)
        self.assertEqual(result["raw_gate_reasons"]["failed_tick_counts"]["pair_completeness"], 1)

    def test_unknown_orphan_and_non_track_bindings_fail_closed(self):
        for mutate in (
            lambda decision: decision.update(
                applied_candidate_id="frame:orphan",
                applied_source="vla",
                applied_candidate_source="vla",
                safety_executed_candidate_id="frame:orphan",
                safety_executed_source="vla",
            ),
            lambda decision: decision.update(
                applied_source="mrm",
                applied_candidate_source="mrm",
                applied_candidate_id=None,
                applied_mode="TRACK_APPROVED",
            ),
            lambda decision: decision.update(applied_mode="EMERGENCY_BRAKE"),
        ):
            rows = copy.deepcopy(_runs())
            mutate(rows[0]["decisions"][0])
            result = evaluate_vla75_gate(rows)
            self.assertFalse(result["passed"])
            self.assertIn("provenance", result["failures"])

    def test_executed_and_applied_id_mismatch_is_not_silent(self):
        rows = copy.deepcopy(_runs())
        decision = rows[0]["decisions"][0]
        decision.update(
            applied_candidate_id="frame:expert",
            applied_source="expert",
            applied_candidate_source="expert",
        )
        result = evaluate_vla75_gate(rows)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("executed_applied_id_mismatch" in item for item in result["provenance_failures"])
        )


class VLA75RuntimeAndTrainingTest(unittest.TestCase):
    def test_router_calibration_replays_tick_sequence_instead_of_copying_applied_labels(self):
        rows = []
        for tick in range(16):
            observed = "vla" if tick < 8 else "expert"
            vla_score = 2.0 if observed == "vla" else 0.0
            expert_score = 0.0 if observed == "vla" else 2.0
            rows.append(
                {
                    "pair_id": "sequence-a",
                    "sequence_id": "sequence-a:on",
                    "tick": tick,
                    "applied_source": observed,
                    "vla_prediction": {
                        "deployment_score": vla_score,
                        "preference_utility": vla_score,
                        "trust_probability": 0.95,
                        "unsafe_probability": 0.01,
                    },
                    "expert_prediction": {
                        "deployment_score": expert_score,
                        "preference_utility": expert_score,
                        "trust_probability": 0.95,
                        "unsafe_probability": 0.01,
                    },
                    "vla_unsafe": False,
                    "expert_unsafe": False,
                }
            )
        result = select_vla75_router_config(rows)
        self.assertTrue(result.passed)
        self.assertEqual(result.rows, 16)
        self.assertEqual(result.sequences, 1)
        self.assertAlmostEqual(result.observed_vla_coverage, 0.5)
        # The selected grid point is a replay result: it smooths the reversal
        # and therefore cannot simply copy the 8/8 observed source labels.
        self.assertNotEqual(result.vla_coverage, result.observed_vla_coverage)
        self.assertGreaterEqual(result.switches, 1)

    def test_temporal_stabilizer_holds_then_breaks_on_event(self):
        stabilizer = TemporalPreferenceStabilizer()
        first = stabilizer.update(
            {"vla": 1.0, "expert": 0.9},
            raw_preferred_candidate_id="vla",
            raw_preferred_source="vla",
            candidate_sources={"vla": "vla", "expert": "expert"},
        )
        held = stabilizer.update(
            {"vla": 0.90, "expert": 1.0},
            raw_preferred_candidate_id="expert",
            raw_preferred_source="expert",
            candidate_sources={"vla": "vla", "expert": "expert"},
        )
        self.assertEqual(first[1], "vla")
        self.assertEqual(held[1], "vla")
        broken = stabilizer.update(
            {"vla": 0.1, "expert": 1.5},
            raw_preferred_candidate_id="expert",
            raw_preferred_source="expert",
            candidate_sources={"vla": "vla", "expert": "expert"},
            event_break=True,
        )
        self.assertEqual(broken[1], "expert")
        self.assertEqual(stabilizer.metrics()["switches"], 1)

    def test_group_dro_and_event_loss_are_finite(self):
        weights = group_dro_weights(
            __import__("torch").tensor([1.0, 3.0, 0.5]),
            ["Town01|free_flow|early", "Town03|red|event", "Town01|free_flow|early"],
        )
        self.assertTrue(bool(__import__("torch").isfinite(weights).all()))
        self.assertGreater(float(weights[1]), float(weights[0]))
        preference = __import__("torch").tensor([0.0, 1.0, 3.0])
        loss = event_aware_preference_consistency_loss(
            preference, event_boundary=[False, True, False]
        )
        self.assertEqual(float(loss), 0.0)

    def test_vla75_model_has_fourteen_source_blind_outputs(self):
        import torch

        model = WorldVLA75Model(d_model=32, layers=1, heads=4, ffn=64)
        context = torch.zeros(2, H3_CONTEXT_DIM)
        candidate = torch.zeros(2, H3_CANDIDATE_STEPS, H3_CANDIDATE_DIM)
        output = model(context, candidate)
        self.assertEqual(tuple(output.shape), (2, 14))


class VLA75RunLockTest(unittest.TestCase):
    def test_run_lock_binds_pre_registered_matrix_content(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            for index in range(3):
                path = root / f"seed-{index}.pt"
                path.write_bytes(bytes([index + 1]))
                checkpoints.append(path)
            lock = build_h6_vla75_run_lock(
                ROOT,
                lineage_id="a",
                dataset_id="h6-vla75-hardening",
                matrix_rows=load_h6_vla75_matrix("a", full=False),
                checkpoint_paths=checkpoints,
                calibration={"passed": True},
                scoped_paths=[],
            )
            self.assertTrue(verify_run_lock(lock)["valid"])
            tampered = dict(lock)
            tampered["matrix_pair_ids"] = list(reversed(tampered["matrix_pair_ids"]))
            tampered["lock_sha256"] = stable_sha256(
                {key: value for key, value in tampered.items() if key != "lock_sha256"}
            )
            self.assertFalse(verify_run_lock(tampered)["valid"])

    def test_summary_ensemble_order_and_model_hash_are_bound(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            for index in range(3):
                path = root / f"seed-{index}.pt"
                path.write_bytes(bytes([index + 11]))
                checkpoints.append(path)
            lock = build_h6_vla75_run_lock(
                ROOT,
                lineage_id="a",
                dataset_id="h6-vla75-summary-binding",
                matrix_rows=load_h6_vla75_matrix("a", full=False),
                checkpoint_paths=checkpoints,
                calibration={"passed": True},
                scoped_paths=[],
            )
            summary_models = [
                {
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in checkpoints
            ]
            self.assertTrue(
                verify_summary_checkpoints_against_lock(
                    summary_models, lock, root=ROOT
                )["valid"]
            )
            reordered = list(reversed(summary_models))
            result = verify_summary_checkpoints_against_lock(
                reordered, lock, root=ROOT
            )
            self.assertFalse(result["valid"])
            self.assertIn("checkpoint_order_mismatch", result["failures"])


class VLA75LineageStateTest(unittest.TestCase):
    def test_failed_pilot_is_immutable_and_blocks_full(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = record_formal_lineage_result(
                root,
                "a",
                scope="pilot",
                passed=False,
                dataset_id="h6-vla75-lineage-pilot",
                run_lock_sha256="lock-a",
                evidence_path="pilot/final-delivery.json",
                evidence_sha256="evidence-a",
                gate_result={"ok": False, "failures": ["world_vla_preference"]},
                recorded_wall_time_s=1.0,
            )
            self.assertEqual(state["status"], "PILOT_FAILED")
            self.assertEqual(read_formal_lineage_state(root, "a")["state_sha256"], state["state_sha256"])
            with self.assertRaisesRegex(RuntimeError, "lineage_frozen"):
                assert_formal_lineage_available(
                    root, "a", scope="full", run_lock_sha256="lock-a"
                )
            with self.assertRaisesRegex(RuntimeError, "lineage_frozen"):
                record_formal_lineage_result(
                    root,
                    "a",
                    scope="pilot",
                    passed=True,
                    dataset_id="h6-vla75-lineage-retry",
                    run_lock_sha256="lock-new",
                )

    def test_full_requires_passing_pilot_and_same_lock(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "requires_passing_pilot"):
                record_formal_lineage_result(
                    root,
                    "b",
                    scope="full",
                    passed=False,
                    dataset_id="h6-vla75-lineage-full",
                    run_lock_sha256="lock-b",
                )
            record_formal_lineage_result(
                root,
                "b",
                scope="pilot",
                passed=True,
                dataset_id="h6-vla75-lineage-pilot",
                run_lock_sha256="lock-b",
            )
            with self.assertRaisesRegex(RuntimeError, "run_lock_mismatch"):
                assert_formal_lineage_available(
                    root, "b", scope="full", run_lock_sha256="other-lock"
                )
            completed = record_formal_lineage_result(
                root,
                "b",
                scope="full",
                passed=True,
                dataset_id="h6-vla75-lineage-full",
                run_lock_sha256="lock-b",
            )
            self.assertEqual(completed["status"], "COMPLETED")
            with self.assertRaisesRegex(RuntimeError, "lineage_completed"):
                assert_formal_lineage_available(root, "b", scope="full", run_lock_sha256="lock-b")

    def test_pilot_full_identity_excludes_matrix_scope_and_dataset(self):
        pilot = {
            "lineage_id": "a",
            "config_sha256": "config",
            "model_hash": "model",
            "calibration": {"deployment": {"trust_threshold": 0.5}},
            "scoped_runtime_sha256": {"module.py": "abc"},
            "dataset_id": "h6-vla75-pilot",
            "matrix_pairs": 12,
            "matrix_scope": "pilot",
            "matrix_sha256": "pilot-matrix",
            "matrix_pair_ids": ["p1"],
        }
        full = {
            **pilot,
            "dataset_id": "h6-vla75-full",
            "matrix_pairs": 108,
            "matrix_scope": "full",
            "matrix_sha256": "full-matrix",
            "matrix_pair_ids": ["p1", "p2"],
        }
        self.assertEqual(frozen_run_lock_identity(pilot), frozen_run_lock_identity(full))

    def test_all_lineages_failed_only_after_three_terminal_failures(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(all_formal_lineages_failed(root))
            for lineage in ("a", "b"):
                record_formal_lineage_result(
                    root,
                    lineage,
                    scope="pilot",
                    passed=False,
                    dataset_id=f"h6-vla75-{lineage}",
                    run_lock_sha256=f"lock-{lineage}",
                )
            self.assertFalse(all_formal_lineages_failed(root))
            record_formal_lineage_result(
                root,
                "c",
                scope="pilot",
                passed=False,
                dataset_id="h6-vla75-c",
                run_lock_sha256="lock-c",
            )
            self.assertTrue(all_formal_lineages_failed(root))
            self.assertTrue(formal_lineage_state_path(root, "a").is_file())


if __name__ == "__main__":
    unittest.main()
