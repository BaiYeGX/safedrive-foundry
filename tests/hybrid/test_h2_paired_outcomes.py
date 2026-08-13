from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import (
    ActorInitialState,
    BranchOutcome,
    CandidateSnapshot,
    OracleVerdict,
    PairRecord,
    PairTerminalStatus,
    ResetComparison,
    ResetSignature,
    ScenarioKey,
    compare_reset_signatures,
    h3_feature_view,
)
from data_pipeline.h2.matrix import FIXED_MATRIX, MATRIX_SHA256, PILOT_MATRIX
from data_pipeline.h2.config import H2_CONFIG_SHA256
from data_pipeline.h2.gpu import _parse_query
from data_pipeline.h2.live_contract import trajectory_sha256
from data_pipeline.h2.oracle import label_pair
from data_pipeline.h2.quality import audit_h2_gate
from data_pipeline.h2.store import PairedOutcomeStore
from driving_vla.model.canonicalizer import stable_sha256 as canonical_sha256
from safety_kernel.contracts.types import TrajectoryPoint


H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64


def reset_signature(*, x: float = 1.0, yaw: float = 2.0, speed: float = 3.0) -> ResetSignature:
    return ResetSignature(
        actors=(ActorInitialState("ego", x, 4.0, yaw, speed),),
        route_sha256=H0,
        weather_sha256=H1,
        light_sha256=H2,
        script_sha256="3" * 64,
    )


def comparison(*, comparable: bool = True) -> ResetComparison:
    return ResetComparison(comparable, () if comparable else ("reset",), 0.0, 0.0, 0.0)


def branch(candidate_id: str, candidate_hash: str, **changes: object) -> BranchOutcome:
    values = {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_hash,
        "reset": comparison(),
        "safety_executed": True,
        "safety_input_id": candidate_id,
        "safety_final_id": f"final:{candidate_id}",
        "safety_executed_id": f"final:{candidate_id}",
        "applied_id": f"final:{candidate_id}",
        "pre_binding_trajectory_sha256": candidate_hash,
        "post_binding_trajectory_sha256": candidate_hash,
        "ticks_executed": 50,
        "cleanup_complete": True,
        "collision_count": 0,
        "red_light_violation": False,
        "off_corridor_duration_s": 0.0,
        "route_completed": False,
        "route_progress_m": 10.0,
        "jerk_rms_mps3": 2.0,
    }
    values.update(changes)
    return BranchOutcome(**values)  # type: ignore[arg-type]


def candidate(candidate_id: str, digest: str, source: str, slot: int) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=candidate_id,
        canonical_sha256=digest,
        source=source,
        slot=slot,
        trajectory=({"t": 0.25, "x": 1.0, "y": 2.0, "v": 3.0},),
        guard={"verdict": "PASS", "reject_reasons": []},
        provenance={"canonicalizer_version": "strict-v2"},
    )


def pair_record(dataset_id: str = "dataset-test") -> PairRecord:
    expert = candidate("candidate-a", H0, "expert", 0)
    vla = candidate("candidate-b", H1, "vla", 1)
    left, right = branch(expert.candidate_id, expert.canonical_sha256), branch(vla.candidate_id, vla.canonical_sha256)
    return PairRecord(
        dataset_id=dataset_id,
        scenario=ScenarioKey("Town01", "free_flow", 0, "ClearNoon"),
        matrix_sha256=MATRIX_SHA256,
        anchor={
            "observation_id": "obs",
            "carla_frame": 1,
            "simulation_time_s": 2.0,
            "selection_space": "DISTINCT",
            "swap_invariant": True,
            "routing": {"source": "expert", "slot": 0},
            "observable_snapshot": {
                "ego_x": 0.0,
                "actors": [{"actor_id": "7", "x": 1.0, "source": "observable"}],
            },
        },
        observable_history=({"frame": 1, "ego_x": 0.0},),
        route=((0.0, 0.0), (20.0, 0.0)),
        candidates=(expert, vla),
        terminal_status=PairTerminalStatus.VALID_PAIR,
        branch_order=(expert.candidate_id, vla.candidate_id),
        branches=(left, right),
        vla_forward_count=1,
        capture_reset=reset_signature(),
        label=label_pair(left, right),
    )


class H2ContractsTests(unittest.TestCase):
    def test_fixed_matrix_is_complete_frozen_and_balanced(self) -> None:
        self.assertEqual(len(FIXED_MATRIX), 120)
        self.assertEqual(len(PILOT_MATRIX), 15)
        self.assertEqual(len({row.pair_id for row in FIXED_MATRIX}), 120)
        self.assertEqual(sum(row.expert_slot == 0 for row in FIXED_MATRIX), 60)
        self.assertEqual(len(MATRIX_SHA256), 64)

    def test_reset_comparison_inclusive_limits_and_exact_hashes(self) -> None:
        base = reset_signature()
        at_limit = reset_signature(x=1.05, yaw=2.5, speed=3.1)
        result = compare_reset_signatures(base, at_limit)
        self.assertTrue(result.comparable)
        outside = compare_reset_signatures(base, reset_signature(x=1.050001))
        self.assertFalse(outside.comparable)
        self.assertIn("position_delta_exceeded", outside.reasons)
        mismatched = replace(at_limit, light_sha256="9" * 64)
        self.assertIn("light_sha256_mismatch", compare_reset_signatures(base, mismatched).reasons)

    def test_h3_feature_view_physically_excludes_privileged_fields(self) -> None:
        pair = pair_record()
        original = h3_feature_view(pair, "candidate-a")
        mutated = replace(
            pair,
            candidates=(replace(pair.candidates[0], source="vla", slot=1), pair.candidates[1]),
            branch_order=tuple(reversed(pair.branch_order)),
            branches=tuple(reversed(pair.branches)),
        )
        self.assertEqual(original, h3_feature_view(mutated, "candidate-a"))
        text = json.dumps(original, sort_keys=True).lower()
        for forbidden in ("source", "slot", "branch_order", "actor_future", "oracle", "outcome", "winner"):
            self.assertNotIn(forbidden, text)

    def test_pair_content_hash_is_deterministic(self) -> None:
        first = pair_record()
        second = pair_record()
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.to_dict()["content_sha256"], first.content_sha256)
        self.assertEqual(first.config_sha256, H2_CONFIG_SHA256)

    def test_execution_trajectory_hash_matches_h1_canonical_payload(self) -> None:
        points = tuple(
            TrajectoryPoint(0.25 * (index + 1), float(index), 1.0, 0.1, 0.01, 2.0, 0.2, 0.0)
            for index in range(10)
        )
        expected = canonical_sha256(tuple((p.x, p.y, p.yaw, p.v, p.a, p.kappa) for p in points))
        self.assertEqual(trajectory_sha256(points), expected)


class H2OfflineOracleTests(unittest.TestCase):
    def test_incomplete_is_invalid_and_both_unsafe_unresolved(self) -> None:
        good = branch("a", H0)
        incomplete = replace(branch("b", H1), ticks_executed=49)
        self.assertEqual(label_pair(good, incomplete).verdict, OracleVerdict.INVALID_PAIR)
        unsafe_a = replace(good, collision_count=1)
        unsafe_b = replace(branch("b", H1), off_corridor_duration_s=0.250001)
        self.assertEqual(label_pair(unsafe_a, unsafe_b).verdict, OracleVerdict.UNRESOLVED)

    def test_hard_safety_completion_progress_comfort_and_tie_rules(self) -> None:
        left, right = branch("a", H0), branch("b", H1)
        self.assertEqual(label_pair(replace(left, collision_count=1), right).winner_candidate_id, "b")
        self.assertEqual(label_pair(replace(left, route_completed=True), right).reason, "route_completion")
        self.assertEqual(label_pair(replace(left, route_progress_m=11.0), right).reason, "route_progress")
        comfort = replace(left, route_progress_m=9.75, jerk_rms_mps3=1.0)
        self.assertEqual(label_pair(comfort, right).reason, "comfort_jerk")
        too_slow = replace(comfort, route_progress_m=9.749)
        self.assertEqual(label_pair(too_slow, right).verdict, OracleVerdict.TIE)

    def test_slot_source_and_branch_order_permutation_are_invariant(self) -> None:
        left = replace(branch("physical-a", H0), route_progress_m=12.0)
        right = branch("physical-b", H1)
        normal = label_pair(left, right)
        swapped = label_pair(right, left)
        self.assertEqual(normal, swapped)
        self.assertEqual(normal.winner_candidate_id, "physical-a")


class H2StoreTests(unittest.TestCase):
    def test_atomic_parquet_resume_content_addressing_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PairedOutcomeStore(Path(tmp), "dataset-test")
            image_a = store.write_image(b"not-a-real-png-but-content-addressed")
            image_b = store.write_image(b"not-a-real-png-but-content-addressed")
            self.assertEqual(image_a, image_b)
            timeline_path, timeline_hash = store.write_timeline(
                "pair-a", "candidate-a", ({"frame": i, "x": float(i)} for i in range(50))
            )
            self.assertEqual(len(timeline_hash), 64)
            store.write_event_rows("pair-a", "candidate-a", [])
            store.write_actor_future("pair-a", "candidate-a", [{"frame": 1, "actor_id": 7, "x": 2.0}])
            path = store.write_pair(pair_record())
            store.write_label(
                "Town01__free_flow__s0__ClearNoon",
                {"verdict": "TIE", "winner_candidate_id": None},
            )
            self.assertEqual(store.write_pair(pair_record()), path)
            self.assertTrue(store.has_valid_pair("Town01__free_flow__s0__ClearNoon"))
            valid, reasons = store.verify_manifest()
            self.assertTrue(valid, reasons)
            merged = list(store.iter_labeled_pair_dicts())
            self.assertEqual(merged[0]["label"]["verdict"], "TIE")
            artifact = store.root / timeline_path
            artifact.write_bytes(artifact.read_bytes() + b"corrupt")
            valid, reasons = store.verify_manifest()
            self.assertFalse(valid)
            self.assertTrue(any("artifact_hash_mismatch" in reason for reason in reasons))
            extra = store.root / "unexpected.bin"
            extra.write_bytes(b"unexpected")
            valid, reasons = store.verify_manifest()
            self.assertFalse(valid)
            self.assertIn("unexpected_artifact:unexpected.bin", reasons)

    def test_gpu_query_parser_is_device_level_and_deterministic(self) -> None:
        samples = _parse_query(
            "0, GPU-uuid, NVIDIA GeForce RTX 4080, 16376, 4444\n",
            monotonic_s=10.0,
            elapsed_s=0.1,
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].uuid, "GPU-uuid")
        self.assertAlmostEqual(samples[0].used_gib, 4444 / 1024)

    def test_label_batch_can_defer_manifest_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PairedOutcomeStore(Path(tmp), "dataset-test")
            store.write_manifest()
            store.write_label(
                "Town01__free_flow__s0__ClearNoon",
                {"verdict": "TIE", "winner_candidate_id": None},
                update_manifest=False,
            )
            valid, reasons = store.verify_manifest()
            self.assertFalse(valid)
            self.assertIn("unexpected_artifact:labels/Town01__free_flow__s0__ClearNoon.parquet", reasons)
            store.write_manifest()
            valid, reasons = store.verify_manifest()
            self.assertTrue(valid, reasons)


class H2QualityGateTests(unittest.TestCase):
    def test_full_gate_accepts_balanced_synthetic_contract_rows(self) -> None:
        rows = []
        for index, matrix in enumerate(FIXED_MATRIX):
            valid = True
            expert_id, vla_id = f"e-{index}", f"v-{index}"
            winner = expert_id if index % 2 == 0 else vla_id
            rows.append(
                {
                    "terminal_status": "VALID_PAIR" if valid else "INELIGIBLE",
                    "config_sha256": H2_CONFIG_SHA256,
                    "scenario": matrix.scenario.to_dict(),
                    "anchor": {"selection_space": "DISTINCT", "swap_invariant": True},
                    "candidates": [
                        {"candidate_id": expert_id, "source": "expert", "slot": matrix.expert_slot, "guard": {"verdict": "PASS"}},
                        {"candidate_id": vla_id, "source": "vla", "slot": 1 - matrix.expert_slot, "guard": {"verdict": "PASS"}},
                    ],
                    "branches": [
                        {
                            "candidate_id": expert_id,
                            "safety_input_id": expert_id,
                            "safety_executed_id": f"final-{expert_id}",
                            "applied_id": f"final-{expert_id}",
                            "pre_binding_trajectory_sha256": H0,
                            "post_binding_trajectory_sha256": H0,
                        },
                        {
                            "candidate_id": vla_id,
                            "safety_input_id": vla_id,
                            "safety_executed_id": f"final-{vla_id}",
                            "applied_id": f"final-{vla_id}",
                            "pre_binding_trajectory_sha256": H1,
                            "post_binding_trajectory_sha256": H1,
                        },
                    ],
                    "vla_forward_count": 1,
                    "label": (
                        {"verdict": "CANDIDATE_WIN", "winner_candidate_id": winner}
                        if index < 30
                        else {"verdict": "TIE", "winner_candidate_id": None}
                    ),
                }
            )
        report = audit_h2_gate(rows, scope="full", manifest_valid=True, dataset_bytes=1024, whole_gpu_peak_gb=2.0)
        self.assertTrue(report.passed, report.failures)

    def test_pilot_gate_fails_closed(self) -> None:
        report = audit_h2_gate([], scope="pilot")
        self.assertFalse(report.passed)
        self.assertIn("terminal_count", report.failures)

    def test_live_collector_has_no_oracle_or_tick_master_escape(self) -> None:
        collector = (ROOT / "scripts" / "h2_collect.py").read_text(encoding="utf-8")
        self.assertNotIn("data_pipeline.h2.oracle", collector)
        self.assertNotIn("world.tick(", collector)
        self.assertNotIn("load_world(", collector)
        self.assertNotIn("carla.Client(", collector)


if __name__ == "__main__":
    unittest.main()
