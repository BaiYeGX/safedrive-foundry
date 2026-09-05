"""Regression tests for the frozen H6-CORA C2 data contract."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import (  # noqa: E402
    ActorInitialState,
    ResetSignature,
    compare_reset_signatures,
    stable_sha256,
)
from data_pipeline.h2.live_contract import trajectory_sha256  # noqa: E402
from data_pipeline.h6.cora.config import (  # noqa: E402
    CORA_C2_CONFIG,
    CORA_C2_CONFIG_SHA256,
)
from data_pipeline.h6.cora.contracts import (  # noqa: E402
    FEATURE_SCHEMA,
    CoraBranchOutcome,
    CoraPairEdge,
    CoraProposal,
    CoraRootRecord,
    OutcomeValue,
)
from data_pipeline.h6.cora.feature import (  # noqa: E402
    FORBIDDEN_KEY_TOKENS,
    build_cora_feature_view,
    validate_cora_feature_view,
)
from data_pipeline.h6.cora.interventions import (  # noqa: E402
    FAMILY_OPERATORS,
    derive_interventions,
)
from data_pipeline.h6.cora.loader import load_cora_roots  # noqa: E402
from data_pipeline.h6.cora.matrix import (  # noqa: E402
    CORA_DATA_MATRIX,
    CORA_FORMAL_MATRIX,
    CORA_MATRIX_SHA256,
)
from data_pipeline.h6.cora.outcomes import (  # noqa: E402
    PUBLIC_DERIVATION_VERSION,
    PUBLIC_OUTCOME_HEADS,
    derive_public_outcome_heads,
    validate_public_outcome_heads,
)
from data_pipeline.h6.cora.quality import ALL_OPERATORS, audit_cora_dataset  # noqa: E402
from data_pipeline.h6.cora.run_lock import (  # noqa: E402
    SOURCE_AMENDMENT_SCHEMA,
    SOURCE_AMENDMENT_SCOPE,
    build_run_lock,
    source_identity,
    verify_run_lock,
    verify_source_amendment,
)
from data_pipeline.h6.cora.store import (  # noqa: E402
    CoraDataStore,
    pending_collection_rows,
)
from driving_vla.hybrid.contracts import GuardResult, GuardVerdict  # noqa: E402
from safety_kernel.contracts.types import TrackedObject, TrafficLightObs  # noqa: E402
from tests.hybrid.test_h1_candidates import make_anchor, make_generated_set  # noqa: E402


H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64


def _proposal(proposal_id: str, digest: str, source: str) -> CoraProposal:
    trajectory = tuple(
        {
            "t": 0.25 * (index + 1),
            "x": float(index + 1),
            "y": 0.0,
            "yaw": 0.0,
            "kappa": 0.0,
            "v": 2.0,
            "a": 0.0,
            "jerk": 0.0,
        }
        for index in range(10)
    )
    return CoraProposal(
        proposal_id=proposal_id,
        proposal_sha256=digest,
        root_id="root",
        kind="nominal",
        trajectory=trajectory,
        guard={"verdict": "PASS"},
        audit_source=source,
    )


def _heads() -> dict[str, OutcomeValue]:
    return {
        "progress": OutcomeValue(4.0, "m", True),
        "collision": OutcomeValue(False, "bool", True),
        "repair_success": OutcomeValue(None, "bool", False),
    }


def _branch(
    proposal_id: str,
    proposal_hash: str,
    *,
    ticks: int = 50,
    cleanup: bool = True,
    comparable: bool = True,
) -> CoraBranchOutcome:
    return CoraBranchOutcome(
        root_id="root",
        proposal_id=proposal_id,
        proposal_sha256=proposal_hash,
        split="train",
        guard_verdict="PASS",
        reset={"comparable": comparable},
        safety_input_id=proposal_id,
        safety_input_sha256=proposal_hash,
        pre_repair_id=proposal_id,
        pre_repair_sha256=proposal_hash,
        executable_id=f"exec:{proposal_id}",
        executable_sha256=H2,
        applied_id=f"exec:{proposal_id}",
        applied_sha256=H2,
        decision_kind="ACCEPT",
        applied_mode="TRACK_APPROVED",
        repair_attempted=False,
        repair_success=None,
        would_require_cross_candidate_fallback=False,
        ticks_executed=ticks,
        terminal_reason="HORIZON_COMPLETE",
        cleanup_complete=cleanup,
        heads=_heads(),
    )


class CoraMatrixAndConfigTests(unittest.TestCase):
    def test_frozen_matrix_split_counts_formal_isolation_and_balance(self) -> None:
        self.assertEqual(len(CORA_DATA_MATRIX), 351)
        self.assertEqual(len(CORA_FORMAL_MATRIX), 108)
        self.assertEqual(len({row.root_id for row in CORA_DATA_MATRIX + CORA_FORMAL_MATRIX}), 459)
        counts = {
            split: sum(row.split == split for row in CORA_DATA_MATRIX)
            for split in ("coverage_pilot", "train", "validation", "calibration", "locked_development")
        }
        self.assertEqual(
            counts,
            {
                "coverage_pilot": 27,
                "train": 162,
                "validation": 54,
                "calibration": 54,
                "locked_development": 54,
            },
        )
        development = [row for row in CORA_DATA_MATRIX if row.split != "coverage_pilot"]
        self.assertEqual(sum(row.expert_slot == 0 for row in development), 162)
        self.assertEqual(sum(row.expert_slot == 1 for row in development), 162)
        self.assertTrue(all(not row.collect for row in CORA_FORMAL_MATRIX))
        self.assertEqual(len(CORA_C2_CONFIG_SHA256), 64)
        self.assertEqual(len(CORA_MATRIX_SHA256), 64)

    def test_frozen_seeds_do_not_overlap_historical_or_each_other(self) -> None:
        split_seeds = CORA_C2_CONFIG["splits"]
        values = [int(seed) for seeds in split_seeds.values() for seed in seeds]
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue({0, 1, 2, 3, 89, 97, 101, 103, 107, 109, 113, 127, 131}.isdisjoint(values))

    def test_dedicated_runtime_profile_is_frozen(self) -> None:
        from runtime import load_runtime_profiles

        profiles = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")
        profile = profiles["cora_data"]
        self.assertEqual(profile.fixed_delta_seconds, 0.05)
        self.assertEqual(profile.validate(), [])


class CoraFeatureTests(unittest.TestCase):
    def test_feature_is_physical_allow_list_and_source_blind(self) -> None:
        generated, _runtime = make_generated_set()
        candidate = generated.candidates[0].candidate
        original = build_cora_feature_view(generated.anchor, candidate)
        mutated = build_cora_feature_view(
            generated.anchor,
            replace(
                candidate,
                source=generated.candidates[1].candidate.source,
                dynamics_meta={
                    **dict(candidate.dynamics_meta),
                    "source": "mutated",
                    "slot": 1,
                    "branch_order": ["vla", "expert"],
                    "outcome": {"winner": "vla"},
                },
            ),
        )
        self.assertEqual(original, mutated)
        encoded = json.dumps(original, sort_keys=True).lower()
        for token in FORBIDDEN_KEY_TOKENS:
            self.assertNotIn(token, encoded)

    def test_every_nested_forbidden_token_fails_closed(self) -> None:
        base = {
            "schema_version": FEATURE_SCHEMA,
            "observable": {"ego": {"x": 0.0}, "route": []},
            "trajectory": [],
        }
        for token in FORBIDDEN_KEY_TOKENS:
            payload = json.loads(json.dumps(base))
            payload["observable"]["nested"] = {f"prefix_{token}_suffix": 1}
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "cora_feature_forbidden"):
                validate_cora_feature_view(payload)

    def test_mapping_builder_drops_unknown_privileged_fields(self) -> None:
        trajectory = _proposal("p", H0, "expert").trajectory
        feature = build_cora_feature_view(
            {
                "ego": {"x": 0.0, "y": 0.0},
                "route": [[0.0, 0.0], [10.0, 0.0]],
                "oracle": {"winner": "expert"},
                "actor_future": [{"x": 5.0}],
            },
            trajectory,
        )
        encoded = json.dumps(feature, sort_keys=True).lower()
        self.assertNotIn("oracle", encoded)
        self.assertNotIn("actor_future", encoded)


class CoraInterventionTests(unittest.TestCase):
    @staticmethod
    def _rich_fixture():
        generated, _runtime = make_generated_set()
        anchor = generated.anchor
        snapshot = replace(
            anchor.safety_snapshot,
            actors=(
                TrackedObject("actor", "vehicle", 8.0, 0.4, 0.0, 0.5, 0.0, 4.5, 1.8, anchor.simulation_time_s),
            ),
            traffic_lights=(
                TrafficLightObs("light", "red", 8.0, anchor.simulation_time_s, 8.0, True),
            ),
        )
        anchor = replace(anchor, safety_snapshot=snapshot)
        items = []
        for item in generated.candidates:
            points = tuple(
                replace(
                    point,
                    x=0.5 * (index + 1),
                    y=0.0,
                    yaw=0.0,
                    v=max(0.2, 3.0 - 0.25 * index),
                    a=-1.0,
                    kappa=0.0,
                )
                for index, point in enumerate(item.candidate.points)
            )
            items.append(replace(item, candidate=replace(item.candidate, points=points)))
        return anchor, tuple(items)

    def test_all_seven_operators_are_deterministic_finite_and_bounded(self) -> None:
        anchor, candidates = self._rich_fixture()
        seen = set()
        identities = []
        for family in FAMILY_OPERATORS:
            first = derive_interventions(f"root-{family}", family, anchor, candidates)
            second = derive_interventions(f"root-{family}", family, anchor, candidates)
            self.assertLessEqual(len(first), 2)
            self.assertEqual({row.base_source for row in first}, {"expert", "vla"})
            self.assertEqual(
                [(row.base_source, row.operator, row.status, None if row.candidate is None else trajectory_sha256(row.candidate.candidate.points)) for row in first],
                [(row.base_source, row.operator, row.status, None if row.candidate is None else trajectory_sha256(row.candidate.candidate.points)) for row in second],
            )
            for row in first:
                if row.candidate is None:
                    continue
                seen.add(row.operator)
                proposal = row.to_proposal()
                self.assertIsNotNone(proposal)
                assert proposal is not None
                self.assertEqual(len(proposal.trajectory), 10)
                self.assertEqual(proposal.base_proposal_sha256, trajectory_sha256(next(item.candidate.points for item in candidates if item.provenance.source.value == row.base_source)))
                self.assertTrue(all(float(point["v"]) <= 15.0 + 1e-9 for point in proposal.trajectory))
                identities.append(proposal.proposal_sha256)
        self.assertEqual(seen, set(ALL_OPERATORS))
        self.assertTrue(identities)

    def test_not_applicable_is_explicit_and_does_not_create_fake_proposal(self) -> None:
        generated, _runtime = make_generated_set()
        constant = tuple(
            replace(
                item,
                candidate=replace(
                    item.candidate,
                    points=tuple(replace(point, v=2.0, a=0.0) for point in item.candidate.points),
                ),
            )
            for item in generated.candidates
        )
        results = derive_interventions("root-na", "slow_lead", generated.anchor, constant)
        delayed = next(row for row in results if row.operator == "delayed_brake")
        self.assertEqual(delayed.status, "NOT_APPLICABLE")
        self.assertIsNone(delayed.candidate)
        self.assertIsNone(delayed.to_proposal())

    def test_guard_reject_is_auxiliary_only_but_review_is_core(self) -> None:
        anchor, candidates = self._rich_fixture()

        class FixedGuard:
            def __init__(self, verdict: GuardVerdict) -> None:
                self.verdict = verdict

            def evaluate_candidate(self, _cset, item):
                return GuardResult(item.candidate.candidate_id, self.verdict, (), () if self.verdict is not GuardVerdict.REJECT else ("forced",), 0.0)

        rejected = derive_interventions("root-r", "free_flow", anchor, candidates, guard=FixedGuard(GuardVerdict.REJECT))
        self.assertTrue(all(row.to_proposal() is not None and row.to_proposal().auxiliary_only for row in rejected))
        reviewed = derive_interventions("root-v", "free_flow", anchor, candidates, guard=FixedGuard(GuardVerdict.REVIEW))
        self.assertTrue(all(row.to_proposal() is not None and not row.to_proposal().auxiliary_only for row in reviewed))


class CoraOutcomeContractTests(unittest.TestCase):
    def test_null_mask_never_becomes_zero(self) -> None:
        missing = OutcomeValue(None, "s", False)
        self.assertIsNone(missing.to_dict()["value"])
        with self.assertRaises(ValueError):
            OutcomeValue(0.0, "s", False)

    def test_complete_pair_requires_two_valid_identity_bound_branches(self) -> None:
        left, right = _proposal("left", H0, "expert"), _proposal("right", H1, "vla")
        left_branch, right_branch = _branch("left", H0), _branch("right", H1)
        edge = CoraPairEdge("root", "left", "right", H0, H1, "nominal", True)
        record = CoraRootRecord(
            dataset_id="dataset",
            root_id="root",
            split="train",
            scenario={"map_name": "Town01"},
            matrix_sha256=CORA_MATRIX_SHA256,
            config_sha256=CORA_C2_CONFIG_SHA256,
            anchor_path="anchors/root.json",
            anchor_sha256=H2,
            feature_paths={},
            feature_sha256={},
            proposals=(left, right),
            branches=(left_branch, right_branch),
            edges=(edge,),
            vla_forward_count=1,
            terminal_status="VALID_NOMINAL_PAIR",
        )
        self.assertTrue(record.nominal_pair_outcome_mask)
        self.assertTrue(left_branch.outcome_valid)

    def test_incomplete_cleanup_reset_and_cross_fallback_fail_closed(self) -> None:
        self.assertFalse(_branch("left", H0, ticks=49).outcome_valid)
        self.assertFalse(_branch("left", H0, cleanup=False).outcome_valid)
        self.assertFalse(_branch("left", H0, comparable=False).outcome_valid)
        self.assertFalse(replace(_branch("left", H0), would_require_cross_candidate_fallback=True).outcome_valid)
        left, right = _proposal("left", H0, "expert"), _proposal("right", H1, "vla")
        with self.assertRaisesRegex(ValueError, "masked_edge_invalid_branch"):
            CoraRootRecord(
                dataset_id="dataset",
                root_id="root",
                split="train",
                scenario={},
                matrix_sha256=CORA_MATRIX_SHA256,
                config_sha256=CORA_C2_CONFIG_SHA256,
                anchor_path="a",
                anchor_sha256=H2,
                feature_paths={},
                feature_sha256={},
                proposals=(left, right),
                branches=(_branch("left", H0, ticks=49), _branch("right", H1)),
                edges=(CoraPairEdge("root", "left", "right", H0, H1, "nominal", True),),
                vla_forward_count=1,
                terminal_status="INVALID",
            )

    def test_applied_hash_chain_is_mandatory(self) -> None:
        valid = _branch("left", H0)
        self.assertTrue(valid.identity_valid)
        self.assertFalse(replace(valid, applied_sha256=H1).identity_valid)
        self.assertFalse(replace(valid, pre_repair_sha256=H1).identity_valid)

    def test_all_safety_and_legal_terminal_modes_are_explicit(self) -> None:
        tracked = _branch("left", H0)
        for kind in ("ACCEPT", "QP", "RATO"):
            with self.subTest(kind=kind):
                branch = replace(
                    tracked,
                    decision_kind=kind,
                    repair_attempted=kind in {"QP", "RATO"},
                    repair_success=True if kind in {"QP", "RATO"} else None,
                )
                self.assertTrue(branch.identity_valid)
                self.assertTrue(branch.legal_terminal)
        for kind, mode in (
            ("MINIMAL_RISK", "MINIMAL_RISK_BRAKE"),
            ("EMERGENCY", "EMERGENCY_BRAKE"),
            ("HARD_REJECT", "HOLD_NO_EXEC"),
        ):
            with self.subTest(kind=kind):
                branch = replace(
                    tracked,
                    decision_kind=kind,
                    applied_mode=mode,
                    executable_id=None,
                    executable_sha256=None,
                    applied_id=None,
                    applied_sha256=None,
                    pre_repair_id=None,
                    pre_repair_sha256=None,
                    terminal_reason="MRM_STANDSTILL",
                    ticks_executed=10,
                )
                self.assertTrue(branch.identity_valid)
                self.assertTrue(branch.legal_terminal)
        self.assertTrue(replace(tracked, terminal_reason="ROUTE_COMPLETE", ticks_executed=1).legal_terminal)
        self.assertTrue(replace(tracked, terminal_reason="COLLISION_TERMINAL", ticks_executed=1).legal_terminal)
        self.assertFalse(replace(tracked, terminal_reason="MRM_STANDSTILL", ticks_executed=9).legal_terminal)
        self.assertFalse(replace(tracked, terminal_reason="RUNTIME_FAILURE").legal_terminal)

    def test_public_heads_are_complete_typed_and_do_not_invent_repair_failure(self) -> None:
        branch = _branch("left", H0).to_dict()
        timeline = [
            {
                "tick": index,
                "speed_mps": float(index),
                "route_progress_m": float(index),
                "corridor_distance_m": 2.0 if index >= 2 else 0.2,
                "lateral_acceleration_mps2": float(index) / 10.0,
                "deadline_miss": index == 3,
            }
            for index in range(5)
        ]
        events = [
            {
                "event_type": "collision",
                "other_actor_id": 17,
                "impulse_x": 3.0,
                "impulse_y": 4.0,
                "impulse_z": 0.0,
            }
        ]
        heads = derive_public_outcome_heads(
            branch,
            timeline=timeline,
            events=events,
            corridor_half_width_m=1.0,
            dt_s=0.05,
        )
        self.assertEqual(set(heads), set(PUBLIC_OUTCOME_HEADS))
        self.assertEqual(heads["collision_count"]["value"], 1)
        self.assertEqual(heads["max_collision_impulse"]["value"], 5.0)
        self.assertEqual(heads["collision_other_actor_id"]["value"], 17)
        self.assertAlmostEqual(heads["off_corridor_duration_s"]["value"], 0.15)
        self.assertEqual(heads["controller_deadline_misses"]["value"], 1)
        self.assertFalse(heads["repair_success"]["valid"])
        self.assertIsNone(heads["repair_success"]["value"])
        self.assertFalse(heads["repair_mode"]["valid"])
        self.assertTrue(heads["acceleration_p95_mps2"]["valid"])
        self.assertTrue(heads["jerk_p95_mps3"]["valid"])
        self.assertTrue(
            all(
                item["derivation_version"] == PUBLIC_DERIVATION_VERSION
                for item in heads.values()
            )
        )
        validate_public_outcome_heads(heads)

    def test_public_repair_mode_is_bound_to_actual_qp_or_rato(self) -> None:
        timeline = [
            {
                "speed_mps": float(index),
                "route_progress_m": float(index),
                "corridor_distance_m": 0.0,
                "lateral_acceleration_mps2": 0.0,
                "deadline_miss": False,
            }
            for index in range(4)
        ]
        for kind, mode in (("QP", "LONGITUDINAL_QP"), ("RATO", "RATO_SCP")):
            branch = replace(
                _branch("left", H0),
                decision_kind=kind,
                repair_attempted=True,
                repair_success=True,
            ).to_dict()
            heads = derive_public_outcome_heads(
                branch,
                timeline=timeline,
                events=[],
                corridor_half_width_m=1.0,
            )
            self.assertEqual(heads["repair_mode"]["value"], mode)
            self.assertTrue(heads["repair_success"]["value"])

    def test_reset_threshold_boundaries_are_inclusive_and_excess_fails(self) -> None:
        base = ResetSignature(
            actors=(ActorInitialState("ego", 0.0, 0.0, 0.0, 0.0),),
            route_sha256=H0,
            weather_sha256=H0,
            light_sha256=H0,
            script_sha256=H0,
        )
        boundary = replace(
            base,
            actors=(ActorInitialState("ego", 0.05, 0.0, 0.5, 0.10),),
        )
        self.assertTrue(compare_reset_signatures(base, boundary).comparable)
        for actor in (
            ActorInitialState("ego", 0.0501, 0.0, 0.5, 0.10),
            ActorInitialState("ego", 0.05, 0.0, 0.5001, 0.10),
            ActorInitialState("ego", 0.05, 0.0, 0.5, 0.1001),
        ):
            with self.subTest(actor=actor):
                self.assertFalse(
                    compare_reset_signatures(base, replace(base, actors=(actor,))).comparable
                )


class CoraStoreRunLockAndAuditTests(unittest.TestCase):
    def test_resume_resource_projection_counts_only_pending_roots(self) -> None:
        rows = tuple(SimpleNamespace(root_id=f"root-{index}") for index in range(5))

        class FakeStore:
            @staticmethod
            def has_valid_root(root_id: str) -> bool:
                return root_id in {"root-0", "root-2", "root-4"}

        pending = pending_collection_rows(FakeStore(), rows)
        self.assertEqual([row.root_id for row in pending], ["root-1", "root-3"])
        self.assertEqual(4 * len(pending), 8)

    def test_store_is_atomic_deduplicated_immutable_and_manifest_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CoraDataStore(Path(tmp), "dataset")
            path, digest = store.write_image(b"image")
            self.assertEqual(store.write_image(b"image"), (path, digest))
            target = store.root / "anchors" / "root.json"
            self.assertEqual(store.write_immutable_json(target, {"value": 1}), store.write_immutable_json(target, {"value": 1}))
            with self.assertRaises(FileExistsError):
                store.write_immutable_json(target, {"value": 2})
            store.write_manifest()
            self.assertTrue(store.verify_manifest()[0])
            (store.root / "unexpected.bin").write_bytes(b"unexpected")
            valid, failures = store.verify_manifest()
            self.assertFalse(valid)
            self.assertTrue(any("unexpected_artifact" in failure for failure in failures))

    def test_run_lock_self_hash_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = build_run_lock(
                root,
                environment={"carla": "0.9.16"},
                model={"sha256": H0},
                component_hashes={"guard": H1},
                disk={"free_gib": 100.0},
            )
            self.assertTrue(verify_run_lock(lock)[0])
            tampered = json.loads(json.dumps(lock))
            tampered["resources"]["root_attempt_limit"] = 999
            valid, failures = verify_run_lock(tampered)
            self.assertFalse(valid)
            self.assertIn("run_lock_self_hash", failures)

    def test_source_amendment_binds_both_identities_and_exact_file_hash(self) -> None:
        locked_source = {"tracked_diff_sha256": H0, "untracked_source": []}
        current_source = {"tracked_diff_sha256": H1, "untracked_source": []}
        relative = "safedrive_foundry/data_pipeline/h6/cora/live.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_text("amended source", encoding="utf-8")
            after = hashlib.sha256(target.read_bytes()).hexdigest()
            run_lock = {"run_lock_sha256": H2, "source": locked_source}
            amendment = {
                "schema_version": SOURCE_AMENDMENT_SCHEMA,
                "authorized_scope": SOURCE_AMENDMENT_SCOPE,
                "run_lock_sha256": H2,
                "base_source_identity_sha256": stable_sha256(locked_source),
                "amended_source_identity_sha256": stable_sha256(current_source),
                "files": {
                    relative: {"before_sha256": H0, "after_sha256": after}
                },
            }
            amendment["evidence_sha256"] = stable_sha256(amendment)
            with mock.patch(
                "data_pipeline.h6.cora.run_lock.source_identity",
                return_value=current_source,
            ):
                self.assertTrue(verify_source_amendment(root, run_lock, amendment)[0])
            with mock.patch(
                "data_pipeline.h6.cora.run_lock.source_identity",
                return_value={"tracked_diff_sha256": H2, "untracked_source": []},
            ):
                valid, failures = verify_source_amendment(root, run_lock, amendment)
                self.assertFalse(valid)
                self.assertIn("source_amendment_current_identity", failures)

    def test_preexisting_registry_is_excluded_from_source_identity(self) -> None:
        identity = source_identity(ROOT)
        included = {row["path"] for row in identity["untracked_source"]}
        self.assertNotIn("test_registry.sqlite3", included)
        if (ROOT / "test_registry.sqlite3").exists():
            self.assertIn("test_registry.sqlite3", identity["excluded_preexisting_or_outputs"])

    def test_loader_rejects_calibration_and_locked_development_for_training(self) -> None:
        path = Path("/tmp") / str(CORA_C2_CONFIG["dataset_id"])
        for split in ("calibration", "locked_development", "coverage_pilot"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "training_split_forbidden"):
                load_cora_roots(path, splits=(split,), purpose="training")

    def test_sparse_empty_dataset_fails_quality_gate_without_count_inflation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CoraDataStore(Path(tmp), str(CORA_C2_CONFIG["dataset_id"]))
            store.write_manifest()
            report = audit_cora_dataset(store.root, scope="pilot")
            self.assertFalse(report["passed"])
            self.assertEqual(report["metrics"]["terminal_roots"], 0)
            self.assertEqual(report["metrics"]["valid_nominal_pairs"], 0)
            self.assertIn("pilot_terminal_roots", report["failures"])

    def test_collector_has_no_forbidden_tick_or_connection_bypass(self) -> None:
        collector = (ROOT / "scripts/h6_cora_collect.py").read_text(encoding="utf-8")
        live = (ROOT / "safedrive_foundry/data_pipeline/h6/cora/live.py").read_text(encoding="utf-8")
        combined = collector + live
        for forbidden in ("world" + ".tick(", "carla" + ".Client(", "load" + "_world("):
            self.assertNotIn(forbidden, combined)
        self.assertNotIn("data_pipeline.h2.oracle", combined)
        self.assertNotIn("data_pipeline.h3.model", combined)
        self.assertIn("ScenarioRuntime", live)
        self.assertIn("candidate_set_size\": 1", live)

    def test_cora_uses_opt_in_kinematic_settle_tick_for_reset_identity(self) -> None:
        shared = (ROOT / "scripts/h5_collect.py").read_text(encoding="utf-8")
        live = (ROOT / "safedrive_foundry/data_pipeline/h6/cora/live.py").read_text(encoding="utf-8")
        self.assertIn("kinematic_settle_ticks: int = 0", shared)
        self.assertIn("terminal_progress = initial_progress + len(history)", shared)
        self.assertEqual(live.count("kinematic_settle_ticks=1"), 2)


if __name__ == "__main__":
    unittest.main()
