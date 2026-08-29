"""Fail-closed and training-contract regressions for H6 VLA75."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import H3_CANDIDATE_DIM, H3_CANDIDATE_STEPS, H3_CONTEXT_DIM, stable_sha256  # noqa: E402
from data_pipeline.h6.acceptance import evaluate_vla75_gate  # noqa: E402
from data_pipeline.h6.matrix import (  # noqa: E402
    H6_TRAIN_SEEDS,
    load_h6_training_matrix,
    load_h6_vla75_matrix,
)
from data_pipeline.h6.model import (  # noqa: E402
    GroupDROState,
    WORLD_V3_HEAD_WEIGHTS,
    WORLD_VLA75_EXTRA_HEAD_WEIGHTS,
    WorldVLA75Model,
    WorldVLA75TrainConfig,
    _batch,
    event_aware_preference_consistency_loss,
    group_dro_weights,
    vla75_checkpoint_selection_key,
    train_world_vla75,
    world_vla75_per_sample_loss_report,
)
from data_pipeline.h6.dataset import (  # noqa: E402
    OutcomeCandidateExample,
    OutcomePairExample,
    outcome_examples_lineage_sha256,
)
from data_pipeline.h6.evaluator import (  # noqa: E402
    build_vla75_evaluator,
    file_sha256,
    finalize_training_summary_v2,
    verify_vla75_evaluator,
)
from data_pipeline.h6.calibration import select_vla75_router_config  # noqa: E402
from data_pipeline.h6.run_lock import (  # noqa: E402
    RUN_LOCK_SCHEMA_V1,
    build_h6_vla75_run_lock,
    verify_run_lock,
    verify_summary_checkpoints_against_lock,
    worktree_identity,
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
from data_pipeline.h6.temporal import TemporalSelectorConfig, TemporalSelectorCore  # noqa: E402
from data_pipeline.h5.runtime import H5WorldRouter  # noqa: E402
from scripts.h5_collect import _cleanup_retry_status  # noqa: E402
from scripts.h5_ultimate_benchmark import (  # noqa: E402
    build_latency_smoke_summary,
    verify_latency_smoke_artifact,
)
from scripts.h6_readiness import evaluate_vla75_readiness  # noqa: E402
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
    def test_repair_and_executable_have_independent_candidate_masks(self):
        import torch

        def row(key, source, repair, executable):
            return OutcomeCandidateExample(
                candidate_key=key,
                source=source,
                context=tuple([0.0] * H3_CONTEXT_DIM),
                candidate=tuple(
                    tuple([0.0] * H3_CANDIDATE_DIM)
                    for _ in range(H3_CANDIDATE_STEPS)
                ),
                objective_target=1.0,
                progress_m=1.0,
                route_completed=False,
                collision=False,
                red_light_violation=False,
                offroad=False,
                jerk_rms_mps3=0.0,
                acceleration_rms_mps2=0.0,
                lateral_acceleration_rms_mps2=0.0,
                repair_success=repair,
                trust=True,
                executable=executable,
            )

        pair = OutcomePairExample(
            "masks",
            "Town01",
            "unit",
            89,
            "ClearNoon",
            "train",
            (row("e", "expert", None, True), row("v", "vla", True, None)),
        )
        _, _, targets = _batch([pair], torch.device("cpu"), swap=False)
        outputs = torch.zeros((1, 2, 14))
        original = world_vla75_per_sample_loss_report(outputs, targets)
        changed = {key: value.clone() for key, value in targets.items()}
        changed["repair"][0, 0] = 1000.0
        changed["executable"][0, 1] = 1000.0
        modified = world_vla75_per_sample_loss_report(outputs, changed)
        self.assertAlmostEqual(
            float(original.head_losses["repair"][0]),
            float(modified.head_losses["repair"][0]),
        )
        self.assertAlmostEqual(
            float(original.head_losses["executable"][0]),
            float(modified.head_losses["executable"][0]),
        )
        self.assertTrue(bool(original.head_masks["repair"][0]))
        self.assertTrue(bool(original.head_masks["executable"][0]))

    def test_every_supervised_head_target_is_loss_isolated(self):
        import torch

        def row(key: str, source: str) -> OutcomeCandidateExample:
            return OutcomeCandidateExample(
                candidate_key=key,
                source=source,
                context=tuple([0.2] * H3_CONTEXT_DIM),
                candidate=tuple(
                    tuple([0.3] * H3_CANDIDATE_DIM)
                    for _ in range(H3_CANDIDATE_STEPS)
                ),
                objective_target=2.0 if source == "expert" else 1.0,
                progress_m=2.0,
                route_completed=False,
                collision=False,
                red_light_violation=False,
                offroad=False,
                jerk_rms_mps3=0.5,
                acceleration_rms_mps2=0.4,
                lateral_acceleration_rms_mps2=0.3,
                repair_success=True,
                trust=True,
                executable=True,
            )

        pair = OutcomePairExample(
            "head-isolation",
            "Town01",
            "unit",
            89,
            "ClearNoon",
            "train",
            (row("e", "expert"), row("v", "vla")),
        )
        _, _, targets = _batch([pair], torch.device("cpu"), swap=False)
        outputs = torch.linspace(-1.3, 1.7, 28, dtype=torch.float32).reshape(1, 2, 14)
        baseline = world_vla75_per_sample_loss_report(outputs, targets)

        def candidate_target(name: str, delta: float = 1.0):
            def mutate(values):
                values[name][0, 0] += delta

            return mutate

        def toggle_candidate(name: str):
            def mutate(values):
                values[name][0, 0] = 1.0 - values[name][0, 0]

            return mutate

        def toggle_pair(values):
            values["vla_preference"][0] = 1.0 - values["vla_preference"][0]

        mutators = {
            "objective": candidate_target("objective", 2.0),
            "progress": candidate_target("progress", 2.0),
            "completion": toggle_candidate("completion"),
            "collision": toggle_candidate("collision"),
            "red_light": toggle_candidate("red"),
            "offroad": toggle_candidate("offroad"),
            "comfort": candidate_target("jerk", 1.0),
            "repair": toggle_candidate("repair"),
            "trust": toggle_candidate("trust"),
            "pair_preference": toggle_pair,
            "executable": toggle_candidate("executable"),
        }
        self.assertEqual(set(mutators), set(baseline.head_losses))
        for expected_head, mutate in mutators.items():
            changed_targets = {key: value.clone() for key, value in targets.items()}
            mutate(changed_targets)
            changed = world_vla75_per_sample_loss_report(outputs, changed_targets)
            observed = {
                name
                for name in baseline.head_losses
                if not torch.allclose(
                    baseline.head_losses[name], changed.head_losses[name]
                )
            }
            self.assertEqual(observed, {expected_head}, expected_head)

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
        self.assertEqual(len(result.trace), 16)
        self.assertTrue(
            all(
                item["trace_schema"] == "safedrive.h6.temporal_selector.trace.v1"
                for item in result.trace
            )
        )

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

    def test_persistent_group_dro_reports_empty_group_as_not_measured(self):
        import torch

        state = GroupDROState(["low", "high", "empty"], eta=1.0, floor=0.01)
        empty_before = state.weights["empty"]
        report = state.evaluate_batch(
            torch.tensor([0.25, 3.0]),
            torch.tensor([True, True]),
            ["low", "high"],
            update=True,
        )
        self.assertGreater(report.groups["high"]["weight"], report.groups["low"]["weight"])
        self.assertEqual(report.groups["empty"]["status"], "NOT_MEASURED")
        self.assertEqual(report.groups["empty"]["count"], 0)
        self.assertIsNone(report.groups["empty"]["loss"])
        self.assertAlmostEqual(report.groups["empty"]["weight"], empty_before)
        high_after_first = state.weights["high"]
        empty_after_first = state.weights["empty"]
        second = state.evaluate_batch(
            torch.tensor([0.5]),
            torch.tensor([True]),
            ["low"],
            update=True,
        )
        self.assertEqual(second.groups["high"]["status"], "NOT_MEASURED")
        self.assertAlmostEqual(state.weights["high"], high_after_first)
        self.assertAlmostEqual(state.weights["empty"], empty_after_first)
        self.assertAlmostEqual(sum(state.weights.values()), 1.0)

    def test_selection_key_is_fail_closed_and_source_neutral(self):
        heads = {
            name: {"loss": 0.5, "count": 2}
            for name in tuple(WORLD_V3_HEAD_WEIGHTS) + tuple(WORLD_VLA75_EXTRA_HEAD_WEIGHTS)
        }
        metrics = {
            "schema_version": "safedrive.world.vla75.selection_metrics.v1",
            "evaluator_lineage_sha256": "a" * 64,
            "validation_loss": 1.0,
            "valid_samples": 2,
            "heads": heads,
            "pair_accuracy": 0.75,
            "pair_regret": 0.1,
            "pair_count": 2,
            "groups": {"g": {"status": "MEASURED", "loss": 1.0, "count": 2}},
            "worst_group_loss": 1.0,
            "worst_group_count": 2,
            "candidate_swap_error": 0.0,
            "candidate_swap_count": 56,
            "source_usage": {"vla_rows": 1},
            "config_tuple": "frozen",
        }
        first = vla75_checkpoint_selection_key(metrics)
        changed = copy.deepcopy(metrics)
        changed["source_usage"] = {"vla_rows": 999999}
        self.assertEqual(first, vla75_checkpoint_selection_key(changed))
        changed["raw_world_90_pass"] = True
        changed["actual_applied_75_pass"] = True
        self.assertEqual(first, vla75_checkpoint_selection_key(changed))
        del changed["pair_regret"]
        with self.assertRaisesRegex(ValueError, "metric_missing:pair_regret"):
            vla75_checkpoint_selection_key(changed)
        zero = copy.deepcopy(metrics)
        zero["heads"]["repair"]["count"] = 0
        with self.assertRaisesRegex(ValueError, "head_count_zero:repair"):
            vla75_checkpoint_selection_key(zero)
        no_lineage = copy.deepcopy(metrics)
        del no_lineage["evaluator_lineage_sha256"]
        with self.assertRaisesRegex(
            ValueError, "vla75_selection_evaluator_lineage"
        ):
            vla75_checkpoint_selection_key(no_lineage)


class TemporalSelectorCoreTest(unittest.TestCase):
    def test_offline_and_live_router_traces_match_field_for_field(self):
        from driving_vla.hybrid.contracts import (
            CandidateDifference,
            RoutingResult,
            SelectionSpace,
            WorldDisposition,
        )

        scope = "run|episode|route-1"
        score_rows = (
            (2.0, 1.0),
            (1.9, 2.0),
            (1.8, 2.1),
            (2.4, 1.0),
        )
        rows = []
        for tick, (vla_score, expert_score) in enumerate(score_rows):
            rows.append(
                {
                    "pair_id": scope,
                    "sequence_id": scope,
                    "tick": tick,
                    "phase": "steady",
                    "applied_source": "vla",
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
        offline = select_vla75_router_config(
            rows,
            alpha_grid=(0.5,),
            hold_grid=(2,),
            hysteresis_grid=(0.1,),
            target_actual_vla_coverage=0.0,
            max_switches_per_30s=10,
        )

        class Fallback:
            def route(self, candidate_set):
                ids = tuple(item.candidate.candidate_id for item in candidate_set.candidates)
                return RoutingResult(
                    pass_candidate_ids=ids,
                    rejected_candidate_ids=(),
                    selected_candidate_id=ids[0],
                    selection_space=SelectionSpace.DISTINCT,
                    world=WorldDisposition.DEFERRED_LOW_CONFIDENCE,
                    selector="fallback",
                    reason="fallback",
                    difference=CandidateDifference(
                        max_position_delta_m=1.0, rms_speed_delta_mps=1.0
                    ),
                )

        class Scorer:
            def __init__(self):
                self.index = 0

            def score_pair(self, first, second):
                tick = self.index
                self.index += 1
                by_source = {
                    "vla": score_rows[tick][0],
                    "expert": score_rows[tick][1],
                }
                predictions = []
                for candidate_id, _context, _candidate in (first, second):
                    source = candidate_id.rsplit(":", 1)[-1]
                    value = by_source[source]
                    predictions.append(
                        SimpleNamespace(
                            candidate_key=candidate_id,
                            utility=value,
                            deployment_score=value,
                            preference_utility=value,
                            trust_probability=0.95,
                            unsafe_probability=0.01,
                        )
                    )
                ordered = tuple(
                    item.candidate_key
                    for item in sorted(
                        predictions,
                        key=lambda item: item.deployment_score,
                        reverse=True,
                    )
                )
                return SimpleNamespace(
                    disposition="ranked",
                    selected_candidate_key=ordered[0],
                    defer_reason=None,
                    latency_ms=1.0,
                    predictions=predictions,
                    raw_preference_order=ordered,
                    preference_order=ordered,
                    risk_ceiling=0.2,
                    trust_threshold=0.5,
                )

        router = H5WorldRouter(
            Scorer(),
            Fallback(),
            ema_alpha=0.5,
            min_hold_ticks=2,
            hysteresis_margin=0.1,
            emergency_switch_margin=1.5,
            vla75_mode=True,
        )
        anchor = SimpleNamespace(
            bundle=SimpleNamespace(run_id="run", scenario_id="episode"),
            route_revision="route-1",
        )
        for tick in range(len(rows)):
            ids = {
                source: f"{scope}:{tick}:{source}"
                for source in ("expert", "vla")
            }
            candidate_set = SimpleNamespace(
                anchor=anchor,
                candidates=[
                    SimpleNamespace(
                        guard=SimpleNamespace(passed=True, needs_review=False),
                        candidate=SimpleNamespace(candidate_id=ids[source]),
                        provenance=SimpleNamespace(
                            source=SimpleNamespace(value=source)
                        ),
                    )
                    for source in ("expert", "vla")
                ],
            )
            features = {
                candidate_id: (
                    [0.0] * H3_CONTEXT_DIM,
                    tuple(
                        tuple([0.0] * H3_CANDIDATE_DIM)
                        for _ in range(H3_CANDIDATE_STEPS)
                    ),
                )
                for candidate_id in ids.values()
            }
            router.route(candidate_set, features)

        live_trace = tuple(router.metrics()["history"])
        self.assertEqual(tuple(offline.trace), live_trace)

    def test_source_ema_survives_frame_ids_and_hold_returns_fresh_id(self):
        selector = TemporalSelectorCore(
            TemporalSelectorConfig(ema_alpha=0.5, hold_ticks=3, hysteresis=0.1)
        )
        first = selector.step(
            scope_key="run|episode|route-a",
            source_scores={"expert": 0.0, "vla": 1.0},
            fresh_candidate_ids={"expert": "f1:e", "vla": "f1:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        held = selector.step(
            scope_key="run|episode|route-a",
            source_scores={"expert": 3.0, "vla": 0.9},
            fresh_candidate_ids={"expert": "f2:e", "vla": "f2:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="expert",
        )
        self.assertEqual(first.reason, "choose_initial")
        self.assertEqual(held.disposition, "HOLD")
        self.assertEqual(held.selected_candidate_id, "f2:v")
        self.assertNotEqual(held.selected_candidate_id, first.selected_candidate_id)
        self.assertIn("vla", held.ema_scores)

    def test_boundaries_emergency_risk_event_and_scope_reset(self):
        selector = TemporalSelectorCore(
            TemporalSelectorConfig(
                ema_alpha=1.0,
                hold_ticks=2,
                hysteresis=0.125,
                emergency_switch_margin=1.5,
            )
        )
        ids1 = {"expert": "1:e", "vla": "1:v"}
        selector.step(
            scope_key="s|r1",
            source_scores={"expert": 1.0, "vla": 0.0},
            fresh_candidate_ids=ids1,
            eligible_sources=set(ids1),
            raw_preferred_source="expert",
        )
        hysteresis = selector.step(
            scope_key="s|r1",
            source_scores={"expert": 1.0, "vla": 1.125},
            fresh_candidate_ids={"expert": "2:e", "vla": "2:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        self.assertEqual(hysteresis.reason, "hold_minimum")
        exact_hysteresis = selector.step(
            scope_key="s|r1",
            source_scores={"expert": 1.0, "vla": 1.125},
            fresh_candidate_ids={"expert": "3:e", "vla": "3:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        self.assertEqual(exact_hysteresis.reason, "hold_hysteresis")
        emergency = selector.step(
            scope_key="s|r1",
            source_scores={"expert": 0.0, "vla": 1.5},
            fresh_candidate_ids={"expert": "4:e", "vla": "4:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        self.assertEqual(emergency.reason, "switch_emergency_margin")
        risk = selector.step(
            scope_key="s|r1",
            source_scores={"expert": 0.0, "vla": 2.0},
            fresh_candidate_ids={"expert": "5:e", "vla": "5:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
            unsafe_sources={"vla"},
        )
        self.assertEqual(risk.reason, "switch_emergency_risk")
        event = selector.step(
            scope_key="s|r1",
            source_scores={"expert": 0.0, "vla": 1.0},
            fresh_candidate_ids={"expert": "6:e", "vla": "6:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
            event_break=True,
        )
        self.assertEqual(event.reason, "switch_event_break")
        continued = selector.step(
            scope_key="s|r1",
            source_scores={"expert": 0.0, "vla": 1.0},
            fresh_candidate_ids={"expert": "7:e", "vla": "7:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        self.assertEqual(continued.reason, "choose_continue")
        reset = selector.step(
            scope_key="s|r2",
            source_scores={"expert": 0.0, "vla": 1.0},
            fresh_candidate_ids={"expert": "8:e", "vla": "8:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        self.assertEqual(reset.reason, "choose_initial")
        self.assertEqual(reset.switch_count, 0)

        margin_selector = TemporalSelectorCore(
            TemporalSelectorConfig(
                ema_alpha=1.0,
                hold_ticks=0,
                hysteresis=0.1,
                emergency_switch_margin=1.5,
            )
        )
        margin_selector.step(
            scope_key="margin",
            source_scores={"expert": 1.0, "vla": 0.0},
            fresh_candidate_ids={"expert": "1:e", "vla": "1:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="expert",
        )
        switched = margin_selector.step(
            scope_key="margin",
            source_scores={"expert": 1.0, "vla": 1.2},
            fresh_candidate_ids={"expert": "2:e", "vla": "2:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        self.assertEqual(switched.reason, "switch_margin")

    def test_defer_fallback_order_held_expert_vla_mrm(self):
        selector = TemporalSelectorCore(TemporalSelectorConfig(hold_ticks=0))
        selector.step(
            scope_key="scope",
            source_scores={"expert": 0.0, "vla": 1.0},
            fresh_candidate_ids={"expert": "1:e", "vla": "1:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source="vla",
        )
        held = selector.step(
            scope_key="scope",
            source_scores={},
            fresh_candidate_ids={"expert": "2:e", "vla": "2:v"},
            eligible_sources={"expert", "vla"},
            raw_preferred_source=None,
            learned_defer_reason="ambiguous",
        )
        self.assertEqual((held.reason, held.selected_candidate_id), ("defer_held_source", "2:v"))
        expert = selector.step(
            scope_key="scope",
            source_scores={},
            fresh_candidate_ids={"expert": "3:e"},
            eligible_sources={"expert"},
            raw_preferred_source=None,
        )
        self.assertEqual(expert.reason, "defer_frozen_expert")
        vla = TemporalSelectorCore().step(
            scope_key="other",
            source_scores={},
            fresh_candidate_ids={"vla": "1:v"},
            eligible_sources={"vla"},
            raw_preferred_source=None,
        )
        self.assertEqual(vla.reason, "defer_frozen_vla")
        mrm = TemporalSelectorCore().step(
            scope_key="none",
            source_scores={},
            fresh_candidate_ids={},
            eligible_sources=set(),
            raw_preferred_source=None,
        )
        self.assertEqual(mrm.reason, "defer_no_eligible_mrm")
        self.assertIsNone(mrm.selected_candidate_id)

    def test_vla75_live_defer_entrypoints_all_use_shared_core(self):
        from driving_vla.hybrid.contracts import (
            CandidateDifference,
            RoutingResult,
            SelectionSpace,
            WorldDisposition,
        )

        class Fallback:
            def route(self, candidate_set):
                ids = tuple(
                    item.candidate.candidate_id for item in candidate_set.candidates
                )
                return RoutingResult(
                    pass_candidate_ids=ids,
                    rejected_candidate_ids=(),
                    selected_candidate_id=ids[0] if ids else None,
                    selection_space=SelectionSpace.DISTINCT,
                    world=WorldDisposition.DEFERRED_LOW_CONFIDENCE,
                    selector="fallback",
                    reason="fallback",
                    difference=CandidateDifference(
                        max_position_delta_m=1.0, rms_speed_delta_mps=1.0
                    ),
                )

        class Scorer:
            def __init__(self, *, latency_ms=1.0, disposition="ranked"):
                self.latency_ms = latency_ms
                self.disposition = disposition

            def score_pair(self, first, second):
                return SimpleNamespace(
                    disposition=self.disposition,
                    selected_candidate_key=(
                        first[0] if self.disposition == "ranked" else None
                    ),
                    defer_reason=(
                        None if self.disposition == "ranked" else "low_confidence"
                    ),
                    latency_ms=self.latency_ms,
                    predictions=(),
                )

        def candidate_set(sources=("expert", "vla")):
            return SimpleNamespace(
                anchor=SimpleNamespace(
                    bundle=SimpleNamespace(run_id="run", scenario_id="episode"),
                    route_revision="route",
                ),
                candidates=[
                    SimpleNamespace(
                        guard=SimpleNamespace(passed=True, needs_review=False),
                        candidate=SimpleNamespace(candidate_id=f"frame:{source}"),
                        provenance=SimpleNamespace(
                            source=SimpleNamespace(value=source)
                        ),
                    )
                    for source in sources
                ],
            )

        features = {
            f"frame:{source}": (
                [0.0] * H3_CONTEXT_DIM,
                tuple(
                    tuple([0.0] * H3_CANDIDATE_DIM)
                    for _ in range(H3_CANDIDATE_STEPS)
                ),
            )
            for source in ("expert", "vla")
        }
        cases = (
            (
                "single_candidate",
                H5WorldRouter(Scorer(), Fallback(), vla75_mode=True),
                candidate_set(("vla",)),
                {"frame:vla": features["frame:vla"]},
                "defer_frozen_vla",
            ),
            (
                "feature_missing",
                H5WorldRouter(Scorer(), Fallback(), vla75_mode=True),
                candidate_set(),
                None,
                "defer_frozen_expert",
            ),
            (
                "deadline",
                H5WorldRouter(
                    Scorer(latency_ms=51.0),
                    Fallback(),
                    scorer_deadline_ms=50.0,
                    vla75_mode=True,
                ),
                candidate_set(),
                features,
                "defer_frozen_expert",
            ),
            (
                "low_confidence",
                H5WorldRouter(
                    Scorer(disposition="defer_low_confidence"),
                    Fallback(),
                    vla75_mode=True,
                ),
                candidate_set(),
                features,
                "defer_frozen_expert",
            ),
            (
                "forced_defer",
                H5WorldRouter(
                    Scorer(), Fallback(), force_defer=True, vla75_mode=True
                ),
                candidate_set(),
                features,
                "defer_frozen_expert",
            ),
        )
        for expected_defer, router, candidates, case_features, expected_reason in cases:
            with self.subTest(expected_defer):
                result = router.route(candidates, case_features)
                trace = router.metrics()["history"][-1]
                self.assertEqual(result.reason, expected_reason)
                self.assertEqual(trace["learned_defer_reason"], expected_defer)
                self.assertTrue(trace["disposition"].startswith("DEFER"))


class C1ArtifactTruthfulnessTest(unittest.TestCase):
    def test_readiness_rejects_semantic_placeholders_and_tampering(self):
        import torch

        train_lineage = "1" * 64
        validation_lineage = "2" * 64
        config_hash = "3" * 64
        code_hash = "4" * 64
        worktree_hash = "5" * 64
        required_heads = tuple(WORLD_V3_HEAD_WEIGHTS) + tuple(
            WORLD_VLA75_EXTRA_HEAD_WEIGHTS
        )
        selection = {
            "schema_version": "safedrive.world.vla75.selection_metrics.v1",
            "evaluator_lineage_sha256": validation_lineage,
            "validation_loss": 1.0,
            "valid_samples": 2,
            "heads": {
                name: {"loss": 0.5, "count": 2} for name in required_heads
            },
            "pair_accuracy": 0.5,
            "pair_regret": 0.1,
            "pair_count": 2,
            "groups": {
                "g": {
                    "status": "MEASURED",
                    "loss": 1.0,
                    "count": 2,
                    "weight": 1.0,
                }
            },
            "worst_group_loss": 1.0,
            "worst_group_count": 2,
            "candidate_swap_error": 0.0,
            "candidate_swap_count": 56,
            "source_usage": {"vla_rows": 1},
            "config_tuple": "unit",
        }
        selection_hash = stable_sha256(selection)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_rows = []
            artifact_rows = []
            artifacts = []
            for seed in (11, 23, 37):
                checkpoint = root / f"seed-{seed}.pt"
                torch.save(
                    {
                        "metadata": {
                            "seed": seed,
                            "train_lineage_sha256": train_lineage,
                            "validation_lineage_sha256": validation_lineage,
                            "selection_metrics": selection,
                            "selection_metrics_sha256": selection_hash,
                        }
                    },
                    checkpoint,
                )
                checkpoint_hash = file_sha256(checkpoint)
                artifact = {
                    "schema_version": "safedrive.world.vla75.evaluator.v1",
                    "evidence_state": "MEASURED",
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": checkpoint_hash,
                        "seed": seed,
                        "selection_metrics_sha256": selection_hash,
                    },
                    "inputs": {
                        "training_lineage_sha256": train_lineage,
                        "validation_lineage_sha256": validation_lineage,
                        "config_sha256": config_hash,
                        "code_sha256": code_hash,
                        "worktree_sha256": worktree_hash,
                    },
                    "validation": {
                        "loss": {"status": "MEASURED", "value": 1.0, "count": 2},
                        "heads": {
                            name: {
                                "status": "MEASURED",
                                "loss": 0.5,
                                "count": 2,
                                "weight": float(
                                    WORLD_V3_HEAD_WEIGHTS.get(
                                        name, WORLD_VLA75_EXTRA_HEAD_WEIGHTS.get(name, 1.0)
                                    )
                                ),
                                **(
                                    {"positive_count": 1}
                                    if name in {"collision", "red_light", "offroad"}
                                    else {}
                                ),
                            }
                            for name in required_heads
                        },
                        "pair": {
                            "status": "MEASURED",
                            "accuracy": 0.5,
                            "regret": 0.1,
                            "count": 2,
                        },
                        "groups": {
                            "g": {
                                "status": "MEASURED",
                                "loss": 1.0,
                                "count": 2,
                                "weight": 1.0,
                            }
                        },
                    },
                    "probes": {
                        "candidate_swap": {
                            "status": "MEASURED", "value": 0.0, "count": 56
                        },
                        "source_metadata_swap": {
                            "status": "MEASURED", "value": 0.0, "count": 56
                        },
                        "action_mask": {
                            "status": "MEASURED", "value": 0.1, "count": 2
                        },
                        "context_history_mask": {
                            "status": "MEASURED", "value": 0.1, "count": 2
                        },
                    },
                    "latency": {
                        "status": "MEASURED",
                        "device": "cpu",
                        "iterations": 2,
                        "p50_ms": 1.0,
                        "p95_ms": 1.1,
                        "p99_ms": 1.2,
                        "max_ms": 1.3,
                    },
                    "gpu_peak": {
                        "status": "NOT_MEASURED",
                        "device": "cpu",
                        "incremental_peak_gib": None,
                    },
                    "selection_metrics": copy.deepcopy(selection),
                    "artifact_verification": {
                        "checkpoint_metadata": "VERIFIED",
                        "selection_metrics": "VERIFIED",
                    },
                }
                artifact["inputs"]["input_sha256"] = stable_sha256(
                    {
                        "training": train_lineage,
                        "validation": validation_lineage,
                        "config": config_hash,
                        "code": code_hash,
                        "worktree": worktree_hash,
                    }
                )
                artifact["evaluator_sha256"] = stable_sha256(artifact)
                evaluator_path = root / f"evaluator-{seed}.json"
                evaluator_path.write_text(
                    json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
                )
                artifacts.append(artifact)
                artifact_rows.append(
                    {
                        "seed": seed,
                        "path": str(evaluator_path),
                        "sha256": artifact["evaluator_sha256"],
                        "file_sha256": file_sha256(evaluator_path),
                    }
                )
                checkpoint_rows.append(
                    {
                        "seed": seed,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "schema_version": "safedrive.world.vla75.pair_exec.v1",
                        "evaluator_sha256": artifact["evaluator_sha256"],
                    }
                )

            pilot = load_h6_training_matrix(full=False)
            summary_payload = {
                "contract": "vla75-v2",
                "train_lineage_sha256": train_lineage,
                "validation_lineage_sha256": validation_lineage,
                "training_config_sha256": config_hash,
                "code_sha256": code_hash,
                "worktree_sha256": worktree_hash,
                "models": checkpoint_rows,
                "evaluators": artifact_rows,
                "h6_roots": ["unit"],
                "h6_acceptance_seeds_loaded": False,
                "seeds_loaded": list(H6_TRAIN_SEEDS),
                "h6_train_pair_ids": [
                    row.pair_id for row in pilot if row.scenario.seed == H6_TRAIN_SEEDS[0]
                ],
                "h6_calibration_pair_ids": [
                    row.pair_id for row in pilot if row.scenario.seed == H6_TRAIN_SEEDS[1]
                ],
            }

            def write_summary(changed_artifacts):
                for index, artifact in enumerate(changed_artifacts):
                    artifact["evaluator_sha256"] = stable_sha256(
                        {
                            key: value
                            for key, value in artifact.items()
                            if key != "evaluator_sha256"
                        }
                    )
                    path = Path(artifact_rows[index]["path"])
                    path.write_text(
                        json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    artifact_rows[index]["sha256"] = artifact["evaluator_sha256"]
                    artifact_rows[index]["file_sha256"] = file_sha256(path)
                    checkpoint_rows[index]["evaluator_sha256"] = artifact[
                        "evaluator_sha256"
                    ]
                return finalize_training_summary_v2(summary_payload)

            base = write_summary(copy.deepcopy(artifacts))
            result = evaluate_vla75_readiness(
                base, scope="pilot", lineage_id="a"
            )
            self.assertIn("evaluator_gpu_not_measured:0", result["failures"])

            changed = copy.deepcopy(artifacts)
            changed[0]["gpu_peak"].update(
                status="MEASURED", device="cuda:0", incremental_peak_gib=0.0
            )
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn("evaluator_gpu_not_measured:0", result["failures"])

            changed = copy.deepcopy(artifacts)
            changed[0]["latency"]["p99_ms"] = 0.0
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn("evaluator_latency_p99_ms:0", result["failures"])

            changed = copy.deepcopy(artifacts)
            changed[0]["validation"]["heads"]["repair"].update(
                status="NOT_MEASURED", loss=None, count=0
            )
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn("evaluator_head_not_measured:0:repair", result["failures"])

            changed = copy.deepcopy(artifacts)
            del changed[0]["validation"]["heads"]["repair"]
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn("evaluator_head_missing:0:repair", result["failures"])

            changed = copy.deepcopy(artifacts)
            changed[0]["probes"]["action_mask"]["value"] = 0.0
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn(
                "evaluator_probe_no_sensitivity:0:action_mask", result["failures"]
            )

            changed = copy.deepcopy(artifacts)
            changed[0]["probes"]["candidate_swap"]["value"] = 0.1
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn(
                "evaluator_probe_invariant:0:candidate_swap", result["failures"]
            )

            changed = copy.deepcopy(artifacts)
            changed[0]["inputs"]["validation_lineage_sha256"] = "9" * 64
            changed[0]["inputs"]["input_sha256"] = stable_sha256(
                {
                    "training": train_lineage,
                    "validation": "9" * 64,
                    "config": config_hash,
                    "code": code_hash,
                    "worktree": worktree_hash,
                }
            )
            result = evaluate_vla75_readiness(
                write_summary(changed), scope="pilot", lineage_id="a"
            )
            self.assertIn(
                "evaluator_0:validation_lineage", result["failures"]
            )

            evaluator_path = Path(artifact_rows[0]["path"])
            evaluator_path.write_text(
                evaluator_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            result = evaluate_vla75_readiness(
                base, scope="pilot", lineage_id="a"
            )
            self.assertIn("evaluator_file_hash:0", result["failures"])

            checkpoint_path = Path(checkpoint_rows[0]["checkpoint_path"])
            with checkpoint_path.open("ab") as stream:
                stream.write(b"tamper")
            result = evaluate_vla75_readiness(
                base, scope="pilot", lineage_id="a"
            )
            self.assertIn("checkpoint_hash:0", result["failures"])

    def test_real_cpu_evaluator_is_self_hashed_and_gpu_is_not_measured(self):
        def candidate(key: str, source: str, unsafe: bool) -> OutcomeCandidateExample:
            return OutcomeCandidateExample(
                candidate_key=key,
                source=source,
                context=tuple([0.1 if source == "expert" else 0.2] * H3_CONTEXT_DIM),
                candidate=tuple(
                    tuple([0.1 if source == "expert" else 0.3] * H3_CANDIDATE_DIM)
                    for _ in range(H3_CANDIDATE_STEPS)
                ),
                objective_target=-3.0 if unsafe else 2.0,
                progress_m=1.0 if unsafe else 3.0,
                route_completed=not unsafe,
                collision=unsafe,
                red_light_violation=unsafe,
                offroad=unsafe,
                jerk_rms_mps3=0.5,
                acceleration_rms_mps2=0.4,
                lateral_acceleration_rms_mps2=0.3,
                repair_success=not unsafe,
                trust=not unsafe,
                executable=not unsafe,
                phase="event" if unsafe else "cruise",
                group_key="Town01|unit|ClearNoon",
            )

        rows = [
            OutcomePairExample(
                f"pair-{index}",
                "Town01",
                "unit",
                89,
                "ClearNoon",
                "dev_fold_1",
                (
                    candidate(f"{index}:expert", "expert", False),
                    candidate(f"{index}:vla", "vla", True),
                ),
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "seed-11.pt"
            config = WorldVLA75TrainConfig(
                d_model=32,
                layers=1,
                heads=4,
                ffn=64,
                dropout=0.0,
                batch_size=2,
                max_epochs=1,
                patience=1,
            )
            train_world_vla75(
                rows,
                rows,
                seed=11,
                checkpoint_path=checkpoint,
                device="cpu",
                config=config,
            )
            lineage = outcome_examples_lineage_sha256(rows)
            artifact = build_vla75_evaluator(
                checkpoint,
                rows,
                device="cpu",
                training_input_sha256=lineage,
                config_sha256="a" * 64,
                code_sha256="b" * 64,
                worktree_sha256="c" * 64,
                latency_iterations=2,
            )
            self.assertTrue(verify_vla75_evaluator(artifact, root=ROOT)["valid"])
            self.assertEqual(artifact["gpu_peak"]["status"], "NOT_MEASURED")
            self.assertIsNone(artifact["gpu_peak"]["incremental_peak_gib"])
            tampered = copy.deepcopy(artifact)
            tampered["probes"]["action_mask"]["value"] = 0.0
            self.assertFalse(verify_vla75_evaluator(tampered)["valid"])

            rebound = copy.deepcopy(artifact)
            rebound["selection_metrics"]["validation_loss"] += 1.0
            rebound["evaluator_sha256"] = stable_sha256(
                {
                    key: value
                    for key, value in rebound.items()
                    if key != "evaluator_sha256"
                }
            )
            verification = verify_vla75_evaluator(rebound, root=ROOT)
            self.assertIn(
                "checkpoint_selection_metrics", verification["failures"]
            )

            missing_head = copy.deepcopy(artifact)
            del missing_head["validation"]["heads"]["repair"]
            missing_head["evaluator_sha256"] = stable_sha256(
                {
                    key: value
                    for key, value in missing_head.items()
                    if key != "evaluator_sha256"
                }
            )
            verification = verify_vla75_evaluator(missing_head)
            self.assertIn("head_missing:repair", verification["failures"])

    def test_legacy_summary_is_not_c1_ready_and_readiness_is_self_hashed(self):
        result = evaluate_vla75_readiness(
            {"schema_version": "safedrive.world.vla75.training_summary.v1"},
            scope="pilot",
            lineage_id="a",
        )
        self.assertFalse(result["ready"])
        self.assertIn("legacy_summary_not_c1_ready", result["failures"])
        self.assertEqual(
            result["readiness_sha256"],
            stable_sha256(
                {key: value for key, value in result.items() if key != "readiness_sha256"}
            ),
        )

    def test_v2_summary_missing_evaluators_and_hash_tamper_fail(self):
        summary = finalize_training_summary_v2(
            {
                "contract": "vla75-v2",
                "models": [],
                "evaluators": [],
            }
        )
        result = evaluate_vla75_readiness(
            summary, scope="pilot", lineage_id="a"
        )
        self.assertIn("vla75_ensemble_requires_three_evaluators", result["failures"])
        tampered = dict(summary)
        tampered["contract"] = "tampered"
        result = evaluate_vla75_readiness(
            tampered, scope="pilot", lineage_id="a"
        )
        self.assertIn("summary_hash", result["failures"])

        malformed = copy.deepcopy(summary)
        malformed["models"] = [{"seed": {}}] * 3
        malformed["evaluators"] = [None] * 3
        result = evaluate_vla75_readiness(
            malformed, scope="pilot", lineage_id="a"
        )
        self.assertIn("summary_hash", result["failures"])
        self.assertIn("vla75_model_seed_order", result["failures"])
        self.assertIn("vla75_ensemble_requires_three_evaluators", result["failures"])

    def test_cleanup_residue_never_ticks_world(self):
        class Actor:
            id = 42
            type_id = "vehicle.test"
            is_alive = True

        class World:
            def __init__(self):
                self.ticks = 0

            def get_actors(self):
                return [Actor()]

            def get_settings(self):
                return SimpleNamespace(synchronous_mode=False)

            def tick(self):
                self.ticks += 1

        world = World()
        result = _cleanup_retry_status(world)
        self.assertEqual(result["status"], "NEEDS_USER_ACTION")
        self.assertEqual(result["failure_code"], "CLEANUP_RESIDUE")
        self.assertEqual(result["residue_ids"], [42])
        self.assertFalse(result["tick_advanced"])
        self.assertEqual(world.ticks, 0)
        source = inspect.getsource(__import__("scripts.h5_collect", fromlist=["_"]))
        self.assertNotIn("world.tick()", source)

    def test_clean_scene_allows_one_bounded_retry_without_tick(self):
        class World:
            def __init__(self):
                self.ticks = 0

            def get_actors(self):
                return []

            def get_settings(self):
                return SimpleNamespace(synchronous_mode=False)

            def tick(self):
                self.ticks += 1

        world = World()
        result = _cleanup_retry_status(world)
        self.assertEqual(result["status"], "CLEAN_RETRY_ALLOWED")
        self.assertEqual(result["retry_status"], "ALLOWED_ONCE")
        self.assertTrue(result["settings_confirmed"])
        self.assertTrue(result["tick_owner_confirmed"])
        self.assertEqual(world.ticks, 0)
        self.assertEqual(
            result["cleanup_sha256"],
            stable_sha256(
                {key: value for key, value in result.items() if key != "cleanup_sha256"}
            ),
        )

    def test_random_benchmark_is_never_quality_gate_eligible(self):
        scorer = {
            "device": "cpu",
            "model_state": "random_untrained",
            "p99_latency_ms": 1.0,
            "device_latency_threshold_p99_ms": 15.0,
            "latency_p99_pass": True,
        }
        payload = build_latency_smoke_summary({}, scorer, {}, timestamp=1.0)
        self.assertEqual(payload["status"], "SMOKE_COMPLETED")
        self.assertFalse(payload["quality_gate_eligible"])
        self.assertTrue(verify_latency_smoke_artifact(payload))
        self.assertNotIn("ALL_TARGETS", json.dumps(payload))
        default_path = ROOT / "generated/h6/c1-smoke/ultimate-latency-smoke.json"
        self.assertNotIn("docs/runtime-evidence", str(default_path))


class VLA75RunLockTest(unittest.TestCase):
    def test_c1_run_lock_requires_evaluator_and_input_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            for index in range(3):
                path = root / f"seed-{index}.pt"
                path.write_bytes(bytes([index + 21]))
                checkpoints.append(path)
            missing = build_h6_vla75_run_lock(
                ROOT,
                lineage_id="a",
                dataset_id="h6-vla75-c1-bindings",
                matrix_rows=load_h6_vla75_matrix("a", full=False),
                checkpoint_paths=checkpoints,
                calibration={},
                scoped_paths=[],
            )
            self.assertEqual(
                missing["schema_version"], "safedrive.h6.vla75.run_lock.v2"
            )
            self.assertIn(
                "c1_calibration_bindings_missing",
                verify_run_lock(missing)["failures"],
            )
            bound = build_h6_vla75_run_lock(
                ROOT,
                lineage_id="a",
                dataset_id="h6-vla75-c1-bindings",
                matrix_rows=load_h6_vla75_matrix("a", full=False),
                checkpoint_paths=checkpoints,
                calibration={
                    "deployment": {
                        "c1_bindings": {
                            "evaluator_sha256": ["a" * 64, "b" * 64, "c" * 64],
                            "validation_lineage_sha256": "d" * 64,
                            "training_input_sha256": "e" * 64,
                        }
                    }
                },
                scoped_paths=[],
            )
            self.assertTrue(verify_run_lock(bound)["valid"])

    def test_test_registry_is_excluded_from_worktree_identity(self):
        identity = worktree_identity(ROOT)
        self.assertNotIn(
            "test_registry.sqlite3",
            {item["path"] for item in identity["untracked_files"]},
        )

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
                schema_version=RUN_LOCK_SCHEMA_V1,
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
                schema_version=RUN_LOCK_SCHEMA_V1,
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
