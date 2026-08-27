"""Tests for outcome/trust World v3 and its 90% calibration gate."""

from __future__ import annotations

import math
import json
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import (  # noqa: E402
    H3_CANDIDATE_DIM,
    H3_CANDIDATE_STEPS,
    H3_CONTEXT_DIM,
)
from data_pipeline.h6.calibration import (  # noqa: E402
    CalibrationRow,
    calibrate_vla_deployment,
)
from data_pipeline.h6.acceptance import evaluate_vla90_gate  # noqa: E402
from data_pipeline.h6.config import H6_VLA90_CONFIG_SHA256  # noqa: E402
from data_pipeline.h6.contracts import (  # noqa: E402
    WORLD_V3_OUTPUT_DIM,
    WorldV3Prediction,
)
from data_pipeline.h6.dataset import (  # noqa: E402
    OutcomeCandidateExample,
    OutcomePairExample,
    load_h6_closed_loop_examples,
    load_h6_policy_calibration_examples,
    objective_target,
)
from data_pipeline.h3.dataset import H3DatasetError  # noqa: E402
from data_pipeline.h2.live_contract import route_follow_steer  # noqa: E402
from data_pipeline.h6.model import WorldV3Model, _batch, world_v3_loss  # noqa: E402
from data_pipeline.h6.matrix import (  # noqa: E402
    H6_SEEDS,
    H6_TRAIN_SEEDS,
    _h6_pre_roll_script,
    load_h6_matrix,
    load_h6_training_matrix,
)
from scripts.h5_collect import (  # noqa: E402
    _FrozenTrafficLights,
    _SpectatorFollower,
    _pre_roll_initial_progress,
    _should_force_dynamic_red,
)


def _candidate(key: str, source: str, *, unsafe=False, trust=True):
    return OutcomeCandidateExample(
        candidate_key=key,
        source=source,
        context=tuple([0.0] * H3_CONTEXT_DIM),
        candidate=tuple(
            tuple([0.0] * H3_CANDIDATE_DIM) for _ in range(H3_CANDIDATE_STEPS)
        ),
        objective_target=-10.0 if unsafe else 2.0,
        progress_m=2.0,
        route_completed=False,
        collision=unsafe,
        red_light_violation=False,
        offroad=False,
        jerk_rms_mps3=0.1,
        acceleration_rms_mps2=0.2,
        lateral_acceleration_rms_mps2=0.1,
        repair_success=None,
        trust=trust,
    )


def _prediction(
    key: str,
    *,
    trust_probability: float,
    unsafe_probability: float,
    deployment_score: float = 1.0,
):
    trust_logit = math.log(trust_probability / (1.0 - trust_probability))
    # Put the entire requested union risk in collision and make other hazards tiny.
    risk_logit = math.log(unsafe_probability / (1.0 - unsafe_probability))
    tiny = math.log(1e-6 / (1.0 - 1e-6))
    return WorldV3Prediction(
        candidate_key=key,
        objective_utility=1.0,
        progress_mean_m=2.0,
        progress_logvar=-2.0,
        completion_logit=0.0,
        collision_logit=risk_logit,
        red_light_logit=tiny,
        offroad_logit=tiny,
        jerk_mean_log1p=0.0,
        acceleration_mean_mps2=0.0,
        lateral_acceleration_mean_mps2=0.0,
        repair_success_logit=0.0,
        trust_logit=trust_logit,
        deployment_score=deployment_score,
    )


class WorldV3ModelTest(unittest.TestCase):
    def test_kinematic_pre_roll_always_starts_from_authored_route_origin(self):
        route = ((0.0, 0.0), (10.0, 0.0))
        kinematic = SimpleNamespace(
            script={"pre_roll_kinematic": True}, route=route
        )
        physical = SimpleNamespace(script={}, route=route)
        self.assertEqual(_pre_roll_initial_progress(kinematic, 10.0, 0.0), 0.0)
        self.assertAlmostEqual(
            _pre_roll_initial_progress(physical, 10.0, 0.0), 10.0
        )

    def test_h6_pre_roll_rejects_zero_motion_anchors(self):
        script = _h6_pre_roll_script(
            {
                "pre_roll_ticks": 20,
                "pre_roll_target_speed_mps": 2.0,
                "pre_roll_kp": 0.3,
                "pre_roll_max_throttle": 0.35,
            }
        )
        self.assertEqual(script["pre_roll_ticks"], 20)
        self.assertEqual(script["pre_roll_target_speed_mps"], 4.0)
        self.assertEqual(script["pre_roll_min_ready_speed_mps"], 2.5)
        self.assertEqual(script["pre_roll_max_extra_ticks"], 80)
        self.assertTrue(script["pre_roll_route_follow"])
        self.assertTrue(script["pre_roll_kinematic"])
        self.assertEqual(script["pre_roll_kinematic_speed_mps"], 3.0)
        self.assertTrue(script["spectator_follow_ego"])

    def test_h6_pre_roll_preserves_short_red_light_timing(self):
        script = _h6_pre_roll_script(
            {"pre_roll_ticks": 10, "pre_roll_target_speed_mps": 8.0}
        )
        self.assertEqual(script["pre_roll_ticks"], 10)
        self.assertEqual(script["pre_roll_target_speed_mps"], 8.0)

    def test_h6_red_light_starts_green_then_changes_on_declared_tick(self):
        script = _h6_pre_roll_script(
            {"red_at_capture": False, "red_after_tick": 5}
        )
        scenario = SimpleNamespace(script=script)
        self.assertFalse(_should_force_dynamic_red(scenario, 4))
        self.assertTrue(_should_force_dynamic_red(scenario, 5))

        class FakeLight:
            def __init__(self):
                self.state = "Red"

            def freeze(self, _value):
                return None

            def set_state(self, value):
                self.state = value

        light = FakeLight()
        frozen = _FrozenTrafficLights(SimpleNamespace(), scenario)
        frozen.saved = [(light, "Red", False)]
        frozen.force_green()
        self.assertEqual(str(light.state).split(".")[-1], "Green")

        dynamic = SimpleNamespace(
            red_light={"trigger_x": 0.0, "trigger_y": 0.0},
            script={"red_at_capture": False},
        )
        frozen = _FrozenTrafficLights(SimpleNamespace(), dynamic)
        frozen.saved = [(light, "Red", False)]
        light.state = "Red"
        frozen.reset_for_arm()
        self.assertEqual(str(light.state).split(".")[-1], "Green")

        static_red = SimpleNamespace(
            red_light={"trigger_x": 0.0, "trigger_y": 0.0},
            script={"red_at_capture": True},
        )
        frozen = _FrozenTrafficLights(SimpleNamespace(), static_red)
        frozen.saved = [(light, "Green", False)]
        frozen._target_light = lambda: light
        frozen.reset_for_arm()
        self.assertEqual(str(light.state).split(".")[-1], "Red")

    def test_route_follow_steer_is_zero_straight_and_turns_toward_corner(self):
        straight = route_follow_steer(
            0.0, 0.0, 0.0, 3.0, ((0.0, 0.0), (20.0, 0.0))
        )
        corner = route_follow_steer(
            8.0, 0.0, 0.0, 3.0, ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
        )
        self.assertAlmostEqual(straight, 0.0)
        self.assertGreater(corner, 0.0)
        self.assertLessEqual(corner, 0.35)

    def test_spectator_follower_updates_while_decision_work_is_in_flight(self):
        class FakeSpectator:
            def __init__(self):
                self.transforms = []

            def set_transform(self, transform):
                self.transforms.append(transform)

        transform = SimpleNamespace(
            location=SimpleNamespace(x=10.0, y=5.0, z=1.0),
            rotation=SimpleNamespace(yaw=0.0),
        )
        spectator = FakeSpectator()
        runtime = SimpleNamespace(
            _actors={"ego": SimpleNamespace(get_transform=lambda: transform)},
            world=SimpleNamespace(get_spectator=lambda: spectator),
        )
        scenario = SimpleNamespace(
            script={"spectator_follow_ego": True, "spectator_follow_hz": 50.0}
        )
        follower = _SpectatorFollower(runtime, scenario)
        follower.start()
        time.sleep(0.07)
        follower.stop()
        self.assertGreaterEqual(follower.updates, 2)
        self.assertIsNone(follower.last_error)
        self.assertEqual(len(spectator.transforms), follower.updates)

    def test_training_matrix_is_disjoint_from_formal_acceptance(self):
        training = load_h6_training_matrix(full=True)
        acceptance = load_h6_matrix(full=True)
        self.assertTrue(set(H6_TRAIN_SEEDS).isdisjoint(H6_SEEDS))
        self.assertTrue(
            {row.pair_id for row in training}.isdisjoint(
                {row.pair_id for row in acceptance}
            )
        )

    def test_training_pilot_balances_family_support_across_both_seeds(self):
        pilot = load_h6_training_matrix(full=False)
        self.assertEqual(len(pilot), 24)
        for map_name in ("Town01", "Town03", "Town05"):
            for family in (
                "free_flow",
                "emergency_lead_brake",
                "aggressive_cut_in",
                "red_light_dilemma",
            ):
                seeds = {
                    row.scenario.seed
                    for row in pilot
                    if row.scenario.map_name == map_name
                    and row.scenario.family == family
                }
                self.assertEqual(seeds, set(H6_TRAIN_SEEDS))

    def test_h6_loader_uses_real_closed_loop_arm_outcomes(self):
        context = [0.0] * H3_CONTEXT_DIM
        candidate = [[0.0] * H3_CANDIDATE_DIM for _ in range(H3_CANDIDATE_STEPS)]
        scenario = {
            "map_name": "Town03",
            "family": "free_flow",
            "seed": H6_TRAIN_SEEDS[0],
            "weather": "ClearNoon",
        }
        common = {
            "pair_id": "Town03__free_flow__s89__ClearNoon",
            "scenario": scenario,
            "manifest_kind": "h6_fresh_training",
            "physical_sha256": "same",
            "ok": True,
            "ticks_executed": 10,
            "collision_count": 0,
            "red_light_violation": False,
            "off_corridor_duration_s": 0.0,
            "route_completed": False,
            "jerk_rms_mps3": 0.1,
            "acceleration_rms_mps2": 0.2,
            "lateral_acceleration_rms_mps2": 0.1,
            "initial_route_progress_m": 0.0,
        }
        off = {
            **common,
            "arm": "off",
            "route_progress_m": 3.0,
            "expert_executed_ticks": 10,
            "vla_executed_ticks": 0,
            "decisions": [],
        }
        on = {
            **common,
            "arm": "on",
            "route_progress_m": 3.2,
            "expert_executed_ticks": 1,
            "vla_executed_ticks": 9,
            "reset_comparison": {"comparable": True},
            "decisions": [
                {
                    "tick": 0,
                    "simulation_time_s": 1.0,
                    "executed_source": "vla",
                    "world_features": {
                        "frame:expert": {"context": context, "candidate": candidate},
                        "frame:vla": {"context": context, "candidate": candidate},
                    },
                    "repair": None,
                    "guard": {
                        "frame:expert": {"verdict": "PASS"},
                        "frame:vla": {"verdict": "PASS"},
                    },
                    "arbitration": {
                        "notes": [],
                        "audits": [
                            {"candidate_id": "frame:expert", "source": "classic", "final_ok": True, "reject_reasons": []},
                            {"candidate_id": "frame:vla", "source": "vla_fast", "final_ok": True, "reject_reasons": []},
                        ],
                    },
                }
            ],
            "timeline": [
                {
                    "tick": 0,
                    "carla_frame": 11,
                    "simulation_time_s": 1.05,
                    "route_progress_m": 0.2,
                    "corridor_distance_m": 0.0,
                    "acceleration_mps2": 0.2,
                    "lateral_acceleration_mps2": 0.1,
                }
            ],
            "events": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            runs.mkdir()
            (runs / "pair__off.json").write_text(json.dumps(off), encoding="utf-8")
            (runs / "pair__on.json").write_text(json.dumps(on), encoding="utf-8")
            examples = load_h6_closed_loop_examples(
                Path(directory),
                seeds=(H6_TRAIN_SEEDS[0],),
                split="h6_train",
            )
            self.assertEqual(len(examples), 1)
            expert, vla = examples[0].candidates
            self.assertEqual((expert.source, vla.source), ("expert", "vla"))
            self.assertFalse(expert.outcome_observed)
            self.assertTrue(vla.outcome_observed)
            self.assertEqual((examples[0].arm, examples[0].tick), ("on", 0))
            self.assertEqual(expert.progress_m, 0.0)
            self.assertAlmostEqual(vla.progress_m, 10.0)
            self.assertTrue(expert.safety_observed)
            self.assertTrue(vla.safety_observed)
            self.assertTrue(vla.trust)
            policy_rows = load_h6_policy_calibration_examples(
                Path(directory), seeds=(H6_TRAIN_SEEDS[0],)
            )
            self.assertEqual(len(policy_rows), 1)
            policy_expert, policy_vla = policy_rows[0].candidates
            self.assertEqual(
                (policy_expert.progress_m, policy_vla.progress_m), (3.0, 3.2)
            )
            self.assertAlmostEqual(policy_rows[0].actual_vla_coverage, 0.9)
            policy_pair = policy_rows[0].as_outcome_pair(
                split="h6_train_policy"
            )
            self.assertTrue(
                all(item.outcome_observed for item in policy_pair.candidates)
            )
            self.assertEqual(policy_pair.family, "free_flow")
            with self.assertRaisesRegex(H3DatasetError, "acceptance_seed"):
                load_h6_closed_loop_examples(
                    Path(directory), seeds=(H6_SEEDS[0],), split="bad"
                )

    def test_h6_loader_accepts_mixed_episode_and_masks_unexecuted_candidate(self):
        context = [0.0] * H3_CONTEXT_DIM
        candidate = [[0.0] * H3_CANDIDATE_DIM for _ in range(H3_CANDIDATE_STEPS)]
        common = {
            "pair_id": "Town03__free_flow__s89__ClearNoon",
            "scenario": {
                "map_name": "Town03",
                "family": "free_flow",
                "seed": H6_TRAIN_SEEDS[0],
                "weather": "ClearNoon",
            },
            "manifest_kind": "h6_fresh_training",
            "physical_sha256": "same",
            "ok": True,
            "ticks_executed": 10,
            "route_progress_m": 3.0,
            "collision_count": 0,
            "red_light_violation": False,
            "off_corridor_duration_s": 0.0,
            "initial_route_progress_m": 0.0,
        }
        off = {
            **common,
            "arm": "off",
            "expert_executed_ticks": 10,
            "vla_executed_ticks": 0,
            "decisions": [],
        }
        on = {
            **common,
            "arm": "on",
            "expert_executed_ticks": 2,
            "vla_executed_ticks": 8,
            "reset_comparison": {"comparable": True},
            "decisions": [
                {
                    "tick": 0,
                    "simulation_time_s": 1.0,
                    "executed_source": "vla",
                    "world_features": {
                        "frame:expert": {"context": context, "candidate": candidate},
                        "frame:vla": {"context": context, "candidate": candidate},
                    },
                    "guard": {
                        "frame:expert": {"verdict": "PASS"},
                        "frame:vla": {"verdict": "PASS"},
                    },
                    "arbitration": {
                        "audits": [
                            {"candidate_id": "frame:expert", "source": "classic", "final_ok": True, "reject_reasons": []},
                            {"candidate_id": "frame:vla", "source": "vla_fast", "final_ok": True, "reject_reasons": []},
                        ]
                    },
                }
            ],
            "timeline": [
                {
                    "tick": 0,
                    "carla_frame": 11,
                    "simulation_time_s": 1.05,
                    "route_progress_m": 0.1,
                    "corridor_distance_m": 0.0,
                    "acceleration_mps2": 0.0,
                    "lateral_acceleration_mps2": 0.0,
                }
            ],
            "events": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            runs.mkdir()
            (runs / "pair__off.json").write_text(json.dumps(off), encoding="utf-8")
            (runs / "pair__on.json").write_text(json.dumps(on), encoding="utf-8")
            examples = load_h6_closed_loop_examples(
                Path(directory),
                seeds=(H6_TRAIN_SEEDS[0],),
                split="h6_train",
            )
            self.assertEqual(len(examples), 1)
            expert, vla = examples[0].candidates
            self.assertFalse(expert.outcome_observed)
            self.assertTrue(vla.outcome_observed)

    def test_shared_model_has_comprehensive_outputs_and_finite_loss(self):
        model = WorldV3Model(d_model=32, layers=1, heads=4, ffn=64, dropout=0.0)
        context = torch.zeros((2, H3_CONTEXT_DIM), dtype=torch.float32)
        candidate = torch.zeros(
            (2, H3_CANDIDATE_STEPS, H3_CANDIDATE_DIM), dtype=torch.float32
        )
        self.assertEqual(tuple(model(context, candidate).shape), (2, WORLD_V3_OUTPUT_DIM))
        pair = OutcomePairExample(
            "pair",
            "Town01",
            "free_flow",
            0,
            "ClearNoon",
            "dev_fold_1",
            (_candidate("expert", "expert"), _candidate("vla", "vla", unsafe=True, trust=False)),
        )
        contexts, candidates, targets = _batch([pair], torch.device("cpu"), swap=False)
        outputs = torch.stack(
            [model(contexts[:, index], candidates[:, index]) for index in range(2)], dim=1
        )
        loss, pieces = world_v3_loss(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(pieces),
            {"objective_reg", "pair", "progress", "completion", "collision", "red", "offroad", "comfort", "repair", "trust"},
        )

    def test_unobserved_counterfactual_targets_do_not_change_loss(self):
        expert = _candidate("expert", "expert")
        vla = replace(
            _candidate("vla", "vla"),
            outcome_observed=False,
            safety_observed=False,
            repair_success=None,
        )
        pair = OutcomePairExample(
            "masked",
            "Town01",
            "free_flow",
            0,
            "ClearNoon",
            "h6_train",
            (expert, vla),
        )
        _, _, targets = _batch([pair], torch.device("cpu"), swap=False)
        outputs = torch.randn((1, 2, WORLD_V3_OUTPUT_DIM), generator=torch.Generator().manual_seed(7))
        original, _ = world_v3_loss(outputs, targets)
        changed = {key: value.clone() for key, value in targets.items()}
        for key in (
            "objective", "progress", "completion", "collision", "red",
            "offroad", "jerk", "accel", "lat_accel", "trust",
        ):
            changed[key][0, 1] += 1000.0
        modified, _ = world_v3_loss(outputs, changed)
        self.assertAlmostEqual(float(original), float(modified), places=7)

    def test_objective_penalizes_each_hard_hazard(self):
        safe = {"route_progress_m": 3.0}
        collision = {**safe, "collision_count": 1}
        red = {**safe, "red_light_violation": True}
        offroad = {**safe, "off_corridor_duration_s": 1.0}
        self.assertLess(objective_target(collision), objective_target(safe))
        self.assertLess(objective_target(red), objective_target(safe))
        self.assertLess(objective_target(offroad), objective_target(safe))


class VLADeploymentCalibrationTest(unittest.TestCase):
    def test_calibration_can_reach_90_by_excluding_predicted_unsafe_tail(self):
        rows = []
        for index in range(10):
            risky = index == 9
            rows.append(
                CalibrationRow(
                    pair_id=str(index),
                    vla_prediction=_prediction(
                        "vla", trust_probability=0.95, unsafe_probability=0.90 if risky else 0.02
                    ),
                    expert_prediction=_prediction("expert", trust_probability=0.9, unsafe_probability=0.01),
                    vla_unsafe=risky,
                    expert_unsafe=False,
                    vla_progress_m=2.0,
                    expert_progress_m=2.0,
                )
            )
        calibration = calibrate_vla_deployment(rows)
        self.assertTrue(calibration.passed)
        self.assertGreaterEqual(calibration.vla_coverage, 0.90)
        self.assertLessEqual(calibration.unsafe_delta, 0.01)

    def test_calibration_fails_instead_of_faking_90(self):
        rows = [
            CalibrationRow(
                pair_id=str(index),
                vla_prediction=_prediction("vla", trust_probability=0.95, unsafe_probability=0.01),
                expert_prediction=_prediction("expert", trust_probability=0.9, unsafe_probability=0.01),
                vla_unsafe=True,
                expert_unsafe=False,
                vla_progress_m=2.0,
                expert_progress_m=2.0,
            )
            for index in range(10)
        ]
        calibration = calibrate_vla_deployment(rows)
        self.assertFalse(calibration.passed)
        self.assertEqual(
            calibration.reason,
            "no_dev_threshold_meets_coverage_safety_progress_and_high_trust",
        )

    def test_policy_first_tick_cannot_fake_tickwise_90_percent_coverage(self):
        tick_rows = []
        for index in range(10):
            prefers_vla = index < 8
            tick_rows.append(
                CalibrationRow(
                    pair_id=f"tick-{index}",
                    vla_prediction=_prediction(
                        "vla",
                        deployment_score=2.0 if prefers_vla else 0.5,
                        trust_probability=0.95,
                        unsafe_probability=0.01,
                    ),
                    expert_prediction=_prediction(
                        "expert",
                        deployment_score=1.0,
                        trust_probability=0.95,
                        unsafe_probability=0.01,
                    ),
                    vla_unsafe=False,
                    expert_unsafe=False,
                    vla_progress_m=2.0,
                    expert_progress_m=1.0,
                )
            )
        policy_rows = [
            CalibrationRow(
                pair_id="policy-first-tick",
                vla_prediction=_prediction(
                    "vla",
                    deployment_score=2.0,
                    trust_probability=0.95,
                    unsafe_probability=0.01,
                ),
                expert_prediction=_prediction(
                    "expert",
                    deployment_score=1.0,
                    trust_probability=0.95,
                    unsafe_probability=0.01,
                ),
                vla_unsafe=False,
                expert_unsafe=False,
                vla_progress_m=3.0,
                expert_progress_m=1.0,
            )
        ]
        calibration = calibrate_vla_deployment(
            tick_rows, policy_rows=policy_rows
        )
        self.assertFalse(calibration.passed)
        self.assertEqual(calibration.rows, 10)
        self.assertEqual(calibration.policy_rows, 1)

    def test_high_trust_does_not_fake_a_high_vla_world_score(self):
        rows = [
            CalibrationRow(
                pair_id=str(index),
                vla_prediction=_prediction(
                    "vla",
                    trust_probability=0.99,
                    unsafe_probability=0.01,
                    deployment_score=0.0,
                ),
                expert_prediction=_prediction(
                    "expert",
                    trust_probability=0.9,
                    unsafe_probability=0.01,
                    deployment_score=1.0,
                ),
                vla_unsafe=False,
                expert_unsafe=False,
                vla_progress_m=2.0,
                expert_progress_m=2.0,
            )
            for index in range(10)
        ]
        self.assertFalse(calibrate_vla_deployment(rows).passed)


class ActualExecutionAcceptanceTest(unittest.TestCase):
    @staticmethod
    def _runs(vla_ticks=50):
        decisions = []
        for tick in range(50):
            source = "vla" if tick < vla_ticks else "mrm"
            decisions.append(
                {
                    "tick": tick,
                    "executed_source": source,
                    "scorer_latency_ms": 2.0,
                    "guard": {
                        "frame:vla": {"verdict": "REVIEW"},
                        "frame:expert": {"verdict": "PASS"},
                    },
                    "routing": {
                        "selected_candidate_id": "frame:vla",
                        "pass_candidate_ids": ["frame:vla", "frame:expert"],
                        "preference_order": ["frame:vla", "frame:expert"],
                    },
                    "world_score": {
                        "trust_threshold": 0.5,
                        "risk_ceiling": 0.2,
                        "predictions": [
                            {
                                "candidate_key": "frame:vla",
                                "deployment_score": 2.0,
                                "trust_probability": 0.9,
                                "unsafe_probability": 0.05,
                            },
                            {
                                "candidate_key": "frame:expert",
                                "deployment_score": 1.0,
                                "trust_probability": 0.9,
                                "unsafe_probability": 0.05,
                            },
                        ],
                    },
                    "repair": None,
                    "arbitration": {"notes": [], "audits": []},
                }
            )
        target = {
            "pair_id": "p1",
            "arm": "on",
            "ok": True,
            "ticks_executed": 50,
            "decisions": decisions,
            "route_progress_m": 3.0,
            "collision_count": 0,
            "red_light_violation": False,
            "off_corridor_duration_s": 0.0,
            "actual_switch_count": 0,
            "scorer_deadline_misses": 0,
            "config_sha256": H6_VLA90_CONFIG_SHA256,
            "manifest_kind": "h6_fresh",
            "physical_sha256": "physical-p1",
            "reset_comparison": {"comparable": True},
        }
        baseline_decisions = [
            {"tick": tick, "executed_source": "expert"} for tick in range(50)
        ]
        baseline = {
            **target,
            "arm": "off",
            "decisions": baseline_decisions,
            "ticks_executed": 50,
            "vla_executed_ticks": 0,
            "expert_executed_ticks": 50,
            "reset_comparison": None,
        }
        return [target, baseline]

    def test_gate_counts_actual_execution_not_world_preference(self):
        result = evaluate_vla90_gate(self._runs(vla_ticks=50), target_vla_coverage=0.90)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["coverage"]["actual_vla"], 1.0)

    def test_gate_fails_when_actual_vla_is_below_90(self):
        result = evaluate_vla90_gate(self._runs(vla_ticks=44), target_vla_coverage=0.90)
        self.assertFalse(result["passed"])
        self.assertIn("actual_vla_coverage", result["failures"])

    def test_gate_requires_world_to_really_prefer_vla(self):
        runs = self._runs(vla_ticks=50)
        for decision in runs[0]["decisions"]:
            decision["routing"]["selected_candidate_id"] = "frame:expert"
            decision["routing"]["preference_order"] = ["frame:expert", "frame:vla"]
            decision["world_score"]["predictions"][0]["deployment_score"] = 0.0
        result = evaluate_vla90_gate(runs, target_vla_coverage=0.90)
        self.assertFalse(result["passed"])
        self.assertIn("world_vla_preference", result["failures"])

    def test_single_surviving_vla_does_not_fake_world_high_score(self):
        runs = self._runs(vla_ticks=50)
        for decision in runs[0]["decisions"]:
            decision["world_score"]["predictions"] = [
                decision["world_score"]["predictions"][0]
            ]
        result = evaluate_vla90_gate(runs, target_vla_coverage=0.90)
        self.assertFalse(result["passed"])
        self.assertIn("world_vla_preference", result["failures"])
        self.assertIn("provenance", result["failures"])

    def test_vla_contaminated_baseline_is_rejected(self):
        runs = self._runs(vla_ticks=50)
        runs[1]["vla_executed_ticks"] = 1
        runs[1]["decisions"][0]["executed_source"] = "vla"
        result = evaluate_vla90_gate(runs, target_vla_coverage=0.90)
        self.assertFalse(result["passed"])
        self.assertIn("provenance", result["failures"])
        self.assertTrue(
            any(
                item.startswith("baseline_not_classic_only")
                for item in result["provenance_failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
