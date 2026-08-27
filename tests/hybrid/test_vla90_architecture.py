"""Regression tests for the VLA-primary Guard/World/Safety contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from safety_kernel.arbitration.pipeline import ArbitrationPipeline  # noqa: E402
from safety_kernel.config import load_safety_config  # noqa: E402
from safety_kernel.contracts.schema import SCHEMA_VERSION  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ComponentAvailability,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyMode,
    TrackedObject,
    TrafficLightObs,
    TrajectoryPoint,
)
from safety_kernel.validator.checks import check_collision, check_rules  # noqa: E402
from safety_kernel.validator.engine import TrajectoryValidator  # noqa: E402


def _points(*, y: float = 0.0, speed: float = 2.0, step: float = 0.5):
    return tuple(
        TrajectoryPoint(
            t=0.25 * (index + 1),
            x=step * (index + 1),
            y=y,
            yaw=0.0,
            kappa=0.0,
            v=speed,
            a=0.0,
        )
        for index in range(10)
    )


def _candidate(candidate_id: str, source: CandidateSource, *, y: float = 0.0):
    return PolicyCandidate(
        candidate_id=candidate_id,
        source=source,
        generated_time_s=1.0,
        valid_until_s=2.0,
        probability=0.9,
        uncertainty=0.2 if source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW} else 0.0,
        points=_points(y=y),
    )


def _snapshot(*, actors=(), lights=()):
    return ObservableSnapshot(
        run_id="run-vla90",
        frame_id="frame-20",
        scenario_id="scenario-vla90",
        simulation_time_s=1.0,
        wall_time_s=10.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=2.0,
        observed_time_s=1.0,
        freshness_s=0.0,
        speed_limit_mps=12.0,
        actors=tuple(actors),
        traffic_lights=tuple(lights),
        corridor_centerline=tuple((float(index), 0.0) for index in range(40)),
        corridor_half_width_m=1.75,
    )


class GeometryAndRuleFixTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_safety_config()

    def test_adjacent_lane_is_not_circle_false_positive(self):
        actor = TrackedObject(
            actor_id="adjacent",
            class_name="vehicle.test",
            x=0.5,
            y=3.6,
            yaw=0.0,
            vx=0.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=1.0,
        )
        margin = check_collision(
            _candidate("vla", CandidateSource.VLA_FAST),
            _snapshot(actors=(actor,)),
            self.cfg,
        )
        self.assertGreater(margin.margin, 0.0)

    def test_actor_prediction_uses_first_candidate_time(self):
        actor = TrackedObject(
            actor_id="moving",
            class_name="vehicle.test",
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=20.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=1.0,
        )
        fast_points = _points(speed=20.0, step=5.0)
        candidate = PolicyCandidate(
            candidate_id="vla-fast",
            source=CandidateSource.VLA_FAST,
            generated_time_s=1.0,
            valid_until_s=2.0,
            probability=0.9,
            points=fast_points,
        )
        margin = check_collision(candidate, _snapshot(actors=(actor,)), self.cfg)
        self.assertLess(margin.margin, 0.0)
        self.assertAlmostEqual(margin.first_violation_time_s, 0.25)

    def test_red_light_allows_brakeable_approach_but_rejects_crossing(self):
        light = TrafficLightObs(
            light_id="red",
            state="red",
            distance_m=8.0,
            stop_line_distance_m=8.0,
            controls_ego_lane=True,
            observed_time_s=1.0,
        )
        brakeable = PolicyCandidate(
            candidate_id="brakeable",
            source=CandidateSource.VLA_FAST,
            generated_time_s=1.0,
            valid_until_s=2.0,
            probability=0.9,
            points=_points(speed=4.0, step=0.5),
        )
        crossing = PolicyCandidate(
            candidate_id="crossing",
            source=CandidateSource.VLA_FAST,
            generated_time_s=1.0,
            valid_until_s=2.0,
            probability=0.9,
            points=_points(speed=4.0, step=1.0),
        )
        self.assertGreaterEqual(check_rules(brakeable, _snapshot(lights=(light,)), self.cfg).margin, 0.0)
        crossing_margin = check_rules(crossing, _snapshot(lights=(light,)), self.cfg)
        self.assertLess(crossing_margin.margin, 0.0)
        self.assertIn(crossing_margin.message, {"red_light_unstoppable", "red_light_crossing"})


class PreferenceAndFallbackTest(unittest.TestCase):
    def test_vla_is_repair_attempted_then_same_tick_expert_is_used(self):
        cfg = load_safety_config()
        validator = TrajectoryValidator(cfg)
        pipeline = ArbitrationPipeline(cfg, validator)
        vla = _candidate("vla-first", CandidateSource.VLA_FAST, y=3.0)
        expert = _candidate("expert-fallback", CandidateSource.CLASSIC)
        candidate_set = PolicyCandidateSet(
            run_id="run-vla90",
            frame_id="frame-20",
            scenario_id="scenario-vla90",
            model_id="world-v3-test",
            carla_frame=20,
            simulation_time_s=1.0,
            wall_time_s=10.0,
            candidates=(vla, expert),
            schema_version=SCHEMA_VERSION,
            preference_order=(vla.candidate_id, expert.candidate_id),
        )
        repair_calls = []

        def repair_fn(**kwargs):
            repair_calls.append(tuple(c.candidate_id for c in kwargs["candidate_set"].candidates))
            return None, None, []

        result = pipeline.run_candidate_pipeline(
            obs=_snapshot(),
            candidate_set=candidate_set,
            availability=ComponentAvailability(classic=True, vla=True, world=True, safety=True),
            mode=SafetyMode.NORMAL,
            now_s=1.0,
            repair_fn=repair_fn,
            state_emergency=False,
        )
        self.assertEqual(repair_calls[0], (vla.candidate_id,))
        self.assertEqual(result.decision.executed_trajectory_id, expert.candidate_id)
        self.assertEqual(result.decision.decision_kind.value, "CLASSIC_FALLBACK")
        self.assertEqual(result.arbitration.ranked_ids, (vla.candidate_id, expert.candidate_id))
        self.assertIn("preferred_vla_repair_failed_try_expert", result.arbitration.notes)

    def test_lower_ranked_candidate_is_safety_audited_without_changing_vla_winner(self):
        cfg = load_safety_config()
        pipeline = ArbitrationPipeline(cfg, TrajectoryValidator(cfg))
        vla = _candidate("vla-winner", CandidateSource.VLA_FAST)
        expert = _candidate("expert-audit-only", CandidateSource.CLASSIC, y=3.0)
        candidate_set = PolicyCandidateSet(
            run_id="run-vla90",
            frame_id="frame-20",
            scenario_id="scenario-vla90",
            model_id="world-v3-test",
            carla_frame=21,
            simulation_time_s=1.0,
            wall_time_s=10.0,
            candidates=(vla, expert),
            schema_version=SCHEMA_VERSION,
            preference_order=(vla.candidate_id, expert.candidate_id),
        )
        result = pipeline.run_candidate_pipeline(
            obs=_snapshot(),
            candidate_set=candidate_set,
            availability=ComponentAvailability(
                classic=True, vla=True, world=True, safety=True
            ),
            mode=SafetyMode.NORMAL,
            now_s=1.0,
            repair_fn=lambda **_kwargs: (None, None, []),
            state_emergency=False,
        )
        self.assertEqual(result.decision.executed_trajectory_id, vla.candidate_id)
        audits = {audit.candidate_id: audit for audit in result.arbitration.audits}
        self.assertTrue(audits[vla.candidate_id].final_ok)
        self.assertFalse(audits[expert.candidate_id].final_ok)
        self.assertFalse(audits[expert.candidate_id].selected)
        self.assertFalse(
            any(
                expert.candidate_id in reason
                for reason in result.decision.reject_reasons
            )
        )


if __name__ == "__main__":
    unittest.main()
