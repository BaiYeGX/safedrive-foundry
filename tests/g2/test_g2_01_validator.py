"""G2-01 Trajectory Validator hard-check tests."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from safety_kernel import (  # noqa: E402
    SCHEMA_VERSION,
    ComponentAvailability,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
    TrajectoryPoint,
    TrajectoryValidator,
    load_safety_config,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
    TrackedObject,
    TrafficLightObs,
)


def _straight(
    *,
    n: int = 16,
    dt: float = 0.25,
    v: float = 5.0,
    y: float = 0.0,
    kappa: float = 0.0,
    a: float = 0.0,
) -> tuple[TrajectoryPoint, ...]:
    pts = []
    x = 0.0
    speed = v
    for i in range(n):
        t = i * dt
        pts.append(TrajectoryPoint(t=t, x=x, y=y, yaw=0.0, kappa=kappa, v=speed, a=a, jerk=0.0))
        x += speed * dt
        speed = max(0.0, speed + a * dt)
    return tuple(pts)


def _obs(**kwargs) -> ObservableSnapshot:
    base = dict(
        run_id="run-v",
        frame_id="f0",
        scenario_id="sc",
        simulation_time_s=1.0,
        wall_time_s=10.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=5.0,
        observed_time_s=1.0,
        corridor_centerline=tuple((float(i), 0.0) for i in range(0, 80)),
        corridor_half_width_m=1.75,
        privilege=ObservationPrivilege.OBSERVABLE,
    )
    base.update(kwargs)
    return ObservableSnapshot(**base)


def _set(points=None, **cand_kw) -> PolicyCandidateSet:
    now = 1.0
    pts = points if points is not None else _straight()
    cand = PolicyCandidate(
        candidate_id=cand_kw.pop("candidate_id", "legal"),
        source=cand_kw.pop("source", CandidateSource.CLASSIC),
        generated_time_s=cand_kw.pop("generated_time_s", now),
        valid_until_s=cand_kw.pop("valid_until_s", now + 0.2),
        probability=cand_kw.pop("probability", 1.0),
        points=pts,
        **cand_kw,
    )
    return PolicyCandidateSet(
        run_id="run-v",
        frame_id="f0",
        scenario_id="sc",
        model_id="classic",
        carla_frame=1,
        simulation_time_s=now,
        wall_time_s=10.0,
        candidates=(cand,),
        schema_version=SCHEMA_VERSION,
    )


class G201ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.validator = TrajectoryValidator(self.cfg)
        self.avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)

    def test_legal_classic_accepted_without_learning(self) -> None:
        result = self.validator.validate_candidates(_set(), _obs(), availability=self.avail)
        self.assertEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertIsNotNone(result.accepted)
        self.assertFalse(result.decision.learning_modules_required)

    def test_nan_rejected(self) -> None:
        pts = list(_straight())
        bad = list(pts)
        p = pts[3]
        bad[3] = TrajectoryPoint(t=p.t, x=float("nan"), y=p.y, yaw=p.yaw, kappa=p.kappa, v=p.v, a=p.a, jerk=p.jerk)
        # Schema path: build candidate directly (bypass serialize finite check in from_dict).
        cand_set = _set(points=tuple(bad))
        result = self.validator.validate_candidates(cand_set, _obs(), availability=self.avail)
        self.assertNotEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertTrue(any("numeric" in r for r in result.decision.reject_reasons))

    def test_expired_rejected(self) -> None:
        result = self.validator.validate_candidates(
            _set(generated_time_s=0.0, valid_until_s=0.5),
            _obs(simulation_time_s=1.0),
            now_s=1.0,
            availability=self.avail,
        )
        self.assertNotEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertTrue(any("freshness" in r for r in result.decision.reject_reasons))

    def test_out_of_order_time_rejected(self) -> None:
        pts = list(_straight())
        p5 = pts[5]
        p4 = pts[4]
        pts[5] = TrajectoryPoint(t=p4.t - 0.01, x=p5.x, y=p5.y, yaw=p5.yaw, kappa=p5.kappa, v=p5.v, a=p5.a, jerk=p5.jerk)
        result = self.validator.validate_candidates(_set(points=tuple(pts)), _obs(), availability=self.avail)
        self.assertTrue(any("time_order" in r for r in result.decision.reject_reasons))

    def test_extreme_speed_rejected(self) -> None:
        result = self.validator.validate_candidates(_set(points=_straight(v=40.0)), _obs(), availability=self.avail)
        self.assertTrue(any("dynamics" in r for r in result.decision.reject_reasons))

    def test_collision_envelope_rejected(self) -> None:
        actor = TrackedObject(
            actor_id="a1",
            class_name="vehicle",
            x=5.0,
            y=0.0,
            yaw=0.0,
            vx=0.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=1.0,
        )
        result = self.validator.validate_candidates(
            _set(points=_straight(v=5.0)),
            _obs(actors=(actor,)),
            availability=self.avail,
        )
        self.assertTrue(any("collision" in r for r in result.decision.reject_reasons))

    def test_offroad_rejected(self) -> None:
        result = self.validator.validate_candidates(
            _set(points=_straight(y=10.0)),
            _obs(),
            availability=self.avail,
        )
        self.assertTrue(any("road" in r for r in result.decision.reject_reasons))

    def test_red_light_rule_rejected(self) -> None:
        light = TrafficLightObs(light_id="tl1", state="red", distance_m=4.0, observed_time_s=1.0)
        result = self.validator.validate_candidates(
            _set(points=_straight(v=8.0)),
            _obs(traffic_lights=(light,)),
            availability=self.avail,
        )
        self.assertTrue(any("rules" in r for r in result.decision.reject_reasons))

    def test_teleport_trackability_rejected(self) -> None:
        pts = list(_straight())
        p = pts[8]
        pts[8] = TrajectoryPoint(t=p.t, x=p.x + 50.0, y=p.y, yaw=p.yaw, kappa=p.kappa, v=p.v, a=p.a, jerk=p.jerk)
        result = self.validator.validate_candidates(_set(points=tuple(pts)), _obs(), availability=self.avail)
        self.assertTrue(any("trackability" in r for r in result.decision.reject_reasons))

    def test_oracle_privilege_rejected_at_runtime(self) -> None:
        result = self.validator.validate_candidates(
            _set(),
            _obs(privilege=ObservationPrivilege.ORACLE),
            availability=self.avail,
        )
        self.assertNotEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertTrue(any("privilege" in r for r in result.decision.reject_reasons))

    def test_vla_source_dropped_when_vla_unavailable(self) -> None:
        vla = _set(source=CandidateSource.VLA_FAST, candidate_id="vla1")
        result = self.validator.validate_candidates(vla, _obs(), availability=self.avail)
        self.assertNotEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertTrue(any("vla_unavailable" in r for r in result.decision.reject_reasons))

    def test_learning_all_failed_classic_fallback_path(self) -> None:
        # No candidates at all → MINIMAL_RISK when classic "available" but empty set rejected.
        empty = PolicyCandidateSet(
            run_id="run-v",
            frame_id="f0",
            scenario_id="sc",
            model_id="none",
            carla_frame=1,
            simulation_time_s=1.0,
            wall_time_s=10.0,
            candidates=(),
            schema_version=SCHEMA_VERSION,
        )
        result = self.validator.validate_candidates(empty, _obs(), availability=self.avail)
        self.assertIn(
            result.decision.decision_kind,
            {DecisionKind.CLASSIC_FALLBACK, DecisionKind.MINIMAL_RISK, DecisionKind.EMERGENCY},
        )
        self.assertIsNotNone(result.decision.fallback_request)
        self.assertTrue(result.events)  # must not silently swallow

    def test_state_check_stale_observation(self) -> None:
        result = self.validator.check_state(_obs(observed_time_s=0.0, simulation_time_s=1.0), now_s=1.0)
        self.assertNotEqual(result.decision.decision_kind, DecisionKind.ACCEPT)


if __name__ == "__main__":
    unittest.main()
