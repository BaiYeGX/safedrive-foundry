"""G2-01 G1 trajectory replay, property tests, learning-fail paths."""

from __future__ import annotations

import copy
import math
import random
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
    SafetyMode,
    TrajectoryPoint,
    TrajectoryValidator,
    load_safety_config,
)
from safety_kernel.adapters.g1_trajectory import (  # noqa: E402
    g1_plan_result_to_candidate_set,
    load_g1_trajectory_json,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
)
from safety_kernel.state_machine import SafetyStateMachine  # noqa: E402

G1_SAMPLE = ROOT / "docs/architecture/evidence/g1-04/sample_follow_trajectory.json"


def _corridor_for_traj(points: list[dict], half: float = 2.0) -> tuple[tuple[float, float], ...]:
    return tuple((float(p["x"]), float(p["y"])) for p in points[::2])


class G201ReplayFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_safety_config()
        cls.plan = load_g1_trajectory_json(G1_SAMPLE)

    def setUp(self) -> None:
        self.validator = TrajectoryValidator(self.cfg)
        self.sm = SafetyStateMachine(self.cfg)
        self.classic_only = ComponentAvailability(
            classic=True,
            vla=False,
            world=False,
            safety=True,
            detail={"vla": "offline", "world": "offline"},
        )

    def test_g1_follow_trajectory_replay_accepts(self) -> None:
        self.assertTrue(G1_SAMPLE.is_file())
        cset = g1_plan_result_to_candidate_set(
            self.plan,
            run_id="g2-01-replay",
            frame_id="replay-0",
            scenario_id="g1-04-follow",
            simulation_time_s=0.0,
            wall_time_s=0.0,
            now_s=0.0,
        )
        self.assertEqual(len(cset.candidates), 1)
        pts = self.plan["trajectory"]["points"]
        obs = ObservableSnapshot(
            run_id="g2-01-replay",
            frame_id="replay-0",
            scenario_id="g1-04-follow",
            simulation_time_s=0.0,
            wall_time_s=0.0,
            ego_x=float(pts[0]["x"]),
            ego_y=float(pts[0]["y"]),
            ego_yaw=float(pts[0]["yaw"]),
            ego_v=float(pts[0]["v"]),
            observed_time_s=0.0,
            corridor_centerline=_corridor_for_traj(pts),
            corridor_half_width_m=2.5,
            privilege=ObservationPrivilege.OBSERVABLE,
        )
        result = self.validator.validate_candidates(cset, obs, availability=self.classic_only, now_s=0.0)
        self.assertEqual(result.decision.decision_kind, DecisionKind.ACCEPT, msg=result.decision.reject_reasons)
        self.assertIsNotNone(result.accepted)
        # Learning fully off — still works.
        self.assertTrue(self.classic_only.learning_all_failed)

    def test_intentional_fault_injection_matrix(self) -> None:
        base_plan = copy.deepcopy(self.plan)
        pts = base_plan["trajectory"]["points"]
        cases = {
            "nan_v": lambda p: {**p[0], "v": float("nan")},
            "shuffle_time": None,
            "huge_kappa": lambda p: {**p[5], "kappa": 2.0},
        }
        # nan
        plan_nan = copy.deepcopy(base_plan)
        plan_nan["trajectory"]["points"][0]["v"] = float("nan")
        # time disorder
        plan_ord = copy.deepcopy(base_plan)
        plan_ord["trajectory"]["points"][4]["t"] = plan_ord["trajectory"]["points"][3]["t"] - 0.05
        # extreme kappa
        plan_k = copy.deepcopy(base_plan)
        plan_k["trajectory"]["points"][5]["kappa"] = 2.0

        for name, plan in [("nan", plan_nan), ("order", plan_ord), ("kappa", plan_k)]:
            with self.subTest(case=name):
                cset = g1_plan_result_to_candidate_set(
                    plan,
                    run_id="fault",
                    frame_id=name,
                    scenario_id="inject",
                    simulation_time_s=0.0,
                    now_s=0.0,
                )
                obs = ObservableSnapshot(
                    run_id="fault",
                    frame_id=name,
                    scenario_id="inject",
                    simulation_time_s=0.0,
                    wall_time_s=0.0,
                    ego_x=0.0,
                    ego_y=0.0,
                    ego_yaw=0.0,
                    ego_v=5.0,
                    observed_time_s=0.0,
                    corridor_centerline=_corridor_for_traj(pts),
                    corridor_half_width_m=3.0,
                )
                result = self.validator.validate_candidates(cset, obs, availability=self.classic_only, now_s=0.0)
                self.assertNotEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
                self.assertTrue(result.decision.reject_reasons or result.decision.fallback_request)

    def test_property_random_corruptions_deterministic_reject(self) -> None:
        rng = random.Random(42)
        base = g1_plan_result_to_candidate_set(
            self.plan,
            run_id="prop",
            frame_id="0",
            scenario_id="prop",
            simulation_time_s=0.0,
            now_s=0.0,
        )
        cand = base.candidates[0]
        pts = list(cand.points)
        obs = ObservableSnapshot(
            run_id="prop",
            frame_id="0",
            scenario_id="prop",
            simulation_time_s=0.0,
            wall_time_s=0.0,
            ego_x=pts[0].x,
            ego_y=pts[0].y,
            ego_yaw=pts[0].yaw,
            ego_v=pts[0].v,
            observed_time_s=0.0,
            corridor_centerline=tuple((p.x, p.y) for p in pts[::2]),
            corridor_half_width_m=3.0,
        )
        rejects = 0
        for i in range(40):
            corrupted = list(pts)
            idx = rng.randrange(len(corrupted))
            p = corrupted[idx]
            mode = rng.choice(["nan", "inf", "speed", "teleport", "time"])
            if mode == "nan":
                corrupted[idx] = TrajectoryPoint(p.t, float("nan"), p.y, p.yaw, p.kappa, p.v, p.a, p.jerk)
            elif mode == "inf":
                corrupted[idx] = TrajectoryPoint(p.t, p.x, float("inf"), p.yaw, p.kappa, p.v, p.a, p.jerk)
            elif mode == "speed":
                corrupted[idx] = TrajectoryPoint(p.t, p.x, p.y, p.yaw, p.kappa, 99.0, p.a, p.jerk)
            elif mode == "teleport":
                corrupted[idx] = TrajectoryPoint(p.t, p.x + 80.0, p.y + 80.0, p.yaw, p.kappa, p.v, p.a, p.jerk)
            else:
                corrupted[idx] = TrajectoryPoint(p.t - 1.0, p.x, p.y, p.yaw, p.kappa, p.v, p.a, p.jerk)
            bad = PolicyCandidate(
                candidate_id=f"bad-{i}",
                source=CandidateSource.CLASSIC,
                generated_time_s=0.0,
                valid_until_s=0.2,
                probability=1.0,
                points=tuple(corrupted),
            )
            cset = PolicyCandidateSet(
                run_id="prop",
                frame_id=str(i),
                scenario_id="prop",
                model_id="classic",
                carla_frame=i,
                simulation_time_s=0.0,
                wall_time_s=0.0,
                candidates=(bad,),
                schema_version=SCHEMA_VERSION,
            )
            result = self.validator.validate_candidates(cset, obs, availability=self.classic_only, now_s=0.0)
            if result.decision.decision_kind != DecisionKind.ACCEPT:
                rejects += 1
        self.assertEqual(rejects, 40)

    def test_learning_fail_emergency_when_no_classic(self) -> None:
        avail = ComponentAvailability(classic=False, vla=False, world=False, safety=True)
        empty = PolicyCandidateSet(
            run_id="x",
            frame_id="f",
            scenario_id="s",
            model_id="none",
            carla_frame=0,
            simulation_time_s=0.0,
            wall_time_s=0.0,
            candidates=(),
            schema_version=SCHEMA_VERSION,
        )
        obs = ObservableSnapshot(
            run_id="x",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=0.0,
            wall_time_s=0.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=0.0,
            observed_time_s=0.0,
        )
        result = self.validator.validate_candidates(empty, obs, availability=avail, now_s=0.0)
        self.assertEqual(result.decision.decision_kind, DecisionKind.EMERGENCY)
        tr = self.sm.step(result.decision, avail, now_s=0.0, frame_id="f")
        self.assertIsNotNone(tr)
        self.assertEqual(self.sm.mode, SafetyMode.EMERGENCY)


if __name__ == "__main__":
    unittest.main()
