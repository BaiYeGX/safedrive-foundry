"""G2-01 SafetyKernel facade, prefilter/final, classic after VLA drop, ROS dicts."""

from __future__ import annotations

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
    SafetyKernel,
    SafetyMode,
    TrajectoryPoint,
    ValidationStage,
    load_safety_config,
)
from safety_kernel.adapters import (  # noqa: E402
    candidate_to_trajectory_dict,
    decision_to_policy_decision_dict,
    decision_to_safety_status_dict,
)
from safety_kernel.contracts.types import CandidateSource, DecisionKind, ObservationPrivilege  # noqa: E402
from safety_kernel.oracle_offline import OracleUsedAtRuntimeError, evaluate_with_oracle  # noqa: E402


def _pts(n: int = 16, v: float = 5.0, y: float = 0.0) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(TrajectoryPoint(t=0.25 * i, x=x, y=y, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
        x += v * 0.25
    return tuple(out)


def _obs(now: float = 1.0) -> ObservableSnapshot:
    return ObservableSnapshot(
        run_id="k",
        frame_id="f",
        scenario_id="s",
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=5.0,
        observed_time_s=now,
        corridor_centerline=tuple((float(i), 0.0) for i in range(0, 100)),
        corridor_half_width_m=2.0,
        privilege=ObservationPrivilege.OBSERVABLE,
    )


def _cand(cid: str, source: CandidateSource, *, now: float = 1.0, pts=None, **kw) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=cid,
        source=source,
        generated_time_s=now,
        valid_until_s=now + 0.2,
        probability=kw.pop("probability", 1.0),
        points=pts or _pts(),
        **kw,
    )


class G201KernelIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.kernel = SafetyKernel(self.cfg)

    def test_tick_accepts_classic_only(self) -> None:
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="classic",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("c1", CandidateSource.CLASSIC),),
            schema_version=SCHEMA_VERSION,
        )
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        result = self.kernel.tick(_obs(), cset, availability=avail, now_s=1.0)
        self.assertEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertEqual(result.mode, SafetyMode.NORMAL)
        self.assertEqual(result.accepted_trajectory_id, "c1")

    def test_vla_dropped_classic_selected(self) -> None:
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="mixed",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(
                _cand("vla1", CandidateSource.VLA_FAST, probability=0.9),
                _cand("classic1", CandidateSource.CLASSIC, probability=0.5),
            ),
            schema_version=SCHEMA_VERSION,
        )
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        result = self.kernel.tick(_obs(), cset, availability=avail, now_s=1.0)
        self.assertEqual(result.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertEqual(result.decision.final_candidate_id, "classic1")
        self.assertTrue(any("vla_unavailable" in r for r in result.decision.reject_reasons))
        self.assertTrue(any("selected_classic_after_learning_drop" in r for r in result.decision.reject_reasons))

    def test_prefilter_allows_soft_road_extreme_skipped_on_prefilter_only(self) -> None:
        # Extreme lateral offset fails FINAL road but PREFILTER only checks numeric/schema/time.
        offroad = _pts(y=20.0)
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("off", CandidateSource.CLASSIC, pts=offroad),),
            schema_version=SCHEMA_VERSION,
        )
        pre = self.kernel.validator.validate_candidates(
            cset, _obs(), now_s=1.0, stage=ValidationStage.PREFILTER
        )
        fin = self.kernel.validator.validate_candidates(
            cset, _obs(), now_s=1.0, stage=ValidationStage.FINAL
        )
        self.assertEqual(pre.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertNotEqual(fin.decision.decision_kind, DecisionKind.ACCEPT)

    def test_hard_reject_all_illegal(self) -> None:
        bad = list(_pts())
        p = bad[2]
        bad[2] = TrajectoryPoint(p.t, float("nan"), p.y, p.yaw, p.kappa, p.v, p.a, p.jerk)
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("bad", CandidateSource.CLASSIC, pts=tuple(bad)),),
            schema_version=SCHEMA_VERSION,
        )
        result = self.kernel.tick(_obs(), cset, now_s=1.0)
        self.assertIn(
            result.decision.decision_kind,
            {DecisionKind.HARD_REJECT, DecisionKind.MINIMAL_RISK},
        )
        self.assertIsNotNone(result.decision.fallback_request)
        self.assertTrue(result.events)

    def test_ros_adapters_shape(self) -> None:
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("c1", CandidateSource.CLASSIC),),
            schema_version=SCHEMA_VERSION,
        )
        result = self.kernel.tick(_obs(), cset, now_s=1.0)
        status = decision_to_safety_status_dict(result.decision, mode=result.mode)
        self.assertIn(status["level"], {1, 2, 3, 4})
        self.assertFalse(status["oracle_observation"])
        policy = decision_to_policy_decision_dict(result.decision)
        self.assertEqual(policy["policy_kind"], 1)  # CLASSIC_EXPERT
        traj = candidate_to_trajectory_dict(result.candidate_result.accepted)  # type: ignore[union-attr]
        self.assertEqual(traj["trajectory_id"], "c1")
        self.assertGreaterEqual(len(traj["points"]), 3)

    def test_state_emergency_skips_candidates(self) -> None:
        """Non-finite ego / Oracle privilege: state lock, no candidate path or repair."""
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("c1", CandidateSource.CLASSIC),),
            schema_version=SCHEMA_VERSION,
        )
        nan_obs = ObservableSnapshot(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            wall_time_s=1.0,
            ego_x=float("nan"),
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            observed_time_s=1.0,
            corridor_centerline=tuple((float(i), 0.0) for i in range(0, 100)),
            corridor_half_width_m=2.0,
            privilege=ObservationPrivilege.OBSERVABLE,
        )
        k = SafetyKernel(self.cfg)
        before = len(k.validator.latency_candidate_ms)
        res = k.tick(nan_obs, cset, now_s=1.0)
        self.assertEqual(res.decision.decision_kind, DecisionKind.EMERGENCY)
        self.assertIsNone(res.decision.executed_trajectory_id)
        self.assertTrue(res.repair_result is None or not res.repair_result.success)
        self.assertEqual(len(k.validator.latency_candidate_ms), before)
        self.assertIn("state_locked", " ".join(res.arbitration.stages) if res.arbitration else "")

        oracle_obs = ObservableSnapshot(
            run_id="k",
            frame_id="f2",
            scenario_id="s",
            simulation_time_s=1.0,
            wall_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            observed_time_s=1.0,
            corridor_centerline=tuple((float(i), 0.0) for i in range(0, 100)),
            corridor_half_width_m=2.0,
            privilege=ObservationPrivilege.ORACLE,
        )
        res2 = k.tick(oracle_obs, cset, now_s=1.0)
        self.assertEqual(res2.decision.decision_kind, DecisionKind.EMERGENCY)
        self.assertIn("state_locked_no_candidate_override", res2.decision.reject_reasons)

    def test_qp_ros_confidence_positive(self) -> None:
        from safety_kernel.contracts.types import TrafficLightObs

        lights = (TrafficLightObs("tl", "red", 6.0, 1.0),)
        obs = ObservableSnapshot(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            wall_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=8.0,
            observed_time_s=1.0,
            corridor_centerline=tuple((float(i), 0.0) for i in range(0, 100)),
            corridor_half_width_m=2.0,
            traffic_lights=lights,
            privilege=ObservationPrivilege.OBSERVABLE,
        )
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("fast", CandidateSource.CLASSIC, pts=_pts(v=8.0)),),
            schema_version=SCHEMA_VERSION,
        )
        res = self.kernel.tick(obs, cset, now_s=1.0)
        if res.decision.decision_kind is DecisionKind.QP and res.decision.executed_trajectory_id:
            policy = decision_to_policy_decision_dict(res.decision)
            self.assertGreater(policy["confidence"], 0.0)

    def test_oracle_offline_ok_and_runtime_guard(self) -> None:
        cand = _cand("c1", CandidateSource.CLASSIC)
        oracle_obs = ObservableSnapshot(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            wall_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            observed_time_s=1.0,
            corridor_centerline=tuple((float(i), 0.0) for i in range(0, 100)),
            corridor_half_width_m=2.0,
            privilege=ObservationPrivilege.ORACLE,
            oracle_fields={"true_ttc": 9.9},
        )
        offline = evaluate_with_oracle(cand, oracle_obs, self.cfg, now_s=1.0)
        self.assertFalse(offline.hard_reject)
        self.assertEqual(offline.privilege, ObservationPrivilege.ORACLE)
        with self.assertRaises(OracleUsedAtRuntimeError):
            evaluate_with_oracle(cand, oracle_obs, self.cfg, now_s=1.0, allow_oracle=False)
        # Runtime validator must reject oracle privilege.
        runtime = self.kernel.validator.validate_candidates(
            PolicyCandidateSet(
                run_id="k",
                frame_id="f",
                scenario_id="s",
                model_id="c",
                carla_frame=0,
                simulation_time_s=1.0,
                wall_time_s=1.0,
                candidates=(cand,),
                schema_version=SCHEMA_VERSION,
            ),
            oracle_obs,
            now_s=1.0,
        )
        self.assertNotEqual(runtime.decision.decision_kind, DecisionKind.ACCEPT)

    def test_metrics_snapshot_fields(self) -> None:
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("c1", CandidateSource.CLASSIC),),
            schema_version=SCHEMA_VERSION,
        )
        self.kernel.tick(_obs(), cset, now_s=1.0)
        snap = self.kernel.metrics_snapshot()
        self.assertEqual(snap["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(snap["contracts_schema_hash"]), 64)
        self.assertGreaterEqual(snap["state_latency"]["count"], 1)
        self.assertGreaterEqual(snap["candidate_latency"]["count"], 1)

    def test_speed_limit_rule(self) -> None:
        cset = PolicyCandidateSet(
            run_id="k",
            frame_id="f",
            scenario_id="s",
            model_id="c",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("fast", CandidateSource.CLASSIC, pts=_pts(v=12.0)),),
            schema_version=SCHEMA_VERSION,
        )
        obs = _obs()
        obs = ObservableSnapshot(
            run_id=obs.run_id,
            frame_id=obs.frame_id,
            scenario_id=obs.scenario_id,
            simulation_time_s=obs.simulation_time_s,
            wall_time_s=obs.wall_time_s,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            observed_time_s=1.0,
            speed_limit_mps=8.0,
            corridor_centerline=obs.corridor_centerline,
            corridor_half_width_m=2.0,
        )
        result = self.kernel.validator.validate_candidates(cset, obs, now_s=1.0)
        self.assertTrue(any("rules" in r for r in result.decision.reject_reasons))


if __name__ == "__main__":
    unittest.main()
