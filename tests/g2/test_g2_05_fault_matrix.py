"""G2-05 offline fault matrix + baseline comparison (no live CARLA)."""

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
    RepairInterface,
    RepairMode,
    SafetyKernel,
    TrajectoryPoint,
    load_safety_config,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
    TrackedObject,
)
from safety_kernel.faults import (  # noqa: E402
    DEFAULT_MATRIX,
    FaultId,
    apply_fault_to_obs,
    apply_fault_to_set,
    expected_action_holds,
)


def _pts(v: float = 6.0, n: int = 16) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(TrajectoryPoint(t=0.25 * i, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
        x += v * 0.25
    return tuple(out)


def _base_obs(now: float = 1.0) -> ObservableSnapshot:
    lead = TrackedObject("lead", "vehicle", 18.0, 0.0, 0.0, 3.0, 0.0, 4.5, 1.8, now)
    return ObservableSnapshot(
        run_id="g2-05",
        frame_id="f0",
        scenario_id="fault",
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=6.0,
        observed_time_s=now,
        actors=(lead,),
        corridor_centerline=tuple((float(i), 0.0) for i in range(0, 150)),
        corridor_half_width_m=2.5,
        privilege=ObservationPrivilege.OBSERVABLE,
        schema_version=SCHEMA_VERSION,
        coordinate_frame="map",
    )


def _cset(pts=None, now: float = 1.0) -> PolicyCandidateSet:
    cand = PolicyCandidate(
        candidate_id="raw",
        source=CandidateSource.CLASSIC,
        generated_time_s=now,
        valid_until_s=now + 0.2,
        probability=1.0,
        points=pts or _pts(),
    )
    return PolicyCandidateSet(
        run_id="g2-05",
        frame_id="f0",
        scenario_id="fault",
        model_id="classic",
        carla_frame=0,
        simulation_time_s=now,
        wall_time_s=now,
        candidates=(cand,),
        schema_version=SCHEMA_VERSION,
        coordinate_frame="map",
    )


class G205FaultMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.iface = RepairInterface(self.cfg)
        self.avail = ComponentAvailability(classic=True, vla=True, world=False, safety=True)

    def _fresh_kernel(self) -> SafetyKernel:
        return SafetyKernel(self.cfg)

    def test_matrix_specs_complete(self) -> None:
        required = {
            FaultId.STALE_OBS,
            FaultId.PACKET_DROP_ACTOR,
            FaultId.OUT_OF_ORDER_TIME,
            FaultId.LOCALIZATION_BIAS,
            FaultId.MISSED_ACTOR,
            FaultId.ACTOR_OFFSET,
            FaultId.SOLVER_STALE_CANDIDATE,
            FaultId.NUMERIC_NAN,
            FaultId.VISION_SOFT_DEGRADE,
            FaultId.LOW_ATTACHMENT,
            FaultId.ACTUATOR_SATURATION,
            FaultId.SOLVER_TIMEOUT,
            FaultId.MODEL_TIMEOUT,
        }
        ids = {f.fault_id for f in DEFAULT_MATRIX}
        self.assertTrue(required.issubset(ids), msg=f"missing={required - ids}")
        for fault in DEFAULT_MATRIX:
            d = fault.to_dict()
            for key in ("fault_id", "severity", "start_s", "duration_s", "seed", "expected_action", "recovery_s"):
                self.assertIn(key, d)

    def test_every_fault_expected_action(self) -> None:
        """P1-2: each matrix fault must satisfy its expected_action, not only non-empty kind."""
        for fault in DEFAULT_MATRIX:
            kernel = self._fresh_kernel()
            obs = apply_fault_to_obs(_base_obs(), fault, now_s=1.0)
            cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
            res = kernel.tick(obs, cset, now_s=1.0, availability=self.avail)
            repair_ok = None if res.repair_result is None else res.repair_result.success
            ok = expected_action_holds(
                fault,
                res.decision.decision_kind.value,
                repair_success=repair_ok,
            )
            self.assertTrue(
                ok,
                msg=(
                    f"{fault.fault_id.value}: expected_action={fault.expected_action} "
                    f"got kind={res.decision.decision_kind.value} repair={repair_ok} "
                    f"reasons={res.decision.reject_reasons[:6]}"
                ),
            )
            self.assertFalse(res.decision.learning_modules_required)

    def test_stale_obs_full_tick_state_lock(self) -> None:
        """Stale obs must state-lock: no ACCEPT/QP/RATO on untrusted observation."""
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.STALE_OBS)
        obs = apply_fault_to_obs(_base_obs(), fault, now_s=1.0)
        cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
        res = self._fresh_kernel().tick(obs, cset, now_s=1.0, availability=self.avail)
        self.assertIn(
            res.decision.decision_kind,
            {DecisionKind.MINIMAL_RISK, DecisionKind.EMERGENCY},
        )
        self.assertNotIn(
            res.decision.decision_kind,
            {DecisionKind.ACCEPT, DecisionKind.QP, DecisionKind.RATO},
        )
        self.assertIsNone(res.decision.executed_trajectory_id)
        self.assertFalse(res.decision.learning_modules_required)
        self.assertTrue(res.repair_result is None or not res.repair_result.success)
        self.assertIn("state_locked_no_candidate_override", res.decision.reject_reasons)

    def test_numeric_nan_hard_reject_no_learning(self) -> None:
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.NUMERIC_NAN)
        obs = _base_obs()
        cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
        res = self._fresh_kernel().tick(obs, cset, now_s=1.0, availability=self.avail)
        self.assertIn(
            res.decision.decision_kind,
            {DecisionKind.HARD_REJECT, DecisionKind.MINIMAL_RISK, DecisionKind.CLASSIC_FALLBACK},
        )
        self.assertFalse(res.decision.learning_modules_required)

    def test_out_of_order_hard_reject(self) -> None:
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.OUT_OF_ORDER_TIME)
        obs = _base_obs()
        cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
        res = self._fresh_kernel().tick(obs, cset, now_s=1.0, availability=self.avail)
        self.assertNotEqual(res.decision.decision_kind, DecisionKind.ACCEPT)

    def test_stale_candidate_no_false_qp_success(self) -> None:
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.SOLVER_STALE_CANDIDATE)
        obs = _base_obs()
        cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
        cand = cset.candidates[0]
        qp = self.iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0)
        self.assertFalse(qp.success)

    def test_solver_timeout_no_execute(self) -> None:
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.SOLVER_TIMEOUT)
        obs = apply_fault_to_obs(_base_obs(), fault, now_s=1.0)
        cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
        cand = cset.candidates[0]
        qp = self.iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0)
        self.assertFalse(qp.success)
        self.assertEqual(qp.solver_trace.status.value, "timeout")
        res = self._fresh_kernel().tick(obs, cset, now_s=1.0, availability=self.avail)
        self.assertNotIn(res.decision.decision_kind, {DecisionKind.QP, DecisionKind.RATO})

    def test_actuator_saturation_not_accept(self) -> None:
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.ACTUATOR_SATURATION)
        res = self._fresh_kernel().tick(
            apply_fault_to_obs(_base_obs(), fault, now_s=1.0),
            apply_fault_to_set(_cset(), fault, now_s=1.0),
            now_s=1.0,
            availability=self.avail,
        )
        self.assertNotEqual(res.decision.decision_kind, DecisionKind.ACCEPT)

    def test_model_timeout_prefers_classic(self) -> None:
        fault = next(f for f in DEFAULT_MATRIX if f.fault_id is FaultId.MODEL_TIMEOUT)
        res = self._fresh_kernel().tick(
            apply_fault_to_obs(_base_obs(), fault, now_s=1.0),
            apply_fault_to_set(_cset(), fault, now_s=1.0),
            now_s=1.0,
            availability=self.avail,
        )
        # Timed-out VLA must not be executed; classic sibling or fallback.
        self.assertNotEqual(res.decision.final_candidate_id, "raw")
        if res.decision.final_candidate_id is not None:
            self.assertEqual(res.decision.final_candidate_id, "classic_fallback")

    def test_learning_off_classic_still_runs_all_faults(self) -> None:
        """Learning modules closed: Classic+Safety path still produces decisions."""
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        for fault in DEFAULT_MATRIX:
            kernel = self._fresh_kernel()
            obs = apply_fault_to_obs(_base_obs(), fault, now_s=1.0)
            cset = apply_fault_to_set(_cset(), fault, now_s=1.0)
            res = kernel.tick(obs, cset, now_s=1.0, availability=avail)
            self.assertIsNotNone(res.decision.decision_kind)
            self.assertFalse(res.decision.learning_modules_required)

    def test_baseline_modes_compare_follow(self) -> None:
        """Raw / Rule / HardReject / Longitudinal / RATO interface still unified under fault-free follow."""
        obs = _base_obs()
        cand = _cset().candidates[0]
        lead = TrackedObject("lead", "vehicle", 10.0, 0.0, 0.0, 2.0, 0.0, 4.5, 1.8, 1.0)
        obs = ObservableSnapshot(
            run_id=obs.run_id,
            frame_id=obs.frame_id,
            scenario_id=obs.scenario_id,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=8.0,
            observed_time_s=1.0,
            actors=(lead,),
            corridor_centerline=obs.corridor_centerline,
            corridor_half_width_m=2.5,
            privilege=ObservationPrivilege.OBSERVABLE,
            schema_version=SCHEMA_VERSION,
            coordinate_frame="map",
        )
        cmp = self.iface.compare_all(cand, obs, now_s=1.0, reject_hints=["c:collision"])
        self.assertIn("raw", cmp)
        self.assertIn("longitudinal", cmp)
        self.assertIn("rato", cmp)
        self.assertFalse(cmp["hard_reject"].success)


if __name__ == "__main__":
    unittest.main()
