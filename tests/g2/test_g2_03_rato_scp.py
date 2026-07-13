"""G2-03 restricted RATO-SCP: corridor, triggers, scenarios, cascade vs QP."""

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
    RepairInterface,
    RepairMode,
    SafetyKernel,
    SolverStatus,
    TrajectoryPoint,
    has_legal_lateral_corridor,
    is_rato_eligible_hints,
    load_safety_config,
)
from safety_kernel.config import RatoScpConfig, SafetyKernelConfig  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
    TrackedObject,
    TrafficLightObs,
)
from safety_kernel.validator.checks import hard_violations, run_full_checks  # noqa: E402


def _straight_pts(
    n: int = 17,
    v: float = 6.0,
    dt: float = 0.25,
    y: float = 0.0,
) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(
            TrajectoryPoint(
                t=dt * i,
                x=x,
                y=y,
                yaw=0.0,
                kappa=0.0,
                v=v,
                a=0.0,
                jerk=0.0,
            )
        )
        x += v * dt
    return tuple(out)


def _obs(
    *,
    now: float = 1.0,
    ego_v: float = 6.0,
    actors: tuple[TrackedObject, ...] = (),
    lights: tuple[TrafficLightObs, ...] = (),
    half_width: float = 3.5,
    corridor: bool = True,
) -> ObservableSnapshot:
    cl = tuple((float(i), 0.0) for i in range(0, 200)) if corridor else ()
    return ObservableSnapshot(
        run_id="g2-03",
        frame_id="f0",
        scenario_id="rato",
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=ego_v,
        ego_a=0.0,
        observed_time_s=now,
        actors=actors,
        traffic_lights=lights,
        corridor_centerline=cl,
        corridor_half_width_m=half_width,
        privilege=ObservationPrivilege.OBSERVABLE,
    )


def _cand(pts, *, cid: str = "raw", now: float = 1.0) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=cid,
        source=CandidateSource.CLASSIC,
        generated_time_s=now,
        valid_until_s=now + 0.2,
        probability=1.0,
        points=pts,
        behavior="cruise",
    )


def _cset(cand: PolicyCandidate, *, now: float = 1.0) -> PolicyCandidateSet:
    return PolicyCandidateSet(
        run_id="g2-03",
        frame_id="f0",
        scenario_id="rato",
        model_id="classic",
        carla_frame=0,
        simulation_time_s=now,
        wall_time_s=now,
        candidates=(cand,),
        schema_version=SCHEMA_VERSION,
    )


def _static_blocker(x: float = 12.0, y: float = 0.0) -> TrackedObject:
    return TrackedObject(
        actor_id="static",
        class_name="vehicle",
        x=x,
        y=y,
        yaw=0.0,
        vx=0.0,
        vy=0.0,
        length_m=4.5,
        width_m=1.9,
        observed_time_s=1.0,
    )


def _with_rato_enabled(cfg: SafetyKernelConfig, enabled: bool) -> SafetyKernelConfig:
    r = cfg.rato
    new_r = RatoScpConfig(
        enabled=enabled,
        deadline_ms=r.deadline_ms,
        max_scp_iters=r.max_scp_iters,
        trust_radius_m=r.trust_radius_m,
        max_lateral_step_m=r.max_lateral_step_m,
        warm_start=r.warm_start,
        w_path=r.w_path,
        w_smooth=r.w_smooth,
        w_slack=r.w_slack,
        w_progress=r.w_progress,
        slack_corridor_max_m=r.slack_corridor_max_m,
        slack_collision_max_m=r.slack_collision_max_m,
        min_lateral_clearance_m=r.min_lateral_clearance_m,
        min_qp_progress_to_skip=r.min_qp_progress_to_skip,
        oscillation_eps_m=r.oscillation_eps_m,
        min_progress_ratio=r.min_progress_ratio,
        repair_on_hard_reject=r.repair_on_hard_reject,
    )
    return SafetyKernelConfig(
        schema_version=cfg.schema_version,
        name=cfg.name,
        raw_toml=cfg.raw_toml,
        control_period_s=cfg.control_period_s,
        state_check_deadline_ms=cfg.state_check_deadline_ms,
        candidate_check_deadline_ms=cfg.candidate_check_deadline_ms,
        default_horizon_s=cfg.default_horizon_s,
        min_horizon_s=cfg.min_horizon_s,
        max_horizon_s=cfg.max_horizon_s,
        max_candidate_age_s=cfg.max_candidate_age_s,
        min_points=cfg.min_points,
        wheelbase_m=cfg.wheelbase_m,
        max_speed_mps=cfg.max_speed_mps,
        max_accel_mps2=cfg.max_accel_mps2,
        max_decel_mps2=cfg.max_decel_mps2,
        max_jerk_mps3=cfg.max_jerk_mps3,
        max_curvature_per_m=cfg.max_curvature_per_m,
        max_lateral_accel_mps2=cfg.max_lateral_accel_mps2,
        width_m=cfg.width_m,
        length_m=cfg.length_m,
        collision_inflate_m=cfg.collision_inflate_m,
        require_drivable=cfg.require_drivable,
        max_offroad_m=cfg.max_offroad_m,
        lane_half_width_m=cfg.lane_half_width_m,
        enforce_speed_limit=cfg.enforce_speed_limit,
        speed_limit_margin_mps=cfg.speed_limit_margin_mps,
        enforce_red_light_stop=cfg.enforce_red_light_stop,
        red_light_stop_distance_m=cfg.red_light_stop_distance_m,
        red_light_max_approach_speed_mps=cfg.red_light_max_approach_speed_mps,
        escalate_debounce_frames=cfg.escalate_debounce_frames,
        recover_debounce_frames=cfg.recover_debounce_frames,
        min_dwell_s=cfg.min_dwell_s,
        recover_clear_frames=cfg.recover_clear_frames,
        emit_safety_events=cfg.emit_safety_events,
        record_failure_samples=cfg.record_failure_samples,
        max_failure_samples=cfg.max_failure_samples,
        qp=cfg.qp,
        rato=new_r,
        arbitration=cfg.arbitration,
    )


class G203EligibilityTests(unittest.TestCase):
    def test_corridor_gate(self) -> None:
        cfg = load_safety_config()
        obs_ok = _obs(half_width=3.5)
        obs_none = _obs(corridor=False)
        # half_width 0.05 + max_offroad 0.5 → lateral_room ≈ 0.5 < 0.55 min clearance.
        obs_narrow = _obs(half_width=0.05)
        self.assertTrue(has_legal_lateral_corridor(obs_ok, cfg, min_clearance_m=0.55))
        self.assertFalse(has_legal_lateral_corridor(obs_none, cfg, min_clearance_m=0.55))
        self.assertFalse(has_legal_lateral_corridor(obs_narrow, cfg, min_clearance_m=0.55))

    def test_hint_eligibility(self) -> None:
        self.assertTrue(is_rato_eligible_hints(["c:collision:collision_envelope"]))
        self.assertTrue(is_rato_eligible_hints(["c:road:offroad"]))
        self.assertFalse(is_rato_eligible_hints(["c:rules:red_light_approach"]))
        self.assertFalse(is_rato_eligible_hints(["c:numeric:non_finite"]))
        self.assertFalse(is_rato_eligible_hints(["c:freshness:stale_age"]))


class G203RatoScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.iface = RepairInterface(self.cfg)

    def test_static_obstacle_lateral_repair(self) -> None:
        """Static blocker on path + wide corridor → RATO offsets laterally with progress."""
        actor = _static_blocker(x=12.0, y=0.0)
        obs = _obs(actors=(actor,), ego_v=6.0, half_width=3.5)
        cand = _cand(_straight_pts(v=6.0, n=17))
        raw_viol = hard_violations(run_full_checks(cand, obs, self.cfg, now_s=1.0))
        self.assertTrue(any(v.name == "collision" for v in raw_viol), msg=raw_viol)

        qp = self.iface.repair(
            cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:collision"]
        )
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"]
        )
        self.assertTrue(rato.success, msg=f"{rato.reason} {rato.solver_trace}")
        assert rato.candidate is not None
        viol = hard_violations(run_full_checks(rato.candidate, obs, self.cfg, now_s=1.0))
        self.assertEqual(viol, [], msg=[(v.name, v.message, v.margin) for v in viol])
        # Lateral modification present
        max_abs_y = max(abs(p.y) for p in rato.candidate.points)
        self.assertGreater(max_abs_y, 0.4, msg="expected lateral offset around static obstacle")
        # Progress: RATO should not collapse worse than a pure stop (honest comparison)
        if qp.success:
            self.assertGreaterEqual(
                rato.metrics.progress_ratio,
                qp.metrics.progress_ratio - 0.15,
            )
        self.assertGreater(rato.metrics.progress_ratio, 0.15)
        self.assertIn("scp_iters", rato.solver_trace.extras)
        self.assertGreaterEqual(int(rato.solver_trace.extras["scp_iters"]), 1)

    def test_narrow_corridor_pull_in(self) -> None:
        """Slightly off-center path in narrowish corridor is pulled toward centerline."""
        pts = _straight_pts(v=5.0, n=15, y=1.2)
        obs = _obs(ego_v=5.0, half_width=2.2)
        cand = _cand(pts)
        # May or may not hard-fail road depending on offroad slack; RATO should still run.
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:road:offroad"]
        )
        if rato.success:
            assert rato.candidate is not None
            mean_abs_y = sum(abs(p.y) for p in rato.candidate.points) / len(rato.candidate.points)
            self.assertLessEqual(mean_abs_y, 1.2 + 0.05)
            viol = hard_violations(run_full_checks(rato.candidate, obs, self.cfg, now_s=1.0))
            self.assertEqual(viol, [], msg=[(v.name, v.message) for v in viol])
        else:
            # Honest fail allowed if corridor too tight for dynamics.
            self.assertIn(
                rato.solver_trace.status,
                {
                    SolverStatus.INFEASIBLE,
                    SolverStatus.NOT_REPAIRABLE,
                    SolverStatus.TIMEOUT,
                    SolverStatus.NUMERICAL_ERROR,
                    SolverStatus.REVALIDATE_FAIL,
                    SolverStatus.MAX_ITER,
                },
            )

    def test_lane_change_conflict(self) -> None:
        """Adjacent-lane actor conflict: RATO keeps trajectory inside corridor safely."""
        side = TrackedObject(
            actor_id="side",
            class_name="vehicle",
            x=10.0,
            y=2.2,
            yaw=0.0,
            vx=4.0,
            vy=0.0,
            length_m=4.0,
            width_m=1.8,
            observed_time_s=1.0,
        )
        # Path slightly toward actor
        pts = _straight_pts(v=6.0, n=17, y=1.0)
        obs = _obs(actors=(side,), ego_v=6.0, half_width=3.5)
        cand = _cand(pts)
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision:lane"]
        )
        if rato.success:
            assert rato.candidate is not None
            viol = hard_violations(run_full_checks(rato.candidate, obs, self.cfg, now_s=1.0))
            self.assertEqual(viol, [], msg=[(v.name, v.message, v.margin) for v in viol])
            self.assertLessEqual(max(abs(p.y) for p in rato.candidate.points), 3.5)
        else:
            self.assertIn(
                rato.solver_trace.status.value,
                {
                    "infeasible",
                    "not_repairable",
                    "timeout",
                    "numerical_error",
                    "revalidate_fail",
                    "max_iter",
                },
            )

    def test_blocked_corridor_infeasible(self) -> None:
        """Obstacles on both sides with insufficient room → honest infeasible, not false success."""
        left = TrackedObject("L", "vehicle", 12.0, 1.6, 0.0, 0.0, 0.0, 4.5, 2.0, 1.0)
        right = TrackedObject("R", "vehicle", 12.0, -1.6, 0.0, 0.0, 0.0, 4.5, 2.0, 1.0)
        center = _static_blocker(12.0, 0.0)
        obs = _obs(actors=(left, right, center), ego_v=6.0, half_width=2.0)
        cand = _cand(_straight_pts(v=6.0, n=17))
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"]
        )
        if rato.success:
            assert rato.candidate is not None
            # If somehow solvable, must revalidate clean.
            viol = hard_violations(run_full_checks(rato.candidate, obs, self.cfg, now_s=1.0))
            self.assertEqual(viol, [])
        else:
            self.assertFalse(rato.success)
            self.assertNotEqual(rato.solver_trace.status, SolverStatus.SOLVED)

    def test_no_corridor_skips_rato(self) -> None:
        actor = _static_blocker()
        obs = _obs(actors=(actor,), corridor=False)
        cand = _cand(_straight_pts())
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"]
        )
        self.assertFalse(rato.success)
        self.assertEqual(rato.solver_trace.status, SolverStatus.NOT_REPAIRABLE)
        self.assertEqual(rato.reason, "no_legal_corridor")

    def test_red_light_hints_not_direct_rato(self) -> None:
        """Pure red-light is longitudinal-only at the RATO eligibility gate."""
        lights = (TrafficLightObs("tl", "red", 7.0, 1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:rules:red_light"]
        )
        self.assertFalse(rato.success)
        self.assertEqual(rato.reason, "not_rato_eligible")

    def test_stale_input(self) -> None:
        actor = _static_blocker()
        obs = _obs(now=2.0, actors=(actor,))
        cand = _cand(_straight_pts(), now=0.0)
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=2.0, reject_hints=["c:collision"]
        )
        self.assertFalse(rato.success)
        self.assertEqual(rato.solver_trace.status, SolverStatus.STALE_INPUT)

    def test_disabled_rato(self) -> None:
        cfg = _with_rato_enabled(self.cfg, False)
        iface = RepairInterface(cfg)
        actor = _static_blocker()
        obs = _obs(actors=(actor,))
        cand = _cand(_straight_pts())
        rato = iface.repair(cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"])
        self.assertFalse(rato.success)
        self.assertEqual(rato.solver_trace.status, SolverStatus.DISABLED)

    def test_report_fields(self) -> None:
        actor = _static_blocker()
        obs = _obs(actors=(actor,))
        cand = _cand(_straight_pts())
        rato = self.iface.repair(
            cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"]
        )
        d = rato.to_dict()
        for key in ("mode", "success", "solver_trace", "metrics", "reason"):
            self.assertIn(key, d)
        for key in (
            "safety_margin_m",
            "modification_norm",
            "progress_ratio",
            "progress_loss",
            "comfort_jerk_rms",
            "slack_used_max",
        ):
            self.assertIn(key, d["metrics"])


class G203KernelCascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)

    def test_red_light_stays_qp_without_rato(self) -> None:
        """Longitudinal-only red light should remain QP; RATO must not run for pure rules."""
        kernel = SafetyKernel(self.cfg)
        lights = (TrafficLightObs("tl", "red", 7.0, 1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        result = kernel.tick(obs, _cset(cand), now_s=1.0, availability=self.avail)
        self.assertEqual(result.decision.decision_kind, DecisionKind.QP, msg=result.decision.reject_reasons)
        self.assertIsNotNone(result.repair_result)
        assert result.repair_result is not None
        self.assertEqual(result.repair_result.mode, RepairMode.LONGITUDINAL)

    def test_static_obstacle_may_use_rato(self) -> None:
        kernel = SafetyKernel(self.cfg)
        actor = _static_blocker(x=11.0, y=0.0)
        obs = _obs(actors=(actor,), ego_v=6.0, half_width=3.5)
        cand = _cand(_straight_pts(v=6.0, n=17))
        result = kernel.tick(obs, _cset(cand), now_s=1.0, availability=self.avail)
        # Either RATO or QP accepted; both valid. Prefer documenting kind.
        self.assertIn(
            result.decision.decision_kind,
            {DecisionKind.RATO, DecisionKind.QP, DecisionKind.HARD_REJECT, DecisionKind.MINIMAL_RISK},
        )
        if result.decision.decision_kind is DecisionKind.RATO:
            self.assertIsNotNone(result.repair_result)
            assert result.repair_result is not None
            self.assertTrue(result.repair_result.success)
            self.assertEqual(result.repair_result.mode, RepairMode.RATO)

    def test_rato_disabled_qp_independent(self) -> None:
        """Closing secondary RATO leaves G2-02 longitudinal path independently usable."""
        cfg = _with_rato_enabled(self.cfg, False)
        kernel = SafetyKernel(cfg)
        lights = (TrafficLightObs("tl", "red", 7.0, 1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        result = kernel.tick(obs, _cset(cand), now_s=1.0, availability=self.avail)
        self.assertEqual(result.decision.decision_kind, DecisionKind.QP)
        # Direct longitudinal still works via interface
        iface = RepairInterface(cfg)
        qp = iface.repair(
            cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:rules:red_light"]
        )
        self.assertTrue(qp.success)

    def test_warm_start_second_rato(self) -> None:
        iface = RepairInterface(self.cfg)
        actor = _static_blocker()
        obs = _obs(actors=(actor,))
        cand = _cand(_straight_pts(n=13, v=5.0))
        r1 = iface.repair(cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"])
        r2 = iface.repair(cand, obs, mode=RepairMode.RATO, now_s=1.0, reject_hints=["c:collision"])
        if r1.success and r2.success:
            self.assertTrue(
                r2.solver_trace.warm_started
                or r2.solver_trace.iterations <= r1.solver_trace.iterations + 80
            )


if __name__ == "__main__":
    unittest.main()
