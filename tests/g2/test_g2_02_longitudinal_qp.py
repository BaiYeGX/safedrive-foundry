"""G2-02 longitudinal QP repair: scenarios, baselines, contracts."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

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
    is_longitudinally_repairable,
    load_safety_config,
    osqp_available,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
    TrackedObject,
    TrafficLightObs,
)
from safety_kernel.validator.checks import run_full_checks, hard_violations  # noqa: E402
from safety_kernel.repair.longitudinal_qp import _arc_lengths, _lead_s_profile  # noqa: E402


def _straight_pts(
    n: int = 17,
    v: float = 8.0,
    dt: float = 0.25,
    a: float = 0.0,
) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(
            TrajectoryPoint(
                t=dt * i,
                x=x,
                y=0.0,
                yaw=0.0,
                kappa=0.0,
                v=v,
                a=a,
                jerk=0.0,
            )
        )
        x += v * dt
    return tuple(out)


def _obs(
    *,
    now: float = 1.0,
    ego_v: float = 8.0,
    actors: tuple[TrackedObject, ...] = (),
    lights: tuple[TrafficLightObs, ...] = (),
    speed_limit: float | None = None,
) -> ObservableSnapshot:
    return ObservableSnapshot(
        run_id="g2-02",
        frame_id="f0",
        scenario_id="longitudinal",
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=ego_v,
        ego_a=0.0,
        observed_time_s=now,
        speed_limit_mps=speed_limit,
        actors=actors,
        traffic_lights=lights,
        corridor_centerline=tuple((float(i), 0.0) for i in range(0, 200)),
        corridor_half_width_m=2.5,
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
        run_id="g2-02",
        frame_id="f0",
        scenario_id="longitudinal",
        model_id="classic",
        carla_frame=0,
        simulation_time_s=now,
        wall_time_s=now,
        candidates=(cand,),
        schema_version=SCHEMA_VERSION,
    )


class G202RepairabilityTests(unittest.TestCase):
    def test_repairable_tokens(self) -> None:
        self.assertTrue(is_longitudinally_repairable(["c1:rules:red_light_approach"]))
        self.assertTrue(is_longitudinally_repairable(["c1:collision:collision_envelope"]))
        self.assertTrue(is_longitudinally_repairable(["c1:dynamics:speed"]))
        self.assertFalse(is_longitudinally_repairable(["c1:numeric:non_finite_at_index_0"]))
        self.assertFalse(is_longitudinally_repairable(["c1:road:offroad"]))
        self.assertFalse(is_longitudinally_repairable(["c1:freshness:stale_age:1.0"]))

    def test_lead_envelope_includes_both_vehicle_half_lengths(self) -> None:
        cfg = load_safety_config()
        points = _straight_pts(n=5, v=2.0)
        actor = TrackedObject(
            actor_id="lead",
            class_name="vehicle",
            x=10.0,
            y=0.0,
            yaw=0.0,
            vx=0.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=1.0,
        )
        obs = _obs(ego_v=2.0, actors=(actor,))
        times = np.array([point.t for point in points], dtype=float)
        cap = _lead_s_profile(
            obs,
            _arc_lengths(points),
            points,
            times,
            cfg,
            min_gap_m=cfg.qp.min_gap_m,
            time_headway_s=cfg.qp.time_headway_s,
        )
        expected = 10.0 - (
            cfg.qp.min_gap_m
            + 0.5 * cfg.length_m
            + 0.5 * actor.length_m
            + cfg.collision_inflate_m
        )
        self.assertAlmostEqual(float(cap[0]), expected)

    def test_lead_prediction_is_aligned_to_first_candidate_timestamp(self) -> None:
        cfg = load_safety_config()
        points = tuple(
            TrajectoryPoint(
                t=0.25 * (index + 1),
                x=0.5 * index,
                y=0.0,
                yaw=0.0,
                kappa=0.0,
                v=2.0,
                a=0.0,
                jerk=0.0,
            )
            for index in range(5)
        )
        actor = TrackedObject(
            actor_id="moving-lead",
            class_name="vehicle",
            x=10.0,
            y=0.0,
            yaw=0.0,
            vx=4.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=1.0,
        )
        obs = _obs(now=1.0, ego_v=2.0, actors=(actor,))
        times = np.array([point.t for point in points], dtype=float)
        cap = _lead_s_profile(
            obs,
            _arc_lengths(points),
            points,
            times,
            cfg,
            min_gap_m=cfg.qp.min_gap_m,
            time_headway_s=cfg.qp.time_headway_s,
        )
        clearance = (
            cfg.qp.min_gap_m
            + actor.vx * cfg.qp.time_headway_s
            + 0.5 * cfg.length_m
            + 0.5 * actor.length_m
            + cfg.collision_inflate_m
        )
        self.assertAlmostEqual(float(cap[0]), 10.0 + 4.0 * 0.25 - clearance)


class G202LongitudinalQPScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.iface = RepairInterface(self.cfg)
        self.kernel = SafetyKernel(self.cfg)
        self.avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)

    def test_red_light_normal_repair(self) -> None:
        """Cruise into nearby red light → QP slows; better than raw."""
        # Validator only enforces red when distance <= red_light_stop_distance_m (8m).
        lights = (TrafficLightObs(light_id="tl1", state="red", distance_m=7.0, observed_time_s=1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        # Raw fails rules (near red with non-trivial approach speed).
        raw_margins = run_full_checks(cand, obs, self.cfg, now_s=1.0)
        self.assertTrue(any(m.name == "rules" and m.margin < 0 for m in raw_margins))

        result = self.iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["raw:rules:red_light"])
        self.assertTrue(result.success, msg=f"{result.reason} {result.solver_trace}")
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        # Approach speeds in first 2s must satisfy red-light rule (+tiny slack).
        early_v = [p.v for p in result.candidate.points if p.t <= 2.0]
        self.assertTrue(
            max(early_v) <= self.cfg.red_light_max_approach_speed_mps + self.cfg.qp.slack_speed_max_mps + 1e-3
        )
        reval = run_full_checks(result.candidate, obs, self.cfg, now_s=1.0)
        viol = hard_violations(reval)
        self.assertEqual(viol, [], msg=[(v.name, v.message, v.margin) for v in viol])
        self.assertGreaterEqual(result.metrics.safety_margin_m, -self.cfg.qp.slack_stop_max_m - 0.5)
        self.assertFalse(result.metrics.unjustified_stop)
        self.assertIn(result.solver_trace.status, {SolverStatus.SOLVED, SolverStatus.SOLVED_INACCURATE})

    def test_red_light_boundary_and_unsolvable(self) -> None:
        """Boundary: stop just within distance; unsolvable: stop behind ego."""
        # Boundary solvable (within 8m rule window)
        lights_ok = (TrafficLightObs(light_id="tl", state="red", distance_m=8.0, observed_time_s=1.0),)
        obs_ok = _obs(lights=lights_ok, ego_v=4.0)
        cand = _cand(_straight_pts(v=4.0, n=13))
        r_ok = self.iface.repair(cand, obs_ok, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:rules:red_light"])
        self.assertTrue(r_ok.success, msg=f"{r_ok.reason} {r_ok.solver_trace}")

        # Unsolvable: stop line already behind / zero with high residual demand — actor blocks immediately.
        actor = TrackedObject(
            actor_id="blocker",
            class_name="vehicle",
            x=1.0,
            y=0.0,
            yaw=0.0,
            vx=0.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=1.0,
        )
        lights_tight = (TrafficLightObs(light_id="tl", state="red", distance_m=0.2, observed_time_s=1.0),)
        obs_bad = _obs(lights=lights_tight, actors=(actor,), ego_v=12.0)
        cand_fast = _cand(_straight_pts(v=12.0, n=17))
        r_bad = self.iface.repair(
            cand_fast,
            obs_bad,
            mode=RepairMode.LONGITUDINAL,
            now_s=1.0,
            reject_hints=["c:collision:collision_envelope", "c:rules:red_light"],
        )
        # Either fails solver/revalidate path or succeeds with near-stop; must not claim unjustified free progress.
        if r_bad.success:
            self.assertIsNotNone(r_bad.candidate)
            assert r_bad.candidate is not None
            self.assertLess(max(p.v for p in r_bad.candidate.points[:4]), 3.0)
        else:
            self.assertIn(
                r_bad.solver_trace.status,
                {
                    SolverStatus.INFEASIBLE,
                    SolverStatus.MAX_ITER,
                    SolverStatus.TIMEOUT,
                    SolverStatus.NUMERICAL_ERROR,
                    SolverStatus.REVALIDATE_FAIL,
                    SolverStatus.NOT_REPAIRABLE,
                },
            )

    def test_following_lead_vehicle(self) -> None:
        """Lead vehicle ahead slower → QP reduces progress / speed vs raw crash."""
        lead = TrackedObject(
            actor_id="lead",
            class_name="vehicle",
            x=18.0,
            y=0.0,
            yaw=0.0,
            vx=3.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.8,
            observed_time_s=1.0,
        )
        obs = _obs(actors=(lead,), ego_v=8.0)
        cand = _cand(_straight_pts(v=8.0, n=17))
        raw_v = hard_violations(run_full_checks(cand, obs, self.cfg, now_s=1.0))
        self.assertTrue(any(v.name == "collision" for v in raw_v), msg=raw_v)

        result = self.iface.repair(
            cand,
            obs,
            mode=RepairMode.LONGITUDINAL,
            now_s=1.0,
            reject_hints=["raw:collision:collision_envelope"],
        )
        self.assertTrue(result.success, msg=f"{result.reason} {result.solver_trace}")
        assert result.candidate is not None
        viol = hard_violations(run_full_checks(result.candidate, obs, self.cfg, now_s=1.0))
        self.assertEqual(viol, [], msg=[(v.name, v.message) for v in viol])
        self.assertLess(result.candidate.points[-1].v, 8.0 + 0.1)
        self.assertGreater(result.metrics.progress_ratio, 0.1)

    def test_cut_in_actor(self) -> None:
        """Cut-in: lateral actor with forward component creates envelope risk."""
        cut = TrackedObject(
            actor_id="cut",
            class_name="vehicle",
            x=8.0,
            y=1.2,
            yaw=0.0,
            vx=3.0,
            vy=-0.4,
            length_m=4.0,
            width_m=1.8,
            observed_time_s=1.0,
        )
        obs = _obs(actors=(cut,), ego_v=9.0)
        cand = _cand(_straight_pts(v=9.0, n=17))
        result = self.iface.repair(
            cand,
            obs,
            mode=RepairMode.LONGITUDINAL,
            now_s=1.0,
            reject_hints=["raw:collision:collision_envelope"],
        )
        # Accept success or honest infeasible — both valid under tight geometry.
        if result.success:
            assert result.candidate is not None
            viol = hard_violations(run_full_checks(result.candidate, obs, self.cfg, now_s=1.0))
            self.assertEqual(viol, [], msg=[(v.name, v.message, v.margin) for v in viol])
            self.assertLessEqual(result.metrics.modification_norm, 20.0)
        else:
            self.assertIn(result.solver_trace.status.value, {
                "infeasible", "max_iter", "timeout", "numerical_error", "revalidate_fail", "not_repairable",
            })

    def test_hard_brake_dynamics(self) -> None:
        """Excessive accel/jerk profile is smoothed within vehicle limits."""
        pts = []
        x = 0.0
        v = 8.0
        for i in range(17):
            a = 5.0 if i < 4 else -7.0  # violates accel/decel
            pts.append(TrajectoryPoint(t=0.25 * i, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=a, jerk=20.0))
            x += max(0.0, v) * 0.25
            v = max(0.0, v + a * 0.25)
        cand = _cand(tuple(pts))
        obs = _obs(ego_v=8.0)
        raw_viol = hard_violations(run_full_checks(cand, obs, self.cfg, now_s=1.0))
        self.assertTrue(any(v.name == "dynamics" for v in raw_viol), msg=raw_viol)
        result = self.iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:dynamics:accel"])
        self.assertTrue(result.success, msg=result.reason)
        assert result.candidate is not None
        for p in result.candidate.points:
            self.assertLessEqual(p.a, self.cfg.max_accel_mps2 + 1e-3)
            self.assertGreaterEqual(p.a, -self.cfg.max_decel_mps2 - 1e-3)
            self.assertLessEqual(abs(p.jerk), self.cfg.max_jerk_mps3 + 0.5)

    def test_lateral_acceleration_is_repaired_by_speed_not_hidden(self) -> None:
        points = tuple(
            TrajectoryPoint(
                t=point.t,
                x=point.x,
                y=point.y,
                yaw=point.yaw,
                kappa=0.20,
                v=8.0,
                a=0.0,
                jerk=0.0,
            )
            for point in _straight_pts(n=17, v=8.0)
        )
        cand = _cand(points)
        result = self.iface.repair(
            cand,
            _obs(ego_v=8.0),
            mode=RepairMode.LONGITUDINAL,
            now_s=1.0,
            reject_hints=["raw:dynamics:lat_accel"],
        )
        self.assertTrue(result.success, msg=f"{result.reason} {result.solver_trace}")
        assert result.candidate is not None
        self.assertLessEqual(
            max(abs(point.kappa) * point.v * point.v for point in result.candidate.points),
            self.cfg.max_lateral_accel_mps2 + 1e-3,
        )

    def test_stale_input_contract(self) -> None:
        cand = _cand(_straight_pts(), now=0.0)
        obs = _obs(now=2.0)
        result = self.iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=2.0)
        self.assertFalse(result.success)
        self.assertEqual(result.solver_trace.status, SolverStatus.STALE_INPUT)


class G202BaselineComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.iface = RepairInterface(self.cfg)

    def test_unified_modes_exist(self) -> None:
        modes = self.iface.available_modes()
        self.assertEqual(
            set(modes),
            {
                RepairMode.RAW,
                RepairMode.RULE,
                RepairMode.HARD_REJECT,
                RepairMode.LONGITUDINAL,
                RepairMode.RATO,
            },
        )

    def test_qp_vs_rule_hardreject_red_light(self) -> None:
        lights = (TrafficLightObs(light_id="tl1", state="red", distance_m=7.0, observed_time_s=1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        cmp = self.iface.compare_all(cand, obs, now_s=1.0, reject_hints=["c:rules:red_light"])
        self.assertFalse(cmp["hard_reject"].success)
        self.assertTrue(cmp["raw"].success)  # raw returns path but is unsafe
        raw_viol = hard_violations(run_full_checks(cmp["raw"].candidate, obs, self.cfg, now_s=1.0))  # type: ignore[arg-type]
        self.assertTrue(raw_viol)
        # Longitudinal should repair when feasible.
        self.assertTrue(cmp["longitudinal"].success, msg=f"{cmp['longitudinal'].reason} {cmp['longitudinal'].solver_trace}")
        long_viol = hard_violations(
            run_full_checks(cmp["longitudinal"].candidate, obs, self.cfg, now_s=1.0)  # type: ignore[arg-type]
        )
        self.assertEqual(long_viol, [])
        # Comfort: QP jerk rms should be finite and typically <= rule (honest if not).
        qj = cmp["longitudinal"].metrics.comfort_jerk_rms
        rj = cmp["rule"].metrics.comfort_jerk_rms
        self.assertTrue(math.isfinite(qj) and math.isfinite(rj))
        # Progress: QP should not collapse more than rule by huge margin without safety need.
        self.assertGreaterEqual(
            cmp["longitudinal"].metrics.progress_ratio,
            min(0.05, cmp["rule"].metrics.progress_ratio),
        )
        # Report fields present
        for key in ("safety_margin_m", "modification_norm", "progress_loss", "unjustified_stop"):
            self.assertIn(key, cmp["longitudinal"].metrics.to_dict())


class G202KernelIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.kernel = SafetyKernel(self.cfg)

    def test_kernel_qp_decision_on_red_light(self) -> None:
        lights = (TrafficLightObs(light_id="tl1", state="red", distance_m=7.0, observed_time_s=1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        result = self.kernel.tick(
            obs,
            _cset(cand),
            now_s=1.0,
            availability=ComponentAvailability(classic=True, vla=False, world=False, safety=True),
        )
        self.assertEqual(result.decision.decision_kind, DecisionKind.QP, msg=result.decision.reject_reasons)
        self.assertIsNotNone(result.decision.post_repair_trajectory_id)
        self.assertIsNotNone(result.repair_result)
        assert result.repair_result is not None
        self.assertTrue(result.repair_result.success)
        self.assertEqual(result.mode.name, "NORMAL")

    def test_kernel_non_repairable_stays_reject(self) -> None:
        # Numeric/schema failure cannot be repaired by QP or RATO.
        pts = _straight_pts(v=5.0)
        bad = list(pts)
        # Inject NaN so both longitudinal and RATO skip as not_repairable.
        p0 = bad[0]
        bad[0] = TrajectoryPoint(
            t=p0.t, x=float("nan"), y=p0.y, yaw=p0.yaw, kappa=p0.kappa, v=p0.v, a=p0.a, jerk=p0.jerk
        )
        obs = _obs(ego_v=5.0)
        cand = _cand(tuple(bad))
        result = self.kernel.tick(obs, _cset(cand), now_s=1.0)
        self.assertIn(
            result.decision.decision_kind,
            {DecisionKind.HARD_REJECT, DecisionKind.MINIMAL_RISK},
        )

    def test_warm_start_second_solve(self) -> None:
        lights = (TrafficLightObs(light_id="tl1", state="red", distance_m=7.0, observed_time_s=1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0, n=13))
        iface = RepairInterface(self.cfg)
        r1 = iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:rules:red_light"])
        r2 = iface.repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:rules:red_light"])
        self.assertTrue(r1.success and r2.success, msg=(r1.reason, r2.reason))
        self.assertTrue(r2.solver_trace.warm_started or r2.solver_trace.iterations <= r1.solver_trace.iterations + 50)

    def test_osqp_backend_label_honest(self) -> None:
        lights = (TrafficLightObs(light_id="tl1", state="red", distance_m=7.0, observed_time_s=1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        r = RepairInterface(self.cfg).repair(cand, obs, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:rules"])
        self.assertTrue(r.success, msg=r.reason)
        # Prefer real OSQP when tools/wsl_site_packages (or env) provides it.
        if osqp_available():
            self.assertEqual(r.solver_trace.backend, "osqp")
        else:
            self.assertIn(r.solver_trace.backend, {"scipy_slsqp", "numpy_admm"})

    def test_kernel_reset_clears_warm_start(self) -> None:
        lights = (TrafficLightObs(light_id="tl1", state="red", distance_m=7.0, observed_time_s=1.0),)
        obs = _obs(lights=lights, ego_v=5.0)
        cand = _cand(_straight_pts(v=5.0))
        cset = _cset(cand)
        self.kernel.tick(obs, cset, now_s=1.0)
        self.kernel.reset()
        long_b = self.kernel.repair._backends[RepairMode.LONGITUDINAL]
        self.assertIsNone(getattr(long_b, "_last_z", "missing"))


if __name__ == "__main__":
    unittest.main()
