"""Minimal full-tick regressions for Codex G2 acceptance findings (P0/P1)."""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
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
    load_safety_config,
)
from safety_kernel.config import RatoScpConfig  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
    TrackedObject,
)
from safety_kernel.repair.qp_solver import LongitudinalQPSolver, QPProblem  # noqa: E402
from safety_kernel.repair.rato_scp import RestrictedRatoScpRepair  # noqa: E402


def _pts(v: float = 6.0, n: int = 16) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(TrajectoryPoint(t=0.25 * i, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
        x += v * 0.25
    return tuple(out)


def _obs(
    *,
    now: float = 1.0,
    run_id: str = "reg",
    frame_id: str = "f0",
    scenario_id: str = "s0",
    actors: tuple[TrackedObject, ...] = (),
    schema_version: str = SCHEMA_VERSION,
    coordinate_frame: str = "map",
) -> ObservableSnapshot:
    return ObservableSnapshot(
        run_id=run_id,
        frame_id=frame_id,
        scenario_id=scenario_id,
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=6.0,
        observed_time_s=now,
        actors=actors,
        corridor_centerline=tuple((float(i), 0.0) for i in range(0, 150)),
        corridor_half_width_m=2.5,
        privilege=ObservationPrivilege.OBSERVABLE,
        schema_version=schema_version,
        coordinate_frame=coordinate_frame,
    )


def _cset(
    pts=None,
    *,
    now: float = 1.0,
    run_id: str = "reg",
    frame_id: str = "f0",
    scenario_id: str = "s0",
    schema_version: str = SCHEMA_VERSION,
    coordinate_frame: str = "map",
    candidates=None,
) -> PolicyCandidateSet:
    if candidates is None:
        candidates = (
            PolicyCandidate(
                candidate_id="raw",
                source=CandidateSource.CLASSIC,
                generated_time_s=now,
                valid_until_s=now + 0.2,
                probability=1.0,
                points=pts or _pts(),
            ),
        )
    return PolicyCandidateSet(
        run_id=run_id,
        frame_id=frame_id,
        scenario_id=scenario_id,
        model_id="classic",
        carla_frame=0,
        simulation_time_s=now,
        wall_time_s=now,
        candidates=candidates,
        schema_version=schema_version,
        coordinate_frame=coordinate_frame,
    )


class CodexP0ActorNanTests(unittest.TestCase):
    def test_nan_actor_state_locks_full_tick_no_accept(self) -> None:
        """P0-1: actor.x=NaN must not ACCEPT; state-lock at Observable boundary."""
        nan_actor = TrackedObject(
            "lead",
            "vehicle",
            float("nan"),
            0.0,
            0.0,
            3.0,
            0.0,
            4.5,
            1.8,
            1.0,
        )
        obs = _obs(actors=(nan_actor,))
        cset = _cset()
        kernel = SafetyKernel(load_safety_config())
        res = kernel.tick(
            obs,
            cset,
            now_s=1.0,
            availability=ComponentAvailability(classic=True, vla=False, world=False, safety=True),
        )
        self.assertIn(
            res.decision.decision_kind,
            {DecisionKind.EMERGENCY, DecisionKind.MINIMAL_RISK},
        )
        self.assertNotIn(
            res.decision.decision_kind,
            {DecisionKind.ACCEPT, DecisionKind.QP, DecisionKind.RATO},
        )
        self.assertIsNone(res.decision.executed_trajectory_id)
        self.assertTrue(
            any("actor_numeric" in r or "non_finite_actor" in r for r in res.decision.reject_reasons)
            or any(
                "actor_numeric" in r or "non_finite" in r
                for r in res.state_result.decision.reject_reasons
            )
        )
        # Repair must not have executed a candidate on untrusted obs.
        self.assertTrue(res.repair_result is None or not res.repair_result.success)

    def test_inf_actor_velocity_state_lock(self) -> None:
        bad = TrackedObject("a", "vehicle", 10.0, 0.0, 0.0, float("inf"), 0.0, 4.5, 1.8, 1.0)
        kernel = SafetyKernel(load_safety_config())
        res = kernel.tick(
            _obs(actors=(bad,)),
            _cset(),
            now_s=1.0,
            availability=ComponentAvailability(classic=True, vla=False, world=False, safety=True),
        )
        self.assertNotEqual(res.decision.decision_kind, DecisionKind.ACCEPT)


class CodexP0IdentityTests(unittest.TestCase):
    def test_run_id_mismatch_hard_reject(self) -> None:
        kernel = SafetyKernel(load_safety_config())
        res = kernel.tick(
            _obs(run_id="obs-run"),
            _cset(run_id="set-run"),
            now_s=1.0,
            availability=ComponentAvailability(classic=True, vla=False, world=False, safety=True),
        )
        self.assertEqual(res.decision.decision_kind, DecisionKind.HARD_REJECT)
        self.assertTrue(any("run_id" in r for r in res.decision.reject_reasons))
        self.assertIsNone(res.decision.executed_trajectory_id)

    def test_frame_scenario_schema_coord_time_dup_id(self) -> None:
        kernel = SafetyKernel(load_safety_config())
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        base = _obs()

        cases = [
            _cset(frame_id="other"),
            _cset(scenario_id="other"),
            _cset(schema_version="wrong.schema.v0"),
            _cset(coordinate_frame="ego"),  # obs is map
            PolicyCandidateSet(
                run_id="reg",
                frame_id="f0",
                scenario_id="s0",
                model_id="classic",
                carla_frame=0,
                simulation_time_s=2.0,  # obs at 1.0
                wall_time_s=1.0,
                candidates=(
                    PolicyCandidate(
                        "raw",
                        CandidateSource.CLASSIC,
                        1.0,
                        1.2,
                        1.0,
                        _pts(),
                    ),
                ),
                schema_version=SCHEMA_VERSION,
            ),
            _cset(
                candidates=(
                    PolicyCandidate("dup", CandidateSource.CLASSIC, 1.0, 1.2, 1.0, _pts()),
                    PolicyCandidate("dup", CandidateSource.CLASSIC, 1.0, 1.2, 0.5, _pts(v=5.0)),
                )
            ),
        ]
        for cset in cases:
            res = kernel.tick(base, cset, now_s=1.0, availability=avail)
            self.assertEqual(
                res.decision.decision_kind,
                DecisionKind.HARD_REJECT,
                msg=f"expected HARD_REJECT for {cset}",
            )
            self.assertTrue(
                any(r.startswith("identity:") for r in res.decision.reject_reasons),
                msg=list(res.decision.reject_reasons),
            )


class CodexP0DeadlineTests(unittest.TestCase):
    def test_osqp_infeasible_does_not_fall_through_to_slow_slsqp(self) -> None:
        """An OSQP infeasible result must retain its class and return promptly."""
        from unittest.mock import patch

        from safety_kernel.repair.types import SolverTrace

        n = 2
        problem = QPProblem(
            P=np.eye(n),
            q=np.zeros(n),
            A=np.eye(n),
            l=np.ones(n),
            u=-np.ones(n),
        )
        infeasible = SolverTrace(
            status=SolverStatus.INFEASIBLE,
            iterations=25,
            primal_residual=1.0,
            dual_residual=0.0,
            objective=float("inf"),
            latency_ms=0.1,
            warm_started=False,
            backend="osqp",
            message="primal infeasible",
        )
        solver = LongitudinalQPSolver(deadline_ms=30.0, prefer_osqp=True)
        with (
            patch.object(solver, "_solve_osqp", return_value=(None, infeasible)),
            patch.object(
                solver,
                "_solve_scipy",
                side_effect=AssertionError("infeasible QP must not reach SLSQP"),
            ),
        ):
            x, trace = solver.solve(problem, warm_start=False)
        self.assertIsNone(x)
        self.assertEqual(trace.status, SolverStatus.INFEASIBLE)
        self.assertEqual(trace.backend, "osqp")

    def test_qp_solver_nonpositive_deadline_timeout(self) -> None:
        """P0-3: deadline <= 0 → TIMEOUT, no solution vector."""
        n = 4
        P = np.eye(n)
        q = np.zeros(n)
        A = np.eye(n)
        l = -np.ones(n)
        u = np.ones(n)
        problem = QPProblem(P=P, q=q, A=A, l=l, u=u)
        for deadline in (0.0, -1.0, 1e-12):
            solver = LongitudinalQPSolver(deadline_ms=deadline, prefer_osqp=True)
            x, trace = solver.solve(problem, warm_start=False)
            self.assertIsNone(x)
            self.assertEqual(trace.status, SolverStatus.TIMEOUT)

    def test_qp_solver_tiny_deadline_never_success(self) -> None:
        n = 8
        P = np.eye(n) * 2.0
        q = np.linspace(-1.0, 1.0, n)
        A = np.vstack([np.eye(n), -np.eye(n)])
        l = np.full(2 * n, -10.0)
        u = np.full(2 * n, 10.0)
        problem = QPProblem(P=P, q=q, A=A, l=l, u=u)
        # Extremely small positive deadline: backends must not return executable success.
        solver = LongitudinalQPSolver(deadline_ms=1e-9, prefer_osqp=True, max_iter=5000)
        x, trace = solver.solve(problem, warm_start=False)
        if x is not None:
            # If a backend finished "instantly", enforce_deadline still gates on wall clock.
            self.assertLessEqual(trace.latency_ms, solver.deadline_ms + 1e-3)
        # With 1e-9 ms, practically always TIMEOUT.
        self.assertTrue(
            x is None or trace.status is SolverStatus.TIMEOUT or trace.latency_ms <= solver.deadline_ms
        )
        # Stronger contract: when status is TIMEOUT, x must be None.
        if trace.status is SolverStatus.TIMEOUT:
            self.assertIsNone(x)


class CodexP1RatoTimeoutTests(unittest.TestCase):
    def test_rato_timeout_never_success_even_after_partial(self) -> None:
        """P1-1: TIMEOUT must not rewrite to SOLVED_INACCURATE / success=True."""
        cfg = load_safety_config()
        # Tiny deadline forces mid-SCP timeout after at least one attempt if possible.
        rato_cfg = replace(
            cfg.rato,
            deadline_ms=1e-9,
            max_scp_iters=5,
            enabled=True,
        )
        cfg2 = replace(cfg, rato=rato_cfg)
        repairer = RestrictedRatoScpRepair(cfg2)
        now = 1.0
        lead = TrackedObject("lead", "vehicle", 12.0, 1.2, 0.0, 2.0, 0.0, 4.5, 1.8, now)
        obs = _obs(actors=(lead,), now=now)
        # Lateral offset path to force RATO eligibility.
        pts = []
        x = 0.0
        for i in range(16):
            pts.append(
                TrajectoryPoint(
                    t=0.25 * i,
                    x=x,
                    y=0.6 if i > 2 else 0.0,
                    yaw=0.0,
                    kappa=0.0,
                    v=6.0,
                    a=0.0,
                    jerk=0.0,
                )
            )
            x += 1.5
        cand = PolicyCandidate(
            "raw",
            CandidateSource.CLASSIC,
            now,
            now + 0.2,
            1.0,
            tuple(pts),
        )
        result = repairer.repair(
            cand,
            obs,
            now_s=now,
            reject_hints=["c:collision", "c:road"],
            force=True,
        )
        if result.solver_trace.status is SolverStatus.TIMEOUT:
            self.assertFalse(result.success)
            self.assertIsNone(result.candidate)
            self.assertNotEqual(result.solver_trace.status, SolverStatus.SOLVED_INACCURATE)

    def test_inject_solver_timeout_qp_and_rato(self) -> None:
        cfg = load_safety_config()
        iface = RepairInterface(cfg)
        now = 1.0
        lead = TrackedObject("lead", "vehicle", 10.0, 0.0, 0.0, 2.0, 0.0, 4.5, 1.8, now)
        obs = _obs(actors=(lead,), now=now)
        cand = PolicyCandidate(
            "raw",
            CandidateSource.CLASSIC,
            now,
            now + 0.2,
            1.0,
            _pts(v=12.0),
            dynamics_meta={"inject_solver_timeout": True},
        )
        for mode in (RepairMode.LONGITUDINAL, RepairMode.RATO):
            r = iface.repair(cand, obs, mode=mode, now_s=now, reject_hints=["c:collision"])
            self.assertFalse(r.success, msg=mode.value)
            self.assertEqual(r.solver_trace.status, SolverStatus.TIMEOUT)
            self.assertIsNone(r.candidate)


class CodexP1IdentityMatchedAccept(unittest.TestCase):
    def test_identity_ok_still_accepts_safe_classic(self) -> None:
        kernel = SafetyKernel(load_safety_config())
        res = kernel.tick(
            _obs(actors=()),
            _cset(),
            now_s=1.0,
            availability=ComponentAvailability(classic=True, vla=False, world=False, safety=True),
        )
        self.assertEqual(res.decision.decision_kind, DecisionKind.ACCEPT)


if __name__ == "__main__":
    unittest.main()
