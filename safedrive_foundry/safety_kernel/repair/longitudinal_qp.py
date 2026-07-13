"""Longitudinal trajectory repair: speed/accel/jerk only, path (x,y,yaw,kappa) fixed.

Reduced-space QP free variables:
  z = [v0, a_0..a_{n-2}, slack_stop_*, slack_lead_*, slack_v_*]
s,v follow discrete double-integrator maps (no dynamics equality block).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    ObservableSnapshot,
    PolicyCandidate,
    TrajectoryPoint,
)
from safety_kernel.repair.qp_solver import LongitudinalQPSolver, QPProblem
from safety_kernel.repair.types import (
    RepairMetrics,
    RepairMode,
    RepairResult,
    SolverStatus,
    SolverTrace,
)


def _arc_lengths(points: Sequence[TrajectoryPoint]) -> np.ndarray:
    s = np.zeros(len(points), dtype=float)
    for i in range(1, len(points)):
        ds = math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
        s[i] = s[i - 1] + max(ds, 1e-6)
    return s


def _interp_path(
    path_s: np.ndarray,
    points: Sequence[TrajectoryPoint],
    query_s: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)
    yaws = np.unwrap(np.array([p.yaw for p in points], dtype=float))
    kappas = np.array([p.kappa for p in points], dtype=float)
    s_end = float(path_s[-1])
    q = np.clip(query_s, 0.0, max(s_end, 1e-6))
    x = np.interp(q, path_s, xs)
    y = np.interp(q, path_s, ys)
    yaw = np.interp(q, path_s, yaws)
    kappa = np.interp(q, path_s, kappas)
    return list(zip(x.tolist(), y.tolist(), yaw.tolist(), kappa.tolist(), strict=True))


def _lead_s_profile(
    obs: ObservableSnapshot,
    path_s: np.ndarray,
    points: Sequence[TrajectoryPoint],
    times: np.ndarray,
    cfg: SafetyKernelConfig,
    *,
    min_gap_m: float,
    time_headway_s: float,
) -> np.ndarray:
    n = len(times)
    s_max = np.full(n, path_s[-1] + 80.0, dtype=float)
    if not obs.actors or len(points) < 1:
        return s_max

    if len(points) >= 2:
        dx = points[1].x - points[0].x
        dy = points[1].y - points[0].y
        path_len = math.hypot(dx, dy)
    else:
        path_len = 0.0
    if path_len < 1e-6:
        ux, uy = math.cos(obs.ego_yaw), math.sin(obs.ego_yaw)
    else:
        ux, uy = dx / path_len, dy / path_len

    t0 = float(times[0])
    for actor in obs.actors:
        if actor.lost:
            continue
        rel_x = actor.x - points[0].x
        rel_y = actor.y - points[0].y
        s_actor0 = rel_x * ux + rel_y * uy
        if s_actor0 < -1.0:
            continue
        v_long = actor.vx * ux + actor.vy * uy
        for k, t in enumerate(times):
            dt = float(t - t0)
            s_actor = s_actor0 + v_long * dt
            gap = (
                min_gap_m
                + max(0.0, v_long) * time_headway_s
                + 0.5 * cfg.length_m
                + cfg.collision_inflate_m
            )
            s_max[k] = min(s_max[k], s_actor - gap)
    return s_max


def _stop_s(obs: ObservableSnapshot, cfg: SafetyKernelConfig, buffer_m: float) -> float | None:
    stop = None
    if not cfg.enforce_red_light_stop:
        return None
    for light in obs.traffic_lights:
        if light.state != "red":
            continue
        # Only enforce stop geometry for lights inside validator horizon.
        if light.distance_m > cfg.red_light_stop_distance_m + 1e-6:
            continue
        cand = max(0.05, float(light.distance_m) - buffer_m)
        stop = cand if stop is None else min(stop, cand)
    return stop


def is_longitudinally_repairable(reject_messages: Sequence[str]) -> bool:
    if not reject_messages:
        return False
    repairable_tokens = (
        "dynamics",
        "collision",
        "rules",
        "speed",
        "accel",
        "jerk",
        "red_light",
        "speed_limit",
        "collision_envelope",
    )
    hard_block = (
        "numeric",
        "schema",
        "freshness",
        "time_order",
        "road",
        "offroad",
        "trackability",
        "privilege",
        "teleport",
        "yaw_rate",
        "missing_candidate",
    )
    joined = " ".join(reject_messages).lower()
    if any(tok in joined for tok in hard_block):
        return False
    return any(tok in joined for tok in repairable_tokens)


def _build_sv_maps(n: int, n_a: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Maps z=[v0,a...] → v,s.

    v[k] = v0 + dt * sum_{i<k} a[i]
    s[0] = 0
    s[k] = sum_{i<k} (v[i]*dt + 0.5*a[i]*dt^2)
    """
    n_var_dyn = 1 + n_a
    V = np.zeros((n, n_var_dyn))
    S = np.zeros((n, n_var_dyn))
    for k in range(n):
        V[k, 0] = 1.0
        for i in range(min(k, n_a)):
            V[k, 1 + i] = dt
    for k in range(1, n):
        for i in range(k):
            # s[k] += v[i]*dt
            S[k] += V[i] * dt
            if i < n_a:
                S[k, 1 + i] += 0.5 * dt * dt
    return V, S


class LongitudinalQPRepair:
    """Default longitudinal QP repairer (path-fixed, reduced-space OSQP-form)."""

    def __init__(self, config: SafetyKernelConfig) -> None:
        self.config = config
        self._last_z: np.ndarray | None = None
        qp = config.qp
        self.solver = LongitudinalQPSolver(
            max_iter=qp.max_iter,
            abs_tol=qp.abs_tol,
            rel_tol=qp.rel_tol,
            deadline_ms=qp.deadline_ms,
            prefer_osqp=qp.solver in {"osqp", "osqp_auto"},
        )

    def clear_warm_start(self) -> None:
        self._last_z = None
        self.solver.clear_warm_start()

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult:
        cfg = self.config
        qp = cfg.qp
        now = obs.simulation_time_s if now_s is None else now_s
        empty = RepairMetrics(
            safety_margin_m=-1.0,
            modification_norm=0.0,
            unjustified_stop=False,
            progress_ratio=0.0,
            progress_loss=1.0,
            comfort_jerk_rms=0.0,
            comfort_accel_rms=0.0,
            slack_used_max=0.0,
            terminal_speed_mps=0.0,
            terminal_s_m=0.0,
        )

        if not qp.enabled:
            return self._fail(candidate, SolverStatus.DISABLED, "qp_disabled", empty)

        # Deterministic offline fault injection (G2-05): never return a success on timeout inject.
        if bool(candidate.dynamics_meta.get("inject_solver_timeout")):
            return self._fail(
                candidate,
                SolverStatus.TIMEOUT,
                "timeout",
                empty,
                message="inject_solver_timeout",
            )

        age = now - candidate.generated_time_s
        if age > cfg.max_candidate_age_s or candidate.valid_until_s < now:
            return self._fail(
                candidate,
                SolverStatus.STALE_INPUT,
                "stale_input",
                empty,
                message=f"age={age:.4f}",
            )
        if len(candidate.points) < cfg.min_points:
            return self._fail(candidate, SolverStatus.NOT_REPAIRABLE, "too_few_points", empty)
        if reject_hints and not is_longitudinally_repairable(reject_hints):
            return self._fail(candidate, SolverStatus.NOT_REPAIRABLE, "not_repairable", empty)

        points = candidate.points
        n = len(points)
        n_a = n - 1
        path_s = _arc_lengths(points)
        times = np.array([p.t for p in points], dtype=float)
        dts = np.diff(times)
        dts = dts[dts > 1e-6]
        dt = float(np.median(dts)) if len(dts) else cfg.control_period_s
        dt = max(dt, 1e-3)

        v_ref = np.array([max(0.0, p.v) for p in points], dtype=float)
        s_ref = path_s.copy()
        v_ego = max(0.0, float(obs.ego_v if math.isfinite(obs.ego_v) else points[0].v))
        a_ego = float(obs.ego_a if math.isfinite(obs.ego_a) else points[0].a)

        stop_s = _stop_s(obs, cfg, qp.stop_line_buffer_m)
        s_lead = _lead_s_profile(
            obs, path_s, points, times, cfg, min_gap_m=qp.min_gap_m, time_headway_s=qp.time_headway_s
        )

        v_max = cfg.max_speed_mps
        if cfg.enforce_speed_limit and obs.speed_limit_mps is not None:
            v_max = min(v_max, obs.speed_limit_mps + cfg.speed_limit_margin_mps)

        # z = [v0, a..., sl_stop(n), sl_lead(n), sl_v(n)]
        n_dyn = 1 + n_a
        off_v0 = 0
        off_a = 1
        off_ss = n_dyn
        off_sl = off_ss + n
        off_sv = off_sl + n
        n_var = off_sv + n

        V_dyn, S_dyn = _build_sv_maps(n, n_a, dt)  # maps dyn part only

        # Expand to full z maps
        V = np.zeros((n, n_var))
        S = np.zeros((n, n_var))
        V[:, :n_dyn] = V_dyn
        S[:, :n_dyn] = S_dyn

        # Cost 1/2 z'P z + q'z
        P = np.zeros((n_var, n_var))
        q = np.zeros(n_var)

        # Soft initial velocity to ego (allow red-light drop).
        P[off_v0, off_v0] += 2.0 * qp.w_v_ref * 4.0
        q[off_v0] += -2.0 * qp.w_v_ref * 4.0 * v_ego

        # Track v_ref and s_ref
        W_v = qp.w_v_ref * np.eye(n)
        P += 2.0 * (V.T @ W_v @ V)
        q += 2.0 * (V.T @ W_v @ (-v_ref))

        w_s = np.full(n, 0.5 * qp.w_progress)
        w_s[-1] = 2.0 * qp.w_progress
        W_s = np.diag(w_s)
        P += 2.0 * (S.T @ W_s @ S)
        q += 2.0 * (S.T @ W_s @ (-s_ref))

        # Accel / jerk comfort
        for k in range(n_a):
            P[off_a + k, off_a + k] += 2.0 * qp.w_a
        for k in range(1, n_a):
            c = qp.w_jerk / (dt * dt)
            P[off_a + k, off_a + k] += 2.0 * c
            P[off_a + k - 1, off_a + k - 1] += 2.0 * c
            P[off_a + k, off_a + k - 1] -= 2.0 * c
            P[off_a + k - 1, off_a + k] -= 2.0 * c
        if n_a >= 1:
            c0 = qp.w_jerk / (dt * dt)
            P[off_a, off_a] += 2.0 * c0
            q[off_a] += -2.0 * c0 * a_ego

        for k in range(n):
            P[off_ss + k, off_ss + k] += 2.0 * qp.w_slack
            P[off_sl + k, off_sl + k] += 2.0 * qp.w_slack
            P[off_sv + k, off_sv + k] += 2.0 * qp.w_slack

        # Ridge
        P += 1e-8 * np.eye(n_var)

        # Constraints
        rows: list[np.ndarray] = []
        lows: list[float] = []
        ups: list[float] = []

        def add(row: np.ndarray, lo: float, hi: float) -> None:
            rows.append(row)
            lows.append(lo)
            ups.append(hi)

        # a bounds
        for k in range(n_a):
            r = np.zeros(n_var)
            r[off_a + k] = 1.0
            add(r, -cfg.max_decel_mps2, cfg.max_accel_mps2)

        # jerk bounds
        if n_a >= 1:
            r = np.zeros(n_var)
            r[off_a] = 1.0 / dt
            add(r, -cfg.max_jerk_mps3 + a_ego / dt, cfg.max_jerk_mps3 + a_ego / dt)
        for k in range(1, n_a):
            r = np.zeros(n_var)
            r[off_a + k] = 1.0 / dt
            r[off_a + k - 1] = -1.0 / dt
            add(r, -cfg.max_jerk_mps3, cfg.max_jerk_mps3)

        red_active = stop_s is not None
        # v0 bounds
        r = np.zeros(n_var)
        r[off_v0] = 1.0
        add(r, 0.0, v_max)

        for k in range(n):
            # 0 <= v
            add(V[k].copy(), 0.0, float("inf"))
            # v - slack_v <= v_max
            r = V[k].copy()
            r[off_sv + k] -= 1.0
            add(r, -float("inf"), v_max)
            # slack_v bounds
            r = np.zeros(n_var)
            r[off_sv + k] = 1.0
            add(r, 0.0, qp.slack_speed_max_mps)

            # lead: s - slack_lead <= s_lead
            r = S[k].copy()
            r[off_sl + k] -= 1.0
            add(r, -float("inf"), float(s_lead[k]))
            r = np.zeros(n_var)
            r[off_sl + k] = 1.0
            add(r, 0.0, qp.slack_lead_max_m)

            # stop line + stop slack (fixed 0 when no red)
            if red_active:
                r = S[k].copy()
                r[off_ss + k] -= 1.0
                add(r, -float("inf"), float(stop_s))
                r = np.zeros(n_var)
                r[off_ss + k] = 1.0
                add(r, 0.0, qp.slack_stop_max_m)
            else:
                r = np.zeros(n_var)
                r[off_ss + k] = 1.0
                add(r, 0.0, 0.0)

            # Red approach speed for t<=2s (matches validator — hard, no slack).
            if red_active and (times[k] - times[0]) <= 2.0 + 1e-9:
                add(V[k].copy(), -float("inf"), cfg.red_light_max_approach_speed_mps)

        A = np.vstack(rows)
        l_vec = np.array(lows, dtype=float)
        u_vec = np.array(ups, dtype=float)
        big = 1.0e5
        l_vec = np.where(np.isfinite(l_vec), l_vec, -big)
        u_vec = np.where(np.isfinite(u_vec), u_vec, big)

        # Seed: constant decel if stop/lead, else track ref.
        z0 = np.zeros(n_var)
        if red_active:
            z0[off_v0] = min(v_ego, cfg.red_light_max_approach_speed_mps)
            a_cmd = -min(cfg.max_decel_mps2, max(0.5, (v_ego * v_ego) / (2.0 * max(float(stop_s), 0.5))))
            z0[off_a:off_a + n_a] = a_cmd
        elif float(np.min(s_lead)) < s_ref[-1] - 0.5:
            z0[off_v0] = v_ego
            s_cap = max(0.5, float(np.min(s_lead)))
            a_cmd = -min(cfg.max_decel_mps2, max(0.1, (v_ego * v_ego) / (2.0 * s_cap)))
            z0[off_a:off_a + n_a] = a_cmd
        else:
            z0[off_v0] = v_ego
            for k in range(n_a):
                z0[off_a + k] = float(
                    np.clip((v_ref[k + 1] - v_ref[k]) / dt, -cfg.max_decel_mps2, cfg.max_accel_mps2)
                )

        if qp.warm_start and self._last_z is not None and self._last_z.size == n_var:
            z0 = 0.5 * z0 + 0.5 * self._last_z
            self.solver.set_warm_start(z0)
        else:
            self.solver.set_warm_start(z0)

        problem = QPProblem(P=P, q=q, A=A, l=l_vec, u=u_vec)
        z, trace = self.solver.solve(problem, warm_start=qp.warm_start)

        if z is None or trace.status not in {SolverStatus.SOLVED, SolverStatus.SOLVED_INACCURATE}:
            return RepairResult(
                mode=RepairMode.LONGITUDINAL,
                success=False,
                candidate=None,
                pre_repair_id=candidate.candidate_id,
                post_repair_id=None,
                solver_trace=trace,
                metrics=empty,
                reason=f"solver_{trace.status.value}",
            )

        status = trace.status
        v_opt = np.maximum(0.0, V @ z)
        s_opt = np.maximum(S @ z, 0.0)
        a_opt = z[off_a:off_a + n_a]
        slack_max = float(
            max(
                np.max(np.abs(z[off_ss:off_ss + n])),
                np.max(np.abs(z[off_sl:off_sl + n])),
                np.max(np.abs(z[off_sv:off_sv + n])),
                0.0,
            )
        )

        path_samples = _interp_path(path_s, points, s_opt)
        new_pts: list[TrajectoryPoint] = []
        for k in range(n):
            x, y, yaw, kappa = path_samples[k]
            a = float(a_opt[k]) if k < n_a else (float(a_opt[-1]) if n_a else 0.0)
            a = float(np.clip(a, -cfg.max_decel_mps2, cfg.max_accel_mps2))
            if k == 0:
                jerk = (a - a_ego) / dt
            elif k < n_a:
                jerk = (float(a_opt[k]) - float(a_opt[k - 1])) / dt
            else:
                jerk = 0.0
            jerk = float(np.clip(jerk, -cfg.max_jerk_mps3, cfg.max_jerk_mps3))
            v_k = float(max(0.0, v_opt[k]))
            if red_active and (times[k] - times[0]) <= 2.0 + 1e-9:
                v_k = min(v_k, cfg.red_light_max_approach_speed_mps)
            v_k = min(v_k, v_max)
            new_pts.append(
                TrajectoryPoint(
                    t=float(times[k]),
                    x=float(x),
                    y=float(y),
                    yaw=float(yaw),
                    kappa=float(kappa),
                    v=v_k,
                    a=a,
                    jerk=jerk,
                )
            )

        post_id = f"{candidate.candidate_id}__qp"
        repaired = PolicyCandidate(
            candidate_id=post_id,
            source=candidate.source,
            generated_time_s=candidate.generated_time_s,
            valid_until_s=candidate.valid_until_s,
            probability=candidate.probability,
            points=tuple(new_pts),
            behavior=candidate.behavior or "longitudinal_qp_repair",
            critical_actor=candidate.critical_actor,
            conflict_type=candidate.conflict_type,
            risk_horizon_s=candidate.risk_horizon_s,
            intended_action=candidate.intended_action or "qp_repair",
            uncertainty=candidate.uncertainty,
            availability=True,
            dynamics_meta={
                **dict(candidate.dynamics_meta),
                "repair": "longitudinal_qp",
                "pre_repair_id": candidate.candidate_id,
                "solver_backend": trace.backend,
                "solver_status": status.value,
                "osqp_form": "reduced_space_v0_a_slacks",
            },
        )

        v_exec = np.array([p.v for p in new_pts], dtype=float)
        v_orig = np.array([p.v for p in points], dtype=float)
        mod = float(np.linalg.norm(v_exec - v_orig) / math.sqrt(n))
        progress_ratio = float(s_opt[-1] / max(float(s_ref[-1]), 1e-3))
        mean_v = float(np.mean(v_exec))
        constrained = red_active or float(np.min(s_lead)) < float(s_ref[-1]) - 1.0
        unjustified = (mean_v < 0.3 and progress_ratio < qp.min_progress_ratio) and not constrained
        lead_margin = float(np.min(s_lead - s_opt))
        stop_margin = float(float(stop_s) - float(np.max(s_opt))) if red_active else 1.0
        metrics = RepairMetrics(
            safety_margin_m=min(lead_margin, stop_margin),
            modification_norm=mod,
            unjustified_stop=unjustified,
            progress_ratio=progress_ratio,
            progress_loss=max(0.0, 1.0 - progress_ratio),
            comfort_jerk_rms=float(np.sqrt(np.mean(np.array([p.jerk for p in new_pts]) ** 2))),
            comfort_accel_rms=float(np.sqrt(np.mean(np.array([p.a for p in new_pts]) ** 2))),
            slack_used_max=slack_max,
            terminal_speed_mps=float(v_exec[-1]),
            terminal_s_m=float(s_opt[-1]),
        )

        if unjustified:
            return RepairResult(
                mode=RepairMode.LONGITUDINAL,
                success=False,
                candidate=None,
                pre_repair_id=candidate.candidate_id,
                post_repair_id=None,
                solver_trace=SolverTrace(
                    status=SolverStatus.INFEASIBLE,
                    iterations=trace.iterations,
                    primal_residual=trace.primal_residual,
                    dual_residual=trace.dual_residual,
                    objective=trace.objective,
                    latency_ms=trace.latency_ms,
                    warm_started=trace.warm_started,
                    backend=trace.backend,
                    message="unjustified_zero_speed",
                ),
                metrics=metrics,
                reason="unjustified_zero_speed",
            )

        self._last_z = z.copy()
        return RepairResult(
            mode=RepairMode.LONGITUDINAL,
            success=True,
            candidate=repaired,
            pre_repair_id=candidate.candidate_id,
            post_repair_id=post_id,
            solver_trace=trace,
            metrics=metrics,
            reason="ok",
        )

    def _fail(
        self,
        candidate: PolicyCandidate,
        status: SolverStatus,
        reason: str,
        metrics: RepairMetrics,
        *,
        message: str = "",
    ) -> RepairResult:
        return RepairResult(
            mode=RepairMode.LONGITUDINAL,
            success=False,
            candidate=None,
            pre_repair_id=candidate.candidate_id,
            post_repair_id=None,
            solver_trace=SolverTrace(
                status=status,
                iterations=0,
                primal_residual=0.0,
                dual_residual=0.0,
                objective=0.0,
                latency_ms=0.0,
                warm_started=False,
                backend="none",
                message=message or reason,
            ),
            metrics=metrics,
            reason=reason,
        )


