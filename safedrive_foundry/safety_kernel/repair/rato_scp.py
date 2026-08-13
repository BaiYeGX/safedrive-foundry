"""Restricted 2D RATO-SCP: finite-iteration SCP on Frenet lateral offsets.

Not a second local planner. Lateral repair only when a legal corridor exists;
path longitudinal timing is re-profiled after geometry repair.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    ObservableSnapshot,
    PolicyCandidate,
    TrajectoryPoint,
)
from safety_kernel.repair.corridor import (
    build_corridor_frame,
    frenet_of_trajectory,
    has_legal_lateral_corridor,
    is_rato_eligible_hints,
    project_xy,
    xy_from_frenet,
)
from safety_kernel.repair.qp_solver import LongitudinalQPSolver, QPProblem
from safety_kernel.repair.types import (
    RepairMetrics,
    RepairMode,
    RepairResult,
    SolverStatus,
    SolverTrace,
)


def _actor_radius(length_m: float, width_m: float, cfg: SafetyKernelConfig) -> float:
    return 0.5 * math.hypot(length_m, width_m) + cfg.collision_inflate_m + 0.5 * cfg.width_m


def _empty_metrics() -> RepairMetrics:
    return RepairMetrics(
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


def _path_progress_m(points: Sequence[TrajectoryPoint]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
    return total


def _yaw_kappa(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    yaw = np.zeros(n)
    kappa = np.zeros(n)
    for k in range(n):
        if k + 1 < n:
            dx = xs[k + 1] - xs[k]
            dy = ys[k + 1] - ys[k]
        else:
            dx = xs[k] - xs[k - 1]
            dy = ys[k] - ys[k - 1]
        yaw[k] = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else (yaw[k - 1] if k else 0.0)
    for k in range(1, n - 1):
        ds = math.hypot(xs[k + 1] - xs[k - 1], ys[k + 1] - ys[k - 1])
        if ds > 1e-6:
            kappa[k] = float((yaw[k + 1] - yaw[k - 1]) / ds)
    if n >= 2:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
    return yaw, kappa


class RestrictedRatoScpRepair:
    """Secondary restricted RATO-SCP repairer (Frenet lateral SCP + speed re-profile)."""

    def __init__(self, config: SafetyKernelConfig) -> None:
        self.config = config
        self._last_d: np.ndarray | None = None
        rato = config.rato
        # Per-iteration QP budget: leave headroom for SCP outer loop.
        per_iter_deadline = max(25.0, 0.7 * rato.deadline_ms / max(1, rato.max_scp_iters))
        self.solver = LongitudinalQPSolver(
            max_iter=6000,
            abs_tol=1.0e-4,
            rel_tol=1.0e-4,
            deadline_ms=per_iter_deadline,
            prefer_osqp=True,
        )

    def clear_warm_start(self) -> None:
        self._last_d = None
        self.solver.clear_warm_start()

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
        force: bool = False,
    ) -> RepairResult:
        cfg = self.config
        rato = cfg.rato
        now = obs.simulation_time_s if now_s is None else now_s
        empty = _empty_metrics()
        t0 = time.perf_counter()

        if not rato.enabled and not force:
            return self._fail(candidate, SolverStatus.DISABLED, "rato_disabled", empty)

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
        if reject_hints and not is_rato_eligible_hints(reject_hints) and not force:
            return self._fail(candidate, SolverStatus.NOT_REPAIRABLE, "not_rato_eligible", empty)
        if not has_legal_lateral_corridor(obs, cfg, min_clearance_m=rato.min_lateral_clearance_m):
            return self._fail(candidate, SolverStatus.NOT_REPAIRABLE, "no_legal_corridor", empty)

        frame = build_corridor_frame(obs, cfg)
        assert frame is not None
        points = candidate.points
        n = len(points)
        times = np.array([p.t for p in points], dtype=float)
        s_ref, d_ref, _, _, _, _ = frenet_of_trajectory(points, frame)
        d_bound = frame.lateral_room_m

        # Seed lateral offsets: free-side dodge guess, optional warm start blend.
        d_cur = self._seed_lateral(
            d_ref=d_ref,
            s_ref=s_ref,
            times=times,
            obs=obs,
            frame=frame,
            d_bound=d_bound,
        )
        if rato.warm_start and self._last_d is not None and self._last_d.size == n:
            d_cur = 0.5 * d_cur + 0.5 * np.clip(self._last_d, -d_bound, d_bound)

        total_qp_iters = 0
        last_backend = "none"
        last_status = SolverStatus.MAX_ITER
        slack_max = 0.0
        scp_iters_done = 0
        prev_obj = float("inf")
        oscillation = False
        deadline_hit = False
        warm_any = False
        d_history: list[np.ndarray] = []

        for scp_i in range(rato.max_scp_iters):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms >= rato.deadline_ms:
                deadline_hit = True
                last_status = SolverStatus.TIMEOUT
                break

            z, trace, slack_i = self._solve_lateral_qp(
                d_cur=d_cur,
                d_ref=d_ref,
                s_ref=s_ref,
                d_bound=d_bound,
                times=times,
                points=points,
                obs=obs,
                n=n,
            )
            total_qp_iters += trace.iterations
            last_backend = trace.backend
            warm_any = warm_any or trace.warm_started
            scp_iters_done = scp_i + 1

            if z is None or trace.status not in {SolverStatus.SOLVED, SolverStatus.SOLVED_INACCURATE}:
                last_status = trace.status
                if scp_i == 0:
                    return RepairResult(
                        mode=RepairMode.RATO,
                        success=False,
                        candidate=None,
                        pre_repair_id=candidate.candidate_id,
                        post_repair_id=None,
                        solver_trace=SolverTrace(
                            status=trace.status,
                            iterations=total_qp_iters,
                            primal_residual=trace.primal_residual,
                            dual_residual=trace.dual_residual,
                            objective=trace.objective,
                            latency_ms=(time.perf_counter() - t0) * 1000.0,
                            warm_started=warm_any,
                            backend=last_backend,
                            message=trace.message or "scp_subproblem_fail",
                            extras={"scp_iters": scp_iters_done},
                        ),
                        metrics=empty,
                        reason=f"scp_subproblem_{trace.status.value}",
                    )
                break

            d_next = z[:n].copy()
            slack_max = max(slack_max, slack_i)
            # Trust-region already in QP; clip to corridor with slack.
            d_next = np.clip(d_next, -d_bound - rato.slack_corridor_max_m, d_bound + rato.slack_corridor_max_m)

            # Oscillation: alternating sign of step without objective drop.
            if d_history:
                step = d_next - d_cur
                prev_step = d_cur - d_history[-1]
                if (
                    float(np.linalg.norm(step)) > rato.oscillation_eps_m
                    and float(np.dot(step, prev_step)) < -1e-6
                    and abs(trace.objective - prev_obj) < 1e-3
                ):
                    oscillation = True
                    last_status = SolverStatus.INFEASIBLE
                    break
            d_history.append(d_cur.copy())
            delta = float(np.linalg.norm(d_next - d_cur))
            d_cur = d_next
            last_status = trace.status
            prev_obj = trace.objective
            if delta < rato.oscillation_eps_m:
                break

        if deadline_hit and scp_iters_done == 0:
            return self._fail(
                candidate,
                SolverStatus.TIMEOUT,
                "timeout",
                empty,
                message="deadline_before_first_iter",
            )
        if oscillation:
            return RepairResult(
                mode=RepairMode.RATO,
                success=False,
                candidate=None,
                pre_repair_id=candidate.candidate_id,
                post_repair_id=None,
                solver_trace=SolverTrace(
                    status=SolverStatus.INFEASIBLE,
                    iterations=total_qp_iters,
                    primal_residual=0.0,
                    dual_residual=0.0,
                    objective=prev_obj,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    warm_started=warm_any,
                    backend=last_backend,
                    message="oscillation",
                    extras={"scp_iters": scp_iters_done},
                ),
                metrics=empty,
                reason="oscillation",
            )

        # Optional residual lateral nudge if still slightly in collision envelope.
        d_cur = self._nudge_clear(d_cur, s_ref, times, obs, frame, d_bound)

        # Reconstruct path + speed profile (constant-time grid, arc-length consistent v).
        xs, ys, _, _ = xy_from_frenet(s_ref, d_cur, frame)
        yaw, kappa = _yaw_kappa(xs, ys)
        kappa = np.clip(kappa, -cfg.max_curvature_per_m, cfg.max_curvature_per_m)

        new_pts = self._reprofile_speed(
            times=times,
            xs=xs,
            ys=ys,
            yaw=yaw,
            kappa=kappa,
            v_seed=np.array([max(0.0, p.v) for p in points], dtype=float),
            obs=obs,
            ego_v=max(0.0, float(obs.ego_v if math.isfinite(obs.ego_v) else points[0].v)),
            ego_a=float(obs.ego_a if math.isfinite(obs.ego_a) else points[0].a),
        )

        post_id = f"{candidate.candidate_id}__rato"
        repaired = PolicyCandidate(
            candidate_id=post_id,
            source=candidate.source,
            generated_time_s=candidate.generated_time_s,
            valid_until_s=candidate.valid_until_s,
            probability=candidate.probability,
            points=tuple(new_pts),
            behavior=candidate.behavior or "rato_scp_repair",
            critical_actor=candidate.critical_actor,
            conflict_type=candidate.conflict_type,
            risk_horizon_s=candidate.risk_horizon_s,
            intended_action=candidate.intended_action or "rato_repair",
            uncertainty=candidate.uncertainty,
            availability=True,
            dynamics_meta={
                **dict(candidate.dynamics_meta),
                "repair": "rato_scp",
                "pre_repair_id": candidate.candidate_id,
                "scp_iters": scp_iters_done,
                "solver_backend": last_backend,
                "solver_status": last_status.value,
            },
        )

        # Metrics
        lat_mod = float(np.linalg.norm(d_cur - d_ref) / math.sqrt(n))
        v_exec = np.array([p.v for p in new_pts], dtype=float)
        v_orig = np.array([p.v for p in points], dtype=float)
        mod = float(math.sqrt(lat_mod**2 + (np.linalg.norm(v_exec - v_orig) / math.sqrt(n)) ** 2))
        prog_ref = max(_path_progress_m(points), 1e-3)
        prog_new = _path_progress_m(new_pts)
        progress_ratio = float(prog_new / prog_ref)
        # Collision margin estimate
        safety_margin = self._min_collision_margin(new_pts, obs)
        # Corridor margin
        _, d_final, _, _, _, _ = frenet_of_trajectory(new_pts, frame)
        corridor_margin = float(d_bound - np.max(np.abs(d_final)))
        safety_margin = min(safety_margin, corridor_margin)
        mean_v = float(np.mean(v_exec))
        unjustified = mean_v < 0.25 and progress_ratio < rato.min_progress_ratio

        metrics = RepairMetrics(
            safety_margin_m=safety_margin,
            modification_norm=mod,
            unjustified_stop=unjustified,
            progress_ratio=progress_ratio,
            progress_loss=max(0.0, 1.0 - progress_ratio),
            comfort_jerk_rms=float(np.sqrt(np.mean(np.array([p.jerk for p in new_pts]) ** 2))),
            comfort_accel_rms=float(np.sqrt(np.mean(np.array([p.a for p in new_pts]) ** 2))),
            slack_used_max=slack_max,
            terminal_speed_mps=float(v_exec[-1]),
            terminal_s_m=float(prog_new),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if deadline_hit:
            status = SolverStatus.TIMEOUT
        elif latency_ms > rato.deadline_ms:
            status = SolverStatus.TIMEOUT
        else:
            status = last_status if last_status in {SolverStatus.SOLVED, SolverStatus.SOLVED_INACCURATE} else SolverStatus.SOLVED_INACCURATE

        if unjustified:
            return RepairResult(
                mode=RepairMode.RATO,
                success=False,
                candidate=None,
                pre_repair_id=candidate.candidate_id,
                post_repair_id=None,
                solver_trace=SolverTrace(
                    status=SolverStatus.INFEASIBLE,
                    iterations=total_qp_iters,
                    primal_residual=0.0,
                    dual_residual=0.0,
                    objective=prev_obj,
                    latency_ms=latency_ms,
                    warm_started=warm_any,
                    backend=last_backend,
                    message="unjustified_zero_speed",
                    extras={"scp_iters": scp_iters_done},
                ),
                metrics=metrics,
                reason="unjustified_zero_speed",
            )

        # Contract: any timeout enters freeze/fallback chain — never execute a late/partial solution.
        if status is SolverStatus.TIMEOUT:
            return RepairResult(
                mode=RepairMode.RATO,
                success=False,
                candidate=None,
                pre_repair_id=candidate.candidate_id,
                post_repair_id=None,
                solver_trace=SolverTrace(
                    status=SolverStatus.TIMEOUT,
                    iterations=total_qp_iters,
                    primal_residual=0.0,
                    dual_residual=0.0,
                    objective=prev_obj if math.isfinite(prev_obj) else 0.0,
                    latency_ms=latency_ms,
                    warm_started=warm_any,
                    backend=last_backend,
                    message="deadline_exceeded" if not deadline_hit else "deadline_after_partial",
                    extras={
                        "scp_iters": scp_iters_done,
                        "deadline_ms": rato.deadline_ms,
                        "partial_discarded": True,
                    },
                ),
                metrics=metrics,
                reason="timeout",
            )

        self._last_d = d_cur.copy()
        return RepairResult(
            mode=RepairMode.RATO,
            success=True,
            candidate=repaired,
            pre_repair_id=candidate.candidate_id,
            post_repair_id=post_id,
            solver_trace=SolverTrace(
                status=status,
                iterations=total_qp_iters,
                primal_residual=0.0,
                dual_residual=0.0,
                objective=prev_obj if math.isfinite(prev_obj) else 0.0,
                latency_ms=latency_ms,
                warm_started=warm_any,
                backend=last_backend,
                message="ok",
                extras={
                    "scp_iters": scp_iters_done,
                    "lateral_mod_rms": lat_mod,
                    "d_bound_m": d_bound,
                    "deadline_ms": rato.deadline_ms,
                },
            ),
            metrics=metrics,
            reason="ok",
        )

    def _solve_lateral_qp(
        self,
        *,
        d_cur: np.ndarray,
        d_ref: np.ndarray,
        s_ref: np.ndarray,
        d_bound: float,
        times: np.ndarray,
        points: Sequence[TrajectoryPoint],
        obs: ObservableSnapshot,
        n: int,
    ) -> tuple[np.ndarray | None, SolverTrace, float]:
        """One SCP iteration: QP on lateral offsets d with linearized collision half-planes."""
        cfg = self.config
        rato = cfg.rato
        # z = [d_0..d_{n-1}, slack_corridor_n, slack_coll_n]
        off_d = 0
        off_sc = n
        off_sl = off_sc + n
        n_var = off_sl + n

        P = np.zeros((n_var, n_var))
        q = np.zeros(n_var)

        # Track reference lateral (minimal intervention).
        for k in range(n):
            P[off_d + k, off_d + k] += 2.0 * rato.w_path
            q[off_d + k] += -2.0 * rato.w_path * float(d_ref[k])
            # Prefer progress: mild pull toward 0 offset (center) for free space.
            P[off_d + k, off_d + k] += 2.0 * 0.05 * rato.w_progress

        # Lateral smoothness
        for k in range(n - 1):
            c = rato.w_smooth
            P[off_d + k, off_d + k] += 2.0 * c
            P[off_d + k + 1, off_d + k + 1] += 2.0 * c
            P[off_d + k, off_d + k + 1] -= 2.0 * c
            P[off_d + k + 1, off_d + k] -= 2.0 * c

        for k in range(n):
            P[off_sc + k, off_sc + k] += 2.0 * rato.w_slack
            P[off_sl + k, off_sl + k] += 2.0 * rato.w_slack

        P += 1e-8 * np.eye(n_var)

        rows: list[np.ndarray] = []
        lows: list[float] = []
        ups: list[float] = []

        def add(row: np.ndarray, lo: float, hi: float) -> None:
            rows.append(row)
            lows.append(lo)
            ups.append(hi)

        # Corridor bounds with slack: -d_bound - sc <= d <= d_bound + sc, 0<=sc<=slack_max
        for k in range(n):
            r = np.zeros(n_var)
            r[off_d + k] = 1.0
            r[off_sc + k] = -1.0
            add(r, -float("inf"), d_bound)
            r = np.zeros(n_var)
            r[off_d + k] = 1.0
            r[off_sc + k] = 1.0
            add(r, -d_bound, float("inf"))
            r = np.zeros(n_var)
            r[off_sc + k] = 1.0
            add(r, 0.0, rato.slack_corridor_max_m)

        # Trust region around current SCP linearization point
        for k in range(n):
            r = np.zeros(n_var)
            r[off_d + k] = 1.0
            add(r, float(d_cur[k]) - rato.trust_radius_m, float(d_cur[k]) + rato.trust_radius_m)

        # Lateral step between consecutive points
        for k in range(n - 1):
            r = np.zeros(n_var)
            r[off_d + k + 1] = 1.0
            r[off_d + k] = -1.0
            add(r, -rato.max_lateral_step_m, rato.max_lateral_step_m)

        # Anchor first point near current lateral (continuity), allow modest dodge.
        r = np.zeros(n_var)
        r[off_d] = 1.0
        add(r, float(d_cur[0]) - 0.75, float(d_cur[0]) + 0.75)

        # Collision: Frenet lateral separation + Cartesian half-plane linearization.
        # Lateral form is required when the obstacle sits on-path (n·u≈0 for pure 2D form).
        frame = build_corridor_frame(obs, cfg)
        assert frame is not None
        xs_lin, ys_lin, nxs, nys = xy_from_frenet(s_ref, d_cur, frame)
        t0p = float(times[0])

        for k in range(n):
            dt = float(times[k] - t0p)
            px, py = float(xs_lin[k]), float(ys_lin[k])
            nx, ny = float(nxs[k]), float(nys[k])
            s_k = float(s_ref[k])
            for actor in obs.actors:
                if actor.lost:
                    continue
                ax = actor.x + actor.vx * dt
                ay = actor.y + actor.vy * dt
                radius = _actor_radius(actor.length_m, actor.width_m, cfg)
                a_s, a_d, _, _, _, _ = project_xy(frame, ax, ay)
                # Longitudinal proximity gate: only constrain when s-near the actor.
                long_gate = 0.5 * actor.length_m + cfg.collision_inflate_m + 0.5 * cfg.length_m + 0.5
                ds = abs(s_k - a_s)
                if ds > long_gate:
                    continue
                # Scale separation demand by longitudinal overlap (1 at actor s, 0 at gate edge).
                overlap = max(0.0, 1.0 - ds / max(long_gate, 1e-3))

                # --- Frenet lateral separation (primary for on-path static obstacles) ---
                # Match validator envelope at full overlap; taper near gate to keep QP feasible.
                d_sep_full = min(radius + 0.05, d_bound - 0.05)
                d_sep_full = max(
                    d_sep_full,
                    0.5 * actor.width_m + 0.5 * cfg.width_m + cfg.collision_inflate_m,
                )
                d_sep = d_sep_full * (0.35 + 0.65 * overlap)
                # Choose free side: away from actor lateral, biased by current offset.
                if abs(float(d_cur[k]) - a_d) > 1e-3:
                    side = 1.0 if float(d_cur[k]) >= a_d else -1.0
                else:
                    side = 1.0 if a_d <= 0.0 else -1.0
                    if abs(float(d_ref[k])) > 0.05:
                        side = 1.0 if float(d_ref[k]) > 0 else -1.0
                # side * (d - a_d) + slack >= d_sep
                r = np.zeros(n_var)
                r[off_d + k] = side
                r[off_sl + k] = 1.0
                add(r, d_sep + side * a_d, float("inf"))

                # --- Cartesian linearized distance (secondary; only when well-conditioned) ---
                # Keep radius consistent with lateral d_sep to avoid over-constraint at
                # near-coincident linearization points (dist≈0 with full circumradius).
                soft_r = d_sep
                vx, vy = px - ax, py - ay
                dist = math.hypot(vx, vy)
                if dist > 0.35:
                    ux, uy = vx / dist, vy / dist
                    n_dot_u = nx * ux + ny * uy
                    if abs(n_dot_u) > 0.35 and dist < soft_r + 2.0:
                        # dist0 + n_dot_u*(d - d0) + slack >= soft_r
                        rhs = soft_r - dist + n_dot_u * float(d_cur[k])
                        r = np.zeros(n_var)
                        r[off_d + k] = n_dot_u
                        r[off_sl + k] = 1.0
                        add(r, rhs, float("inf"))

            r = np.zeros(n_var)
            r[off_sl + k] = 1.0
            add(r, 0.0, rato.slack_collision_max_m)

        A = np.vstack(rows)
        l_vec = np.array(lows, dtype=float)
        u_vec = np.array(ups, dtype=float)
        big = 1.0e5
        l_vec = np.where(np.isfinite(l_vec), l_vec, -big)
        u_vec = np.where(np.isfinite(u_vec), u_vec, big)

        z0 = np.zeros(n_var)
        z0[:n] = d_cur
        # Seed lateral dodge toward free side of nearby actors.
        for k in range(n):
            s_k = float(s_ref[k])
            best_dodge = float(d_cur[k])
            for actor in obs.actors:
                if actor.lost:
                    continue
                dt = float(times[k] - t0p)
                ax = actor.x + actor.vx * dt
                ay = actor.y + actor.vy * dt
                a_s, a_d, _, _, _, _ = project_xy(frame, ax, ay)
                radius = _actor_radius(actor.length_m, actor.width_m, cfg)
                long_gate = 0.5 * actor.length_m + cfg.collision_inflate_m + 0.5 * cfg.length_m + 1.0
                if abs(s_k - a_s) > long_gate + 2.0:
                    continue
                side = 1.0 if a_d <= 0.0 else -1.0
                if abs(float(d_cur[k]) - a_d) > 1e-3:
                    side = 1.0 if float(d_cur[k]) >= a_d else -1.0
                rad = _actor_radius(actor.length_m, actor.width_m, cfg)
                d_sep_seed = min(d_bound * 0.95, rad + 0.1)
                target = a_d + side * d_sep_seed
                target = float(np.clip(target, -d_bound * 0.98, d_bound * 0.98))
                if abs(target - a_d) > abs(best_dodge - a_d):
                    best_dodge = target
            z0[k] = best_dodge

        if rato.warm_start:
            self.solver.set_warm_start(z0)
        problem = QPProblem(P=P, q=q, A=A, l=l_vec, u=u_vec)
        z, trace = self.solver.solve(problem, warm_start=rato.warm_start)
        slack_i = 0.0
        if z is not None:
            slack_i = float(
                max(
                    np.max(np.abs(z[off_sc:off_sc + n])),
                    np.max(np.abs(z[off_sl:off_sl + n])),
                    0.0,
                )
            )
        return z, trace, slack_i

    def _seed_lateral(
        self,
        *,
        d_ref: np.ndarray,
        s_ref: np.ndarray,
        times: np.ndarray,
        obs: ObservableSnapshot,
        frame: object,
        d_bound: float,
    ) -> np.ndarray:
        """Initial lateral profile: reference + free-side dodge near actors."""
        cfg = self.config
        d = d_ref.copy()
        t0p = float(times[0])
        n = len(d)
        for k in range(n):
            s_k = float(s_ref[k])
            best = float(d[k])
            for actor in obs.actors:
                if actor.lost:
                    continue
                dt = float(times[k] - t0p)
                ax = actor.x + actor.vx * dt
                ay = actor.y + actor.vy * dt
                a_s, a_d, _, _, _, _ = project_xy(frame, ax, ay)  # type: ignore[arg-type]
                long_gate = 0.5 * actor.length_m + cfg.collision_inflate_m + 0.5 * cfg.length_m + 2.0
                if abs(s_k - a_s) > long_gate + 3.0:
                    continue
                rad = _actor_radius(actor.length_m, actor.width_m, cfg)
                side = 1.0 if a_d <= 0.0 else -1.0
                if abs(float(d_ref[k]) - a_d) > 1e-3:
                    side = 1.0 if float(d_ref[k]) >= a_d else -1.0
                target = a_d + side * min(d_bound * 0.95, rad + 0.15)
                # Smooth ramp weight by longitudinal distance to actor.
                w = max(0.0, 1.0 - abs(s_k - a_s) / max(long_gate + 3.0, 1.0))
                blended = (1.0 - w) * float(d_ref[k]) + w * target
                if abs(blended - a_d) > abs(best - a_d):
                    best = blended
            d[k] = float(np.clip(best, -d_bound, d_bound))
        # One-pass lateral smoothing for trackability.
        for _ in range(2):
            d2 = d.copy()
            for k in range(1, n - 1):
                d2[k] = 0.25 * d[k - 1] + 0.5 * d[k] + 0.25 * d[k + 1]
            d = np.clip(d2, -d_bound, d_bound)
        return d

    def _nudge_clear(
        self,
        d_cur: np.ndarray,
        s_ref: np.ndarray,
        times: np.ndarray,
        obs: ObservableSnapshot,
        frame: object,
        d_bound: float,
    ) -> np.ndarray:
        """Small lateral push if residual collision remains after SCP."""
        cfg = self.config
        d = d_cur.copy()
        xs, ys, _, _ = xy_from_frenet(s_ref, d, frame)  # type: ignore[arg-type]
        t0p = float(times[0])
        for _ in range(3):
            worst = 0.0
            worst_k = -1
            worst_side = 1.0
            for k in range(len(d)):
                dt = float(times[k] - t0p)
                px, py = float(xs[k]), float(ys[k])
                for actor in obs.actors:
                    if actor.lost:
                        continue
                    ax = actor.x + actor.vx * dt
                    ay = actor.y + actor.vy * dt
                    radius = _actor_radius(actor.length_m, actor.width_m, cfg)
                    margin = math.hypot(px - ax, py - ay) - radius
                    if margin < worst:
                        worst = margin
                        worst_k = k
                        a_s, a_d, _, _, _, _ = project_xy(frame, ax, ay)  # type: ignore[arg-type]
                        worst_side = 1.0 if float(d[k]) >= a_d else -1.0
            if worst >= 0.02 or worst_k < 0:
                break
            # Push neighborhood around worst index (scale with residual depth).
            push = min(0.45, 0.20 + abs(worst))
            for k in range(max(0, worst_k - 3), min(len(d), worst_k + 4)):
                d[k] = float(np.clip(d[k] + worst_side * push, -d_bound, d_bound))
            xs, ys, _, _ = xy_from_frenet(s_ref, d, frame)  # type: ignore[arg-type]
        return d

    def _reprofile_speed(
        self,
        *,
        times: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        yaw: np.ndarray,
        kappa: np.ndarray,
        v_seed: np.ndarray,
        obs: ObservableSnapshot,
        ego_v: float,
        ego_a: float,
    ) -> list[TrajectoryPoint]:
        """Simple feasible speed profile on fixed repaired path (not full QP)."""
        cfg = self.config
        n = len(times)
        dts = np.diff(times)
        dt_med = float(np.median(dts[dts > 1e-6])) if np.any(dts > 1e-6) else cfg.control_period_s
        dt_med = max(dt_med, 1e-3)

        # Segment lengths along new path — speeds must cover geometry for trackability.
        seg = np.zeros(n)
        for i in range(1, n):
            seg[i] = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])

        v_max = cfg.max_speed_mps
        if cfg.enforce_speed_limit and obs.speed_limit_mps is not None:
            v_max = min(v_max, obs.speed_limit_mps + cfg.speed_limit_margin_mps)

        # Curvature speed limit
        v_lim = np.full(n, v_max)
        for k in range(n):
            if abs(kappa[k]) > 1e-4:
                v_curv = math.sqrt(max(0.1, cfg.max_lateral_accel_mps2 / max(abs(kappa[k]), 1e-6)))
                v_lim[k] = min(v_lim[k], v_curv)

        # Geometry-consistent speed: cover path step within trackability envelope.
        # max_step ≈ v_avg * dt + 0.5*a_max*dt^2 + 1.0  →  require v_avg large enough.
        v = np.zeros(n)
        v[0] = min(max(ego_v, float(v_seed[0])), float(v_lim[0]))
        for k in range(1, n):
            dt = max(float(times[k] - times[k - 1]), 1e-3)
            # Need (v[k-1]+v[k])/2 * dt + 0.5*a_max*dt^2 + 1.0 >= seg[k]
            need = 2.0 * max(0.0, seg[k] - 1.0 - 0.5 * cfg.max_accel_mps2 * dt * dt) / dt - v[k - 1]
            # Extra margin 0.25 m/s for numerical comfort.
            v_geom = max(0.0, need + 0.25, float(v_seed[k]) * 0.6)
            v_coast = v[k - 1] + cfg.max_accel_mps2 * dt
            v[k] = min(float(v_lim[k]), max(v_geom, min(v_coast, float(v_lim[k]))))
            if v[k] < v_geom and v_geom <= float(v_lim[k]):
                v[k] = v_geom

        # Forward re-pass: enforce geometry with higher priority than comfort.
        for k in range(1, n):
            dt = max(float(times[k] - times[k - 1]), 1e-3)
            max_step = (v[k - 1] + v[k]) * 0.5 * dt + 0.5 * cfg.max_accel_mps2 * dt * dt + 1.0
            if seg[k] > max_step:
                # Solve for v[k]: (v0+v1)/2*dt + c = seg → v1 = 2*(seg-c)/dt - v0
                c = 0.5 * cfg.max_accel_mps2 * dt * dt + 1.0
                v_need = 2.0 * max(0.0, seg[k] - c) / dt - v[k - 1] + 0.5
                v[k] = min(float(v_lim[k]), max(v[k], v_need, 0.0))
                # If still impossible, raise v[k-1] as well within lim.
                max_step2 = (v[k - 1] + v[k]) * 0.5 * dt + c
                if seg[k] > max_step2:
                    v_need0 = 2.0 * max(0.0, seg[k] - c) / dt - v[k] + 0.5
                    v[k - 1] = min(float(v_lim[k - 1]), max(v[k - 1], v_need0, 0.0))

        # Backward pass: mild decel smoothing, never drop below geometry need.
        for k in range(n - 2, -1, -1):
            dt = max(float(times[k + 1] - times[k]), 1e-3)
            v_dec = v[k + 1] + cfg.max_decel_mps2 * dt
            c = 0.5 * cfg.max_accel_mps2 * dt * dt + 1.0
            v_geom = 2.0 * max(0.0, seg[k + 1] - c) / dt - v[k + 1] + 0.25
            v[k] = min(float(v_lim[k]), max(min(v[k], v_dec), v_geom, 0.0))
        # Red light approach (same window as validator)
        if cfg.enforce_red_light_stop:
            for light in obs.traffic_lights:
                if light.state != "red":
                    continue
                if light.controls_ego_lane is False:
                    continue
                distance = light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m
                if distance > cfg.red_light_stop_distance_m:
                    continue
                for k in range(n):
                    if times[k] - times[0] <= 2.0 + 1e-9:
                        v[k] = min(v[k], cfg.red_light_max_approach_speed_mps)

        # If path steps still exceed trackability, shorten lateral geometry by scaling d→center.
        # (Applied by recomputing positions is not available; instead cap kappa and smooth yaw.)
        yaw = np.unwrap(yaw.copy())
        for k in range(1, n):
            dt = max(float(times[k] - times[k - 1]), 1e-3)
            max_dyaw = abs(v[k - 1]) * cfg.max_curvature_per_m * dt + 0.35
            dyaw = yaw[k] - yaw[k - 1]
            if abs(dyaw) > max_dyaw:
                yaw[k] = yaw[k - 1] + math.copysign(max_dyaw, dyaw)
        kappa = np.clip(kappa, -cfg.max_curvature_per_m, cfg.max_curvature_per_m)

        pts: list[TrajectoryPoint] = []
        for k in range(n):
            if k == 0:
                a = float(np.clip((v[0] - ego_v) / dt_med, -cfg.max_decel_mps2, cfg.max_accel_mps2))
                jerk = float(np.clip((a - ego_a) / dt_med, -cfg.max_jerk_mps3, cfg.max_jerk_mps3))
            else:
                dt = max(float(times[k] - times[k - 1]), 1e-3)
                a = float(np.clip((v[k] - v[k - 1]) / dt, -cfg.max_decel_mps2, cfg.max_accel_mps2))
                a_prev = pts[-1].a
                jerk = float(np.clip((a - a_prev) / dt, -cfg.max_jerk_mps3, cfg.max_jerk_mps3))
            # Lateral accel check: |kappa| v^2 with tiny margin under hard limit.
            v_k = float(max(0.0, v[k]))
            kap = float(np.clip(kappa[k], -cfg.max_curvature_per_m, cfg.max_curvature_per_m))
            lat_lim = cfg.max_lateral_accel_mps2 * 0.98
            if abs(kap) * v_k * v_k > lat_lim:
                v_k = math.sqrt(lat_lim / max(abs(kap), 1e-6))
                v[k] = v_k
                if k > 0:
                    dt = max(float(times[k] - times[k - 1]), 1e-3)
                    a = float(np.clip((v_k - pts[-1].v) / dt, -cfg.max_decel_mps2, cfg.max_accel_mps2))
                    jerk = float(np.clip((a - pts[-1].a) / dt, -cfg.max_jerk_mps3, cfg.max_jerk_mps3))
            # Also clamp kappa if speed cannot be reduced further.
            if abs(kap) * v_k * v_k > lat_lim and v_k < 0.5:
                kap = math.copysign(lat_lim / max(v_k * v_k, 1e-6), kap) if v_k > 1e-3 else 0.0
            pts.append(
                TrajectoryPoint(
                    t=float(times[k]),
                    x=float(xs[k]),
                    y=float(ys[k]),
                    yaw=float(yaw[k]),
                    kappa=float(kap),
                    v=v_k,
                    a=a,
                    jerk=jerk,
                )
            )
        return pts

    def _min_collision_margin(
        self,
        points: Sequence[TrajectoryPoint],
        obs: ObservableSnapshot,
    ) -> float:
        cfg = self.config
        if not obs.actors:
            return 1.0
        worst = float("inf")
        t0 = points[0].t
        for p in points:
            dt = p.t - t0
            for actor in obs.actors:
                if actor.lost:
                    continue
                ax = actor.x + actor.vx * dt
                ay = actor.y + actor.vy * dt
                radius = _actor_radius(actor.length_m, actor.width_m, cfg)
                worst = min(worst, math.hypot(p.x - ax, p.y - ay) - radius)
        return float(worst if math.isfinite(worst) else 1.0)

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
            mode=RepairMode.RATO,
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
