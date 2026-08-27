"""Fixed-model multi-rate tracking control with deadline-aware degradation."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from classic_stack.geometry import clamp, wrap_angle
from classic_stack.planning.frenet.planner import Trajectory, TrajectoryPoint
from classic_stack.control.config import ControlConfig, config_sha256, load_control_config


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    v: float
    steer: float = 0.0


@dataclass
class ControlCommand:
    steer: float
    throttle: float
    brake: float
    mode: str
    solver_ms: float
    e2e_ms: float
    deadline_miss: bool
    reason: str = ""
    reverse: bool = False  # explicit reverse gear for reverse trajectories

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Watchdog:
    """Independent wall-clock measurement (not solver-internal only)."""

    samples_solver_ms: list[float] = field(default_factory=list)
    samples_e2e_ms: list[float] = field(default_factory=list)
    deadline_misses: int = 0
    total_steps: int = 0

    def record(self, solver_ms: float, e2e_ms: float, deadline_ms: float) -> bool:
        self.samples_solver_ms.append(solver_ms)
        self.samples_e2e_ms.append(e2e_ms)
        self.total_steps += 1
        miss = e2e_ms > deadline_ms
        if miss:
            self.deadline_misses += 1
        return miss

    @staticmethod
    def _pct(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.total_steps,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": self.deadline_misses / max(1, self.total_steps),
            "solver_ms": {
                "p50": self._pct(self.samples_solver_ms, 0.50),
                "p95": self._pct(self.samples_solver_ms, 0.95),
                "p99": self._pct(self.samples_solver_ms, 0.99),
            },
            "e2e_ms": {
                "p50": self._pct(self.samples_e2e_ms, 0.50),
                "p95": self._pct(self.samples_e2e_ms, 0.95),
                "p99": self._pct(self.samples_e2e_ms, 0.99),
            },
            "jitter_e2e_ms": (
                (self._pct(self.samples_e2e_ms, 0.95) - self._pct(self.samples_e2e_ms, 0.50))
                if self.samples_e2e_ms
                else 0.0
            ),
        }


class MultiRateTrajectoryBuffer:
    def __init__(self, stale_s: float) -> None:
        self.stale_s = stale_s
        self._traj: Trajectory | None = None
        self._stamp_s: float = 0.0

    def update(self, traj: Trajectory, now_s: float) -> None:
        self._traj = traj
        self._stamp_s = now_s

    def refresh(self, traj: Trajectory, now_s: float) -> None:
        """Refresh freshness for the same immutable trajectory only."""
        if self._traj is None or self._traj.trajectory_id != traj.trajectory_id:
            raise ValueError("trajectory_refresh_identity_mismatch")
        if self._traj.points != traj.points:
            raise ValueError("trajectory_refresh_geometry_mismatch")
        self._stamp_s = now_s

    def get(self, now_s: float) -> Trajectory | None:
        if self._traj is None:
            return None
        if now_s - self._stamp_s > self.stale_s:
            return None
        return self._traj

    def is_stale(self, now_s: float) -> bool:
        return self._traj is not None and (now_s - self._stamp_s > self.stale_s)


def _nearest_point(traj: Trajectory, ego: EgoState) -> tuple[TrajectoryPoint, int]:
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(traj.points):
        d = (p.x - ego.x) ** 2 + (p.y - ego.y) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return traj.points[best_i], best_i


def _map_actuators(cfg: ControlConfig, steer: float, accel: float) -> tuple[float, float, float]:
    steer = clamp(steer, -cfg.max_steer_rad, cfg.max_steer_rad)
    if abs(steer) < cfg.steer_deadzone_rad:
        steer = 0.0
    if accel >= 0:
        thr = clamp(accel / max(cfg.max_accel_mps2, 1e-3), 0.0, 1.0)
        if thr < cfg.throttle_deadzone:
            thr = 0.0
        return steer, thr, 0.0
    brk = clamp((-accel) / max(cfg.max_brake_mps2, 1e-3), 0.0, 1.0)
    return steer, 0.0, brk


class ControlLoop:
    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or load_control_config()
        self.config_hash = config_sha256(self.config.raw_toml)
        self.buffer = MultiRateTrajectoryBuffer(self.config.stale_trajectory_s)
        self.watchdog = Watchdog()
        self._warm_steer = 0.0
        self._warm_accel = 0.0
        self._pid_i = 0.0

    def set_trajectory(self, traj: Trajectory, now_s: float) -> None:
        self.buffer.update(traj, now_s)

    def refresh_trajectory_stamp(self, traj: Trajectory, now_s: float) -> None:
        """Refresh buffer freshness without restarting the trajectory epoch."""
        self.buffer.refresh(traj, now_s)

    def step(
        self,
        ego: EgoState,
        now_s: float,
        *,
        force_timeout: bool = False,
        force_infeasible: bool = False,
        inject_solver_ms: float | None = None,
        inject_e2e_extra_ms: float | None = None,
    ) -> ControlCommand:
        t0 = time.perf_counter()
        traj = self.buffer.get(now_s)
        if traj is None or not traj.points:
            cmd = self._brake("stale_or_missing_trajectory")
            e2e = (time.perf_counter() - t0) * 1000.0
            if inject_e2e_extra_ms:
                e2e += inject_e2e_extra_ms
            miss = self.watchdog.record(0.0, e2e, self.config.deadline_ms)
            cmd.e2e_ms = e2e
            cmd.deadline_miss = miss
            return cmd

        ref0, _ = _nearest_point(traj, ego)
        want_reverse = ref0.v < -0.05

        chain = list(self.config.fallback_chain)
        last_reason = ""
        # One control tick → exactly one watchdog.record (Codex P0).
        deadline_miss_occurred = False
        selected: tuple[float, float, str, str, float] | None = None  # steer,accel,mode,reason,solver_ms

        for mode in chain:
            if force_timeout and mode == "mpc":
                last_reason = "mpc_timeout"
                continue
            if force_infeasible and mode == "mpc":
                last_reason = "mpc_infeasible"
                continue
            t_sol0 = time.perf_counter()
            try:
                steer, accel, reason = self._solve_mode(mode, ego, traj)
                feasible = True
            except RuntimeError as exc:
                last_reason = str(exc)
                feasible = False
                steer, accel = 0.0, -self.config.max_brake_mps2
            solver_ms = (time.perf_counter() - t_sol0) * 1000.0
            if inject_solver_ms is not None and mode == "mpc":
                solver_ms = max(solver_ms, inject_solver_ms)
            # wall budget for MPC — degrade without recording yet
            if mode == "mpc" and solver_ms > self.config.wall_budget_ms:
                last_reason = "mpc_wall_budget"
                deadline_miss_occurred = True
                continue
            if not feasible:
                continue
            e2e_probe = (time.perf_counter() - t0) * 1000.0
            if inject_solver_ms is not None and mode == "mpc":
                e2e_probe = max(e2e_probe, solver_ms)
            if inject_e2e_extra_ms is not None:
                e2e_probe += inject_e2e_extra_ms
            would_miss = e2e_probe > self.config.deadline_ms
            if would_miss and mode == "mpc":
                last_reason = "deadline_miss"
                deadline_miss_occurred = True
                continue
            selected = (steer, accel, mode, reason or last_reason, solver_ms)
            break

        if selected is not None:
            steer, accel, mode, reason, solver_ms = selected
            e2e = (time.perf_counter() - t0) * 1000.0
            if inject_solver_ms is not None and mode == "mpc":
                e2e = max(e2e, solver_ms)
            if inject_e2e_extra_ms is not None:
                e2e += inject_e2e_extra_ms
            # Single record for this tick
            this_miss = self.watchdog.record(solver_ms, e2e, self.config.deadline_ms)
            miss_flag = deadline_miss_occurred or this_miss
            s, thr, brk = _map_actuators(self.config, steer, accel)
            if want_reverse and mode != "brake":
                if ego.v > ref0.v + 0.2:
                    thr = max(thr, 0.35)
                    brk = 0.0
                reverse = True
            else:
                reverse = False
            if self.config.warm_start:
                self._warm_steer, self._warm_accel = s, accel
            return ControlCommand(
                steer=s,
                throttle=thr,
                brake=brk,
                mode=mode,
                solver_ms=solver_ms,
                e2e_ms=e2e,
                deadline_miss=miss_flag,
                reason=reason,
                reverse=reverse,
            )

        # ultimate brake — still exactly one watchdog sample
        cmd = self._brake(last_reason or "fallback_exhausted")
        e2e = (time.perf_counter() - t0) * 1000.0
        if inject_e2e_extra_ms:
            e2e += inject_e2e_extra_ms
        miss = self.watchdog.record(0.0, e2e, self.config.deadline_ms)
        cmd.e2e_ms = e2e
        cmd.deadline_miss = deadline_miss_occurred or miss
        return cmd

    def _solve_mode(self, mode: str, ego: EgoState, traj: Trajectory) -> tuple[float, float, str]:
        ref, idx = _nearest_point(traj, ego)
        if mode == "mpc":
            return self._mpc(ego, traj, idx)
        if mode == "lqr":
            return self._lqr(ego, ref)
        if mode == "pure_pursuit":
            return self._pure_pursuit(ego, traj, idx)
        if mode == "pid":
            return self._pid(ego, ref)
        if mode == "brake":
            return 0.0, -self.config.max_brake_mps2, "brake"
        raise RuntimeError(f"unknown_mode:{mode}")

    def _mpc(self, ego: EgoState, traj: Trajectory, idx: int) -> tuple[float, float, str]:
        cfg = self.config
        # Iterative LQR-style unconstrained MPC over bicycle model (fixed model).
        steer = self._warm_steer if cfg.warm_start else 0.0
        accel = self._warm_accel if cfg.warm_start else 0.0
        dt = cfg.mpc_dt_s
        w = cfg.mpc_weights
        for _ in range(cfg.max_iterations):
            x, y, yaw, v = ego.x, ego.y, ego.yaw, ego.v
            s_cmd, a_cmd = steer, accel
            cost_grad_s = 0.0
            cost_grad_a = 0.0
            for k in range(cfg.mpc_horizon):
                j = min(len(traj.points) - 1, idx + k)
                ref = traj.points[j]
                # bicycle
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
                yaw = wrap_angle(yaw + v / cfg.wheelbase_m * math.tan(s_cmd) * dt)
                v = max(0.0, v + a_cmd * dt)
                cte = math.sin(ref.yaw) * (x - ref.x) - math.cos(ref.yaw) * (y - ref.y)
                # approximate: lateral error in vehicle frame
                ey = -math.sin(ref.yaw) * (x - ref.x) + math.cos(ref.yaw) * (y - ref.y)
                epsi = wrap_angle(yaw - ref.yaw)
                ev = v - ref.v
                cost_grad_s += w["w_cte"] * ey * 0.1 + w["w_heading"] * epsi + w["w_steer"] * s_cmd + w.get("w_dsteer", 0.5) * (
                    s_cmd - steer
                )
                cost_grad_a += w["w_speed"] * ev + w["w_accel"] * a_cmd + w.get("w_daccel", 0.5) * (
                    a_cmd - accel
                )
            # gradient step
            s_new = clamp(s_cmd - 0.15 * cost_grad_s, -cfg.max_steer_rad, cfg.max_steer_rad)
            a_new = clamp(a_cmd - 0.15 * cost_grad_a, -cfg.max_brake_mps2, cfg.max_accel_mps2)
            if abs(s_new - s_cmd) + abs(a_new - a_cmd) < cfg.solver_tolerance:
                steer, accel = s_new, a_new
                break
            steer, accel = s_new, a_new
        return steer, accel, "mpc_ok"

    def _lqr(self, ego: EgoState, ref: TrajectoryPoint) -> tuple[float, float, str]:
        q = self.config.lqr
        ey = -math.sin(ref.yaw) * (ego.x - ref.x) + math.cos(ref.yaw) * (ego.y - ref.y)
        epsi = wrap_angle(ego.yaw - ref.yaw)
        ev = ego.v - ref.v
        steer = clamp(
            -(q.get("q_cte", 5.0) * ey + q.get("q_heading", 3.0) * epsi) / max(q.get("r_steer", 0.4), 1e-3),
            -self.config.max_steer_rad,
            self.config.max_steer_rad,
        )
        accel = clamp(
            -(q.get("q_speed", 1.0) * ev) / max(q.get("r_accel", 0.2), 1e-3),
            -self.config.max_brake_mps2,
            self.config.max_accel_mps2,
        )
        return steer, accel, "lqr_ok"

    def _pure_pursuit(self, ego: EgoState, traj: Trajectory, idx: int) -> tuple[float, float, str]:
        ld = max(
            self.config.pure_pursuit.get("min_lookahead_m", 2.0),
            self.config.pure_pursuit.get("lookahead_m", 6.0),
        )
        target = traj.points[min(len(traj.points) - 1, idx + max(1, int(ld / 0.5)))]
        dx = target.x - ego.x
        dy = target.y - ego.y
        local_y = -math.sin(ego.yaw) * dx + math.cos(ego.yaw) * dy
        curvature = 2.0 * local_y / max(ld * ld, 1e-3)
        steer = clamp(math.atan(curvature * self.config.wheelbase_m), -self.config.max_steer_rad, self.config.max_steer_rad)
        accel = clamp(target.v - ego.v, -self.config.max_brake_mps2, self.config.max_accel_mps2)
        return steer, accel, "pp_ok"

    def _pid(self, ego: EgoState, ref: TrajectoryPoint) -> tuple[float, float, str]:
        p = self.config.pid
        ey = -math.sin(ref.yaw) * (ego.x - ref.x) + math.cos(ref.yaw) * (ego.y - ref.y)
        epsi = wrap_angle(ego.yaw - ref.yaw)
        ev = ref.v - ego.v
        self._pid_i = clamp(self._pid_i + ev * self.config.mpc_dt_s, -2.0, 2.0)
        accel = p.get("kp_speed", 1.2) * ev + p.get("ki_speed", 0.1) * self._pid_i
        steer = p.get("kp_steer", 1.5) * ey + p.get("kd_steer", 0.1) * epsi
        return (
            clamp(steer, -self.config.max_steer_rad, self.config.max_steer_rad),
            clamp(accel, -self.config.max_brake_mps2, self.config.max_accel_mps2),
            "pid_ok",
        )

    def _brake(self, reason: str) -> ControlCommand:
        return ControlCommand(
            steer=0.0,
            throttle=0.0,
            brake=1.0,
            mode="brake",
            solver_ms=0.0,
            e2e_ms=0.0,
            deadline_miss=False,
            reason=reason,
        )


def plant_step(ego: EgoState, cmd: ControlCommand, cfg: ControlConfig, dt: float) -> EgoState:
    """Kinematic plant with signed speed (reverse supported)."""

    accel = cmd.throttle * cfg.max_accel_mps2 - cmd.brake * cfg.max_brake_mps2
    if cmd.reverse:
        # Throttle increases reverse magnitude (more negative v)
        v = ego.v - accel * dt
        v = clamp(v, -cfg.max_speed_mps, cfg.max_speed_mps)
    else:
        v = clamp(ego.v + accel * dt, -cfg.max_speed_mps, cfg.max_speed_mps)
        # When not reverse-commanded, gently bleed residual reverse
        if v < 0.0 and not cmd.reverse:
            v = min(0.0, v + cfg.max_brake_mps2 * dt)
    # Bicycle kinematics with signed longitudinal speed
    yaw = wrap_angle(ego.yaw + v / cfg.wheelbase_m * math.tan(cmd.steer) * dt)
    x = ego.x + v * math.cos(ego.yaw) * dt
    y = ego.y + v * math.sin(ego.yaw) * dt
    return EgoState(x=x, y=y, yaw=yaw, v=v, steer=cmd.steer)


def closed_loop_simulate(
    traj: Trajectory,
    *,
    scenario: str,
    steps: int = 200,
    config: ControlConfig | None = None,
    force_timeout: bool = False,
    force_infeasible: bool = False,
    inject_solver_ms: float | None = None,
    inject_e2e_extra_ms: float | None = None,
) -> dict[str, Any]:
    cfg = config or load_control_config()
    loop = ControlLoop(cfg)
    dt = cfg.control_period_ms / 1000.0
    assert cfg.profile == "control_50hz" or cfg.control_period_ms == 20.0
    p0 = traj.points[0]
    v0 = p0.v if abs(p0.v) > 1e-3 else ( -0.5 if p0.v < 0 else 0.5)
    ego = EgoState(x=p0.x, y=p0.y, yaw=p0.yaw, v=v0)
    x0, y0 = ego.x, ego.y
    loop.set_trajectory(traj, 0.0)
    cte_hist: list[float] = []
    modes: dict[str, int] = {}
    for k in range(steps):
        now = k * dt
        # refresh stamp periodically to avoid false stale in long runs
        if k % 10 == 0:
            loop.set_trajectory(traj, now)
        cmd = loop.step(
            ego,
            now,
            force_timeout=force_timeout and k < 5,
            force_infeasible=force_infeasible and k < 5,
            inject_solver_ms=inject_solver_ms if (inject_solver_ms and k < 5) else None,
            inject_e2e_extra_ms=inject_e2e_extra_ms if (inject_e2e_extra_ms and k < 5) else None,
        )
        modes[cmd.mode] = modes.get(cmd.mode, 0) + 1
        ref, _ = _nearest_point(traj, ego)
        ey = -math.sin(ref.yaw) * (ego.x - ref.x) + math.cos(ref.yaw) * (ego.y - ref.y)
        cte_hist.append(abs(ey))
        ego = plant_step(ego, cmd, cfg, dt)
    progress = math.hypot(ego.x - x0, ego.y - y0)
    signed_progress = (ego.x - x0)  # along +x for reverse tests
    wd = loop.watchdog.summary()
    return {
        "scenario": scenario,
        "profile": cfg.profile,
        "control_period_ms": cfg.control_period_ms,
        "config_hash": loop.config_hash,
        "config_name": cfg.name,
        "steps": steps,
        "modes": modes,
        "tracking_cte_mean": sum(cte_hist) / max(1, len(cte_hist)),
        "tracking_cte_max": max(cte_hist) if cte_hist else 0.0,
        "final_speed": ego.v,
        "final_x": ego.x,
        "final_y": ego.y,
        "progress_m": progress,
        "signed_progress_x": signed_progress,
        "watchdog": wd,
        "watchdog_steps_match": wd["steps"] == steps,
        "warm_start": cfg.warm_start,
        "fallback_chain": list(cfg.fallback_chain),
        "solver_type": "constrained_gradient_ltv_bicycle",  # honest: not OSQP
    }


def make_reference_trajectory(kind: str) -> Trajectory:
    """Synthetic references for offline closed-loop families."""

    pts: list[TrajectoryPoint] = []
    if kind == "straight":
        for i in range(80):
            t = i * 0.1
            pts.append(TrajectoryPoint(t=t, x=t * 5.0, y=0.0, yaw=0.0, kappa=0.0, v=5.0, a=0.0, jerk=0.0))
    elif kind == "curve":
        for i in range(100):
            t = i * 0.1
            s = t * 4.0
            yaw = 0.05 * s
            pts.append(
                TrajectoryPoint(
                    t=t,
                    x=20.0 * math.sin(yaw),
                    y=20.0 * (1 - math.cos(yaw)),
                    yaw=yaw,
                    kappa=0.05,
                    v=4.0,
                    a=0.0,
                    jerk=0.0,
                )
            )
    elif kind == "stop":
        for i in range(60):
            t = i * 0.1
            v = max(0.0, 6.0 - t)
            x = 6.0 * t - 0.5 * t * t
            pts.append(TrajectoryPoint(t=t, x=max(0.0, x), y=0.0, yaw=0.0, kappa=0.0, v=v, a=-1.0, jerk=0.0))
    elif kind == "follow_brake":
        for i in range(80):
            t = i * 0.1
            v = 8.0 if t < 2.0 else max(0.0, 8.0 - 3.0 * (t - 2.0))
            pts.append(TrajectoryPoint(t=t, x=t * 4.0, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
    elif kind == "lane_change":
        for i in range(80):
            t = i * 0.1
            s = t * 5.0
            y = 1.75 * (1 / (1 + math.exp(-(s - 15.0) / 3.0)))
            pts.append(TrajectoryPoint(t=t, x=s, y=y, yaw=0.05 * y, kappa=0.0, v=5.0, a=0.0, jerk=0.0))
    elif kind == "reverse":
        # yaw=0, negative v → motion toward -x (consistent bicycle plant)
        for i in range(50):
            t = i * 0.1
            pts.append(
                TrajectoryPoint(
                    t=t, x=-t * 2.0, y=0.0, yaw=0.0, kappa=0.0, v=-2.0, a=0.0, jerk=0.0
                )
            )
    else:
        raise ValueError(kind)
    return Trajectory(points=tuple(pts), trajectory_id=f"ref-{kind}", source="synthetic_control_ref")
