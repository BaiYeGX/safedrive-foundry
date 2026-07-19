"""Constrained lateral MPC for a spatial VLA path.

This controller is intentionally G3-specific.  It does not alter the frozen G1
baseline and it never consumes map lane geometry.  A small linear bicycle-model
QP optimizes steering rate over a real prediction horizon; longitudinal speed is
derived from the VLA speed head and capped by path curvature.
"""

from __future__ import annotations

import math
import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np

from driving_vla.runtime.path_manager import EgoPose, SpatialPath, wrap_angle
from safety_kernel.repair.qp_solver import LongitudinalQPSolver, QPProblem
from safety_kernel.repair.types import SolverStatus


@dataclass(frozen=True)
class VLAMPCConfig:
    control_dt_s: float = 0.05
    prediction_dt_s: float = 0.10
    horizon: int = 20
    wheelbase_m: float = 2.70
    max_steer_rad: float = 0.60
    max_steer_rate_rps: float = 0.35
    max_steer_accel_rps2: float = 1.50
    max_lateral_accel_mps2: float = 1.00
    curve_limit_quantile: float = 0.90
    max_speed_mps: float = 2.50
    max_accel_mps2: float = 1.50
    max_brake_mps2: float = 3.00
    path_end_margin_m: float = 2.00
    # Freshness envelope (do NOT hard-cut to 0 at the first hard threshold):
    #   age <= soft  → full speed cap
    #   soft < age < hard → ramp max → crawl
    #   hard <= age < zero → crawl → 0
    #   age >= zero → 0
    # Avoids intersection reject storms parking the car before reanchor can land.
    path_stale_soft_s: float = 1.00
    path_stale_hard_s: float = 2.50
    path_stale_zero_s: float = 5.00
    path_stale_crawl_mps: float = 1.00
    min_linearization_speed_mps: float = 0.60
    # How far above measured speed the lateral model may look ahead (m/s).
    # Must NOT jump to path.target (e.g. 6 m/s) while the car crawls at ~2 m/s —
    # that over-reads path curvature and multiplies steer sign flips.
    linearization_speed_slack_mps: float = 0.50
    weight_lateral: float = 6.0
    weight_heading: float = 4.0
    weight_steer: float = 1.0
    weight_steer_rate: float = 2.0
    weight_steer_accel: float = 10.0
    terminal_scale: float = 3.0
    longitudinal_kp: float = 1.0
    longitudinal_ki: float = 0.10
    solver_deadline_ms: float = 30.0


@dataclass(frozen=True)
class VLAMPCCommand:
    steer_rad: float
    steer_rate_rps: float
    accel_mps2: float
    target_speed_mps: float
    lateral_error_m: float
    heading_error_rad: float
    reference_curvature: float
    curve_speed_limit_mps: float
    horizon_speed_limit_mps: float
    freshness_speed_limit_mps: float
    path_age_s: float
    # fresh | soft_ramp | crawl | hard_stop
    freshness_regime: str
    mode: str
    solver_status: str
    solver_backend: str
    solver_ms: float


def _difference_matrix(n: int) -> np.ndarray:
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        d[i, i] = 1.0
        if i > 0:
            d[i, i - 1] = -1.0
    return d


class ConstrainedVLAMPC:
    """Linearized path-coordinate MPC with hard steering constraints."""

    def __init__(self, config: VLAMPCConfig | None = None) -> None:
        self.config = config or VLAMPCConfig()
        if not (
            0.0 <= self.config.path_stale_soft_s
            <= self.config.path_stale_hard_s
            <= self.config.path_stale_zero_s
        ):
            raise ValueError(
                "path freshness thresholds must satisfy 0 <= soft <= hard <= zero"
            )
        if self.config.path_stale_crawl_mps < 0.0:
            raise ValueError("path_stale_crawl_mps must be non-negative")
        self._solver = LongitudinalQPSolver(
            deadline_ms=self.config.solver_deadline_ms,
            max_iter=4000,
            abs_tol=1e-4,
            rel_tol=1e-4,
            prefer_osqp=True,
        )
        self._steer_rad = 0.0
        self._steer_rate_rps = 0.0
        self._speed_integral = 0.0
        self._progress_s = 0.0
        self._path_stamp_s: float | None = None

    @property
    def steer_rad(self) -> float:
        return self._steer_rad

    def reset(self, *, steer_rad: float = 0.0) -> None:
        self._steer_rad = float(steer_rad)
        self._steer_rate_rps = 0.0
        self._speed_integral = 0.0
        self._progress_s = 0.0
        self._path_stamp_s = None
        self._solver.clear_warm_start()

    @staticmethod
    def _tracking_errors(path: SpatialPath, ego: EgoPose, progress_s: float) -> tuple[float, float, float, float]:
        s0 = path.project_s(ego.x, ego.y, hint_s=progress_s)
        rx, ry, ryaw, rk = path.sample(s0)
        yaw_ref = float(ryaw[0])
        dx, dy = ego.x - float(rx[0]), ego.y - float(ry[0])
        lateral = -math.sin(yaw_ref) * dx + math.cos(yaw_ref) * dy
        heading = wrap_angle(ego.yaw - yaw_ref)
        return s0, lateral, heading, float(rk[0])

    def _reference(self, path: SpatialPath, s0: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        distances = np.arange(1, cfg.horizon + 1, dtype=float) * max(
            speed, cfg.min_linearization_speed_mps
        ) * cfg.prediction_dt_s
        _x, _y, _yaw, kappa = path.sample(s0 + distances)
        delta_ff = np.arctan(cfg.wheelbase_m * np.asarray(kappa, dtype=float))
        delta_ff = np.clip(delta_ff, -cfg.max_steer_rad, cfg.max_steer_rad)
        return np.asarray(kappa, dtype=float), delta_ff

    def _condensed_model(
        self,
        *,
        x0: np.ndarray,
        speed: float,
        kappa_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return X = base + influence @ steering_rate_sequence."""
        cfg = self.config
        n = cfg.horizon
        dt = cfg.prediction_dt_s
        v = max(speed, cfg.min_linearization_speed_mps)
        a = np.array(
            [
                [1.0, v * dt, 0.0],
                [0.0, 1.0, v / cfg.wheelbase_m * dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        b = np.array([0.0, 0.0, dt], dtype=float)
        state = np.asarray(x0, dtype=float).copy()
        influence = np.zeros((3, n), dtype=float)
        bases: list[np.ndarray] = []
        influences: list[np.ndarray] = []
        for k in range(n):
            affine = np.array([0.0, -v * float(kappa_ref[k]) * dt, 0.0], dtype=float)
            state = a @ state + affine
            influence = a @ influence
            influence[:, k] += b
            bases.append(state.copy())
            influences.append(influence.copy())
        return np.concatenate(bases), np.vstack(influences)

    def _problem(
        self,
        *,
        base: np.ndarray,
        influence: np.ndarray,
        delta_ff: np.ndarray,
        speed: float,
    ) -> QPProblem:
        cfg = self.config
        n = cfg.horizon
        q_diag = np.tile(
            np.array([cfg.weight_lateral, cfg.weight_heading, cfg.weight_steer], dtype=float), n
        )
        q_diag[-3:] *= cfg.terminal_scale
        qbar = np.diag(q_diag)
        target = np.zeros(3 * n, dtype=float)
        target[2::3] = delta_ff
        d = _difference_matrix(n)
        d_offset = np.zeros(n, dtype=float)
        d_offset[0] = -self._steer_rate_rps

        p = 2.0 * (
            influence.T @ qbar @ influence
            + cfg.weight_steer_rate * np.eye(n)
            + cfg.weight_steer_accel * (d.T @ d)
        )
        q = 2.0 * (
            influence.T @ qbar @ (base - target)
            + cfg.weight_steer_accel * d.T @ d_offset
        )
        p += 1e-7 * np.eye(n)

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        # Steering-rate input bounds.
        rows.extend(np.eye(n))
        lower.extend([-cfg.max_steer_rate_rps] * n)
        upper.extend([cfg.max_steer_rate_rps] * n)

        # Predicted steering angle bounds, tightened by lateral acceleration.
        lat_delta = cfg.max_steer_rad
        if speed > 0.3:
            lat_delta = min(
                lat_delta,
                math.atan(cfg.max_lateral_accel_mps2 * cfg.wheelbase_m / max(speed * speed, 1e-6)),
            )
        for k in range(n):
            row = influence[3 * k + 2]
            bias = float(base[3 * k + 2])
            rows.append(row)
            lower.append(-lat_delta - bias)
            upper.append(lat_delta - bias)

        # Steering acceleration bounds; first row is relative to last applied rate.
        accel_bound = cfg.max_steer_accel_rps2 * cfg.prediction_dt_s
        for k in range(n):
            rows.append(d[k])
            offset = d_offset[k]
            lower.append(-accel_bound - offset)
            upper.append(accel_bound - offset)

        return QPProblem(
            P=p,
            q=q,
            A=np.asarray(rows, dtype=float),
            l=np.asarray(lower, dtype=float),
            u=np.asarray(upper, dtype=float),
        )

    def _longitudinal(
        self,
        path: SpatialPath,
        ego: EgoPose,
        kappa_ref: np.ndarray,
        *,
        progress_s: float,
        now_s: float | None,
    ) -> tuple[float, float, float, float, float, float, str]:
        cfg = self.config
        # A single interpolation spike must not collapse straight-line speed.
        # Sustained curvature still survives this robust percentile unchanged.
        max_kappa = (
            float(np.quantile(np.abs(kappa_ref), np.clip(cfg.curve_limit_quantile, 0.0, 1.0)))
            if kappa_ref.size
            else 0.0
        )
        curve_limit = cfg.max_speed_mps
        if max_kappa > 1e-4:
            curve_limit = math.sqrt(cfg.max_lateral_accel_mps2 / max_kappa)
        remaining = max(0.0, path.length_m - progress_s - cfg.path_end_margin_m)
        horizon_limit = math.sqrt(2.0 * cfg.max_brake_mps2 * remaining)

        path_age = 0.0 if now_s is None else max(0.0, float(now_s) - path.stamp_s)
        soft = float(cfg.path_stale_soft_s)
        hard = float(cfg.path_stale_hard_s)
        zero = max(hard, float(getattr(cfg, "path_stale_zero_s", 5.0)))
        crawl = min(
            float(cfg.max_speed_mps),
            max(0.0, float(getattr(cfg, "path_stale_crawl_mps", 1.0))),
        )
        if path_age <= soft:
            freshness_limit = cfg.max_speed_mps
            freshness_regime = "fresh"
        elif path_age >= zero:
            freshness_limit = 0.0
            freshness_regime = "hard_stop"
        elif path_age >= hard:
            # Crawl → 0 between hard and zero (not an instant park at hard).
            t = (zero - path_age) / max(zero - hard, 1e-6)
            t = float(np.clip(t, 0.0, 1.0))
            freshness_limit = crawl * t
            freshness_regime = "crawl" if freshness_limit > 1e-3 else "hard_stop"
        else:
            # soft → hard: max → crawl
            t = (hard - path_age) / max(hard - soft, 1e-6)
            t = float(np.clip(t, 0.0, 1.0))
            freshness_limit = crawl + (cfg.max_speed_mps - crawl) * t
            freshness_regime = "soft_ramp"

        target = max(
            0.0,
            min(
                path.target_speed_mps,
                cfg.max_speed_mps,
                curve_limit,
                horizon_limit,
                freshness_limit,
            ),
        )
        error = target - ego.speed_mps
        self._speed_integral = float(np.clip(self._speed_integral + error * cfg.control_dt_s, -2.0, 2.0))
        accel = cfg.longitudinal_kp * error + cfg.longitudinal_ki * self._speed_integral
        return (
            target,
            float(np.clip(accel, -cfg.max_brake_mps2, cfg.max_accel_mps2)),
            curve_limit,
            horizon_limit,
            freshness_limit,
            path_age,
            freshness_regime,
        )

    def _fallback_steer(self, lateral: float, heading: float, curvature: float) -> float:
        cfg = self.config
        feedforward = math.atan(cfg.wheelbase_m * curvature)
        desired = feedforward - 0.35 * lateral - 0.80 * heading
        max_delta = cfg.max_steer_rate_rps * cfg.control_dt_s
        return float(np.clip(desired, self._steer_rad - max_delta, self._steer_rad + max_delta))

    def step(
        self,
        path: SpatialPath,
        ego: EgoPose,
        *,
        measured_steer_rad: float | None = None,
        now_s: float | None = None,
    ) -> VLAMPCCommand:
        cfg = self.config
        if self._path_stamp_s is None or abs(path.stamp_s - self._path_stamp_s) > 1e-9:
            self._progress_s = 0.0
            self._path_stamp_s = path.stamp_s
        if measured_steer_rad is not None and math.isfinite(measured_steer_rad):
            self._steer_rad = float(np.clip(measured_steer_rad, -cfg.max_steer_rad, cfg.max_steer_rad))

        s0, lateral, heading, curvature0 = self._tracking_errors(path, ego, self._progress_s)
        self._progress_s = max(self._progress_s, s0)
        prediction_horizon_s = cfg.prediction_dt_s * cfg.horizon
        reachable_speed = max(0.0, float(ego.speed_mps)) + cfg.max_accel_mps2 * prediction_horizon_s
        # Linearize near the *actual/short-term executable* speed, not the cruise cap.
        exec_speed = min(
            max(0.0, float(ego.speed_mps)) + max(0.0, float(cfg.linearization_speed_slack_mps)),
            max(0.0, float(path.target_speed_mps)),
            float(cfg.max_speed_mps),
            float(reachable_speed),
        )
        linear_speed = max(float(cfg.min_linearization_speed_mps), float(exec_speed))
        kappa_ref, delta_ff = self._reference(path, s0, linear_speed)
        x0 = np.array([lateral, heading, self._steer_rad], dtype=float)
        base, influence = self._condensed_model(x0=x0, speed=linear_speed, kappa_ref=kappa_ref)
        problem = self._problem(base=base, influence=influence, delta_ff=delta_ff, speed=linear_speed)
        # OSQP 1.x emits "Polishing not needed" despite verbose=False.  The
        # structured trace below retains solver failures without flooding the
        # 20 Hz demo console.
        with redirect_stdout(io.StringIO()):
            solution, trace = self._solver.solve(problem, warm_start=True)

        solved = solution is not None and trace.status in {SolverStatus.SOLVED, SolverStatus.SOLVED_INACCURATE}
        if solved:
            requested_rate = float(solution[0])
            mode = "mpc"
        else:
            fallback = self._fallback_steer(lateral, heading, curvature0)
            requested_rate = (fallback - self._steer_rad) / max(cfg.control_dt_s, 1e-6)
            mode = "bounded_fallback"

        # The optimized rate is defined on prediction_dt; apply a physical rate at control_dt.
        requested_rate = float(np.clip(requested_rate, -cfg.max_steer_rate_rps, cfg.max_steer_rate_rps))
        max_rate_change = cfg.max_steer_accel_rps2 * cfg.control_dt_s
        applied_rate = float(
            np.clip(requested_rate, self._steer_rate_rps - max_rate_change, self._steer_rate_rps + max_rate_change)
        )
        steer = float(
            np.clip(
                self._steer_rad + applied_rate * cfg.control_dt_s,
                -cfg.max_steer_rad,
                cfg.max_steer_rad,
            )
        )
        self._steer_rad = steer
        self._steer_rate_rps = applied_rate
        (
            target_speed,
            accel,
            curve_limit,
            horizon_limit,
            freshness_limit,
            path_age,
            freshness_regime,
        ) = self._longitudinal(
            path,
            ego,
            kappa_ref,
            progress_s=s0,
            now_s=now_s,
        )
        return VLAMPCCommand(
            steer_rad=steer,
            steer_rate_rps=applied_rate,
            accel_mps2=accel,
            target_speed_mps=target_speed,
            lateral_error_m=lateral,
            heading_error_rad=heading,
            reference_curvature=curvature0,
            curve_speed_limit_mps=curve_limit,
            horizon_speed_limit_mps=horizon_limit,
            freshness_speed_limit_mps=freshness_limit,
            path_age_s=path_age,
            freshness_regime=freshness_regime,
            mode=mode,
            solver_status=trace.status.value,
            solver_backend=trace.backend,
            solver_ms=trace.latency_ms,
        )
