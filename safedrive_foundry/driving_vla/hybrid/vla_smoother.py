"""Intent-preserving kinematic filter for VLA trajectories.

The filter follows the VLA polyline and adjusts only the speed profile needed
for acceleration, jerk and lateral-acceleration feasibility.  Position, yaw,
curvature and speed are then recomputed together, so metadata can never claim a
safe curvature while the x/y path still contains the original sharp turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from driving_vla.adapter.policy_adapter import TrajectoryArray
from driving_vla.hybrid.contracts import ObservableAnchor


@dataclass(frozen=True)
class VLASmootherConfig:
    """Physical bounds for the deterministic filter."""
    # Physical limits for passenger sedan (Tesla Model 3 / CARLA vehicle)
    wheelbase_m: float = 2.875
    max_steer_rad: float = 0.60
    # Stay inside the deployed Safety kernel (3.0 m/s^2 and 3.0 m/s^2
    # lateral) with a small numerical margin.  The previous 3.5 values made a
    # trajectory pass this filter and then predictably fail final Safety.
    max_accel_mps2: float = 2.8
    max_decel_mps2: float = 5.0
    max_jerk_mps3: float = 4.0
    max_speed_mps: float = 35.0
    max_lateral_accel_mps2: float = 2.8

    dt_s: float = 0.25
    steps: int = 10


def _solve_longitudinal_profile(
    raw_v: np.ndarray,
    ego_v: float,
    ego_a: float,
    cfg: VLASmootherConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve constrained longitudinal speed and acceleration profile."""
    n = len(raw_v)
    v = np.zeros(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)

    v_prev = ego_v
    a_prev = ego_a

    for i in range(n):
        target_v = np.clip(raw_v[i], 0.0, cfg.max_speed_mps)

        # Desired unconstrained acceleration to reach target_v in dt
        desired_a = (target_v - v_prev) / cfg.dt_s

        # Bound acceleration step by maximum jerk: |a - a_prev| <= j_max * dt
        max_a_from_jerk = a_prev + cfg.max_jerk_mps3 * cfg.dt_s
        min_a_from_jerk = a_prev - cfg.max_jerk_mps3 * cfg.dt_s

        # Intersect with global vehicle limits
        upper_a = min(cfg.max_accel_mps2, max_a_from_jerk)
        lower_a = max(-cfg.max_decel_mps2, min_a_from_jerk)

        actual_a = np.clip(desired_a, lower_a, upper_a)
        actual_v = max(0.0, v_prev + actual_a * cfg.dt_s)

        a[i] = actual_a
        v[i] = actual_v

        v_prev = actual_v
        a_prev = actual_a

    return v, a


def smooth_vla_trajectory(
    trajectory: TrajectoryArray,
    anchor: ObservableAnchor | None = None,
    config: VLASmootherConfig | None = None,
) -> TrajectoryArray:
    """Filter speed and resample on the original VLA path consistently."""
    cfg = config or VLASmootherConfig()
    pts = np.array(trajectory.points_xy_yaw_v_a_kappa, dtype=np.float64)
    if len(pts) != cfg.steps:
        return trajectory

    # Extract ego physical state
    if anchor is not None and anchor.safety_snapshot is not None:
        snap = anchor.safety_snapshot
        ego_x = float(snap.ego_x)
        ego_y = float(snap.ego_y)
        ego_yaw = float(snap.ego_yaw)
        ego_v = max(0.0, float(snap.ego_v))
        ego_a = float(snap.ego_a)
    else:
        ego_x = float(pts[0, 0])
        ego_y = float(pts[0, 1])
        ego_yaw = float(pts[0, 2])
        ego_v = max(0.0, float(pts[0, 3]))
        ego_a = 0.0

    raw_v = pts[:, 3]

    # Geometry remains the VLA intent.  Include the current ego pose as s=0.
    path = np.vstack(([ego_x, ego_y], pts[:, :2]))
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segment)))
    if not np.isfinite(arclength).all() or arclength[-1] <= 1e-6:
        return trajectory

    def integrate_speeds(speeds: np.ndarray) -> np.ndarray:
        positions = np.zeros(len(speeds), dtype=np.float64)
        previous = ego_v
        cumulative = 0.0
        for index, speed in enumerate(speeds):
            cumulative += 0.5 * (previous + speed) * cfg.dt_s
            positions[index] = cumulative
            previous = speed
        if positions[-1] > arclength[-1]:
            positions *= arclength[-1] / max(positions[-1], 1e-9)
        return positions

    def sample_geometry(s_positions: np.ndarray):
        target_x = np.interp(s_positions, arclength, path[:, 0])
        target_y = np.interp(s_positions, arclength, path[:, 1])
        x = np.zeros(len(target_x), dtype=np.float64)
        y = np.zeros(len(target_y), dtype=np.float64)
        yaw = np.zeros(len(target_x), dtype=np.float64)
        kappa = np.zeros(len(target_x), dtype=np.float64)
        previous_x, previous_y, previous_yaw = ego_x, ego_y, ego_yaw
        steering_kappa = math.tan(cfg.max_steer_rad) / cfg.wheelbase_m
        for index in range(len(x)):
            dx, dy = target_x[index] - previous_x, target_y[index] - previous_y
            distance = math.hypot(dx, dy)
            desired_yaw = math.atan2(dy, dx) if distance > 1e-4 else previous_yaw
            desired_change = math.atan2(
                math.sin(desired_yaw - previous_yaw),
                math.cos(desired_yaw - previous_yaw),
            )
            # Keep the VLA's turn intent.  Clipping geometry by instantaneous
            # speed made a fast vehicle continue almost straight through a
            # bend and leave the route.  Steering reachability bounds the
            # shape; lateral acceleration is handled by the anticipatory speed
            # pass below and, for an already-too-late first point, final Safety.
            max_change = steering_kappa * max(distance, 1e-3)
            dyaw = float(np.clip(desired_change, -max_change, max_change))
            yaw[index] = previous_yaw + dyaw
            x[index] = previous_x + distance * math.cos(yaw[index])
            y[index] = previous_y + distance * math.sin(yaw[index])
            kappa[index] = dyaw / max(distance, 1e-3)
            previous_x, previous_y, previous_yaw = x[index], y[index], yaw[index]
        return x, y, yaw, kappa

    # Geometry depends on the distance implied by the speed profile, while the
    # safe speed cap depends on the resulting geometry.  A single pass can
    # therefore create a new sharp point after it has already computed the
    # cap.  Iterate a small fixed number of times and only tighten targets.
    speed_targets = np.clip(raw_v, 0.0, cfg.max_speed_mps)
    smooth_v = np.zeros_like(speed_targets)
    smooth_a = np.zeros_like(speed_targets)
    smooth_x = np.zeros_like(speed_targets)
    smooth_y = np.zeros_like(speed_targets)
    smooth_yaw = np.zeros_like(speed_targets)
    smooth_kappa = np.zeros_like(speed_targets)
    for _ in range(6):
        smooth_v, smooth_a = _solve_longitudinal_profile(
            speed_targets, ego_v, ego_a, cfg
        )
        s_positions = integrate_speeds(smooth_v)
        smooth_x, smooth_y, smooth_yaw, smooth_kappa = sample_geometry(s_positions)
        lateral_caps = np.array(
            [
                cfg.max_speed_mps
                if abs(curvature) <= 1e-6
                else math.sqrt(0.98 * cfg.max_lateral_accel_mps2 / abs(curvature))
                for curvature in smooth_kappa
            ],
            dtype=np.float64,
        )
        tightened = np.minimum(speed_targets, lateral_caps)
        # A pointwise curvature cap starts braking only when the car is already
        # at the curve.  Propagate each future cap backward through the path so
        # the VLA begins slowing while the bend is still ahead.
        for index in range(len(tightened) - 2, -1, -1):
            distance = max(0.0, float(s_positions[index + 1] - s_positions[index]))
            braking_cap = math.sqrt(
                max(
                    0.0,
                    float(tightened[index + 1]) ** 2
                    + 2.0 * cfg.max_decel_mps2 * distance,
                )
            )
            tightened[index] = min(float(tightened[index]), braking_cap)
        if np.max(np.abs(tightened - speed_targets)) <= 1e-4:
            break
        speed_targets = tightened

    smoothed_matrix = np.column_stack([
        smooth_x,
        smooth_y,
        smooth_yaw,
        smooth_v,
        smooth_a,
        smooth_kappa,
    ])

    return TrajectoryArray(
        points_xy_yaw_v_a_kappa=tuple(tuple(float(val) for val in row) for row in smoothed_matrix),
        probability=trajectory.probability,
        uncertainty=trajectory.uncertainty,
        candidate_id=trajectory.candidate_id,
        behavior=trajectory.behavior,
        intended_action=trajectory.intended_action,
    )


__all__ = ["VLASmootherConfig", "smooth_vla_trajectory"]
