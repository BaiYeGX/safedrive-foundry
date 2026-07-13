"""Centerline + constant-speed baseline for fair comparison."""

from __future__ import annotations

import time
from typing import Sequence

from classic_stack.geometry import FrenetFrame, ReferencePath
from classic_stack.planning.frenet.config import FrenetSTConfig
from classic_stack.planning.frenet.planner import (
    ActorState,
    PlanRequest,
    PlanResult,
    StaticObstacle,
    Trajectory,
    TrajectoryPoint,
)


class CenterlineConstantSpeedPlanner:
    """Follow d=0 at constant speed (no ST optimization)."""

    def __init__(self, config: FrenetSTConfig) -> None:
        self.config = config

    def plan(self, request: PlanRequest) -> PlanResult:
        t0 = time.perf_counter()
        frame = FrenetFrame(request.reference, self.config.vehicle)
        v = min(request.v0 if request.v0 > 0.1 else 6.0, self.config.vehicle.max_speed_mps)
        if request.scenario_kind == "stop":
            v = max(request.v0, 0.1)
        dt = self.config.st_dt_s
        horizon = self.config.st_t_horizon_s
        points: list[TrajectoryPoint] = []
        s = 0.0
        speed = v
        for k in range(int(horizon / dt) + 1):
            t = k * dt
            if request.scenario_kind == "stop":
                # linear brake to 0
                speed = max(0.0, v * (1.0 - t / max(horizon, 1e-3)))
            pose = frame.frenet_to_cartesian(s, 0.0)
            kappa = frame.curvature_proxy(s, 0.0)
            points.append(
                TrajectoryPoint(
                    t=t,
                    x=pose.x,
                    y=pose.y,
                    yaw=pose.yaw,
                    kappa=kappa,
                    v=speed,
                    a=0.0,
                    jerk=0.0,
                )
            )
            s += speed * dt
            if s >= request.reference.length:
                break
        # crude collision: if any static on centerline nearby, fail
        for obs in request.static_obstacles:
            os, od = request.reference.project(obs.x, obs.y)
            if abs(od) < obs.radius_m + 0.5 * self.config.vehicle.width_m and 0 <= os <= s:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return PlanResult(
                    ok=False,
                    failure_code="BASELINE_STATIC_COLLISION",
                    reject_reasons={"static_collision": 1},
                    candidates=1,
                    wall_time_ms=elapsed,
                    cost_terms={},
                    trajectory=None,
                    planner_name="centerline_constant_speed",
                )
        traj = Trajectory(points=tuple(points), trajectory_id="baseline-centerline", source="centerline_constant_speed")
        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            ok=True,
            failure_code=None,
            reject_reasons={},
            candidates=1,
            wall_time_ms=elapsed,
            cost_terms={"progress": -s},
            trajectory=traj,
            planner_name="centerline_constant_speed",
        )
