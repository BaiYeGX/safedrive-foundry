"""Fixed-sampling Frenet lattice + S-T speed planner (G1-04 baseline)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from classic_stack.geometry import FrenetFrame, ReferencePath, clamp
from classic_stack.planning.frenet.config import FrenetSTConfig, config_sha256, load_frenet_st_config
from classic_stack.planning.speed.prediction import ActorState, predict_actors_st
from classic_stack.planning.speed.st_dp import build_st_occupancy, smooth_jerk, solve_st_dp


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float
    kappa: float
    v: float
    a: float
    jerk: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Trajectory:
    points: tuple[TrajectoryPoint, ...]
    trajectory_id: str
    source: str = "frenet_st"
    risk_cost: float = 0.0
    tracking_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "source": self.source,
            "risk_cost": self.risk_cost,
            "tracking_cost": self.tracking_cost,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass(frozen=True)
class StaticObstacle:
    x: float
    y: float
    radius_m: float = 1.0


# Re-export for public API
__all_actors__ = ActorState


@dataclass(frozen=True)
class PlanRequest:
    reference: ReferencePath
    v0: float = 6.0
    a0: float = 0.0
    s0: float = 0.0
    d0: float = 0.0
    scenario_kind: str = "cruise"  # follow|stop|lane_change|cut_in|avoid
    actors: tuple[ActorState, ...] = ()
    static_obstacles: tuple[StaticObstacle, ...] = ()
    target_speed_mps: float | None = None
    stop_s: float | None = None
    preferred_offset_m: float | None = None
    seed: int = 0


@dataclass
class PlanResult:
    ok: bool
    failure_code: str | None
    reject_reasons: dict[str, int]
    candidates: int
    wall_time_ms: float
    cost_terms: dict[str, float]
    trajectory: Trajectory | None
    planner_name: str = "frenet_st"
    config_hash: str | None = None
    config_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failure_code": self.failure_code,
            "reject_reasons": dict(self.reject_reasons),
            "candidates": self.candidates,
            "wall_time_ms": self.wall_time_ms,
            "cost_terms": dict(self.cost_terms),
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "planner_name": self.planner_name,
            "config_hash": self.config_hash,
            "config_name": self.config_name,
        }


class FrenetPlanner:
    """Fixed uniform Frenet sampling + ST-DP speed + jerk smoothing."""

    def __init__(self, config: FrenetSTConfig | None = None) -> None:
        self.config = config or load_frenet_st_config()
        self.config_hash = config_sha256(self.config.raw_toml)

    def plan(self, request: PlanRequest) -> PlanResult:
        t0 = time.perf_counter()
        rejects: dict[str, int] = {}
        frame = FrenetFrame(request.reference, self.config.vehicle)
        samples = self._sample_paths(request, frame, rejects)
        if not samples:
            return self._fail("NO_SPATIAL_CANDIDATE", rejects, 0, t0)

        ranked = sorted(samples, key=lambda item: item["spatial_cost"])
        ranked = ranked[: self.config.max_candidates]
        best: dict[str, Any] | None = None
        best_cost = float("inf")

        # Shared actor prediction once in world frame
        pred = predict_actors_st(
            request.actors,
            request.reference,
            self.config,
            ego_s=request.s0,
            ego_v=request.v0,
        )

        for cand in ranked:
            path = cand["path"]  # list of (s,d)
            # shift occupancy to path s relative to ego s0
            shifted = []
            for sample in pred:
                shifted.append(
                    {
                        "actor_id": sample["actor_id"],
                        "t": sample["t"],
                        "s_min": float(sample["s_min"]) - request.s0,
                        "s_max": float(sample["s_max"]) - request.s0,
                    }
                )
            grid = build_st_occupancy(shifted, self.config)
            v_target = cand["v_target"]
            stop_s = request.stop_s
            if request.scenario_kind == "stop" and stop_s is None:
                stop_s = min(25.0, request.reference.length * 0.5)
            profile, err = solve_st_dp(
                grid,
                v0=request.v0,
                v_target=v_target,
                vehicle=self.config.vehicle,
                stop_at_s=stop_s,
            )
            if err or not profile:
                rejects[err or "ST_FAIL"] = rejects.get(err or "ST_FAIL", 0) + 1
                continue
            from classic_stack.planning.speed.st_dp import validate_profile_kinematics

            kin_err = validate_profile_kinematics(profile, vehicle=self.config.vehicle)
            if kin_err:
                rejects[kin_err] = rejects.get(kin_err, 0) + 1
                continue
            smooth = smooth_jerk(
                profile,
                window=self.config.smooth_window,
                max_jerk=self.config.vehicle.max_jerk_mps3,
                vehicle=self.config.vehicle,
            )
            # Re-check kinematics on smoothed (t,s,v,a) ignoring jerk column
            traj_points: list[TrajectoryPoint] = []
            dyn_cost = 0.0
            feasible = True
            for t, s_rel, v, a, jerk in smooth:
                s_abs = request.s0 + s_rel
                # interpolate d along path by s
                d = self._d_at_s(path, s_abs)
                if abs(d) > self.config.road_half_width_m:
                    rejects["road_boundary"] = rejects.get("road_boundary", 0) + 1
                    feasible = False
                    break
                kappa = frame.curvature_proxy(s_abs, d)
                if abs(kappa) > self.config.vehicle.max_curvature_per_m:
                    rejects["curvature"] = rejects.get("curvature", 0) + 1
                    feasible = False
                    break
                if abs(a) > self.config.vehicle.max_accel_mps2 + 1e-6 and a > 0:
                    rejects["accel"] = rejects.get("accel", 0) + 1
                    feasible = False
                    break
                if a < -self.config.vehicle.max_decel_mps2 - 1e-6:
                    rejects["decel"] = rejects.get("decel", 0) + 1
                    feasible = False
                    break
                if abs(jerk) > self.config.vehicle.max_jerk_mps3 + 1e-3:
                    rejects["jerk"] = rejects.get("jerk", 0) + 1
                    # soft: clamp rather than hard fail after smoothing
                    jerk = clamp(jerk, -self.config.vehicle.max_jerk_mps3, self.config.vehicle.max_jerk_mps3)
                if not self._static_ok(request, frame, s_abs, d):
                    rejects["static_collision"] = rejects.get("static_collision", 0) + 1
                    feasible = False
                    break
                pose = frame.frenet_to_cartesian(s_abs, d)
                traj_points.append(
                    TrajectoryPoint(t=t, x=pose.x, y=pose.y, yaw=pose.yaw, kappa=kappa, v=v, a=a, jerk=jerk)
                )
                dyn_cost += (
                    self.config.costs.get("w_accel", 0.3) * a * a
                    + self.config.costs.get("w_jerk", 0.2) * jerk * jerk
                    + self.config.costs.get("w_curvature", 0.5) * kappa * kappa
                )
            if not feasible or len(traj_points) < 2:
                continue
            # dynamic proximity cost
            if self._dynamic_collision(traj_points, request):
                rejects["dynamic_collision"] = rejects.get("dynamic_collision", 0) + 1
                continue
            total = cand["spatial_cost"] + dyn_cost - self.config.costs.get("w_progress", 0.8) * traj_points[-1].t
            if total < best_cost:
                best_cost = total
                best = {
                    "points": traj_points,
                    "spatial_cost": cand["spatial_cost"],
                    "dyn_cost": dyn_cost,
                    "v_target": v_target,
                    "offset": cand["offset"],
                }

        if best is None:
            return self._fail("NO_FEASIBLE_SPEED", rejects, len(ranked), t0)

        traj = Trajectory(
            points=tuple(best["points"]),
            trajectory_id=f"frenet-{request.scenario_kind}-{request.seed}",
            source="frenet_st",
            risk_cost=float(best["dyn_cost"]),
            tracking_cost=float(best["spatial_cost"]),
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            ok=True,
            failure_code=None,
            reject_reasons=rejects,
            candidates=len(ranked),
            wall_time_ms=elapsed,
            cost_terms={
                "spatial": float(best["spatial_cost"]),
                "dynamic": float(best["dyn_cost"]),
                "offset": float(best["offset"]),
                "v_target": float(best["v_target"]),
            },
            trajectory=traj,
            planner_name="frenet_st",
            config_hash=self.config_hash,
            config_name=self.config.name,
        )

    def _sample_paths(
        self,
        request: PlanRequest,
        frame: FrenetFrame,
        rejects: dict[str, int],
    ) -> list[dict[str, Any]]:
        offsets = list(self.config.lateral_offsets_m)
        if request.preferred_offset_m is not None:
            offsets = sorted(set(offsets + [request.preferred_offset_m]))
        if request.scenario_kind == "lane_change" and request.preferred_offset_m is None:
            offsets = sorted(set(offsets + [1.75, -1.75]))
        speeds = list(self.config.target_speeds_mps)
        if request.target_speed_mps is not None:
            speeds = sorted(set(speeds + [request.target_speed_mps]))
        if request.scenario_kind == "stop":
            speeds = sorted(set(speeds + [0.0, 2.0]))
        out: list[dict[str, Any]] = []
        for d_end in offsets:
            if abs(d_end) > self.config.road_half_width_m:
                rejects["road_boundary"] = rejects.get("road_boundary", 0) + 1
                continue
            for T in self.config.horizon_s:
                path = self._cubic_lateral_path(request.s0, request.d0, d_end, T)
                # spatial validity
                ok = True
                kappa_pen = 0.0
                for s, d in path:
                    if abs(d) > self.config.road_half_width_m:
                        rejects["road_boundary"] = rejects.get("road_boundary", 0) + 1
                        ok = False
                        break
                    kappa = frame.curvature_proxy(s, d)
                    if abs(kappa) > self.config.vehicle.max_curvature_per_m:
                        rejects["curvature"] = rejects.get("curvature", 0) + 1
                        ok = False
                        break
                    kappa_pen += abs(kappa)
                    if not self._static_ok(request, frame, s, d):
                        rejects["static_collision"] = rejects.get("static_collision", 0) + 1
                        ok = False
                        break
                if not ok:
                    continue
                for v_t in speeds:
                    if v_t < 0 or v_t > self.config.vehicle.max_speed_mps:
                        rejects["speed"] = rejects.get("speed", 0) + 1
                        continue
                    spatial = (
                        self.config.costs.get("w_offset", 1.0) * (d_end - (request.preferred_offset_m or 0.0)) ** 2
                        + self.config.costs.get("w_speed", 0.4) * (v_t - (request.target_speed_mps or v_t)) ** 2
                        + self.config.costs.get("w_curvature", 0.5) * kappa_pen
                    )
                    out.append({"path": path, "offset": d_end, "v_target": v_t, "spatial_cost": spatial, "T": T})
        return out

    def _cubic_lateral_path(
        self, s0: float, d0: float, d1: float, horizon_s: float
    ) -> list[tuple[float, float]]:
        # Map time horizon to s horizon via nominal speed mid of targets
        s_len = max(self.config.ds_m * 4, 8.0 * horizon_s / 4.0)
        n = max(4, int(s_len / self.config.ds_m))
        path: list[tuple[float, float]] = []
        for i in range(n + 1):
            u = i / n
            # smoothstep
            d = d0 + (d1 - d0) * (3 * u * u - 2 * u * u * u)
            s = s0 + u * s_len
            path.append((s, d))
        return path

    def _d_at_s(self, path: Sequence[tuple[float, float]], s: float) -> float:
        if s <= path[0][0]:
            return path[0][1]
        if s >= path[-1][0]:
            return path[-1][1]
        for i in range(len(path) - 1):
            s0, d0 = path[i]
            s1, d1 = path[i + 1]
            if s0 <= s <= s1:
                r = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
                return d0 + r * (d1 - d0)
        return path[-1][1]

    def _static_ok(self, request: PlanRequest, frame: FrenetFrame, s: float, d: float) -> bool:
        pose = frame.frenet_to_cartesian(s, d)
        for obs in request.static_obstacles:
            if math.hypot(pose.x - obs.x, pose.y - obs.y) < obs.radius_m + 0.5 * self.config.vehicle.width_m:
                return False
        return True

    def _dynamic_collision(self, points: Sequence[TrajectoryPoint], request: PlanRequest) -> bool:
        # Check ego disc vs predicted actor discs at nearest times
        for actor in request.actors:
            for p in points:
                # constant velocity replay for check
                x = actor.x + actor.speed_mps * math.cos(actor.yaw) * p.t
                y = actor.y + actor.speed_mps * math.sin(actor.yaw) * p.t
                thr = 0.5 * (actor.length_m + self.config.vehicle.length_m) * 0.6
                if math.hypot(p.x - x, p.y - y) < thr:
                    return True
        return False

    def _fail(self, code: str, rejects: dict[str, int], candidates: int, t0: float) -> PlanResult:
        return PlanResult(
            ok=False,
            failure_code=code,
            reject_reasons=rejects,
            candidates=candidates,
            wall_time_ms=(time.perf_counter() - t0) * 1000.0,
            cost_terms={},
            trajectory=None,
            planner_name="frenet_st",
            config_hash=self.config_hash,
            config_name=self.config.name,
        )
