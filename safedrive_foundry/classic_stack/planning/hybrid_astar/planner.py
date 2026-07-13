"""Hybrid A* planner, Dubins grid baseline, and feature-based selector."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from classic_stack.geometry import wrap_angle
from classic_stack.planning.frenet.planner import Trajectory, TrajectoryPoint
from classic_stack.planning.hybrid_astar.config import (
    HybridAstarConfig,
    config_sha256,
    load_hybrid_astar_config,
)
from classic_stack.planning.hybrid_astar.reeds_shepp import dubins_path_length, reeds_shepp_path


@dataclass(frozen=True)
class ObstacleMap:
    """Axis-aligned world with circular obstacles and optional walls."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    circles: tuple[tuple[float, float, float], ...] = ()  # x,y,r
    boxes: tuple[tuple[float, float, float, float], ...] = ()  # xmin,ymin,xmax,ymax

    def in_bounds(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def collides(self, x: float, y: float, radius: float = 0.9) -> bool:
        if not self.in_bounds(x, y):
            return True
        for cx, cy, r in self.circles:
            if math.hypot(x - cx, y - cy) < r + radius:
                return True
        for bx0, by0, bx1, by1 in self.boxes:
            if bx0 - radius <= x <= bx1 + radius and by0 - radius <= y <= by1 + radius:
                return True
        return False


@dataclass(frozen=True)
class ManeuverRequest:
    start: tuple[float, float, float]  # x,y,yaw
    goal: tuple[float, float, float]
    world: ObstacleMap
    require_reverse: bool = False
    narrow: bool = False
    blocked: bool = False
    seed: int = 0
    max_expansions: int | None = None
    time_budget_ms: float | None = None


@dataclass
class ManeuverResult:
    ok: bool
    failure_code: str | None
    trajectory: Trajectory | None
    path_length_m: float
    gear_switches: int
    nodes_expanded: int
    nodes_reopened: int
    analytic_hits: int
    wall_time_ms: float
    partial: bool
    open_size: int
    closed_size: int
    planner_name: str
    reject_reasons: dict[str, int] = field(default_factory=dict)
    config_hash: str | None = None
    config_name: str | None = None
    curvature_mean: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failure_code": self.failure_code,
            "path_length_m": self.path_length_m,
            "gear_switches": self.gear_switches,
            "nodes_expanded": self.nodes_expanded,
            "nodes_reopened": self.nodes_reopened,
            "analytic_hits": self.analytic_hits,
            "wall_time_ms": self.wall_time_ms,
            "partial": self.partial,
            "open_size": self.open_size,
            "closed_size": self.closed_size,
            "planner_name": self.planner_name,
            "reject_reasons": dict(self.reject_reasons),
            "config_hash": self.config_hash,
            "config_name": self.config_name,
            "curvature_mean": self.curvature_mean,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
        }


@dataclass(order=True)
class _Node:
    f: float
    g: float
    x: float = field(compare=False)
    y: float = field(compare=False)
    yaw: float = field(compare=False)
    gear: int = field(compare=False)  # +1 forward, -1 reverse
    steer: float = field(compare=False)
    parent: int | None = field(compare=False)
    idx: int = field(compare=False)


class HybridAstarPlanner:
    def __init__(self, config: HybridAstarConfig | None = None) -> None:
        self.config = config or load_hybrid_astar_config()
        self.config_hash = config_sha256(self.config.raw_toml)

    def plan(self, request: ManeuverRequest) -> ManeuverResult:
        t0 = time.perf_counter()
        cfg = self.config
        max_exp = request.max_expansions or cfg.max_expansions
        budget = (request.time_budget_ms or cfg.time_budget_ms) / 1000.0
        radius = cfg.min_turning_radius_m
        disc_r = 0.5 * cfg.width_m

        start = request.start
        goal = request.goal
        if request.world.collides(start[0], start[1], disc_r):
            return self._fail("START_COLLISION", t0, 0, 0, 0, 0, 0)
        if request.world.collides(goal[0], goal[1], disc_r):
            return self._fail("GOAL_COLLISION", t0, 0, 0, 0, 0, 0)

        yaw_res = 2.0 * math.pi / cfg.yaw_bins
        open_heap: list[_Node] = []
        nodes: list[_Node] = []
        g_score: dict[tuple[int, int, int, int], float] = {}
        closed: set[tuple[int, int, int, int]] = set()
        rejects: dict[str, int] = {}
        reopened = 0
        analytic_hits = 0
        expansions = 0
        best_idx = 0
        best_goal_dist = math.hypot(start[0] - goal[0], start[1] - goal[1])

        def key(x: float, y: float, yaw: float, gear: int) -> tuple[int, int, int, int]:
            return (
                int(round(x / cfg.xy_resolution_m)),
                int(round(y / cfg.xy_resolution_m)),
                int(round(wrap_angle(yaw) / yaw_res)) % cfg.yaw_bins,
                gear,
            )

        def heuristic(x: float, y: float, yaw: float) -> float:
            return dubins_path_length(x, y, yaw, goal[0], goal[1], goal[2], radius)

        h0 = heuristic(*start)
        root = _Node(f=h0, g=0.0, x=start[0], y=start[1], yaw=start[2], gear=1, steer=0.0, parent=None, idx=0)
        nodes.append(root)
        heapq.heappush(open_heap, root)
        g_score[key(start[0], start[1], start[2], 1)] = 0.0

        directions = (1, -1) if cfg.allow_reverse else (1,)
        goal_found_idx: int | None = None

        while open_heap and expansions < max_exp:
            if time.perf_counter() - t0 > budget:
                rejects["TIMEOUT"] = rejects.get("TIMEOUT", 0) + 1
                break
            cur = heapq.heappop(open_heap)
            k = key(cur.x, cur.y, cur.yaw, cur.gear)
            if k in closed:
                continue
            closed.add(k)
            expansions += 1

            dist = math.hypot(cur.x - goal[0], cur.y - goal[1])
            if dist < best_goal_dist:
                best_goal_dist = dist
                best_idx = cur.idx

            if (
                dist <= cfg.goal_xy_tol_m
                and abs(wrap_angle(cur.yaw - goal[2])) <= cfg.goal_yaw_tol_rad
            ):
                goal_found_idx = cur.idx
                break

            # Analytic RS expansion (endpoint must match sample; no snap-to-goal)
            if expansions % cfg.analytic_expansion_every == 0 or dist < cfg.analytic_when_dist_m:
                rs = reeds_shepp_path(
                    cur.x,
                    cur.y,
                    cur.yaw,
                    goal[0],
                    goal[1],
                    goal[2],
                    radius,
                    xy_tol=cfg.goal_xy_tol_m,
                    yaw_tol=cfg.goal_yaw_tol_rad,
                )
                if rs is not None:
                    samples = rs.sample(cur.x, cur.y, cur.yaw, radius, ds=cfg.step_m)
                    # Collision on full sample including last point
                    if samples and all(
                        not request.world.collides(px, py, disc_r) for px, py, _ in samples
                    ):
                        ex, ey, eyaw = samples[-1]
                        if (
                            math.hypot(ex - goal[0], ey - goal[1]) <= cfg.goal_xy_tol_m
                            and abs(wrap_angle(eyaw - goal[2])) <= cfg.goal_yaw_tol_rad
                        ):
                            analytic_hits += 1
                            g_new = cur.g + rs.length
                            parent_i = cur.idx
                            # gear from signed segment length via consecutive samples motion
                            for i_s, (px, py, pyaw) in enumerate(samples[1:]):
                                # approximate gear: forward if movement aligns with heading
                                if i_s == 0:
                                    gear = cur.gear
                                else:
                                    px0, py0, yaw0 = samples[i_s]
                                    move_yaw = math.atan2(py - py0, px - px0)
                                    gear = 1 if abs(wrap_angle(move_yaw - yaw0)) < math.pi / 2 else -1
                                n = _Node(
                                    f=g_new,
                                    g=g_new,
                                    x=px,
                                    y=py,
                                    yaw=pyaw,
                                    gear=gear,
                                    steer=0.0,
                                    parent=parent_i,
                                    idx=len(nodes),
                                )
                                nodes.append(n)
                                parent_i = n.idx
                            goal_found_idx = parent_i
                            break

            for gear in directions:
                for steer in cfg.steer_set:
                    nx, ny, nyaw = self._propagate(cur.x, cur.y, cur.yaw, steer, gear, cfg)
                    if request.world.collides(nx, ny, disc_r):
                        rejects["collision"] = rejects.get("collision", 0) + 1
                        continue
                    step_cost = cfg.w_length * cfg.step_m
                    if gear < 0:
                        step_cost *= cfg.w_reverse
                    if gear != cur.gear:
                        step_cost += cfg.w_gear * cfg.gear_switch_penalty
                    step_cost += cfg.w_steer * abs(steer - cur.steer) * cfg.steer_switch_penalty
                    g_new = cur.g + step_cost
                    nk = key(nx, ny, nyaw, gear)
                    if nk in closed:
                        continue
                    prev = g_score.get(nk)
                    if prev is not None and g_new >= prev - 1e-9:
                        continue
                    if prev is not None:
                        reopened += 1
                    g_score[nk] = g_new
                    h = heuristic(nx, ny, nyaw)
                    node = _Node(
                        f=g_new + h,
                        g=g_new,
                        x=nx,
                        y=ny,
                        yaw=nyaw,
                        gear=gear,
                        steer=steer,
                        parent=cur.idx,
                        idx=len(nodes),
                    )
                    nodes.append(node)
                    heapq.heappush(open_heap, node)

        elapsed = (time.perf_counter() - t0) * 1000.0
        if goal_found_idx is None:
            if cfg.partial_solution and nodes:
                # partial: best so far
                path_nodes = self._reconstruct(nodes, best_idx)
                traj, length, gears, kappa_mean = self._to_trajectory(path_nodes, source="hybrid_astar_partial")
                return ManeuverResult(
                    ok=False,
                    failure_code="PARTIAL_SOLUTION",
                    trajectory=traj,
                    path_length_m=length,
                    gear_switches=gears,
                    nodes_expanded=expansions,
                    nodes_reopened=reopened,
                    analytic_hits=analytic_hits,
                    wall_time_ms=elapsed,
                    partial=True,
                    open_size=len(open_heap),
                    closed_size=len(closed),
                    planner_name="hybrid_astar",
                    reject_reasons=rejects,
                    config_hash=self.config_hash,
                    config_name=cfg.name,
                    curvature_mean=kappa_mean,
                )
            code = "TIMEOUT" if "TIMEOUT" in rejects else "NO_PATH"
            return self._fail(code, t0, expansions, reopened, analytic_hits, len(open_heap), len(closed), rejects)

        path_nodes = self._reconstruct(nodes, goal_found_idx)
        traj, length, gears, kappa_mean = self._to_trajectory(path_nodes, source="hybrid_astar")
        return ManeuverResult(
            ok=True,
            failure_code=None,
            trajectory=traj,
            path_length_m=length,
            gear_switches=gears,
            nodes_expanded=expansions,
            nodes_reopened=reopened,
            analytic_hits=analytic_hits,
            wall_time_ms=elapsed,
            partial=False,
            open_size=len(open_heap),
            closed_size=len(closed),
            planner_name="hybrid_astar",
            reject_reasons=rejects,
            config_hash=self.config_hash,
            config_name=cfg.name,
            curvature_mean=kappa_mean,
        )

    def _propagate(
        self, x: float, y: float, yaw: float, steer: float, gear: int, cfg: HybridAstarConfig
    ) -> tuple[float, float, float]:
        # bicycle model
        beta = math.tan(steer) / max(cfg.wheelbase_m, 1e-3)
        ds = cfg.step_m * (1 if gear > 0 else -1)
        if abs(beta) < 1e-6:
            return x + ds * math.cos(yaw), y + ds * math.sin(yaw), yaw
        yaw_n = wrap_angle(yaw + ds * beta)
        x_n = x + (math.sin(yaw_n) - math.sin(yaw)) / beta
        y_n = y - (math.cos(yaw_n) - math.cos(yaw)) / beta
        return x_n, y_n, yaw_n

    def _reconstruct(self, nodes: Sequence[_Node], idx: int) -> list[_Node]:
        chain: list[_Node] = []
        cur: int | None = idx
        guard = 0
        while cur is not None and guard < 100000:
            node = nodes[cur]
            chain.append(node)
            cur = node.parent
            guard += 1
        chain.reverse()
        return chain

    def _to_trajectory(
        self, path: Sequence[_Node], *, source: str
    ) -> tuple[Trajectory, float, int, float]:
        if not path:
            return Trajectory(points=(), trajectory_id="empty", source=source), 0.0, 0, 0.0
        pts: list[TrajectoryPoint] = []
        length = 0.0
        gears = 0
        kappas: list[float] = []
        t = 0.0
        v = 2.0
        for i, node in enumerate(path):
            if i > 0:
                prev = path[i - 1]
                ds = math.hypot(node.x - prev.x, node.y - prev.y)
                length += ds
                t += ds / max(v, 0.1)
                if node.gear != prev.gear:
                    gears += 1
                # curvature proxy from steer
                kappa = math.tan(node.steer) / max(self.config.wheelbase_m, 1e-3)
                kappas.append(abs(kappa))
            else:
                kappa = 0.0
            pts.append(
                TrajectoryPoint(
                    t=t,
                    x=node.x,
                    y=node.y,
                    yaw=node.yaw,
                    kappa=math.tan(node.steer) / max(self.config.wheelbase_m, 1e-3),
                    v=v if node.gear > 0 else -v,
                    a=0.0,
                    jerk=0.0,
                )
            )
        mean_k = sum(kappas) / len(kappas) if kappas else 0.0
        traj = Trajectory(points=tuple(pts), trajectory_id=f"{source}-{len(pts)}", source=source)
        return traj, length, gears, mean_k

    def _fail(
        self,
        code: str,
        t0: float,
        expansions: int,
        reopened: int,
        analytic: int,
        open_n: int,
        closed_n: int,
        rejects: dict[str, int] | None = None,
    ) -> ManeuverResult:
        return ManeuverResult(
            ok=False,
            failure_code=code,
            trajectory=None,
            path_length_m=0.0,
            gear_switches=0,
            nodes_expanded=expansions,
            nodes_reopened=reopened,
            analytic_hits=analytic,
            wall_time_ms=(time.perf_counter() - t0) * 1000.0,
            partial=False,
            open_size=open_n,
            closed_size=closed_n,
            planner_name="hybrid_astar",
            reject_reasons=rejects or {},
            config_hash=self.config_hash,
            config_name=self.config.name,
        )


class GridAstarDubinsPlanner:
    """Holonomic grid A* + Dubins length baseline for fair comparison."""

    def __init__(self, config: HybridAstarConfig | None = None) -> None:
        self.config = config or load_hybrid_astar_config()
        self.config_hash = config_sha256(self.config.raw_toml)

    def plan(self, request: ManeuverRequest) -> ManeuverResult:
        t0 = time.perf_counter()
        res = self.config.xy_resolution_m
        disc = 0.5 * self.config.width_m
        sx, sy, _ = request.start
        gx, gy, gyaw = request.goal
        start_i = (int(round(sx / res)), int(round(sy / res)))
        goal_i = (int(round(gx / res)), int(round(gy / res)))

        def cell_ok(i: int, j: int) -> bool:
            return not request.world.collides(i * res, j * res, disc)

        if not cell_ok(*start_i) or not cell_ok(*goal_i):
            return ManeuverResult(
                ok=False,
                failure_code="GRID_ENDPOINT_COLLISION",
                trajectory=None,
                path_length_m=0.0,
                gear_switches=0,
                nodes_expanded=0,
                nodes_reopened=0,
                analytic_hits=0,
                wall_time_ms=(time.perf_counter() - t0) * 1000.0,
                partial=False,
                open_size=0,
                closed_size=0,
                planner_name="grid_astar_dubins",
                config_hash=self.config_hash,
                config_name=self.config.name,
            )

        open_h: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_h, (0.0, 0.0, start_i))
        came: dict[tuple[int, int], tuple[int, int] | None] = {start_i: None}
        gcost = {start_i: 0.0}
        expansions = 0
        found = False
        while open_h and expansions < self.config.max_expansions:
            f, g, cur = heapq.heappop(open_h)
            expansions += 1
            if cur == goal_i:
                found = True
                break
            ci, cj = cur
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ni, nj = ci + di, cj + dj
                if not cell_ok(ni, nj):
                    continue
                step = res * (math.sqrt(2) if di and dj else 1.0)
                ng = g + step
                nxt = (ni, nj)
                if ng < gcost.get(nxt, 1e18):
                    gcost[nxt] = ng
                    h = math.hypot((ni - goal_i[0]) * res, (nj - goal_i[1]) * res)
                    # dubins bias
                    h = 0.5 * h + 0.5 * dubins_path_length(
                        ni * res, nj * res, 0.0, gx, gy, gyaw, self.config.min_turning_radius_m
                    )
                    heapq.heappush(open_h, (ng + h, ng, nxt))
                    came[nxt] = cur
        elapsed = (time.perf_counter() - t0) * 1000.0
        if not found:
            return ManeuverResult(
                ok=False,
                failure_code="NO_PATH",
                trajectory=None,
                path_length_m=0.0,
                gear_switches=0,
                nodes_expanded=expansions,
                nodes_reopened=0,
                analytic_hits=0,
                wall_time_ms=elapsed,
                partial=False,
                open_size=len(open_h),
                closed_size=len(came),
                planner_name="grid_astar_dubins",
                config_hash=self.config_hash,
                config_name=self.config.name,
            )
        # reconstruct
        chain: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal_i
        while cur is not None:
            chain.append(cur)
            cur = came[cur]
        chain.reverse()
        pts: list[TrajectoryPoint] = []
        length = 0.0
        t = 0.0
        for n, (i, j) in enumerate(chain):
            x, y = i * res, j * res
            if n > 0:
                pi, pj = chain[n - 1]
                length += math.hypot((i - pi) * res, (j - pj) * res)
                t = length / 2.0
            yaw = math.atan2(gy - y, gx - x) if n < len(chain) - 1 else gyaw
            pts.append(TrajectoryPoint(t=t, x=x, y=y, yaw=yaw, kappa=0.0, v=2.0, a=0.0, jerk=0.0))
        traj = Trajectory(points=tuple(pts), trajectory_id="grid-dubins", source="grid_astar_dubins")
        return ManeuverResult(
            ok=True,
            failure_code=None,
            trajectory=traj,
            path_length_m=length,
            gear_switches=0,
            nodes_expanded=expansions,
            nodes_reopened=0,
            analytic_hits=0,
            wall_time_ms=elapsed,
            partial=False,
            open_size=len(open_h),
            closed_size=len(came),
            planner_name="grid_astar_dubins",
            config_hash=self.config_hash,
            config_name=self.config.name,
        )


class PlannerSelector:
    """Feature-based planner choice — never hardcodes scenario names."""

    def __init__(self, config: HybridAstarConfig | None = None) -> None:
        self.config = config or load_hybrid_astar_config()
        self.hybrid = HybridAstarPlanner(self.config)
        self.grid = GridAstarDubinsPlanner(self.config)

    def select(self, request: ManeuverRequest) -> str:
        # Features only: reverse need, narrow corridor, blocked road.
        if request.require_reverse or request.narrow or request.blocked:
            return "hybrid_astar"
        return "grid_astar_dubins"

    def plan(self, request: ManeuverRequest) -> ManeuverResult:
        choice = self.select(request)
        if choice == "hybrid_astar":
            return self.hybrid.plan(request)
        return self.grid.plan(request)
