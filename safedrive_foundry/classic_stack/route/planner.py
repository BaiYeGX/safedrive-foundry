"""Deterministic multi-objective A* route corridor planner."""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from classic_stack.map.lane_graph import LaneEdge, LaneGraph, LaneNode, MapError


class RouteError(ValueError):
    """Raised when a route request is invalid or no legal corridor exists."""


@dataclass(frozen=True)
class RouteRequest:
    start_node_id: str
    goal_node_ids: tuple[str, ...]
    seed: int = 0
    allow_lane_change: bool = True
    allow_junction: bool = True
    max_expansions: int = 10000
    lane_change_penalty: float = 4.0
    junction_penalty: float = 2.0
    speed_weight: float = 0.15
    forbidden_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.goal_node_ids:
            raise RouteError("goal_node_ids must not be empty")
        if self.max_expansions <= 0:
            raise RouteError("max_expansions must be positive")


@dataclass(frozen=True)
class CorridorStep:
    node_id: str
    road_id: str
    lane_id: int
    edge_id: str | None
    maneuver: str
    semantic: str
    cost_from_start: float
    speed_limit_mps: float
    junction_id: str | None
    oracle_fields: tuple[str, ...] = (
        "node_id",
        "edge_id",
        "maneuver",
        "semantic",
        "cost_from_start",
        "junction_id",
    )
    observable_fields: tuple[str, ...] = ("node_id", "road_id", "lane_id", "maneuver", "semantic")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteCorridor:
    route_id: str
    map_name: str
    map_hash: str
    start_node_id: str
    goal_node_id: str
    seed: int
    steps: tuple[CorridorStep, ...]
    total_cost: float
    road_sequence: tuple[str, ...]
    lane_change_count: int
    junction_count: int
    semantics: Mapping[str, Any] = field(default_factory=dict)

    def node_ids(self) -> tuple[str, ...]:
        return tuple(step.node_id for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "map_name": self.map_name,
            "map_hash": self.map_hash,
            "start_node_id": self.start_node_id,
            "goal_node_id": self.goal_node_id,
            "seed": self.seed,
            "steps": [step.to_dict() for step in self.steps],
            "total_cost": self.total_cost,
            "road_sequence": list(self.road_sequence),
            "lane_change_count": self.lane_change_count,
            "junction_count": self.junction_count,
            "semantics": dict(self.semantics),
        }


@dataclass(frozen=True)
class RouteResult:
    ok: bool
    corridor: RouteCorridor | None
    expansions: int
    failure_code: str | None = None
    failure_node_id: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "corridor": None if self.corridor is None else self.corridor.to_dict(),
            "expansions": self.expansions,
            "failure_code": self.failure_code,
            "failure_node_id": self.failure_node_id,
            "message": self.message,
        }


def _route_id(map_hash: str, request: RouteRequest, goal: str, node_path: Sequence[str]) -> str:
    payload = {
        "map_hash": map_hash,
        "start": request.start_node_id,
        "goal": goal,
        "seed": request.seed,
        "path": list(node_path),
        "allow_lane_change": request.allow_lane_change,
        "allow_junction": request.allow_junction,
        "lane_change_penalty": request.lane_change_penalty,
        "junction_penalty": request.junction_penalty,
        "speed_weight": request.speed_weight,
        "forbidden": list(request.forbidden_node_ids),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"route-{digest[:16]}"


def _maneuver_for_edge(edge: LaneEdge | None, node: LaneNode) -> str:
    if edge is None:
        return "FOLLOW"
    if edge.kind == "lane_change_left":
        return "LEFT"
    if edge.kind == "lane_change_right":
        return "RIGHT"
    if edge.kind == "junction":
        return "STRAIGHT"
    if node.stop_line_s is not None:
        return "STOP"
    return "FOLLOW"


def _semantic_for_step(edge: LaneEdge | None, node: LaneNode) -> str:
    if edge is not None and edge.kind.startswith("lane_change"):
        return "lane_change"
    if edge is not None and edge.kind == "junction" or node.junction_id:
        return "junction"
    if node.stop_line_s is not None or node.signal_ids:
        return "regulated_road"
    return "road"


class RoutePlanner:
    """Multi-objective A* over the lane graph with deterministic tie breaks."""

    def __init__(self, graph: LaneGraph) -> None:
        self.graph = graph

    def plan(self, request: RouteRequest) -> RouteResult:
        if request.start_node_id not in self.graph.nodes:
            raise RouteError(f"unknown start_node_id {request.start_node_id}")
        goals = []
        for goal in request.goal_node_ids:
            if goal not in self.graph.nodes:
                raise RouteError(f"unknown goal_node_id {goal}")
            goals.append(goal)
        forbidden = set(request.forbidden_node_ids)
        if request.start_node_id in forbidden:
            return RouteResult(
                ok=False,
                corridor=None,
                expansions=0,
                failure_code="START_FORBIDDEN",
                failure_node_id=request.start_node_id,
                message="start node is forbidden",
            )

        goal_set = set(goals)
        # Heuristic: shortest remaining length using uniform lower bound.
        min_edge = min((edge.cost for edge in self.graph.edges.values()), default=1.0)
        min_edge = max(0.1, min_edge)

        def heuristic(node_id: str) -> float:
            # Deterministic admissible-ish heuristic using graph distance proxy:
            # zero for goals, otherwise min edge cost. Keeps planning stable offline.
            return 0.0 if node_id in goal_set else min_edge

        # priority: (f, h, tie_seed_component, node_id, g, path_nodes, path_edges)
        counter = 0
        start_h = heuristic(request.start_node_id)
        open_heap: list[tuple[float, float, int, str, float, tuple[str, ...], tuple[str, ...]]] = []
        heapq.heappush(
            open_heap,
            (
                start_h,
                start_h,
                self._tie(request.seed, request.start_node_id, counter),
                request.start_node_id,
                0.0,
                (request.start_node_id,),
                tuple(),
            ),
        )
        best_g: dict[str, float] = {request.start_node_id: 0.0}
        expansions = 0

        while open_heap:
            f_cost, h_cost, _tie, node_id, g_cost, path_nodes, path_edges = heapq.heappop(open_heap)
            del f_cost, h_cost
            expansions += 1
            if expansions > request.max_expansions:
                return RouteResult(
                    ok=False,
                    corridor=None,
                    expansions=expansions,
                    failure_code="MAX_EXPANSIONS",
                    failure_node_id=node_id,
                    message="search exceeded max_expansions",
                )
            if node_id in goal_set:
                return RouteResult(
                    ok=True,
                    corridor=self._build_corridor(request, node_id, path_nodes, path_edges, g_cost),
                    expansions=expansions,
                )

            for edge in self._candidate_edges(node_id, request):
                if edge.target in forbidden:
                    continue
                step_cost = self._edge_cost(edge, request)
                candidate_g = g_cost + step_cost
                previous = best_g.get(edge.target)
                if previous is not None and candidate_g >= previous - 1e-9:
                    continue
                best_g[edge.target] = candidate_g
                counter += 1
                next_h = heuristic(edge.target)
                heapq.heappush(
                    open_heap,
                    (
                        candidate_g + next_h,
                        next_h,
                        self._tie(request.seed, edge.target, counter),
                        edge.target,
                        candidate_g,
                        path_nodes + (edge.target,),
                        path_edges + (edge.edge_id,),
                    ),
                )

        return RouteResult(
            ok=False,
            corridor=None,
            expansions=expansions,
            failure_code="UNREACHABLE",
            failure_node_id=request.start_node_id,
            message="no legal corridor to any goal",
        )

    def _candidate_edges(self, node_id: str, request: RouteRequest) -> list[LaneEdge]:
        edges = []
        for edge in self.graph.successors(node_id):
            if edge.kind.startswith("lane_change") and not request.allow_lane_change:
                continue
            if edge.kind == "junction" and not request.allow_junction:
                continue
            edges.append(edge)
        # Deterministic expansion order.
        edges.sort(key=lambda item: (item.kind, item.target, item.edge_id))
        return edges

    def _edge_cost(self, edge: LaneEdge, request: RouteRequest) -> float:
        node = self.graph.nodes[edge.target]
        cost = float(edge.cost)
        if edge.kind.startswith("lane_change"):
            cost += request.lane_change_penalty
        if edge.kind == "junction" or node.junction_id:
            cost += request.junction_penalty
        # Prefer higher speed corridors slightly (lower cost).
        cost += request.speed_weight * max(0.0, 20.0 - node.speed_limit_mps)
        return cost

    @staticmethod
    def _tie(seed: int, node_id: str, counter: int) -> int:
        digest = hashlib.sha256(f"{seed}:{node_id}:{counter}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _build_corridor(
        self,
        request: RouteRequest,
        goal_node_id: str,
        path_nodes: Sequence[str],
        path_edges: Sequence[str],
        total_cost: float,
    ) -> RouteCorridor:
        steps: list[CorridorStep] = []
        lane_change_count = 0
        junction_count = 0
        road_sequence: list[str] = []

        for index, node_id in enumerate(path_nodes):
            node = self.graph.nodes[node_id]
            edge = self.graph.edges[path_edges[index - 1]] if index > 0 else None
            if edge is not None and edge.kind.startswith("lane_change"):
                lane_change_count += 1
            if (edge is not None and edge.kind == "junction") or node.junction_id:
                junction_count += 1
            if not road_sequence or road_sequence[-1] != node.road_id:
                road_sequence.append(node.road_id)
            cost_from_start = 0.0
            if index > 0:
                # Reconstruct cumulative cost approximately from step list length * edge costs.
                # Exact g is known only at goal; recompute prefix costs for auditability.
                cost_from_start = 0.0
                running = 0.0
                for edge_id in path_edges[:index]:
                    running += self._edge_cost(self.graph.edges[edge_id], request)
                cost_from_start = running
            steps.append(
                CorridorStep(
                    node_id=node.node_id,
                    road_id=node.road_id,
                    lane_id=node.lane_id,
                    edge_id=None if edge is None else edge.edge_id,
                    maneuver=_maneuver_for_edge(edge, node),
                    semantic=_semantic_for_step(edge, node),
                    cost_from_start=cost_from_start,
                    speed_limit_mps=node.speed_limit_mps,
                    junction_id=node.junction_id,
                )
            )

        # Ensure final total matches planner g.
        if steps:
            final_steps = list(steps)
            final_steps[-1] = CorridorStep(
                **{
                    **steps[-1].to_dict(),
                    "cost_from_start": total_cost,
                }
            )
            # dataclass rebuild
            steps = [
                CorridorStep(
                    node_id=item["node_id"] if isinstance(item, dict) else item.node_id,
                    road_id=item["road_id"] if isinstance(item, dict) else item.road_id,
                    lane_id=item["lane_id"] if isinstance(item, dict) else item.lane_id,
                    edge_id=item["edge_id"] if isinstance(item, dict) else item.edge_id,
                    maneuver=item["maneuver"] if isinstance(item, dict) else item.maneuver,
                    semantic=item["semantic"] if isinstance(item, dict) else item.semantic,
                    cost_from_start=item["cost_from_start"] if isinstance(item, dict) else item.cost_from_start,
                    speed_limit_mps=item["speed_limit_mps"] if isinstance(item, dict) else item.speed_limit_mps,
                    junction_id=item["junction_id"] if isinstance(item, dict) else item.junction_id,
                )
                for item in final_steps
            ]

        route_id = _route_id(self.graph.map_hash, request, goal_node_id, path_nodes)
        semantics = {
            "contains_road": any(step.semantic == "road" for step in steps),
            "contains_lane_change": lane_change_count > 0,
            "contains_junction": junction_count > 0,
            "oracle_inputs": ["lane_graph", "signal_topology", "speed_limits", "multi_goal_set"],
            "observable_inputs": ["start_node", "goal_nodes", "forbidden_nodes", "route_progress"],
            "maneuvers": [step.maneuver for step in steps],
        }
        return RouteCorridor(
            route_id=route_id,
            map_name=self.graph.map_name,
            map_hash=self.graph.map_hash,
            start_node_id=request.start_node_id,
            goal_node_id=goal_node_id,
            seed=request.seed,
            steps=tuple(steps),
            total_cost=total_cost,
            road_sequence=tuple(road_sequence),
            lane_change_count=lane_change_count,
            junction_count=junction_count,
            semantics=semantics,
        )
