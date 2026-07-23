#!/usr/bin/env python3
"""Live CARLA classic expert with complex routes (turns / U-turn / junctions).

- Optional one-shot map load via --load-map (then never switch again).
- Complex route: waypoint-graph A* between distant spawns with yaw change.
- Maneuver tags: STRAIGHT / TURN / U_TURN / JUNCTION.
- Planning: Frenet (normal) + Hybrid A* (U-turn / reverse / blocked).
- Control: G1-06 ControlLoop @ 50Hz + spectator chase-cam.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import carla
except ImportError:  # offline unit tests only need helpers (StuckWatch, cruise, CTE)
    carla = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RouteNode:
    """Unified path node (waypoint or synthetic U-turn sample)."""

    x: float
    y: float
    yaw: float  # radians
    segment: str  # approach | uturn | longtail
    reverse: bool = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.behavior import BehaviorEvent, BehaviorStateMachine  # noqa: E402
from classic_stack.control import load_control_config  # noqa: E402
from classic_stack.control.adaptation import RaceControlLoop  # noqa: E402
from classic_stack.control.controller import ControlLoop, EgoState  # noqa: E402
from classic_stack.geometry import ReferencePath  # noqa: E402
from classic_stack.planning.frenet import FrenetPlanner, load_frenet_st_config  # noqa: E402
from classic_stack.planning.frenet.planner import PlanRequest, Trajectory, TrajectoryPoint  # noqa: E402
from classic_stack.planning.hybrid_astar import HybridAstarPlanner, load_hybrid_astar_config  # noqa: E402
from classic_stack.planning.hybrid_astar.planner import ManeuverRequest, ObstacleMap  # noqa: E402
from classic_stack.risk import evaluate_risk_field  # noqa: E402
from runtime import (  # noqa: E402
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402


def _map_basename(name: str) -> str:
    return name.split("/")[-1].replace("_Opt", "")


def set_spectator_follow(world: carla.World, vehicle: carla.Vehicle) -> None:
    try:
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        yaw_rad = math.radians(rotation.yaw)
        cam_loc = carla.Location(
            location.x - 7.0 * math.cos(yaw_rad),
            location.y - 7.0 * math.sin(yaw_rad),
            location.z + 3.0,
        )
        spectator.set_transform(
            carla.Transform(cam_loc, carla.Rotation(pitch=-14.0, yaw=rotation.yaw, roll=0.0))
        )
    except Exception:
        pass


def ego_state_from_actor(actor) -> EgoState:
    tf = actor.get_transform()
    vel = actor.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    ctrl = actor.get_control()
    return EgoState(
        x=float(tf.location.x),
        y=float(tf.location.y),
        yaw=math.radians(float(tf.rotation.yaw)),
        v=float(speed),
        steer=float(ctrl.steer) * 0.6,
    )


def draw_route(world: carla.World, locs: list[carla.Location], color: carla.Color, life: float = 90.0) -> None:
    debug = world.debug
    for i in range(len(locs) - 1):
        a = locs[i] + carla.Location(z=0.5)
        b = locs[i + 1] + carla.Location(z=0.5)
        debug.draw_line(a, b, thickness=0.2, color=color, life_time=life)
    if locs:
        debug.draw_string(locs[-1] + carla.Location(z=1.5), "GOAL", False, carla.Color(255, 40, 40), life)


def draw_segmented_route(world: carla.World, nodes: list[RouteNode], life: float = 90.0) -> None:
    """Green=approach, magenta=U-turn, cyan=long-tail."""

    colors = {
        "approach": carla.Color(40, 220, 80),
        "uturn": carla.Color(255, 40, 220),
        "longtail": carla.Color(40, 180, 255),
    }
    debug = world.debug
    for i in range(len(nodes) - 1):
        a = carla.Location(nodes[i].x, nodes[i].y, 0.5)
        b = carla.Location(nodes[i + 1].x, nodes[i + 1].y, 0.5)
        col = colors.get(nodes[i].segment, carla.Color(40, 220, 80))
        debug.draw_line(a, b, thickness=0.25, color=col, life_time=life)
    if nodes:
        # Labels for each segment start
        seen: set[str] = set()
        for n in nodes:
            if n.segment in seen:
                continue
            seen.add(n.segment)
            label = {"approach": "START", "uturn": "U-TURN", "longtail": "LONGTAIL"}.get(n.segment, n.segment)
            debug.draw_string(
                carla.Location(n.x, n.y, 2.0),
                label,
                False,
                colors.get(n.segment, carla.Color(255, 255, 255)),
                life,
            )
        last = nodes[-1]
        debug.draw_string(
            carla.Location(last.x, last.y, 2.0),
            "GOAL",
            False,
            carla.Color(255, 40, 40),
            life,
        )


def waypoint_neighbors(wp: carla.Waypoint, step: float = 2.5) -> list[carla.Waypoint]:
    out: list[carla.Waypoint] = []
    out.extend(wp.next(step))
    # explore junctions / lane changes
    for lane in (wp.get_left_lane(), wp.get_right_lane()):
        if lane is None:
            continue
        if lane.lane_type != carla.LaneType.Driving:
            continue
        if lane.lane_id * wp.lane_id < 0:
            continue  # opposite direction
        out.extend(lane.next(step))
    # at junction, try all next options with smaller step
    if wp.is_junction:
        out.extend(wp.next(1.5))
    return out


def complex_route_astar(
    carla_map: carla.Map,
    start_tf: carla.Transform,
    goal_tf: carla.Transform,
    *,
    max_expansions: int = 8000,
) -> list[carla.Waypoint]:
    """A* on driving waypoints between two transforms."""

    start = carla_map.get_waypoint(start_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    goal = carla_map.get_waypoint(goal_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if start is None or goal is None:
        raise RuntimeError("cannot project start/goal to road")

    def key(wp: carla.Waypoint) -> int:
        return int(wp.id)

    def h(wp: carla.Waypoint) -> float:
        return wp.transform.location.distance(goal.transform.location)

    open_h: list[tuple[float, int, int]] = []  # f, counter, wp_id
    counter = 0
    gscore = {key(start): 0.0}
    came: dict[int, int | None] = {key(start): None}
    id_to_wp = {key(start): start}
    heapq.heappush(open_h, (h(start), counter, key(start)))
    expansions = 0
    found = None

    while open_h and expansions < max_expansions:
        _, _, cid = heapq.heappop(open_h)
        cur = id_to_wp[cid]
        expansions += 1
        if cur.transform.location.distance(goal.transform.location) < 4.0:
            found = cid
            # link goal
            if key(goal) not in came:
                came[key(goal)] = cid
                id_to_wp[key(goal)] = goal
                found = key(goal)
            break
        for nxt in waypoint_neighbors(cur):
            nid = key(nxt)
            step = cur.transform.location.distance(nxt.transform.location)
            # prefer turns at junctions slightly (long-tail coverage), small penalty otherwise
            turn_cost = 0.0
            dyaw = abs(
                (nxt.transform.rotation.yaw - cur.transform.rotation.yaw + 180) % 360 - 180
            )
            if dyaw > 25:
                turn_cost = 0.5  # still allow; small cost so A* takes them when useful
            if nxt.is_junction:
                turn_cost *= 0.5
            ng = gscore[cid] + step + turn_cost
            if ng < gscore.get(nid, 1e18):
                gscore[nid] = ng
                came[nid] = cid
                id_to_wp[nid] = nxt
                counter += 1
                heapq.heappush(open_h, (ng + h(nxt), counter, nid))

    if found is None:
        raise RuntimeError(f"complex A* failed expansions={expansions}")

    chain: list[carla.Waypoint] = []
    cur_id: int | None = found
    guard = 0
    while cur_id is not None and guard < 20000:
        chain.append(id_to_wp[cur_id])
        cur_id = came.get(cur_id)
        guard += 1
    chain.reverse()
    return chain


def route_complexity(wps: list[carla.Waypoint]) -> dict:
    length = 0.0
    turn_deg = 0.0
    junctions = 0
    max_dyaw = 0.0
    for i in range(1, len(wps)):
        length += wps[i - 1].transform.location.distance(wps[i].transform.location)
        dyaw = abs(
            (wps[i].transform.rotation.yaw - wps[i - 1].transform.rotation.yaw + 180) % 360 - 180
        )
        turn_deg += dyaw
        max_dyaw = max(max_dyaw, dyaw)
        if wps[i].is_junction:
            junctions += 1
    start_yaw = wps[0].transform.rotation.yaw
    end_yaw = wps[-1].transform.rotation.yaw
    net_yaw = abs((end_yaw - start_yaw + 180) % 360 - 180)
    tags = []
    if junctions >= 1:
        tags.append("JUNCTION")
    if max_dyaw > 40 or turn_deg > 60:
        tags.append("TURN")
    if net_yaw > 140 or turn_deg > 150:
        tags.append("U_TURN_OR_LOOP")
    if length > 120:
        tags.append("LONG")
    if not tags:
        tags.append("SIMPLE")
    return {
        "length_m": length,
        "turn_deg_sum": turn_deg,
        "max_segment_dyaw": max_dyaw,
        "net_yaw_change": net_yaw,
        "junctions": junctions,
        "waypoints": len(wps),
        "tags": tags,
    }


def wp_to_node(wp: carla.Waypoint, segment: str) -> RouteNode:
    loc = wp.transform.location
    return RouteNode(
        x=float(loc.x),
        y=float(loc.y),
        yaw=math.radians(float(wp.transform.rotation.yaw)),
        segment=segment,
        reverse=False,
    )


def follow_road(
    start_wp: carla.Waypoint,
    distance_m: float,
    *,
    step: float = 2.5,
    prefer_turn: bool = True,
    rng: random.Random | None = None,
) -> list[carla.Waypoint]:
    """Walk driving graph forward; at junctions prefer larger heading change (long-tail)."""

    out: list[carla.Waypoint] = []
    cur = start_wp
    traveled = 0.0
    guard = 0
    while traveled < distance_m and guard < 400:
        guard += 1
        opts = list(cur.next(step))
        if not opts:
            break
        if prefer_turn and len(opts) > 1:

            def _dyaw(n: carla.Waypoint) -> float:
                return abs((n.transform.rotation.yaw - cur.transform.rotation.yaw + 180.0) % 360.0 - 180.0)

            ranked = sorted(opts, key=_dyaw, reverse=True)
            if rng is not None and rng.random() < 0.75:
                nxt = ranked[0]
            else:
                nxt = opts[0]
        else:
            nxt = opts[0]
        out.append(nxt)
        traveled += cur.transform.location.distance(nxt.transform.location)
        cur = nxt
    return out


def clear_world_npcs(world: carla.World, *, keep_ids: set[int] | None = None) -> dict:
    """Destroy all vehicles/walkers/controllers except keep_ids. Path-follow only."""

    keep = set(keep_ids or ())
    destroyed = {"vehicle": 0, "walker": 0, "controller": 0}
    actors = world.get_actors()
    for pattern, key in (
        ("vehicle.*", "vehicle"),
        ("walker.*", "walker"),
        ("controller.*", "controller"),
    ):
        for actor in actors.filter(pattern):
            try:
                aid = int(actor.id)
            except Exception:
                continue
            if aid in keep:
                continue
            try:
                actor.destroy()
                destroyed[key] += 1
            except Exception:
                pass
    return destroyed


def road_uturn_wps(
    start_wp: carla.Waypoint,
    *,
    max_steps: int = 120,
    step: float = 2.0,
) -> list[carla.Waypoint]:
    """On-road U-turn: stay on driving graph, greedily reverse heading (~180°).

    Avoids free-space arcs that cut into sidewalks / lamp posts / parked props.
    """

    start_yaw = start_wp.transform.rotation.yaw
    path: list[carla.Waypoint] = []
    cur = start_wp
    best_path: list[carla.Waypoint] = []
    best_net = 0.0
    for _ in range(max_steps):
        opts = list(cur.next(step))
        if not opts:
            # try slightly larger step at dead-ends
            opts = list(cur.next(step + 1.0))
        if not opts:
            break

        def _score(n: carla.Waypoint) -> float:
            net = abs((n.transform.rotation.yaw - start_yaw + 180.0) % 360.0 - 180.0)
            imm = abs((n.transform.rotation.yaw - cur.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
            # prefer junction options (more likely true turnaround)
            j = 8.0 if n.is_junction else 0.0
            return net + 0.35 * imm + j

        nxt = max(opts, key=_score)
        path.append(nxt)
        cur = nxt
        net = abs((cur.transform.rotation.yaw - start_yaw + 180.0) % 360.0 - 180.0)
        if net > best_net:
            best_net = net
            best_path = list(path)
        if net >= 150.0 and len(path) >= 10:
            return path
    # accept best partial turnaround if we got a real heading flip
    if best_net >= 120.0 and len(best_path) >= 8:
        return best_path
    return path


def path_length_m(nodes: list[RouteNode]) -> float:
    total = 0.0
    for i in range(1, len(nodes)):
        total += math.hypot(nodes[i].x - nodes[i - 1].x, nodes[i].y - nodes[i - 1].y)
    return total


def route_complexity_nodes(nodes: list[RouteNode]) -> dict:
    length = path_length_m(nodes)
    turn_deg = 0.0
    max_dyaw = 0.0
    for i in range(1, len(nodes)):
        dyaw = abs((math.degrees(nodes[i].yaw - nodes[i - 1].yaw) + 180.0) % 360.0 - 180.0)
        turn_deg += dyaw
        max_dyaw = max(max_dyaw, dyaw)
    net_yaw = 0.0
    if nodes:
        net_yaw = abs((math.degrees(nodes[-1].yaw - nodes[0].yaw) + 180.0) % 360.0 - 180.0)
    segs = {n.segment for n in nodes}
    tags = []
    if "uturn" in segs:
        tags.append("U_TURN")
    if turn_deg > 60 or max_dyaw > 40:
        tags.append("TURN")
    if length > 120:
        tags.append("LONG")
    if "longtail" in segs:
        tags.append("LONGTAIL")
    if "approach" in segs:
        tags.append("APPROACH")
    if not tags:
        tags.append("SIMPLE")
    return {
        "length_m": length,
        "turn_deg_sum": turn_deg,
        "max_segment_dyaw": max_dyaw,
        "net_yaw_change": net_yaw,
        "junctions": sum(1 for n in nodes if n.segment == "longtail") // 8,
        "waypoints": len(nodes),
        "tags": tags,
        "segments": {
            s: sum(1 for n in nodes if n.segment == s) for s in ("approach", "uturn", "longtail")
        },
    }


def build_full_demo_route(
    world: carla.World,
    rng: random.Random,
) -> tuple[list[RouteNode], dict, carla.Transform]:
    """Full show route: approach (turns) → explicit U-turn → long multi-turn tail."""

    carla_map = world.get_map()
    spawns = list(carla_map.get_spawn_points())
    if len(spawns) < 2:
        raise RuntimeError("map has fewer than 2 spawn points")
    rng.shuffle(spawns)

    best: tuple[list[RouteNode], dict, carla.Transform] | None = None
    best_score = -1.0

    for start_tf in spawns[: min(24, len(spawns))]:
        start_wp = carla_map.get_waypoint(
            start_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if start_wp is None:
            continue
        # --- approach: ~45–70 m, prefer junctions/turns ---
        approach_wps = follow_road(
            start_wp, rng.uniform(45.0, 70.0), prefer_turn=True, rng=rng
        )
        if len(approach_wps) < 8:
            continue
        approach_nodes = [wp_to_node(start_wp, "approach")] + [
            wp_to_node(w, "approach") for w in approach_wps
        ]
        # --- on-road U-turn (no free-space arc → no lamp-post collisions) ---
        uturn_wps = road_uturn_wps(approach_wps[-1])
        if len(uturn_wps) < 10:
            continue
        uturn_nodes = [wp_to_node(w, "uturn") for w in uturn_wps]
        start_yaw = approach_wps[-1].transform.rotation.yaw
        end_yaw = uturn_wps[-1].transform.rotation.yaw
        net_u = abs((end_yaw - start_yaw + 180.0) % 360.0 - 180.0)
        if net_u < 120.0:
            continue  # not a real U-turn / turnaround
        after_wp = uturn_wps[-1]
        after_tf = after_wp.transform
        # Prefer goal: far + heading difference + not too close to U-turn
        candidates = []
        for g in spawns:
            d = after_tf.location.distance(g.location)
            if d < 90.0:
                continue
            yaw_gap = abs((after_tf.rotation.yaw - g.rotation.yaw + 180.0) % 360.0 - 180.0)
            candidates.append((g, d + 0.35 * yaw_gap))
        if not candidates:
            continue
        candidates.sort(key=lambda t: t[1], reverse=True)
        leg_ok = False
        long_nodes: list[RouteNode] = []
        for goal_tf, _ in candidates[:12]:
            try:
                leg_wps = complex_route_astar(carla_map, after_tf, goal_tf, max_expansions=14000)
            except Exception:
                continue
            if len(leg_wps) < 35:
                continue
            meta_leg = route_complexity(leg_wps)
            if meta_leg["turn_deg_sum"] < 50 and meta_leg["junctions"] < 2:
                continue
            long_nodes = [wp_to_node(w, "longtail") for w in leg_wps]
            leg_ok = True
            break
        if not leg_ok:
            # fallback: farthest spawn
            goal_tf = max(spawns, key=lambda t: after_tf.location.distance(t.location))
            try:
                leg_wps = complex_route_astar(carla_map, after_tf, goal_tf, max_expansions=15000)
                long_nodes = [wp_to_node(w, "longtail") for w in leg_wps]
            except Exception:
                continue

        nodes = approach_nodes + uturn_nodes + long_nodes
        meta = route_complexity_nodes(nodes)
        if "U_TURN" not in meta["tags"]:
            continue
        score = (
            meta["length_m"] * 0.3
            + meta["turn_deg_sum"]
            + 80.0  # guaranteed U-turn bonus
            + 0.4 * meta["net_yaw_change"]
        )
        if score > best_score:
            best_score = score
            spawn_tf = start_wp.transform
            spawn_tf.location.z = max(spawn_tf.location.z, 0.5)
            best = (nodes, meta, spawn_tf)
            if meta["length_m"] >= 180 and meta["turn_deg_sum"] >= 200:
                break

    if best is None:
        raise RuntimeError("failed to build full demo route with U-turn")
    return best


def nodes_to_trajectory(
    nodes: list[RouteNode],
    *,
    cruise_mps: float = 6.5,
    uturn_mps: float = 3.2,
    traj_id: str = "full-demo",
) -> Trajectory:
    """Dense geometric path → timed trajectory for ControlLoop (MPC) tracking."""

    pts: list[TrajectoryPoint] = []
    dist = 0.0
    prev: RouteNode | None = None
    for n in nodes:
        if prev is not None:
            dist += math.hypot(n.x - prev.x, n.y - prev.y)
        v = uturn_mps if n.segment == "uturn" else cruise_mps
        if n.reverse:
            v = -abs(v)
        if prev is None:
            t = 0.0
        else:
            seg_v = uturn_mps if n.segment == "uturn" or prev.segment == "uturn" else cruise_mps
            seg_v = max(1.5, abs(seg_v))
            t = pts[-1].t + math.hypot(n.x - prev.x, n.y - prev.y) / seg_v
        pts.append(
            TrajectoryPoint(
                t=t,
                x=n.x,
                y=n.y,
                yaw=n.yaw,
                kappa=0.0,
                v=abs(v),
                a=0.0,
                jerk=0.0,
            )
        )
        prev = n
    if len(pts) < 2:
        raise RuntimeError("need >=2 route nodes")
    return Trajectory(points=tuple(pts), trajectory_id=traj_id, source="full_demo_dense")


def plan_along_nodes(nodes: list[RouteNode], *, cruise_mps: float = 6.5) -> tuple[Trajectory, dict]:
    if len(nodes) < 4:
        raise RuntimeError("route too short")
    traj = nodes_to_trajectory(nodes, cruise_mps=cruise_mps)
    return traj, {
        "planner": "waypoint_dense",
        "ok": True,
        "nodes": len(nodes),
        "segments": {s: sum(1 for n in nodes if n.segment == s) for s in ("approach", "uturn", "longtail")},
        "modules": ["waypoint_dense"],
    }


def nodes_to_reference(nodes: list[RouteNode]) -> ReferencePath:
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    yaws = [n.yaw for n in nodes]
    return ReferencePath.from_xy(xs, ys, yaws)


def plan_frenet_on_nodes(
    frenet: FrenetPlanner,
    nodes: list[RouteNode],
    ego: EgoState,
    *,
    seed: int,
    cruise_mps: float = 6.0,
    scenario_kind: str = "follow",
) -> tuple[Trajectory | None, dict]:
    """True G1-04 Frenet+ST on the remaining route polyline (not waypoint timing)."""

    if len(nodes) < 4:
        return None, {"planner": "frenet_st", "ok": False, "failure": "ROUTE_TOO_SHORT"}
    ref = nodes_to_reference(nodes)
    s0, d0 = ref.project(ego.x, ego.y)
    s0 = max(0.0, min(s0, max(0.0, ref.length - 8.0)))
    if abs(d0) > 3.5:
        d0 = 0.0
    req = PlanRequest(
        reference=ref,
        v0=max(0.8, abs(ego.v)),
        s0=s0,
        d0=d0,
        scenario_kind=scenario_kind,
        target_speed_mps=cruise_mps,
        seed=seed,
    )
    result = frenet.plan(req)
    meta = {
        "planner": "frenet_st",
        "ok": result.ok,
        "failure": result.failure_code,
        "candidates": result.candidates,
        "ms": result.wall_time_ms,
        "reject": result.reject_reasons,
        "modules": ["FrenetPlanner", "ST-DP"],
        "config_hash": result.config_hash,
    }
    if result.ok and result.trajectory is not None:
        return result.trajectory, meta
    return None, meta


def plan_hybrid_local(
    hybrid: HybridAstarPlanner,
    ego: EgoState,
    goal_node: RouteNode,
    *,
    seed: int,
) -> tuple[Trajectory | None, dict]:
    """G1-05 Hybrid A* local maneuver toward a nearby route node (e.g. U-turn segment)."""

    world = ObstacleMap(
        xmin=ego.x - 45,
        xmax=ego.x + 45,
        ymin=ego.y - 45,
        ymax=ego.y + 45,
        boxes=(),
    )
    req = ManeuverRequest(
        start=(ego.x, ego.y, ego.yaw),
        goal=(goal_node.x, goal_node.y, goal_node.yaw),
        world=world,
        require_reverse=goal_node.segment == "uturn",
        seed=seed,
    )
    result = hybrid.plan(req)
    meta = {
        "planner": "hybrid_astar",
        "ok": result.ok,
        "partial": result.partial,
        "failure": result.failure_code,
        "nodes": result.nodes_expanded,
        "analytic_hits": result.analytic_hits,
        "ms": result.wall_time_ms,
        "gears": result.gear_switches,
        "modules": ["HybridAstarPlanner", "ReedsShepp"],
        "config_hash": result.config_hash,
    }
    if result.ok and result.trajectory is not None:
        return result.trajectory, meta
    return None, meta


def segment_cruise_mps(segment: str, *, approach: float = 5.5, uturn: float = 3.0, longtail: float = 5.5) -> float:
    if segment == "uturn":
        return uturn
    if segment == "longtail":
        return longtail
    return approach


def route_cross_track(nodes: list[RouteNode], x: float, y: float) -> float:
    """Signed lateral distance to nearest route segment (left +)."""

    best_d = 0.0
    best = 1e18
    for i in range(len(nodes) - 1):
        x0, y0 = nodes[i].x, nodes[i].y
        x1, y1 = nodes[i + 1].x, nodes[i + 1].y
        vx, vy = x1 - x0, y1 - y0
        seg2 = vx * vx + vy * vy
        if seg2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((x - x0) * vx + (y - y0) * vy) / seg2))
        px, py = x0 + t * vx, y0 + t * vy
        dist = math.hypot(x - px, y - py)
        if dist < best:
            best = dist
            cross = vx * (y - y0) - vy * (x - x0)
            best_d = math.copysign(dist, cross)
    return best_d


class StuckWatch:
    """Detect no-progress stall (v low and route progress frozen)."""

    def __init__(
        self,
        *,
        v_max: float = 0.35,
        prog_eps: float = 0.4,
        hold_s: float = 4.0,
        grace_s: float = 3.0,
    ) -> None:
        self.v_max = v_max
        self.prog_eps = prog_eps
        self.hold_s = hold_s
        self.grace_s = grace_s
        self._t0: float | None = None
        self._since: float | None = None
        self._last_prog = 0.0

    def update(self, *, now_s: float, v: float, progress_m: float) -> bool:
        if self._t0 is None:
            self._t0 = now_s
            self._last_prog = progress_m
            return False
        if now_s - self._t0 < self.grace_s:
            self._last_prog = progress_m
            return False
        moved = progress_m - self._last_prog
        if v < self.v_max and moved < self.prog_eps:
            if self._since is None:
                self._since = now_s
            elif now_s - self._since >= self.hold_s:
                return True
        else:
            self._since = None
            self._last_prog = progress_m
        return False


def plan_stack(
    stack: str,
    *,
    frenet: FrenetPlanner,
    hybrid: HybridAstarPlanner,
    nodes: list[RouteNode],
    ego: EgoState,
    seed: int,
    segment: str,
    cruise_mps: float = 6.0,
    uturn_planner: str = "dense",
) -> tuple[Trajectory, dict]:
    """Select planner by stack flag. Always records actual modules used.

    stacks:
      waypoint — dense route timing only (demo bypass)
      basic    — Frenet+ST on route, dense fallback
      full     — Frenet+ST; uturn via dense|hybrid|frenet; RaceControl elsewhere
    """

    stack = stack.lower().strip()
    uturn_planner = uturn_planner.lower().strip()
    modules: list[str] = []
    stats = {
        "hybrid_calls": 0,
        "hybrid_ok": 0,
        "rs_analytic_hits": 0,
        "rs_trajectory_executed": False,
        "fallback": False,
    }

    if stack == "waypoint":
        traj, meta = plan_along_nodes(nodes, cruise_mps=cruise_mps)
        return traj, {**meta, **stats}

    # U-turn: default dense (stable, on-road); hybrid optional
    if segment == "uturn" and len(nodes) >= 4:
        if uturn_planner == "dense" or (stack != "full" and uturn_planner != "frenet"):
            traj, meta = plan_along_nodes(nodes, cruise_mps=cruise_mps)
            meta = {
                **meta,
                **stats,
                "planner": "waypoint_dense_uturn",
                "modules": ["waypoint_dense_uturn"],
                "stack": stack,
                "segment": segment,
                "uturn_planner": "dense",
            }
            return traj, meta
        if uturn_planner == "hybrid" and stack == "full":
            goal_i = min(len(nodes) - 1, 12)
            stats["hybrid_calls"] = 1
            h_traj, h_meta = plan_hybrid_local(hybrid, ego, nodes[goal_i], seed=seed)
            modules.extend(h_meta.get("modules", []))
            stats["rs_analytic_hits"] = int(h_meta.get("nodes") is not None and h_meta.get("ok"))
            # analytic hits from planner result field if present
            if h_meta.get("ok"):
                stats["hybrid_ok"] = 1
            # Prefer storing analytic from hybrid meta — re-read if we add field
            if h_traj is not None:
                stats["rs_trajectory_executed"] = True
                stats["rs_analytic_hits"] = int(h_meta.get("analytic_hits") or 0)
                stats["hybrid_ok"] = 1
                h_meta = {
                    **h_meta,
                    **stats,
                    "stack": stack,
                    "segment": segment,
                    "uturn_planner": "hybrid",
                    "modules": modules,
                }
                return h_traj, h_meta

    f_traj, f_meta = plan_frenet_on_nodes(
        frenet, nodes, ego, seed=seed, cruise_mps=cruise_mps, scenario_kind="follow"
    )
    modules.extend(f_meta.get("modules", []))
    if f_traj is not None:
        f_meta = {
            **f_meta,
            **stats,
            "stack": stack,
            "segment": segment,
            "modules": modules,
            "uturn_planner": uturn_planner if segment == "uturn" else None,
        }
        return f_traj, f_meta

    traj, meta = plan_along_nodes(nodes, cruise_mps=cruise_mps)
    stats["fallback"] = True
    meta = {
        **meta,
        **stats,
        "planner": "waypoint_dense_fallback",
        "stack": stack,
        "segment": segment,
        "frenet_fail": f_meta.get("failure"),
        "modules": modules + ["waypoint_dense_fallback"],
        "ok": True,
        "fallback": True,
    }
    return traj, meta


def to_carla_control(
    cmd,
    control_cfg,
    state: EgoState,
    *,
    segment: str = "approach",
    cross_track: float = 0.0,
    uturn_throttle_cap: float = 0.28,
) -> Any:
    if carla is None:
        raise RuntimeError("carla not available")
    c = carla.VehicleControl()
    c.steer = float(max(-1.0, min(1.0, cmd.steer / max(control_cfg.max_steer_rad, 1e-3))))
    thr = float(cmd.throttle)
    reverse = bool(getattr(cmd, "reverse", False))
    abs_d = abs(cross_track)
    soft = 0.6 if segment == "uturn" else 0.8
    hard = 1.2 if segment == "uturn" else 1.6

    if cmd.mode != "brake" and not reverse:
        if segment == "uturn":
            # No aggressive kick on U-turn (prevents plow into curbs/lamps)
            if state.v < 0.8:
                thr = max(thr, 0.35)
            thr = min(thr, uturn_throttle_cap)
        else:
            if state.v < 1.0:
                thr = max(thr, 0.55)
            elif state.v < 4.0:
                thr = max(thr, 0.40)
            elif state.v < 6.0:
                thr = max(thr, 0.28)
        if abs_d > hard:
            thr = min(thr, 0.25)
        elif abs_d > soft:
            thr *= 0.6
    c.throttle = float(max(0.0, min(1.0, thr)))
    c.brake = float(cmd.brake if cmd.mode == "brake" or cmd.brake > 0.05 else 0.0)
    if c.throttle > 0.05:
        c.brake = 0.0
    c.reverse = reverse
    return c


def auto_ticks_for_route(length_m: float, control_period_ms: float, *, hold_s: float = 5.0) -> int:
    """Budget enough ticks to finish whole path + hold at goal (car stays visible)."""

    # ~5.0 m/s average including U-turn slowdown; 1.45× margin for tracking lag
    drive_s = max(40.0, (length_m / 5.0) * 1.45)
    total_s = drive_s + hold_s
    dt = max(control_period_ms / 1000.0, 0.01)
    return int(math.ceil(total_s / dt))


def ensure_map(client: carla.Client, map_token: str, *, force_load_world: bool = False) -> carla.World:
    """Stay on current map by default.

    Runtime ``client.load_world`` frequently triggers UE
    ``shader compilation failures are fatal`` on this host. Prefer cold-start
    with the desired map in ``carla_start.toml`` arguments, then restart Server.
    """

    world = client.get_world()
    cur = _map_basename(world.get_map().name)
    want = _map_basename(map_token)
    if cur == want:
        print(f"already on {world.get_map().name}", flush=True)
        return world
    if not force_load_world:
        print(
            f"WARN: map is {cur}, wanted {want}. Skipping load_world "
            f"(use --force-load-world to override; risk of shader fatal). "
            f"To change map safely: kill CarlaUE4, set carla_start.toml map, sdf sim ensure.",
            flush=True,
        )
        return world
    print(f"FORCE load_world once: {cur} -> {want} (shader fatal risk)...", flush=True)
    client.set_timeout(180.0)
    world = client.load_world(want)
    time.sleep(3.0)
    print(f"now map={world.get_map().name}", flush=True)
    return world


def main() -> int:
    if carla is None:
        print("carla module not installed; live run requires CARLA Python API", flush=True)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="max control ticks; 0 = auto from route length (finish whole path)",
    )
    parser.add_argument(
        "--load-map",
        type=str,
        default="",
        help="desired map name; only loads if --force-load-world (default: stay on current)",
    )
    parser.add_argument(
        "--force-load-world",
        action="store_true",
        help="allow client.load_world (unsafe on this host; may freeze CARLA)",
    )
    parser.add_argument("--replan-every", type=int, default=40)
    parser.add_argument("--hold-s", type=float, default=6.0, help="seconds to keep car at goal before despawn")
    parser.add_argument(
        "--stack",
        type=str,
        default="full",
        choices=("waypoint", "basic", "full"),
        help="waypoint=dense bypass; basic=Frenet+ControlLoop; full=Frenet+Hybrid U-turn+RaceControl",
    )
    parser.add_argument(
        "--max-route-m",
        type=float,
        default=0.0,
        help="if >0, trim route to this length for shorter full-stack demos",
    )
    parser.add_argument(
        "--uturn-planner",
        type=str,
        default="dense",
        choices=("dense", "hybrid", "frenet"),
        help="U-turn segment planner (default dense = on-road stable)",
    )
    parser.add_argument("--stuck-speed", type=float, default=0.35)
    parser.add_argument("--stuck-hold-s", type=float, default=4.0)
    parser.add_argument("--stuck-grace-s", type=float, default=3.0)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(1, 10_000_000)
    rng = random.Random(seed)
    evidence_dir = ROOT / "docs/runtime-evidence/g1-live"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"g1-live-{args.stack}-{seed}-{int(time.time())}"

    print(
        f"=== LIVE classic expert stack={args.stack} "
        f"(approach + U-TURN + longtail) ===",
        flush=True,
    )
    resolver = ConnectionResolver(ROOT, timeout_seconds=15.0)
    report = resolver.preflight()
    if report.status != "READY":
        report = resolver.ensure(startup_timeout_seconds=90.0)
    if report.status != "READY":
        print(json.dumps(report.to_dict(), indent=2))
        return 2

    client, report = resolver.connect(report=report)
    client.set_timeout(30.0)
    if args.load_map:
        world = ensure_map(client, args.load_map, force_load_world=bool(args.force_load_world))
    else:
        world = client.get_world()
        print("stay on current map (no load_world)", flush=True)
    map_name = str(world.get_map().name)
    print(f"map={map_name} seed={seed}", flush=True)

    route_nodes, complexity, start_tf = build_full_demo_route(world, rng)
    if args.max_route_m and args.max_route_m > 40.0:
        # Trim path by arc length for short full-stack demos
        acc = 0.0
        cut = len(route_nodes)
        for i in range(1, len(route_nodes)):
            acc += math.hypot(
                route_nodes[i].x - route_nodes[i - 1].x,
                route_nodes[i].y - route_nodes[i - 1].y,
            )
            if acc >= args.max_route_m:
                cut = i + 1
                break
        route_nodes = route_nodes[: max(cut, 12)]
        complexity = route_complexity_nodes(route_nodes)
        print(f"trimmed route to ~{args.max_route_m}m → {complexity['length_m']:.1f}m", flush=True)
    goal_node = route_nodes[-1]
    goal_loc = carla.Location(goal_node.x, goal_node.y, start_tf.location.z)
    print(f"full route: {json.dumps(complexity)}", flush=True)
    print(
        f"start=({start_tf.location.x:.1f},{start_tf.location.y:.1f}) "
        f"goal=({goal_node.x:.1f},{goal_node.y:.1f}) nodes={len(route_nodes)}",
        flush=True,
    )
    print(
        "legend: GREEN=approach  MAGENTA=U-TURN  CYAN=longtail  RED label=GOAL",
        flush=True,
    )

    try:
        world.get_spectator().set_transform(
            carla.Transform(
                start_tf.location + carla.Location(z=12.0, x=-8.0),
                carla.Rotation(pitch=-35.0, yaw=start_tf.rotation.yaw),
            )
        )
    except Exception:
        pass

    behavior = BehaviorStateMachine(route_id=f"full-{seed}")
    behavior.handle(BehaviorEvent.ROUTE_TURN, timestamp_s=0.0)

    frenet = FrenetPlanner(load_frenet_st_config())
    hybrid = HybridAstarPlanner(load_hybrid_astar_config())
    control_cfg = load_control_config()
    # full stack uses RaceControlLoop (adaptive); basic/waypoint use fixed ControlLoop
    if args.stack == "full":
        control = RaceControlLoop(control_cfg, variant="full")
        control_kind = "RaceControlLoop(full)"
    else:
        control = ControlLoop(control_cfg)
        control_kind = "ControlLoop(fixed)"

    ticks = args.ticks
    if ticks <= 0:
        ticks = auto_ticks_for_route(
            complexity["length_m"], control_cfg.control_period_ms, hold_s=args.hold_s
        )
    hold_ticks = int(args.hold_s / max(control_cfg.control_period_ms / 1000.0, 0.01))
    print(
        f"stack={args.stack} control={control_kind} "
        f"ticks={ticks} (~{ticks * control_cfg.control_period_ms / 1000.0:.0f}s) "
        f"hold={hold_ticks} route_m={complexity['length_m']:.0f}",
        flush=True,
    )

    identity = RunIdentity(
        experiment_id=f"g1-live-stack-{args.stack}",
        run_id=run_id,
        scenario_id=f"{args.stack}-{_map_basename(map_name)}",
        attempt_id=int(seed % 1_000_000),
        server_epoch=str(int(time.time())),
        producer_version="g1-live-stack-3",
        schema_version="safedrive.runtime.v1",
    )
    profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["control_50hz"]
    registry_path = evidence_dir / f"{run_id}-registry.sqlite"
    lease_path = ROOT / ".runtime" / f"tick-lease-{run_id}.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)

    spec = ScenarioSpec(
        scenario_id=identity.scenario_id,
        map_name=map_name,
        actors=(
            ActorSpec(
                name="ego",
                blueprint="vehicle.tesla.model3",
                transform=start_tf,
                role="ego",
                spawn_order=0,
                autopilot=False,
            ),
        ),
        sensors=(),
        traffic_manager_port=8700 + int(seed % 50),
        traffic_manager_seed=seed,
        sensor_timeout_seconds=2.0,
    )
    runtime = ScenarioRuntime(
        client=client,
        identity=identity,
        profile=profile,
        registry=RunRegistry(registry_path),
        lease_path=lease_path,
        owner="sdf.g1-live.full",
    )

    history: list[dict] = []
    plan_log: list[dict] = []
    ok = False
    failure = None
    min_goal_dist = 1e9
    max_route_i = 0
    arrived = False
    hold_left = 0
    traj: Trajectory | None = None
    uturn_seen = False
    modules_used: set[str] = set()
    stuck_watch = StuckWatch(
        v_max=args.stuck_speed,
        hold_s=args.stuck_hold_s,
        grace_s=args.stuck_grace_s,
    )
    race_id_samples = 0
    race_tighten_max = 0.0
    hybrid_calls = 0
    hybrid_ok_count = 0
    rs_analytic_hits_sum = 0
    rs_trajectory_executed = False
    fallback_count = 0
    max_abs_cte = 0.0
    force_replan = False
    prev_ego_for_id: EgoState | None = None
    prev_cmd_for_id = None

    try:
        # Empty scene before spawn — path-follow only, no traffic clutter.
        cleared0 = clear_world_npcs(world)
        print(f"cleared NPCs before spawn: {cleared0}", flush=True)

        runtime.start(spec)
        ego = runtime._actors["ego"]
        # Remove any leftover vehicles/walkers again (keep ego only).
        cleared1 = clear_world_npcs(runtime.world, keep_ids={int(ego.id)})
        print(f"cleared NPCs after spawn (kept ego): {cleared1}", flush=True)

        # Long-lived segmented path (refresh while driving so it never vanishes first)
        draw_life = max(120.0, complexity["length_m"] / 4.0 + 60.0)
        draw_segmented_route(runtime.world, route_nodes, life=draw_life)
        print(
            f"spawned ego id={ego.id}; NO traffic; stack={args.stack}; "
            f"control={control_kind}; run_id={run_id}",
            flush=True,
        )
        set_spectator_follow(runtime.world, ego)
        runtime.tick(carla.VehicleControl(brake=1.0))
        set_spectator_follow(runtime.world, ego)
        runtime.tick(carla.VehicleControl(throttle=0.55, brake=0.0))
        set_spectator_follow(runtime.world, ego)

        state0 = ego_state_from_actor(ego)
        traj, meta = plan_stack(
            args.stack,
            frenet=frenet,
            hybrid=hybrid,
            nodes=route_nodes,
            ego=state0,
            seed=seed,
            segment=route_nodes[0].segment,
            cruise_mps=segment_cruise_mps(route_nodes[0].segment),
            uturn_planner=args.uturn_planner,
        )
        plan_log.append({"k": 0, **meta})
        for m in meta.get("modules") or []:
            modules_used.add(str(m))
        hybrid_calls += int(meta.get("hybrid_calls") or 0)
        hybrid_ok_count += int(meta.get("hybrid_ok") or 0)
        rs_analytic_hits_sum += int(meta.get("analytic_hits") or meta.get("rs_analytic_hits") or 0)
        rs_trajectory_executed = rs_trajectory_executed or bool(meta.get("rs_trajectory_executed"))
        fallback_count += int(bool(meta.get("fallback")))
        print(f"initial plan: {meta} n_pts={len(traj.points)}", flush=True)
        control.set_trajectory(traj, 0.0)

        risk = evaluate_risk_field(
            ego_v=max(1.0, state0.v),
            actors=[{"s": 25.0, "v": 4.0}],
            uncertainty_scale=1.2,
            track="observable",
        )
        print(f"RiskField observable={risk.observable_score:.3f}", flush=True)

        for k in range(ticks):
            now = k * (control_cfg.control_period_ms / 1000.0)
            state = ego_state_from_actor(ego)
            behavior.handle(BehaviorEvent.TICK, timestamp_s=now)

            loc = ego.get_transform().location
            nearest_i = min(
                range(len(route_nodes)),
                key=lambda i: math.hypot(route_nodes[i].x - loc.x, route_nodes[i].y - loc.y),
            )
            if nearest_i > max_route_i and nearest_i <= max_route_i + 30:
                max_route_i = nearest_i
            elif nearest_i > max_route_i + 30:
                nearest_i = max_route_i

            seg = route_nodes[max_route_i].segment
            if seg == "uturn":
                uturn_seen = True

            cte = route_cross_track(route_nodes, loc.x, loc.y)
            max_abs_cte = max(max_abs_cte, abs(cte))
            hard = 1.2 if seg == "uturn" else 1.6
            if abs(cte) > hard:
                force_replan = True

            do_replan = (not arrived) and (
                force_replan or (k > 0 and k % max(1, args.replan_every) == 0)
            )
            if do_replan:
                force_replan = False
                remain = route_nodes[max(0, max_route_i) :]
                if len(remain) >= 6:
                    new_traj, meta = plan_stack(
                        args.stack,
                        frenet=frenet,
                        hybrid=hybrid,
                        nodes=remain,
                        ego=state,
                        seed=seed + k,
                        segment=seg,
                        cruise_mps=segment_cruise_mps(seg),
                        uturn_planner=args.uturn_planner,
                    )
                    plan_log.append({"k": k, **meta})
                    for m in meta.get("modules") or []:
                        modules_used.add(str(m))
                    hybrid_calls += int(meta.get("hybrid_calls") or 0)
                    hybrid_ok_count += int(meta.get("hybrid_ok") or 0)
                    rs_analytic_hits_sum += int(
                        meta.get("analytic_hits") or meta.get("rs_analytic_hits") or 0
                    )
                    rs_trajectory_executed = rs_trajectory_executed or bool(
                        meta.get("rs_trajectory_executed")
                    )
                    fallback_count += int(bool(meta.get("fallback")))
                    traj = new_traj
            control.set_trajectory(traj, now)

            if arrived:
                c = carla.VehicleControl(throttle=0.0, brake=1.0)
                cmd_mode = "hold"
                cmd_miss = False
                header = runtime.tick(c)
                set_spectator_follow(runtime.world, ego)
                hold_left -= 1
            else:
                cmd = control.step(state, now)
                c = to_carla_control(
                    cmd,
                    control_cfg,
                    state,
                    segment=seg,
                    cross_track=cte,
                )
                header = runtime.tick(c)
                set_spectator_follow(runtime.world, ego)
                # Race identification (full stack only)
                if args.stack == "full" and isinstance(control, RaceControlLoop):
                    state_after = ego_state_from_actor(ego)
                    dt_s = control_cfg.control_period_ms / 1000.0
                    control.identify(state, cmd, state_after, dt_s)
                    race_id_samples = control.est.samples
                    race_tighten_max = max(race_tighten_max, float(control.tighten))
                cmd_mode = cmd.mode
                cmd_miss = cmd.deadline_miss

            loc = ego.get_transform().location
            dist = math.hypot(loc.x - goal_loc.x, loc.y - goal_loc.y)
            min_goal_dist = min(min_goal_dist, dist)
            route_progress_m = 0.0
            for i in range(1, max_route_i + 1):
                route_progress_m += math.hypot(
                    route_nodes[i].x - route_nodes[i - 1].x,
                    route_nodes[i].y - route_nodes[i - 1].y,
                )
            history.append(
                {
                    "k": k,
                    "frame": header.carla_frame,
                    "x": loc.x,
                    "y": loc.y,
                    "dist_goal": dist,
                    "route_i": max_route_i,
                    "route_progress_m": route_progress_m,
                    "segment": seg,
                    "mode": cmd_mode,
                    "behavior": behavior.state.value,
                    "v": state.v,
                    "cte": cte,
                    "deadline_miss": cmd_miss,
                    "uturn_seen": uturn_seen,
                }
            )

            # Stuck detection
            now_s = now
            if not arrived and stuck_watch.update(
                now_s=now_s, v=state.v, progress_m=route_progress_m
            ):
                failure = "STUCK_NO_PROGRESS"
                ok = False
                print(
                    f"STUCK at tick={k} v={state.v:.2f} prog={route_progress_m:.1f}m "
                    f"seg={seg} cte={cte:+.2f}m — aborting cleanly",
                    flush=True,
                )
                break

            if k % 80 == 0:
                draw_segmented_route(runtime.world, route_nodes, life=45.0)
            if k > 0 and k % 200 == 0:
                clear_world_npcs(runtime.world, keep_ids={int(ego.id)})
            if k % 50 == 0:
                print(
                    f"tick={k}/{ticks} v={state.v:.2f} dist={dist:.1f}m "
                    f"i={max_route_i}/{len(route_nodes)} prog={route_progress_m:.0f}m "
                    f"seg={seg} cte={cte:+.2f} ctrl={cmd_mode}",
                    flush=True,
                )

            if not arrived and (dist < 8.0 or max_route_i >= len(route_nodes) - 2):
                arrived = True
                ok = True
                hold_left = hold_ticks
                print(
                    f"ARRIVED goal at tick={k} prog={route_progress_m:.0f}m "
                    f"uturn_seen={uturn_seen}; holding {args.hold_s:.0f}s before despawn",
                    flush=True,
                )
            if arrived and hold_left <= 0:
                break

        if not ok and history:
            route_prog = history[-1].get("route_progress_m", 0.0)
            frac = route_prog / max(complexity["length_m"], 1.0)
            if frac >= 0.85 and uturn_seen:
                ok = True
                failure = "NEAR_COMPLETE"
            elif route_prog >= 60.0 and uturn_seen:
                ok = True
                failure = "PROGRESS_WITH_UTURN_NOT_FULL"
            else:
                failure = "INCOMPLETE_ROUTE"
        runtime.complete()
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        try:
            runtime.abort(type(exc).__name__)
        except Exception:
            try:
                runtime.close()
            except Exception:
                pass
        ok = False

    modes: dict[str, int] = {}
    for h in history:
        modes[h["mode"]] = modes.get(h["mode"], 0) + 1
    seg_ticks = {}
    for h in history:
        seg_ticks[h.get("segment", "?")] = seg_ticks.get(h.get("segment", "?"), 0) + 1

    # RaceControlLoop exposes .base.watchdog; ControlLoop has .watchdog
    if hasattr(control, "watchdog"):
        wd = control.watchdog.summary()
        chash = control.config_hash
    else:
        wd = control.base.watchdog.summary()
        chash = control.base.config_hash

    modules_list = sorted(modules_used)

    payload = {
        "schema": "safedrive.g1.live_stack.v4",
        "run_id": run_id,
        "ok": ok,
        "failure": failure,
        "seed": seed,
        "map": map_name,
        "stack": args.stack,
        "uturn_planner": args.uturn_planner,
        "control_kind": control_kind,
        "modules_used": modules_list,
        "race_identification_samples": race_id_samples,
        "race_tightening_max": race_tighten_max,
        "hybrid_calls": hybrid_calls,
        "hybrid_ok_count": hybrid_ok_count,
        "rs_analytic_hits_sum": rs_analytic_hits_sum,
        "rs_trajectory_executed": rs_trajectory_executed,
        "fallback_count": fallback_count,
        "max_abs_cte_m": max_abs_cte,
        "claims": {
            "waypoint_bypass": args.stack == "waypoint",
            "frenet_in_loop": "FrenetPlanner" in modules_list,
            "hybrid_in_loop": hybrid_calls > 0 or "HybridAstarPlanner" in modules_list,
            "race_control_wrapper_in_loop": args.stack == "full",
            "race_identification_active": race_id_samples > 0,
            "rs_analytic_used": rs_analytic_hits_sum > 0 and rs_trajectory_executed,
            "uturn_dense_default": args.uturn_planner == "dense",
            "not_full_stack_if_only_fallback": fallback_count > 0
            and "FrenetPlanner" not in modules_list,
        },
        "complexity": complexity,
        "arrived": arrived,
        "uturn_seen": uturn_seen,
        "max_route_index": max_route_i,
        "max_route_progress_m": history[-1].get("route_progress_m") if history else 0.0,
        "route_nodes": len(route_nodes),
        "ticks_budget": ticks,
        "ticks_run": len(history),
        "min_goal_dist_m": min_goal_dist if min_goal_dist < 1e8 else None,
        "final_goal_dist_m": history[-1]["dist_goal"] if history else None,
        "start_goal_dist_m": history[0]["dist_goal"] if history else None,
        "segment_ticks": seg_ticks,
        "connection": report.to_dict(),
        "frenet_hash": frenet.config_hash,
        "hybrid_hash": hybrid.config_hash,
        "control_hash": chash,
        "watchdog": wd,
        "mode_counts": modes,
        "plan_log": plan_log[:40],
        "history_tail": history[-50:],
        "pipeline": [
            "ScenarioRuntime",
            "BehaviorStateMachine",
            "route(approach+uturn+longtail)",
            f"stack={args.stack}",
            control_kind,
            "spectator_follow",
            "no_traffic",
        ],
    }
    out = evidence_dir / f"{run_id}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Attempt always; success pointer only when ok
    (evidence_dir / "latest_attempt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if ok:
        (evidence_dir / "latest_success.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "stack": args.stack,
                "modules": modules_list,
                "claims": payload["claims"],
                "arrived": arrived,
                "uturn_seen": uturn_seen,
                "map": map_name,
                "seed": seed,
                "tags": complexity["tags"],
                "turn_deg": round(complexity["turn_deg_sum"], 1),
                "route_m": round(complexity["length_m"], 1),
                "segments": complexity.get("segments"),
                "ticks": len(history),
                "ticks_budget": ticks,
                "prog_m": None if not history else round(history[-1].get("route_progress_m", 0.0), 1),
                "min_goal_m": round(min_goal_dist, 2) if min_goal_dist < 1e8 else None,
                "final_goal_m": None if not history else round(history[-1]["dist_goal"], 2),
                "seg_ticks": seg_ticks,
                "modes": modes,
                "failure": failure,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"evidence={out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
