"""ROS-facing adapter stubs for classic map / route / behavior outputs.

These adapters convert pure-Python classic_stack objects into dictionaries that
mirror sdf_interfaces field names. They do not create ROS nodes or call CARLA.
"""

from __future__ import annotations

from typing import Any, Mapping

from classic_stack.behavior.state_machine import BehaviorGoal, BehaviorTransition
from classic_stack.route.planner import RouteCorridor


MANEUVER_TO_ROUTE_MSG = {
    "UNKNOWN": 0,
    "FOLLOW": 1,
    "LEFT": 2,
    "RIGHT": 3,
    "STRAIGHT": 4,
    "STOP": 5,
}


def corridor_to_route_dict(corridor: RouteCorridor, *, frame: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project a RouteCorridor into a Route.msg-shaped dictionary."""

    waypoints = []
    maneuvers = []
    for step in corridor.steps:
        node_xy = (0.0, 0.0)
        waypoints.append(
            {
                "pose": {
                    "position": {"x": node_xy[0], "y": node_xy[1], "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "header_frame_id": "carla_map",
                "node_id": step.node_id,
                "road_id": step.road_id,
                "lane_id": step.lane_id,
                "semantic": step.semantic,
            }
        )
        maneuvers.append(MANEUVER_TO_ROUTE_MSG.get(step.maneuver, 0))
    return {
        "frame": dict(frame or {}),
        "route_id": corridor.route_id,
        "waypoints": waypoints,
        "maneuver": maneuvers,
        "map_hash": corridor.map_hash,
        "map_name": corridor.map_name,
        "oracle_inputs": list(corridor.semantics.get("oracle_inputs", [])),
        "observable_inputs": list(corridor.semantics.get("observable_inputs", [])),
    }


def behavior_goal_to_dict(goal: BehaviorGoal, *, frame: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = goal.to_dict()
    payload["frame"] = dict(frame or {})
    return payload


def behavior_transition_to_dict(transition: BehaviorTransition) -> dict[str, Any]:
    return transition.to_dict()
