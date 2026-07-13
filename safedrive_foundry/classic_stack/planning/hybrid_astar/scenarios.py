"""Fixed offline maps for G1-05 maneuver families."""

from __future__ import annotations

import math

from classic_stack.planning.hybrid_astar.planner import ManeuverRequest, ObstacleMap

MANEUVER_KINDS = ("blocked_detour", "three_point_turn", "reverse_park", "dead_end_escape")


def make_maneuver(kind: str, *, seed: int = 0) -> ManeuverRequest:
    if kind == "blocked_detour":
        # Straight road blocked in the middle; start west, goal east.
        world = ObstacleMap(
            xmin=-2.0,
            xmax=40.0,
            ymin=-8.0,
            ymax=8.0,
            boxes=((12.0, -1.2, 16.0, 1.2),),  # blockage on centerline
        )
        return ManeuverRequest(
            start=(0.0, 0.0, 0.0),
            goal=(30.0, 0.0, 0.0),
            world=world,
            blocked=True,
            seed=seed,
        )
    if kind == "three_point_turn":
        world = ObstacleMap(xmin=-5.0, xmax=15.0, ymin=-6.0, ymax=6.0, boxes=())
        return ManeuverRequest(
            start=(0.0, 0.0, 0.0),
            goal=(0.0, 0.0, math.pi),
            world=world,
            require_reverse=True,
            seed=seed,
        )
    if kind == "reverse_park":
        # Parallel bay behind vehicle on the right.
        world = ObstacleMap(
            xmin=-2.0,
            xmax=20.0,
            ymin=-8.0,
            ymax=4.0,
            boxes=(
                (6.0, -6.0, 8.0, -2.0),  # rear car
                (14.0, -6.0, 16.0, -2.0),  # front car
            ),
        )
        return ManeuverRequest(
            start=(2.0, 0.0, 0.0),
            goal=(11.0, -4.0, 0.0),
            world=world,
            require_reverse=True,
            narrow=True,
            seed=seed,
        )
    if kind == "dead_end_escape":
        # Cul-de-sac: must reverse then leave.
        world = ObstacleMap(
            xmin=-2.0,
            xmax=20.0,
            ymin=-4.0,
            ymax=4.0,
            boxes=(
                (8.0, -4.0, 20.0, -1.5),
                (8.0, 1.5, 20.0, 4.0),
                (18.0, -1.5, 20.0, 1.5),  # end wall
            ),
        )
        return ManeuverRequest(
            start=(16.0, 0.0, 0.0),
            goal=(0.0, 0.0, math.pi),
            world=world,
            require_reverse=True,
            narrow=True,
            seed=seed,
        )
    raise ValueError(f"unknown maneuver kind: {kind}")
