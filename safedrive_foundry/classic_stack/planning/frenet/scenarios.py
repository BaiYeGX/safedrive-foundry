"""Fixed synthetic scenarios for G1-04 acceptance (offline)."""

from __future__ import annotations

from classic_stack.geometry import ReferencePath
from classic_stack.planning.frenet.planner import ActorState, PlanRequest, StaticObstacle


def straight_reference(length_m: float = 80.0, ds: float = 2.0) -> ReferencePath:
    n = int(length_m / ds) + 1
    xs = [i * ds for i in range(n)]
    ys = [0.0] * n
    return ReferencePath.from_xy(xs, ys)


def make_scenario(kind: str, *, seed: int = 0) -> PlanRequest:
    """Build one of: follow, stop, lane_change, cut_in, avoid."""

    ref = straight_reference()
    if kind == "follow":
        lead = ActorState(
            actor_id="lead",
            x=25.0,
            y=0.0,
            yaw=0.0,
            speed_mps=5.0,
            model="idm",
        )
        return PlanRequest(
            reference=ref,
            v0=10.0,
            scenario_kind="follow",
            actors=(lead,),
            target_speed_mps=8.0,
            seed=seed,
        )
    if kind == "stop":
        return PlanRequest(
            reference=ref,
            v0=8.0,
            scenario_kind="stop",
            stop_s=20.0,
            target_speed_mps=0.0,
            seed=seed,
        )
    if kind == "lane_change":
        # Obstacle on centerline; preferred offset 1.75 must clear disc:
        # clearance need radius + 0.5*width ≈ radius+0.95 < 1.75 → radius < 0.8
        return PlanRequest(
            reference=ref,
            v0=8.0,
            scenario_kind="lane_change",
            preferred_offset_m=1.75,
            target_speed_mps=8.0,
            static_obstacles=(StaticObstacle(x=18.0, y=0.0, radius_m=0.65),),
            seed=seed,
        )
    if kind == "cut_in":
        cutter = ActorState(
            actor_id="cut_in",
            x=12.0,
            y=1.6,
            yaw=0.0,
            speed_mps=7.0,
            model="cv",
        )
        return PlanRequest(
            reference=ref,
            v0=11.0,
            scenario_kind="cut_in",
            actors=(cutter,),
            target_speed_mps=6.0,
            seed=seed,
        )
    if kind == "avoid":
        return PlanRequest(
            reference=ref,
            v0=7.0,
            scenario_kind="avoid",
            preferred_offset_m=-1.75,
            static_obstacles=(StaticObstacle(x=24.0, y=0.0, radius_m=0.65),),
            target_speed_mps=7.0,
            seed=seed,
        )
    raise ValueError(f"unknown scenario kind: {kind}")


SCENARIO_KINDS = ("follow", "stop", "lane_change", "cut_in", "avoid")
