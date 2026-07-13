"""Interpretable actor prediction: CV / CTRV / IDM for S-T occupancy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from classic_stack.geometry import ReferencePath
from classic_stack.planning.frenet.config import FrenetSTConfig


@dataclass(frozen=True)
class ActorState:
    actor_id: str
    x: float
    y: float
    yaw: float
    speed_mps: float
    length_m: float = 4.5
    width_m: float = 1.9
    model: str = "cv"  # cv | ctrv | idm
    accel_mps2: float = 0.0
    yaw_rate_rps: float = 0.0


def _idm_accel(
    v: float,
    v_lead: float,
    gap: float,
    *,
    v0: float,
    t_gap: float,
    s0: float,
    a_max: float,
    b_comf: float,
) -> float:
    if gap <= 1e-3:
        return -a_max
    s_star = s0 + max(0.0, v * t_gap + v * (v - v_lead) / (2.0 * math.sqrt(max(a_max * b_comf, 1e-6))))
    return a_max * (1.0 - (v / max(v0, 1e-3)) ** 4 - (s_star / gap) ** 2)


def predict_actors_st(
    actors: Sequence[ActorState],
    reference: ReferencePath,
    config: FrenetSTConfig,
    *,
    ego_s: float = 0.0,
    ego_v: float = 0.0,
) -> list[dict[str, float | str]]:
    """Return list of occupancy samples: s_min,s_max,t,actor_id."""

    samples: list[dict[str, float | str]] = []
    dt = config.pred_dt_s
    steps = max(1, int(config.pred_horizon_s / dt))
    for actor in actors:
        s0, _ = reference.project(actor.x, actor.y)
        x, y, yaw, v = actor.x, actor.y, actor.yaw, actor.speed_mps
        half = 0.5 * actor.length_m + config.st_inflation_m
        for k in range(steps + 1):
            t = k * dt
            if actor.model == "ctrv":
                if abs(actor.yaw_rate_rps) < 1e-4:
                    x += v * math.cos(yaw) * dt
                    y += v * math.sin(yaw) * dt
                else:
                    x += (v / actor.yaw_rate_rps) * (
                        math.sin(yaw + actor.yaw_rate_rps * dt) - math.sin(yaw)
                    )
                    y += (v / actor.yaw_rate_rps) * (
                        -math.cos(yaw + actor.yaw_rate_rps * dt) + math.cos(yaw)
                    )
                    yaw += actor.yaw_rate_rps * dt
            elif actor.model == "idm":
                # Lead is ego if actor is ahead on reference, else free road.
                gap = max(0.5, (s0 + v * t) - ego_s - 0.5 * config.vehicle.length_m)
                a = _idm_accel(
                    v,
                    ego_v,
                    gap,
                    v0=config.vehicle.max_speed_mps,
                    t_gap=config.idm_time_gap_s,
                    s0=config.idm_min_gap_m,
                    a_max=config.idm_max_accel,
                    b_comf=config.idm_comf_decel,
                )
                v = max(0.0, v + a * dt)
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
            else:
                # constant velocity
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
            s, _ = reference.project(x, y)
            samples.append(
                {
                    "actor_id": actor.actor_id,
                    "t": float(t),
                    "s_min": float(s - half),
                    "s_max": float(s + half),
                }
            )
            if k == 0:
                s0 = s
    return samples
