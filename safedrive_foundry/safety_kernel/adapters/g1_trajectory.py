"""Convert G1 Frenet/Hybrid trajectory dicts into PolicyCandidateSet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.contracts.types import (
    CandidateSource,
    PolicyCandidate,
    PolicyCandidateSet,
    TrajectoryPoint,
)


def _points_from_g1(raw_points: list[Mapping[str, Any]]) -> tuple[TrajectoryPoint, ...]:
    points: list[TrajectoryPoint] = []
    for item in raw_points:
        points.append(
            TrajectoryPoint(
                t=float(item["t"]),
                x=float(item["x"]),
                y=float(item["y"]),
                yaw=float(item["yaw"]),
                kappa=float(item["kappa"]),
                v=float(item["v"]),
                a=float(item["a"]),
                jerk=float(item.get("jerk", 0.0)),
            )
        )
    return tuple(points)


def g1_trajectory_dict_to_candidate(
    traj: Mapping[str, Any],
    *,
    now_s: float,
    valid_for_s: float = 0.20,
    source: CandidateSource = CandidateSource.CLASSIC,
    probability: float = 1.0,
) -> PolicyCandidate:
    points = _points_from_g1(list(traj["points"]))
    trajectory_id = str(traj.get("trajectory_id") or traj.get("id") or "g1-traj")
    return PolicyCandidate(
        candidate_id=trajectory_id,
        source=source,
        generated_time_s=now_s,
        valid_until_s=now_s + valid_for_s,
        probability=probability,
        points=points,
        behavior=str(traj.get("behavior", "follow") or "follow"),
        risk_horizon_s=float(points[-1].t - points[0].t) if points else 0.0,
        intended_action=str(traj.get("source", "classic") or "classic"),
        uncertainty=float(traj.get("risk_cost", 0.0) or 0.0) * 0.01,
        availability=True,
        dynamics_meta={
            "risk_cost": traj.get("risk_cost", 0.0),
            "tracking_cost": traj.get("tracking_cost", 0.0),
            "g1_source": traj.get("source"),
        },
    )


def g1_plan_result_to_candidate_set(
    plan: Mapping[str, Any],
    *,
    run_id: str,
    frame_id: str,
    scenario_id: str,
    model_id: str = "classic_g1",
    carla_frame: int = 0,
    simulation_time_s: float = 0.0,
    wall_time_s: float = 0.0,
    now_s: float | None = None,
) -> PolicyCandidateSet:
    now = simulation_time_s if now_s is None else now_s
    traj = plan.get("trajectory")
    candidates: list[PolicyCandidate] = []
    if plan.get("ok") and isinstance(traj, Mapping) and traj.get("points"):
        candidates.append(g1_trajectory_dict_to_candidate(traj, now_s=now, source=CandidateSource.CLASSIC))
    return PolicyCandidateSet(
        run_id=run_id,
        frame_id=frame_id,
        scenario_id=scenario_id,
        model_id=model_id,
        carla_frame=carla_frame,
        simulation_time_s=simulation_time_s,
        wall_time_s=wall_time_s,
        candidates=tuple(candidates),
        schema_version=SCHEMA_VERSION,
        coordinate_frame="map",
    )


def load_g1_trajectory_json(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("G1 trajectory JSON must be an object")
    return data
