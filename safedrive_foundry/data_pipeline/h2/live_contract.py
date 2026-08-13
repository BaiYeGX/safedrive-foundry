"""Source-neutral live collection helpers (never imports the offline Oracle)."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from safety_kernel.contracts.serialize import candidate_to_dict
from safety_kernel.contracts.types import PolicyCandidate, TrajectoryPoint

from .contracts import (
    ActorInitialState,
    BranchOutcome,
    CandidateSnapshot,
    ResetComparison,
    ResetSignature,
    stable_sha256,
)


COLLECTOR_VERSION = "h2-paired-collector-v2-route-stop-preroll"
SCENARIO_ALGORITHM_VERSION = "h2-fixed-carla-scenarios-v3-route-stop-preroll"


def trajectory_sha256(points: Sequence[TrajectoryPoint]) -> str:
    # Match the H1 canonicalizer's authoritative x/y/yaw/v/a/kappa payload.
    return stable_sha256(
        tuple((point.x, point.y, point.yaw, point.v, point.a, point.kappa) for point in points)
    )


def candidate_snapshot(hybrid: Any, *, slot: int) -> CandidateSnapshot:
    payload = candidate_to_dict(hybrid.candidate)
    canonical = trajectory_sha256(hybrid.candidate.points)
    if canonical != hybrid.provenance.canonical_sha256:
        raise ValueError("captured candidate canonical hash mismatch")
    return CandidateSnapshot(
        candidate_id=hybrid.candidate.candidate_id,
        canonical_sha256=canonical,
        source=hybrid.provenance.source.value,
        slot=slot,
        trajectory=tuple(payload["points"]),
        guard={} if hybrid.guard is None else hybrid.guard.to_dict(),
        provenance=hybrid.provenance.to_dict(),
    )


def actor_initial_state(role: str, actor: Any) -> ActorInitialState:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    return ActorInitialState(
        role=role,
        x=float(transform.location.x),
        y=float(transform.location.y),
        yaw_deg=float(transform.rotation.yaw),
        speed_mps=math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2),
    )


def reset_signature(
    actors: Mapping[str, Any],
    *,
    route: Sequence[tuple[float, float]],
    weather: Mapping[str, float],
    lights: Iterable[Mapping[str, Any]],
    script: Mapping[str, Any],
) -> ResetSignature:
    return ResetSignature(
        actors=tuple(actor_initial_state(role, actors[role]) for role in sorted(actors)),
        route_sha256=stable_sha256([[float(x), float(y)] for x, y in route]),
        weather_sha256=stable_sha256(dict(weather)),
        light_sha256=stable_sha256(sorted((dict(item) for item in lights), key=lambda item: str(item.get("role", "")))),
        script_sha256=stable_sha256(dict(script)),
    )


def point_to_polyline_distance(x: float, y: float, route: Sequence[tuple[float, float]]) -> float:
    return route_projection(x, y, route)[1]


def route_projection(x: float, y: float, route: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if len(route) < 2:
        return 0.0, math.inf
    best_s, best_distance, accumulated = 0.0, math.inf, 0.0
    for index in range(1, len(route)):
        ax, ay = route[index - 1]
        bx, by = route[index]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        fraction = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        px, py = ax + fraction * dx, ay + fraction * dy
        distance = math.hypot(x - px, y - py)
        segment = math.sqrt(length_sq)
        if distance < best_distance:
            best_distance = distance
            best_s = accumulated + fraction * segment
        accumulated += segment
    return best_s, best_distance


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values) / len(values)) if values else 0.0


def kinematic_metrics(timeline: Sequence[Mapping[str, Any]], *, dt_s: float = 0.05) -> dict[str, float]:
    speeds = [float(item["speed_mps"]) for item in timeline]
    accelerations = [(speeds[i] - speeds[i - 1]) / dt_s for i in range(1, len(speeds))]
    jerks = [(accelerations[i] - accelerations[i - 1]) / dt_s for i in range(1, len(accelerations))]
    lateral_accelerations = [float(item.get("lateral_acceleration_mps2", 0.0)) for item in timeline]
    return {
        "acceleration_rms_mps2": rms(accelerations),
        "jerk_rms_mps3": rms(jerks),
        "lateral_acceleration_rms_mps2": rms(lateral_accelerations),
    }


def make_branch_outcome(
    *,
    candidate: PolicyCandidate,
    reset: ResetComparison,
    decision: Any,
    applied: Any,
    pre_binding_sha256: str,
    post_binding_sha256: str,
    timeline: Sequence[Mapping[str, Any]],
    cleanup_complete: bool,
    collision_count: int,
    red_light_violation: bool,
    off_corridor_duration_s: float,
    route_completed: bool,
    route_progress_m: float,
    timeline_path: str,
    actor_future_path: str,
    event_path: str,
    branch_latency_s: float,
    whole_gpu_peak_gb: float,
    errors: Sequence[str] = (),
) -> BranchOutcome:
    metrics = kinematic_metrics(timeline)
    return BranchOutcome(
        candidate_id=candidate.candidate_id,
        candidate_sha256=pre_binding_sha256,
        reset=reset,
        safety_executed=bool(applied.is_track_approved),
        safety_input_id=candidate.candidate_id,
        safety_final_id=decision.final_candidate_id,
        safety_executed_id=decision.executed_trajectory_id,
        applied_id=applied.executed_id,
        pre_binding_trajectory_sha256=pre_binding_sha256,
        post_binding_trajectory_sha256=post_binding_sha256,
        ticks_executed=len(timeline),
        cleanup_complete=cleanup_complete,
        collision_count=collision_count,
        red_light_violation=red_light_violation,
        off_corridor_duration_s=off_corridor_duration_s,
        route_completed=route_completed,
        route_progress_m=route_progress_m,
        jerk_rms_mps3=metrics["jerk_rms_mps3"],
        acceleration_rms_mps2=metrics["acceleration_rms_mps2"],
        lateral_acceleration_rms_mps2=metrics["lateral_acceleration_rms_mps2"],
        deadline_misses=sum(bool(item.get("deadline_miss")) for item in timeline),
        timeline_path=timeline_path,
        actor_future_path=actor_future_path,
        event_path=event_path,
        branch_latency_s=branch_latency_s,
        whole_gpu_peak_gb=whole_gpu_peak_gb,
        errors=tuple(errors),
    )


__all__ = [
    "COLLECTOR_VERSION", "SCENARIO_ALGORITHM_VERSION", "actor_initial_state",
    "candidate_snapshot", "kinematic_metrics", "make_branch_outcome",
    "point_to_polyline_distance", "reset_signature", "rms", "route_projection",
    "trajectory_sha256",
]
