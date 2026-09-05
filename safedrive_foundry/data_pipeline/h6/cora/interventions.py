"""Deterministic, offline-only C2 trajectory interventions."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from data_pipeline.h2.contracts import stable_sha256
from data_pipeline.h2.live_contract import route_projection, trajectory_sha256
from driving_vla.hybrid.contracts import (
    CandidateProvenance,
    GuardResult,
    HybridCandidate,
    HybridCandidateSet,
    HybridSource,
    ObservableAnchor,
)
from driving_vla.hybrid.guard import CandidateGuard
from safety_kernel.contracts.serialize import candidate_to_dict
from safety_kernel.contracts.types import PolicyCandidate, TrajectoryPoint

from .config import CORA_C2_CONFIG
from .contracts import CoraProposal


FAMILY_OPERATORS: Mapping[str, tuple[str, str]] = {
    "free_flow": ("speed_scale_up", "curvature_bump"),
    "slow_lead": ("delayed_brake", "shortened_stopping_margin"),
    "stopped_lead": ("shortened_stopping_margin", "obstacle_envelope_approach"),
    "cut_in": ("speed_scale_up", "lateral_offset_toward_conflict"),
    "red_light_hold": ("delayed_brake", "stop_line_crossing"),
    "emergency_lead_brake": ("delayed_brake", "obstacle_envelope_approach"),
    "aggressive_cut_in": ("lateral_offset_toward_conflict", "obstacle_envelope_approach"),
    "red_light_dilemma": ("delayed_brake", "stop_line_crossing"),
    "cross_traffic_conflict": ("speed_scale_up", "obstacle_envelope_approach"),
}


class InterventionNotApplicable(ValueError):
    pass


@dataclass(frozen=True)
class InterventionResult:
    root_id: str
    base_source: str
    operator: str
    status: str
    candidate: HybridCandidate | None = None
    guard: GuardResult | None = None
    error: str = ""

    @property
    def finite_proposal(self) -> bool:
        return self.candidate is not None

    @property
    def guard_eligible(self) -> bool:
        return bool(self.guard and self.guard.passed)

    def to_proposal(self) -> CoraProposal | None:
        if self.candidate is None:
            return None
        item = self.candidate
        payload = candidate_to_dict(item.candidate)
        base = dict(item.candidate.dynamics_meta).get("cora_intervention", {})
        return CoraProposal(
            proposal_id=item.candidate.candidate_id,
            proposal_sha256=trajectory_sha256(item.candidate.points),
            root_id=self.root_id,
            kind="offline_intervention",
            trajectory=tuple(payload["points"]),
            guard={} if self.guard is None else self.guard.to_dict(),
            base_proposal_id=str(base["base_proposal_id"]),
            base_proposal_sha256=str(base["base_proposal_sha256"]),
            audit_source=self.base_source,
            operator=self.operator,
            magnitude=dict(base["magnitude"]),
            provenance=item.provenance.to_dict(),
            auxiliary_only=not self.guard_eligible,
            status=self.status,
        )


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _interpolate_route(route: Sequence[tuple[float, float]], target_s: float) -> tuple[float, float]:
    if len(route) < 2:
        raise InterventionNotApplicable("route_missing")
    accumulated = 0.0
    for index in range(1, len(route)):
        ax, ay = route[index - 1]
        bx, by = route[index]
        length = math.hypot(bx - ax, by - ay)
        if length <= 1e-9:
            continue
        if accumulated + length >= target_s:
            ratio = max(0.0, min(1.0, (target_s - accumulated) / length))
            return ax + ratio * (bx - ax), ay + ratio * (by - ay)
        accumulated += length
    raise InterventionNotApplicable("route_target_out_of_range")


def _recompute(
    anchor: ObservableAnchor,
    original: Sequence[TrajectoryPoint],
    xy: Sequence[tuple[float, float]],
) -> tuple[TrajectoryPoint, ...]:
    if len(original) != 10 or len(xy) != 10:
        raise InterventionNotApplicable("trajectory_not_t10")
    dt = 0.25
    previous_x, previous_y = float(anchor.bundle.ego_x), float(anchor.bundle.ego_y)
    previous_yaw = float(anchor.bundle.ego_yaw)
    previous_v = max(0.0, float(anchor.bundle.ego_v))
    previous_a = 0.0
    output: list[TrajectoryPoint] = []
    for index, (x, y) in enumerate(xy):
        if not all(math.isfinite(value) for value in (x, y)):
            raise InterventionNotApplicable("non_finite_xy")
        dx, dy = float(x) - previous_x, float(y) - previous_y
        ds = math.hypot(dx, dy)
        yaw = previous_yaw if ds <= 1e-9 else math.atan2(dy, dx)
        v = min(15.0, ds / dt)
        a = (v - previous_v) / dt
        kappa = _wrap(yaw - previous_yaw) / max(ds, 1e-3)
        jerk = (a - previous_a) / dt
        output.append(
            TrajectoryPoint(
                t=(index + 1) * dt,
                x=float(x),
                y=float(y),
                yaw=_wrap(yaw),
                kappa=float(kappa),
                v=float(v),
                a=float(a),
                jerk=float(jerk),
            )
        )
        previous_x, previous_y, previous_yaw = float(x), float(y), yaw
        previous_v, previous_a = v, a
    return tuple(output)


def _speed_scale(anchor: ObservableAnchor, points: Sequence[TrajectoryPoint]) -> tuple[TrajectoryPoint, ...]:
    factor = float(CORA_C2_CONFIG["interventions"]["speed_scale"])
    dt = 0.25
    previous = (float(anchor.bundle.ego_x), float(anchor.bundle.ego_y))
    xy: list[tuple[float, float]] = []
    for point in points:
        dx, dy = point.x - previous[0], point.y - previous[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            xy.append(previous)
            continue
        scaled = min(length * factor, 15.0 * dt)
        current = (previous[0] + dx / length * scaled, previous[1] + dy / length * scaled)
        xy.append(current)
        previous = current
    return _recompute(anchor, points, xy)


def _delayed_brake(anchor: ObservableAnchor, points: Sequence[TrajectoryPoint]) -> tuple[TrajectoryPoint, ...]:
    speeds = [float(item.v) for item in points]
    if not any(speeds[index] < speeds[index - 1] - 0.10 for index in range(1, len(speeds))):
        raise InterventionNotApplicable("no_braking_profile")
    delayed = [speeds[0], speeds[0], *speeds[:-2]]
    previous = (float(anchor.bundle.ego_x), float(anchor.bundle.ego_y))
    xy: list[tuple[float, float]] = []
    for point, target_v in zip(points, delayed):
        dx, dy = point.x - previous[0], point.y - previous[1]
        length = math.hypot(dx, dy)
        yaw = point.yaw if length <= 1e-9 else math.atan2(dy, dx)
        step = min(15.0, max(0.0, target_v)) * 0.25
        current = (previous[0] + math.cos(yaw) * step, previous[1] + math.sin(yaw) * step)
        xy.append(current)
        previous = current
    return _recompute(anchor, points, xy)


def _extend_terminal(
    anchor: ObservableAnchor, points: Sequence[TrajectoryPoint], distance_m: float
) -> tuple[TrajectoryPoint, ...]:
    xy: list[tuple[float, float]] = []
    terminal_yaw = float(points[-1].yaw)
    for index, point in enumerate(points):
        ratio = ((index + 1) / len(points)) ** 2
        xy.append(
            (
                float(point.x) + math.cos(terminal_yaw) * distance_m * ratio,
                float(point.y) + math.sin(terminal_yaw) * distance_m * ratio,
            )
        )
    return _recompute(anchor, points, xy)


def _stop_line_crossing(anchor: ObservableAnchor, points: Sequence[TrajectoryPoint]) -> tuple[TrajectoryPoint, ...]:
    controlled = [
        light
        for light in anchor.safety_snapshot.traffic_lights
        if light.controls_ego_lane is not False and light.stop_line_distance_m is not None
    ]
    if not controlled:
        raise InterventionNotApplicable("stop_line_geometry_missing")
    stop_distance = min(float(light.stop_line_distance_m) for light in controlled)
    route = tuple((float(x), float(y)) for x, y in anchor.bundle.route_xy)
    ego_s, _ = route_projection(anchor.bundle.ego_x, anchor.bundle.ego_y, route)
    last_s, _ = route_projection(points[-1].x, points[-1].y, route)
    target_s = ego_s + stop_distance + float(CORA_C2_CONFIG["interventions"]["stop_line_crossing_m"])
    denom = max(last_s - ego_s, 1e-6)
    xy = []
    for point in points:
        point_s, _ = route_projection(point.x, point.y, route)
        ratio = max(0.0, min(1.0, (point_s - ego_s) / denom))
        xy.append(_interpolate_route(route, ego_s + ratio * (target_s - ego_s)))
    return _recompute(anchor, points, xy)


def _lateral_offset(
    anchor: ObservableAnchor, points: Sequence[TrajectoryPoint], root_id: str
) -> tuple[TrajectoryPoint, ...]:
    actors = anchor.safety_snapshot.actors
    sign = 1.0 if int(stable_sha256({"lateral": root_id}), 16) % 2 == 0 else -1.0
    if actors:
        actor = min(
            actors,
            key=lambda item: math.hypot(item.x - points[-1].x, item.y - points[-1].y),
        )
        normal_x, normal_y = -math.sin(points[-1].yaw), math.cos(points[-1].yaw)
        sign = 1.0 if (actor.x - points[-1].x) * normal_x + (actor.y - points[-1].y) * normal_y >= 0 else -1.0
    maximum = float(CORA_C2_CONFIG["interventions"]["lateral_offset_m"])
    xy = []
    for index, point in enumerate(points):
        u = (index + 1) / len(points)
        smooth = u * u * (3.0 - 2.0 * u)
        offset = sign * maximum * smooth
        xy.append((point.x - math.sin(point.yaw) * offset, point.y + math.cos(point.yaw) * offset))
    return _recompute(anchor, points, xy)


def _curvature_bump(anchor: ObservableAnchor, points: Sequence[TrajectoryPoint]) -> tuple[TrajectoryPoint, ...]:
    amplitude = float(CORA_C2_CONFIG["interventions"]["curvature_bump_m"])
    xy = []
    for point in points:
        offset = amplitude * math.sin(2.0 * math.pi * point.t / 2.5)
        xy.append((point.x - math.sin(point.yaw) * offset, point.y + math.cos(point.yaw) * offset))
    return _recompute(anchor, points, xy)


def _obstacle_approach(anchor: ObservableAnchor, points: Sequence[TrajectoryPoint]) -> tuple[TrajectoryPoint, ...]:
    actors = tuple(actor for actor in anchor.safety_snapshot.actors if not actor.lost)
    if not actors:
        raise InterventionNotApplicable("observable_actor_missing")
    reduction = float(CORA_C2_CONFIG["interventions"]["obstacle_clearance_reduction_m"])
    min_time = float(CORA_C2_CONFIG["interventions"]["obstacle_min_time_s"])
    xy = []
    for point in points:
        if point.t < min_time:
            xy.append((point.x, point.y))
            continue
        actor = min(actors, key=lambda item: math.hypot(item.x - point.x, item.y - point.y))
        dx, dy = actor.x - point.x, actor.y - point.y
        distance = math.hypot(dx, dy)
        if distance <= reduction + 1e-6:
            xy.append((point.x, point.y))
        else:
            xy.append((point.x + dx / distance * reduction, point.y + dy / distance * reduction))
    return _recompute(anchor, points, xy)


def _apply(
    operator: str,
    anchor: ObservableAnchor,
    points: Sequence[TrajectoryPoint],
    root_id: str,
) -> tuple[TrajectoryPoint, ...]:
    if operator == "speed_scale_up":
        return _speed_scale(anchor, points)
    if operator == "delayed_brake":
        return _delayed_brake(anchor, points)
    if operator == "shortened_stopping_margin":
        return _extend_terminal(
            anchor, points, float(CORA_C2_CONFIG["interventions"]["shortened_stopping_margin_m"])
        )
    if operator == "stop_line_crossing":
        return _stop_line_crossing(anchor, points)
    if operator == "lateral_offset_toward_conflict":
        return _lateral_offset(anchor, points, root_id)
    if operator == "curvature_bump":
        return _curvature_bump(anchor, points)
    if operator == "obstacle_envelope_approach":
        return _obstacle_approach(anchor, points)
    raise ValueError(f"unknown_cora_intervention:{operator}")


def _magnitude(operator: str) -> dict[str, Any]:
    values = CORA_C2_CONFIG["interventions"]
    mapping = {
        "speed_scale_up": {"scale": values["speed_scale"], "max_speed_mps": values["max_speed_mps"]},
        "delayed_brake": {"delay_s": values["delayed_brake_s"]},
        "shortened_stopping_margin": {"distance_m": values["shortened_stopping_margin_m"]},
        "stop_line_crossing": {"distance_m": values["stop_line_crossing_m"]},
        "lateral_offset_toward_conflict": {"offset_m": values["lateral_offset_m"]},
        "curvature_bump": {"amplitude_m": values["curvature_bump_m"], "center_s": 1.25},
        "obstacle_envelope_approach": {
            "clearance_reduction_m": values["obstacle_clearance_reduction_m"],
            "minimum_time_s": values["obstacle_min_time_s"],
        },
    }
    return dict(mapping[operator])


def derive_interventions(
    root_id: str,
    family: str,
    anchor: ObservableAnchor,
    candidates: Sequence[HybridCandidate],
    *,
    guard: CandidateGuard | None = None,
) -> tuple[InterventionResult, ...]:
    if family not in FAMILY_OPERATORS:
        raise ValueError(f"cora_intervention_family:{family}")
    by_source = {item.provenance.source.value: item for item in candidates}
    operators = FAMILY_OPERATORS[family]
    if int(stable_sha256({"root_id": root_id, "assignment": "cora-c2"}), 16) % 2:
        operators = tuple(reversed(operators))
    evaluator = guard or CandidateGuard()
    results: list[InterventionResult] = []
    for source, operator in zip(("expert", "vla"), operators):
        base = by_source.get(source)
        if base is None:
            results.append(InterventionResult(root_id, source, operator, "BASE_MISSING", error="base_missing"))
            continue
        try:
            points = _apply(operator, anchor, base.candidate.points, root_id)
            proposal_hash = trajectory_sha256(points)
            base_hash = trajectory_sha256(base.candidate.points)
            proposal_id = f"{base.candidate.candidate_id}:cora:{operator}"
            metadata = {
                "base_proposal_id": base.candidate.candidate_id,
                "base_proposal_sha256": base_hash,
                "operator": operator,
                "magnitude": _magnitude(operator),
                "offline_only": True,
            }
            candidate = replace(
                base.candidate,
                candidate_id=proposal_id,
                points=points,
                intended_action=f"cora_intervention:{operator}",
                dynamics_meta={**dict(base.candidate.dynamics_meta), "cora_intervention": metadata},
            )
            provenance = CandidateProvenance(
                source=HybridSource.EXPERT if source == "expert" else HybridSource.VLA,
                candidate_id=proposal_id,
                observation_id=base.provenance.observation_id,
                frame_id=base.provenance.frame_id,
                carla_frame=base.provenance.carla_frame,
                simulation_time_s=base.provenance.simulation_time_s,
                route_revision=base.provenance.route_revision,
                generator_id="cora-offline-intervention@1.0.0",
                generator_hash=stable_sha256({"operator": operator, "magnitude": _magnitude(operator)}),
                raw_sha256=proposal_hash,
                canonical_sha256=proposal_hash,
                canonicalizer_version="safedrive.cora.intervention_canonicalizer.v1",
                canonicalization_error_m=0.0,
                coverage_shortfall_m=0.0,
                generation_latency_s=0.0,
                generated_wall_time_s=base.provenance.generated_wall_time_s,
                freshness_s=base.provenance.freshness_s,
                coordinate_frame="map",
            )
            item = HybridCandidate(candidate=candidate, provenance=provenance)
            guard_result = evaluator.evaluate_candidate(HybridCandidateSet(anchor=anchor, candidates=(item,)), item)
            item = item.with_guard(guard_result)
            results.append(
                InterventionResult(
                    root_id,
                    source,
                    operator,
                    "GUARD_ELIGIBLE" if guard_result.passed else "AUXILIARY_REJECT",
                    candidate=item,
                    guard=guard_result,
                )
            )
        except InterventionNotApplicable as exc:
            results.append(InterventionResult(root_id, source, operator, "NOT_APPLICABLE", error=str(exc)))
        except (ValueError, ArithmeticError) as exc:
            results.append(
                InterventionResult(
                    root_id,
                    source,
                    operator,
                    "CANONICALIZATION_FAILED",
                    error=f"{type(exc).__name__}:{exc}",
                )
            )
    if len(results) > int(CORA_C2_CONFIG["interventions"]["max_per_root"]):
        raise AssertionError("cora_intervention_limit")
    return tuple(results)


def derive_scaled_intervention(
    root_id: str,
    anchor: ObservableAnchor,
    base: HybridCandidate,
    operator: str,
    multiplier: float,
    *,
    guard: CandidateGuard | None = None,
) -> InterventionResult:
    """Build one deterministic 1x/2x/3x screening candidate.

    The base operator is evaluated once, then its geometric displacement is
    scaled around the nominal points.  Kinematic consistency is checked before
    the candidate is exposed to Guard; no CARLA outcome is inferred here.
    """
    if multiplier not in (1.0, 2.0, 3.0):
        raise ValueError("cora_screen_multiplier")
    source = base.provenance.source.value
    try:
        base_points = tuple(base.candidate.points)
        modified = _apply(operator, anchor, base_points, root_id)
        xy = []
        for original, changed in zip(base_points, modified):
            x = float(original.x) + multiplier * (float(changed.x) - float(original.x))
            y = float(original.y) + multiplier * (float(changed.y) - float(original.y))
            xy.append((x, y))
        # Reject before recomputation if the requested geometry would exceed
        # the fixed speed contract; _recompute intentionally caps only normal
        # generated trajectories and must not hide an over-limit screen arm.
        dt = 0.25
        previous = (float(anchor.bundle.ego_x), float(anchor.bundle.ego_y))
        for x, y in xy:
            if math.hypot(x - previous[0], y - previous[1]) / dt > 15.0 + 1e-9:
                raise InterventionNotApplicable("screen_speed_limit")
            previous = (x, y)
        points = _recompute(anchor, base_points, xy)
        proposal_hash = trajectory_sha256(points)
        metadata = {
            "base_proposal_id": base.candidate.candidate_id,
            "base_proposal_sha256": trajectory_sha256(base_points),
            "operator": operator,
            "magnitude": {**_magnitude(operator), "multiplier": multiplier},
            "offline_only": True,
            "screening": True,
        }
        candidate_id = f"{base.candidate.candidate_id}:cora:{operator}:x{int(multiplier)}"
        candidate = replace(
            base.candidate,
            candidate_id=candidate_id,
            points=points,
            intended_action=f"cora_screen:{operator}:x{int(multiplier)}",
            dynamics_meta={**dict(base.candidate.dynamics_meta), "cora_intervention": metadata},
        )
        provenance = replace(
            base.provenance,
            candidate_id=candidate_id,
            generator_id="cora-offline-screen@1.0.0",
            generator_hash=stable_sha256({"operator": operator, "multiplier": multiplier}),
            raw_sha256=proposal_hash,
            canonical_sha256=proposal_hash,
            generation_latency_s=0.0,
            freshness_s=base.provenance.freshness_s,
        )
        item = HybridCandidate(candidate=candidate, provenance=provenance)
        guard_result = (guard or CandidateGuard()).evaluate_candidate(
            HybridCandidateSet(anchor=anchor, candidates=(item,)), item)
        return InterventionResult(root_id, source, operator,
            "GUARD_ELIGIBLE" if guard_result.passed else "AUXILIARY_REJECT",
            candidate=item.with_guard(guard_result), guard=guard_result)
    except InterventionNotApplicable:
        raise
    except (ValueError, ArithmeticError) as exc:
        return InterventionResult(root_id, source, operator, "CANONICALIZATION_FAILED", error=f"{type(exc).__name__}:{exc}")


__all__ = [
    "FAMILY_OPERATORS",
    "InterventionNotApplicable",
    "InterventionResult",
    "derive_interventions",
    "derive_scaled_intervention",
]
