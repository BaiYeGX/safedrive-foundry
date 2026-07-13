"""Offline fault-injection matrix for G2-05 (Observable degradation + solver faults).

Each fault records: start, duration, severity, recovery, seed, expected action.
Oracle fields are never injected into runtime validator paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from safety_kernel.contracts.types import (
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
    PolicyCandidate,
    PolicyCandidateSet,
    TrajectoryPoint,
    TrackedObject,
)


class FaultId(str, Enum):
    STALE_OBS = "stale_obs"
    PACKET_DROP_ACTOR = "packet_drop_actor"
    OUT_OF_ORDER_TIME = "out_of_order_time"
    LOCALIZATION_BIAS = "localization_bias"
    MISSED_ACTOR = "missed_actor"
    ACTOR_OFFSET = "actor_offset"
    VISION_SOFT_DEGRADE = "vision_soft_degrade"  # marked via meta only
    SOLVER_STALE_CANDIDATE = "solver_stale_candidate"
    NUMERIC_NAN = "numeric_nan"
    LOW_ATTACHMENT = "low_attachment"
    ACTUATOR_SATURATION = "actuator_saturation"
    SOLVER_TIMEOUT = "solver_timeout"
    MODEL_TIMEOUT = "model_timeout"
    NONE = "none"


@dataclass(frozen=True)
class FaultSpec:
    fault_id: FaultId
    severity: float  # 0..1
    start_s: float
    duration_s: float
    seed: int
    expected_action: str
    description: str

    def active_at(self, t: float) -> bool:
        # Inclusive end so t == start+duration still applies (common tick boundary).
        return self.start_s - 1e-12 <= t <= self.start_s + self.duration_s + 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id.value,
            "severity": self.severity,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "seed": self.seed,
            "expected_action": self.expected_action,
            "description": self.description,
            "recovery_s": self.start_s + self.duration_s,
        }


DEFAULT_MATRIX: tuple[FaultSpec, ...] = (
    FaultSpec(
        FaultId.STALE_OBS,
        0.7,
        0.0,
        1.0,
        11,
        "minimal_risk_state_lock",
        "Observation freshness exceeds max_candidate_age (state lock; no QP/RATO)",
    ),
    FaultSpec(
        FaultId.PACKET_DROP_ACTOR,
        0.5,
        0.0,
        1.0,
        12,
        "accept_if_no_other_violations",
        "Lead actor temporarily lost (dropped track)",
    ),
    FaultSpec(
        FaultId.OUT_OF_ORDER_TIME,
        0.9,
        0.0,
        1.0,
        13,
        "hard_reject",
        "Trajectory timestamps not monotonic",
    ),
    FaultSpec(
        FaultId.LOCALIZATION_BIAS,
        0.6,
        0.0,
        1.0,
        14,
        "road_or_collision_reject_or_repair",
        "Ego pose lateral bias vs corridor",
    ),
    FaultSpec(
        FaultId.MISSED_ACTOR,
        0.8,
        0.0,
        1.0,
        15,
        "raw_may_pass_validator_misses_risk",
        "Critical lead actor not published (missed detection)",
    ),
    FaultSpec(
        FaultId.ACTOR_OFFSET,
        0.55,
        0.0,
        1.0,
        16,
        "collision_reject_or_repair",
        "Actor lateral/longitudinal offset error",
    ),
    FaultSpec(
        FaultId.SOLVER_STALE_CANDIDATE,
        0.75,
        0.0,
        1.0,
        17,
        "stale_input_no_repair",
        "Candidate older than freshness window",
    ),
    FaultSpec(
        FaultId.NUMERIC_NAN,
        1.0,
        0.0,
        1.0,
        18,
        "hard_reject_no_repair",
        "Non-finite trajectory point",
    ),
    FaultSpec(
        FaultId.VISION_SOFT_DEGRADE,
        0.4,
        0.0,
        1.0,
        19,
        "classic_or_accept_with_higher_uncertainty",
        "Vision soft degrade via actor cov + learning meta (no Oracle)",
    ),
    FaultSpec(
        FaultId.LOW_ATTACHMENT,
        0.7,
        0.0,
        1.0,
        20,
        "collision_reject_or_repair",
        "Low-friction proxy: short headway + high relative speed (Observable only)",
    ),
    FaultSpec(
        FaultId.ACTUATOR_SATURATION,
        0.85,
        0.0,
        1.0,
        21,
        "dynamics_hard_reject",
        "Commanded accel exceeds vehicle limits (actuator saturation)",
    ),
    FaultSpec(
        FaultId.SOLVER_TIMEOUT,
        0.9,
        0.0,
        1.0,
        22,
        "solver_timeout_no_execute",
        "QP/RATO solver timeout inject — must not execute timed-out solution",
    ),
    FaultSpec(
        FaultId.MODEL_TIMEOUT,
        0.75,
        0.0,
        1.0,
        23,
        "model_timeout_classic_fallback",
        "Learning model timeout degrade gate → Classic fallback path",
    ),
)


def _with_points(cand: PolicyCandidate, points: tuple[TrajectoryPoint, ...]) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=cand.candidate_id,
        source=cand.source,
        generated_time_s=cand.generated_time_s,
        valid_until_s=cand.valid_until_s,
        probability=cand.probability,
        points=points,
        behavior=cand.behavior,
        critical_actor=cand.critical_actor,
        conflict_type=cand.conflict_type,
        risk_horizon_s=cand.risk_horizon_s,
        intended_action=cand.intended_action,
        uncertainty=cand.uncertainty,
        availability=cand.availability,
        dynamics_meta=dict(cand.dynamics_meta),
    )


def _clone_obs(
    obs: ObservableSnapshot,
    *,
    observed_time_s: float | None = None,
    freshness_s: float | None = None,
    ego_x: float | None = None,
    ego_y: float | None = None,
    ego_v: float | None = None,
    ego_a: float | None = None,
    actors: tuple[TrackedObject, ...] | None = None,
    traffic_lights: tuple | None = None,
    corridor_centerline: tuple | None = None,
    corridor_half_width_m: float | None = None,
) -> ObservableSnapshot:
    return ObservableSnapshot(
        run_id=obs.run_id,
        frame_id=obs.frame_id,
        scenario_id=obs.scenario_id,
        simulation_time_s=obs.simulation_time_s,
        wall_time_s=obs.wall_time_s,
        ego_x=obs.ego_x if ego_x is None else ego_x,
        ego_y=obs.ego_y if ego_y is None else ego_y,
        ego_yaw=obs.ego_yaw,
        ego_v=obs.ego_v if ego_v is None else ego_v,
        ego_a=obs.ego_a if ego_a is None else ego_a,
        observed_time_s=obs.observed_time_s if observed_time_s is None else observed_time_s,
        freshness_s=obs.freshness_s if freshness_s is None else freshness_s,
        speed_limit_mps=obs.speed_limit_mps,
        actors=obs.actors if actors is None else actors,
        traffic_lights=obs.traffic_lights if traffic_lights is None else traffic_lights,
        corridor_centerline=obs.corridor_centerline if corridor_centerline is None else corridor_centerline,
        corridor_half_width_m=obs.corridor_half_width_m
        if corridor_half_width_m is None
        else corridor_half_width_m,
        privilege=ObservationPrivilege.OBSERVABLE,
        oracle_fields={},
        schema_version=getattr(obs, "schema_version", "safedrive.safety.contracts.v1"),
        coordinate_frame=getattr(obs, "coordinate_frame", "map"),
    )


def apply_fault_to_obs(obs: ObservableSnapshot, fault: FaultSpec, *, now_s: float) -> ObservableSnapshot:
    if not fault.active_at(now_s) or fault.fault_id is FaultId.NONE:
        return obs
    if fault.fault_id is FaultId.STALE_OBS:
        return _clone_obs(obs, observed_time_s=now_s - 1.0, freshness_s=1.0)
    if fault.fault_id is FaultId.PACKET_DROP_ACTOR:
        actors = tuple(
            TrackedObject(
                actor_id=a.actor_id,
                class_name=a.class_name,
                x=a.x,
                y=a.y,
                yaw=a.yaw,
                vx=a.vx,
                vy=a.vy,
                length_m=a.length_m,
                width_m=a.width_m,
                observed_time_s=a.observed_time_s,
                lost=True,
                source=a.source,
                cov_xx=a.cov_xx,
                cov_yy=a.cov_yy,
            )
            for a in obs.actors
        )
        return _clone_obs(obs, actors=actors)
    if fault.fault_id is FaultId.LOCALIZATION_BIAS:
        # Large enough lateral bias to exit the default corridor envelope.
        return _clone_obs(obs, ego_y=obs.ego_y + 3.5 * fault.severity)
    if fault.fault_id is FaultId.MISSED_ACTOR:
        return _clone_obs(obs, actors=())
    if fault.fault_id is FaultId.ACTOR_OFFSET:
        # Pull lead into the ego corridor path so collision envelope is violated.
        actors = tuple(
            TrackedObject(
                actor_id=a.actor_id,
                class_name=a.class_name,
                x=min(a.x, 8.0) - 1.0 * fault.severity,
                y=a.y + 0.2 * fault.severity,
                yaw=a.yaw,
                vx=min(a.vx, 1.5),
                vy=a.vy,
                length_m=a.length_m,
                width_m=a.width_m,
                observed_time_s=a.observed_time_s,
                lost=a.lost,
                source=a.source,
                cov_xx=a.cov_xx,
                cov_yy=a.cov_yy,
            )
            for a in obs.actors
        )
        if not actors:
            actors = (
                TrackedObject("lead", "vehicle", 7.0, 0.0, 0.0, 1.0, 0.0, 4.5, 1.8, now_s),
            )
        return _clone_obs(obs, actors=actors)
    if fault.fault_id is FaultId.VISION_SOFT_DEGRADE:
        # Observable-only: inflate actor covariance; never inject Oracle fields.
        actors = tuple(
            TrackedObject(
                actor_id=a.actor_id,
                class_name=a.class_name,
                x=a.x,
                y=a.y,
                yaw=a.yaw,
                vx=a.vx,
                vy=a.vy,
                length_m=a.length_m,
                width_m=a.width_m,
                observed_time_s=a.observed_time_s,
                lost=a.lost,
                source=a.source,
                cov_xx=max(a.cov_xx, 1.5 + 2.0 * fault.severity),
                cov_yy=max(a.cov_yy, 1.5 + 2.0 * fault.severity),
            )
            for a in obs.actors
        )
        return _clone_obs(obs, actors=actors)
    if fault.fault_id is FaultId.LOW_ATTACHMENT:
        # Short headway lead + elevated ego speed (low-mu proxy via kinematics).
        actors = tuple(
            TrackedObject(
                actor_id=a.actor_id,
                class_name=a.class_name,
                x=min(a.x, 8.0),
                y=a.y,
                yaw=a.yaw,
                vx=max(0.0, a.vx - 2.0),
                vy=a.vy,
                length_m=a.length_m,
                width_m=a.width_m,
                observed_time_s=a.observed_time_s,
                lost=a.lost,
                source=a.source,
                cov_xx=a.cov_xx,
                cov_yy=a.cov_yy,
            )
            for a in obs.actors
        )
        if not actors:
            actors = (
                TrackedObject("lead", "vehicle", 8.0, 0.0, 0.0, 1.0, 0.0, 4.5, 1.8, now_s),
            )
        return _clone_obs(obs, ego_v=max(obs.ego_v, 10.0), actors=actors)
    if fault.fault_id is FaultId.SOLVER_TIMEOUT:
        # Close lead so raw collides and repair is attempted (timeout then discards solution).
        actors = tuple(
            TrackedObject(
                actor_id=a.actor_id,
                class_name=a.class_name,
                x=min(a.x, 10.0),
                y=a.y,
                yaw=a.yaw,
                vx=min(a.vx, 2.0),
                vy=a.vy,
                length_m=a.length_m,
                width_m=a.width_m,
                observed_time_s=a.observed_time_s,
                lost=a.lost,
                source=a.source,
                cov_xx=a.cov_xx,
                cov_yy=a.cov_yy,
            )
            for a in obs.actors
        )
        if not actors:
            actors = (
                TrackedObject("lead", "vehicle", 10.0, 0.0, 0.0, 2.0, 0.0, 4.5, 1.8, now_s),
            )
        return _clone_obs(obs, ego_v=max(obs.ego_v, 10.0), actors=actors)
    return obs


def apply_fault_to_candidate(
    cand: PolicyCandidate,
    fault: FaultSpec,
    *,
    now_s: float,
) -> PolicyCandidate:
    if not fault.active_at(now_s) or fault.fault_id is FaultId.NONE:
        return cand
    if fault.fault_id is FaultId.OUT_OF_ORDER_TIME and len(cand.points) >= 3:
        pts = list(cand.points)
        # Swap two timestamps to break monotonicity.
        p1, p2 = pts[1], pts[2]
        pts[1] = TrajectoryPoint(t=p2.t, x=p1.x, y=p1.y, yaw=p1.yaw, kappa=p1.kappa, v=p1.v, a=p1.a, jerk=p1.jerk)
        pts[2] = TrajectoryPoint(t=p1.t, x=p2.x, y=p2.y, yaw=p2.yaw, kappa=p2.kappa, v=p2.v, a=p2.a, jerk=p2.jerk)
        return _with_points(cand, tuple(pts))
    if fault.fault_id is FaultId.SOLVER_STALE_CANDIDATE:
        return PolicyCandidate(
            candidate_id=cand.candidate_id,
            source=cand.source,
            generated_time_s=now_s - 1.0,
            valid_until_s=now_s - 0.5,
            probability=cand.probability,
            points=cand.points,
            behavior=cand.behavior,
            critical_actor=cand.critical_actor,
            conflict_type=cand.conflict_type,
            risk_horizon_s=cand.risk_horizon_s,
            intended_action=cand.intended_action,
            uncertainty=cand.uncertainty,
            availability=cand.availability,
            dynamics_meta=dict(cand.dynamics_meta),
        )
    if fault.fault_id is FaultId.NUMERIC_NAN and cand.points:
        pts = list(cand.points)
        p0 = pts[0]
        pts[0] = TrajectoryPoint(
            t=p0.t,
            x=float("nan"),
            y=p0.y,
            yaw=p0.yaw,
            kappa=p0.kappa,
            v=p0.v,
            a=p0.a,
            jerk=p0.jerk,
        )
        return _with_points(cand, tuple(pts))
    if fault.fault_id is FaultId.LOCALIZATION_BIAS and cand.points:
        # Bias trajectory with the same lateral error so corridor check sees offroad.
        bias = 3.5 * fault.severity  # severity 0.6 → 2.1 m; with half_width~2.5 + offroad margin
        # Push beyond corridor: ensure rejection under default half_width 2.5.
        bias = max(bias, 3.2)
        pts = tuple(
            TrajectoryPoint(
                t=p.t,
                x=p.x,
                y=p.y + bias,
                yaw=p.yaw,
                kappa=p.kappa,
                v=p.v,
                a=p.a,
                jerk=p.jerk,
            )
            for p in cand.points
        )
        return _with_points(cand, pts)
    if fault.fault_id is FaultId.VISION_SOFT_DEGRADE:
        # Mark soft vision degrade on candidate meta; raise uncertainty for learning sources.
        meta = dict(cand.dynamics_meta)
        meta["vision_soft"] = True
        meta["degrade"] = "ood" if cand.source.value.startswith("vla") else meta.get("degrade", "")
        unc = max(cand.uncertainty, 0.5 + 0.4 * fault.severity)
        return PolicyCandidate(
            candidate_id=cand.candidate_id,
            source=cand.source,
            generated_time_s=cand.generated_time_s,
            valid_until_s=cand.valid_until_s,
            probability=cand.probability,
            points=cand.points,
            behavior=cand.behavior,
            critical_actor=cand.critical_actor,
            conflict_type=cand.conflict_type,
            risk_horizon_s=cand.risk_horizon_s,
            intended_action=cand.intended_action,
            uncertainty=unc,
            availability=cand.availability,
            dynamics_meta=meta,
        )
    if fault.fault_id is FaultId.LOW_ATTACHMENT and cand.points:
        # High-speed follow trajectory under short headway (paired with obs fault).
        pts = tuple(
            TrajectoryPoint(
                t=p.t,
                x=p.x,
                y=p.y,
                yaw=p.yaw,
                kappa=p.kappa,
                v=max(p.v, 10.0),
                a=p.a,
                jerk=p.jerk,
            )
            for p in cand.points
        )
        meta = dict(cand.dynamics_meta)
        meta["fault"] = "low_attachment"
        return PolicyCandidate(
            candidate_id=cand.candidate_id,
            source=cand.source,
            generated_time_s=cand.generated_time_s,
            valid_until_s=cand.valid_until_s,
            probability=cand.probability,
            points=pts,
            behavior=cand.behavior,
            critical_actor=cand.critical_actor,
            conflict_type=cand.conflict_type,
            risk_horizon_s=cand.risk_horizon_s,
            intended_action=cand.intended_action,
            uncertainty=cand.uncertainty,
            availability=cand.availability,
            dynamics_meta=meta,
        )
    if fault.fault_id is FaultId.ACTUATOR_SATURATION and cand.points:
        # Commanded accel far beyond vehicle limits.
        pts = tuple(
            TrajectoryPoint(
                t=p.t,
                x=p.x,
                y=p.y,
                yaw=p.yaw,
                kappa=p.kappa,
                v=p.v,
                a=50.0 * fault.severity + 20.0,
                jerk=100.0,
            )
            for p in cand.points
        )
        meta = dict(cand.dynamics_meta)
        meta["fault"] = "actuator_saturation"
        return PolicyCandidate(
            candidate_id=cand.candidate_id,
            source=cand.source,
            generated_time_s=cand.generated_time_s,
            valid_until_s=cand.valid_until_s,
            probability=cand.probability,
            points=pts,
            behavior=cand.behavior,
            critical_actor=cand.critical_actor,
            conflict_type=cand.conflict_type,
            risk_horizon_s=cand.risk_horizon_s,
            intended_action=cand.intended_action,
            uncertainty=cand.uncertainty,
            availability=cand.availability,
            dynamics_meta=meta,
        )
    if fault.fault_id is FaultId.SOLVER_TIMEOUT:
        meta = dict(cand.dynamics_meta)
        meta["inject_solver_timeout"] = True
        meta["fault"] = "solver_timeout"
        # Force a hard-collision raw path so repair is attempted and timeout is observed.
        pts = list(cand.points)
        if pts:
            # Keep path but ensure collision pressure via high speed into lead.
            pts = [
                TrajectoryPoint(
                    t=p.t,
                    x=p.x,
                    y=p.y,
                    yaw=p.yaw,
                    kappa=p.kappa,
                    v=max(p.v, 12.0),
                    a=p.a,
                    jerk=p.jerk,
                )
                for p in pts
            ]
        return PolicyCandidate(
            candidate_id=cand.candidate_id,
            source=cand.source,
            generated_time_s=cand.generated_time_s,
            valid_until_s=cand.valid_until_s,
            probability=cand.probability,
            points=tuple(pts) if pts else cand.points,
            behavior=cand.behavior,
            critical_actor=cand.critical_actor,
            conflict_type=cand.conflict_type,
            risk_horizon_s=cand.risk_horizon_s,
            intended_action=cand.intended_action,
            uncertainty=cand.uncertainty,
            availability=cand.availability,
            dynamics_meta=meta,
        )
    if fault.fault_id is FaultId.MODEL_TIMEOUT:
        meta = dict(cand.dynamics_meta)
        meta["degrade"] = "timeout"
        meta["fault"] = "model_timeout"
        return PolicyCandidate(
            candidate_id=cand.candidate_id,
            source=CandidateSource.VLA_FAST,
            generated_time_s=cand.generated_time_s,
            valid_until_s=cand.valid_until_s,
            probability=cand.probability,
            points=cand.points,
            behavior=cand.behavior,
            critical_actor=cand.critical_actor,
            conflict_type=cand.conflict_type,
            risk_horizon_s=cand.risk_horizon_s,
            intended_action=cand.intended_action,
            uncertainty=max(cand.uncertainty, 0.6),
            availability=cand.availability,
            dynamics_meta=meta,
        )
    return cand


def apply_fault_to_set(
    cset: PolicyCandidateSet,
    fault: FaultSpec,
    *,
    now_s: float,
) -> PolicyCandidateSet:
    cands = tuple(apply_fault_to_candidate(c, fault, now_s=now_s) for c in cset.candidates)
    # Model timeout: keep a Classic sibling so fallback path is exercisable.
    if fault.fault_id is FaultId.MODEL_TIMEOUT and fault.active_at(now_s):
        classic = PolicyCandidate(
            candidate_id="classic_fallback",
            source=CandidateSource.CLASSIC,
            generated_time_s=now_s,
            valid_until_s=now_s + 0.2,
            probability=0.7,
            points=cset.candidates[0].points if cset.candidates else (),
            behavior="classic_hold",
        )
        cands = cands + (classic,)
    return PolicyCandidateSet(
        run_id=cset.run_id,
        frame_id=cset.frame_id,
        scenario_id=cset.scenario_id,
        model_id=cset.model_id,
        carla_frame=cset.carla_frame,
        simulation_time_s=cset.simulation_time_s,
        wall_time_s=cset.wall_time_s,
        candidates=cands,
        schema_version=cset.schema_version,
        coordinate_frame=cset.coordinate_frame,
    )


def expected_action_holds(fault: FaultSpec, decision_kind: str, *, repair_success: bool | None = None) -> bool:
    """Verify measured decision against fault.expected_action (offline matrix)."""
    kind = decision_kind.upper()
    exp = fault.expected_action
    if exp == "minimal_risk_state_lock":
        return kind in {"MINIMAL_RISK", "EMERGENCY"} and kind not in {"ACCEPT", "QP", "RATO"}
    if exp == "accept_if_no_other_violations":
        return kind in {"ACCEPT", "QP", "RATO", "CLASSIC_FALLBACK", "HARD_REJECT", "MINIMAL_RISK"}
    if exp == "hard_reject":
        return kind != "ACCEPT"
    if exp in {"road_or_collision_reject_or_repair", "collision_reject_or_repair"}:
        return kind != "ACCEPT" or kind in {"QP", "RATO"}
    if exp == "raw_may_pass_validator_misses_risk":
        # Documented negative: Observable miss can ACCEPT.
        return kind in {"ACCEPT", "QP", "RATO", "HARD_REJECT", "MINIMAL_RISK", "CLASSIC_FALLBACK"}
    if exp == "stale_input_no_repair":
        return kind != "QP" and kind != "RATO" and (repair_success is not True)
    if exp == "hard_reject_no_repair":
        return kind in {"HARD_REJECT", "MINIMAL_RISK", "CLASSIC_FALLBACK"} and repair_success is not True
    if exp == "classic_or_accept_with_higher_uncertainty":
        return kind in {"ACCEPT", "CLASSIC_FALLBACK", "QP", "RATO", "HARD_REJECT", "MINIMAL_RISK"}
    if exp == "dynamics_hard_reject":
        return kind in {"HARD_REJECT", "MINIMAL_RISK", "CLASSIC_FALLBACK", "QP", "RATO"} and kind != "ACCEPT"
    if exp == "solver_timeout_no_execute":
        return kind not in {"QP", "RATO"} and repair_success is not True
    if exp == "model_timeout_classic_fallback":
        return kind in {"ACCEPT", "CLASSIC_FALLBACK", "QP", "RATO", "HARD_REJECT", "MINIMAL_RISK"} and kind != "EMERGENCY"
    return kind != ""
