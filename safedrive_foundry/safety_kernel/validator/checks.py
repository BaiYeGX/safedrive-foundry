"""Atomic hard checks for Safety Validator (Observable inputs only).

Collision envelope: constant-velocity actor prediction + circular inflation
(not full polygon / map-aware collision). Red-light rule is a near-zone approach
speed gate aligned with longitudinal QP constraints — not a full stop-line planner.
"""

from __future__ import annotations

import math
from typing import Iterable

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    ConstraintMargin,
    ObservableSnapshot,
    ObservationPrivilege,
    PolicyCandidate,
    PolicyCandidateSet,
    TrackedObject,
)


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def check_privilege(obs: ObservableSnapshot) -> ConstraintMargin:
    """Runtime path must not consume Oracle privilege snapshots."""
    if obs.privilege is ObservationPrivilege.ORACLE:
        return ConstraintMargin(
            name="privilege",
            margin=-1.0,
            hard=True,
            message="runtime validator rejects Oracle-privilege observations",
        )
    if obs.oracle_fields:
        return ConstraintMargin(
            name="privilege",
            margin=-1.0,
            hard=True,
            message="oracle_fields present on runtime observation",
        )
    return ConstraintMargin(name="privilege", margin=1.0, hard=True, message="observable_ok")


def check_numeric(candidate: PolicyCandidate) -> ConstraintMargin:
    if not candidate.points:
        return ConstraintMargin(
            name="numeric",
            margin=-1.0,
            hard=True,
            message="empty_trajectory",
        )
    for i, p in enumerate(candidate.points):
        if not _finite(p.t, p.x, p.y, p.yaw, p.kappa, p.v, p.a, p.jerk):
            return ConstraintMargin(
                name="numeric",
                margin=-1.0,
                hard=True,
                first_violation_time_s=p.t,
                message=f"non_finite_at_index_{i}",
            )
    if not _finite(candidate.generated_time_s, candidate.valid_until_s, candidate.probability, candidate.uncertainty):
        return ConstraintMargin(name="numeric", margin=-1.0, hard=True, message="non_finite_meta")
    return ConstraintMargin(name="numeric", margin=1.0, hard=True, message="ok")


def check_schema_fields(candidate: PolicyCandidate, cfg: SafetyKernelConfig) -> ConstraintMargin:
    if not candidate.candidate_id:
        return ConstraintMargin(name="schema", margin=-1.0, hard=True, message="missing_candidate_id")
    if len(candidate.points) < cfg.min_points:
        return ConstraintMargin(
            name="schema",
            margin=-1.0,
            hard=True,
            message=f"too_few_points:{len(candidate.points)}<{cfg.min_points}",
        )
    if candidate.probability < 0.0 or candidate.probability > 1.0:
        return ConstraintMargin(name="schema", margin=-1.0, hard=True, message="probability_out_of_range")
    horizon = candidate.horizon_s
    if horizon + 1e-9 < cfg.min_horizon_s or horizon - 1e-9 > cfg.max_horizon_s:
        return ConstraintMargin(
            name="schema",
            margin=-1.0,
            hard=True,
            message=f"horizon_out_of_range:{horizon:.3f}",
        )
    return ConstraintMargin(name="schema", margin=1.0, hard=True, message="ok")


def check_freshness(
    candidate: PolicyCandidate,
    *,
    now_s: float,
    cfg: SafetyKernelConfig,
) -> ConstraintMargin:
    age = now_s - candidate.generated_time_s
    if age > cfg.max_candidate_age_s:
        return ConstraintMargin(
            name="freshness",
            margin=cfg.max_candidate_age_s - age,
            hard=True,
            message=f"stale_age:{age:.4f}",
        )
    if candidate.valid_until_s < now_s:
        return ConstraintMargin(
            name="freshness",
            margin=candidate.valid_until_s - now_s,
            hard=True,
            message="expired",
        )
    if candidate.generated_time_s > now_s + 1e-3:
        return ConstraintMargin(
            name="freshness",
            margin=-1.0,
            hard=True,
            message="generated_in_future",
        )
    return ConstraintMargin(
        name="freshness",
        margin=min(cfg.max_candidate_age_s - age, candidate.valid_until_s - now_s),
        hard=True,
        message="ok",
    )


def check_time_order(candidate: PolicyCandidate) -> ConstraintMargin:
    prev_t = None
    for p in candidate.points:
        if prev_t is not None and p.t <= prev_t + 1e-9:
            return ConstraintMargin(
                name="time_order",
                margin=-1.0,
                hard=True,
                first_violation_time_s=p.t,
                message="non_monotonic_time",
            )
        prev_t = p.t
    return ConstraintMargin(name="time_order", margin=1.0, hard=True, message="ok")


def _point_to_polyline_distance(x: float, y: float, polyline: tuple[tuple[float, float], ...]) -> float:
    if len(polyline) < 2:
        if not polyline:
            return float("inf")
        px, py = polyline[0]
        return math.hypot(x - px, y - py)
    best = float("inf")
    for i in range(len(polyline) - 1):
        x0, y0 = polyline[i]
        x1, y1 = polyline[i + 1]
        dx, dy = x1 - x0, y1 - y0
        if dx == 0.0 and dy == 0.0:
            dist = math.hypot(x - x0, y - y0)
        else:
            t = ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist = math.hypot(x - proj_x, y - proj_y)
        if dist < best:
            best = dist
    return best


def check_road(
    candidate: PolicyCandidate,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
) -> ConstraintMargin:
    if not cfg.require_drivable:
        return ConstraintMargin(name="road", margin=1.0, hard=True, message="disabled")
    if not obs.corridor_centerline:
        # Without corridor geometry, only check ego proximity start (cannot invent map).
        return ConstraintMargin(name="road", margin=0.0, hard=False, message="no_corridor_soft")
    half = obs.corridor_half_width_m if obs.corridor_half_width_m > 0 else cfg.lane_half_width_m
    limit = half + cfg.max_offroad_m
    worst = float("inf")
    first_t = None
    for p in candidate.points:
        dist = _point_to_polyline_distance(p.x, p.y, obs.corridor_centerline)
        margin = limit - dist
        if margin < worst:
            worst = margin
        if margin < 0.0 and first_t is None:
            first_t = p.t
    if worst < 0.0:
        return ConstraintMargin(
            name="road",
            margin=worst,
            hard=True,
            first_violation_time_s=first_t,
            message="offroad",
        )
    return ConstraintMargin(name="road", margin=worst, hard=True, message="ok")


def check_dynamics(candidate: PolicyCandidate, cfg: SafetyKernelConfig) -> ConstraintMargin:
    worst = float("inf")
    first_t = None
    msg = "ok"
    for p in candidate.points:
        margins = [
            cfg.max_speed_mps - abs(p.v),
            cfg.max_accel_mps2 - p.a if p.a >= 0 else cfg.max_decel_mps2 + p.a,
            cfg.max_jerk_mps3 - abs(p.jerk),
            cfg.max_curvature_per_m - abs(p.kappa),
            cfg.max_lateral_accel_mps2 - abs(p.kappa) * (p.v * p.v),
        ]
        local = min(margins)
        if local < worst:
            worst = local
            if local < 0.0 and first_t is None:
                first_t = p.t
                if margins[0] < 0:
                    msg = "speed"
                elif margins[1] < 0:
                    msg = "accel"
                elif margins[2] < 0:
                    msg = "jerk"
                elif margins[3] < 0:
                    msg = "curvature"
                else:
                    msg = "lat_accel"
    if worst < 0.0:
        return ConstraintMargin(
            name="dynamics",
            margin=worst,
            hard=True,
            first_violation_time_s=first_t,
            message=msg,
        )
    return ConstraintMargin(name="dynamics", margin=worst if math.isfinite(worst) else 1.0, hard=True, message="ok")


def _obb_signed_margin(
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    actor_x: float,
    actor_y: float,
    actor_yaw: float,
    actor_length_m: float,
    actor_width_m: float,
    cfg: SafetyKernelConfig,
) -> float:
    """SAT separation for two oriented vehicle boxes.

    Positive means a separating gap exists.  Negative means the boxes overlap,
    with the magnitude representing the shallowest penetration.  This avoids
    the old half-diagonal circles which treated adjacent lanes as collisions.
    """
    ego_axes = (
        (math.cos(ego_yaw), math.sin(ego_yaw)),
        (-math.sin(ego_yaw), math.cos(ego_yaw)),
    )
    actor_axes = (
        (math.cos(actor_yaw), math.sin(actor_yaw)),
        (-math.sin(actor_yaw), math.cos(actor_yaw)),
    )
    delta = (actor_x - ego_x, actor_y - ego_y)
    ego_half = (0.5 * cfg.length_m, 0.5 * cfg.width_m)
    actor_half = (0.5 * actor_length_m, 0.5 * actor_width_m)
    worst_gap = -float("inf")
    for axis in (*ego_axes, *actor_axes):
        center_distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        ego_radius = (
            ego_half[0] * abs(ego_axes[0][0] * axis[0] + ego_axes[0][1] * axis[1])
            + ego_half[1] * abs(ego_axes[1][0] * axis[0] + ego_axes[1][1] * axis[1])
        )
        actor_radius = (
            actor_half[0] * abs(actor_axes[0][0] * axis[0] + actor_axes[0][1] * axis[1])
            + actor_half[1] * abs(actor_axes[1][0] * axis[0] + actor_axes[1][1] * axis[1])
        )
        gap = center_distance - ego_radius - actor_radius - cfg.collision_inflate_m
        worst_gap = max(worst_gap, gap)
    return worst_gap


def check_collision(
    candidate: PolicyCandidate,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
) -> ConstraintMargin:
    # Defense-in-depth: non-finite / illegal actors must not yield ACCEPT.
    for actor in obs.actors:
        if actor.lost:
            continue
        if not _finite(
            actor.x,
            actor.y,
            actor.yaw,
            actor.vx,
            actor.vy,
            actor.length_m,
            actor.width_m,
            actor.observed_time_s,
            actor.cov_xx,
            actor.cov_yy,
        ):
            return ConstraintMargin(
                name="collision",
                margin=-1.0,
                hard=True,
                actor_id=actor.actor_id,
                message="non_finite_actor",
            )
        if actor.length_m <= 0.0 or actor.width_m <= 0.0:
            return ConstraintMargin(
                name="collision",
                margin=-1.0,
                hard=True,
                actor_id=actor.actor_id,
                message="illegal_actor_size",
            )
    worst = float("inf")
    first_t = None
    actor_hit = None
    for p in candidate.points:
        # Actor state is measured at actor.observed_time_s.  Candidate times are
        # relative to the current anchor, so the first point at t=0.25 must use
        # actor position + velocity*0.25 (the old code lagged every actor by one
        # planning step).
        for actor in obs.actors:
            if actor.lost:
                continue
            dt = max(0.0, obs.simulation_time_s - actor.observed_time_s + p.t)
            ax = actor.x + actor.vx * dt
            ay = actor.y + actor.vy * dt
            margin = _obb_signed_margin(
                ego_x=p.x,
                ego_y=p.y,
                ego_yaw=p.yaw,
                actor_x=ax,
                actor_y=ay,
                actor_yaw=actor.yaw,
                actor_length_m=actor.length_m,
                actor_width_m=actor.width_m,
                cfg=cfg,
            )
            if not math.isfinite(margin):
                return ConstraintMargin(
                    name="collision",
                    margin=-1.0,
                    hard=True,
                    first_violation_time_s=p.t,
                    actor_id=actor.actor_id,
                    message="non_finite_distance",
                )
            if margin < worst:
                worst = margin
            if margin < 0.0 and first_t is None:
                first_t = p.t
                actor_hit = actor.actor_id
    if not obs.actors:
        return ConstraintMargin(name="collision", margin=1.0, hard=True, message="no_actors")
    if not math.isfinite(worst):
        return ConstraintMargin(
            name="collision",
            margin=-1.0,
            hard=True,
            message="non_finite_margin",
        )
    if worst < 0.0:
        return ConstraintMargin(
            name="collision",
            margin=worst,
            hard=True,
            first_violation_time_s=first_t,
            actor_id=actor_hit,
            message="collision_envelope",
        )
    return ConstraintMargin(name="collision", margin=worst, hard=True, message="ok")


def check_rules(
    candidate: PolicyCandidate,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
) -> ConstraintMargin:
    worst = float("inf")
    first_t = None
    rule_id = None
    msg = "ok"
    if cfg.enforce_speed_limit and obs.speed_limit_mps is not None:
        limit = obs.speed_limit_mps + cfg.speed_limit_margin_mps
        for p in candidate.points:
            margin = limit - abs(p.v)
            if margin < worst:
                worst = margin
            if margin < 0.0 and first_t is None:
                first_t = p.t
                rule_id = "speed_limit"
                msg = "speed_limit"
    if cfg.enforce_red_light_stop:
        for light in obs.traffic_lights:
            if light.state != "red":
                continue
            if light.controls_ego_lane is False:
                continue
            distance = light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m
            if distance > cfg.red_light_stop_distance_m:
                continue
            # Judge the actual stop-line plan, not every approach point.  A car
            # may legally approach faster than 1 m/s if its planned braking can
            # still stop before the line.  Candidate path length is the only
            # observable route-progress proxy available in this contract.
            if distance < -1.0:
                continue
            travelled = 0.0
            previous_x, previous_y = obs.ego_x, obs.ego_y
            terminal_speed = 0.0
            crossed_at = None
            for p in candidate.points:
                travelled += math.hypot(p.x - previous_x, p.y - previous_y)
                previous_x, previous_y = p.x, p.y
                terminal_speed = abs(p.v)
                if travelled > distance + 0.25 and crossed_at is None:
                    crossed_at = p.t

            # Evaluate the trajectory as a plan.  Requiring every approach
            # point to be able to stop *again from that point* incorrectly
            # rejected trajectories that were already braking.  A legal plan
            # must not cross the line and, at its terminal state, must retain
            # enough room for a bounded stop beyond the finite horizon.
            crossing_margin = distance - travelled + 0.25
            terminal_remaining = distance - travelled
            terminal_stopping_distance = (
                terminal_speed * terminal_speed
                / max(0.1, 2.0 * cfg.max_decel_mps2)
                + 0.50
            )
            stopping_margin = terminal_remaining - terminal_stopping_distance
            margin = min(crossing_margin, stopping_margin)
            if margin < worst:
                worst = margin
            if margin < 0.0 and first_t is None:
                first_t = crossed_at if crossed_at is not None else candidate.points[-1].t
                rule_id = f"red_light:{light.light_id}"
                msg = (
                    "red_light_crossing"
                    if crossed_at is not None
                    else "red_light_unstoppable"
                )
    if worst < 0.0:
        return ConstraintMargin(
            name="rules",
            margin=worst,
            hard=True,
            first_violation_time_s=first_t,
            rule_id=rule_id,
            message=msg,
        )
    if not math.isfinite(worst):
        worst = 1.0
    return ConstraintMargin(name="rules", margin=worst, hard=True, message=msg)


def check_trackability(candidate: PolicyCandidate, cfg: SafetyKernelConfig) -> ConstraintMargin:
    """Feasibility of successive points under vehicle limits (tracking envelope)."""
    if len(candidate.points) < 2:
        return ConstraintMargin(name="trackability", margin=-1.0, hard=True, message="too_short")
    worst = float("inf")
    first_t = None
    msg = "ok"
    for i in range(1, len(candidate.points)):
        p0 = candidate.points[i - 1]
        p1 = candidate.points[i]
        dt = p1.t - p0.t
        if dt <= 1e-6:
            return ConstraintMargin(
                name="trackability",
                margin=-1.0,
                hard=True,
                first_violation_time_s=p1.t,
                message="zero_dt",
            )
        dist = math.hypot(p1.x - p0.x, p1.y - p0.y)
        # Expected distance roughly average speed * dt; allow large slack but reject teleport jumps.
        max_step = (abs(p0.v) + abs(p1.v)) * 0.5 * dt + 0.5 * cfg.max_accel_mps2 * dt * dt + 1.0
        step_margin = max_step - dist
        # Heading change vs yaw-rate capability ~ v * kappa
        dyaw = abs((p1.yaw - p0.yaw + math.pi) % (2 * math.pi) - math.pi)
        max_dyaw = abs(p0.v) * cfg.max_curvature_per_m * dt + 0.35
        yaw_margin = max_dyaw - dyaw
        local = min(step_margin, yaw_margin)
        if local < worst:
            worst = local
        if local < 0.0 and first_t is None:
            first_t = p1.t
            msg = "teleport" if step_margin < yaw_margin else "yaw_rate"
    if worst < 0.0:
        return ConstraintMargin(
            name="trackability",
            margin=worst,
            hard=True,
            first_violation_time_s=first_t,
            message=msg,
        )
    return ConstraintMargin(name="trackability", margin=worst, hard=True, message="ok")


def run_prefilter_checks(
    candidate: PolicyCandidate,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
    *,
    now_s: float,
) -> list[ConstraintMargin]:
    """Hard pre-validation before expensive World/arbitration (industrial chain)."""
    return [
        check_privilege(obs),
        check_schema_fields(candidate, cfg),
        check_numeric(candidate),
        check_freshness(candidate, now_s=now_s, cfg=cfg),
        check_time_order(candidate),
    ]


def run_full_checks(
    candidate: PolicyCandidate,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
    *,
    now_s: float,
) -> list[ConstraintMargin]:
    return [
        *run_prefilter_checks(candidate, obs, cfg, now_s=now_s),
        check_road(candidate, obs, cfg),
        check_dynamics(candidate, cfg),
        check_collision(candidate, obs, cfg),
        check_rules(candidate, obs, cfg),
        check_trackability(candidate, cfg),
    ]


def check_actor_integrity(obs: ObservableSnapshot) -> ConstraintMargin:
    """Reject non-finite / illegal TrackedObject fields at the Observable boundary."""
    for actor in obs.actors:
        if not actor.actor_id:
            return ConstraintMargin(
                name="actor_numeric",
                margin=-1.0,
                hard=True,
                actor_id=actor.actor_id or None,
                message="missing_actor_id",
            )
        if not _finite(
            actor.x,
            actor.y,
            actor.yaw,
            actor.vx,
            actor.vy,
            actor.length_m,
            actor.width_m,
            actor.observed_time_s,
            actor.cov_xx,
            actor.cov_yy,
        ):
            return ConstraintMargin(
                name="actor_numeric",
                margin=-1.0,
                hard=True,
                actor_id=actor.actor_id,
                message="non_finite_actor",
            )
        if actor.length_m <= 0.0 or actor.width_m <= 0.0:
            return ConstraintMargin(
                name="actor_numeric",
                margin=-1.0,
                hard=True,
                actor_id=actor.actor_id,
                message="illegal_actor_size",
            )
    for light in obs.traffic_lights:
        if not _finite(light.distance_m, light.observed_time_s):
            return ConstraintMargin(
                name="actor_numeric",
                margin=-1.0,
                hard=True,
                rule_id=light.light_id,
                message="non_finite_traffic_light",
            )
        if light.stop_line_distance_m is not None and not _finite(light.stop_line_distance_m):
            return ConstraintMargin(
                name="actor_numeric",
                margin=-1.0,
                hard=True,
                rule_id=light.light_id,
                message="non_finite_stop_line",
            )
    return ConstraintMargin(name="actor_numeric", margin=1.0, hard=True, message="ok")


# Max |set.sim_time - obs.sim_time| for identity (seconds).
IDENTITY_TIME_TOL_S = 0.05
ALLOWED_COORDINATE_FRAMES = frozenset({"map", "ego", "odom"})


def check_set_identity_and_contract(
    candidate_set: PolicyCandidateSet,
    obs: ObservableSnapshot,
    *,
    expected_schema_version: str,
    time_tol_s: float = IDENTITY_TIME_TOL_S,
) -> list[ConstraintMargin]:
    """Hard identity / contract consistency between PolicyCandidateSet and ObservableSnapshot."""
    margins: list[ConstraintMargin] = []
    if candidate_set.run_id != obs.run_id:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"run_id_mismatch:{candidate_set.run_id!r}!={obs.run_id!r}",
            )
        )
    if candidate_set.frame_id != obs.frame_id:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"frame_id_mismatch:{candidate_set.frame_id!r}!={obs.frame_id!r}",
            )
        )
    if candidate_set.scenario_id != obs.scenario_id:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"scenario_id_mismatch:{candidate_set.scenario_id!r}!={obs.scenario_id!r}",
            )
        )
    set_schema = candidate_set.schema_version
    obs_schema = getattr(obs, "schema_version", expected_schema_version)
    if set_schema != expected_schema_version:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"schema_version_mismatch_set:{set_schema!r}",
            )
        )
    if obs_schema != expected_schema_version:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"schema_version_mismatch_obs:{obs_schema!r}",
            )
        )
    if set_schema != obs_schema:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"schema_version_set_obs_mismatch:{set_schema!r}!={obs_schema!r}",
            )
        )
    set_frame = candidate_set.coordinate_frame or "map"
    obs_frame = getattr(obs, "coordinate_frame", "map") or "map"
    if set_frame not in ALLOWED_COORDINATE_FRAMES:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"coordinate_frame_invalid:{set_frame!r}",
            )
        )
    if obs_frame not in ALLOWED_COORDINATE_FRAMES:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"coordinate_frame_invalid_obs:{obs_frame!r}",
            )
        )
    if set_frame != obs_frame:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"coordinate_frame_mismatch:{set_frame!r}!={obs_frame!r}",
            )
        )
    dt = abs(float(candidate_set.simulation_time_s) - float(obs.simulation_time_s))
    if dt > time_tol_s:
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"simulation_time_mismatch:dt={dt:.6f}",
            )
        )
    ids = [c.candidate_id for c in candidate_set.candidates]
    if len(ids) != len(set(ids)):
        # Find first duplicate for message.
        seen: set[str] = set()
        dup = ""
        for cid in ids:
            if cid in seen:
                dup = cid
                break
            seen.add(cid)
        margins.append(
            ConstraintMargin(
                name="identity",
                margin=-1.0,
                hard=True,
                message=f"duplicate_candidate_id:{dup}",
            )
        )
    if not margins:
        margins.append(ConstraintMargin(name="identity", margin=1.0, hard=True, message="ok"))
    return margins


def run_state_checks(
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
    *,
    now_s: float,
) -> list[ConstraintMargin]:
    """Lightweight 50Hz monitoring (no candidate trajectory)."""
    margins = [check_privilege(obs)]
    # observed_time_s is authoritative (0.0 is a valid timestamp at episode start).
    age = now_s - obs.observed_time_s
    freshness_margin = cfg.max_candidate_age_s - age
    margins.append(
        ConstraintMargin(
            name="obs_freshness",
            margin=freshness_margin,
            hard=True,
            message="stale_observation" if freshness_margin < 0 else "ok",
        )
    )
    if not _finite(obs.ego_x, obs.ego_y, obs.ego_yaw, obs.ego_v, obs.ego_a):
        margins.append(ConstraintMargin(name="ego_numeric", margin=-1.0, hard=True, message="non_finite_ego"))
    else:
        margins.append(ConstraintMargin(name="ego_numeric", margin=1.0, hard=True, message="ok"))
    margins.append(check_actor_integrity(obs))
    speed_m = cfg.max_speed_mps - abs(obs.ego_v) if math.isfinite(obs.ego_v) else -1.0
    margins.append(
        ConstraintMargin(
            name="ego_speed",
            margin=speed_m,
            hard=True,
            message="ego_overspeed" if speed_m < 0 else "ok",
        )
    )
    return margins


def hard_violations(margins: Iterable[ConstraintMargin]) -> list[ConstraintMargin]:
    return [m for m in margins if m.hard and m.margin < 0.0]
