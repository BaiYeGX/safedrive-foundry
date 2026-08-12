"""Branch outcome aggregation for R2 primary 2.5 s horizon.

Oracle diagnostics (clearance/TTC) are computed offline from privileged traces.
They must not be fed back into runtime control.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from driving_vla.evaluation.paired_contract import (
    EXPECTED_PRIMARY_TICKS,
    ORACLE_TRACE_FLAGS,
    PRIMARY_HORIZON_S,
)


@dataclass(frozen=True)
class TickRecord:
    """One control tick of runtime-observable fields (+ optional oracle fields)."""

    tick_index: int
    simulation_time_s: float
    ego_x: float
    ego_y: float
    ego_yaw_rad: float
    ego_v: float
    selected_candidate_id: str
    executed_candidate_id: str
    source_id: str
    path_age_s: float
    freshness_regime: str
    mpc_mode: str
    mpc_status: str
    mpc_latency_s: float
    collision: bool = False
    collision_impulse: float = 0.0
    offroad: bool = False
    lane_invasion: bool = False
    route_progress_m: float = 0.0
    longitudinal_accel: float = 0.0
    lateral_accel: float = 0.0
    jerk: float = 0.0
    steer_rate: float = 0.0
    # Oracle-only (must be flagged when serialized)
    actor_clearance_m: float | None = None
    ttc_s: float | None = None
    oracle_only: bool = False

    def runtime_dict(self) -> dict[str, Any]:
        """Fields allowed in control / runtime traces."""
        return {
            "tick_index": int(self.tick_index),
            "simulation_time_s": float(self.simulation_time_s),
            "ego_x": float(self.ego_x),
            "ego_y": float(self.ego_y),
            "ego_yaw_rad": float(self.ego_yaw_rad),
            "ego_v": float(self.ego_v),
            "selected_candidate_id": self.selected_candidate_id,
            "executed_candidate_id": self.executed_candidate_id,
            "source_id": self.source_id,
            "path_age_s": float(self.path_age_s),
            "freshness_regime": self.freshness_regime,
            "mpc_mode": self.mpc_mode,
            "mpc_status": self.mpc_status,
            "mpc_latency_s": float(self.mpc_latency_s),
            "collision": bool(self.collision),
            "collision_impulse": float(self.collision_impulse),
            "offroad": bool(self.offroad),
            "lane_invasion": bool(self.lane_invasion),
            "route_progress_m": float(self.route_progress_m),
            "longitudinal_accel": float(self.longitudinal_accel),
            "lateral_accel": float(self.lateral_accel),
            "jerk": float(self.jerk),
            "steer_rate": float(self.steer_rate),
        }

    def oracle_dict(self) -> dict[str, Any]:
        d = {
            **ORACLE_TRACE_FLAGS,
            "tick_index": int(self.tick_index),
            "simulation_time_s": float(self.simulation_time_s),
            "actor_clearance_m": self.actor_clearance_m,
            "ttc_s": self.ttc_s,
            "collision": bool(self.collision),
            "collision_impulse": float(self.collision_impulse),
            "offroad": bool(self.offroad),
            "lane_invasion": bool(self.lane_invasion),
            "route_progress_m": float(self.route_progress_m),
        }
        return d


@dataclass(frozen=True)
class BranchOutcomeMetrics:
    candidate_id: str
    candidate_index: int
    completed_primary_horizon: bool
    completed_primary_ticks: int
    collision_episode_count: int
    first_collision_time_s: float | None
    collision_impulse_sum: float
    offroad_fraction: float
    first_offroad_time_s: float | None
    lane_invasion_episode_count: int
    minimum_actor_clearance_m: float | None
    minimum_ttc_s: float | None
    route_progress_delta_m: float
    distance_m: float
    longitudinal_accel_abs_p95: float
    longitudinal_accel_abs_max: float
    jerk_abs_p95: float
    jerk_abs_max: float
    lateral_accel_abs_p95: float
    lateral_accel_abs_max: float
    steer_rate_abs_p95: float
    steer_rate_abs_max: float
    mpc_solved_count: int
    mpc_timeout_count: int
    mpc_fallback_count: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": int(self.candidate_index),
            "completed_primary_horizon": bool(self.completed_primary_horizon),
            "completed_primary_ticks": int(self.completed_primary_ticks),
            "collision_episode_count": int(self.collision_episode_count),
            "first_collision_time_s": self.first_collision_time_s,
            "collision_impulse_sum": float(self.collision_impulse_sum),
            "offroad_fraction": float(self.offroad_fraction),
            "first_offroad_time_s": self.first_offroad_time_s,
            "lane_invasion_episode_count": int(self.lane_invasion_episode_count),
            "minimum_actor_clearance_m": self.minimum_actor_clearance_m,
            "minimum_ttc_s": self.minimum_ttc_s,
            "route_progress_delta_m": float(self.route_progress_delta_m),
            "distance_m": float(self.distance_m),
            "longitudinal_accel_abs_p95": float(self.longitudinal_accel_abs_p95),
            "longitudinal_accel_abs_max": float(self.longitudinal_accel_abs_max),
            "jerk_abs_p95": float(self.jerk_abs_p95),
            "jerk_abs_max": float(self.jerk_abs_max),
            "lateral_accel_abs_p95": float(self.lateral_accel_abs_p95),
            "lateral_accel_abs_max": float(self.lateral_accel_abs_max),
            "steer_rate_abs_p95": float(self.steer_rate_abs_p95),
            "steer_rate_abs_max": float(self.steer_rate_abs_max),
            "mpc_solved_count": int(self.mpc_solved_count),
            "mpc_timeout_count": int(self.mpc_timeout_count),
            "mpc_fallback_count": int(self.mpc_fallback_count),
            "extra": dict(self.extra),
        }


def _percentile_abs(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(abs(float(v)) for v in values)
    if len(xs) == 1:
        return xs[0]
    # Nearest-rank
    k = max(0, min(len(xs) - 1, int(math.ceil(p / 100.0 * len(xs))) - 1))
    return xs[k]


def _count_episodes(flags: Sequence[bool]) -> int:
    """Count rising-edge episodes in a boolean time series."""
    count = 0
    prev = False
    for f in flags:
        cur = bool(f)
        if cur and not prev:
            count += 1
        prev = cur
    return count


def _first_true_time(flags: Sequence[bool], times: Sequence[float]) -> float | None:
    for f, t in zip(flags, times):
        if f:
            return float(t)
    return None


def aggregate_branch_outcome(
    ticks: Sequence[TickRecord],
    *,
    candidate_id: str,
    candidate_index: int,
    primary_horizon_s: float = PRIMARY_HORIZON_S,
    expected_ticks: int = EXPECTED_PRIMARY_TICKS,
) -> BranchOutcomeMetrics:
    """Aggregate primary-horizon metrics. Ticks beyond horizon are ignored for main oracle."""
    primary = [t for t in ticks if float(t.simulation_time_s) <= primary_horizon_s + 1e-9]
    # If ticks are indexed without absolute horizon times, fall back to first N.
    if not primary and ticks:
        primary = list(ticks[:expected_ticks])

    n = len(primary)
    times = [float(t.simulation_time_s) for t in primary]
    collisions = [bool(t.collision) for t in primary]
    offroads = [bool(t.offroad) for t in primary]
    lanes = [bool(t.lane_invasion) for t in primary]

    impulse_sum = sum(float(t.collision_impulse) for t in primary if t.collision)
    offroad_fraction = (sum(1 for x in offroads if x) / n) if n else 1.0

    clearances = [float(t.actor_clearance_m) for t in primary if t.actor_clearance_m is not None]
    ttcs = [float(t.ttc_s) for t in primary if t.ttc_s is not None and math.isfinite(float(t.ttc_s))]

    progress0 = float(primary[0].route_progress_m) if primary else 0.0
    progress1 = float(primary[-1].route_progress_m) if primary else 0.0
    route_delta = progress1 - progress0

    dist = 0.0
    for i in range(1, n):
        dx = primary[i].ego_x - primary[i - 1].ego_x
        dy = primary[i].ego_y - primary[i - 1].ego_y
        dist += math.hypot(dx, dy)

    long_a = [float(t.longitudinal_accel) for t in primary]
    lat_a = [float(t.lateral_accel) for t in primary]
    jerks = [float(t.jerk) for t in primary]
    steers = [float(t.steer_rate) for t in primary]

    mpc_solved = sum(1 for t in primary if str(t.mpc_status).lower() == "solved")
    mpc_timeout = sum(1 for t in primary if str(t.mpc_status).lower() == "timeout")
    mpc_fallback = sum(1 for t in primary if str(t.mpc_status).lower() == "fallback")

    return BranchOutcomeMetrics(
        candidate_id=candidate_id,
        candidate_index=int(candidate_index),
        completed_primary_horizon=n >= expected_ticks,
        completed_primary_ticks=n,
        collision_episode_count=_count_episodes(collisions),
        first_collision_time_s=_first_true_time(collisions, times),
        collision_impulse_sum=float(impulse_sum),
        offroad_fraction=float(offroad_fraction),
        first_offroad_time_s=_first_true_time(offroads, times),
        lane_invasion_episode_count=_count_episodes(lanes),
        minimum_actor_clearance_m=(min(clearances) if clearances else None),
        minimum_ttc_s=(min(ttcs) if ttcs else None),
        route_progress_delta_m=float(route_delta),
        distance_m=float(dist),
        longitudinal_accel_abs_p95=_percentile_abs(long_a, 95),
        longitudinal_accel_abs_max=_percentile_abs(long_a, 100),
        jerk_abs_p95=_percentile_abs(jerks, 95),
        jerk_abs_max=_percentile_abs(jerks, 100),
        lateral_accel_abs_p95=_percentile_abs(lat_a, 95),
        lateral_accel_abs_max=_percentile_abs(lat_a, 100),
        steer_rate_abs_p95=_percentile_abs(steers, 95),
        steer_rate_abs_max=_percentile_abs(steers, 100),
        mpc_solved_count=mpc_solved,
        mpc_timeout_count=mpc_timeout,
        mpc_fallback_count=mpc_fallback,
    )


def ttc_risk_bucket(ttc_s: float | None) -> int:
    """Lower is worse. null / large TTC → safest bucket 3."""
    if ttc_s is None or not math.isfinite(float(ttc_s)):
        return 3
    t = float(ttc_s)
    if t < 0.5:
        return 0
    if t < 1.0:
        return 1
    if t < 1.5:
        return 2
    return 3


def make_synthetic_ticks(
    *,
    n: int = EXPECTED_PRIMARY_TICKS,
    dt_s: float = 0.05,
    candidate_id: str,
    speed_mps: float = 5.0,
    collision_at: float | None = None,
    offroad_fraction: float = 0.0,
    clearance_m: float | None = 5.0,
    ttc_s: float | None = 2.0,
    jerk: float = 0.5,
    mpc_status: str = "solved",
    mpc_timeout_ticks: int = 0,
) -> list[TickRecord]:
    """Helper for offline unit tests (not used in live runner)."""
    ticks: list[TickRecord] = []
    offroad_n = int(round(offroad_fraction * n))
    for i in range(n):
        t = (i + 1) * dt_s
        coll = collision_at is not None and t + 1e-9 >= collision_at
        status = mpc_status
        if i >= n - mpc_timeout_ticks:
            status = "timeout"
        ticks.append(
            TickRecord(
                tick_index=i,
                simulation_time_s=t,
                ego_x=speed_mps * t,
                ego_y=0.0,
                ego_yaw_rad=0.0,
                ego_v=speed_mps,
                selected_candidate_id=candidate_id,
                executed_candidate_id=candidate_id,
                source_id=f"anchor:{candidate_id}",
                path_age_s=t,
                freshness_regime="soft" if t > 1.0 else "fresh",
                mpc_mode="tracking",
                mpc_status=status,
                mpc_latency_s=0.01,
                collision=coll,
                collision_impulse=100.0 if coll else 0.0,
                offroad=(i < offroad_n),
                route_progress_m=speed_mps * t,
                longitudinal_accel=0.0,
                lateral_accel=0.0,
                jerk=jerk,
                steer_rate=0.0,
                actor_clearance_m=clearance_m,
                ttc_s=ttc_s,
                oracle_only=True,
            )
        )
    return ticks
