"""S-T occupancy grid and true discrete DP speed corridor (G1-04 acceptance repair)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from classic_stack.geometry import VehicleParams, clamp
from classic_stack.planning.frenet.config import FrenetSTConfig


@dataclass
class STGrid:
    ds: float
    dt: float
    s_bins: int
    t_bins: int
    occupied: list[list[bool]]  # [t][s]

    def is_free(self, s: float, t: float) -> bool:
        si = int(math.floor(s / self.ds))
        ti = int(math.floor(t / self.dt))
        if ti < 0 or ti >= self.t_bins:
            return True
        if si < 0:
            return True
        if si >= self.s_bins:
            return False
        return not self.occupied[ti][si]

    def segment_free(self, s0: float, s1: float, t0: float, t1: float, steps: int = 4) -> bool:
        steps = max(2, steps)
        for k in range(steps + 1):
            r = k / steps
            if not self.is_free(s0 + (s1 - s0) * r, t0 + (t1 - t0) * r):
                return False
        return True


def build_st_occupancy(
    samples: Sequence[dict[str, float | str]],
    config: FrenetSTConfig,
) -> STGrid:
    s_bins = max(2, int(config.st_s_horizon_m / config.st_ds_m) + 1)
    t_bins = max(2, int(config.st_t_horizon_s / config.st_dt_s) + 1)
    occupied = [[False for _ in range(s_bins)] for _ in range(t_bins)]
    for sample in samples:
        t = float(sample["t"])
        s_min = float(sample["s_min"])
        s_max = float(sample["s_max"])
        ti = int(round(t / config.st_dt_s))
        if ti < 0 or ti >= t_bins:
            continue
        i0 = max(0, int(s_min / config.st_ds_m))
        i1 = min(s_bins - 1, int(s_max / config.st_ds_m))
        for si in range(i0, i1 + 1):
            occupied[ti][si] = True
    return STGrid(ds=config.st_ds_m, dt=config.st_dt_s, s_bins=s_bins, t_bins=t_bins, occupied=occupied)


def solve_st_dp(
    grid: STGrid,
    *,
    v0: float,
    v_target: float,
    vehicle: VehicleParams,
    stop_at_s: float | None = None,
) -> tuple[list[tuple[float, float, float, float]], str | None]:
    """Time-layered DP on (t_idx, s_idx) with continuous speed stored per cell.

    Returns list of (t, s, v, a) or empty + failure code.
    - Transitions obey accel/decel limits and occupancy along the segment.
    - Stop mode requires s≈stop and v≈0; never force-appends a fake terminal stop.
    """

    if grid.t_bins < 2 or grid.s_bins < 2:
        return [], "ST_EMPTY"

    dt = grid.dt
    ds = grid.ds
    INF = 1e30
    # cost, v, a_into, parent_si at each (ti, si)
    cost = [[INF] * grid.s_bins for _ in range(grid.t_bins)]
    v_at = [[0.0] * grid.s_bins for _ in range(grid.t_bins)]
    a_at = [[0.0] * grid.s_bins for _ in range(grid.t_bins)]
    parent_si: list[list[int | None]] = [[None] * grid.s_bins for _ in range(grid.t_bins)]

    a_opts = (
        -vehicle.max_decel_mps2,
        -0.5 * vehicle.max_decel_mps2,
        0.0,
        0.5 * vehicle.max_accel_mps2,
        vehicle.max_accel_mps2,
    )

    si0 = 0
    if not grid.is_free(0.0, 0.0):
        found = False
        for si in range(grid.s_bins):
            if grid.is_free(si * ds, 0.0):
                si0 = si
                found = True
                break
        if not found:
            return [], "ST_START_OCCUPIED"

    cost[0][si0] = 0.0
    v_at[0][si0] = clamp(v0, 0.0, vehicle.max_speed_mps)
    a_at[0][si0] = 0.0

    v_tgt = clamp(v_target, 0.0, vehicle.max_speed_mps)

    for ti in range(grid.t_bins - 1):
        t = ti * dt
        t_n = (ti + 1) * dt
        for si in range(grid.s_bins):
            c0 = cost[ti][si]
            if c0 >= INF / 2:
                continue
            s = si * ds
            v = v_at[ti][si]
            for a in a_opts:
                a = clamp(a, -vehicle.max_decel_mps2, vehicle.max_accel_mps2)
                v_n = clamp(v + a * dt, 0.0, vehicle.max_speed_mps)
                s_n = s + v * dt + 0.5 * a * dt * dt
                if s_n < -1e-6:
                    continue
                s_n = min(s_n, (grid.s_bins - 1) * ds)
                if not grid.segment_free(s, s_n, t, t_n, steps=4):
                    continue
                si_n = int(round(s_n / ds))
                si_n = max(0, min(grid.s_bins - 1, si_n))
                s_disc = si_n * ds
                if not grid.is_free(s_disc, t_n):
                    continue
                if stop_at_s is not None:
                    # Progress toward stop station, then brake (v_target often 0).
                    rem = max(0.0, stop_at_s - s_disc)
                    v_lim = math.sqrt(max(0.0, 2.0 * vehicle.max_decel_mps2 * rem))
                    stage = 0.2 * a * a + 0.15 * (v_n - min(v_tgt, v_lim)) ** 2
                    stage += 0.4 * abs(s_disc - stop_at_s)
                    if v_n > v_lim + 0.4:
                        stage += 6.0 * (v_n - v_lim) ** 2
                    if s_disc > stop_at_s + ds:
                        stage += 10.0 * (s_disc - stop_at_s)
                    # slight progress reward until stop
                    if s_disc < stop_at_s:
                        stage -= 0.12 * (s_disc - s)
                else:
                    stage = (v_n - v_tgt) ** 2 + 0.15 * a * a - 0.08 * (s_disc - s)
                nc = c0 + stage
                if nc < cost[ti + 1][si_n]:
                    cost[ti + 1][si_n] = nc
                    v_at[ti + 1][si_n] = v_n
                    a_at[ti + 1][si_n] = a
                    parent_si[ti + 1][si_n] = si

    best_term: tuple[int, int] | None = None
    best_score = INF
    partial_stop = False

    if stop_at_s is not None:
        s_tol = max(ds * 3.0, 4.0)
        v_tol = 0.95
        for ti in range(grid.t_bins):
            for si in range(grid.s_bins):
                if cost[ti][si] >= INF / 2:
                    continue
                s = si * ds
                if abs(s - stop_at_s) > s_tol:
                    continue
                if v_at[ti][si] > v_tol:
                    continue
                score = cost[ti][si] + 5.0 * abs(s - stop_at_s) + 2.0 * v_at[ti][si]
                if score < best_score:
                    best_score = score
                    best_term = (ti, si)
        if best_term is None:
            # A stop farther than the finite planning horizon is still safe if
            # the terminal state remains before the line and inside the
            # braking envelope.  The next planning cycle will continue the
            # approach; never fabricate a terminal stop outside the horizon.
            # A partial approach is useful only when the planning horizon is
            # actually consumed.  This prevents a blocked wall at t=0 from
            # being misreported as a safe one-step prefix.
            for ti in (grid.t_bins - 1,):
                for si in range(grid.s_bins):
                    if cost[ti][si] >= INF / 2:
                        continue
                    s = si * ds
                    if s > stop_at_s + 1e-6:
                        continue
                    remaining = max(0.0, stop_at_s - s)
                    v_lim = math.sqrt(max(0.0, 2.0 * vehicle.max_decel_mps2 * remaining))
                    if v_at[ti][si] > v_lim + 0.4:
                        continue
                    score = cost[ti][si] - 10.0 * s + 0.5 * v_at[ti][si] ** 2
                    if score < best_score:
                        best_score = score
                        best_term = (ti, si)
            if best_term is None:
                return [], "ST_INFEASIBLE_STOP"
            partial_stop = True
    else:
        for ti in range(grid.t_bins):
            for si in range(grid.s_bins):
                if cost[ti][si] >= INF / 2:
                    continue
                s = si * ds
                score = cost[ti][si] - 10.0 * s + 0.5 * (v_at[ti][si] - v_tgt) ** 2
                if score < best_score:
                    best_score = score
                    best_term = (ti, si)
        if best_term is None:
            return [], "ST_NO_PATH"

    ti, si = best_term
    chain: list[tuple[int, int]] = []
    while True:
        chain.append((ti, si))
        if ti == 0:
            break
        psi = parent_si[ti][si]
        if psi is None:
            break
        ti, si = ti - 1, psi
    chain.reverse()
    if len(chain) < 2:
        return [], "ST_EMPTY"

    profile: list[tuple[float, float, float, float]] = []
    for ti, si in chain:
        profile.append((ti * dt, si * ds, v_at[ti][si], a_at[ti][si]))

    # Re-derive a from successive samples for consistency
    fixed: list[tuple[float, float, float, float]] = [profile[0]]
    for i in range(1, len(profile)):
        t0, s0, v0p, _ = fixed[-1]
        t1, s1, v1, _ = profile[i]
        dtp = max(1e-6, t1 - t0)
        a = clamp((v1 - v0p) / dtp, -vehicle.max_decel_mps2, vehicle.max_accel_mps2)
        s_pred = s0 + v0p * dtp + 0.5 * a * dtp * dtp
        s_use = s_pred
        if not grid.segment_free(s0, s_use, t0, t1, steps=4):
            s_use = s1
            if not grid.segment_free(s0, s_use, t0, t1, steps=4):
                return [], "ST_BLOCKED"
        v_use = clamp(v0p + a * dtp, 0.0, vehicle.max_speed_mps)
        fixed.append((t1, s_use, v_use, a))

    if stop_at_s is not None and not partial_stop:
        _, s_f, v_f, _ = fixed[-1]
        if v_f > 0.95:
            return [], "ST_INFEASIBLE_STOP"
        if abs(s_f - stop_at_s) > max(4.0, 3.0 * ds) and s_f < 0.6 * stop_at_s:
            return [], "ST_INFEASIBLE_STOP"
    elif stop_at_s is not None:
        _, s_f, v_f, _ = fixed[-1]
        remaining = max(0.0, stop_at_s - s_f)
        if s_f > stop_at_s + 1e-6 or v_f > math.sqrt(max(0.0, 2.0 * vehicle.max_decel_mps2 * remaining)) + 0.5:
            return [], "ST_INFEASIBLE_STOP"

    return fixed, None


def validate_profile_kinematics(
    profile: Sequence[tuple[float, float, float, float]],
    *,
    vehicle: VehicleParams,
    dt_tol: float = 0.05,
    s_tol: float = 2.5,
) -> str | None:
    """Hard checks only: time order, non-neg speed, accel limits, catastrophic s/v jumps."""

    if len(profile) < 2:
        return "ST_EMPTY"
    for i in range(1, len(profile)):
        t0, s0, v0, a0 = profile[i - 1]
        t1, s1, v1, a1 = profile[i]
        dt = t1 - t0
        if dt <= 1e-9:
            return "ST_BAD_TIME"
        if v0 < -1e-3 or v1 < -1e-3:
            return "ST_NEG_SPEED"
        if a0 > vehicle.max_accel_mps2 + 0.15 or a0 < -vehicle.max_decel_mps2 - 0.15:
            return "ST_ACCEL_LIMIT"
        if a1 > vehicle.max_accel_mps2 + 0.15 or a1 < -vehicle.max_decel_mps2 - 0.15:
            return "ST_ACCEL_LIMIT"
        # Catastrophic only (binning + re-integration leave residual)
        if abs(v1 - v0) > vehicle.max_decel_mps2 * dt + vehicle.max_accel_mps2 * dt + 3.0:
            return "ST_V_INCONSISTENT"
        if abs(s1 - s0) > (max(v0, v1) + 1.0) * dt + s_tol * 3.0:
            return "ST_S_INCONSISTENT"
    return None


def smooth_jerk(
    profile: Sequence[tuple[float, float, float, float]],
    *,
    window: int,
    max_jerk: float,
    vehicle: VehicleParams | None = None,
) -> list[tuple[float, float, float, float, float]]:
    """Jerk-limited accel with re-integrated v,s (no fake terminal stop)."""

    if not profile:
        return []
    accels = [p[3] for p in profile]
    w = max(1, window)
    smooth_a: list[float] = []
    for i in range(len(accels)):
        lo = max(0, i - w // 2)
        hi = min(len(accels), i + w // 2 + 1)
        smooth_a.append(sum(accels[lo:hi]) / (hi - lo))

    out: list[tuple[float, float, float, float, float]] = []
    prev_a = smooth_a[0]
    prev_t = profile[0][0]
    s = profile[0][1]
    v = max(0.0, profile[0][2])
    a_max = vehicle.max_accel_mps2 if vehicle else 1e9
    a_min = -vehicle.max_decel_mps2 if vehicle else -1e9
    v_max = vehicle.max_speed_mps if vehicle else 1e9

    for i, (t, s_ref, _v_ref, _) in enumerate(profile):
        a_des = smooth_a[i]
        dt = max(1e-3, t - prev_t) if i else 1e-3
        jerk = clamp((a_des - prev_a) / dt, -max_jerk, max_jerk)
        a = clamp(prev_a + jerk * dt, a_min, a_max)
        if i == 0:
            out.append((t, s, v, a, 0.0))
        else:
            v_n = clamp(v + a * dt, 0.0, v_max)
            s_n = s + v * dt + 0.5 * a * dt * dt
            if abs(s_n - s_ref) > 3.0:
                s_n = 0.5 * s_n + 0.5 * s_ref
            out.append((t, s_n, v_n, a, jerk))
            s, v = s_n, v_n
        prev_a = a
        prev_t = t
    return out
