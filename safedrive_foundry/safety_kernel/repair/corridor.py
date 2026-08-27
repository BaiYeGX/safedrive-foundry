"""Frenet convex corridor utilities for restricted RATO-SCP (G2-03).

Builds a local s/d frame from Observable corridor centerline and reports
whether a legal lateral clearance exists for secondary 2D repair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import ObservableSnapshot, TrajectoryPoint


@dataclass(frozen=True)
class CorridorFrame:
    """Sampled centerline with arc-length and unit tangents/normals."""

    centerline: tuple[tuple[float, float], ...]
    s: np.ndarray  # (M,)
    x: np.ndarray
    y: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    half_width_m: float
    vehicle_half_width_m: float

    @property
    def legal(self) -> bool:
        return len(self.centerline) >= 2 and self.s[-1] > 1e-3 and self.half_width_m > 0.0

    @property
    def lateral_room_m(self) -> float:
        """Usable |d| for ego-center path points.

        Matches validator road geometry: distance(center, centerline) <= half_width
        (half_width already includes max_offroad). Only a tiny numerical margin is
        reserved so repaired points do not sit exactly on the hard boundary.
        """
        return max(0.0, self.half_width_m - 0.05)


def _polyline_s(xy: Sequence[tuple[float, float]]) -> np.ndarray:
    s = np.zeros(len(xy), dtype=float)
    for i in range(1, len(xy)):
        dx = xy[i][0] - xy[i - 1][0]
        dy = xy[i][1] - xy[i - 1][1]
        s[i] = s[i - 1] + math.hypot(dx, dy)
    return s


def build_corridor_frame(obs: ObservableSnapshot, cfg: SafetyKernelConfig) -> CorridorFrame | None:
    cl = obs.corridor_centerline
    if not cl or len(cl) < 2:
        return None
    half = obs.corridor_half_width_m if obs.corridor_half_width_m > 0 else cfg.lane_half_width_m
    # Effective road bound includes max_offroad slack used by validator.
    half_eff = half + cfg.max_offroad_m
    vehicle_half = 0.5 * cfg.width_m + 0.05
    s = _polyline_s(cl)
    if float(s[-1]) < 1e-3:
        return None
    xs = np.array([p[0] for p in cl], dtype=float)
    ys = np.array([p[1] for p in cl], dtype=float)
    tx = np.zeros(len(cl), dtype=float)
    ty = np.zeros(len(cl), dtype=float)
    for i in range(len(cl)):
        if i + 1 < len(cl):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
        else:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
        nrm = math.hypot(dx, dy)
        if nrm < 1e-9:
            tx[i], ty[i] = 1.0, 0.0
        else:
            tx[i], ty[i] = dx / nrm, dy / nrm
    # Left-normal for d > 0.
    nx, ny = -ty, tx
    return CorridorFrame(
        centerline=tuple(cl),
        s=s,
        x=xs,
        y=ys,
        tx=tx,
        ty=ty,
        nx=nx,
        ny=ny,
        half_width_m=float(half_eff),
        vehicle_half_width_m=float(vehicle_half),
    )


def project_xy(frame: CorridorFrame, x: float, y: float) -> tuple[float, float, float, float, float, float]:
    """Project map (x,y) → (s, d, cx, cy, nx, ny) onto centerline."""
    best_i = 0
    best_dist2 = float("inf")
    best_s = 0.0
    best_d = 0.0
    best_t = 0.0
    for i in range(len(frame.s) - 1):
        x0, y0 = float(frame.x[i]), float(frame.y[i])
        x1, y1 = float(frame.x[i + 1]), float(frame.y[i + 1])
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg_len2))
        px = x0 + t * dx
        py = y0 + t * dy
        dist2 = (x - px) ** 2 + (y - py) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_i = i
            best_t = t
            best_s = float(frame.s[i] + t * (frame.s[i + 1] - frame.s[i]))
            # Signed lateral via segment normal.
            nrm = math.hypot(dx, dy)
            if nrm < 1e-9:
                nx, ny = float(frame.nx[i]), float(frame.ny[i])
            else:
                nx, ny = -dy / nrm, dx / nrm
            best_d = (x - px) * nx + (y - py) * ny
            best_cx, best_cy = px, py
            best_nx, best_ny = nx, ny
    if best_dist2 == float("inf"):
        return 0.0, 0.0, float(frame.x[0]), float(frame.y[0]), float(frame.nx[0]), float(frame.ny[0])
    # Interpolate tangent frame for reconstruction.
    i = best_i
    nx = (1.0 - best_t) * float(frame.nx[i]) + best_t * float(frame.nx[min(i + 1, len(frame.nx) - 1)])
    ny = (1.0 - best_t) * float(frame.ny[i]) + best_t * float(frame.ny[min(i + 1, len(frame.ny) - 1)])
    nrm = math.hypot(nx, ny)
    if nrm < 1e-9:
        nx, ny = best_nx, best_ny
    else:
        nx, ny = nx / nrm, ny / nrm
    return best_s, best_d, best_cx, best_cy, nx, ny


def frenet_of_trajectory(
    points: Sequence[TrajectoryPoint],
    frame: CorridorFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return s, d, cx, cy, nx, ny arrays for each trajectory point."""
    n = len(points)
    s = np.zeros(n)
    d = np.zeros(n)
    cx = np.zeros(n)
    cy = np.zeros(n)
    nx = np.zeros(n)
    ny = np.zeros(n)
    for k, p in enumerate(points):
        s[k], d[k], cx[k], cy[k], nx[k], ny[k] = project_xy(frame, p.x, p.y)
    return s, d, cx, cy, nx, ny


def xy_from_frenet(
    s_arr: np.ndarray,
    d_arr: np.ndarray,
    frame: CorridorFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map (s,d) → (x,y) and return local normals used."""
    n = len(s_arr)
    xs = np.zeros(n)
    ys = np.zeros(n)
    nxs = np.zeros(n)
    nys = np.zeros(n)
    s_cl = frame.s
    for k in range(n):
        s = float(np.clip(s_arr[k], s_cl[0], s_cl[-1]))
        # Locate segment.
        i = int(np.searchsorted(s_cl, s, side="right") - 1)
        i = max(0, min(i, len(s_cl) - 2))
        s0, s1 = float(s_cl[i]), float(s_cl[i + 1])
        t = 0.0 if s1 - s0 < 1e-9 else (s - s0) / (s1 - s0)
        t = max(0.0, min(1.0, t))
        cx = (1.0 - t) * float(frame.x[i]) + t * float(frame.x[i + 1])
        cy = (1.0 - t) * float(frame.y[i]) + t * float(frame.y[i + 1])
        nx = (1.0 - t) * float(frame.nx[i]) + t * float(frame.nx[i + 1])
        ny = (1.0 - t) * float(frame.ny[i]) + t * float(frame.ny[i + 1])
        nrm = math.hypot(nx, ny)
        if nrm < 1e-9:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = nx / nrm, ny / nrm
        xs[k] = cx + float(d_arr[k]) * nx
        ys[k] = cy + float(d_arr[k]) * ny
        nxs[k], nys[k] = nx, ny
    return xs, ys, nxs, nys


def has_legal_lateral_corridor(
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
    *,
    min_clearance_m: float,
) -> bool:
    """True when Observable corridor provides usable lateral room for RATO."""
    frame = build_corridor_frame(obs, cfg)
    if frame is None or not frame.legal:
        return False
    return frame.lateral_room_m + 1e-9 >= min_clearance_m


def is_rato_eligible_hints(reject_messages: Sequence[str]) -> bool:
    """RATO may only attempt lateral-relevant repairable failures."""
    if not reject_messages:
        # Empty hints: allow when caller (cascade) already decided to try.
        return True
    hard_block = (
        "numeric",
        "schema",
        "freshness",
        "time_order",
        "privilege",
        "missing_candidate",
    )
    lateral_tokens = (
        "collision",
        "road",
        "offroad",
        "cut_in",
        "static",
        "lateral",
        "lat_accel",
        "curvature",
        "narrow",
        "lane",
        "trackability",
        "teleport",
        "yaw_rate",
    )
    # Pure red-light / speed without collision/road should stay longitudinal-only
    # unless also marked lateral (cascade may still force try when QP progress low).
    joined = " ".join(reject_messages).lower()
    if any(tok in joined for tok in hard_block):
        return False
    if any(tok in joined for tok in lateral_tokens):
        return True
    # Longitudinal-only tokens: not eligible for *direct* RATO without cascade override.
    long_only = ("red_light", "rules", "speed", "accel", "jerk", "dynamics")
    if any(tok in joined for tok in long_only) and not any(tok in joined for tok in lateral_tokens):
        return False
    return True
