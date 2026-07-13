"""Reference path and Frenet frame utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .vehicle import VehicleParams, wrap_angle


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class ReferencePath:
    """Piecewise-linear centerline parameterized by arc length s."""

    points: tuple[Pose2D, ...]
    s: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2 or len(self.points) != len(self.s):
            raise ValueError("ReferencePath needs >=2 points with matching s")
        if self.s[0] != 0.0:
            raise ValueError("ReferencePath.s must start at 0")

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @classmethod
    def from_xy(cls, xs: Sequence[float], ys: Sequence[float], yaws: Sequence[float] | None = None) -> "ReferencePath":
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("xs/ys must have equal length >= 2")
        points: list[Pose2D] = []
        s_vals: list[float] = [0.0]
        for i, (x, y) in enumerate(zip(xs, ys)):
            if yaws is not None:
                yaw = float(yaws[i])
            elif i + 1 < len(xs):
                yaw = math.atan2(ys[i + 1] - y, xs[i + 1] - x)
            else:
                yaw = math.atan2(y - ys[i - 1], x - xs[i - 1])
            points.append(Pose2D(float(x), float(y), wrap_angle(yaw)))
            if i > 0:
                ds = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
                s_vals.append(s_vals[-1] + ds)
        return cls(points=tuple(points), s=tuple(s_vals))

    def sample(self, s_query: float) -> Pose2D:
        s_query = max(0.0, min(self.length, s_query))
        for i in range(len(self.s) - 1):
            if self.s[i] <= s_query <= self.s[i + 1] or i == len(self.s) - 2:
                s0, s1 = self.s[i], self.s[i + 1]
                p0, p1 = self.points[i], self.points[i + 1]
                ratio = 0.0 if s1 <= s0 else (s_query - s0) / (s1 - s0)
                x = p0.x + ratio * (p1.x - p0.x)
                y = p0.y + ratio * (p1.y - p0.y)
                yaw = wrap_angle(p0.yaw + ratio * wrap_angle(p1.yaw - p0.yaw))
                return Pose2D(x, y, yaw)
        return self.points[-1]

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Return (s, d) for Cartesian point via nearest segment."""

        best_s = 0.0
        best_d = 0.0
        best_dist = float("inf")
        for i in range(len(self.points) - 1):
            p0, p1 = self.points[i], self.points[i + 1]
            vx, vy = p1.x - p0.x, p1.y - p0.y
            seg_len2 = vx * vx + vy * vy
            if seg_len2 < 1e-12:
                continue
            t = ((x - p0.x) * vx + (y - p0.y) * vy) / seg_len2
            t = max(0.0, min(1.0, t))
            proj_x = p0.x + t * vx
            proj_y = p0.y + t * vy
            dist = math.hypot(x - proj_x, y - proj_y)
            if dist < best_dist:
                best_dist = dist
                best_s = self.s[i] + t * (self.s[i + 1] - self.s[i])
                # signed lateral: left positive
                cross = vx * (y - p0.y) - vy * (x - p0.x)
                best_d = math.copysign(dist, cross)
        return best_s, best_d


class FrenetFrame:
    def __init__(self, reference: ReferencePath, vehicle: VehicleParams | None = None) -> None:
        self.reference = reference
        self.vehicle = vehicle or VehicleParams()

    def frenet_to_cartesian(self, s: float, d: float, yaw_frenet: float = 0.0) -> Pose2D:
        ref = self.reference.sample(s)
        # normal pointing left of path
        nx = -math.sin(ref.yaw)
        ny = math.cos(ref.yaw)
        return Pose2D(ref.x + d * nx, ref.y + d * ny, wrap_angle(ref.yaw + yaw_frenet))

    def curvature_proxy(self, s: float, d: float, ds: float = 1.0) -> float:
        """Finite-difference curvature of the offset path."""

        p0 = self.frenet_to_cartesian(max(0.0, s - ds), d)
        p1 = self.frenet_to_cartesian(s, d)
        p2 = self.frenet_to_cartesian(min(self.reference.length, s + ds), d)
        a = math.hypot(p1.x - p0.x, p1.y - p0.y)
        b = math.hypot(p2.x - p1.x, p2.y - p1.y)
        c = math.hypot(p2.x - p0.x, p2.y - p0.y)
        if a * b * c < 1e-9:
            return 0.0
        # triangle area method
        area2 = abs((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x))
        return 2.0 * area2 / max(a * b * c, 1e-9)
