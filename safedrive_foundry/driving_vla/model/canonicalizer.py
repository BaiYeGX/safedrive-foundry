"""Strict deterministic canonicalization for the H-route trajectory contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from driving_vla.adapter.policy_adapter import TrajectoryArray
from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS

CANONICALIZER_VERSION = "safedrive.trajectory_canonicalizer.v2"


class CanonicalizationError(ValueError):
    """Raised when canonicalization would require inventing trajectory semantics."""


@dataclass(frozen=True)
class UpstreamPathSpeed:
    """Native path plus speed output, in ego or map coordinates."""

    path_xy: tuple[tuple[float, float], ...]
    speed_mps: tuple[float, ...]
    frame: str = "ego"


@dataclass(frozen=True)
class UpstreamTimedTrajectory:
    """Native timed points: ``t,x,y,yaw,v,a,kappa``."""

    points: tuple[tuple[float, float, float, float, float, float, float], ...]
    frame: str = "map"


@dataclass(frozen=True)
class CanonicalizationReport:
    version: str
    input_sha256: str
    canonical_sha256: str
    source_points: int
    canonical_points: int
    max_resample_error_m: float
    coverage_shortfall_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalizationResult:
    trajectory: TrajectoryArray
    report: CanonicalizationReport


def stable_sha256(payload: Any) -> str:
    """Hash numeric trajectory payloads using a stable JSON encoding."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cum_arclength(path: Sequence[tuple[float, float]]) -> list[float]:
    s = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        s.append(s[-1] + math.hypot(dx, dy))
    return s


_cum_arclength = cum_arclength


def interp_xy(
    path: Sequence[tuple[float, float]], s_list: Sequence[float], s_query: float
) -> tuple[float, float, float]:
    """Interpolate ``x,y,yaw`` at arc length without extrapolation."""

    if len(path) < 2 or not s_list or s_list[-1] <= 1e-9:
        raise CanonicalizationError("degenerate_path")
    if s_query < -1e-9 or s_query > s_list[-1] + 1e-9:
        raise CanonicalizationError(
            f"path_query_out_of_range:{s_query:.6f}>{s_list[-1]:.6f}"
        )
    sq = max(0.0, min(float(s_query), float(s_list[-1])))
    for i in range(1, len(s_list)):
        if s_list[i] >= sq - 1e-12:
            s0, s1 = s_list[i - 1], s_list[i]
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            seg = max(s1 - s0, 1e-12)
            r = (sq - s0) / seg
            return (
                x0 + r * (x1 - x0),
                y0 + r * (y1 - y0),
                math.atan2(y1 - y0, x1 - x0),
            )
    x0, y0 = path[-2]
    x1, y1 = path[-1]
    return x1, y1, math.atan2(y1 - y0, x1 - x0)


_interp_xy = interp_xy


def _speed_at(speeds: Sequence[float], t: float, dt: float = DT_S) -> float:
    """Interpolate samples whose first value belongs to ``t=dt``."""

    if not speeds:
        raise CanonicalizationError("missing_speed_profile")
    idx = max(0.0, t / dt - 1.0)
    i0 = max(0, min(int(math.floor(idx)), len(speeds) - 1))
    i1 = min(i0 + 1, len(speeds) - 1)
    if i0 == i1:
        return float(speeds[i0])
    r = idx - i0
    return float(speeds[i0]) * (1.0 - r) + float(speeds[i1]) * r


def _finite_rows(rows: Sequence[Sequence[float]], *, expected: int, name: str) -> None:
    for i, row in enumerate(rows):
        if len(row) != expected:
            raise CanonicalizationError(f"{name}_row_width:{i}:{len(row)}!={expected}")
        if not all(math.isfinite(float(value)) for value in row):
            raise CanonicalizationError(f"{name}_non_finite:{i}")


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _interp_angle(a0: float, a1: float, ratio: float) -> float:
    return _wrap_angle(a0 + ratio * _wrap_angle(a1 - a0))


class TrajectoryCanonicalizer:
    """Strict conversion to the fixed H1 ``T10/dt=.25/horizon=2.5`` contract."""

    version = CANONICALIZER_VERSION

    def __init__(
        self,
        *,
        t_steps: int = T_STEPS,
        dt_s: float = DT_S,
        horizon_s: float = HORIZON_S,
    ) -> None:
        self.t_steps = int(t_steps)
        self.dt_s = float(dt_s)
        self.horizon_s = float(horizon_s)
        if self.t_steps <= 0 or self.dt_s <= 0.0:
            raise ValueError("t_steps and dt_s must be positive")
        if abs(self.t_steps * self.dt_s - self.horizon_s) > 1e-9:
            raise ValueError("canonical horizon must equal T * dt")

    @property
    def sample_times(self) -> tuple[float, ...]:
        return tuple((i + 1) * self.dt_s for i in range(self.t_steps))

    def canonicalize(
        self,
        upstream: UpstreamPathSpeed,
        *,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        origin_yaw: float = 0.0,
        to_map: bool = True,
    ) -> TrajectoryArray:
        return self.canonicalize_with_report(
            upstream,
            origin_xy=origin_xy,
            origin_yaw=origin_yaw,
            to_map=to_map,
        ).trajectory

    def canonicalize_with_report(
        self,
        upstream: UpstreamPathSpeed,
        *,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        origin_yaw: float = 0.0,
        to_map: bool = True,
    ) -> CanonicalizationResult:
        path = tuple((float(x), float(y)) for x, y in upstream.path_xy)
        speeds = tuple(float(value) for value in upstream.speed_mps)
        _finite_rows(path, expected=2, name="path")
        if len(path) < 2:
            raise CanonicalizationError("degenerate_path")
        if upstream.frame not in {"ego", "map"}:
            raise CanonicalizationError(f"unsupported_coordinate_frame:{upstream.frame}")
        if not speeds or not all(math.isfinite(v) and v >= 0.0 for v in speeds):
            raise CanonicalizationError("invalid_speed_profile")
        s_list = cum_arclength(path)
        if s_list[-1] <= 1e-6:
            raise CanonicalizationError("degenerate_path")

        source_payload = {"kind": "path_speed", "path": path, "speed": speeds, "frame": upstream.frame}
        points: list[tuple[float, float, float, float, float, float]] = []
        s_pos = 0.0
        previous_v = _speed_at(speeds, self.dt_s, self.dt_s)
        previous_yaw: float | None = None
        previous_xy: tuple[float, float] | None = None
        coverage_shortfall = 0.0
        max_projection_error = 0.0

        for t in self.sample_times:
            v = _speed_at(speeds, t, self.dt_s)
            s_pos += 0.5 * (previous_v + v) * self.dt_s
            if s_pos > s_list[-1] + 1e-6:
                coverage_shortfall = max(coverage_shortfall, s_pos - s_list[-1])
                raise CanonicalizationError(
                    f"insufficient_path_coverage:shortfall_m={coverage_shortfall:.6f}"
                )
            x_path, y_path, yaw_path = interp_xy(path, s_list, s_pos)
            if upstream.frame == "ego" and to_map:
                x, y, yaw = self._ego_to_map(
                    x_path, y_path, yaw_path, origin_xy, origin_yaw
                )
            else:
                x, y, yaw = x_path, y_path, yaw_path
            a = 0.0 if not points else (v - previous_v) / self.dt_s
            if previous_yaw is None or previous_xy is None:
                kappa = 0.0
            else:
                ds = math.hypot(x - previous_xy[0], y - previous_xy[1])
                kappa = _wrap_angle(yaw - previous_yaw) / max(ds, 1e-3)
            points.append((x, y, _wrap_angle(yaw), v, a, kappa))
            previous_v, previous_yaw, previous_xy = v, yaw, (x, y)

        trajectory = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(points),
            probability=1.0,
            uncertainty=0.1,
            candidate_id="tau0",
            intended_action="nominal",
        )
        report = CanonicalizationReport(
            version=self.version,
            input_sha256=stable_sha256(source_payload),
            canonical_sha256=stable_sha256(trajectory.points_xy_yaw_v_a_kappa),
            source_points=len(path),
            canonical_points=len(points),
            max_resample_error_m=max_projection_error,
            coverage_shortfall_m=coverage_shortfall,
        )
        return CanonicalizationResult(trajectory=trajectory, report=report)

    def canonicalize_timed(
        self,
        upstream: UpstreamTimedTrajectory,
        *,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        origin_yaw: float = 0.0,
        to_map: bool = True,
    ) -> CanonicalizationResult:
        rows = tuple(tuple(float(value) for value in row) for row in upstream.points)
        _finite_rows(rows, expected=7, name="timed")
        if len(rows) < 2:
            raise CanonicalizationError("timed_trajectory_too_short")
        if upstream.frame not in {"ego", "map"}:
            raise CanonicalizationError(f"unsupported_coordinate_frame:{upstream.frame}")
        for i in range(1, len(rows)):
            if rows[i][0] <= rows[i - 1][0] + 1e-9:
                raise CanonicalizationError(f"timed_non_monotonic:{i}")
        if rows[0][0] > self.sample_times[0] + 1e-9:
            raise CanonicalizationError("timed_start_after_first_sample")
        if rows[-1][0] < self.horizon_s - 1e-9:
            raise CanonicalizationError(
                f"timed_horizon_too_short:{rows[-1][0]:.6f}<{self.horizon_s:.6f}"
            )

        out: list[tuple[float, float, float, float, float, float]] = []
        cursor = 1
        for t in self.sample_times:
            while cursor < len(rows) and rows[cursor][0] < t - 1e-12:
                cursor += 1
            if cursor >= len(rows):
                raise CanonicalizationError("timed_horizon_too_short")
            p0, p1 = rows[cursor - 1], rows[cursor]
            denom = max(p1[0] - p0[0], 1e-12)
            ratio = max(0.0, min(1.0, (t - p0[0]) / denom))
            x = p0[1] + ratio * (p1[1] - p0[1])
            y = p0[2] + ratio * (p1[2] - p0[2])
            yaw = _interp_angle(p0[3], p1[3], ratio)
            v = p0[4] + ratio * (p1[4] - p0[4])
            a = p0[5] + ratio * (p1[5] - p0[5])
            kappa = p0[6] + ratio * (p1[6] - p0[6])
            if upstream.frame == "ego" and to_map:
                x, y, yaw = self._ego_to_map(x, y, yaw, origin_xy, origin_yaw)
            out.append((x, y, _wrap_angle(yaw), v, a, kappa))

        trajectory = TrajectoryArray(
            points_xy_yaw_v_a_kappa=tuple(out),
            probability=1.0,
            uncertainty=0.0,
            candidate_id="tau0",
            behavior="follow",
            intended_action="expert",
        )
        # Reconstruct raw in-window positions from the T10 samples to expose the
        # geometric loss caused by reducing a dense timed trajectory to ten points.
        timed_xy = [(t, x, y) for t, x, y, *_ in rows if self.dt_s <= t <= self.horizon_s]
        max_error = 0.0
        canonical_txy = list(zip(self.sample_times, (p[0] for p in out), (p[1] for p in out)))
        for t, x, y in timed_xy:
            if t <= canonical_txy[0][0]:
                rx, ry = canonical_txy[0][1], canonical_txy[0][2]
            else:
                rx, ry = canonical_txy[-1][1], canonical_txy[-1][2]
                for i in range(1, len(canonical_txy)):
                    if canonical_txy[i][0] >= t:
                        q0, q1 = canonical_txy[i - 1], canonical_txy[i]
                        r = (t - q0[0]) / max(q1[0] - q0[0], 1e-12)
                        rx = q0[1] + r * (q1[1] - q0[1])
                        ry = q0[2] + r * (q1[2] - q0[2])
                        break
            max_error = max(max_error, math.hypot(x - rx, y - ry))

        report = CanonicalizationReport(
            version=self.version,
            input_sha256=stable_sha256({"kind": "timed", "points": rows, "frame": upstream.frame}),
            canonical_sha256=stable_sha256(trajectory.points_xy_yaw_v_a_kappa),
            source_points=len(rows),
            canonical_points=len(out),
            max_resample_error_m=max_error,
            coverage_shortfall_m=0.0,
        )
        return CanonicalizationResult(trajectory=trajectory, report=report)

    @staticmethod
    def _ego_to_map(
        xe: float,
        ye: float,
        yaw_e: float,
        origin_xy: tuple[float, float],
        origin_yaw: float,
    ) -> tuple[float, float, float]:
        c, s = math.cos(origin_yaw), math.sin(origin_yaw)
        return (
            origin_xy[0] + c * xe - s * ye,
            origin_xy[1] + s * xe + c * ye,
            origin_yaw + yaw_e,
        )


__all__ = [
    "CANONICALIZER_VERSION",
    "CanonicalizationError",
    "CanonicalizationReport",
    "CanonicalizationResult",
    "TrajectoryCanonicalizer",
    "UpstreamPathSpeed",
    "UpstreamTimedTrajectory",
    "cum_arclength",
    "interp_xy",
    "stable_sha256",
]
