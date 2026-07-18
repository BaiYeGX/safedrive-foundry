"""Spatial path conditioning for low-rate VLA plans and high-rate tracking.

The model owns path geometry.  This module only aligns successive model paths
in space, preserves a short already-committed prefix, rejects discontinuous
updates, and exposes a dense geometric reference to the low-level controller.
No CARLA map or lane centerline is consumed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # pragma: no cover - scipy is present in the project venv
    PchipInterpolator = None  # type: ignore[assignment]


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _cumulative_s(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))


def _drop_near_duplicates(xy: Sequence[tuple[float, float]], eps: float = 0.03) -> np.ndarray:
    kept: list[tuple[float, float]] = []
    for px, py in xy:
        point = (float(px), float(py))
        if not kept or math.hypot(point[0] - kept[-1][0], point[1] - kept[-1][1]) >= eps:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def _interp(s: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(query)
    if values.size == 1:
        return np.full_like(query, float(values[0]))
    if PchipInterpolator is not None and values.size >= 3:
        return np.asarray(PchipInterpolator(s, values, extrapolate=False)(query), dtype=float)
    return np.interp(query, s, values)


def _segment_intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 < -1e-8 and o3 * o4 < -1e-8


@dataclass(frozen=True)
class EgoPose:
    x: float
    y: float
    yaw: float
    speed_mps: float = 0.0


@dataclass(frozen=True)
class SpatialPath:
    """Dense map-frame path, parameterized only by arc length."""

    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    target_speed_mps: float
    stamp_s: float
    source_id: str = "vla"

    @property
    def length_m(self) -> float:
        return float(self.s[-1]) if self.s.size else 0.0

    def sample(self, query_s: float | np.ndarray) -> tuple[np.ndarray, ...]:
        q = np.atleast_1d(np.asarray(query_s, dtype=float))
        if self.s.size == 0:
            z = np.zeros_like(q)
            return z, z, z, z
        q = np.clip(q, float(self.s[0]), float(self.s[-1]))
        x = np.interp(q, self.s, self.x)
        y = np.interp(q, self.s, self.y)
        cy = np.interp(q, self.s, np.cos(self.yaw))
        sy = np.interp(q, self.s, np.sin(self.yaw))
        yaw = np.arctan2(sy, cy)
        kappa = np.interp(q, self.s, self.kappa)
        return x, y, yaw, kappa

    def project_s(self, x: float, y: float, *, hint_s: float | None = None) -> float:
        """Project a point to the path using segments, optionally in a local window."""
        if self.s.size < 2:
            return 0.0
        lo = 0
        hi = self.s.size - 1
        if hint_s is not None:
            lo = max(0, int(np.searchsorted(self.s, hint_s - 3.0)) - 1)
            hi = min(self.s.size - 1, int(np.searchsorted(self.s, hint_s + 12.0)) + 1)
        best_s, best_d2 = float(self.s[lo]), float("inf")
        p = np.array([float(x), float(y)], dtype=float)
        for i in range(lo, hi):
            a = np.array([self.x[i], self.y[i]], dtype=float)
            b = np.array([self.x[i + 1], self.y[i + 1]], dtype=float)
            ab = b - a
            denom = float(ab @ ab)
            u = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
            nearest = a + u * ab
            d2 = float((p - nearest) @ (p - nearest))
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self.s[i] + u * (self.s[i + 1] - self.s[i]))
        return best_s

    def relative_samples(self, ego: EgoPose, distances: np.ndarray) -> tuple[np.ndarray, ...]:
        s0 = self.project_s(ego.x, ego.y)
        return self.sample(s0 + np.asarray(distances, dtype=float))

    def as_xy(self) -> list[tuple[float, float]]:
        return list(zip(self.x.astype(float).tolist(), self.y.astype(float).tolist(), strict=True))


@dataclass(frozen=True)
class PathQuality:
    ok: bool
    reason: str
    forward_ratio: float
    self_intersects: bool
    max_abs_curvature: float
    length_m: float
    switch_lateral_5m: float = 0.0
    switch_heading_5m_deg: float = 0.0


@dataclass(frozen=True)
class PathManagerConfig:
    resample_ds_m: float = 0.20
    min_path_length_m: float = 5.0
    horizon_m: float = 20.0
    min_forward_ratio: float = 0.90
    max_abs_curvature: float = 0.20
    max_switch_lateral_5m: float = 1.00
    max_switch_heading_5m_deg: float = 12.0
    committed_prefix_min_m: float = 1.5
    committed_prefix_time_s: float = 0.60
    tail_blend_m: float = 4.0
    ensemble_weights: tuple[float, ...] = (0.60, 0.30, 0.10)


@dataclass(frozen=True)
class PathUpdate:
    accepted: bool
    reason: str
    raw: SpatialPath | None
    committed: SpatialPath | None
    quality: PathQuality


def spatial_path_from_xy(
    points_xy: Sequence[tuple[float, float]],
    *,
    ego: EgoPose,
    target_speed_mps: float,
    stamp_s: float,
    ds_m: float = 0.20,
    source_id: str = "vla",
) -> SpatialPath | None:
    """Create a dense spatial path while preserving the model's geometry."""
    pts = _drop_near_duplicates(points_xy)
    if pts.shape[0] < 2:
        return None
    if math.hypot(float(pts[0, 0]) - ego.x, float(pts[0, 1]) - ego.y) > 0.03:
        pts = np.vstack((np.array([[ego.x, ego.y]], dtype=float), pts))
    s_nodes = _cumulative_s(pts[:, 0], pts[:, 1])
    if s_nodes[-1] < 0.05:
        return None
    n = max(2, int(math.floor(float(s_nodes[-1]) / max(ds_m, 0.05))) + 1)
    s = np.linspace(0.0, float(s_nodes[-1]), n, dtype=float)
    x = _interp(s_nodes, pts[:, 0], s)
    y = _interp(s_nodes, pts[:, 1], s)
    dx = np.gradient(x, s, edge_order=1)
    dy = np.gradient(y, s, edge_order=1)
    yaw = np.unwrap(np.arctan2(dy, dx))
    kappa = np.gradient(yaw, s, edge_order=1)
    return SpatialPath(
        s=s,
        x=x,
        y=y,
        yaw=np.asarray([wrap_angle(v) for v in yaw], dtype=float),
        kappa=np.asarray(kappa, dtype=float),
        target_speed_mps=max(0.0, float(target_speed_mps)),
        stamp_s=float(stamp_s),
        source_id=str(source_id),
    )


class VLAPathManager:
    """Receding spatial ensemble with a committed near-field prefix."""

    def __init__(self, config: PathManagerConfig | None = None) -> None:
        self.config = config or PathManagerConfig()
        self._raw_history: list[SpatialPath] = []
        self._committed: SpatialPath | None = None

    @property
    def committed(self) -> SpatialPath | None:
        return self._committed

    def reset(self) -> None:
        self._raw_history.clear()
        self._committed = None

    @staticmethod
    def _forward_ratio(path: SpatialPath, ego: EgoPose) -> float:
        c, s = math.cos(ego.yaw), math.sin(ego.yaw)
        local_x = c * (path.x - ego.x) + s * (path.y - ego.y)
        return float(np.mean(local_x >= -0.20)) if local_x.size else 0.0

    @staticmethod
    def _self_intersects(path: SpatialPath) -> bool:
        stride = max(1, int(round(0.5 / max(float(np.median(np.diff(path.s))), 0.1))))
        pts = np.column_stack((path.x[::stride], path.y[::stride]))
        for i in range(max(0, pts.shape[0] - 1)):
            for j in range(i + 2, pts.shape[0] - 1):
                if j == i + 1:
                    continue
                if _segment_intersects(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                    return True
        return False

    def _quality(self, raw: SpatialPath | None, ego: EgoPose) -> PathQuality:
        if raw is None:
            return PathQuality(False, "degenerate", 0.0, False, float("inf"), 0.0)
        cfg = self.config
        forward = self._forward_ratio(raw, ego)
        intersects = self._self_intersects(raw)
        curvature_core = np.abs(raw.kappa[(raw.s >= 0.5) & (raw.s <= raw.length_m - 0.5)])
        if curvature_core.size == 0:
            curvature_core = np.abs(raw.kappa)
        max_kappa = float(np.max(curvature_core)) if curvature_core.size else float("inf")
        lat_jump = 0.0
        head_jump_deg = 0.0
        if self._committed is not None:
            distance = min(5.0, raw.length_m, self._committed.length_m)
            rx, ry, ryaw, _ = raw.relative_samples(ego, np.array([distance]))
            ox, oy, oyaw, _ = self._committed.relative_samples(ego, np.array([distance]))
            normal_x, normal_y = -math.sin(float(oyaw[0])), math.cos(float(oyaw[0]))
            lat_jump = abs((float(rx[0]) - float(ox[0])) * normal_x + (float(ry[0]) - float(oy[0])) * normal_y)
            head_jump_deg = abs(math.degrees(wrap_angle(float(ryaw[0]) - float(oyaw[0]))))

        checks = (
            (raw.length_m >= cfg.min_path_length_m, "too_short"),
            (forward >= cfg.min_forward_ratio, "not_forward"),
            (not intersects, "self_intersection"),
            (max_kappa <= cfg.max_abs_curvature, "curvature_limit"),
            (lat_jump <= cfg.max_switch_lateral_5m, "lateral_switch"),
            (head_jump_deg <= cfg.max_switch_heading_5m_deg, "heading_switch"),
        )
        reason = "ok"
        ok = True
        for passed, name in checks:
            if not passed:
                ok, reason = False, name
                break
        return PathQuality(ok, reason, forward, intersects, max_kappa, raw.length_m, lat_jump, head_jump_deg)

    def _ensemble(self, ego: EgoPose, stamp_s: float) -> SpatialPath:
        cfg = self.config
        max_len = min(cfg.horizon_m, min(p.length_m for p in self._raw_history))
        distances = np.arange(0.0, max_len + 0.5 * cfg.resample_ds_m, cfg.resample_ds_m)
        weights = np.asarray(cfg.ensemble_weights[: len(self._raw_history)], dtype=float)
        weights /= max(float(weights.sum()), 1e-9)
        xs, ys = [], []
        for path in self._raw_history:
            x, y, _yaw, _kappa = path.relative_samples(ego, distances)
            xs.append(x)
            ys.append(y)
        x_new = np.average(np.vstack(xs), axis=0, weights=weights)
        y_new = np.average(np.vstack(ys), axis=0, weights=weights)
        # Geometry benefits from temporal ensembling.  Speed does not: the
        # speed planner already slew-limits acceleration, and averaging here
        # would delay a newly requested VLA brake.
        target_speed = self._raw_history[0].target_speed_mps

        if self._committed is not None:
            old_x, old_y, _old_yaw, _old_k = self._committed.relative_samples(ego, distances)
            commit_m = max(cfg.committed_prefix_min_m, max(0.0, ego.speed_mps) * cfg.committed_prefix_time_s)
            alpha = np.clip((distances - commit_m) / max(cfg.tail_blend_m, 1e-3), 0.0, 1.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            x_new = (1.0 - alpha) * old_x + alpha * x_new
            y_new = (1.0 - alpha) * old_y + alpha * y_new

        dense = spatial_path_from_xy(
            list(zip(x_new.tolist(), y_new.tolist(), strict=True)),
            ego=ego,
            target_speed_mps=target_speed,
            stamp_s=stamp_s,
            ds_m=cfg.resample_ds_m,
            source_id="vla_committed",
        )
        assert dense is not None
        return dense

    def update(
        self,
        points_xy: Sequence[tuple[float, float]],
        *,
        ego: EgoPose,
        target_speed_mps: float,
        stamp_s: float,
        source_id: str = "vla",
    ) -> PathUpdate:
        raw = spatial_path_from_xy(
            points_xy,
            ego=ego,
            target_speed_mps=target_speed_mps,
            stamp_s=stamp_s,
            ds_m=self.config.resample_ds_m,
            source_id=source_id,
        )
        quality = self._quality(raw, ego)
        if not quality.ok:
            # A geometrically invalid update must not steer the car, but a VLA
            # brake request still has authority.  Preserve the old geometry
            # and stamp (so freshness continues to age) while lowering speed.
            if self._committed is not None:
                lowered_speed = min(self._committed.target_speed_mps, max(0.0, float(target_speed_mps)))
                if lowered_speed < self._committed.target_speed_mps:
                    self._committed = replace(self._committed, target_speed_mps=lowered_speed)
            return PathUpdate(False, quality.reason, raw, self._committed, quality)

        assert raw is not None
        self._raw_history.insert(0, raw)
        del self._raw_history[len(self.config.ensemble_weights) :]
        self._committed = self._ensemble(ego, stamp_s)
        return PathUpdate(True, "accepted", raw, self._committed, quality)
