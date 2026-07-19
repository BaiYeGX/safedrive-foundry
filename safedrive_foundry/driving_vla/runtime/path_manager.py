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


def curvature_profile_from_xy(
    points_xy: Sequence[tuple[float, float]],
    *,
    ds_m: float | None = None,
    quantile: float = 0.90,
) -> dict[str, float | int | bool]:
    """Native (vertex) or dense (PCHIP) absolute-curvature stats for evidence.

    When ``ds_m`` is None the raw polyline is used without densification.
    When ``ds_m`` is set, the same densify path as ``spatial_path_from_xy`` is used
    so live rejects can be attributed to native geometry vs interpolation spikes.
    """
    pts = _drop_near_duplicates(points_xy)
    empty = {
        "n_points": int(pts.shape[0]),
        "length_m": 0.0,
        "max_abs_curvature": float("inf"),
        "curvature_quantile": float("inf"),
        "dense": bool(ds_m is not None),
        "ok": False,
    }
    if pts.shape[0] < 2:
        return empty
    s_nodes = _cumulative_s(pts[:, 0], pts[:, 1])
    length = float(s_nodes[-1])
    if length < 0.05:
        empty["length_m"] = length
        return empty
    if ds_m is None:
        s = s_nodes
        x, y = pts[:, 0], pts[:, 1]
    else:
        n = max(2, int(math.floor(length / max(float(ds_m), 0.05))) + 1)
        s = np.linspace(0.0, length, n, dtype=float)
        x = _interp(s_nodes, pts[:, 0], s)
        y = _interp(s_nodes, pts[:, 1], s)
    dx = np.gradient(x, s, edge_order=1)
    dy = np.gradient(y, s, edge_order=1)
    yaw = np.unwrap(np.arctan2(dy, dx))
    kappa = np.gradient(yaw, s, edge_order=1)
    if ds_m is None:
        core = np.abs(kappa[1:-1]) if kappa.size >= 3 else np.abs(kappa)
    else:
        core = np.abs(kappa[(s >= 0.5) & (s <= length - 0.5)])
        if core.size == 0:
            core = np.abs(kappa)
    max_k = float(np.max(core)) if core.size else float("inf")
    q = float(np.quantile(core, np.clip(quantile, 0.0, 1.0))) if core.size else float("inf")
    return {
        "n_points": int(s.size),
        "length_m": length,
        "max_abs_curvature": max_k,
        "curvature_quantile": q,
        "dense": bool(ds_m is not None),
        "ok": True,
    }


def compare_native_dense_curvature(
    points_xy: Sequence[tuple[float, float]],
    *,
    ds_m: float = 0.20,
    quantile: float = 0.90,
    hard_max: float = 1.00,
) -> dict[str, object]:
    """Attribute a hard curvature reject to native geometry vs dense densify."""
    native = curvature_profile_from_xy(points_xy, ds_m=None, quantile=quantile)
    dense = curvature_profile_from_xy(points_xy, ds_m=ds_m, quantile=quantile)
    n_max = float(native["max_abs_curvature"])  # type: ignore[arg-type]
    d_max = float(dense["max_abs_curvature"])  # type: ignore[arg-type]
    if n_max > hard_max and d_max > hard_max:
        source = "native_and_dense"
    elif d_max > hard_max >= n_max:
        source = "dense_pchip_spike"
    elif n_max > hard_max >= d_max:
        source = "native_only"
    else:
        source = "neither_exceeds_hard"
    return {
        "native": native,
        "dense": dense,
        "hard_max": float(hard_max),
        "attribution": source,
    }


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
    curvature_quantile: float
    length_m: float
    switch_lateral_5m: float = 0.0
    switch_heading_5m_deg: float = 0.0
    # History-aware lateral mode (signed ego-frame lateral at probe distance).
    signed_lateral_raw_m: float = 0.0
    signed_lateral_committed_m: float = 0.0
    lateral_mode_raw: int = 0
    lateral_mode_committed: int = 0


@dataclass(frozen=True)
class PathManagerConfig:
    resample_ds_m: float = 0.20
    min_path_length_m: float = 5.0
    horizon_m: float = 20.0
    min_forward_ratio: float = 0.90
    max_abs_curvature: float = 0.20
    curvature_quantile: float = 0.90
    hard_max_abs_curvature: float = 1.00
    max_switch_lateral_5m: float = 1.00
    max_switch_heading_5m_deg: float = 12.0
    # RTC-style near commitment + far update.  Freeze only the short prefix that
    # is already being executed, then transition once from old to latest.  A
    # previous new→old→new construction introduced two artificial seams.
    near_freeze_min_m: float = 0.80
    near_freeze_time_s: float = 0.25
    far_blend_m: float = 6.00
    # Retained as a compatibility field for older configs; no longer used to
    # switch back to latest in the near field.
    ego_snap_m: float = 0.0
    committed_prefix_min_m: float = 0.80
    committed_prefix_time_s: float = 0.25
    tail_blend_m: float = 6.00
    # After blend+PCHIP densify, re-check committed κ; fallback to latest-only if spiked.
    recheck_committed_curvature: bool = True
    # Geometric multi-history average is disabled by default: averaging straight
    # and pre-lane-change modes creates non-physical mid paths (ACT mode jump).
    use_latest_only_for_far: bool = True
    ensemble_weights: tuple[float, ...] = (0.60, 0.30, 0.10)
    # Legacy field names retained for config compatibility.  Reanchor is no
    # longer part of the accept path; these define the coarse-nav reverse veto
    # and its path probe distance.
    reanchor_nav_max_heading_deg: float = 95.0
    reanchor_probe_distance_m: float = 5.0
    # History-aware lateral mode (experimental). Default OFF: ego-frame lat@10m
    # sign flips are legal on S-curves and must not be treated as lane-change modes.
    enable_lateral_mode_flip: bool = False
    lateral_mode_probe_m: float = 10.0
    lateral_mode_deadband_m: float = 0.50
    lateral_mode_flip_min_m: float = 1.20
    # If coarse nav is nearly straight ahead, diagnose preemptive side bias.
    enable_early_lane_change: bool = True
    nav_straight_lat_m: float = 2.50
    path_vs_nav_max_extra_lat_m: float = 2.00


@dataclass(frozen=True)
class PathUpdate:
    accepted: bool
    reason: str
    raw: SpatialPath | None
    committed: SpatialPath | None
    quality: PathQuality
    reanchor_pending_count: int = 0


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

    def _probe_pose(
        self, path: SpatialPath, ego: EgoPose, distance_m: float | None = None
    ) -> tuple[float, float, float]:
        """Map-frame (x, y, yaw) at a fixed arc distance along the path."""
        dist = float(
            distance_m
            if distance_m is not None
            else self.config.reanchor_probe_distance_m
        )
        dist = min(max(0.5, dist), max(0.5, path.length_m * 0.9))
        x, y, yaw, _k = path.relative_samples(ego, np.array([dist], dtype=float))
        return float(x[0]), float(y[0]), float(yaw[0])

    def _signed_lateral_ego(
        self, path: SpatialPath, ego: EgoPose, distance_m: float | None = None
    ) -> float:
        """Ego-frame lateral of the path point at ``distance_m`` (left +)."""
        dist = float(
            distance_m if distance_m is not None else self.config.lateral_mode_probe_m
        )
        dist = min(max(0.5, dist), max(0.5, path.length_m * 0.9))
        px, py, _yaw = self._probe_pose(path, ego, dist)
        dx, dy = px - ego.x, py - ego.y
        return float(-math.sin(ego.yaw) * dx + math.cos(ego.yaw) * dy)

    @staticmethod
    def _lateral_mode(signed_lat_m: float, deadband_m: float) -> int:
        if signed_lat_m > deadband_m:
            return 1
        if signed_lat_m < -deadband_m:
            return -1
        return 0

    def _nav_lateral_ego(
        self, ego: EgoPose, nav_target_map_xy: tuple[float, float] | None
    ) -> float | None:
        if nav_target_map_xy is None:
            return None
        dx = float(nav_target_map_xy[0]) - ego.x
        dy = float(nav_target_map_xy[1]) - ego.y
        return float(-math.sin(ego.yaw) * dx + math.cos(ego.yaw) * dy)

    def _intrinsic_failure(self, raw: SpatialPath, ego: EgoPose) -> str | None:
        """Hard geometric failures independent of reference continuity."""
        cfg = self.config
        if raw.length_m < cfg.min_path_length_m:
            return "too_short"
        if self._forward_ratio(raw, ego) < cfg.min_forward_ratio:
            return "not_forward"
        if self._self_intersects(raw):
            return "self_intersection"
        curvature_core = np.abs(raw.kappa[(raw.s >= 0.5) & (raw.s <= raw.length_m - 0.5)])
        if curvature_core.size == 0:
            curvature_core = np.abs(raw.kappa)
        max_kappa = float(np.max(curvature_core)) if curvature_core.size else float("inf")
        if max_kappa > cfg.hard_max_abs_curvature:
            return "curvature_hard_limit"
        # The robust/quantile curvature limit is a continuity diagnostic, not an
        # intrinsic invalidity.  Legitimate junction turns can exceed it for a
        # few VLA frames.  Only the hard pointwise bound is fail-closed.
        return None

    def _nav_alignment_ok(
        self,
        raw: SpatialPath,
        ego: EgoPose,
        nav_target_map_xy: tuple[float, float] | None,
    ) -> bool:
        """True when the VLA path is not clearly reverse of the coarse nav target."""
        if nav_target_map_xy is None:
            return True
        tx, ty = float(nav_target_map_xy[0]), float(nav_target_map_xy[1])
        nav_dx, nav_dy = tx - ego.x, ty - ego.y
        nav_norm = math.hypot(nav_dx, nav_dy)
        if nav_norm < 1.0:
            return True
        _px, _py, path_yaw = self._probe_pose(raw, ego)
        path_dx, path_dy = math.cos(path_yaw), math.sin(path_yaw)
        cos_ang = (path_dx * nav_dx + path_dy * nav_dy) / nav_norm
        cos_ang = max(-1.0, min(1.0, cos_ang))
        ang_deg = abs(math.degrees(math.acos(cos_ang)))
        return ang_deg <= float(self.config.reanchor_nav_max_heading_deg)

    def _switch_metrics(
        self, raw: SpatialPath, ego: EgoPose
    ) -> tuple[float, float]:
        lat_jump = 0.0
        head_jump_deg = 0.0
        if self._committed is None:
            return lat_jump, head_jump_deg
        distance = min(5.0, raw.length_m, self._committed.length_m)
        rx, ry, ryaw, _ = raw.relative_samples(ego, np.array([distance]))
        ox, oy, oyaw, _ = self._committed.relative_samples(ego, np.array([distance]))
        normal_x, normal_y = -math.sin(float(oyaw[0])), math.cos(float(oyaw[0]))
        lat_jump = abs(
            (float(rx[0]) - float(ox[0])) * normal_x + (float(ry[0]) - float(oy[0])) * normal_y
        )
        head_jump_deg = abs(math.degrees(wrap_angle(float(ryaw[0]) - float(oyaw[0]))))
        return lat_jump, head_jump_deg

    def _quality(
        self,
        raw: SpatialPath | None,
        ego: EgoPose,
        nav_target_map_xy: tuple[float, float] | None = None,
    ) -> PathQuality:
        if raw is None:
            return PathQuality(
                False,
                "degenerate",
                0.0,
                False,
                float("inf"),
                float("inf"),
                0.0,
            )
        cfg = self.config
        forward = self._forward_ratio(raw, ego)
        intersects = self._self_intersects(raw)
        curvature_core = np.abs(raw.kappa[(raw.s >= 0.5) & (raw.s <= raw.length_m - 0.5)])
        if curvature_core.size == 0:
            curvature_core = np.abs(raw.kappa)
        max_kappa = float(np.max(curvature_core)) if curvature_core.size else float("inf")
        curvature_quantile = (
            float(
                np.quantile(
                    curvature_core,
                    np.clip(cfg.curvature_quantile, 0.0, 1.0),
                )
            )
            if curvature_core.size
            else float("inf")
        )
        lat_jump, head_jump_deg = self._switch_metrics(raw, ego)
        lat_raw = self._signed_lateral_ego(raw, ego)
        lat_com = (
            self._signed_lateral_ego(self._committed, ego)
            if self._committed is not None
            else 0.0
        )
        mode_raw = self._lateral_mode(lat_raw, cfg.lateral_mode_deadband_m)
        mode_com = self._lateral_mode(lat_com, cfg.lateral_mode_deadband_m)
        mode_flip_ok = True
        if (
            bool(cfg.enable_lateral_mode_flip)
            and self._committed is not None
            and mode_com != 0
            and mode_raw != 0
            and mode_com != mode_raw
            and abs(lat_raw) >= float(cfg.lateral_mode_flip_min_m)
        ):
            mode_flip_ok = False
        # Coarse nav nearly straight ahead, but VLA path pre-biases hard into a side lane.
        nav_lat_ok = True
        nav_lat = self._nav_lateral_ego(ego, nav_target_map_xy)
        if (
            bool(cfg.enable_early_lane_change)
            and nav_lat is not None
            and abs(nav_lat) <= float(cfg.nav_straight_lat_m)
        ):
            if abs(lat_raw) > abs(nav_lat) + float(cfg.path_vs_nav_max_extra_lat_m):
                nav_lat_ok = False
        # The ordered list chooses the primary diagnostic.  update() separately
        # decides whether that diagnostic is hard-invalid or continuity-only.
        checks = (
            (raw.length_m >= cfg.min_path_length_m, "too_short"),
            (forward >= cfg.min_forward_ratio, "not_forward"),
            (not intersects, "self_intersection"),
            (max_kappa <= cfg.hard_max_abs_curvature, "curvature_hard_limit"),
            (curvature_quantile <= cfg.max_abs_curvature, "curvature_limit"),
            (lat_jump <= cfg.max_switch_lateral_5m, "lateral_switch"),
            (head_jump_deg <= cfg.max_switch_heading_5m_deg, "heading_switch"),
            (mode_flip_ok, "lateral_mode_flip"),
            (nav_lat_ok, "early_lane_change"),
        )
        reason = "ok"
        ok = True
        for passed, name in checks:
            if not passed:
                ok, reason = False, name
                break
        return PathQuality(
            ok,
            reason,
            forward,
            intersects,
            max_kappa,
            curvature_quantile,
            raw.length_m,
            lat_jump,
            head_jump_deg,
            lat_raw,
            lat_com,
            mode_raw,
            mode_com,
        )

    def _path_curvature_core(self, path: SpatialPath) -> np.ndarray:
        # MPC samples the reference from the first prediction step, so a seam in
        # the first 0.2–0.5 m is control-relevant rather than a disposable endpoint
        # artefact.  Exclude only one resampling cell for numerical gradients.
        margin = max(0.05, float(self.config.resample_ds_m))
        core = np.abs(
            path.kappa[
                (path.s >= margin)
                & (path.s <= max(margin, path.length_m - margin))
            ]
        )
        if core.size == 0:
            core = np.abs(path.kappa)
        return core

    def _committed_curvature_ok(self, path: SpatialPath) -> bool:
        """Post-blend densify must not invent harder turns than raw quality gates."""
        cfg = self.config
        core = self._path_curvature_core(path)
        if core.size == 0:
            return False
        max_k = float(np.max(core))
        q = float(np.quantile(core, np.clip(cfg.curvature_quantile, 0.0, 1.0)))
        if max_k > float(cfg.hard_max_abs_curvature):
            return False
        # Blend artefacts often spike local max while quantile stays moderate —
        # still reject if max is well above soft limit.
        if max_k > max(float(cfg.max_abs_curvature) * 2.5, float(cfg.max_abs_curvature) + 0.15):
            return False
        if q > float(cfg.max_abs_curvature) * 1.25:
            return False
        return True

    def _dense_from_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ego: EgoPose,
        target_speed_mps: float,
        stamp_s: float,
        source_id: str,
    ) -> SpatialPath:
        # The sampled prefix starts at the ego projection on the reference.  Do
        # not let ``spatial_path_from_xy`` prepend the measured ego position when
        # projection error exceeds 3 cm: that extra segment caused a 0.2–0.4 m
        # heading kink and multi-radian/m near-field curvature.  The tracker
        # already projects the measured ego onto this path.
        build_ego = EgoPose(
            float(x[0]),
            float(y[0]),
            float(ego.yaw),
            float(ego.speed_mps),
        )
        dense = spatial_path_from_xy(
            list(zip(x.tolist(), y.tolist(), strict=True)),
            ego=build_ego,
            target_speed_mps=target_speed_mps,
            stamp_s=stamp_s,
            ds_m=self.config.resample_ds_m,
            source_id=source_id,
        )
        assert dense is not None
        return dense

    def _ensemble(self, ego: EgoPose, stamp_s: float) -> SpatialPath:
        """RTC-style commit: one old-prefix→latest transition; recheck κ.

        Samples share ego-relative arc stations (world frame), not point indices.
        The first sample remains the projected path point; the tracker projects
        measured ego onto it.  If blend densify creates κ spikes, fall back to
        latest-only (still pure VLA geometry).
        """
        cfg = self.config
        latest = self._raw_history[0]
        max_len = min(float(cfg.horizon_m), float(latest.length_m))
        if self._committed is not None:
            max_len = min(
                float(cfg.horizon_m),
                max(float(latest.length_m), float(self._committed.length_m)),
            )
        max_len = max(max_len, 1.0)
        distances = np.arange(0.0, max_len + 0.5 * cfg.resample_ds_m, cfg.resample_ds_m)
        if distances.size < 2:
            distances = np.asarray([0.0, max_len], dtype=float)

        if cfg.use_latest_only_for_far or len(self._raw_history) == 1:
            x_new, y_new, _yaw, _k = latest.relative_samples(ego, distances)
        else:
            lat0 = self._signed_lateral_ego(latest, ego)
            mode0 = self._lateral_mode(lat0, cfg.lateral_mode_deadband_m)
            xs, ys, wts = [], [], []
            for path, w in zip(
                self._raw_history, cfg.ensemble_weights[: len(self._raw_history)], strict=False
            ):
                lat = self._signed_lateral_ego(path, ego)
                mode = self._lateral_mode(lat, cfg.lateral_mode_deadband_m)
                if mode0 != 0 and mode != 0 and mode != mode0:
                    continue
                x, y, _yaw, _k = path.relative_samples(ego, distances)
                xs.append(x)
                ys.append(y)
                wts.append(float(w))
            if not xs:
                x_new, y_new, _yaw, _k = latest.relative_samples(ego, distances)
            else:
                w = np.asarray(wts, dtype=float)
                w /= max(float(w.sum()), 1e-9)
                x_new = np.average(np.vstack(xs), axis=0, weights=w)
                y_new = np.average(np.vstack(ys), axis=0, weights=w)

        target_speed = latest.target_speed_mps
        x_latest, y_latest = x_new.copy(), y_new.copy()

        if self._committed is not None:
            old_x, old_y, _old_yaw, _old_k = self._committed.relative_samples(ego, distances)
            freeze_m = max(
                float(cfg.near_freeze_min_m),
                float(cfg.committed_prefix_min_m),
                max(0.0, ego.speed_mps)
                * max(float(cfg.near_freeze_time_s), float(cfg.committed_prefix_time_s)),
            )
            blend_m = max(float(cfg.far_blend_m), float(cfg.tail_blend_m), 1e-3)
            # alpha=0 → old, alpha=1 → latest.  There is deliberately one
            # transition only; the former latest→old→latest shape caused the
            # yellow committed path to oscillate even when both inputs were sane.
            alpha = np.zeros_like(distances)
            for i, d in enumerate(distances):
                if d > freeze_m:
                    t = (d - freeze_m) / blend_m
                    t = float(np.clip(t, 0.0, 1.0))
                    t = t * t * (3.0 - 2.0 * t)
                    alpha[i] = t
            x_new = (1.0 - alpha) * old_x + alpha * x_latest
            y_new = (1.0 - alpha) * old_y + alpha * y_latest

        dense = self._dense_from_xy(
            x_new,
            y_new,
            ego=ego,
            target_speed_mps=target_speed,
            stamp_s=stamp_s,
            source_id="vla_committed",
        )
        if (
            self._committed is not None
            and bool(cfg.recheck_committed_curvature)
            and not self._committed_curvature_ok(dense)
        ):
            # Blend invented a non-physical seam — execute pure latest VLA path.
            dense = self._dense_from_xy(
                x_latest,
                y_latest,
                ego=ego,
                target_speed_mps=target_speed,
                stamp_s=stamp_s,
                source_id="vla_committed_latest_fallback",
            )
        return dense

    def _apply_speed_only_on_reject(
        self, target_speed_mps: float
    ) -> None:
        if self._committed is None:
            return
        lowered_speed = min(self._committed.target_speed_mps, max(0.0, float(target_speed_mps)))
        if lowered_speed < self._committed.target_speed_mps:
            self._committed = replace(self._committed, target_speed_mps=lowered_speed)

    def _commit_path(self, raw: SpatialPath, ego: EgoPose, stamp_s: float) -> SpatialPath:
        self._raw_history.insert(0, raw)
        del self._raw_history[len(self.config.ensemble_weights) :]
        self._committed = self._ensemble(ego, stamp_s)
        return self._committed

    def update(
        self,
        points_xy: Sequence[tuple[float, float]],
        *,
        ego: EgoPose,
        target_speed_mps: float,
        stamp_s: float,
        source_id: str = "vla",
        nav_target_map_xy: tuple[float, float] | None = None,
    ) -> PathUpdate:
        raw = spatial_path_from_xy(
            points_xy,
            ego=ego,
            target_speed_mps=target_speed_mps,
            stamp_s=stamp_s,
            ds_m=self.config.resample_ds_m,
            source_id=source_id,
        )
        quality = self._quality(raw, ego, nav_target_map_xy=nav_target_map_xy)

        if raw is None:
            self._apply_speed_only_on_reject(target_speed_mps)
            return PathUpdate(False, "degenerate", None, self._committed, quality, 0)

        # Fail closed only for intrinsic invalid geometry or a path clearly
        # opposite the coarse navigation direction.  Continuity mismatches
        # (heading/lateral switch, soft curvature, heuristic lane bias) are not
        # invalid VLA intent at a junction; committing them through the bounded
        # prefix transition refreshes the reference without waiting for a stop.
        intrinsic = self._intrinsic_failure(raw, ego)
        if intrinsic is not None:
            self._apply_speed_only_on_reject(target_speed_mps)
            return PathUpdate(False, intrinsic, raw, self._committed, quality, 0)
        if not self._nav_alignment_ok(raw, ego, nav_target_map_xy):
            self._apply_speed_only_on_reject(target_speed_mps)
            return PathUpdate(
                False,
                "nav_reverse",
                raw,
                self._committed,
                quality,
                0,
            )

        committed = self._commit_path(raw, ego, stamp_s)
        reason = "accepted" if quality.ok else f"accepted_soft_{quality.reason}"
        return PathUpdate(True, reason, raw, committed, quality, 0)
