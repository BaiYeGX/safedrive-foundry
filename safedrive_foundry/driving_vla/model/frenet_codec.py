"""Frenet residual codec for R2-X Spatial K2 V2.

Maps native polyline (s, n) residuals to map-frame paths without fixed L/R bias.
"""

from __future__ import annotations

import math
from typing import Sequence

from driving_vla.model.canonicalizer import cum_arclength


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 1.0, 0.0
    return dx / n, dy / n


def path_frame(path_xy: Sequence[tuple[float, float]]) -> tuple[list[float], list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (s_list, tangents, normals) for each vertex."""
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 2:
        return [0.0], [(1.0, 0.0)], [(0.0, 1.0)]
    s_list = cum_arclength(pts)
    tans: list[tuple[float, float]] = []
    norms: list[tuple[float, float]] = []
    for i in range(len(pts)):
        if i + 1 < len(pts):
            tx, ty = _unit(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        else:
            tx, ty = _unit(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        tans.append((tx, ty))
        # left normal
        norms.append((-ty, tx))
    return list(s_list), tans, norms


def sample_native_at_s(
    path_xy: Sequence[tuple[float, float]],
    s_list: Sequence[float],
    tans: Sequence[tuple[float, float]],
    norms: Sequence[tuple[float, float]],
    s_query: float,
) -> tuple[float, float, float, float, float]:
    """Return (x, y, yaw, nx, ny) on native path at arc length s_query."""
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 2:
        x, y = pts[0] if pts else (0.0, 0.0)
        return x, y, 0.0, 0.0, 1.0
    s_q = max(0.0, min(float(s_query), float(s_list[-1])))
    # find segment
    j = 0
    for i in range(len(s_list) - 1):
        if s_list[i + 1] + 1e-12 >= s_q:
            j = i
            break
        j = i
    s0, s1 = float(s_list[j]), float(s_list[j + 1])
    denom = max(s1 - s0, 1e-9)
    u = (s_q - s0) / denom
    x = pts[j][0] + u * (pts[j + 1][0] - pts[j][0])
    y = pts[j][1] + u * (pts[j + 1][1] - pts[j][1])
    tx, ty = tans[j]
    nx, ny = norms[j]
    yaw = math.atan2(ty, tx)
    return x, y, yaw, nx, ny


def project_point_to_polyline(
    path_xy: Sequence[tuple[float, float]],
    point_xy: tuple[float, float],
) -> tuple[float, float, float, float, float, float]:
    """Project point onto polyline segments (not vertex-only search).

    Returns ``(s, d, x_proj, y_proj, nx, ny)`` where +d is left-normal offset.
    """
    pts = [(float(x), float(y)) for x, y in path_xy]
    px, py = float(point_xy[0]), float(point_xy[1])
    if len(pts) < 2:
        x0, y0 = pts[0] if pts else (px, py)
        return 0.0, 0.0, x0, y0, 0.0, 1.0
    s_list = cum_arclength(pts)
    best = None  # (dist2, s, d, x, y, nx, ny)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-18:
            continue
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        qx, qy = x0 + t * dx, y0 + t * dy
        tx, ty = _unit(dx, dy)
        nx, ny = -ty, tx  # left normal
        d_lat = (px - qx) * nx + (py - qy) * ny
        # residual after removing lateral (should be ~0 if on normal line)
        rx = (px - qx) - d_lat * nx
        ry = (py - qy) - d_lat * ny
        dist2 = rx * rx + ry * ry + d_lat * d_lat  # true Euclidean to segment point
        # Prefer true nearest: use Euclidean distance to projection
        dist2 = (px - qx) ** 2 + (py - qy) ** 2
        s = float(s_list[i]) + t * (float(s_list[i + 1]) - float(s_list[i]))
        cand = (dist2, s, d_lat, qx, qy, nx, ny)
        if best is None or dist2 < best[0] - 1e-15:
            best = cand
        elif abs(dist2 - best[0]) <= 1e-15 and s < best[1]:
            best = cand
    if best is None:
        x0, y0 = pts[0]
        return 0.0, 0.0, x0, y0, 0.0, 1.0
    _d2, s, d, qx, qy, nx, ny = best
    return float(s), float(d), float(qx), float(qy), float(nx), float(ny)


def project_path_to_frenet(
    native_path_xy: Sequence[tuple[float, float]],
    path_xy: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Project each path vertex onto native polyline → (s_list, d_list)."""
    s_out: list[float] = []
    d_out: list[float] = []
    prev_s = 0.0
    for i, p in enumerate(path_xy):
        s, d, _x, _y, _nx, _ny = project_point_to_polyline(native_path_xy, (p[0], p[1]))
        # enforce weak monotonic s for reporting (clamp forward)
        if i > 0 and s + 1e-9 < prev_s:
            s = prev_s
        prev_s = s
        s_out.append(s)
        d_out.append(d)
    return tuple(s_out), tuple(d_out)


def softplus(x: float) -> float:
    # stable softplus
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def envelope_at_index(i: int, n: int, *, ramp_points: int = 3) -> float:
    """Near-field envelope in [0,1]: first samples suppressed, then full residual."""
    if n <= 1:
        return 0.0
    if i <= 0:
        return 0.0
    if ramp_points <= 1:
        return 1.0
    return min(1.0, float(i) / float(ramp_points))


def decode_frenet_residual_path(
    native_path_xy: Sequence[tuple[float, float]],
    raw_delta_s: Sequence[float],
    raw_d: Sequence[float],
    *,
    max_lateral_m: float = 1.0,
    max_delta_s_per_step_m: float = 3.0,
    ramp_points: int = 3,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...], tuple[float, ...]]:
    """Decode residual sequences into map path + frenet s/d samples.

    - Δs via softplus for non-negative step advances (monotonic s).
    - d via tanh * max_lateral * envelope (no fixed sign bias in codec).
    """
    pts = [(float(x), float(y)) for x, y in native_path_xy]
    if len(pts) < 2:
        raise ValueError("native_path_xy needs >=2 points")
    n = min(len(pts), len(raw_delta_s), len(raw_d))
    if n < 2:
        raise ValueError("residual sequences too short")
    s_list, tans, norms = path_frame(pts)
    s_vals: list[float] = []
    d_vals: list[float] = []
    out: list[tuple[float, float]] = []
    s_cur = 0.0
    for i in range(n):
        ds = softplus(float(raw_delta_s[i]))
        ds = min(ds, float(max_delta_s_per_step_m))
        if i == 0:
            s_cur = 0.0
        else:
            s_cur = min(s_cur + ds, float(s_list[-1]))
        env = envelope_at_index(i, n, ramp_points=ramp_points)
        d = env * float(max_lateral_m) * math.tanh(float(raw_d[i]))
        x, y, _yaw, nx, ny = sample_native_at_s(pts, s_list, tans, norms, s_cur)
        out.append((x + d * nx, y + d * ny))
        s_vals.append(s_cur)
        d_vals.append(d)
    return tuple(out), tuple(s_vals), tuple(d_vals)


def max_xy_separation(
    path_a: Sequence[tuple[float, float]],
    path_b: Sequence[tuple[float, float]],
) -> float:
    n = min(len(path_a), len(path_b))
    if n == 0:
        return 0.0
    return max(
        math.hypot(float(path_a[i][0]) - float(path_b[i][0]), float(path_a[i][1]) - float(path_b[i][1]))
        for i in range(n)
    )


def first_point_error(
    path: Sequence[tuple[float, float]],
    ego_xy: tuple[float, float],
) -> float:
    if not path:
        return float("inf")
    return math.hypot(float(path[0][0]) - float(ego_xy[0]), float(path[0][1]) - float(ego_xy[1]))


def smooth_scalar_series(values: Sequence[float], *, passes: int = 2) -> list[float]:
    """3-point moving average; preserves endpoints (near-field continuity)."""
    xs = [float(v) for v in values]
    if len(xs) < 3:
        return xs
    for _ in range(max(1, int(passes))):
        out = [xs[0]]
        for i in range(1, len(xs) - 1):
            out.append((xs[i - 1] + xs[i] + xs[i + 1]) / 3.0)
        out.append(xs[-1])
        xs = out
    return xs


def path_max_abs_curvature(path_xy: Sequence[tuple[float, float]]) -> float:
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 3:
        return 0.0
    max_k = 0.0
    for i in range(1, len(pts) - 1):
        dx0, dy0 = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        dx1, dy1 = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        ds0 = math.hypot(dx0, dy0)
        ds1 = math.hypot(dx1, dy1)
        if ds0 < 1e-6 or ds1 < 1e-6:
            continue
        a0 = math.atan2(dy0, dx0)
        a1 = math.atan2(dy1, dx1)
        dpsi = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        kappa = abs(dpsi) / max(0.5 * (ds0 + ds1), 1e-6)
        max_k = max(max_k, kappa)
    return max_k


def smooth_path_xy(
    path_xy: Sequence[tuple[float, float]],
    *,
    passes: int = 2,
    hard_max_abs_curvature: float | None = None,
) -> tuple[tuple[float, float], ...]:
    """Smooth polyline in XY; optional iterative damp if kappa exceeds live hard limit.

    Used so proposal paths can pass PathManager hard_max_abs_curvature=1.0 without
    silently relaxing the live threshold.
    """
    pts = [(float(x), float(y)) for x, y in path_xy]
    if len(pts) < 3:
        return tuple(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for _ in range(max(1, int(passes))):
        xs = smooth_scalar_series(xs, passes=1)
        ys = smooth_scalar_series(ys, passes=1)
        # lock first point (near-field)
        xs[0], ys[0] = pts[0][0], pts[0][1]
    out = list(zip(xs, ys))
    if hard_max_abs_curvature is None:
        return tuple(out)
    # progressive extra smoothing until under hard limit or cap
    for _ in range(8):
        if path_max_abs_curvature(out) <= hard_max_abs_curvature + 1e-6:
            break
        xs = smooth_scalar_series([p[0] for p in out], passes=1)
        ys = smooth_scalar_series([p[1] for p in out], passes=1)
        xs[0], ys[0] = pts[0][0], pts[0][1]
        out = list(zip(xs, ys))
    return tuple(out)
