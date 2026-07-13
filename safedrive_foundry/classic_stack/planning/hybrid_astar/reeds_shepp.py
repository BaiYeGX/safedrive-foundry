"""Reeds–Shepp / Dubins analytic helpers for Hybrid A* expansions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from classic_stack.geometry import wrap_angle


@dataclass(frozen=True)
class RSPath:
    segments: tuple[tuple[str, float], ...]  # (L|R|S, signed arc/path length meters)
    length: float

    def sample(
        self,
        x0: float,
        y0: float,
        yaw0: float,
        radius: float,
        ds: float = 0.3,
    ) -> list[tuple[float, float, float]]:
        x, y, yaw = x0, y0, yaw0
        out = [(x, y, yaw)]
        for kind, length in self.segments:
            remaining = abs(length)
            sign = 1.0 if length >= 0.0 else -1.0
            while remaining > 1e-9:
                step = min(ds, remaining) * sign
                if kind == "S":
                    x += step * math.cos(yaw)
                    y += step * math.sin(yaw)
                elif kind == "L":
                    dtheta = step / radius
                    yaw_n = wrap_angle(yaw + dtheta)
                    x += radius * (math.sin(yaw_n) - math.sin(yaw))
                    y += -radius * (math.cos(yaw_n) - math.cos(yaw))
                    yaw = yaw_n
                elif kind == "R":
                    dtheta = -step / radius
                    yaw_n = wrap_angle(yaw + dtheta)
                    x += -radius * (math.sin(yaw_n) - math.sin(yaw))
                    y += radius * (math.cos(yaw_n) - math.cos(yaw))
                    yaw = yaw_n
                remaining -= abs(step)
                out.append((x, y, yaw))
        return out


def _mod2pi(angle: float) -> float:
    v = math.fmod(angle, 2.0 * math.pi)
    if v < -math.pi:
        v += 2.0 * math.pi
    elif v >= math.pi:
        v -= 2.0 * math.pi
    return v


def reeds_shepp_path(
    x0: float,
    y0: float,
    yaw0: float,
    x1: float,
    y1: float,
    yaw1: float,
    radius: float,
    *,
    xy_tol: float = 0.35,
    yaw_tol: float = 0.25,
) -> RSPath | None:
    """Return shortest RS path whose *sampled* endpoint matches the goal within tolerance.

    Invalid analytic families that do not reach the goal are discarded (no snap).
    """

    if radius <= 1e-6:
        return None
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(yaw0), math.sin(yaw0)
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = wrap_angle(yaw1 - yaw0)

    best: RSPath | None = None
    for segs in _all_families(x, y, phi):
        length = sum(abs(val) for _, val in segs) * radius
        scaled = tuple((k, v * radius) for k, v in segs)
        cand = RSPath(segments=scaled, length=length)
        samples = cand.sample(x0, y0, yaw0, radius, ds=max(0.2, radius * 0.08))
        if not samples:
            continue
        ex, ey, eyaw = samples[-1]
        if math.hypot(ex - x1, ey - y1) > xy_tol:
            continue
        if abs(wrap_angle(eyaw - yaw1)) > yaw_tol:
            continue
        if best is None or cand.length < best.length:
            best = cand
    return best


def segment_gears(path: RSPath) -> list[int]:
    """+1 forward / -1 reverse per RS segment (signed length)."""

    return [1 if length >= 0.0 else -1 for _kind, length in path.segments]


def dubins_path_length(
    x0: float,
    y0: float,
    yaw0: float,
    x1: float,
    y1: float,
    yaw1: float,
    radius: float,
) -> float:
    path = reeds_shepp_path(x0, y0, yaw0, x1, y1, yaw1, radius)
    if path is None:
        return math.hypot(x1 - x0, y1 - y0)
    return path.length


def _all_families(x: float, y: float, phi: float) -> list[list[tuple[str, float]]]:
    out: list[list[tuple[str, float]]] = []
    for fn in (_lsl, _rsr, _lsr, _rsl, _lrl, _rlr):
        res = fn(x, y, phi)
        if res is not None:
            out.append(res)
    return out


def _lsl(x: float, y: float, phi: float) -> list[tuple[str, float]] | None:
    t = _mod2pi(math.atan2(y - 1.0 + math.cos(phi), x - math.sin(phi)))
    u = math.hypot(x - math.sin(phi), y - 1.0 + math.cos(phi))
    v = _mod2pi(phi - t)
    return [("L", t), ("S", u), ("L", v)]


def _rsr(x: float, y: float, phi: float) -> list[tuple[str, float]] | None:
    t = _mod2pi(-math.atan2(-y - 1.0 + math.cos(phi), x + math.sin(phi)))
    u = math.hypot(x + math.sin(phi), -y - 1.0 + math.cos(phi))
    v = _mod2pi(-phi - t)
    return [("R", t), ("S", u), ("R", v)]


def _lsr(x: float, y: float, phi: float) -> list[tuple[str, float]] | None:
    u1 = math.hypot(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 * u1 < 4.0:
        return None
    u = math.sqrt(max(0.0, u1 * u1 - 4.0))
    theta = math.atan2(2.0, u)
    t = _mod2pi(math.atan2(y - 1.0 - math.cos(phi), x + math.sin(phi)) + theta)
    v = _mod2pi(t - phi)
    return [("L", t), ("S", u), ("R", v)]


def _rsl(x: float, y: float, phi: float) -> list[tuple[str, float]] | None:
    u1 = math.hypot(x - math.sin(phi), -y - 1.0 - math.cos(phi))
    if u1 * u1 < 4.0:
        return None
    u = math.sqrt(max(0.0, u1 * u1 - 4.0))
    theta = math.atan2(2.0, u)
    t = _mod2pi(-math.atan2(-y - 1.0 - math.cos(phi), x - math.sin(phi)) + theta)
    v = _mod2pi(-phi - t)
    return [("R", t), ("S", u), ("L", v)]


def _lrl(x: float, y: float, phi: float) -> list[tuple[str, float]] | None:
    u1 = math.hypot(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 > 4.0:
        return None
    u = _mod2pi(2.0 * math.pi - math.acos(max(-1.0, min(1.0, 0.25 * u1))))
    t = _mod2pi(math.atan2(y - 1.0 + math.cos(phi), x - math.sin(phi)) + 0.5 * u)
    v = _mod2pi(phi - t + u)
    return [("L", t), ("R", -u), ("L", v)]


def _rlr(x: float, y: float, phi: float) -> list[tuple[str, float]] | None:
    u1 = math.hypot(x + math.sin(phi), y + 1.0 - math.cos(phi))
    if u1 > 4.0:
        return None
    u = _mod2pi(2.0 * math.pi - math.acos(max(-1.0, min(1.0, 0.25 * u1))))
    t = _mod2pi(-math.atan2(y + 1.0 - math.cos(phi), x + math.sin(phi)) + 0.5 * u)
    v = _mod2pi(-phi - t + u)
    return [("R", t), ("L", -u), ("R", v)]
