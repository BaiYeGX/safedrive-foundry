"""Official SimLingo input contract helpers (targets, camera preprocess, config).

Keeps map geometry as coarse navigation only — never as a tracked centerline.
Implements RoutePlanner pop logic and BGR→JPEG→RGB crop matching agent_simlingo.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SimLingoContractConfig:
    """Live defaults follow official SimLingo eval; legacy is opt-in for A/B."""

    official_contract: bool = True
    # Camera (official training/eval)
    camera_width: int = 1024
    camera_height: int = 512
    camera_xyz: tuple[float, float, float] = (-1.5, 0.0, 2.0)
    camera_fov_deg: float = 110.0
    # RoutePlanner (agent_simlingo)
    route_planner_min_distance_m: float = 7.5
    route_planner_max_distance_m: float = 50.0
    densify_ds_m: float = 1.0
    # Legacy fixed arc targets (only when official_contract=False)
    legacy_target_d1_m: float = 15.0
    legacy_target_d2_m: float = 30.0
    # Preprocess
    jpeg_quality: int = 95  # cv2 default quality scale used by imencode
    crop_bottom_frac_num: float = 4.8  # official: (h * 4.8) // 16
    crop_bottom_frac_den: float = 16.0
    # PathManager experimental gates
    lateral_mode_flip_enabled: bool = False
    early_lane_change_enabled: bool = True
    # Prompt fragment when no far_command map is available (follow-road mode)
    command_text: str = "Command: follow the road."

    def evidence_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preprocess_chain"] = (
            "BGRA[:3]=BGR → cv2.imencode(.jpg) → imdecode → BGR2RGB → "
            "crop_bottom(4.8/16) → InternVL2 dynamic_preprocess 448"
            if self.official_contract
            else "RGB → PIL JPEG → crop_bottom(4.8/16) → InternVL2 448 (legacy)"
        )
        d["target_source"] = (
            "RoutePlanner[1],[2] after densify~1m + min_distance pop"
            if self.official_contract
            else f"legacy arc +{self.legacy_target_d1_m}/+{self.legacy_target_d2_m}m"
        )
        d["geometry_head"] = "pred_route → PathManager/MPC"
        d["speed_head"] = "pred_speed_wps → speed only"
        return d


def legacy_contract_config() -> SimLingoContractConfig:
    """Previous G3 defaults for offline A/B (640-class targets + RGB-JPEG path)."""
    return SimLingoContractConfig(
        official_contract=False,
        camera_width=640,
        camera_height=320,
        lateral_mode_flip_enabled=True,
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def densify_polyline_xy(
    points: Sequence[tuple[float, float]],
    *,
    ds_m: float = 1.0,
) -> list[tuple[float, float]]:
    """Resample a 2D polyline at approximately ``ds_m`` spacing."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 2:
        return pts
    # cumulative arc
    s_nodes = [0.0]
    for i in range(1, len(pts)):
        s_nodes.append(
            s_nodes[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        )
    total = s_nodes[-1]
    if total < 1e-6:
        return [pts[0], pts[-1]]
    ds = max(float(ds_m), 1e-3)
    n = max(2, int(math.floor(total / ds)) + 1)
    out: list[tuple[float, float]] = []
    for k in range(n):
        s_q = min(total, k * ds)
        # last sample snaps to end
        if k == n - 1:
            s_q = total
        # locate segment
        j = 1
        while j < len(s_nodes) and s_nodes[j] < s_q:
            j += 1
        j = min(j, len(s_nodes) - 1)
        s0, s1 = s_nodes[j - 1], s_nodes[j]
        u = 0.0 if s1 <= s0 else (s_q - s0) / (s1 - s0)
        x = pts[j - 1][0] + u * (pts[j][0] - pts[j - 1][0])
        y = pts[j - 1][1] + u * (pts[j][1] - pts[j - 1][1])
        out.append((float(x), float(y)))
    return out


def inverse_conversion_2d(
    point: tuple[float, float] | np.ndarray,
    translation: tuple[float, float] | np.ndarray,
    yaw: float,
) -> tuple[float, float]:
    """Official Rᵀ (point − translation). +x forward, +y CARLA-right."""
    px, py = float(point[0]), float(point[1])
    tx, ty = float(translation[0]), float(translation[1])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - tx, py - ty
    return (c * dx + s * dy, -s * dx + c * dy)


def ego_to_map(
    ego_xy: tuple[float, float],
    translation: tuple[float, float],
    yaw: float,
) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = float(ego_xy[0]), float(ego_xy[1])
    return (
        float(translation[0]) + c * x - s * y,
        float(translation[1]) + s * x + c * y,
    )


class RoutePlannerXY:
    """Carla-free port of team_code.nav_planner.RoutePlanner pop logic (XY only)."""

    def __init__(self, min_distance: float = 7.5, max_distance: float = 50.0) -> None:
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.route: deque[np.ndarray] = deque()
        self.route_distances: deque[float] = deque()
        self.is_last = False

    def set_route(self, points_xy: Sequence[tuple[float, float]]) -> None:
        self.route.clear()
        self.route_distances.clear()
        self.is_last = False
        for p in points_xy:
            self.route.append(np.array([float(p[0]), float(p[1]), 0.0], dtype=float))
        if len(self.route) < 1:
            return
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i] - self.route[i - 1]
            self.route_distances.append(float(math.hypot(float(diff[0]), float(diff[1]))))

    def run_step(self, gps_xy: tuple[float, float]) -> list[tuple[float, float]]:
        gps = np.array([float(gps_xy[0]), float(gps_xy[1]), 0.0], dtype=float)
        if len(self.route) <= 2:
            self.is_last = True
            return [(float(p[0]), float(p[1])) for p in self.route]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += self.route_distances[i]
            diff = self.route[i] - gps
            distance = float(math.hypot(float(diff[0]), float(diff[1])))
            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()

        return [(float(p[0]), float(p[1])) for p in self.route]


@dataclass
class NavigationTargetResult:
    target_ego_1: tuple[float, float]
    target_ego_2: tuple[float, float]
    target_map_1: tuple[float, float]
    target_map_2: tuple[float, float]
    progress_s: float
    valid: bool
    remaining_route_m: float
    target1_distance_m: float
    target2_distance_m: float
    target_separation_m: float
    dense_route_n: int
    mode: str  # "official_route_planner" | "legacy_arc"


def _polyline_s(points: Sequence[tuple[float, float]]) -> np.ndarray:
    s = [0.0]
    for i in range(1, len(points)):
        s.append(s[-1] + math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]))
    return np.asarray(s, dtype=float)


def _project_progress(
    route: Sequence[tuple[float, float]],
    route_s: np.ndarray,
    x: float,
    y: float,
    hint_s: float,
) -> float:
    if len(route) < 2:
        return 0.0
    best_s, best_d2 = float(hint_s), float("inf")
    lo = max(0.0, float(hint_s) - 2.0)
    hi = min(float(route_s[-1]), float(hint_s) + 40.0)
    for i in range(len(route) - 1):
        if route_s[i + 1] < lo or route_s[i] > hi:
            continue
        a = np.asarray(route[i], dtype=float)
        b = np.asarray(route[i + 1], dtype=float)
        p = np.asarray([x, y], dtype=float)
        ab = b - a
        denom = float(ab @ ab)
        u = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        q = a + u * ab
        d2 = float((p - q) @ (p - q))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(route_s[i] + u * (route_s[i + 1] - route_s[i]))
    return max(float(hint_s), best_s)


def _point_at_s(
    route: Sequence[tuple[float, float]], route_s: np.ndarray, s_query: float
) -> tuple[float, float]:
    s_query = float(np.clip(s_query, 0.0, float(route_s[-1])))
    for i in range(1, len(route_s)):
        if route_s[i] >= s_query:
            u = (s_query - route_s[i - 1]) / max(route_s[i] - route_s[i - 1], 1e-9)
            x = route[i - 1][0] + u * (route[i][0] - route[i - 1][0])
            y = route[i - 1][1] + u * (route[i][1] - route[i - 1][1])
            return (float(x), float(y))
    return (float(route[-1][0]), float(route[-1][1]))


def navigation_targets_official(
    route_xy: Sequence[tuple[float, float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    progress_hint_s: float = 0.0,
    config: SimLingoContractConfig | None = None,
) -> NavigationTargetResult:
    """Densify → keep forward suffix → RoutePlanner pop → remaining [1],[2] → Rᵀ.

    Official agent keeps a deque that advances with the vehicle so ``route[0]`` is
    near the ego.  When we inject a full map polyline we first drop points behind
    the projected progress (keep a short rear margin), then apply the same pop
    logic — otherwise ``[1]``/``[2]`` can be behind the car on mid-route starts.
    """
    cfg = config or SimLingoContractConfig()
    raw = [(float(x), float(y)) for x, y in route_xy]
    dense = densify_polyline_xy(raw, ds_m=cfg.densify_ds_m)
    dense_s = _polyline_s(dense)
    progress = _project_progress(dense, dense_s, ego_x, ego_y, progress_hint_s)

    # Keep a small rear margin so min_distance pop still sees nearby behind pts.
    rear_margin_m = min(2.0, float(cfg.route_planner_min_distance_m))
    start_s = max(0.0, progress - rear_margin_m)
    start_idx = 0
    for i, s in enumerate(dense_s):
        if s >= start_s:
            start_idx = i
            break
    forward = dense[start_idx:]
    if len(forward) < 2:
        forward = dense[-2:] if len(dense) >= 2 else dense

    planner = RoutePlannerXY(cfg.route_planner_min_distance_m, cfg.route_planner_max_distance_m)
    planner.set_route(forward)
    remaining = planner.run_step((ego_x, ego_y))

    if len(remaining) > 2:
        m1, m2 = remaining[1], remaining[2]
    elif len(remaining) > 1:
        m1 = m2 = remaining[1]
    elif remaining:
        m1 = m2 = remaining[0]
    else:
        m1 = m2 = (ego_x, ego_y)

    # Prefer ahead targets if pop left a behind sample (numerical edge).
    e_try = inverse_conversion_2d(m1, (ego_x, ego_y), ego_yaw)
    if e_try[0] < 0.5 and len(remaining) > 3:
        for j in range(1, min(len(remaining) - 1, 12)):
            cand = remaining[j]
            e_c = inverse_conversion_2d(cand, (ego_x, ego_y), ego_yaw)
            if e_c[0] >= 0.5:
                m1 = remaining[j]
                m2 = remaining[min(j + 1, len(remaining) - 1)]
                break

    e1 = inverse_conversion_2d(m1, (ego_x, ego_y), ego_yaw)
    e2 = inverse_conversion_2d(m2, (ego_x, ego_y), ego_yaw)
    t1_dist = math.hypot(e1[0], e1[1])
    t2_dist = math.hypot(e2[0], e2[1])
    sep = math.hypot(m2[0] - m1[0], m2[1] - m1[1])
    remaining_m = float(dense_s[-1]) - progress if dense_s.size else 0.0
    # Official does not hard-fail like our legacy 28 m check; keep soft validity
    # so short tails still feed the nearest remaining points.
    valid = (
        len(remaining) >= 2
        and t1_dist >= 1.0
        and remaining_m >= 3.0
    )
    return NavigationTargetResult(
        target_ego_1=e1,
        target_ego_2=e2,
        target_map_1=m1,
        target_map_2=m2,
        progress_s=progress,
        valid=valid,
        remaining_route_m=remaining_m,
        target1_distance_m=t1_dist,
        target2_distance_m=t2_dist,
        target_separation_m=sep,
        dense_route_n=len(dense),
        mode="official_route_planner",
    )


def navigation_targets_legacy(
    route_xy: Sequence[tuple[float, float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    progress_hint_s: float = 0.0,
    config: SimLingoContractConfig | None = None,
) -> NavigationTargetResult:
    """Legacy fixed +15 / +30 m along arc (previous G3 behaviour)."""
    cfg = config or legacy_contract_config()
    route = [(float(x), float(y)) for x, y in route_xy]
    route_s = _polyline_s(route)
    progress = _project_progress(route, route_s, ego_x, ego_y, progress_hint_s)
    m1 = _point_at_s(route, route_s, progress + cfg.legacy_target_d1_m)
    m2 = _point_at_s(route, route_s, progress + cfg.legacy_target_d2_m)
    e1 = inverse_conversion_2d(m1, (ego_x, ego_y), ego_yaw)
    e2 = inverse_conversion_2d(m2, (ego_x, ego_y), ego_yaw)
    remaining_m = float(route_s[-1]) - progress
    t1_dist = math.hypot(e1[0], e1[1])
    sep = math.hypot(m2[0] - m1[0], m2[1] - m1[1])
    valid = remaining_m >= 28.0 and t1_dist >= 3.0 and sep >= 2.0
    return NavigationTargetResult(
        target_ego_1=e1,
        target_ego_2=e2,
        target_map_1=m1,
        target_map_2=m2,
        progress_s=progress,
        valid=valid,
        remaining_route_m=remaining_m,
        target1_distance_m=t1_dist,
        target2_distance_m=math.hypot(e2[0], e2[1]),
        target_separation_m=sep,
        dense_route_n=len(route),
        mode="legacy_arc",
    )


def navigation_targets(
    route_xy: Sequence[tuple[float, float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    progress_hint_s: float = 0.0,
    config: SimLingoContractConfig | None = None,
) -> NavigationTargetResult:
    cfg = config or SimLingoContractConfig()
    if cfg.official_contract:
        return navigation_targets_official(
            route_xy,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_yaw=ego_yaw,
            progress_hint_s=progress_hint_s,
            config=cfg,
        )
    return navigation_targets_legacy(
        route_xy,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        progress_hint_s=progress_hint_s,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Camera preprocess
# ---------------------------------------------------------------------------


def crop_bottom_official(rgb: np.ndarray, *, num: float = 4.8, den: float = 16.0) -> np.ndarray:
    h = int(rgb.shape[0])
    cut = int(h - (h * num) // den)
    return rgb[:cut]


def preprocess_camera_official_bgr(bgr: np.ndarray, *, jpeg_quality: int = 95) -> np.ndarray:
    """Official agent_simlingo chain starting from BGR uint8 HxWx3.

    BGR → JPEG encode/decode → BGR2RGB → crop bottom 4.8/16.
    Returns RGB uint8 (cropped), ready for InternVL dynamic_preprocess.
    """
    arr = np.asarray(bgr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 BGR, got {arr.shape}")
    bgr3 = np.ascontiguousarray(arr[:, :, :3])
    if bgr3.dtype != np.uint8:
        bgr3 = np.clip(bgr3, 0, 255).astype(np.uint8)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "official SimLingo camera contract requires OpenCV (cv2). "
            "Install into the project venv: pip install opencv-python-headless"
        ) from exc
    ok, encoded = cv2.imencode(
        ".jpg", bgr3, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("cv2.imdecode failed")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return crop_bottom_official(rgb)


def preprocess_camera_legacy_rgb(rgb: np.ndarray) -> np.ndarray:
    """Legacy path: RGB → PIL JPEG roundtrip → crop (previous G3)."""
    import io
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
    rgb3 = np.ascontiguousarray(arr[:, :, :3])
    if rgb3.dtype != np.uint8:
        rgb3 = np.clip(rgb3, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb3).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return crop_bottom_official(out)


def carla_bgra_to_bgr(bgra: np.ndarray) -> np.ndarray:
    """CARLA raw BGRA → BGR (first three channels, no swap)."""
    arr = np.asarray(bgra)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx4 BGRA, got {arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3])


def build_official_prompt(
    *,
    speed_mps: float,
    command_text: str | None = None,
    eval_route_as: str = "target_point",
    use_cot: bool = True,
) -> str:
    """Match agent_simlingo prompt construction for target_point eval.

    Official target_point mode (agent_simlingo ~472–537):
      prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
      prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
    Do **not** inject an extra "Command: follow the road" unless callers pass
    command_text explicitly for non-target_point experiments.
    """
    speed = float(round(speed_mps, 1))
    if eval_route_as in {"target_point", "target_point_command"} or command_text is None:
        prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
    else:
        cmd = str(command_text).strip()
        if not cmd.endswith("."):
            cmd = cmd + "."
        prompt_tp = cmd
    if use_cot:
        return f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
    return f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."


__all__ = [
    "SimLingoContractConfig",
    "legacy_contract_config",
    "RoutePlannerXY",
    "NavigationTargetResult",
    "densify_polyline_xy",
    "inverse_conversion_2d",
    "ego_to_map",
    "navigation_targets",
    "navigation_targets_official",
    "navigation_targets_legacy",
    "preprocess_camera_official_bgr",
    "preprocess_camera_legacy_rgb",
    "carla_bgra_to_bgr",
    "crop_bottom_official",
    "build_official_prompt",
]
