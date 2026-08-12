"""Observable-only coordinate transforms and tensor builders."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ACTOR_FEATURES,
    CANDIDATE_FEATURES,
    EGO_FEATURES,
    HISTORY,
    K,
    MAX_ACTORS,
    MAX_ROAD_POINTS,
    MAX_ROAD_POLYLINES,
    ROAD_FEATURES,
    T,
    WorldContractError,
)

FORBIDDEN_OBSERVABLE_KEYS = frozenset(
    {
        "oracle",
        "oracle_winner",
        "oracle_only",
        "actor_future",
        "true_actor_future",
        "privileged_future",
        "scripted_intent",
        "minimum_ttc_oracle",
    }
)


def assert_observable_only(payload: Mapping[str, Any]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_s = str(key)
                if key_s in FORBIDDEN_OBSERVABLE_KEYS or key_s.startswith("oracle_"):
                    raise WorldContractError(f"oracle/privileged key in observable payload: {path}{key_s}")
                walk(child, f"{path}{key_s}.")
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                walk(child, f"{path}{i}.")

    walk(payload, "")


def world_to_ego(
    x: float,
    y: float,
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
) -> tuple[float, float]:
    dx, dy = float(x) - float(ego_x), float(y) - float(ego_y)
    c, s = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
    return c * dx + s * dy, -s * dx + c * dy


def vector_world_to_ego(vx: float, vy: float, *, ego_yaw_rad: float) -> tuple[float, float]:
    c, s = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
    return c * float(vx) + s * float(vy), -s * float(vx) + c * float(vy)


def _history_indices(values: Sequence[Any], count: int = HISTORY) -> list[Any | None]:
    tail = list(values)[-count:]
    return [None] * (count - len(tail)) + tail


def build_ego_history(
    history: Sequence[Mapping[str, Any]],
    *,
    anchor_pose: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((HISTORY, EGO_FEATURES), dtype=np.float32)
    mask = np.zeros(HISTORY, dtype=bool)
    ex, ey, eyaw = anchor_pose
    for i, row in enumerate(_history_indices(history)):
        if row is None:
            continue
        x, y = world_to_ego(row["x"], row["y"], ego_x=ex, ego_y=ey, ego_yaw_rad=eyaw)
        vx, vy = vector_world_to_ego(row.get("vx", 0.0), row.get("vy", 0.0), ego_yaw_rad=eyaw)
        ax, ay = vector_world_to_ego(row.get("ax", 0.0), row.get("ay", 0.0), ego_yaw_rad=eyaw)
        yaw = float(row.get("yaw_rad", eyaw)) - eyaw
        out[i] = (
            x,
            y,
            math.sin(yaw),
            math.cos(yaw),
            vx,
            vy,
            ax,
            ay,
            float(row.get("yaw_rate", 0.0)),
            1.0,
            float(row.get("dt", 0.0)),
        )
        mask[i] = bool(row.get("valid", True))
    return out, mask


def build_actor_history(
    actors: Sequence[Mapping[str, Any]],
    *,
    anchor_pose: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    ex, ey, eyaw = anchor_pose

    def risk_key(actor: Mapping[str, Any]) -> tuple[float, str]:
        current = list(actor.get("history", []))[-1]
        x, y = world_to_ego(current["x"], current["y"], ego_x=ex, ego_y=ey, ego_yaw_rad=eyaw)
        vx, _ = vector_world_to_ego(
            current.get("vx", 0.0), current.get("vy", 0.0), ego_yaw_rad=eyaw
        )
        distance = math.hypot(x, y)
        closing_bonus = max(0.0, -vx) * 2.0
        corridor_bonus = 10.0 if x >= -5.0 and abs(y) <= 6.0 else 0.0
        score = distance - closing_bonus - corridor_bonus
        return score, str(actor.get("track_id_hash", ""))

    selected = sorted(
        [a for a in actors if a.get("history")],
        key=risk_key,
    )[:MAX_ACTORS]
    out = np.zeros((MAX_ACTORS, HISTORY, ACTOR_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_ACTORS, HISTORY), dtype=bool)
    ids: list[str] = []
    for ai, actor in enumerate(selected):
        ids.append(str(actor.get("track_id_hash", f"actor-{ai}")))
        length = float(actor.get("length", 4.5))
        width = float(actor.get("width", 1.8))
        covariance = float(actor.get("covariance", 0.0))
        actor_type = float(actor.get("type_id", 0.0))
        for hi, row in enumerate(_history_indices(actor["history"])):
            if row is None:
                continue
            if not bool(row.get("valid", True)):
                continue
            x, y = world_to_ego(row["x"], row["y"], ego_x=ex, ego_y=ey, ego_yaw_rad=eyaw)
            vx, vy = vector_world_to_ego(
                row.get("vx", 0.0), row.get("vy", 0.0), ego_yaw_rad=eyaw
            )
            yaw = float(row.get("yaw_rad", eyaw)) - eyaw
            out[ai, hi] = (
                x,
                y,
                math.sin(yaw),
                math.cos(yaw),
                vx,
                vy,
                float(row.get("ax", 0.0)),
                float(row.get("ay", 0.0)),
                length,
                width,
                covariance,
                float(row.get("time_since_seen", 0.0)),
                actor_type,
                1.0,
            )
            mask[ai, hi] = True
    return out, mask, tuple(ids)


def build_road_context(
    polylines: Sequence[Mapping[str, Any]],
    *,
    anchor_pose: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    ex, ey, eyaw = anchor_pose
    out = np.zeros(
        (MAX_ROAD_POLYLINES, MAX_ROAD_POINTS, ROAD_FEATURES), dtype=np.float32
    )
    mask = np.zeros((MAX_ROAD_POLYLINES, MAX_ROAD_POINTS), dtype=bool)
    for pi, poly in enumerate(list(polylines)[:MAX_ROAD_POLYLINES]):
        points = list(poly.get("points", []))[:MAX_ROAD_POINTS]
        kind = float(poly.get("type_id", pi))
        for qi, point in enumerate(points):
            x, y = world_to_ego(point[0], point[1], ego_x=ex, ego_y=ey, ego_yaw_rad=eyaw)
            if qi + 1 < len(points):
                dx, dy = points[qi + 1][0] - point[0], points[qi + 1][1] - point[1]
            elif qi:
                dx, dy = point[0] - points[qi - 1][0], point[1] - points[qi - 1][1]
            else:
                dx, dy = 1.0, 0.0
            tangent = math.atan2(dy, dx) - eyaw
            out[pi, qi] = (
                x,
                y,
                math.sin(tangent),
                math.cos(tangent),
                kind,
                float(poly.get("speed_limit_mps", 0.0)),
            )
            mask[pi, qi] = True
    return out, mask


def build_candidate_tensor(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, tuple[str | None, str | None]]:
    out = np.zeros((K, T, CANDIDATE_FEATURES), dtype=np.float32)
    mask = np.zeros(K, dtype=bool)
    reasons: list[str | None] = [None, None]
    for index in range(K):
        if index >= len(candidates):
            reasons[index] = "MISSING_CANDIDATE"
            continue
        candidate = candidates[index]
        if not bool(candidate.get("available", True)):
            reasons[index] = str(candidate.get("unavailable_reason") or "UNAVAILABLE")
            continue
        points = list(candidate.get("points", []))
        if len(points) != T:
            raise WorldContractError(f"candidate {index} requires exactly T={T} points")
        for ti, point in enumerate(points):
            out[index, ti] = (
                float(point["x"]),
                float(point["y"]),
                math.sin(float(point["yaw"])),
                math.cos(float(point["yaw"])),
                float(point["v"]),
                float(point["a"]),
                float(point["kappa"]),
                float(point.get("time", (ti + 1) * 0.25)),
            )
        mask[index] = True
    return out, mask, (reasons[0], reasons[1])
