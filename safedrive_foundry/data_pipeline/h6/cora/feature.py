"""Physically allow-listed, metadata-source-blind CORA feature view."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from driving_vla.hybrid.contracts import ObservableAnchor
from safety_kernel.contracts.serialize import point_to_dict
from safety_kernel.contracts.types import PolicyCandidate, TrajectoryPoint

from .contracts import FEATURE_SCHEMA


FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "source",
        "slot",
        "branch_order",
        "provenance",
        "guard",
        "safety",
        "actor_future",
        "outcome",
        "winner",
        "oracle",
        "regression",
        "formal_answer",
        "operator",
        "label",
    }
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _key_forbidden(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(token == normalized or token in normalized for token in FORBIDDEN_KEY_TOKENS)


def validate_cora_feature_view(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FEATURE_SCHEMA:
        raise ValueError("cora_feature_schema")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _key_forbidden(str(key)):
                    raise ValueError(f"cora_feature_forbidden:{path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"cora_feature_non_finite:{path}")

    walk(payload, "feature")


def _trajectory_rows(trajectory: PolicyCandidate | Sequence[TrajectoryPoint] | Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    points: Any = trajectory.points if isinstance(trajectory, PolicyCandidate) else trajectory
    rows: list[dict[str, float]] = []
    for item in points:
        raw = point_to_dict(item) if isinstance(item, TrajectoryPoint) else dict(item)
        rows.append(
            {
                "t": float(raw["t"]),
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "yaw": float(raw["yaw"]),
                "kappa": float(raw["kappa"]),
                "v": float(raw["v"]),
                "a": float(raw["a"]),
                "jerk": float(raw.get("jerk", 0.0)),
            }
        )
    if len(rows) != 10:
        raise ValueError(f"cora_feature_trajectory_t10:{len(rows)}")
    return rows


def _from_anchor(anchor: ObservableAnchor) -> dict[str, Any]:
    bundle = anchor.bundle
    snapshot = anchor.safety_snapshot
    return {
        "ego": {
            "x": float(bundle.ego_x),
            "y": float(bundle.ego_y),
            "yaw": float(bundle.ego_yaw),
            "v": float(bundle.ego_v),
            "a": float(snapshot.ego_a),
        },
        "ego_history": [
            {"x": float(x), "y": float(y), "yaw": float(yaw), "v": float(v)}
            for x, y, yaw, v in bundle.ego_history
        ],
        "route": [{"x": float(x), "y": float(y)} for x, y in bundle.route_xy],
        "actors": [
            {
                "class_name": str(actor.class_name),
                "x": float(actor.x),
                "y": float(actor.y),
                "yaw": float(actor.yaw),
                "vx": float(actor.vx),
                "vy": float(actor.vy),
                "length_m": float(actor.length_m),
                "width_m": float(actor.width_m),
                "lost": bool(actor.lost),
                "cov_xx": float(actor.cov_xx),
                "cov_yy": float(actor.cov_yy),
            }
            for actor in snapshot.actors
        ],
        "traffic_lights": [
            {
                "state": str(light.state),
                "distance_m": float(light.distance_m),
                "stop_line_distance_m": (
                    None if light.stop_line_distance_m is None else float(light.stop_line_distance_m)
                ),
                "controls_ego_lane": light.controls_ego_lane,
            }
            for light in snapshot.traffic_lights
        ],
        "speed_limit_mps": (
            None if snapshot.speed_limit_mps is None else float(snapshot.speed_limit_mps)
        ),
        "corridor_centerline": [
            {"x": float(x), "y": float(y)} for x, y in snapshot.corridor_centerline
        ],
        "corridor_half_width_m": float(snapshot.corridor_half_width_m),
    }


def _from_mapping(anchor: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = anchor.get("observable_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    persisted_anchor = anchor.get("observable_anchor")
    persisted_anchor = persisted_anchor if isinstance(persisted_anchor, Mapping) else {}

    ego = anchor.get("ego")
    if not isinstance(ego, Mapping):
        ego = persisted_anchor.get("ego")
    if not isinstance(ego, Mapping) and all(
        key in snapshot for key in ("ego_x", "ego_y", "ego_yaw", "ego_v")
    ):
        ego = {
            "x": snapshot["ego_x"],
            "y": snapshot["ego_y"],
            "yaw": snapshot["ego_yaw"],
            "v": snapshot["ego_v"],
            "a": snapshot.get("ego_a", 0.0),
        }
    if not isinstance(ego, Mapping):
        raise ValueError("cora_feature_ego_missing")

    route = anchor.get("route", anchor.get("route_xy"))
    if not isinstance(route, (tuple, list)):
        route = snapshot.get("route", snapshot.get("route_xy"))
    if not isinstance(route, (tuple, list)):
        raise ValueError("cora_feature_route_missing")

    history = anchor.get("ego_history", anchor.get("observable_history", ()))
    normalized_history = []
    for row in history if isinstance(history, (tuple, list)) else ():
        if not isinstance(row, Mapping):
            continue
        normalized_history.append(
            {
                "x": float(row.get("x", row.get("ego_x"))),
                "y": float(row.get("y", row.get("ego_y"))),
                "yaw": float(row.get("yaw", row.get("ego_yaw"))),
                "v": float(row.get("v", row.get("ego_speed_mps"))),
            }
        )

    normalized_route = []
    for point in route:
        if isinstance(point, Mapping):
            normalized_route.append({"x": float(point["x"]), "y": float(point["y"])})
        else:
            normalized_route.append({"x": float(point[0]), "y": float(point[1])})

    actor_rows = anchor.get("actors", snapshot.get("actors", ()))
    normalized_actors = []
    for actor in actor_rows if isinstance(actor_rows, (tuple, list)) else ():
        if not isinstance(actor, Mapping):
            continue
        normalized_actors.append(
            {
                "class_name": str(actor["class_name"]),
                "x": float(actor["x"]),
                "y": float(actor["y"]),
                "yaw": float(actor["yaw"]),
                "vx": float(actor["vx"]),
                "vy": float(actor["vy"]),
                "length_m": float(actor["length_m"]),
                "width_m": float(actor["width_m"]),
                "lost": bool(actor["lost"]),
                "cov_xx": float(actor["cov_xx"]),
                "cov_yy": float(actor["cov_yy"]),
            }
        )

    light_rows = anchor.get("traffic_lights", snapshot.get("traffic_lights", ()))
    normalized_lights = []
    for light in light_rows if isinstance(light_rows, (tuple, list)) else ():
        if not isinstance(light, Mapping):
            continue
        stop_line = light.get("stop_line_distance_m")
        normalized_lights.append(
            {
                "state": str(light["state"]),
                "distance_m": float(light["distance_m"]),
                "stop_line_distance_m": None if stop_line is None else float(stop_line),
                "controls_ego_lane": light.get("controls_ego_lane"),
            }
        )

    centerline = anchor.get("corridor_centerline", snapshot.get("corridor_centerline", ()))
    normalized_centerline = []
    for point in centerline if isinstance(centerline, (tuple, list)) else ():
        if isinstance(point, Mapping):
            normalized_centerline.append({"x": float(point["x"]), "y": float(point["y"])})
        else:
            normalized_centerline.append({"x": float(point[0]), "y": float(point[1])})

    allowed: dict[str, Any] = {
        "ego": {
            "x": float(ego["x"]),
            "y": float(ego["y"]),
            "yaw": float(ego.get("yaw", 0.0)),
            "v": float(ego.get("v", 0.0)),
            "a": float(ego.get("a", snapshot.get("ego_a", 0.0))),
        },
        "ego_history": normalized_history,
        "route": normalized_route,
        "actors": normalized_actors,
        "traffic_lights": normalized_lights,
        "speed_limit_mps": (
            None
            if anchor.get("speed_limit_mps", snapshot.get("speed_limit_mps")) is None
            else float(anchor.get("speed_limit_mps", snapshot.get("speed_limit_mps")))
        ),
        "corridor_centerline": normalized_centerline,
        "corridor_half_width_m": float(
            anchor.get("corridor_half_width_m", snapshot.get("corridor_half_width_m", 0.0))
        ),
    }
    for optional in ("navigation_command", "frozen_sensor_features"):
        if optional in anchor:
            allowed[optional] = _jsonable(anchor[optional])
    return allowed


def build_cora_feature_view(
    anchor: ObservableAnchor | Mapping[str, Any],
    trajectory: PolicyCandidate | Sequence[TrajectoryPoint] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": FEATURE_SCHEMA,
        "observable": _from_anchor(anchor) if isinstance(anchor, ObservableAnchor) else _from_mapping(anchor),
        "trajectory": _trajectory_rows(trajectory),
    }
    validate_cora_feature_view(payload)
    return payload


__all__ = [
    "FORBIDDEN_KEY_TOKENS",
    "build_cora_feature_view",
    "validate_cora_feature_view",
]
