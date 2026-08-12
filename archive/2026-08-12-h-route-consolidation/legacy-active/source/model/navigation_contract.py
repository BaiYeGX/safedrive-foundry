"""Runtime-observable navigation and lane-topology contract for K2 V3.

The route manoeuvre is mission input, never a model-selected alternative.
Map geometry is used to classify/bind the mission and to validate candidates;
it is not substituted for SimLingo's native ``pred_route`` at execution time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

NAVIGATION_SCHEMA_V3 = "safedrive.navigation_context.v3"


class RouteManeuver(str, Enum):
    FOLLOW_STRAIGHT = "FOLLOW_STRAIGHT"
    FOLLOW_CURVE_LEFT = "FOLLOW_CURVE_LEFT"
    FOLLOW_CURVE_RIGHT = "FOLLOW_CURVE_RIGHT"
    JUNCTION_STRAIGHT = "JUNCTION_STRAIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    ROUTE_CHANGE_LEFT = "ROUTE_CHANGE_LEFT"
    ROUTE_CHANGE_RIGHT = "ROUTE_CHANGE_RIGHT"


class TargetLaneSide(str, Enum):
    NONE = "NONE"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class TrafficSignalState(str, Enum):
    UNKNOWN = "UNKNOWN"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    STOP_SIGN = "STOP_SIGN"


def _enum_value(value: Enum | str) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _xy(route_xy: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in route_xy)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def route_heading_change_rad(route_xy: Sequence[Sequence[float]]) -> float:
    """Return physical heading change, positive left in CARLA coordinates.

    CARLA uses a left-handed frame: for a vehicle facing +x, +y is to its
    right and positive yaw is a physical right turn.  Negating the ordinary
    mathematical XY heading delta keeps all public maneuver semantics in the
    vehicle-centric convention (left positive, right negative).
    """
    points = _xy(route_xy)
    headings: list[float] = []
    for first, second in zip(points, points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if math.hypot(dx, dy) > 1.0e-6:
            headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    return -sum(_wrap_angle(b - a) for a, b in zip(headings, headings[1:]))


def classify_route_maneuver(
    route_xy: Sequence[Sequence[float]],
    *,
    junction_flags: Sequence[bool] = (),
    route_change_side: TargetLaneSide | str = TargetLaneSide.NONE,
    curve_min_deg: float = 12.0,
    junction_turn_min_deg: float = 35.0,
) -> RouteManeuver:
    """Classify an upstream mission polyline without consulting model output."""
    side = TargetLaneSide(_enum_value(route_change_side))
    if side is TargetLaneSide.LEFT:
        return RouteManeuver.ROUTE_CHANGE_LEFT
    if side is TargetLaneSide.RIGHT:
        return RouteManeuver.ROUTE_CHANGE_RIGHT

    delta_deg = math.degrees(route_heading_change_rad(route_xy))
    in_junction = any(bool(value) for value in junction_flags)
    if in_junction:
        if delta_deg >= float(junction_turn_min_deg):
            return RouteManeuver.TURN_LEFT
        if delta_deg <= -float(junction_turn_min_deg):
            return RouteManeuver.TURN_RIGHT
        return RouteManeuver.JUNCTION_STRAIGHT
    if delta_deg >= float(curve_min_deg):
        return RouteManeuver.FOLLOW_CURVE_LEFT
    if delta_deg <= -float(curve_min_deg):
        return RouteManeuver.FOLLOW_CURVE_RIGHT
    return RouteManeuver.FOLLOW_STRAIGHT


@dataclass(frozen=True)
class LaneAccessV3:
    side: TargetLaneSide
    exists: bool = False
    driving: bool = False
    same_direction: bool = False
    lane_change_allowed: bool = False
    currently_clear: bool = False
    road_id: int | None = None
    lane_id: int | None = None
    lane_width_m: float = 0.0
    centerline_xy: tuple[tuple[float, float], ...] = ()
    marking_type: str = "UNKNOWN"

    @property
    def authorized(self) -> bool:
        return bool(
            self.exists
            and self.driving
            and self.same_direction
            and self.lane_change_allowed
            and self.currently_clear
            and len(self.centerline_xy) >= 2
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["side"] = self.side.value
        value["centerline_xy"] = [list(point) for point in self.centerline_xy]
        value["authorized"] = self.authorized
        return value


@dataclass(frozen=True)
class RouteContextV3:
    maneuver: RouteManeuver
    route_xy: tuple[tuple[float, float], ...]
    origin_road_id: int | None = None
    origin_lane_id: int | None = None
    origin_lane_width_m: float = 3.5
    origin_lane_centerline_xy: tuple[tuple[float, float], ...] = ()
    target_road_id: int | None = None
    target_lane_id: int | None = None
    entry_signature: str = ""
    exit_signature: str = ""
    is_junction: bool = False
    has_crosswalk: bool = False
    stop_line_distance_m: float | None = None
    traffic_signal_state: TrafficSignalState = TrafficSignalState.UNKNOWN
    left_lane: LaneAccessV3 = LaneAccessV3(TargetLaneSide.LEFT)
    right_lane: LaneAccessV3 = LaneAccessV3(TargetLaneSide.RIGHT)
    conflict_zone_id: str = ""
    schema_version: str = NAVIGATION_SCHEMA_V3
    route_hash: str = ""
    topology_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != NAVIGATION_SCHEMA_V3:
            raise ValueError(f"unsupported navigation schema: {self.schema_version}")
        if len(self.route_xy) < 2:
            raise ValueError("route context requires at least two route points")
        computed_route = self.compute_route_hash()
        computed_topology = self.compute_topology_hash()
        if self.route_hash and self.route_hash != computed_route:
            raise ValueError("route_hash does not match route content")
        if self.topology_hash and self.topology_hash != computed_topology:
            raise ValueError("topology_hash does not match topology content")
        object.__setattr__(self, "route_hash", computed_route)
        object.__setattr__(self, "topology_hash", computed_topology)

    def route_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "maneuver": self.maneuver.value,
            "route_xy": [list(point) for point in self.route_xy],
            "origin_road_id": self.origin_road_id,
            "origin_lane_id": self.origin_lane_id,
            "target_road_id": self.target_road_id,
            "target_lane_id": self.target_lane_id,
            "entry_signature": self.entry_signature,
            "exit_signature": self.exit_signature,
        }

    def topology_payload(self) -> dict[str, Any]:
        payload = {
            "route_hash": self.compute_route_hash(),
            "is_junction": self.is_junction,
            "has_crosswalk": self.has_crosswalk,
            "stop_line_distance_m": self.stop_line_distance_m,
            "traffic_signal_state": self.traffic_signal_state.value,
            "left_lane": self.left_lane.to_dict(),
            "right_lane": self.right_lane.to_dict(),
            "conflict_zone_id": self.conflict_zone_id,
        }
        # Preserve cold-read compatibility with the first V3 development
        # artifacts, whose implicit CARLA default was 3.5 m.
        if abs(float(self.origin_lane_width_m) - 3.5) > 1.0e-9:
            payload["origin_lane_width_m"] = float(self.origin_lane_width_m)
        if self.origin_lane_centerline_xy:
            payload["origin_lane_centerline_xy"] = [
                list(point) for point in self.origin_lane_centerline_xy
            ]
        return payload

    def compute_route_hash(self) -> str:
        return canonical_sha256(self.route_payload())

    def compute_topology_hash(self) -> str:
        return canonical_sha256(self.topology_payload())

    def lane(self, side: TargetLaneSide | str) -> LaneAccessV3:
        target = TargetLaneSide(_enum_value(side))
        if target is TargetLaneSide.LEFT:
            return self.left_lane
        if target is TargetLaneSide.RIGHT:
            return self.right_lane
        raise ValueError("NONE has no adjacent lane")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.route_payload(),
            **self.topology_payload(),
            "origin_lane_width_m": float(self.origin_lane_width_m),
            "origin_lane_centerline_xy": [
                list(point) for point in self.origin_lane_centerline_xy
            ],
            "route_hash": self.route_hash,
            "topology_hash": self.topology_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RouteContextV3":
        def lane(raw: Mapping[str, Any] | None, side: TargetLaneSide) -> LaneAccessV3:
            source = dict(raw or {})
            return LaneAccessV3(
                side=side,
                exists=bool(source.get("exists", False)),
                driving=bool(source.get("driving", False)),
                same_direction=bool(source.get("same_direction", False)),
                lane_change_allowed=bool(source.get("lane_change_allowed", False)),
                currently_clear=bool(source.get("currently_clear", False)),
                road_id=source.get("road_id"),
                lane_id=source.get("lane_id"),
                lane_width_m=float(source.get("lane_width_m", 0.0)),
                centerline_xy=_xy(source.get("centerline_xy") or ()),
                marking_type=str(source.get("marking_type") or "UNKNOWN"),
            )

        return cls(
            maneuver=RouteManeuver(str(value["maneuver"])),
            route_xy=_xy(value.get("route_xy") or ()),
            origin_road_id=value.get("origin_road_id"),
            origin_lane_id=value.get("origin_lane_id"),
            origin_lane_width_m=float(value.get("origin_lane_width_m", 3.5)),
            origin_lane_centerline_xy=_xy(
                value.get("origin_lane_centerline_xy") or ()
            ),
            target_road_id=value.get("target_road_id"),
            target_lane_id=value.get("target_lane_id"),
            entry_signature=str(value.get("entry_signature") or ""),
            exit_signature=str(value.get("exit_signature") or ""),
            is_junction=bool(value.get("is_junction", False)),
            has_crosswalk=bool(value.get("has_crosswalk", False)),
            stop_line_distance_m=(
                None
                if value.get("stop_line_distance_m") is None
                else float(value["stop_line_distance_m"])
            ),
            traffic_signal_state=TrafficSignalState(
                str(value.get("traffic_signal_state") or "UNKNOWN")
            ),
            left_lane=lane(value.get("left_lane"), TargetLaneSide.LEFT),
            right_lane=lane(value.get("right_lane"), TargetLaneSide.RIGHT),
            conflict_zone_id=str(value.get("conflict_zone_id") or ""),
            schema_version=str(value.get("schema_version") or NAVIGATION_SCHEMA_V3),
            route_hash=str(value.get("route_hash") or ""),
            topology_hash=str(value.get("topology_hash") or ""),
        )


def build_route_context(
    route_xy: Sequence[Sequence[float]],
    *,
    junction_flags: Sequence[bool] = (),
    route_change_side: TargetLaneSide | str = TargetLaneSide.NONE,
    maneuver: RouteManeuver | str | None = None,
    **kwargs: Any,
) -> RouteContextV3:
    points = _xy(route_xy)
    selected = (
        RouteManeuver(_enum_value(maneuver))
        if maneuver is not None
        else classify_route_maneuver(
            points,
            junction_flags=junction_flags,
            route_change_side=route_change_side,
        )
    )
    return RouteContextV3(
        maneuver=selected,
        route_xy=points,
        is_junction=bool(any(junction_flags)),
        **kwargs,
    )


__all__ = [
    "NAVIGATION_SCHEMA_V3",
    "LaneAccessV3",
    "RouteContextV3",
    "RouteManeuver",
    "TargetLaneSide",
    "TrafficSignalState",
    "build_route_context",
    "canonical_sha256",
    "classify_route_maneuver",
    "route_heading_change_rad",
]
