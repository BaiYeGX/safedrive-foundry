"""Candidate-blind current-state adaptor for the R2 Basic 1v1 contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ObservableActor1v1:
    actor_present: bool
    actor_id: int | None = None
    actor_lon_m: float | None = None
    actor_lat_m: float | None = None
    actor_speed_mps: float | None = None
    actor_rel_lon_speed_mps: float | None = None
    current_ttc_s: float | None = None
    distance_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ObservableActor1v1":
        source = dict(value or {})
        return cls(
            actor_present=bool(source.get("actor_present", False)),
            actor_id=source.get("actor_id"),
            actor_lon_m=source.get("actor_lon_m"),
            actor_lat_m=source.get("actor_lat_m"),
            actor_speed_mps=source.get("actor_speed_mps"),
            actor_rel_lon_speed_mps=source.get("actor_rel_lon_speed_mps"),
            current_ttc_s=source.get("current_ttc_s"),
            distance_m=source.get("distance_m"),
        )


def observe_basic1v1_actor(
    *,
    ego: Any,
    actors: Iterable[Any],
) -> ObservableActor1v1:
    """Observe the nearest non-ego actor in the ego frame.

    Only current transforms and velocities are read.  Candidate identity,
    scripted future and Oracle outcomes are not inputs.
    """
    ego_tf = ego.get_transform()
    ego_velocity = ego.get_velocity()
    yaw = math.radians(float(ego_tf.rotation.yaw))
    fx, fy = math.cos(yaw), math.sin(yaw)
    ego_forward_speed = (
        float(ego_velocity.x) * fx + float(ego_velocity.y) * fy
    )
    nearest: tuple[float, Any, float, float] | None = None
    for actor in actors:
        if int(getattr(actor, "id", -1)) == int(getattr(ego, "id", -2)):
            continue
        try:
            actor_tf = actor.get_transform()
            dx = float(actor_tf.location.x - ego_tf.location.x)
            dy = float(actor_tf.location.y - ego_tf.location.y)
        except Exception:  # noqa: BLE001
            continue
        distance = math.hypot(dx, dy)
        if nearest is None or distance < nearest[0]:
            nearest = (distance, actor, dx, dy)
    if nearest is None:
        return ObservableActor1v1(actor_present=False)

    distance, actor, dx, dy = nearest
    actor_velocity = actor.get_velocity()
    actor_forward_speed = (
        float(actor_velocity.x) * fx + float(actor_velocity.y) * fy
    )
    lon = dx * fx + dy * fy
    # CARLA uses a left-handed map frame: with yaw=0, physical left is -Y.
    # Keep the public observable semantic intuitive (+left, -right).
    lat = dx * fy - dy * fx
    closing = ego_forward_speed - actor_forward_speed
    ttc = (
        lon / closing
        if lon > 0.0 and closing > 1.0e-3
        else math.inf
    )
    return ObservableActor1v1(
        actor_present=True,
        actor_id=int(getattr(actor, "id", -1)),
        actor_lon_m=float(lon),
        actor_lat_m=float(lat),
        actor_speed_mps=math.hypot(
            float(actor_velocity.x), float(actor_velocity.y)
        ),
        actor_rel_lon_speed_mps=float(actor_forward_speed - ego_forward_speed),
        current_ttc_s=float(min(ttc, 30.0)),
        distance_m=float(distance),
    )


def basic1v1_conflict_active(
    *,
    scenario_family: str,
    scene: ObservableActor1v1 | Mapping[str, Any] | None,
) -> bool:
    """Current-state risk latch input for teacher/contract smoke.

    Family selects the registered interaction geometry.  The decision itself
    uses only the current actor state, so a cleared actor deterministically
    returns the teacher to CLEAR and permits resume/rejoin.
    """
    observation = (
        scene
        if isinstance(scene, ObservableActor1v1)
        else ObservableActor1v1.from_mapping(scene)
    )
    family = str(scenario_family or "").lower()
    if family in {"clear", "empty", "clear_no_alternative"}:
        return False
    if not observation.actor_present:
        return False
    lon = float(observation.actor_lon_m or 0.0)
    lat = float(observation.actor_lat_m or 0.0)
    speed = float(observation.actor_speed_mps or 0.0)
    ttc = float(observation.current_ttc_s or 30.0)
    distance = float(observation.distance_m or math.hypot(lon, lat))

    if "lead" in family or "brake" in family or "follow" in family:
        return bool(
            0.0 < lon < 30.0
            and abs(lat) <= 2.2
            and (lon < 12.0 or ttc < 6.0 or (speed < 0.5 and lon < 25.0))
        )
    if "obstruction" in family or "narrow" in family:
        return bool(-2.0 < lon < 35.0 and abs(lat) <= 2.2)
    if "cut" in family:
        return bool(-3.0 < lon < 28.0 and (abs(lat) < 4.5 or ttc < 5.0))
    if "cross" in family:
        return bool(distance < 28.0 and -10.0 < lon < 30.0)
    if "merge" in family or "yield" in family:
        return bool(distance < 20.0 and -8.0 < lon < 25.0)
    return False


def conflict_side_from_scene(
    scene: ObservableActor1v1 | Mapping[str, Any] | None,
    *,
    deadband_m: float = 0.25,
) -> str:
    observation = (
        scene
        if isinstance(scene, ObservableActor1v1)
        else ObservableActor1v1.from_mapping(scene)
    )
    if not observation.actor_present or observation.actor_lat_m is None:
        return "none"
    lateral = float(observation.actor_lat_m)
    if lateral > float(deadband_m):
        return "left"
    if lateral < -float(deadband_m):
        return "right"
    return "center"


__all__ = [
    "ObservableActor1v1",
    "basic1v1_conflict_active",
    "conflict_side_from_scene",
    "observe_basic1v1_actor",
]
