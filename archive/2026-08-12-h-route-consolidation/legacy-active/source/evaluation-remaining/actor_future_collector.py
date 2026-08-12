"""Oracle-only actor future collector for paired CARLA branches.

The collector piggybacks on the existing paired runner's tick loop. It never
creates a CARLA client and never calls ``world.tick()``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "safedrive.oracle_actor_future_trace.v0"
MANIFEST_SCHEMA = "safedrive.oracle_actor_future_trace_manifest.v0"
TARGET_TIMES_S = tuple(0.25 * (i + 1) for i in range(10))
ROAD_CONTEXT_SCHEMA = "safedrive.observable_road_context.v1"


class ActorFutureCollectorError(RuntimeError):
    pass


def stable_actor_key(
    *,
    scenario_id: str,
    seed_id: str,
    name: str,
    role: str,
    blueprint: str,
) -> str:
    raw = "|".join(
        str(x).strip()
        for x in (scenario_id, seed_id, name, role, blueprint)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def capture_observable_scene_t0(
    *,
    scenario_id: str,
    seed_id: str,
    spawned_actors: Iterable[Any],
    route_waypoints: Sequence[Sequence[float]],
    map_name: str,
    family: str | None = None,
    simulation_time_s: float,
    frame: int,
    road_polylines: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture a runtime-observable proxy scene at branch cold-start.

    This contains current measured state only. No scripted intent, actor future,
    oracle metric, or outcome is serialized.
    """

    ego_payload: dict[str, Any] | None = None
    actors: list[dict[str, Any]] = []
    for spawned in spawned_actors:
        actor = spawned.actor
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        angular = actor.get_angular_velocity()
        acceleration = actor.get_acceleration()
        extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
        row = {
            "name": str(spawned.name),
            "role": str(spawned.role),
            "blueprint": str(spawned.blueprint),
            "track_id_hash": stable_actor_key(
                scenario_id=scenario_id,
                seed_id=seed_id,
                name=str(spawned.name),
                role=str(spawned.role),
                blueprint=str(spawned.blueprint),
            ),
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw_rad": math.radians(float(transform.rotation.yaw)),
            "vx": float(velocity.x),
            "vy": float(velocity.y),
            "vz": float(velocity.z),
            "ax": float(acceleration.x),
            "ay": float(acceleration.y),
            "yaw_rate": math.radians(float(angular.z)),
            "length": 2.0 * float(getattr(extent, "x", 2.25)),
            "width": 2.0 * float(getattr(extent, "y", 0.9)),
            "valid": True,
            "time_since_seen": 0.0,
            "covariance": 0.0,
        }
        if str(spawned.role) == "ego":
            ego_payload = row
        else:
            actors.append(row)
    if ego_payload is None:
        raise ActorFutureCollectorError("observable scene has no ego")
    payload = {
        "schema_version": "safedrive.observable_scene_t0.v0",
        "observable_only": True,
        "source": "carla_sil_actor_proxy_v0",
        "scenario_id": str(scenario_id),
        "seed_id": str(seed_id),
        "map_name": str(map_name),
        "simulation_time_s": float(simulation_time_s),
        "frame": int(frame),
        "ego": ego_payload,
        "actors": sorted(actors, key=lambda x: x["track_id_hash"]),
        "route_waypoints": [
            [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
            for point in route_waypoints
        ],
        "road_polylines": [dict(polyline) for polyline in (road_polylines or ())],
    }
    # Scenario family is an offline grouping label. V4 passes ``None`` so it
    # cannot enter the observable runtime scene or head input; historical
    # V0/V3 callers may retain the audit field.
    if family is not None:
        payload["family"] = str(family)
    return payload


def capture_observable_road_context(
    *,
    world: Any,
    ego: Any,
    route_waypoints: Sequence[Sequence[float]],
    max_points: int = 24,
) -> list[dict[str, Any]]:
    """Build route/center/boundary polylines from current CARLA map only."""
    carla_map = world.get_map()
    location = ego.get_transform().location
    waypoint = carla_map.get_waypoint(location, project_to_road=True)
    route = [
        [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
        for point in route_waypoints
    ][:max_points]
    output: list[dict[str, Any]] = [
        {
            "kind": "route",
            "type_id": 0.0,
            "speed_limit_mps": float(getattr(ego, "get_speed_limit", lambda: 0.0)())
            / 3.6,
            "points": route,
        }
    ]
    if waypoint is None:
        return output
    centers = []
    current = waypoint
    for _ in range(max_points):
        transform = current.transform
        centers.append(
            [
                float(transform.location.x),
                float(transform.location.y),
                float(transform.location.z),
            ]
        )
        following = current.next(1.0)
        if not following:
            break
        current = following[0]
    output.append(
        {
            "kind": "lane_center",
            "type_id": 1.0,
            "speed_limit_mps": float(getattr(ego, "get_speed_limit", lambda: 0.0)())
            / 3.6,
            "points": centers,
        }
    )
    for side, sign, type_id in (("left_boundary", 1.0, 2.0), ("right_boundary", -1.0, 3.0)):
        points = []
        current = waypoint
        for _ in range(max_points):
            transform = current.transform
            yaw = math.radians(float(transform.rotation.yaw))
            offset = 0.5 * float(current.lane_width) * sign
            points.append(
                [
                    float(transform.location.x) - math.sin(yaw) * offset,
                    float(transform.location.y) + math.cos(yaw) * offset,
                    float(transform.location.z),
                ]
            )
            following = current.next(1.0)
            if not following:
                break
            current = following[0]
        output.append(
            {
                "kind": side,
                "type_id": type_id,
                "speed_limit_mps": 0.0,
                "points": points,
            }
        )
    traffic_light = getattr(ego, "get_traffic_light", lambda: None)()
    if traffic_light is not None:
        transform = traffic_light.get_transform()
        output.append(
            {
                "kind": "traffic_control",
                "type_id": 4.0,
                "speed_limit_mps": 0.0,
                "state": str(traffic_light.get_state()),
                "points": [
                    [
                        float(transform.location.x),
                        float(transform.location.y),
                        float(transform.location.z),
                    ]
                ],
            }
        )
    return output


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class ActorFrame:
    time_s: float
    frame: int
    actor_key: str
    name: str
    role: str
    blueprint: str
    x: float
    y: float
    z: float
    yaw_rad: float
    vx: float
    vy: float
    vz: float
    length: float
    width: float
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "oracle_only": True,
            "consumed_by_control": False,
            **self.__dict__,
        }


class ActorFutureCollector:
    def __init__(
        self,
        *,
        scenario_id: str,
        seed_id: str,
        pair_id: str,
        attempt_id: int,
        branch_index: int,
        anchor_artifact_hash: str,
        registry_hash: str,
        model_hash: str,
        guard_hash: str,
        executor_hash: str,
    ) -> None:
        self.scenario_id = str(scenario_id)
        self.seed_id = str(seed_id)
        self.pair_id = str(pair_id)
        self.attempt_id = int(attempt_id)
        self.branch_index = int(branch_index)
        self.anchor_artifact_hash = str(anchor_artifact_hash)
        self.registry_hash = str(registry_hash)
        self.model_hash = str(model_hash)
        self.guard_hash = str(guard_hash)
        self.executor_hash = str(executor_hash)
        self._frames: list[ActorFrame] = []
        self._seen_frame_actor: set[tuple[int, str]] = set()
        self.duplicate_count = 0
        self.out_of_order_count = 0
        self._last_time = -math.inf

    def record(
        self,
        *,
        time_s: float,
        frame: int,
        actors: Iterable[Any],
    ) -> None:
        time_value = float(time_s)
        if time_value + 1e-9 < self._last_time:
            self.out_of_order_count += 1
        self._last_time = max(self._last_time, time_value)
        for spawned in actors:
            if str(getattr(spawned, "role", "")) == "ego":
                continue
            actor = spawned.actor
            tf = actor.get_transform()
            velocity = actor.get_velocity()
            extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
            length = 2.0 * float(getattr(extent, "x", 2.25))
            width = 2.0 * float(getattr(extent, "y", 0.9))
            key = stable_actor_key(
                scenario_id=self.scenario_id,
                seed_id=self.seed_id,
                name=str(spawned.name),
                role=str(spawned.role),
                blueprint=str(spawned.blueprint),
            )
            unique = (int(frame), key)
            if unique in self._seen_frame_actor:
                self.duplicate_count += 1
                continue
            self._seen_frame_actor.add(unique)
            self._frames.append(
                ActorFrame(
                    time_s=time_value,
                    frame=int(frame),
                    actor_key=key,
                    name=str(spawned.name),
                    role=str(spawned.role),
                    blueprint=str(spawned.blueprint),
                    x=float(tf.location.x),
                    y=float(tf.location.y),
                    z=float(tf.location.z),
                    yaw_rad=math.radians(float(tf.rotation.yaw)),
                    vx=float(velocity.x),
                    vy=float(velocity.y),
                    vz=float(velocity.z),
                    length=length,
                    width=width,
                )
            )

    @property
    def frames(self) -> tuple[ActorFrame, ...]:
        return tuple(self._frames)

    def resample(self) -> dict[str, list[dict[str, Any] | None]]:
        by_actor: dict[str, list[ActorFrame]] = {}
        for frame in self._frames:
            by_actor.setdefault(frame.actor_key, []).append(frame)
        output: dict[str, list[dict[str, Any] | None]] = {}
        for actor_key, rows in by_actor.items():
            rows.sort(key=lambda x: (x.time_s, x.frame))
            values: list[dict[str, Any] | None] = []
            for target in TARGET_TIMES_S:
                nearest = min(rows, key=lambda x: abs(x.time_s - target))
                if abs(nearest.time_s - target) > 0.075:
                    values.append(None)
                else:
                    values.append(nearest.to_dict())
            output[actor_key] = values
        return output

    def finalize(self, branch_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
        oracle_dir = Path(branch_dir) / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        trace_path = oracle_dir / "actor_future_trace.jsonl"
        manifest_path = oracle_dir / "actor_future_trace_manifest.json"
        if not overwrite and (trace_path.exists() or manifest_path.exists()):
            raise ActorFutureCollectorError(
                f"refusing to overwrite actor future trace under {oracle_dir}"
            )
        ordered = sorted(self._frames, key=lambda x: (x.time_s, x.frame, x.actor_key))
        trace_path.write_text(
            "".join(_json_line(frame.to_dict()) + "\n" for frame in ordered),
            encoding="utf-8",
        )
        actor_table: dict[str, dict[str, str]] = {}
        for frame in ordered:
            actor_table.setdefault(
                frame.actor_key,
                {
                    "name": frame.name,
                    "role": frame.role,
                    "blueprint": frame.blueprint,
                },
            )
        resampled = self.resample()
        valid_slots = sum(
            value is not None for values in resampled.values() for value in values
        )
        total_slots = max(1, len(resampled) * len(TARGET_TIMES_S))
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "oracle_only": True,
            "consumed_by_control": False,
            "pair_id": self.pair_id,
            "attempt_id": self.attempt_id,
            "branch_index": self.branch_index,
            "scenario_id": self.scenario_id,
            "seed_id": self.seed_id,
            "anchor_artifact_hash": self.anchor_artifact_hash,
            "registry_hash": self.registry_hash,
            "model_hash": self.model_hash,
            "guard_hash": self.guard_hash,
            "executor_hash": self.executor_hash,
            "sampling_rate_hz": 20.0,
            "target_times_s": list(TARGET_TIMES_S),
            "interpolation_rule": "nearest_within_0.075s",
            "actor_identity_table": actor_table,
            "frame_count": len(ordered),
            "duplicate_count": self.duplicate_count,
            "out_of_order_count": self.out_of_order_count,
            "valid_target_slots": valid_slots,
            "total_target_slots": total_slots,
            "target_coverage": valid_slots / total_slots,
            "raw_trace_sha256": _file_sha256(trace_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest


def load_actor_future_trace(
    trace_path: Path,
) -> tuple[ActorFrame, ...]:
    rows: list[ActorFrame] = []
    with Path(trace_path).open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("schema_version") != SCHEMA_VERSION:
                raise ActorFutureCollectorError(
                    f"{trace_path}:{line_number}: unsupported trace schema"
                )
            if value.get("oracle_only") is not True or value.get("consumed_by_control") is not False:
                raise ActorFutureCollectorError(
                    f"{trace_path}:{line_number}: oracle namespace flags invalid"
                )
            fields = {
                name: value[name]
                for name in ActorFrame.__dataclass_fields__
            }
            rows.append(ActorFrame(**fields))
    return tuple(rows)


def resample_actor_frames(
    rows: Sequence[ActorFrame],
    *,
    target_times_s: Sequence[float] = TARGET_TIMES_S,
    tolerance_s: float = 0.075,
) -> Mapping[str, tuple[ActorFrame | None, ...]]:
    by_actor: dict[str, list[ActorFrame]] = {}
    for row in rows:
        by_actor.setdefault(row.actor_key, []).append(row)
    result: dict[str, tuple[ActorFrame | None, ...]] = {}
    for key, frames in by_actor.items():
        ordered = sorted(frames, key=lambda x: (x.time_s, x.frame))
        aligned: list[ActorFrame | None] = []
        for target in target_times_s:
            nearest = min(ordered, key=lambda x: abs(x.time_s - target))
            aligned.append(
                nearest if abs(nearest.time_s - float(target)) <= tolerance_s else None
            )
        result[key] = tuple(aligned)
    return result
