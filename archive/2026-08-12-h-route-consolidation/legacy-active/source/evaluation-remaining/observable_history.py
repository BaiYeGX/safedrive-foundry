"""Runtime-observable history ring buffer for joint R2/World collection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .actor_future_collector import stable_actor_key

HISTORY_SCHEMA = "safedrive.observable_history.v1"
HISTORY_COUNT = 5
HISTORY_DT_S = 0.20


class ObservableHistoryError(RuntimeError):
    pass


def _actor_state(spawned: Any, *, scenario_id: str, seed_id: str) -> dict[str, Any]:
    actor = spawned.actor
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    angular = actor.get_angular_velocity()
    # Bounding-box dimensions are immutable fixture metadata.  Reading
    # ``actor.bounding_box`` through CARLA's proxy at every history frame is
    # both unnecessary and, on 0.9.16 Windows↔WSL, can block for the entire RPC
    # timeout while ordinary transform/velocity calls remain healthy.
    requested_extent = getattr(
        getattr(spawned, "requested", None), "bounding_box_extent_m", None
    )
    extent = None
    if requested_extent is None:
        extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
    if requested_extent is not None:
        length = 2.0 * float(requested_extent[0])
        width = 2.0 * float(requested_extent[1])
    else:
        length = 2.0 * float(getattr(extent, "x", 2.25))
        width = 2.0 * float(getattr(extent, "y", 0.9))
    return {
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
        "length": length,
        "width": width,
        "valid": True,
        "time_since_seen": 0.0,
        "covariance": 0.0,
    }


@dataclass(frozen=True)
class ObservableFrame:
    simulation_time_s: float
    frame: int
    ego: dict[str, Any]
    actors: tuple[dict[str, Any], ...]


class ObservableHistoryRecorder:
    """Records exact observable state and emits five anchor-aligned samples."""

    def __init__(self, *, scenario_id: str, seed_id: str) -> None:
        self.scenario_id = str(scenario_id)
        self.seed_id = str(seed_id)
        self._frames: deque[ObservableFrame] = deque(maxlen=256)

    def record(
        self,
        *,
        simulation_time_s: float,
        frame: int,
        spawned_actors: Iterable[Any],
    ) -> None:
        ego = None
        actors: list[dict[str, Any]] = []
        for spawned in spawned_actors:
            state = _actor_state(
                spawned, scenario_id=self.scenario_id, seed_id=self.seed_id
            )
            if str(spawned.role) == "ego":
                ego = state
            else:
                actors.append(state)
        if ego is None:
            raise ObservableHistoryError("history frame has no ego")
        if self._frames and float(simulation_time_s) <= self._frames[-1].simulation_time_s:
            raise ObservableHistoryError("history time must be strictly increasing")
        self._frames.append(
            ObservableFrame(
                simulation_time_s=float(simulation_time_s),
                frame=int(frame),
                ego=ego,
                actors=tuple(sorted(actors, key=lambda row: row["track_id_hash"])),
            )
        )

    def finalize(self, *, anchor_time_s: float) -> dict[str, Any]:
        targets = [
            float(anchor_time_s) - HISTORY_DT_S * (HISTORY_COUNT - 1 - index)
            for index in range(HISTORY_COUNT)
        ]
        selected: list[ObservableFrame] = []
        for target in targets:
            candidates = [
                frame for frame in self._frames if frame.simulation_time_s <= target + 1e-6
            ]
            if not candidates:
                raise ObservableHistoryError(
                    f"history coverage missing at t={target:.3f}; "
                    f"frames={len(self._frames)}"
                )
            chosen = min(candidates, key=lambda frame: abs(frame.simulation_time_s - target))
            if abs(chosen.simulation_time_s - target) > 0.075:
                raise ObservableHistoryError(
                    f"history sample misaligned by "
                    f"{abs(chosen.simulation_time_s - target):.3f}s"
                )
            selected.append(chosen)
        if len({frame.frame for frame in selected}) != HISTORY_COUNT:
            raise ObservableHistoryError("history samples must use five distinct frames")
        actor_keys = sorted(
            {
                row["track_id_hash"]
                for frame in selected
                for row in frame.actors
            }
        )
        actor_histories = []
        for key in actor_keys:
            history = []
            prototype = None
            for frame, target in zip(selected, targets):
                current = next(
                    (row for row in frame.actors if row["track_id_hash"] == key), None
                )
                if current is not None:
                    prototype = current
                    history.append(
                        {
                            **current,
                            "dt": float(frame.simulation_time_s - anchor_time_s),
                        }
                    )
                else:
                    history.append({"valid": False, "dt": float(target - anchor_time_s)})
            actor_histories.append(
                {
                    "track_id_hash": key,
                    "name": str((prototype or {}).get("name", key)),
                    "role": str((prototype or {}).get("role", "actor")),
                    "blueprint": str((prototype or {}).get("blueprint", "unknown")),
                    "length": float((prototype or {}).get("length", 4.5)),
                    "width": float((prototype or {}).get("width", 1.8)),
                    "history": history,
                }
            )
        ego_history = [
            {
                **frame.ego,
                "dt": float(frame.simulation_time_s - anchor_time_s),
            }
            for frame in selected
        ]
        return {
            "schema_version": HISTORY_SCHEMA,
            "observable_only": True,
            "history_count": HISTORY_COUNT,
            "history_dt_s": HISTORY_DT_S,
            "anchor_time_s": float(anchor_time_s),
            "frame_ids": [frame.frame for frame in selected],
            "ego_history": ego_history,
            "actors": actor_histories,
        }


def merge_history_into_scene(
    scene: dict[str, Any],
    history: dict[str, Any],
) -> dict[str, Any]:
    if history.get("schema_version") != HISTORY_SCHEMA:
        raise ObservableHistoryError("history schema mismatch")
    merged = dict(scene)
    merged["history_schema_version"] = HISTORY_SCHEMA
    merged["ego_history"] = list(history["ego_history"])
    merged["actor_histories"] = list(history["actors"])
    merged["history_frame_ids"] = list(history["frame_ids"])
    return merged
