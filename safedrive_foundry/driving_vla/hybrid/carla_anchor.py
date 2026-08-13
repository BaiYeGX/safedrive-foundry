"""CARLA observable-anchor helpers shared by bounded H1/H2 live tools."""

from __future__ import annotations

import io
import math
from typing import Any, Sequence

import numpy as np
from PIL import Image

from classic_stack.control.controller import EgoState
from driving_vla.adapter.policy_adapter import ObservationBundle
from driving_vla.hybrid.contracts import ObservableAnchor
from driving_vla.hybrid.generators import route_revision_sha256
from driving_vla.model.simlingo_runtime import SIMLINGO_CAMERA_XYZ
from safety_kernel.contracts.types import ObservableSnapshot, TrackedObject, TrafficLightObs


def map_basename(name: str) -> str:
    return str(name).split("/")[-1].replace("_Opt", "")


def image_rgb(measurement: Any) -> np.ndarray:
    width, height = int(measurement.width), int(measurement.height)
    bgra = np.frombuffer(measurement.raw_data, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])


def image_png_bytes(measurement: Any) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(image_rgb(measurement), mode="RGB").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def ego_state(actor: Any) -> tuple[EgoState, float]:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
    accel = math.sqrt(float(acceleration.x) ** 2 + float(acceleration.y) ** 2 + float(acceleration.z) ** 2)
    return (
        EgoState(
            x=float(transform.location.x),
            y=float(transform.location.y),
            yaw=math.radians(float(transform.rotation.yaw)),
            v=speed,
            steer=float(actor.get_control().steer) * 0.6,
        ),
        accel,
    )


def observable_actors(world: Any, ego: Any, simulation_time_s: float) -> tuple[TrackedObject, ...]:
    ego_location = ego.get_transform().location
    output: list[TrackedObject] = []
    for actor in world.get_actors():
        if int(actor.id) == int(ego.id):
            continue
        type_id = str(getattr(actor, "type_id", ""))
        if not type_id.startswith(("vehicle.", "walker.")):
            continue
        transform = actor.get_transform()
        if math.hypot(float(transform.location.x - ego_location.x), float(transform.location.y - ego_location.y)) > 50.0:
            continue
        velocity = actor.get_velocity()
        extent = actor.bounding_box.extent
        output.append(
            TrackedObject(
                actor_id=str(actor.id),
                class_name=type_id,
                x=float(transform.location.x),
                y=float(transform.location.y),
                yaw=math.radians(float(transform.rotation.yaw)),
                vx=float(velocity.x),
                vy=float(velocity.y),
                length_m=max(0.1, 2.0 * float(extent.x)),
                width_m=max(0.1, 2.0 * float(extent.y)),
                observed_time_s=simulation_time_s,
            )
        )
    return tuple(sorted(output, key=lambda item: item.actor_id))


def observable_lights(
    world: Any,
    ego: Any,
    simulation_time_s: float,
    route: Sequence[tuple[float, float]],
) -> tuple[TrafficLightObs, ...]:
    ego_location = ego.get_transform().location
    output: list[TrafficLightObs] = []
    for actor in world.get_actors():
        if not str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light"):
            continue
        location = actor.get_transform().location
        distance = math.hypot(float(location.x - ego_location.x), float(location.y - ego_location.y))
        if distance > 50.0 or min(math.hypot(float(location.x) - x, float(location.y) - y) for x, y in route) > 10.0:
            continue
        state = str(actor.get_state()).split(".")[-1].lower()
        output.append(
            TrafficLightObs(
                light_id=str(actor.id),
                state=state,
                distance_m=distance,
                observed_time_s=simulation_time_s,
            )
        )
    return tuple(sorted(output, key=lambda item: item.light_id))


def safety_snapshot(
    runtime: Any,
    header: Any,
    route: tuple[tuple[float, float], ...],
    *,
    frame_id: str,
) -> ObservableSnapshot:
    ego = runtime._actors["ego"]
    state, acceleration = ego_state(ego)
    current_waypoint = runtime.world.get_map().get_waypoint(ego.get_transform().location, project_to_road=True)
    half_width = 1.75 if current_waypoint is None else max(0.5, float(current_waypoint.lane_width) * 0.5)
    simulation_time = float(header.simulation_time)
    return ObservableSnapshot(
        run_id=runtime.identity.run_id,
        frame_id=frame_id,
        scenario_id=runtime.identity.scenario_id,
        simulation_time_s=simulation_time,
        wall_time_s=float(header.wall_time),
        ego_x=state.x,
        ego_y=state.y,
        ego_yaw=state.yaw,
        ego_v=state.v,
        ego_a=acceleration,
        observed_time_s=simulation_time,
        freshness_s=0.0,
        speed_limit_mps=max(0.0, float(ego.get_speed_limit()) / 3.6),
        actors=observable_actors(runtime.world, ego, simulation_time),
        traffic_lights=observable_lights(runtime.world, ego, simulation_time, route),
        corridor_centerline=route,
        corridor_half_width_m=half_width,
        coordinate_frame="map",
    )


def build_anchor(
    runtime: Any,
    header: Any,
    route: tuple[tuple[float, float], ...],
    *,
    ego_history: Sequence[tuple[float, float, float, float]] = (),
) -> ObservableAnchor:
    measurement = runtime.sensor_measurement("front_camera", header.carla_frame)
    if int(measurement.frame) != int(header.carla_frame):
        raise RuntimeError("camera_frame_mismatch")
    state, _ = ego_state(runtime._actors["ego"])
    observation_id = f"{runtime.identity.run_id}:frame-{header.carla_frame}"
    bundle = ObservationBundle(
        run_id=runtime.identity.run_id,
        frame_id=observation_id,
        scenario_id=runtime.identity.scenario_id,
        simulation_time_s=float(header.simulation_time),
        wall_time_s=float(header.wall_time),
        carla_frame=int(header.carla_frame),
        ego_x=state.x,
        ego_y=state.y,
        ego_yaw=state.yaw,
        ego_v=state.v,
        route_xy=route,
        front_rgb=image_rgb(measurement),
        ego_history=tuple(ego_history) or ((state.x, state.y, state.yaw, state.v),),
        meta={
            "official_contract": True,
            "image_layout": "rgb",
            "prompt_mode": "target_point",
            "camera_mount_xyz": SIMLINGO_CAMERA_XYZ,
        },
    )
    snapshot = safety_snapshot(runtime, header, route, frame_id=observation_id)
    return ObservableAnchor(
        observation_id=observation_id,
        bundle=bundle,
        safety_snapshot=snapshot,
        route_revision=route_revision_sha256(route),
        sensor_frames={"front_camera": int(measurement.frame)},
        sensor_timestamps_s={"front_camera": float(measurement.timestamp)},
    )


__all__ = [
    "build_anchor", "ego_state", "image_png_bytes", "image_rgb", "map_basename",
    "observable_actors", "observable_lights", "safety_snapshot",
]
