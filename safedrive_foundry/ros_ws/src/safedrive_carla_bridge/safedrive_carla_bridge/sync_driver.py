"""CARLA/ROS 2 G0 tick-master node.

This is the only component allowed to call ``world.tick()``.  The existing
``carla_status_bridge`` remains read-only and must not be launched as a second
clock publisher for a synchronized run.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import uuid
from typing import Sequence

import carla
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from .sync_contract import (
    CommandFrames,
    ContractViolation,
    FrameEnvelope,
    FrameLedger,
    SyncConfig,
    TickMasterRegistry,
    apply_sync_settings,
    canonical_state_hash,
    restore_carla_settings,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def config_from_environment() -> SyncConfig:
    config = SyncConfig(
        tick_master=os.environ.get("SDF_TICK_MASTER", "sdf.g0-05.sync"),
        fixed_delta_seconds=_env_float("SDF_FIXED_DELTA_SECONDS", 0.05),
        substepping=os.environ.get("SDF_SUBSTEPPING", "true").lower() in {"1", "true", "yes"},
        max_substep_delta_time=_env_float("SDF_MAX_SUBSTEP_DELTA_TIME", 0.01),
        max_substeps=_env_int("SDF_MAX_SUBSTEPS", 5),
        clock_topic=os.environ.get("SDF_CLOCK_TOPIC", "/clock"),
        snapshot_topic=os.environ.get("SDF_SNAPSHOT_TOPIC", "/safedrive/carla/status"),
        timestamp_tolerance_seconds=_env_float("SDF_TIMESTAMP_TOLERANCE_SECONDS", 1e-6),
    )
    config.assert_valid()
    return config


class CarlaSyncDriver(Node):
    def __init__(self, client: carla.Client, world: carla.World, config: SyncConfig, host: str, port: int) -> None:
        super().__init__("safedrive_carla_sync_driver")
        self._client = client
        self._world = world
        self._config = config
        self._host = host
        self._port = port
        self._registry = TickMasterRegistry(config.tick_master)
        self._registry.claim(config.tick_master)
        self._original_settings = apply_sync_settings(world, config)
        self._ledger = FrameLedger()
        self._episode_id = uuid.uuid4().hex
        self._event_seq = 0
        self._closed = False
        self._status_publisher = self.create_publisher(String, config.snapshot_topic, 10)
        self._clock_publisher = self.create_publisher(Clock, config.clock_topic, 10)
        self.get_logger().info(
            f"tick master={config.tick_master} fixed_delta={config.fixed_delta_seconds} "
            f"substep={config.max_substep_delta_time}x{config.max_substeps} "
            f"CARLA={host}:{port} clock={config.clock_topic} status={config.snapshot_topic}"
        )

    def tick_once(self) -> dict[str, object]:
        self._event_seq += 1
        frame = int(self._world.tick())
        snapshot = self._world.get_snapshot()
        if int(snapshot.frame) != frame:
            raise ContractViolation(
                "snapshot_frame_mismatch",
                f"world.tick returned {frame}, snapshot is {snapshot.frame}",
            )
        simulation_seconds = float(snapshot.timestamp.elapsed_seconds)
        state_hash = canonical_state_hash(
            {
                "frame": frame,
                "simulation_seconds": simulation_seconds,
                "map": self._world.get_map().name,
            }
        )
        envelope = FrameEnvelope(
            episode_id=self._episode_id,
            carla_frame=frame,
            simulation_seconds=simulation_seconds,
            delta_seconds=float(snapshot.timestamp.delta_seconds),
            snapshot_frame=int(snapshot.frame),
            message_frame=frame,
            clock_frame=frame,
            clock_seconds=simulation_seconds,
            event_seq=self._event_seq,
            state_hash=state_hash,
            command_frames=CommandFrames(frame, frame, frame),
            timestamp_tolerance_seconds=self._config.timestamp_tolerance_seconds,
        )
        accepted = self._ledger.ingest(envelope, source="tick")
        if not accepted.accepted:
            raise ContractViolation(accepted.code, accepted.message)

        seconds = int(simulation_seconds)
        nanoseconds = int(round((simulation_seconds - seconds) * 1_000_000_000))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        clock_message = Clock()
        clock_message.clock.sec = seconds
        clock_message.clock.nanosec = nanoseconds

        payload = envelope.to_dict()
        payload.update(
            {
                "schema": "safedrive.carla.sync.v1",
                "tick_master": self._config.tick_master,
                "clock_topic": self._config.clock_topic,
                "publisher_wall_time": time.time(),
                "publisher_host": socket.gethostname(),
                "map": self._world.get_map().name,
                "endpoint": f"{self._host}:{self._port}",
            }
        )
        status_message = String()
        status_message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        # Keep this order explicit: one tick -> one snapshot -> one /clock and
        # status record, all derived from the same FrameEnvelope.
        self._clock_publisher.publish(clock_message)
        self._status_publisher.publish(status_message)
        return payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            restore_carla_settings(self._world, self._original_settings)
        finally:
            self._registry.release(self._config.tick_master)

    def destroy_node(self) -> bool:
        self.close()
        return super().destroy_node()


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SafeDrive G0-05 CARLA tick master")
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    parser.add_argument("--timeout", type=float, default=_env_float("CARLA_TIMEOUT_SECONDS", 10.0))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("CARLA_SYNC_STEPS", "0")), help="0 means run until shutdown")
    parser.add_argument("--expected-version", default=os.environ.get("CARLA_EXPECTED_VERSION", "0.9.16"))
    parsed = parser.parse_args(args)
    config = config_from_environment()

    client = carla.Client(parsed.host, parsed.port)
    client.set_timeout(parsed.timeout)
    client_version = str(client.get_client_version())
    server_version = str(client.get_server_version())
    if client_version != parsed.expected_version or server_version != parsed.expected_version:
        raise RuntimeError(
            f"CARLA version mismatch: expected={parsed.expected_version}, "
            f"client={client_version}, server={server_version}"
        )
    world = client.get_world()

    rclpy.init(args=None)
    node: CarlaSyncDriver | None = None
    try:
        node = CarlaSyncDriver(client, world, config, parsed.host, parsed.port)
        completed = 0
        while rclpy.ok() and (parsed.steps <= 0 or completed < parsed.steps):
            node.tick_once()
            completed += 1
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
