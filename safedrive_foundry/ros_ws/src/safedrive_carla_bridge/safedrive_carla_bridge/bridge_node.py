"""Publish minimal CARLA snapshots as ROS 2 String messages.

This bridge intentionally never calls world.tick(); deterministic ownership is
reserved for the CARLA driver or the G0-05 synchronization task.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid

import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .sync_contract import FrameEnvelope, FrameLedger, canonical_state_hash


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


class CarlaStatusBridge(Node):
    def __init__(self, client: carla.Client, world: carla.World, host: str, port: int) -> None:
        super().__init__("safedrive_carla_status_bridge")
        topic = os.environ.get("CARLA_STATUS_TOPIC", "/safedrive/carla/status")
        hz = _env_float("CARLA_STATUS_HZ", 10.0)
        if hz <= 0:
            raise RuntimeError("CARLA_STATUS_HZ must be positive")
        self._client = client
        self._world = world
        self._host = host
        self._port = port
        self._episode_id = uuid.uuid4().hex
        self._event_seq = 0
        self._ledger = FrameLedger()
        self._publisher = self.create_publisher(String, topic, 10)
        self._timer = self.create_timer(1.0 / hz, self._publish_snapshot)
        self.get_logger().info(
            f"connected CARLA client={client.get_client_version()} "
            f"server={client.get_server_version()} map={world.get_map().name} "
            f"endpoint={host}:{port} topic={topic}"
        )

    def _publish_snapshot(self) -> None:
        try:
            snapshot = self._world.get_snapshot()
            self._event_seq += 1
            simulation_seconds = float(snapshot.timestamp.elapsed_seconds)
            state_hash = canonical_state_hash(
                {
                    "frame": int(snapshot.frame),
                    "simulation_seconds": simulation_seconds,
                    "map": self._world.get_map().name,
                }
            )
            envelope = FrameEnvelope(
                episode_id=self._episode_id,
                carla_frame=int(snapshot.frame),
                simulation_seconds=simulation_seconds,
                delta_seconds=float(snapshot.timestamp.delta_seconds),
                snapshot_frame=int(snapshot.frame),
                message_frame=int(snapshot.frame),
                clock_frame=int(snapshot.frame),
                clock_seconds=simulation_seconds,
                event_seq=self._event_seq,
                state_hash=state_hash,
            )
            result = self._ledger.ingest(envelope, source="message")
            if not result.accepted:
                # A read-only bridge must never hide duplicate or skipped
                # frames. The tick master remains the only component allowed
                # to advance CARLA.
                self.get_logger().warning(f"snapshot frame rejected [{result.code}]: {result.message}")
                return
            payload = envelope.to_dict()
            payload.update(
                {
                    "schema": "safedrive.carla.status.v2",
                    "map": self._world.get_map().name,
                    "endpoint": f"{self._host}:{self._port}",
                    "publisher_wall_time": time.time(),
                    "publisher_host": socket.gethostname(),
                    "tick_owner": "sdf.g0-05.sync",
                    "clock_topic": "/clock",
                }
            )
            message = String()
            message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            self._publisher.publish(message)
        except Exception as exc:  # surface runtime disconnects without crashing silently
            self.get_logger().error(f"CARLA snapshot read failed: {type(exc).__name__}: {exc}")


def main(args: list[str] | None = None) -> None:
    host = os.environ.get("CARLA_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("CARLA_PORT", "2000"))
    except ValueError as exc:
        raise RuntimeError("CARLA_PORT must be an integer") from exc

    client = carla.Client(host, port)
    client.set_timeout(_env_float("CARLA_TIMEOUT_SECONDS", 10.0))
    try:
        server_version = client.get_server_version()
        client_version = client.get_client_version()
        expected_version = os.environ.get("CARLA_EXPECTED_VERSION", "0.9.16")
        if server_version != expected_version or client_version != expected_version:
            raise RuntimeError(
                f"CARLA version mismatch: expected={expected_version}, "
                f"client={client_version}, server={server_version}"
            )
        world = client.get_world()
    except RuntimeError as exc:
        if "CARLA version mismatch" in str(exc):
            raise
        raise RuntimeError(
            f"CARLA connection failed at {host}:{port}; "
            "check server, port, firewall and Python API version"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"CARLA connection failed at {host}:{port}; "
            "check server, port, firewall and Python API version"
        ) from exc

    rclpy.init(args=args)
    node = CarlaStatusBridge(client, world, host, port)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
