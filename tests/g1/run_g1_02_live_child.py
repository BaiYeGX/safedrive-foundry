#!/usr/bin/env python3
"""One isolated real-CARLA attempt; the parent owns the Registry."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import carla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from runtime import (  # noqa: E402
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402


def transform_from_dict(value: dict[str, float]) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=value["x"], y=value["y"], z=value["z"]),
        carla.Rotation(pitch=value["pitch"], yaw=value["yaw"], roll=value["roll"]),
    )


def read_spec(path: Path) -> ScenarioSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actors = tuple(
        ActorSpec(
            item["name"], item["blueprint"], transform_from_dict(item["transform"]),
            item["role"], item["spawn_order"], item.get("autopilot", False),
        )
        for item in payload["actors"]
    )
    sensors = tuple(
        SensorSpec(
            item["name"], item["blueprint"], transform_from_dict(item["transform"]),
            item["parent"], item["spawn_order"],
        )
        for item in payload["sensors"]
    )
    return ScenarioSpec(
        payload["scenario_id"], payload["map_name"], actors, sensors,
        payload["traffic_manager_port"], payload["traffic_manager_seed"],
        payload["sensor_timeout_seconds"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--crash-after-tick", action="store_true")
    args = parser.parse_args()

    # PYTHONFAULTHANDLER=1 is set by the parent; enable explicitly as well so
    # Python-side thread stacks are preserved in child.stderr on native abort.
    import faulthandler

    faulthandler.enable()
    identity = RunIdentity(**json.loads(args.identity.read_text(encoding="utf-8")))
    spec = read_spec(args.scenario)
    profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=float(os.environ.get("CARLA_TIMEOUT_SECONDS", "10")))
    connection_report = resolver.preflight()
    client, connection_report = resolver.connect(report=connection_report)
    world = client.get_world()
    before = world.get_settings()
    args.state.write_text(
        json.dumps(
            {
                "attempt_id": identity.attempt_id,
                "run_id": identity.run_id,
                "actor_ids": {},
                "sensor_ids": {},
                "world_settings_before": {
                    "synchronous_mode": before.synchronous_mode,
                    "fixed_delta_seconds": before.fixed_delta_seconds,
                    "substepping": before.substepping,
                    "max_substep_delta_time": before.max_substep_delta_time,
                    "max_substeps": before.max_substeps,
                },
                "traffic_manager_port": spec.traffic_manager_port,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runtime = ScenarioRuntime(
        client=client,
        identity=identity,
        profile=profile,
        registry=RunRegistry(args.registry),
        lease_path=args.lease,
        adopt_existing_running=True,
    )
    runtime.start(spec)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    state["actor_ids"] = {name: int(actor.id) for name, actor in runtime._actors.items()}
    state["sensor_ids"] = {name: int(sensor.id) for name, sensor in runtime._sensors.items()}
    args.state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    header = runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
    state["frame"] = header.carla_frame
    state["sensor_frames"] = runtime._barrier.snapshot() if runtime._barrier is not None else {}
    args.state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    if args.crash_after_tick:
        os.abort()
    runtime.complete()
    state["completed_wall_time"] = time.time()
    args.state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
