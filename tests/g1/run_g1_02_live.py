#!/usr/bin/env python3
"""Live CARLA acceptance harness for G1-02.

This is deliberately outside the runtime implementation.  It exercises the
existing ScenarioRuntime against CARLA 0.9.16 and writes reviewable evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402

from runtime import (  # noqa: E402
    ActorSpec,
    RunRegistry,
    RuntimeIdentityFactory,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    TickLeaseUnavailable,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver  # noqa: E402
from runtime.scenario_runtime import RuntimeViolation  # noqa: E402


EXPECTED_VERSION = "0.9.16"
SCENARIO_ID = "Town10HD_Opt.g1-02-live"


class TrafficManagerRecorder:
    def __init__(self, delegate: Any, events: list[dict[str, Any]]) -> None:
        self.delegate = delegate
        self.events = events

    def set_synchronous_mode(self, enabled: bool) -> None:
        self.events.append({"call": "set_synchronous_mode", "value": enabled})
        self.delegate.set_synchronous_mode(enabled)

    def set_random_device_seed(self, seed: int) -> None:
        self.events.append({"call": "set_random_device_seed", "value": seed})
        self.delegate.set_random_device_seed(seed)


class ClientRecorder:
    def __init__(self, delegate: carla.Client, tm_events: list[dict[str, Any]]) -> None:
        self.delegate = delegate
        self.tm_events = tm_events

    def get_world(self) -> carla.World:
        return self.delegate.get_world()

    def load_world(self, map_name: str) -> carla.World:
        return self.delegate.load_world(map_name)

    def get_trafficmanager(self, port: int) -> TrafficManagerRecorder:
        self.tm_events.append({"call": "get_trafficmanager", "port": port})
        return TrafficManagerRecorder(self.delegate.get_trafficmanager(port), self.tm_events)


def identity(attempt_id: int):
    return RuntimeIdentityFactory.create(
        {
            "experiment_id": "g1-02-live-acceptance",
            "scenario_id": SCENARIO_ID,
            "attempt_id": attempt_id,
            "server_epoch": "carla-0.9.16-live-20260712",
            "producer_version": "g1-02-live-v1",
        }
    )


def free_spawn_points(world: carla.World, count: int) -> list[carla.Transform]:
    occupied = [actor.get_location() for actor in world.get_actors().filter("vehicle.*")]
    result: list[carla.Transform] = []
    for transform in world.get_map().get_spawn_points():
        location = transform.location
        if any(location.distance(other) < 5.0 for other in occupied):
            continue
        if any(location.distance(other.location) < 8.0 for other in result):
            continue
        result.append(transform)
        if len(result) == count:
            return result
    raise RuntimeError(f"need {count} free spawn points, found {len(result)}")


def scenario(
    world: carla.World,
    *,
    with_ego: bool,
    with_npc: bool = False,
    with_camera: bool = False,
    sensor_timeout_seconds: float = 2.0,
) -> ScenarioSpec:
    points = free_spawn_points(world, 2 if with_npc else 1) if with_ego else []
    actors: list[ActorSpec] = []
    sensors: list[SensorSpec] = []
    if with_ego:
        actors.append(ActorSpec("ego", "vehicle.tesla.model3", points[0], "ego", 0))
    if with_npc:
        actors.append(ActorSpec("npc", "vehicle.audi.tt", points[1], "npc", 1, True))
    if with_camera:
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.2))
        sensors.append(SensorSpec("front_camera", "sensor.camera.rgb", camera_transform, "ego", 0))
    return ScenarioSpec(
        scenario_id=SCENARIO_ID,
        map_name=world.get_map().name,
        actors=tuple(actors),
        sensors=tuple(sensors),
        traffic_manager_port=8000,
        traffic_manager_seed=20260712,
        sensor_timeout_seconds=sensor_timeout_seconds,
    )


def actor_ids(runtime: ScenarioRuntime) -> dict[str, int]:
    result = {name: int(actor.id) for name, actor in runtime._actors.items()}
    result.update({name: int(sensor.id) for name, sensor in runtime._sensors.items()})
    return result


def assert_cleaned(world: carla.World, ids: dict[str, int]) -> None:
    deadline = time.monotonic() + 2.0
    remaining: dict[str, int] = dict(ids)
    while remaining and time.monotonic() < deadline:
        alive_ids = {int(actor.id) for actor in world.get_actors()}
        remaining = {name: actor_id for name, actor_id in ids.items() if actor_id in alive_ids}
        if remaining:
            time.sleep(0.01)
    if remaining:
        raise AssertionError(f"actors remain after cleanup: {remaining}")


def registry_rows(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT run_id, scenario_id, attempt_id, config_hash, status, "
            "started_wall_time, ended_wall_time, failure_code, actor_manifest_json "
            "FROM scenario_attempts ORDER BY attempt_id"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["actor_manifest"] = json.loads(item.pop("actor_manifest_json"))
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    trace_path = args.evidence_dir / "live_trace.jsonl"

    def trace(phase: str, **values: Any) -> None:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"wall_time": time.time(), "phase": phase, **values}, sort_keys=True) + "\n")

    registry_path = args.evidence_dir / "run_registry.sqlite3"
    lease_path = args.evidence_dir / "tick_lease.lock"
    registry = RunRegistry(registry_path)
    profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]

    resolver = ConnectionResolver(ROOT, expected_version=EXPECTED_VERSION, timeout_seconds=float(os.environ.get("CARLA_TIMEOUT_SECONDS", "10")))
    connection_report = resolver.preflight()
    client, connection_report = resolver.connect(report=connection_report)
    host, port = connection_report.host, connection_report.port
    client_version, server_version = connection_report.client_version, connection_report.server_version
    world = client.get_world()
    initial_actor_ids = {int(actor.id) for actor in world.get_actors()}
    initial_settings = world.get_settings()
    tm_events: list[dict[str, Any]] = []
    recorded_client = ClientRecorder(client, tm_events)
    retained_runtimes: list[ScenarioRuntime] = []
    results: dict[str, Any] = {
        "schema": "safedrive.g1-02.live-evidence.v1",
        "endpoint": f"{host}:{port}",
        "client_version": client_version,
        "server_version": server_version,
        "map": world.get_map().name,
        "profile": profile.name,
        "initial_actor_count": len(initial_actor_ids),
        "checks": {},
    }

    # Empty lifecycle and unique lease use no world tick or actors.
    empty_spec = scenario(world, with_ego=False)
    first = ScenarioRuntime(client=recorded_client, identity=identity(0), profile=profile,
                            registry=registry, lease_path=lease_path)
    second = ScenarioRuntime(client=recorded_client, identity=identity(1), profile=profile,
                             registry=registry, lease_path=lease_path)
    retained_runtimes.extend((first, second))
    trace("empty_start")
    first.start(empty_spec)
    lease_rejected = False
    try:
        second.start(empty_spec)
    except TickLeaseUnavailable:
        lease_rejected = True
    if not lease_rejected:
        raise AssertionError("second runtime acquired tick lease")
    first.complete()
    trace("empty_complete")
    results["checks"]["empty_and_unique_lease"] = {"passed": True, "second_rejected": True}

    # Single ego: real control, tick and cleanup.
    single_spec = scenario(world, with_ego=True)
    single = ScenarioRuntime(client=recorded_client, identity=identity(2), profile=profile,
                             registry=registry, lease_path=lease_path)
    retained_runtimes.append(single)
    trace("single_start")
    single.start(single_spec)
    single_ids = actor_ids(single)
    single_header = single.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
    single.complete()
    assert_cleaned(world, single_ids)
    trace("single_complete", frame=single_header.carla_frame)
    results["checks"]["single_ego"] = {
        "passed": True, "frame": single_header.carla_frame,
        "simulation_time": single_header.simulation_time, "spawned": single_ids,
    }

    # Repeat the same NPC+camera manifest twice to verify deterministic ordering,
    # sensor-frame barrier, TM configuration calls and zero residue.
    manifests: list[list[dict[str, Any]]] = []
    repeat_results: list[dict[str, Any]] = []
    for attempt in (3, 4):
        trace("repeat_prepare", attempt=attempt)
        live_spec = scenario(world, with_ego=True, with_npc=True, with_camera=True)
        runtime = ScenarioRuntime(client=recorded_client, identity=identity(attempt), profile=profile,
                                  registry=registry, lease_path=lease_path)
        retained_runtimes.append(runtime)
        trace("repeat_start", attempt=attempt)
        runtime.start(live_spec)
        trace("repeat_spawned", attempt=attempt, ids=actor_ids(runtime))
        spawned = actor_ids(runtime)
        header = runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
        sensor_frames = dict(runtime._barrier.received) if runtime._barrier is not None else {}
        if sensor_frames != {"front_camera": header.carla_frame}:
            raise AssertionError(f"sensor/frame mismatch header={header.carla_frame} sensors={sensor_frames}")
        manifests.append(ScenarioRuntime.actor_manifest(live_spec))
        runtime.complete()
        trace("repeat_runtime_complete", attempt=attempt)
        assert_cleaned(world, spawned)
        trace("repeat_cleanup_verified", attempt=attempt)
        repeat_results.append(
            {"attempt": attempt, "frame": header.carla_frame, "sensor_frames": sensor_frames,
             "spawned": spawned, "cleaned": True}
        )
    if manifests[0] != manifests[1]:
        raise AssertionError("actor manifests differ across repeated runs")
    results["checks"]["npc_camera_repeat"] = {
        "passed": True, "manifest": manifests[0], "runs": repeat_results,
    }

    # Stop the real sensor before tick to force a live barrier timeout.  tick()
    # must mark the attempt interrupted and clean every actor/sensor.
    timeout_spec = scenario(world, with_ego=True, with_camera=True, sensor_timeout_seconds=0.05)
    timeout_runtime = ScenarioRuntime(client=recorded_client, identity=identity(5), profile=profile,
                                      registry=registry, lease_path=lease_path)
    retained_runtimes.append(timeout_runtime)
    trace("timeout_start")
    timeout_runtime.start(timeout_spec)
    timeout_ids = actor_ids(timeout_runtime)
    timeout_runtime._sensors["front_camera"].stop()
    timeout_error = None
    try:
        timeout_runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
    except RuntimeViolation as exc:
        timeout_error = str(exc)
    if timeout_error is None or "sensor_barrier_timeout" not in timeout_error:
        raise AssertionError(f"expected sensor barrier timeout, got {timeout_error!r}")
    assert_cleaned(world, timeout_ids)
    trace("timeout_cleanup_verified", error=timeout_error)
    results["checks"]["abnormal_cleanup"] = {
        "passed": True, "error": timeout_error, "spawned": timeout_ids, "cleaned": True,
    }

    rows = registry_rows(registry_path)
    statuses = {int(row["attempt_id"]): row["status"] for row in rows}
    expected_statuses = {0: "COMPLETED", 2: "COMPLETED", 3: "COMPLETED", 4: "COMPLETED", 5: "INTERRUPTED"}
    if statuses != expected_statuses:
        raise AssertionError(f"registry status mismatch expected={expected_statuses} actual={statuses}")
    if any(row["ended_wall_time"] is None or not row["config_hash"] for row in rows):
        raise AssertionError("registry missing terminal time or config hash")

    final_actor_ids = {int(actor.id) for actor in world.get_actors()}
    added_ids = sorted(final_actor_ids - initial_actor_ids)
    if added_ids:
        raise AssertionError(f"live acceptance left actor ids behind: {added_ids}")
    final_settings = world.get_settings()
    settings_restored = (
        final_settings.synchronous_mode == initial_settings.synchronous_mode
        and final_settings.fixed_delta_seconds == initial_settings.fixed_delta_seconds
        and final_settings.substepping == initial_settings.substepping
        and final_settings.max_substep_delta_time == initial_settings.max_substep_delta_time
        and final_settings.max_substeps == initial_settings.max_substeps
    )
    if not settings_restored:
        raise AssertionError("CARLA world settings were not restored")
    if not any(event == {"call": "set_random_device_seed", "value": 20260712} for event in tm_events):
        raise AssertionError("Traffic Manager seed call was not observed")
    if not any(event == {"call": "set_synchronous_mode", "value": True} for event in tm_events):
        raise AssertionError("Traffic Manager synchronous mode call was not observed")

    results["checks"]["traffic_manager"] = {"passed": True, "events": tm_events}
    results["checks"]["registry"] = {"passed": True, "rows": rows}
    results["checks"]["global_cleanup"] = {
        "passed": True, "initial_actor_count": len(initial_actor_ids),
        "final_actor_count": len(final_actor_ids), "added_actor_ids": added_ids,
        "world_settings_restored": settings_restored,
    }
    results["completed_wall_time"] = time.time()
    results["status"] = "PASS"
    evidence_path = args.evidence_dir / "live_acceptance.json"
    evidence_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    trace("acceptance_pass")
    print(json.dumps({"status": "PASS", "evidence": str(evidence_path), "registry": str(registry_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
