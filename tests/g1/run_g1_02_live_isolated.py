#!/usr/bin/env python3
"""Parent supervisor for repeated isolated G1-02 real-CARLA attempts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


SCENARIO_ID = "Town10HD_Opt.g1-02-isolated"


def transform_dict(transform: carla.Transform) -> dict[str, float]:
    return {
        "x": transform.location.x, "y": transform.location.y, "z": transform.location.z,
        "pitch": transform.rotation.pitch, "yaw": transform.rotation.yaw, "roll": transform.rotation.roll,
    }


def identity(attempt: int) -> RunIdentity:
    return RunIdentity(
        experiment_id="g1-02-isolated-acceptance",
        run_id=f"g1-02-isolated-{attempt:03d}",
        scenario_id=SCENARIO_ID,
        attempt_id=attempt,
        server_epoch="carla-0.9.16-live-20260712",
        producer_version="g1-02-isolated-v1",
    )


def free_spawn_points(world: carla.World, count: int) -> list[carla.Transform]:
    occupied = [actor.get_location() for actor in world.get_actors().filter("vehicle.*")]
    result: list[carla.Transform] = []
    for transform in world.get_map().get_spawn_points():
        if any(transform.location.distance(other) < 5.0 for other in occupied):
            continue
        if any(transform.location.distance(other.location) < 8.0 for other in result):
            continue
        result.append(transform)
        if len(result) == count:
            return result
    raise RuntimeError(f"need {count} free spawn points, found {len(result)}")


def make_spec(world: carla.World) -> ScenarioSpec:
    ego, npc = free_spawn_points(world, 2)
    return ScenarioSpec(
        SCENARIO_ID,
        world.get_map().name,
        actors=(
            ActorSpec("ego", "vehicle.tesla.model3", ego, "ego", 0),
            ActorSpec("npc", "vehicle.audi.tt", npc, "npc", 1, True),
        ),
        sensors=(
            SensorSpec("front_camera", "sensor.camera.rgb", carla.Transform(carla.Location(x=1.5, z=2.2)), "ego", 0),
        ),
        traffic_manager_port=8000,
        traffic_manager_seed=20260712,
        sensor_timeout_seconds=2.0,
    )


def state_settings(world: carla.World) -> dict[str, Any]:
    settings = world.get_settings()
    return {
        "synchronous_mode": settings.synchronous_mode,
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "substepping": settings.substepping,
        "max_substep_delta_time": settings.max_substep_delta_time,
        "max_substeps": settings.max_substeps,
    }


def compensate_cleanup(client: carla.Client, state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    world = client.get_world()
    ids = [*state.get("sensor_ids", {}).values(), *state.get("actor_ids", {}).values()]
    active = {int(actor.id): actor for actor in world.get_actors()}
    destroyed: list[int] = []
    for actor_id in ids:
        actor = active.get(int(actor_id))
        if actor is None:
            continue
        with_error = None
        try:
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        except BaseException as exc:  # parent records but continues all cleanup
            with_error = type(exc).__name__
        if with_error is None:
            destroyed.append(int(actor_id))
    with_error = None
    try:
        client.get_trafficmanager(int(state.get("traffic_manager_port", 8000))).set_synchronous_mode(False)
    except BaseException as exc:
        with_error = type(exc).__name__
    before = state.get("world_settings_before")
    if before:
        settings = world.get_settings()
        for field, value in before.items():
            setattr(settings, field, value)
        world.apply_settings(settings)
    deadline = time.monotonic() + 3.0
    target_ids = {int(value) for value in ids}
    while time.monotonic() < deadline:
        active_ids = {int(actor.id) for actor in world.get_actors()}
        residue = sorted(target_ids & active_ids)
        if not residue:
            break
        time.sleep(0.05)
    active_ids = {int(actor.id) for actor in world.get_actors()}
    return {
        "destroyed_ids": destroyed,
        "residue_ids": sorted(target_ids & active_ids),
        "settings_restored": state_settings(world) == before if before else None,
        "traffic_manager_error": with_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--crash-run", type=int, default=-1)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    registry_path = args.evidence_dir / "run_registry.sqlite3"
    lease_path = args.evidence_dir / "tick_lease.lock"
    registry = RunRegistry(registry_path)
    profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=float(os.environ.get("CARLA_TIMEOUT_SECONDS", "10")))
    connection_report = resolver.preflight()
    client, connection_report = resolver.connect(report=connection_report)
    host, port = connection_report.host, connection_report.port
    world = client.get_world()
    evidence: dict[str, Any] = {
        "schema": "safedrive.g1-02.isolated-evidence.v1",
        "endpoint": f"{host}:{port}",
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "map": world.get_map().name,
        "runs": [],
    }
    child_path = ROOT / "tests/g1/run_g1_02_live_child.py"
    for attempt in range(args.runs):
        run_identity = identity(attempt)
        spec = make_spec(world)
        run_dir = args.evidence_dir / f"attempt-{attempt:02d}"
        run_dir.mkdir()
        spec_path = run_dir / "scenario.json"
        identity_path = run_dir / "identity.json"
        state_path = run_dir / "state.json"
        spec_path.write_text(
            json.dumps(
                {
                    "scenario_id": spec.scenario_id, "map_name": spec.map_name,
                    "actors": [
                        {
                            "name": item.name, "blueprint": item.blueprint, "role": item.role,
                            "spawn_order": item.spawn_order, "autopilot": item.autopilot,
                            "transform": transform_dict(item.transform),
                        }
                        for item in spec.actors
                    ],
                    "sensors": [
                        {
                            "name": item.name, "blueprint": item.blueprint, "parent": item.parent,
                            "spawn_order": item.spawn_order, "transform": transform_dict(item.transform),
                        }
                        for item in spec.sensors
                    ],
                    "traffic_manager_port": spec.traffic_manager_port,
                    "traffic_manager_seed": spec.traffic_manager_seed,
                    "sensor_timeout_seconds": spec.sensor_timeout_seconds,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        identity_path.write_text(json.dumps(run_identity.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        state_path.write_text(json.dumps({"world_settings_before": state_settings(world), "traffic_manager_port": spec.traffic_manager_port}, indent=2), encoding="utf-8")
        registry.begin(run_identity, ScenarioRuntime.config_hash(spec, profile), ScenarioRuntime.actor_manifest(spec))
        env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
        command = [
            sys.executable, str(child_path), "--scenario", str(spec_path), "--identity", str(identity_path),
            "--registry", str(registry_path), "--lease", str(lease_path), "--state", str(state_path),
        ]
        if attempt == args.crash_run:
            command.append("--crash-after-tick")
        with (run_dir / "child.stdout").open("w", encoding="utf-8") as stdout, (run_dir / "child.stderr").open("w", encoding="utf-8") as stderr:
            child = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
        row = registry.record(run_identity.run_id)
        run_result: dict[str, Any] = {"attempt": attempt, "returncode": child.returncode, "status_before_parent": row["status"] if row else None}
        if child.returncode != 0:
            registry.mark_crashed(run_identity.run_id, exit_code=child.returncode, detail="isolated child non-zero exit")
            run_result["parent_cleanup"] = compensate_cleanup(client, state_path)
        else:
            run_result["parent_cleanup"] = {"not_needed": True}
            if registry.status(run_identity.run_id) != RunRegistry.COMPLETED:
                registry.mark_crashed(run_identity.run_id, exit_code=child.returncode, detail="child exited without COMPLETED")
                run_result["parent_cleanup"] = compensate_cleanup(client, state_path)
        run_result["status"] = registry.status(run_identity.run_id)
        evidence["runs"].append(run_result)
        expected_status = RunRegistry.CRASHED if attempt == args.crash_run else RunRegistry.COMPLETED
        if run_result["status"] != expected_status or run_result["parent_cleanup"].get("residue_ids"):
            raise RuntimeError(f"isolated attempt failed: {run_result}")
    evidence["status"] = "PASS"
    evidence["registry_rows"] = [registry.record(identity(i).run_id) for i in range(args.runs)]
    (args.evidence_dir / "isolated_acceptance.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "evidence": str(args.evidence_dir / 'isolated_acceptance.json')}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
