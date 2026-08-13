#!/usr/bin/env python3
"""Materialize and collect H2 paired outcomes without importing the offline Oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
import torch  # noqa: E402

from classic_stack.control.controller import ControlLoop  # noqa: E402
from data_pipeline.h2.carla_scenarios import (  # noqa: E402
    PhysicalScenario,
    materialize_physical_scenario,
    transform_from_dict,
)
from data_pipeline.h2.contracts import (  # noqa: E402
    PairRecord,
    PairTerminalStatus,
    compare_reset_signatures,
    stable_sha256,
)
from data_pipeline.h2.config import H2_CONFIG_SHA256, config_identity  # noqa: E402
from data_pipeline.h2.gpu import GPUMemorySampler  # noqa: E402
from data_pipeline.h2.live_contract import (  # noqa: E402
    COLLECTOR_VERSION,
    candidate_snapshot,
    make_branch_outcome,
    reset_signature,
    route_projection,
    trajectory_sha256,
)
from data_pipeline.h2.matrix import FIXED_MATRIX, MATRIX_SHA256, PILOT_MATRIX  # noqa: E402
from data_pipeline.h2.store import PairedOutcomeStore, file_sha256  # noqa: E402
from driving_vla.hybrid import (  # noqa: E402
    ClassicExpertGenerator,
    HybridCandidateSet,
    NominalVLAGenerator,
    SelectionSpace,
    generate_hybrid_set,
    simlingo_generator_hash,
)
from driving_vla.hybrid.carla_anchor import (  # noqa: E402
    build_anchor,
    ego_state,
    image_png_bytes,
    map_basename,
    safety_snapshot,
)
from driving_vla.hybrid.guard import CandidateGuard  # noqa: E402
from driving_vla.hybrid.router import FrozenH1Router  # noqa: E402
from driving_vla.model.nominal_policy import NominalVLAPolicy  # noqa: E402
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
)
from driving_vla.runtime.safety_control_bind import (  # noqa: E402
    apply_safety_control,
    resolve_executable_candidate,
    safety_points_to_ctrl,
)
from runtime import (  # noqa: E402
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver, READY  # noqa: E402
from safety_kernel.contracts.types import ComponentAvailability, PolicyCandidateSet  # noqa: E402
from safety_kernel.kernel import SafetyKernel  # noqa: E402


DATA_ROOT = ROOT / "generated" / "h2" / "paired-outcomes"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h2"
PROFILE = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]


def _git_text(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return (result.stdout or "").strip()


def _worktree_identity() -> dict[str, Any]:
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True, check=False).stdout
    raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout
    rows: list[dict[str, Any]] = []
    for encoded in sorted(item for item in raw.split(b"\0") if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if path.is_file():
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return {
        "commit": _git_text("rev-parse", "HEAD"),
        "branch": _git_text("branch", "--show-current"),
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_manifest_sha256": stable_sha256(rows),
        "untracked_files": rows,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _connection(map_name: str) -> tuple[Any, Any, Any]:
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=10.0)
    report = resolver.preflight()
    if report.status != READY:
        raise RuntimeError(f"CARLA_NOT_READY:{report.error_code}:{report.error_message}")
    if map_basename(str(report.map)) != map_name:
        raise RuntimeError(f"MAP_MISMATCH:{report.map}!={map_name}")
    client, _ = resolver.connect(report=report)
    return client, client.get_world(), report


def _require_clean_scene(world: Any) -> None:
    residue = [
        int(actor.id)
        for actor in world.get_actors()
        if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker."))
        and bool(getattr(actor, "is_alive", True))
    ]
    if residue:
        raise RuntimeError(f"NEEDS_CLEAN_SCENE:{residue}")


def _manifest_part_path(store: PairedOutcomeStore, map_name: str) -> Path:
    return store.root / "scenario-parts" / f"{map_name}.json"


def materialize_map(dataset_id: str, map_name: str) -> dict[str, Any]:
    store = PairedOutcomeStore(DATA_ROOT, dataset_id)
    client, world, report = _connection(map_name)
    del client
    _require_clean_scene(world)
    rows = [materialize_physical_scenario(world, entry) for entry in FIXED_MATRIX if entry.scenario.map_name == map_name]
    if len(rows) != 40:
        raise RuntimeError(f"MATERIALIZATION_COUNT:{len(rows)}")
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h2.physical_manifest_part.v1",
        "dataset_id": dataset_id,
        "map_name": map_name,
        "matrix_sha256": MATRIX_SHA256,
        "config_sha256": H2_CONFIG_SHA256,
        "config": config_identity(),
        "worktree": _worktree_identity(),
        "connection": report.to_dict(),
        "rows": [row.to_dict() for row in rows],
    }
    payload["part_sha256"] = stable_sha256(payload)
    path = _manifest_part_path(store, map_name)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"physical manifest part changed: {path}")
    else:
        _atomic_json(path, payload)
    store.write_manifest()
    return {"ok": True, "path": str(path), "rows": len(rows), "part_sha256": payload["part_sha256"]}


def freeze_manifest(dataset_id: str) -> dict[str, Any]:
    store = PairedOutcomeStore(DATA_ROOT, dataset_id)
    parts = []
    rows: list[dict[str, Any]] = []
    for map_name in ("Town01", "Town03", "Town05"):
        part = json.loads(_manifest_part_path(store, map_name).read_text(encoding="utf-8"))
        expected = part.pop("part_sha256")
        if (
            stable_sha256(part) != expected
            or part["matrix_sha256"] != MATRIX_SHA256
            or part.get("config_sha256") != H2_CONFIG_SHA256
        ):
            raise RuntimeError(f"MANIFEST_PART_HASH:{map_name}")
        part["part_sha256"] = expected
        parts.append(part)
        rows.extend(part["rows"])
    if len(rows) != 120 or len({row["pair_id"] for row in rows}) != 120:
        raise RuntimeError("PHYSICAL_MANIFEST_NOT_120_UNIQUE")
    worktrees = {
        (
            part["worktree"]["commit"],
            part["worktree"]["worktree_diff_sha256"],
            part["worktree"]["untracked_manifest_sha256"],
        )
        for part in parts
    }
    if len(worktrees) != 1:
        raise RuntimeError("MATERIALIZATION_WORKTREE_CHANGED")
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h2.physical_manifest.v1",
        "dataset_id": dataset_id,
        "matrix_sha256": MATRIX_SHA256,
        "config_sha256": H2_CONFIG_SHA256,
        "config": config_identity(),
        "worktree": parts[0]["worktree"],
        "part_sha256": {part["map_name"]: part["part_sha256"] for part in parts},
        "rows": sorted(rows, key=lambda row: int(row["matrix_index"])),
    }
    payload["physical_manifest_sha256"] = stable_sha256(payload)
    path = store.root / "scenario_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise FileExistsError("frozen physical manifest already exists with different content")
    _atomic_json(path, payload)
    store._atomic_parquet(  # one frozen, reviewable Parquet table
        store.root / "scenario_manifest.parquet",
        [
            {
                "pair_id": row["pair_id"],
                "physical_sha256": row["physical_sha256"],
                "record_json": json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            }
            for row in payload["rows"]
        ],
    )
    store.write_manifest()
    return {
        "ok": True,
        "rows": 120,
        "path": str(path),
        "physical_manifest_sha256": payload["physical_manifest_sha256"],
    }


def restart_smoke(dataset_id: str, map_name: str) -> dict[str, Any]:
    """One bounded Runtime tick after a cold restart; no model or pair collection."""

    store = PairedOutcomeStore(DATA_ROOT, dataset_id)
    client, world, report = _connection(map_name)
    _require_clean_scene(world)
    physical = materialize_physical_scenario(
        world,
        next(
            row for row in FIXED_MATRIX
            if row.scenario.map_name == map_name
            and row.scenario.family == "free_flow"
            and row.scenario.seed == 0
            and row.scenario.weather == "ClearNoon"
        ),
    )
    evidence_dir = EVIDENCE_ROOT / dataset_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(evidence_dir / "restart_smoke_registry.sqlite3")
    run_id = f"{map_name}-restart-smoke-{time.time_ns()}"
    runtime = _runtime(client, physical, run_id=run_id, phase="restart-smoke", sensors=(), registry=registry)
    try:
        header = runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
        runtime.complete()
    except BaseException:
        runtime.abort("restart_smoke_failure")
        raise
    cleanup = registry.record(run_id)
    if cleanup is None or cleanup["status"] != "COMPLETED":
        raise RuntimeError("RESTART_SMOKE_CLEANUP_INCOMPLETE")
    _require_clean_scene(world)
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h2.restart_smoke.v1",
        "dataset_id": dataset_id,
        "map_name": map_name,
        "connection": report.to_dict(),
        "config": config_identity(),
        "config_sha256": H2_CONFIG_SHA256,
        "carla_frame": int(header.carla_frame),
        "simulation_time_s": float(header.simulation_time),
        "physical_scenario_sha256": physical.physical_sha256,
        "cleanup": cleanup,
        "worktree": _worktree_identity(),
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    path = evidence_dir / f"restart-smoke-{map_name}.json"
    _atomic_json(path, payload)
    store.write_manifest()
    return {"ok": True, "map": map_name, "evidence": str(path), "evidence_sha256": payload["evidence_sha256"]}


def _load_physical_manifest(store: PairedOutcomeStore) -> tuple[dict[str, PhysicalScenario], dict[str, Any]]:
    path = store.root / "scenario_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("physical_manifest_sha256")
    if stable_sha256(payload) != expected:
        raise RuntimeError("PHYSICAL_MANIFEST_HASH_MISMATCH")
    payload["physical_manifest_sha256"] = expected
    current = _worktree_identity()
    frozen = payload["worktree"]
    for name in ("commit", "worktree_diff_sha256", "untracked_manifest_sha256"):
        if current[name] != frozen[name]:
            raise RuntimeError(f"WORKTREE_CHANGED_SINCE_MATERIALIZATION:{name}")
    if payload.get("config_sha256") != H2_CONFIG_SHA256:
        raise RuntimeError("H2_CONFIG_CHANGED_SINCE_MATERIALIZATION")
    rows = {str(row["pair_id"]): PhysicalScenario.from_dict(row) for row in payload["rows"]}
    return rows, payload


class _FrozenTrafficLights:
    def __init__(self, world: Any, scenario: PhysicalScenario) -> None:
        self.world = world
        self.scenario = scenario
        self.saved: list[tuple[Any, Any, bool]] = []

    def __enter__(self) -> "_FrozenTrafficLights":
        lights = [
            actor for actor in self.world.get_actors()
            if str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light")
        ]
        for light in lights:
            is_frozen = bool(light.is_frozen()) if hasattr(light, "is_frozen") else False
            self.saved.append((light, light.get_state(), is_frozen))
            light.freeze(True)
        if self.scenario.red_light is not None:
            x, y = float(self.scenario.red_light["trigger_x"]), float(self.scenario.red_light["trigger_y"])
            target = min(
                lights,
                key=lambda light: math.hypot(float(light.get_transform().location.x) - x, float(light.get_transform().location.y) - y),
            )
            target.set_state(carla.TrafficLightState.Red)
        return self

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "role": f"light-{index}",
                "x": float(light.get_transform().location.x),
                "y": float(light.get_transform().location.y),
                "state": str(light.get_state()).split(".")[-1],
            }
            for index, (light, _, _) in enumerate(sorted(self.saved, key=lambda item: int(item[0].id)))
        ]

    def __exit__(self, *_: object) -> None:
        for light, state, was_frozen in self.saved:
            light.set_state(state)
            light.freeze(was_frozen)


def _actor_specs(scenario: PhysicalScenario) -> tuple[ActorSpec, ...]:
    actors = [
        ActorSpec("ego", "vehicle.tesla.model3", transform_from_dict(scenario.ego_transform), "ego", 0, False)
    ]
    for index, npc in enumerate(scenario.npc_actors, start=1):
        actors.append(
            ActorSpec(
                str(npc["role"]), str(npc["blueprint"]), transform_from_dict(npc["transform"]), "npc", index, False
            )
        )
    return tuple(actors)


def _camera_spec() -> SensorSpec:
    width, height = SIMLINGO_CAMERA_NATIVE_SIZE
    return SensorSpec(
        "front_camera",
        "sensor.camera.rgb",
        carla.Transform(carla.Location(x=SIMLINGO_CAMERA_XYZ[0], y=SIMLINGO_CAMERA_XYZ[1], z=SIMLINGO_CAMERA_XYZ[2])),
        "ego",
        0,
        {"image_size_x": str(width), "image_size_y": str(height), "fov": str(SIMLINGO_CAMERA_FOV_DEG)},
    )


def _event_specs() -> tuple[SensorSpec, ...]:
    identity = carla.Transform()
    return (
        SensorSpec("collision", "sensor.other.collision", identity, "ego", 0, delivery="event"),
        SensorSpec("lane_invasion", "sensor.other.lane_invasion", identity, "ego", 1, delivery="event"),
    )


def _npc_controls(scenario: PhysicalScenario) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    if "lead_control" in scenario.script:
        command = scenario.script["lead_control"]
        controls["lead"] = carla.VehicleControl(
            throttle=float(command["throttle"]), brake=float(command["brake"]), steer=float(command["steer"])
        )
    if "cutter_control" in scenario.script:
        command = scenario.script["cutter_control"]
        controls["cutter"] = carla.VehicleControl(
            throttle=float(command["throttle"]), brake=float(command["brake"]), steer=float(command["steer"])
        )
    return controls


def _runtime(
    client: Any,
    scenario: PhysicalScenario,
    *,
    run_id: str,
    phase: str,
    sensors: Sequence[SensorSpec],
    registry: RunRegistry,
) -> ScenarioRuntime:
    identity = RunIdentity(
        experiment_id="h2-paired-outcomes",
        run_id=run_id,
        scenario_id=f"h2-{scenario.pair_id}-{phase}",
        attempt_id=0,
        server_epoch="carla-0.9.16-h2",
        producer_version=COLLECTOR_VERSION,
    )
    spec = ScenarioSpec(
        scenario_id=identity.scenario_id,
        map_name=scenario.scenario.map_name,
        actors=_actor_specs(scenario),
        sensors=tuple(sensors),
        traffic_manager_seed=scenario.scenario.seed,
        sensor_timeout_seconds=10.0,
        weather=scenario.weather,
    )
    runtime = ScenarioRuntime(
        client=client,
        identity=identity,
        profile=PROFILE,
        registry=registry,
        lease_path=ROOT / ".runtime" / "tick-lease.lock",
        owner="sdf.h2.paired_collector",
    )
    runtime.start(spec)
    return runtime


def _actors_by_role(runtime: ScenarioRuntime) -> dict[str, Any]:
    return {name: actor for name, actor in runtime._actors.items()}


def _actual_weather(runtime: ScenarioRuntime, scenario: PhysicalScenario) -> dict[str, float]:
    weather = runtime.world.get_weather()
    return {name: float(getattr(weather, name)) for name in sorted(scenario.weather)}


def _pre_roll(
    runtime: ScenarioRuntime,
    scenario: PhysicalScenario,
    *,
    store: PairedOutcomeStore | None = None,
) -> tuple[Any, list[dict[str, Any]], list[tuple[float, float, float, float]]]:
    history: list[dict[str, Any]] = []
    ego_history: list[tuple[float, float, float, float]] = []
    header = None
    for index in range(20):
        controls = {"ego": carla.VehicleControl(throttle=0.0, brake=1.0), **_npc_controls(scenario)}
        header = runtime.tick_controls(controls)
        state, acceleration = ego_state(runtime._actors["ego"])
        ego_history.append((state.x, state.y, state.yaw, state.v))
        row: dict[str, Any] = {
            "index": index,
            "carla_frame": int(header.carla_frame),
            "simulation_time_s": float(header.simulation_time),
            "ego_x": state.x,
            "ego_y": state.y,
            "ego_yaw": state.yaw,
            "ego_speed_mps": state.v,
            "ego_acceleration_mps2": acceleration,
        }
        if store is not None:
            measurement = runtime.sensor_measurement("front_camera", header.carla_frame)
            image_path, image_sha = store.write_image(image_png_bytes(measurement))
            row.update({"image_path": image_path, "image_sha256": image_sha})
        history.append(row)
    assert header is not None
    return header, history, ego_history


def _event_rows(runtime: ScenarioRuntime, start_frame: int, end_frame: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    collision_count = 0
    for measurement in runtime.sensor_events("collision", since_frame=start_frame, through_frame=end_frame):
        collision_count += 1
        impulse = measurement.normal_impulse
        rows.append(
            {
                "event_type": "collision",
                "frame": int(measurement.frame),
                "other_actor_id": int(getattr(measurement.other_actor, "id", -1)),
                "impulse_x": float(impulse.x), "impulse_y": float(impulse.y), "impulse_z": float(impulse.z),
            }
        )
    for measurement in runtime.sensor_events("lane_invasion", since_frame=start_frame, through_frame=end_frame):
        rows.append(
            {
                "event_type": "lane_invasion",
                "frame": int(measurement.frame),
                "markings": ",".join(sorted(str(marking.type) for marking in measurement.crossed_lane_markings)),
            }
        )
    return rows, collision_count


def _capture(
    client: Any,
    scenario: PhysicalScenario,
    store: PairedOutcomeStore,
    registry: RunRegistry,
    lights: _FrozenTrafficLights,
    classic: ClassicExpertGenerator,
    vla: NominalVLAGenerator,
    policy: NominalVLAPolicy,
) -> dict[str, Any]:
    run_id = f"{scenario.pair_id}-capture-{time.time_ns()}"
    runtime = _runtime(client, scenario, run_id=run_id, phase="capture", sensors=(_camera_spec(),), registry=registry)
    before_forward = policy.forward_count
    try:
        header, history, ego_history = _pre_roll(runtime, scenario, store=store)
        anchor = build_anchor(runtime, header, scenario.route, ego_history=ego_history)
        captured_reset = reset_signature(
            _actors_by_role(runtime), route=scenario.route, weather=_actual_weather(runtime, scenario),
            lights=lights.snapshot(), script=scenario.script,
        )
        generated = generate_hybrid_set(anchor, classic, vla)
        guarded = CandidateGuard().evaluate(generated)
        routing = FrozenH1Router().route(guarded)
        swapped = HybridCandidateSet(
            anchor=guarded.anchor,
            candidates=tuple(reversed(guarded.candidates)),
            attempts=tuple(reversed(guarded.attempts)),
        )
        swapped_route = FrozenH1Router().route(swapped)
        swap_invariant = (
            routing.pass_candidate_ids == swapped_route.pass_candidate_ids
            and routing.selection_space == swapped_route.selection_space
            and routing.difference == swapped_route.difference
        )
        runtime.complete()
        cleanup = registry.record(run_id)
        return {
            "anchor": anchor,
            "history": history,
            "guarded": guarded,
            "routing": routing,
            "capture_reset": captured_reset,
            "vla_forward_count": policy.forward_count - before_forward,
            "swap_invariant": swap_invariant,
            "cleanup": cleanup,
        }
    except BaseException:
        runtime.abort("capture_failure")
        raise


def _branch(
    client: Any,
    scenario: PhysicalScenario,
    store: PairedOutcomeStore,
    registry: RunRegistry,
    lights: _FrozenTrafficLights,
    captured_reset: Any,
    candidate: Any,
    gpu_sampler: GPUMemorySampler | None = None,
) -> tuple[Any, dict[str, str]]:
    run_id = f"{scenario.pair_id}-{candidate.candidate_id}-{time.time_ns()}"
    started = time.perf_counter()
    runtime = _runtime(client, scenario, run_id=run_id, phase="branch", sensors=_event_specs(), registry=registry)
    timeline: list[dict[str, Any]] = []
    actor_future: list[dict[str, Any]] = []
    errors: list[str] = []
    pre_hash = trajectory_sha256(candidate.points)
    reset_cmp = compare_reset_signatures(captured_reset, captured_reset)
    decision = None
    applied = None
    first_frame = -1
    last_frame = -1
    corridor_half_width = 2.0
    try:
        header, _, _ = _pre_roll(runtime, scenario)
        branch_reset = reset_signature(
            _actors_by_role(runtime), route=scenario.route, weather=_actual_weather(runtime, scenario),
            lights=lights.snapshot(), script=scenario.script,
        )
        reset_cmp = compare_reset_signatures(captured_reset, branch_reset)
        frame_id = f"{runtime.identity.run_id}:frame-{header.carla_frame}"
        snapshot = safety_snapshot(runtime, header, scenario.route, frame_id=frame_id)
        bound = replace(
            candidate,
            generated_time_s=float(header.simulation_time),
            valid_until_s=float(header.simulation_time) + 0.25,
            dynamics_meta={
                **dict(candidate.dynamics_meta),
                "h2_identity_binding": {
                    "run_id": runtime.identity.run_id,
                    "frame_id": frame_id,
                    "simulation_time_s": float(header.simulation_time),
                    "points_unchanged": True,
                },
            },
        )
        post_hash = trajectory_sha256(bound.points)
        candidate_set = PolicyCandidateSet(
            run_id=runtime.identity.run_id,
            frame_id=frame_id,
            scenario_id=runtime.identity.scenario_id,
            model_id="h2-forced-single-candidate@1.0.0",
            carla_frame=int(header.carla_frame),
            simulation_time_s=float(header.simulation_time),
            wall_time_s=float(header.wall_time),
            candidates=(bound,),
            schema_version="safedrive.safety.contracts.v1",
            coordinate_frame="map",
        )
        source = str(getattr(candidate.source, "value", candidate.source))
        availability = ComponentAvailability(
            classic=source == "classic", vla=source.startswith("vla"), world=False, safety=True,
            detail={"world": "H2_NOT_IMPLEMENTED", "forced_candidate_id": candidate.candidate_id},
        )
        safety = SafetyKernel()
        tick_result = safety.tick(snapshot, candidate_set, now_s=float(header.simulation_time), availability=availability)
        decision = tick_result.decision
        control_loop = ControlLoop()
        initial_ego, _ = ego_state(runtime._actors["ego"])
        applied = apply_safety_control(decision, candidate_set, control_loop, initial_ego, float(header.simulation_time))
        executable, resolve_notes = resolve_executable_candidate(decision, candidate_set)
        corridor_half_width = float(snapshot.corridor_half_width_m)
        if not reset_cmp.comparable:
            errors.extend(reset_cmp.reasons)
        elif not applied.is_track_approved or executable is None:
            errors.append("safety_execution_unresolvable:" + ",".join(resolve_notes))
        elif executable.source != candidate.source:
            errors.append("cross_source_fallback")
        else:
            route_start, _ = route_projection(initial_ego.x, initial_ego.y, scenario.route)
            previous_yaw = initial_ego.yaw
            current_header = header
            for tick in range(50):
                now_s = float(current_header.simulation_time)
                if tick == 0:
                    command = applied
                else:
                    if tick % 4 == 0:
                        control_loop.set_trajectory(safety_points_to_ctrl(executable.points), now_s)
                    current_ego, _ = ego_state(runtime._actors["ego"])
                    command = control_loop.step(current_ego, now_s)
                controls = {
                    "ego": carla.VehicleControl(
                        throttle=float(command.throttle), brake=float(command.brake), steer=float(command.steer),
                        reverse=bool(getattr(command, "reverse", False)),
                    ),
                    **_npc_controls(scenario),
                }
                tick_started = time.perf_counter()
                current_header = runtime.tick_controls(controls)
                tick_wall_ms = (time.perf_counter() - tick_started) * 1000.0
                state, acceleration = ego_state(runtime._actors["ego"])
                progress, corridor_distance = route_projection(state.x, state.y, scenario.route)
                yaw_rate = ((state.yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi) / 0.05
                previous_yaw = state.yaw
                timeline.append(
                    {
                        "tick": tick,
                        "carla_frame": int(current_header.carla_frame),
                        "simulation_time_s": float(current_header.simulation_time),
                        "x": state.x, "y": state.y, "yaw": state.yaw, "speed_mps": state.v,
                        "acceleration_mps2": acceleration,
                        "lateral_acceleration_mps2": state.v * yaw_rate,
                        "route_progress_m": progress - route_start,
                        "corridor_distance_m": corridor_distance,
                        "throttle": float(command.throttle), "brake": float(command.brake), "steer": float(command.steer),
                        "control_mode": str(getattr(command, "applied_mode", getattr(command, "mode", ""))),
                        "deadline_miss": bool(getattr(command, "deadline_miss", False)),
                        "tick_wall_ms": tick_wall_ms,
                    }
                )
                for role, actor in sorted(runtime._actors.items()):
                    if role == "ego":
                        continue
                    actor_state, _ = ego_state(actor)
                    actor_future.append(
                        {
                            "tick": tick, "carla_frame": int(current_header.carla_frame), "role": role,
                            "actor_id": int(actor.id), "x": actor_state.x, "y": actor_state.y,
                            "yaw": actor_state.yaw, "speed_mps": actor_state.v,
                        }
                    )
            first_frame = int(timeline[0]["carla_frame"])
            last_frame = int(timeline[-1]["carla_frame"])
        event_rows, collision_count = _event_rows(runtime, first_frame, last_frame) if first_frame >= 0 else ([], 0)
        runtime.complete()
        cleanup = registry.record(run_id)
        cleanup_complete = bool(cleanup and cleanup["status"] == "COMPLETED")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
        runtime.abort(type(exc).__name__)
        cleanup = registry.record(run_id)
        cleanup_complete = False
        event_rows, collision_count = [], 0
        post_hash = pre_hash

    timeline_path = actor_path = event_path = ""
    artifacts: dict[str, str] = {}
    if timeline:
        timeline_path, artifacts["timeline"] = store.write_timeline(scenario.pair_id, candidate.candidate_id, timeline)
        actor_payload = actor_future or [{"tick": -1, "carla_frame": -1, "role": "none", "actor_id": -1, "x": 0.0, "y": 0.0, "yaw": 0.0, "speed_mps": 0.0}]
        actor_path, artifacts["actor_future"] = store.write_actor_future(
            scenario.pair_id, candidate.candidate_id, actor_payload
        )
        event_path, artifacts["events"] = store.write_event_rows(scenario.pair_id, candidate.candidate_id, event_rows)
    final_progress = float(timeline[-1]["route_progress_m"]) if timeline else 0.0
    off_duration = 0.05 * sum(float(row["corridor_distance_m"]) > corridor_half_width for row in timeline)
    red_violation = bool(
        scenario.red_light is not None
        and timeline
        and final_progress > float(scenario.red_light["stop_progress_m"]) + 1.0
    )
    route_length = sum(
        math.hypot(scenario.route[i][0] - scenario.route[i - 1][0], scenario.route[i][1] - scenario.route[i - 1][1])
        for i in range(1, len(scenario.route))
    )
    if decision is None or applied is None:
        raise RuntimeError("branch failed before Safety decision could be recorded")
    outcome = make_branch_outcome(
        candidate=candidate,
        reset=reset_cmp,
        decision=decision,
        applied=applied,
        pre_binding_sha256=pre_hash,
        post_binding_sha256=post_hash,
        timeline=timeline,
        cleanup_complete=cleanup_complete,
        collision_count=collision_count,
        red_light_violation=red_violation,
        off_corridor_duration_s=off_duration,
        route_completed=final_progress >= route_length - 2.0,
        route_progress_m=final_progress,
        timeline_path=timeline_path,
        actor_future_path=actor_path,
        event_path=event_path,
        branch_latency_s=time.perf_counter() - started,
        whole_gpu_peak_gb=0.0 if gpu_sampler is None else gpu_sampler.peak_used_gib(),
        errors=errors,
    )
    return outcome, artifacts


def _collect_map_impl(
    dataset_id: str,
    map_name: str,
    scope: str,
    *,
    gpu_sampler: GPUMemorySampler | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    store = PairedOutcomeStore(DATA_ROOT, dataset_id)
    scenarios, physical_manifest = _load_physical_manifest(store)
    client, world, report = _connection(map_name)
    _require_clean_scene(world)
    evidence_dir = EVIDENCE_ROOT / dataset_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(evidence_dir / "run_registry.sqlite3")
    policy = NominalVLAPolicy(keep_on_gpu=True)
    policy.ensure_loaded()
    classic = ClassicExpertGenerator()
    vla = NominalVLAGenerator(policy, generator_hash=simlingo_generator_hash(policy))
    selected = PILOT_MATRIX if scope == "pilot" else FIXED_MATRIX
    rows = [row for row in selected if row.scenario.map_name == map_name]
    results: list[dict[str, Any]] = []
    for matrix in rows:
        pair_id = matrix.pair_id
        if store.has_valid_pair(pair_id):
            results.append({"pair_id": pair_id, "status": "RESUMED"})
            continue
        scenario = scenarios[pair_id]
        pair_errors: list[str] = []
        candidates = ()
        branches = ()
        artifacts: dict[str, str] = {}
        capture_payload: dict[str, Any] | None = None
        with _FrozenTrafficLights(world, scenario) as lights:
            try:
                capture_payload = _capture(client, scenario, store, registry, lights, classic, vla, policy)
                guarded = capture_payload["guarded"]
                routing = capture_payload["routing"]
                candidates = tuple(
                    candidate_snapshot(item, slot=(scenario.expert_slot if item.provenance.source.value == "expert" else 1 - scenario.expert_slot))
                    for item in guarded.candidates
                )
                eligible = (
                    len(candidates) == 2
                    and all(item.guard is not None and item.guard.passed for item in guarded.candidates)
                    and routing.selection_space is SelectionSpace.DISTINCT
                    and capture_payload["vla_forward_count"] == 1
                    and capture_payload["swap_invariant"]
                    and capture_payload["cleanup"] is not None
                    and capture_payload["cleanup"]["status"] == "COMPLETED"
                )
                if eligible:
                    by_source = {item.provenance.source.value: item.candidate for item in guarded.candidates}
                    branch_rows = []
                    for source in scenario.branch_order:
                        outcome, branch_artifacts = _branch(
                            client, scenario, store, registry, lights,
                            capture_payload["capture_reset"], by_source[source], gpu_sampler,
                        )
                        branch_rows.append(outcome)
                        for name, digest in branch_artifacts.items():
                            artifacts[f"{outcome.candidate_id}:{name}"] = digest
                    branches = tuple(branch_rows)
                    terminal = PairTerminalStatus.VALID_PAIR if all(branch.complete for branch in branches) else PairTerminalStatus.INVALID_PAIR
                else:
                    terminal = PairTerminalStatus.INELIGIBLE
            except BaseException as exc:
                terminal = PairTerminalStatus.CAPTURE_FAILED
                pair_errors.append(f"{type(exc).__name__}:{exc}")

        anchor_payload: dict[str, Any] = {
            "physical_scenario_sha256": scenario.physical_sha256,
            "physical_manifest_sha256": physical_manifest["physical_manifest_sha256"],
            "selection_space": None,
            "swap_invariant": False,
        }
        history: tuple[Mapping[str, Any], ...] = ()
        vla_forward_count = 0
        capture_reset = None
        branch_order_ids: tuple[str, ...] = ()
        if capture_payload is not None:
            anchor_payload.update(capture_payload["anchor"].to_dict())
            anchor_payload["observable_snapshot"] = asdict(capture_payload["anchor"].safety_snapshot)
            anchor_payload["selection_space"] = capture_payload["routing"].selection_space.value
            anchor_payload["routing"] = capture_payload["routing"].to_dict()
            anchor_payload["swap_invariant"] = capture_payload["swap_invariant"]
            history = tuple(capture_payload["history"])
            vla_forward_count = int(capture_payload["vla_forward_count"])
            capture_reset = capture_payload["capture_reset"]
            ids_by_source = {item.source: item.candidate_id for item in candidates}
            if set(ids_by_source) == {"expert", "vla"}:
                branch_order_ids = tuple(ids_by_source[source] for source in scenario.branch_order)
        pair = PairRecord(
            dataset_id=dataset_id,
            scenario=scenario.scenario,
            matrix_sha256=MATRIX_SHA256,
            config_sha256=H2_CONFIG_SHA256,
            anchor=anchor_payload,
            observable_history=history,
            route=scenario.route,
            candidates=tuple(candidates),
            terminal_status=terminal,
            branch_order=branch_order_ids,
            branches=tuple(branches),
            vla_forward_count=vla_forward_count,
            capture_reset=capture_reset,
            artifact_hashes=artifacts,
            errors=tuple(pair_errors),
        )
        store.write_pair(pair)
        results.append({"pair_id": pair_id, "status": terminal.value, "content_sha256": pair.content_sha256})
        _require_clean_scene(world)
    store.write_manifest()
    payload = {
        "schema_version": "safedrive.h2.collect_map_evidence.v1",
        "dataset_id": dataset_id,
        "map_name": map_name,
        "scope": scope,
        "connection": report.to_dict(),
        "config": config_identity(),
        "config_sha256": H2_CONFIG_SHA256,
        "worktree": _worktree_identity(),
        "model_forward_count": policy.forward_count,
        "model_peak_vram_mb": policy.last_peak_vram_mb,
        "results": results,
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    _atomic_json(evidence_dir / f"collect-{scope}-{map_name}.json", payload)
    return {"ok": True, "map": map_name, "scope": scope, "pairs": len(results), "evidence_sha256": payload["evidence_sha256"]}


def collect_map(dataset_id: str, map_name: str, scope: str) -> dict[str, Any]:
    """Collect one map while recording device-level GPU memory evidence."""

    sampler = GPUMemorySampler(interval_s=0.1, gpu_index=0).start()
    evidence_dir = EVIDENCE_ROOT / dataset_id
    try:
        result = _collect_map_impl(dataset_id, map_name, scope, gpu_sampler=sampler)
    except BaseException as exc:
        gpu = sampler.stop()
        _atomic_json(
            evidence_dir / f"collect-{scope}-{map_name}-failure.json",
            {
                "schema_version": "safedrive.h2.collect_map_failure.v1",
                "dataset_id": dataset_id,
                "map_name": map_name,
                "scope": scope,
                "config": config_identity(),
                "error": f"{type(exc).__name__}:{exc}",
                "gpu": gpu,
                "worktree": _worktree_identity(),
            },
        )
        raise
    gpu = sampler.stop()
    evidence = evidence_dir / f"collect-{scope}-{map_name}.json"
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["gpu"] = gpu
    payload["whole_gpu_peak_gb"] = float(gpu.get("peak_used_gib", 0.0))
    payload.pop("evidence_sha256", None)
    payload["evidence_sha256"] = stable_sha256(payload)
    _atomic_json(evidence, payload)
    result["evidence_sha256"] = payload["evidence_sha256"]
    result["whole_gpu_peak_gb"] = payload["whole_gpu_peak_gb"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    restart = subparsers.add_parser("restart-smoke")
    restart.add_argument("--map", required=True, choices=("Town01", "Town03", "Town05"))
    materialize = subparsers.add_parser("materialize-map")
    materialize.add_argument("--map", required=True, choices=("Town01", "Town03", "Town05"))
    subparsers.add_parser("freeze-manifest")
    collect = subparsers.add_parser("collect-map")
    collect.add_argument("--map", required=True, choices=("Town01", "Town03", "Town05"))
    collect.add_argument("--scope", required=True, choices=("pilot", "full"))
    args = parser.parse_args()
    try:
        if args.command == "restart-smoke":
            result = restart_smoke(args.dataset_id, args.map)
        elif args.command == "materialize-map":
            result = materialize_map(args.dataset_id, args.map)
        elif args.command == "freeze-manifest":
            result = freeze_manifest(args.dataset_id)
        else:
            result = collect_map(args.dataset_id, args.map, args.scope)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
