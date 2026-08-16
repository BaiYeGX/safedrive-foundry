#!/usr/bin/env python3
"""Real CARLA H5 closed-loop collector: off/on/defer arms for frozen scenarios.

This script is the live execution layer.  It does not read H4 test labels,
Oracle labels, or Regression inputs.  It only uses frozen physical scenarios,
frozen H4 World checkpoints, and the H5 router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
import torch  # noqa: E402

from classic_stack.control.controller import ControlLoop  # noqa: E402
from data_pipeline.h2.carla_scenarios import PhysicalScenario, transform_from_dict  # noqa: E402
from data_pipeline.h2.contracts import (  # noqa: E402
    ActorInitialState,
    ResetSignature,
    compare_reset_signatures,
    stable_sha256,
)
from data_pipeline.h2.gpu import GPUMemorySampler  # noqa: E402
from data_pipeline.h2.live_contract import (  # noqa: E402
    actor_initial_state,
    kinematic_metrics,
    reset_signature,
    route_projection,
    trajectory_sha256,
)
from data_pipeline.h3.model import load_model  # noqa: E402
from data_pipeline.h4.contracts import FINAL_CHECKPOINTS, H4_CONFIG  # noqa: E402
from data_pipeline.h4.runtime import NormalizedWorldScorer  # noqa: E402
from data_pipeline.h5.config import H5_CONFIG, H5_CONFIG_SHA256  # noqa: E402
from data_pipeline.h5.matrix import h5_matrix_sha256, load_h5_matrix  # noqa: E402
from data_pipeline.h5.store import H5Store  # noqa: E402
from driving_vla.hybrid import (  # noqa: E402
    ClassicExpertGenerator,
    NominalVLAGenerator,
    generate_hybrid_set,
    simlingo_generator_hash,
)
from driving_vla.hybrid.carla_anchor import (  # noqa: E402
    build_anchor,
    ego_state,
    image_png_bytes,
    map_basename,
)
from driving_vla.hybrid.pipeline import H1CandidatePipeline, ego_history_entry  # noqa: E402
from driving_vla.hybrid.router import FrozenH1Router  # noqa: E402
from driving_vla.model.nominal_policy import NominalVLAPolicy  # noqa: E402
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
)
from driving_vla.runtime.safety_control_bind import apply_safety_control  # noqa: E402
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

from data_pipeline.h5.runtime import H5WorldRouter  # noqa: E402

DATA_ROOT = ROOT / "generated" / "h5"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h5"
PROFILE = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]


def _git_text(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return (result.stdout or "").strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worktree_identity() -> dict[str, Any]:
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True, check=False).stdout
    raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout
    rows = []
    for encoded in sorted(item for item in raw.split(b"\0") if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if path.is_file():
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
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


def _require_clean_scene(world: Any) -> None:
    residue = [
        int(actor.id)
        for actor in world.get_actors()
        if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker."))
        and bool(getattr(actor, "is_alive", True))
    ]
    if residue:
        raise RuntimeError(f"NEEDS_CLEAN_SCENE:{residue}")


def _force_clean_scene(world: Any) -> None:
    """Destroy all vehicle/walker actors before an infrastructure retry."""
    for actor in world.get_actors():
        type_id = str(getattr(actor, "type_id", ""))
        if type_id.startswith(("vehicle.", "walker.")) and bool(getattr(actor, "is_alive", True)):
            try:
                actor.destroy()
            except Exception:
                pass
    try:
        if bool(world.get_settings().synchronous_mode):
            world.tick()
    except Exception:
        pass
    time.sleep(1.0)
    _require_clean_scene(world)


def _connection(map_name: str) -> tuple[Any, Any, Any]:
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=10.0)
    report = resolver.preflight()
    if report.status != READY:
        raise RuntimeError(f"CARLA_NOT_READY:{report.error_code}:{report.error_message}")
    if map_basename(str(report.map)) != map_name:
        raise RuntimeError(f"MAP_MISMATCH:{report.map}!={map_name}")
    client, _ = resolver.connect(report=report)
    return client, client.get_world(), report


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
            if self.scenario.script.get("red_at_capture", True):
                self.force_red()
            else:
                self.force_green()
        return self

    def _target_light(self):
        lights = [actor for actor in self.world.get_actors() if str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light")]
        if not lights or self.scenario.red_light is None:
            return None
        x, y = float(self.scenario.red_light["trigger_x"]), float(self.scenario.red_light["trigger_y"])
        return min(lights, key=lambda light: math.hypot(float(light.get_transform().location.x) - x, float(light.get_transform().location.y) - y))

    def force_red(self) -> None:
        target = self._target_light()
        if target is not None:
            target.freeze(True)
            target.set_state(carla.TrafficLightState.Red)

    def force_green(self) -> None:
        for light, _state, _was_frozen in self.saved:
            try:
                light.freeze(True)
                light.set_state(carla.TrafficLightState.Green)
            except Exception:
                pass
        lights = [actor for actor in self.world.get_actors() if str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light")]
        if not lights or self.scenario.red_light is None:
            return
        x, y = float(self.scenario.red_light["trigger_x"]), float(self.scenario.red_light["trigger_y"])
        target = min(lights, key=lambda light: math.hypot(float(light.get_transform().location.x) - x, float(light.get_transform().location.y) - y))
        target.freeze(True)
        target.set_state(carla.TrafficLightState.Red)

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


def _npc_controls(scenario: PhysicalScenario, *, tick_index: int | None = None, pre_roll: bool = False) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    script = scenario.script

    def choose(role: str, event_key: str) -> Mapping[str, Any] | None:
        event_control = script.get(f"{role}_control")
        pre_roll_control = script.get(f"{role}_pre_roll_control")
        event_tick = int(script.get(f"{role}_{event_key}_tick", 0)) if f"{role}_{event_key}_tick" in script else 0
        if pre_roll:
            return pre_roll_control or event_control
        if event_tick and tick_index is not None and tick_index < event_tick:
            return pre_roll_control or event_control
        return event_control

    for role in ("lead", "cutter", "cross"):
        command = choose(role, "brake" if role == "lead" else ("cut_in" if role == "cutter" else "conflict"))
        if command is not None and role in {str(item["role"]) for item in scenario.npc_actors}:
            controls[role] = carla.VehicleControl(
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
        experiment_id="h5-world-on-off",
        run_id=run_id,
        scenario_id=f"h5-{scenario.pair_id}-{phase}",
        attempt_id=0,
        server_epoch="carla-0.9.16-h5",
        producer_version="h5-closed-loop-v1",
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
        lease_path=ROOT / ".runtime" / f"tick-lease-h5-{run_id}.lock",
        owner="sdf.h5.collector",
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
) -> tuple[Any, list[dict[str, Any]], list[tuple[float, float, float, float]]]:
    history: list[dict[str, Any]] = []
    ego_history: list[tuple[float, float, float, float]] = []
    header = None
    pre_roll_ticks = int(scenario.script.get("pre_roll_ticks", 20))
    for index in range(pre_roll_ticks):
        family = str(scenario.script.get("family", scenario.scenario.family))
        if family in {
            "free_flow", "slow_lead", "cut_in",
            "emergency_lead_brake", "aggressive_cut_in", "red_light_dilemma", "cross_traffic_conflict",
        }:
            target = float(scenario.script.get("pre_roll_target_speed_mps", 2.0))
            kp = float(scenario.script.get("pre_roll_kp", 0.30))
            current, _ = ego_state(runtime._actors["ego"])
            throttle = max(0.0, min(float(scenario.script.get("pre_roll_max_throttle", 0.35)), kp * (target - current.v)))
            ego_control = carla.VehicleControl(throttle=throttle, brake=0.0)
        else:
            ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        controls = {"ego": ego_control, **_npc_controls(scenario, pre_roll=True)}
        header = runtime.tick_controls(controls)
        state, acceleration = ego_state(runtime._actors["ego"])
        ego_history.append((state.x, state.y, state.yaw, state.v))
        history.append(
            {
                "index": index,
                "carla_frame": int(header.carla_frame),
                "simulation_time_s": float(header.simulation_time),
                "ego_x": state.x,
                "ego_y": state.y,
                "ego_yaw": state.yaw,
                "ego_speed_mps": state.v,
                "ego_acceleration_mps2": acceleration,
            }
        )
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


def _load_scorer() -> NormalizedWorldScorer:
    evidence = json.loads((ROOT / "docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json").read_text())
    stats = evidence["normalization_stats"]
    stats_list = [(float(item["mean"]), float(item["std"])) for item in stats["items"]]
    models = []
    for seed, info in FINAL_CHECKPOINTS.items():
        path = ROOT / info["path"]
        actual = _file_sha256(path)
        if actual != info["sha256"]:
            raise ValueError(f"checkpoint_sha_mismatch:{path}:{actual}")
        model, _ = load_model(path, device="cuda")
        models.append(model)
    return NormalizedWorldScorer(
        models,
        stats_list,
        device="cuda",
        temperature=float(H5_CONFIG["temperature"]),
        risk_defer_probability=float(H5_CONFIG["risk_defer_probability"]),
        probability_temperature_floor=float(H5_CONFIG["probability_temperature_floor"]),
    )


def _make_pipeline(arm: str, scorer: NormalizedWorldScorer) -> H1CandidatePipeline:
    if arm == "off":
        return H1CandidatePipeline(router=FrozenH1Router())
    router = H5WorldRouter(
        scorer,
        min_hold_ticks=int(H5_CONFIG["router"]["min_hold_ticks"]),
        hysteresis_margin=float(H5_CONFIG["router"]["hysteresis_margin"]),
        emergency_switch_margin=float(H5_CONFIG["router"]["emergency_switch_margin"]),
        single_pass_grace_ticks=int(H5_CONFIG["router"]["single_pass_grace_ticks"]),
        force_defer=(arm == "defer"),
        scorer_deadline_ms=float(H5_CONFIG["runtime"]["scorer_deadline_ms"]),
    )
    return H1CandidatePipeline(router=router)


def _run_arm(
    client: Any,
    scenario: PhysicalScenario,
    *,
    arm: str,
    arm_order_index: int,
    pipeline: H1CandidatePipeline,
    classic: ClassicExpertGenerator,
    vla: NominalVLAGenerator,
    policy: NominalVLAPolicy,
    registry: RunRegistry,
    lights: _FrozenTrafficLights,
    reference_reset: Mapping[str, Any] | None,
    gpu_sampler: GPUMemorySampler | None,
) -> dict[str, Any]:
    run_id = f"{scenario.pair_id}-{arm}-{time.time_ns()}"
    started = time.perf_counter()
    runtime = _runtime(
        client, scenario, run_id=run_id, phase=arm, sensors=(_camera_spec(), *_event_specs()), registry=registry
    )
    timeline: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []
    first_frame = -1
    last_frame = -1
    cleanup_complete = False
    route_length = sum(
        math.hypot(scenario.route[i][0] - scenario.route[i - 1][0], scenario.route[i][1] - scenario.route[i - 1][1])
        for i in range(1, len(scenario.route))
    )
    try:
        header, history, ego_history = _pre_roll(runtime, scenario)
        initial_state, _ = ego_state(runtime._actors["ego"])
        initial_progress, _ = route_projection(initial_state.x, initial_state.y, scenario.route)
        current_reset = reset_signature(
            _actors_by_role(runtime), route=scenario.route, weather=_actual_weather(runtime, scenario),
            lights=lights.snapshot(), script=scenario.script,
        )
        reset_cmp = (
            None if reference_reset is None
            else compare_reset_signatures(reference_reset, current_reset)
        )
        # Use the same observable history for all arms.
        pipeline.seed_ego_history(history)
        control_loop = ControlLoop()
        before_forward = policy.forward_count
        for tick in range(int(H5_CONFIG["matrix"]["decision_ticks"])):
            anchor = build_anchor(runtime, header, scenario.route, ego_history=ego_history)
            seed = int(stable_sha256({"pair_id": scenario.pair_id, "tick": tick}), 16) % (2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generated = generate_hybrid_set(anchor, classic, vla)
            if len(generated.candidates) != 2 or not all(attempt.success for attempt in generated.attempts):
                errors.append(f"generation_incomplete_tick:{tick}:sources={[attempt.source.value for attempt in generated.attempts if not attempt.success]}")
            result = pipeline.decide(generated)
            routing = result.routing
            guarded = result.guarded_set
            selected_source = None
            if routing.selected_candidate_id is not None:
                for item in guarded.candidates:
                    if item.candidate.candidate_id == routing.selected_candidate_id:
                        selected_source = item.provenance.source.value
                        break
            safety_decision = result.safety.decision
            ego, _ = ego_state(runtime._actors["ego"])
            selected_candidates = tuple(
                item.candidate
                for item in guarded.candidates
                if item.candidate.candidate_id == routing.selected_candidate_id
            )
            applied = apply_safety_control(
                safety_decision,
                guarded.to_policy_candidate_set(selected_candidates),
                control_loop,
                ego,
                float(header.simulation_time),
            )
            router_metrics = pipeline.router.metrics() if hasattr(pipeline.router, "metrics") else {}
            scorer_latency = None
            scorer_disposition = None
            scorer_reason = None
            if hasattr(pipeline.router, "last_score"):
                score = pipeline.router.last_score
                if score is not None:
                    scorer_latency = float(score.latency_ms)
                    scorer_disposition = str(score.disposition)
                    scorer_reason = score.defer_reason
            controls = {
                "ego": carla.VehicleControl(
                    throttle=float(applied.throttle), brake=float(applied.brake), steer=float(applied.steer),
                    reverse=bool(getattr(applied, "reverse", False)),
                ),
                **_npc_controls(scenario, tick_index=tick),
            }
            tick_started = time.perf_counter()
            header = runtime.tick_controls(controls)
            tick_wall_ms = (time.perf_counter() - tick_started) * 1000.0
            state, acceleration = ego_state(runtime._actors["ego"])
            progress, corridor_distance = route_projection(state.x, state.y, scenario.route)
            previous_yaw = ego_history[-1][2] if ego_history else state.yaw
            yaw_rate = ((state.yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi) / 0.05
            ego_history.append((state.x, state.y, state.yaw, state.v))
            timeline.append(
                {
                    "tick": tick,
                    "carla_frame": int(header.carla_frame),
                    "simulation_time_s": float(header.simulation_time),
                    "x": state.x, "y": state.y, "yaw": state.yaw, "speed_mps": state.v,
                    "acceleration_mps2": acceleration,
                    "lateral_acceleration_mps2": state.v * yaw_rate,
                    "route_progress_m": progress,
                    "corridor_distance_m": corridor_distance,
                    "throttle": float(applied.throttle), "brake": float(applied.brake), "steer": float(applied.steer),
                    "control_mode": str(getattr(applied, "applied_mode", "")),
                    "deadline_miss": False,
                    "tick_wall_ms": tick_wall_ms,
                }
            )
            decisions.append(
                {
                    "tick": tick,
                    "carla_frame": int(header.carla_frame),
                    "simulation_time_s": float(header.simulation_time),
                    "anchor_id": anchor.observation_id,
                    "candidate_sha256": {
                        item.provenance.source.value: item.provenance.canonical_sha256
                        for item in guarded.candidates
                    },
                    "guard": {item.candidate.candidate_id: item.guard.to_dict() for item in guarded.candidates},
                    "routing": routing.to_dict(),
                    "selected_source": selected_source,
                    "scorer_latency_ms": scorer_latency,
                    "scorer_disposition": scorer_disposition,
                    "scorer_defer_reason": scorer_reason,
                    "safety_decision_kind": str(getattr(safety_decision.decision_kind, "value", safety_decision.decision_kind)),
                    "applied_mode": str(getattr(applied, "applied_mode", "")),
                    "switch_count": int(router_metrics.get("switch_count", 0)),
                    "defer_count": int(router_metrics.get("defer_count", 0)),
                    "fallback_count": int(router_metrics.get("defer_count", 0)) if arm != "off" else 0,
                    "generation_latency_s": {
                        attempt.source.value: attempt.generation_latency_s for attempt in generated.attempts
                    },
                    "tick_wall_ms": tick_wall_ms,
                }
            )
        first_frame = int(timeline[0]["carla_frame"]) if timeline else -1
        last_frame = int(timeline[-1]["carla_frame"]) if timeline else -1
        event_rows, collision_count = _event_rows(runtime, first_frame, last_frame) if first_frame >= 0 else ([], 0)
        runtime.complete()
        cleanup = registry.record(run_id)
        cleanup_complete = bool(cleanup and cleanup["status"] == "COMPLETED")
        final_abs_progress = float(timeline[-1]["route_progress_m"]) if timeline else initial_progress
        final_progress = final_abs_progress - initial_progress
        off_duration = 0.05 * sum(float(row["corridor_distance_m"]) > 2.0 for row in timeline)
        red_violation = bool(
            scenario.red_light is not None
            and timeline
            and final_abs_progress > float(scenario.red_light["stop_progress_m"]) + 1.0
        )
        metrics = kinematic_metrics(timeline, dt_s=0.05) if timeline else {"jerk_rms_mps3": 0.0, "acceleration_rms_mps2": 0.0, "lateral_acceleration_rms_mps2": 0.0}
        scorer_ms = [float(d["scorer_latency_ms"]) for d in decisions if d.get("scorer_latency_ms") is not None]
        scorer_deadline_misses = sum(1 for d in decisions if d.get("scorer_latency_ms") is not None and float(d["scorer_latency_ms"]) > float(H5_CONFIG["runtime"]["scorer_deadline_ms"]))
        safety_fallback_count = sum(1 for d in decisions if d.get("applied_mode") != "TRACK_APPROVED")
        router_metrics = pipeline.router.metrics() if hasattr(pipeline.router, "metrics") else {}
        fallback_count = int(router_metrics.get("defer_count", 0)) if arm != "off" else 0
        payload = {
            "schema_version": H5_CONFIG["schema_version"],
            "dataset_id": "",
            "run_id": run_id,
            "pair_id": scenario.pair_id,
            "scenario": scenario.scenario.to_dict(),
            "physical_sha256": scenario.physical_sha256,
            "manifest_kind": "",
            "arm": arm,
            "arm_order_index": arm_order_index,
            "reset_signature": current_reset.to_dict(),
            "reset_comparison": None if reset_cmp is None else reset_cmp.to_dict(),
            "route_progress_m": final_progress,
            "route_completed": final_progress >= route_length - 2.0,
            "collision_count": collision_count,
            "red_light_violation": red_violation,
            "off_corridor_duration_s": off_duration,
            "jerk_rms_mps3": metrics["jerk_rms_mps3"],
            "acceleration_rms_mps2": metrics["acceleration_rms_mps2"],
            "lateral_acceleration_rms_mps2": metrics["lateral_acceleration_rms_mps2"],
            "switch_count": int(router_metrics.get("switch_count", 0)),
            "defer_count": int(router_metrics.get("defer_count", 0)),
            "fallback_count": fallback_count,
            "safety_fallback_count": safety_fallback_count,
            "deadline_misses": 0,
            "scorer_deadline_misses": scorer_deadline_misses,
            "p50_scorer_ms": sorted(scorer_ms)[len(scorer_ms)//2] if scorer_ms else None,
            "p95_scorer_ms": sorted(scorer_ms)[int(len(scorer_ms)*0.95)] if scorer_ms else None,
            "p99_scorer_ms": sorted(scorer_ms)[int(len(scorer_ms)*0.99)] if scorer_ms else None,
            "whole_gpu_peak_gb": 0.0 if gpu_sampler is None else gpu_sampler.peak_used_gib(),
            "vla_forward_count": policy.forward_count - before_forward,
            "ticks_executed": len(timeline),
            "cleanup_complete": cleanup_complete,
            "ok": cleanup_complete and len(timeline) == int(H5_CONFIG["matrix"]["decision_ticks"]),
            "errors": tuple(errors),
            "decisions": tuple(decisions),
            "timeline": tuple(timeline),
            "events": tuple(event_rows),
            "worktree": _worktree_identity(),
            "config_sha256": H5_CONFIG_SHA256,
        }
        return payload
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
        runtime.abort(type(exc).__name__)
        cleanup = registry.record(run_id)
        return {
            "schema_version": H5_CONFIG["schema_version"],
            "dataset_id": "",
            "run_id": run_id,
            "pair_id": scenario.pair_id,
            "scenario": scenario.scenario.to_dict(),
            "physical_sha256": scenario.physical_sha256,
            "manifest_kind": "",
            "arm": arm,
            "arm_order_index": arm_order_index,
            "reset_signature": {},
            "reset_comparison": None,
            "route_progress_m": 0.0,
            "route_completed": False,
            "collision_count": 0,
            "red_light_violation": False,
            "off_corridor_duration_s": 0.0,
            "jerk_rms_mps3": 0.0,
            "acceleration_rms_mps2": 0.0,
            "lateral_acceleration_rms_mps2": 0.0,
            "switch_count": 0,
            "defer_count": 0,
            "fallback_count": 0,
            "safety_fallback_count": 0,
            "deadline_misses": 0,
            "scorer_deadline_misses": 0,
            "p50_scorer_ms": None,
            "p95_scorer_ms": None,
            "p99_scorer_ms": None,
            "whole_gpu_peak_gb": 0.0 if gpu_sampler is None else gpu_sampler.peak_used_gib(),
            "vla_forward_count": 0,
            "ticks_executed": 0,
            "cleanup_complete": False,
            "ok": False,
            "errors": tuple(errors),
            "decisions": (),
            "timeline": (),
            "events": (),
            "worktree": _worktree_identity(),
            "config_sha256": H5_CONFIG_SHA256,
        }


def _collect_map_impl(
    dataset_id: str,
    map_name: str,
    scope: str,
    *,
    pair_id: str | None = None,
    gpu_sampler: GPUMemorySampler | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    store = H5Store(DATA_ROOT, dataset_id)
    scenarios = load_h5_matrix(ROOT, full=(scope == "full"))
    rows = [s for s in scenarios if s.scenario.map_name == map_name]
    if pair_id:
        rows = [s for s in rows if s.pair_id == pair_id]
    if not rows:
        raise RuntimeError(f"NO_SCENARIOS_FOR_MAP:{map_name}")
    client, world, report = _connection(map_name)
    _require_clean_scene(world)
    evidence_dir = EVIDENCE_ROOT / dataset_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(evidence_dir / f"run_registry_{map_name}_{scope}.sqlite3")
    policy = NominalVLAPolicy(keep_on_gpu=True)
    policy.ensure_loaded()
    classic = ClassicExpertGenerator()
    vla = NominalVLAGenerator(policy, generator_hash=simlingo_generator_hash(policy))
    scorer = _load_scorer()
    results = []
    for scenario in rows:
        physical = scenario.physical
        reference_reset = None
        with _FrozenTrafficLights(world, physical) as lights:
            for arm_index, arm in enumerate(scenario.arm_order):
                if store.has_run(scenario.pair_id, arm):
                    if reference_reset is None:
                        existing = store.read_run(scenario.pair_id, arm)
                        rs = existing.get("reset_signature")
                        if rs:
                            reference_reset = ResetSignature(
                                actors=tuple(ActorInitialState(**a) for a in rs.get("actors", ())),
                                route_sha256=str(rs.get("route_sha256", "")),
                                weather_sha256=str(rs.get("weather_sha256", "")),
                                light_sha256=str(rs.get("light_sha256", "")),
                                script_sha256=str(rs.get("script_sha256", "")),
                            )
                    results.append({"pair_id": scenario.pair_id, "arm": arm, "status": "RESUMED"})
                    continue
                pipeline = _make_pipeline(arm, scorer)
                run = None
                for attempt in range(2):
                    try:
                        run = _run_arm(
                            client, physical, arm=arm, arm_order_index=arm_index,
                            pipeline=pipeline, classic=classic, vla=vla, policy=policy,
                            registry=registry, lights=lights, reference_reset=reference_reset,
                            gpu_sampler=gpu_sampler,
                        )
                        break
                    except Exception as exc:
                        if attempt >= 1:
                            raise
                        # Infrastructure-only retry: runtime failed to start (e.g. spawn_failed).
                        _force_clean_scene(world)
                        time.sleep(2.0)
                assert run is not None
                run["dataset_id"] = dataset_id
                run["manifest_kind"] = scenario.manifest_kind
                run["config_sha256"] = H5_CONFIG_SHA256
                store.write_run(run)
                if reference_reset is None and run.get("reset_signature"):
                    # Convert dict to a lightweight object for compare_reset_signatures.
                    rs = run["reset_signature"]
                    reference_reset = ResetSignature(
                        actors=tuple(ActorInitialState(**a) for a in rs.get("actors", ())),
                        route_sha256=str(rs.get("route_sha256", "")),
                        weather_sha256=str(rs.get("weather_sha256", "")),
                        light_sha256=str(rs.get("light_sha256", "")),
                        script_sha256=str(rs.get("script_sha256", "")),
                    )
                results.append({"pair_id": scenario.pair_id, "arm": arm, "status": "OK" if run["ok"] else "FAILED", "run_id": run["run_id"]})
        _require_clean_scene(world)
    store.write_manifest()
    payload = {
        "schema_version": "safedrive.h5.collect_evidence.v1",
        "dataset_id": dataset_id,
        "map_name": map_name,
        "scope": scope,
        "connection": report.to_dict(),
        "config_sha256": H5_CONFIG_SHA256,
        "worktree": _worktree_identity(),
        "matrix_sha256": h5_matrix_sha256(scenarios),
        "results": results,
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    _atomic_json(evidence_dir / f"collect-{scope}-{map_name}.json", payload)
    return {"ok": True, "map": map_name, "scope": scope, "pairs": len(rows), "evidence_sha256": payload["evidence_sha256"]}


def collect_map(dataset_id: str, map_name: str, scope: str, *, pair_id: str | None = None) -> dict[str, Any]:
    sampler = GPUMemorySampler(interval_s=0.1, gpu_index=0).start()
    evidence_dir = EVIDENCE_ROOT / dataset_id
    try:
        result = _collect_map_impl(dataset_id, map_name, scope, pair_id=pair_id, gpu_sampler=sampler)
    except BaseException as exc:
        gpu = sampler.stop()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            evidence_dir / f"collect-{scope}-{map_name}-failure.json",
            {
                "schema_version": "safedrive.h5.collect_failure.v1",
                "dataset_id": dataset_id,
                "map_name": map_name,
                "scope": scope,
                "error": f"{type(exc).__name__}:{exc}",
                "gpu": gpu,
            },
        )
        raise
    sampler.stop()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--pair-id", default=None)
    args = parser.parse_args()
    result = collect_map(args.dataset_id, args.map, args.scope, pair_id=args.pair_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
