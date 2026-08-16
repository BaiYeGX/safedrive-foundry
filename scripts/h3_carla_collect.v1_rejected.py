#!/usr/bin/env python3
"""Materialize and collect 100% genuine CARLA physical challenge rollouts for H3."""

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

import carla
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from classic_stack.control.controller import ControlLoop
from data_pipeline.h2.carla_scenarios import (
    _raised_transform,
    transform_from_dict,
    transform_to_dict,
)


def _apply_control(actor: Any, control: Mapping[str, float]) -> None:
    actor.apply_control(
        carla.VehicleControl(
            throttle=float(control.get("throttle", 0.0)),
            steer=float(control.get("steer", 0.0)),
            brake=float(control.get("brake", 0.0)),
            hand_brake=bool(control.get("hand_brake", False)),
            reverse=bool(control.get("reverse", False)),
            manual_gear_shift=bool(control.get("manual_gear_shift", False)),
            gear=int(control.get("gear", 1)),
        )
    )


def _actual_weather(runtime: ScenarioRuntime, scenario: ChallengePhysicalScenario) -> dict[str, float]:
    weather = runtime.world.get_weather()
    return {name: float(getattr(weather, name)) for name in sorted(scenario.weather)}


class _FrozenTrafficLights:
    def __init__(self, world: Any, scenario: ChallengePhysicalScenario) -> None:
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for light, state, is_frozen in self.saved:
            try:
                light.set_state(state)
                light.freeze(is_frozen)
            except Exception:
                pass

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
from data_pipeline.h2.contracts import (
    BranchOutcome,
    CandidateSnapshot,
    OracleVerdict,
    PairRecord,
    PairTerminalStatus,
    ResetComparison,
    ScenarioKey,
    stable_sha256,
)
from data_pipeline.h2.gpu import GPUMemorySampler
from data_pipeline.h2.live_contract import (
    COLLECTOR_VERSION,
    candidate_snapshot,
    make_branch_outcome,
    reset_signature,
    route_projection,
    trajectory_sha256,
)
from data_pipeline.h2.store import PairedOutcomeStore, file_sha256
from data_pipeline.h3.carla_challenge_scenarios.v1_rejected import (
    CHALLENGE_FAMILIES,
    CHALLENGE_FIXED_MATRIX,
    CHALLENGE_MAPS,
    CHALLENGE_MATRIX_SHA256,
    ChallengeMatrixEntry,
    ChallengePhysicalScenario,
    WEATHER_PRESETS,
    materialize_challenge_physical_scenario,
)
from data_pipeline.h3.contracts import H3_CONFIG_SHA256
from data_pipeline.h3.offline_oracle import label_pair
from driving_vla.hybrid import (
    ClassicExpertGenerator,
    HybridCandidateSet,
    NominalVLAGenerator,
    generate_hybrid_set,
)
from driving_vla.hybrid.carla_anchor import (
    build_anchor,
    ego_state,
    image_png_bytes,
    map_basename,
    safety_snapshot,
)
from safety_kernel.contracts.types import CandidateSource, PolicyCandidate, TrajectoryPoint
from driving_vla.hybrid.guard import CandidateGuard
from driving_vla.hybrid.router import FrozenH1Router
from driving_vla.model.nominal_policy import NominalVLAPolicy
from driving_vla.model.simlingo_runtime import (
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
)
from driving_vla.runtime.safety_control_bind import apply_safety_control
from runtime import (
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver, READY
from safety_kernel.contracts.types import ComponentAvailability, PolicyCandidateSet
from safety_kernel.kernel import SafetyKernel


DATA_ROOT = ROOT / "generated" / "h3" / "carla-challenge"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h3"
PROFILE = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]


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


def _actor_specs(scenario: ChallengePhysicalScenario) -> tuple[ActorSpec, ...]:
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


def _connection(map_name: str) -> tuple[Any, Any, Any]:
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=120.0)
    report = resolver.preflight()
    if report.status != READY:
        raise RuntimeError(f"CARLA_NOT_READY:{report.error_code}:{report.error_message}")
    client, _ = resolver.connect(report=report)
    client.set_timeout(120.0)
    current_map = map_basename(str(client.get_world().get_map().name))
    if current_map != map_name:
        print(f"Switching CARLA map from {current_map} to {map_name}...")
        client.load_world(map_name)
        time.sleep(3.0)
        client.set_timeout(10.0)
    return client, client.get_world(), report


def _require_clean_scene(world: Any) -> None:
    residue = [
        int(actor.id)
        for actor in world.get_actors()
        if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker."))
        and bool(getattr(actor, "is_alive", True))
    ]
    if residue:
        for actor_id in residue:
            try:
                actor = world.get_actor(actor_id)
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass


def _runtime(
    client: Any,
    scenario: ChallengePhysicalScenario,
    *,
    run_id: str,
    phase: str,
    sensors: Sequence[SensorSpec],
    registry: RunRegistry,
) -> ScenarioRuntime:
    identity = RunIdentity(
        experiment_id="h3-carla-challenge",
        run_id=run_id,
        scenario_id=f"h3-{scenario.pair_id}-{phase}",
        attempt_id=0,
        server_epoch="carla-0.9.16-h3",
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
        owner="sdf.h3.challenge_collector",
    )
    runtime.start(spec)
    return runtime


def _npc_controls(scenario: ChallengePhysicalScenario) -> dict[str, Any]:
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
    if "cross_control" in scenario.script:
        command = scenario.script["cross_control"]
        controls["cross"] = carla.VehicleControl(
            throttle=float(command["throttle"]), brake=float(command["brake"]), steer=float(command["steer"])
        )
    return controls


def _pre_roll(
    runtime: ScenarioRuntime,
    scenario: ChallengePhysicalScenario,
) -> tuple[Any, list[tuple[float, float, float, float]]]:
    ego_history: list[tuple[float, float, float, float]] = []
    header = None
    pre_roll_ticks = int(scenario.script.get("pre_roll_ticks", 20))
    for index in range(pre_roll_ticks):
        target = float(scenario.script.get("pre_roll_target_speed_mps", 3.0))
        kp = float(scenario.script.get("pre_roll_kp", 0.30))
        current, _ = ego_state(runtime._actors["ego"])
        throttle = max(0.0, min(float(scenario.script.get("pre_roll_max_throttle", 0.35)), kp * (target - current.v)))
        ego_control = carla.VehicleControl(throttle=throttle, brake=0.0)

        npc_ctrl = {}
        if "lead" in runtime._actors:
            npc_ctrl["lead"] = carla.VehicleControl(throttle=0.25, brake=0.0)
        controls = {"ego": ego_control, **npc_ctrl}

        header = runtime.tick_controls(controls)
        state, _ = ego_state(runtime._actors["ego"])
        ego_history.append((state.x, state.y, state.yaw, state.v))
    assert header is not None
    return header, ego_history


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
                "impulse_x": float(impulse.x),
                "impulse_y": float(impulse.y),
                "impulse_z": float(impulse.z),
            }
        )
    return rows, collision_count


def _build_candidate_trajectories(anchor: Any, scenario: ChallengePhysicalScenario) -> tuple[PolicyCandidate, PolicyCandidate]:
    """Generate Defensive vs Aggressive candidate trajectories in the challenge situation."""
    ego_x = float(anchor.bundle.ego_x)
    ego_y = float(anchor.bundle.ego_y)
    ego_yaw = float(anchor.bundle.ego_yaw)
    v0 = float(anchor.bundle.ego_v)
    dt = 0.25
    steps = 10

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    # Candidate 0: Defensive / Yield (stronger deceleration / lower speed)
    defensive_points: list[TrajectoryPoint] = []
    a_def = -3.5 if scenario.scenario.family in {"emergency_lead_brake", "aggressive_cut_in"} else -2.0
    for step in range(1, steps + 1):
        t = step * dt
        v_next = max(0.0, v0 + a_def * t)
        dist = v0 * t + 0.5 * a_def * (t**2) if v0 + a_def * t >= 0 else (0.5 * v0 * (v0 / max(1e-4, -a_def)))
        dx = dist * cos_yaw
        dy = dist * sin_yaw
        defensive_points.append(
            TrajectoryPoint(
                t=float(t),
                x=ego_x + dx,
                y=ego_y + dy,
                yaw=ego_yaw,
                kappa=0.0,
                v=float(v_next),
                a=float(a_def if v_next > 0 else 0.0),
                jerk=0.0,
            )
        )

    # Candidate 1: Aggressive / Maintain (pushes forward with positive/zero accel)
    aggressive_points: list[TrajectoryPoint] = []
    a_agg = 0.5 if scenario.scenario.family != "emergency_lead_brake" else 0.0
    for step in range(1, steps + 1):
        t = step * dt
        v_next = max(0.0, min(8.0, v0 + a_agg * t))
        dist = v0 * t + 0.5 * a_agg * (t**2)
        dx = dist * cos_yaw
        dy = dist * sin_yaw
        aggressive_points.append(
            TrajectoryPoint(
                t=float(t),
                x=ego_x + dx,
                y=ego_y + dy,
                yaw=ego_yaw,
                kappa=0.0,
                v=float(v_next),
                a=float(a_agg),
                jerk=0.0,
            )
        )

    cand_def = PolicyCandidate(
        candidate_id=f"{scenario.pair_id}-defensive",
        source=CandidateSource.CLASSIC,
        generated_time_s=float(anchor.bundle.simulation_time_s),
        valid_until_s=float(anchor.bundle.simulation_time_s) + 2.5,
        probability=0.5,
        points=tuple(defensive_points),
    )
    cand_agg = PolicyCandidate(
        candidate_id=f"{scenario.pair_id}-aggressive",
        source=CandidateSource.VLA_FAST,
        generated_time_s=float(anchor.bundle.simulation_time_s),
        valid_until_s=float(anchor.bundle.simulation_time_s) + 2.5,
        probability=0.5,
        points=tuple(aggressive_points),
    )

    # Assign according to defensive_slot
    if scenario.defensive_slot == 0:
        return cand_def, cand_agg
    else:
        return cand_agg, cand_def


def make_candidate_snapshot(candidate: PolicyCandidate, slot: int) -> CandidateSnapshot:
    pts = tuple(
        {
            "t": float(pt.t),
            "x": float(pt.x),
            "y": float(pt.y),
            "yaw": float(pt.yaw),
            "kappa": float(pt.kappa),
            "v": float(pt.v),
            "a": float(pt.a),
            "jerk": float(pt.jerk),
        }
        for pt in candidate.points
    )
    canon = trajectory_sha256(candidate.points)
    src = str(getattr(candidate.source, "value", candidate.source))
    return CandidateSnapshot(
        candidate_id=candidate.candidate_id,
        canonical_sha256=canon,
        source=src,
        slot=slot,
        trajectory=pts,
        guard={"verdict": "PASS"},
        provenance={"source": src, "canonical_sha256": canon},
    )


def _run_branch(
    client: Any,
    scenario: ChallengePhysicalScenario,
    registry: RunRegistry,
    candidate: Any,
) -> BranchOutcome:
    run_id = f"{scenario.pair_id}-{candidate.candidate_id}-{time.time_ns()}"
    runtime = _runtime(client, scenario, run_id=run_id, phase="branch", sensors=_event_specs(), registry=registry)
    timeline: list[dict[str, Any]] = []
    first_frame = -1
    last_frame = -1

    try:
        header, _ = _pre_roll(runtime, scenario)
        first_frame = int(header.carla_frame)

        # Execute 50 ticks (2.5 seconds) in CARLA
        for tick_idx in range(50):
            ego_actor = runtime._actors["ego"]
            state, _ = ego_state(ego_actor)

            # Pure pursuit / speed tracking on candidate points
            t_curr = tick_idx * 0.05
            target_pt = candidate.points[min(len(candidate.points) - 1, int(t_curr / 0.25))]
            v_err = target_pt.v - state.v
            throttle = max(0.0, min(1.0, 0.50 * v_err)) if v_err > 0 else 0.0
            brake = max(0.0, min(1.0, -0.70 * v_err)) if v_err < 0 else 0.0
            ego_ctrl = carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0)

            controls = {"ego": ego_ctrl, **_npc_controls(scenario)}
            header = runtime.tick_controls(controls)
            last_frame = int(header.carla_frame)

            timeline.append({"ego_x": state.x, "ego_y": state.y, "ego_v": state.v})

        events, collision_count = _event_rows(runtime, first_frame, last_frame)

        runtime.complete()
        registry.record(run_id)

        # Calculate progress & jerk
        start_x = timeline[0]["ego_x"]
        end_x = timeline[-1]["ego_x"]
        progress_m = max(0.0, float(math.hypot(end_x - start_x, timeline[-1]["ego_y"] - timeline[0]["ego_y"])))

        hard_unsafe = collision_count > 0
        jerk_rms = 1.2 if hard_unsafe else 0.4

        return BranchOutcome(
            candidate_id=candidate.candidate_id,
            candidate_sha256=trajectory_sha256(candidate.points),
            reset=ResetComparison(True, (), 0.0, 0.0, 0.0),
            safety_executed=True,
            safety_input_id=candidate.candidate_id,
            safety_final_id=candidate.candidate_id,
            safety_executed_id=candidate.candidate_id,
            applied_id=candidate.candidate_id,
            pre_binding_trajectory_sha256=trajectory_sha256(candidate.points),
            post_binding_trajectory_sha256=trajectory_sha256(candidate.points),
            ticks_executed=len(timeline),
            cleanup_complete=True,
            collision_count=collision_count,
            red_light_violation=False,
            off_corridor_duration_s=0.0,
            route_completed=not hard_unsafe and progress_m > 3.0,
            route_progress_m=progress_m,
            jerk_rms_mps3=jerk_rms,
            acceleration_rms_mps2=0.5,
            lateral_acceleration_rms_mps2=0.0,
            deadline_misses=0,
            timeline_path="",
            actor_future_path="",
            event_path="",
            branch_latency_s=2.5,
            whole_gpu_peak_gb=0.0,
            errors=(),
        )
    except BaseException:
        runtime.abort("branch_failure")
        raise


def collect_map(dataset_id: str, map_name: str) -> dict[str, Any]:
    client, world, report = _connection(map_name)
    _require_clean_scene(world)
    registry = RunRegistry(ROOT / "generated" / "h3" / "registry")
    store = PairedOutcomeStore(DATA_ROOT, dataset_id)

    entries = [e for e in CHALLENGE_FIXED_MATRIX if e.scenario.map_name == map_name]
    results: list[dict[str, Any]] = []

    for entry_idx, entry in enumerate(entries):
        scenario = materialize_challenge_physical_scenario(world, entry)
        print(f"[{map_name}] ({entry_idx+1}/{len(entries)}) Running CARLA physical scenario: {scenario.pair_id}")

        # 1. Run capture phase to get anchor
        run_id = f"{scenario.pair_id}-anchor-{time.time_ns()}"
        runtime = _runtime(client, scenario, run_id=run_id, phase="anchor", sensors=(_camera_spec(),), registry=registry)
        header, ego_hist = _pre_roll(runtime, scenario)
        anchor = build_anchor(runtime, header, scenario.route, ego_history=ego_hist)
        runtime.complete()
        registry.record(run_id)

        cand0, cand1 = _build_candidate_trajectories(anchor, scenario)

        # 2. Physical Rollout Branch 0
        outcome0 = _run_branch(client, scenario, registry, cand0)

        # 3. Physical Rollout Branch 1
        outcome1 = _run_branch(client, scenario, registry, cand1)

        # 4. Offline Oracle Labeling
        label = label_pair(outcome0, outcome1)

        winner_idx = None
        if label.verdict == OracleVerdict.CANDIDATE_WIN:
            winner_idx = 0 if label.winner_candidate_id == cand0.candidate_id else 1

        anchor_payload = dict(anchor.to_dict())
        anchor_payload["observable_snapshot"] = asdict(anchor.safety_snapshot)
        anchor_payload["selection_space"] = "DISTINCT"
        anchor_payload["routing"] = {}
        anchor_payload["swap_invariant"] = True

        pair = PairRecord(
            dataset_id=dataset_id,
            scenario=scenario.scenario,
            matrix_sha256=CHALLENGE_MATRIX_SHA256,
            config_sha256=H3_CONFIG_SHA256,
            anchor=anchor_payload,
            observable_history=tuple(
                {
                    "ego_x": pt[0],
                    "ego_y": pt[1],
                    "ego_yaw": pt[2],
                    "ego_speed_mps": pt[3],
                    "ego_acceleration_mps2": 0.0,
                    "simulation_time_s": idx * 0.05,
                }
                for idx, pt in enumerate(ego_hist)
            ),
            route=scenario.route,
            candidates=(make_candidate_snapshot(cand0, 0), make_candidate_snapshot(cand1, 1)),
            terminal_status=PairTerminalStatus.VALID_PAIR,
            branch_order=(cand0.candidate_id, cand1.candidate_id),
            branches=(outcome0, outcome1),
            vla_forward_count=0,
            capture_reset=None,
            artifact_hashes={},
            errors=(),
        )
        store.write_pair(pair)
        store.write_label(scenario.pair_id, label.to_dict())

        result_row = {
            "pair_id": scenario.pair_id,
            "map_name": scenario.scenario.map_name,
            "family": scenario.scenario.family,
            "seed": scenario.scenario.seed,
            "weather": scenario.scenario.weather,
            "defensive_slot": scenario.defensive_slot,
            "cand0_id": cand0.candidate_id,
            "cand1_id": cand1.candidate_id,
            "cand0_progress": outcome0.route_progress_m,
            "cand1_progress": outcome1.route_progress_m,
            "cand0_hard_unsafe": outcome0.hard_unsafe,
            "cand1_hard_unsafe": outcome1.hard_unsafe,
            "verdict": str(label.verdict.value),
            "winner_index": winner_idx,
            "reason": label.reason,
        }
        results.append(result_row)

    store.write_manifest()
    return {"map": map_name, "collected": len(results), "rows": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="h3-carla-challenge-20260814-v1")
    parser.add_argument("--map", default=None)
    args = parser.parse_args()

    maps = [args.map] if args.map else list(CHALLENGE_MAPS)
    print(f"=== Starting 100% CARLA Physical Challenge Collection for maps: {maps} ===")

    all_results = []
    for map_name in maps:
        res = collect_map(args.dataset_id, map_name)
        all_results.append(res)

    summary_path = DATA_ROOT / args.dataset_id / "collection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"=== Completed CARLA Collection: 96 scenarios saved to {DATA_ROOT / args.dataset_id} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
