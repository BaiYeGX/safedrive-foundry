#!/usr/bin/env python3
"""Real CARLA H5 closed-loop collector: off/on/defer arms for frozen scenarios.

This script is the live execution layer.  It does not read H4 test labels,
Oracle labels, or Regression inputs.  It only uses frozen physical scenarios,
frozen H4 World checkpoints, and the H5 router.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
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
    route_follow_steer,
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
from driving_vla.hybrid.router import ClassicOnlyRouter, FrozenH1Router  # noqa: E402
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
    diff_sha256 = hashlib.sha256(diff).hexdigest()
    head = _git_text("rev-parse", "HEAD")
    branch = _git_text("branch", "--show-current")
    return {
        "head": head,
        "commit": head,
        "branch": branch,
        "worktree_diff_sha256": diff_sha256,
        "untracked_manifest_sha256": stable_sha256(rows),
        "untracked_files": rows,
        # The scoped run-lock and every decision record use the same complete
        # identity.  Keep this separate from the scoped source hash: evidence
        # may be written after a run without changing the code lock, while the
        # full dirty-tree identity still makes that distinction auditable.
        "full_worktree_hash": stable_sha256(
            {
                "head": head,
                "diff_sha256": diff_sha256,
                "untracked": rows,
            }
        ),
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


class _CollectionNeedsUserAction(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(str(payload.get("failure_code", "NEEDS_USER_ACTION")))
        self.payload = dict(payload)


def _cleanup_retry_status(
    world: Any,
    *,
    lease_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Inspect cleanup state without destroying actors or advancing CARLA."""

    residue_ids: list[int] = []
    errors: list[str] = []
    try:
        residue_ids = sorted(
            int(actor.id)
            for actor in world.get_actors()
            if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker."))
            and bool(getattr(actor, "is_alive", True))
        )
    except Exception as exc:
        errors.append(f"actors_unavailable:{type(exc).__name__}")
    try:
        settings = world.get_settings()
        synchronous_mode = getattr(settings, "synchronous_mode", None)
        settings_confirmed = synchronous_mode is False
        if not settings_confirmed:
            errors.append(f"settings_not_restored:synchronous_mode={synchronous_mode!r}")
    except Exception as exc:
        synchronous_mode = None
        settings_confirmed = False
        errors.append(f"settings_unavailable:{type(exc).__name__}")

    active_owners: list[str] = []
    observed_owners: list[str] = []
    observed_paths: list[str] = []
    for lease_path in sorted({Path(item) for item in lease_paths}, key=str):
        if not lease_path.exists():
            continue
        observed_paths.append(str(lease_path))
        try:
            with lease_path.open("r", encoding="utf-8") as handle:
                try:
                    lease_payload = json.load(handle)
                    owner = str(lease_payload.get("owner") or "unknown")
                except (TypeError, ValueError, json.JSONDecodeError):
                    owner = "unknown"
                observed_owners.append(owner)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    active_owners.append(owner)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            errors.append(f"tick_lease_unavailable:{lease_path.name}:{type(exc).__name__}")
    lease_confirmed = not active_owners and not any(
        item.startswith("tick_lease_unavailable:") for item in errors
    )
    if active_owners:
        errors.append(f"tick_lease_active:{','.join(sorted(set(active_owners)))}")
    tick_owner = (
        f"active:{','.join(sorted(set(active_owners)))}"
        if active_owners
        else (
            f"free:last={','.join(sorted(set(observed_owners)))}"
            if observed_owners
            else "free:no_lease_files"
        )
    )
    clean = not residue_ids and settings_confirmed and lease_confirmed and not errors
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h6.collection_cleanup_status.v1",
        "status": "CLEAN_RETRY_ALLOWED" if clean else "NEEDS_USER_ACTION",
        "failure_code": None if clean else "CLEANUP_RESIDUE",
        "residue_ids": residue_ids,
        "tick_owner": tick_owner,
        "tick_owner_confirmed": lease_confirmed,
        "tick_lease_paths": observed_paths,
        "settings_synchronous_mode": synchronous_mode,
        "settings_confirmed": settings_confirmed,
        "tick_advanced": False,
        "cleanup_status": "CLEAN" if clean else "RESIDUE_OR_OWNER_UNCONFIRMED",
        "retry_status": "ALLOWED_ONCE" if clean else "STOPPED",
        "errors": errors,
    }
    payload["cleanup_sha256"] = stable_sha256(payload)
    return payload


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
        self.reset_for_arm()
        return self

    def reset_for_arm(self) -> None:
        """Restore the authored light state before every paired arm.

        CARLA's ambient light phase is not part of a non-red scenario.  Freeze
        all lights green there.  Dynamic red scenarios always restart green
        and switch at their declared tick, so an off-arm red cannot leak into
        the following on arm.
        """

        self.force_green()
        if self.scenario.red_light is not None and self.scenario.script.get(
            "red_at_capture", True
        ):
            self.force_red()

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


def _follow_ego_spectator(runtime: ScenarioRuntime, scenario: PhysicalScenario) -> None:
    """Move only CARLA's viewer behind the ego; never touch sensor inputs."""

    if not bool(scenario.script.get("spectator_follow_ego", False)):
        return
    ego_transform = runtime._actors["ego"].get_transform()
    yaw_rad = math.radians(float(ego_transform.rotation.yaw))
    distance = float(scenario.script.get("spectator_follow_distance_m", 8.0))
    height = float(scenario.script.get("spectator_follow_height_m", 4.0))
    location = carla.Location(
        x=float(ego_transform.location.x) - distance * math.cos(yaw_rad),
        y=float(ego_transform.location.y) - distance * math.sin(yaw_rad),
        z=float(ego_transform.location.z) + height,
    )
    rotation = carla.Rotation(
        pitch=float(scenario.script.get("spectator_follow_pitch_deg", -15.0)),
        yaw=float(ego_transform.rotation.yaw),
        roll=0.0,
    )
    runtime.world.get_spectator().set_transform(carla.Transform(location, rotation))


class _SpectatorFollower:
    """Keep the CARLA viewer attached while model/planner work is in flight."""

    def __init__(self, runtime: ScenarioRuntime, scenario: PhysicalScenario) -> None:
        self.runtime = runtime
        self.scenario = scenario
        self.enabled = bool(scenario.script.get("spectator_follow_ego", False))
        self.period_s = 1.0 / max(
            1.0, float(scenario.script.get("spectator_follow_hz", 20.0))
        )
        self.updates = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._update_once()
        self._thread = threading.Thread(
            target=self._run,
            name="sdf-spectator-follow",
            daemon=True,
        )
        self._thread.start()

    def _update_once(self) -> None:
        try:
            _follow_ego_spectator(self.runtime, self.scenario)
            self.updates += 1
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}:{exc}"

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self._update_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.25, 2.0 * self.period_s))
            self._thread = None


def _should_force_dynamic_red(scenario: PhysicalScenario, tick: int) -> bool:
    if not bool(scenario.script.get("dynamic_traffic_light_timing", False)):
        return False
    red_after_tick = scenario.script.get("red_after_tick")
    return red_after_tick is not None and int(tick) >= int(red_after_tick)


def _phase_boundaries(script: Mapping[str, Any]) -> dict[str, int]:
    """Return frozen, offline-only event boundaries from the scenario script."""

    keys = ("lead_brake_tick", "cutter_cut_in_tick", "red_after_tick", "cross_conflict_tick")
    values = {
        name: int(script[key])
        for name in keys
        if script.get(name) is not None
    }
    return values


def _offline_phase(script: Mapping[str, Any], tick: int, total_ticks: int) -> str:
    """Group evidence without exposing phase/family/source to World online."""

    boundaries = _phase_boundaries(script)
    if boundaries:
        event_tick = min(boundaries.values())
        if int(tick) < event_tick:
            return "pre_event"
        if int(tick) == event_tick:
            return "intervention"
        return "recovery"
    third = max(1, int(total_ticks) // 3)
    return "early" if int(tick) < third else "late" if int(tick) >= 2 * third else "middle"


def _actual_weather(runtime: ScenarioRuntime, scenario: PhysicalScenario) -> dict[str, float]:
    weather = runtime.world.get_weather()
    return {name: float(getattr(weather, name)) for name in sorted(scenario.weather)}


def _route_pose(route: Sequence[tuple[float, float]], progress_m: float):
    remaining = max(0.0, float(progress_m))
    for index in range(1, len(route)):
        x0, y0 = route[index - 1]
        x1, y1 = route[index]
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 1e-9:
            continue
        if remaining <= length:
            ratio = remaining / length
            return (
                x0 + ratio * (x1 - x0),
                y0 + ratio * (y1 - y0),
                math.atan2(y1 - y0, x1 - x0),
            )
        remaining -= length
    x0, y0 = route[-2]
    x1, y1 = route[-1]
    return x1, y1, math.atan2(y1 - y0, x1 - x0)


def _pre_roll_initial_progress(
    scenario: PhysicalScenario,
    ego_x: float,
    ego_y: float,
) -> float:
    """Choose the staging origin for a live arm.

    Kinematic H6 staging is a deterministic reset procedure, so it must start
    from the authored route origin.  Re-projecting the just-spawned actor can
    occasionally snap to the terminal segment on curved/loop-like CARLA
    routes, which teleports the arm to the route end before evaluation.
    """

    if bool(scenario.script.get("pre_roll_kinematic", False)):
        return 0.0
    return float(route_projection(ego_x, ego_y, scenario.route)[0])


def _pre_roll(
    runtime: ScenarioRuntime,
    scenario: PhysicalScenario,
    *,
    kinematic_settle_ticks: int = 0,
) -> tuple[Any, list[dict[str, Any]], list[tuple[float, float, float, float]]]:
    history: list[dict[str, Any]] = []
    ego_history: list[tuple[float, float, float, float]] = []
    header = None
    pre_roll_ticks = int(scenario.script.get("pre_roll_ticks", 20))
    min_ready_speed = scenario.script.get("pre_roll_min_ready_speed_mps")
    max_extra_ticks = int(scenario.script.get("pre_roll_max_extra_ticks", 0))
    total_limit = pre_roll_ticks + (max_extra_ticks if min_ready_speed is not None else 0)
    last_speed = 0.0
    kinematic = bool(scenario.script.get("pre_roll_kinematic", False))
    settle_ticks = int(kinematic_settle_ticks)
    if settle_ticks < 0:
        raise ValueError("kinematic_settle_ticks_must_be_non_negative")
    ego_actor = runtime._actors["ego"]
    initial_state, _ = ego_state(ego_actor)
    initial_progress = _pre_roll_initial_progress(
        scenario,
        initial_state.x,
        initial_state.y,
    )
    kinematic_speed = float(
        scenario.script.get(
            "pre_roll_kinematic_speed_mps",
            scenario.script.get("pre_roll_target_speed_mps", 3.0),
        )
    )
    if kinematic:
        ego_actor.set_simulate_physics(False)
    for index in range(total_limit):
        family = str(scenario.script.get("family", scenario.scenario.family))
        if family in {
            "free_flow", "slow_lead", "cut_in",
            "emergency_lead_brake", "aggressive_cut_in", "red_light_dilemma", "cross_traffic_conflict",
        }:
            target = float(scenario.script.get("pre_roll_target_speed_mps", 2.0))
            kp = float(scenario.script.get("pre_roll_kp", 0.30))
            current, _ = ego_state(runtime._actors["ego"])
            throttle = max(0.0, min(float(scenario.script.get("pre_roll_max_throttle", 0.35)), kp * (target - current.v)))
            steer = 0.0
            if bool(scenario.script.get("pre_roll_route_follow", False)):
                steer = route_follow_steer(
                    current.x,
                    current.y,
                    current.yaw,
                    current.v,
                    scenario.route,
                    min_lookahead_m=float(scenario.script.get("pre_roll_min_lookahead_m", 4.0)),
                    speed_lookahead_s=float(scenario.script.get("pre_roll_speed_lookahead_s", 1.2)),
                    max_normalized_steer=float(scenario.script.get("pre_roll_max_steer", 0.35)),
                )
            ego_control = carla.VehicleControl(throttle=throttle, brake=0.0, steer=steer)
        else:
            ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        if kinematic:
            progress = initial_progress + (index + 1) * kinematic_speed * PROFILE.fixed_delta_seconds
            x, y, yaw = _route_pose(scenario.route, progress)
            current_transform = ego_actor.get_transform()
            ego_actor.set_transform(
                carla.Transform(
                    carla.Location(x=x, y=y, z=float(current_transform.location.z)),
                    carla.Rotation(yaw=math.degrees(yaw)),
                )
            )
            ego_control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)
        controls = {"ego": ego_control, **_npc_controls(scenario, pre_roll=True)}
        header = runtime.tick_controls(controls)
        _follow_ego_spectator(runtime, scenario)
        state, acceleration = ego_state(ego_actor)
        if kinematic:
            state = type(state)(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                v=kinematic_speed,
                steer=state.steer,
            )
            acceleration = 0.0
        last_speed = float(state.v)
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
                "ego_steer": float(ego_control.steer),
            }
        )
        if (
            index + 1 >= pre_roll_ticks
            and (min_ready_speed is None or last_speed >= float(min_ready_speed))
        ):
            break
    assert header is not None
    # CARLA can expose the penultimate kinematic transform for one frame when
    # physics is toggled around a freshly spawned vehicle.  C2 requests one
    # opt-in settle tick that repeats the exact terminal pose, so capture and
    # every branch observe the same authored anchor without advancing the
    # world outside ScenarioRuntime.  Historical H5 callers keep the default
    # of zero and therefore retain their frozen behaviour.
    if kinematic and settle_ticks:
        terminal_progress = initial_progress + len(history) * kinematic_speed * PROFILE.fixed_delta_seconds
        terminal_x, terminal_y, terminal_yaw = _route_pose(scenario.route, terminal_progress)
        for _ in range(settle_ticks):
            current_transform = ego_actor.get_transform()
            ego_actor.set_transform(
                carla.Transform(
                    carla.Location(
                        x=terminal_x,
                        y=terminal_y,
                        z=float(current_transform.location.z),
                    ),
                    carla.Rotation(yaw=math.degrees(terminal_yaw)),
                )
            )
            controls = {
                "ego": carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0),
                **_npc_controls(scenario, pre_roll=True),
            }
            header = runtime.tick_controls(controls)
            _follow_ego_spectator(runtime, scenario)
            state, _ = ego_state(ego_actor)
            state = type(state)(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                v=kinematic_speed,
                steer=state.steer,
            )
            last_speed = float(state.v)
            ego_history.append((state.x, state.y, state.yaw, state.v))
            history.append(
                {
                    "index": len(history),
                    "carla_frame": int(header.carla_frame),
                    "simulation_time_s": float(header.simulation_time),
                    "ego_x": state.x,
                    "ego_y": state.y,
                    "ego_yaw": state.yaw,
                    "ego_speed_mps": state.v,
                    "ego_acceleration_mps2": 0.0,
                    "ego_steer": 0.0,
                }
            )
    if kinematic:
        ego_actor.set_simulate_physics(True)
        ego_actor.set_target_velocity(
            carla.Vector3D(
                x=kinematic_speed * math.cos(state.yaw),
                y=kinematic_speed * math.sin(state.yaw),
                z=0.0,
            )
        )
        ego_actor.set_target_angular_velocity(carla.Vector3D())
    if min_ready_speed is not None and last_speed < float(min_ready_speed):
        raise RuntimeError(
            f"PRE_ROLL_NOT_READY:speed={last_speed:.6f}<"
            f"{float(min_ready_speed):.6f}:ticks={len(history)}"
        )
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


def _make_pipeline(
    arm: str,
    scorer: NormalizedWorldScorer,
    *,
    classic_only_off: bool = False,
    router_config: Mapping[str, Any] | None = None,
) -> H1CandidatePipeline:
    if arm == "off":
        return H1CandidatePipeline(
            router=ClassicOnlyRouter() if classic_only_off else FrozenH1Router(),
            # H6 needs source-blind features for the actually executed Classic
            # baseline ticks, while the router remains strictly Classic-only.
            record_world_features=classic_only_off,
        )
    is_vla75 = bool(
        str(getattr(scorer, "schema_version", "")).endswith(
            "vla75.pair_exec.v1"
        )
    )
    frozen_router = dict(router_config or {})
    # Formal v75 runs use the development-selected values carried by the
    # training summary/run-lock.  Keep all historical v1 defaults untouched.
    vla75_ema_alpha = float(frozen_router.get("ema_alpha", 0.50))
    vla75_hold_ticks = int(frozen_router.get("hold_ticks", 10))
    vla75_hysteresis = float(frozen_router.get("hysteresis", 0.10))
    vla75_emergency_margin = float(
        frozen_router.get("emergency_switch_margin", 1.5)
    )
    if is_vla75 and not (
        0.0 < vla75_ema_alpha <= 1.0
        and vla75_hold_ticks >= 1
        and vla75_hysteresis >= 0.0
        and vla75_emergency_margin >= vla75_hysteresis
    ):
        raise ValueError("vla75_router_calibration_invalid")
    router = H5WorldRouter(
        scorer,
        min_hold_ticks=(
            vla75_hold_ticks
            if is_vla75
            else int(H5_CONFIG["router"]["min_hold_ticks"])
        ),
        # VLA75's temporal values are frozen by the development grid.  The
        # v1 H5 values remain untouched; this branch only activates for the
        # independent v2 scorer.
        hysteresis_margin=(
            vla75_hysteresis
            if is_vla75
            else float(H5_CONFIG["router"]["hysteresis_margin"])
        ),
        emergency_switch_margin=(
            vla75_emergency_margin
            if is_vla75
            else float(H5_CONFIG["router"]["emergency_switch_margin"])
        ),
        single_pass_grace_ticks=int(H5_CONFIG["router"]["single_pass_grace_ticks"]),
        force_defer=(arm == "defer"),
        scorer_deadline_ms=float(H5_CONFIG["runtime"]["scorer_deadline_ms"]),
        ema_alpha=vla75_ema_alpha if is_vla75 else None,
        vla75_mode=is_vla75,
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
    run_lock_sha256: str | None = None,
    run_schema_version: str = H5_CONFIG["schema_version"],
    physical_sha256: str | None = None,
    provenance_model_hash: str | None = None,
    provenance_feature_schema: str | None = None,
    worktree_identity_override: Mapping[str, Any] | None = None,
    router_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = f"{scenario.pair_id}-{arm}-{time.time_ns()}"
    started = time.perf_counter()
    runtime = _runtime(
        client, scenario, run_id=run_id, phase=arm, sensors=(_camera_spec(), *_event_specs()), registry=registry
    )
    spectator_follower = _SpectatorFollower(runtime, scenario)
    spectator_follower.start()
    # Formal v2 runs carry the immutable run-lock identity into every
    # decision.  Evidence files are written while collection is in progress
    # and would otherwise mutate the dynamic untracked manifest between the
    # first and last arm, making an honest lock appear to change mid-run.
    worktree_snapshot = dict(worktree_identity_override or _worktree_identity())
    timeline: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    executed_sources: list[str] = []
    world_incremental_gpu_samples: list[float] = []
    errors: list[str] = []
    previous_event_signature: tuple[Any, ...] | None = None
    first_frame = -1
    last_frame = -1
    cleanup_complete = False
    route_length = sum(
        math.hypot(scenario.route[i][0] - scenario.route[i - 1][0], scenario.route[i][1] - scenario.route[i - 1][1])
        for i in range(1, len(scenario.route))
    )
    phase_boundaries = _phase_boundaries(scenario.script)
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
            if _should_force_dynamic_red(scenario, tick):
                lights.force_red()
            anchor = build_anchor(runtime, header, scenario.route, ego_history=ego_history)
            seed = int(stable_sha256({"pair_id": scenario.pair_id, "tick": tick}), 16) % (2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generated = generate_hybrid_set(anchor, classic, vla)
            if len(generated.candidates) != 2 or not all(attempt.success for attempt in generated.attempts):
                failed = [attempt for attempt in generated.attempts if not attempt.success]
                errors.append(
                    f"generation_incomplete_tick:{tick}:"
                    + "|".join(f"{attempt.source.value}={attempt.error}" for attempt in failed)
                )
            event_signature = tuple(
                sorted(
                    (
                        item.provenance.source.value,
                        str(item.candidate.critical_actor or ""),
                        str(item.candidate.conflict_type or ""),
                        str(item.candidate.behavior or ""),
                    )
                    for item in generated.candidates
                )
            )
            event_break = bool(_should_force_dynamic_red(scenario, tick)) or (
                previous_event_signature is not None
                and event_signature != previous_event_signature
            )
            result = pipeline.decide(generated, event_break=event_break)
            previous_event_signature = event_signature
            routing = result.routing
            guarded = result.guarded_set
            selected_source = None
            if routing.selected_candidate_id is not None:
                for item in guarded.candidates:
                    if item.candidate.candidate_id == routing.selected_candidate_id:
                        selected_source = item.provenance.source.value
                        break
            safety_decision = result.safety.decision
            accepted_candidate = safety_decision.accepted_candidate
            executed_source = None
            if accepted_candidate is not None:
                if accepted_candidate.source.value.startswith("vla"):
                    executed_source = "vla"
                elif accepted_candidate.source.value == "classic":
                    executed_source = "expert"
            if executed_source is None:
                executed_source = "mrm"
            executed_sources.append(executed_source)
            ego, _ = ego_state(runtime._actors["ego"])
            # Safety receives the complete Guard-eligible preference order.
            # Passing only the routed head would make a same-tick Expert
            # fallback look orphaned and would prevent the kernel from
            # exercising its normal candidate-by-candidate final validation.
            safety_ids = set(getattr(result, "safety_input_ids", ()))
            selected_candidates = tuple(
                item.candidate
                for item in guarded.candidates
                if item.candidate.candidate_id in safety_ids
            )
            if not selected_candidates and routing.selected_candidate_id is not None:
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
            applied_mode = str(
                getattr(
                    getattr(applied, "applied_mode", ""),
                    "value",
                    getattr(applied, "applied_mode", ""),
                )
            )
            candidate_by_id = {
                item.candidate.candidate_id: item for item in guarded.candidates
            }

            def _candidate_source(candidate_id: str | None) -> str | None:
                if candidate_id is None:
                    return None
                item = candidate_by_id.get(str(candidate_id))
                if item is None:
                    if accepted_candidate is not None and str(accepted_candidate.candidate_id) == str(candidate_id):
                        source = str(accepted_candidate.source.value)
                        return "vla" if source.startswith("vla") else "expert" if source in {"classic", "expert"} else source
                    return None
                source = str(item.provenance.source.value)
                return "vla" if source.startswith("vla") else "expert" if source in {"classic", "expert"} else source

            safety_executed_id = safety_decision.executed_trajectory_id
            safety_executed_source = _candidate_source(safety_executed_id)
            applied_id = getattr(applied, "executed_id", None)
            applied_source = _candidate_source(applied_id)
            if applied_source is None:
                applied_source = "mrm"
            repair_payload = (
                None
                if result.safety.repair_result is None
                else result.safety.repair_result.to_dict()
            )
            repair_margins = tuple((repair_payload or {}).get("margins") or ())
            repair_final_validation = (
                None
                if repair_payload is None
                else bool(
                    repair_payload.get("success")
                    and all(
                        not bool(item.get("hard"))
                        or float(item.get("margin", -1.0)) >= 0.0
                        for item in repair_margins
                    )
                )
            )
            if repair_payload is not None:
                repair_payload = {
                    **repair_payload,
                    "final_validation": repair_final_validation,
                }
            router_metrics = pipeline.router.metrics() if hasattr(pipeline.router, "metrics") else {}
            scorer_latency = None
            scorer_disposition = None
            scorer_reason = None
            world_score = None
            if hasattr(pipeline.router, "last_score"):
                score = pipeline.router.last_score
                if score is not None:
                    scorer_latency = float(score.latency_ms)
                    scorer_disposition = str(score.disposition)
                    scorer_reason = score.defer_reason
                    if hasattr(score, "to_dict"):
                        world_score = score.to_dict()
            world_scorer = getattr(pipeline.router, "scorer", None)
            model_hash = (
                None
                if world_score is None
                else world_score.get("model_hash")
            ) or getattr(world_scorer, "model_hash", None)
            feature_schema = (
                None
                if world_score is None
                else world_score.get("feature_schema")
            ) or getattr(world_scorer, "feature_schema", None)
            if feature_schema is None and world_scorer is not None:
                feature_schema = "safedrive.h3.world_scorer.v2"
            model_hash = model_hash or provenance_model_hash
            feature_schema = feature_schema or provenance_feature_schema
            world_incremental_gpu_gib = (
                None
                if world_score is None
                else world_score.get("world_incremental_gpu_gib")
            )
            if world_incremental_gpu_gib is None:
                world_incremental_gpu_gib = getattr(
                    world_scorer, "last_incremental_gpu_gib", None
                )
            if world_incremental_gpu_gib is not None:
                try:
                    value = float(world_incremental_gpu_gib)
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    world_incremental_gpu_samples.append(value)
            raw_preferred_id = routing.raw_preferred_candidate_id
            if raw_preferred_id is None and isinstance(world_score, Mapping):
                raw_order = world_score.get("raw_preference_order") or world_score.get(
                    "raw_preference_order", ()
                )
                if raw_order:
                    raw_preferred_id = str(raw_order[0])
            raw_preferred_source = routing.raw_preferred_source
            if raw_preferred_source is None and raw_preferred_id is not None:
                raw_item = candidate_by_id.get(str(raw_preferred_id))
                if raw_item is not None:
                    value = str(raw_item.provenance.source.value)
                    raw_preferred_source = (
                        "vla" if value.startswith("vla")
                        else "expert" if value in {"classic", "expert"}
                        else value
                    )
            raw_gate_reasons = dict(routing.raw_gate_reasons)
            if not raw_gate_reasons and isinstance(world_score, Mapping):
                raw_gate_reasons = dict(world_score.get("raw_gate_reasons") or {})
            if run_schema_version == "safedrive.h6.vla75.run.v2" and not raw_gate_reasons:
                # The Classic-only baseline intentionally does not invoke
                # World, but its v2 decision record still has to carry a
                # complete, explicit raw-gate audit rather than an omitted
                # field that could be mistaken for missing provenance.
                raw_gate_reasons = {
                    "score": False,
                    "pair_preference": False,
                    "trust": False,
                    "risk": False,
                    "pair_completeness": False,
                }
            controls = {
                "ego": carla.VehicleControl(
                    throttle=float(applied.throttle), brake=float(applied.brake), steer=float(applied.steer),
                    reverse=bool(getattr(applied, "reverse", False)),
                ),
                **_npc_controls(scenario, tick_index=tick),
            }
            tick_started = time.perf_counter()
            header = runtime.tick_controls(controls)
            _follow_ego_spectator(runtime, scenario)
            tick_wall_ms = (time.perf_counter() - tick_started) * 1000.0
            state, acceleration = ego_state(runtime._actors["ego"])
            progress, corridor_distance = route_projection(state.x, state.y, scenario.route)
            previous_yaw = ego_history[-1][2] if ego_history else state.yaw
            yaw_rate = ((state.yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi) / 0.05
            ego_history.append((state.x, state.y, state.yaw, state.v))
            timeline.append(
                {
                    "tick": tick,
                    "phase": _offline_phase(
                        scenario.script, tick, int(H5_CONFIG["matrix"]["decision_ticks"])
                    ),
                    "carla_frame": int(header.carla_frame),
                    "simulation_time_s": float(header.simulation_time),
                    "x": state.x, "y": state.y, "yaw": state.yaw, "speed_mps": state.v,
                    "acceleration_mps2": acceleration,
                    "lateral_acceleration_mps2": state.v * yaw_rate,
                    "route_progress_m": progress,
                    "corridor_distance_m": corridor_distance,
                    "throttle": float(applied.throttle), "brake": float(applied.brake), "steer": float(applied.steer),
                    "control_mode": applied_mode,
                    "deadline_miss": False,
                    "tick_wall_ms": tick_wall_ms,
                }
            )
            decisions.append(
                {
                    "tick": tick,
                    "phase": _offline_phase(
                        scenario.script, tick, int(H5_CONFIG["matrix"]["decision_ticks"])
                    ),
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
                    "selected_candidate_id": routing.selected_candidate_id,
                    "selected_candidate_source": selected_source,
                    "raw_preferred_candidate_id": raw_preferred_id,
                    "raw_preferred_source": raw_preferred_source,
                    "raw_gate_reasons": raw_gate_reasons,
                    "stabilized_preferred_candidate_id": routing.stabilized_preferred_candidate_id,
                    "stabilized_preferred_source": routing.stabilized_preferred_source,
                    "safety_executed_candidate_id": safety_executed_id,
                    "safety_executed_source": safety_executed_source or "mrm",
                    "applied_candidate_id": applied_id,
                    "applied_source": applied_source,
                    "applied_candidate_source": applied_source,
                    "executed_candidate_id": safety_decision.executed_trajectory_id,
                    "executed_source": executed_source,
                    "red_light_active": bool(
                        scenario.red_light is not None
                        and (
                            bool(scenario.script.get("red_at_capture", True))
                            or _should_force_dynamic_red(scenario, tick)
                        )
                    ),
                    "scorer_latency_ms": scorer_latency,
                    "scorer_disposition": scorer_disposition,
                    "scorer_defer_reason": scorer_reason,
                    "world_score": world_score,
                    "world_incremental_gpu_gib": world_incremental_gpu_gib,
                    "model_hash": model_hash,
                    "feature_schema": feature_schema,
                    "worktree_hash": worktree_snapshot.get("full_worktree_hash"),
                    "safety_decision_kind": str(getattr(safety_decision.decision_kind, "value", safety_decision.decision_kind)),
                    "applied_mode": applied_mode,
                    "switch_count": int(router_metrics.get("switch_count", 0)),
                    "defer_count": int(router_metrics.get("defer_count", 0)),
                    "fallback_count": int(router_metrics.get("defer_count", 0)) if arm != "off" else 0,
                    "generation_latency_s": {
                        attempt.source.value: attempt.generation_latency_s for attempt in generated.attempts
                    },
                    "generation_attempts": [attempt.to_dict() for attempt in generated.attempts],
                    "candidates": {
                        item.candidate.candidate_id: item.to_dict() for item in guarded.candidates
                    },
                    "world_features": {
                        key: {"context": list(value[0]), "candidate": [list(row) for row in value[1]]}
                        for key, value in (result.world_features or {}).items()
                    },
                    "repair": repair_payload,
                    "repair_input_id": None if repair_payload is None else repair_payload.get("pre_repair_id"),
                    "repair_output_id": None if repair_payload is None else repair_payload.get("post_repair_id"),
                    "repair_method": None if repair_payload is None else repair_payload.get("mode"),
                    "repair_success": None if repair_payload is None else bool(repair_payload.get("success")),
                    "repair_final_validation": repair_final_validation,
                    "candidate_hashes": {
                        item.candidate.candidate_id: item.provenance.canonical_sha256
                        for item in guarded.candidates
                    },
                    "arbitration": None if result.safety.arbitration is None else result.safety.arbitration.to_dict(),
                    "tick_wall_ms": tick_wall_ms,
                }
            )
        first_frame = int(timeline[0]["carla_frame"]) if timeline else -1
        last_frame = int(timeline[-1]["carla_frame"]) if timeline else -1
        event_rows, collision_count = _event_rows(runtime, first_frame, last_frame) if first_frame >= 0 else ([], 0)
        spectator_follower.stop()
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
        safety_fallback_count = sum(
            1 for decision in decisions
            if decision.get("applied_mode") != "TRACK_APPROVED"
        )
        router_metrics = pipeline.router.metrics() if hasattr(pipeline.router, "metrics") else {}
        fallback_count = int(router_metrics.get("defer_count", 0)) if arm != "off" else 0
        actual_switch_count = sum(
            executed_sources[index] != executed_sources[index - 1]
            for index in range(1, len(executed_sources))
            if run_schema_version == "safedrive.h6.vla75.run.v2"
            or "mrm" not in {executed_sources[index], executed_sources[index - 1]}
        )
        payload = {
            "schema_version": run_schema_version,
            "dataset_id": "",
            "run_id": run_id,
            "pair_id": scenario.pair_id,
            "scenario": scenario.scenario.to_dict(),
            "physical_sha256": physical_sha256 or scenario.physical_sha256,
            "manifest_kind": "",
            "arm": arm,
            "arm_order_index": arm_order_index,
            "reset_signature": current_reset.to_dict(),
            "reset_comparison": None if reset_cmp is None else reset_cmp.to_dict(),
            "initial_route_progress_m": initial_progress,
            "red_light_stop_progress_m": (
                None
                if scenario.red_light is None
                else float(scenario.red_light["stop_progress_m"])
            ),
            "phase_boundaries": phase_boundaries,
            "route_progress_m": final_progress,
            "route_completed": final_progress >= route_length - 2.0,
            "collision_count": collision_count,
            "red_light_violation": red_violation,
            "off_corridor_duration_s": off_duration,
            "jerk_rms_mps3": metrics["jerk_rms_mps3"],
            "acceleration_rms_mps2": metrics["acceleration_rms_mps2"],
            "lateral_acceleration_rms_mps2": metrics["lateral_acceleration_rms_mps2"],
            "switch_count": int(router_metrics.get("switch_count", 0)),
            "actual_switch_count": actual_switch_count,
            "defer_count": int(router_metrics.get("defer_count", 0)),
            "fallback_count": fallback_count,
            "safety_fallback_count": safety_fallback_count,
            "deadline_misses": 0,
            "scorer_deadline_misses": scorer_deadline_misses,
            "p50_scorer_ms": sorted(scorer_ms)[len(scorer_ms)//2] if scorer_ms else None,
            "p95_scorer_ms": sorted(scorer_ms)[int(len(scorer_ms)*0.95)] if scorer_ms else None,
            "p99_scorer_ms": sorted(scorer_ms)[int(len(scorer_ms)*0.99)] if scorer_ms else None,
            "whole_gpu_peak_gb": 0.0 if gpu_sampler is None else gpu_sampler.peak_used_gib(),
            "world_incremental_gpu_gib": (
                max(world_incremental_gpu_samples)
                if world_incremental_gpu_samples
                else None
            ),
            "vla_forward_count": policy.forward_count - before_forward,
            "vla_executed_ticks": executed_sources.count("vla"),
            "expert_executed_ticks": executed_sources.count("expert"),
            "mrm_ticks": executed_sources.count("mrm"),
            "ticks_executed": len(timeline),
            "cleanup_complete": cleanup_complete,
            "ok": (
                cleanup_complete
                and len(timeline) == int(H5_CONFIG["matrix"]["decision_ticks"])
                and not any(str(error).startswith("generation_incomplete_tick") for error in errors)
            ),
            "errors": tuple(errors),
            "pre_roll": tuple(history),
            "decisions": tuple(decisions),
            "timeline": tuple(timeline),
            "events": tuple(event_rows),
            "spectator_follow_updates": spectator_follower.updates,
            "spectator_follow_error": spectator_follower.last_error,
            "worktree": worktree_snapshot,
            "config_sha256": H5_CONFIG_SHA256,
            "run_lock_sha256": run_lock_sha256,
            "router_calibration": dict(router_config or {}),
        }
        return payload
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
        spectator_follower.stop()
        runtime.abort(type(exc).__name__)
        cleanup = registry.record(run_id)
        return {
            "schema_version": run_schema_version,
            "dataset_id": "",
            "run_id": run_id,
            "pair_id": scenario.pair_id,
            "scenario": scenario.scenario.to_dict(),
            "physical_sha256": physical_sha256 or scenario.physical_sha256,
            "manifest_kind": "",
            "arm": arm,
            "arm_order_index": arm_order_index,
            "reset_signature": {},
            "reset_comparison": None,
            "initial_route_progress_m": None,
            "red_light_stop_progress_m": None,
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
            "world_incremental_gpu_gib": None,
            "vla_forward_count": 0,
            "ticks_executed": 0,
            "cleanup_complete": False,
            "ok": False,
            "errors": tuple(errors),
            "pre_roll": (),
            "decisions": (),
            "timeline": (),
            "events": (),
            "spectator_follow_updates": spectator_follower.updates,
            "spectator_follow_error": spectator_follower.last_error,
            "worktree": worktree_snapshot,
            "config_sha256": H5_CONFIG_SHA256,
            "run_lock_sha256": run_lock_sha256,
            "router_calibration": dict(router_config or {}),
        }


def _collect_map_impl(
    dataset_id: str,
    map_name: str,
    scope: str,
    *,
    pair_id: str | None = None,
    gpu_sampler: GPUMemorySampler | None = None,
    scorer_override: Any | None = None,
    arms_override: Sequence[str] | None = None,
    experiment_config_sha256: str = H5_CONFIG_SHA256,
    evidence_root: Path = EVIDENCE_ROOT,
    scenarios_override: Sequence[Any] | None = None,
    scenario_materializer: Any | None = None,
    classic_only_off: bool = False,
    run_lock_sha256: str | None = None,
    run_schema_version: str = H5_CONFIG["schema_version"],
    worktree_identity_override: Mapping[str, Any] | None = None,
    router_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    store = H5Store(DATA_ROOT, dataset_id)
    scenarios = (
        tuple(scenarios_override)
        if scenarios_override is not None
        else load_h5_matrix(ROOT, full=(scope == "full"))
    )
    rows = [s for s in scenarios if s.scenario.map_name == map_name]
    if pair_id:
        rows = [s for s in rows if s.pair_id == pair_id]
    if not rows:
        raise RuntimeError(f"NO_SCENARIOS_FOR_MAP:{map_name}")
    client, world, report = _connection(map_name)
    _require_clean_scene(world)
    evidence_dir = evidence_root / dataset_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(evidence_dir / f"run_registry_{map_name}_{scope}.sqlite3")
    policy = NominalVLAPolicy(keep_on_gpu=True)
    policy.ensure_loaded()
    classic = ClassicExpertGenerator()
    vla = NominalVLAGenerator(policy, generator_hash=simlingo_generator_hash(policy))
    scorer = scorer_override or _load_scorer()
    results = []
    for scenario in rows:
        physical = scenario.physical
        if physical is None and scenario_materializer is not None:
            physical = scenario_materializer(world, scenario)
        if physical is None:
            raise RuntimeError(f"physical_scenario_missing:{scenario.pair_id}")
        reference_reset = None
        with _FrozenTrafficLights(world, physical) as lights:
            arms = tuple(arms_override) if arms_override is not None else scenario.arm_order
            for arm_index, arm in enumerate(arms):
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
                lights.reset_for_arm()
                pipeline = _make_pipeline(
                    arm,
                    scorer,
                    classic_only_off=classic_only_off,
                    router_config=router_config,
                )
                run = None
                for attempt in range(2):
                    try:
                        run = _run_arm(
                            client, physical, arm=arm, arm_order_index=arm_index,
                            pipeline=pipeline, classic=classic, vla=vla, policy=policy,
                            registry=registry, lights=lights, reference_reset=reference_reset,
                            gpu_sampler=gpu_sampler,
                            run_lock_sha256=run_lock_sha256,
                            run_schema_version=run_schema_version,
                            physical_sha256=physical.physical_sha256,
                            provenance_model_hash=getattr(scorer, "model_hash", None),
                            provenance_feature_schema=getattr(
                                scorer, "feature_schema", "safedrive.h3.world_scorer.v2"
                            ),
                            worktree_identity_override=worktree_identity_override,
                            router_config=router_config,
                        )
                        break
                    except Exception as exc:
                        if attempt >= 1:
                            raise
                        # Infrastructure-only retry is permitted once and
                        # only after Runtime has already left a clean scene.
                        # The collector never repairs cleanup by owning a tick
                        # or destroying actors outside ScenarioRuntime.
                        cleanup = _cleanup_retry_status(
                            world,
                            lease_paths=(
                                ROOT / ".runtime/tick-lease.lock",
                                *ROOT.glob(".runtime/tick-lease-h5-*.lock"),
                            ),
                        )
                        if cleanup["status"] != "CLEAN_RETRY_ALLOWED":
                            raise _CollectionNeedsUserAction(
                                {
                                    **cleanup,
                                    "status": "NEEDS_USER_ACTION",
                                    "failure_code": "CLEANUP_RESIDUE",
                                    "initial_error": f"{type(exc).__name__}:{exc}",
                                }
                            ) from exc
                assert run is not None
                run["dataset_id"] = dataset_id
                run["manifest_kind"] = scenario.manifest_kind
                run["config_sha256"] = experiment_config_sha256
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
        "config_sha256": experiment_config_sha256,
        "run_lock_sha256": run_lock_sha256,
        "worktree": _worktree_identity(),
        "matrix_sha256": h5_matrix_sha256(scenarios),
        "results": results,
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    _atomic_json(evidence_dir / f"collect-{scope}-{map_name}.json", payload)
    return {"ok": True, "map": map_name, "scope": scope, "pairs": len(rows), "evidence_sha256": payload["evidence_sha256"]}


def collect_map(
    dataset_id: str,
    map_name: str,
    scope: str,
    *,
    pair_id: str | None = None,
    scorer_override: Any | None = None,
    arms_override: Sequence[str] | None = None,
    experiment_config_sha256: str = H5_CONFIG_SHA256,
    evidence_root: Path = EVIDENCE_ROOT,
    scenarios_override: Sequence[Any] | None = None,
    scenario_materializer: Any | None = None,
    classic_only_off: bool = False,
    run_lock_sha256: str | None = None,
    run_schema_version: str = H5_CONFIG["schema_version"],
    worktree_identity_override: Mapping[str, Any] | None = None,
    router_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sampler = GPUMemorySampler(interval_s=0.1, gpu_index=0).start()
    evidence_dir = evidence_root / dataset_id
    try:
        result = _collect_map_impl(
            dataset_id,
            map_name,
            scope,
            pair_id=pair_id,
            gpu_sampler=sampler,
            scorer_override=scorer_override,
            arms_override=arms_override,
            experiment_config_sha256=experiment_config_sha256,
            evidence_root=evidence_root,
            scenarios_override=scenarios_override,
            scenario_materializer=scenario_materializer,
            classic_only_off=classic_only_off,
            run_lock_sha256=run_lock_sha256,
            run_schema_version=run_schema_version,
            worktree_identity_override=worktree_identity_override,
            router_config=router_config,
        )
    except BaseException as exc:
        gpu = sampler.stop()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        cleanup_payload = (
            dict(exc.payload)
            if isinstance(exc, _CollectionNeedsUserAction)
            else {
                "status": "FAILED",
                "failure_code": "COLLECTION_EXCEPTION",
                "residue_ids": [],
                "tick_owner": "ScenarioRuntime",
                "tick_owner_confirmed": False,
                "tick_advanced": False,
                "cleanup_status": "NOT_MEASURED",
                "retry_status": "STOPPED",
            }
        )
        failure_payload = {
            "schema_version": "safedrive.h6.collection_failure.v1",
            "dataset_id": dataset_id,
            "map_name": map_name,
            "scope": scope,
            "error": f"{type(exc).__name__}:{exc}",
            "gpu": gpu,
            **cleanup_payload,
        }
        failure_payload.pop("cleanup_sha256", None)
        failure_payload["failure_sha256"] = stable_sha256(failure_payload)
        _atomic_json(
            evidence_dir / f"collect-{scope}-{map_name}-failure.json",
            failure_payload,
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
