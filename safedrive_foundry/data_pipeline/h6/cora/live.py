"""Live C2 collector implementation; deliberately contains no offline Oracle."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import contextlib
from enum import Enum
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import carla
import torch

from classic_stack.control.controller import ControlLoop
from data_pipeline.h2.contracts import compare_reset_signatures, stable_sha256
from data_pipeline.h2.carla_scenarios import PhysicalScenario
from data_pipeline.h2.gpu import GPUMemorySampler
from data_pipeline.h2.live_contract import (
    kinematic_metrics,
    reset_signature,
    route_projection,
    trajectory_sha256,
)
from data_pipeline.h2.store import file_sha256
from driving_vla.hybrid import (
    ClassicExpertGenerator,
    NominalVLAGenerator,
    generate_hybrid_set,
    simlingo_generator_hash,
)
from driving_vla.hybrid.carla_anchor import (
    build_anchor,
    ego_state,
    image_png_bytes,
    map_basename,
    safety_snapshot,
)
from driving_vla.hybrid.guard import CandidateGuard
from driving_vla.model.nominal_policy import NominalVLAPolicy
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime
from driving_vla.runtime.safety_control_bind import (
    AppliedMode,
    apply_safety_control,
    resolve_executable_candidate,
    safety_points_to_ctrl,
)
from runtime import (
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver, READY
from safety_kernel.contracts.serialize import candidate_to_dict
from safety_kernel.contracts.types import ComponentAvailability, PolicyCandidateSet
from safety_kernel.kernel import SafetyKernel

from scripts.h5_collect import (
    _FrozenTrafficLights,
    _actor_specs,
    _actors_by_role,
    _actual_weather,
    _camera_spec,
    _event_rows,
    _event_specs,
    _follow_ego_spectator,
    _npc_controls,
    _pre_roll,
    _should_force_dynamic_red,
)

from .config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256, config_identity
from .contracts import (
    CoraBranchOutcome,
    CoraPairEdge,
    CoraProposal,
    CoraRootRecord,
    OutcomeValue,
)
from .feature import build_cora_feature_view
from .interventions import InterventionResult, derive_interventions
from .matrix import (
    CORA_DATA_MATRIX,
    CORA_FORMAL_MATRIX,
    CORA_MATRIX_SHA256,
    CORA_SMOKE_ROOT_IDS,
    CoraMatrixRow,
)
from .run_lock import (
    build_run_lock,
    source_identity,
    verify_run_lock,
    verify_source_amendment,
)
from .scenarios import materialize_cora_physical_scenario
from .store import CoraDataStore, pending_collection_rows


ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / "generated" / "h6" / "cora"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h6" / str(CORA_C2_CONFIG["dataset_id"])
PROFILE = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["cora_data"]
COLLECTION_SCHEMA = "safedrive.cora.collection_summary.v1"
PHYSICAL_PART_SCHEMA = "safedrive.cora.physical_manifest_part.v1"
PHYSICAL_SCHEMA = "safedrive.cora.physical_manifest.v1"
SENSOR_CONTRACT_SHA256 = stable_sha256(
    {
        "front_camera": {
            "native": [1024, 512],
            "fov_deg": 110.0,
            "mount_xyz": [-1.5, 0.0, 2.0],
        },
        "branch_events": ["collision", "lane_invasion"],
    }
)
WORLD_SETTINGS_SHA256 = stable_sha256(
    {
        "profile": "cora_data",
        "synchronous_mode": PROFILE.synchronous_mode,
        "substepping": PROFILE.substepping,
        "fixed_delta_seconds": PROFILE.fixed_delta_seconds,
        "max_substep_delta_time": PROFILE.max_substep_delta_time,
        "max_substeps": PROFILE.max_substeps,
    }
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


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
        payload = {
            "schema_version": "safedrive.cora.needs_user_action.v1",
            "dataset_id": CORA_C2_CONFIG["dataset_id"],
            "reason": "unexpected_live_actor_residue",
            "actor_ids": residue,
            "timestamp_ns": time.time_ns(),
        }
        payload["evidence_sha256"] = stable_sha256(payload)
        _atomic_json(EVIDENCE_ROOT / f"needs-user-action-{time.time_ns()}.json", payload)
        raise RuntimeError(f"NEEDS_USER_ACTION:LIVE_ACTOR_RESIDUE:{residue}")


def _dataset_store(dataset_id: str) -> CoraDataStore:
    if dataset_id != str(CORA_C2_CONFIG["dataset_id"]):
        raise ValueError("cora_dataset_id_not_frozen")
    return CoraDataStore(DATA_ROOT, dataset_id)


def _part_path(store: CoraDataStore, map_name: str) -> Path:
    return store.root / "scenario-parts" / f"{map_name}.json"


def materialize_map(dataset_id: str, map_name: str) -> dict[str, Any]:
    store = _dataset_store(dataset_id)
    client, world, report = _connection(map_name)
    del client
    _require_clean_scene(world)
    selected = [row for row in CORA_DATA_MATRIX if row.scenario.map_name == map_name]
    physical = [materialize_cora_physical_scenario(world, row) for row in selected]
    if len(physical) != 117:
        raise RuntimeError(f"CORA_MATERIALIZATION_COUNT:{map_name}:{len(physical)}")
    payload: dict[str, Any] = {
        "schema_version": PHYSICAL_PART_SCHEMA,
        "dataset_id": dataset_id,
        "map_name": map_name,
        "config": config_identity(),
        "config_sha256": CORA_C2_CONFIG_SHA256,
        "matrix_sha256": CORA_MATRIX_SHA256,
        "source": source_identity(ROOT),
        "connection": report.to_dict(),
        "rows": [item.to_dict() for item in physical],
    }
    payload["part_sha256"] = stable_sha256(payload)
    path = _part_path(store, map_name)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"cora_physical_part_conflict:{path}")
    else:
        store.write_immutable_json(path, payload)
    store.write_manifest()
    return {"ok": True, "map": map_name, "rows": len(physical), "part_sha256": payload["part_sha256"]}


def _model_identity() -> dict[str, Any]:
    runtime = SimLingoNeuralRuntime()
    checkpoint = Path(runtime.ckpt_path)
    hydra = Path(runtime.hydra_config)
    internvl = Path(runtime.internvl_root)
    if not checkpoint.is_file() or not hydra.is_file() or not internvl.is_dir():
        raise FileNotFoundError("cora_nominal_vla_assets_missing")
    config_candidates = sorted(
        path for path in internvl.glob("*.json") if path.is_file()
    )
    return {
        "model_id": NominalVLAPolicy.model_id,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "hydra_path": str(hydra.relative_to(ROOT)),
        "hydra_sha256": file_sha256(hydra),
        "internvl_root": str(internvl.relative_to(ROOT)),
        "internvl_config": [
            {"path": str(path.relative_to(internvl)), "sha256": file_sha256(path)}
            for path in config_candidates
        ],
    }


def _component_hashes() -> dict[str, str]:
    relative_paths = (
        "safedrive_foundry/classic_stack/planning/frenet/planner.py",
        "safedrive_foundry/driving_vla/hybrid/generators.py",
        "safedrive_foundry/driving_vla/hybrid/guard.py",
        "safedrive_foundry/driving_vla/runtime/safety_control_bind.py",
        "safedrive_foundry/safety_kernel/kernel.py",
        "safedrive_foundry/classic_stack/control/controller.py",
        "safedrive_foundry/runtime/scenario_runtime.py",
    )
    output: dict[str, str] = {}
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = file_sha256(path)
    return output


def _environment_identity(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10.0,
    )
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError(f"cora_gpu_identity_failed:{gpu.returncode}:{gpu.stderr.strip()}")
    return {
        "carla_client_version": getattr(carla, "__version__", "0.9.16"),
        "connections": {str(part["map_name"]): dict(part["connection"]) for part in parts},
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_query": gpu.stdout.strip(),
        "profile": "cora_data",
        "world_settings_sha256": WORLD_SETTINGS_SHA256,
        "sensor_contract_sha256": SENSOR_CONTRACT_SHA256,
        "tick_owner": "sdf.h6.cora.collector",
        "ros_tick_master_allowed": False,
    }


def _disk_identity(store: CoraDataStore) -> dict[str, Any]:
    usage = shutil.disk_usage(store.root)
    return {
        "path": str(store.root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / 1024**3,
        "required_floor_gib": CORA_C2_CONFIG["resources"]["free_disk_floor_gib"],
    }


def freeze_manifest(dataset_id: str) -> dict[str, Any]:
    store = _dataset_store(dataset_id)
    parts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for map_name in CORA_C2_CONFIG["maps"]:
        part = json.loads(_part_path(store, str(map_name)).read_text(encoding="utf-8"))
        expected = part.pop("part_sha256", None)
        if expected != stable_sha256(part):
            raise ValueError(f"cora_physical_part_hash:{map_name}")
        part["part_sha256"] = expected
        if part.get("config_sha256") != CORA_C2_CONFIG_SHA256 or part.get("matrix_sha256") != CORA_MATRIX_SHA256:
            raise ValueError(f"cora_physical_part_identity:{map_name}")
        parts.append(part)
        rows.extend(part["rows"])
    if len(rows) != 351 or len({str(row["pair_id"]) for row in rows}) != 351:
        raise ValueError("cora_physical_manifest_count")
    source_ids = {stable_sha256(part["source"]) for part in parts}
    if len(source_ids) != 1 or stable_sha256(source_identity(ROOT)) not in source_ids:
        raise ValueError("cora_source_changed_during_materialization")
    physical: dict[str, Any] = {
        "schema_version": PHYSICAL_SCHEMA,
        "dataset_id": dataset_id,
        "config_sha256": CORA_C2_CONFIG_SHA256,
        "matrix_sha256": CORA_MATRIX_SHA256,
        "formal_matrix_sha256": stable_sha256([row.to_dict() for row in CORA_FORMAL_MATRIX]),
        "formal_collected": False,
        "part_sha256": {str(part["map_name"]): part["part_sha256"] for part in parts},
        "source": parts[0]["source"],
        "rows": sorted(rows, key=lambda row: int(row["matrix_index"])),
    }
    physical["physical_manifest_sha256"] = stable_sha256(physical)
    physical_path = store.root / "scenario-manifest.json"
    store.write_immutable_json(physical_path, physical)
    physical_parquet = store.root / "scenario-manifest.parquet"
    if not physical_parquet.exists():
        store._atomic_parquet(
            physical_parquet,
            [
                {
                    "root_id": row["pair_id"],
                    "physical_sha256": row["physical_sha256"],
                    "record_json": json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                }
                for row in physical["rows"]
            ],
        )
    disk = _disk_identity(store)
    if float(disk["free_gib"]) < float(CORA_C2_CONFIG["resources"]["free_disk_floor_gib"]):
        raise RuntimeError(f"cora_disk_below_floor:{disk['free_gib']:.3f}")
    lock = build_run_lock(
        ROOT,
        environment=_environment_identity(parts),
        model=_model_identity(),
        component_hashes=_component_hashes(),
        disk=disk,
    )
    valid, failures = verify_run_lock(lock)
    if not valid:
        raise ValueError(f"cora_run_lock_invalid:{failures}")
    store.write_immutable_json(store.root / "run-lock.json", lock)
    _atomic_json(EVIDENCE_ROOT / "run-lock.json", lock)
    store.write_manifest()
    return {
        "ok": True,
        "rows": len(rows),
        "physical_manifest_sha256": physical["physical_manifest_sha256"],
        "run_lock_sha256": lock["run_lock_sha256"],
    }


def _load_frozen(store: CoraDataStore) -> tuple[dict[str, PhysicalScenario], dict[str, Any], dict[str, Any]]:
    physical_path = store.root / "scenario-manifest.json"
    lock_path = store.root / "run-lock.json"
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    expected_physical = physical.pop("physical_manifest_sha256", None)
    if expected_physical != stable_sha256(physical):
        raise ValueError("cora_physical_manifest_self_hash")
    physical["physical_manifest_sha256"] = expected_physical
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    valid, failures = verify_run_lock(lock)
    if not valid:
        raise ValueError(f"cora_run_lock:{failures}")
    if stable_sha256(source_identity(ROOT)) != stable_sha256(lock["source"]):
        amendment_paths = sorted(EVIDENCE_ROOT.glob("source-amendment-*.json"))
        amendment_failures: list[str] = []
        accepted = 0
        for amendment_path in amendment_paths:
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            amendment_valid, failures = verify_source_amendment(ROOT, lock, amendment)
            if amendment_valid:
                accepted += 1
            else:
                amendment_failures.extend(
                    f"{amendment_path.name}:{failure}" for failure in failures
                )
        if accepted != 1 or amendment_failures:
            raise ValueError(
                "cora_source_changed_since_run_lock:"
                f"accepted_amendments={accepted}:failures={amendment_failures}"
            )
    if physical.get("formal_collected") or lock.get("formal_collected"):
        raise ValueError("cora_formal_collection_forbidden")
    scenarios = {
        str(row["pair_id"]): PhysicalScenario.from_dict(row)
        for row in physical["rows"]
    }
    if set(scenarios) != {row.root_id for row in CORA_DATA_MATRIX}:
        raise ValueError("cora_physical_manifest_matrix_mismatch")
    return scenarios, physical, lock


def _runtime(
    client: Any,
    scenario: PhysicalScenario,
    *,
    run_id: str,
    phase: str,
    sensors: Sequence[Any],
    registry: RunRegistry,
) -> ScenarioRuntime:
    identity = RunIdentity(
        experiment_id="h6-cora-c2",
        run_id=run_id,
        scenario_id=f"cora-{scenario.pair_id}-{phase}",
        attempt_id=0,
        server_epoch="carla-0.9.16-cora-c2",
        producer_version="h6-cora-c2-collector-v1",
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
        owner="sdf.h6.cora.collector",
    )
    runtime.start(spec)
    return runtime


def _capture(
    client: Any,
    scenario: PhysicalScenario,
    store: CoraDataStore,
    registry: RunRegistry,
    lights: _FrozenTrafficLights,
    classic: ClassicExpertGenerator,
    vla: NominalVLAGenerator,
    policy: NominalVLAPolicy,
) -> dict[str, Any]:
    run_id = f"{scenario.pair_id}-capture-{time.time_ns()}"
    runtime = _runtime(
        client,
        scenario,
        run_id=run_id,
        phase="capture",
        sensors=(_camera_spec(),),
        registry=registry,
    )
    before_forward = policy.forward_count
    try:
        lights.reset_for_arm()
        header, history, ego_history = _pre_roll(
            runtime,
            scenario,
            kinematic_settle_ticks=1,
        )
        measurement = runtime.sensor_measurement("front_camera", header.carla_frame)
        image_path, image_sha256 = store.write_image(image_png_bytes(measurement))
        anchor = build_anchor(runtime, header, scenario.route, ego_history=ego_history)
        captured_reset = reset_signature(
            _actors_by_role(runtime),
            route=scenario.route,
            weather=_actual_weather(runtime, scenario),
            lights=lights.snapshot(),
            script=scenario.script,
        )
        generated = generate_hybrid_set(anchor, classic, vla)
        guarded = CandidateGuard().evaluate(generated)
        forward_count = policy.forward_count - before_forward
        runtime.complete()
        cleanup = registry.record(run_id)
        if not cleanup or cleanup["status"] != "COMPLETED":
            raise RuntimeError("cora_capture_cleanup_incomplete")
        return {
            "anchor": anchor,
            "history": history,
            "guarded": guarded,
            "capture_reset": captured_reset,
            "vla_forward_count": forward_count,
            "image_path": image_path,
            "image_sha256": image_sha256,
            "cleanup": cleanup,
        }
    except BaseException:
        with contextlib.suppress(BaseException):
            runtime.abort("cora_capture_failure")
        raise


def _proposal_from_hybrid(root_id: str, item: Any) -> CoraProposal:
    candidate = item.candidate
    payload = candidate_to_dict(candidate)
    proposal_hash = trajectory_sha256(candidate.points)
    if proposal_hash != item.provenance.canonical_sha256:
        raise ValueError("cora_nominal_proposal_hash_mismatch")
    return CoraProposal(
        proposal_id=candidate.candidate_id,
        proposal_sha256=proposal_hash,
        root_id=root_id,
        kind="nominal",
        trajectory=tuple(payload["points"]),
        guard={} if item.guard is None else item.guard.to_dict(),
        audit_source=item.provenance.source.value,
        provenance=item.provenance.to_dict(),
        auxiliary_only=False,
        status="GUARD_ELIGIBLE" if item.guard is not None and item.guard.passed else "GUARD_REJECT",
    )


def _reset_payload(
    captured: Any,
    branch: Any,
    *,
    proposal_sha256: str,
) -> dict[str, Any]:
    comparison = compare_reset_signatures(
        captured,
        branch,
        position_limit_m=float(CORA_C2_CONFIG["reset"]["position_delta_m"]),
        yaw_limit_deg=float(CORA_C2_CONFIG["reset"]["yaw_delta_deg"]),
        speed_limit_mps=float(CORA_C2_CONFIG["reset"]["speed_delta_mps"]),
    )
    payload = comparison.to_dict()
    payload.update(
        {
            "capture": captured.to_dict(),
            "branch": branch.to_dict(),
            "world_settings_sha256": WORLD_SETTINGS_SHA256,
            "capture_sensor_sha256": SENSOR_CONTRACT_SHA256,
            "branch_sensor_sha256": SENSOR_CONTRACT_SHA256,
            "capture_proposal_sha256": proposal_sha256,
            "branch_proposal_sha256": proposal_sha256,
        }
    )
    payload["comparable"] = bool(
        comparison.comparable
        and payload["capture_sensor_sha256"] == payload["branch_sensor_sha256"]
        and payload["capture_proposal_sha256"] == payload["branch_proposal_sha256"]
    )
    return payload


def _head(value: Any, unit: str, *, valid: bool = True) -> OutcomeValue:
    return OutcomeValue(value=value if valid else None, unit=unit, valid=valid)


def _minimum_clearance(
    timeline: Sequence[Mapping[str, Any]], actor_future: Sequence[Mapping[str, Any]]
) -> tuple[float | None, float | None]:
    actors_by_tick: dict[int, list[Mapping[str, Any]]] = {}
    for row in actor_future:
        actors_by_tick.setdefault(int(row["tick"]), []).append(row)
    minimum = math.inf
    minimum_ttc = math.inf
    for row in timeline:
        tick = int(row["tick"])
        ego_speed = max(0.0, float(row["speed_mps"]))
        for actor in actors_by_tick.get(tick, ()):
            distance = math.hypot(float(actor["x"]) - float(row["x"]), float(actor["y"]) - float(row["y"]))
            minimum = min(minimum, distance)
            closing = ego_speed - max(0.0, float(actor.get("speed_mps", 0.0)))
            if closing > 1e-3:
                minimum_ttc = min(minimum_ttc, distance / closing)
    return (
        None if not math.isfinite(minimum) else minimum,
        None if not math.isfinite(minimum_ttc) else minimum_ttc,
    )


def _outcome_heads(
    *,
    proposal: CoraProposal,
    timeline: Sequence[Mapping[str, Any]],
    actor_future: Sequence[Mapping[str, Any]],
    collision_count: int,
    collision_severity: float | None,
    red_light: bool,
    offroad: bool,
    route_complete: bool,
    terminal_reason: str,
    cleanup_complete: bool,
    decision: Any | None,
    executable: Any | None,
) -> dict[str, OutcomeValue]:
    metrics = kinematic_metrics(timeline, dt_s=float(CORA_C2_CONFIG["timing"]["fixed_delta_seconds"]))
    clearance, ttc = _minimum_clearance(timeline, actor_future)
    decision_kind = None if decision is None else str(getattr(decision.decision_kind, "value", decision.decision_kind))
    repair_attempted = decision_kind in {"QP", "RATO"}
    repair_success = bool(
        repair_attempted and decision and decision.post_repair_trajectory_id
    )
    executable_valid = executable is not None
    fallback_needed = bool(decision and decision.fallback_request is not None)
    progress = float(timeline[-1]["route_progress_m"]) if timeline else None
    return {
        "progress": _head(progress, "m", valid=progress is not None),
        "completion": _head(route_complete, "bool", valid=bool(timeline)),
        "collision": _head(collision_count > 0, "bool", valid=bool(timeline)),
        "collision_severity": _head(collision_severity, "N*s", valid=collision_severity is not None),
        "red_light": _head(red_light, "bool", valid=bool(timeline)),
        "offroad": _head(offroad, "bool", valid=bool(timeline)),
        "minimum_ttc": _head(ttc, "s", valid=ttc is not None),
        "minimum_clearance": _head(clearance, "m", valid=clearance is not None),
        "acceleration_rms": _head(metrics["acceleration_rms_mps2"], "m/s^2", valid=bool(timeline)),
        "jerk_rms": _head(metrics["jerk_rms_mps3"], "m/s^3", valid=bool(timeline)),
        "lateral_acceleration_rms": _head(metrics["lateral_acceleration_rms_mps2"], "m/s^2", valid=bool(timeline)),
        "deadline_misses": _head(sum(bool(row.get("deadline_miss")) for row in timeline), "count", valid=bool(timeline)),
        "guard_eligible": _head(str(proposal.guard.get("verdict")) in {"PASS", "REVIEW"}, "bool"),
        "safety_executable": _head(executable_valid, "bool", valid=decision is not None),
        "repair_attempted": _head(repair_attempted, "bool", valid=decision is not None),
        "repair_success": _head(repair_success, "bool", valid=decision is not None),
        "executable": _head(executable_valid, "bool", valid=decision is not None),
        "mrm": _head(decision_kind in {"MINIMAL_RISK", "HARD_REJECT"}, "bool", valid=decision is not None),
        "emergency": _head(decision_kind == "EMERGENCY", "bool", valid=decision is not None),
        "fallback_needed": _head(fallback_needed, "bool", valid=decision is not None),
        "ticks": _head(len(timeline), "count"),
        "terminal": _head(terminal_reason, "enum"),
        "cleanup": _head(cleanup_complete, "bool"),
    }


def _branch(
    client: Any,
    scenario: PhysicalScenario,
    store: CoraDataStore,
    registry: RunRegistry,
    lights: _FrozenTrafficLights,
    captured_reset: Any,
    proposal: CoraProposal,
    candidate: Any,
    *,
    split: str,
    repair_protocol: bool = False,
) -> CoraBranchOutcome:
    run_id = f"{scenario.pair_id}-{stable_sha256(proposal.proposal_id)[:12]}-{time.time_ns()}"
    runtime = _runtime(
        client,
        scenario,
        run_id=run_id,
        phase="branch",
        sensors=_event_specs(),
        registry=registry,
    )
    timeline: list[dict[str, Any]] = []
    actor_future: list[dict[str, Any]] = []
    errors: list[str] = []
    reset_payload: dict[str, Any] = {"comparable": False, "reasons": ["branch_not_started"]}
    decision = None
    executable = None
    applied = None
    pre_repair_id = None
    pre_repair_sha256 = None
    executable_id = None
    executable_sha256 = None
    terminal_reason = "RUNTIME_FAILURE"
    cleanup_complete = False
    event_rows: list[dict[str, Any]] = []
    collision_count = 0
    collision_severity: float | None = None
    red_light_violation = False
    offroad = False
    route_complete = False
    cross_candidate_fallback = False
    trace = None
    try:
        lights.reset_for_arm()
        header, _, _ = _pre_roll(
            runtime,
            scenario,
            kinematic_settle_ticks=1,
        )
        branch_reset = reset_signature(
            _actors_by_role(runtime),
            route=scenario.route,
            weather=_actual_weather(runtime, scenario),
            lights=lights.snapshot(),
            script=scenario.script,
        )
        reset_payload = _reset_payload(
            captured_reset,
            branch_reset,
            proposal_sha256=proposal.proposal_sha256,
        )
        if not reset_payload["comparable"]:
            terminal_reason = "RESET_MISMATCH"
            errors.extend(str(item) for item in reset_payload.get("reasons", ()))
        else:
            frame_id = f"{runtime.identity.run_id}:frame-{header.carla_frame}"
            snapshot = safety_snapshot(runtime, header, scenario.route, frame_id=frame_id)
            bound = replace(
                candidate,
                generated_time_s=float(header.simulation_time),
                valid_until_s=float(header.simulation_time) + 0.25,
                dynamics_meta={
                    **dict(candidate.dynamics_meta),
                    "cora_identity_binding": {
                        "parent_proposal_id": proposal.proposal_id,
                        "parent_proposal_sha256": proposal.proposal_sha256,
                        "run_id": runtime.identity.run_id,
                        "frame_id": frame_id,
                        "points_unchanged": True,
                    },
                },
            )
            if trajectory_sha256(bound.points) != proposal.proposal_sha256:
                raise ValueError("cora_branch_binding_changed_proposal")
            candidate_set = PolicyCandidateSet(
                run_id=runtime.identity.run_id,
                frame_id=frame_id,
                scenario_id=runtime.identity.scenario_id,
                model_id="cora-forced-single-candidate@1.0.0",
                carla_frame=int(header.carla_frame),
                simulation_time_s=float(header.simulation_time),
                wall_time_s=float(header.wall_time),
                candidates=(bound,),
                schema_version="safedrive.safety.contracts.v1",
                coordinate_frame="map",
            )
            source = str(getattr(bound.source, "value", bound.source))
            availability = ComponentAvailability(
                classic=source == "classic",
                vla=source.startswith("vla"),
                world=False,
                safety=True,
                detail={
                    "world": "CORA_OFFLINE_DATA_COLLECTION",
                    "forced_candidate_id": proposal.proposal_id,
                    "candidate_set_size": 1,
                    "cross_candidate_fallback": "PROHIBITED",
                },
            )
            kernel = SafetyKernel()
            result = kernel.tick(
                snapshot,
                candidate_set,
                now_s=float(header.simulation_time),
                availability=availability,
            )
            decision = result.decision
            control_loop = ControlLoop()
            initial_ego, _ = ego_state(runtime._actors["ego"])
            if repair_protocol:
                from .repair_labels import safety_trace
                trace = safety_trace(result, kernel.repair_traces, proposal.proposal_id)
                trace["initial_xy"] = [initial_ego.x, initial_ego.y]
            applied = apply_safety_control(
                decision,
                candidate_set,
                control_loop,
                initial_ego,
                float(header.simulation_time),
            )
            executable, resolve_notes = resolve_executable_candidate(decision, candidate_set)
            pre_repair_id = decision.pre_repair_trajectory_id
            if trace and trace["repair_attempted"]:
                pre_repair_id = proposal.proposal_id
            pre_repair_sha256 = (
                proposal.proposal_sha256 if pre_repair_id == proposal.proposal_id else None
            )
            if executable is not None:
                executable_id = executable.candidate_id
                executable_sha256 = trajectory_sha256(executable.points)
                cross_candidate_fallback = executable.source != bound.source
            if decision.fallback_request is not None:
                target = str(getattr(decision.fallback_request.target, "value", decision.fallback_request.target))
                if target == "CLASSIC" and source != "classic":
                    cross_candidate_fallback = True
            if cross_candidate_fallback:
                errors.append("cross_candidate_fallback")
            route_start, _ = route_projection(initial_ego.x, initial_ego.y, scenario.route)
            route_length = sum(
                math.hypot(
                    scenario.route[index][0] - scenario.route[index - 1][0],
                    scenario.route[index][1] - scenario.route[index - 1][1],
                )
                for index in range(1, len(scenario.route))
            )
            current_header = header
            previous_yaw = initial_ego.yaw
            standstill_ticks = 0
            first_frame = -1
            last_frame = -1
            corridor_half_width = float(snapshot.corridor_half_width_m)
            for tick in range(int(CORA_C2_CONFIG["timing"]["branch_ticks"])):
                now_s = float(current_header.simulation_time)
                if _should_force_dynamic_red(scenario, tick):
                    lights.force_red()
                if applied.applied_mode is AppliedMode.TRACK_APPROVED:
                    if tick == 0:
                        command = applied
                    else:
                        if tick % int(CORA_C2_CONFIG["timing"]["trajectory_stamp_refresh_ticks"]) == 0:
                            assert executable is not None
                            control_loop.refresh_trajectory_stamp(safety_points_to_ctrl(executable.points), now_s)
                        current_ego, _ = ego_state(runtime._actors["ego"])
                        command = control_loop.step(current_ego, now_s)
                else:
                    command = applied
                controls = {
                    "ego": carla.VehicleControl(
                        throttle=float(command.throttle),
                        brake=float(command.brake),
                        steer=float(command.steer),
                        reverse=bool(getattr(command, "reverse", False)),
                    ),
                    **_npc_controls(scenario, tick_index=tick),
                }
                tick_started = time.perf_counter()
                current_header = runtime.tick_controls(controls)
                _follow_ego_spectator(runtime, scenario)
                tick_wall_ms = (time.perf_counter() - tick_started) * 1000.0
                state, acceleration = ego_state(runtime._actors["ego"])
                progress, corridor_distance = route_projection(state.x, state.y, scenario.route)
                yaw_rate = (
                    (state.yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
                ) / float(CORA_C2_CONFIG["timing"]["fixed_delta_seconds"])
                previous_yaw = state.yaw
                row = {
                    "tick": tick,
                    "carla_frame": int(current_header.carla_frame),
                    "simulation_time_s": float(current_header.simulation_time),
                    "x": state.x,
                    "y": state.y,
                    "yaw": state.yaw,
                    "speed_mps": state.v,
                    "acceleration_mps2": acceleration,
                    "lateral_acceleration_mps2": state.v * yaw_rate,
                    "route_progress_m": progress - route_start,
                    "corridor_distance_m": corridor_distance,
                    "throttle": float(command.throttle),
                    "brake": float(command.brake),
                    "steer": float(command.steer),
                    "control_mode": str(
                        getattr(
                            getattr(command, "applied_mode", "TRACK_APPROVED"),
                            "value",
                            getattr(command, "applied_mode", "TRACK_APPROVED"),
                        )
                    ),
                    "deadline_miss": bool(getattr(command, "deadline_miss", False)),
                    "tick_wall_ms": tick_wall_ms,
                }
                if repair_protocol:
                    target_light = lights._target_light()
                    row["traffic_light_state"] = None if target_light is None else str(target_light.get_state())
                timeline.append(row)
                for role, actor in sorted(runtime._actors.items()):
                    if role == "ego":
                        continue
                    actor_state, _ = ego_state(actor)
                    actor_future.append(
                        {
                            "tick": tick,
                            "carla_frame": int(current_header.carla_frame),
                            "role": role,
                            "actor_id": int(actor.id),
                            "x": actor_state.x,
                            "y": actor_state.y,
                            "yaw": actor_state.yaw,
                            "speed_mps": actor_state.v,
                        }
                    )
                if first_frame < 0:
                    first_frame = int(current_header.carla_frame)
                last_frame = int(current_header.carla_frame)
                collision_now = runtime.sensor_events(
                    "collision",
                    since_frame=last_frame,
                    through_frame=last_frame,
                )
                if collision_now:
                    terminal_reason = "COLLISION_TERMINAL"
                    break
                route_complete = (progress if repair_protocol else progress - route_start) >= route_length - 2.0
                if route_complete:
                    terminal_reason = "ROUTE_COMPLETE"
                    break
                if not applied.is_track_approved and state.v <= float(CORA_C2_CONFIG["timing"]["mrm_standstill_speed_mps"]):
                    standstill_ticks += 1
                    if standstill_ticks >= int(CORA_C2_CONFIG["timing"]["mrm_standstill_ticks"]):
                        terminal_reason = "MRM_STANDSTILL"
                        break
                else:
                    standstill_ticks = 0
            else:
                terminal_reason = "HORIZON_COMPLETE"
            if first_frame >= 0:
                event_rows, collision_count = _event_rows(runtime, first_frame, last_frame)
                impulses = [
                    math.sqrt(
                        float(row.get("impulse_x", 0.0)) ** 2
                        + float(row.get("impulse_y", 0.0)) ** 2
                        + float(row.get("impulse_z", 0.0)) ** 2
                    )
                    for row in event_rows
                    if row.get("event_type") == "collision"
                ]
                collision_severity = max(impulses) if impulses else (0.0 if timeline else None)
            offroad = any(
                float(row["corridor_distance_m"]) > corridor_half_width for row in timeline
            )
            final_progress = float(timeline[-1]["route_progress_m"]) if timeline else 0.0
            red_light_violation = bool(
                scenario.red_light is not None
                and timeline
                and final_progress > float(scenario.red_light["stop_progress_m"]) + 1.0
            )
        runtime.complete()
        cleanup = registry.record(run_id)
        cleanup_complete = bool(cleanup and cleanup["status"] == "COMPLETED")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
        with contextlib.suppress(BaseException):
            runtime.abort(type(exc).__name__)
        cleanup_complete = False
        terminal_reason = "RUNTIME_FAILURE"

    artifacts: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    if trace is not None:
        path, digest = store.write_immutable_json(
            store.root / "safety-traces" / f"{scenario.pair_id}__{proposal.proposal_sha256}.json", trace)
        artifacts["safety_trace"], artifact_hashes["safety_trace"] = path, digest
    if timeline:
        path, digest = store.write_rows("timeline", scenario.pair_id, proposal.proposal_sha256, timeline)
        artifacts["timeline"], artifact_hashes["timeline"] = path, digest
        future_rows = actor_future or [
            {"tick": -1, "carla_frame": -1, "role": "none", "actor_id": -1, "x": 0.0, "y": 0.0, "yaw": 0.0, "speed_mps": 0.0}
        ]
        path, digest = store.write_rows("actor_future", scenario.pair_id, proposal.proposal_sha256, future_rows)
        artifacts["actor_future"], artifact_hashes["actor_future"] = path, digest
        path, digest = store.write_rows("events", scenario.pair_id, proposal.proposal_sha256, event_rows)
        artifacts["events"], artifact_hashes["events"] = path, digest
    heads = _outcome_heads(
        proposal=proposal,
        timeline=timeline,
        actor_future=actor_future,
        collision_count=collision_count,
        collision_severity=collision_severity,
        red_light=red_light_violation,
        offroad=offroad,
        route_complete=route_complete,
        terminal_reason=terminal_reason,
        cleanup_complete=cleanup_complete,
        decision=decision,
        executable=executable,
    )
    label_payload = {
        "schema_version": "safedrive.cora.outcome_labels.v1",
        "root_id": scenario.pair_id,
        "proposal_id": proposal.proposal_id,
        "proposal_sha256": proposal.proposal_sha256,
        "heads": {name: value.to_dict() for name, value in heads.items()},
    }
    label_payload["label_sha256"] = stable_sha256(label_payload)
    label_path, label_sha = store.write_label(
        scenario.pair_id, proposal.proposal_sha256, label_payload
    )
    artifacts["label"], artifact_hashes["label"] = label_path, label_sha
    decision_kind = None if decision is None else str(getattr(decision.decision_kind, "value", decision.decision_kind))
    repair_attempted = decision_kind in {"QP", "RATO"}
    repair_success = None if not repair_attempted else executable is not None
    if trace is not None:
        repair_attempted = trace["repair_attempted"]
        repair_success = trace["repair_success"]
    return CoraBranchOutcome(
        root_id=scenario.pair_id,
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        split=split,
        guard_verdict=str(proposal.guard.get("verdict", "REJECT")),
        reset=reset_payload,
        safety_input_id=None if decision is None else proposal.proposal_id,
        safety_input_sha256=None if decision is None else proposal.proposal_sha256,
        pre_repair_id=pre_repair_id,
        pre_repair_sha256=pre_repair_sha256,
        executable_id=executable_id,
        executable_sha256=executable_sha256,
        applied_id=None if applied is None else applied.executed_id,
        applied_sha256=(
            executable_sha256
            if applied is not None
            and applied.applied_mode is AppliedMode.TRACK_APPROVED
            and applied.executed_id == executable_id
            else None
        ),
        decision_kind=decision_kind,
        applied_mode=None if applied is None else applied.applied_mode.value,
        repair_attempted=repair_attempted,
        repair_success=repair_success,
        would_require_cross_candidate_fallback=cross_candidate_fallback,
        ticks_executed=len(timeline),
        terminal_reason=terminal_reason,
        cleanup_complete=cleanup_complete,
        heads=heads,
        artifact_paths=artifacts,
        artifact_sha256=artifact_hashes,
        errors=tuple(errors),
        auxiliary_only=proposal.auxiliary_only,
    )


def _selected_rows(map_name: str, scope: str) -> tuple[CoraMatrixRow, ...]:
    if scope == "smoke":
        selected = [row for row in CORA_DATA_MATRIX if row.root_id in CORA_SMOKE_ROOT_IDS]
    elif scope == "pilot":
        selected = [row for row in CORA_DATA_MATRIX if row.split == "coverage_pilot"]
    elif scope == "development":
        selected = [row for row in CORA_DATA_MATRIX if row.split != "coverage_pilot"]
    else:
        raise ValueError(f"cora_collection_scope:{scope}")
    return tuple(row for row in selected if row.scenario.map_name == map_name)


def _resource_snapshot(store: CoraDataStore) -> dict[str, Any]:
    usage = shutil.disk_usage(store.root)
    dataset_bytes = sum(path.stat().st_size for path in store.root.rglob("*") if path.is_file())
    roots = list(store.pairs_dir.glob("*.json"))
    branch_attempts = 0
    for path in roots:
        try:
            branch_attempts += len(json.loads(path.read_text(encoding="utf-8")).get("branches", ()))
        except (OSError, json.JSONDecodeError):
            branch_attempts += int(CORA_C2_CONFIG["interventions"]["max_per_root"]) + 2
    return {
        "free_bytes": usage.free,
        "free_gib": usage.free / 1024**3,
        "dataset_bytes": dataset_bytes,
        "dataset_gib": dataset_bytes / 1024**3,
        "root_attempts": len(roots),
        "branch_attempts": branch_attempts,
    }


def _enforce_resources(store: CoraDataStore, *, additional_branches: int = 0) -> dict[str, Any]:
    snapshot = _resource_snapshot(store)
    limits = CORA_C2_CONFIG["resources"]
    failures = []
    if snapshot["free_gib"] < float(limits["free_disk_floor_gib"]):
        failures.append("free_disk_floor")
    if snapshot["dataset_gib"] > float(limits["dataset_limit_gib"]):
        failures.append("dataset_size")
    if snapshot["root_attempts"] > int(limits["root_attempt_limit"]):
        failures.append("root_attempt_limit")
    if snapshot["branch_attempts"] + additional_branches > int(limits["branch_attempt_limit"]):
        failures.append("branch_attempt_limit")
    if failures:
        raise RuntimeError(f"cora_resource_gate:{failures}:{snapshot}")
    return snapshot


def _prior_collection_wall_s() -> float:
    total = 0.0
    if not EVIDENCE_ROOT.is_dir():
        return total
    for path in EVIDENCE_ROOT.glob("collect-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            total += max(0.0, float(payload.get("elapsed_s", 0.0)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return total


def _write_terminal_capture_failure(
    store: CoraDataStore,
    row: CoraMatrixRow,
    scenario: PhysicalScenario,
    *,
    physical_manifest_sha256: str,
    run_lock_sha256: str,
    vla_forward_count: int,
    error: BaseException,
) -> CoraRootRecord:
    anchor_payload: dict[str, Any] = {
        "schema_version": "safedrive.cora.root_anchor.v1",
        "dataset_id": store.dataset_id,
        "root_id": row.root_id,
        "scenario": row.scenario.to_dict(),
        "physical_sha256": scenario.physical_sha256,
        "physical_manifest_sha256": physical_manifest_sha256,
        "run_lock_sha256": run_lock_sha256,
        "capture_status": "FAILED",
        "error": f"{type(error).__name__}:{error}",
    }
    anchor_payload["anchor_content_sha256"] = stable_sha256(anchor_payload)
    anchor_path, anchor_sha = store.write_anchor(row.root_id, anchor_payload)
    return CoraRootRecord(
        dataset_id=store.dataset_id,
        root_id=row.root_id,
        split=row.split,
        scenario={
            **row.scenario.to_dict(),
            "physical_sha256": scenario.physical_sha256,
            "run_lock_sha256": run_lock_sha256,
        },
        matrix_sha256=CORA_MATRIX_SHA256,
        config_sha256=CORA_C2_CONFIG_SHA256,
        anchor_path=anchor_path,
        anchor_sha256=anchor_sha,
        feature_paths={},
        feature_sha256={},
        proposals=(),
        branches=(),
        edges=(),
        vla_forward_count=vla_forward_count,
        terminal_status="CAPTURE_FAILED",
        missingness=(
            {
                "stage": "capture",
                "reason": f"{type(error).__name__}:{error}",
                "valid": False,
            },
        ),
    )


def _collect_root(
    client: Any,
    world: Any,
    store: CoraDataStore,
    registry: RunRegistry,
    row: CoraMatrixRow,
    scenario: PhysicalScenario,
    *,
    physical_manifest_sha256: str,
    run_lock_sha256: str,
    classic: ClassicExpertGenerator,
    vla: NominalVLAGenerator,
    policy: NominalVLAPolicy,
    repair_protocol: bool = False,
    intervention_factory: Any = None,
    before_branch: Any = None,
) -> CoraRootRecord:
    before_forward = policy.forward_count
    with _FrozenTrafficLights(world, scenario) as lights:
        try:
            capture = _capture(client, scenario, store, registry, lights, classic, vla, policy)
        except BaseException as exc:
            return _write_terminal_capture_failure(
                store,
                row,
                scenario,
                physical_manifest_sha256=physical_manifest_sha256,
                run_lock_sha256=run_lock_sha256,
                vla_forward_count=policy.forward_count - before_forward,
                error=exc,
            )
        anchor = capture["anchor"]
        guarded = capture["guarded"]
        vla_forward_count = int(capture["vla_forward_count"])
        missingness: list[dict[str, Any]] = []
        if vla_forward_count != 1:
            missingness.append(
                {
                    "stage": "generation",
                    "reason": f"vla_forward_count:{vla_forward_count}",
                    "valid": False,
                }
            )
        nominal_items = tuple(guarded.candidates)
        nominal_proposals = tuple(_proposal_from_hybrid(row.root_id, item) for item in nominal_items)
        nominal_by_source = {
            item.provenance.source.value: (item, proposal)
            for item, proposal in zip(nominal_items, nominal_proposals)
        }
        for source in ("expert", "vla"):
            if source not in nominal_by_source:
                missingness.append({"stage": "generation", "reason": f"nominal_{source}_missing", "valid": False})
        interventions = (intervention_factory or derive_interventions)(
            row.root_id,
            row.scenario.family,
            anchor,
            nominal_items,
        )
        intervention_pairs: list[tuple[InterventionResult, CoraProposal]] = []
        for result in interventions:
            proposal = result.to_proposal()
            if proposal is None:
                missingness.append(
                    {
                        "stage": "intervention",
                        "source": result.base_source,
                        "operator": result.operator,
                        "reason": result.status,
                        "detail": result.error,
                        "valid": False,
                    }
                )
            else:
                intervention_pairs.append((result, proposal))
        all_proposals = nominal_proposals + tuple(proposal for _, proposal in intervention_pairs)
        if len(all_proposals) > 4:
            raise AssertionError("cora_root_proposal_limit")
        anchor_payload: dict[str, Any] = {
            "schema_version": "safedrive.cora.root_anchor.v1",
            "dataset_id": store.dataset_id,
            "root_id": row.root_id,
            "split": row.split,
            "scenario": row.scenario.to_dict(),
            "matrix_index": row.matrix_index,
            "branch_order": list(row.branch_order),
            "expert_slot": row.expert_slot,
            "physical_sha256": scenario.physical_sha256,
            "physical_manifest_sha256": physical_manifest_sha256,
            "run_lock_sha256": run_lock_sha256,
            "observable_anchor": anchor.to_dict(),
            "observable_snapshot": _jsonable(asdict(anchor.safety_snapshot)),
            "observable_history": _jsonable(capture["history"]),
            "route": [[float(x), float(y)] for x, y in scenario.route],
            "image_path": capture["image_path"],
            "image_sha256": capture["image_sha256"],
            "capture_reset": capture["capture_reset"].to_dict(),
            "sensor_contract_sha256": SENSOR_CONTRACT_SHA256,
            "world_settings_sha256": WORLD_SETTINGS_SHA256,
            "vla_forward_count": vla_forward_count,
            "capture_cleanup": capture["cleanup"],
        }
        anchor_payload["anchor_content_sha256"] = stable_sha256(anchor_payload)
        anchor_path, anchor_sha = store.write_anchor(row.root_id, anchor_payload)
        feature_paths: dict[str, str] = {}
        feature_hashes: dict[str, str] = {}
        candidate_by_id: dict[str, Any] = {
            item.candidate.candidate_id: item.candidate for item in nominal_items
        }
        for result, proposal in intervention_pairs:
            assert result.candidate is not None
            candidate_by_id[proposal.proposal_id] = result.candidate.candidate
        for proposal in all_proposals:
            store.write_proposal(
                row.root_id,
                proposal.proposal_sha256,
                proposal.to_dict(),
                intervention=proposal.kind == "offline_intervention",
            )
            feature = build_cora_feature_view(anchor, candidate_by_id[proposal.proposal_id])
            path, digest = store.write_feature(row.root_id, proposal.proposal_sha256, feature)
            feature_paths[proposal.proposal_sha256] = path
            feature_hashes[proposal.proposal_sha256] = digest
        branches: list[CoraBranchOutcome] = []
        if vla_forward_count == 1:
            ordered_nominal = [
                nominal_by_source[source]
                for source in row.branch_order
                if source in nominal_by_source
            ]
            for item, proposal in ordered_nominal:
                if str(proposal.guard.get("verdict")) not in {"PASS", "REVIEW"}:
                    missingness.append(
                        {
                            "stage": "nominal_branch",
                            "source": proposal.audit_source,
                            "reason": "guard_reject",
                            "valid": False,
                        }
                    )
                    continue
                if before_branch is not None:
                    before_branch(row.root_id, proposal.proposal_id)
                branches.append(
                    _branch(
                        client,
                        scenario,
                        store,
                        registry,
                        lights,
                        capture["capture_reset"],
                        proposal,
                        item.candidate,
                        split=row.split,
                        repair_protocol=repair_protocol,
                    )
                )
            for result, proposal in sorted(
                intervention_pairs,
                key=lambda item: stable_sha256({"root": row.root_id, "proposal": item[1].proposal_id}),
            ):
                assert result.candidate is not None
                if before_branch is not None:
                    before_branch(row.root_id, proposal.proposal_id)
                branches.append(
                    _branch(
                        client,
                        scenario,
                        store,
                        registry,
                        lights,
                        capture["capture_reset"],
                        proposal,
                        result.candidate.candidate,
                        split=row.split,
                        repair_protocol=repair_protocol,
                    )
                )
        branch_by_id = {branch.proposal_id: branch for branch in branches}
        edges: list[CoraPairEdge] = []
        if set(nominal_by_source) == {"expert", "vla"}:
            left_source, right_source = row.branch_order
            left = nominal_by_source[left_source][1]
            right = nominal_by_source[right_source][1]
            left_branch = branch_by_id.get(left.proposal_id)
            right_branch = branch_by_id.get(right.proposal_id)
            pair_mask = bool(
                vla_forward_count == 1
                and left_branch
                and right_branch
                and left_branch.outcome_valid
                and right_branch.outcome_valid
                and not left_branch.would_require_cross_candidate_fallback
                and not right_branch.would_require_cross_candidate_fallback
            )
            edges.append(
                CoraPairEdge(
                    root_id=row.root_id,
                    left_proposal_id=left.proposal_id,
                    right_proposal_id=right.proposal_id,
                    left_proposal_sha256=left.proposal_sha256,
                    right_proposal_sha256=right.proposal_sha256,
                    edge_kind="nominal",
                    pair_outcome_mask=pair_mask,
                )
            )
        for _result, intervention in intervention_pairs:
            base = next(
                (proposal for proposal in nominal_proposals if proposal.proposal_id == intervention.base_proposal_id),
                None,
            )
            if base is None:
                missingness.append({"stage": "intervention_edge", "reason": "base_missing", "valid": False})
                continue
            intervention_branch = branch_by_id.get(intervention.proposal_id)
            base_branch = branch_by_id.get(base.proposal_id)
            edge_mask = bool(
                not intervention.auxiliary_only
                and intervention_branch
                and base_branch
                and intervention_branch.outcome_valid
                and base_branch.outcome_valid
            )
            edges.append(
                CoraPairEdge(
                    root_id=row.root_id,
                    left_proposal_id=intervention.proposal_id,
                    right_proposal_id=base.proposal_id,
                    left_proposal_sha256=intervention.proposal_sha256,
                    right_proposal_sha256=base.proposal_sha256,
                    edge_kind="intervention_base",
                    pair_outcome_mask=edge_mask,
                )
            )
        nominal_mask = any(edge.edge_kind == "nominal" and edge.pair_outcome_mask for edge in edges)
        terminal_status = "VALID_NOMINAL_PAIR" if nominal_mask else "INCOMPLETE_NOMINAL_PAIR"
        return CoraRootRecord(
            dataset_id=store.dataset_id,
            root_id=row.root_id,
            split=row.split,
            scenario={
                **row.scenario.to_dict(),
                "physical_sha256": scenario.physical_sha256,
                "run_lock_sha256": run_lock_sha256,
                "branch_order": list(row.branch_order),
                "expert_slot": row.expert_slot,
            },
            matrix_sha256=CORA_MATRIX_SHA256,
            config_sha256=CORA_C2_CONFIG_SHA256,
            anchor_path=anchor_path,
            anchor_sha256=anchor_sha,
            feature_paths=feature_paths,
            feature_sha256=feature_hashes,
            proposals=all_proposals,
            branches=tuple(branches),
            edges=tuple(edges),
            vla_forward_count=vla_forward_count,
            terminal_status=terminal_status,
            missingness=tuple(missingness),
        )


def collect_map(dataset_id: str, map_name: str, scope: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    store = _dataset_store(dataset_id)
    manifest_ok, manifest_failures = store.verify_manifest()
    if not manifest_ok:
        raise ValueError(f"cora_pre_collection_manifest:{manifest_failures}")
    scenarios, physical_manifest, run_lock = _load_frozen(store)
    client, world, report = _connection(map_name)
    _require_clean_scene(world)
    rows = _selected_rows(map_name, scope)
    expected = {"smoke": 1, "pilot": 9, "development": 108}[scope]
    if len(rows) != expected:
        raise AssertionError(f"cora_scope_count:{scope}:{map_name}:{len(rows)}")
    resources_now = _resource_snapshot(store)
    pending_rows = pending_collection_rows(store, rows)
    if resources_now["root_attempts"] + len(pending_rows) > int(CORA_C2_CONFIG["resources"]["root_attempt_limit"]):
        raise RuntimeError("cora_projected_root_attempt_limit")
    _enforce_resources(store, additional_branches=4 * len(pending_rows))
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(EVIDENCE_ROOT / "run-registry.sqlite3")
    sampler = GPUMemorySampler(interval_s=0.1, gpu_index=0).start()
    started = time.perf_counter()
    prior_wall_s = _prior_collection_wall_s()
    results: list[dict[str, Any]] = []
    policy = NominalVLAPolicy(keep_on_gpu=True)
    try:
        policy.ensure_loaded()
        generator_hash = simlingo_generator_hash(policy)
        classic = ClassicExpertGenerator()
        vla = NominalVLAGenerator(policy, generator_hash=generator_hash)
        for row in rows:
            aggregate_wall_s = prior_wall_s + (time.perf_counter() - started)
            if aggregate_wall_s > float(CORA_C2_CONFIG["resources"]["aggregate_wall_limit_hours"]) * 3600.0:
                raise RuntimeError(f"cora_aggregate_wall_limit:{aggregate_wall_s:.3f}")
            if store.has_valid_root(row.root_id):
                results.append({"root_id": row.root_id, "status": "RESUMED"})
                continue
            existing = store.pairs_dir / f"{row.root_id}.json"
            if existing.exists():
                raise ValueError(f"cora_existing_terminal_root_invalid:{row.root_id}")
            _enforce_resources(store, additional_branches=4)
            record = _collect_root(
                client,
                world,
                store,
                registry,
                row,
                scenarios[row.root_id],
                physical_manifest_sha256=str(physical_manifest["physical_manifest_sha256"]),
                run_lock_sha256=str(run_lock["run_lock_sha256"]),
                classic=classic,
                vla=vla,
                policy=policy,
            )
            store.write_root(record)
            results.append(
                {
                    "root_id": row.root_id,
                    "status": record.terminal_status,
                    "vla_forward_count": record.vla_forward_count,
                    "nominal_pair_outcome_mask": record.nominal_pair_outcome_mask,
                    "branches": len(record.branches),
                    "content_sha256": record.to_dict()["content_sha256"],
                }
            )
            _require_clean_scene(world)
    except BaseException as exc:
        gpu = sampler.stop()
        failure: dict[str, Any] = {
            "schema_version": "safedrive.cora.collection_failure.v1",
            "dataset_id": dataset_id,
            "map_name": map_name,
            "scope": scope,
            "error": f"{type(exc).__name__}:{exc}",
            "elapsed_s": time.perf_counter() - started,
            "gpu": gpu,
            "results": results,
            "resource": _resource_snapshot(store),
        }
        failure["evidence_sha256"] = stable_sha256(failure)
        _atomic_json(EVIDENCE_ROOT / f"collect-{scope}-{map_name}-failure-{time.time_ns()}.json", failure)
        raise
    gpu = sampler.stop()
    elapsed_s = time.perf_counter() - started
    peak = float(gpu.get("peak_used_gib", 0.0))
    if peak > float(CORA_C2_CONFIG["resources"]["whole_gpu_peak_limit_gib"]):
        raise RuntimeError(f"cora_gpu_peak_limit:{peak:.6f}")
    store.write_manifest()
    manifest_ok, manifest_failures = store.verify_manifest()
    if not manifest_ok:
        raise ValueError(f"cora_post_collection_manifest:{manifest_failures}")
    payload: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA,
        "dataset_id": dataset_id,
        "map_name": map_name,
        "scope": scope,
        "connection": report.to_dict(),
        "run_lock_sha256": run_lock["run_lock_sha256"],
        "physical_manifest_sha256": physical_manifest["physical_manifest_sha256"],
        "elapsed_s": elapsed_s,
        "gpu": gpu,
        "whole_gpu_peak_gib": peak,
        "model_forward_count_process": policy.forward_count,
        "model_peak_vram_mb": policy.last_peak_vram_mb,
        "results": results,
        "resource": _resource_snapshot(store),
        "manifest_valid": True,
    }
    payload["collection_summary_sha256"] = stable_sha256(payload)
    _atomic_json(EVIDENCE_ROOT / f"collect-{scope}-{map_name}.json", payload)
    return {
        "ok": True,
        "map": map_name,
        "scope": scope,
        "roots": len(results),
        "whole_gpu_peak_gib": peak,
        "collection_summary_sha256": payload["collection_summary_sha256"],
    }


__all__ = [
    "COLLECTION_SCHEMA",
    "DATA_ROOT",
    "EVIDENCE_ROOT",
    "collect_map",
    "freeze_manifest",
    "materialize_map",
]
