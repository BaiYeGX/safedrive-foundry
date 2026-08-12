"""R2 live paired orchestrator: one anchor forward + two frozen-candidate branches.

Branch phases never call VLA forward. Exact spawn only. Oracle traces stay offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.evaluation.comparability import (
    BranchExecutionReport,
    evaluate_pair_comparability,
)
from driving_vla.evaluation.actor_future_collector import (
    ActorFutureCollector,
    capture_observable_road_context,
    capture_observable_scene_t0,
)
from driving_vla.evaluation.fixture_runtime import (
    FixtureError,
    FixtureSession,
    _reapply_exact_state,
    apply_weather,
    cleanup_session,
    connect_world,
    measure_initial_state,
    open_fixture_session,
    restore_async,
)
from driving_vla.evaluation.oracle import evaluate_pair_oracle
from driving_vla.evaluation.outcome_metrics import (
    TickRecord,
    aggregate_branch_outcome,
)
from driving_vla.evaluation.paired_contract import (
    EXPECTED_PRIMARY_TICKS,
    K2_ANCHOR_SCHEMA,
    K2_ANCHOR_SCHEMA_V2,
    PRIMARY_HORIZON_S,
    K2AnchorArtifactV1,
    K2AnchorArtifactV2,
    ObservationFingerprint,
    SerializedCandidate,
    assert_no_oracle_in_control_payload,
    build_model_retimer_hash,
    compute_pair_id,
    compute_run_id,
    content_hash,
    sha256_hex,
)
from driving_vla.evaluation.k2_spatial_artifact import (
    bundle_from_artifact_v2,
)
from driving_vla.evaluation.k2_v3_artifact import (
    K2_ANCHOR_SCHEMA_V3,
    K2AnchorArtifactV3,
)
from driving_vla.evaluation.runner_contract import (
    PAIR_STATUS_COMPLETED,
    PAIR_STATUS_FAILED,
    RETRY_POLICY_NO_AUTO_RETRY,
    ExpectedPairHashes,
    RunnerContractError,
    append_ledger_if_new,
    attempt_dir_for,
    build_completed_manifest,
    build_failed_manifest,
    finalize_branch_failure_codes,
    ledger_path_for_evidence_root,
    normalize_retry_policy,
    plan_pair_attempt,
    require_frozen_registry,
    resolve_no_auto_retry_action,
)
from driving_vla.evaluation.scenario_registry import (
    ScenarioRegistryV1,
    ScenarioSeedFixture,
    load_scenario_registry,
)
from driving_vla.model.k2_builder import (
    GUARD_OK,
    K2Diagnostics,
    K2ExecutionSpec,
    K2PredictionBundle,
)
from driving_vla.model.neural_policy import NeuralV1Policy, NeuralV2Policy
from driving_vla.model.simlingo_contract import carla_bgra_to_bgr
from driving_vla.model.simlingo_runtime import (
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
    SimLingoNeuralRuntime,
)
from driving_vla.runtime.k2_execution import (
    apply_k2_to_executors,
    select_k2,
    select_k2_semantic_v3,
    select_k2_spatial,
)
from driving_vla.runtime.path_manager import EgoPose, PathManagerConfig, VLAPathManager
from driving_vla.runtime.vehicle_geometry import vehicle_geometry_from_carla_vehicle
from driving_vla.runtime.vla_mpc_tracker import ConstrainedVLAMPC, VLAMPCConfig
from driving_vla.runtime.vla_speed_planner import VLASpeedConfig, VLASpeedPlanner

EXECUTOR_CONFIG_ID = "g3_stable_vla_mpc_r2_v1"
EXECUTOR_CONFIG_HASH = content_hash(
    {
        "id": EXECUTOR_CONFIG_ID,
        "control_dt_s": 0.05,
        "primary_horizon_s": PRIMARY_HORIZON_S,
        "expected_ticks": EXPECTED_PRIMARY_TICKS,
        "path_stale_soft_s": 1.0,
        "path_stale_hard_s": 2.5,
        "path_stale_zero_s": 5.0,
        "mpc_horizon": 20,
        "prediction_dt_s": 0.10,
    },
    nibble=32,
)

SPECTATOR_BACK_M = 8.0
SPECTATOR_UP_M = 5.5
SPECTATOR_PITCH_DEG = -20.0


@dataclass
class SensorBundle:
    camera: Any = None
    collision: Any = None
    lane: Any = None
    image_lock: threading.Lock = field(default_factory=threading.Lock)
    event_lock: threading.Lock = field(default_factory=threading.Lock)
    latest_image: dict[str, Any] = field(
        default_factory=lambda: {
            "rgb": None,
            "layout": "bgr",
            "frame": -1,
            "received": 0,
        }
    )
    collision_episodes: int = 0
    collision_impulse_sum: float = 0.0
    first_collision_time_s: float | None = None
    lane_invasion_episodes: int = 0
    _lane_prev: bool = False
    _collision_active: bool = False


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(obj), sort_keys=True, default=str) + "\n")


def compute_chase_cam_world(
    *,
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    back_m: float = SPECTATOR_BACK_M,
    up_m: float = SPECTATOR_UP_M,
) -> tuple[float, float, float, float]:
    """Return a behind-and-above spectator pose without importing CARLA."""
    yaw_rad = math.radians(float(yaw_deg))
    return (
        float(x) - float(back_m) * math.cos(yaw_rad),
        float(y) - float(back_m) * math.sin(yaw_rad),
        float(z) + float(up_m),
        float(yaw_deg),
    )


def set_spectator_follow(world: Any, ego: Any) -> tuple[bool, str | None]:
    """Move CARLA's spectator behind the ego; never affect simulation control."""
    try:
        import carla

        tf = ego.get_transform()
        cx, cy, cz, yaw = compute_chase_cam_world(
            x=float(tf.location.x),
            y=float(tf.location.y),
            z=float(tf.location.z),
            yaw_deg=float(tf.rotation.yaw),
        )
        world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=cx, y=cy, z=cz),
                carla.Rotation(
                    pitch=float(SPECTATOR_PITCH_DEG),
                    yaw=yaw,
                    roll=0.0,
                ),
            )
        )
        return True, None
    except Exception as exc:  # spectator is viewing-only; execution must continue
        return False, f"{type(exc).__name__}: {exc}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _route_xy(fixture: ScenarioSeedFixture) -> tuple[tuple[float, float], ...]:
    return tuple((float(w[0]), float(w[1])) for w in fixture.route.waypoints)


def _nav_targets_ego(
    route_xy: tuple[tuple[float, float], ...],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    d1: float = 15.0,
    d2: float = 30.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Map-route points → ego-frame targets (fallback arc if route short)."""
    if len(route_xy) < 2:
        return (d1, 0.0), (d2, 0.0)
    # nearest route index
    best_i = 0
    best_d = 1e18
    for i, (x, y) in enumerate(route_xy):
        d = (x - ego_x) ** 2 + (y - ego_y) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    # advance along route by approx d1/d2
    def _point_at(dist: float) -> tuple[float, float]:
        acc = 0.0
        i = best_i
        while i + 1 < len(route_xy) and acc < dist:
            x0, y0 = route_xy[i]
            x1, y1 = route_xy[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            if acc + seg >= dist:
                t = (dist - acc) / max(seg, 1e-6)
                mx = x0 + t * (x1 - x0)
                my = y0 + t * (y1 - y0)
                return mx, my
            acc += seg
            i += 1
        return route_xy[-1]

    m1 = _point_at(d1)
    m2 = _point_at(d2)
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)

    def to_ego(mx: float, my: float) -> tuple[float, float]:
        dx, dy = mx - ego_x, my - ego_y
        return (float(c * dx - s * dy), float(s * dx + c * dy))

    return to_ego(*m1), to_ego(*m2)


def load_anchor_artifact_any(
    data: bytes,
) -> K2AnchorArtifactV1 | K2AnchorArtifactV2 | K2AnchorArtifactV3:
    """Deserialize V1, V2 or V3 anchor artifact by schema (fail-closed)."""
    payload = json.loads(data.decode("utf-8"))
    schema = str(payload.get("schema_version") or "")
    if schema == K2_ANCHOR_SCHEMA_V2:
        return K2AnchorArtifactV2.from_dict(payload)
    if schema == K2_ANCHOR_SCHEMA:
        return K2AnchorArtifactV1.from_dict(payload)
    if schema == K2_ANCHOR_SCHEMA_V3:
        return K2AnchorArtifactV3.from_dict(payload)
    raise FixtureError(f"unsupported_anchor_schema:{schema}")


def select_from_anchor_artifact(
    art: K2AnchorArtifactV1 | K2AnchorArtifactV2 | K2AnchorArtifactV3,
    *,
    force_index: int,
    allow_nominal_only_fallback: bool = False,
):
    """Cold-select a forced V1/V2/V3 branch without another VLA forward."""
    if isinstance(art, K2AnchorArtifactV3):
        return select_k2_semantic_v3(
            art.bundle,
            mode="force",
            force_index=force_index,
        )
    if isinstance(art, K2AnchorArtifactV2):
        bundle = bundle_from_artifact_v2(art)
        if (
            allow_nominal_only_fallback
            and force_index == 0
            and bundle.guard_status != GUARD_OK
            and set(bundle.guard_reasons) == {"SPATIAL_COLLAPSE_ELIGIBLE"}
        ):
            # Candidate 0 is the exact native nominal anchor. The set-level
            # rejection only says candidate 1 collapsed; R3 K_eff=1 collection
            # may execute nominal while retaining the original rejected
            # artifact and an explicit fallback audit.
            bundle = replace(bundle, guard_status=GUARD_OK, guard_reasons=())
        return select_k2_spatial(bundle, mode="force", force_index=force_index)
    bundle = bundle_from_artifact(art)
    return select_k2(bundle, mode="force", force_index=force_index)


def bundle_from_artifact(art: K2AnchorArtifactV1) -> K2PredictionBundle:
    """Rebuild runtime K2PredictionBundle from frozen artifact (no VLA)."""
    cands: list[TrajectoryArray] = []
    specs: dict[str, K2ExecutionSpec] = {}
    for sc in art.candidates:
        cands.append(
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=sc.points_xy_yaw_v_a_kappa,
                probability=sc.probability,
                candidate_id=sc.candidate_id,
            )
        )
        specs[sc.candidate_id] = K2ExecutionSpec(
            candidate_id=sc.candidate_id,
            spatial_path_xy=sc.spatial_path_xy,
            speed_samples_mps=sc.speed_samples_mps,
            timed_trajectory_hash=sc.timed_trajectory_hash,
            native_path_hash=sc.native_path_hash,
            branch_type=sc.branch_type,
        )
    diag_raw = dict(art.diagnostics)
    diag = K2Diagnostics(
        mean_speed_gap_mps=float(diag_raw.get("mean_speed_gap_mps", 0.0)),
        final_progress_gap_m=float(diag_raw.get("final_progress_gap_m", 0.0)),
        max_position_separation_m=float(diag_raw.get("max_position_separation_m", 0.0)),
        mean_position_separation_m=float(diag_raw.get("mean_position_separation_m", 0.0)),
        collapsed=bool(diag_raw.get("collapsed", False)),
        collapse_reason=diag_raw.get("collapse_reason"),
        selection_space_eligible=bool(diag_raw.get("selection_space_eligible", True)),
        path_speed_cap_active=bool(diag_raw.get("path_speed_cap_active", False)),
        position_integration_error_max_m=float(
            diag_raw.get("position_integration_error_max_m", 0.0)
        ),
        acceleration_error_max_mps2=float(diag_raw.get("acceleration_error_max_mps2", 0.0)),
        yaw_tangent_error_max_rad=float(diag_raw.get("yaw_tangent_error_max_rad", 0.0)),
        curvature_error_max_per_m=float(diag_raw.get("curvature_error_max_per_m", 0.0)),
        curvature_error_p95_per_m=float(diag_raw.get("curvature_error_p95_per_m", 0.0)),
        native_path_cross_track_error_max_m=float(
            diag_raw.get("native_path_cross_track_error_max_m", 0.0)
        ),
    )
    return K2PredictionBundle(
        observation_identity={
            "pair_id": art.pair_id,
            "scenario_id": art.scenario_id,
            "seed_id": art.seed_id,
            "anchor_run_id": art.anchor_run_id,
        },
        model_id=art.model_id,
        config_hash=art.config_hash,
        retimer_version=art.retimer_version,
        native_path_xy=art.native_path_xy,
        native_path_hash=art.native_path_hash,
        candidates=tuple(cands),
        execution_specs=specs,
        top1_index=art.top1_index,
        probability_source=art.probability_source,
        probability_margin=0.0,
        branch_type=art.branch_type,
        diagnostics=diag,
        guard_status=art.guard_status,
        guard_reasons=art.guard_reasons,
    )


def artifact_from_bundle(
    bundle: K2PredictionBundle,
    *,
    pair_id: str,
    scenario_id: str,
    seed_id: str,
    anchor_run_id: str,
    anchor_carla_frame: int,
    anchor_simulation_time_s: float,
    requested_initial_state_hash: str,
    measured_initial_state_hash: str,
    observation_fingerprint: ObservationFingerprint,
    model_checkpoint_hash: str,
    retimer_hash: str,
    executor_config_hash: str,
) -> K2AnchorArtifactV1:
    cands: list[SerializedCandidate] = []
    for i, c in enumerate(bundle.candidates):
        spec = bundle.execution_specs[c.candidate_id]
        cands.append(
            SerializedCandidate(
                candidate_id=c.candidate_id,
                candidate_index=i,
                probability=float(c.probability),
                points_xy_yaw_v_a_kappa=c.points_xy_yaw_v_a_kappa,
                spatial_path_xy=spec.spatial_path_xy,
                speed_samples_mps=spec.speed_samples_mps,
                timed_trajectory_hash=spec.timed_trajectory_hash,
                native_path_hash=spec.native_path_hash,
                branch_type=spec.branch_type,
            )
        )
    d = bundle.diagnostics
    return K2AnchorArtifactV1(
        schema_version=K2_ANCHOR_SCHEMA,
        pair_id=pair_id,
        scenario_id=scenario_id,
        seed_id=seed_id,
        anchor_run_id=anchor_run_id,
        anchor_carla_frame=int(anchor_carla_frame),
        anchor_simulation_time_s=float(anchor_simulation_time_s),
        requested_initial_state_hash=requested_initial_state_hash,
        measured_initial_state_hash=measured_initial_state_hash,
        observation_fingerprint=observation_fingerprint,
        model_id=bundle.model_id,
        model_checkpoint_hash=model_checkpoint_hash,
        config_hash=bundle.config_hash,
        retimer_version=bundle.retimer_version,
        retimer_hash=retimer_hash,
        executor_config_hash=executor_config_hash,
        native_path_xy=bundle.native_path_xy,
        native_path_hash=bundle.native_path_hash,
        candidates=tuple(cands),
        top1_index=int(bundle.top1_index),
        guard_status=bundle.guard_status,
        guard_reasons=tuple(bundle.guard_reasons),
        guard_metrics={},
        probability_source=bundle.probability_source,
        branch_type=bundle.branch_type,
        diagnostics={
            "mean_speed_gap_mps": d.mean_speed_gap_mps,
            "final_progress_gap_m": d.final_progress_gap_m,
            "max_position_separation_m": d.max_position_separation_m,
            "mean_position_separation_m": d.mean_position_separation_m,
            "collapsed": d.collapsed,
            "collapse_reason": d.collapse_reason,
            "selection_space_eligible": d.selection_space_eligible,
            "path_speed_cap_active": d.path_speed_cap_active,
            "position_integration_error_max_m": d.position_integration_error_max_m,
            "acceleration_error_max_mps2": d.acceleration_error_max_mps2,
            "yaw_tangent_error_max_rad": d.yaw_tangent_error_max_rad,
            "curvature_error_max_per_m": d.curvature_error_max_per_m,
            "curvature_error_p95_per_m": d.curvature_error_p95_per_m,
            "native_path_cross_track_error_max_m": d.native_path_cross_track_error_max_m,
        },
    )


def _attach_sensors(world: Any, ego: Any) -> SensorBundle:
    import carla

    sb = SensorBundle()
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_w, cam_h = SIMLINGO_CAMERA_NATIVE_SIZE
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(SIMLINGO_CAMERA_FOV_DEG))
    cam_bp.set_attribute("sensor_tick", "0.05")
    sb.camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(*SIMLINGO_CAMERA_XYZ)),
        attach_to=ego,
    )
    camera_parent = getattr(sb.camera, "parent", None)
    if camera_parent is None or int(camera_parent.id) != int(ego.id):
        raise FixtureError("front_rgb_camera_not_attached_to_ego")
    sb.collision = world.spawn_actor(
        world.get_blueprint_library().find("sensor.other.collision"),
        carla.Transform(),
        attach_to=ego,
    )
    sb.lane = world.spawn_actor(
        world.get_blueprint_library().find("sensor.other.lane_invasion"),
        carla.Transform(),
        attach_to=ego,
    )

    def on_image(image: Any) -> None:
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4
        )
        frame = carla_bgra_to_bgr(bgra)
        with sb.image_lock:
            sb.latest_image["rgb"] = frame
            sb.latest_image["layout"] = "bgr"
            sb.latest_image["frame"] = int(image.frame)
            sb.latest_image["received"] = int(sb.latest_image["received"]) + 1

    def on_collision(event: Any) -> None:
        impulse = event.normal_impulse
        mag = math.sqrt(
            float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2
        )
        with sb.event_lock:
            if not sb._collision_active:
                sb.collision_episodes += 1
                sb._collision_active = True
                if sb.first_collision_time_s is None:
                    sb.first_collision_time_s = float(
                        getattr(getattr(event, "timestamp", None), "elapsed_seconds", 0.0)
                        or 0.0
                    )
            sb.collision_impulse_sum += mag

    def on_lane(event: Any) -> None:
        with sb.event_lock:
            if not sb._lane_prev:
                sb.lane_invasion_episodes += 1
            sb._lane_prev = True

    sb.camera.listen(on_image)
    sb.collision.listen(on_collision)
    sb.lane.listen(on_lane)
    return sb


def _destroy_sensors(sb: SensorBundle, *, client: Any | None = None) -> None:
    sensor_ids: list[int] = []
    for s in (sb.lane, sb.collision, sb.camera):
        if s is None:
            continue
        try:
            s.stop()
        except Exception:
            pass
        try:
            sensor_ids.append(int(s.id))
        except Exception:
            pass
    # Sensor actors own callback closures that retain this SensorBundle.  Drop
    # every Python proxy before server-side destruction to avoid accumulating
    # destroyed proxies across a long campaign.
    sb.lane = None
    sb.collision = None
    sb.camera = None
    with sb.image_lock:
        sb.latest_image["rgb"] = None
    if sensor_ids and client is not None:
        try:
            import carla

            client.apply_batch_sync(
                [carla.command.DestroyActor(sensor_id) for sensor_id in sensor_ids],
                False,
            )
        except Exception:
            pass


def _hold_all_braked(session: FixtureSession) -> None:
    import carla

    for sp in session.spawned:
        if hasattr(sp.actor, "apply_control"):
            sp.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))


def _wait_camera(
    sb: SensorBundle,
    world: Any,
    session: FixtureSession | None = None,
    *,
    min_frames: int = 3,
    max_ticks: int = 80,
) -> np.ndarray:
    for _ in range(max_ticks):
        if session is not None:
            _hold_all_braked(session)
        world.tick()
        with sb.image_lock:
            if (
                sb.latest_image["rgb"] is not None
                and int(sb.latest_image["received"]) >= min_frames
            ):
                return np.ascontiguousarray(sb.latest_image["rgb"])
    raise FixtureError("SENSOR_SYNC_FAILURE: camera frames not ready")


def _pin_decision_state(session: FixtureSession, world: Any) -> None:
    """Re-pin registry transform/velocity at decision time (after camera wait)."""
    _reapply_exact_state(session)
    _release_decision_controls(session)
    world.tick()
    _reapply_exact_state(session)
    _release_decision_controls(session)
    world.tick()


def _release_decision_controls(session: FixtureSession) -> None:
    """Clear camera-wait braking before history/anchor measurement."""
    import carla

    for spawned in session.spawned:
        if spawned.role == "ego" and hasattr(spawned.actor, "apply_control"):
            spawned.actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=0.0,
                    steer=0.0,
                    hand_brake=False,
                    reverse=False,
                )
            )
    _apply_npc_scripts(session, 0.0)


def _prepare_decision_state(
    session: FixtureSession,
    world: Any,
    *,
    collect_observable_history: bool,
) -> dict[str, Any] | None:
    """Rebuild exact state and optionally replay one observable pre-anchor second."""
    if not collect_observable_history:
        _pin_decision_state(session, world)
        return None
    from driving_vla.evaluation.observable_history import ObservableHistoryRecorder

    recorder = ObservableHistoryRecorder(
        scenario_id=session.fixture.scenario_id,
        seed_id=session.fixture.seed_id,
    )
    _reapply_exact_state(session)
    _release_decision_controls(session)
    for index in range(21):
        if index:
            _apply_npc_scripts(session, index * 0.05)
            world.tick()
        snapshot = world.get_snapshot()
        recorder.record(
            simulation_time_s=float(snapshot.timestamp.elapsed_seconds),
            frame=int(snapshot.frame),
            spawned_actors=session.spawned,
        )
    anchor_time = float(world.get_snapshot().timestamp.elapsed_seconds)
    return recorder.finalize(anchor_time_s=anchor_time)


def _ego_pose(ego: Any) -> EgoPose:
    tf = ego.get_transform()
    v = ego.get_velocity()
    speed = math.hypot(float(v.x), float(v.y))
    return EgoPose(
        x=float(tf.location.x),
        y=float(tf.location.y),
        yaw=math.radians(float(tf.rotation.yaw)),
        speed_mps=float(speed),
    )


def _observable_actor_scene(
    session: FixtureSession,
    pose: EgoPose,
) -> dict[str, Any]:
    """Current candidate-blind actor state in the ego frame.

    This is captured before candidate construction and contains no scripted
    future, candidate identity, or oracle result.
    """
    conflict_actor = next(
        (item.actor for item in session.spawned if item.role != "ego"),
        None,
    )
    if conflict_actor is None:
        return {
            "actor_present": False,
            "actor_lon_m": None,
            "actor_lat_m": None,
            "actor_speed_mps": None,
        }
    actor_tf = conflict_actor.get_transform()
    actor_v = conflict_actor.get_velocity()
    dx = float(actor_tf.location.x - pose.x)
    dy = float(actor_tf.location.y - pose.y)
    fx, fy = math.cos(pose.yaw), math.sin(pose.yaw)
    return {
        "actor_present": True,
        "actor_lon_m": dx * fx + dy * fy,
        "actor_lat_m": -dx * fy + dy * fx,
        "actor_speed_mps": math.hypot(float(actor_v.x), float(actor_v.y)),
    }


def _route_progress_m(
    route_xy: tuple[tuple[float, float], ...], ego_x: float, ego_y: float
) -> float:
    if len(route_xy) < 2:
        return 0.0
    best_s = 0.0
    best_d = 1e18
    acc = 0.0
    for i in range(len(route_xy) - 1):
        x0, y0 = route_xy[i]
        x1, y1 = route_xy[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        # project ego onto segment
        if seg < 1e-6:
            continue
        t = max(0.0, min(1.0, ((ego_x - x0) * (x1 - x0) + (ego_y - y0) * (y1 - y0)) / (seg * seg)))
        px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        d = math.hypot(ego_x - px, ego_y - py)
        s = acc + t * seg
        if d < best_d:
            best_d = d
            best_s = s
        acc += seg
    return float(best_s)


def _min_actor_clearance(ego: Any, others: list[Any]) -> float | None:
    if not others:
        return None
    et = ego.get_transform().location
    best = None
    for o in others:
        ol = o.get_transform().location
        d = math.hypot(float(et.x - ol.x), float(et.y - ol.y))
        # subtract approximate half-lengths
        d_adj = max(0.0, d - 3.5)
        best = d_adj if best is None else min(best, d_adj)
    return best


def _ttc_s(ego: Any, others: list[Any]) -> float | None:
    """Simple longitudinal closing TTC; null if no closing actor."""
    if not others:
        return None
    et = ego.get_transform()
    ev = ego.get_velocity()
    yaw = math.radians(float(et.rotation.yaw))
    fx, fy = math.cos(yaw), math.sin(yaw)
    ego_speed = float(ev.x) * fx + float(ev.y) * fy
    best = None
    for o in others:
        ot = o.get_transform()
        ov = o.get_velocity()
        dx = float(ot.location.x - et.location.x)
        dy = float(ot.location.y - et.location.y)
        along = dx * fx + dy * fy
        if along <= 0.5:
            continue
        other_speed = float(ov.x) * fx + float(ov.y) * fy
        closing = ego_speed - other_speed
        if closing <= 0.05:
            continue
        ttc = along / closing
        if ttc > 0:
            best = ttc if best is None else min(best, ttc)
    return best


def _is_offroad(world: Any, ego: Any) -> bool:
    loc = ego.get_transform().location
    wp = world.get_map().get_waypoint(loc, project_to_road=True)
    if wp is None:
        return True
    d = math.hypot(float(loc.x - wp.transform.location.x), float(loc.y - wp.transform.location.y))
    return d > 2.5


def _apply_npc_scripts(session: FixtureSession, t_s: float) -> None:
    # Keep NPC and traffic-control phases on the same simulation-time clock.
    # Candidate identity is not an input to either script.
    from driving_vla.evaluation.fixture_runtime import _apply_scripts

    _apply_scripts(
        session,
        simulation_time_since_anchor_s=float(t_s),
        include_ego=False,
    )


def _make_executors(ego: Any) -> tuple[VLAPathManager, VLASpeedPlanner, ConstrainedVLAMPC, float]:
    geom = vehicle_geometry_from_carla_vehicle(ego)
    wheelbase = float(geom.wheelbase_m)
    max_steer = float(geom.max_steer_rad)
    path_manager = VLAPathManager(
        PathManagerConfig(
            max_abs_curvature=0.30,
            max_switch_lateral_5m=1.0,
            max_switch_heading_5m_deg=12.0,
        )
    )
    speed_planner = VLASpeedPlanner(
        VLASpeedConfig(max_speed_mps=8.0, calibration_gain=1.0, max_accel_mps2=2.50)
    )
    # A cold branch rebuild must initialize the stateful slew limiter from the
    # measured vehicle state.  Leaving it at the constructor's 0 m/s makes the
    # one-shot anchor update rise to only ~0.1 m/s, forcing every candidate to
    # full brake and erasing longitudinal selection space.
    velocity = ego.get_velocity()
    measured_speed = math.sqrt(
        float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
    )
    speed_planner.reset(target_speed_mps=measured_speed)
    tracker = ConstrainedVLAMPC(
        VLAMPCConfig(
            control_dt_s=0.05,
            prediction_dt_s=0.10,
            horizon=20,
            wheelbase_m=wheelbase,
            max_steer_rad=max_steer,
            max_speed_mps=8.0,
            path_stale_soft_s=1.0,
            path_stale_hard_s=2.5,
            path_stale_zero_s=5.0,
            solver_deadline_ms=30.0,
        )
    )
    return path_manager, speed_planner, tracker, max_steer


@dataclass
class BranchLiveResult:
    report: BranchExecutionReport
    metrics: Any
    ticks: list[TickRecord]
    summary: dict[str, Any]
    control_payload_ok: bool
    control_seq: list[dict[str, Any]] = field(default_factory=list)


def run_anchor_v2_from_bundle(
    bundle: Any,
    *,
    pair_id: str,
    scenario_id: str,
    seed_id: str,
    anchor_run_id: str,
    anchor_carla_frame: int,
    anchor_simulation_time_s: float,
    requested_initial_state_hash: str,
    measured_initial_state_hash: str,
    observation_fingerprint: ObservationFingerprint,
    model_checkpoint_hash: str,
    executor_config_hash: str,
    evidence_lineage: str = "spatial_mode_head",
) -> Any:
    """Freeze a Guarded V2 bundle into K2AnchorArtifactV2 (no second forward)."""
    from driving_vla.evaluation.k2_spatial_artifact import artifact_from_bundle_v2

    return artifact_from_bundle_v2(
        bundle,
        pair_id=pair_id,
        scenario_id=scenario_id,
        seed_id=seed_id,
        anchor_run_id=anchor_run_id,
        anchor_carla_frame=anchor_carla_frame,
        anchor_simulation_time_s=anchor_simulation_time_s,
        requested_initial_state_hash=requested_initial_state_hash,
        measured_initial_state_hash=measured_initial_state_hash,
        observation_fingerprint=observation_fingerprint,
        model_checkpoint_hash=model_checkpoint_hash,
        executor_config_hash=executor_config_hash,
        evidence_lineage=evidence_lineage,
    )


def run_anchor_v2(
    *,
    client: Any,
    world: Any,
    fixture: ScenarioSeedFixture,
    policy: NeuralV2Policy,
    pair_id: str,
    model_checkpoint_hash: str,
    registry_hash: str,
    evidence_dir: Path,
    forward_counter: dict[str, int],
    executor_config_hash: str = EXECUTOR_CONFIG_HASH,
    spectator_follow: bool = True,
    collect_observable_history: bool = False,
) -> tuple[K2AnchorArtifactV2, dict[str, Any]]:
    """X5G: one NeuralV2Policy forward → Guard V2 → freeze V2 artifact.

    Branch phases must use the frozen artifact only (no second VLA forward).
    Does not modify V1 ``run_anchor`` semantics.
    """
    from driving_vla.model.driving_feature import DrivingFeatureError
    from driving_vla.model.k2_spatial_types import GUARD_OK as SPATIAL_GUARD_OK

    apply_weather(world, fixture.weather)
    session = open_fixture_session(client, world, fixture, settle_ticks=8)
    sb: SensorBundle | None = None
    try:
        ego = next(s.actor for s in session.spawned if s.role == "ego")
        if spectator_follow:
            set_spectator_follow(world, ego)
        sb = _attach_sensors(world, ego)
        image = _wait_camera(sb, world, session)
        observable_history = _prepare_decision_state(
            session,
            world,
            collect_observable_history=collect_observable_history,
        )
        if observable_history is not None:
            with sb.image_lock:
                image = np.ascontiguousarray(sb.latest_image["rgb"])
        else:
            image = _wait_camera(sb, world, session, min_frames=1, max_ticks=10)
        measured = measure_initial_state(session)
        pose = _ego_pose(ego)
        observable_actor_scene = _observable_actor_scene(session, pose)
        route_xy = _route_xy(fixture)
        tp1, tp2 = _nav_targets_ego(
            route_xy, ego_x=pose.x, ego_y=pose.y, ego_yaw=pose.yaw
        )
        snap = world.get_snapshot()
        frame_id = f"anchor-v2-{int(snap.frame)}"
        obs = ObservationBundle(
            run_id=pair_id,
            frame_id=frame_id,
            scenario_id=fixture.scenario_id,
            simulation_time_s=float(snap.timestamp.elapsed_seconds),
            wall_time_s=time.time(),
            carla_frame=int(snap.frame),
            ego_x=pose.x,
            ego_y=pose.y,
            ego_yaw=pose.yaw,
            ego_v=pose.speed_mps,
            route_xy=route_xy,
            front_rgb=image,
            meta={
                "official_contract": True,
                "image_layout": "bgr",
                "target_ego_1": tp1,
                "target_ego_2": tp2,
                "command_text": None,
                "prompt_mode": "target_point",
                "camera_mount_xyz": list(SIMLINGO_CAMERA_XYZ),
                "observable_scene_v1": observable_actor_scene,
            },
        )
        forward_counter["n"] = int(forward_counter.get("n", 0)) + 1
        if forward_counter["n"] != 1:
            raise FixtureError(
                f"v2 anchor forward_count must be 1 at first call, got {forward_counter['n']}"
            )
        t0 = time.perf_counter()
        try:
            bundle = policy.predict_bundle(obs)
        except DrivingFeatureError as exc:
            raise FixtureError(f"V2_FEATURE_FAIL_CLOSED: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if int(policy.last_forward_count) != 1:
            raise FixtureError(
                f"v2 policy reported forward_count={policy.last_forward_count}, expected 1"
            )
        gstat = str(getattr(bundle, "guard_status", "") or "")
        # Soft-warn: still freeze artifact so force smoke can diagnose rejects.
        # Hard fail only if candidates missing / schema broken.
        if not getattr(bundle, "candidates", None) or len(bundle.candidates) != 2:
            raise FixtureError(
                f"V2_BUNDLE_INVALID: guard={gstat} reasons={getattr(bundle, 'guard_reasons', ())}"
            )

        rgb_sha = sha256_hex(np.ascontiguousarray(image).tobytes())
        k2_hash = content_hash(
            {
                "native": getattr(bundle, "native_path_hash", ""),
                "cids": list(bundle.candidate_ids())
                if hasattr(bundle, "candidate_ids")
                else [],
                "config": getattr(bundle, "config_hash", ""),
                "fwd": getattr(bundle, "backbone_forward_id", ""),
            },
            nibble=32,
        )
        obs_fp = ObservationFingerprint(
            front_rgb_sha256=rgb_sha,
            image_height=int(image.shape[0]),
            image_width=int(image.shape[1]),
            image_channels=int(image.shape[2]) if image.ndim == 3 else 1,
            image_layout="bgr",
            ego_observable={
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
                "v": pose.speed_mps,
            },
            route_targets=[list(tp1), list(tp2)],
            camera_frame={
                "mount_xyz": list(SIMLINGO_CAMERA_XYZ),
                "fov": SIMLINGO_CAMERA_FOV_DEG,
                "carla_frame": int(snap.frame),
            },
            k2_bundle_hash=k2_hash,
        )
        anchor_run_id = compute_run_id(pair_id=pair_id, role="anchor", attempt_id=0)
        native = getattr(policy, "last_native", None)
        art = run_anchor_v2_from_bundle(
            bundle,
            pair_id=pair_id,
            scenario_id=fixture.scenario_id,
            seed_id=fixture.seed_id,
            anchor_run_id=anchor_run_id,
            anchor_carla_frame=int(snap.frame),
            anchor_simulation_time_s=float(snap.timestamp.elapsed_seconds),
            requested_initial_state_hash=fixture.requested_initial_state_hash(),
            measured_initial_state_hash=measured.measured_hash(),
            observation_fingerprint=obs_fp,
            model_checkpoint_hash=model_checkpoint_hash,
            executor_config_hash=executor_config_hash,
            evidence_lineage="spatial_mode_head",
        )
        raw = art.to_json_bytes()
        art2 = K2AnchorArtifactV2.from_json_bytes(raw)
        if art2.artifact_content_hash() != art.artifact_content_hash():
            raise FixtureError("v2 anchor serialize round-trip hash mismatch")

        adir = evidence_dir / "anchor"
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "anchor_bundle_v2.json").write_bytes(raw)
        np.save(adir / "anchor_front_rgb.npy", image)
        _write_json(adir / "initial_state_raw.json", measured.raw_dict())
        feat_lineage = {
            "schema_version": "safedrive.r2x.v2_anchor_lineage.v1",
            "backbone_forward_id": str(getattr(bundle, "backbone_forward_id", "") or ""),
            "spatial_head_checkpoint_hash": str(
                getattr(bundle, "spatial_head_checkpoint_hash", "") or ""
            ),
            "base_checkpoint_hash": str(
                getattr(bundle, "base_checkpoint_hash", "") or model_checkpoint_hash
            ),
            "driving_feature_hash": str(
                (getattr(bundle, "set_diagnostics", {}) or {}).get(
                    "driving_feature_hash", ""
                )
            ),
            "driving_feature_ok": bool(
                (getattr(bundle, "set_diagnostics", {}) or {}).get(
                    "driving_feature_ok", False
                )
            ),
            "native_feature_hash": str(
                getattr(native, "driving_feature_hash", "") if native else ""
            ),
            "native_raw_hash": str(
                getattr(native, "driving_feature_raw_hash", "") if native else ""
            ),
            "native_feature_source": str(
                getattr(native, "driving_feature_source", "") if native else ""
            ),
        }
        _write_json(adir / "feature_lineage.json", feat_lineage)
        mean64 = list(getattr(native, "driving_feature", ()) or ())
        full_pool = list(getattr(native, "driving_feature_full_pool", ()) or ())
        raw_tokens_audit = int(pair_id[:8], 16) % 20 == 0
        condition_variant = (
            fixture.scenario_id.split("__", 1)[1]
            if "__" in fixture.scenario_id
            else "base"
        )
        conflict_side = "none"
        actor_lon_m = observable_actor_scene["actor_lon_m"]
        actor_lat_m = observable_actor_scene["actor_lat_m"]
        actor_speed_mps = observable_actor_scene["actor_speed_mps"]
        if actor_lat_m is not None and abs(float(actor_lat_m)) >= 0.25:
            conflict_side = "left" if float(actor_lat_m) > 0.0 else "right"
        feature_payload = {
            "schema_version": "safedrive.r23_anchor_feature.v1",
            "anchor_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "scenario_family": fixture.family,
            "lineage_id": fixture.scenario_id.split("__", 1)[0],
            "episode_id": fixture.scenario_id.split("__", 1)[0],
            "condition_variant": condition_variant,
            "conflict_side": conflict_side,
            "actor_lon_m": actor_lon_m,
            "actor_lat_m": actor_lat_m,
            "actor_speed_mps": actor_speed_mps,
            "adjacent_lane_authorized": bool(
                "obstruction_layout=adjacent_same_direction"
                in str(fixture.notes or "")
            ),
            "ego_v": pose.speed_mps,
            "native_path_xy": [
                [float(x), float(y)]
                for x, y in getattr(native, "path_map_xy", ())
            ],
            "driving_feature": mean64,
            "driving_feature_hash": str(
                getattr(native, "driving_feature_hash", "") or ""
            ),
            "driving_feature_source": str(
                getattr(native, "driving_feature_source", "") or ""
            ),
            "full_pool_saved": True,
            "raw_tokens_audit": raw_tokens_audit,
            "full_pool": full_pool,
            "full_pool_hash": str(
                getattr(native, "driving_feature_full_pool_hash", "") or ""
            ),
            "model_checkpoint_hash": model_checkpoint_hash,
            "observation_hash": content_hash(obs_fp.to_dict(), nibble=64),
            "backbone_forward_id": str(getattr(bundle, "backbone_forward_id", "") or ""),
        }
        _write_json(adir / "feature.json", feature_payload)
        if observable_history is not None:
            _write_json(adir / "observable_history.json", observable_history)
        meta = {
            "pair_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "registry_hash": registry_hash,
            "artifact_schema": K2_ANCHOR_SCHEMA_V2,
            "artifact_content_hash": art.artifact_content_hash(),
            "guard_status": art.guard_status,
            "forward_count": 1,
            "latency_ms": latency_ms,
            "peak_vram_mb": float(policy.last_peak_vram_mb),
            "candidate_ids": [
                c.candidate_id for c in art.candidates
            ],
            "top1_index": art.top1_index,
            "native_path_hash": art.native_path_hash,
            "backbone_forward_id": art.backbone_forward_id,
            "spatial_head_checkpoint_hash": art.spatial_head_checkpoint_hash,
            "requested_initial_state_hash": art.requested_initial_state_hash,
            "measured_initial_state_hash": art.measured_initial_state_hash,
            "feature_lineage": feat_lineage,
            "policy_model_id": NeuralV2Policy.model_id,
            "observable_history_collected": observable_history is not None,
        }
        _write_json(adir / "run_config.json", meta)
        print(
            f"[anchor-v2] guard={art.guard_status} hash={art.artifact_content_hash()[:16]}… "
            f"latency_ms={latency_ms:.1f} fwd={art.backbone_forward_id[:24]}",
            flush=True,
        )
        return art, meta
    finally:
        if sb is not None:
            _destroy_sensors(sb, client=client)
        cleanup_session(session, soft=True)


def run_anchor(
    *,
    client: Any,
    world: Any,
    fixture: ScenarioSeedFixture,
    policy: NeuralV1Policy,
    pair_id: str,
    model_checkpoint_hash: str,
    retimer_hash: str,
    registry_hash: str,
    evidence_dir: Path,
    forward_counter: dict[str, int],
    spectator_follow: bool = True,
) -> tuple[K2AnchorArtifactV1, dict[str, Any]]:
    """Cold rebuild + one real forward + artifact save. No branch control. (V1 longitudinal)"""
    apply_weather(world, fixture.weather)
    session = open_fixture_session(client, world, fixture, settle_ticks=8)
    sb: SensorBundle | None = None
    try:
        ego = next(s.actor for s in session.spawned if s.role == "ego")
        if spectator_follow:
            set_spectator_follow(world, ego)
        sb = _attach_sensors(world, ego)
        image = _wait_camera(sb, world, session)
        # Decision-time pin: measured initial state must match across branches.
        _pin_decision_state(session, world)
        image = _wait_camera(sb, world, session, min_frames=1, max_ticks=10)
        measured = measure_initial_state(session)
        pose = _ego_pose(ego)
        route_xy = _route_xy(fixture)
        tp1, tp2 = _nav_targets_ego(
            route_xy, ego_x=pose.x, ego_y=pose.y, ego_yaw=pose.yaw
        )
        snap = world.get_snapshot()
        frame_id = f"anchor-{int(snap.frame)}"
        obs = ObservationBundle(
            run_id=pair_id,
            frame_id=frame_id,
            scenario_id=fixture.scenario_id,
            simulation_time_s=float(snap.timestamp.elapsed_seconds),
            wall_time_s=time.time(),
            carla_frame=int(snap.frame),
            ego_x=pose.x,
            ego_y=pose.y,
            ego_yaw=pose.yaw,
            ego_v=pose.speed_mps,
            route_xy=route_xy,
            front_rgb=image,
            meta={
                "official_contract": True,
                "image_layout": "bgr",
                "target_ego_1": tp1,
                "target_ego_2": tp2,
                "command_text": None,
                "prompt_mode": "target_point",
                "camera_mount_xyz": list(SIMLINGO_CAMERA_XYZ),
            },
        )
        # Exactly one forward
        forward_counter["n"] = int(forward_counter.get("n", 0)) + 1
        if forward_counter["n"] != 1:
            raise FixtureError(
                f"anchor forward_count must be 1 at first call, got {forward_counter['n']}"
            )
        t0 = time.perf_counter()
        bundle = policy.predict_bundle(obs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if int(policy.last_forward_count) != 1:
            raise FixtureError(
                f"policy reported forward_count={policy.last_forward_count}, expected 1"
            )
        if bundle.guard_status != GUARD_OK:
            raise FixtureError(
                f"GUARD_REJECT: {bundle.guard_status} reasons={bundle.guard_reasons}"
            )

        rgb_sha = sha256_hex(np.ascontiguousarray(image).tobytes())
        k2_hash = content_hash(
            {
                "native": bundle.native_path_hash,
                "cids": list(bundle.candidate_ids()),
                "config": bundle.config_hash,
            },
            nibble=32,
        )
        obs_fp = ObservationFingerprint(
            front_rgb_sha256=rgb_sha,
            image_height=int(image.shape[0]),
            image_width=int(image.shape[1]),
            image_channels=int(image.shape[2]) if image.ndim == 3 else 1,
            image_layout="bgr",
            ego_observable={
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
                "v": pose.speed_mps,
            },
            route_targets=[list(tp1), list(tp2)],
            camera_frame={
                "mount_xyz": list(SIMLINGO_CAMERA_XYZ),
                "fov": SIMLINGO_CAMERA_FOV_DEG,
                "carla_frame": int(snap.frame),
            },
            k2_bundle_hash=k2_hash,
        )
        anchor_run_id = compute_run_id(pair_id=pair_id, role="anchor", attempt_id=0)
        art = artifact_from_bundle(
            bundle,
            pair_id=pair_id,
            scenario_id=fixture.scenario_id,
            seed_id=fixture.seed_id,
            anchor_run_id=anchor_run_id,
            anchor_carla_frame=int(snap.frame),
            anchor_simulation_time_s=float(snap.timestamp.elapsed_seconds),
            requested_initial_state_hash=fixture.requested_initial_state_hash(),
            measured_initial_state_hash=measured.measured_hash(),
            observation_fingerprint=obs_fp,
            model_checkpoint_hash=model_checkpoint_hash,
            retimer_hash=retimer_hash,
            executor_config_hash=EXECUTOR_CONFIG_HASH,
        )
        # Round-trip hash check
        raw = art.to_json_bytes()
        art2 = K2AnchorArtifactV1.from_json_bytes(raw)
        if art2.artifact_content_hash() != art.artifact_content_hash():
            raise FixtureError("anchor serialize round-trip hash mismatch")

        adir = evidence_dir / "anchor"
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "anchor_bundle.json").write_bytes(raw)
        np.save(adir / "anchor_front_rgb.npy", image)
        _write_json(adir / "initial_state_raw.json", measured.raw_dict())
        meta = {
            "pair_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "registry_hash": registry_hash,
            "artifact_content_hash": art.artifact_content_hash(),
            "guard_status": art.guard_status,
            "forward_count": 1,
            "latency_ms": latency_ms,
            "peak_vram_mb": float(policy.last_peak_vram_mb),
            "candidate_ids": list(bundle.candidate_ids()),
            "top1_index": art.top1_index,
            "native_path_hash": art.native_path_hash,
            "requested_initial_state_hash": art.requested_initial_state_hash,
            "measured_initial_state_hash": art.measured_initial_state_hash,
        }
        _write_json(adir / "run_config.json", meta)
        print(
            f"[anchor] guard={art.guard_status} hash={art.artifact_content_hash()[:16]}… "
            f"latency_ms={latency_ms:.1f}",
            flush=True,
        )
        return art, meta
    finally:
        if sb is not None:
            _destroy_sensors(sb, client=client)
        # soft: dead-server must not burn minutes on destroy timeouts
        cleanup_session(session, soft=True)


def run_branch(
    *,
    client: Any,
    world: Any,
    fixture: ScenarioSeedFixture,
    artifact: K2AnchorArtifactV1 | K2AnchorArtifactV2 | K2AnchorArtifactV3,
    force_index: int,
    pair_id: str,
    registry_hash: str,
    model_retimer_hash: str,
    evidence_dir: Path,
    forward_counter: dict[str, int],
    policy: NeuralV1Policy | None = None,
    spectator_follow: bool = True,
    spectator_wall_pace_s: float = 0.0,
    attempt_id: int = 0,
    collect_actor_future: bool = True,
    allow_nominal_only_fallback: bool = False,
    collect_observable_history: bool = False,
    include_scenario_family: bool = True,
) -> BranchLiveResult:
    """Cold rebuild, load frozen artifact, force-execute candidate for 2.5 s.

    Supports V1, Spatial V2 and route-bound V3 artifacts. Branch never
    re-forwards VLA.
    """
    assert force_index in (0, 1)
    apply_weather(world, fixture.weather)
    session = open_fixture_session(client, world, fixture, settle_ticks=8)
    sb: SensorBundle | None = None
    pending: dict[str, Any] | None = None
    cleanup_ok = True
    cleanup_error: str | None = None
    role = f"branch_{force_index}"
    bdir = evidence_dir / f"branch-{force_index}"
    bdir.mkdir(parents=True, exist_ok=True)
    try:
        ego = next(s.actor for s in session.spawned if s.role == "ego")
        spectator_updates = 0
        spectator_failures = 0
        spectator_last_error: str | None = None
        if spectator_follow:
            ok, err = set_spectator_follow(world, ego)
            spectator_updates += int(ok)
            spectator_failures += int(not ok)
            spectator_last_error = err
            print(
                f"[spectator] branch-{force_index} follow={'ON' if ok else 'FAILED'} "
                f"pace={max(0.0, float(spectator_wall_pace_s)):.2f}s/tick",
                flush=True,
            )
            if not ok:
                raise FixtureError(
                    f"spectator_follow_initialization_failed:{err}"
                )
        npcs = [s.actor for s in session.spawned if s.role != "ego"]
        sb = _attach_sensors(world, ego)
        # Branch-start RGB for diagnostics only (no VLA)
        branch_rgb = _wait_camera(sb, world, session)
        observable_history = _prepare_decision_state(
            session,
            world,
            collect_observable_history=collect_observable_history,
        )
        if observable_history is not None:
            with sb.image_lock:
                branch_rgb = np.ascontiguousarray(sb.latest_image["rgb"])
        else:
            branch_rgb = _wait_camera(sb, world, session, min_frames=1, max_ticks=10)
        measured = measure_initial_state(session)
        # Prove no second forward: call site must not use policy.predict_bundle
        forwards_before = int(forward_counter.get("n", 0))
        selection = select_from_anchor_artifact(
            artifact,
            force_index=force_index,
            allow_nominal_only_fallback=allow_nominal_only_fallback,
        )
        path_manager, speed_planner, tracker, max_steer = _make_executors(ego)
        pose = _ego_pose(ego)
        snap0 = world.get_snapshot()
        t0 = float(snap0.timestamp.elapsed_seconds)
        observable_scene_t0 = capture_observable_scene_t0(
            scenario_id=fixture.scenario_id,
            seed_id=fixture.seed_id,
            spawned_actors=session.spawned,
            route_waypoints=fixture.route.waypoints,
            map_name=fixture.map_name,
            family=fixture.family if include_scenario_family else None,
            simulation_time_s=t0,
            frame=int(snap0.frame),
            road_polylines=capture_observable_road_context(
                world=world,
                ego=ego,
                route_waypoints=fixture.route.waypoints,
            ),
        )
        if observable_history is not None:
            from driving_vla.evaluation.observable_history import merge_history_into_scene

            observable_scene_t0 = merge_history_into_scene(
                observable_scene_t0, observable_history
            )
        _write_json(bdir / "observable_scene_t0.json", observable_scene_t0)
        frame_id = f"branch{force_index}-{int(snap0.frame)}"
        applied = apply_k2_to_executors(
            selection,
            speed_planner=speed_planner,
            path_manager=path_manager,
            ego=pose,
            stamp_s=t0,
            frame_id=frame_id,
            dt_s=0.05,
            ego_speed_mps=pose.speed_mps,
            nav_target_map_xy=_route_xy(fixture)[min(1, len(_route_xy(fixture)) - 1)],
        )
        if applied.selected_candidate_id != selection.candidate_id:
            raise FixtureError("EXECUTION_BINDING_FAILURE: selected id mismatch")
        if selection.candidate_id not in applied.source_id:
            raise FixtureError("EXECUTION_BINDING_FAILURE: source_id missing candidate")
        committed_at_apply = applied.path_update.committed
        committed_path_hash = None
        if committed_at_apply is not None:
            from driving_vla.model.k2_spatial_types import stable_hash_xy

            committed_path_hash = stable_hash_xy(
                tuple(
                    (float(x), float(y))
                    for x, y in zip(
                        committed_at_apply.x.tolist(),
                        committed_at_apply.y.tolist(),
                    )
                )
            )

        route_xy = _route_xy(fixture)
        actor_future_collector = (
            ActorFutureCollector(
                scenario_id=fixture.scenario_id,
                seed_id=fixture.seed_id,
                pair_id=pair_id,
                attempt_id=attempt_id,
                branch_index=force_index,
                anchor_artifact_hash=artifact.artifact_content_hash(),
                registry_hash=registry_hash,
                model_hash=model_retimer_hash,
                guard_hash=str(getattr(artifact, "config_hash", "") or model_retimer_hash),
                executor_hash=EXECUTOR_CONFIG_HASH,
            )
            if collect_actor_future
            else None
        )
        ticks: list[TickRecord] = []
        control_seq: list[dict[str, Any]] = []
        selected_ok = True
        executed_ok = True
        source_ok = True
        mpc_solved = 0
        mpc_timeout = 0
        mpc_fallback = 0
        prev_v = pose.speed_mps
        prev_steer = 0.0
        prev_a = 0.0
        n_ticks = EXPECTED_PRIMARY_TICKS
        outcome_path = bdir / "outcome_trace.jsonl"
        oracle_path = bdir / "oracle_trace.jsonl"
        if outcome_path.exists():
            outcome_path.unlink()
        if oracle_path.exists():
            oracle_path.unlink()

        first_coll_tick: int | None = None
        for i in range(n_ticks):
            with sb.event_lock:
                sb._collision_active = False
                sb._lane_prev = False

            snap = world.get_snapshot()
            sim_s = float(snap.timestamp.elapsed_seconds)
            t_rel = (i + 1) * 0.05
            _apply_npc_scripts(session, t_rel)
            pose = _ego_pose(ego)
            committed = path_manager.committed
            if committed is None:
                raise FixtureError("EXECUTION_BINDING_FAILURE: no committed path")
            try:
                ctrl_now = ego.get_control()
                measured_steer = float(ctrl_now.steer) * max_steer
            except Exception:
                measured_steer = None
            cmd = tracker.step(committed, pose, measured_steer_rad=measured_steer, now_s=sim_s)
            status = str(cmd.solver_status).lower()
            if "solved" in status:
                mpc_status = "solved"
                mpc_solved += 1
            elif "timeout" in status or "deadline" in status:
                mpc_status = "timeout"
                mpc_timeout += 1
            elif cmd.mode == "bounded_fallback" or cmd.mode != "mpc":
                mpc_status = "fallback"
                mpc_fallback += 1
            else:
                mpc_status = "solved"
                mpc_solved += 1

            steer_norm = float(np.clip(cmd.steer_rad / max(max_steer, 1e-6), -1.0, 1.0))
            accel = float(cmd.accel_mps2)
            throttle = float(np.clip(accel / 2.5, 0.0, 1.0))
            brake = float(np.clip(-accel / 3.0, 0.0, 1.0))
            import carla

            ego.apply_control(
                carla.VehicleControl(steer=steer_norm, throttle=throttle, brake=brake)
            )

            sid_now = str(getattr(committed, "source_id", "") or "")
            if selection.candidate_id not in sid_now:
                source_ok = False
            if applied.selected_candidate_id != selection.candidate_id:
                selected_ok = False
            if applied.executed_candidate_id != selection.candidate_id:
                executed_ok = False

            with sb.event_lock:
                ep = int(sb.collision_episodes)
                impulse = float(sb.collision_impulse_sum)
                lane_ep = int(sb.lane_invasion_episodes)
            if ep > 0 and first_coll_tick is None:
                first_coll_tick = i
            coll_now = first_coll_tick is not None and i >= first_coll_tick

            clearance = _min_actor_clearance(ego, npcs)
            ttc = _ttc_s(ego, npcs)
            offroad = _is_offroad(world, ego)
            progress = _route_progress_m(route_xy, pose.x, pose.y)
            long_a = float(cmd.accel_mps2)
            jerk = (long_a - prev_a) / 0.05
            prev_a = long_a
            steer_rate = abs(steer_norm - prev_steer) / 0.05
            prev_v = pose.speed_mps
            prev_steer = steer_norm

            tick = TickRecord(
                tick_index=i,
                simulation_time_s=t_rel,
                ego_x=pose.x,
                ego_y=pose.y,
                ego_yaw_rad=pose.yaw,
                ego_v=pose.speed_mps,
                selected_candidate_id=selection.candidate_id,
                executed_candidate_id=applied.executed_candidate_id,
                source_id=applied.source_id,
                path_age_s=float(cmd.path_age_s),
                freshness_regime=str(cmd.freshness_regime),
                mpc_mode=str(cmd.mode),
                mpc_status=mpc_status,
                mpc_latency_s=float(cmd.solver_ms) / 1000.0,
                collision=bool(coll_now),
                collision_impulse=float(impulse) if coll_now else 0.0,
                offroad=offroad,
                lane_invasion=lane_ep > 0,
                route_progress_m=progress,
                longitudinal_accel=long_a,
                lateral_accel=float(cmd.lateral_error_m),
                jerk=float(jerk),
                steer_rate=float(steer_rate),
                actor_clearance_m=clearance,
                ttc_s=ttc,
                oracle_only=True,
            )
            ticks.append(tick)
            rt = tick.runtime_dict()
            assert_no_oracle_in_control_payload(rt)
            _append_jsonl(outcome_path, rt)
            _append_jsonl(oracle_path, tick.oracle_dict())
            control_seq.append(
                {
                    "tick": i,
                    "t_s": t_rel,
                    "throttle": throttle,
                    "brake": brake,
                    "steer": steer_norm,
                    "mpc_status": mpc_status,
                    "solver_ms": float(cmd.solver_ms),
                    "selected_candidate_id": selection.candidate_id,
                    "source_id": applied.source_id,
                }
            )
            if spectator_follow:
                ok, err = set_spectator_follow(world, ego)
                spectator_updates += int(ok)
                spectator_failures += int(not ok)
                if err is not None:
                    spectator_last_error = err
                if not ok:
                    raise FixtureError(
                        f"spectator_follow_update_failed:{err}"
                    )
            world.tick()
            if actor_future_collector is not None:
                future_snapshot = world.get_snapshot()
                actor_future_collector.record(
                    time_s=t_rel,
                    frame=int(future_snapshot.frame),
                    actors=session.spawned,
                )
            if spectator_wall_pace_s > 0.0:
                time.sleep(float(spectator_wall_pace_s))

        if int(forward_counter.get("n", 0)) != forwards_before:
            raise FixtureError(
                f"branch re-forward forbidden: before={forwards_before} after={forward_counter.get('n')}"
            )

        metrics = aggregate_branch_outcome(
            ticks,
            candidate_id=selection.candidate_id,
            candidate_index=force_index,
        )
        from driving_vla.evaluation.maneuver_completion import (
            evaluate_overtake_completion,
        )

        final_pose = _ego_pose(ego)
        final_actor_scene = _observable_actor_scene(session, final_pose)
        maneuver_completion = evaluate_overtake_completion(
            family=fixture.family,
            route_xy=route_xy,
            ticks=ticks,
            final_actor_lon_m=final_actor_scene.get("actor_lon_m"),
        )
        if sb.collision_episodes > 0:
            from dataclasses import replace

            metrics = replace(
                metrics,
                collision_episode_count=int(sb.collision_episodes),
                first_collision_time_s=(
                    None if first_coll_tick is None else (first_coll_tick + 1) * 0.05
                ),
                collision_impulse_sum=float(sb.collision_impulse_sum),
            )
        actor_future_manifest = (
            actor_future_collector.finalize(bdir)
            if actor_future_collector is not None
            else None
        )

        # Stash pre-cleanup payload; BranchExecutionReport is finalized AFTER cleanup.
        pending = {
            "selection": selection,
            "measured": measured,
            "ticks": ticks,
            "metrics": metrics,
            "control_seq": control_seq,
            "selected_ok": selected_ok,
            "executed_ok": executed_ok,
            "source_ok": source_ok,
            "mpc_solved": mpc_solved,
            "mpc_timeout": mpc_timeout,
            "mpc_fallback": mpc_fallback,
            "branch_rgb": branch_rgb,
            "collision_episodes": int(sb.collision_episodes),
            "lane_invasion_episodes": int(sb.lane_invasion_episodes),
            "collision_impulse_sum": float(sb.collision_impulse_sum),
            "spectator_follow_enabled": bool(spectator_follow),
            "spectator_updates": int(spectator_updates),
            "spectator_failures": int(spectator_failures),
            "spectator_last_error": spectator_last_error,
            "spectator_wall_pace_s": float(spectator_wall_pace_s),
            "actor_future_manifest": actor_future_manifest,
            "maneuver_completion": maneuver_completion.to_dict(),
        }
    except Exception:
        fail = {"error": traceback.format_exc()}
        _write_json(bdir / "failure.json", fail)
        pending = None
        raise
    finally:
        cleanup_ok = True
        cleanup_error: str | None = None
        if sb is not None:
            try:
                _destroy_sensors(sb, client=client)
            except Exception as exc:  # noqa: BLE001
                cleanup_ok = False
                cleanup_error = f"sensor_destroy:{exc}"
        try:
            cleanup_session(session, soft=False)
        except Exception as exc:  # noqa: BLE001
            cleanup_ok = False
            cleanup_error = (
                f"{cleanup_error}; session:{exc}" if cleanup_error else f"session:{exc}"
            )
        if not cleanup_ok:
            _write_json(
                bdir / "cleanup_error.json",
                {"error": cleanup_error, "cleanup_ok": False},
            )

    # Final report only after cleanup (success path).
    if pending is None:
        raise FixtureError("branch pending payload missing after cleanup")
    selection = pending["selection"]
    measured = pending["measured"]
    metrics = pending["metrics"]
    failure_codes = finalize_branch_failure_codes(cleanup_ok=cleanup_ok)
    report = BranchExecutionReport(
        candidate_index=force_index,
        candidate_id=selection.candidate_id,
        anchor_artifact_hash=artifact.artifact_content_hash(),
        measured_initial_state=measured,
        registry_hash=registry_hash,
        scenario_id=fixture.scenario_id,
        seed_id=fixture.seed_id,
        model_retimer_hash=model_retimer_hash,
        executor_config_hash=EXECUTOR_CONFIG_HASH,
        completed_primary_ticks=len(pending["ticks"]),
        mpc_solved_ticks=int(pending["mpc_solved"]),
        mpc_timeout_ticks=int(pending["mpc_timeout"]),
        mpc_fallback_ticks=int(pending["mpc_fallback"]),
        selected_ids_consistent=bool(pending["selected_ok"]),
        executed_ids_consistent=bool(pending["executed_ok"]),
        source_ids_consistent=bool(pending["source_ok"]),
        sensor_sync_ok=True,
        tick_owner_ok=True,
        cleanup_ok=bool(cleanup_ok),
        spawn_ok=True,
        server_ok=True,
        failure_codes=failure_codes,
        weather=fixture.weather.to_dict(),
        traffic_light_state=dict(fixture.traffic_light),
        script_phase=dict(measured.actor_script_phase),
    )
    summary = {
        "role": role,
        "candidate_id": selection.candidate_id,
        "candidate_index": force_index,
        "artifact_hash": artifact.artifact_content_hash(),
        "metrics": metrics.to_dict(),
        "mpc_solved": pending["mpc_solved"],
        "mpc_timeout": pending["mpc_timeout"],
        "mpc_fallback": pending["mpc_fallback"],
        "collision_episodes": pending["collision_episodes"],
        "lane_invasion_episodes": pending["lane_invasion_episodes"],
        "branch_rgb_sha256": sha256_hex(
            np.ascontiguousarray(pending["branch_rgb"]).tobytes()
        ),
        "forwards_during_branch": 0,
        "forwards_total": int(forward_counter.get("n", 0)),
        "cleanup_ok": bool(cleanup_ok),
        "cleanup_error": cleanup_error,
        "committed_path_hash": committed_path_hash,
        "source_id": applied.source_id,
        "spectator_follow_enabled": pending["spectator_follow_enabled"],
        "spectator_updates": pending["spectator_updates"],
        "spectator_failures": pending["spectator_failures"],
        "spectator_last_error": pending["spectator_last_error"],
        "spectator_wall_pace_s": pending["spectator_wall_pace_s"],
        "actor_future_trace": pending["actor_future_manifest"],
        "maneuver_completion": pending["maneuver_completion"],
        "reactive_actor_present": any(
            spawned.role != "ego"
            and spawned.requested.script.script_type == "reactive_yield"
            for spawned in session.spawned
        ),
    }
    _write_json(bdir / "branch_summary.json", summary)
    _write_json(bdir / "initial_state_raw.json", measured.raw_dict())
    _write_json(bdir / "control_seq.json", {"ticks": pending["control_seq"]})
    _write_json(
        bdir / "run_config.json",
        {
            "pair_id": pair_id,
            "force_index": force_index,
            "candidate_id": selection.candidate_id,
            "artifact_hash": artifact.artifact_content_hash(),
            "cleanup_ok": bool(cleanup_ok),
        },
    )
    _write_json(
        bdir / "collision_episodes.json",
        {
            "episode_count": pending["collision_episodes"],
            "impulse_sum": pending["collision_impulse_sum"],
        },
    )
    print(
        f"[branch-{force_index}] cand={selection.candidate_id} ticks={len(pending['ticks'])} "
        f"mpc_solved={pending['mpc_solved']} coll={pending['collision_episodes']} "
        f"cleanup_ok={cleanup_ok}",
        flush=True,
    )
    return BranchLiveResult(
        report=report,
        metrics=metrics,
        ticks=pending["ticks"],
        summary=summary,
        control_payload_ok=True,
        control_seq=pending["control_seq"],
    )


def _identity_hashes_without_carla(
    *,
    device: str = "cuda",
) -> tuple[str, str, str, str]:
    """Compute model/retimer hashes without connecting CARLA (may load CUDA).

    Checkpoint file hash does not require GPU; model_id/config from k2 defaults.
    """
    from driving_vla.model.k2_builder import load_k2_config
    from driving_vla.model.neural_policy import NeuralV1Policy
    from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

    # Resolve default ckpt path without loading weights.
    runtime_probe = SimLingoNeuralRuntime(device=device)
    ckpt = Path(runtime_probe.ckpt_path)
    model_checkpoint_hash = _file_sha256(ckpt) if ckpt.is_file() else "missing_ckpt"
    k2_cfg = load_k2_config()
    retimer_hash = content_hash(
        {"retimer_version": k2_cfg.retimer_version, "cfg": k2_cfg.config_hash()},
        nibble=16,
    )
    model_id = NeuralV1Policy.model_id
    model_retimer_hash = build_model_retimer_hash(
        model_id=model_id,
        model_checkpoint_hash=model_checkpoint_hash,
        config_hash=k2_cfg.config_hash(),
        retimer_version=k2_cfg.retimer_version,
        retimer_hash=retimer_hash,
    )
    return model_checkpoint_hash, retimer_hash, model_retimer_hash, k2_cfg.config_hash()


def build_spatial_run_identity(
    spatial_head_checkpoint: Path | str,
) -> dict[str, str]:
    """Build immutable V2 head/config identity without loading CARLA."""
    from driving_vla.model.k2_spatial_types import load_k2_spatial_config

    ckpt = Path(spatial_head_checkpoint)
    if not ckpt.is_file():
        raise RunnerContractError(f"spatial head checkpoint not found: {ckpt}")
    cfg = load_k2_spatial_config()
    return {
        "policy_type": "NeuralV2Policy",
        "policy_model_id": NeuralV2Policy.model_id,
        "spatial_head_checkpoint_hash": _file_sha256(ckpt),
        "spatial_k2_config_hash": cfg.config_hash(),
    }


def bind_spatial_model_retimer_hash(
    base_model_retimer_hash: str,
    spatial_identity: Mapping[str, str],
) -> str:
    """Bind the Spatial head/config to pair IDs and comparability identity."""
    required = (
        "policy_type",
        "policy_model_id",
        "spatial_head_checkpoint_hash",
        "spatial_k2_config_hash",
    )
    missing = [key for key in required if not str(spatial_identity.get(key, ""))]
    if missing:
        raise RunnerContractError(
            f"spatial identity missing required fields: {missing}"
        )
    return content_hash(
        {
            "base_model_retimer_hash": str(base_model_retimer_hash),
            **{key: str(spatial_identity[key]) for key in required},
        },
        nibble=64,
    )


def run_pair(
    *,
    registry_path: Path | str,
    scenario_id: str,
    seed_id: str,
    evidence_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 2000,
    branch_order: tuple[int, int] | None = None,
    device: str = "cuda",
    registry_manifest_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    force_attempt_id: int | None = None,
    retry_policy: str = RETRY_POLICY_NO_AUTO_RETRY,
    shared_policy: Any | None = None,
    shared_identity: tuple[str, str, str] | None = None,
    carla_timeout_s: float = 60.0,
    spatial_k2: bool = False,
    spatial_head_checkpoint: str | Path | None = None,
    spatial_run_identity: Mapping[str, str] | None = None,
    allow_singleton: bool = False,
    collect_observable_history: bool = False,
) -> dict[str, Any]:
    """Full pair: frozen-registry gate → idempotent read or planned attempt run.

    ``shared_policy`` / ``shared_identity`` let run-set load SimLingo once and
    reuse across 12 pairs (avoids 12× GPU reload thrash that can stall CARLA RPC).

    When ``spatial_k2`` is True, uses NeuralV2Policy + run_anchor_v2 (dual residual).
    """
    evidence_root = Path(evidence_root)
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    retry = normalize_retry_policy(retry_policy)

    # 1) Fail-closed frozen registry BEFORE CARLA connect.
    reg, freeze_audit, man_path = require_frozen_registry(
        registry_path,
        manifest_path=registry_manifest_path,
        repo_root=repo,
    )
    fixture = reg.get(scenario_id, seed_id)
    registry_hash = freeze_audit["registry_sha256"]

    # 2) Identity hashes (file/config only; no CARLA). Prefer shared from run-set.
    if shared_identity is not None:
        model_checkpoint_hash, retimer_hash, model_retimer_hash = shared_identity
    else:
        model_checkpoint_hash, retimer_hash, model_retimer_hash, _cfg_hash = (
            _identity_hashes_without_carla(device=device)
        )
    pair_policy_identity: dict[str, str] = {}
    if spatial_k2:
        if not spatial_head_checkpoint:
            raise RunnerContractError("spatial_k2 requires spatial_head_checkpoint")
        local_spatial_identity = build_spatial_run_identity(spatial_head_checkpoint)
        if spatial_run_identity is not None:
            for key, value in local_spatial_identity.items():
                if str(spatial_run_identity.get(key, "")) != value:
                    raise RunnerContractError(
                        f"spatial run identity mismatch on {key}: "
                        f"planned={spatial_run_identity.get(key)!r} live={value!r}"
                    )
            pair_policy_identity = {
                key: str(spatial_run_identity[key])
                for key in local_spatial_identity
            }
        else:
            pair_policy_identity = local_spatial_identity
        if shared_identity is None:
            base_model_retimer_hash = model_retimer_hash
        else:
            (
                base_checkpoint_hash,
                base_retimer_hash,
                base_model_retimer_hash,
                _base_cfg_hash,
            ) = _identity_hashes_without_carla(device=device)
            if base_checkpoint_hash != model_checkpoint_hash:
                raise RunnerContractError(
                    "shared spatial identity base checkpoint hash mismatch"
                )
            if base_retimer_hash != retimer_hash:
                raise RunnerContractError(
                    "shared spatial identity base retimer hash mismatch"
                )
        expected_spatial_model_retimer_hash = bind_spatial_model_retimer_hash(
            base_model_retimer_hash, pair_policy_identity
        )
        if shared_identity is not None and (
            model_retimer_hash != expected_spatial_model_retimer_hash
        ):
            raise RunnerContractError(
                "shared spatial model_retimer_hash does not bind live head/config"
            )
        model_retimer_hash = expected_spatial_model_retimer_hash
    pair_id = compute_pair_id(
        scenario_registry_hash=registry_hash,
        scenario_id=scenario_id,
        seed_id=seed_id,
        model_checkpoint_config_retimer_hash=model_retimer_hash,
        executor_config_hash=EXECUTOR_CONFIG_HASH,
    )
    expected = ExpectedPairHashes(
        pair_id=pair_id,
        scenario_id=scenario_id,
        seed_id=seed_id,
        registry_sha256=registry_hash,
        model_retimer_hash=model_retimer_hash,
        executor_config_hash=EXECUTOR_CONFIG_HASH,
    )

    # 3) Attempt selection under frozen retry policy.
    if force_attempt_id is not None:
        if retry != RETRY_POLICY_NO_AUTO_RETRY:
            raise RunnerContractError("force_attempt_id requires no_auto_retry")
        action = resolve_no_auto_retry_action(
            evidence_root,
            pair_id=pair_id,
            planned_attempt_id=int(force_attempt_id),
            expected=expected,
        )
        if action["action"] == "idempotent_read":
            out = dict(action["existing_manifest"])
            out["idempotent_read"] = True
            out["attempt_id"] = int(action["attempt_id"])
            out["attempt_dir"] = str(Path(action["attempt_dir"]).as_posix())
            out["frozen_registry_audit"] = freeze_audit
            out["retained_failed"] = False
            print(
                f"[pair] IDEMPOTENT_READ id={pair_id} attempt={action['attempt_id']}",
                flush=True,
            )
            return out
        if action["action"] == "retain_failed":
            out = dict(action["existing_manifest"])
            out["idempotent_read"] = False
            out["retained_failed"] = True
            out["attempt_id"] = int(action["attempt_id"])
            out["attempt_dir"] = str(Path(action["attempt_dir"]).as_posix())
            out["frozen_registry_audit"] = freeze_audit
            out["status"] = PAIR_STATUS_FAILED
            out["comparable"] = False
            print(
                f"[pair] RETAIN_FAILED id={pair_id} attempt={action['attempt_id']} "
                f"(no_auto_retry)",
                flush=True,
            )
            return out
        attempt_id = int(action["attempt_id"])
        attempt_dir = Path(action["attempt_dir"])
        attempt_dir.mkdir(parents=True, exist_ok=False)
    else:
        plan = plan_pair_attempt(evidence_root, expected)
        if plan.mode == "idempotent_read" and plan.existing_manifest is not None:
            out = dict(plan.existing_manifest)
            out["idempotent_read"] = True
            out["attempt_id"] = plan.attempt_id
            out["attempt_dir"] = str(plan.attempt_dir.as_posix())
            out["frozen_registry_audit"] = freeze_audit
            print(
                f"[pair] IDEMPOTENT_READ id={pair_id} attempt={plan.attempt_id} "
                f"dir={plan.attempt_dir}",
                flush=True,
            )
            return out
        # Standalone (non-run-set) path may still allocate next attempt_id.
        attempt_dir = plan.attempt_dir
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_id = plan.attempt_id

    if branch_order is None:
        branch_order = (0, 1) if seed_id == "seed_a" else (1, 0)

    print(
        f"[pair] NEW_RUN id={pair_id} attempt={attempt_id} {scenario_id}/{seed_id} "
        f"order={branch_order} registry_manifest={man_path}",
        flush=True,
    )

    client = None
    world = None
    forward_counter: dict[str, int] = {"n": 0}
    anchor_failure_identity: dict[str, Any] = {}

    def _write_failed(exc: BaseException, *, phase: str) -> dict[str, Any]:
        fail_man = build_failed_manifest(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            registry_sha256=registry_hash,
            model_retimer_hash=model_retimer_hash,
            executor_config_hash=EXECUTOR_CONFIG_HASH,
            attempt_id=attempt_id,
            error=str(exc),
            extra={
                "family": fixture.family,
                "phase": phase,
                "traceback": traceback.format_exc(),
                "attempt_dir": str(attempt_dir.as_posix()),
                "frozen_registry_audit": freeze_audit,
                "idempotent_read": False,
                "comparable": False,
                **pair_policy_identity,
                **anchor_failure_identity,
            },
        )
        try:
            _write_json(attempt_dir / "pair_manifest.json", fail_man)
            _write_json(attempt_dir / "failure.json", fail_man)
        except Exception:
            pass
        return fail_man

    try:
        try:
            client, world = connect_world(
                host=host,
                port=port,
                map_name=fixture.map_name,
                sim_dt_s=0.05,
                sync=True,
                timeout_s=float(carla_timeout_s),
                retries=3,
            )
        except Exception as exc:
            _write_failed(exc, phase="connect_world")
            raise

        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for R2 pair run")

        if shared_policy is not None:
            policy = shared_policy
            policy.ensure_loaded()
        else:
            runtime = SimLingoNeuralRuntime(device=device)
            load = runtime.load()
            if not load.ok:
                raise RuntimeError(f"SimLingo load failed: {load.error}")
            # Reconfirm ckpt hash matches preflight identity (fail-closed).
            ckpt = Path(runtime.ckpt_path)
            live_ckpt_hash = _file_sha256(ckpt) if ckpt.is_file() else "missing_ckpt"
            if live_ckpt_hash != model_checkpoint_hash:
                raise RunnerContractError(
                    f"checkpoint hash changed between preflight and load: "
                    f"{model_checkpoint_hash[:16]} vs {live_ckpt_hash[:16]}"
                )
            if spatial_k2:
                if not spatial_head_checkpoint:
                    raise RunnerContractError(
                        "spatial_k2 requires spatial_head_checkpoint"
                    )
                policy = NeuralV2Policy(
                    runtime=runtime,
                    keep_on_gpu=True,
                    spatial_head_checkpoint=str(spatial_head_checkpoint),
                    device=device,
                    require_driving_feature=True,
                )
            else:
                policy = NeuralV1Policy(runtime=runtime, keep_on_gpu=True)
            policy.ensure_loaded()

        use_v2 = bool(spatial_k2) or isinstance(policy, NeuralV2Policy)
        if use_v2:
            if not isinstance(policy, NeuralV2Policy):
                raise RunnerContractError(
                    "spatial_k2 requires NeuralV2Policy shared_policy"
                )
            art, anchor_meta = run_anchor_v2(
                client=client,
                world=world,
                fixture=fixture,
                policy=policy,
                pair_id=pair_id,
                model_checkpoint_hash=model_checkpoint_hash,
                registry_hash=registry_hash,
                evidence_dir=attempt_dir,
                forward_counter=forward_counter,
                collect_observable_history=collect_observable_history,
            )
            art_path = attempt_dir / "anchor" / "anchor_bundle_v2.json"
        else:
            art, anchor_meta = run_anchor(
                client=client,
                world=world,
                fixture=fixture,
                policy=policy,
                pair_id=pair_id,
                model_checkpoint_hash=model_checkpoint_hash,
                retimer_hash=retimer_hash,
                registry_hash=registry_hash,
                evidence_dir=attempt_dir,
                forward_counter=forward_counter,
            )
            art_path = attempt_dir / "anchor" / "anchor_bundle.json"
        if forward_counter["n"] != 1:
            raise FixtureError(
                f"expected exactly 1 forward after anchor, got {forward_counter['n']}"
            )

        art_disk = load_anchor_artifact_any(art_path.read_bytes())
        if art_disk.artifact_content_hash() != art.artifact_content_hash():
            raise FixtureError("disk artifact hash mismatch")
        anchor_failure_identity.update(
            {
                "artifact_content_hash": art_disk.artifact_content_hash(),
                "candidate_ids": [
                    str(candidate.candidate_id) for candidate in art_disk.candidates
                ],
                "top1_candidate_index": int(art_disk.top1_index),
                "top1_candidate_id": str(
                    art_disk.candidates[int(art_disk.top1_index)].candidate_id
                ),
            }
        )

        results: dict[int, BranchLiveResult] = {}
        available_indices = set(range(len(art_disk.candidates)))
        if isinstance(art_disk, K2AnchorArtifactV2):
            available_indices = {
                int(candidate.candidate_index)
                for candidate in art_disk.candidates
                if bool(candidate.available)
            }
            if set(art_disk.guard_reasons) == {"SPATIAL_COLLAPSE_ELIGIBLE"}:
                available_indices = {0}
        if not allow_singleton and available_indices != {0, 1}:
            # Preserve historical R2 fail-closed behavior.
            available_indices = {0, 1}
        if allow_singleton and 0 not in available_indices:
            raise FixtureError("R3 singleton collection requires available nominal candidate 0")
        for idx in branch_order:
            if idx not in available_indices:
                _write_json(
                    attempt_dir / f"branch-{idx}" / "unavailable.json",
                    {
                        "candidate_index": idx,
                        "candidate_id": str(art_disk.candidates[idx].candidate_id),
                        "available": False,
                        "unavailable_reason": str(
                            "SPATIAL_COLLAPSE_ELIGIBLE"
                            if set(getattr(art_disk, "guard_reasons", ()))
                            == {"SPATIAL_COLLAPSE_ELIGIBLE"}
                            else getattr(
                                art_disk.candidates[idx], "availability_reason", None
                            )
                            or "NO_ALTERNATIVE"
                        ),
                        "executed": False,
                        "ranking_mask": False,
                    },
                )
                continue
            results[idx] = run_branch(
                client=client,
                world=world,
                fixture=fixture,
                artifact=art_disk,
                force_index=idx,
                pair_id=pair_id,
                registry_hash=registry_hash,
                model_retimer_hash=model_retimer_hash,
                evidence_dir=attempt_dir,
                forward_counter=forward_counter,
                policy=policy,
                attempt_id=attempt_id,
                allow_nominal_only_fallback=allow_singleton,
                collect_observable_history=collect_observable_history,
            )
            if forward_counter["n"] != 1:
                raise FixtureError(
                    f"forward count changed during branch {idx}: {forward_counter['n']}"
                )

        b0 = results[0]
        if len(results) == 1 and allow_singleton:
            missing_index = 1
            unavailable = _read_json_or_none(
                attempt_dir / "branch-1" / "unavailable.json"
            ) or {
                "candidate_index": 1,
                "available": False,
                "unavailable_reason": "NO_ALTERNATIVE",
                "executed": False,
                "ranking_mask": False,
            }
            comp_dict = {
                "status": "SINGLETON_NO_RANKING",
                "comparable": False,
                "reasons": ["NO_ALTERNATIVE"],
                "failure_codes": [],
                "details": {"executed_branches": [0], "candidate_mask": [1, 0]},
            }
            oracle_dict = {
                "pair_id": pair_id,
                "scenario_id": scenario_id,
                "seed_id": seed_id,
                "family": fixture.family,
                "comparable": False,
                "top1_candidate_id": str(art_disk.candidates[0].candidate_id),
                "top1_candidate_index": 0,
                "oracle_candidate_id": None,
                "oracle_candidate_index": None,
                "oracle_decision_level": None,
                "decision_reason": "K_eff=1_NO_RANKING_NEEDED",
                "pair_label": "NO_ALTERNATIVE",
                "both_bad": False,
                "outcome_delta": {},
                "failure_reasons": ["NO_ALTERNATIVE"],
                "relative_winner_if_both_bad": None,
            }
            _write_json(attempt_dir / "pair_comparability.json", comp_dict)
            _write_json(attempt_dir / "pair_oracle.json", oracle_dict)
            manifest = build_completed_manifest(
                pair_id=pair_id,
                scenario_id=scenario_id,
                seed_id=seed_id,
                family=fixture.family,
                registry_sha256=registry_hash,
                model_retimer_hash=model_retimer_hash,
                executor_config_hash=EXECUTOR_CONFIG_HASH,
                artifact_content_hash=art_disk.artifact_content_hash(),
                attempt_id=attempt_id,
                branch_order=branch_order,
                forward_count_total=int(forward_counter["n"]),
                comparable=False,
                comparability=comp_dict,
                oracle=oracle_dict,
                anchor=anchor_meta,
                branch_0=b0.summary,
                branch_1=unavailable,
                extra={
                    "idempotent_read": False,
                    "attempt_dir": str(attempt_dir.as_posix()),
                    "frozen_registry_audit": freeze_audit,
                    "branch_cleanup_ok": {"0": bool(b0.report.cleanup_ok), "1": True},
                    "candidate_mask": [1, 0],
                    "ranking_mask": False,
                    "r3_singleton_audit": True,
                    **pair_policy_identity,
                    "candidate_ids": [
                        str(candidate.candidate_id) for candidate in art_disk.candidates
                    ],
                    "top1_candidate_index": int(art_disk.top1_index),
                    "top1_candidate_id": str(
                        art_disk.candidates[int(art_disk.top1_index)].candidate_id
                    ),
                },
            )
            ledger = ledger_path_for_evidence_root(evidence_root)
            appended = append_ledger_if_new(
                ledger,
                {
                    **oracle_dict,
                    "attempt_id": attempt_id,
                    "artifact_content_hash": art_disk.artifact_content_hash(),
                    "registry_sha256": registry_hash,
                    "status": PAIR_STATUS_COMPLETED,
                },
            )
            manifest["ledger_appended"] = appended
            _write_json(attempt_dir / "pair_manifest.json", manifest)
            print(
                f"[pair] singleton candidate=0 reason={unavailable['unavailable_reason']} "
                f"forwards={forward_counter['n']} attempt={attempt_id} "
                f"ledger_appended={appended}",
                flush=True,
            )
            return manifest
        b1 = results[1]
        comp = evaluate_pair_comparability(
            anchor=art_disk,
            branch0=b0.report,
            branch1=b1.report,
            expected_registry_hash=registry_hash,
            expected_scenario_id=scenario_id,
            expected_seed_id=seed_id,
            expected_model_retimer_hash=model_retimer_hash,
            expected_executor_config_hash=EXECUTOR_CONFIG_HASH,
        )
        _write_json(attempt_dir / "pair_comparability.json", comp.to_dict())

        oracle = evaluate_pair_oracle(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            family=fixture.family,
            comparable=comp.comparable,
            top1_index=art_disk.top1_index,
            metrics0=b0.metrics if comp.comparable else None,
            metrics1=b1.metrics if comp.comparable else None,
            incomparable_reasons=comp.reasons,
        )
        _write_json(attempt_dir / "pair_oracle.json", oracle.to_dict())

        manifest = build_completed_manifest(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            family=fixture.family,
            registry_sha256=registry_hash,
            model_retimer_hash=model_retimer_hash,
            executor_config_hash=EXECUTOR_CONFIG_HASH,
            artifact_content_hash=art_disk.artifact_content_hash(),
            attempt_id=attempt_id,
            branch_order=branch_order,
            forward_count_total=int(forward_counter["n"]),
            comparable=comp.comparable,
            comparability=comp.to_dict(),
            oracle=oracle.to_dict(),
            anchor=anchor_meta,
            branch_0=b0.summary,
            branch_1=b1.summary,
            extra={
                "idempotent_read": False,
                "attempt_dir": str(attempt_dir.as_posix()),
                "frozen_registry_audit": freeze_audit,
                "branch_cleanup_ok": {
                    "0": bool(b0.report.cleanup_ok),
                    "1": bool(b1.report.cleanup_ok),
                },
                **pair_policy_identity,
                "candidate_ids": [
                    str(candidate.candidate_id) for candidate in art_disk.candidates
                ],
                "top1_candidate_index": int(art_disk.top1_index),
                "top1_candidate_id": str(
                    art_disk.candidates[int(art_disk.top1_index)].candidate_id
                ),
            },
        )
        _write_json(attempt_dir / "pair_manifest.json", manifest)

        # Ledger: one row per attempt; never duplicate on re-read (idempotent).
        ledger = ledger_path_for_evidence_root(evidence_root)
        ledger_row = {
            **oracle.to_dict(),
            "attempt_id": attempt_id,
            "artifact_content_hash": art_disk.artifact_content_hash(),
            "registry_sha256": registry_hash,
            "status": PAIR_STATUS_COMPLETED,
        }
        appended = append_ledger_if_new(ledger, ledger_row)
        manifest["ledger_appended"] = appended
        _write_json(attempt_dir / "pair_manifest.json", manifest)

        print(
            f"[pair] comparable={comp.comparable} label={oracle.pair_label} "
            f"oracle={oracle.oracle_candidate_id} forwards={forward_counter['n']} "
            f"attempt={attempt_id} ledger_appended={appended}",
            flush=True,
        )
        return manifest
    except Exception as exc:
        # connect_world failures already wrote FAILED manifest
        if not (attempt_dir / "pair_manifest.json").is_file():
            _write_failed(exc, phase="run_pair")
        raise
    finally:
        if world is not None:
            restore_async(world)
