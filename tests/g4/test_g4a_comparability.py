"""R2 offline: initial-state tolerances and failure classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.comparability import (  # noqa: E402
    STATUS_COMPARABLE,
    STATUS_INCOMPARABLE,
    BranchExecutionReport,
    classify_external_failure,
    evaluate_pair_comparability,
)
from driving_vla.evaluation.paired_contract import (  # noqa: E402
    K2_ANCHOR_SCHEMA,
    MEASURED_STATE_SCHEMA,
    ActorSnapshot,
    K2AnchorArtifactV1,
    MeasuredInitialState,
    ObservationFingerprint,
    SerializedCandidate,
    TransformPose,
    VelocityState,
)


def _pose(x=0.0, y=0.0, yaw=0.0) -> TransformPose:
    return TransformPose(x, y, 0.5, 0.0, 0.0, yaw)


def _vel(vx=5.0, vy=0.0) -> VelocityState:
    return VelocityState(vx, vy, 0.0)


def _actor(name: str, role: str, x: float = 0.0, yaw: float = 0.0, vx: float = 5.0) -> ActorSnapshot:
    return ActorSnapshot(
        name=name,
        role=role,
        blueprint="vehicle.tesla.model3",
        transform=_pose(x=x, yaw=yaw),
        velocity=_vel(vx=vx),
        control={"throttle": 0.0, "brake": 0.0, "steer": 0.0},
        script_phase="init",
        bounding_box_extent_m=(2.3, 1.0, 0.75),
    )


def _measured(*, ego_x: float = 0.0, frame: int = 10, lead_x: float = 20.0) -> MeasuredInitialState:
    return MeasuredInitialState(
        schema_version=MEASURED_STATE_SCHEMA,
        map_name="Town03",
        open_drive_identity="Town03",
        world_settings={"fixed_delta_seconds": 0.05, "synchronous_mode": True},
        weather={"cloudiness": 10.0, "wetness": 0.0},
        actors=(
            _actor("ego", "ego", x=ego_x, vx=8.0),
            _actor("lead", "npc", x=lead_x, vx=7.0),
        ),
        traffic_light_state={"policy": "freeze_green", "state": "Green"},
        route_anchor={"identity": "r"},
        sensor_calibration={"front_rgb": {"fov": 110.0}},
        carla_server_epoch="epoch1",
        carla_version="0.9.15",
        simulation_frame=frame,
        simulation_time_s=0.5,
        actor_script_phase={"ego": "init", "lead": "init"},
    )


def _candidate(idx: int, cid: str) -> SerializedCandidate:
    pts = tuple((float(i + 1), 0.0, 0.0, 5.0, 0.0, 0.0) for i in range(10))
    path = tuple((float(i), 0.0) for i in range(20))
    return SerializedCandidate(
        candidate_id=cid,
        candidate_index=idx,
        probability=0.5,
        points_xy_yaw_v_a_kappa=pts,
        spatial_path_xy=path,
        speed_samples_mps=tuple(5.0 for _ in range(10)),
        timed_trajectory_hash=f"th{idx}",
        native_path_hash="npath",
        branch_type="longitudinal_temporal",
    )


def _anchor(*, guard: str = "OK") -> K2AnchorArtifactV1:
    obs = ObservationFingerprint(
        front_rgb_sha256="abc",
        image_height=512,
        image_width=1024,
        image_channels=3,
        image_layout="HWC",
        ego_observable={"x": 0.0, "y": 0.0, "v": 8.0},
        route_targets=[(50.0, 0.0)],
        camera_frame={"parent": "ego"},
        k2_bundle_hash="kb1",
    )
    return K2AnchorArtifactV1(
        schema_version=K2_ANCHOR_SCHEMA,
        pair_id="pair_test",
        scenario_id="lead_brake_moderate",
        seed_id="seed_a",
        anchor_run_id="anchor-1",
        anchor_carla_frame=10,
        anchor_simulation_time_s=0.5,
        requested_initial_state_hash="req",
        measured_initial_state_hash="meas",
        observation_fingerprint=obs,
        model_id="simlingo",
        model_checkpoint_hash="ckpt",
        config_hash="cfg",
        retimer_version="safedrive.k2_retimer.v1",
        retimer_hash="rh",
        executor_config_hash="ex",
        native_path_xy=tuple((float(i), 0.0) for i in range(20)),
        native_path_hash="npath",
        candidates=(_candidate(0, "v1_nominal"), _candidate(1, "v1_conservative")),
        top1_index=0,
        guard_status=guard,
        guard_reasons=() if guard == "OK" else ("test_reject",),
    )


def _branch(
    idx: int,
    *,
    measured: MeasuredInitialState | None = None,
    anchor_hash: str | None = None,
    ticks: int = 50,
    mpc_solved: int = 50,
    mpc_timeout: int = 0,
    mpc_fallback: int = 0,
    ids_ok: bool = True,
    registry_hash: str = "reg",
    model_hash: str = "model",
    exec_hash: str = "ex",
    weather: dict | None = None,
    lights: dict | None = None,
    **kwargs,
) -> BranchExecutionReport:
    art = _anchor()
    ah = anchor_hash if anchor_hash is not None else art.artifact_content_hash()
    return BranchExecutionReport(
        candidate_index=idx,
        candidate_id="v1_nominal" if idx == 0 else "v1_conservative",
        anchor_artifact_hash=ah,
        measured_initial_state=measured or _measured(),
        registry_hash=registry_hash,
        scenario_id="lead_brake_moderate",
        seed_id="seed_a",
        model_retimer_hash=model_hash,
        executor_config_hash=exec_hash,
        completed_primary_ticks=ticks,
        mpc_solved_ticks=mpc_solved,
        mpc_timeout_ticks=mpc_timeout,
        mpc_fallback_ticks=mpc_fallback,
        selected_ids_consistent=ids_ok,
        executed_ids_consistent=ids_ok,
        source_ids_consistent=ids_ok,
        weather=weather if weather is not None else {"cloudiness": 10.0},
        traffic_light_state=lights if lights is not None else {"policy": "freeze_green", "state": "Green"},
        **kwargs,
    )


class G4AComparabilityTest(unittest.TestCase):
    def test_comparable_happy_path(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_COMPARABLE)
        self.assertTrue(r.comparable)

    def test_absolute_frame_mismatch_not_incomparable(self) -> None:
        """Cold rebuilds never share absolute CARLA frame IDs."""
        art = _anchor()
        m0 = _measured(frame=10)
        m1 = _measured(frame=999)
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, measured=m0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, measured=m1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_COMPARABLE)
        self.assertEqual(r.details["simulation_frames"]["branch1"], 999)

    def test_position_tolerance(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, measured=_measured(ego_x=0.0), anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, measured=_measured(ego_x=0.05), anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_INCOMPARABLE)
        self.assertIn("INITIAL_STATE_MISMATCH", r.failure_codes)

    def test_yaw_tolerance(self) -> None:
        art = _anchor()
        m0 = _measured()
        actors = list(m0.actors)
        bad_ego = ActorSnapshot(
            name="ego",
            role="ego",
            blueprint=actors[0].blueprint,
            transform=_pose(yaw=1.0),
            velocity=actors[0].velocity,
            control=actors[0].control,
            script_phase="init",
            bounding_box_extent_m=actors[0].bounding_box_extent_m,
        )
        m1 = MeasuredInitialState(
            schema_version=m0.schema_version,
            map_name=m0.map_name,
            open_drive_identity=m0.open_drive_identity,
            world_settings=m0.world_settings,
            weather=m0.weather,
            actors=(bad_ego, actors[1]),
            traffic_light_state=m0.traffic_light_state,
            route_anchor=m0.route_anchor,
            sensor_calibration=m0.sensor_calibration,
            carla_server_epoch=m0.carla_server_epoch,
            carla_version=m0.carla_version,
            simulation_frame=m0.simulation_frame,
            simulation_time_s=m0.simulation_time_s,
            actor_script_phase=m0.actor_script_phase,
        )
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, measured=m0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, measured=m1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_INCOMPARABLE)
        self.assertTrue(any("yaw_tol" in x for x in r.reasons))

    def test_velocity_tolerance(self) -> None:
        art = _anchor()
        m0 = _measured()
        actors = list(m0.actors)
        bad_ego = ActorSnapshot(
            name="ego",
            role="ego",
            blueprint=actors[0].blueprint,
            transform=actors[0].transform,
            velocity=_vel(vx=8.2),
            control=actors[0].control,
            script_phase="init",
            bounding_box_extent_m=actors[0].bounding_box_extent_m,
        )
        m1 = MeasuredInitialState(
            schema_version=m0.schema_version,
            map_name=m0.map_name,
            open_drive_identity=m0.open_drive_identity,
            world_settings=m0.world_settings,
            weather=m0.weather,
            actors=(bad_ego, actors[1]),
            traffic_light_state=m0.traffic_light_state,
            route_anchor=m0.route_anchor,
            sensor_calibration=m0.sensor_calibration,
            carla_server_epoch=m0.carla_server_epoch,
            carla_version=m0.carla_version,
            simulation_frame=m0.simulation_frame,
            simulation_time_s=m0.simulation_time_s,
            actor_script_phase=m0.actor_script_phase,
        )
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, measured=m0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, measured=m1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_INCOMPARABLE)
        self.assertIn("INITIAL_STATE_MISMATCH", r.failure_codes)

    def test_script_phase_mismatch(self) -> None:
        art = _anchor()
        m0 = _measured()
        actors = list(m0.actors)
        bad_lead = ActorSnapshot(
            name="lead",
            role="npc",
            blueprint=actors[1].blueprint,
            transform=actors[1].transform,
            velocity=actors[1].velocity,
            control=actors[1].control,
            script_phase="brake_phase",
            bounding_box_extent_m=actors[1].bounding_box_extent_m,
        )
        m1 = MeasuredInitialState(
            schema_version=m0.schema_version,
            map_name=m0.map_name,
            open_drive_identity=m0.open_drive_identity,
            world_settings=m0.world_settings,
            weather=m0.weather,
            actors=(actors[0], bad_lead),
            traffic_light_state=m0.traffic_light_state,
            route_anchor=m0.route_anchor,
            sensor_calibration=m0.sensor_calibration,
            carla_server_epoch=m0.carla_server_epoch,
            carla_version=m0.carla_version,
            simulation_frame=m0.simulation_frame,
            simulation_time_s=m0.simulation_time_s,
            actor_script_phase=m0.actor_script_phase,
        )
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, measured=m0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, measured=m1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertIn("SCRIPT_PHASE_MISMATCH", r.failure_codes)

    def test_anchor_bundle_mismatch(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, anchor_hash="wrong_hash"),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertIn("ANCHOR_BUNDLE_MISMATCH", r.failure_codes)

    def test_guard_reject(self) -> None:
        art = _anchor(guard="REJECT")
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertIn("GUARD_REJECT", r.failure_codes)

    def test_incomplete_horizon(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, ticks=40, mpc_solved=40, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertIn("INCOMPLETE_PRIMARY_HORIZON", r.failure_codes)

    def test_mpc_deadline(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(
                0,
                mpc_solved=45,
                mpc_timeout=5,
                anchor_hash=art.artifact_content_hash(),
            ),
            branch1=_branch(1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertIn("MPC_DEADLINE_UNRELIABLE", r.failure_codes)

    def test_mpc_48_solved_ok(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(
                0,
                mpc_solved=48,
                mpc_timeout=2,
                anchor_hash=art.artifact_content_hash(),
            ),
            branch1=_branch(
                1,
                mpc_solved=48,
                mpc_fallback=2,
                anchor_hash=art.artifact_content_hash(),
            ),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertEqual(r.status, STATUS_COMPARABLE)

    def test_execution_binding_failure(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(0, ids_ok=False, anchor_hash=art.artifact_content_hash()),
            branch1=_branch(1, anchor_hash=art.artifact_content_hash()),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertIn("EXECUTION_BINDING_FAILURE", r.failure_codes)

    def test_sensor_tick_cleanup_server(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(
                0,
                sensor_sync_ok=False,
                anchor_hash=art.artifact_content_hash(),
            ),
            branch1=_branch(
                1,
                cleanup_ok=False,
                tick_owner_ok=False,
                server_ok=False,
                anchor_hash=art.artifact_content_hash(),
            ),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        for code in (
            "SENSOR_SYNC_FAILURE",
            "CLEANUP_FAILURE",
            "TICK_OWNER_CONFLICT",
            "SERVER_CRASH_OR_HANG",
        ):
            self.assertIn(code, r.failure_codes)

    def test_weather_and_light_mismatch(self) -> None:
        art = _anchor()
        r = evaluate_pair_comparability(
            anchor=art,
            branch0=_branch(
                0,
                weather={"cloudiness": 10.0},
                lights={"state": "Green"},
                anchor_hash=art.artifact_content_hash(),
            ),
            branch1=_branch(
                1,
                weather={"cloudiness": 50.0},
                lights={"state": "Red"},
                anchor_hash=art.artifact_content_hash(),
            ),
            expected_registry_hash="reg",
            expected_scenario_id="lead_brake_moderate",
            expected_seed_id="seed_a",
            expected_model_retimer_hash="model",
            expected_executor_config_hash="ex",
        )
        self.assertTrue(
            "INITIAL_STATE_MISMATCH" in r.failure_codes or "SCRIPT_PHASE_MISMATCH" in r.failure_codes
        )

    def test_classify_buckets(self) -> None:
        self.assertEqual(classify_external_failure("SERVER_CRASH_OR_HANG"), "EXTERNAL_SERVER")
        self.assertEqual(classify_external_failure("SPAWN_FAILED"), "FIXTURE_OR_RUNNER")
        self.assertEqual(classify_external_failure("GUARD_REJECT"), "ANCHOR_OR_BINDING")
        self.assertEqual(classify_external_failure("MPC_DEADLINE_UNRELIABLE"), "EXECUTOR_DEADLINE")


if __name__ == "__main__":
    unittest.main()
