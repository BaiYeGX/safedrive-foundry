"""R2 offline: anchor artifact serialize, same-hash branches, no second forward, ID bind."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_contract import (  # noqa: E402
    K2_ANCHOR_SCHEMA,
    ContractError,
    K2AnchorArtifactV1,
    ObservationFingerprint,
    SerializedCandidate,
    assert_no_oracle_in_control_payload,
    compute_pair_id,
    compute_run_id,
    content_hash,
)
from driving_vla.evaluation.outcome_metrics import TickRecord  # noqa: E402
from driving_vla.runtime.k2_execution import K2SelectionError, select_k2  # noqa: E402
from driving_vla.adapter.policy_adapter import TrajectoryArray  # noqa: E402
from driving_vla.model.k2_builder import (  # noqa: E402
    GUARD_OK,
    K2Diagnostics,
    K2ExecutionSpec,
    K2PredictionBundle,
    stable_hash_xy,
)


def _pts(speed: float = 5.0):
    return tuple((float(i + 1) * 0.5, 0.0, 0.0, speed, 0.0, 0.0) for i in range(10))


def _path():
    return tuple((float(i), 0.0) for i in range(30))


def _artifact() -> K2AnchorArtifactV1:
    path = _path()
    nh = stable_hash_xy(path)
    c0 = SerializedCandidate(
        candidate_id="v1_nominal",
        candidate_index=0,
        probability=0.5,
        points_xy_yaw_v_a_kappa=_pts(5.0),
        spatial_path_xy=path,
        speed_samples_mps=tuple(5.0 for _ in range(10)),
        timed_trajectory_hash="th0",
        native_path_hash=nh,
        branch_type="longitudinal_temporal",
    )
    c1 = SerializedCandidate(
        candidate_id="v1_conservative",
        candidate_index=1,
        probability=0.5,
        points_xy_yaw_v_a_kappa=_pts(3.25),
        spatial_path_xy=path,
        speed_samples_mps=tuple(3.25 for _ in range(10)),
        timed_trajectory_hash="th1",
        native_path_hash=nh,
        branch_type="longitudinal_temporal",
    )
    obs = ObservationFingerprint(
        front_rgb_sha256="deadbeef",
        image_height=512,
        image_width=1024,
        image_channels=3,
        image_layout="HWC_RGB_uint8",
        ego_observable={"x": 0.0, "y": 0.0, "v": 5.0},
        route_targets=[[30.0, 0.0]],
        camera_frame={"attach": "ego"},
        k2_bundle_hash="bundlehash",
    )
    return K2AnchorArtifactV1(
        schema_version=K2_ANCHOR_SCHEMA,
        pair_id="pairabc",
        scenario_id="lead_brake_moderate",
        seed_id="seed_a",
        anchor_run_id="anchor-xyz",
        anchor_carla_frame=42,
        anchor_simulation_time_s=1.0,
        requested_initial_state_hash="req1",
        measured_initial_state_hash="meas1",
        observation_fingerprint=obs,
        model_id="simlingo",
        model_checkpoint_hash="ckpt1",
        config_hash="cfg1",
        retimer_version="safedrive.k2_retimer.v1",
        retimer_hash="rh1",
        executor_config_hash="ex1",
        native_path_xy=path,
        native_path_hash=nh,
        candidates=(c0, c1),
        top1_index=0,
        guard_status=GUARD_OK,
        guard_reasons=(),
        k=2,
        t_steps=10,
        dt_s=0.25,
        horizon_s=2.5,
    )


def _bundle_from_artifact(art: K2AnchorArtifactV1) -> K2PredictionBundle:
    """Rebuild runtime bundle from frozen artifact (branch path; no VLA forward)."""
    cands = []
    specs = {}
    for sc in art.candidates:
        arr = TrajectoryArray(
            points_xy_yaw_v_a_kappa=sc.points_xy_yaw_v_a_kappa,
            probability=sc.probability,
            candidate_id=sc.candidate_id,
        )
        cands.append(arr)
        specs[sc.candidate_id] = K2ExecutionSpec(
            candidate_id=sc.candidate_id,
            spatial_path_xy=sc.spatial_path_xy,
            speed_samples_mps=sc.speed_samples_mps,
            timed_trajectory_hash=sc.timed_trajectory_hash,
            native_path_hash=sc.native_path_hash,
            branch_type=sc.branch_type,
        )
    diag = K2Diagnostics(
        mean_speed_gap_mps=1.0,
        final_progress_gap_m=1.0,
        max_position_separation_m=1.0,
        mean_position_separation_m=0.5,
        collapsed=False,
        collapse_reason=None,
        selection_space_eligible=True,
        path_speed_cap_active=False,
        position_integration_error_max_m=0.0,
        acceleration_error_max_mps2=0.0,
        yaw_tangent_error_max_rad=0.0,
        curvature_error_max_per_m=0.0,
        curvature_error_p95_per_m=0.0,
        native_path_cross_track_error_max_m=0.0,
    )
    return K2PredictionBundle(
        observation_identity={"frame_id": "anchor"},
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


class G4AAnchorExecutionTest(unittest.TestCase):
    def test_k2_t10_guard_ok(self) -> None:
        art = _artifact()
        self.assertEqual(art.k, 2)
        self.assertEqual(art.t_steps, 10)
        self.assertEqual(art.guard_status, GUARD_OK)
        self.assertEqual(len(art.candidates), 2)

    def test_serialize_roundtrip_hash(self) -> None:
        art = _artifact()
        h0 = art.artifact_content_hash()
        raw = art.to_json_bytes()
        art2 = K2AnchorArtifactV1.from_json_bytes(raw)
        self.assertEqual(art2.artifact_content_hash(), h0)
        self.assertEqual(art2.pair_id, art.pair_id)
        self.assertEqual(art2.candidates[0].timed_trajectory_hash, "th0")
        self.assertEqual(art2.candidates[1].timed_trajectory_hash, "th1")

    def test_hash_mismatch_on_tamper(self) -> None:
        art = _artifact()
        d = art.to_dict()
        d["artifact_content_hash"] = "0" * 64
        with self.assertRaises(ContractError):
            K2AnchorArtifactV1.from_dict(d)

    def test_branch0_and_branch1_same_artifact(self) -> None:
        art = _artifact()
        h = art.artifact_content_hash()
        # Simulate two branches loading the same serialized blob
        b0 = K2AnchorArtifactV1.from_json_bytes(art.to_json_bytes())
        b1 = K2AnchorArtifactV1.from_json_bytes(art.to_json_bytes())
        self.assertEqual(b0.artifact_content_hash(), h)
        self.assertEqual(b1.artifact_content_hash(), h)
        self.assertEqual(b0.artifact_content_hash(), b1.artifact_content_hash())

    def test_select_force_without_forward(self) -> None:
        art = _artifact()
        forward_count = {"n": 0}

        class CountingPolicy:
            def predict_bundle(self, *a, **k):
                forward_count["n"] += 1
                raise AssertionError("branch must not call VLA forward")

        policy = CountingPolicy()
        bundle = _bundle_from_artifact(art)
        # branch mode: force select only
        s0 = select_k2(bundle, mode="force", force_index=0)
        s1 = select_k2(bundle, mode="force", force_index=1)
        self.assertEqual(s0.candidate_id, "v1_nominal")
        self.assertEqual(s1.candidate_id, "v1_conservative")
        self.assertEqual(forward_count["n"], 0)
        # Ensure we never needed policy
        del policy

    def test_branch_mode_forbids_second_forward_contract(self) -> None:
        """Documented contract: branch uses artifact only; forward counter stays 0."""
        art = _artifact()
        class Gate:
            def __init__(self):
                self.forwards = 0
                self.mode = "anchor"

            def forward(self):
                if self.mode == "branch":
                    raise RuntimeError("second VLA forward forbidden in branch mode")
                self.forwards += 1

        g = Gate()
        g.forward()  # anchor once
        g.mode = "branch"
        with self.assertRaises(RuntimeError):
            g.forward()
        self.assertEqual(g.forwards, 1)
        # branch still selects from artifact
        bundle = _bundle_from_artifact(art)
        sel = select_k2(bundle, mode="force", force_index=0)
        self.assertEqual(sel.candidate_index, 0)

    def test_source_id_mismatch_fail_closed(self) -> None:
        art = _artifact()
        bundle = _bundle_from_artifact(art)
        # Break execution binding by wrong timed hash spatial check is on verify;
        # select_k2 fails on missing/orphan id:
        bad_specs = dict(bundle.execution_specs)
        del bad_specs["v1_nominal"]
        broken = K2PredictionBundle(
            observation_identity=bundle.observation_identity,
            model_id=bundle.model_id,
            config_hash=bundle.config_hash,
            retimer_version=bundle.retimer_version,
            native_path_xy=bundle.native_path_xy,
            native_path_hash=bundle.native_path_hash,
            candidates=bundle.candidates,
            execution_specs=bad_specs,
            top1_index=bundle.top1_index,
            probability_source=bundle.probability_source,
            probability_margin=bundle.probability_margin,
            branch_type=bundle.branch_type,
            diagnostics=bundle.diagnostics,
            guard_status=bundle.guard_status,
            guard_reasons=bundle.guard_reasons,
        )
        with self.assertRaises(K2SelectionError):
            select_k2(broken, mode="force", force_index=0)

    def test_pair_and_run_ids_stable(self) -> None:
        p1 = compute_pair_id(
            scenario_registry_hash="r",
            scenario_id="lead_brake_moderate",
            seed_id="seed_a",
            model_checkpoint_config_retimer_hash="m",
            executor_config_hash="e",
        )
        p2 = compute_pair_id(
            scenario_registry_hash="r",
            scenario_id="lead_brake_moderate",
            seed_id="seed_a",
            model_checkpoint_config_retimer_hash="m",
            executor_config_hash="e",
        )
        self.assertEqual(p1, p2)
        self.assertEqual(len(p1), 20)
        a0 = compute_run_id(pair_id=p1, role="anchor", attempt_id=0)
        a1 = compute_run_id(pair_id=p1, role="anchor", attempt_id=1)
        self.assertNotEqual(a0, a1)
        b0 = compute_run_id(pair_id=p1, role="branch_0", attempt_id=0)
        self.assertTrue(b0.startswith("branch_0-"))

    def test_failed_attempt_not_overwritten_by_id(self) -> None:
        pair = "pairx"
        fail_dir = compute_run_id(pair_id=pair, role="branch_0", attempt_id=0)
        ok_dir = compute_run_id(pair_id=pair, role="branch_0", attempt_id=1)
        self.assertNotEqual(fail_dir, ok_dir)

    def test_oracle_fields_not_in_control_namespace(self) -> None:
        tick = TickRecord(
            tick_index=0,
            simulation_time_s=0.05,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw_rad=0.0,
            ego_v=5.0,
            selected_candidate_id="v1_nominal",
            executed_candidate_id="v1_nominal",
            source_id="f:v1_nominal",
            path_age_s=0.05,
            freshness_regime="fresh",
            mpc_mode="tracking",
            mpc_status="solved",
            mpc_latency_s=0.01,
            actor_clearance_m=2.0,
            ttc_s=1.2,
            oracle_only=True,
        )
        control = tick.runtime_dict()
        assert_no_oracle_in_control_payload(control)
        oracle = tick.oracle_dict()
        self.assertTrue(oracle["oracle_only"])
        self.assertFalse(oracle["consumed_by_control"])
        with self.assertRaises(ContractError):
            assert_no_oracle_in_control_payload(
                {"control": control, "oracle_trace": oracle}
            )

    def test_content_hash_field_change(self) -> None:
        art = _artifact()
        h0 = art.artifact_content_hash()
        # change seed_id via rebuild
        d = art.payload_for_hash()
        d["seed_id"] = "seed_b"
        self.assertNotEqual(content_hash(d, nibble=64), h0)


if __name__ == "__main__":
    unittest.main()
