"""Offline tests for paired_live artifact conversion (no CARLA)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import TrajectoryArray  # noqa: E402
from driving_vla.evaluation.paired_contract import (  # noqa: E402
    K2_ANCHOR_SCHEMA,
    ObservationFingerprint,
    assert_no_oracle_in_control_payload,
)
from driving_vla.evaluation.paired_live import (  # noqa: E402
    EXECUTOR_CONFIG_HASH,
    artifact_from_bundle,
    bundle_from_artifact,
)
from driving_vla.evaluation.outcome_metrics import TickRecord  # noqa: E402
from driving_vla.model.k2_builder import (  # noqa: E402
    GUARD_OK,
    K2Diagnostics,
    K2ExecutionSpec,
    K2PredictionBundle,
    stable_hash_xy,
)
from driving_vla.runtime.k2_execution import select_k2  # noqa: E402


def _bundle() -> K2PredictionBundle:
    path = tuple((float(i), 0.0) for i in range(30))
    nh = stable_hash_xy(path)
    pts0 = tuple((float(i + 1) * 0.5, 0.0, 0.0, 5.0, 0.0, 0.0) for i in range(10))
    pts1 = tuple((float(i + 1) * 0.3, 0.0, 0.0, 3.0, 0.0, 0.0) for i in range(10))
    c0 = TrajectoryArray(points_xy_yaw_v_a_kappa=pts0, probability=0.5, candidate_id="v1_nominal")
    c1 = TrajectoryArray(points_xy_yaw_v_a_kappa=pts1, probability=0.5, candidate_id="v1_conservative")
    specs = {
        "v1_nominal": K2ExecutionSpec(
            candidate_id="v1_nominal",
            spatial_path_xy=path,
            speed_samples_mps=tuple(5.0 for _ in range(10)),
            timed_trajectory_hash="th0",
            native_path_hash=nh,
            branch_type="longitudinal_temporal",
        ),
        "v1_conservative": K2ExecutionSpec(
            candidate_id="v1_conservative",
            spatial_path_xy=path,
            speed_samples_mps=tuple(3.0 for _ in range(10)),
            timed_trajectory_hash="th1",
            native_path_hash=nh,
            branch_type="longitudinal_temporal",
        ),
    }
    diag = K2Diagnostics(
        mean_speed_gap_mps=2.0,
        final_progress_gap_m=2.0,
        max_position_separation_m=2.0,
        mean_position_separation_m=1.0,
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
        observation_identity={"frame": "t"},
        model_id="sdf-vla-v1-neural@0.1.0",
        config_hash="cfg",
        retimer_version="safedrive.k2_retimer.v1",
        native_path_xy=path,
        native_path_hash=nh,
        candidates=(c0, c1),
        execution_specs=specs,
        top1_index=0,
        probability_source="fixed_equal_prior_unscaled",
        probability_margin=0.0,
        branch_type="longitudinal_temporal",
        diagnostics=diag,
        guard_status=GUARD_OK,
        guard_reasons=(),
    )


class PairedLiveOfflineTest(unittest.TestCase):
    def test_artifact_roundtrip_select_force(self) -> None:
        b = _bundle()
        obs = ObservationFingerprint(
            front_rgb_sha256="x",
            image_height=512,
            image_width=1024,
            image_channels=3,
            image_layout="bgr",
            ego_observable={"v": 5.0},
            route_targets=[[10.0, 0.0]],
            camera_frame={},
            k2_bundle_hash="k",
        )
        art = artifact_from_bundle(
            b,
            pair_id="p",
            scenario_id="lead_brake_moderate",
            seed_id="seed_a",
            anchor_run_id="a",
            anchor_carla_frame=1,
            anchor_simulation_time_s=1.0,
            requested_initial_state_hash="r",
            measured_initial_state_hash="m",
            observation_fingerprint=obs,
            model_checkpoint_hash="c",
            retimer_hash="rh",
            executor_config_hash=EXECUTOR_CONFIG_HASH,
        )
        self.assertEqual(art.schema_version, K2_ANCHOR_SCHEMA)
        raw = art.to_json_bytes()
        art2 = type(art).from_json_bytes(raw)
        b2 = bundle_from_artifact(art2)
        s0 = select_k2(b2, mode="force", force_index=0)
        s1 = select_k2(b2, mode="force", force_index=1)
        self.assertEqual(s0.candidate_id, "v1_nominal")
        self.assertEqual(s1.candidate_id, "v1_conservative")
        self.assertEqual(art2.artifact_content_hash(), art.artifact_content_hash())

    def test_control_payload_no_oracle(self) -> None:
        t = TickRecord(
            tick_index=0,
            simulation_time_s=0.05,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw_rad=0.0,
            ego_v=1.0,
            selected_candidate_id="v1_nominal",
            executed_candidate_id="v1_nominal",
            source_id="f:v1_nominal",
            path_age_s=0.05,
            freshness_regime="fresh",
            mpc_mode="mpc",
            mpc_status="solved",
            mpc_latency_s=0.01,
            actor_clearance_m=3.0,
            ttc_s=1.5,
            oracle_only=True,
        )
        assert_no_oracle_in_control_payload(t.runtime_dict())


if __name__ == "__main__":
    unittest.main()
