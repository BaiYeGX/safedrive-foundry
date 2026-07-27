"""Spatial K2 V2 artifact ↔ runtime bundle conversion (no VLA re-forward)."""

from __future__ import annotations

from typing import Any, Mapping

from driving_vla.evaluation.paired_contract import (
    K2_ANCHOR_SCHEMA_V2,
    K2AnchorArtifactV2,
    ObservationFingerprint,
    SerializedCandidateV2,
    content_hash,
)
from driving_vla.model.k2_spatial_guard import attach_spatial_guard
from driving_vla.model.k2_spatial_types import (
    BRANCH_TYPE_V2,
    SCHEMA_V2,
    K2CandidateV2,
    K2ExecutionSpecV2,
    K2PredictionBundleV2,
    load_k2_spatial_config,
    stable_hash_xy,
)


def bundle_from_artifact_v2(art: K2AnchorArtifactV2) -> K2PredictionBundleV2:
    """Cold rebuild of runtime V2 bundle from frozen artifact (no VLA)."""
    cands: list[K2CandidateV2] = []
    specs: dict[str, K2ExecutionSpecV2] = {}
    for sc in art.candidates:
        cands.append(
            K2CandidateV2(
                candidate_id=sc.candidate_id,
                mode_id=sc.mode_id,
                available=sc.available,
                availability_reason=sc.availability_reason,
                probability=sc.probability,
                points_xy_yaw_v_a_kappa=sc.points_xy_yaw_v_a_kappa,
                frenet_s=sc.frenet_s,
                frenet_d=sc.frenet_d,
                proposal_path_hash=sc.proposal_path_hash,
                timed_trajectory_hash=sc.timed_trajectory_hash,
                native_anchor_hash=sc.native_anchor_hash,
                head_lineage=sc.head_lineage,
                spatial_path_xy=sc.spatial_path_xy,
            )
        )
        specs[sc.candidate_id] = K2ExecutionSpecV2(
            candidate_id=sc.candidate_id,
            spatial_path_xy=sc.spatial_path_xy,
            speed_samples_mps=sc.speed_samples_mps,
            spatial_path_hash=sc.proposal_path_hash,
            timed_trajectory_hash=sc.timed_trajectory_hash,
            native_anchor_hash=sc.native_anchor_hash,
            branch_type=sc.branch_type,
            mode_id=sc.mode_id,
            available=sc.available,
            head_lineage=sc.head_lineage,
        )
    if len(cands) != 2:
        raise ValueError(f"V2 artifact must have 2 candidates, got {len(cands)}")
    return K2PredictionBundleV2(
        schema_version=SCHEMA_V2,
        observation_identity={
            "pair_id": art.pair_id,
            "scenario_id": art.scenario_id,
            "seed_id": art.seed_id,
            "anchor_run_id": art.anchor_run_id,
        },
        model_id=art.model_id,
        config_hash=art.config_hash,
        base_checkpoint_hash=art.model_checkpoint_hash,
        spatial_head_checkpoint_hash=art.spatial_head_checkpoint_hash,
        backbone_forward_id=art.backbone_forward_id,
        native_path_xy=art.native_path_xy,
        native_path_hash=art.native_path_hash,
        candidates=(cands[0], cands[1]),
        execution_specs=specs,
        top1_index=art.top1_index,
        probability_source=art.probability_source,
        branch_type=art.branch_type,
        guard_status=art.guard_status,
        guard_reasons=art.guard_reasons,
        set_diagnostics=dict(art.set_diagnostics or art.guard_metrics),
    )


def artifact_from_bundle_v2(
    bundle: K2PredictionBundleV2,
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
) -> K2AnchorArtifactV2:
    cands: list[SerializedCandidateV2] = []
    for i, c in enumerate(bundle.candidates):
        spec = bundle.execution_specs[c.candidate_id]
        cands.append(
            SerializedCandidateV2(
                candidate_id=c.candidate_id,
                candidate_index=i,
                mode_id=c.mode_id,
                available=c.available,
                availability_reason=c.availability_reason,
                probability=float(c.probability),
                points_xy_yaw_v_a_kappa=c.points_xy_yaw_v_a_kappa,
                frenet_s=c.frenet_s,
                frenet_d=c.frenet_d,
                spatial_path_xy=spec.spatial_path_xy,
                speed_samples_mps=spec.speed_samples_mps,
                proposal_path_hash=c.proposal_path_hash,
                timed_trajectory_hash=c.timed_trajectory_hash,
                native_anchor_hash=c.native_anchor_hash,
                head_lineage=c.head_lineage,
                branch_type=spec.branch_type or BRANCH_TYPE_V2,
            )
        )
    return K2AnchorArtifactV2(
        schema_version=K2_ANCHOR_SCHEMA_V2,
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
        spatial_head_checkpoint_hash=bundle.spatial_head_checkpoint_hash,
        config_hash=bundle.config_hash,
        backbone_forward_id=bundle.backbone_forward_id,
        executor_config_hash=executor_config_hash,
        native_path_xy=bundle.native_path_xy,
        native_path_hash=bundle.native_path_hash,
        candidates=tuple(cands),
        top1_index=bundle.top1_index,
        guard_status=bundle.guard_status,
        guard_reasons=bundle.guard_reasons,
        guard_metrics=dict(bundle.set_diagnostics or {}),
        probability_source=bundle.probability_source,
        branch_type=bundle.branch_type,
        set_diagnostics=dict(bundle.set_diagnostics or {}),
        evidence_lineage=evidence_lineage,
    )


def make_dummy_observation_fingerprint(
    *,
    k2_bundle_hash: str = "dummy",
) -> ObservationFingerprint:
    return ObservationFingerprint(
        front_rgb_sha256="0" * 64,
        image_height=224,
        image_width=224,
        image_channels=3,
        image_layout="HWC",
        ego_observable={"speed_mps": 5.0},
        route_targets=[],
        camera_frame={"frame_id": "offline"},
        k2_bundle_hash=k2_bundle_hash,
    )


def guarded_bundle_roundtrip(bundle: K2PredictionBundleV2) -> K2PredictionBundleV2:
    """Attach guard then serialize/deserialize via artifact to prove cold rebuild."""
    cfg = load_k2_spatial_config()
    g = attach_spatial_guard(bundle, config=cfg, require_diversity_if_eligible=True)
    fp = make_dummy_observation_fingerprint(
        k2_bundle_hash=content_hash({"native": g.native_path_hash}, nibble=16)
    )
    art = artifact_from_bundle_v2(
        g,
        pair_id="offline-probe",
        scenario_id="contract_probe",
        seed_id="seed_a",
        anchor_run_id="anchor-offline",
        anchor_carla_frame=0,
        anchor_simulation_time_s=0.0,
        requested_initial_state_hash="0" * 64,
        measured_initial_state_hash="0" * 64,
        observation_fingerprint=fp,
        model_checkpoint_hash=g.base_checkpoint_hash,
        executor_config_hash="offline_executor",
        evidence_lineage="contract_probe",
    )
    raw = art.to_json_bytes()
    art2 = K2AnchorArtifactV2.from_json_bytes(raw)
    return bundle_from_artifact_v2(art2)
