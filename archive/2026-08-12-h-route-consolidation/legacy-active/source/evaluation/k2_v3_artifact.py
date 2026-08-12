"""Cold, hash-checked K2 V3 artifact serialization.

The artifact contains the already-generated SimLingo anchor and semantic
candidate.  Deserialization never runs the VLA again and always re-runs the
V3 Contract Guard before the bundle can reach an executor or World.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from driving_vla.model.k2_v3_guard import attach_k2_v3_guard
from driving_vla.model.k2_v3_types import (
    AlternativeKind,
    K2CandidateV3,
    K2PredictionBundleV3,
    ManeuverPhase,
)
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    canonical_sha256,
)

K2_ANCHOR_SCHEMA_V3 = "safedrive.k2_anchor.v3"


def _candidate_from_mapping(value: Mapping[str, Any]) -> K2CandidateV3:
    return K2CandidateV3(
        candidate_id=str(value["candidate_id"]),
        alternative_kind=AlternativeKind(str(value["alternative_kind"])),
        route_maneuver=RouteManeuver(str(value["route_maneuver"])),
        available=bool(value["available"]),
        availability_reason=str(value.get("availability_reason") or ""),
        probability=float(value["probability"]),
        target_lane_side=TargetLaneSide(str(value["target_lane_side"])),
        maneuver_phase=ManeuverPhase(str(value["maneuver_phase"])),
        spatial_path_xy=tuple(
            (float(point[0]), float(point[1]))
            for point in value["spatial_path_xy"]
        ),
        points_xy_yaw_v_a_kappa=tuple(
            tuple(float(component) for component in row[:6])
            for row in value["points_xy_yaw_v_a_kappa"]
        ),
        route_hash=str(value["route_hash"]),
        topology_hash=str(value["topology_hash"]),
        corridor_hash=str(value["corridor_hash"]),
        intent_hash=str(value["intent_hash"]),
        spatial_path_hash=str(value["spatial_path_hash"]),
        timed_trajectory_hash=str(value["timed_trajectory_hash"]),
        origin_lane_signature=str(value.get("origin_lane_signature") or ""),
        target_lane_signature=str(value.get("target_lane_signature") or ""),
        head_lineage=str(value.get("head_lineage") or "spatial_mode_head_v3"),
        feature_content_hash=str(value.get("feature_content_hash") or ""),
        raw_head_output_hash=str(value.get("raw_head_output_hash") or ""),
        metadata=dict(value.get("metadata") or {}),
    )


def bundle_from_mapping_v3(
    value: Mapping[str, Any],
    *,
    require_guard_ok: bool = True,
) -> K2PredictionBundleV3:
    candidates = tuple(
        _candidate_from_mapping(item) for item in value.get("candidates") or ()
    )
    if len(candidates) != 2:
        raise ValueError(f"K2 V3 artifact requires 2 candidates, got {len(candidates)}")
    bundle = K2PredictionBundleV3(
        route_context=RouteContextV3.from_mapping(value["route_context"]),
        candidates=(candidates[0], candidates[1]),
        observation_identity=dict(value.get("observation_identity") or {}),
        model_id=str(value["model_id"]),
        config_hash=str(value["config_hash"]),
        base_checkpoint_hash=str(value["base_checkpoint_hash"]),
        spatial_head_checkpoint_hash=str(
            value["spatial_head_checkpoint_hash"]
        ),
        backbone_forward_id=str(value["backbone_forward_id"]),
        top1_index=int(value.get("top1_index", 0)),
        probability_source=str(
            value.get("probability_source") or "fixed_prior_uncalibrated"
        ),
        schema_version=str(value["schema_version"]),
        branch_type=str(value["branch_type"]),
        guard_status=str(value.get("guard_status") or "UNVALIDATED"),
        guard_reasons=tuple(str(reason) for reason in value.get("guard_reasons") or ()),
        guard_metrics=dict(value.get("guard_metrics") or {}),
        bundle_hash=str(value["bundle_hash"]),
    )
    guarded = attach_k2_v3_guard(bundle)
    if require_guard_ok and guarded.guard_status != "OK":
        raise ValueError(
            "K2 V3 artifact rejected by Guard: "
            + ",".join(guarded.guard_reasons)
        )
    return guarded


def artifact_json_bytes_v3(bundle: K2PredictionBundleV3) -> bytes:
    """Serialize a V3 bundle using canonical JSON for durable evidence."""
    guarded = attach_k2_v3_guard(bundle)
    if guarded.guard_status != "OK":
        raise ValueError(
            "cannot freeze rejected K2 V3 bundle: "
            + ",".join(guarded.guard_reasons)
        )
    return json.dumps(
        guarded.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def bundle_from_artifact_json_v3(
    payload: bytes | bytearray | memoryview | str,
    *,
    require_guard_ok: bool = True,
) -> K2PredictionBundleV3:
    raw = payload.decode("utf-8") if not isinstance(payload, str) else payload
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("K2 V3 artifact root must be an object")
    return bundle_from_mapping_v3(value, require_guard_ok=require_guard_ok)


@dataclass(frozen=True)
class K2AnchorArtifactV3:
    """One guarded V3 bundle plus the exact paired-branch anchor identity."""

    pair_id: str
    scenario_id: str
    seed_id: str
    anchor_run_id: str
    anchor_carla_frame: int
    anchor_simulation_time_s: float
    requested_initial_state_hash: str
    measured_initial_state_hash: str
    observation_fingerprint: Mapping[str, Any]
    model_checkpoint_hash: str
    executor_config_hash: str
    bundle: K2PredictionBundleV3
    schema_version: str = K2_ANCHOR_SCHEMA_V3

    def __post_init__(self) -> None:
        if self.schema_version != K2_ANCHOR_SCHEMA_V3:
            raise ValueError(
                f"unsupported K2 V3 anchor schema {self.schema_version}"
            )
        if self.bundle.guard_status != "OK":
            raise ValueError("K2 V3 anchor requires Guard OK bundle")

    @property
    def candidates(self) -> tuple[K2CandidateV3, K2CandidateV3]:
        return self.bundle.candidates

    @property
    def top1_index(self) -> int:
        return int(self.bundle.top1_index)

    @property
    def guard_status(self) -> str:
        return str(self.bundle.guard_status)

    @property
    def guard_reasons(self) -> tuple[str, ...]:
        return tuple(self.bundle.guard_reasons)

    @property
    def config_hash(self) -> str:
        return str(self.bundle.config_hash)

    @property
    def route_context(self) -> RouteContextV3:
        return self.bundle.route_context

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "seed_id": self.seed_id,
            "anchor_run_id": self.anchor_run_id,
            "anchor_carla_frame": int(self.anchor_carla_frame),
            "anchor_simulation_time_s": float(
                self.anchor_simulation_time_s
            ),
            "requested_initial_state_hash": (
                self.requested_initial_state_hash
            ),
            "measured_initial_state_hash": self.measured_initial_state_hash,
            "observation_fingerprint": dict(
                self.observation_fingerprint
            ),
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "executor_config_hash": self.executor_config_hash,
            "bundle": self.bundle.to_dict(),
        }

    def artifact_content_hash(self) -> str:
        return canonical_sha256(self.content_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.content_payload(),
            "artifact_content_hash": self.artifact_content_hash(),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "K2AnchorArtifactV3":
        artifact = cls(
            schema_version=str(value["schema_version"]),
            pair_id=str(value["pair_id"]),
            scenario_id=str(value["scenario_id"]),
            seed_id=str(value["seed_id"]),
            anchor_run_id=str(value["anchor_run_id"]),
            anchor_carla_frame=int(value["anchor_carla_frame"]),
            anchor_simulation_time_s=float(
                value["anchor_simulation_time_s"]
            ),
            requested_initial_state_hash=str(
                value["requested_initial_state_hash"]
            ),
            measured_initial_state_hash=str(
                value["measured_initial_state_hash"]
            ),
            observation_fingerprint=dict(
                value.get("observation_fingerprint") or {}
            ),
            model_checkpoint_hash=str(value["model_checkpoint_hash"]),
            executor_config_hash=str(value["executor_config_hash"]),
            bundle=bundle_from_mapping_v3(
                value["bundle"],
                require_guard_ok=True,
            ),
        )
        declared = str(value.get("artifact_content_hash") or "")
        if declared and declared != artifact.artifact_content_hash():
            raise ValueError("K2 V3 anchor artifact content hash mismatch")
        return artifact

    @classmethod
    def from_json_bytes(
        cls,
        payload: bytes | bytearray | memoryview | str,
    ) -> "K2AnchorArtifactV3":
        raw = payload.decode("utf-8") if not isinstance(payload, str) else payload
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("K2 V3 anchor root must be an object")
        return cls.from_dict(value)


__all__ = [
    "artifact_json_bytes_v3",
    "bundle_from_artifact_json_v3",
    "bundle_from_mapping_v3",
    "K2_ANCHOR_SCHEMA_V3",
    "K2AnchorArtifactV3",
]
