"""Joint R2/World collection contracts.

This module is deliberately independent from the frozen R2 12-pair contract.
It describes one collection anchor and a resumable campaign made from immutable
12-pair shards.  Oracle fields remain labels and are never part of runtime
model input.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

COLLECTION_SAMPLE_SCHEMA = "safedrive.r23_collection_sample.v1"
CAMPAIGN_SCHEMA = "safedrive.r23_campaign.v1"
CAMPAIGN_CHECKPOINT_SCHEMA = "safedrive.r23_campaign_checkpoint.v1"
SHARD_SIZE = 12
VALID_SPLITS = frozenset({"train", "val", "test", "holdout"})
VALID_PHASES = frozenset({"r2_calibration", "world_formal"})
TECHNICAL_FAILURE_CODES = frozenset(
    {
        "CARLA_CONNECT_FAILURE",
        "CARLA_RPC_TIMEOUT",
        "SENSOR_SYNC_FAILURE",
        "SPAWN_FAILURE",
        "SERVER_FAILURE",
        "CLEANUP_FAILURE",
    }
)


class R23CollectionError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _require_hash(value: str, name: str) -> str:
    token = str(value or "").lower()
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise R23CollectionError(f"{name} must be a 64-character sha256")
    return token


@dataclass(frozen=True)
class CollectionSlot:
    slot_id: str
    phase: str
    shard_id: str
    shard_index: int
    slot_index: int
    lineage_id: str
    family: str
    map_name: str
    split: str
    condition_variant: str
    seed_id: str
    anchor_variant: str
    scenario_id: str
    reserve: bool = False

    def validate(self) -> None:
        for name in (
            "slot_id",
            "shard_id",
            "lineage_id",
            "family",
            "map_name",
            "condition_variant",
            "seed_id",
            "anchor_variant",
            "scenario_id",
        ):
            if not str(getattr(self, name)).strip():
                raise R23CollectionError(f"slot.{name} must be non-empty")
        if self.phase not in VALID_PHASES:
            raise R23CollectionError(f"unsupported phase {self.phase!r}")
        if self.split not in VALID_SPLITS:
            raise R23CollectionError(f"unsupported split {self.split!r}")
        if self.shard_index < 0 or not 0 <= self.slot_index < SHARD_SIZE:
            raise R23CollectionError("invalid shard/slot index")
        expected = content_hash(
            {
                "phase": self.phase,
                "lineage_id": self.lineage_id,
                "condition_variant": self.condition_variant,
                "seed_id": self.seed_id,
                "anchor_variant": self.anchor_variant,
            }
        )[:24]
        if self.slot_id != expected:
            raise R23CollectionError(
                f"slot_id mismatch for {self.scenario_id}: {self.slot_id} != {expected}"
            )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CollectionSlot":
        item = cls(
            slot_id=str(value["slot_id"]),
            phase=str(value["phase"]),
            shard_id=str(value["shard_id"]),
            shard_index=int(value["shard_index"]),
            slot_index=int(value["slot_index"]),
            lineage_id=str(value["lineage_id"]),
            family=str(value["family"]),
            map_name=str(value["map_name"]),
            split=str(value["split"]),
            condition_variant=str(value["condition_variant"]),
            seed_id=str(value["seed_id"]),
            anchor_variant=str(value["anchor_variant"]),
            scenario_id=str(value["scenario_id"]),
            reserve=bool(value.get("reserve", False)),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class R23CollectionSampleV1:
    slot: CollectionSlot
    status: str
    observable_path: str | None
    anchor_artifact_path: str | None
    feature_path: str | None
    branch_paths: tuple[str | None, str | None]
    technical_failure_codes: tuple[str, ...] = ()
    unavailable_reasons: tuple[str | None, str | None] = (None, None)
    provenance: Mapping[str, str] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.slot.validate()
        if self.status not in {"COMPLETED", "FAILED", "SINGLETON"}:
            raise R23CollectionError(f"invalid sample status {self.status!r}")
        if len(self.branch_paths) != 2 or len(self.unavailable_reasons) != 2:
            raise R23CollectionError("branch/reason tuples must have K=2")
        unknown = set(self.technical_failure_codes).difference(TECHNICAL_FAILURE_CODES)
        if unknown:
            raise R23CollectionError(f"unknown technical failure codes: {sorted(unknown)}")
        required_hashes = (
            "registry_sha256",
            "r2_checkpoint_sha256",
            "executor_sha256",
            "collection_config_sha256",
        )
        for name in required_hashes:
            _require_hash(str(self.provenance.get(name, "")), f"provenance.{name}")
        if self.status != "FAILED":
            for name, value in (
                ("observable_path", self.observable_path),
                ("anchor_artifact_path", self.anchor_artifact_path),
                ("feature_path", self.feature_path),
            ):
                if not str(value or "").strip():
                    raise R23CollectionError(f"{name} required for {self.status}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": COLLECTION_SAMPLE_SCHEMA,
            "slot": self.slot.to_dict(),
            "status": self.status,
            "observable_path": self.observable_path,
            "anchor_artifact_path": self.anchor_artifact_path,
            "feature_path": self.feature_path,
            "branch_paths": list(self.branch_paths),
            "technical_failure_codes": list(self.technical_failure_codes),
            "unavailable_reasons": list(self.unavailable_reasons),
            "provenance": dict(self.provenance),
            "audit": dict(self.audit),
        }


def validate_campaign_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if str(manifest.get("schema_version")) != CAMPAIGN_SCHEMA:
        raise R23CollectionError("campaign schema mismatch")
    stored_hash = str(manifest.get("manifest_content_hash", ""))
    body = dict(manifest)
    body.pop("manifest_content_hash", None)
    measured_hash = content_hash(body)
    if stored_hash != measured_hash:
        raise R23CollectionError("campaign content hash mismatch")
    phase = str(manifest.get("phase"))
    if phase not in VALID_PHASES:
        raise R23CollectionError(f"invalid campaign phase {phase!r}")
    slots = [CollectionSlot.from_dict(value) for value in manifest.get("slots", [])]
    if not slots or len(slots) % SHARD_SIZE:
        raise R23CollectionError("campaign slots must be non-empty and divisible by 12")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise R23CollectionError("duplicate slot_id")
    lineage_splits: dict[str, set[str]] = {}
    for slot in slots:
        if slot.phase != phase:
            raise R23CollectionError("slot phase differs from campaign phase")
        lineage_splits.setdefault(slot.lineage_id, set()).add(slot.split)
    overlap = {key: values for key, values in lineage_splits.items() if len(values) > 1}
    if overlap:
        raise R23CollectionError(f"lineage split overlap: {sorted(overlap)}")
    r2_hash = str(manifest.get("r2_checkpoint_sha256", ""))
    _require_hash(r2_hash, "r2_checkpoint_sha256")
    if phase == "world_formal" and r2_hash == "0" * 64:
        raise R23CollectionError("world formal campaign requires frozen R2 checkpoint")
    return {
        "manifest_content_hash": measured_hash,
        "n_slots": len(slots),
        "n_shards": len(slots) // SHARD_SIZE,
        "n_reserve": sum(slot.reserve for slot in slots),
        "n_lineages": len(lineage_splits),
    }


def build_campaign_manifest(
    *,
    campaign_id: str,
    phase: str,
    slots: Sequence[CollectionSlot],
    r2_checkpoint_sha256: str,
    collection_config_sha256: str,
    base_slot_target: int,
    completed_target: int,
) -> dict[str, Any]:
    for slot in slots:
        slot.validate()
    body = {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": str(campaign_id),
        "phase": str(phase),
        "shard_size": SHARD_SIZE,
        "base_slot_target": int(base_slot_target),
        "completed_target": int(completed_target),
        "r2_checkpoint_sha256": _require_hash(
            r2_checkpoint_sha256, "r2_checkpoint_sha256"
        ),
        "collection_config_sha256": _require_hash(
            collection_config_sha256, "collection_config_sha256"
        ),
        "slots": [slot.to_dict() for slot in slots],
    }
    manifest = {**body, "manifest_content_hash": content_hash(body)}
    validate_campaign_manifest(manifest)
    return manifest


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit = validate_campaign_manifest(manifest)
    if str(checkpoint.get("schema_version")) != CAMPAIGN_CHECKPOINT_SCHEMA:
        raise R23CollectionError("checkpoint schema mismatch")
    if str(checkpoint.get("manifest_content_hash")) != audit["manifest_content_hash"]:
        raise R23CollectionError("checkpoint campaign hash mismatch")
    results = list(checkpoint.get("results", []))
    slots = [CollectionSlot.from_dict(value) for value in manifest["slots"]]
    if len(results) > len(slots):
        raise R23CollectionError("checkpoint has more results than slots")
    for index, result in enumerate(results):
        if str(result.get("slot_id")) != slots[index].slot_id:
            raise R23CollectionError("checkpoint results must be a contiguous slot prefix")
    if int(checkpoint.get("last_completed_index", -1)) != len(results) - 1:
        raise R23CollectionError("checkpoint last_completed_index mismatch")
    return {"n_results": len(results), "next_index": len(results)}


def technical_failure_only(result: Mapping[str, Any]) -> bool:
    codes = tuple(str(code) for code in result.get("failure_codes", ()))
    return bool(codes) and set(codes).issubset(TECHNICAL_FAILURE_CODES)


def validate_completed_joint_artifacts(
    *,
    attempt_dir: Path,
    expected_scenario_id: str,
    expected_seed_id: str,
) -> dict[str, Any]:
    """Fail closed on incomplete or cross-forward joint collection artifacts."""

    anchor_dir = attempt_dir / "anchor"
    feature_path = anchor_dir / "feature.json"
    lineage_path = anchor_dir / "feature_lineage.json"
    artifact_path = anchor_dir / "anchor_bundle_v2.json"
    rgb_path = anchor_dir / "anchor_front_rgb.npy"
    history_path = anchor_dir / "observable_history.json"
    required = (feature_path, lineage_path, artifact_path, rgb_path, history_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise R23CollectionError(
            f"completed slot missing joint anchor artifacts: {missing}"
        )

    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if str(feature.get("scenario_id")) != str(expected_scenario_id):
        raise R23CollectionError("feature scenario_id mismatch")
    if str(feature.get("seed_id")) != str(expected_seed_id):
        raise R23CollectionError("feature seed_id mismatch")
    mean64 = list(feature.get("driving_feature") or ())
    if len(mean64) != 64:
        raise R23CollectionError(
            f"mean64 feature must contain 64 values, got {len(mean64)}"
        )
    ego_v = float(feature.get("ego_v") or 0.0)
    if ego_v < 0.75:
        raise R23CollectionError(
            f"anchor speed below meaningful collection floor: {ego_v:.3f} m/s"
        )
    forward_ids = {
        str(feature.get("backbone_forward_id") or ""),
        str(lineage.get("backbone_forward_id") or ""),
        str(artifact.get("backbone_forward_id") or ""),
    }
    if "" in forward_ids or len(forward_ids) != 1:
        raise R23CollectionError(
            f"RGB/feature/artifact forward lineage mismatch: {sorted(forward_ids)}"
        )
    feature_hashes = {
        str(feature.get("driving_feature_hash") or ""),
        str(lineage.get("driving_feature_hash") or ""),
        str(lineage.get("native_feature_hash") or ""),
    }
    if "" in feature_hashes or len(feature_hashes) != 1:
        raise R23CollectionError(
            f"driving feature hash mismatch: {sorted(feature_hashes)}"
        )
    if int(history.get("history_count", 0)) != 5:
        raise R23CollectionError("anchor history must contain five frames")
    frame_ids = list(history.get("frame_ids") or ())
    ego_history = list(history.get("ego_history") or ())
    if len(frame_ids) != 5 or len(set(frame_ids)) != 5 or len(ego_history) != 5:
        raise R23CollectionError("anchor history frames must be five real distinct frames")

    branch_audits: dict[str, Any] = {}
    for branch_index in range(2):
        branch_dir = attempt_dir / f"branch-{branch_index}"
        unavailable_path = branch_dir / "unavailable.json"
        scene_path = branch_dir / "observable_scene_t0.json"
        if unavailable_path.is_file():
            branch_audits[str(branch_index)] = {"available": False}
            continue
        if not scene_path.is_file():
            raise R23CollectionError(
                f"executed branch {branch_index} missing observable_scene_t0.json"
            )
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        ego_rows = list(scene.get("ego_history") or ())
        actor_rows = list(scene.get("actor_histories") or ())
        if len(ego_rows) != 5:
            raise R23CollectionError(
                f"branch {branch_index} ego history must contain five frames"
            )
        for actor in actor_rows:
            if len(list(actor.get("history") or ())) != 5:
                raise R23CollectionError(
                    f"branch {branch_index} actor history must contain five masks/states"
                )
        branch_audits[str(branch_index)] = {
            "available": True,
            "n_actor_histories": len(actor_rows),
        }
    return {
        "backbone_forward_id": next(iter(forward_ids)),
        "driving_feature_hash": next(iter(feature_hashes)),
        "history_count": 5,
        "anchor_speed_mps": ego_v,
        "branch_history": branch_audits,
    }
