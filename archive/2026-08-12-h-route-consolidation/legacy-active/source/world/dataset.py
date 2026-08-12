"""ActionBranchDatasetV0 builder, immutable shards, split and quality audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from driving_vla.evaluation.actor_future_collector import (
    load_actor_future_trace,
    resample_actor_frames,
)

from .contracts import (
    FUTURE_FEATURES,
    K,
    MAX_ACTORS,
    OUTCOME_FEATURES,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V1,
    T,
    ActionBranchSample,
    SampleIdentity,
    WorldContractError,
    content_hash,
)
from .observable_builder import (
    assert_observable_only,
    build_actor_history,
    build_ego_history,
    build_road_context,
    vector_world_to_ego,
    world_to_ego,
)

DATASET_MANIFEST_SCHEMA = "safedrive.action_branch_dataset_manifest.v0"
DATASET_MANIFEST_SCHEMA_V1 = "safedrive.action_branch_dataset_manifest.v1"
SPLIT_MANIFEST_SCHEMA = "safedrive.action_branch_split.v0"
SPLIT_MANIFEST_SCHEMA_V1 = "safedrive.action_branch_split.v1"
QUALITY_REPORT_SCHEMA = "safedrive.action_branch_quality.v0"

OUTCOME_NAMES = (
    "collision",
    "offroad_fraction",
    "minimum_ttc_s",
    "minimum_clearance_m",
    "route_progress_m",
    "comfort_jerk_p95",
    "mpc_reliability",
    "first_collision_time_s",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WorldContractError(f"required file missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorldContractError(f"expected JSON object: {path}")
    return value


def _candidate_tensor_from_artifact(
    artifact: Mapping[str, Any],
    *,
    anchor_pose: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[str | None, str | None]]:
    tensor = np.zeros((K, T, 8), dtype=np.float32)
    mask = np.zeros(K, dtype=bool)
    reasons: list[str | None] = [None, None]
    ex, ey, eyaw = anchor_pose
    candidates = list(artifact.get("candidates", []))
    by_index = {int(x["candidate_index"]): x for x in candidates}
    set_level_collapse = set(artifact.get("guard_reasons", [])) == {
        "SPATIAL_COLLAPSE_ELIGIBLE"
    }
    for index in range(K):
        candidate = by_index.get(index)
        if candidate is None:
            reasons[index] = "MISSING_CANDIDATE"
            continue
        if not bool(candidate.get("available", True)) or (
            index == 1 and set_level_collapse
        ):
            reasons[index] = str(
                "SPATIAL_COLLAPSE_ELIGIBLE"
                if index == 1 and set_level_collapse
                else candidate.get("availability_reason") or "UNAVAILABLE"
            )
            continue
        points = list(candidate.get("points_xy_yaw_v_a_kappa", []))
        if len(points) != T:
            raise WorldContractError(f"candidate {index} has {len(points)} points, expected {T}")
        for ti, point in enumerate(points):
            x, y = world_to_ego(
                float(point[0]), float(point[1]), ego_x=ex, ego_y=ey, ego_yaw_rad=eyaw
            )
            yaw = float(point[2]) - eyaw
            tensor[index, ti] = (
                x,
                y,
                math.sin(yaw),
                math.cos(yaw),
                float(point[3]),
                float(point[4]),
                float(point[5]),
                (ti + 1) * 0.25,
            )
        mask[index] = True
    return tensor, mask, (reasons[0], reasons[1])


def _outcome_vector(metrics: Mapping[str, Any]) -> np.ndarray:
    ttc = metrics.get("minimum_ttc_s")
    first_collision = metrics.get("first_collision_time_s")
    return np.asarray(
        [
            1.0 if int(metrics.get("collision_episode_count", 0)) > 0 else 0.0,
            float(metrics.get("offroad_fraction", 0.0)),
            2.5 if ttc is None else min(2.5, max(0.0, float(ttc))),
            float(metrics.get("minimum_actor_clearance_m") or 100.0),
            float(metrics.get("route_progress_delta_m", 0.0)),
            float(metrics.get("jerk_abs_p95", 0.0)),
            float(metrics.get("mpc_solved_count", 0))
            / max(1.0, float(metrics.get("completed_primary_ticks", 50))),
            2.5 if first_collision is None else min(2.5, max(0.0, float(first_collision))),
        ],
        dtype=np.float32,
    )


def _future_tensor(
    *,
    trace_path: Path,
    actor_ids: Sequence[str],
    anchor_pose: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    rows = load_actor_future_trace(trace_path)
    aligned = resample_actor_frames(rows)
    values = np.zeros((MAX_ACTORS, T, FUTURE_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_ACTORS, T), dtype=bool)
    ex, ey, eyaw = anchor_pose
    for ai, actor_id in enumerate(actor_ids[:MAX_ACTORS]):
        frames = aligned.get(actor_id)
        if frames is None:
            continue
        for ti, frame in enumerate(frames):
            if frame is None:
                continue
            x, y = world_to_ego(
                frame.x, frame.y, ego_x=ex, ego_y=ey, ego_yaw_rad=eyaw
            )
            vx, vy = vector_world_to_ego(frame.vx, frame.vy, ego_yaw_rad=eyaw)
            yaw = frame.yaw_rad - eyaw
            values[ai, ti] = (x, y, math.sin(yaw), math.cos(yaw), vx, vy)
            mask[ai, ti] = True
    return values, mask


def sample_from_attempt(attempt_dir: Path) -> ActionBranchSample:
    """Build one immutable sample from a completed R3 paired attempt."""
    attempt_dir = Path(attempt_dir)
    manifest_path = attempt_dir / "pair_manifest.json"
    if not manifest_path.is_file():
        # V3/V4 live runners write pair_report first; the V4 runner also emits
        # a compatibility pair_manifest before returning.  Keeping this
        # fallback makes interrupted development attempts inspectable without
        # changing the frozen V0 reader contract.
        manifest_path = attempt_dir / "pair_report.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("status", "COMPLETED")).upper() == "FAILED":
        raise WorldContractError(f"failed attempt cannot supervise World: {attempt_dir}")
    artifact_path = next(
        (
            attempt_dir / "anchor" / name
            for name in (
                "anchor_bundle_v2.json",
                "anchor_bundle_v3.json",
                "anchor_bundle_v4.json",
            )
            if (attempt_dir / "anchor" / name).is_file()
        ),
        None,
    )
    if artifact_path is None:
        raise WorldContractError(f"anchor artifact missing: {attempt_dir}")
    raw_artifact = _read_json(artifact_path)
    # V3/V4 artifacts wrap the fixed candidates under ``bundle``.  Normalize
    # only the reader-local view; the frozen source artifact remains untouched.
    artifact = dict(raw_artifact)
    if isinstance(raw_artifact.get("bundle"), Mapping):
        bundle = dict(raw_artifact["bundle"])
        artifact.update(
            {
                "candidates": [
                    {"candidate_index": index, **dict(candidate)}
                    for index, candidate in enumerate(bundle.get("candidates") or ())
                ],
                "config_hash": bundle.get("config_hash", ""),
                "executor_config_hash": raw_artifact.get(
                    "executor_config_hash", ""
                ),
            }
        )
    oracle_path = attempt_dir / "pair_oracle.json"
    comparability_path = attempt_dir / "pair_comparability.json"
    pair_report_path = attempt_dir / "pair_report.json"
    pair_report = _read_json(pair_report_path) if pair_report_path.is_file() else {}
    oracle = _read_json(oracle_path) if oracle_path.is_file() else dict(pair_report.get("oracle") or {})
    comparability = (
        _read_json(comparability_path)
        if comparability_path.is_file()
        else dict(pair_report.get("comparability") or {})
    )
    scene0 = _read_json(attempt_dir / "branch-0" / "observable_scene_t0.json")
    assert_observable_only(scene0)

    ego = scene0["ego"]
    anchor_pose = (float(ego["x"]), float(ego["y"]), float(ego["yaw_rad"]))
    ego_rows = list(scene0.get("ego_history") or [{**ego, "dt": 0.0}])
    ego_history, ego_mask = build_ego_history(ego_rows, anchor_pose=anchor_pose)
    actor_rows = list(scene0.get("actor_histories") or [])
    if not actor_rows:
        actor_rows = [
            {
                **actor,
                "history": [{**actor, "dt": 0.0}],
            }
            for actor in scene0.get("actors", [])
        ]
    actor_history, actor_history_mask, actor_ids = build_actor_history(
        actor_rows,
        anchor_pose=anchor_pose,
    )
    route_points = scene0.get("route_waypoints", [])
    road_polylines = list(scene0.get("road_polylines") or [])
    if not road_polylines:
        road_polylines = [
            {"points": route_points, "type_id": 0.0, "speed_limit_mps": 0.0}
        ]
    road, road_mask = build_road_context(
        road_polylines,
        anchor_pose=anchor_pose,
    )
    candidates, candidate_mask, unavailable_reasons = _candidate_tensor_from_artifact(
        artifact,
        anchor_pose=anchor_pose,
    )

    actor_future = np.zeros((K, MAX_ACTORS, T, FUTURE_FEATURES), dtype=np.float32)
    actor_future_mask = np.zeros((K, MAX_ACTORS, T), dtype=bool)
    outcomes = np.zeros((K, OUTCOME_FEATURES), dtype=np.float32)
    outcome_mask = np.zeros(K, dtype=bool)
    for index in range(K):
        branch_dir = attempt_dir / f"branch-{index}"
        if not candidate_mask[index] or not branch_dir.is_dir():
            continue
        summary = _read_json(branch_dir / "branch_summary.json")
        metrics = summary.get("metrics")
        if not isinstance(metrics, Mapping):
            raise WorldContractError(f"branch {index} has no outcome metrics")
        if (
            not bool(metrics.get("completed_primary_horizon"))
            or int(metrics.get("mpc_timeout_count", 0)) > 0
            or int(metrics.get("mpc_fallback_count", 0)) > 0
        ):
            continue
        future, future_mask = _future_tensor(
            trace_path=branch_dir / "oracle" / "actor_future_trace.jsonl",
            actor_ids=actor_ids,
            anchor_pose=anchor_pose,
        )
        actor_future[index] = future
        actor_future_mask[index] = future_mask
        outcomes[index] = _outcome_vector(metrics)
        outcome_mask[index] = True

    comparable = bool(comparability.get("comparable", False)) and bool(outcome_mask.all())
    pair_label = str(oracle.get("pair_label", "")).upper()
    rank_mask = comparable and pair_label in {
        "TIE",
        "TOP1_BEST",
        "CANDIDATE_1_BEST",
        "BOTH_BAD",
    }
    tie_target = pair_label == "TIE" and rank_mask
    winner = oracle.get("outcome_delta", {}).get("winner_index")
    if tie_target:
        rank_target = 0.0
    elif winner == 0:
        rank_target = 1.0
    elif winner == 1:
        rank_target = -1.0
    else:
        rank_mask = False
        rank_target = 0.0
    rank_weight = (
        0.25 if rank_mask and pair_label == "BOTH_BAD" else 1.0 if rank_mask else 0.0
    )

    scenario_id = str(artifact["scenario_id"])
    scenario_lineage = re.sub(r"^r3v\d+_", "", scenario_id).split("__", 1)[0]
    seed_id = str(artifact["seed_id"])
    family = str(oracle.get("family") or manifest.get("family") or "unknown")
    condition_variant = str(
        pair_report.get("condition_variant")
        or manifest.get("condition_variant")
        or (scenario_id.split("__", 1)[1] if "__" in scenario_id else "base")
    )
    repeat_group = str(
        pair_report.get("repeat_group")
        or manifest.get("repeat_group")
        or scenario_lineage
    )
    aa_noise_identity = str(
        pair_report.get("aa_noise_identity")
        or manifest.get("aa_noise_identity")
        or content_hash(
            {
                "namespace": "r3_aa_noise_probe",
                "repeat_group": repeat_group,
                "candidate_id": "v3_nominal_progress",
            }
        )
    ).lower()
    map_name = str(scene0.get("map_name") or "unknown")
    source_hash = file_sha256(manifest_path)
    observation_hash = content_hash(
        {
            "ego_history": ego_history.tolist(),
            "actor_history": actor_history.tolist(),
            "road": road.tolist(),
        }
    )
    identity = SampleIdentity(
        sample_id=f"{artifact.get('pair_id', manifest.get('pair_id', 'unknown'))}-a{int(manifest.get('attempt_id', 0))}",
        pair_id=str(artifact.get("pair_id") or manifest.get("pair_id") or ""),
        scenario_id=scenario_id,
        seed_id=seed_id,
        group_key=f"{map_name}|{family}|{scenario_lineage}",
        family=family,
        map_name=map_name,
        initial_state_hash=str(artifact["measured_initial_state_hash"]),
        observation_hash=observation_hash,
        anchor_artifact_hash=str(
            artifact.get("artifact_content_hash")
            or manifest.get("artifact_content_hash")
            or ""
        ),
        model_hash=str(
            artifact.get("model_checkpoint_hash")
            or manifest.get("r2_checkpoint_sha256")
            or bundle.get("base_checkpoint_hash", "")
            if isinstance(raw_artifact.get("bundle"), Mapping)
            else artifact.get("model_checkpoint_hash", "")
        ),
        guard_hash=str(artifact.get("config_hash") or ""),
        executor_hash=str(artifact.get("executor_config_hash") or ""),
        source_manifest_hash=source_hash,
    )
    sample = ActionBranchSample(
        identity=identity,
        ego_history=ego_history,
        ego_history_mask=ego_mask,
        actor_history=actor_history,
        actor_history_mask=actor_history_mask,
        road=road,
        road_mask=road_mask,
        candidates=candidates,
        candidate_mask=candidate_mask,
        actor_future=actor_future,
        actor_future_mask=actor_future_mask,
        outcomes=outcomes,
        outcome_mask=outcome_mask,
        rank_target=rank_target,
        rank_mask=rank_mask,
        rank_weight=rank_weight,
        tie_target=tie_target,
        comparable=comparable,
        unavailable_reasons=unavailable_reasons,
        audit={
            "pair_label": pair_label,
            "both_bad": bool(oracle.get("both_bad", False)),
            "source_attempt_dir": str(attempt_dir),
            "actor_ids": list(actor_ids),
            "reactive_actor_present": bool(
                _read_json(attempt_dir / "branch-0" / "branch_summary.json").get(
                    "reactive_actor_present", False
                )
            ),
            "oracle_labels_separate": True,
            "namespace": str(
                raw_artifact.get("namespace")
                or manifest.get("namespace")
                or pair_report.get("namespace")
                or "r3_teacher_development"
            ),
            "r2_checkpoint_sha256": str(
                raw_artifact.get("r2_checkpoint_sha256")
                or manifest.get("r2_checkpoint_sha256")
                or pair_report.get("checkpoint_sha256")
                or ""
            ).lower(),
            "condition_variant": condition_variant,
            "repeat_group": repeat_group,
            "aa_noise_identity": aa_noise_identity,
            "actor_controller_kind": str(
                pair_report.get("actor_controller_kind")
                or manifest.get("actor_controller_kind")
                or "fixed"
            ),
            "artifact_schema": str(raw_artifact.get("schema_version") or ""),
            "artifact_file_sha256": file_sha256(artifact_path),
            "branch_summary_sha256": {
                str(index): file_sha256(attempt_dir / f"branch-{index}" / "branch_summary.json")
                for index in range(K)
                if (attempt_dir / f"branch-{index}" / "branch_summary.json").is_file()
            },
            "branch_future_trace_sha256": {
                str(index): file_sha256(
                    attempt_dir / f"branch-{index}" / "oracle" / "actor_future_trace.jsonl"
                )
                for index in range(K)
                if (
                    attempt_dir / f"branch-{index}" / "oracle" / "actor_future_trace.jsonl"
                ).is_file()
            },
            "observable_scene_sha256": {
                str(index): file_sha256(attempt_dir / f"branch-{index}" / "observable_scene_t0.json")
                for index in range(K)
                if (attempt_dir / f"branch-{index}" / "observable_scene_t0.json").is_file()
            },
            "v4_token_raw_content_hash": str(
                (raw_artifact.get("bundle") or {})
                .get("observation_identity", {})
                .get("v4_token_raw_content_hash", "")
                if isinstance(raw_artifact.get("bundle"), Mapping)
                else ""
            ),
            "formal_trace_path": str(
                attempt_dir / "anchor" / "formal_trace_v4.json"
            )
            if (attempt_dir / "anchor" / "formal_trace_v4.json").is_file()
            else "",
            "semantic_rescue_count": int(
                pair_report.get("semantic_rescue_count", 0)
            ),
            "scenario_family_runtime_use": int(
                pair_report.get("scenario_family_runtime_use", 0)
            ),
        },
    )
    sample.validate()
    return sample


def assign_group_splits(
    samples: Sequence[ActionBranchSample],
    *,
    seed: int = 3407,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict[str, str]:
    if not 0.0 < train_fraction < 1.0:
        raise WorldContractError("train_fraction must be in (0,1)")
    if not 0.0 <= val_fraction < 1.0 or train_fraction + val_fraction >= 1.0:
        raise WorldContractError("invalid val_fraction")
    group_to_split: dict[str, str] = {}
    for group in sorted({sample.identity.group_key for sample in samples}):
        digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < train_fraction:
            split = "train"
        elif value < train_fraction + val_fraction:
            split = "val"
        else:
            split = "test"
        group_to_split[group] = split
    return {sample.identity.sample_id: group_to_split[sample.identity.group_key] for sample in samples}


def quality_report(
    samples: Sequence[ActionBranchSample],
    split_by_sample: Mapping[str, str],
    *,
    min_comparable: int = 256,
    min_decisive: int = 64,
    min_wins_per_slot: int = 16,
    min_future_coverage: float = 0.95,
    min_test_samples: int = 0,
    min_test_dual: int = 0,
    min_test_decisive: int = 0,
    min_test_wins_per_slot: int = 0,
) -> dict[str, Any]:
    for sample in samples:
        sample.validate()
    sample_ids = [sample.identity.sample_id for sample in samples]
    duplicate_count = len(sample_ids) - len(set(sample_ids))
    group_splits: dict[str, set[str]] = {}
    for sample in samples:
        group_splits.setdefault(sample.identity.group_key, set()).add(
            split_by_sample[sample.identity.sample_id]
        )
    group_overlap = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    comparable = [sample for sample in samples if sample.comparable]
    decisive = [sample for sample in comparable if sample.rank_mask and not sample.tie_target]
    wins0 = sum(sample.rank_target > 0 for sample in decisive)
    wins1 = sum(sample.rank_target < 0 for sample in decisive)
    valid_slots = sum(int(sample.actor_future_mask.sum()) for sample in samples)
    expected_slots = sum(
        int(sample.outcome_mask.sum()) * max(1, int(sample.actor_history_mask[:, -1].sum())) * T
        for sample in samples
    )
    coverage = valid_slots / max(1, expected_slots)
    split_quality: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        subset = [
            sample
            for sample in samples
            if split_by_sample[sample.identity.sample_id] == split
        ]
        subset_comparable = [sample for sample in subset if sample.comparable]
        subset_decisive = [
            sample
            for sample in subset_comparable
            if sample.rank_mask and not sample.tie_target
        ]
        split_quality[split] = {
            "samples": len(subset),
            "dual": sum(int(sample.candidate_mask.sum()) == 2 for sample in subset),
            "comparable": len(subset_comparable),
            "decisive": len(subset_decisive),
            "winner_slot_0": sum(sample.rank_target > 0 for sample in subset_decisive),
            "winner_slot_1": sum(sample.rank_target < 0 for sample in subset_decisive),
        }
    hard = {
        "schema_validation_100pct": True,
        "finite_values_100pct": True,
        "identity_binding_100pct": True,
        "runtime_oracle_namespace_leakage": 0,
        "split_group_overlap": len(group_overlap),
        "duplicate_sample_identity": duplicate_count,
        "comparable_minimum": len(comparable) >= min_comparable,
        "decisive_minimum": len(decisive) >= min_decisive,
        "winner_balance_minimum": min(wins0, wins1) >= min_wins_per_slot,
        "future_coverage_minimum": coverage >= min_future_coverage,
        "test_sample_minimum": split_quality["test"]["samples"] >= min_test_samples,
        "test_dual_minimum": split_quality["test"]["dual"] >= min_test_dual,
        "test_decisive_minimum": split_quality["test"]["decisive"] >= min_test_decisive,
        "test_winner_balance_minimum": min(
            split_quality["test"]["winner_slot_0"],
            split_quality["test"]["winner_slot_1"],
        )
        >= min_test_wins_per_slot,
    }
    return {
        "schema_version": QUALITY_REPORT_SCHEMA,
        "sample_count": len(samples),
        "comparable_count": len(comparable),
        "decisive_count": len(decisive),
        "tie_count": sum(sample.tie_target for sample in comparable),
        "winner_slot_0": wins0,
        "winner_slot_1": wins1,
        "future_valid_slots": valid_slots,
        "future_expected_slots": expected_slots,
        "future_coverage": coverage,
        "family_counts": {
            family: sum(sample.identity.family == family for sample in samples)
            for family in sorted({sample.identity.family for sample in samples})
        },
        "split_counts": {
            split: sum(value == split for value in split_by_sample.values())
            for split in ("train", "val", "test")
        },
        "split_quality": split_quality,
        "group_overlap": group_overlap,
        "hard_gates": hard,
        "all_hard_gates_pass": all(
            value is True
            if isinstance(value, bool)
            else value == 0
            for value in hard.values()
        ),
    }


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def write_dataset(
    samples: Sequence[ActionBranchSample],
    root: Path,
    *,
    split_by_sample: Mapping[str, str] | None = None,
    shard_size: int = 128,
    quality_thresholds: Mapping[str, Any] | None = None,
    namespace: str = "",
    r2_checkpoint_sha256: str = "",
) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise WorldContractError(f"dataset root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "shards").mkdir()
    ordered = sorted(samples, key=lambda x: x.identity.sample_id)
    if split_by_sample is None:
        split_by_sample = assign_group_splits(ordered)
    missing = {x.identity.sample_id for x in ordered}.difference(split_by_sample)
    if missing:
        raise WorldContractError(f"split missing samples: {sorted(missing)}")

    namespace = str(namespace or "")
    r2_checkpoint_sha256 = str(r2_checkpoint_sha256 or "").lower()
    if namespace:
        if namespace in {"r3_r2_blind_holdout", "r2_blind", "formal_blind"}:
            raise WorldContractError("R2 blind namespace is holdout-only, not a training dataset")
        if len(r2_checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in r2_checkpoint_sha256
        ):
            raise WorldContractError("V1 dataset requires a 64-hex R2 checkpoint hash")
    source_path = root / "source_index.jsonl"
    shard_records: list[dict[str, Any]] = []
    source_lines: list[str] = []
    for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
        chunk = ordered[start : start + shard_size]
        for sample in chunk:
            sample.validate()
        arrays = {
            name: np.stack([getattr(sample, name) for sample in chunk])
            for name in (
                "ego_history",
                "ego_history_mask",
                "actor_history",
                "actor_history_mask",
                "road",
                "road_mask",
                "candidates",
                "candidate_mask",
                "actor_future",
                "actor_future_mask",
                "outcomes",
                "outcome_mask",
            )
        }
        arrays.update(
            {
                "rank_target": np.asarray([x.rank_target for x in chunk], dtype=np.float32),
                "rank_mask": np.asarray([x.rank_mask for x in chunk], dtype=np.bool_),
                "rank_weight": np.asarray(
                    [x.rank_weight for x in chunk], dtype=np.float32
                ),
                "tie_target": np.asarray([x.tie_target for x in chunk], dtype=np.bool_),
                "comparable": np.asarray([x.comparable for x in chunk], dtype=np.bool_),
            }
        )
        shard_path = root / "shards" / f"shard_{shard_index:05d}.npz"
        np.savez_compressed(shard_path, **arrays)
        shard_records.append(
            {
                "path": str(shard_path.relative_to(root)),
                "sha256": file_sha256(shard_path),
                "sample_count": len(chunk),
                "start_index": start,
            }
        )
        for local_index, sample in enumerate(chunk):
            record = sample.source_record(
                schema_version=SCHEMA_VERSION_V1 if namespace else SCHEMA_VERSION
            )
            record.update(
                {
                    "shard": shard_records[-1]["path"],
                    "local_index": local_index,
                    "split": split_by_sample[sample.identity.sample_id],
                    "namespace": namespace or "r3_teacher_development",
                    "campaign_namespace": namespace or "r3_teacher_development",
                    "r2_checkpoint_sha256": r2_checkpoint_sha256,
                }
            )
            source_lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
    source_path.write_text("\n".join(source_lines) + ("\n" if source_lines else ""), encoding="utf-8")
    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_V1 if namespace else SPLIT_MANIFEST_SCHEMA,
        "group_key_contract": "map|family|scenario_lineage",
        "assignments": dict(sorted(split_by_sample.items())),
        "assignment_hash": content_hash(dict(sorted(split_by_sample.items()))),
        "namespace": namespace or "r3_teacher_development",
        "campaign_namespace": namespace or "r3_teacher_development",
        "r2_checkpoint_sha256": r2_checkpoint_sha256,
    }
    _exclusive_json(root / "split_manifest.json", split_manifest)
    report = quality_report(
        ordered,
        split_by_sample,
        **dict(quality_thresholds or {}),
    )
    report["namespace"] = namespace or "r3_teacher_development"
    report["r2_checkpoint_sha256"] = r2_checkpoint_sha256
    _exclusive_json(root / "quality_report.json", report)
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_V1 if namespace else DATASET_MANIFEST_SCHEMA,
        "sample_schema_version": SCHEMA_VERSION_V1 if namespace else SCHEMA_VERSION,
        "sample_count": len(ordered),
        "outcome_names": list(OUTCOME_NAMES),
        "namespace": namespace or "r3_teacher_development",
        "campaign_namespace": namespace or "r3_teacher_development",
        "r2_checkpoint_sha256": r2_checkpoint_sha256,
        "shards": shard_records,
        "source_index_sha256": file_sha256(source_path),
        "split_manifest_sha256": file_sha256(root / "split_manifest.json"),
        "quality_report_sha256": file_sha256(root / "quality_report.json"),
        "frozen": True,
    }
    manifest["dataset_content_hash"] = content_hash(manifest)
    _exclusive_json(root / "dataset_manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class DatasetRow:
    sample: ActionBranchSample
    split: str


class ActionBranchDataset:
    def __init__(self, root: Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        self.manifest = _read_json(self.root / "dataset_manifest.json")
        if self.manifest.get("schema_version") not in {
            DATASET_MANIFEST_SCHEMA,
            DATASET_MANIFEST_SCHEMA_V1,
        }:
            raise WorldContractError("unsupported dataset manifest")
        if verify_hashes:
            for shard in self.manifest["shards"]:
                path = self.root / shard["path"]
                if file_sha256(path) != shard["sha256"]:
                    raise WorldContractError(f"shard hash mismatch: {path}")
            for field, filename in (
                ("source_index_sha256", "source_index.jsonl"),
                ("split_manifest_sha256", "split_manifest.json"),
                ("quality_report_sha256", "quality_report.json"),
            ):
                if file_sha256(self.root / filename) != self.manifest[field]:
                    raise WorldContractError(f"manifest hash mismatch: {filename}")
        self._records: list[dict[str, Any]] = []
        with (self.root / "source_index.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self._records.append(json.loads(line))
        self._shards: dict[str, Mapping[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._records)

    def _shard(self, relative: str) -> Mapping[str, np.ndarray]:
        if relative not in self._shards:
            loaded = np.load(self.root / relative, allow_pickle=False)
            self._shards[relative] = {name: loaded[name] for name in loaded.files}
        return self._shards[relative]

    def __getitem__(self, index: int) -> DatasetRow:
        record = self._records[index]
        arrays = self._shard(record["shard"])
        i = int(record["local_index"])
        sample = ActionBranchSample(
            identity=SampleIdentity.from_dict(record["identity"]),
            ego_history=arrays["ego_history"][i],
            ego_history_mask=arrays["ego_history_mask"][i],
            actor_history=arrays["actor_history"][i],
            actor_history_mask=arrays["actor_history_mask"][i],
            road=arrays["road"][i],
            road_mask=arrays["road_mask"][i],
            candidates=arrays["candidates"][i],
            candidate_mask=arrays["candidate_mask"][i],
            actor_future=arrays["actor_future"][i],
            actor_future_mask=arrays["actor_future_mask"][i],
            outcomes=arrays["outcomes"][i],
            outcome_mask=arrays["outcome_mask"][i],
            rank_target=float(arrays["rank_target"][i]),
            rank_mask=bool(arrays["rank_mask"][i]),
            rank_weight=float(arrays["rank_weight"][i]),
            tie_target=bool(arrays["tie_target"][i]),
            comparable=bool(arrays["comparable"][i]),
            unavailable_reasons=tuple(record["unavailable_reasons"]),
            audit=record.get("audit", {}),
        )
        sample.validate()
        return DatasetRow(sample=sample, split=str(record["split"]))

    def iter_split(self, split: str) -> Iterator[ActionBranchSample]:
        for index, record in enumerate(self._records):
            if record["split"] == split:
                yield self[index].sample


class ActionBranchDatasetV1(ActionBranchDataset):
    """Strict reader for the final-head V1 namespace.

    The generic :class:`ActionBranchDataset` remains the backwards-compatible
    V0/V1 reader used by historical development evidence.
    """

    def __init__(self, root: Path, *, verify_hashes: bool = True) -> None:
        super().__init__(root, verify_hashes=verify_hashes)
        if self.manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA_V1:
            raise WorldContractError("ActionBranchDatasetV1 requires a V1 manifest")
        if str(self.manifest.get("sample_schema_version") or "") != SCHEMA_VERSION_V1:
            raise WorldContractError("ActionBranchDatasetV1 sample schema mismatch")
        if str(self.manifest.get("namespace") or "") != "r3_final_head_formal":
            raise WorldContractError("ActionBranchDatasetV1 requires final-head namespace")
        if str(self.manifest.get("campaign_namespace") or "") != "r3_final_head_formal":
            raise WorldContractError("ActionBranchDatasetV1 campaign namespace mismatch")
        bound_hash = str(self.manifest.get("r2_checkpoint_sha256") or "").lower()
        if len(bound_hash) != 64 or any(char not in "0123456789abcdef" for char in bound_hash):
            raise WorldContractError("ActionBranchDatasetV1 requires a bound R2 checkpoint hash")
        for index, record in enumerate(self._records):
            if str(record.get("schema_version") or "") != SCHEMA_VERSION_V1:
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} has a non-V1 schema"
                )
            if str(record.get("namespace") or "") != "r3_final_head_formal":
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} namespace mismatch"
                )
            if str(record.get("campaign_namespace") or "") != "r3_final_head_formal":
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} campaign namespace mismatch"
                )
            if str(record.get("r2_checkpoint_sha256") or "").lower() != bound_hash:
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} checkpoint binding mismatch"
                )
            repeat_group = str(record.get("audit", {}).get("repeat_group") or "")
            aa_identity = str(record.get("audit", {}).get("aa_noise_identity") or "").lower()
            if not repeat_group:
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} repeat_group missing"
                )
            if len(aa_identity) != 64 or any(char not in "0123456789abcdef" for char in aa_identity):
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} A/A identity is not 64-hex"
                )
            expected_aa_identity = content_hash(
                {
                    "namespace": "r3_aa_noise_probe",
                    "repeat_group": repeat_group,
                    "candidate_id": "v3_nominal_progress",
                }
            ).lower()
            if aa_identity != expected_aa_identity:
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} A/A identity binding mismatch"
                )
            candidate_mask = list(record.get("candidate_mask") or ())
            unavailable_mask = list(record.get("candidate_unavailable_mask") or ())
            if len(candidate_mask) != 2 or len(unavailable_mask) != 2:
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} candidate mask fields missing"
                )
            if any(
                int(bool(candidate_mask[pos])) + int(bool(unavailable_mask[pos])) != 1
                for pos in range(2)
            ):
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} candidate availability masks overlap"
                )
            for field in ("tie_mask", "both_bad_mask", "incomparable_mask"):
                if field not in record:
                    raise WorldContractError(
                        f"ActionBranchDatasetV1 sample {index} {field} missing"
                    )
            if bool(record["tie_mask"]) != bool(record.get("tie_target", False)):
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} tie mask mismatch"
                )
            if bool(record["both_bad_mask"]) != bool(
                record.get("audit", {}).get("both_bad", False)
            ):
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} BOTH_BAD mask mismatch"
                )
            if bool(record["incomparable_mask"]) == bool(record.get("comparable", False)):
                raise WorldContractError(
                    f"ActionBranchDatasetV1 sample {index} incomparable mask mismatch"
                )
