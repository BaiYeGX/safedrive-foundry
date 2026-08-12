"""Dataset and offline promotion gates for K2 V3 semantic heads."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from driving_vla.model.k2_v3_types import AlternativeKind
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    canonical_sha256,
)

DATASET_SCHEMA = "safedrive.k2_v3_semantic_dataset.v1"
EXPECTED_FULL_SPLIT_COUNTS = {"train": 252, "dev": 60, "test": 48}


class R2V3DatasetError(ValueError):
    pass


def _teacher_event_matches_slot(
    event: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> bool:
    expected_kind = AlternativeKind(str(slot["alternative_kind"]))
    actual_kind = AlternativeKind(
        str(event.get("alternative_slot_kind") or AlternativeKind.NONE.value)
    )
    expected_available = expected_kind is not AlternativeKind.NONE
    return bool(
        str(event.get("vla_version")) == "v3"
        and str(event.get("route_maneuver")) == str(slot["maneuver"])
        and actual_kind is expected_kind
        and bool(event.get("alternative_slot_available")) == expected_available
        and bool(event.get("accepted"))
    )


def build_v3_dataset_row(
    *,
    slot: Mapping[str, Any],
    route_context: Mapping[str, Any],
    event: Mapping[str, Any],
    source_evidence_dir: str = "",
) -> dict[str, Any]:
    """Convert one frozen teacher anchor into the learned-head row contract."""
    if not _teacher_event_matches_slot(event, slot):
        raise R2V3DatasetError(
            f"{slot.get('slot_id')}: event does not match frozen semantic slot"
        )
    # The route file is the immutable coarse/topology anchor.  Traffic control
    # is a current observable and may be UNKNOWN when that anchor was authored
    # (for example, before the ego reaches the trigger volume).  Bind the
    # teacher's first accepted observation into the training context so a
    # red-light/stop-sign sample is not indistinguishable from CLEAR.  The
    # route hash is intentionally left untouched; only the dynamic topology
    # fields are refreshed from the same-forward event.
    context_mapping = dict(route_context)
    observed_signal = str(event.get("traffic_signal_state") or "").upper()
    if observed_signal in {"GREEN", "YELLOW", "RED", "STOP_SIGN"}:
        context_mapping["traffic_signal_state"] = observed_signal
        context_mapping["stop_line_distance_m"] = (
            None
            if event.get("stop_line_distance_m") is None
            else float(event["stop_line_distance_m"])
        )
        context_mapping.pop("topology_hash", None)
    context = RouteContextV3.from_mapping(context_mapping)
    if context.maneuver.value != str(slot["maneuver"]):
        raise R2V3DatasetError(
            f"{slot.get('slot_id')}: route maneuver mismatch"
        )
    if str(event.get("route_hash") or "") != context.route_hash:
        raise R2V3DatasetError(
            f"{slot.get('slot_id')}: route hash mismatch"
        )
    feature = event.get("driving_feature")
    if not isinstance(feature, Sequence) or isinstance(feature, (str, bytes)):
        raise R2V3DatasetError(
            f"{slot.get('slot_id')}: same-forward driving feature missing"
        )
    feature_values = [float(value) for value in feature]
    if len(feature_values) != 64 or not any(
        abs(value) > 1.0e-12 for value in feature_values
    ):
        raise R2V3DatasetError(
            f"{slot.get('slot_id')}: driving feature must be nonzero mean64"
        )
    native_path = event.get("raw_path_map_xy")
    if not isinstance(native_path, Sequence) or len(native_path) < 2:
        raise R2V3DatasetError(
            f"{slot.get('slot_id')}: native path missing"
        )
    alternative_kind = AlternativeKind(
        str(event["alternative_slot_kind"])
    )
    target_side = TargetLaneSide(
        str(
            event.get("alternative_slot_target_lane_side")
            or TargetLaneSide.NONE.value
        )
    )
    if alternative_kind is AlternativeKind.NONE:
        target_side = TargetLaneSide.NONE
    speed_samples = [
        float(value) for value in event.get("vla_speed_samples_mps") or ()
    ]
    base_speed_mps = (
        max(speed_samples)
        if speed_samples
        else max(0.0, float(event.get("desired_speed_mps") or 0.0))
    )
    ego_value = event.get("resolved_vla_input_speed_mps")
    if ego_value is None:
        ego_value = dict(event.get("ego") or {}).get("speed_mps", 0.0)
    candidate_valid = dict(event.get("candidate_valid") or {})
    alternative_valid = bool(
        candidate_valid.get("v3_alternative", False)
    )
    nominal_valid = bool(
        candidate_valid.get("v3_nominal_progress", False)
    )
    row_body = {
        "sample_id": str(slot["slot_id"]),
        "lineage_id": str(slot["lineage_id"]),
        "split": str(slot["split"]),
        "map_name": str(slot["map_name"]),
        "template_id": str(slot["template_id"]),
        "family": str(slot["family"]),
        "condition": str(slot["condition"]),
        "seed_id": str(slot["seed_id"]),
        "route_fixture_id": str(slot["route_fixture_id"]),
        "actor_script_id": str(slot["actor_script_id"]),
        "route_context": context.to_dict(),
        "native_path_xy": [
            [float(point[0]), float(point[1])] for point in native_path
        ],
        "ego_v": max(0.0, float(ego_value)),
        "base_speed_mps": max(0.0, float(base_speed_mps)),
        "driving_feature": feature_values,
        "driving_feature_hash": str(
            event.get("driving_feature_hash") or ""
        ),
        "driving_feature_raw_hash": str(
            event.get("driving_feature_raw_hash") or ""
        ),
        "observable_scene": dict(
            event.get("observable_scene_v1") or {}
        ),
        "alternative_kind": alternative_kind.value,
        "target_lane_side": target_side.value,
        "alternative_available": bool(
            event["alternative_slot_available"]
        ),
        "availability_reason": str(
            event.get("alternative_slot_availability_reason") or ""
        ),
        "avoid_offset_m": (
            0.6
            if alternative_kind is AlternativeKind.SPATIAL_AVOID
            else 0.5
        ),
        "temporal_speed_scale": 0.0,
        "departure_start": 0.08,
        "departure_end": 0.42,
        "rejoin_start": 0.58,
        "rejoin_end": 0.92,
        "teacher_guard_accepted": (
            alternative_valid
            if bool(event["alternative_slot_available"])
            else nominal_valid
        ),
        "teacher_path_manager_accepted": bool(event.get("accepted")),
        "source_camera_frame": int(event.get("camera_frame") or 0),
        "source_evidence_dir": str(source_evidence_dir),
    }
    return {
        **row_body,
        "sample_content_hash": canonical_sha256(row_body),
    }


def build_v3_dataset_from_evidence(
    *,
    manifest: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one immutable row per manifest slot from completed teacher runs."""
    slots = list(manifest.get("slots") or ())
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for slot in slots:
        slot_id = str(slot["slot_id"])
        evidence = evidence_root / slot_id
        try:
            route_context = json.loads(
                (evidence / "route_context_v3.json").read_text(
                    encoding="utf-8"
                )
            )
            event_files = sorted(
                path
                for path in evidence.glob("vla_events_*.json")
                if "partial" not in path.name
            )
            if len(event_files) != 1:
                raise R2V3DatasetError(
                    f"expected one completed VLA event file, got {len(event_files)}"
                )
            events = json.loads(event_files[0].read_text(encoding="utf-8"))
            matches = [
                event
                for event in events
                if _teacher_event_matches_slot(event, slot)
            ]
            if not matches:
                raise R2V3DatasetError(
                    "no accepted teacher anchor matching frozen slot"
                )
            rows.append(
                build_v3_dataset_row(
                    slot=slot,
                    route_context=route_context,
                    event=matches[0],
                    source_evidence_dir=str(evidence.as_posix()),
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "slot_id": slot_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if failures:
        sample = "; ".join(
            f"{row['slot_id']}={row['error']}" for row in failures[:5]
        )
        raise R2V3DatasetError(
            f"dataset evidence incomplete: {len(failures)} slots; {sample}"
        )
    audit = validate_v3_dataset_rows(rows)
    audit.update(
        {
            "manifest_hash": str(manifest.get("manifest_hash") or ""),
            "evidence_root": str(evidence_root.as_posix()),
            "row_content_hash": canonical_sha256(
                [row["sample_content_hash"] for row in rows]
            ),
        }
    )
    return rows, audit


def validate_v3_dataset_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either the frozen 144 pilot or the complete 360 campaign."""
    phase = str(manifest.get("phase") or "")
    slots = list(manifest.get("slots") or ())
    if phase == "calibration":
        expected_count = 360
        expected_splits = EXPECTED_FULL_SPLIT_COUNTS
    elif phase == "calibration_pilot":
        expected_count = 144
        expected_splits = Counter(
            str(slot.get("split") or "") for slot in slots
        )
        if str(manifest.get("source_calibration_manifest_hash") or "") == "":
            raise R2V3DatasetError(
                "pilot manifest must bind the full calibration manifest"
            )
    else:
        raise R2V3DatasetError(f"unsupported dataset phase: {phase}")
    if len(slots) != expected_count:
        raise R2V3DatasetError(
            f"{phase} requires {expected_count} slots, got {len(slots)}"
        )
    slot_ids = [str(slot.get("slot_id") or "") for slot in slots]
    if "" in slot_ids or len(set(slot_ids)) != expected_count:
        raise R2V3DatasetError("dataset manifest slot ids must be unique")
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    route_splits: dict[str, set[str]] = defaultdict(set)
    for slot in slots:
        split = str(slot.get("split") or "")
        if split not in {"train", "dev", "test"}:
            raise R2V3DatasetError(f"invalid manifest split: {split}")
        lineage_splits[str(slot.get("lineage_id") or "")].add(split)
        route_splits[str(slot.get("route_fixture_id") or "")].add(split)
    if any(len(values) != 1 for values in lineage_splits.values()):
        raise R2V3DatasetError("manifest lineage overlap across splits")
    if any(len(values) != 1 for values in route_splits.values()):
        raise R2V3DatasetError("manifest route overlap across splits")
    split_counts = Counter(str(slot["split"]) for slot in slots)
    if phase == "calibration" and dict(split_counts) != expected_splits:
        raise R2V3DatasetError(
            f"full calibration split mismatch: {dict(split_counts)}"
        )
    return {
        "phase": phase,
        "slot_count": expected_count,
        "slot_ids": set(slot_ids),
        "split_counts": dict(split_counts),
        "lineage_count": len(lineage_splits),
        "route_count": len(route_splits),
    }


def validate_v3_dataset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_all_splits: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise R2V3DatasetError("K2 V3 dataset is empty")
    required = {
        "sample_id",
        "lineage_id",
        "split",
        "route_fixture_id",
        "route_context",
        "native_path_xy",
        "ego_v",
        "base_speed_mps",
        "alternative_kind",
        "target_lane_side",
        "alternative_available",
    }
    sample_ids: set[str] = set()
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    route_splits: dict[str, set[str]] = defaultdict(set)
    route_hash_splits: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise R2V3DatasetError(
                f"row {index} missing fields: {','.join(missing)}"
            )
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise R2V3DatasetError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        split = str(row["split"])
        if split not in {"train", "dev", "test"}:
            raise R2V3DatasetError(f"invalid split: {split}")
        lineage_splits[str(row["lineage_id"])].add(split)
        route_splits[str(row["route_fixture_id"])].add(split)
        try:
            AlternativeKind(str(row["alternative_kind"]))
            TargetLaneSide(str(row["target_lane_side"]))
            context = RouteContextV3.from_mapping(row["route_context"])
        except (KeyError, TypeError, ValueError) as exc:
            raise R2V3DatasetError(
                f"row {index} has invalid semantic/navigation label"
            ) from exc
        route_hash_splits[context.route_hash].add(split)
        path = row["native_path_xy"]
        if not isinstance(path, Sequence) or len(path) < 2:
            raise R2V3DatasetError(f"row {index} native path too short")
    leaking_lineages = sorted(
        lineage for lineage, splits in lineage_splits.items() if len(splits) != 1
    )
    leaking_routes = sorted(
        route for route, splits in route_splits.items() if len(splits) != 1
    )
    leaking_route_hashes = sorted(
        route_hash
        for route_hash, splits in route_hash_splits.items()
        if len(splits) != 1
    )
    if leaking_lineages:
        raise R2V3DatasetError(
            "lineage overlap across splits: " + ",".join(leaking_lineages[:5])
        )
    if leaking_routes:
        raise R2V3DatasetError(
            "route overlap across splits: " + ",".join(leaking_routes[:5])
        )
    if leaking_route_hashes:
        raise R2V3DatasetError(
            "actual route hash overlap across splits: "
            + ",".join(leaking_route_hashes[:5])
        )
    split_counts = Counter(str(row["split"]) for row in rows)
    if require_all_splits and set(split_counts) != {"train", "dev", "test"}:
        raise R2V3DatasetError("dataset must contain train/dev/test")
    return {
        "schema_version": DATASET_SCHEMA,
        "samples": len(rows),
        "lineages": len(lineage_splits),
        "routes": len(route_splits),
        "actual_route_hashes": len(route_hash_splits),
        "split_counts": dict(split_counts),
        "lineage_overlap": 0,
        "route_overlap": 0,
        "actual_route_hash_overlap": 0,
    }


@dataclass(frozen=True)
class OfflinePromotionThresholdsV3:
    semantic_accuracy_min: float = 0.90
    direction_accuracy_min: float = 0.95
    availability_recall_min: float = 0.80
    availability_specificity_min: float = 0.80
    none_closure_min: float = 1.00
    route_maneuver_consistency_min: float = 1.00
    legal_route_target_rate_min: float = 1.00
    guard_mpc_acceptance_min: float = 0.90


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_offline_prediction_records_v3(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: OfflinePromotionThresholdsV3 = OfflinePromotionThresholdsV3(),
) -> dict[str, Any]:
    if not records:
        raise R2V3DatasetError("offline prediction records are empty")
    semantic_ok = 0
    direction_ok = 0
    direction_total = 0
    true_positive = false_negative = true_negative = false_positive = 0
    none_ok = none_total = 0
    route_ok = legal_ok = guard_mpc_ok = 0
    for row in records:
        target_kind = str(row["target_kind"])
        predicted_kind = str(row["predicted_kind"])
        target_side = str(row["target_side"])
        predicted_side = str(row["predicted_side"])
        target_available = bool(row["target_available"])
        predicted_available = bool(row["predicted_available"])
        semantic_ok += predicted_kind == target_kind
        if target_side in {"LEFT", "RIGHT"}:
            direction_total += 1
            direction_ok += predicted_side == target_side
        if target_available and predicted_available:
            true_positive += 1
        elif target_available:
            false_negative += 1
        elif predicted_available:
            false_positive += 1
        else:
            true_negative += 1
        if target_kind == AlternativeKind.NONE.value:
            none_total += 1
            none_ok += (
                predicted_kind == AlternativeKind.NONE.value
                and not predicted_available
            )
        route_ok += (
            str(row["input_route_maneuver"])
            == str(row["output_route_maneuver"])
        )
        legal_ok += bool(row.get("legal_route_target"))
        guard_mpc_ok += bool(row.get("guard_accepted")) and bool(
            row.get("mpc_accepted")
        )
    count = len(records)
    metrics = {
        "sample_count": count,
        "semantic_accuracy": semantic_ok / count,
        "direction_accuracy": _safe_rate(direction_ok, direction_total),
        "direction_denominator": direction_total,
        "availability_recall": _safe_rate(
            true_positive, true_positive + false_negative
        ),
        "availability_specificity": _safe_rate(
            true_negative, true_negative + false_positive
        ),
        "none_closure_rate": _safe_rate(none_ok, none_total),
        "none_denominator": none_total,
        "route_maneuver_consistency": route_ok / count,
        "legal_route_target_rate": legal_ok / count,
        "guard_mpc_acceptance": guard_mpc_ok / count,
    }
    gates = {
        "semantic_accuracy": metrics["semantic_accuracy"]
        >= thresholds.semantic_accuracy_min,
        "direction_accuracy": direction_total > 0
        and metrics["direction_accuracy"] >= thresholds.direction_accuracy_min,
        "availability_recall": metrics["availability_recall"]
        >= thresholds.availability_recall_min,
        "availability_specificity": metrics["availability_specificity"]
        >= thresholds.availability_specificity_min,
        "none_closure": none_total > 0
        and metrics["none_closure_rate"] >= thresholds.none_closure_min,
        "route_maneuver_consistency": metrics["route_maneuver_consistency"]
        >= thresholds.route_maneuver_consistency_min,
        "legal_route_target_rate": metrics["legal_route_target_rate"]
        >= thresholds.legal_route_target_rate_min,
        "guard_mpc_acceptance": metrics["guard_mpc_acceptance"]
        >= thresholds.guard_mpc_acceptance_min,
    }
    return {
        "schema_version": "safedrive.k2_v3_offline_gate.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "thresholds": dict(thresholds.__dict__),
    }


__all__ = [
    "DATASET_SCHEMA",
    "OfflinePromotionThresholdsV3",
    "R2V3DatasetError",
    "build_v3_dataset_from_evidence",
    "build_v3_dataset_row",
    "evaluate_offline_prediction_records_v3",
    "validate_v3_dataset_manifest",
    "validate_v3_dataset_rows",
]
