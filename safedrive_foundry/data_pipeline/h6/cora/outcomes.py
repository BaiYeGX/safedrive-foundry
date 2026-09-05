"""Public C2 outcome heads derived from immutable CARLA branch artifacts.

The live collector stored a compact set of loss-facing aliases before the
final public head names were frozen.  This offline-only module expands those
aliases together with the immutable timeline/event shards.  It never changes
or re-runs a recorded rollout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_pipeline.h2.contracts import stable_sha256

from .store import CoraDataStore


PUBLIC_LABEL_SCHEMA = "safedrive.cora.outcome_labels.v2"
PUBLIC_DERIVATION_VERSION = "cora-c2-public-labeler-v2"
PUBLIC_OUTCOME_HEADS = (
    "route_progress_m",
    "route_completed",
    "local_goal_completed",
    "collision_count",
    "max_collision_impulse",
    "collision_other_actor_id",
    "red_light_violation",
    "off_corridor_duration_s",
    "max_corridor_deviation_m",
    "minimum_ttc_s",
    "minimum_clearance_m",
    "acceleration_rms_mps2",
    "acceleration_p95_mps2",
    "jerk_rms_mps3",
    "jerk_p95_mps3",
    "lateral_acceleration_rms_mps2",
    "lateral_acceleration_p95_mps2",
    "controller_deadline_misses",
    "guard_verdict",
    "safety_decision_kind",
    "repair_attempted",
    "repair_success",
    "repair_mode",
    "executable",
    "minimal_risk",
    "would_require_cross_candidate_fallback",
    "ticks_executed",
    "terminal_reason",
    "cleanup_status",
)


def _head(value: Any, unit: str, *, valid: bool = True) -> dict[str, Any]:
    if not valid:
        value = None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("cora_public_outcome_non_finite")
    return {
        "value": value,
        "unit": unit,
        "valid": bool(valid),
        "derivation_version": PUBLIC_DERIVATION_VERSION,
    }


def _legacy(branch: Mapping[str, Any], name: str) -> tuple[Any, bool]:
    item = branch.get("heads", {}).get(name)
    if not isinstance(item, Mapping) or not bool(item.get("valid", False)):
        return None, False
    return item.get("value"), True


def _rms(values: Sequence[float]) -> float | None:
    return None if not values else math.sqrt(sum(value * value for value in values) / len(values))


def _p95_abs(values: Sequence[float]) -> float | None:
    """Return deterministic linear-interpolation P95 of absolute magnitudes."""

    if not values:
        return None
    ordered = sorted(abs(float(value)) for value in values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _kinematic_series(
    timeline: Sequence[Mapping[str, Any]], *, dt_s: float
) -> tuple[list[float], list[float], list[float]]:
    speeds = [float(row["speed_mps"]) for row in timeline]
    accelerations = [
        (speeds[index] - speeds[index - 1]) / dt_s
        for index in range(1, len(speeds))
    ]
    jerks = [
        (accelerations[index] - accelerations[index - 1]) / dt_s
        for index in range(1, len(accelerations))
    ]
    lateral = [float(row.get("lateral_acceleration_mps2", 0.0)) for row in timeline]
    return accelerations, jerks, lateral


def derive_public_outcome_heads(
    branch: Mapping[str, Any],
    *,
    timeline: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    corridor_half_width_m: float | None,
    dt_s: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Derive every public C2 head from one immutable branch.

    Missing quantities stay ``value=null, valid=false``.  In particular,
    repair success is undefined when Safety did not actually attempt repair.
    """

    if dt_s <= 0.0 or not math.isfinite(dt_s):
        raise ValueError("cora_public_outcome_dt")
    timeline_valid = bool(timeline)
    progress_legacy, progress_valid = _legacy(branch, "progress")
    completion, completion_valid = _legacy(branch, "completion")
    red_light, red_light_valid = _legacy(branch, "red_light")
    minimum_ttc, minimum_ttc_valid = _legacy(branch, "minimum_ttc")
    minimum_clearance, minimum_clearance_valid = _legacy(branch, "minimum_clearance")
    acceleration_rms, acceleration_rms_valid = _legacy(branch, "acceleration_rms")
    jerk_rms, jerk_rms_valid = _legacy(branch, "jerk_rms")
    lateral_rms, lateral_rms_valid = _legacy(branch, "lateral_acceleration_rms")
    executable, executable_valid = _legacy(branch, "executable")

    progress = (
        float(timeline[-1]["route_progress_m"])
        if timeline_valid and "route_progress_m" in timeline[-1]
        else progress_legacy
    )
    progress_is_valid = timeline_valid or progress_valid

    collisions = [row for row in events if str(row.get("event_type")) == "collision"]
    impulses: list[tuple[float, Any]] = []
    for event in collisions:
        impulse = math.sqrt(
            float(event.get("impulse_x", 0.0) or 0.0) ** 2
            + float(event.get("impulse_y", 0.0) or 0.0) ** 2
            + float(event.get("impulse_z", 0.0) or 0.0) ** 2
        )
        impulses.append((impulse, event.get("other_actor_id")))
    largest = max(impulses, key=lambda item: item[0]) if impulses else None

    corridor_values = [
        abs(float(row["corridor_distance_m"]))
        for row in timeline
        if row.get("corridor_distance_m") is not None
    ]
    corridor_valid = bool(corridor_values) and corridor_half_width_m is not None
    off_duration = None
    if corridor_valid:
        off_duration = sum(
            value > float(corridor_half_width_m) for value in corridor_values
        ) * dt_s

    accelerations, jerks, lateral = _kinematic_series(timeline, dt_s=dt_s)
    repair_attempted = bool(branch.get("repair_attempted", False))
    repair_success = branch.get("repair_success")
    decision_kind = branch.get("decision_kind")
    repair_modes = {"QP": "LONGITUDINAL_QP", "RATO": "RATO_SCP"}

    heads = {
        "route_progress_m": _head(progress, "m", valid=progress_is_valid),
        "route_completed": _head(completion, "bool", valid=completion_valid),
        "local_goal_completed": _head(completion, "bool", valid=completion_valid),
        "collision_count": _head(len(collisions), "count", valid=timeline_valid),
        "max_collision_impulse": _head(
            0.0 if largest is None else largest[0], "N*s", valid=timeline_valid
        ),
        "collision_other_actor_id": _head(
            None if largest is None else largest[1],
            "actor_id",
            valid=largest is not None and largest[1] is not None,
        ),
        "red_light_violation": _head(red_light, "bool", valid=red_light_valid),
        "off_corridor_duration_s": _head(off_duration, "s", valid=corridor_valid),
        "max_corridor_deviation_m": _head(
            max(corridor_values) if corridor_values else None,
            "m",
            valid=bool(corridor_values),
        ),
        "minimum_ttc_s": _head(minimum_ttc, "s", valid=minimum_ttc_valid),
        "minimum_clearance_m": _head(
            minimum_clearance, "m", valid=minimum_clearance_valid
        ),
        "acceleration_rms_mps2": _head(
            acceleration_rms if acceleration_rms_valid else _rms(accelerations),
            "m/s^2",
            valid=acceleration_rms_valid or bool(accelerations),
        ),
        "acceleration_p95_mps2": _head(
            _p95_abs(accelerations), "m/s^2", valid=bool(accelerations)
        ),
        "jerk_rms_mps3": _head(
            jerk_rms if jerk_rms_valid else _rms(jerks),
            "m/s^3",
            valid=jerk_rms_valid or bool(jerks),
        ),
        "jerk_p95_mps3": _head(_p95_abs(jerks), "m/s^3", valid=bool(jerks)),
        "lateral_acceleration_rms_mps2": _head(
            lateral_rms if lateral_rms_valid else _rms(lateral),
            "m/s^2",
            valid=lateral_rms_valid or bool(lateral),
        ),
        "lateral_acceleration_p95_mps2": _head(
            _p95_abs(lateral), "m/s^2", valid=bool(lateral)
        ),
        "controller_deadline_misses": _head(
            sum(bool(row.get("deadline_miss")) for row in timeline),
            "count",
            valid=timeline_valid,
        ),
        "guard_verdict": _head(
            branch.get("guard_verdict"),
            "enum",
            valid=branch.get("guard_verdict") in {"PASS", "REVIEW", "REJECT"},
        ),
        "safety_decision_kind": _head(
            decision_kind, "enum", valid=decision_kind is not None
        ),
        "repair_attempted": _head(
            repair_attempted, "bool", valid=decision_kind is not None
        ),
        "repair_success": _head(
            repair_success,
            "bool",
            valid=repair_attempted and isinstance(repair_success, bool),
        ),
        "repair_mode": _head(
            repair_modes.get(str(decision_kind)),
            "enum",
            valid=repair_attempted and str(decision_kind) in repair_modes,
        ),
        "executable": _head(executable, "bool", valid=executable_valid),
        "minimal_risk": _head(
            str(decision_kind) in {"MINIMAL_RISK", "HARD_REJECT"},
            "bool",
            valid=decision_kind is not None,
        ),
        "would_require_cross_candidate_fallback": _head(
            bool(branch.get("would_require_cross_candidate_fallback", False)),
            "bool",
            valid=decision_kind is not None,
        ),
        "ticks_executed": _head(
            int(branch.get("ticks_executed", len(timeline))), "count", valid=True
        ),
        "terminal_reason": _head(
            branch.get("terminal_reason"),
            "enum",
            valid=branch.get("terminal_reason") is not None,
        ),
        "cleanup_status": _head(
            "COMPLETED" if bool(branch.get("cleanup_complete")) else "INCOMPLETE",
            "enum",
            valid=True,
        ),
    }
    validate_public_outcome_heads(heads)
    return heads


def validate_public_outcome_heads(heads: Mapping[str, Any], *, derivation_version: str = PUBLIC_DERIVATION_VERSION) -> None:
    if set(heads) != set(PUBLIC_OUTCOME_HEADS):
        missing = sorted(set(PUBLIC_OUTCOME_HEADS) - set(heads))
        extra = sorted(set(heads) - set(PUBLIC_OUTCOME_HEADS))
        raise ValueError(f"cora_public_outcome_keys:missing={missing}:extra={extra}")
    for name in PUBLIC_OUTCOME_HEADS:
        item = heads[name]
        if not isinstance(item, Mapping):
            raise ValueError(f"cora_public_outcome_mapping:{name}")
        if set(item) != {"value", "unit", "valid", "derivation_version"}:
            raise ValueError(f"cora_public_outcome_fields:{name}")
        if not item.get("unit") or item.get("derivation_version") != derivation_version:
            raise ValueError(f"cora_public_outcome_metadata:{name}")
        if not bool(item.get("valid")) and item.get("value") is not None:
            raise ValueError(f"cora_public_outcome_invalid_non_null:{name}")
        value = item.get("value")
        if bool(item.get("valid")) and isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"cora_public_outcome_non_finite:{name}")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment admission
        raise RuntimeError("cora_public_labels_require_pyarrow") from exc
    return pq.read_table(path).to_pylist()


def materialize_public_labels(store: CoraDataStore, *, dt_s: float = 0.05) -> dict[str, Any]:
    """Add immutable v2 public-label sidecars beside the original labels."""

    written = 0
    reused = 0
    root_count = 0
    for record in store.iter_roots():
        root_count += 1
        anchor_path = store.root / str(record.get("anchor_path", ""))
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        corridor_half_width_m = anchor.get("observable_snapshot", {}).get(
            "corridor_half_width_m"
        )
        for branch in record.get("branches", ()):
            paths = branch.get("artifact_paths", {})
            if "timeline" not in paths or "events" not in paths:
                raise ValueError(
                    f"cora_public_label_branch_artifacts:{record.get('root_id')}:{branch.get('proposal_id')}"
                )
            heads = derive_public_outcome_heads(
                branch,
                timeline=_read_parquet(store.root / str(paths["timeline"])),
                events=_read_parquet(store.root / str(paths["events"])),
                corridor_half_width_m=(
                    None if corridor_half_width_m is None else float(corridor_half_width_m)
                ),
                dt_s=dt_s,
            )
            payload = {
                "schema_version": PUBLIC_LABEL_SCHEMA,
                "root_id": record["root_id"],
                "proposal_id": branch["proposal_id"],
                "proposal_sha256": branch["proposal_sha256"],
                "source_label_path": paths.get("label"),
                "source_label_sha256": branch.get("artifact_sha256", {}).get("label"),
                "heads": heads,
            }
            payload["public_label_sha256"] = stable_sha256(payload)
            target = store.public_label_path(
                str(record["root_id"]), str(branch["proposal_sha256"])
            )
            existed = target.is_file()
            store.write_immutable_json(target, payload)
            reused += int(existed)
            written += int(not existed)
    if written:
        store.write_manifest()
    return {
        "schema_version": PUBLIC_LABEL_SCHEMA,
        "root_count": root_count,
        "written": written,
        "reused": reused,
        "label_count": written + reused,
    }


def read_public_label(store: CoraDataStore, root_id: str, proposal_sha256: str) -> dict[str, Any]:
    path = store.public_label_path(root_id, proposal_sha256)
    if not path.is_file():
        raise FileNotFoundError(f"cora_public_label_missing:{root_id}:{proposal_sha256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("public_label_sha256", None)
    if expected != stable_sha256(payload):
        raise ValueError(f"cora_public_label_self_hash:{root_id}:{proposal_sha256}")
    payload["public_label_sha256"] = expected
    if payload.get("schema_version") != PUBLIC_LABEL_SCHEMA:
        raise ValueError(f"cora_public_label_schema:{root_id}:{proposal_sha256}")
    if payload.get("root_id") != root_id or payload.get("proposal_sha256") != proposal_sha256:
        raise ValueError(f"cora_public_label_identity:{root_id}:{proposal_sha256}")
    validate_public_outcome_heads(payload.get("heads", {}))
    return payload


__all__ = [
    "PUBLIC_DERIVATION_VERSION",
    "PUBLIC_LABEL_SCHEMA",
    "PUBLIC_OUTCOME_HEADS",
    "derive_public_outcome_heads",
    "materialize_public_labels",
    "read_public_label",
    "validate_public_outcome_heads",
]
