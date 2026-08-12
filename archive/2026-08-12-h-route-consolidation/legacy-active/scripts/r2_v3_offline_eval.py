#!/usr/bin/env python3
"""Evaluate one learned K2 V3 head on frozen held-out semantic rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.executability_metrics import (  # noqa: E402
    evaluate_branch_executability,
)
from driving_vla.evaluation.r2_v3_dataset import (  # noqa: E402
    evaluate_offline_prediction_records_v3,
    validate_v3_dataset_rows,
)
from driving_vla.model.k2_v3_guard import attach_k2_v3_guard  # noqa: E402
from driving_vla.model.navigation_contract import RouteContextV3  # noqa: E402
from driving_vla.model.semantic_mode_heads import (  # noqa: E402
    SpatialSemanticHeadRuntimeV3,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("dataset row must be an object")
                rows.append(value)
    validate_v3_dataset_rows(rows)
    return rows


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--records-out", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = Path(args.data)
    checkpoint = Path(args.checkpoint)
    rows = [
        row
        for row in _load_jsonl(dataset)
        if str(row["split"]) == str(args.split)
    ]
    if not rows:
        raise ValueError(f"dataset has no {args.split} rows")
    runtime = SpatialSemanticHeadRuntimeV3(
        device=str(args.device),
        checkpoint_path=str(checkpoint),
    )
    records: list[dict] = []
    for row in rows:
        route = RouteContextV3.from_mapping(row["route_context"])
        prediction = runtime.predict(
            native_path_xy=row["native_path_xy"],
            route_context=route,
            ego_v=float(row["ego_v"]),
            base_speed_mps=float(row["base_speed_mps"]),
            driving_feature=row.get("driving_feature"),
            observable_scene=row.get("observable_scene"),
        )
        bundle = attach_k2_v3_guard(
            runtime.build_bundle(
                native_path_xy=row["native_path_xy"],
                route_context=route,
                ego_v=float(row["ego_v"]),
                base_speed_mps=float(row["base_speed_mps"]),
                driving_feature=row.get("driving_feature"),
                observable_scene=row.get("observable_scene"),
                observation_identity={
                    "frame_id": str(row["sample_id"]),
                    "feature_content_hash": str(
                        row.get("driving_feature_hash") or ""
                    ),
                    "raw_head_output_hash": "",
                },
                backbone_forward_id=f"offline:{row['sample_id']}",
                base_checkpoint_hash="simlingo-frozen-dataset",
                spatial_head_checkpoint_hash=_sha256(checkpoint),
            )
        )
        alternative = bundle.candidates[1]
        predicted_available = bool(alternative.available)
        selected_index = 1 if predicted_available else 0
        selected = bundle.candidates[selected_index]
        candidate_valid = dict(
            bundle.guard_metrics.get("candidate_valid") or {}
        )
        guard_accepted = bool(
            candidate_valid.get(selected.candidate_id, False)
        )
        executable = evaluate_branch_executability(
            path_xy=selected.spatial_path_xy,
            speed_samples_mps=selected.speed_samples_mps,
            ego_v=float(row["ego_v"]),
            path_manager_accepted=guard_accepted,
            t10_points_xy_yaw_v=selected.points_xy_yaw_v_a_kappa,
        )
        output_route = selected.route_maneuver.value
        route_consistent = output_route == route.maneuver.value
        legal_route_target = bool(
            route_consistent
            and selected.route_hash == route.route_hash
            and selected.topology_hash == route.topology_hash
            and guard_accepted
        )
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "target_kind": str(row["alternative_kind"]),
                "predicted_kind": prediction.alternative_kind.value,
                "target_side": str(row["target_lane_side"]),
                "predicted_side": prediction.target_lane_side.value,
                "target_available": bool(row["alternative_available"]),
                "predicted_available": predicted_available,
                "availability_probability": (
                    prediction.availability_probability
                ),
                "input_route_maneuver": route.maneuver.value,
                "output_route_maneuver": output_route,
                "legal_route_target": legal_route_target,
                "guard_accepted": guard_accepted,
                "mpc_accepted": bool(executable.live_prefilter),
                "guard_status": bundle.guard_status,
                "guard_reasons": list(bundle.guard_reasons),
                "selected_candidate_id": selected.candidate_id,
                "executability": executable.to_dict(),
            }
        )
    gate = evaluate_offline_prediction_records_v3(records)
    report = {
        **gate,
        "schema_version": "safedrive.k2_v3_offline_report.v1",
        "split": str(args.split),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "record_count": len(records),
        "all_outputs_finite": all(
            math.isfinite(float(row["availability_probability"]))
            for row in records
        ),
    }
    if not report["all_outputs_finite"]:
        report["passed"] = False
    _write_exclusive(Path(args.records_out), records)
    _write_exclusive(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
