#!/usr/bin/env python3
"""Relabel Spatial K2 development data with the exact Guard-v3 teacher.

This is development-only.  The observed v4 exam is explicitly retired into
development, so a future formal R2-K still requires a fresh blind exam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.development_split import (  # noqa: E402
    assign_balanced_episode_split,
)
from driving_vla.model.spatial_k2_teacher import (  # noqa: E402
    load_teacher_config,
    select_defensive_teacher,
    teacher_label_to_dict,
)
from driving_vla.model.teacher_offline_stages import (  # noqa: E402
    production_execution_stages,
)

DEFAULT_SOURCE = (
    ROOT / "docs/runtime-evidence/r2x-training/dataset-v6-formal/samples.jsonl"
)
DEFAULT_OUT = (
    ROOT / "docs/runtime-evidence/r2x-training/dataset-v9-development"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relabel(row: dict[str, Any], *, config, stages) -> dict[str, Any]:
    native = row.get("native_path_xy") or []
    feature = row.get("driving_feature")
    if len(native) < 2:
        raise ValueError("native_path_xy missing")
    if not feature:
        raise ValueError("driving_feature missing")
    family = str(row.get("scenario_family") or "unknown")
    side = str(row.get("conflict_side") or "none")
    ego_v = float(row.get("ego_v") or 5.0)
    label = select_defensive_teacher(
        scenario_family=family,
        conflict_side=side,
        n_path=len(native),
        config=config,
        execution_stages=stages,
        allow_unit_test_stub_metrics=False,
        privileged_scene={
            "scenario_family": family,
            "conflict_side": side,
            "ego_v": ego_v,
        },
        native_path_xy=native,
        ego_v=ego_v,
    )
    output = dict(row)
    output["source_sample_id"] = str(row.get("sample_id") or "")
    output["sample_id"] = "v9_" + _sha(
        json.dumps(
            {
                "source_sample_id": output["source_sample_id"],
                "observation_hash": row.get("observation_hash")
                or row.get("driving_feature_hash"),
                "teacher_config_hash": label.config_hash,
                "selected_candidate_id": label.selected_candidate_id,
                "alternative_available": label.alternative_available,
            },
            sort_keys=True,
        )
    )[:12]
    output["alternative_available"] = bool(label.alternative_available)
    output["availability_reason"] = label.availability_reason
    output["nominal"] = label.nominal_residual
    output["defensive"] = label.defensive_residual or {
        "raw_delta_s": [0.0] * len(native),
        "raw_d": [0.0] * len(native),
        "speed_scale": 1.0,
        "head_lineage": "no_alternative",
    }
    output["teacher"] = teacher_label_to_dict(label)
    output["teacher_config_hash"] = label.config_hash
    output["teacher_schema_version"] = config.schema_version
    output["teacher_id"] = config.teacher_id
    output["teacher_version"] = "production_v4_legal_pool"
    output["dataset_version"] = "v9-development-legal-pool"
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    source = Path(args.source)
    out = Path(args.out)
    if not source.is_file():
        print(f"missing source: {source}", flush=True)
        return 2
    if out.exists():
        print(f"refuse to overwrite existing output: {out}", flush=True)
        return 3

    source_rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = load_teacher_config()
    if config.schema_version != "safedrive.k2_spatial_teacher.v4":
        print(f"wrong teacher schema: {config.schema_version}", flush=True)
        return 4
    stages = production_execution_stages()
    relabeled: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        relabeled.append(_relabel(row, config=config, stages=stages))
        if (index + 1) % 10 == 0:
            print(f"relabeled {index + 1}/{len(source_rows)}", flush=True)

    rows, split_manifest = assign_balanced_episode_split(
        relabeled, seed=args.seed
    )
    out.mkdir(parents=True, exist_ok=False)
    samples = out / "samples.jsonl"
    with samples.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    train_features = np.asarray(
        [
            [float(value) for value in row["driving_feature"]]
            for row in rows
            if row["split_id"] == "train"
        ],
        dtype=np.float64,
    )
    stats = {
        "n": int(train_features.shape[0]),
        "dim": int(train_features.shape[1]),
        "mean": train_features.mean(axis=0).tolist(),
        "std": (train_features.std(axis=0) + 1.0e-6).tolist(),
        "fit_split": "train",
    }
    (out / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    available_rows = [row for row in rows if row["alternative_available"]]
    subfloor = [
        row["sample_id"]
        for row in available_rows
        if max(abs(float(value)) for value in row["defensive"]["raw_d"]) < 0.5
    ]
    card = {
        "schema_version": "safedrive.r2x.dataset.v9_development",
        "status": "DEVELOPMENT_ONLY" if not subfloor else "INVALID_SUBFLOOR_LABEL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source.as_posix()),
        "source_sha256": _file_sha(source),
        "samples_sha256": _file_sha(samples),
        "teacher_schema_version": config.schema_version,
        "teacher_id": config.teacher_id,
        "teacher_config_hash": config.config_hash(),
        "n_samples": len(rows),
        "split_counts": dict(Counter(row["split_id"] for row in rows)),
        "family_counts": dict(Counter(row["scenario_family"] for row in rows)),
        "side_counts": dict(Counter(row["conflict_side"] for row in rows)),
        "alt_counts": {
            str(key): value
            for key, value in Counter(
                bool(row["alternative_available"]) for row in rows
            ).items()
        },
        "available_subfloor_label_ids": subfloor,
        "old_v4_exam_retired_to_development": True,
        "formal_train_candidate": False,
        "r2k_pilot_allowed": False,
        "new_blind_exam_registry_required": True,
        "episode_leakage": False,
    }
    (out / "dataset_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(card, indent=2, sort_keys=True), flush=True)
    return 0 if not subfloor else 5


if __name__ == "__main__":
    raise SystemExit(main())
