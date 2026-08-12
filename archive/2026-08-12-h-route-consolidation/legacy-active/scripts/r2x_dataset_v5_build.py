#!/usr/bin/env python3
"""X5D: build immutable dataset-v5-real from v4 real features + production teacher.

Does NOT overwrite dataset-v4-real. Relabels with production execution stages
(Guard / PM / steer-κ / MPC) and stratified train/val/holdout split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.spatial_k2_teacher import (  # noqa: E402
    load_teacher_config,
    select_defensive_teacher,
    teacher_label_to_dict,
)
from driving_vla.model.teacher_offline_stages import (  # noqa: E402
    production_execution_stages,
)

DEFAULT_V4 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real/samples.jsonl"
DEFAULT_OUT = ROOT / "docs/runtime-evidence/r2x-training/dataset-v5-real"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_obj(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, default=str))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _episode_id(r: dict) -> str:
    return str(
        r.get("episode_id")
        or f"{r.get('scenario_id','unk')}__{r.get('seed_id','unk')}__{r.get('variant','base')}"
    )


def _split_by_episode(
    rows: list[dict],
    *,
    seed: int = 11,
    train_frac: float = 0.70,
    val_frac: float = 0.20,
) -> dict[str, str]:
    """Map episode_id → split. Stratify by family; leave holdout for formal eval."""
    ep_fam: dict[str, str] = {}
    for r in rows:
        e = _episode_id(r)
        ep_fam.setdefault(e, str(r.get("scenario_family") or "unk"))
    by_fam: dict[str, list[str]] = {}
    for e, fam in ep_fam.items():
        by_fam.setdefault(fam, []).append(e)
    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for fam, eps in by_fam.items():
        eps = sorted(eps)
        rng.shuffle(eps)
        n = len(eps)
        if n == 1:
            assign[eps[0]] = "train"
            continue
        if n == 2:
            assign[eps[0]] = "train"
            assign[eps[1]] = "val"
            continue
        n_va = max(1, int(round(val_frac * n)))
        n_ho = max(1, int(round((1.0 - train_frac - val_frac) * n)))
        if n_va + n_ho >= n:
            n_ho = max(0, n - 2)
            n_va = 1
        n_tr = n - n_va - n_ho
        for i, e in enumerate(eps):
            if i < n_tr:
                assign[e] = "train"
            elif i < n_tr + n_va:
                assign[e] = "val"
            else:
                assign[e] = "holdout"
    return assign


def _norm_stats(train_feats: list[list[float]]) -> dict:
    x = np.asarray(train_feats, dtype=np.float64)
    if x.size == 0:
        return {"mean": [], "std": [], "n": 0, "hash": ""}
    mu = x.mean(axis=0)
    sig = x.std(axis=0) + 1e-6
    stats = {
        "mean": [float(v) for v in mu.tolist()],
        "std": [float(v) for v in sig.tolist()],
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
    }
    stats["hash"] = _hash_obj({"mean": stats["mean"], "std": stats["std"]})[:16]
    return stats


def _relabel_row(
    row: dict,
    *,
    stages: dict,
    teacher_cfg,
) -> dict:
    native = row.get("native_path_xy") or []
    if len(native) < 2:
        raise ValueError(f"missing native_path_xy for {row.get('sample_id')}")
    n_path = len(native)
    ego_v = float(row.get("ego_v") or 5.0)
    label = select_defensive_teacher(
        scenario_family=str(row.get("scenario_family") or ""),
        conflict_side=str(row.get("conflict_side") or ""),
        n_path=n_path,
        config=teacher_cfg,
        execution_stages=stages,
        allow_unit_test_stub_metrics=False,
        privileged_scene={
            "scenario_family": row.get("scenario_family"),
            "conflict_side": row.get("conflict_side"),
            "ego_v": ego_v,
        },
        native_path_xy=native,
        ego_v=ego_v,
    )
    td = teacher_label_to_dict(label)
    out = dict(row)
    out["alternative_available"] = bool(label.alternative_available)
    out["availability_reason"] = label.availability_reason
    out["nominal"] = label.nominal_residual
    out["defensive"] = label.defensive_residual or {
        "raw_delta_s": [0.0] * n_path,
        "raw_d": [0.0] * n_path,
        "speed_scale": 1.0,
        "head_lineage": "no_alternative",
    }
    out["teacher"] = td
    out["teacher_config_hash"] = label.config_hash
    out["teacher_version"] = "production_v2_stages"
    out["dataset_version"] = "v5-real"
    out["source_dataset"] = "dataset-v4-real"
    out["source_sample_id"] = row.get("sample_id")
    # recompute sample_id with new teacher hash
    out["sample_id"] = "v5_" + _sha(
        json.dumps(
            {
                "src": row.get("sample_id"),
                "obs": row.get("observation_hash"),
                "tch": label.config_hash,
                "avail": label.alternative_available,
                "reason": label.availability_reason,
            },
            sort_keys=True,
        )
    )[:12]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-samples", type=str, default=str(DEFAULT_V4))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    v4_path = Path(args.v4_samples)
    out = Path(args.out)
    if not v4_path.is_file():
        print(f"missing v4 samples: {v4_path}", flush=True)
        return 2
    if out.exists() and (out / "samples.jsonl").is_file():
        print(f"dataset-v5 already exists at {out}; refuse overwrite", flush=True)
        # allow rebuild only if marker says incomplete
        card = out / "dataset_card.json"
        if card.is_file():
            c = json.loads(card.read_text(encoding="utf-8"))
            if c.get("status") == "OK":
                print("status=OK; use a new --out path to regenerate", flush=True)
                return 0

    out.mkdir(parents=True, exist_ok=True)
    rows_in = [
        json.loads(line)
        for line in v4_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows_in:
        print("empty v4", flush=True)
        return 2

    stages = production_execution_stages()
    teacher_cfg = load_teacher_config()
    # production must require stages
    assert teacher_cfg.require_execution_filters, "teacher config must require stages"

    print(f"relabeling n={len(rows_in)} with production teacher…", flush=True)
    labeled: list[dict] = []
    for i, row in enumerate(rows_in):
        try:
            labeled.append(_relabel_row(row, stages=stages, teacher_cfg=teacher_cfg))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {row.get('sample_id')}: {exc}", flush=True)
            raise
        if (i + 1) % 5 == 0 or i + 1 == len(rows_in):
            n_alt = sum(1 for r in labeled if r.get("alternative_available"))
            print(
                f"  {i+1}/{len(rows_in)} alt={n_alt}/{len(labeled)}",
                flush=True,
            )

    split_map = _split_by_episode(labeled, seed=args.seed)
    for r in labeled:
        r["split_id"] = split_map[_episode_id(r)]

    # Write samples
    samples_path = out / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as fh:
        for r in labeled:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")

    train_feats = [
        list(r["driving_feature"])
        for r in labeled
        if r.get("split_id") == "train" and r.get("driving_feature")
    ]
    stats = _norm_stats(train_feats)
    (out / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )

    split_counts = Counter(r["split_id"] for r in labeled)
    fam_counts = Counter(r.get("scenario_family") for r in labeled)
    alt_counts = Counter(bool(r.get("alternative_available")) for r in labeled)
    reason_counts = Counter(r.get("availability_reason") for r in labeled)
    split_alt = Counter(
        (r["split_id"], bool(r.get("alternative_available"))) for r in labeled
    )

    card = {
        "schema_version": "safedrive.r2x.dataset.v5",
        "status": "OK",
        "dataset_id": "dataset-v5-real",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(labeled),
        "source_v4_samples": str(v4_path.as_posix()),
        "source_v4_sha256": _file_sha256(v4_path),
        "samples_sha256": _file_sha256(samples_path),
        "teacher_config_hash": teacher_cfg.config_hash(),
        "teacher_version": "production_v2_stages",
        "require_execution_filters": True,
        "stages": list(stages.keys()),
        "split_counts": dict(split_counts),
        "family_counts": dict(fam_counts),
        "alt_counts": {str(k): v for k, v in alt_counts.items()},
        "reason_counts": dict(reason_counts),
        "split_alt": {f"{a}/{b}": c for (a, b), c in split_alt.items()},
        "immutable": True,
        "overwrite_forbidden": True,
        "normalization_hash": stats.get("hash"),
        "is_real_simlingo_feature": all(
            r.get("is_real_simlingo_feature") for r in labeled
        ),
    }
    (out / "dataset_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )

    split_manifest = {
        "seed": args.seed,
        "episode_split": split_map,
        "counts": dict(split_counts),
        "policy": "stratified_by_family_episode",
        "holdout_for": "formal_offline_and_x5h_selection_only",
    }
    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )

    teacher_manifest = {
        "teacher_id": teacher_cfg.teacher_id,
        "config_hash": teacher_cfg.config_hash(),
        "require_execution_filters": True,
        "stages": list(stages.keys()),
        "n_alt": int(alt_counts.get(True, 0)),
        "n_no_alt": int(alt_counts.get(False, 0)),
        "reasons": dict(reason_counts),
    }
    (out / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest, indent=2) + "\n", encoding="utf-8"
    )

    lineage = {
        "source_dataset": "dataset-v4-real",
        "source_samples_sha256": card["source_v4_sha256"],
        "v5_samples_sha256": card["samples_sha256"],
        "teacher_config_hash": teacher_cfg.config_hash(),
        "feature_hashes": [
            r.get("driving_feature_hash") for r in labeled[:5]
        ],  # sample
        "n": len(labeled),
    }
    (out / "lineage_report.json").write_text(
        json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
    )

    leakage = {
        "frame_level_split": False,
        "episode_level_split": True,
        "holdout_disjoint": True,
        "train_eps": sorted(
            {e for e, s in split_map.items() if s == "train"}
        ),
        "val_eps": sorted({e for e, s in split_map.items() if s == "val"}),
        "holdout_eps": sorted(
            {e for e, s in split_map.items() if s == "holdout"}
        ),
    }
    # verify no overlap
    tr, va, ho = set(leakage["train_eps"]), set(leakage["val_eps"]), set(
        leakage["holdout_eps"]
    )
    leakage["overlap_train_val"] = sorted(tr & va)
    leakage["overlap_train_holdout"] = sorted(tr & ho)
    leakage["overlap_val_holdout"] = sorted(va & ho)
    leakage["ok"] = not (
        leakage["overlap_train_val"]
        or leakage["overlap_train_holdout"]
        or leakage["overlap_val_holdout"]
    )
    (out / "leakage_report.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(card, indent=2), flush=True)
    return 0 if leakage["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
