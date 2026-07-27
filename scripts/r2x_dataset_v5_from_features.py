#!/usr/bin/env python3
"""Build teacher-labeled samples from live feature anchors + optional merge with v4/v5."""

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
from driving_vla.model.teacher_offline_stages import production_execution_stages  # noqa: E402


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _hash_obj(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, default=str))


def _load_feature_anchors(d: Path) -> list[dict]:
    rows = []
    ad = d / "anchors"
    if not ad.is_dir():
        for p in sorted(d.glob("**/feature.json")):
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        return rows
    for p in sorted(ad.glob("*/feature.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def _episode_id(r: dict) -> str:
    return str(
        r.get("episode_id")
        or f"{r.get('scenario_id','unk')}__{r.get('seed_id','unk')}__{r.get('variant','base')}"
    )


def _split_by_episode(rows: list[dict], seed: int = 13) -> dict[str, str]:
    ep_fam: dict[str, str] = {}
    for r in rows:
        ep_fam.setdefault(_episode_id(r), str(r.get("scenario_family") or "unk"))
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
        n_va = max(1, int(round(0.2 * n)))
        n_ho = max(1, int(round(0.1 * n)))
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


def feature_to_sample(f: dict, *, stages, teacher_cfg) -> dict:
    native = f.get("native_path_xy")
    if not native or len(native) < 2:
        raise ValueError(f"native_path missing {f.get('anchor_id')}")
    drive = f.get("driving_feature") or f.get("mean64")
    if not drive:
        raise ValueError(f"driving_feature missing {f.get('anchor_id')}")
    ego_v = float(f.get("ego_v") or 5.0)
    fam = str(f.get("scenario_family") or "unknown")
    side = str(f.get("conflict_side") or "none")
    n_path = len(native)
    label = select_defensive_teacher(
        scenario_family=fam,
        conflict_side=side,
        n_path=n_path,
        config=teacher_cfg,
        execution_stages=stages,
        allow_unit_test_stub_metrics=False,
        privileged_scene={
            "scenario_family": fam,
            "conflict_side": side,
            "ego_v": ego_v,
        },
        native_path_xy=native,
        ego_v=ego_v,
    )
    td = teacher_label_to_dict(label)
    sample_id = "v5_" + _sha(
        json.dumps(
            {
                "aid": f.get("anchor_id"),
                "obs": f.get("observation_hash") or f.get("driving_feature_hash"),
                "tch": label.config_hash,
                "avail": label.alternative_available,
            },
            sort_keys=True,
        )
    )[:12]
    return {
        "sample_id": sample_id,
        "anchor_id": f.get("anchor_id"),
        "episode_id": _episode_id(f),
        "scenario_id": f.get("scenario_id"),
        "seed_id": f.get("seed_id"),
        "variant": f.get("variant") or "base",
        "scenario_family": fam,
        "conflict_side": side,
        "ego_v": ego_v,
        "base_speed_mps": float(f.get("base_speed_mps") or ego_v),
        "native_path_xy": native,
        "driving_feature": list(drive),
        "driving_feature_hash": f.get("driving_feature_hash")
        or f.get("mean64_hash")
        or _hash_obj(list(drive))[:16],
        "driving_feature_ok": True,
        "driving_feature_source": f.get("driving_feature_source") or "carla_live",
        "is_real_simlingo_feature": True,
        "alternative_available": bool(label.alternative_available),
        "availability_reason": label.availability_reason,
        "nominal": label.nominal_residual,
        "defensive": label.defensive_residual
        or {
            "raw_delta_s": [0.0] * n_path,
            "raw_d": [0.0] * n_path,
            "speed_scale": 1.0,
            "head_lineage": "no_alternative",
        },
        "teacher": td,
        "teacher_config_hash": label.config_hash,
        "teacher_version": "production_v2_stages",
        "dataset_version": "v5-real-live",
        "model_checkpoint_hash": f.get("model_checkpoint_hash"),
        "observation_hash": f.get("observation_hash") or f.get("driving_feature_hash"),
        "simlingo_mode": f.get("simlingo_mode") or "live",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features-dir",
        default=str(
            ROOT / "docs/runtime-evidence/r2x-training/dataset-v5-live-features"
        ),
    )
    ap.add_argument(
        "--merge-samples",
        default=str(
            ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real/samples.jsonl"
        ),
        help="optional prior samples.jsonl to merge (re-labeled if --relabel-merge)",
    )
    ap.add_argument("--relabel-merge", action="store_true")
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/runtime-evidence/r2x-training/dataset-v5b-real"),
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and (out / "samples.jsonl").is_file() and not args.force:
        print(f"exists {out}; use --force", flush=True)
        return 0
    if out.exists() and args.force:
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    stages = production_execution_stages()
    teacher_cfg = load_teacher_config()
    feats = _load_feature_anchors(Path(args.features_dir))
    print(f"live features n={len(feats)}", flush=True)
    labeled: list[dict] = []
    for i, f in enumerate(feats):
        labeled.append(feature_to_sample(f, stages=stages, teacher_cfg=teacher_cfg))
        if (i + 1) % 5 == 0:
            n_alt = sum(1 for r in labeled if r["alternative_available"])
            print(f"  feat {i+1}/{len(feats)} alt={n_alt}", flush=True)

    merge_path = Path(args.merge_samples)
    if merge_path.is_file():
        print(f"merging {merge_path}", flush=True)
        for line in merge_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.relabel_merge:
                # re-label via teacher using existing fields
                f = {
                    "anchor_id": row.get("sample_id"),
                    "episode_id": row.get("episode_id"),
                    "scenario_id": row.get("scenario_id"),
                    "seed_id": row.get("seed_id"),
                    "variant": row.get("variant"),
                    "scenario_family": row.get("scenario_family"),
                    "conflict_side": row.get("conflict_side"),
                    "ego_v": row.get("ego_v"),
                    "base_speed_mps": row.get("base_speed_mps"),
                    "native_path_xy": row.get("native_path_xy"),
                    "driving_feature": row.get("driving_feature"),
                    "driving_feature_hash": row.get("driving_feature_hash"),
                    "observation_hash": row.get("observation_hash"),
                    "model_checkpoint_hash": row.get("model_checkpoint_hash"),
                }
                try:
                    labeled.append(
                        feature_to_sample(f, stages=stages, teacher_cfg=teacher_cfg)
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip merge {row.get('sample_id')}: {exc}", flush=True)
            else:
                # keep prior labels if already production teacher
                if row.get("teacher_version") == "production_v2_stages":
                    labeled.append(row)
                else:
                    f = {
                        "anchor_id": row.get("sample_id"),
                        "episode_id": row.get("episode_id"),
                        "scenario_id": row.get("scenario_id"),
                        "seed_id": row.get("seed_id"),
                        "variant": row.get("variant"),
                        "scenario_family": row.get("scenario_family"),
                        "conflict_side": row.get("conflict_side"),
                        "ego_v": row.get("ego_v"),
                        "base_speed_mps": row.get("base_speed_mps"),
                        "native_path_xy": row.get("native_path_xy"),
                        "driving_feature": row.get("driving_feature"),
                        "driving_feature_hash": row.get("driving_feature_hash"),
                        "observation_hash": row.get("observation_hash"),
                        "model_checkpoint_hash": row.get("model_checkpoint_hash"),
                    }
                    labeled.append(
                        feature_to_sample(f, stages=stages, teacher_cfg=teacher_cfg)
                    )

    # dedupe by driving_feature_hash + episode
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in labeled:
        k = f"{r.get('driving_feature_hash')}|{_episode_id(r)}|{r.get('alternative_available')}"
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    labeled = uniq

    split_map = _split_by_episode(labeled, seed=args.seed)
    for r in labeled:
        r["split_id"] = split_map[_episode_id(r)]

    sp = out / "samples.jsonl"
    with sp.open("w", encoding="utf-8") as fh:
        for r in labeled:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")

    train_feats = [
        list(r["driving_feature"])
        for r in labeled
        if r.get("split_id") == "train" and r.get("driving_feature")
    ]
    x = np.asarray(train_feats, dtype=np.float64) if train_feats else np.zeros((0, 1))
    stats = {
        "mean": x.mean(0).tolist() if x.size else [],
        "std": (x.std(0) + 1e-6).tolist() if x.size else [],
        "n": int(x.shape[0]) if x.size else 0,
        "dim": int(x.shape[1]) if x.size else 0,
    }
    stats["hash"] = _hash_obj({"mean": stats["mean"], "std": stats["std"]})[:16]
    (out / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )

    card = {
        "schema_version": "safedrive.r2x.dataset.v5b",
        "status": "OK",
        "dataset_id": out.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(labeled),
        "samples_sha256": _file_sha(sp),
        "features_dir": str(Path(args.features_dir).as_posix()),
        "merge_samples": str(merge_path.as_posix()) if merge_path.is_file() else None,
        "teacher_config_hash": teacher_cfg.config_hash(),
        "split_counts": dict(Counter(r["split_id"] for r in labeled)),
        "family_counts": dict(Counter(r.get("scenario_family") for r in labeled)),
        "alt_counts": {
            str(k): v
            for k, v in Counter(bool(r.get("alternative_available")) for r in labeled).items()
        },
        "reason_counts": dict(Counter(r.get("availability_reason") for r in labeled)),
        "immutable": True,
    }
    (out / "dataset_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    (out / "split_manifest.json").write_text(
        json.dumps({"seed": args.seed, "episode_split": split_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(card, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
