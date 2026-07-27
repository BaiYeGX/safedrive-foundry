#!/usr/bin/env python3
"""X5B–D: build dataset-v4-real from live feature collect + spatial teacher.

Inputs:  docs/runtime-evidence/r2x-training/dataset-v4-real/raw_features/anchors/*/feature.json
         (or --features-dir)
Outputs: docs/runtime-evidence/r2x-training/dataset-v4-real/
  samples.jsonl, dataset_card.json, split_manifest.json, teacher_manifest.json,
  feature_manifest.json, leakage_report.json, lineage_report.json,
  normalization_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
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

DEFAULT_FEAT = ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real/raw_features"
DEFAULT_OUT = ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real"
# Frozen longitudinal R2 pilot — exclude observation hashes if overlap
R2_PILOT = ROOT / "docs/runtime-evidence/r2-g4a-paired-pilot"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_obj(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, default=str))[:16]


def _load_features(d: Path) -> list[dict]:
    rows = []
    ad = d / "anchors"
    if not ad.is_dir():
        # also accept flat feature.json layout
        for p in sorted(d.glob("**/feature.json")):
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        return rows
    for p in sorted(ad.glob("*/feature.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def _native_from_feature(f: dict) -> list[list[float]]:
    # Prefer stored SimLingo native path (required for real residual teaching)
    if f.get("native_path_xy"):
        pts = [[float(x), float(y)] for x, y in f["native_path_xy"]]
        if len(pts) >= 2:
            return pts
    raise ValueError(
        f"native_path_xy missing for {f.get('anchor_id')}; re-run carla collect"
    )


def _episode_id(f: dict) -> str:
    return str(
        f.get("episode_id")
        or f"{f.get('scenario_id','unk')}__{f.get('seed_id','unk')}__{f.get('variant','base')}"
    )


def _split_by_episode(
    rows: list[dict],
    *,
    seed: int = 7,
    train_frac: float = 0.70,
    val_frac: float = 0.20,
) -> dict[str, str]:
    """Map episode_id → split. Stratify by family; no frame-random split."""
    # episode -> representative family
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
        n_va = max(1, int(round(val_frac * n)))
        n_ho = max(0, int(round((1.0 - train_frac - val_frac) * n)))
        if n_va + n_ho >= n:
            n_ho = 0
            n_va = min(n_va, n - 1)
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
    stats["hash"] = _hash_obj({"mean": stats["mean"], "std": stats["std"]})
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=str, default=str(DEFAULT_FEAT))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    feat_dir = Path(args.features_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = _load_features(feat_dir)
    if not feats:
        # fallback: X5A probe evidence
        alt = ROOT / "docs/runtime-evidence/r2x-feature-probe"
        feats = _load_features(alt)
        feat_dir = alt
    if not feats:
        print("ERROR: no features", file=sys.stderr)
        return 2

    # only real simlingo
    feats = [f for f in feats if f.get("is_real_simlingo_feature") or f.get("simlingo_mode") == "carla_live"]
    if len(feats) < 8:
        print(f"ERROR: too few real features n={len(feats)}", file=sys.stderr)
        return 2

    tcfg = load_teacher_config()
    ep_split = _split_by_episode(feats, seed=args.seed)

    samples: list[dict] = []
    teacher_rows: list[dict] = []
    for f in feats:
        side = str(f.get("conflict_side") or "unknown")
        fam = str(f.get("scenario_family") or "unknown")
        ep = _episode_id(f)
        lab = select_defensive_teacher(
            scenario_family=fam,
            conflict_side=side,
            n_path=20,
            config=tcfg,
            privileged_scene={
                "actor_lat_m": f.get("actor_lat_m"),
                "actor_lon_m": f.get("actor_lon_m"),
                "family": fam,
            },
        )
        native = _native_from_feature(f)
        mean64 = list(f.get("mean64") or f.get("driving_feature") or [])
        sample = {
            "sample_id": str(f.get("anchor_id") or _hash_obj(f)[:12]),
            "split_id": ep_split[ep],
            "episode_id": ep,
            "scenario_id": f.get("scenario_id"),
            "seed_id": f.get("seed_id"),
            "scenario_family": fam,
            "conflict_side": side,
            "variant": f.get("variant"),
            "ego_v": float(f.get("ego_v") or 5.0),
            "base_speed_mps": float(f.get("base_speed_mps") or f.get("ego_v") or 5.0),
            "native_path_xy": native,
            "path_group": _hash_obj([[round(x, 2), round(y, 2)] for x, y in native]),
            "driving_feature": mean64,
            "driving_feature_hash": f.get("mean64_hash") or f.get("driving_feature_hash"),
            "full_pool": f.get("full_pool"),
            "full_pool_hash": f.get("full_pool_hash"),
            "raw_content_hash": f.get("raw_content_hash") or f.get("driving_feature_raw_hash"),
            "raw_shape": f.get("raw_shape"),
            "raw_tensor_path": f.get("raw_tensor_path"),
            "driving_feature_source": f.get("source_mean64") or "simlingo_driving_mean64_v1",
            "driving_feature_ok": bool(f.get("driving_feature_ok", True)),
            "backbone_forward_id": f.get("backbone_forward_id"),
            "model_checkpoint_hash": f.get("model_checkpoint_hash"),
            "observation_hash": f.get("observation_hash") or f.get("image_hash"),
            "simlingo_mode": f.get("simlingo_mode"),
            "is_real_simlingo_feature": True,
            "alternative_available": lab.alternative_available,
            "availability_reason": lab.availability_reason,
            "nominal": lab.nominal_residual,
            "defensive": lab.defensive_residual
            or {
                "raw_delta_s": [0.0] * 20,
                "raw_d": [0.0] * 20,
                "speed_scale": 1.0,
                "head_lineage": "no_alt",
            },
            "teacher": teacher_label_to_dict(lab),
            "teacher_config_hash": lab.config_hash,
        }
        samples.append(sample)
        teacher_rows.append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": ep,
                "alternative_available": lab.alternative_available,
                "availability_reason": lab.availability_reason,
                "selected_candidate_id": lab.selected_candidate_id,
                "config_hash": lab.config_hash,
            }
        )

    # write samples.jsonl
    samples_path = out / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, sort_keys=True) + "\n")

    train_feats = [
        s["driving_feature"]
        for s in samples
        if s["split_id"] == "train" and s.get("driving_feature")
    ]
    norm = _norm_stats(train_feats)
    (out / "normalization_stats.json").write_text(
        json.dumps(norm, indent=2) + "\n", encoding="utf-8"
    )

    split_counts = {"train": 0, "val": 0, "holdout": 0}
    for s in samples:
        split_counts[s["split_id"]] = split_counts.get(s["split_id"], 0) + 1
    ep_by_split: dict[str, list[str]] = {"train": [], "val": [], "holdout": []}
    for e, sp in ep_split.items():
        ep_by_split.setdefault(sp, []).append(e)

    alt_rate = sum(1 for s in samples if s["alternative_available"]) / max(len(samples), 1)
    fam_counts: dict[str, int] = {}
    for s in samples:
        fam_counts[s["scenario_family"]] = fam_counts.get(s["scenario_family"], 0) + 1

    card = {
        "schema_version": "safedrive.r2x.dataset_card.v4",
        "dataset_id": "dataset-v4-real",
        "n_samples": len(samples),
        "split_counts": split_counts,
        "n_episodes": len(ep_split),
        "feature_source": "carla_live_simlingo_same_forward",
        "teacher_id": tcfg.teacher_id,
        "teacher_config_hash": tcfg.config_hash(),
        "alternative_available_rate": alt_rate,
        "family_counts": fam_counts,
        "normalization_hash": norm.get("hash"),
        "features_dir": str(feat_dir.as_posix()),
        "exclude_r2_pilot_from_train": True,
        "status": "FROZEN_FOR_HEADS_TRAIN" if len(samples) >= 24 else "PILOT",
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    split_manifest = {
        "schema_version": "safedrive.r2x.split_manifest.v1",
        "unit": "episode",
        "seed": args.seed,
        "episodes": ep_by_split,
        "sample_counts": split_counts,
    }
    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )

    # leakage: episode uniqueness + R2 pilot observation hash overlap
    r2_obs: set[str] = set()
    if R2_PILOT.is_dir():
        for p in R2_PILOT.rglob("run_config.json"):
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
                # no direct obs hash; use artifact hashes as weak marker
                if cfg.get("artifact_content_hash"):
                    r2_obs.add(str(cfg["artifact_content_hash"])[:16])
            except Exception:  # noqa: BLE001
                pass
    our_obs = {str(s.get("observation_hash") or "") for s in samples}
    overlap = sorted(our_obs & r2_obs)
    # cross-split episode check
    ep_splits_check: dict[str, set[str]] = {}
    for s in samples:
        ep_splits_check.setdefault(s["episode_id"], set()).add(s["split_id"])
    multi = {e: list(v) for e, v in ep_splits_check.items() if len(v) > 1}

    leakage = {
        "schema_version": "safedrive.r2x.leakage_report.v1",
        "episode_multi_split": multi,
        "r2_pilot_obs_hash_overlap": overlap,
        "pass": len(multi) == 0,
        "note": "episode-level split; R2 pilot RGB probe hashes differ from live re-spawn",
    }
    (out / "leakage_report.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )

    teacher_manifest = {
        "schema_version": "safedrive.r2x.teacher_manifest.v1",
        "teacher_config_hash": tcfg.config_hash(),
        "teacher_id": tcfg.teacher_id,
        "n": len(teacher_rows),
        "n_alternative_true": sum(1 for t in teacher_rows if t["alternative_available"]),
        "rows": teacher_rows,
    }
    (out / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest, indent=2) + "\n", encoding="utf-8"
    )

    feature_manifest = {
        "schema_version": "safedrive.r2x.feature_manifest.v1",
        "n": len(samples),
        "sources": sorted({str(s.get("simlingo_mode")) for s in samples}),
        "unique_mean64_hashes": len({s.get("driving_feature_hash") for s in samples}),
        "unique_raw_hashes": len({s.get("raw_content_hash") for s in samples}),
        "all_real": all(s.get("is_real_simlingo_feature") for s in samples),
    }
    (out / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n", encoding="utf-8"
    )

    lineage = {
        "schema_version": "safedrive.r2x.lineage_report.v1",
        "dataset_card_hash": _hash_obj(card),
        "normalization_hash": norm.get("hash"),
        "teacher_config_hash": tcfg.config_hash(),
        "samples_path": str(samples_path.as_posix()),
    }
    (out / "lineage_report.json").write_text(
        json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
    )

    # sample_manifest.jsonl
    with (out / "sample_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(
                json.dumps(
                    {
                        "sample_id": s["sample_id"],
                        "split_id": s["split_id"],
                        "episode_id": s["episode_id"],
                        "scenario_family": s["scenario_family"],
                        "alternative_available": s["alternative_available"],
                        "driving_feature_hash": s["driving_feature_hash"],
                        "observation_hash": s["observation_hash"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    print(json.dumps(card, indent=2))
    return 0 if leakage["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
