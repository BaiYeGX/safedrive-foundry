#!/usr/bin/env python3
"""Build formal train set: EXCLUDE R2 12-pair exam fixtures from train/val.

Exam pairs remain holdout-only for R2-K / regression (forbid_r2_pilot_in_train).
Sources: v4 samples + live features, re-labeled with production teacher.
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

from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    REQUIRED_SCENARIO_IDS,
    REQUIRED_SEEDS,
)
from driving_vla.model.spatial_k2_teacher import (  # noqa: E402
    load_teacher_config,
    select_defensive_teacher,
    teacher_label_to_dict,
)
from driving_vla.model.teacher_offline_stages import production_execution_stages  # noqa: E402

EXAM_PAIRS = {
    (sid, seed) for sid in REQUIRED_SCENARIO_IDS for seed in REQUIRED_SEEDS
}

DEFAULT_OUT = ROOT / "docs/runtime-evidence/r2x-training/dataset-v6-formal"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _is_exam(scenario_id: str | None, seed_id: str | None) -> bool:
    return (str(scenario_id or ""), str(seed_id or "")) in EXAM_PAIRS


def _load_v4(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def _load_features(d: Path) -> list[dict]:
    ad = d / "anchors"
    rows = []
    if not ad.is_dir():
        return rows
    for p in sorted(ad.glob("*/feature.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def _relabel(row: dict, *, stages, cfg) -> dict:
    native = row.get("native_path_xy") or []
    if len(native) < 2:
        raise ValueError("native_path_xy missing")
    drive = row.get("driving_feature") or row.get("mean64")
    if not drive:
        raise ValueError("driving_feature missing")
    ego_v = float(row.get("ego_v") or 5.0)
    fam = str(row.get("scenario_family") or "unknown")
    side = str(row.get("conflict_side") or "none")
    n_path = len(native)
    lab = select_defensive_teacher(
        scenario_family=fam,
        conflict_side=side,
        n_path=n_path,
        config=cfg,
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
    sid = str(row.get("scenario_id") or "")
    seed = str(row.get("seed_id") or "")
    exam = _is_exam(sid, seed)
    sample_id = "v6_" + _sha(
        json.dumps(
            {
                "sid": sid,
                "seed": seed,
                "obs": row.get("observation_hash") or row.get("driving_feature_hash"),
                "tch": lab.config_hash,
                "avail": lab.alternative_available,
            },
            sort_keys=True,
        )
    )[:12]
    return {
        "sample_id": sample_id,
        "episode_id": row.get("episode_id")
        or f"{sid}__{seed}__{row.get('variant') or 'base'}",
        "scenario_id": sid,
        "seed_id": seed,
        "variant": row.get("variant") or "base",
        "scenario_family": fam,
        "conflict_side": side,
        "ego_v": ego_v,
        "base_speed_mps": float(row.get("base_speed_mps") or ego_v),
        "native_path_xy": native,
        "driving_feature": list(drive),
        "driving_feature_hash": row.get("driving_feature_hash")
        or _sha(json.dumps(list(drive)))[:16],
        "driving_feature_ok": True,
        "driving_feature_source": row.get("driving_feature_source") or "merged",
        "is_real_simlingo_feature": bool(row.get("is_real_simlingo_feature", True)),
        "is_r2_exam_fixture": exam,
        "alternative_available": bool(lab.alternative_available),
        "availability_reason": lab.availability_reason,
        "nominal": lab.nominal_residual,
        "defensive": lab.defensive_residual
        or {
            "raw_delta_s": [0.0] * n_path,
            "raw_d": [0.0] * n_path,
            "speed_scale": 1.0,
            "head_lineage": "no_alternative",
        },
        "teacher": teacher_label_to_dict(lab),
        "teacher_config_hash": lab.config_hash,
        "teacher_version": "production_v2_stages",
        "dataset_version": "v6-formal",
        "observation_hash": row.get("observation_hash")
        or row.get("driving_feature_hash"),
        "model_checkpoint_hash": row.get("model_checkpoint_hash"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v4",
        default=str(
            ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real/samples.jsonl"
        ),
    )
    ap.add_argument(
        "--features",
        default=str(
            ROOT / "docs/runtime-evidence/r2x-training/dataset-v5-live-features"
        ),
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=17)
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
    cfg = load_teacher_config()
    raw_rows: list[dict] = []
    for r in _load_v4(Path(args.v4)):
        raw_rows.append(r)
    for f in _load_features(Path(args.features)):
        raw_rows.append(f)

    labeled: list[dict] = []
    for i, r in enumerate(raw_rows):
        try:
            labeled.append(_relabel(r, stages=stages, cfg=cfg))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {r.get('sample_id') or r.get('anchor_id')}: {exc}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  labeled {i+1}/{len(raw_rows)}", flush=True)

    # Dedup by feature hash + scenario + seed
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in labeled:
        k = f"{r.get('driving_feature_hash')}|{r.get('scenario_id')}|{r.get('seed_id')}"
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    labeled = uniq

    # Split: exam fixtures → holdout_exam only; rest stratified train/val/holdout_diag
    exam = [r for r in labeled if r["is_r2_exam_fixture"]]
    pool = [r for r in labeled if not r["is_r2_exam_fixture"]]
    # episode split on pool
    ep_fam: dict[str, str] = {}
    for r in pool:
        ep_fam.setdefault(r["episode_id"], r["scenario_family"])
    by_fam: dict[str, list[str]] = {}
    for e, fam in ep_fam.items():
        by_fam.setdefault(fam, []).append(e)
    rng = random.Random(args.seed)
    assign: dict[str, str] = {}
    for fam, eps in by_fam.items():
        eps = sorted(eps)
        rng.shuffle(eps)
        n = len(eps)
        if n == 1:
            assign[eps[0]] = "train"
            continue
        n_va = max(1, int(round(0.2 * n))) if n >= 3 else 1
        n_ho = max(0, int(round(0.1 * n))) if n >= 5 else 0
        if n_va + n_ho >= n:
            n_ho = 0
            n_va = min(1, n - 1)
        n_tr = n - n_va - n_ho
        for i, e in enumerate(eps):
            if i < n_tr:
                assign[e] = "train"
            elif i < n_tr + n_va:
                assign[e] = "val"
            else:
                assign[e] = "holdout_diag"
    for r in pool:
        r["split_id"] = assign[r["episode_id"]]
    for r in exam:
        r["split_id"] = "holdout_exam"

    all_rows = pool + exam
    sp = out / "samples.jsonl"
    with sp.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")

    train_feats = [
        list(r["driving_feature"])
        for r in all_rows
        if r["split_id"] == "train" and r.get("driving_feature")
    ]
    x = np.asarray(train_feats, dtype=np.float64) if train_feats else np.zeros((0, 64))
    stats = {
        "mean": x.mean(0).tolist() if x.size else [],
        "std": (x.std(0) + 1e-6).tolist() if x.size else [],
        "n": int(x.shape[0]) if x.size else 0,
        "dim": int(x.shape[1]) if x.ndim == 2 else 0,
    }
    stats["hash"] = _sha(json.dumps({"mean": stats["mean"], "std": stats["std"]}))[
        :16
    ]
    (out / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )

    # Leakage audit
    train_pairs = {
        (r["scenario_id"], r["seed_id"])
        for r in all_rows
        if r["split_id"] == "train"
    }
    exam_in_train = sorted(train_pairs & EXAM_PAIRS)
    leakage = {
        "forbid_r2_pilot_in_train": True,
        "exam_pairs": sorted(list(EXAM_PAIRS)),
        "exam_in_train": exam_in_train,
        "ok": len(exam_in_train) == 0,
        "n_exam_holdout": len(exam),
        "n_pool": len(pool),
    }
    (out / "leakage_report.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )

    card = {
        "schema_version": "safedrive.r2x.dataset.v6_formal",
        "status": "OK" if leakage["ok"] else "LEAKAGE_FAIL",
        "dataset_id": "dataset-v6-formal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(all_rows),
        "samples_sha256": _file_sha(sp),
        "teacher_config_hash": cfg.config_hash(),
        "split_counts": dict(Counter(r["split_id"] for r in all_rows)),
        "family_counts": dict(Counter(r["scenario_family"] for r in all_rows)),
        "alt_counts": {
            str(k): v
            for k, v in Counter(bool(r["alternative_available"]) for r in all_rows).items()
        },
        "exam_holdout_n": len(exam),
        "forbid_r2_pilot_in_train": True,
        "leakage_ok": leakage["ok"],
        "normalization_hash": stats["hash"],
        "formal_train_candidate": leakage["ok"],
    }
    (out / "dataset_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    (out / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "exam_pairs_holdout": sorted(list(EXAM_PAIRS)),
                "policy": "exam_holdout_plus_stratified_pool",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(card, indent=2), flush=True)
    return 0 if leakage["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
