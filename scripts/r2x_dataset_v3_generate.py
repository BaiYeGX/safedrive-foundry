#!/usr/bin/env python3
"""X5B/C offline dataset-v3: real-feature *contract* samples (not scene_proxy labels).

Until CARLA READY, features are synthetic L/R-biased tensors extracted via
``extract_driving_feature_bundle`` (same API as live SimLingo path).
Teacher defensive labels are **scene-conditioned** from conflict_side (not seed parity).

When real feature collect lands, replace feature fields only — keep teacher schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.driving_feature import extract_driving_feature_bundle  # noqa: E402

OUT = ROOT / "docs/runtime-evidence/r2x-training/dataset-v3"


def _hash_obj(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _path_group(native: list) -> str:
    q = [[round(float(x), 3), round(float(y), 3)] for x, y in native]
    return _hash_obj(q)


def _native(i: int) -> list[list[float]]:
    ang = 0.05 * (i % 16)
    curv = 0.01 * ((i // 4) % 5)
    x, y, th = float(i) * 0.3, float(i) * 0.1, ang
    out = []
    for _ in range(20):
        out.append([x, y])
        th += curv
        x += math.cos(th) * 1.2
        y += math.sin(th) * 1.2
    return out


def _feature_for_side(side: str, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    arr = rng.randn(1, 8, 96).astype(np.float32) * 0.15
    if side == "left":
        arr[:, :, :16] += 2.0
    elif side == "right":
        arr[:, :, 16:32] += 2.0
    elif side == "center":
        arr[:, :, 40:48] += 1.2
    # empty: near-zero structured noise only
    b = extract_driving_feature_bundle(arr, require=True)
    return {
        "driving_feature": list(b.mean64),
        "driving_feature_hash": b.mean64_hash,
        "full_pool": list(b.full_pool),
        "full_pool_hash": b.full_pool_hash,
        "raw_shape": list(b.raw_shape),
        "raw_content_hash": b.raw_content_hash,
        "driving_feature_source": "simlingo_driving_mean64_v1_synthetic_standin",
        "driving_feature_ok": True,
        "feature_pipeline_note": (
            "synthetic adaptor tensor through live extract API; "
            "replace with real SimLingo when CARLA READY"
        ),
    }


def _teacher(side: str, n: int, amb: bool, rng: random.Random) -> tuple[dict, dict, bool, str]:
    nom = {
        "raw_delta_s": [0.55 + 0.02 * rng.random() for _ in range(n)],
        "raw_d": [0.0] * n,
        "speed_scale": 1.0,
        "head_lineage": "teacher_label",
    }
    if not amb or side == "empty":
        return (
            nom,
            {
                "raw_delta_s": list(nom["raw_delta_s"]),
                "raw_d": [0.0] * n,
                "speed_scale": 1.0,
                "head_lineage": "teacher_label",
            },
            False,
            "NO_ALTERNATIVE",
        )
    # defensive away from conflict
    if side == "left":
        sign = -1.0
    elif side == "right":
        sign = 1.0
    else:
        sign = 1.0 if rng.random() > 0.5 else -1.0
    alt_d = [sign * min(1.5, 0.12 * i) for i in range(n)]
    return (
        nom,
        {
            "raw_delta_s": [0.45 + 0.02 * rng.random() for _ in range(n)],
            "raw_d": alt_d,
            "speed_scale": 0.85,
            "head_lineage": "teacher_label",
        },
        True,
        "scene_defensive_corridor",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ~30% empty for availability specificity
    families = [
        "left_cut_in",
        "right_cut_in",
        "lead_brake",
        "crossing",
        "empty",
        "empty",
        "empty",
        "left_cut_in",
        "right_cut_in",
        "lead_brake",
    ]
    train, val = [], []
    train_g, val_g = set(), set()
    i = 0
    while len(train) < args.n_train or len(val) < args.n_val:
        native = _native(i)
        pg = _path_group(native)
        if pg in train_g or pg in val_g:
            i += 1
            continue
        fill_val = len(val) < args.n_val and (
            len(train) >= args.n_train
            or (len(val) / max(args.n_val, 1)) <= (len(train) / max(args.n_train, 1))
        )
        if fill_val:
            fam = families[len(val) % len(families)]
            split, bucket, groups = "val", val, val_g
            sid = f"v3_val_{len(val)}"
            seed = args.seed + 5000 + len(val)
        else:
            if len(train) >= args.n_train:
                i += 1
                if i > 20000:
                    break
                continue
            fam = families[len(train) % len(families)]
            split, bucket, groups = "train", train, train_g
            sid = f"v3_train_{len(train)}"
            seed = args.seed + 1000 + len(train)
        if "left" in fam:
            side = "left"
        elif "right" in fam:
            side = "right"
        elif "empty" in fam:
            side = "empty"
        else:
            side = "center"
        amb = side != "empty" and (len(bucket) % 5) != 0
        rng = random.Random(seed)
        nom, alt, available, reason = _teacher(side, 20, amb, rng)
        feat = _feature_for_side(side, seed)
        ego_v = 4.0 + (seed % 5) * 0.3
        sample = {
            "sample_id": sid,
            "split_id": split,
            "scenario_family": fam,
            "scenario_group": f"{fam}:{pg[:8]}",
            "path_group": pg,
            "conflict_side": side,
            "alternative_available": available,
            "availability_reason": reason,
            "ego_v": ego_v,
            "base_speed_mps": ego_v / 0.9,
            "native_path_xy": native,
            "native_path_hash": _hash_obj(native),
            "nominal": nom,
            "defensive": alt,
            "teacher": {
                "source": "scene_conditioned_lattice",
                "version": "r2x_teacher_v3",
                "privileged_for_label_only": True,
                "note": "defensive side from conflict_side; not R2 oracle",
            },
            **feat,
            "leakage_audit": {"from_r2_pilot_outcome": False},
        }
        bucket.append(sample)
        groups.add(pg)
        i += 1
        if i > 20000:
            break

    samples = train + val
    sp = out / "samples.jsonl"
    with sp.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, sort_keys=True) + "\n")

    leak = {
        "schema_version": "safedrive.r2x.leakage.v3",
        "overlap_train_val_path_groups": sorted(train_g & val_g),
        "n_unique_train": len(train_g),
        "n_unique_val": len(val_g),
        "ok": len(train_g & val_g) == 0,
    }
    split_m = {
        "schema_version": "safedrive.r2x.split.v3",
        "counts": {"train": len(train), "val": len(val)},
        "feature": "simlingo_api_synthetic_standin",
        "teacher": "scene_conditioned_v3",
        "samples_path": str(sp.as_posix()),
        "content_hash": _hash_obj([s["sample_id"] for s in samples]),
    }
    (out / "split_manifest.json").write_text(json.dumps(split_m, indent=2) + "\n", encoding="utf-8")
    (out / "leakage_report.json").write_text(json.dumps(leak, indent=2) + "\n", encoding="utf-8")
    (out / "dataset_card.json").write_text(
        json.dumps(
            {
                "name": "r2x_dataset_v3",
                "note": "Feature API stand-in until CARLA SimLingo collect; teacher scene-conditioned",
                "n": len(samples),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_m, indent=2))
    print("leakage_ok", leak["ok"])
    return 0 if leak["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
