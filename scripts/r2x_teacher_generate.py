#!/usr/bin/env python3
"""Generate offline dual-label Spatial K2 training samples (geometry teacher).

Teacher labels are Frenet residuals relative to native path — NOT R2 oracle winners.
Split is by **path geometry group** (not sample id only) to avoid train/val leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "runtime-evidence" / "r2x-training" / "dataset-v1"
PILOT = ROOT / "docs" / "runtime-evidence" / "r2-g4a-paired-pilot"


def _hash_obj(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _stable_seed_from_str(s: str, base: int = 0) -> int:
    """Process-stable seed (never use built-in hash() — salted across processes)."""
    h = hashlib.sha256(f"{base}:{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _native_from_bundle(bundle: dict) -> list[list[float]]:
    path = bundle.get("native_path_xy") or []
    return [[float(p[0]), float(p[1])] for p in path]


def _path_group_key(native: list[list[float]]) -> str:
    """Geometry group for split de-duplication (quantize to reduce float noise)."""
    q = [[round(float(x), 3), round(float(y), 3)] for x, y in native]
    return _hash_obj(q)


def make_sample(
    *,
    sample_id: str,
    native: list[list[float]],
    ego_v: float,
    base_speed: float,
    split: str,
    family: str,
    ambiguity: bool,
    seed: int,
    path_group: str,
    scenario_group: str,
) -> dict:
    rng = random.Random(seed)
    n = max(2, len(native))
    nom_d = [0.0] * n
    nom_ds = [0.6 + 0.05 * rng.random() for _ in range(n)]
    sign = 1.0 if (seed % 2 == 0) else -1.0
    if ambiguity:
        alt_d = [sign * min(1.8, 0.12 * i) for i in range(n)]
        alt_ds = [0.5 + 0.05 * rng.random() for _ in range(n)]
        available = True
        reason = "synthetic_defensive_corridor"
    else:
        alt_d = [0.0] * n
        alt_ds = list(nom_ds)
        available = False
        reason = "NO_ALTERNATIVE"
    return {
        "sample_id": sample_id,
        "split_id": split,
        "scenario_family": family,
        "scenario_group": scenario_group,
        "path_group": path_group,
        "ambiguity_type": "synthetic_corridor" if ambiguity else "empty_or_stop",
        "alternative_available": available,
        "availability_reason": reason,
        "ego_v": ego_v,
        "base_speed_mps": base_speed,
        "native_path_xy": native,
        "native_path_hash": _hash_obj(native),
        "nominal": {
            "raw_delta_s": nom_ds,
            "raw_d": nom_d,
            "speed_scale": 1.0,
            "head_lineage": "teacher_label",
        },
        "defensive": {
            "raw_delta_s": alt_ds,
            "raw_d": alt_d,
            "speed_scale": 0.85 if ambiguity else 1.0,
            "head_lineage": "teacher_label",
        },
        "teacher": {
            "source": "offline_frenet_lattice",
            "version": "r2x_teacher_v2",
            "privileged": False,
            "note": "geometry-only bootstrap labels; not scene-conditioned CARLA teacher",
        },
        "leakage_audit": {
            "from_r2_pilot_outcome": False,
            "from_regression": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        type=str,
        default=str(OUT),
        help="output dataset directory",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # More unique geometries: vary angle + curvature + length so train/val can split by path_group
    natives: list[tuple[str, list[list[float]], float, str]] = []
    for i in range(64):
        ang = (i % 16) * 0.12
        curv = 0.02 * ((i // 4) % 5)
        native = []
        x, y, th = 0.0, 0.0, ang
        for j in range(20):
            native.append([x, y])
            th = th + curv
            x = x + math.cos(th) * 1.2
            y = y + math.sin(th) * 1.2
        natives.append((f"syn_{i}", native, 4.0 + (i % 5) * 0.5, "synthetic"))

    # Regression: pilot geometry only, deterministic seeds
    reg_samples = []
    man_path = PILOT / "run_set_manifest.json"
    rep_path = PILOT / "run_set_report.json"
    if man_path.is_file() and rep_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        for slot, row in zip(man.get("pairs") or [], rep.get("pair_results") or []):
            if not row.get("comparable"):
                continue
            pid = row["pair_id"]
            aid = int(row["attempt_id"])
            bundle_p = (
                PILOT / "pairs" / pid / f"attempt_{aid}" / "anchor" / "anchor_bundle.json"
            )
            if not bundle_p.is_file():
                continue
            b = json.loads(bundle_p.read_text(encoding="utf-8"))
            native = _native_from_bundle(b)
            if len(native) < 2:
                continue
            pg = _path_group_key(native)
            reg_samples.append(
                make_sample(
                    sample_id=f"r2hold_{pid}_{aid}",
                    native=native,
                    ego_v=5.0,
                    base_speed=6.0,
                    split="regression",
                    family=str(slot.get("family") or "unknown"),
                    ambiguity=True,
                    seed=_stable_seed_from_str(pid, base=args.seed),
                    path_group=pg,
                    scenario_group=f"r2pilot:{pid}",
                )
            )

    # Assign unique path_groups to train vs val without overlap
    rng = random.Random(args.seed)
    # Build one sample per native geometry first, then expand with ego/speed variants
    # within the same split only.
    indexed = list(enumerate(natives))
    rng.shuffle(indexed)

    # Reserve distinct path groups for train and val
    n_need = args.n_train + args.n_val
    # Use at least as many unique geometries as samples when possible
    train: list[dict] = []
    val: list[dict] = []
    train_groups: set[str] = set()
    val_groups: set[str] = set()
    reg_groups = {_path_group_key(s["native_path_xy"]) for s in reg_samples}

    for gi, (orig_i, (name, native, spd, fam)) in enumerate(indexed):
        if len(train) >= args.n_train and len(val) >= args.n_val:
            break
        pg = _path_group_key(native)
        if pg in reg_groups:
            continue  # never put pilot geometry into train/val
        # Prefer filling train first with unique groups, then val with remaining unique
        if len(train) < args.n_train and pg not in val_groups:
            amb = (gi % 3) != 0
            sample = make_sample(
                sample_id=f"{name}_train_{len(train)}",
                native=native,
                ego_v=spd * 0.9,
                base_speed=spd,
                split="train",
                family=fam,
                ambiguity=amb,
                seed=args.seed + 1000 + len(train),
                path_group=pg,
                scenario_group=f"syn:{name}",
            )
            train.append(sample)
            train_groups.add(pg)
        elif len(val) < args.n_val and pg not in train_groups:
            amb = (gi % 3) != 0
            sample = make_sample(
                sample_id=f"{name}_val_{len(val)}",
                native=native,
                ego_v=spd * 0.9,
                base_speed=spd,
                split="val",
                family=fam,
                ambiguity=amb,
                seed=args.seed + 5000 + len(val),
                path_group=pg,
                scenario_group=f"syn:{name}",
            )
            val.append(sample)
            val_groups.add(pg)

    # If still short, create extra unique geometries rather than reusing groups
    def _extra_native(k: int) -> list[list[float]]:
        ang = 0.05 * k
        curv = 0.01 * (k % 7)
        native = []
        x, y, th = float(k) * 0.01, float(k) * 0.02, ang
        for j in range(20):
            native.append([x, y])
            th = th + curv
            x = x + math.cos(th) * 1.15
            y = y + math.sin(th) * 1.15
        return native

    k_extra = 0
    while len(train) < args.n_train or len(val) < args.n_val:
        native = _extra_native(k_extra)
        k_extra += 1
        pg = _path_group_key(native)
        if pg in train_groups or pg in val_groups or pg in reg_groups:
            continue
        if len(train) < args.n_train:
            train.append(
                make_sample(
                    sample_id=f"extra_train_{len(train)}",
                    native=native,
                    ego_v=4.5,
                    base_speed=5.0,
                    split="train",
                    family="synthetic_extra",
                    ambiguity=(len(train) % 3) != 0,
                    seed=args.seed + 9000 + len(train),
                    path_group=pg,
                    scenario_group=f"extra:{len(train)}",
                )
            )
            train_groups.add(pg)
        else:
            val.append(
                make_sample(
                    sample_id=f"extra_val_{len(val)}",
                    native=native,
                    ego_v=4.5,
                    base_speed=5.0,
                    split="val",
                    family="synthetic_extra",
                    ambiguity=(len(val) % 3) != 0,
                    seed=args.seed + 19000 + len(val),
                    path_group=pg,
                    scenario_group=f"extra_val:{len(val)}",
                )
            )
            val_groups.add(pg)

    samples = train + val + reg_samples
    samples_path = out / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, sort_keys=True) + "\n")

    path_overlap_train_val = sorted(train_groups & val_groups)
    path_overlap_train_reg = sorted(train_groups & reg_groups)
    path_overlap_val_reg = sorted(val_groups & reg_groups)
    id_inter = set(s["sample_id"] for s in train) & set(
        s["sample_id"] for s in reg_samples
    )

    leakage = {
        "schema_version": "safedrive.r2x.leakage.v2",
        "train_ids": [s["sample_id"] for s in train],
        "val_ids": [s["sample_id"] for s in val],
        "regression_ids": [s["sample_id"] for s in reg_samples],
        "train_path_groups": sorted(train_groups),
        "val_path_groups": sorted(val_groups),
        "regression_path_groups": sorted(reg_groups),
        "overlap_train_regression_ids": sorted(id_inter),
        "overlap_train_val_path_groups": path_overlap_train_val,
        "overlap_train_regression_path_groups": path_overlap_train_reg,
        "overlap_val_regression_path_groups": path_overlap_val_reg,
        "n_unique_train_paths": len(train_groups),
        "n_unique_val_paths": len(val_groups),
        "ok": (
            len(id_inter) == 0
            and len(path_overlap_train_val) == 0
            and len(path_overlap_train_reg) == 0
        ),
    }

    split_manifest = {
        "schema_version": "safedrive.r2x.split.v2",
        "counts": {
            "train": len(train),
            "val": len(val),
            "regression": len(reg_samples),
        },
        "unique_path_groups": {
            "train": len(train_groups),
            "val": len(val_groups),
            "regression": len(reg_groups),
        },
        "forbid_overlap_with_r2_pilot_outcomes": True,
        "forbid_path_group_overlap_train_val": True,
        "regression_uses_pilot_geometry_only": True,
        "seed_policy": "sha256_stable_not_python_hash",
        "samples_path": str(samples_path.as_posix()),
        "content_hash": _hash_obj([s["sample_id"] for s in samples]),
        "path_group_hash": _hash_obj(sorted(train_groups | val_groups | reg_groups)),
    }

    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out / "leakage_report.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )
    (out / "dataset_card.json").write_text(
        json.dumps(
            {
                "name": "r2x_spatial_k2_dataset_v1",
                "teacher": "offline_frenet_lattice",
                "n_samples": len(samples),
                "split_policy": "path_group_disjoint_train_val",
                "note": (
                    "Bootstrap dual labels with path-group de-duplication; "
                    "not production dual-expert CARLA scene teacher."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_manifest, indent=2))
    print(
        "leakage_ok",
        leakage["ok"],
        "n",
        len(samples),
        "train_paths",
        len(train_groups),
        "val_paths",
        len(val_groups),
    )
    return 0 if leakage["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
