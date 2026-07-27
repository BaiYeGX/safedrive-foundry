#!/usr/bin/env python3
"""R2-X2 scene-conditioned dual-label teacher (bootstrap without CARLA live).

Teacher selection may use privileged-style scenario parameters (conflict side,
clearance, TTC) to *choose* the defensive residual. Head features are only
runtime-matched ``scene_proxy_feature`` (same function used at inference).

This is NOT geometry-only seed-parity; defensive lateral sign follows conflict
side so the head can learn "yield away from cut-in".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path_insert = str(ROOT / "safedrive_foundry")
import sys

sys.path.insert(0, sys_path_insert)

from driving_vla.model.driving_feature import (  # noqa: E402
    feature_vector_hash,
    scene_proxy_feature,
)

OUT = ROOT / "docs" / "runtime-evidence" / "r2x-training" / "dataset-v2"


def _hash_obj(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _stable_seed(s: str, base: int = 0) -> int:
    h = hashlib.sha256(f"{base}:{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _path_group_key(native: list[list[float]]) -> str:
    q = [[round(float(x), 3), round(float(y), 3)] for x, y in native]
    return _hash_obj(q)


def _make_native(i: int) -> list[list[float]]:
    """Unique geometry per index (origin offset prevents path_group exhaustion)."""
    ang = (i % 16) * 0.11
    curv = 0.015 * ((i // 3) % 6)
    step = 1.15 + 0.01 * (i % 5)
    native: list[list[float]] = []
    # unique origin so train/val never exhaust a small geometry lattice
    x, y, th = float(i) * 0.37, float(i) * 0.19, ang + 0.003 * i
    for _ in range(20):
        native.append([x, y])
        th = th + curv
        x = x + math.cos(th) * step
        y = y + math.sin(th) * step
    return native


def teacher_residuals(
    *,
    n: int,
    ambiguity: bool,
    conflict_side: str,
    clearance_m: float,
    ttc_s: float,
    rng: random.Random,
) -> tuple[dict, dict, bool, str]:
    """Fixed-slot labels: nominal≈0 lateral; defensive away from conflict.

    Privileged-style params select the label; they are stored under teacher
    only, not in driving_feature.
    """
    nom_d = [0.0] * n
    nom_ds = [0.55 + 0.05 * rng.random() for _ in range(n)]
    if not ambiguity:
        return (
            {
                "raw_delta_s": nom_ds,
                "raw_d": nom_d,
                "speed_scale": 1.0,
                "head_lineage": "teacher_label",
            },
            {
                "raw_delta_s": list(nom_ds),
                "raw_d": [0.0] * n,
                "speed_scale": 1.0,
                "head_lineage": "teacher_label",
            },
            False,
            "NO_ALTERNATIVE",
        )

    # defensive direction: away from conflict
    if conflict_side == "left":
        sign = -1.0  # shift right
    elif conflict_side == "right":
        sign = 1.0  # shift left
    elif conflict_side == "center":
        # lead brake: prefer mild right + stronger slowdown
        sign = 1.0 if rng.random() > 0.5 else -1.0
    else:
        sign = 1.0

    # amplitude scales with urgency (privileged for teacher only)
    urgency = 1.0
    if ttc_s < 3.0 or clearance_m < 6.0:
        urgency = 1.4
    elif ttc_s > 10.0 and clearance_m > 15.0:
        urgency = 0.7

    amp = min(1.8, 0.14 * urgency)
    alt_d = [sign * min(1.8, amp * i) for i in range(n)]
    alt_ds = [0.45 + 0.05 * rng.random() for _ in range(n)]
    speed = 0.75 if urgency >= 1.2 else 0.85
    return (
        {
            "raw_delta_s": nom_ds,
            "raw_d": nom_d,
            "speed_scale": 1.0,
            "head_lineage": "teacher_label",
        },
        {
            "raw_delta_s": alt_ds,
            "raw_d": alt_d,
            "speed_scale": speed,
            "head_lineage": "teacher_label",
        },
        True,
        "scene_defensive_corridor",
    )


def make_sample(
    *,
    sample_id: str,
    native: list[list[float]],
    split: str,
    family: str,
    seed: int,
    path_group: str,
    ambiguity: bool,
) -> dict:
    rng = random.Random(seed)
    # scenario layout (observable at runtime via relative pose)
    if family.startswith("cut"):
        side = "left" if (seed % 2 == 0) else "right"
        rel_x, rel_y = 12.0, (3.5 if side == "left" else -3.5)
        clearance = 5.0 + rng.random() * 4.0
        ttc = 2.5 + rng.random() * 2.0
        actor_v = 6.0
    elif family.startswith("lead"):
        side = "center"
        rel_x, rel_y = 10.0 + rng.random() * 5.0, 0.2 * (rng.random() - 0.5)
        clearance = 6.0 + rng.random() * 6.0
        ttc = 3.0 + rng.random() * 4.0
        actor_v = 3.0
    elif family.startswith("cross"):
        side = "left" if seed % 3 == 0 else "right"
        rel_x, rel_y = 8.0, (6.0 if side == "left" else -6.0)
        clearance = 7.0 + rng.random() * 5.0
        ttc = 4.0 + rng.random() * 3.0
        actor_v = 5.0
    else:
        side = "none"
        rel_x, rel_y = 0.0, 0.0
        clearance = 25.0
        ttc = 30.0
        actor_v = 0.0

    ego_v = 4.0 + (seed % 5) * 0.4
    base_speed = ego_v / 0.9
    nom, alt, available, reason = teacher_residuals(
        n=len(native),
        ambiguity=ambiguity and side != "none",
        conflict_side=side,
        clearance_m=clearance,
        ttc_s=ttc,
        rng=rng,
    )
    # Runtime-matched feature (NO privileged future; only current observables)
    drive = scene_proxy_feature(
        conflict_side=side if available or side != "none" else "none",
        actor_rel_x_m=rel_x,
        actor_rel_y_m=rel_y,
        actor_speed_mps=actor_v,
        clearance_m=clearance,
        ttc_s=ttc,
        ego_v=ego_v,
        scenario_family=family,
    )
    return {
        "sample_id": sample_id,
        "split_id": split,
        "scenario_family": family,
        "scenario_group": f"{family}:{path_group[:8]}",
        "path_group": path_group,
        "ambiguity_type": "scene_conflict" if available else "empty_or_no_alt",
        "alternative_available": available,
        "availability_reason": reason,
        "conflict_side": side,
        "ego_v": ego_v,
        "base_speed_mps": base_speed,
        "native_path_xy": native,
        "native_path_hash": _hash_obj(native),
        "observables": {
            "conflict_side": side,
            "actor_rel_x_m": rel_x,
            "actor_rel_y_m": rel_y,
            "actor_speed_mps": actor_v,
            "clearance_m": clearance,
            "ttc_s": ttc,
        },
        "driving_feature": drive,
        "driving_feature_hash": feature_vector_hash(drive),
        "driving_feature_source": "scene_proxy_v1",
        "nominal": nom,
        "defensive": alt,
        "teacher": {
            "source": "scene_conditioned_frenet",
            "version": "r2x_teacher_scene_v1",
            "privileged_for_label_only": True,
            "privileged_fields": ["urgency_from_ttc_clearance"],
            "note": (
                "Privileged urgency selects amplitude; head sees only observables "
                "via scene_proxy_feature (same at train and runtime)."
            ),
        },
        "leakage_audit": {
            "from_r2_pilot_outcome": False,
            "from_regression": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ~30% empty for availability specificity; rest conflict families
    families = [
        "cut_in",
        "cut_in",
        "lead_brake",
        "lead_brake",
        "crossing",
        "crossing",
        "empty",
        "empty",
        "empty",
    ]
    train, val = [], []
    train_groups: set[str] = set()
    val_groups: set[str] = set()
    # Fill train and val from interleaved unique indices (even→train, odd→val)
    # until both quotas met, guaranteeing path_group disjointness.
    i = 0
    while len(train) < args.n_train or len(val) < args.n_val:
        native = _make_native(i)
        pg = _path_group_key(native)
        want_train = len(train) < args.n_train
        want_val = len(val) < args.n_val
        if pg in train_groups or pg in val_groups:
            i += 1
            if i > 20000:
                break
            continue
        fill_val = want_val and (
            not want_train
            or (len(val) / max(args.n_val, 1)) <= (len(train) / max(args.n_train, 1))
        )
        if fill_val and want_val:
            # family mix inside val (same schedule as train)
            fam = families[len(val) % len(families)]
            amb = fam != "empty" and (len(val) % 5) != 0
            val.append(
                make_sample(
                    sample_id=f"sc_val_{len(val)}",
                    native=native,
                    split="val",
                    family=fam,
                    seed=args.seed + 5000 + len(val),
                    path_group=pg,
                    ambiguity=amb,
                )
            )
            val_groups.add(pg)
        elif want_train:
            fam = families[len(train) % len(families)]
            amb = fam != "empty" and (len(train) % 5) != 0
            train.append(
                make_sample(
                    sample_id=f"sc_train_{len(train)}",
                    native=native,
                    split="train",
                    family=fam,
                    seed=args.seed + 1000 + len(train),
                    path_group=pg,
                    ambiguity=amb,
                )
            )
            train_groups.add(pg)
        i += 1
        if i > 20000:
            break

    samples = train + val
    samples_path = out / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, sort_keys=True) + "\n")

    overlap = sorted(train_groups & val_groups)
    leakage = {
        "schema_version": "safedrive.r2x.leakage.v2",
        "train_ids": [s["sample_id"] for s in train],
        "val_ids": [s["sample_id"] for s in val],
        "train_path_groups": sorted(train_groups),
        "val_path_groups": sorted(val_groups),
        "overlap_train_val_path_groups": overlap,
        "n_unique_train_paths": len(train_groups),
        "n_unique_val_paths": len(val_groups),
        "ok": len(overlap) == 0,
    }
    split_manifest = {
        "schema_version": "safedrive.r2x.split.v2",
        "counts": {"train": len(train), "val": len(val)},
        "unique_path_groups": {
            "train": len(train_groups),
            "val": len(val_groups),
        },
        "forbid_path_group_overlap_train_val": True,
        "seed_policy": "sha256_stable",
        "teacher": "scene_conditioned_frenet_v1",
        "feature": "scene_proxy_v1",
        "samples_path": str(samples_path.as_posix()),
        "content_hash": _hash_obj([s["sample_id"] for s in samples]),
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
                "name": "r2x_spatial_k2_dataset_v2_scene",
                "teacher": "scene_conditioned_frenet",
                "feature": "scene_proxy_matched_train_runtime",
                "n_samples": len(samples),
                "note": "Bootstrap scene teacher; replace with CARLA-collected when available.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    n_elig = sum(1 for s in samples if s["alternative_available"])
    print(json.dumps(split_manifest, indent=2))
    print("leakage_ok", leakage["ok"], "n", len(samples), "eligible", n_elig)
    return 0 if leakage["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
