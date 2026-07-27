#!/usr/bin/env python3
"""Build balanced development data after retiring the observed v4 exam."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.development_split import (  # noqa: E402
    assign_balanced_episode_split,
)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "docs/runtime-evidence/r2x-training/dataset-v6-formal/samples.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(
            ROOT
            / "docs/runtime-evidence/r2x-training/dataset-v7-development"
        ),
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = Path(args.source)
    out = Path(args.out)
    if (out / "samples.jsonl").is_file() and not args.force:
        print(f"exists {out}; use --force", flush=True)
        return 0
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source_rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows, split_manifest = assign_balanced_episode_split(
        source_rows, seed=args.seed
    )
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
    card = {
        "schema_version": "safedrive.r2x.dataset.v7_development",
        "status": "DEVELOPMENT_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source.as_posix()),
        "source_sha256": _file_sha(source),
        "samples_sha256": _file_sha(samples),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
