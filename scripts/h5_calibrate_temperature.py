#!/usr/bin/env python3
"""Calibrate H5 ranking probability temperature on H3 dev folds only.

This script never opens H4 test labels.  It uses the same H4 normalization
stats and frozen checkpoints, then selects a temperature that minimizes ECE on
decisive dev examples while enforcing a probability floor of 0.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h3.dataset import load_examples  # noqa: E402
from data_pipeline.h3.evaluate import sigmoid  # noqa: E402
from data_pipeline.h3.model import load_model, predict_model  # noqa: E402
from data_pipeline.h4.contracts import FINAL_CHECKPOINTS, H4_CONFIG  # noqa: E402


def _ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    total = len(probs)
    if total == 0:
        return 1.0
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        indices = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not indices:
            continue
        conf = sum(probs[i] for i in indices) / len(indices)
        acc = sum(labels[i] for i in indices) / len(indices)
        ece += (len(indices) / total) * abs(acc - conf)
    return ece


def main() -> int:
    split_path = ROOT / "docs/runtime-evidence/h3/h3-v2-20260815d-final/split_manifest.json"
    split = json.loads(split_path.read_text())
    roots = [
        ROOT / "generated/h2/paired-outcomes" / H4_CONFIG["h2_dataset_id"],
        ROOT / "generated/h3/carla-challenge-v2" / H4_CONFIG["challenge_dataset_id"],
    ]
    dev = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        dev.extend(load_examples(roots, split, split=fold))

    evidence = json.loads(
        (ROOT / "docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json").read_text()
    )
    stats = {str(item["seed"]): (float(item["mean"]), float(item["std"])) for item in evidence["normalization_stats"]["items"]}
    models = {}
    for seed, info in FINAL_CHECKPOINTS.items():
        model, _ = load_model(ROOT / info["path"], device="cpu")
        models[seed] = model

    deltas = []
    labels = []
    for example in dev:
        if example.winner_index is None:
            continue
        utils = [[], []]
        for seed, model in models.items():
            pred0, pred1 = predict_model(model, example, device="cpu")
            mean, std = stats[seed]
            utils[0].append((pred0.utility - mean) / max(1e-9, std))
            utils[1].append((pred1.utility - mean) / max(1e-9, std))
        delta = sum(utils[0]) / len(utils[0]) - sum(utils[1]) / len(utils[1])
        deltas.append(delta)
        labels.append(1 if example.winner_index == 0 else 0)

    best_t = None
    best_ece = float("inf")
    results = []
    for t in [i / 100.0 for i in range(5, 501)]:  # 0.05 .. 5.00
        probs = [sigmoid(d / max(0.05, t)) for d in deltas]
        ece = _ece(probs, labels)
        results.append({"temperature": t, "ece": ece})
        if ece < best_ece - 1e-12:
            best_ece = ece
            best_t = t
    if best_t is None:
        best_t = 0.5
        best_ece = 1.0
    payload = {
        "schema_version": "safedrive.h5.temperature_calibration.v1",
        "h4_run_id": "h4-locked-20260816-final",
        "method": "dev_ece_grid",
        "temperature": float(best_t),
        "ece": float(best_ece),
        "probability_temperature_floor": 0.5,
        "n_decisive_dev": len(deltas),
        "grid_size": len(results),
        "best_grid": results,
    }
    payload["sha256"] = stable_sha256(payload)
    out = ROOT / "generated/h5/temperature_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "temperature": best_t, "ece": best_ece, "evidence": str(out), "sha256": payload["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
