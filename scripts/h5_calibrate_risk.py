#!/usr/bin/env python3
"""Calibrate the H5 risk defer threshold on H3 dev folds only.

The risk head is a separate binary head; its logits are not temperature-
calibrated like the pairwise ranking head.  This script chooses a probability
threshold that separates hard-unsafe dev candidates from safe candidates, and
writes the result to ``generated/h5/risk_calibration.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.dataset import load_examples  # noqa: E402
from data_pipeline.h3.evaluate import sigmoid  # noqa: E402
from data_pipeline.h3.model import load_model, predict_model  # noqa: E402
from data_pipeline.h4.contracts import (  # noqa: E402
    FINAL_CHECKPOINTS,
    H4_CONFIG,
)
from data_pipeline.h3.contracts import stable_sha256  # noqa: E402


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

    models = []
    for seed, info in FINAL_CHECKPOINTS.items():
        model, _ = load_model(ROOT / info["path"], device="cpu")
        models.append(model)

    safe = []
    unsafe = []
    for example in dev:
        for index, candidate in enumerate(example.candidates):
            logits = []
            for model in models:
                p0, p1 = predict_model(model, example, device="cpu")
                logits.append((p0 if index == 0 else p1).risk_logit)
            risk_logit = sum(logits) / len(logits)
            probability = sigmoid(risk_logit)
            item = {
                "pair_id": example.pair_id,
                "candidate_index": index,
                "risk_logit": float(risk_logit),
                "risk_probability": float(probability),
                "hard_unsafe": bool(candidate.risk),
            }
            (unsafe if candidate.risk else safe).append(item)

    if not unsafe:
        threshold = 1.0
        method = "no_unsafe_dev_examples"
    elif safe:
        max_safe = max(item["risk_probability"] for item in safe)
        min_unsafe = min(item["risk_probability"] for item in unsafe)
        if max_safe < min_unsafe:
            threshold = (max_safe + min_unsafe) / 2.0
            method = "separable_midpoint"
        else:
            # No clean separation; choose threshold maximizing F1 over a grid.
            best, best_f1 = 0.5, -1.0
            for t in [i / 100.0 for i in range(1, 100)]:
                tp = sum(item["risk_probability"] > t for item in unsafe)
                fp = sum(item["risk_probability"] > t for item in safe)
                fn = len(unsafe) - tp
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-9, precision + recall)
                if f1 > best_f1:
                    best, best_f1 = t, f1
            threshold = best
            method = "f1_grid"
    else:
        threshold = 0.0
        method = "all_dev_safe"

    payload = {
        "schema_version": "safedrive.h5.risk_calibration.v1",
        "h4_run_id": "h4-locked-20260816-final",
        "method": method,
        "risk_defer_probability": float(threshold),
        "dev_candidates": len(safe) + len(unsafe),
        "safe_candidates": len(safe),
        "unsafe_candidates": len(unsafe),
        "max_safe_probability": max((item["risk_probability"] for item in safe), default=None),
        "min_unsafe_probability": min((item["risk_probability"] for item in unsafe), default=None),
        "items": safe + unsafe,
    }
    payload["sha256"] = stable_sha256(payload)
    out = ROOT / "generated/h5/risk_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "threshold": threshold, "method": method, "evidence": str(out), "sha256": payload["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
