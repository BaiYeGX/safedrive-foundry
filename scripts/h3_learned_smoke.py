#!/usr/bin/env python3
"""H3.1 training-path smoke on dev only.

Trains a tiny learned-scene-gate model for a bounded number of epochs and
checks that the new H3.1 options (learned gate, risk-aware ranking, natural
context ablation) work end-to-end.  This script never reads test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.dataset import load_examples  # noqa: E402
from data_pipeline.h3.evaluate import prediction_rows  # noqa: E402
from data_pipeline.h3.model import load_model, train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--risk-ranking-weight", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/runtime-evidence/h3/h3-learned-smoke.json")
    args = parser.parse_args()

    split = json.loads((ROOT / "docs/runtime-evidence/h3/h3-v2-20260815d-final/split_manifest.json").read_text())
    roots = [
        ROOT / "generated/h2/paired-outcomes/h2-gatepass-20260813-routefix",
        ROOT / "generated/h3/carla-challenge-v2/h3-challenge-v2-20260815d-dev",
    ]
    dev = []
    for fold in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        dev.extend(load_examples(roots, split, split=fold))
    train = dev[:12]
    val = dev[12:16]
    checkpoint = args.output.with_suffix(".pt")

    started = time.time()
    result = train_model(
        train, val,
        seed=123,
        checkpoint_path=checkpoint,
        device="cpu",
        max_epochs=args.epochs,
        patience=args.epochs,
        scene_gate_mode="learned",
        risk_ranking_weight=args.risk_ranking_weight,
    )
    model, metadata = load_model(checkpoint, device="cpu")
    natural_rows = prediction_rows([model], val, device="cpu", context_mask_mode="actors")

    payload = {
        "schema_version": "safedrive.h3.learned_smoke.v1",
        "ok": True,
        "epochs": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "scene_gate_mode": metadata.get("scene_gate_mode"),
        "risk_ranking_weight": metadata.get("risk_ranking_weight"),
        "train_examples": len(train),
        "val_examples": len(val),
        "natural_actor_ablation_rows": len(natural_rows),
        "wall_time_s": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
