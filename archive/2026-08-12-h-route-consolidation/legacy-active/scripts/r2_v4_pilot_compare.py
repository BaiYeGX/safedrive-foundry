#!/usr/bin/env python3
"""Run the four pre-registered R2 V4 representation pilot comparisons.

The pilot is deliberately a development operation: it may read train/val
rows from the 144-anchor manifest, but it refuses any Town13/``test`` row and
never writes a formal checkpoint.  The resulting report is the only input to
the first V4 head repair decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "scripts"))

from r2_v4_train_heads import TOKEN_MODES, load_rows, train  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pilot_only(rows: list[Any]) -> None:
    if not rows:
        raise ValueError("pilot dataset is empty")
    if any(str(row.split).lower() == "test" for row in rows):
        raise ValueError("V4 pilot must not contain a locked test split")
    town13 = [row for row in rows if "town13" in row.group.lower()]
    if town13:
        raise ValueError("V4 pilot must not read Town13/root groups")
    if len(rows) != 144:
        raise ValueError(f"V4 pilot must contain exactly 144 anchors, got {len(rows)}")
    if not {row.split for row in rows}.issubset({"train", "val"}):
        raise ValueError("V4 pilot split must be train/val only")


def compare(args: Namespace) -> dict[str, Any]:
    data = Path(args.data).resolve()
    rows = load_rows(data)
    _pilot_only(rows)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for mode in TOKEN_MODES:
        run_name = f"pilot144-{mode.replace('+', '-')}-seed-{int(args.seed)}"
        train_args = Namespace(
            data=str(data),
            output_root=str(output_root),
            run_name=run_name,
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            dropout=float(args.dropout),
            eval_every=int(args.eval_every),
            patience_evals=int(args.patience_evals),
            seed=int(args.seed),
            device=str(args.device),
            overfit_32=False,
            evaluate_test=False,
            token_mode=mode,
            group_balanced=bool(args.group_balanced),
            class_balanced=bool(args.class_balanced),
            availability_threshold=float(getattr(args, "availability_threshold", 0.5)),
        )
        report = train(train_args)
        reports[mode] = report

    val_scores = {mode: float(report["val"].get("macro_f1", 0.0)) for mode, report in reports.items()}
    structured = val_scores["structured-only"]
    mean64 = val_scores["mean64"]
    token = val_scores["token-aware"]
    full = val_scores["token-aware+history"]
    gates = {
        "full_ood_macro_f1": full >= 0.75,
        "full_no_kind_recall_below_0_50": min(
            float(value) for value in reports["token-aware+history"]["val"].get("kind_recall", {}).values()
        ) >= 0.50,
        "token_aware_vs_mean64": token - mean64 >= 0.10,
        "token_aware_vs_structured": token - structured >= 0.03,
    }
    result = {
        "schema_version": "safedrive.r2_v4.pilot_compare.v1",
        "data": str(data),
        "data_sha256": _sha256(data),
        "anchor_count": len(rows),
        "town13_read": False,
        "seed": int(args.seed),
        "val_macro_f1": val_scores,
        "reports": reports,
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "selected_mode": "token-aware+history" if all(gates.values()) else None,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-balanced", action="store_true")
    parser.add_argument("--class-balanced", action="store_true")
    parser.add_argument("--availability-threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = compare(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pilot_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
