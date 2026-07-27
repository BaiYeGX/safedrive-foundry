#!/usr/bin/env python3
"""Promote a trained head to formal-OK only after blind isolation + offline gates.

Does not invent selection-space. Binds the checkpoint to one frozen, previously
unobserved registry and refuses overlap with training samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.checkpoint_contract import (  # noqa: E402
    write_checkpoint_manifest,
    file_sha256,
)
from driving_vla.evaluation.runner_contract import require_frozen_registry  # noqa: E402


def _fsha(p: Path) -> str:
    return file_sha256(p, full=True)


def _dataset_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        scenario_id = str(row.get("scenario_id") or "").strip()
        seed_id = str(row.get("seed_id") or "").strip()
        if not scenario_id or not seed_id:
            raise ValueError(f"dataset row {line_no} missing scenario_id/seed_id")
        pairs.add((scenario_id, seed_id))
    return pairs


def evaluate_formal_inputs(
    *,
    offline: dict,
    card: dict,
    train_pairs: set[tuple[str, str]],
    blind_pairs: set[tuple[str, str]],
    registry_version: str,
) -> dict:
    learned = dict(offline.get("learned_confidence_diagnostic") or {})
    eligible_guard = float(offline.get("eligible_guard_ok_rate") or 0.0)
    eligible_spatial = float(offline.get("eligible_spatial_sep_rate") or 0.0)
    eligible_valid = float(offline.get("eligible_proposal_valid_rate") or 0.0)
    recall = float(learned.get("recall") or 0.0)
    specificity = float(learned.get("specificity") or 0.0)
    overlap = sorted(train_pairs & blind_pairs)
    gates = {
        "offline_schema_v3": str(offline.get("schema_version")) == "safedrive.r2x.offline_exec.v3",
        "head_status_ok": str(offline.get("head_status") or "") == "OK",
        "eligible_guard_ok": eligible_guard >= 0.90,
        "eligible_spatial_ok": eligible_spatial >= 0.70,
        "eligible_proposal_valid_ok": eligible_valid >= 0.70,
        "new_blind_registry": registry_version != "v1",
        "blind_pair_overlap_zero": not overlap,
        "dataset_requires_new_blind": bool(card.get("new_blind_exam_registry_required")),
        "dataset_not_pilot_authorized": not bool(card.get("r2k_pilot_allowed")),
    }
    return {
        "ok": all(gates.values()),
        "gates": gates,
        "overlap": [{"scenario_id": s, "seed_id": d} for s, d in overlap],
        "eligible_guard_ok_rate": eligible_guard,
        "eligible_spatial_sep_rate": eligible_spatial,
        "eligible_proposal_valid_rate": eligible_valid,
        "learned_confidence_recall": recall,
        "learned_confidence_specificity": specificity,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--offline-report", required=True)
    ap.add_argument("--dataset-card", default="")
    ap.add_argument("--blind-registry", required=True)
    ap.add_argument("--registry-manifest", required=True)
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    data = Path(args.data)
    off = Path(args.offline_report)
    if not ckpt.is_file() or not data.is_file() or not off.is_file():
        print("missing inputs", flush=True)
        return 2

    offline = json.loads(off.read_text(encoding="utf-8"))
    card: dict = {}
    if args.dataset_card and Path(args.dataset_card).is_file():
        card = json.loads(Path(args.dataset_card).read_text(encoding="utf-8"))
    registry, freeze_audit, _ = require_frozen_registry(
        args.blind_registry,
        manifest_path=args.registry_manifest,
    )
    audit = evaluate_formal_inputs(
        offline=offline,
        card=card,
        train_pairs=_dataset_pairs(data),
        blind_pairs=set(registry.pairs()),
        registry_version=registry.registry_version,
    )
    if not audit["ok"]:
        print(json.dumps({"status": "REFUSE", "reason": "formal_input_gate", "audit": audit}, indent=2))
        return 3

    eg = float(audit["eligible_guard_ok_rate"])
    spec = float(audit["learned_confidence_specificity"])
    recall = float(audit["learned_confidence_recall"])
    head_status = str(offline.get("head_status") or "")

    formal_allowed = [
        "formal_offline",
        "x5h_acceptance",
        "r2k_pilot",
        "offline_diagnostic",
        "historical_comparison",
    ]
    extra = {
        "data_path": str(data.as_posix()),
        "data_sha256": _fsha(data),
        "offline_report": str(off.as_posix()),
        "offline_report_sha256": _fsha(off),
        "offline_head_status": head_status,
        "eligible_guard_ok_rate": eg,
        "availability_specificity": spec,
        "availability_recall": recall,
        "eligible_spatial_sep_rate": audit["eligible_spatial_sep_rate"],
        "eligible_proposal_valid_rate": audit["eligible_proposal_valid_rate"],
        "pass_spatial": offline.get("pass_spatial"),
        "pass_availability": offline.get("pass_availability"),
        "pass_quality": offline.get("pass_quality"),
        "dataset_card": str(Path(args.dataset_card).as_posix()) if args.dataset_card else "",
        "blind_registry_path": str(Path(args.blind_registry).as_posix()),
        "blind_registry_manifest": str(Path(args.registry_manifest).as_posix()),
        "blind_registry_sha256": freeze_audit["registry_sha256"],
        "blind_registry_version": registry.registry_version,
        "blind_pair_overlap_zero": True,
        "formal_input_audit": audit,
        "forbid_r2_pilot_in_train": True,
        "collapse_sep_m": 0.50,
        "formal_note": (
            "OK authorizes formal offline/X5H/R2-K measurement under frozen 0.50m "
            "gate; does not claim selection space without pilot Oracle labels"
        ),
        "run_name": args.run_name,
    }
    man = write_checkpoint_manifest(
        ckpt.parent / "CHECKPOINT_STATUS.json",
        checkpoint_path=ckpt,
        status="OK",
        allowed_uses=formal_allowed,
        forbidden_uses=[],
        reasons=[
            "new_frozen_blind_registry",
            "blind_pair_overlap_zero",
            "offline_v3_gates_met",
            "sha256_bound",
            f"offline_head_status={head_status}",
        ],
        extra=extra,
    )
    # Root alias for smoke / r2k entry
    root = ROOT / "docs/runtime-evidence/r2x-training/checkpoints"
    alias = root / "FORMAL_HEAD_REQUIRED.pt"
    alias.write_bytes(ckpt.read_bytes())
    write_checkpoint_manifest(
        root / "CHECKPOINT_STATUS.json",
        checkpoint_path=alias,
        status="OK",
        allowed_uses=formal_allowed,
        forbidden_uses=[],
        reasons=[
            "formal_alias_blind_registry_bound",
            "new_frozen_blind_registry",
            "blind_pair_overlap_zero",
            "offline_v3_gates_met",
            f"source={ckpt.as_posix()}",
        ],
        extra={**extra, "source_checkpoint": str(ckpt.as_posix())},
    )
    print(json.dumps({"status": "OK", "manifest": man, "alias": str(alias)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
