#!/usr/bin/env python3
"""Authorize a trained Spatial head for development CARLA smoke only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.checkpoint_contract import (  # noqa: E402
    file_sha256,
    write_checkpoint_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offline-report", required=True)
    parser.add_argument("--dataset-card", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    report_path = Path(args.offline_report)
    card_path = Path(args.dataset_card)
    if not all(path.is_file() for path in (checkpoint, report_path, card_path)):
        print("missing input", flush=True)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    gates_ok = all(
        bool(report.get(field))
        for field in (
            "pass_quality",
            "pass_spatial",
            "pass_proposal_validity",
            "pipeline_contract_ok",
        )
    )
    if (
        report.get("schema_version") != "safedrive.r2x.offline_exec.v3"
        or report.get("availability_semantics") != "executability_only_v1"
        or not gates_ok
    ):
        print(
            json.dumps(
                {"status": "REFUSE", "reason": "offline_v3_gates_not_met"},
                indent=2,
            )
        )
        return 3
    if bool(card.get("formal_train_candidate")) or bool(
        card.get("r2k_pilot_allowed")
    ):
        print(
            json.dumps(
                {"status": "REFUSE", "reason": "expected_development_only_data"},
                indent=2,
            )
        )
        return 4
    manifest = write_checkpoint_manifest(
        checkpoint.parent / "CHECKPOINT_STATUS.json",
        checkpoint_path=checkpoint,
        status="HEAD_TRAINED_NOT_FORMAL",
        allowed_uses=[
            "offline_diagnostic",
            "historical_comparison",
            "development_live_smoke",
        ],
        forbidden_uses=[
            "formal_offline",
            "x5h_acceptance",
            "r2k_pilot",
        ],
        reasons=[
            "offline_v3_executability_gates_met",
            "development_live_smoke_only",
            "not_formal_not_r2k",
        ],
        extra={
            "offline_report": str(report_path.as_posix()),
            "offline_report_sha256": file_sha256(report_path),
            "dataset_card": str(card_path.as_posix()),
            "dataset_card_sha256": file_sha256(card_path),
            "availability_semantics": "executability_only_v1",
            "formal_eligible": False,
            "r2k_pilot_allowed": False,
        },
    )
    print(json.dumps({"status": "OK", "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
