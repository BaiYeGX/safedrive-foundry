#!/usr/bin/env python3
"""Freeze one selected R2 V4 checkpoint for the blind live gates.

The training checkpoint is deliberately not formal while val selection is in
progress.  After exactly one Town13 evaluation of the selected checkpoint,
this transition grants only the blind-audit use.  Final ``OK`` promotion stays
in ``r2_v4_promote_formal.py`` and still requires smoke/core/audit evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.checkpoint_contract import (  # noqa: E402
    STATUS_R2V4_PENDING_BLIND,
    write_checkpoint_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path, condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"{path.name}: {message}")


def freeze(checkpoint: Path, formal_report: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    formal_report = formal_report.resolve()
    _require(checkpoint, checkpoint.is_file(), "checkpoint file missing")
    report = _read(formal_report)
    checkpoint_hash = _sha(checkpoint)
    rows = list(report.get("reports") or [])
    _require(
        formal_report,
        bool(report.get("locked_test_opened_once")),
        "Town13 locked test was not opened",
    )
    _require(
        formal_report,
        int(report.get("test_evaluation_count", 0)) == 1
        and sum(1 for row in rows if row.get("test") is not None) == 1,
        "Town13 must be evaluated exactly once",
    )
    _require(formal_report, bool(report.get("all_hard_gates_pass")), "formal gates failed")
    bound = [row for row in rows if str(row.get("checkpoint_sha256") or "").lower() == checkpoint_hash]
    _require(formal_report, len(bound) == 1, "formal report is not bound to exactly one checkpoint")

    status_path = checkpoint.parent / "CHECKPOINT_STATUS.json"
    if status_path.exists():
        existing = _read(status_path)
        old_hash = str(existing.get("checkpoint_sha256") or "").lower()
        _require(status_path, old_hash == checkpoint_hash, "status/checkpoint hash mismatch")
        if str(existing.get("status") or "") == "OK":
            raise ValueError("checkpoint is already final OK; refusing pending-blind downgrade")
        backup = checkpoint.parent / "CHECKPOINT_STATUS.pre_pending_blind.json"
        if not backup.exists():
            backup.write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = write_checkpoint_manifest(
        status_path,
        checkpoint_path=checkpoint,
        status=STATUS_R2V4_PENDING_BLIND,
        allowed_uses=["r2v4_blind_audit"],
        forbidden_uses=["r2v4_formal", "r3_final_head_formal", "world_campaign"],
        reasons=["Town06 val selection and exactly one Town13 test passed; blind gates pending"],
        extra={
            "formal_eval_sha256": _sha(formal_report),
            "selected_checkpoint_sha256": checkpoint_hash,
            "locked_test_opened_once": True,
        },
    )
    result = {
        "schema_version": "safedrive.r2_v4.pending_blind.v1",
        "status": manifest["status"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "formal_eval_sha256": _sha(formal_report),
        "output": str(status_path),
    }
    output = checkpoint.parent / "PENDING_BLIND_PROMOTION.json"
    if output.exists():
        old = _read(output)
        _require(output, str(old.get("checkpoint_sha256") or "") == checkpoint_hash, "existing pending record mismatch")
    else:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formal-report", required=True)
    args = parser.parse_args()
    result = freeze(Path(args.checkpoint), Path(args.formal_report))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
