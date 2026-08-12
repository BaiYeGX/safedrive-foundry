#!/usr/bin/env python3
"""Build the immutable K2 V3 semantic dataset from frozen teacher evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r2_v3_dataset import (  # noqa: E402
    build_v3_dataset_from_evidence,
    validate_v3_dataset_manifest,
)
from driving_vla.model.navigation_contract import canonical_sha256  # noqa: E402


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", required=True)
    args = parser.parse_args()

    manifest = json.loads(
        Path(args.manifest).read_text(encoding="utf-8")
    )
    stored_hash = str(manifest.get("manifest_hash") or "")
    body = dict(manifest)
    body.pop("manifest_hash", None)
    if stored_hash != canonical_sha256(body):
        raise ValueError("calibration manifest hash mismatch")
    manifest_audit = validate_v3_dataset_manifest(manifest)
    rows, audit = build_v3_dataset_from_evidence(
        manifest=manifest,
        evidence_root=Path(args.evidence_root),
    )
    row_ids = {str(row["sample_id"]) for row in rows}
    if row_ids != manifest_audit["slot_ids"]:
        raise ValueError("dataset rows do not exactly cover frozen manifest slots")
    dataset_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    audit.update(
        {
            "phase": manifest_audit["phase"],
            "dataset_sha256": hashlib.sha256(
                dataset_text.encode("utf-8")
            ).hexdigest(),
            "exact_manifest_coverage": True,
        }
    )
    _write_exclusive(
        Path(args.out),
        dataset_text,
    )
    _write_exclusive(
        Path(args.audit_out),
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "status": "DATASET_BUILT",
                "dataset": str(Path(args.out)),
                "audit": str(Path(args.audit_out)),
                **audit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
