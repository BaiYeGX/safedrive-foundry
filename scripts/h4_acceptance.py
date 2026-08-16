#!/usr/bin/env python3
"""Verify H4 locked-evaluation evidence hash and gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-gate-pass", action="store_true", default=True)
    args = parser.parse_args()
    path = ROOT / "docs" / "runtime-evidence" / "h4" / args.run_id / "final-delivery.json"
    if not path.exists():
        print(json.dumps({"ok": False, "error": f"EVIDENCE_MISSING:{path}"}, sort_keys=True))
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("evidence_sha256")
    body = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    actual = stable_sha256(body)
    gate_passed = bool(payload.get("gate", {}).get("passed", False))
    failures = list(payload.get("gate", {}).get("failures", []))
    ok = actual == expected and (not args.require_gate_pass or gate_passed)
    print(json.dumps({"ok": ok, "run_id": args.run_id, "evidence_sha256_valid": actual == expected,
                      "gate_status": payload.get("gate_status"), "failures": failures,
                      "evidence": str(path)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
