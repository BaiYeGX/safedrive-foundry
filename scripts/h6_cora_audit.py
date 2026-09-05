#!/usr/bin/env python3
"""Audit C2 root integrity, leakage, coverage and frozen gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import stable_sha256  # noqa: E402
from data_pipeline.h6.cora.config import CORA_C2_CONFIG  # noqa: E402
from data_pipeline.h6.cora.quality import audit_cora_dataset  # noqa: E402
from data_pipeline.h6.cora.run_lock import verify_run_lock  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit(dataset_id: str, scope: str, *, trust_pilot_manifest: bool = False) -> dict:
    repair_id = "h6-cora-c2-repair-20260905-v2"
    if dataset_id == repair_id:
        from data_pipeline.h6.cora.repair import audit as repair_audit
        dataset = ROOT / "generated" / "h6" / "cora" / dataset_id
        payload = repair_audit(dataset)
        evidence = ROOT / "docs" / "runtime-evidence" / "h6" / dataset_id / "data-quality.json"
        _atomic_json(evidence, payload)
        return {"ok": True, "passed": bool(payload["passed"]), "failures": payload["failures"],
                "evidence": str(evidence), "data_quality_sha256": payload.get("final_delivery_sha256")}
    if dataset_id != str(CORA_C2_CONFIG["dataset_id"]):
        raise ValueError("cora_audit_dataset_not_frozen")
    dataset = ROOT / "generated" / "h6" / "cora" / dataset_id
    lock = json.loads((dataset / "run-lock.json").read_text(encoding="utf-8"))
    lock_valid, lock_failures = verify_run_lock(lock)
    if trust_pilot_manifest:
        if scope != "development":
            raise ValueError("cora_trusted_manifest_scope")
        pilot_path = ROOT / "docs" / "runtime-evidence" / "h6" / dataset_id / "data-quality-pilot.json"
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if not (
            bool(pilot.get("passed"))
            and bool(pilot.get("metrics", {}).get("manifest_valid"))
            and bool(pilot.get("public_label_audit", {}).get("passed"))
            and bool(pilot.get("feature_audit", {}).get("passed"))
            and bool(pilot.get("inventory_audit", {}).get("passed"))
        ):
            raise ValueError("cora_pilot_manifest_not_reusable")
    payload = audit_cora_dataset(
        dataset,
        scope=scope,
        trust_pilot_manifest=trust_pilot_manifest,
    )
    payload["run_lock"] = {
        "passed": lock_valid,
        "failures": list(lock_failures),
        "run_lock_sha256": lock.get("run_lock_sha256"),
    }
    failures = list(payload.get("failures", ()))
    if not lock_valid:
        failures.append("run_lock")
    payload["failures"] = list(dict.fromkeys(failures))
    payload["passed"] = not payload["failures"]
    payload.pop("data_quality_sha256", None)
    payload["data_quality_sha256"] = stable_sha256(payload)
    evidence = ROOT / "docs" / "runtime-evidence" / "h6" / dataset_id / f"data-quality-{scope}.json"
    _atomic_json(evidence, payload)
    if scope == "development":
        _atomic_json(evidence.parent / "data-quality.json", payload)
    return {
        "ok": True,
        "passed": payload["passed"],
        "failures": payload["failures"],
        "evidence": str(evidence),
        "data_quality_sha256": payload["data_quality_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=str(CORA_C2_CONFIG["dataset_id"]))
    parser.add_argument("--scope", required=True, choices=("pilot", "development"))
    parser.add_argument(
        "--trust-pilot-manifest",
        action="store_true",
        help="For development only, reuse the immediately preceding passed pilot manifest audit.",
    )
    args = parser.parse_args()
    try:
        result = audit(
            args.dataset_id,
            args.scope,
            trust_pilot_manifest=args.trust_pilot_manifest,
        )
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
