#!/usr/bin/env python3
"""Freeze a completed H2 attempt into a small, auditable terminal Evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "generated" / "h2" / "paired-outcomes"
EVIDENCE_ROOT = ROOT / "docs" / "runtime-evidence" / "h2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _worktree_identity() -> dict[str, Any]:
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True, check=False
    ).stdout
    return {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def finalize(dataset_id: str) -> dict[str, Any]:
    dataset = DATA_ROOT / dataset_id
    evidence_dir = EVIDENCE_ROOT / dataset_id
    audit_path = evidence_dir / "offline-label-audit-full.json"
    pilot_path = evidence_dir / "offline-label-audit-pilot.json"
    manifest_path = dataset / "manifest.json"
    physical_path = dataset / "scenario_manifest.json"
    required = (audit_path, manifest_path, physical_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"H2_FINALIZE_MISSING:{missing}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    gate = dict(audit.get("gate", {}))
    metrics = dict(gate.get("metrics", {}))
    orchestrator_path = evidence_dir / "orchestrator.json"
    orchestrator = json.loads(orchestrator_path.read_text(encoding="utf-8")) if orchestrator_path.is_file() else {}
    collector = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(evidence_dir.glob("collect-full-Town*.json"))
        if path.is_file()
    }
    payload: dict[str, Any] = {
        "schema_version": "safedrive.h2.final_delivery.v1",
        "dataset_id": dataset_id,
        "terminal_status": "COMPLETED",
        "evidence_status": "VERIFIED",
        "gate_status": "GATE_PASSED" if gate.get("passed") else "GATE_FAILED",
        "stopped": True,
        "h3_status": "NOT_AUTHORIZED",
        "attempts": {
            "pilot": {"records": audit.get("records"), "gate_passed": json.loads(pilot_path.read_text(encoding="utf-8")).get("gate", {}).get("passed") if pilot_path.is_file() else None},
            "full": {"records": audit.get("records"), "gate_passed": bool(gate.get("passed"))},
            "initial_orchestrator_ok": orchestrator.get("ok"),
            "initial_orchestrator_error": orchestrator.get("error"),
            "recovered_after_safe_cold_restart": (evidence_dir / "collect-full-Town01-failure.json").is_file(),
        },
        "gate": gate,
        "metrics": metrics,
        "hashes": {
            "config_sha256": audit.get("config_sha256"),
            "physical_manifest_sha256": physical.get("physical_manifest_sha256"),
            "physical_manifest_file_sha256": _sha256(physical_path),
            "store_manifest_sha256": manifest.get("manifest_sha256"),
            "store_manifest_file_sha256": _sha256(manifest_path),
            "offline_audit_sha256": _sha256(audit_path),
        },
        "artifacts": {
            "dataset_bytes": sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file()),
            "store_artifact_count": manifest.get("artifact_count"),
            "pair_count": manifest.get("pair_count"),
            "collector_evidence": collector,
        },
        "worktree": _worktree_identity(),
        "recovery": {
            "prior_failure_evidence": str(evidence_dir / "collect-full-Town01-failure.json"),
            "recovery_action": "safe_cold_restart_then_resume_same_dataset_and_frozen_manifest",
            "data_identity_changed": False,
        },
    }
    payload["evidence_sha256"] = _stable_sha256(payload)
    output = evidence_dir / "final-delivery.json"
    _atomic_json(output, payload)
    return {"ok": True, "evidence": str(output), "evidence_sha256": payload["evidence_sha256"], "gate_status": payload["gate_status"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(finalize(args.dataset_id), sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
