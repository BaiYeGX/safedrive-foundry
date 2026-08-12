#!/usr/bin/env python3
"""Run pre-authored K2 V3 core/audit pairs with one shared learned policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_live_v3 import (  # noqa: E402
    build_v3_model_identity,
    run_pair_v3,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import canonical_sha256  # noqa: E402
from driving_vla.model.neural_policy import NeuralV3Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SimLingoNeuralRuntime,
)


def _read_frozen(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if stored != canonical_sha256(body):
        raise ValueError(f"{path}: manifest hash mismatch")
    return value


def _write(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authoring-root", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = _read_frozen(Path(args.manifest))
    map_name = str(args.map)
    cases = [
        dict(case)
        for case in manifest.get("cases") or ()
        if str(case.get("map_name") or "") == map_name
    ]
    if not cases:
        raise ValueError(f"no {manifest['phase']} cases for {map_name}")
    evidence_root = Path(args.evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=False)
    identity = build_v3_model_identity(args.checkpoint)
    runtime = SimLingoNeuralRuntime(device=str(args.device))
    load = runtime.load()
    if not load.ok:
        raise RuntimeError(f"SimLingo load failed: {load.error}")
    policy = NeuralV3Policy(
        runtime=runtime,
        semantic_head_checkpoint=str(args.checkpoint),
        teacher_mode=False,
        keep_on_gpu=True,
        lazy=False,
        device=str(args.device),
        checkpoint_use="r2v3_blind_audit",
    )
    policy.ensure_loaded()

    rows = []
    for index, case in enumerate(cases):
        fixture_id = str(case["fixture_id"])
        registry_path = (
            Path(args.authoring_root)
            / str(manifest["phase"])
            / map_name
            / fixture_id
            / "scenario_registry.toml"
        )
        registry = load_scenario_registry(registry_path)
        seed_id = str(case.get("seed_id") or "seed_a")
        fixture = registry.get(fixture_id, seed_id)
        row = {
            "case_id": str(case.get("case_id") or case.get("pair_id")),
            "fixture_id": fixture_id,
            "family": str(case["family"]),
            "maneuver": str(case["maneuver"]),
            "repeat_group": str(case.get("repeat_group") or ""),
            "checkpoint_sha256": identity["checkpoint_sha256"],
        }
        try:
            pair = run_pair_v3(
                registry=registry,
                fixture=fixture,
                checkpoint=args.checkpoint,
                evidence_dir=evidence_root / fixture_id,
                device=str(args.device),
                shared_policy=policy,
                branch_order=(
                    (0, 1)
                    if seed_id == "seed_a"
                    else (1, 0)
                ),
            )
            row.update(pair)
            row["status"] = "COMPLETED"
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "status": "FAILED",
                    "comparable": False,
                    "decisive": False,
                    "winner": None,
                    "candidate1_available": False,
                    "safe_candidate_exists": False,
                    "both_bad": True,
                    "guard_mpc_failure": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        # Preserve the pre-registered grouping after merging the live report.
        row["repeat_group"] = str(case.get("repeat_group") or "")
        rows.append(row)
        _write(
            evidence_root / "progress.json",
            {
                "schema_version": "safedrive.r2_v3.formal_pairs_progress.v1",
                "phase": str(manifest["phase"]),
                "map_name": map_name,
                "source_manifest_hash": manifest["manifest_hash"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "rows": rows,
            },
        )
        print(
            f"[formal-pair] {index + 1}/{len(cases)} {fixture_id} "
            f"{row['status']} comparable={row.get('comparable')}",
            flush=True,
        )
    report = {
        "schema_version": "safedrive.r2_v3.formal_pairs.v1",
        "phase": str(manifest["phase"]),
        "map_name": map_name,
        "source_manifest_hash": manifest["manifest_hash"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "n_run": len(rows),
        "n_completed": sum(row["status"] == "COMPLETED" for row in rows),
        "rows": rows,
    }
    _write(evidence_root / "pair_campaign_report.json", report, exclusive=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if int(report["n_completed"]) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
