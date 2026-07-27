#!/usr/bin/env python3
"""Read-only R2-D planner probe against real Evidence (does NOT write run_set_manifest)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_live import (  # noqa: E402
    EXECUTOR_CONFIG_HASH,
    _identity_hashes_without_carla,
)
from driving_vla.evaluation.runner_contract import (  # noqa: E402
    build_run_set_manifest,
    require_frozen_registry,
)


def main() -> int:
    evidence_pairs = ROOT / "docs/runtime-evidence/r2-g4a-paired-pilot/pairs"
    frozen = (
        ROOT
        / "docs/runtime-evidence/r2-g4a-paired-pilot/registry/registry_manifest.json"
    )
    reg_path = ROOT / "safedrive_foundry/config/g4a/scenario_registry_v1.toml"
    reg, audit, _ = require_frozen_registry(
        reg_path, manifest_path=frozen, repo_root=ROOT
    )
    _ckpt, _rh, model_retimer_hash, _ = _identity_hashes_without_carla(device="cpu")
    # Build in-memory only — never write run_set_manifest.json
    plan = build_run_set_manifest(
        registry=reg,
        freeze_audit=audit,
        pairs_root=evidence_pairs,
        model_retimer_hash=model_retimer_hash,
        executor_config_hash=EXECUTOR_CONFIG_HASH,
        model_checkpoint_hash=_ckpt,
        retimer_hash=_rh,
    )
    first = plan["pairs"][0]
    summary = {
        "wrote_manifest": False,
        "pairs_root": str(evidence_pairs.as_posix()),
        "first_slot": {
            "scenario_id": first["scenario_id"],
            "seed_id": first["seed_id"],
            "pair_id": first["pair_id"],
            "planned_attempt_id": first["planned_attempt_id"],
            "planned_attempt_dir_rel": first["planned_attempt_dir_rel"],
            "occupied_attempts_at_freeze": first.get("occupied_attempts_at_freeze"),
        },
        "attempt_id_histogram": {},
        "manifest_content_hash": plan["manifest_content_hash"],
    }
    hist: dict[str, int] = {}
    for p in plan["pairs"]:
        k = str(p["planned_attempt_id"])
        hist[k] = hist.get(k, 0) + 1
    summary["attempt_id_histogram"] = hist
    print(json.dumps(summary, indent=2))
    # Expectation for current repo Evidence: R2-C legacy on lead_brake_moderate/seed_a
    if first["scenario_id"] == "lead_brake_moderate" and first["seed_id"] == "seed_a":
        if int(first["planned_attempt_id"]) != 1:
            print(
                f"PROBE_UNEXPECTED: expected planned_attempt_id=1, got {first['planned_attempt_id']}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
