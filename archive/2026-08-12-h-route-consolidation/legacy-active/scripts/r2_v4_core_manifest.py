#!/usr/bin/env python3
"""Freeze the 12-pair R2 V4 core blind subset before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from r2_v4_audit_manifest import CONDITION_BY_FAMILY, AUDIT_SEEDS, build_audit_manifest
from r2_v4_campaign import build_manifest


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _condition_scenario_id(lineage_id: str, condition: str) -> str:
    return f"{lineage_id}__{condition}"


def build_core_manifest() -> dict[str, Any]:
    campaign = build_manifest()
    lineages = sorted(campaign["lineages"], key=lambda row: row["lineage_id"])
    # Two lineages per family, all on train/val maps, with two cold seeds.  The
    # first seed of each lineage is selected to keep exactly 12 pairs while
    # preserving all six family labels and multiple maps.
    selected: list[dict[str, Any]] = []
    for family in campaign["families"]:
        candidates = [
            row
            for row in lineages
            if row["family"] == family and row["split"] != "test"
        ]
        selected.extend(candidates[:2])
    pairs: list[dict[str, Any]] = []
    for lineage in selected:
        condition = CONDITION_BY_FAMILY[str(lineage["family"])]
        seed = AUDIT_SEEDS[0]
        pairs.append(
            {
                "pair_id": f"{lineage['lineage_id']}__core__{seed}",
                "scenario_id": _condition_scenario_id(str(lineage['lineage_id']), condition),
                "seed_id": seed,
                "lineage_id": lineage["lineage_id"],
                "root_group": lineage["root_group"],
                "map_name": lineage["map_name"],
                "family": lineage["family"],
                "route_maneuver": lineage["route_maneuver"],
                "condition_variant": condition,
                "split": lineage["split"],
            }
        )
    body = {
        "schema_version": "safedrive.r2.v4.core_blind_manifest.v1",
        "campaign_id": "r2-v4-core-blind-12-v1",
        "source_campaign_manifest_hash": campaign["manifest_hash"],
        "pairs": pairs,
        "pair_count": len(pairs),
        "outcome_used_in_authoring": False,
        "candidate1_cross_family_required": True,
    }
    if len(pairs) != 12 or len({row["family"] for row in pairs}) != 6:
        raise AssertionError("R2 V4 core manifest coverage mismatch")
    body["manifest_hash"] = _sha(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite core manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = build_core_manifest()
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest_hash": value["manifest_hash"], "pairs": 12}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
