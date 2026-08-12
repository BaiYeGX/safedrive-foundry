#!/usr/bin/env python3
"""Freeze the 84-fixture × 3-cold-seed R2 V4 blind audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from r2_v4_campaign import build_manifest

AUDIT_SEEDS = ("cold_a", "cold_b", "cold_c")
CONDITION_BY_FAMILY = {
    "lead_braking": "mild_conflict_left",
    "cut_in": "mild_conflict_left",
    "crossing": "mild_conflict_left",
    "merge": "mild_conflict_left",
    "obstruction": "hard_conflict_right",
    "clear": "actor_absent_green",
}
SCHEMA = "safedrive.r2.v4.blind_audit_manifest.v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _condition_scenario_id(lineage_id: str, condition: str) -> str:
    """Registry identity is condition-level; ``seed_id`` is the repeat key."""
    return f"{lineage_id}__{condition}"


def build_audit_manifest(campaign: dict[str, Any] | None = None) -> dict[str, Any]:
    campaign = campaign or build_manifest()
    lineages = [dict(row) for row in campaign.get("lineages", [])]
    if len(lineages) != 84:
        raise ValueError(f"R2 V4 audit requires 84 lineages, got {len(lineages)}")
    fixtures: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for lineage in lineages:
        family = str(lineage["family"])
        condition = CONDITION_BY_FAMILY[family]
        fixture_id = f"{lineage['lineage_id']}__audit"
        fixture = {
            "fixture_id": fixture_id,
            "lineage_id": str(lineage["lineage_id"]),
            "root_group": str(lineage["root_group"]),
            "map_name": str(lineage["map_name"]),
            "split": str(lineage["split"]),
            "family": family,
            "condition_variant": condition,
            "route_maneuver": str(lineage["route_maneuver"]),
            "cold_rebuild": True,
        }
        fixtures.append(fixture)
        for seed in AUDIT_SEEDS:
            pairs.append(
                {
                    **fixture,
                    "pair_id": f"{fixture_id}__{seed}",
                    "scenario_id": _condition_scenario_id(str(lineage['lineage_id']), condition),
                    "seed_id": seed,
                }
            )
    body = {
        "schema_version": SCHEMA,
        "campaign_id": "r2-v4-84-fixture-252-pair-blind-v1",
        "source_campaign_manifest_hash": str(campaign.get("manifest_hash") or ""),
        "fixtures": fixtures,
        "pairs": pairs,
        "pair_count": len(pairs),
        "cold_seed_count": len(AUDIT_SEEDS),
        "locked_test": True,
        "outcome_used_in_authoring": False,
    }
    if len(fixtures) != 84 or len(pairs) != 252:
        raise AssertionError("R2 V4 audit count mismatch")
    body["manifest_hash"] = _sha(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = (
        json.loads(Path(args.campaign_manifest).read_text(encoding="utf-8"))
        if args.campaign_manifest
        else build_manifest()
    )
    manifest = build_audit_manifest(campaign)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite audit manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest_hash": manifest["manifest_hash"], "pairs": 252}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
