#!/usr/bin/env python3
"""Freeze the final-head R3 ActionBranch campaign manifest.

The manifest is created only after a formal R2 checkpoint hash is available.
Its test lineages and A/A noise probes are predeclared before branch outcomes
are observed; technical reserves contain train/val lineages only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MAPS = ("Town03", "Town04", "Town05", "Town06", "Town10HD", "Town12", "Town13")
FAMILIES = ("lead_braking", "cut_in", "crossing", "merge", "obstruction", "clear")
CONDITIONS = (
    ("mild_fixed_dry_day", "fixed"),
    ("medium_fixed_wet_day", "fixed"),
    ("hard_fixed_rain_dusk", "fixed"),
    ("mild_reactive_dry_night", "reactive"),
    ("medium_reactive_wet_dusk", "reactive"),
    ("hard_reactive_rain_night", "reactive"),
)
SEEDS = ("seed_a", "seed_b")
SCHEMA = "safedrive.r3.final_head_campaign.v1"


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _aa_identity(repeat_group: str) -> str:
    return _sha(
        {
            "namespace": "r3_aa_noise_probe",
            "repeat_group": str(repeat_group),
            "candidate_id": "v3_nominal_progress",
        }
    )


def _split(map_name: str) -> str:
    if map_name == "Town06":
        return "val"
    if map_name == "Town13":
        return "test"
    return "train"


def _require_hash(value: str) -> str:
    value = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("formal R3 campaign requires a 64-hex R2 checkpoint SHA256")
    return value


def _validate_campaign_coverage(
    lineages: list[dict[str, Any]], slots: list[dict[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    """Freeze map/family/controller coverage before any branch outcome exists."""
    coverage: dict[str, dict[str, list[str]]] = {}
    expected_maps = {
        "train": {"Town03", "Town04", "Town05", "Town10HD", "Town12"},
        "val": {"Town06"},
        "test": {"Town13"},
    }
    for split, maps in expected_maps.items():
        split_lineages = [row for row in lineages if str(row["split"]) == split]
        if {str(row["map_name"]) for row in split_lineages} != maps:
            raise AssertionError(f"R3 {split} map coverage mismatch")
        families = {str(row["family"]) for row in split_lineages}
        if families != set(FAMILIES):
            raise AssertionError(f"R3 {split} must cover all six families")
        split_slots = [row for row in slots if str(row["split"]) == split]
        controllers = {"fixed", "reactive"}
        actual_controllers = {str(row["actor_controller_kind"]) for row in split_slots}
        if actual_controllers != controllers:
            raise AssertionError(f"R3 {split} must cover fixed and reactive conditions")
        conditions = {str(row["condition_variant"]) for row in split_slots}
        if conditions != {name for name, _controller in CONDITIONS}:
            raise AssertionError(f"R3 {split} condition matrix is incomplete")
        coverage[split] = {
            "maps": sorted(maps),
            "families": sorted(families),
            "controllers": sorted(actual_controllers),
            "conditions": sorted(conditions),
        }

    split_by_root: dict[str, set[str]] = {}
    for row in lineages:
        split_by_root.setdefault(str(row["root_group"]), set()).add(str(row["split"]))
    if any(len(value) != 1 for value in split_by_root.values()):
        raise AssertionError("R3 root-lineage split overlap")
    if len({str(row["scenario_id"]) for row in slots}) != len(slots):
        raise AssertionError("R3 campaign contains duplicate scenario slots")
    return coverage


def build_campaign(r2_checkpoint_sha256: str) -> dict[str, Any]:
    checkpoint = _require_hash(r2_checkpoint_sha256)
    lineages: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    index = 0
    for map_name in MAPS:
        for family in FAMILIES:
            for geometry_index in range(4):
                lineage_id = f"r3v1-{map_name.lower()}-{family}-g{geometry_index:02d}"
                lineage = {
                    "lineage_id": lineage_id,
                    "root_group": lineage_id,
                    "map_name": map_name,
                    "family": family,
                    "geometry_index": geometry_index,
                    "split": _split(map_name),
                }
                lineages.append(lineage)
                for condition, controller in CONDITIONS:
                    for seed in SEEDS:
                        identity = {
                            "namespace": "r3_final_head_formal",
                            "lineage_id": lineage_id,
                            "condition": condition,
                            "seed": seed,
                        }
                        slots.append(
                            {
                                "slot_id": _sha(identity)[:24],
                                "scenario_id": f"{lineage_id}__{condition}__{seed}",
                                "lineage_id": lineage_id,
                                "root_group": lineage_id,
                                "repeat_group": lineage_id,
                                "aa_noise_identity": _aa_identity(lineage_id),
                                "map_name": map_name,
                                "family": family,
                                "condition_variant": condition,
                                "actor_controller_kind": controller,
                                "seed_id": seed,
                                "split": _split(map_name),
                                "namespace": "r3_final_head_formal",
                                "r2_checkpoint_sha256": checkpoint,
                                "candidate_count": 2,
                                "history_count": 5,
                                "future_steps": 10,
                                "future_dt_s": 0.25,
                            }
                        )
                index += 1
    # Pre-freeze a balanced technical reserve: seven lineages per family,
    # selected round-robin over the six non-test maps.  This keeps reserve
    # replenishment from becoming a result-dependent winner/family sampler.
    reserve_lineages: list[dict[str, Any]] = []
    for family in FAMILIES:
        candidates = [
            lineage
            for lineage in lineages
            if lineage["family"] == family and lineage["split"] != "test"
        ]
        by_map = {
            map_name: [item for item in candidates if item["map_name"] == map_name]
            for map_name in MAPS
            if map_name != "Town13"
        }
        ordered_candidates = [
            by_map[map_name][0]
            for map_name in MAPS
            if map_name != "Town13"
        ]
        ordered_candidates.append(by_map["Town03"][1])
        for lineage in ordered_candidates:
            reserve_lineages.append(
                {
                    **lineage,
                    "lineage_id": lineage["lineage_id"] + "__reserve",
                    "root_group": lineage["lineage_id"] + "__reserve",
                    "reserve_source_lineage_id": lineage["lineage_id"],
                }
            )
    reserve_slots: list[dict[str, Any]] = []
    for lineage in reserve_lineages:
        for condition, controller in CONDITIONS:
            for seed in SEEDS:
                identity = {
                    "namespace": "r3_final_head_formal_reserve",
                    "lineage_id": lineage["lineage_id"],
                    "condition": condition,
                    "seed": seed,
                }
                reserve_slots.append(
                    {
                        "slot_id": _sha(identity)[:24],
                        "scenario_id": f"{lineage['lineage_id']}__{condition}__{seed}",
                        "lineage_id": lineage["lineage_id"],
                        "root_group": lineage["lineage_id"],
                        "repeat_group": lineage["lineage_id"],
                        "aa_noise_identity": _aa_identity(lineage["lineage_id"]),
                        "map_name": lineage["map_name"],
                        "family": lineage["family"],
                        "condition_variant": condition,
                        "actor_controller_kind": controller,
                        "seed_id": seed,
                        "split": lineage["split"],
                        "namespace": "r3_final_head_formal_reserve",
                        "r2_checkpoint_sha256": checkpoint,
                        "reserve": True,
                    }
                )
    # Freeze a balanced train/val development gate before any branch outcome
    # is observed.  Select 64 root lineages round-robin across all six
    # non-test maps and six families, then take four fixed/reactive-balanced
    # seed/condition slots per side (8 slots per lineage = 512).  This keeps
    # Town06 in the first development gate instead of silently making it a
    # train-only pilot.
    dev_lineages: list[dict[str, Any]] = []
    grouped = {
        (map_name, family): [
            row
            for row in lineages
            if row["map_name"] == map_name
            and row["family"] == family
            and row["split"] in {"train", "val"}
        ]
        for map_name in MAPS
        if map_name != "Town13"
        for family in FAMILIES
    }
    for round_index in range(2):
        for key in grouped:
            values = grouped[key]
            if round_index < len(values):
                dev_lineages.append(values[round_index])
    dev_lineages = dev_lineages[:64]
    slot_lookup = {str(row["scenario_id"]): row for row in slots}
    development_slots: list[dict[str, Any]] = []
    for lineage in dev_lineages:
        prefix = str(lineage["lineage_id"])
        for condition, _controller in CONDITIONS:
            development_slots.append(
                slot_lookup[f"{prefix}__{condition}__seed_a"]
            )
        for condition in ("mild_fixed_dry_day", "mild_reactive_dry_night"):
            development_slots.append(
                slot_lookup[f"{prefix}__{condition}__seed_b"]
            )
    if len(development_slots) != 512:
        raise AssertionError("R3 development gate must contain exactly 512 train/val slots")
    noise_probe = [
        {
            "repeat_group": lineage["lineage_id"],
            "aa_noise_identity": _aa_identity(lineage["lineage_id"]),
            "lineage_id": lineage["lineage_id"],
            "scenario_id": f"{lineage['lineage_id']}__mild_fixed_dry_day__seed_a",
            "seed_id": "seed_a",
            "condition_variant": "mild_fixed_dry_day",
            "map_name": lineage["map_name"],
            "family": lineage["family"],
            "split": lineage["split"],
            "namespace": "r3_aa_noise_probe",
            "r2_checkpoint_sha256": checkpoint,
            "candidate_id": "candidate_0",
        }
        for lineage in lineages
    ]
    body = {
        "schema_version": SCHEMA,
        "campaign_id": "r3-final-head-action-branches-town13-ood-v1",
        "namespace": "r3_final_head_formal",
        "r2_checkpoint_sha256": checkpoint,
        "maps": list(MAPS),
        "families": list(FAMILIES),
        "conditions": [name for name, _controller in CONDITIONS],
        "lineages": lineages,
        "slots": slots,
        "reserve_lineages": reserve_lineages,
        "reserve_slots": reserve_slots,
        "development_slots": development_slots,
        "noise_probe": noise_probe,
        "targets": {
            "base_slots": 2016,
            "reserve_slots": 504,
            "development_slots": 512,
            "completed_min": 2000,
            "comparable_min": 1000,
            "decisive_min": 250,
            "winner_each_min": 100,
        },
        "split_policy": {
            "train": ["Town03", "Town04", "Town05", "Town10HD", "Town12"],
            "val": ["Town06"],
            "test": ["Town13"],
        },
    }
    if len(lineages) != 168 or len(slots) != 2016:
        raise AssertionError(f"R3 base count mismatch: {len(lineages)} / {len(slots)}")
    if len(reserve_lineages) != 42 or len(reserve_slots) != 504:
        raise AssertionError("R3 reserve count mismatch")
    body["split_coverage"] = _validate_campaign_coverage(lineages, slots)
    body["manifest_hash"] = _sha(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = build_campaign(args.r2_checkpoint_sha256)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest_hash": campaign["manifest_hash"], "base_slots": 2016, "reserve_slots": 504}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
