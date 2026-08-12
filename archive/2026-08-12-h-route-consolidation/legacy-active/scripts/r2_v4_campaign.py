#!/usr/bin/env python3
"""Freeze the R2 V4 counterfactual calibration manifest.

This authoring step is pure/offline.  CARLA geometry and actor scripts are
bound by the normal authoring runner later; this manifest only freezes the
root-lineage, map-OOD split, condition matrix and expected semantic labels so
collection cannot select examples after observing outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MAPS = ("Town03", "Town04", "Town05", "Town06", "Town10HD", "Town12", "Town13")
FAMILIES = ("lead_braking", "cut_in", "crossing", "merge", "obstruction", "clear")
CONDITIONS = (
    "actor_absent_green",
    "actor_nonconflict",
    "mild_conflict_left",
    "hard_conflict_right",
    "legal_adjacent_lane",
    "red_signal",
)
SEEDS = ("seed_a", "seed_b")
MANEUVERS = (
    "FOLLOW_STRAIGHT",
    "FOLLOW_CURVE_LEFT",
    "FOLLOW_CURVE_RIGHT",
    "JUNCTION_STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ROUTE_CHANGE_LEFT",
    "ROUTE_CHANGE_RIGHT",
)
EXPECTED_KINDS = ("NONE", "TEMPORAL_YIELD", "SPATIAL_AVOID", "SPATIAL_OVERTAKE")
EXPECTED_SIDES = ("NONE", "LEFT", "RIGHT")
SCHEMA = "safedrive.r2.v4.calibration_manifest.v2"

# Frozen authoring metadata.  These fields are consumed only by the geometry
#/actor authoring runner; none may enter the V4 runtime observation.
CONDITION_TABLE = {
    "actor_absent_green": {
        "actor_presence": "absent",
        "conflict": "none",
        "traffic_signal": "green",
        "adjacent_lane_legality": "not_applicable",
        "expected_role": "clear_control",
    },
    "actor_nonconflict": {
        "actor_presence": "present",
        "conflict": "none",
        "traffic_signal": "green",
        "adjacent_lane_legality": "not_applicable",
        "expected_role": "nonconflict_control",
    },
    "mild_conflict_left": {
        "actor_presence": "present",
        "conflict": "mild",
        "traffic_signal": "green",
        "conflict_side": "left",
        "adjacent_lane_legality": "legal",
        "expected_role": "mild_counterfactual",
    },
    "hard_conflict_right": {
        "actor_presence": "present",
        "conflict": "hard",
        "traffic_signal": "green",
        "conflict_side": "right",
        "adjacent_lane_legality": "legal",
        "expected_role": "hard_counterfactual",
    },
    "legal_adjacent_lane": {
        "actor_presence": "present",
        "conflict": "hard",
        "traffic_signal": "green",
        "adjacent_lane_legality": "legal",
        "expected_role": "spatial_legal_control",
    },
    "red_signal": {
        "actor_presence": "present",
        "conflict": "hard",
        "traffic_signal": "red",
        "adjacent_lane_legality": "illegal",
        "expected_role": "temporal_signal_control",
    },
}


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _split(map_name: str) -> str:
    if map_name == "Town06":
        return "val"
    if map_name == "Town13":
        return "test"
    return "train"


def _expected(family: str, condition: str) -> tuple[str, str, bool]:
    if condition in {"actor_absent_green", "actor_nonconflict"}:
        return "NONE", "NONE", False
    if condition == "red_signal":
        return "TEMPORAL_YIELD", "NONE", True
    if family == "clear":
        return "TEMPORAL_YIELD", "NONE", True
    if family == "obstruction":
        if condition == "legal_adjacent_lane":
            return "SPATIAL_OVERTAKE", "LEFT", True
        if condition == "hard_conflict_right":
            return "SPATIAL_OVERTAKE", "RIGHT", True
        return "TEMPORAL_YIELD", "NONE", True
    if family == "cut_in":
        if condition in {"mild_conflict_left", "legal_adjacent_lane"}:
            return "SPATIAL_AVOID", "RIGHT", True
        if condition == "hard_conflict_right":
            return "SPATIAL_AVOID", "LEFT", True
        return "TEMPORAL_YIELD", "NONE", True
    if family in {"crossing", "merge", "lead_braking"}:
        return "TEMPORAL_YIELD", "NONE", True
    return "NONE", "NONE", False


def _expected_params(family: str, condition: str, kind: str) -> list[float]:
    """Frozen normalized supervision for the six bounded decoder controls.

    The values are labels from the counterfactual condition, never runtime
    rules.  Unused dimensions remain zero and are masked by the trainer; the
    decoder still consumes every dimension that is supervised for the kind.
    """
    if kind == "NONE":
        return [0.0] * 6
    if kind == "TEMPORAL_YIELD":
        speed_scale = 0.65 if condition.startswith("hard") else 0.45
        return [0.0, speed_scale, 0.0, 0.0, 0.0, 0.0]
    offset = 0.75 if condition.startswith("hard") else 0.45
    if family == "obstruction" and condition == "legal_adjacent_lane":
        offset = 0.60
    return [offset, 0.0, 0.12, 0.55, 0.62, 0.82]


def _validate_manifest_coverage(
    lineages: list[dict[str, Any]], slots: list[dict[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    """Validate all pre-collection label coverage without reading outcomes.

    A frozen campaign is only useful if every map split can exercise the full
    semantic contract.  Keep this check deterministic and label-only so it
    cannot be satisfied by choosing examples after CARLA results are known.
    """
    coverage: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in slots if str(row["split"]) == split]
        if not rows:
            raise AssertionError(f"R2 V4 split has no anchors: {split}")
        expected = {
            "kind": {str(row["expected_kind"]) for row in rows},
            "side": {str(row["expected_side"]) for row in rows},
            "availability": {str(bool(row["expected_available"])) for row in rows},
            "maneuver": {str(row["route_maneuver"]) for row in rows},
        }
        if set(EXPECTED_KINDS) - expected["kind"]:
            raise AssertionError(
                f"R2 V4 {split} is missing kind coverage: "
                f"{sorted(set(EXPECTED_KINDS) - expected['kind'])}"
            )
        if set(EXPECTED_SIDES) - expected["side"]:
            raise AssertionError(
                f"R2 V4 {split} is missing side coverage: "
                f"{sorted(set(EXPECTED_SIDES) - expected['side'])}"
            )
        if expected["availability"] != {"False", "True"}:
            raise AssertionError(f"R2 V4 {split} must cover available and unavailable anchors")
        if set(MANEUVERS) - expected["maneuver"]:
            raise AssertionError(
                f"R2 V4 {split} is missing maneuver coverage: "
                f"{sorted(set(MANEUVERS) - expected['maneuver'])}"
            )
        coverage[split] = {
            key: sorted(value) for key, value in expected.items()
        }

    # Every interaction lineage (the clear family is the explicit no-actor
    # control) must contain at least two label states.  This prevents a
    # condition matrix from silently degenerating into repeated NONE rows.
    slot_by_lineage: dict[str, list[dict[str, Any]]] = {}
    for row in slots:
        slot_by_lineage.setdefault(str(row["lineage_id"]), []).append(row)
    lineage_family = {
        str(row["lineage_id"]): str(row["family"]) for row in lineages
    }
    for lineage_id, rows in slot_by_lineage.items():
        states = {
            (str(row["expected_kind"]), str(row["expected_side"]), bool(row["expected_available"]))
            for row in rows
        }
        if len(states) < 2:
            raise AssertionError(
                f"R2 V4 lineage has no label-changing counterfactuals: {lineage_id}"
            )
    return coverage


def build_manifest() -> dict[str, Any]:
    lineages: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    lineage_index = 0
    for map_index, map_name in enumerate(MAPS):
        for family in FAMILIES:
            for geometry_index in range(2):
                lineage_id = f"r2v4-{map_name.lower()}-{family}-g{geometry_index:02d}"
                split = _split(map_name)
                lineage = {
                    "lineage_id": lineage_id,
                    "map_name": map_name,
                    "family": family,
                    "geometry_index": geometry_index,
                    "split": split,
                    "route_maneuver": MANEUVERS[lineage_index % len(MANEUVERS)],
                    "root_group": lineage_id,
                }
                lineages.append(lineage)
                for condition in CONDITIONS:
                    kind, side, available = _expected(family, condition)
                    expected_params = _expected_params(family, condition, kind)
                    for seed in SEEDS:
                        scenario_id = f"{lineage_id}__{condition}__{seed}"
                        identity = {
                            "lineage_id": lineage_id,
                            "condition": condition,
                            "seed": seed,
                        }
                        slots.append(
                            {
                                "slot_id": _sha(identity)[:24],
                                "scenario_id": scenario_id,
                                "lineage_id": lineage_id,
                                "root_group": lineage_id,
                                "map_name": map_name,
                                "family": family,
                                "condition_variant": condition,
                                "seed_id": seed,
                                "split": split,
                                "route_maneuver": lineage["route_maneuver"],
                                "expected_kind": kind,
                                "expected_side": side,
                                "expected_available": available,
                                "expected_maneuver_params": expected_params,
                                "raw_tokens_required": True,
                                "scenario_family_runtime_forbidden": True,
                            }
                        )
                lineage_index += 1
    body = {
        "schema_version": SCHEMA,
        "campaign_id": "r2-v4-counterfactual-town13-ood-v2",
        "maps": list(MAPS),
        "families": list(FAMILIES),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "condition_table": CONDITION_TABLE,
        "runtime_forbidden_fields": [
            "scenario_family",
            "family",
            "oracle_outcome",
            "actor_future",
        ],
        "lineages": lineages,
        "slots": slots,
        "split_policy": {
            "train": ["Town03", "Town04", "Town05", "Town10HD", "Town12"],
            "val": ["Town06"],
            "test": ["Town13"],
        },
        "pilot_lineages": [
            lineage["lineage_id"]
            for lineage in lineages
            if lineage["map_name"] in {"Town03", "Town06"}
            and lineage["geometry_index"] == 0
        ],
    }
    if len(lineages) != 84 or len(slots) != 1008:
        raise AssertionError(f"R2 V4 counts mismatch: {len(lineages)} / {len(slots)}")
    if len(body["pilot_lineages"]) != 12:
        raise AssertionError("R2 V4 pilot must contain 12 root lineages")
    group_splits: dict[str, set[str]] = {}
    for row in slots:
        group_splits.setdefault(row["root_group"], set()).add(row["split"])
    if any(len(value) != 1 for value in group_splits.values()):
        raise AssertionError("R2 V4 root-lineage split overlap")
    body["split_coverage"] = _validate_manifest_coverage(lineages, slots)
    body["manifest_hash"] = _sha(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output", default="")
    args = parser.parse_args()
    manifest = build_manifest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.pilot_output:
        pilot_ids = set(manifest["pilot_lineages"])
        pilot = {
            **manifest,
            "campaign_id": "r2-v4-counterfactual-pilot-v1",
            "lineages": [row for row in manifest["lineages"] if row["lineage_id"] in pilot_ids],
            "slots": [row for row in manifest["slots"] if row["lineage_id"] in pilot_ids],
        }
        pilot.pop("manifest_hash", None)
        pilot["manifest_hash"] = _sha(pilot)
        pilot_path = Path(args.pilot_output)
        pilot_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_path.write_text(json.dumps(pilot, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "manifest_hash": manifest["manifest_hash"], "lineages": 84, "slots": 1008}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
