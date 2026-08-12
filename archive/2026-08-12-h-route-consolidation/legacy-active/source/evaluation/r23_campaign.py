"""Deterministic planning and audit helpers for R2/World collection campaigns."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .r23_collection import (
    CAMPAIGN_CHECKPOINT_SCHEMA,
    CollectionSlot,
    R23CollectionError,
    build_campaign_manifest,
    content_hash,
    validate_campaign_manifest,
    validate_checkpoint,
    write_json_atomic,
    write_json_exclusive,
)

MAPS = ("Town03", "Town10HD", "Town12")
V2_MAPS = (
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town12",
)
FAMILIES = (
    "lead_braking",
    "cut_in",
    "crossing",
    "merge",
    "obstruction",
    "clear",
)
SEEDS = ("seed_a", "seed_b")
CONDITIONS = tuple(f"condition_{index:02d}" for index in range(6))
V2_CONDITIONS = (
    "mild_fixed_dry_day",
    "medium_fixed_wet_day",
    "hard_fixed_rain_dusk",
    "mild_reactive_dry_night",
    "medium_reactive_wet_dusk",
    "hard_reactive_rain_night",
)
V2_ALLOWED_FAMILIES_BY_MAP = {
    map_name: frozenset(FAMILIES) for map_name in V2_MAPS
}
# Town07 is deliberately used for rural/narrow-road interactions, not for
# synthetic multi-lane cut-in or motorway merge fixtures.
V2_ALLOWED_FAMILIES_BY_MAP["Town07"] = frozenset(
    {"lead_braking", "crossing", "obstruction", "clear"}
)


@dataclass(frozen=True)
class LineageSpec:
    lineage_id: str
    phase: str
    map_name: str
    family: str
    split: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


def _balanced_lineages(
    *,
    phase: str,
    counts: Mapping[str, int],
    prefix: str,
) -> list[LineageSpec]:
    lineages: list[LineageSpec] = []
    global_index = 0
    for split, count in counts.items():
        for local_index in range(count):
            map_name = MAPS[local_index % len(MAPS)]
            family = FAMILIES[local_index % len(FAMILIES)]
            lineages.append(
                LineageSpec(
                    lineage_id=f"{prefix}_{global_index:03d}_{map_name}_{family}",
                    phase=phase,
                    map_name=map_name,
                    family=family,
                    split=split,
                )
            )
            global_index += 1
    return lineages


def default_lineage_bank() -> dict[str, Any]:
    r2 = _balanced_lineages(
        phase="r2_calibration",
        counts={"train": 21, "val": 5, "holdout": 4},
        prefix="r2cal",
    )
    world = _balanced_lineages(
        phase="world_formal",
        counts={"train": 60, "val": 12, "test": 12},
        prefix="world",
    )
    body = {
        "schema_version": "safedrive.r23_lineage_bank.v1",
        "maps": list(MAPS),
        "families": list(FAMILIES),
        "lineages": [lineage.to_dict() for lineage in (*r2, *world)],
        "geometry_status": "AUTHORING_REQUIRED",
        "geometry_contract": (
            "each lineage must bind frozen CARLA transforms, route corridor, "
            "conflict point and actor scripts before dry-run"
        ),
    }
    return {**body, "bank_content_hash": content_hash(body)}


def _balanced_pairs_v2(
    *,
    maps: Sequence[str],
    n_lineages: int,
) -> list[tuple[str, str]]:
    if n_lineages % len(maps) or n_lineages % len(FAMILIES):
        raise R23CollectionError(
            "V2 lineage count must divide evenly across maps and families"
        )
    map_target = {name: n_lineages // len(maps) for name in maps}
    family_target = {name: n_lineages // len(FAMILIES) for name in FAMILIES}
    source = "__source__"
    sink = "__sink__"
    map_nodes = {name: f"map:{name}" for name in maps}
    family_nodes = {name: f"family:{name}" for name in FAMILIES}

    def solve(edge_cap: int) -> dict[tuple[str, str], int] | None:
        capacity: dict[str, dict[str, int]] = defaultdict(dict)

        def add_edge(left: str, right: str, value: int) -> None:
            capacity[left][right] = int(value)
            capacity[right].setdefault(left, 0)

        for map_name in maps:
            add_edge(source, map_nodes[map_name], map_target[map_name])
            allowed = V2_ALLOWED_FAMILIES_BY_MAP.get(
                map_name, frozenset(FAMILIES)
            )
            for family in FAMILIES:
                if family in allowed:
                    add_edge(
                        map_nodes[map_name],
                        family_nodes[family],
                        edge_cap,
                    )
        for family in FAMILIES:
            add_edge(family_nodes[family], sink, family_target[family])
        residual = {
            left: dict(rights) for left, rights in capacity.items()
        }
        flow = 0
        while True:
            parent: dict[str, str | None] = {source: None}
            queue = deque([source])
            while queue and sink not in parent:
                left = queue.popleft()
                for right in sorted(residual[left]):
                    if residual[left][right] > 0 and right not in parent:
                        parent[right] = left
                        queue.append(right)
            if sink not in parent:
                break
            amount = n_lineages
            node = sink
            while parent[node] is not None:
                previous = str(parent[node])
                amount = min(amount, residual[previous][node])
                node = previous
            node = sink
            while parent[node] is not None:
                previous = str(parent[node])
                residual[previous][node] -= amount
                residual[node][previous] += amount
                node = previous
            flow += amount
        if flow != n_lineages:
            return None
        return {
            (map_name, family): (
                capacity[map_nodes[map_name]][family_nodes[family]]
                - residual[map_nodes[map_name]][family_nodes[family]]
            )
            for map_name in maps
            for family in FAMILIES
            if family_nodes[family] in capacity[map_nodes[map_name]]
        }

    initial_cap = max(
        1, math.ceil(n_lineages / (len(maps) * len(FAMILIES)))
    )
    counts = None
    for edge_cap in range(initial_cap, max(map_target.values()) + 1):
        counts = solve(edge_cap)
        if counts is not None:
            break
    if counts is None:
        raise R23CollectionError(
            "unable to satisfy V2 map/family marginal targets"
        )
    pairs: list[tuple[str, str]] = []
    for repeat in range(max(counts.values())):
        for map_name in maps:
            for family in FAMILIES:
                if counts.get((map_name, family), 0) > repeat:
                    pairs.append((map_name, family))
    if len(pairs) != n_lineages:
        raise R23CollectionError("V2 flow expansion count mismatch")
    return pairs


def _assign_v2_splits(
    pairs: Sequence[tuple[str, str]],
    *,
    split_counts: Mapping[str, int],
) -> list[tuple[str, str, str]]:
    remaining = list(pairs)
    assigned: list[tuple[str, str, str]] = []
    # Small evaluation splits are filled first, maximizing unseen map/family
    # coverage.  The large train split receives the deterministic remainder.
    ordered_splits = sorted(split_counts, key=lambda name: split_counts[name])
    eval_splits = [name for name in ordered_splits if name != "train"]
    for split_position, split in enumerate(eval_splits):
        count = int(split_counts[split])
        seen_maps: set[str] = set()
        seen_families: set[str] = set()
        for _ in range(count):
            if not remaining:
                raise R23CollectionError("insufficient V2 lineage pairs")
            remaining_map_counts = Counter(item[0] for item in remaining)
            remaining_family_counts = Counter(item[1] for item in remaining)
            future_coverage_sets = len(eval_splits) - split_position
            ranked = sorted(
                (
                    item
                    for item in enumerate(remaining)
                    if (
                        remaining_map_counts[item[1][0]] > future_coverage_sets
                        and remaining_family_counts[item[1][1]]
                        > future_coverage_sets
                    )
                ),
                key=lambda item: (
                    item[1][0] in seen_maps,
                    item[1][1] in seen_families,
                    item[1][0],
                    item[1][1],
                    item[0],
                ),
            )
            if not ranked:
                raise R23CollectionError(
                    f"unable to preserve future split coverage while filling {split}"
                )
            index, (map_name, family) = ranked[0]
            remaining.pop(index)
            assigned.append((map_name, family, split))
            seen_maps.add(map_name)
            seen_families.add(family)
        if seen_maps != set(map_name for map_name, _ in pairs):
            raise R23CollectionError(f"{split} does not cover every V2 map")
        if seen_families != set(FAMILIES):
            raise R23CollectionError(f"{split} does not cover every V2 family")
    train_count = int(split_counts.get("train", 0))
    if len(remaining) != train_count:
        raise R23CollectionError(
            f"V2 train remainder mismatch: {len(remaining)} != {train_count}"
        )
    assigned.extend((map_name, family, "train") for map_name, family in remaining)
    counts = Counter(split for _, _, split in assigned)
    if counts != Counter({key: int(value) for key, value in split_counts.items()}):
        raise R23CollectionError("V2 split counts mismatch")
    return assigned


def _v2_lineages(
    *,
    phase: str,
    maps: Sequence[str],
    split_counts: Mapping[str, int],
    prefix: str,
) -> list[LineageSpec]:
    n_lineages = sum(int(value) for value in split_counts.values())
    assigned = _assign_v2_splits(
        _balanced_pairs_v2(maps=maps, n_lineages=n_lineages),
        split_counts=split_counts,
    )
    # Keep split order stable for contiguous campaign reporting.
    split_order = {name: index for index, name in enumerate(split_counts)}
    assigned.sort(key=lambda row: (split_order[row[2]], row[0], row[1]))
    return [
        LineageSpec(
            lineage_id=f"{prefix}_{index:03d}_{map_name}_{family}",
            phase=phase,
            map_name=map_name,
            family=family,
            split=split,
        )
        for index, (map_name, family, split) in enumerate(assigned)
    ]


def default_lineage_bank_v2(
    *,
    maps: Sequence[str] = V2_MAPS,
) -> dict[str, Any]:
    maps = tuple(str(item) for item in maps)
    if len(maps) != 7 or len(set(maps)) != 7:
        raise R23CollectionError("V2 requires seven distinct maps")
    r2 = _v2_lineages(
        phase="r2_calibration",
        maps=maps,
        split_counts={"train": 60, "val": 12, "holdout": 12},
        prefix="r2v2",
    )
    world = _v2_lineages(
        phase="world_formal",
        maps=maps,
        split_counts={"train": 120, "val": 24, "test": 24},
        prefix="worldv2",
    )
    body = {
        "schema_version": "safedrive.r23_lineage_bank.v2",
        "maps": list(maps),
        "families": list(FAMILIES),
        "conditions": list(V2_CONDITIONS),
        "condition_matrix_version": "r23-condition-matrix-v2",
        "lineages": [lineage.to_dict() for lineage in (*r2, *world)],
        "geometry_status": "AUTHORING_REQUIRED",
        "geometry_contract": (
            "each lineage binds one unique topology signature, frozen CARLA "
            "transforms, route corridor, conflict point and actor scripts"
        ),
    }
    return {**body, "bank_content_hash": content_hash(body)}


def load_lineage_bank(path: Path) -> list[LineageSpec]:
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = str(value.get("bank_content_hash", ""))
    body = dict(value)
    body.pop("bank_content_hash", None)
    if stored != content_hash(body):
        raise R23CollectionError("lineage bank content hash mismatch")
    schema = str(value.get("schema_version"))
    if schema not in {
        "safedrive.r23_lineage_bank.v1",
        "safedrive.r23_lineage_bank.v2",
    }:
        raise R23CollectionError("lineage bank schema mismatch")
    lineages = [
        LineageSpec(
            lineage_id=str(row["lineage_id"]),
            phase=str(row["phase"]),
            map_name=str(row["map_name"]),
            family=str(row["family"]),
            split=str(row["split"]),
        )
        for row in value.get("lineages", [])
    ]
    if len({item.lineage_id for item in lineages}) != len(lineages):
        raise R23CollectionError("duplicate lineage_id in bank")
    configured_maps = {str(item) for item in value.get("maps", ())}
    if set(item.map_name for item in lineages) != configured_maps:
        raise R23CollectionError("lineage bank must cover all configured maps")
    if schema.endswith(".v2"):
        for phase, expected in (("r2_calibration", 84), ("world_formal", 168)):
            selected = [item for item in lineages if item.phase == phase]
            if len(selected) != expected:
                raise R23CollectionError(
                    f"V2 {phase} lineage count mismatch: {len(selected)}"
                )
            if set(item.map_name for item in selected) != configured_maps:
                raise R23CollectionError(f"V2 {phase} map coverage mismatch")
            if set(item.family for item in selected) != set(FAMILIES):
                raise R23CollectionError(f"V2 {phase} family coverage mismatch")
    return lineages


def _slots_for_lineage(
    lineage: LineageSpec,
    *,
    shard_index: int,
    reserve: bool,
    condition_offset: int = 0,
    conditions: Sequence[str] = CONDITIONS,
) -> list[CollectionSlot]:
    shard_id = f"{lineage.phase}-shard-{shard_index:03d}"
    slots: list[CollectionSlot] = []
    if len(conditions) != 6:
        raise R23CollectionError("each R23 lineage requires exactly six conditions")
    for condition_index, base_condition in enumerate(conditions):
        condition = (
            f"reserve_{condition_index:02d}"
            if reserve
            else base_condition
        )
        for seed_index, seed in enumerate(SEEDS):
            slot_index = condition_index * len(SEEDS) + seed_index
            anchor = f"anchor_{condition_index + condition_offset:02d}"
            scenario = f"{lineage.lineage_id}__{condition}"
            raw = {
                "phase": lineage.phase,
                "lineage_id": lineage.lineage_id,
                "condition_variant": condition,
                "seed_id": seed,
                "anchor_variant": anchor,
            }
            slots.append(
                CollectionSlot(
                    slot_id=content_hash(raw)[:24],
                    phase=lineage.phase,
                    shard_id=shard_id,
                    shard_index=shard_index,
                    slot_index=slot_index,
                    lineage_id=lineage.lineage_id,
                    family=lineage.family,
                    map_name=lineage.map_name,
                    split=lineage.split,
                    condition_variant=condition,
                    seed_id=seed,
                    anchor_variant=anchor,
                    scenario_id=scenario,
                    reserve=reserve,
                )
            )
    return slots


def build_phase_manifest(
    *,
    lineages: Sequence[LineageSpec],
    phase: str,
    campaign_id: str,
    r2_checkpoint_sha256: str,
    collection_config_sha256: str,
    expected_base_lineages: int | None = None,
    reserve_lineages: int | None = None,
    completed_target: int | None = None,
    campaign_version: str = "v1",
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    selected = [item for item in lineages if item.phase == phase]
    expected = (
        int(expected_base_lineages)
        if expected_base_lineages is not None
        else (30 if phase == "r2_calibration" else 84)
    )
    if len(selected) != expected:
        raise R23CollectionError(
            f"{phase} requires {expected} base lineages, got {len(selected)}"
        )
    slots: list[CollectionSlot] = []
    for shard_index, lineage in enumerate(selected):
        slots.extend(
            _slots_for_lineage(
                lineage,
                shard_index=shard_index,
                reserve=False,
                conditions=conditions,
            )
        )
    base_target = len(slots)
    if completed_target is None:
        completed_target = base_target if phase == "r2_calibration" else 1000
    reserve_count = (
        int(reserve_lineages)
        if reserve_lineages is not None
        else (16 if phase == "world_formal" else 0)
    )
    if reserve_count and phase != "world_formal":
        raise R23CollectionError("technical reserve is only valid for world_formal")
    if reserve_count:
        if campaign_version == "v1" and reserve_lineages is None:
            reserve_sources = (
                selected[:12]
                + [item for item in selected if item.split == "val"][:2]
                + [item for item in selected if item.split == "test"][:2]
            )
        else:
            reserve_sources = [
                selected[index % len(selected)] for index in range(reserve_count)
            ]
        for reserve_index, lineage in enumerate(reserve_sources):
            reserve_lineage = (
                lineage
                if campaign_version == "v1"
                else LineageSpec(
                    lineage_id=(
                        f"{lineage.lineage_id}"
                        f"__technical_reserve_{reserve_index:03d}"
                    ),
                    phase=lineage.phase,
                    map_name=lineage.map_name,
                    family=lineage.family,
                    split=lineage.split,
                )
            )
            slots.extend(
                _slots_for_lineage(
                    reserve_lineage,
                    shard_index=len(selected) + reserve_index,
                    reserve=True,
                    condition_offset=6,
                    conditions=conditions,
                )
            )
    return build_campaign_manifest(
        campaign_id=campaign_id,
        phase=phase,
        slots=slots,
        r2_checkpoint_sha256=r2_checkpoint_sha256,
        collection_config_sha256=collection_config_sha256,
        base_slot_target=base_target,
        completed_target=int(completed_target),
    )


def create_campaign_layout(
    root: Path,
    *,
    lineage_bank: Path,
    r2_checkpoint_sha256: str,
    collection_config_sha256: str,
    phases: Sequence[str] = ("r2_calibration", "world_formal"),
    phase_specs: Mapping[str, Mapping[str, Any]] | None = None,
    campaign_version: str = "v1",
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    lineages = load_lineage_bank(lineage_bank)
    outputs = {}
    for phase in phases:
        if phase not in {"r2_calibration", "world_formal"}:
            raise R23CollectionError(f"unsupported phase {phase!r}")
        phase_root = root / phase
        spec = dict((phase_specs or {}).get(phase, {}))
        manifest = build_phase_manifest(
            lineages=lineages,
            phase=phase,
            campaign_id=f"r23-{phase}-{campaign_version}",
            r2_checkpoint_sha256=(
                r2_checkpoint_sha256
            ),
            collection_config_sha256=collection_config_sha256,
            expected_base_lineages=spec.get("base_lineages"),
            reserve_lineages=spec.get("reserve_lineages"),
            completed_target=spec.get("completed_target"),
            campaign_version=campaign_version,
            conditions=conditions,
        )
        write_json_exclusive(phase_root / "campaign_manifest.json", manifest)
        by_shard: dict[str, list[CollectionSlot]] = defaultdict(list)
        for slot in (CollectionSlot.from_dict(row) for row in manifest["slots"]):
            by_shard[slot.shard_id].append(slot)
        for shard_id, shard_slots in by_shard.items():
            shard = phase_root / "shards" / shard_id
            shard.mkdir(parents=True, exist_ok=True)
            write_json_exclusive(
                shard / "shard_plan.json",
                {
                    "schema_version": "safedrive.r23_shard_plan.v1",
                    "campaign_manifest_hash": manifest["manifest_content_hash"],
                    "shard_id": shard_id,
                    "slots": [
                        slot.to_dict()
                        for slot in sorted(shard_slots, key=lambda item: item.slot_index)
                    ],
                },
            )
        checkpoint = {
            "schema_version": CAMPAIGN_CHECKPOINT_SCHEMA,
            "manifest_content_hash": manifest["manifest_content_hash"],
            "last_completed_index": -1,
            "results": [],
        }
        write_json_exclusive(phase_root / "campaign_checkpoint.json", checkpoint)
        outputs[phase] = validate_campaign_manifest(manifest)
    return outputs


def append_checkpoint_result(
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    audit = validate_checkpoint(checkpoint, manifest)
    slots = [CollectionSlot.from_dict(value) for value in manifest["slots"]]
    if audit["next_index"] >= len(slots):
        raise R23CollectionError("campaign is already complete")
    expected = slots[audit["next_index"]]
    if str(result.get("slot_id")) != expected.slot_id:
        raise R23CollectionError("result is not for the next planned slot")
    results = [*checkpoint["results"], dict(result)]
    updated = {
        **checkpoint,
        "last_completed_index": len(results) - 1,
        "results": results,
    }
    validate_checkpoint(updated, manifest)
    write_json_atomic(checkpoint_path, updated)
    return updated


def campaign_status(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(
        (root / "campaign_checkpoint.json").read_text(encoding="utf-8")
    )
    manifest_audit = validate_campaign_manifest(manifest)
    checkpoint_audit = validate_checkpoint(checkpoint, manifest)
    counters = Counter(str(row.get("status", "UNKNOWN")) for row in checkpoint["results"])
    failure_codes = Counter(
        str(code)
        for row in checkpoint["results"]
        for code in row.get("failure_codes", ())
    )
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    slot_by_id = {
        slot.slot_id: slot
        for slot in (CollectionSlot.from_dict(row) for row in manifest["slots"])
    }
    for row in checkpoint["results"]:
        slot = slot_by_id[str(row["slot_id"])]
        by_split[slot.split][str(row.get("status", "UNKNOWN"))] += 1
    return {
        **manifest_audit,
        **checkpoint_audit,
        "status_counts": dict(counters),
        "failure_codes": dict(failure_codes),
        "split_status_counts": {
            key: dict(value) for key, value in sorted(by_split.items())
        },
    }
