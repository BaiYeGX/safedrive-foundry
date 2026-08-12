"""Frozen R2 V3 collection plans and World-ready acceptance gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from driving_vla.model.k2_v3_types import AlternativeKind
from driving_vla.model.navigation_contract import (
    RouteManeuver,
    canonical_sha256,
)

SCHEMA = "safedrive.r2_world_ready.v3"
CALIBRATION_MAPS = ("Town03", "Town04", "Town05", "Town10HD", "Town12")
AUDIT_MAPS = CALIBRATION_MAPS + ("Town06", "Town13")
CALIBRATION_CONDITIONS = ("mild", "medium", "hard")
CALIBRATION_SEEDS = ("seed_a", "seed_b")
WORLD_SEEDS = ("seed_a", "seed_b", "seed_c", "seed_d")


@dataclass(frozen=True)
class TemplateV3:
    template_id: str
    family: str
    maneuver_cycle: tuple[RouteManeuver, ...]
    alternative_kind: AlternativeKind
    audit_bucket: str


TEMPLATES = (
    TemplateV3(
        "straight_curve",
        "clear",
        (
            RouteManeuver.FOLLOW_STRAIGHT,
            RouteManeuver.FOLLOW_CURVE_LEFT,
            RouteManeuver.FOLLOW_CURVE_RIGHT,
        ),
        AlternativeKind.NONE,
        "interaction",
    ),
    TemplateV3(
        "junction_straight",
        "clear",
        (RouteManeuver.JUNCTION_STRAIGHT,),
        AlternativeKind.NONE,
        "road_or_traffic",
    ),
    TemplateV3(
        "turn_left_right",
        "clear",
        (RouteManeuver.TURN_LEFT, RouteManeuver.TURN_RIGHT),
        AlternativeKind.NONE,
        "road_or_traffic",
    ),
    TemplateV3(
        "route_change_left_right",
        "clear",
        (
            RouteManeuver.ROUTE_CHANGE_LEFT,
            RouteManeuver.ROUTE_CHANGE_RIGHT,
        ),
        AlternativeKind.NONE,
        "road_or_traffic",
    ),
    TemplateV3(
        "follow_stop",
        "lead_braking",
        (RouteManeuver.FOLLOW_STRAIGHT,),
        AlternativeKind.TEMPORAL_YIELD,
        "interaction",
    ),
    TemplateV3(
        "cut_in_left",
        "cut_in",
        (RouteManeuver.FOLLOW_STRAIGHT,),
        AlternativeKind.SPATIAL_AVOID,
        "interaction",
    ),
    TemplateV3(
        "cut_in_right",
        "cut_in",
        (RouteManeuver.FOLLOW_STRAIGHT,),
        AlternativeKind.SPATIAL_AVOID,
        "interaction",
    ),
    TemplateV3(
        "merge_yield",
        "merge",
        (RouteManeuver.TURN_RIGHT, RouteManeuver.FOLLOW_STRAIGHT),
        AlternativeKind.TEMPORAL_YIELD,
        "interaction",
    ),
    TemplateV3(
        "crossing_turn_yield",
        "crossing",
        (RouteManeuver.TURN_LEFT, RouteManeuver.TURN_RIGHT),
        AlternativeKind.TEMPORAL_YIELD,
        "interaction",
    ),
    TemplateV3(
        "overtake_left",
        "obstruction",
        (RouteManeuver.FOLLOW_STRAIGHT,),
        AlternativeKind.SPATIAL_OVERTAKE,
        "interaction",
    ),
    TemplateV3(
        "overtake_right",
        "obstruction",
        (RouteManeuver.FOLLOW_STRAIGHT,),
        AlternativeKind.SPATIAL_OVERTAKE,
        "interaction",
    ),
    TemplateV3(
        "traffic_control",
        "traffic_control",
        (
            RouteManeuver.JUNCTION_STRAIGHT,
            RouteManeuver.TURN_LEFT,
            RouteManeuver.TURN_RIGHT,
        ),
        AlternativeKind.TEMPORAL_YIELD,
        "road_or_traffic",
    ),
)


def _topology_balanced_template_slots(
    map_name: str,
) -> tuple[tuple[str, TemplateV3], ...]:
    """Return stable slot identities independently of fixture authoring."""
    del map_name
    return tuple(
        (slot.template_id, slot)
        for slot in TEMPLATES
    )


@dataclass(frozen=True)
class CampaignSlotV3:
    slot_id: str
    phase: str
    lineage_id: str
    split: str
    map_name: str
    template_id: str
    family: str
    maneuver: RouteManeuver
    alternative_kind: AlternativeKind
    condition: str
    seed_id: str
    route_fixture_id: str
    actor_script_id: str
    oracle_version: str = "oracle_v2_clearance_saturated"

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["maneuver"] = self.maneuver.value
        value["alternative_kind"] = self.alternative_kind.value
        return value


def _lineage_rows(maps: Sequence[str]) -> list[tuple[int, int, str, TemplateV3]]:
    return [
        (map_index, template_index, map_name, template)
        for map_index, map_name in enumerate(maps)
        for template_index, template in enumerate(TEMPLATES)
    ]


def _split_by_frozen_hash(
    rows: Sequence[tuple[int, int, str, TemplateV3]],
) -> dict[tuple[str, str], str]:
    ordered = sorted(
        rows,
        key=lambda row: canonical_sha256(
            {"map": row[2], "template": row[3].template_id, "version": SCHEMA}
        ),
    )
    counts = (("train", 42), ("dev", 10), ("test", 8))
    if len(ordered) != sum(count for _, count in counts):
        raise ValueError("calibration must contain exactly 60 lineages")
    result: dict[tuple[str, str], str] = {}
    offset = 0
    for split, count in counts:
        for row in ordered[offset : offset + count]:
            result[(row[2], row[3].template_id)] = split
        offset += count
    return result


def build_calibration_manifest_v3() -> dict[str, Any]:
    rows = _lineage_rows(CALIBRATION_MAPS)
    splits = _split_by_frozen_hash(rows)
    slots: list[CampaignSlotV3] = []
    for map_index, template_index, map_name, template in rows:
        lineage = f"r2v3-cal-{map_name}-{template.template_id}"
        maneuver = template.maneuver_cycle[
            (map_index + template_index) % len(template.maneuver_cycle)
        ]
        for condition in CALIBRATION_CONDITIONS:
            for seed_id in CALIBRATION_SEEDS:
                slot_id = (
                    f"{lineage}-{condition}-{seed_id}"
                )
                slots.append(
                    CampaignSlotV3(
                        slot_id=slot_id,
                        phase="calibration",
                        lineage_id=lineage,
                        split=splits[(map_name, template.template_id)],
                        map_name=map_name,
                        template_id=template.template_id,
                        family=template.family,
                        maneuver=maneuver,
                        alternative_kind=template.alternative_kind,
                        condition=condition,
                        seed_id=seed_id,
                        route_fixture_id=f"{lineage}-route-v1",
                        actor_script_id=f"{lineage}-{condition}-actors-v1",
                    )
                )
    body = {
        "schema_version": SCHEMA,
        "phase": "calibration",
        "maps": list(CALIBRATION_MAPS),
        "templates": [template.template_id for template in TEMPLATES],
        "conditions": list(CALIBRATION_CONDITIONS),
        "seeds": list(CALIBRATION_SEEDS),
        "split_slots": {"train": 252, "dev": 60, "test": 48},
        "pilot_maps": list(CALIBRATION_MAPS[:2]),
        "pilot_slots": 144,
        "frozen_before_results": [
            "split",
            "route_fixture_id",
            "entry_exit",
            "actor_script_id",
            "oracle_v2",
            "thresholds",
        ],
        "slots": [slot.to_dict() for slot in slots],
    }
    validate_calibration_manifest_v3(body)
    return {**body, "manifest_hash": canonical_sha256(body)}


def validate_calibration_manifest_v3(manifest: Mapping[str, Any]) -> None:
    slots = list(manifest.get("slots") or ())
    if len(slots) != 360:
        raise ValueError(f"calibration requires 360 slots, got {len(slots)}")
    if len({str(slot["slot_id"]) for slot in slots}) != 360:
        raise ValueError("calibration slot ids must be unique")
    lineage_slots: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for slot in slots:
        lineage_slots[str(slot["lineage_id"])].append(slot)
    if len(lineage_slots) != 60:
        raise ValueError("calibration requires 60 unique lineages")
    if any(len(items) != 6 for items in lineage_slots.values()):
        raise ValueError("each calibration lineage must contain six slots")
    split_counts = Counter(str(slot["split"]) for slot in slots)
    if split_counts != Counter({"train": 252, "dev": 60, "test": 48}):
        raise ValueError(f"calibration split mismatch: {split_counts}")
    route_splits: dict[str, set[str]] = defaultdict(set)
    for slot in slots:
        route_splits[str(slot["route_fixture_id"])].add(str(slot["split"]))
    if any(len(splits) != 1 for splits in route_splits.values()):
        raise ValueError("route fixture leaks across splits")
    pilot = [
        slot for slot in slots if str(slot["map_name"]) in CALIBRATION_MAPS[:2]
    ]
    if len(pilot) != 144:
        raise ValueError("two-map pilot must contain 144 slots")


LONG_SMOKE_CASES = (
    ("straight_clear", RouteManeuver.FOLLOW_STRAIGHT, "clear"),
    ("curve_left_clear", RouteManeuver.FOLLOW_CURVE_LEFT, "clear"),
    ("curve_right_clear", RouteManeuver.FOLLOW_CURVE_RIGHT, "clear"),
    ("junction_straight", RouteManeuver.JUNCTION_STRAIGHT, "clear"),
    ("turn_left", RouteManeuver.TURN_LEFT, "clear"),
    ("turn_right", RouteManeuver.TURN_RIGHT, "clear"),
    ("route_change_left", RouteManeuver.ROUTE_CHANGE_LEFT, "clear"),
    ("route_change_right", RouteManeuver.ROUTE_CHANGE_RIGHT, "clear"),
    ("follow_stop_resume", RouteManeuver.FOLLOW_STRAIGHT, "lead_braking"),
    ("cut_in_left", RouteManeuver.FOLLOW_STRAIGHT, "cut_in"),
    ("cut_in_right", RouteManeuver.FOLLOW_STRAIGHT, "cut_in"),
    ("left_turn_crossing_yield", RouteManeuver.TURN_LEFT, "crossing"),
    ("right_turn_merge_yield", RouteManeuver.TURN_RIGHT, "merge"),
    ("overtake_left_rejoin", RouteManeuver.FOLLOW_STRAIGHT, "obstruction"),
    ("overtake_right_rejoin", RouteManeuver.FOLLOW_STRAIGHT, "obstruction"),
    ("red_green_resume_route", RouteManeuver.TURN_LEFT, "traffic_control"),
)


def build_long_smoke_manifest_v3(*, learned: bool) -> dict[str, Any]:
    mode = "learned" if learned else "teacher_contract"
    cases = [
        {
            "case_id": f"r2v3-{mode}-{case_id}",
            "mode": mode,
            "maneuver": maneuver.value,
            "family": family,
            "duration_s": 20.0,
            "completion_required": True,
        }
        for case_id, maneuver, family in LONG_SMOKE_CASES
    ]
    body = {
        "schema_version": SCHEMA,
        "phase": "long_smoke",
        "mode": mode,
        "cases": cases,
        "required_passes": 16,
    }
    return {**body, "manifest_hash": canonical_sha256(body)}


def build_core_blind_manifest_v3() -> dict[str, Any]:
    cases = [
        (
            "turn_left",
            "turn_left_right",
            "clear",
            RouteManeuver.TURN_LEFT,
            AlternativeKind.NONE,
            False,
        ),
        (
            "turn_right",
            "turn_left_right",
            "clear",
            RouteManeuver.TURN_RIGHT,
            AlternativeKind.NONE,
            False,
        ),
        (
            "junction_straight",
            "crossing_turn_yield",
            "crossing",
            RouteManeuver.JUNCTION_STRAIGHT,
            AlternativeKind.TEMPORAL_YIELD,
            True,
        ),
        (
            "curve",
            "cut_in_left",
            "cut_in",
            RouteManeuver.FOLLOW_CURVE_LEFT,
            AlternativeKind.SPATIAL_AVOID,
            True,
        ),
        (
            "follow",
            "follow_stop",
            "lead_braking",
            RouteManeuver.FOLLOW_STRAIGHT,
            AlternativeKind.TEMPORAL_YIELD,
            True,
        ),
        (
            "cut_in_left",
            "cut_in_left",
            "cut_in",
            RouteManeuver.FOLLOW_STRAIGHT,
            AlternativeKind.SPATIAL_AVOID,
            True,
        ),
        (
            "cut_in_right",
            "cut_in_right",
            "cut_in",
            RouteManeuver.FOLLOW_STRAIGHT,
            AlternativeKind.SPATIAL_AVOID,
            True,
        ),
        (
            "overtake_left",
            "overtake_left",
            "obstruction",
            RouteManeuver.FOLLOW_STRAIGHT,
            AlternativeKind.SPATIAL_OVERTAKE,
            True,
        ),
        (
            "overtake_right",
            "overtake_right",
            "obstruction",
            RouteManeuver.FOLLOW_STRAIGHT,
            AlternativeKind.SPATIAL_OVERTAKE,
            True,
        ),
        (
            "crossing_left",
            "crossing_turn_yield",
            "crossing",
            RouteManeuver.TURN_LEFT,
            AlternativeKind.TEMPORAL_YIELD,
            True,
        ),
        (
            "merge_right",
            "merge_yield",
            "merge",
            RouteManeuver.TURN_RIGHT,
            AlternativeKind.TEMPORAL_YIELD,
            True,
        ),
        (
            "traffic_control",
            "traffic_control",
            "traffic_control",
            RouteManeuver.TURN_LEFT,
            AlternativeKind.TEMPORAL_YIELD,
            True,
        ),
    ]
    # Keep the twelve core fixtures spread across all seven audit maps.
    core_map_order = (
        "Town13",
        "Town04",
        "Town05",
        "Town10HD",
        "Town12",
        "Town06",
        "Town03",
        "Town03",
        "Town04",
        "Town05",
        "Town10HD",
        "Town12",
    )
    body = {
        "schema_version": SCHEMA,
        "phase": "core_blind",
        "cases": [
            {
                "pair_id": f"r2v3-core-{index:02d}-{name}",
                "fixture_id": f"blind-only-{index:02d}-{name}",
                "route_fixture_id": f"blind-only-{index:02d}-{name}-route-v1",
                "map_name": core_map_order[index],
                "template_id": template_id,
                "family": family,
                "maneuver": maneuver.value,
                "alternative_kind": alternative_kind.value,
                "dual_candidate_expected": dual,
                "seed_id": (
                    "seed_a" if index % 2 == 0 else "seed_b"
                ),
                "repeat_group": (
                    "symmetric_cut_in"
                    if name in {"cut_in_left", "cut_in_right"}
                    else (
                        "symmetric_overtake"
                        if name
                        in {"overtake_left", "overtake_right"}
                        else ""
                    )
                ),
                "calibration_overlap_forbidden": True,
            }
            for index, (
                name,
                template_id,
                family,
                maneuver,
                alternative_kind,
                dual,
            ) in enumerate(cases)
        ],
        "thresholds": {
            "comparable_min": 10,
            "decisive_min": 4,
            "candidate_0_wins_min": 2,
            "candidate_1_wins_min": 2,
            "candidate_1_win_families_min": 2,
        },
    }
    return {**body, "manifest_hash": canonical_sha256(body)}


def build_unseen_long_audit_manifest_v3() -> dict[str, Any]:
    """Freeze 16 post-checkpoint routes that are absent from every earlier gate."""
    cases = []
    template_by_case = {
        "straight_clear": "straight_curve",
        "curve_left_clear": "straight_curve",
        "curve_right_clear": "straight_curve",
        "junction_straight": "junction_straight",
        "turn_left": "turn_left_right",
        "turn_right": "turn_left_right",
        "route_change_left": "route_change_left_right",
        "route_change_right": "route_change_left_right",
        "follow_stop_resume": "follow_stop",
        "cut_in_left": "cut_in_left",
        "cut_in_right": "cut_in_right",
        "left_turn_crossing_yield": "crossing_turn_yield",
        "right_turn_merge_yield": "merge_yield",
        "overtake_left_rejoin": "overtake_left",
        "overtake_right_rejoin": "overtake_right",
        "red_green_resume_route": "traffic_control",
    }
    for index, (case_id, maneuver, family) in enumerate(LONG_SMOKE_CASES):
        cases.append(
            {
                "case_id": f"r2v3-unseen-{index:02d}-{case_id}",
                "fixture_id": f"unseen-only-{index:02d}-{case_id}",
                "route_fixture_id": (
                    f"unseen-only-{index:02d}-{case_id}-route-v1"
                ),
                "map_name": AUDIT_MAPS[(index + 2) % len(AUDIT_MAPS)],
                "template_id": template_by_case[case_id],
                "family": family,
                "maneuver": maneuver.value,
                "seed_id": (
                    "seed_a" if index % 2 == 0 else "seed_b"
                ),
                "duration_s": 20.0,
                "completion_required": True,
                "calibration_overlap_forbidden": True,
                "core_blind_overlap_forbidden": True,
                "world_ready_audit_overlap_forbidden": True,
            }
        )
    body = {
        "schema_version": SCHEMA,
        "phase": "unseen_long_audit",
        "checkpoint_frozen_before_execution": True,
        "cases": cases,
        "required_passes": 16,
    }
    return {**body, "manifest_hash": canonical_sha256(body)}


def build_world_ready_audit_manifest_v3() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for map_index, map_name in enumerate(AUDIT_MAPS):
        for template_index, (slot_id, template) in enumerate(
            _topology_balanced_template_slots(map_name)
        ):
            maneuver = template.maneuver_cycle[
                (map_index + template_index + 1)
                % len(template.maneuver_cycle)
            ]
            cases.append(
                {
                    "case_id": f"r2v3-audit-{map_name}-{slot_id}",
                    "fixture_id": (
                        f"audit-only-{map_name}-{slot_id}-v1"
                    ),
                    "route_fixture_id": (
                        f"audit-only-{map_name}-{slot_id}-route-v1"
                    ),
                    "map_name": map_name,
                    "template_slot_id": slot_id,
                    "template_id": template.template_id,
                    "family": template.family,
                    "maneuver": maneuver.value,
                    "seed_id": (
                        "seed_a"
                        if (map_index + template_index) % 2 == 0
                        else "seed_b"
                    ),
                    "audit_bucket": template.audit_bucket,
                    "calibration_overlap_forbidden": True,
                    "core_blind_overlap_forbidden": True,
                }
            )
    body = {
        "schema_version": SCHEMA,
        "phase": "world_ready_audit",
        "cases": cases,
        "thresholds": world_ready_thresholds(),
    }
    if len(cases) != 84 or len({case["fixture_id"] for case in cases}) != 84:
        raise AssertionError("World-ready audit must freeze 84 unique fixtures")
    bucket_counts = Counter(case["audit_bucket"] for case in cases)
    if bucket_counts != Counter({"road_or_traffic": 28, "interaction": 56}):
        raise AssertionError(f"World-ready audit 4/8 mix mismatch: {bucket_counts}")
    return {**body, "manifest_hash": canonical_sha256(body)}


def world_ready_thresholds() -> dict[str, Any]:
    return {
        "fatal_events_max": 0,
        "route_completion_overall_min": 0.90,
        "route_completion_per_class_min": 0.80,
        "candidate_1_available_min": 50,
        "comparable_min": 48,
        "decisive_min": 18,
        "candidate_0_wins_min": 6,
        "candidate_1_wins_min": 6,
        "safe_candidate_pair_rate_min": 0.95,
        "both_bad_rate_strict_max": 0.10,
        "systematic_group_min_cases": 3,
        "systematic_guard_mpc_failure_rate_max": 0.20,
    }


def evaluate_core_blind_v3(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 12:
        raise ValueError("core blind requires exactly 12 pair records")
    comparable = sum(bool(row.get("comparable")) for row in records)
    decisive = sum(bool(row.get("decisive")) for row in records)
    wins = Counter(
        int(row["winner"])
        for row in records
        if row.get("winner") in (0, 1)
    )
    c1_families = {
        str(row.get("family"))
        for row in records
        if row.get("winner") == 1 and bool(row.get("decisive"))
    }
    labels_by_fixture: dict[str, set[str]] = defaultdict(set)
    for row in records:
        repeat = str(row.get("repeat_group") or "")
        if repeat:
            labels_by_fixture[repeat].add(str(row.get("pair_label")))
    repeated_consistent = all(
        len(labels) == 1 for labels in labels_by_fixture.values()
    )
    fatal = sum(
        bool(row.get(key))
        for row in records
        for key in ("collision", "offroad", "wrong_exit")
    )
    candidate_contract = all(
        (
            "dual_candidate_expected" not in row
            or "candidate1_available" not in row
            or bool(row.get("candidate1_available"))
            == bool(row.get("dual_candidate_expected"))
        )
        for row in records
    )
    gates = {
        "record_count": len(records) == 12,
        "comparable": comparable >= 10,
        "decisive": decisive >= 4,
        "candidate_0_wins": wins[0] >= 2,
        "candidate_1_wins": wins[1] >= 2,
        "candidate_1_family_spread": len(c1_families) >= 2,
        "repeat_consistency": repeated_consistent,
        "candidate_availability_contract": candidate_contract,
        "fatal_events": fatal == 0,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "comparable": comparable,
            "decisive": decisive,
            "candidate_0_wins": wins[0],
            "candidate_1_wins": wins[1],
            "candidate_1_win_families": sorted(c1_families),
            "fatal_events": fatal,
        },
    }


def evaluate_world_ready_audit_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != 84:
        raise ValueError("World-ready audit requires exactly 84 records")
    thresholds = world_ready_thresholds()
    fatal = sum(
        bool(row.get(key))
        for row in records
        for key in ("collision", "offroad", "wrong_exit")
    )
    completed = sum(bool(row.get("route_completed")) for row in records)
    per_maneuver: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        per_maneuver[str(row["maneuver"])].append(
            bool(row.get("route_completed"))
        )
    per_rate = {
        maneuver: sum(values) / len(values)
        for maneuver, values in per_maneuver.items()
    }
    available = sum(bool(row.get("candidate1_available")) for row in records)
    comparable = sum(bool(row.get("comparable")) for row in records)
    decisive = sum(bool(row.get("decisive")) for row in records)
    wins = Counter(
        int(row["winner"])
        for row in records
        if row.get("winner") in (0, 1)
    )
    safe_rate = sum(bool(row.get("safe_candidate_exists")) for row in records) / 84
    both_bad_rate = sum(bool(row.get("both_bad")) for row in records) / 84

    group_failures: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        failure = bool(row.get("guard_mpc_failure"))
        for key in ("map_name", "maneuver", "family"):
            group_failures[f"{key}:{row.get(key)}"].append(failure)
    systematic = {
        group: sum(values) / len(values)
        for group, values in group_failures.items()
        if len(values) >= int(thresholds["systematic_group_min_cases"])
        and sum(values) / len(values)
        > float(thresholds["systematic_guard_mpc_failure_rate_max"])
    }
    gates = {
        "record_count": len(records) == 84,
        "fatal_events": fatal == 0,
        "route_completion_overall": completed / 84 >= 0.90,
        "route_completion_per_class": bool(per_rate)
        and min(per_rate.values()) >= 0.80,
        "candidate_1_available": available >= 50,
        "comparable": comparable >= 48,
        "decisive": decisive >= 18,
        "candidate_0_wins": wins[0] >= 6,
        "candidate_1_wins": wins[1] >= 6,
        "safe_candidate_pair_rate": safe_rate >= 0.95,
        "both_bad_rate": both_bad_rate < 0.10,
        "no_systematic_guard_mpc_failure": not systematic,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "fatal_events": fatal,
            "route_completion_overall": completed / 84,
            "route_completion_by_maneuver": per_rate,
            "candidate_1_available": available,
            "comparable": comparable,
            "decisive": decisive,
            "candidate_0_wins": wins[0],
            "candidate_1_wins": wins[1],
            "safe_candidate_pair_rate": safe_rate,
            "both_bad_rate": both_bad_rate,
            "systematic_guard_mpc_failures": systematic,
        },
    }


def evaluate_long_smoke_v3(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {case_id for case_id, _maneuver, _family in LONG_SMOKE_CASES}
    present = {str(row.get("case_id", "")).split("-", 3)[-1] for row in records}
    completed = all(
        bool(row.get("completed"))
        and not bool(row.get("collision"))
        and not bool(row.get("offroad"))
        and not bool(row.get("wrong_exit"))
        for row in records
    )
    return {
        "passed": len(records) == 16 and completed and present == expected,
        "completed": sum(bool(row.get("completed")) for row in records),
        "expected": 16,
        "missing": sorted(expected.difference(present)),
    }


def build_world_campaign_manifest_v3(
    *,
    formal_checkpoint_hash: str,
) -> dict[str, Any]:
    checkpoint = str(formal_checkpoint_hash).lower()
    if len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint):
        raise ValueError("formal checkpoint hash must be 64 lowercase hex characters")
    slots: list[dict[str, Any]] = []
    for map_index, map_name in enumerate(AUDIT_MAPS):
        for template_index, (slot_id, template) in enumerate(
            _topology_balanced_template_slots(map_name)
        ):
            maneuver = template.maneuver_cycle[
                (map_index + template_index + 2)
                % len(template.maneuver_cycle)
            ]
            lineage = f"world-v3-{map_name}-{slot_id}"
            for condition in CALIBRATION_CONDITIONS:
                for seed_id in WORLD_SEEDS:
                    slots.append(
                        {
                            "slot_id": (
                                f"{lineage}-{condition}-{seed_id}"
                            ),
                            "lineage_id": lineage,
                            "map_name": map_name,
                            "template_slot_id": slot_id,
                            "template_id": template.template_id,
                            "family": template.family,
                            "maneuver": maneuver.value,
                            "alternative_kind": (
                                template.alternative_kind.value
                            ),
                            "condition": condition,
                            "seed_id": seed_id,
                            "formal_checkpoint_hash": checkpoint,
                            "route_fixture_id": (
                                f"world-only-{lineage}-route-v1"
                            ),
                            "actor_script_id": (
                                f"world-only-{lineage}-{condition}-actors-v1"
                            ),
                            "fixture_id": (
                                f"world-only-{lineage}-{condition}-{seed_id}"
                            ),
                        }
                    )
    body = {
        "schema_version": SCHEMA,
        "phase": "world_campaign",
        "formal_checkpoint_hash": checkpoint,
        "route_schema": "safedrive.navigation_context.v3",
        "candidate_schema": "safedrive.k2.mixed_semantic.v3",
        "oracle_version": "oracle_v2_clearance_saturated",
        "route_fixtures_authored": False,
        "fixture_authoring_boundary": (
            "topology_only_before_world_outcomes"
        ),
        "slots": slots,
        "slot_count": 1008,
        "world_authority": "rank_guard_accepted_candidates_only",
    }
    if len(slots) != 1008 or len({slot["slot_id"] for slot in slots}) != 1008:
        raise AssertionError("World campaign requires 1008 unique slots")
    return {**body, "manifest_hash": canonical_sha256(body)}


__all__ = [
    "AUDIT_MAPS",
    "CALIBRATION_MAPS",
    "LONG_SMOKE_CASES",
    "TEMPLATES",
    "build_calibration_manifest_v3",
    "build_core_blind_manifest_v3",
    "build_long_smoke_manifest_v3",
    "build_unseen_long_audit_manifest_v3",
    "build_world_campaign_manifest_v3",
    "build_world_ready_audit_manifest_v3",
    "evaluate_core_blind_v3",
    "evaluate_long_smoke_v3",
    "evaluate_world_ready_audit_v3",
    "validate_calibration_manifest_v3",
    "world_ready_thresholds",
]
