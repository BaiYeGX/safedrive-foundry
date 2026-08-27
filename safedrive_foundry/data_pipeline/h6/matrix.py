"""Fresh H6 closed-loop matrix, disjoint from all H2-H5 seed lineages."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from data_pipeline.h2.carla_scenarios import materialize_physical_scenario
from data_pipeline.h2.contracts import ScenarioKey, stable_sha256
from data_pipeline.h2.matrix import MatrixEntry
from data_pipeline.h3.challenge_matrix_v2 import (
    CHALLENGE_FAMILIES,
    materialize_challenge_physical_scenario,
)
from data_pipeline.h5.contracts import H5Scenario
from data_pipeline.h6.config import H6_VLA75_FORMAL_LINEAGES, _validate_lineage_id


H6_MAPS = ("Town01", "Town03", "Town05")
H6_BASE_FAMILIES = ("free_flow", "slow_lead", "stopped_lead", "cut_in", "red_light_hold")
H6_FAMILIES = H6_BASE_FAMILIES + tuple(CHALLENGE_FAMILIES)
H6_SEEDS = (101, 103)
H6_TRAIN_SEEDS = (89, 97)
H6_WEATHERS = ("ClearNoon", "CloudyNoon")


def _entry(key: ScenarioKey, index: int) -> MatrixEntry:
    expert_first = int(stable_sha256({"h6": key.to_dict()}), 16) % 2 == 0
    return MatrixEntry(
        scenario=key,
        branch_order=("expert", "vla") if expert_first else ("vla", "expert"),
        expert_slot=0 if expert_first else 1,
        matrix_index=index,
    )


def _matrix(*, seeds: tuple[int, ...], manifest_kind: str, full: bool) -> tuple[H5Scenario, ...]:
    keys = [
        ScenarioKey(map_name, family, seed, weather)
        for map_name in H6_MAPS
        for family in H6_FAMILIES
        for seed in seeds
        for weather in H6_WEATHERS
    ]
    rows = []
    for index, key in enumerate(keys):
        rows.append(
            H5Scenario(
                pair_id=key.pair_id,
                scenario=key,
                physical_sha256="pending_live_materialization",
                manifest_kind=manifest_kind,
                arm_order=("off", "on"),
                physical=None,
            )
        )
    if full:
        return tuple(rows)
    # Formal acceptance keeps twelve held-out rows.  Development needs both
    # training seeds for every pilot family, otherwise the old alternating
    # scheme confounded seed with difficulty (seed 89 saw only free-flow/cut-in
    # while seed 97 saw only emergency/red-light).  The balanced 24-row
    # training pilot gives both the gradient and calibration splits the same
    # family support without touching held-out acceptance seeds.
    priority = ("free_flow", "emergency_lead_brake", "aggressive_cut_in", "red_light_dilemma")
    pilot = []
    for map_index, map_name in enumerate(H6_MAPS):
        for family_index, family in enumerate(priority):
            pilot_seeds = seeds if manifest_kind == "h6_fresh_training" else seeds[:1]
            for pilot_seed in pilot_seeds:
                pilot.append(
                    next(
                        row
                        for row in rows
                        if row.scenario.map_name == map_name
                        and row.scenario.family == family
                        and row.scenario.seed == pilot_seed
                        and row.scenario.weather == H6_WEATHERS[0]
                    )
                )
    return tuple(pilot)


def load_h6_matrix(*, full: bool) -> tuple[H5Scenario, ...]:
    """Held-out formal acceptance matrix; never valid as training input."""

    return _matrix(seeds=H6_SEEDS, manifest_kind="h6_fresh", full=full)


def load_h6_training_matrix(*, full: bool) -> tuple[H5Scenario, ...]:
    """Development collection matrix, seed-disjoint from formal acceptance."""

    return _matrix(
        seeds=H6_TRAIN_SEEDS,
        manifest_kind="h6_fresh_training",
        full=full,
    )


def _vla75_matrix(
    *, lineage_id: str, full: bool
) -> tuple[H5Scenario, ...]:
    """Build one immutable v2 formal matrix.

    The pilot is deliberately the first seed, ClearNoon, and the four
    declared priority families.  Full is the Cartesian product of three maps,
    nine families, two lineage seeds and two weathers (108 pairs).  No row is
    borrowed from the consumed v1 seed or from another lineage.
    """

    lineage = _validate_lineage_id(lineage_id)
    seeds = H6_VLA75_FORMAL_LINEAGES[lineage]
    manifest_kind = f"h6_vla75_fresh_lineage_{lineage}"
    keys = [
        ScenarioKey(map_name, family, seed, weather)
        for map_name in H6_MAPS
        for family in H6_FAMILIES
        for seed in seeds
        for weather in H6_WEATHERS
    ]
    rows = tuple(
        H5Scenario(
            pair_id=key.pair_id,
            scenario=key,
            physical_sha256="pending_live_materialization",
            manifest_kind=manifest_kind,
            arm_order=("off", "on"),
            physical=None,
        )
        for key in keys
    )
    if full:
        return rows
    priority = (
        "free_flow",
        "emergency_lead_brake",
        "aggressive_cut_in",
        "red_light_dilemma",
    )
    first_seed = seeds[0]
    return tuple(
        row
        for map_name in H6_MAPS
        for family in priority
        for row in rows
        if row.scenario.map_name == map_name
        and row.scenario.family == family
        and row.scenario.seed == first_seed
        and row.scenario.weather == H6_WEATHERS[0]
    )


def load_h6_vla75_matrix(
    lineage_id: str, full: bool
) -> tuple[H5Scenario, ...]:
    """Return the formal v2 matrix for ``lineage_id``."""

    return _vla75_matrix(lineage_id=lineage_id, full=full)


def h6_vla75_matrix_sha256(
    lineage_id: str, rows: Sequence[H5Scenario] | None = None
) -> str:
    """Hash lineage identity and matrix rows for the formal run lock."""

    lineage = _validate_lineage_id(lineage_id)
    selected = (
        tuple(rows)
        if rows is not None
        else load_h6_vla75_matrix(lineage, full=True)
    )
    return stable_sha256(
        {
            "schema_version": "safedrive.h6.vla75.matrix.v2",
            "lineage_id": lineage,
            "rows": [
                {
                    "pair_id": row.pair_id,
                    "scenario": row.scenario.to_dict(),
                    "arms": list(row.arm_order),
                    "manifest_kind": row.manifest_kind,
                }
                for row in selected
            ],
        }
    )


def _h6_pre_roll_script(source: dict[str, Any]) -> dict[str, Any]:
    # Preserve each scenario's event timing.  The first H6 pilot forced every
    # family to 80 throttle-only ticks; on curved Town01 routes that drove the
    # ego off route and also moved cut-in actors past their intended anchor.
    # H6 instead keeps the scenario-defined 10/20/40 ticks, follows the route,
    # and adds bounded extra ticks only when the anchor speed is still low.
    script = dict(source)
    script.update(
        {
            "pre_roll_ticks": max(1, int(script.get("pre_roll_ticks", 20))),
            "pre_roll_target_speed_mps": max(
                4.0, float(script.get("pre_roll_target_speed_mps", 0.0))
            ),
            "pre_roll_kp": max(0.5, float(script.get("pre_roll_kp", 0.0))),
            "pre_roll_max_throttle": max(
                0.65, float(script.get("pre_roll_max_throttle", 0.0))
            ),
            "pre_roll_min_ready_speed_mps": 2.5,
            "pre_roll_max_extra_ticks": 80,
            "pre_roll_route_follow": True,
            "pre_roll_min_lookahead_m": 4.0,
            "pre_roll_speed_lookahead_s": 1.2,
            "pre_roll_max_steer": 0.35,
            # Staging is kinematic and route-following so both paired arms
            # start from the same observable pose/speed instead of accumulating
            # 10–20 cm of CARLA throttle/physics variation before evaluation.
            "pre_roll_kinematic": True,
            "pre_roll_kinematic_speed_mps": 3.0,
            "spectator_follow_ego": True,
            "spectator_follow_distance_m": 8.0,
            "spectator_follow_height_m": 4.0,
            "spectator_follow_pitch_deg": -15.0,
            "spectator_follow_hz": 20.0,
            "dynamic_traffic_light_timing": True,
        }
    )
    return script


def materialize_h6_scenario(world: Any, scenario: H5Scenario):
    key = scenario.scenario
    all_rows = (
        load_h6_matrix(full=True)
        + load_h6_training_matrix(full=True)
        + tuple(
            row
            for lineage in H6_VLA75_FORMAL_LINEAGES
            for row in load_h6_vla75_matrix(lineage, full=True)
        )
    )
    # Seed 103 is intentionally present in the historical v1 matrix metadata
    # and in fresh VLA75 lineage A.  Pair ids alone therefore are not a safe
    # lookup key: selecting the first row would silently reuse the old matrix
    # manifest.  Require the lineage/manifest identity first and only fall
    # back to pair-id for legacy callers whose row predates that field.
    index = next(
        (
            index
            for index, row in enumerate(all_rows)
            if row.pair_id == scenario.pair_id
            and row.manifest_kind == scenario.manifest_kind
        ),
        None,
    )
    if index is None:
        index = next(
            index for index, row in enumerate(all_rows) if row.pair_id == scenario.pair_id
        )
    entry = _entry(key, index)
    if key.family in CHALLENGE_FAMILIES:
        physical = materialize_challenge_physical_scenario(world, entry)
    else:
        physical = materialize_physical_scenario(world, entry)
    return replace(physical, script=_h6_pre_roll_script(dict(physical.script)))


def h6_matrix_sha256(rows: Sequence[H5Scenario]) -> str:
    return stable_sha256(
        {
            "lineage": "h6-fresh-seeds-v1",
            "rows": [
                {
                    "scenario": row.scenario.to_dict(),
                    "arms": list(row.arm_order),
                    "manifest_kind": row.manifest_kind,
                }
                for row in rows
            ],
        }
    )


assert len(load_h6_matrix(full=True)) == 108
assert len(load_h6_matrix(full=False)) == 12
assert len(load_h6_training_matrix(full=True)) == 108
assert len(load_h6_training_matrix(full=False)) == 24
assert set(H6_SEEDS).isdisjoint(H6_TRAIN_SEEDS)

__all__ = [
    "H6_FAMILIES",
    "H6_MAPS",
    "H6_SEEDS",
    "H6_TRAIN_SEEDS",
    "H6_WEATHERS",
    "h6_matrix_sha256",
    "_h6_pre_roll_script",
    "load_h6_matrix",
    "load_h6_training_matrix",
    "load_h6_vla75_matrix",
    "h6_vla75_matrix_sha256",
    "materialize_h6_scenario",
]
