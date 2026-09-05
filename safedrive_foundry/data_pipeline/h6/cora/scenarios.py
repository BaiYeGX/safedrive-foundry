"""C2 physical-scenario materialization using the active H2/H3/H6 logic."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from data_pipeline.h2.carla_scenarios import materialize_physical_scenario
from data_pipeline.h2.matrix import MatrixEntry
from data_pipeline.h3.challenge_matrix_v2 import (
    CHALLENGE_FAMILIES,
    materialize_challenge_physical_scenario,
)
from data_pipeline.h6.matrix import _h6_pre_roll_script

from .matrix import CoraMatrixRow


def materialize_cora_physical_scenario(
    world: Any, row: CoraMatrixRow
):
    """Resolve one frozen CORA row without borrowing historical seed tables."""

    entry = MatrixEntry(
        scenario=row.scenario,
        branch_order=row.branch_order,
        expert_slot=row.expert_slot,
        matrix_index=row.matrix_index,
    )
    physical = (
        materialize_challenge_physical_scenario(world, entry)
        if row.scenario.family in CHALLENGE_FAMILIES
        else materialize_physical_scenario(world, entry)
    )
    return replace(
        physical,
        script=_h6_pre_roll_script(dict(physical.script)),
    )


__all__ = ["materialize_cora_physical_scenario"]
