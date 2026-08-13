"""Materialization of the immutable 120-anchor H2 scenario matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import ScenarioKey, stable_sha256


MAPS = ("Town01", "Town03", "Town05")
FAMILIES = ("free_flow", "slow_lead", "stopped_lead", "cut_in", "red_light_hold")
SEEDS = (0, 1, 2, 3)
WEATHERS = ("ClearNoon", "CloudyNoon")
MATRIX_ALGORITHM = "h2-fixed-matrix-v1"


@dataclass(frozen=True)
class MatrixEntry:
    scenario: ScenarioKey
    branch_order: tuple[str, str]
    expert_slot: int
    matrix_index: int

    @property
    def pair_id(self) -> str:
        return self.scenario.pair_id

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pair_id"] = self.pair_id
        return result


def materialize_matrix() -> tuple[MatrixEntry, ...]:
    keys = tuple(
        ScenarioKey(map_name, family, seed, weather)
        for map_name in MAPS
        for family in FAMILIES
        for seed in SEEDS
        for weather in WEATHERS
    )
    # Rank by the pair hash, then alternate. This is hash-derived and exactly
    # balanced across the frozen 120 rows rather than merely probabilistic.
    hash_rank = {
        key.pair_id: rank
        for rank, key in enumerate(sorted(keys, key=lambda item: stable_sha256(item.to_dict())))
    }
    rows: list[MatrixEntry] = []
    for index, key in enumerate(keys):
        expert_first = hash_rank[key.pair_id] % 2 == 0
        rows.append(
            MatrixEntry(
                scenario=key,
                branch_order=("expert", "vla") if expert_first else ("vla", "expert"),
                expert_slot=0 if expert_first else 1,
                matrix_index=index,
            )
        )
    return tuple(rows)


FIXED_MATRIX = materialize_matrix()
PILOT_MATRIX = tuple(
    row for row in FIXED_MATRIX if row.scenario.seed == 0 and row.scenario.weather == "ClearNoon"
)
MATRIX_SHA256 = stable_sha256(
    {"algorithm": MATRIX_ALGORITHM, "rows": [row.to_dict() for row in FIXED_MATRIX]}
)


assert len(FIXED_MATRIX) == 120
assert len(PILOT_MATRIX) == 15
assert sum(row.expert_slot == 0 for row in FIXED_MATRIX) == 60


__all__ = [
    "FAMILIES", "FIXED_MATRIX", "MAPS", "MATRIX_ALGORITHM", "MATRIX_SHA256",
    "MatrixEntry", "PILOT_MATRIX", "SEEDS", "WEATHERS", "materialize_matrix",
]
