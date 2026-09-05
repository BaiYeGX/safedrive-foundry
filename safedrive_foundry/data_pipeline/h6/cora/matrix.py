"""Pre-registered 351-root C2 data matrix and unused formal lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from data_pipeline.h2.contracts import ScenarioKey, stable_sha256

from .config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256


@dataclass(frozen=True)
class CoraMatrixRow:
    scenario: ScenarioKey
    split: str
    branch_order: tuple[str, str]
    expert_slot: int
    matrix_index: int
    collect: bool = True

    @property
    def root_id(self) -> str:
        return self.scenario.pair_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root_id"] = self.root_id
        return payload


def _balanced_rows(
    keys: list[tuple[str, ScenarioKey, bool]], *, start_index: int
) -> tuple[CoraMatrixRow, ...]:
    ranked = {
        key.pair_id: rank
        for rank, (_split, key, _collect) in enumerate(
            sorted(keys, key=lambda item: stable_sha256({"cora_c2": item[1].to_dict()}))
        )
    }
    return tuple(
        CoraMatrixRow(
            scenario=key,
            split=split,
            branch_order=("expert", "vla") if ranked[key.pair_id] % 2 == 0 else ("vla", "expert"),
            expert_slot=0 if ranked[key.pair_id] % 2 == 0 else 1,
            matrix_index=start_index + index,
            collect=collect,
        )
        for index, (split, key, collect) in enumerate(keys)
    )


def materialize_cora_matrix() -> tuple[tuple[CoraMatrixRow, ...], tuple[CoraMatrixRow, ...]]:
    maps = tuple(str(item) for item in CORA_C2_CONFIG["maps"])
    families = tuple(str(item) for item in CORA_C2_CONFIG["families"])
    weather = tuple(str(item) for item in CORA_C2_CONFIG["weather"])
    splits = CORA_C2_CONFIG["splits"]

    pilot_keys = [
        ("coverage_pilot", ScenarioKey(map_name, family, 137, "ClearNoon"), True)
        for map_name in maps
        for family in families
    ]
    development_keys = [
        (split, ScenarioKey(map_name, family, int(seed), weather_name), True)
        for split in ("train", "validation", "calibration", "locked_development")
        for seed in splits[split]
        for map_name in maps
        for family in families
        for weather_name in weather
    ]
    formal_keys = [
        ("reserved_formal", ScenarioKey(map_name, family, int(seed), weather_name), False)
        for seed in splits["reserved_formal"]
        for map_name in maps
        for family in families
        for weather_name in weather
    ]

    pilot = _balanced_rows(pilot_keys, start_index=0)
    development = _balanced_rows(development_keys, start_index=len(pilot))
    formal = _balanced_rows(formal_keys, start_index=0)
    data = pilot + development
    if len(data) != 351 or len(formal) != 108:
        raise AssertionError("cora_c2_matrix_count")
    if len({row.root_id for row in data + formal}) != 459:
        raise AssertionError("cora_c2_matrix_root_overlap")
    if sum(row.expert_slot == 0 for row in development) != 162:
        raise AssertionError("cora_c2_development_slot_balance")
    counts = {split: sum(row.split == split for row in data) for split in {
        "coverage_pilot", "train", "validation", "calibration", "locked_development"
    }}
    if counts != {
        "coverage_pilot": 27,
        "train": 162,
        "validation": 54,
        "calibration": 54,
        "locked_development": 54,
    }:
        raise AssertionError(f"cora_c2_split_counts:{counts}")
    return data, formal


CORA_DATA_MATRIX, CORA_FORMAL_MATRIX = materialize_cora_matrix()
CORA_MATRIX_SHA256 = stable_sha256(
    {
        "schema_version": "safedrive.cora.matrix.v1",
        "algorithm": CORA_C2_CONFIG["matrix_algorithm"],
        "config_sha256": CORA_C2_CONFIG_SHA256,
        "data": [row.to_dict() for row in CORA_DATA_MATRIX],
        "reserved_formal": [row.to_dict() for row in CORA_FORMAL_MATRIX],
    }
)
CORA_SMOKE_ROOT_IDS = frozenset(
    {
        "Town01__free_flow__s137__ClearNoon",
        "Town03__red_light_dilemma__s137__ClearNoon",
        "Town05__aggressive_cut_in__s137__ClearNoon",
    }
)


__all__ = [
    "CORA_DATA_MATRIX",
    "CORA_FORMAL_MATRIX",
    "CORA_MATRIX_SHA256",
    "CORA_SMOKE_ROOT_IDS",
    "CoraMatrixRow",
    "materialize_cora_matrix",
]
