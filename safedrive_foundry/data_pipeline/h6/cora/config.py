"""Frozen C2 protocol configuration and identity."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping

from data_pipeline.h2.contracts import stable_sha256


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "safedrive_foundry" / "config" / "h6" / "cora_c2.toml"


def _load() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        payload = tomllib.load(handle)
    _validate(payload)
    return payload


def _validate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "safedrive.cora.data_config.v1":
        raise ValueError("cora_c2_config_schema")
    if len(payload.get("maps", ())) != 3 or len(payload.get("families", ())) != 9:
        raise ValueError("cora_c2_config_matrix_dimensions")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("cora_c2_config_splits")
    all_seeds = [int(seed) for values in splits.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("cora_c2_config_seed_overlap")
    forbidden = {0, 1, 2, 3, 89, 97, 101, 103, 107, 109, 113, 127, 131}
    if forbidden.intersection(all_seeds):
        raise ValueError("cora_c2_config_historical_seed_overlap")
    if int(payload["resources"]["root_attempt_limit"]) != 351:
        raise ValueError("cora_c2_config_root_limit")
    if int(payload["interventions"]["max_per_root"]) != 2:
        raise ValueError("cora_c2_config_intervention_limit")


CORA_C2_CONFIG = _load()
CORA_C2_CONFIG_SHA256 = stable_sha256(CORA_C2_CONFIG)


def config_identity() -> dict[str, Any]:
    return {
        "path": str(CONFIG_PATH.relative_to(ROOT)),
        "sha256": CORA_C2_CONFIG_SHA256,
        "payload": CORA_C2_CONFIG,
    }


__all__ = [
    "CONFIG_PATH",
    "CORA_C2_CONFIG",
    "CORA_C2_CONFIG_SHA256",
    "config_identity",
]
