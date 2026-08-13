"""Canonical H2 protocol configuration and deterministic identity."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .contracts import stable_sha256


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "safedrive_foundry" / "config" / "h2" / "paired_outcomes.toml"


def _load() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        payload = tomllib.load(handle)
    # TOML parsing returns ordinary dict/list/scalar values; round-tripping via
    # stable_sha256 below is the authoritative normalization for the contract.
    return payload


H2_CONFIG = _load()
H2_CONFIG_SHA256 = stable_sha256(H2_CONFIG)


def config_identity() -> dict[str, Any]:
    """Return the immutable config path, normalized payload and SHA256."""

    return {
        "path": str(CONFIG_PATH.relative_to(ROOT)),
        "sha256": H2_CONFIG_SHA256,
        "payload": H2_CONFIG,
    }


__all__ = ["CONFIG_PATH", "H2_CONFIG", "H2_CONFIG_SHA256", "config_identity"]
