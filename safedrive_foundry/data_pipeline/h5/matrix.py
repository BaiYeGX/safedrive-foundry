"""Frozen H5 scenario matrix from H4 locked test split and physical manifests.

This module reads only split metadata and physical manifests.  It never reads
H4 label shards, Oracle labels, Regression, or any outcome data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.h2.carla_scenarios import PhysicalScenario
from data_pipeline.h2.contracts import ScenarioKey
from data_pipeline.h3.contracts import stable_sha256

from .config import (
    CHALLENGE_PHYSICAL_MANIFEST_REL,
    H2_PHYSICAL_MANIFEST_REL,
    H3_SPLIT_MANIFEST_REL,
    H5_CONFIG,
)
from .contracts import H5Scenario


def _load_json(root: Path, rel: str, *, expected_sha: str | None = None) -> dict[str, Any]:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"missing_required_artifact:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_sha is not None:
        hash_key = "manifest_sha256" if "manifest_sha256" in payload else "physical_manifest_sha256"
        actual = payload.get(hash_key)
        if actual != expected_sha:
            raise ValueError(f"manifest_sha_mismatch:{rel}:{actual}!={expected_sha}")
        # Verify recomputed hash of the payload without the self hash.
        verify = {k: v for k, v in payload.items() if k != hash_key}
        if stable_sha256(verify) != expected_sha:
            raise ValueError(f"manifest_hash_recompute_failed:{rel}")
    return payload


def _scenarios_from_manifest(payload: dict[str, Any]) -> dict[str, PhysicalScenario]:
    out: dict[str, PhysicalScenario] = {}
    for row in payload["rows"]:
        item = PhysicalScenario.from_dict(row)
        if item.pair_id in out:
            raise ValueError(f"duplicate_physical_scenario:{item.pair_id}")
        out[item.pair_id] = item
    return out


def _arm_order(pair_id: str, index: int) -> tuple[str, ...]:
    # Deterministic Latin-square style permutations from the frozen arm order.
    from itertools import permutations

    perms = tuple(tuple(p) for p in permutations(("off", "on", "defer")))
    # Use stable pair hash + index to distribute first/second/third positions.
    digest = int(stable_sha256({"pair_id": pair_id, "seed": H5_CONFIG["matrix"]["arm_order_seed"]}), 16)
    return perms[(digest + index) % len(perms)]


def _pilot_scenarios(scenarios: Sequence[H5Scenario]) -> list[H5Scenario]:
    """Select 12 pilot scenarios balanced by map and family priority."""
    from collections import OrderedDict

    priority = [
        "emergency_lead_brake", "aggressive_cut_in", "red_light_dilemma",
        "cross_traffic_conflict", "cut_in", "free_flow", "slow_lead",
        "stopped_lead", "red_light_hold",
    ]
    chosen: list[H5Scenario] = []
    per_map: dict[str, OrderedDict[str, H5Scenario]] = {}
    for scenario in scenarios:
        per_map.setdefault(scenario.scenario.map_name, OrderedDict()).setdefault(scenario.scenario.family, scenario)
    for map_name in ("Town01", "Town03", "Town05"):
        families = per_map.get(map_name, {})
        selected = []
        for family in priority:
            if family in families and len(selected) < 4:
                selected.append(families[family])
        chosen.extend(selected)
    return chosen[: int(H5_CONFIG["matrix"]["pilot_count"])]


def load_h5_matrix(root: Path, *, full: bool = True) -> tuple[H5Scenario, ...]:
    split = _load_json(root, H3_SPLIT_MANIFEST_REL, expected_sha=H5_CONFIG["h3_split_manifest_sha256"])
    h2_manifest = _load_json(root, H2_PHYSICAL_MANIFEST_REL, expected_sha=H5_CONFIG["h2_physical_manifest_sha256"])
    challenge_manifest = _load_json(
        root, CHALLENGE_PHYSICAL_MANIFEST_REL, expected_sha=H5_CONFIG["challenge_physical_manifest_sha256"]
    )
    h2_rows = _scenarios_from_manifest(h2_manifest)
    challenge_rows = _scenarios_from_manifest(challenge_manifest)

    rows = [
        r for r in split.get("rows", ())
        if r.get("split") == H5_CONFIG["matrix"]["full_split"]
        and bool(r.get("valid_pair", False)) == bool(H5_CONFIG["matrix"]["full_valid_only"])
    ]
    if not rows:
        raise ValueError("empty_h5_test_rows")

    scenarios: list[H5Scenario] = []
    for index, row in enumerate(sorted(rows, key=lambda r: (r["map"], r["family"], r["seed"], r["weather"]))):
        pair_id = str(row["pair_id"])
        if pair_id in challenge_rows:
            physical = challenge_rows[pair_id]
            kind = "challenge"
        elif pair_id in h2_rows:
            physical = h2_rows[pair_id]
            kind = "h2"
        else:
            raise KeyError(f"physical_scenario_missing:{pair_id}")
        key = ScenarioKey(
            map_name=str(row["map"]),
            family=str(row["family"]),
            seed=int(row["seed"]),
            weather=str(row["weather"]),
        )
        scenarios.append(
            H5Scenario(
                pair_id=pair_id,
                scenario=key,
                physical_sha256=physical.physical_sha256,
                manifest_kind=kind,
                arm_order=_arm_order(pair_id, index),
                physical=physical,
            )
        )

    if not full:
        return tuple(_pilot_scenarios(scenarios))
    return tuple(scenarios)


def h5_matrix_sha256(scenarios: Sequence[H5Scenario]) -> str:
    return stable_sha256({"rows": [s.to_dict() for s in scenarios]})


__all__ = ["h5_matrix_sha256", "load_h5_matrix"]
