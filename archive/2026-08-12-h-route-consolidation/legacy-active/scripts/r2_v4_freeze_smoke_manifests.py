#!/usr/bin/env python3
"""Freeze the learned-only R2 V4 16/32-case smoke manifests.

The runner consumes explicit ``scenario_id``/``seed_id`` pairs.  This small
authoring command resolves those pairs from the already-frozen scenario
registry before any head outcome is observed; it never samples or rewrites a
case after collection.  The 32-case manifest is the same 16 route fixtures
under the second registered seed, and every row is marked as an unseen
seed/route evaluation case for the live gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_contract import canonical_json_bytes, content_hash  # noqa: E402
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    R2V3_LONG_SMOKE_SCENARIO_IDS,
    ScenarioSeedFixture,
    load_scenario_registry,
)

SCHEMA = "safedrive.r2.v4.learned_smoke_manifest.v1"


def _manifest_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _fixture_map(fixtures: Iterable[ScenarioSeedFixture]) -> dict[tuple[str, str], ScenarioSeedFixture]:
    result: dict[tuple[str, str], ScenarioSeedFixture] = {}
    for fixture in fixtures:
        key = (str(fixture.scenario_id), str(fixture.seed_id))
        if key in result:
            raise ValueError(f"duplicate smoke fixture identity: {key}")
        result[key] = fixture
    return result


def build_manifest(
    fixtures: Iterable[ScenarioSeedFixture],
    *,
    expected_cases: int,
    source_registry_hashes: list[str] | None = None,
) -> dict[str, Any]:
    if expected_cases not in {16, 32}:
        raise ValueError("learned V4 smoke supports exactly 16 or 32 cases")
    by_key = _fixture_map(fixtures)
    scenario_ids = tuple(str(value) for value in R2V3_LONG_SMOKE_SCENARIO_IDS)
    seeds = ("seed_a",) if expected_cases == 16 else ("seed_a", "seed_b")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for scenario_id in scenario_ids:
            fixture = by_key.get((scenario_id, seed))
            if fixture is None:
                raise ValueError(f"registry missing learned smoke fixture: {scenario_id}/{seed}")
            repeat_group = f"r2v4-learned-smoke|{scenario_id}"
            rows.append(
                {
                    "case_id": f"r2v4-learned-{scenario_id}-{seed}",
                    "scenario_id": scenario_id,
                    "seed_id": seed,
                    "map_name": str(fixture.map_name),
                    "family": str(fixture.family),
                    "maneuver": str((fixture.route.navigation_context or {}).get("maneuver") or ""),
                    "unseen_seed_or_route": expected_cases == 32,
                    "repeat_group": repeat_group,
                    "aa_noise_identity": content_hash(
                        {
                            "namespace": "r3_aa_noise_probe",
                            "repeat_group": repeat_group,
                            "candidate_id": "v3_nominal_progress",
                        }
                    ),
                }
            )
    body = {
        "schema_version": SCHEMA,
        "campaign_id": f"r2-v4-learned-only-{expected_cases}-case-smoke-v1",
        "expected_cases": expected_cases,
        "case_order": "seed_then_frozen_route_fixture",
        "source_registry_sha256": list(source_registry_hashes or []),
        "outcome_used_in_authoring": False,
        "cases": rows,
    }
    if len(rows) != expected_cases:
        raise AssertionError(f"learned smoke count mismatch: {len(rows)} != {expected_cases}")
    return {**body, "manifest_hash": _manifest_hash(body)}


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen smoke manifest: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", action="append", required=True)
    parser.add_argument("--output-16", required=True)
    parser.add_argument("--output-32", required=True)
    args = parser.parse_args()
    registries = [load_scenario_registry(Path(value).resolve()) for value in args.registry]
    fixtures = [fixture for registry in registries for fixture in registry.fixtures]
    hashes = [str(registry.registry_sha256 or registry.compute_registry_sha256()) for registry in registries]
    smoke16 = build_manifest(fixtures, expected_cases=16, source_registry_hashes=hashes)
    smoke32 = build_manifest(fixtures, expected_cases=32, source_registry_hashes=hashes)
    _write_exclusive(Path(args.output_16).resolve(), smoke16)
    _write_exclusive(Path(args.output_32).resolve(), smoke32)
    print(
        json.dumps(
            {
                "output_16": str(Path(args.output_16).resolve()),
                "output_32": str(Path(args.output_32).resolve()),
                "hash_16": smoke16["manifest_hash"],
                "hash_32": smoke32["manifest_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

