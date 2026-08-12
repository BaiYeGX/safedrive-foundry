#!/usr/bin/env python3
"""Bind the V4 calibration manifest to the geometry-only author registries.

The symbolic campaign uses one slot identity per condition *and seed*.  A
Scenario Registry stores the condition as ``scenario_id`` and the cold seed as
``seed_id``.  This command makes that contract explicit without changing the
original frozen campaign: it strips the seed suffix from ``scenario_id``,
records the exact registry and author-manifest hashes for every root lineage,
and emits a new content-hashed campaign/pilot pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _condition_scenario_id(value: str) -> str:
    token = str(value)
    if token.endswith("__seed_a") or token.endswith("__seed_b"):
        return token.rsplit("__", 1)[0]
    return token


def bind(campaign: dict[str, Any], author_root: Path) -> dict[str, Any]:
    author_root = author_root.resolve()
    by_lineage: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(author_root.rglob("author_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lineage = str(manifest.get("lineage_id") or "")
        row = dict(manifest.get("report_row") or {})
        registry_path = Path(str(row.get("registry_path") or manifest_path.parent / "scenario_registry.toml"))
        if not registry_path.is_absolute():
            registry_path = (ROOT / registry_path).resolve()
        if not lineage or not registry_path.is_file():
            raise ValueError(f"invalid author manifest: {manifest_path}")
        by_lineage[lineage] = {
            "registry_path": str(registry_path),
            "registry_sha256": sha256(registry_path),
            "author_manifest_path": str(manifest_path.resolve()),
            "author_manifest_sha256": sha256(manifest_path),
            "topology_sha256": str(manifest.get("topology_sha256") or ""),
            "source_manifest_hash": str(manifest.get("source_manifest_hash") or ""),
        }
    if len(by_lineage) != 84:
        raise ValueError(f"expected 84 author lineages, found {len(by_lineage)}")
    body = json.loads(json.dumps(campaign))
    lineages = []
    for raw in body.get("lineages", []):
        lineage = str(raw["lineage_id"])
        binding = by_lineage.get(lineage)
        if binding is None:
            raise ValueError(f"missing author binding for {lineage}")
        row = dict(raw)
        row["author_binding"] = binding
        lineages.append(row)
    body["lineages"] = lineages
    slots = []
    lineage_map = {str(row["lineage_id"]): row for row in lineages}
    for raw in body.get("slots", []):
        row = dict(raw)
        row["scenario_id"] = _condition_scenario_id(str(row["scenario_id"]))
        binding = dict(lineage_map[str(row["lineage_id"])]["author_binding"])
        row["registry_path"] = binding["registry_path"]
        row["registry_sha256"] = binding["registry_sha256"]
        row["author_manifest_sha256"] = binding["author_manifest_sha256"]
        row["authoring_source_manifest_hash"] = binding["source_manifest_hash"]
        slots.append(row)
    body["slots"] = slots
    body["binding"] = {
        "author_root": str(author_root),
        "author_lineage_count": len(by_lineage),
        "registry_sha256_set": sorted({row["registry_sha256"] for row in by_lineage.values()}),
        "scenario_id_contract": "condition-level scenario_id plus seed_id",
        "source_campaign_manifest_hash": str(campaign.get("manifest_hash") or ""),
    }
    body["schema_version"] = "safedrive.r2.v4.calibration_manifest.v3"
    body.pop("manifest_hash", None)
    body["manifest_hash"] = content_hash(body)
    return body


def pilot_from(campaign: dict[str, Any]) -> dict[str, Any]:
    ids = set(campaign.get("pilot_lineages") or [])
    pilot = dict(campaign)
    pilot["campaign_id"] = "r2-v4-counterfactual-pilot-bound-v1"
    pilot["lineages"] = [row for row in campaign["lineages"] if str(row["lineage_id"]) in ids]
    pilot["slots"] = [row for row in campaign["slots"] if str(row["lineage_id"]) in ids]
    pilot.pop("manifest_hash", None)
    pilot["manifest_hash"] = content_hash(pilot)
    return pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--author-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output", required=True)
    args = parser.parse_args()
    campaign = json.loads(Path(args.campaign).read_text(encoding="utf-8"))
    bound = bind(campaign, Path(args.author_root))
    out = Path(args.output).resolve()
    pilot_out = Path(args.pilot_output).resolve()
    for path, value in ((out, bound), (pilot_out, pilot_from(bound))):
        if path.exists():
            raise SystemExit(f"refusing to overwrite bound manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "pilot_output": str(pilot_out), "manifest_hash": bound["manifest_hash"], "lineages": 84, "slots": len(bound["slots"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
