"""Bounded single-CARLA collector for the C2 repair delta."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data_pipeline.h2.contracts import stable_sha256
from data_pipeline.h2.gpu import GPUMemorySampler
from data_pipeline.h2.carla_scenarios import PhysicalScenario
from data_pipeline.h2.matrix import MatrixEntry
from data_pipeline.h2.contracts import ScenarioKey
from driving_vla.hybrid import ClassicExpertGenerator, NominalVLAGenerator
from driving_vla.model.nominal_policy import NominalVLAPolicy
from runtime import RunRegistry

from .config import CORA_C2_CONFIG
from .contracts import CoraProposal
from .interventions import InterventionNotApplicable, InterventionResult, derive_scaled_intervention
from .live import (
    DATA_ROOT, ROOT, _collect_root, _connection, _require_clean_scene, _resource_snapshot,
)
from .matrix import CoraMatrixRow
from .repair import (
    read_json, read_config, repair_rows, plan_payload, write_plan,
)
from .scenarios import materialize_cora_physical_scenario
from .store import CoraDataStore


REPAIR_EVIDENCE = ROOT / "docs" / "runtime-evidence" / "h6" / "h6-cora-c2-repair-20260905-v2"
BASE = ROOT / "generated" / "h6" / "cora" / "h6-cora-c2-dev-20260830-v1"


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        __import__('os').replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _target(index: int, split: str, diagnostic: bool) -> str:
    if diagnostic:
        return "repair_failure" if index < 6 else "offroad"
    if split == "train":
        return "repair_failure" if index < 20 else "offroad"
    return "repair_failure" if index < 6 else "offroad"


def _recipe_for(row: CoraMatrixRow, target: str, source: str) -> tuple[str, float]:
    # Fixed, pre-registered recipe ordering; no held-out outcome is consulted.
    operators = {
        "repair_failure": {"expert": ("speed_scale_up", 3.0), "vla": ("speed_scale_up", 3.0)},
        "offroad": {"expert": ("lateral_offset_toward_conflict", 3.0), "vla": ("lateral_offset_toward_conflict", 3.0)},
    }
    preferred = operators[target][source]
    # Every family has two pre-registered operators; if the global recipe is
    # inapplicable, use the first family operator at the same multiplier.
    from .interventions import FAMILY_OPERATORS
    if preferred[0] in FAMILY_OPERATORS[row.scenario.family]:
        return preferred
    return FAMILY_OPERATORS[row.scenario.family][0], preferred[1]


def _factory(row: CoraMatrixRow, target: str):
    def factory(root_id: str, family: str, anchor: Any, candidates: Sequence[Any]):
        by_source = {item.provenance.source.value: item for item in candidates}
        results = []
        for source in ("expert", "vla"):
            base = by_source.get(source)
            if base is None:
                continue
            operator, multiplier = _recipe_for(row, target, source)
            try:
                results.append(derive_scaled_intervention(root_id, anchor, base, operator, multiplier))
            except InterventionNotApplicable as exc:
                results.append(InterventionResult(root_id, source, operator, "NOT_APPLICABLE", error=str(exc)))
        return tuple(results)
    return factory


def _new_run_lock(config: Mapping[str, Any], plan: Mapping[str, Any], physical: Sequence[PhysicalScenario]) -> dict[str, Any]:
    payload = {
        "schema_version": "safedrive.cora.repair_run_lock.v1", "dataset_id": config["dataset_id"],
        "base_dataset_id": config["base_dataset_id"], "base_verification": config["base_verification"],
        "plan_sha256": plan["plan_sha256"], "physical_root_count": len(physical),
        "physical_ids": [item.pair_id for item in physical],
        "model_identity_reference": "base_run_lock_recorded_identity_only",
        "created_wall_time_s": time.time(), "formal_collected": False,
    }
    payload["repair_run_lock_sha256"] = stable_sha256(payload)
    return payload


def collect_plan(config_path: Path | str, *, diagnostic: bool = True, batch: int = 0) -> dict[str, Any]:
    config = read_config(config_path)
    dataset = DATA_ROOT / config["dataset_id"]
    store = CoraDataStore(DATA_ROOT, config["dataset_id"])
    rows = repair_rows(config, batch=batch, diagnostic=diagnostic)
    registered = {(str(item["scenario"]["map_name"]), str(item["scenario"]["family"]),
                   int(item["scenario"]["seed"]), str(item["scenario"]["weather"]))
                  for item in read_json(BASE / "scenario-manifest.json").get("rows", ())}
    overlap = [row.root_id for row in rows if (row.scenario.map_name, row.scenario.family,
               row.scenario.seed, row.scenario.weather) in registered]
    if overlap:
        raise RuntimeError(f"repair_seed_overlap:{overlap[:3]}")
    plan = plan_payload(config, rows, batch=None if diagnostic else batch, diagnostic=diagnostic)
    plan_name = "diagnostic-plan.json" if diagnostic else f"batch-{batch + 1}-plan.json"
    if (dataset / plan_name).exists():
        existing = read_json(dataset / plan_name)
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("repair_plan_conflict")
        plan = existing
    else:
        write_plan(dataset, plan, plan_name)
    # One CARLA connection and one ScenarioRuntime tick owner for the entire plan.
    client, world, report = _connection(str(config["map"]))
    _require_clean_scene(world)
    physical = []
    fingerprints = set()
    row_by_id = {row.root_id: row for row in rows}
    for row in rows:
        item = materialize_cora_physical_scenario(world, row)
        fingerprint = stable_sha256({"scenario": row.scenario.to_dict(), "route": item.route,
            "ego_transform": item.ego_transform, "npc_actors": item.npc_actors,
            "weather": item.weather, "script": item.script, "red_light": item.red_light})
        if fingerprint in fingerprints:
            raise RuntimeError(f"repair_duplicate_physical_initial_state:{row.root_id}")
        fingerprints.add(fingerprint)
        physical.append(item)
    physical_manifest = {"schema_version": "safedrive.cora.repair_physical_manifest.v1",
        "dataset_id": config["dataset_id"], "map": config["map"], "plan_sha256": plan["plan_sha256"],
        "rows": [item.to_dict() for item in physical], "formal_collected": False}
    physical_manifest["physical_manifest_sha256"] = stable_sha256(physical_manifest)
    if (dataset / "scenario-manifest.json").exists():
        old = read_json(dataset / "scenario-manifest.json")
        if old.get("physical_manifest_sha256") != physical_manifest["physical_manifest_sha256"]:
            raise RuntimeError("repair_physical_manifest_conflict")
    else:
        store.write_immutable_json(dataset / "scenario-manifest.json", physical_manifest)
    run_lock = _new_run_lock(config, plan, physical)
    if not (dataset / "run-lock.json").exists():
        store.write_immutable_json(dataset / "run-lock.json", run_lock)
    else:
        run_lock = read_json(dataset / "run-lock.json")
    REPAIR_EVIDENCE.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(REPAIR_EVIDENCE / "run-registry.sqlite3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    # Use the recorded generator identity; loading is real, but no old model
    # file hash is recomputed for this repair run.
    old_root = read_json(next((BASE / "pairs").glob("*.json")))
    vla_hash = next(p["provenance"]["generator_hash"] for p in old_root["proposals"] if p.get("audit_source") == "vla" and p.get("kind") == "nominal")
    policy = NominalVLAPolicy(keep_on_gpu=True)
    started = time.perf_counter()
    sampler = GPUMemorySampler(interval_s=0.1, gpu_index=0).start()
    results = []
    try:
        policy.ensure_loaded()
        classic = ClassicExpertGenerator()
        vla = NominalVLAGenerator(policy, generator_hash=vla_hash)
        for index, (row, scenario) in enumerate(zip(rows, physical)):
            if time.perf_counter() - started > float(config["carla_wall_limit_s"]):
                raise RuntimeError("repair_carla_wall_limit")
            if (dataset / "pairs" / f"{row.root_id}.json").exists():
                results.append({"root_id": row.root_id, "status": "RESUMED"})
                continue
            target = _target(index, row.split, diagnostic)
            record = _collect_root(client, world, store, registry, row, scenario,
                physical_manifest_sha256=physical_manifest["physical_manifest_sha256"],
                run_lock_sha256=run_lock["repair_run_lock_sha256"], classic=classic, vla=vla, policy=policy,
                repair_protocol=True, intervention_factory=_factory(row, target))
            payload = record.to_dict()
            content = payload.pop("content_sha256", None)
            payload.update({"root_cluster_id": f"{row.root_id}::capture", "repair_target": target,
                            "diagnostic": diagnostic, "repair_batch": None if diagnostic else batch,
                            "base_dataset_id": config["base_dataset_id"]})
            payload["content_sha256"] = stable_sha256(payload)
            store.write_root(payload)
            results.append({"root_id": row.root_id, "status": record.terminal_status,
                            "branches": len(record.branches), "target": target})
            _require_clean_scene(world)
    finally:
        gpu = sampler.stop()
    elapsed = time.perf_counter() - started
    evidence = {"schema_version": "safedrive.cora.repair_collection.v1", "dataset_id": config["dataset_id"],
        "diagnostic": diagnostic, "batch": None if diagnostic else batch, "plan_sha256": plan["plan_sha256"],
        "map": config["map"], "elapsed_s": elapsed, "gpu": gpu, "results": results,
        "resource": _resource_snapshot(store), "base_verification": config["base_verification"]}
    evidence["collection_sha256"] = stable_sha256(evidence)
    _atomic(REPAIR_EVIDENCE / ("diagnostic-collection.json" if diagnostic else f"batch-{batch + 1}-collection.json"), evidence)
    return {"ok": True, "roots": len(results), "branches": sum(int(r.get("branches", 0)) for r in results),
            "elapsed_s": elapsed, "evidence": str(REPAIR_EVIDENCE)}


__all__ = ["collect_plan"]
