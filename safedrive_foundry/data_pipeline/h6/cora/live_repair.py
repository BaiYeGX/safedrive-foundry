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
    begin_budget_event, close_budget_event, diagnostic_quality, materialize, merged_roots,
    read_json, read_config, repair_rows, plan_payload, write_plan,
)
from .scenarios import materialize_cora_physical_scenario
from .store import CoraDataStore


REPAIR_EVIDENCE = ROOT / "docs" / "runtime-evidence" / "h6" / "h6-cora-c2-repair-20260906-v3"
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


def _factory(row: CoraMatrixRow):
    """Create exactly the intervention entries frozen in the plan row."""
    def factory(root_id: str, family: str, anchor: Any, candidates: Sequence[Any]):
        by_source = {item.provenance.source.value: item for item in candidates}
        results = []
        for source, operator, multiplier in row.repair_recipe:
            base = by_source.get(source)
            if base is None:
                results.append(InterventionResult(root_id, source, operator, "BASE_MISSING", error="base_missing"))
                continue
            try:
                results.append(derive_scaled_intervention(root_id, anchor, base, operator, multiplier))
            except InterventionNotApplicable as exc:
                results.append(InterventionResult(root_id, source, operator, "NOT_APPLICABLE", error=str(exc)))
        return tuple(results)
    return factory


def _physical_fingerprint(payload: Any) -> str:
    """Fingerprint physical initial conditions without IDs or seed labels."""
    if hasattr(payload, "route"):
        values = {
            "route": payload.route,
            "ego_transform": payload.ego_transform,
            "npc_actors": payload.npc_actors,
            "weather": payload.weather,
            "script": payload.script,
            "red_light": payload.red_light,
        }
    else:
        values = {
            "route": payload.get("route"),
            "ego_transform": payload.get("ego_transform"),
            "npc_actors": payload.get("npc_actors"),
            "weather": payload.get("weather"),
            "script": payload.get("script"),
            "red_light": payload.get("red_light"),
        }
    return stable_sha256(values)


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
    phase = "diagnostic" if diagnostic else f"batch-{batch + 1}"
    if not diagnostic:
        materialize(dataset)
        diagnostic_records = [record for record in merged_roots(dataset) if record.get("diagnostic")]
        gate = diagnostic_quality(diagnostic_records, config)
        if not gate["passed"]:
            raise RuntimeError(f"repair_diagnostic_gate_not_passed:{gate}")
    budget_event = begin_budget_event(dataset, phase=phase, limit_s=float(config["carla_wall_limit_s"]))
    started = time.perf_counter()
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
    physical_name = f"scenario-manifest-{phase}.json"
    existing_manifest_paths = [BASE / "scenario-manifest.json"]
    existing_manifest_paths.extend(
        path for path in sorted(dataset.glob("scenario-manifest-*.json")) if path.name != physical_name
    )
    for manifest_path in existing_manifest_paths:
        if manifest_path.is_file():
            for existing in read_json(manifest_path).get("rows", ()):
                fingerprints.add(_physical_fingerprint(existing))
    for row in rows:
        item = materialize_cora_physical_scenario(world, row)
        fingerprint = _physical_fingerprint(item)
        if fingerprint in fingerprints:
            raise RuntimeError(f"repair_duplicate_physical_initial_state:{row.root_id}")
        fingerprints.add(fingerprint)
        physical.append(item)
    physical_manifest = {"schema_version": "safedrive.cora.repair_physical_manifest.v1",
        "dataset_id": config["dataset_id"], "map": config["map"], "plan_sha256": plan["plan_sha256"],
        "phase": phase, "rows": [item.to_dict() for item in physical], "formal_collected": False}
    physical_manifest["physical_manifest_sha256"] = stable_sha256(physical_manifest)
    if (dataset / physical_name).exists():
        old = read_json(dataset / physical_name)
        if old.get("physical_manifest_sha256") != physical_manifest["physical_manifest_sha256"]:
            raise RuntimeError("repair_physical_manifest_conflict")
    else:
        store.write_immutable_json(dataset / physical_name, physical_manifest)
    run_lock = _new_run_lock(config, plan, physical)
    run_lock_name = f"run-lock-{phase}.json"
    if not (dataset / run_lock_name).exists():
        store.write_immutable_json(dataset / run_lock_name, run_lock)
    else:
        run_lock = read_json(dataset / run_lock_name)
    REPAIR_EVIDENCE.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(REPAIR_EVIDENCE / "run-registry.sqlite3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    # Use the recorded generator identity; loading is real, but no old model
    # file hash is recomputed for this repair run.
    old_root = read_json(next((BASE / "pairs").glob("*.json")))
    vla_hash = next(p["provenance"]["generator_hash"] for p in old_root["proposals"] if p.get("audit_source") == "vla" and p.get("kind") == "nominal")
    policy = NominalVLAPolicy(keep_on_gpu=True)
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
            record = _collect_root(client, world, store, registry, row, scenario,
                physical_manifest_sha256=physical_manifest["physical_manifest_sha256"],
                run_lock_sha256=run_lock["repair_run_lock_sha256"], classic=classic, vla=vla, policy=policy,
                repair_protocol=True, intervention_factory=_factory(row))
            payload = record.to_dict()
            content = payload.pop("content_sha256", None)
            payload.update({"root_cluster_id": f"{row.root_id}::capture", "repair_target": row.repair_target,
                            "repair_recipe": [dict(source=s, operator=o, multiplier=m) for s, o, m in row.repair_recipe],
                            "diagnostic": diagnostic, "repair_batch": None if diagnostic else batch,
                            "base_dataset_id": config["base_dataset_id"]})
            payload["content_sha256"] = stable_sha256(payload)
            store.write_root(payload)
            results.append({"root_id": row.root_id, "status": record.terminal_status,
                            "branches": len(record.branches), "target": row.repair_target,
                            "recipe": [dict(source=s, operator=o, multiplier=m) for s, o, m in row.repair_recipe]})
            _require_clean_scene(world)
    finally:
        gpu = sampler.stop()
    elapsed = time.perf_counter() - started
    close_budget_event(dataset, budget_event, elapsed_s=elapsed, status="COMPLETED")
    evidence = {"schema_version": "safedrive.cora.repair_collection.v1", "dataset_id": config["dataset_id"],
        "diagnostic": diagnostic, "batch": None if diagnostic else batch, "phase": phase,
        "plan_sha256": plan["plan_sha256"],
        "map": config["map"], "elapsed_s": elapsed, "gpu": gpu, "results": results,
        "resource": _resource_snapshot(store), "base_verification": config["base_verification"]}
    evidence["collection_sha256"] = stable_sha256(evidence)
    _atomic(REPAIR_EVIDENCE / ("diagnostic-collection.json" if diagnostic else f"batch-{batch + 1}-collection.json"), evidence)
    return {"ok": True, "roots": len(results), "branches": sum(int(r.get("branches", 0)) for r in results),
            "elapsed_s": elapsed, "evidence": str(REPAIR_EVIDENCE)}


__all__ = ["collect_plan"]
