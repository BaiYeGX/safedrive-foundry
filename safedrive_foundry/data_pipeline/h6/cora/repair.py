"""Non-destructive base-plus-delta C2 correction protocol.

Legacy content identities are reused, never rehashed by this reader.
"""
from __future__ import annotations

import collections
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
import tomllib

from data_pipeline.h2.contracts import ScenarioKey, stable_sha256
from .contracts import CoraRootRecord
from .matrix import CoraMatrixRow
from .outcomes import _read_parquet, validate_public_outcome_heads
from .repair_labels import SCHEMA, VERSION, derive_v3
from .store import CoraDataStore

SPLITS = ("train", "validation", "calibration", "locked_development")
REPAIR_MATRIX_SCHEMA = "safedrive.cora.repair_matrix.v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path | str) -> dict:
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "safedrive.cora.repair_protocol.v1":
        raise ValueError("repair_protocol_schema")
    return config


def repair_rows(config: Mapping[str, Any], *, batch: int = 0, diagnostic: bool = False) -> tuple[CoraMatrixRow, ...]:
    """Return the frozen diagnostic or one 48-root formal batch plan."""
    if diagnostic:
        count = int(config["diagnostic_roots"])
        start = int(config["diagnostic_seed_start"])
        split = "diagnostic"
        seed_offsets = range(count)
    else:
        count = int(config["batch_roots"])
        starts = {"train": int(config["train_seed_start"]), "validation": int(config["validation_seed_start"]),
                  "calibration": int(config["calibration_seed_start"]), "locked_development": int(config["locked_development_seed_start"])}
        # The order is part of the plan: 24 train, then 8 per held-out split.
        slots = [("train", starts["train"] + batch * 24 + i) for i in range(24)]
        slots += [("validation", starts["validation"] + batch * 8 + i) for i in range(8)]
        slots += [("calibration", starts["calibration"] + batch * 8 + i) for i in range(8)]
        slots += [("locked_development", starts["locked_development"] + batch * 8 + i) for i in range(8)]
        rows = []
        for index, (slot, seed) in enumerate(slots):
            family = ("emergency_lead_brake" if index % 5 == 0 else "aggressive_cut_in" if index % 5 == 1
                      else "red_light_hold" if index % 5 == 2 else "cut_in" if index % 5 == 3 else "slow_lead")
            weather = "ClearNoon" if index % 2 == 0 else "CloudyNoon"
            key = ScenarioKey(str(config["map"]), family, int(seed), weather)
            rows.append(CoraMatrixRow(key, slot, ("expert", "vla") if index % 2 == 0 else ("vla", "expert"), index, index, True))
        return tuple(rows)
    rows = []
    families = ("emergency_lead_brake", "aggressive_cut_in", "red_light_hold", "cut_in", "slow_lead", "free_flow")
    for index in seed_offsets:
        key = ScenarioKey(str(config["map"]), families[index % len(families)], start + index,
                          "ClearNoon" if index % 2 == 0 else "CloudyNoon")
        rows.append(CoraMatrixRow(key, split, ("expert", "vla") if index % 2 == 0 else ("vla", "expert"), index, index, True))
    return tuple(rows)


def plan_payload(config: Mapping[str, Any], rows: Iterable[CoraMatrixRow], *, batch: int | None = None,
                 diagnostic: bool = False) -> dict[str, Any]:
    material = [row.to_dict() for row in rows]
    payload = {"schema_version": REPAIR_MATRIX_SCHEMA, "dataset_id": config["dataset_id"],
               "base_dataset_id": config["base_dataset_id"], "diagnostic": diagnostic,
               "batch": batch, "rows": material, "root_count": len(material),
               "seed_ranges": {k: config[k] for k in ("diagnostic_seed_start", "train_seed_start", "validation_seed_start", "calibration_seed_start", "locked_development_seed_start")}}
    payload["plan_sha256"] = stable_sha256(payload)
    return payload


def write_plan(dataset: Path, payload: Mapping[str, Any], name: str) -> Path:
    store = CoraDataStore(dataset.parent, dataset.name)
    path = dataset / name
    store.write_immutable_json(path, payload)
    return path


def root_counts(records: Iterable[Mapping[str, Any]]) -> dict:
    buckets = {s: {h: {"positive": set(), "negative": set(), "missing": set()}
        for h in ("collision", "red_light", "offroad", "repair_success", "executable")}
        for s in SPLITS}
    ownership = {}
    for record in records:
        split = record["split"]
        if split not in buckets or record.get("diagnostic"):
            continue
        cluster = record.get("root_cluster_id", record["root_id"])
        if cluster in ownership and ownership[cluster] != split:
            raise ValueError("repair_root_cluster_split_leakage")
        ownership[cluster] = split
        for branch in record.get("branches", ()):
            if (not branch.get("outcome_valid") or branch.get("auxiliary_only")
                    or branch.get("guard_verdict") not in {"PASS", "REVIEW"}):
                continue
            for name, key in (("collision", "collision_count"), ("red_light", "red_light_violation"),
                              ("offroad", "off_corridor_duration_s"), ("repair_success", "repair_success"),
                              ("executable", "executable")):
                head = branch.get("heads", {}).get(key, {})
                category = ("positive" if head.get("value") else "negative") if head.get("valid") else "missing"
                buckets[split][name][category].add(cluster)
    return {s: {h: {c: len(ids) for c, ids in categories.items()} for h, categories in heads.items()}
            for s, heads in buckets.items()}


def coverage_gaps(counts: Mapping[str, Any]) -> list[dict]:
    gaps = []
    for split in SPLITS:
        for head in ("collision", "red_light", "offroad", "repair_success", "executable"):
            binary = head in {"repair_success", "executable"}
            required = (12 if binary else 8) if split == "train" else (3 if binary else 2)
            for category in (("positive", "negative") if binary else ("positive",)):
                actual = counts[split][head][category]
                if actual < required:
                    gaps.append(dict(split=split, head=head, category=category, actual=actual,
                                     required=required, deficit=required-actual))
    return gaps


def diagnostic_quality(records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict:
    """Evaluate the pre-formal diagnostic gate by distinct root cluster.

    A diagnostic root qualifies as a repair failure only when an eligible,
    valid branch records an actual repair attempt and a valid
    ``repair_success=false`` result.  An offroad root is counted when an
    eligible, valid branch has positive off-corridor duration.  Branches do
    not inflate the root counts, and diagnostic rows never enter the core
    split coverage buckets.
    """
    repair_failure_roots: set[str] = set()
    offroad_roots: set[str] = set()
    diagnostic_roots: set[str] = set()
    for record in records:
        if not record.get("diagnostic"):
            continue
        cluster = str(record.get("root_cluster_id", record.get("root_id", "")))
        if not cluster:
            continue
        diagnostic_roots.add(cluster)
        for branch in record.get("branches", ()):
            if (not branch.get("outcome_valid") or branch.get("auxiliary_only")
                    or branch.get("guard_verdict") not in {"PASS", "REVIEW"}):
                continue
            heads = branch.get("heads", {})
            attempted = heads.get("repair_attempted", {})
            success = heads.get("repair_success", {})
            if (attempted.get("valid") and attempted.get("value") is True
                    and success.get("valid") and success.get("value") is False):
                repair_failure_roots.add(cluster)
            offroad = heads.get("off_corridor_duration_s", {})
            if (offroad.get("valid") and offroad.get("value") is not None
                    and float(offroad.get("value")) > 0.0):
                offroad_roots.add(cluster)
    required_failures = int(protocol.get("min_diagnostic_repair_failure_roots", 2))
    required_offroad = int(protocol.get("min_diagnostic_offroad_roots", 1))
    return {
        "diagnostic_roots": len(diagnostic_roots),
        "repair_failure_roots": len(repair_failure_roots),
        "offroad_roots": len(offroad_roots),
        "required_repair_failure_roots": required_failures,
        "required_offroad_roots": required_offroad,
        "passed": len(repair_failure_roots) >= required_failures
        and len(offroad_roots) >= required_offroad,
        "repair_failure_root_ids": sorted(repair_failure_roots),
        "offroad_root_ids": sorted(offroad_roots),
    }


def initialize(config_path: Path | str, project: Path) -> Path:
    config = read_config(config_path)
    parent = project / "generated/h6/cora"
    base = parent / config["base_dataset_id"]
    store = CoraDataStore(parent, config["dataset_id"])
    manifest = read_json(base / "manifest.json")
    lock = read_json(base / "run-lock.json")
    store.write_immutable_json(store.root / "repair-protocol.json", {
        **config, "base_path": str(base.resolve()), "base_manifest_recorded_identity": manifest.get("manifest_sha256"),
        "base_run_lock_recorded_identity": lock.get("run_lock_sha256"), "label_schema": SCHEMA,
        "base_verification": "REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN"})
    return store.root


def materialize(dataset: Path) -> dict:
    protocol = read_json(dataset / "repair-protocol.json")
    base = Path(protocol["base_path"])
    store = CoraDataStore(dataset.parent, dataset.name)
    roots = branches = 0
    for source in (base, dataset):
        manifest_path = source / "scenario-manifest.json"
        physical = {r["pair_id"]: r for r in read_json(manifest_path)["rows"]} if manifest_path.exists() else {}
        for path in sorted((source / "pairs").glob("*.json")):
            record = read_json(path)
            anchor = read_json(source / record["anchor_path"])
            scene = physical[record["root_id"]]
            for branch in record["branches"]:
                artifacts = branch["artifact_paths"]
                timeline = _read_parquet(source / artifacts["timeline"]) if artifacts.get("timeline") else []
                events = _read_parquet(source / artifacts["events"]) if artifacts.get("events") else []
                trace = read_json(source / artifacts["safety_trace"]) if artifacts.get("safety_trace") else None
                heads = derive_v3(branch, timeline=timeline, events=events, anchor=anchor, physical=scene, trace=trace)
                payload = {"schema_version": SCHEMA, "root_id": record["root_id"],
                    "proposal_id": branch["proposal_id"], "source_dataset": source.name,
                    "heads": heads, "derivation": VERSION,
                    "repair_trace_available": trace is not None}
                store.write_immutable_json(dataset / "corrected-labels" /
                    f"{record['root_id']}__{branch['proposal_sha256']}.v3.json", payload)
                branches += 1
            roots += 1
    return {"roots": roots, "branches": branches, "label_schema": SCHEMA}


def merged_roots(dataset: Path) -> tuple[dict, ...]:
    protocol = read_json(dataset / "repair-protocol.json")
    merged = {}
    for source in (Path(protocol["base_path"]), dataset):
        for path in sorted((source / "pairs").glob("*.json")):
            record = read_json(path)
            root = record["root_id"]
            if root in merged:
                raise ValueError(f"repair_duplicate_root:{root}")
            record["artifact_base"] = str(source)
            record.setdefault("root_cluster_id", root)
            record["label_schema"] = SCHEMA
            for branch in record["branches"]:
                label = read_json(dataset / "corrected-labels" / f"{root}__{branch['proposal_sha256']}.v3.json")
                if (label.get("schema_version") != SCHEMA or label.get("root_id") != root
                        or label.get("proposal_id") != branch["proposal_id"]):
                    raise ValueError("repair_label_binding")
                validate_public_outcome_heads(label["heads"], derivation_version=VERSION)
                branch["heads"] = label["heads"]
            merged[root] = record
    records = tuple(merged.values())
    root_counts(records)  # fail closed on cluster split conflicts
    return records


def audit(dataset: Path) -> dict:
    records = merged_roots(dataset)
    counts = root_counts(records)
    gaps = coverage_gaps(counts)
    base = Path(read_json(dataset / "repair-protocol.json")["base_path"])
    report = {"schema_version": "safedrive.cora.repair_quality.v1", "dataset_id": dataset.name,
        "state": "MEASURED", "status": "GATE_FAILED", "passed": False,
        "root_counts": counts, "coverage_gaps": gaps,
        "original_roots": sum(r["artifact_base"] == str(base) for r in records),
        "added_roots": sum(r["artifact_base"] != str(base) for r in records),
        "branch_count": sum(len(r["branches"]) for r in records),
        "independent_roots": len({r["root_cluster_id"] for r in records}),
        "base_verification": "REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN",
        "failures": ["coverage"] if gaps else [],
        "pending_checks": ["delta_integrity", "resources", "regression_tests", "finalizer"],
        "correction": "Previous only-sample-shortage conclusion withdrawn: repair attempt observability, root counting, route/light derivation also required correction."}
    return report


def finalize_repair(dataset: Path, evidence_dir: Path) -> dict:
    """Rebuild the repair delivery report from base references plus deltas."""
    materialize(dataset)
    records = merged_roots(dataset)
    counts = root_counts(records)
    gaps = coverage_gaps(counts)
    protocol = read_json(dataset / "repair-protocol.json")
    collections = []
    for path in sorted(evidence_dir.glob("*-collection.json")):
        collections.append(read_json(path))
    elapsed = sum(float(row.get("elapsed_s", 0.0)) for row in collections)
    admission = read_json(evidence_dir / "admission.json") if (evidence_dir / "admission.json").is_file() else {}
    elapsed += float(admission.get("carla_budget_consumed_s", 0.0))
    branches = sum(len(row.get("branches", ())) for row in records)
    added = [row for row in records if row.get("artifact_base") != str(Path(protocol["base_path"]))]
    limits_ok = len(added) <= int(protocol["max_roots"]) and branches - 1295 <= int(protocol["max_branch_attempts"]) and elapsed <= float(protocol["carla_wall_limit_s"])
    test_report = read_json(evidence_dir / "test-report.json") if (evidence_dir / "test-report.json").is_file() else {"passed": False}
    test_text = str(test_report.get("stdout", "")) + "\n" + str(test_report.get("stderr", ""))
    match = re.search(r"Ran (\d+) tests", test_text)
    if match:
        test_report = {**test_report, "tests_run": int(match.group(1))}
    diagnostic = [r for r in records if r.get("diagnostic")]
    diagnostic_gate = diagnostic_quality(records, protocol)
    diagnostics_ok = bool(diagnostic_gate["passed"])
    passed = not gaps and limits_ok and bool(test_report.get("passed")) and diagnostics_ok
    report = {"schema_version": "safedrive.cora.final_delivery.v2", "dataset_id": dataset.name,
        "state": "VERIFIED" if passed else "MEASURED", "status": "GATE_PASSED" if passed else "GATE_FAILED",
        "passed": passed, "original_roots": 351, "added_roots": len(added),
        "independent_roots": len({r.get("root_cluster_id", r["root_id"]) for r in records}),
        "raw_root_count": len(records), "branch_count": branches,
        "root_counts": counts, "coverage_gaps": gaps,
        "diagnostic_roots": len(diagnostic), "diagnostic_gate": diagnostics_ok,
        "diagnostic_quality": diagnostic_gate,
        "resource": {"aggregate_carla_wall_s": elapsed, "limits_ok": limits_ok,
                      "collection_evidence": collections, "admission": admission},
        "tests": test_report, "base_verification": protocol["base_verification"],
        "hash_scope": "new_delta_artifacts_only; old_data_old_models_reused_recorded_identities",
        "failures": (["coverage"] if gaps else []) + ([] if limits_ok else ["resources"]) +
                    ([] if diagnostics_ok else ["diagnostic_gate"]) +
                    ([] if test_report.get("passed") else ["tests_not_passed"]),
        "correction": "Earlier only-sample-shortage conclusion is superseded by trace, root-deduplicated statistics, and route/light label correction."}
    report["final_delivery_sha256"] = stable_sha256(report)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    dataset_report = dataset / "final-delivery.json"
    temporary = dataset_report.with_name(f".{dataset_report.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, dataset_report)
    (evidence_dir / "data-quality.json").write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"ok": True, "status": report["status"], "passed": passed,
            "added_roots": len(added), "coverage_gaps": len(gaps),
            "evidence": str(evidence_dir / "data-quality.json")}
