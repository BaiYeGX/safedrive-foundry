#!/usr/bin/env python3
"""Freeze one K2 V3 checkpoint before blind, then finalize the same bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r2_world_ready_v3 import (  # noqa: E402
    AUDIT_MAPS,
    CALIBRATION_MAPS,
    build_world_campaign_manifest_v3,
    validate_calibration_manifest_v3,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.checkpoint_contract import (  # noqa: E402
    STATUS_OK,
    write_checkpoint_manifest,
)
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteContextV3,
    canonical_sha256,
)

PENDING_SCHEMA = "safedrive.k2_v3_formal_candidate.v1"
PLAN_FILES = (
    "long_smoke_teacher.json",
    "long_smoke_learned.json",
    "unseen_long_audit_16.json",
    "calibration_360.json",
    "calibration_pilot_144.json",
    "core_blind_12.json",
    "world_ready_audit_84.json",
    "thresholds.json",
    "runtime_contract.json",
)


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _read_frozen(path: str | Path) -> dict[str, Any]:
    value = _read(path)
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if not stored or stored != canonical_sha256(body):
        raise ValueError(f"{path}: manifest hash mismatch")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pass(report: Mapping[str, Any], name: str) -> None:
    passed = report.get("passed")
    if passed is None and name == "overfit_32":
        passed = report.get("overfit_passed")
    if passed is not True:
        raise ValueError(f"{name} report did not pass")


def _validate_plan(
    plan_dir: Path,
    *,
    calibration: Mapping[str, Any],
    core: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    values = {
        name: _read_frozen(plan_dir / name) for name in PLAN_FILES
    }
    expected = {
        "calibration_360.json": calibration,
        "core_blind_12.json": core,
        "world_ready_audit_84.json": audit,
    }
    for name, supplied in expected.items():
        if values[name]["manifest_hash"] != supplied.get("manifest_hash"):
            raise ValueError(f"{name}: supplied manifest differs from plan")
    runtime = values["runtime_contract.json"]
    for relative, frozen_hash in dict(
        runtime.get("implementation_hashes") or {}
    ).items():
        path = ROOT / str(relative)
        if not path.is_file() or _sha(path) != str(frozen_hash):
            raise ValueError(
                f"runtime implementation changed after freeze: {relative}"
            )
    return values


def _validate_campaign_reports(
    reports: list[Mapping[str, Any]],
    *,
    calibration: Mapping[str, Any],
) -> list[str]:
    expected_ids = {
        str(slot["slot_id"]) for slot in calibration["slots"]
    }
    rows: list[Mapping[str, Any]] = []
    hashes: list[str] = []
    for report in reports:
        if str(report.get("manifest_hash") or "") != str(
            calibration["manifest_hash"]
        ):
            raise ValueError("campaign report manifest binding mismatch")
        if str(report.get("mode") or "") != "teacher":
            raise ValueError("formal dataset must come from teacher collection")
        rows.extend(list(report.get("rows") or ()))
        hashes.append(canonical_sha256(report))
    actual_ids = [str(row.get("slot_id") or "") for row in rows]
    if len(actual_ids) != 360 or set(actual_ids) != expected_ids:
        raise ValueError("campaign reports do not exactly cover 360 slots")
    if len(set(actual_ids)) != 360:
        raise ValueError("campaign reports contain duplicate slots")
    if any(
        str(row.get("status") or "")
        not in {"COMPLETED", "COMPLETED_EXISTING"}
        for row in rows
    ):
        raise ValueError("campaign reports contain incomplete slots")
    return hashes


def _manifest_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(manifest.get("slots") or manifest.get("cases") or ())


def _validate_route_banks(
    banks: list[Mapping[str, Any]],
    *,
    manifests: list[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Mapping[str, Any]]]:
    expected: dict[str, Mapping[str, Any]] = {}
    for manifest in manifests:
        for row in _manifest_rows(manifest):
            route_id = str(row["route_fixture_id"])
            previous = expected.get(route_id)
            if previous is not None:
                comparable_keys = (
                    "map_name",
                    "template_id",
                    "family",
                    "maneuver",
                )
                if any(
                    str(previous.get(key) or "")
                    != str(row.get(key) or "")
                    for key in comparable_keys
                ):
                    raise ValueError(
                        f"conflicting frozen route fixture: {route_id}"
                    )
                continue
            expected[route_id] = row
    by_map: dict[str, Mapping[str, Any]] = {}
    routes_by_id: dict[str, Mapping[str, Any]] = {}
    route_hashes: dict[str, tuple[str, str]] = {}
    entry_signatures: dict[tuple[str, str], str] = {}
    for bank in banks:
        if str(bank.get("schema_version") or "") != (
            "safedrive.r2_v3.route_bank.v2"
        ):
            raise ValueError("formal promotion requires route-bank v2")
        map_name = str(bank.get("map_name") or "")
        if map_name in by_map:
            raise ValueError(f"duplicate route bank for {map_name}")
        by_map[map_name] = bank
        constraints = dict(bank.get("authoring_constraints") or {})
        if not (
            constraints.get("map_topology_only")
            and (
                constraints.get("unique_route_hash")
                or constraints.get("cross_phase_route_hash_unique")
            )
            and (
                constraints.get("unique_entry_signature")
                or constraints.get("cross_phase_entry_signature_unique")
            )
            and constraints.get("model_output_used") is False
            and constraints.get("candidate_outcome_used") is False
            and constraints.get("oracle_used") is False
        ):
            raise ValueError(f"{map_name}: route bank authoring contract invalid")
        routes = dict(bank.get("routes_by_fixture_id") or {})
        requirements = dict(bank.get("requirements_by_fixture_id") or {})
        expected_ids = {
            route_id
            for route_id, row in expected.items()
            if str(row["map_name"]) == map_name
        }
        if set(routes) != expected_ids or set(requirements) != expected_ids:
            raise ValueError(f"{map_name}: route bank fixture coverage mismatch")
        for route_id, raw_route in routes.items():
            context = RouteContextV3.from_mapping(
                raw_route["route_context"]
            )
            frozen = expected[route_id]
            if context.maneuver.value != str(frozen["maneuver"]):
                raise ValueError(f"{route_id}: route maneuver mismatch")
            requirement = dict(requirements[route_id])
            phase = str(requirement.get("phase") or "")
            if (
                str(requirement.get("map_name") or "") != map_name
                or str(requirement.get("maneuver") or "")
                != str(frozen["maneuver"])
            ):
                raise ValueError(f"{route_id}: frozen requirement mismatch")
            prior_route = route_hashes.get(context.route_hash)
            if prior_route is not None and (
                phase == "calibration"
                or prior_route != (map_name, phase)
            ):
                raise ValueError(
                    f"actual route hash reused across phases: {route_id}"
                )
            entry_key = (map_name, context.entry_signature)
            prior_entry_phase = entry_signatures.get(entry_key)
            if prior_entry_phase is not None and (
                phase == "calibration" or prior_entry_phase != phase
            ):
                raise ValueError(
                    f"entry signature reused across phases: {route_id}"
                )
            route_hashes[context.route_hash] = (map_name, phase)
            entry_signatures[entry_key] = phase
            routes_by_id[route_id] = raw_route
    if set(by_map) != set(AUDIT_MAPS):
        raise ValueError("route banks must cover all seven audit maps")
    if set(routes_by_id) != set(expected):
        raise ValueError("route banks do not cover every frozen route fixture")
    return (
        {
            map_name: str(bank["manifest_hash"])
            for map_name, bank in by_map.items()
        },
        routes_by_id,
    )


def _registry_hash(path: str | Path) -> str:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise ValueError(f"authored registry missing: {registry_path}")
    return load_scenario_registry(
        registry_path
    ).compute_registry_sha256()


def _validate_campaign_authoring(
    indexes: list[Mapping[str, Any]],
    *,
    calibration: Mapping[str, Any],
    route_bank_hashes: Mapping[str, str],
) -> list[str]:
    by_map = {str(index.get("map_name") or ""): index for index in indexes}
    if set(by_map) != set(CALIBRATION_MAPS) or len(indexes) != 5:
        raise ValueError("campaign authoring must cover five calibration maps")
    for map_name, index in by_map.items():
        if (
            str(index.get("campaign_manifest_hash") or "")
            != str(calibration["manifest_hash"])
            or str(index.get("route_fixture_hash") or "")
            != str(route_bank_hashes[map_name])
            or index.get("outcome_used") is not False
            or index.get("oracle_used") is not False
        ):
            raise ValueError(f"{map_name}: campaign authoring binding mismatch")
        expected = {
            str(slot["lineage_id"])
            for slot in calibration["slots"]
            if str(slot["map_name"]) == map_name
        }
        rows = list(index.get("lineages") or ())
        if {str(row.get("lineage_id") or "") for row in rows} != expected:
            raise ValueError(f"{map_name}: campaign lineage coverage mismatch")
        for row in rows:
            if _registry_hash(row["registry"]) != str(row["registry_hash"]):
                raise ValueError("campaign registry hash changed after authoring")
    return [str(by_map[name]["manifest_hash"]) for name in sorted(by_map)]


def _validate_formal_authoring(
    indexes: list[Mapping[str, Any]],
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    route_bank_hashes: Mapping[str, str],
    routes_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index in indexes:
        key = (
            str(index.get("phase") or ""),
            str(index.get("map_name") or ""),
        )
        if key in by_key:
            raise ValueError(f"duplicate formal authoring index: {key}")
        by_key[key] = index
    expected_keys = {
        (phase, map_name)
        for phase in manifests
        for map_name in AUDIT_MAPS
    }
    if set(by_key) != expected_keys:
        raise ValueError("formal authoring must cover three phases x seven maps")
    for (phase, map_name), index in by_key.items():
        manifest = manifests[phase]
        if (
            str(index.get("source_manifest_hash") or "")
            != str(manifest["manifest_hash"])
            or str(index.get("route_bank_hash") or "")
            != str(route_bank_hashes[map_name])
            or index.get("outcome_used") is not False
            or index.get("oracle_used") is not False
        ):
            raise ValueError(f"{phase}/{map_name}: authoring binding mismatch")
        expected = {
            str(row["fixture_id"]): row
            for row in manifest["cases"]
            if str(row["map_name"]) == map_name
        }
        rows = {
            str(row.get("fixture_id") or ""): row
            for row in index.get("fixtures") or ()
        }
        if set(rows) != set(expected):
            raise ValueError(
                f"{phase}/{map_name}: formal fixture coverage mismatch"
            )
        for fixture_id, row in rows.items():
            frozen = expected[fixture_id]
            route_id = str(frozen["route_fixture_id"])
            context = RouteContextV3.from_mapping(
                routes_by_id[route_id]["route_context"]
            )
            if (
                str(row.get("route_fixture_id") or "") != route_id
                or str(row.get("route_hash") or "") != context.route_hash
                or _registry_hash(row["registry"])
                != str(row["registry_hash"])
            ):
                raise ValueError(
                    f"{phase}/{fixture_id}: authored registry binding mismatch"
                )
    return [
        str(by_key[key]["manifest_hash"]) for key in sorted(by_key)
    ]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint missing: {checkpoint}")
    overfit = _read(args.overfit_report)
    training = _read(args.training_report)
    offline = _read(args.offline_report)
    dataset_audit = _read(args.dataset_audit)
    teacher_smoke = _read(args.teacher_smoke_report)
    learned_smoke = _read(args.learned_smoke_report)
    calibration = _read_frozen(args.calibration_manifest)
    core_manifest = _read_frozen(args.core_manifest)
    audit_manifest = _read_frozen(args.audit_manifest)
    _require_pass(overfit, "overfit_32")
    _require_pass(offline, "offline")
    _require_pass(teacher_smoke, "teacher_long_smoke")
    _require_pass(learned_smoke, "learned_long_smoke")
    validate_calibration_manifest_v3(calibration)
    if not bool(overfit.get("overfit_32")):
        raise ValueError("overfit report is not the required 32-sample check")
    plan = _validate_plan(
        Path(args.plan_dir),
        calibration=calibration,
        core=core_manifest,
        audit=audit_manifest,
    )
    unseen_manifest = plan["unseen_long_audit_16.json"]
    if int(teacher_smoke.get("completed", 0)) != 16:
        raise ValueError("teacher long smoke must complete 16/16")
    if int(learned_smoke.get("completed", 0)) != 16:
        raise ValueError("learned long smoke must complete 16/16")
    checkpoint_hash_before_copy = _sha(checkpoint)
    if bool(training.get("overfit_32")):
        raise ValueError("formal checkpoint cannot be the 32-sample overfit run")
    if (
        str(training.get("checkpoint_sha256") or "")
        != checkpoint_hash_before_copy
        or str(offline.get("checkpoint_sha256") or "")
        != checkpoint_hash_before_copy
        or str(learned_smoke.get("checkpoint_sha256") or "")
        != checkpoint_hash_before_copy
    ):
        raise ValueError(
            "training/offline/learned-smoke checkpoint binding mismatch"
        )
    if teacher_smoke.get("checkpoint_sha256") not in (None, ""):
        raise ValueError("teacher smoke must not use a learned checkpoint")
    if (
        str(teacher_smoke.get("mode") or "") != "teacher"
        or str(learned_smoke.get("mode") or "") != "learned"
    ):
        raise ValueError("long-smoke mode binding mismatch")
    thresholds = plan["thresholds.json"]
    expected_registry_hash = str(
        thresholds.get("long_smoke_registry_hash") or ""
    )
    if (
        str(teacher_smoke.get("registry_hash") or "")
        != expected_registry_hash
        or str(learned_smoke.get("registry_hash") or "")
        != expected_registry_hash
    ):
        raise ValueError("long-smoke registry differs from frozen plan")

    campaign_reports = [_read(path) for path in args.campaign_report]
    campaign_report_hashes = _validate_campaign_reports(
        campaign_reports,
        calibration=calibration,
    )
    expected_dataset = {
        "samples": 360,
        "lineages": 60,
        "routes": 60,
        "actual_route_hashes": 60,
        "split_counts": {"train": 252, "dev": 60, "test": 48},
        "lineage_overlap": 0,
        "route_overlap": 0,
        "actual_route_hash_overlap": 0,
        "exact_manifest_coverage": True,
        "manifest_hash": str(calibration["manifest_hash"]),
    }
    for key, expected in expected_dataset.items():
        if dataset_audit.get(key) != expected:
            raise ValueError(
                f"dataset audit {key} mismatch: "
                f"{dataset_audit.get(key)!r} != {expected!r}"
            )
    dataset_hash = str(dataset_audit.get("dataset_sha256") or "")
    if (
        len(dataset_hash) != 64
        or str(training.get("dataset_sha256") or "") != dataset_hash
        or str(offline.get("dataset_sha256") or "") != dataset_hash
        or int(training.get("dataset_samples") or 0) != 360
        or dict(training.get("dataset_split_counts") or {})
        != {"train": 252, "dev": 60, "test": 48}
    ):
        raise ValueError("formal checkpoint is not bound to the full 360 dataset")
    if (
        str(offline.get("split") or "") != "test"
        or int(offline.get("record_count") or 0) != 48
    ):
        raise ValueError("formal offline gate must use all 48 frozen test rows")

    banks = [_read_frozen(path) for path in args.route_bank]
    route_bank_hashes, routes_by_id = _validate_route_banks(
        banks,
        manifests=[
            calibration,
            core_manifest,
            audit_manifest,
            unseen_manifest,
        ],
    )
    campaign_indexes = [
        _read_frozen(path) for path in args.campaign_authoring_index
    ]
    campaign_authoring_hashes = _validate_campaign_authoring(
        campaign_indexes,
        calibration=calibration,
        route_bank_hashes=route_bank_hashes,
    )
    formal_manifests = {
        "core_blind": core_manifest,
        "world_ready_audit": audit_manifest,
        "unseen_long_audit": unseen_manifest,
    }
    formal_indexes = [
        _read_frozen(path) for path in args.formal_authoring_index
    ]
    formal_authoring_hashes = _validate_formal_authoring(
        formal_indexes,
        manifests=formal_manifests,
        route_bank_hashes=route_bank_hashes,
        routes_by_id=routes_by_id,
    )

    calibration_fixtures = {
        str(slot["route_fixture_id"]) for slot in calibration["slots"]
    }
    core_fixtures = {
        str(case["route_fixture_id"]) for case in core_manifest["cases"]
    }
    audit_fixtures = {
        str(case["route_fixture_id"]) for case in audit_manifest["cases"]
    }
    unseen_fixtures = {
        str(case["route_fixture_id"]) for case in unseen_manifest["cases"]
    }
    if calibration_fixtures.intersection(core_fixtures | audit_fixtures):
        raise ValueError("calibration overlaps blind/audit fixtures")
    if (
        core_fixtures.intersection(audit_fixtures | unseen_fixtures)
        or audit_fixtures.intersection(unseen_fixtures)
        or calibration_fixtures.intersection(unseen_fixtures)
    ):
        raise ValueError("formal route fixtures overlap across frozen phases")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_checkpoint = output_dir / "k2_v3_formal.pt"
    shutil.copy2(checkpoint, frozen_checkpoint)
    checkpoint_hash = _sha(frozen_checkpoint)
    body = {
        "schema_version": PENDING_SCHEMA,
        "status": "FROZEN_PENDING_BLIND",
        "checkpoint": str(frozen_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "calibration_manifest_hash": str(calibration.get("manifest_hash") or ""),
        "core_blind_manifest_hash": str(core_manifest.get("manifest_hash") or ""),
        "world_ready_audit_manifest_hash": str(
            audit_manifest.get("manifest_hash") or ""
        ),
        "unseen_long_audit_manifest_hash": str(
            unseen_manifest.get("manifest_hash") or ""
        ),
        "frozen_plan_hashes": {
            name: str(value["manifest_hash"])
            for name, value in plan.items()
        },
        "route_bank_hashes": route_bank_hashes,
        "campaign_authoring_index_hashes": campaign_authoring_hashes,
        "formal_authoring_index_hashes": formal_authoring_hashes,
        "campaign_report_hashes": campaign_report_hashes,
        "dataset_audit_hash": canonical_sha256(dataset_audit),
        "dataset_sha256": dataset_hash,
        "training_report_hash": canonical_sha256(training),
        "offline_report_hash": canonical_sha256(offline),
        "teacher_smoke_report_hash": canonical_sha256(teacher_smoke),
        "learned_smoke_report_hash": canonical_sha256(learned_smoke),
        "calibration_slots_completed": 360,
        "cross_phase_route_hash_overlap_zero": True,
        "formal_fixtures_frozen_before_blind": True,
        "blind_pair_overlap_zero": True,
        "checkpoint_frozen_before_blind": True,
        "model_selection_from_blind_forbidden": True,
    }
    pending = {**body, "manifest_hash": canonical_sha256(body)}
    pending_path = output_dir / "FORMAL_CANDIDATE.json"
    pending_path.write_text(
        json.dumps(pending, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_manifest(
        output_dir / "CHECKPOINT_STATUS.json",
        checkpoint_path=frozen_checkpoint,
        status="R2V3_FROZEN_PENDING_BLIND",
        allowed_uses=[
            "offline_diagnostic",
            "development_live_smoke",
            "r2v3_blind_audit",
        ],
        forbidden_uses=[
            "formal_offline",
            "r2v3_formal",
            "world_campaign",
        ],
        reasons=["checkpoint_frozen_before_blind", "blind_not_yet_accepted"],
        extra={
            "formal_candidate_manifest": str(pending_path),
            "formal_candidate_manifest_hash": pending["manifest_hash"],
            "core_blind_manifest_hash": pending["core_blind_manifest_hash"],
            "world_ready_audit_manifest_hash": pending[
                "world_ready_audit_manifest_hash"
            ],
            "unseen_long_audit_manifest_hash": pending[
                "unseen_long_audit_manifest_hash"
            ],
            "dataset_sha256": dataset_hash,
            "calibration_slots_completed": 360,
            "formal_fixtures_frozen_before_blind": True,
            "checkpoint_frozen_before_blind": True,
        },
    )
    print(json.dumps(pending, indent=2, sort_keys=True))
    return pending


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    pending_path = Path(args.formal_candidate)
    pending = _read(pending_path)
    if pending.get("status") != "FROZEN_PENDING_BLIND":
        raise ValueError("formal candidate is not pending blind")
    checkpoint = Path(str(pending["checkpoint"]))
    if _sha(checkpoint) != str(pending["checkpoint_sha256"]):
        raise ValueError("frozen checkpoint bytes changed after blind")
    core = _read(args.core_report)
    audit = _read(args.audit_report)
    unseen_long = _read(args.unseen_long_audit_report)
    _require_pass(core, "core_blind")
    _require_pass(audit, "world_ready_audit")
    _require_pass(unseen_long, "unseen_long_audit")
    if int(unseen_long.get("completed", 0)) != 16:
        raise ValueError("unseen learned long audit must complete 16/16")
    for report, name in (
        (core, "core"),
        (audit, "audit"),
        (unseen_long, "unseen_long"),
    ):
        bound = str(report.get("checkpoint_sha256") or "")
        if bound != str(pending["checkpoint_sha256"]):
            raise ValueError(f"{name} report checkpoint binding mismatch")
    source_bindings = (
        (
            core,
            "core",
            str(pending["core_blind_manifest_hash"]),
            12,
        ),
        (
            audit,
            "audit",
            str(pending["world_ready_audit_manifest_hash"]),
            84,
        ),
        (
            unseen_long,
            "unseen_long",
            str(pending["unseen_long_audit_manifest_hash"]),
            16,
        ),
    )
    for report, name, expected_manifest_hash, expected_count in source_bindings:
        if str(report.get("source_manifest_hash") or "") != expected_manifest_hash:
            raise ValueError(f"{name} report manifest binding mismatch")
        rows = report.get("records")
        if rows is None:
            rows = report.get("rows")
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(f"{name} report evidence coverage mismatch")
    output_dir = pending_path.parent
    campaign = build_world_campaign_manifest_v3(
        formal_checkpoint_hash=str(pending["checkpoint_sha256"])
    )
    campaign_path = output_dir / "WORLD_CAMPAIGN_1008.json"
    if campaign_path.exists():
        raise FileExistsError(f"refusing to overwrite {campaign_path}")
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final = {
        **pending,
        "status": "FORMAL_USABLE",
        "core_report_hash": canonical_sha256(core),
        "world_ready_audit_report_hash": canonical_sha256(audit),
        "unseen_long_audit_report_hash": canonical_sha256(unseen_long),
        "world_campaign_manifest": str(campaign_path),
        "world_campaign_manifest_hash": campaign["manifest_hash"],
        "r2_status": "BASIC_ROUTE_AND_1V1_INTERACTION_USABLE",
        "world_gate": "READY",
        "world_training": "NOT_STARTED",
    }
    final["manifest_hash"] = canonical_sha256(
        {key: value for key, value in final.items() if key != "manifest_hash"}
    )
    final_path = output_dir / "FORMAL_ACCEPTANCE.json"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path}")
    final_path.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_manifest(
        output_dir / "CHECKPOINT_STATUS.json",
        checkpoint_path=checkpoint,
        status=STATUS_OK,
        allowed_uses=[
            "offline_diagnostic",
            "development_live_smoke",
            "collection_anchor",
            "formal_offline",
            "r2v3_formal",
            "r2v3_blind_audit",
            "world_campaign",
        ],
        forbidden_uses=[],
        reasons=[
            "same_checkpoint_bytes_passed_pre_registered_blind",
            "world_ready_audit_passed",
            "unseen_long_audit_16_of_16",
        ],
        extra={
            "formal_acceptance_manifest": str(final_path),
            "formal_acceptance_manifest_hash": final["manifest_hash"],
            "world_campaign_manifest": str(campaign_path),
            "world_campaign_manifest_hash": campaign["manifest_hash"],
            "checkpoint_frozen_before_blind": True,
            "blind_pair_overlap_zero": True,
            "r2_status": final["r2_status"],
            "world_gate": final["world_gate"],
            "world_training": final["world_training"],
        },
    )
    print(json.dumps(final, indent=2, sort_keys=True))
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--checkpoint", required=True)
    freeze_parser.add_argument("--output-dir", required=True)
    freeze_parser.add_argument("--plan-dir", required=True)
    freeze_parser.add_argument("--overfit-report", required=True)
    freeze_parser.add_argument("--training-report", required=True)
    freeze_parser.add_argument("--offline-report", required=True)
    freeze_parser.add_argument("--dataset-audit", required=True)
    freeze_parser.add_argument(
        "--campaign-report",
        action="append",
        required=True,
    )
    freeze_parser.add_argument(
        "--route-bank",
        action="append",
        required=True,
    )
    freeze_parser.add_argument(
        "--campaign-authoring-index",
        action="append",
        required=True,
    )
    freeze_parser.add_argument(
        "--formal-authoring-index",
        action="append",
        required=True,
    )
    freeze_parser.add_argument("--teacher-smoke-report", required=True)
    freeze_parser.add_argument("--learned-smoke-report", required=True)
    freeze_parser.add_argument("--calibration-manifest", required=True)
    freeze_parser.add_argument("--core-manifest", required=True)
    freeze_parser.add_argument("--audit-manifest", required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--formal-candidate", required=True)
    finalize_parser.add_argument("--core-report", required=True)
    finalize_parser.add_argument("--audit-report", required=True)
    finalize_parser.add_argument("--unseen-long-audit-report", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    else:
        finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
