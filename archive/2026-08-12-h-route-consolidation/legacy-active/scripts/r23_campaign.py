#!/usr/bin/env python3
"""Plan, inspect and execute resumable joint R2/World collection campaigns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r23_campaign import (  # noqa: E402
    append_checkpoint_result,
    campaign_status,
    create_campaign_layout,
    default_lineage_bank,
    default_lineage_bank_v2,
)
from driving_vla.evaluation.r23_collection import (  # noqa: E402
    CollectionSlot,
    R23CollectionError,
    R23CollectionSampleV1,
    content_hash,
    file_sha256,
    validate_completed_joint_artifacts,
    write_json_exclusive,
)


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def cmd_init_bank(args: argparse.Namespace) -> int:
    path = Path(args.output)
    bank = (
        default_lineage_bank_v2()
        if str(args.version) == "v2"
        else default_lineage_bank()
    )
    write_json_exclusive(path, bank)
    _json(
        {
            "status": "LINEAGE_BANK_CREATED",
            "version": str(args.version),
            "path": str(path),
            "n_lineages": len(bank["lineages"]),
        }
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    checkpoint = Path(args.r2_checkpoint)
    config = Path(args.collection_config)
    if not checkpoint.is_file():
        raise R23CollectionError(f"R2 checkpoint missing: {checkpoint}")
    if not config.is_file():
        raise R23CollectionError(f"collection config missing: {config}")
    config_value = json.loads(config.read_text(encoding="utf-8"))
    schema = str(config_value.get("schema_version") or "")
    if schema not in {
        "safedrive.r23_collection_config.v1",
        "safedrive.r23_collection_config.v2",
    }:
        raise R23CollectionError(f"unsupported collection config: {schema}")
    campaign_version = str(config_value.get("campaign_version") or "v1")
    conditions = tuple(
        str(item)
        for item in config_value.get(
            "conditions",
            [f"condition_{index:02d}" for index in range(6)],
        )
    )
    result = create_campaign_layout(
        Path(args.campaign_root),
        lineage_bank=Path(args.lineage_bank),
        r2_checkpoint_sha256=file_sha256(checkpoint),
        collection_config_sha256=file_sha256(config),
        phases=(args.phase,),
        phase_specs=config_value.get("phase_specs"),
        campaign_version=campaign_version,
        conditions=conditions,
    )
    _json({"status": "PLANNED", "phases": result})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _json(campaign_status(Path(args.phase_root)))
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.collection_config).read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(ROOT / "scripts/r23_map_inventory.py"),
        "--output",
        str(Path(args.output)),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    for map_name in config.get("maps", ()):
        command.extend(["--required-map", str(map_name)])
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


def _shard_command(
    *,
    mode: str,
    shard_root: Path,
    checkpoint: Path,
    host: str,
    port: int,
    device: str,
) -> list[str]:
    registry = shard_root / "scenario_registry.toml"
    registry_manifest = shard_root / "dry-run" / "registry_manifest.json"
    if not registry.is_file():
        raise R23CollectionError(
            f"shard is not authored: {shard_root}; scenario_registry.toml required"
        )
    if mode == "dry-run":
        return [
            sys.executable,
            str(ROOT / "tests/g4/run_g4a_paired.py"),
            "--registry",
            str(registry),
            "--host",
            host,
            "--port",
            str(port),
            "registry-dry-run",
            "--evidence-dir",
            str(shard_root / "dry-run"),
        ]
    if not registry_manifest.is_file():
        raise R23CollectionError(
            f"shard is not frozen: {registry_manifest}; run dry-run first"
        )
    return [
        sys.executable,
        str(ROOT / "scripts/r3_collect_action_branches.py"),
        "--registry",
        str(registry),
        "--registry-manifest",
        str(registry_manifest),
        "--spatial-head-ckpt",
        str(checkpoint),
        "--evidence-dir",
        str(shard_root / "evidence"),
        "--device",
        device,
        "--host",
        host,
        "--port",
        str(port),
    ]


def _campaign_slots(phase_root: Path):
    manifest = json.loads(
        (phase_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    slots = [CollectionSlot.from_dict(row) for row in manifest["slots"]]
    return manifest, slots


def _result_for_slot(slot: CollectionSlot, report: dict) -> dict:
    matches = [
        row
        for row in report.get("pair_results", [])
        if str(row.get("scenario_id")) == slot.scenario_id
        and str(row.get("seed_id")) == slot.seed_id
    ]
    if len(matches) != 1:
        raise R23CollectionError(
            f"run-set report missing/duplicate result for {slot.scenario_id}/{slot.seed_id}"
        )
    row = dict(matches[0])
    failure_codes = list(row.get("failure_codes") or ())
    if not failure_codes:
        error = str(row.get("error") or row.get("failure") or "").lower()
        if "timeout" in error or "time-out" in error:
            failure_codes = ["CARLA_RPC_TIMEOUT"]
        elif "connect" in error:
            failure_codes = ["CARLA_CONNECT_FAILURE"]
        elif "server" in error or "rpc" in error:
            failure_codes = ["SERVER_FAILURE"]
        elif str(row.get("status", "")).upper() == "FAILED":
            failure_codes = ["NON_TECHNICAL_PAIR_FAILURE"]
    run_status = str(row.get("run_status") or row.get("status") or "").upper()
    normalized_status = (
        "COMPLETED"
        if run_status == "COMPLETED"
        or str(row.get("status", "")).upper() in {"COMPARABLE", "INCOMPARABLE"}
        else "FAILED"
    )
    return {
        "slot_id": slot.slot_id,
        "status": normalized_status,
        "pair_id": row.get("pair_id"),
        "attempt_id": row.get("attempt_id"),
        "comparable": bool(row.get("comparable", False)),
        "failure_codes": failure_codes,
        "shard_id": slot.shard_id,
        "reserve": slot.reserve,
        "attempt_dir": row.get("attempt_dir"),
    }


def _write_collection_sample(
    *,
    slot: CollectionSlot,
    result: dict,
    shard_root: Path,
    campaign_manifest: dict,
    r2_checkpoint: Path,
) -> None:
    attempt_raw = str(result.get("attempt_dir") or "")
    attempt_dir = Path(attempt_raw)
    if attempt_raw and not attempt_dir.is_absolute():
        attempt_dir = ROOT / attempt_dir
    if not attempt_dir.is_dir():
        local_pairs = shard_root / "evidence" / "pairs"
        pair_id = str(result.get("pair_id") or "")
        attempt_id = int(result.get("attempt_id") or 0)
        attempt_dir = local_pairs / pair_id / f"attempt_{attempt_id}"
    registry_manifest = json.loads(
        (shard_root / "dry-run" / "registry_manifest.json").read_text(encoding="utf-8")
    )
    pair_manifest = {}
    pair_manifest_path = attempt_dir / "pair_manifest.json"
    if pair_manifest_path.is_file():
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    completed = result["status"] == "COMPLETED"
    artifact_audit = None
    if completed:
        artifact_audit = validate_completed_joint_artifacts(
            attempt_dir=attempt_dir,
            expected_scenario_id=slot.scenario_id,
            expected_seed_id=slot.seed_id,
        )
        if not str(pair_manifest.get("executor_config_hash") or ""):
            raise R23CollectionError("completed slot missing executor_config_hash")
    branch_paths = tuple(
        str((attempt_dir / f"branch-{index}").as_posix())
        if (attempt_dir / f"branch-{index}").is_dir()
        and not (attempt_dir / f"branch-{index}" / "unavailable.json").is_file()
        else None
        for index in range(2)
    )
    unavailable_reasons: list[str | None] = []
    for index in range(2):
        unavailable_path = attempt_dir / f"branch-{index}" / "unavailable.json"
        if unavailable_path.is_file():
            unavailable = json.loads(unavailable_path.read_text(encoding="utf-8"))
            unavailable_reasons.append(
                str(unavailable.get("unavailable_reason") or "NO_ALTERNATIVE")
            )
        else:
            unavailable_reasons.append(None)
    technical_codes = tuple(
        code
        for code in result.get("failure_codes", ())
        if code
        in {
            "CARLA_CONNECT_FAILURE",
            "CARLA_RPC_TIMEOUT",
            "SENSOR_SYNC_FAILURE",
            "SPAWN_FAILURE",
            "SERVER_FAILURE",
            "CLEANUP_FAILURE",
        }
    )
    sample_status = (
        "SINGLETON" if completed and branch_paths[1] is None else
        "COMPLETED" if completed else
        "FAILED"
    )
    sample = R23CollectionSampleV1(
        slot=slot,
        status=sample_status,
        observable_path=(
            str((attempt_dir / "branch-0" / "observable_scene_t0.json").as_posix())
            if completed
            else None
        ),
        anchor_artifact_path=(
            str((attempt_dir / "anchor" / "anchor_bundle_v2.json").as_posix())
            if completed
            else None
        ),
        feature_path=(
            str((attempt_dir / "anchor" / "feature.json").as_posix())
            if completed
            else None
        ),
        branch_paths=branch_paths,  # type: ignore[arg-type]
        technical_failure_codes=technical_codes,
        unavailable_reasons=tuple(unavailable_reasons),  # type: ignore[arg-type]
        provenance={
            "registry_sha256": str(registry_manifest["registry_sha256"]),
            "r2_checkpoint_sha256": file_sha256(r2_checkpoint),
            # The frozen G4 runner uses a 128-bit content identity for backward
            # compatible pair IDs.  The joint schema requires a full SHA-256,
            # so bind that legacy identity into a 256-bit provenance digest
            # without changing any historical runner hash.
            "executor_sha256": content_hash(
                {
                    "executor_config_hash": str(
                        pair_manifest.get("executor_config_hash") or ""
                    )
                }
            ),
            "collection_config_sha256": str(
                campaign_manifest["collection_config_sha256"]
            ),
        },
        audit={
            "pair_id": result.get("pair_id"),
            "attempt_id": result.get("attempt_id"),
            "comparable": result.get("comparable"),
            "failure_codes": list(result.get("failure_codes", ())),
            "attempt_dir": str(attempt_dir.as_posix()),
            "joint_artifact_audit": artifact_audit,
        },
    )
    output = shard_root / "collection_samples" / f"{slot.slot_id}.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != sample.to_dict():
            raise R23CollectionError(f"collection sample mismatch: {output}")
    else:
        write_json_exclusive(output, sample.to_dict())


def cmd_campaign_run(args: argparse.Namespace) -> int:
    phase_root = Path(args.phase_root)
    manifest, slots = _campaign_slots(phase_root)
    checkpoint_path = phase_root / "campaign_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed_prefix = len(checkpoint.get("results", []))
    if completed_prefix and not bool(args.resume) and args.mode == "collect":
        raise R23CollectionError(
            "campaign has prior results; pass --resume to continue immutable slots"
        )
    if completed_prefix % 12:
        raise R23CollectionError(
            "campaign checkpoint must end on a shard boundary; inner run-set owns pair resume"
        )
    by_shard: dict[str, list[CollectionSlot]] = {}
    for slot in slots:
        by_shard.setdefault(slot.shard_id, []).append(slot)
    ordered_shards = sorted(
        by_shard,
        key=lambda key: min(slot.shard_index for slot in by_shard[key]),
    )
    map_only = str(getattr(args, "map_only", "") or "")
    if map_only:
        if args.mode != "dry-run":
            raise R23CollectionError(
                "--map-only is dry-run only; collection checkpoint is contiguous"
            )
        ordered_shards = [
            shard_id
            for shard_id in ordered_shards
            if {slot.map_name for slot in by_shard[shard_id]} == {map_only}
        ]
        if not ordered_shards:
            raise R23CollectionError(f"campaign has no shards for map {map_only}")
    start_shard = 0 if args.mode == "dry-run" else completed_prefix // 12
    limit = int(args.limit_shards) if int(args.limit_shards) > 0 else len(ordered_shards)
    processed = 0
    for shard_position, shard_id in enumerate(ordered_shards[start_shard:], start=start_shard):
        shard_slots = sorted(by_shard[shard_id], key=lambda slot: slot.slot_index)
        reserve = all(slot.reserve for slot in shard_slots)
        if reserve:
            current = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            rows = current.get("results", [])
            completed = sum(str(row.get("status")) in {"COMPLETED", "SINGLETON"} for row in rows)
            technical = sum(
                bool(
                    row.get("failure_codes")
                    and set(row["failure_codes"]).issubset(
                        {
                            "CARLA_CONNECT_FAILURE",
                            "CARLA_RPC_TIMEOUT",
                            "SENSOR_SYNC_FAILURE",
                            "SPAWN_FAILURE",
                            "SERVER_FAILURE",
                            "CLEANUP_FAILURE",
                        }
                    )
                )
                for row in rows
            )
            if completed >= int(manifest["completed_target"]) or technical == 0:
                break
        shard_root = phase_root / "shards" / shard_id
        from scripts.r23_author_scenarios import _ensure_map

        _ensure_map(map_name=shard_slots[0].map_name)
        if args.mode == "collect":
            frozen_registry = shard_root / "dry-run" / "registry_manifest.json"
            if not frozen_registry.is_file():
                dry_command = _shard_command(
                    mode="dry-run",
                    shard_root=shard_root,
                    checkpoint=Path(args.r2_checkpoint),
                    host=args.host,
                    port=args.port,
                    device=args.device,
                )
                dry_code = int(
                    subprocess.run(dry_command, cwd=ROOT, check=False).returncode
                )
                if dry_code != 0 or not frozen_registry.is_file():
                    raise R23CollectionError(
                        f"pre-collection dry-run failed for {shard_id}: exit={dry_code}"
                    )
        command = _shard_command(
            mode=args.mode,
            shard_root=shard_root,
            checkpoint=Path(args.r2_checkpoint),
            host=args.host,
            port=args.port,
            device=args.device,
        )
        code = int(subprocess.run(command, cwd=ROOT, check=False).returncode)
        if args.mode == "dry-run":
            if code != 0:
                return code
        else:
            report_path = shard_root / "evidence" / "run_set_report.json"
            if not report_path.is_file():
                return code or 2
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for slot in shard_slots:
                result = _result_for_slot(slot, report)
                _write_collection_sample(
                    slot=slot,
                    result=result,
                    shard_root=shard_root,
                    campaign_manifest=manifest,
                    r2_checkpoint=Path(args.r2_checkpoint),
                )
                append_checkpoint_result(
                    checkpoint_path,
                    manifest,
                    result,
                )
        processed += 1
        if processed >= limit:
            break
    _json(
        {
            "status": "CAMPAIGN_STEP_COMPLETE",
            "mode": args.mode,
            "processed_shards": processed,
            "campaign": campaign_status(phase_root),
        }
    )
    return 0


def _write_prefrozen_lineage_splits(phase_root: Path) -> Path:
    campaign = json.loads(
        (phase_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    lineage_splits = {}
    for row in campaign["slots"]:
        lineage = str(row["lineage_id"])
        split = str(row["split"])
        previous = lineage_splits.setdefault(lineage, split)
        if previous != split:
            raise R23CollectionError(f"lineage split overlap: {lineage}")
    path = phase_root / "prefrozen_lineage_splits.json"
    payload = {
        "schema_version": "safedrive.r23_prefrozen_splits.v1",
        "campaign_manifest_hash": campaign["manifest_content_hash"],
        "lineage_splits": lineage_splits,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise R23CollectionError("existing prefrozen split manifest mismatch")
    else:
        write_json_exclusive(path, payload)
    return path


def cmd_build_r2(args: argparse.Namespace) -> int:
    phase_root = Path(args.phase_root)
    split_path = _write_prefrozen_lineage_splits(phase_root)
    command = [
        sys.executable,
        str(ROOT / "scripts/r2x_dataset_v5_from_features.py"),
        "--features-dir",
        str(phase_root / "shards"),
        "--merge-samples",
        str(phase_root / "NO_MERGE.jsonl"),
        "--prefrozen-split-manifest",
        str(split_path),
        "--out",
        str(Path(args.output)),
    ]
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


def cmd_build_world(args: argparse.Namespace) -> int:
    phase_root = Path(args.phase_root)
    campaign_manifest = phase_root / "campaign_manifest.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/r3_build_action_branch_dataset.py"),
        "--evidence-root",
        str(phase_root / "shards"),
        "--campaign-manifest",
        str(campaign_manifest),
        "--output",
        str(Path(args.output)),
        "--min-comparable",
        "650",
        "--min-decisive",
        "250",
        "--min-wins-per-slot",
        "100",
        "--min-future-coverage",
        "0.95",
        "--min-test-samples",
        "140",
        "--min-test-dual",
        "96",
        "--min-test-decisive",
        "60",
        "--min-test-wins-per-slot",
        "16",
    ]
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bank = sub.add_parser("init-bank")
    bank.add_argument("--output", required=True)
    bank.add_argument("--version", choices=("v1", "v2"), default="v1")
    bank.set_defaults(func=cmd_init_bank)
    plan = sub.add_parser("plan")
    plan.add_argument("--campaign-root", required=True)
    plan.add_argument("--lineage-bank", required=True)
    plan.add_argument("--r2-checkpoint", required=True)
    plan.add_argument("--collection-config", required=True)
    plan.add_argument(
        "--phase",
        choices=("r2_calibration", "world_formal"),
        default="r2_calibration",
        help="plan R2 first; plan World only after the new R2 head is frozen",
    )
    plan.set_defaults(func=cmd_plan)
    author = sub.add_parser("author")
    author.add_argument("--campaign-root", required=True)
    author.add_argument("--host", default="127.0.0.1")
    author.add_argument("--port", type=int, default=2000)
    author.add_argument("--map-only", default="")
    author.set_defaults(
        func=lambda args: subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/r23_author_scenarios.py"),
                "--campaign-root",
                args.campaign_root,
                "--host",
                args.host,
                "--port",
                str(args.port),
                *(
                    ["--map-only", str(args.map_only)]
                    if str(args.map_only)
                    else []
                ),
            ],
            cwd=ROOT,
            check=False,
        ).returncode
    )
    status = sub.add_parser("status")
    status.add_argument("--phase-root", required=True)
    status.set_defaults(func=cmd_status)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--collection-config", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--host", default="127.0.0.1")
    inventory.add_argument("--port", type=int, default=2000)
    inventory.set_defaults(func=cmd_inventory)
    for name in ("dry-run", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--phase-root", required=True)
        command.add_argument("--r2-checkpoint", required=True)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=2000)
        command.add_argument("--device", default="cuda")
        command.add_argument("--limit-shards", type=int, default=0)
        command.add_argument("--resume", action="store_true")
        command.add_argument(
            "--map-only",
            default="",
            help="dry-run only: select shards from one frozen map",
        )
        command.set_defaults(func=cmd_campaign_run, mode=name)
    build_r2 = sub.add_parser("build-r2")
    build_r2.add_argument("--phase-root", required=True)
    build_r2.add_argument("--output", required=True)
    build_r2.set_defaults(func=cmd_build_r2)
    build_world = sub.add_parser("build-world")
    build_world.add_argument("--phase-root", required=True)
    build_world.add_argument("--output", required=True)
    build_world.set_defaults(func=cmd_build_world)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except R23CollectionError as exc:
        print(f"R23 CAMPAIGN ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
