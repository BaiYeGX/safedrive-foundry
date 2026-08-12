#!/usr/bin/env python3
"""Run a pre-frozen R2 V4 campaign with one resident SimLingo per process."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_live_v4 import run_pair_v4  # noqa: E402
from driving_vla.evaluation.scenario_registry import load_scenario_registry  # noqa: E402
from driving_vla.model.neural_policy import NeuralV4Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402
from r2_v4_author_scenarios import _ensure_map  # noqa: E402


MAP_ORDER = ("Town03", "Town04", "Town05", "Town06", "Town10HD", "Town12", "Town13")


def _ensure_collection_map(map_name: str) -> dict[str, Any]:
    """Cold-start one map block and return its READY endpoint evidence."""

    # Town13 has a reproducible DX12 startup access violation on this host;
    # the supported DX11 fallback is restricted to that block and is recorded
    # in every pair report.  All other maps retain the native DX12 protocol.
    requested_rhi = "dx11" if str(map_name) == "Town13" else "dx12"
    value = dict(_ensure_map(str(map_name), rhi=requested_rhi))
    if str(value.get("status")) != "READY":
        raise RuntimeError(f"CARLA map block is not READY: {value}")
    value.setdefault("requested_map", str(map_name))
    value.setdefault("requested_rhi", requested_rhi)
    value.setdefault("effective_rhi", requested_rhi)
    return value


def _preflight() -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sdf.py"), "sim", "preflight", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        # ``sdf.py --json`` emits one pretty-printed JSON object; parsing only
        # the last line turns a valid READY response into a false parser
        # failure and can leave a resumable campaign looking unstarted.
        value = json.loads(completed.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"preflight JSON unavailable: {completed.stdout[-500:]} {completed.stderr[-300:]}"
        ) from exc
    if value.get("status") != "READY":
        raise RuntimeError(f"CARLA preflight is not READY: {value}")
    return value


def _validate_pilot_report(path: Path) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"pilot report is not valid JSON: {path}") from exc
    if str(report.get("schema_version") or "") != "safedrive.r2_v4.pilot_compare.v1":
        raise SystemExit("pilot report schema mismatch")
    if int(report.get("anchor_count", 0)) != 144:
        raise SystemExit("pilot report must be bound to exactly 144 anchors")
    if bool(report.get("town13_read", True)):
        raise SystemExit("pilot report indicates Town13 was read")
    if not bool(report.get("pilot_pass")):
        raise SystemExit("V4 representation pilot did not pass its promotion gates")


def _validate_existing_pair(
    evidence_dir: Path,
    *,
    slot: dict[str, Any],
    namespace: str,
    checkpoint: Path,
    source_manifest_hash: str,
) -> None:
    """Allow resume only for the exact frozen slot and checkpoint."""
    report_path = evidence_dir / "pair_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"complete pair report is unreadable: {report_path}") from exc
    expected = {
        "scenario_id": str(slot["scenario_id"]),
        "seed_id": str(slot.get("seed_id") or ""),
        "namespace": str(namespace),
        "source_manifest_hash": str(source_manifest_hash),
    }
    for key, value in expected.items():
        if str(report.get(key) or "") != value:
            raise SystemExit(
                f"existing pair evidence binding mismatch for {evidence_dir}: "
                f"{key}={report.get(key)!r}, expected={value!r}"
            )
    checkpoint_hash = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    if str(report.get("checkpoint_sha256") or "").lower() != checkpoint_hash:
        raise SystemExit(f"existing pair checkpoint binding mismatch: {evidence_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", action="append", required=True)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument(
        "--audit-manifest",
        default="",
        help="optional frozen R2 V4 252-pair audit manifest; replaces campaign slots",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--host",
        default="",
        help="optional explicit CARLA host; otherwise use the READY preflight endpoint",
    )
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--namespace", default="r2_v4_formal")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="collect the pre-formal 144-anchor pilot with a development-live checkpoint",
    )
    parser.add_argument(
        "--pilot-report",
        default="",
        help="passing 144-anchor representation pilot; required before R2 expansion",
    )
    parser.add_argument(
        "--checkpoint-use",
        default="",
        help="checkpoint contract use; inferred from namespace when omitted",
    )
    parser.add_argument("--collect-actor-future", action="store_true")
    parser.add_argument(
        "--native-repair-gate",
        default="",
        help="validated JSON from r2_v4_validate_native_repairs.py; required for live collection",
    )
    parser.add_argument(
        "--include-reserve",
        action="store_true",
        help="after the pre-frozen base gate fails, append only the 504 technical reserve slots",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="collect only the pre-frozen R3 512-slot train/val development subset",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        if args.audit_manifest or args.development_only:
            raise SystemExit("--pilot cannot be combined with --audit-manifest or --development-only")
        if str(args.namespace) == "r2_v4_formal":
            args.namespace = "r2_v4_pilot"
        if str(args.namespace) != "r2_v4_pilot":
            raise SystemExit("--pilot requires namespace r2_v4_pilot")
        if args.limit not in {0, 144}:
            raise SystemExit("--pilot must select exactly 144 anchors (use --limit 144)")
        args.limit = 144
    free_gb = shutil.disk_usage(ROOT).free / float(1024**3)
    if free_gb < 80.0 and not args.plan_only:
        raise SystemExit(f"disk guard: {free_gb:.1f} GiB free < 80 GiB")
    campaign_path = Path(args.campaign_manifest).resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    source_manifest_hash = str(campaign.get("manifest_hash") or "")
    if args.audit_manifest:
        audit_path = Path(args.audit_manifest).resolve()
        campaign = json.loads(audit_path.read_text(encoding="utf-8"))
        source_manifest_hash = str(campaign.get("manifest_hash") or "")
        if not campaign.get("pairs"):
            raise SystemExit("audit manifest has no frozen pairs")
        campaign["slots"] = list(campaign["pairs"])
    if args.development_only:
        if args.audit_manifest:
            raise SystemExit("--development-only cannot be combined with the R2 audit manifest")
        development = list(campaign.get("development_slots") or [])
        if len(development) != 512:
            raise SystemExit("campaign must freeze exactly 512 development_slots")
        campaign["slots"] = development
        args.include_reserve = False
    slot_rows = list(campaign.get("slots", []))
    if args.include_reserve:
        slot_rows.extend(list(campaign.get("reserve_slots", [])))
        if len(slot_rows) > 2520:
            raise SystemExit("R3 reserve hard cap exceeded: >2520 slots")
    if args.audit_manifest and str(args.namespace) == "r2_v4_formal":
        # The audit aggregator is namespace-strict.  Refusing the ambiguous
        # default prevents a 252-pair run from being mislabeled as ordinary
        # calibration evidence.
        args.namespace = "r2v4_blind_audit"
    slot_keys = [
        (str(row["scenario_id"]), str(row.get("seed_id") or ""))
        for row in slot_rows
        if str(row.get("namespace") or args.namespace) == str(args.namespace)
        or (
            bool(args.include_reserve)
            and str(row.get("namespace") or "") == f"{args.namespace}_reserve"
        )
    ]
    if len(slot_keys) != len(set(slot_keys)):
        raise SystemExit("frozen campaign contains duplicate (scenario_id, seed_id) slots")
    slots = {
        (str(row["scenario_id"]), str(row.get("seed_id") or "")): row
        for row in slot_rows
        if str(row.get("namespace") or args.namespace) == str(args.namespace)
        or (
            bool(args.include_reserve)
            and str(row.get("namespace") or "") == f"{args.namespace}_reserve"
        )
    }
    fixtures = []
    registries = []
    for raw in args.registry:
        registry = load_scenario_registry(Path(raw))
        registries.append(registry)
        for fixture in registry.fixtures:
            key = (str(fixture.scenario_id), str(fixture.seed_id))
            if key in slots:
                if any(
                    (existing[1].scenario_id, existing[1].seed_id) == key
                    for existing in fixtures
                ):
                    raise SystemExit(f"duplicate fixture across registries: {key}")
                fixtures.append((registry, fixture, slots[key]))
    map_rank = {name: index for index, name in enumerate(MAP_ORDER)}
    fixtures.sort(
        key=lambda item: (
            map_rank.get(str(item[1].map_name), len(MAP_ORDER)),
            str(item[2]["scenario_id"]),
        )
    )
    expected_count = len(slots)
    if args.limit:
        expected_count = min(expected_count, int(args.limit))
    if len(fixtures) < expected_count:
        missing = sorted(
            set(slots)
            - {
                (str(fixture.scenario_id), str(fixture.seed_id))
                for _registry, fixture, _slot in fixtures
            }
        )
        raise SystemExit(
            f"frozen campaign coverage mismatch: expected={expected_count} "
            f"matched={len(fixtures)} missing={len(missing)} first={missing[:3]}"
        )
    if args.limit:
        fixtures = fixtures[: int(args.limit)]
    if args.plan_only:
        print(
            json.dumps(
                {
                    "namespace": args.namespace,
                    "slots_selected": len(fixtures),
                    "disk_free_gib": free_gb,
                    "preflight": "NOT_RUN",
                },
                indent=2,
            )
        )
        return 0
    if not args.native_repair_gate:
        raise SystemExit(
            "live R2 V4 collection requires --native-repair-gate with both maneuver repairs and teacher 16/16"
        )
    gate_path = Path(args.native_repair_gate).resolve()
    if not gate_path.is_file():
        raise SystemExit(f"native repair gate missing: {gate_path}")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"native repair gate is not valid JSON: {gate_path}") from exc
    if str(gate.get("schema_version") or "") != "safedrive.r2.v4.native_repair_gate.v1" or not bool(gate.get("passed")):
        raise SystemExit("native repair gate failed; R2 V4 collection is blocked")
    required_use = {
        "r2_v4_pilot": "development_live_smoke",
        # Calibration creates the training anchors.  It must not require a
        # formal checkpoint trained on those same anchors.
        "r2_v4_calibration": "collection_anchor",
        "r2_v4_formal": "r2v4_formal",
        "r2v4_blind_audit": "r2v4_blind_audit",
        "r3_final_head_formal": "r3_final_head_formal",
    }.get(str(args.namespace))
    if not args.checkpoint_use:
        args.checkpoint_use = required_use or "development_live_smoke"
    if required_use is not None and str(args.checkpoint_use) != required_use:
        raise SystemExit(
            f"namespace {args.namespace} requires --checkpoint-use {required_use}"
        )
    expansion_namespace = str(args.namespace) in {"r2_v4_formal", "r2v4_blind_audit"}
    if expansion_namespace and len(fixtures) > 144:
        if not args.pilot_report:
            raise SystemExit(
                "R2 expansion collection requires --pilot-report from the passing 144-anchor pilot"
            )
        _validate_pilot_report(Path(args.pilot_report).resolve())
    # The first map block is cold-started below; do not attach to whichever
    # map happened to be left open by an earlier campaign.
    resolved_host = ""
    runtime = SimLingoNeuralRuntime(device=args.device)
    load = runtime.load()
    if not load.ok:
        raise SystemExit(f"SimLingo load failed: {load.error}")
    policy = NeuralV4Policy(
        runtime=runtime,
        semantic_head_checkpoint=args.checkpoint,
        keep_on_gpu=True,
        lazy=False,
        device=args.device,
        checkpoint_use=args.checkpoint_use,
    )
    policy.ensure_loaded()
    root = Path(args.evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    current_map = ""
    block_runtime: dict[str, Any] = {}
    for index, (registry, fixture, slot) in enumerate(fixtures):
        if str(fixture.map_name) != current_map:
            block_runtime = _ensure_collection_map(str(fixture.map_name))
            resolved_host = str(args.host or block_runtime.get("host") or "")
            if not resolved_host:
                raise SystemExit(
                    f"READY preflight did not return a CARLA host for {fixture.map_name}"
                )
            current_map = str(fixture.map_name)
        if shutil.disk_usage(root).free / float(1024**3) < 40.0:
            raise SystemExit("disk guard: free space fell below 40 GiB; campaign checkpoint retained")
        evidence_dir = root / str(slot["slot_id"])
        if evidence_dir.exists():
            if (evidence_dir / "pair_report.json").is_file():
                _validate_existing_pair(
                    evidence_dir,
                    slot=slot,
                    namespace=str(args.namespace),
                    checkpoint=Path(args.checkpoint).resolve(),
                    source_manifest_hash=source_manifest_hash,
                )
                continue
            raise SystemExit(
                "refusing to overwrite interrupted V4 evidence directory: "
                f"{evidence_dir}"
            )
        report = run_pair_v4(
            registry=registry,
            fixture=fixture,
            checkpoint=args.checkpoint,
            evidence_dir=evidence_dir,
            host=resolved_host,
            port=args.port,
            device=args.device,
            shared_policy=policy,
            namespace=args.namespace,
            collect_actor_future=bool(args.collect_actor_future),
            checkpoint_use=args.checkpoint_use,
            source_manifest_hash=source_manifest_hash,
            repeat_group=str(slot.get("repeat_group") or ""),
            aa_noise_identity=str(slot.get("aa_noise_identity") or ""),
            carla_runtime={
                "map_name": str(fixture.map_name),
                "host": resolved_host,
                "requested_rhi": str(block_runtime.get("requested_rhi") or ""),
                "effective_rhi": str(block_runtime.get("effective_rhi") or ""),
                "host_source": str(block_runtime.get("host_source") or ""),
            },
        )
        reports.append(report)
        print(json.dumps({"index": index, "slot_id": slot["slot_id"], "comparable": report["comparable"]}, sort_keys=True), flush=True)
    print(json.dumps({"completed_this_run": len(reports), "evidence_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
