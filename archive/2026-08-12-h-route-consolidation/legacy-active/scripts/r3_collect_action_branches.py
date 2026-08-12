#!/usr/bin/env python3
"""Run the existing frozen paired orchestrator with R3 actor-future capture enabled."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", action="append", required=True)
    parser.add_argument("--registry-manifest", default="")
    parser.add_argument("--spatial-head-ckpt", default="")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--host",
        default="",
        help="optional explicit CARLA host; otherwise use the READY preflight endpoint",
    )
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--v4-head-ckpt", default="")
    parser.add_argument("--campaign-manifest", default="")
    parser.add_argument("--namespace", default="r3_final_head_formal")
    parser.add_argument("--include-reserve", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument(
        "--native-repair-gate",
        default="",
        help="validated R2 V4 native repair gate required before live R3 collection",
    )
    args = parser.parse_args()
    registries = [Path(value).resolve() for value in args.registry]
    if any("r2-spatial-k2-pilot-v5-blind" in str(path) for path in registries):
        raise SystemExit("formal blind registry is audit-only and cannot be used for R3 collection")
    if args.v4_head_ckpt:
        if not args.campaign_manifest:
            raise SystemExit("--v4-head-ckpt requires --campaign-manifest")
        command = [
            sys.executable,
            str(ROOT / "scripts/r2_v4_collect_campaign.py"),
            "--registry",
            str(registries[0]),
            "--campaign-manifest",
            str(Path(args.campaign_manifest).resolve()),
            "--checkpoint",
            str(Path(args.v4_head_ckpt).resolve()),
            "--evidence-root",
            str(Path(args.evidence_dir).resolve()),
            "--device",
            args.device,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--namespace",
            args.namespace,
            "--checkpoint-use",
            "r3_final_head_formal",
            "--collect-actor-future",
        ]
        for extra_registry in registries[1:]:
            command.extend(["--registry", str(extra_registry)])
        if args.native_repair_gate:
            command.extend(["--native-repair-gate", str(Path(args.native_repair_gate).resolve())])
        if args.plan_only:
            command.append("--plan-only")
        if args.include_reserve:
            command.append("--include-reserve")
        if args.development_only:
            command.append("--development-only")
        return int(subprocess.run(command, cwd=ROOT).returncode)
    if len(registries) != 1:
        raise SystemExit("legacy R3 collection accepts exactly one --registry")
    registry = registries[0]
    if not args.registry_manifest or not args.spatial_head_ckpt:
        raise SystemExit("legacy R3 collection requires --registry-manifest and --spatial-head-ckpt")
    command = [
        sys.executable,
        str(ROOT / "tests/g4/run_g4a_paired.py"),
        "--registry",
        str(registry),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "run-set",
        "--evidence-dir",
        str(Path(args.evidence_dir)),
        "--registry-manifest",
        str(Path(args.registry_manifest)),
        "--device",
        args.device,
        "--spatial-k2",
        "--spatial-head-ckpt",
        str(Path(args.spatial_head_ckpt)),
        "--checkpoint-use",
        "collection_anchor",
        "--continue-policy",
        "continue_all",
        "--min-comparable",
        "0",
        "--no-aggregate",
        "--collect-observable-history",
    ]
    if args.plan_only:
        command.append("--plan-only")
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
