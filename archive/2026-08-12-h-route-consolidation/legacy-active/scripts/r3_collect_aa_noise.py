#!/usr/bin/env python3
"""Collect the pre-frozen 168 same-action A/A cold-rebuild repeats."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_live_v4 import run_aa_repeat_v4  # noqa: E402
from driving_vla.evaluation.scenario_registry import load_scenario_registry  # noqa: E402
from driving_vla.model.neural_policy import NeuralV4Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402


def _preflight() -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sdf.py"), "sim", "preflight", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    if value.get("status") != "READY":
        raise RuntimeError(f"CARLA preflight is not READY: {value}")
    return value


def _validate_existing_aa(
    path: Path,
    *,
    probe: dict,
    campaign_hash: str,
    checkpoint: Path,
) -> None:
    try:
        report = json.loads((path / "aa_report.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"complete A/A report is unreadable: {path}") from exc
    expected = {
        "scenario_id": str(probe["scenario_id"]),
        "seed_id": str(probe.get("seed_id") or ""),
        "repeat_group": str(probe.get("repeat_group") or ""),
        "aa_noise_identity": str(probe.get("aa_noise_identity") or "").lower(),
        "source_manifest_hash": str(campaign_hash),
    }
    for key, value in expected.items():
        if str(report.get(key) or "").lower() != value:
            raise SystemExit(
                f"existing A/A evidence binding mismatch for {path}: "
                f"{key}={report.get(key)!r}, expected={value!r}"
            )
    checkpoint_hash = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    if str(report.get("checkpoint_sha256") or "").lower() != checkpoint_hash:
        raise SystemExit(f"existing A/A checkpoint binding mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", action="append", required=True)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--host",
        default="",
        help="optional explicit CARLA host; otherwise use the READY preflight endpoint",
    )
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.evidence_root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root.parent).free / float(1024**3)
    if free_gb < 80.0 and not args.plan_only:
        raise SystemExit(f"disk guard: {free_gb:.1f} GiB free < 80 GiB")
    campaign_path = Path(args.campaign_manifest).resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    probe_keys = [
        (str(row["scenario_id"]), str(row.get("seed_id") or ""))
        for row in campaign.get("noise_probe", [])
    ]
    if len(probe_keys) != len(set(probe_keys)):
        raise SystemExit("A/A campaign contains duplicate (scenario_id, seed_id) probes")
    probes = {
        (str(row["scenario_id"]), str(row.get("seed_id") or "")): row
        for row in campaign.get("noise_probe", [])
    }
    fixtures = []
    for raw in args.registry:
        registry = load_scenario_registry(Path(raw))
        for fixture in registry.fixtures:
            key = (str(fixture.scenario_id), str(fixture.seed_id))
            if key in probes:
                if any(
                    (existing[1].scenario_id, existing[1].seed_id) == key
                    for existing in fixtures
                ):
                    raise SystemExit(f"duplicate fixture across registries: {key}")
                fixtures.append((registry, fixture, probes[key]))
    fixtures.sort(key=lambda item: str(item[2]["repeat_group"]))
    expected_count = len(probes)
    if args.limit:
        expected_count = min(expected_count, int(args.limit))
    if len(fixtures) < expected_count:
        missing = sorted(
            set(probes)
            - {
                (str(fixture.scenario_id), str(fixture.seed_id))
                for _registry, fixture, _probe in fixtures
            }
        )
        raise SystemExit(
            f"A/A probe coverage mismatch: expected={expected_count} "
            f"matched={len(fixtures)} missing={len(missing)} first={missing[:3]}"
        )
    if args.limit:
        fixtures = fixtures[: int(args.limit)]
    if args.plan_only:
        print(json.dumps({"namespace": "r3_aa_noise_probe", "slots_selected": len(fixtures), "preflight": "NOT_RUN"}, indent=2))
        return 0
    preflight = _preflight()
    resolved_host = str(args.host or preflight.get("host") or "")
    if not resolved_host:
        raise SystemExit("READY preflight did not return a CARLA host")
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
        checkpoint_use="r3_final_head_formal",
    )
    policy.ensure_loaded()
    root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for registry, fixture, probe in fixtures:
        if shutil.disk_usage(root).free / float(1024**3) < 40.0:
            raise SystemExit("disk guard: free space fell below 40 GiB")
        evidence = root / str(probe["repeat_group"])
        if (evidence / "aa_report.json").is_file():
            _validate_existing_aa(
                evidence,
                probe=probe,
                campaign_hash=str(campaign.get("manifest_hash") or ""),
                checkpoint=Path(args.checkpoint).resolve(),
            )
            continue
        if evidence.exists():
            raise SystemExit(
                "refusing to overwrite interrupted A/A evidence directory: "
                f"{evidence}"
            )
        run_aa_repeat_v4(
            registry=registry,
            fixture=fixture,
            checkpoint=args.checkpoint,
            evidence_dir=evidence,
            host=resolved_host,
            port=args.port,
            device=args.device,
            shared_policy=policy,
            namespace="r3_aa_noise_probe",
            source_manifest_hash=str(campaign.get("manifest_hash") or ""),
            repeat_group=str(probe.get("repeat_group") or ""),
            aa_noise_identity=str(probe.get("aa_noise_identity") or ""),
        )
        completed += 1
    print(json.dumps({"completed_this_run": completed, "evidence_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
