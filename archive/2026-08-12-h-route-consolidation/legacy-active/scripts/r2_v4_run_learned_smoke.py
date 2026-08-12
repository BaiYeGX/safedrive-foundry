#!/usr/bin/env python3
"""Run the frozen learned-only R2 V4 smoke cases.

This is the live counterpart of ``r2_v4_validate_learned_smoke.py``.  Case
identity and outcome summaries are written incrementally, and an existing
case directory is never overwritten.  The runner performs one resident V4
policy load and obtains the CARLA endpoint only from a READY preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_contract import canonical_json_bytes  # noqa: E402
from driving_vla.evaluation.paired_live_v4 import run_pair_v4  # noqa: E402
from driving_vla.evaluation.scenario_registry import load_scenario_registry  # noqa: E402
from driving_vla.model.neural_policy import NeuralV4Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402


SCHEMA = "safedrive.r2.v4.learned_smoke_manifest.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or str(value.get("schema_version") or "") != SCHEMA:
        raise ValueError(f"{path}: invalid learned smoke manifest schema")
    stored = str(value.get("manifest_hash") or "").lower()
    body = dict(value)
    body.pop("manifest_hash", None)
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if stored != actual:
        raise ValueError(f"{path}: learned smoke manifest hash mismatch")
    cases = list(value.get("cases") or [])
    expected = int(value.get("expected_cases") or 0)
    if expected not in {16, 32} or len(cases) != expected:
        raise ValueError(f"{path}: expected_cases/cases mismatch")
    identities = [(str(row.get("scenario_id") or ""), str(row.get("seed_id") or "")) for row in cases]
    if any(not scenario or not seed for scenario, seed in identities):
        raise ValueError(f"{path}: every case requires scenario_id and seed_id")
    if len(set(identities)) != len(identities):
        raise ValueError(f"{path}: duplicate smoke case identity")
    if expected == 32 and not all(bool(row.get("unseen_seed_or_route")) for row in cases):
        raise ValueError(f"{path}: the 32-case smoke must declare unseen seed/route cases")
    return value


def _preflight() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sdf.py"), "sim", "preflight", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"preflight JSON unavailable: {completed.stdout[-500:]}") from exc
    if value.get("status") != "READY":
        raise RuntimeError(f"CARLA preflight is not READY: {value}")
    return value


def _case_key(row: dict[str, Any]) -> str:
    raw = str(row.get("case_id") or f"{row['scenario_id']}__{row['seed_id']}")
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not clean:
        raise ValueError("smoke case has an empty case_id")
    return clean


def _outcome(report: dict[str, Any]) -> dict[str, Any]:
    collision = 0
    offroad = 0
    wrong_exit = 0
    permanent_stall = 0
    for branch in (report.get("branches") or {}).values():
        metrics = dict(branch.get("metrics") or {})
        collision += int(metrics.get("collision_episode_count", branch.get("collision_episodes", 0)) or 0)
        offroad += int(float(metrics.get("offroad_fraction", 0.0) or 0.0) >= 0.02)
        completion = dict(branch.get("maneuver_completion") or {})
        reasons = {str(value).upper() for value in completion.get("reason_codes") or ()}
        wrong_exit += int(any(value.startswith("WRONG_EXIT") for value in reasons))
        permanent_stall += int("PERMANENT_STALL" in reasons)
    fatal = bool(report.get("fatal")) or collision > 0 or offroad > 0 or wrong_exit > 0 or permanent_stall > 0
    return {
        "completed": bool(report.get("strict_success")) and not fatal,
        "collision": collision,
        "offroad": offroad,
        "wrong_exit": wrong_exit,
        "permanent_stall": permanent_stall,
        "fatal": fatal,
    }


def _validate_existing_case(
    report: dict[str, Any],
    *,
    row: dict[str, Any],
    checkpoint_sha256: str,
    source_manifest_hash: str,
) -> None:
    expected = {
        "scenario_id": str(row["scenario_id"]),
        "seed_id": str(row["seed_id"]),
        "namespace": "r2_v4_learned_smoke",
        "source_manifest_hash": str(source_manifest_hash),
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
    }
    for key, value in expected.items():
        if str(report.get(key) or "").lower() != value:
            raise RuntimeError(
                "existing learned-smoke evidence binding mismatch: "
                f"{key}={report.get(key)!r}, expected={value!r}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.cases).resolve()
    manifest = _read_manifest(manifest_path)
    expected_cases = int(manifest["expected_cases"])
    evidence_root = Path(args.evidence_root).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    registries = [load_scenario_registry(Path(value).resolve()) for value in args.registry]
    fixtures: dict[tuple[str, str], tuple[Any, Any]] = {}
    for registry in registries:
        for fixture in registry.fixtures:
            key = (str(fixture.scenario_id), str(fixture.seed_id))
            if key in fixtures:
                raise ValueError(f"duplicate fixture across registries: {key}")
            fixtures[key] = (registry, fixture)
    selected: list[tuple[dict[str, Any], Any, Any]] = []
    for row in manifest["cases"]:
        key = (str(row["scenario_id"]), str(row["seed_id"]))
        if key not in fixtures:
            raise ValueError(f"smoke case missing from supplied registries: {key}")
        registry, fixture = fixtures[key]
        selected.append((dict(row), registry, fixture))
    if args.plan_only:
        return {
            "schema_version": "safedrive.r2.v4.learned_smoke_plan.v1",
            "expected_cases": expected_cases,
            "source_manifest_sha256": _sha256(manifest_path),
            "registry_count": len(registries),
            "preflight": "NOT_RUN",
        }
    if not args.native_repair_gate:
        raise ValueError("learned smoke requires --native-repair-gate")
    gate = json.loads(Path(args.native_repair_gate).resolve().read_text(encoding="utf-8"))
    if str(gate.get("schema_version") or "") != "safedrive.r2.v4.native_repair_gate.v1" or not bool(gate.get("passed")):
        raise ValueError("native repair gate is not passing")
    preflight = _preflight()
    host = str(args.host or preflight.get("host") or "")
    if not host:
        raise ValueError("READY preflight did not return a CARLA host")
    runtime = SimLingoNeuralRuntime(device=args.device)
    load = runtime.load()
    if not load.ok:
        raise RuntimeError(f"SimLingo load failed: {load.error}")
    policy = NeuralV4Policy(
        runtime=runtime,
        semantic_head_checkpoint=args.checkpoint,
        keep_on_gpu=True,
        lazy=False,
        device=args.device,
        checkpoint_use=args.checkpoint_use,
    )
    policy.ensure_loaded()
    rows: list[dict[str, Any]] = []
    for row, registry, fixture in selected:
        case_dir = evidence_root / _case_key(row)
        if (case_dir / "pair_report.json").is_file():
            report = json.loads((case_dir / "pair_report.json").read_text(encoding="utf-8"))
            _validate_existing_case(
                report,
                row=row,
                checkpoint_sha256=_sha256(Path(args.checkpoint)),
                source_manifest_hash=str(manifest.get("manifest_hash") or ""),
            )
        elif case_dir.exists():
            raise RuntimeError(f"refusing to overwrite interrupted smoke evidence: {case_dir}")
        else:
            report = run_pair_v4(
                registry=registry,
                fixture=fixture,
                checkpoint=args.checkpoint,
                evidence_dir=case_dir,
                host=host,
                port=args.port,
                device=args.device,
                shared_policy=policy,
                namespace="r2_v4_learned_smoke",
                collect_actor_future=False,
                checkpoint_use=args.checkpoint_use,
                source_manifest_hash=str(manifest.get("manifest_hash") or ""),
                repeat_group=str(row.get("repeat_group") or ""),
                aa_noise_identity=str(row.get("aa_noise_identity") or ""),
            )
        rows.append({**row, **_outcome(report), "evidence_dir": str(case_dir)})
        progress = {
            "schema_version": "safedrive.r2.v4.learned_smoke_progress.v1",
            "expected_cases": expected_cases,
            "source_manifest_hash": str(manifest.get("manifest_hash") or ""),
            "checkpoint_sha256": _sha256(Path(args.checkpoint)),
            "rows": rows,
        }
        (evidence_root / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "safedrive.r2.v4.learned_smoke_report.v1",
        "expected_cases": expected_cases,
        "completed": sum(bool(row["completed"]) for row in rows),
        "collision": sum(int(row["collision"]) for row in rows),
        "offroad": sum(int(row["offroad"]) for row in rows),
        "wrong_exit": sum(int(row["wrong_exit"]) for row in rows),
        "permanent_stall": sum(int(row["permanent_stall"]) for row in rows),
        "fatal": sum(bool(row["fatal"]) for row in rows),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "source_manifest_hash": str(manifest.get("manifest_hash") or ""),
        "rows": rows,
    }
    output = evidence_root / "learned_smoke_report.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen smoke report: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", action="append", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--native-repair-gate", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--checkpoint-use", default="r2v4_formal")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.plan_only or (result.get("completed") == result.get("expected_cases") and not result.get("fatal")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
