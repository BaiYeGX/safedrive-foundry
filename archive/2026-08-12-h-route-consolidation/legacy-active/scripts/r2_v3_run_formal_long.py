#!/usr/bin/env python3
"""Run learned 15–20 s route/interaction completion on formal V3 fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import canonical_sha256  # noqa: E402
from scripts.r2_v3_run_long_smokes import _validate_case  # noqa: E402

RUNNER = ROOT / "tests/g3/run_g3_vla_mpc_stable.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frozen(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if stored != canonical_sha256(body):
        raise ValueError(f"{path}: manifest hash mismatch")
    return value


def _write(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _conflict_side(template_id: str) -> str:
    if template_id in {"cut_in_left", "overtake_left"}:
        return "left"
    if template_id in {"cut_in_right", "overtake_right"}:
        return "right"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authoring-root", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args()

    manifest = _read_frozen(Path(args.manifest))
    map_name = str(args.map)
    cases = [
        dict(case)
        for case in manifest.get("cases") or ()
        if str(case.get("map_name") or "") == map_name
    ]
    if not cases:
        raise ValueError(f"no {manifest['phase']} cases for {map_name}")
    checkpoint = Path(args.checkpoint)
    checkpoint_hash = _sha(checkpoint)
    root = Path(args.evidence_root)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(cases):
        fixture_id = str(case["fixture_id"])
        seed_id = str(case.get("seed_id") or "seed_a")
        registry_path = (
            Path(args.authoring_root)
            / str(manifest["phase"])
            / map_name
            / fixture_id
            / "scenario_registry.toml"
        )
        registry = load_scenario_registry(registry_path)
        fixture = registry.get(fixture_id, seed_id)
        evidence = root / fixture_id
        evidence.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(RUNNER),
            "--map",
            map_name,
            "--duration-s",
            str(float(fixture.duration_s)),
            "--sim-dt",
            str(float(fixture.sim_dt_s)),
            "--vla-period-s",
            "0.75",
            "--v-ref",
            "6.0",
            "--speed-gain",
            "1.0",
            "--coarse-route-spacing-m",
            "10.0",
            "--vla-version",
            "v3",
            "--v3-mode",
            "learned",
            "--v3-scenario-registry",
            str(registry_path),
            "--v3-scenario-id",
            fixture_id,
            "--v3-seed-id",
            seed_id,
            "--v3-contract-auto-select",
            "--spatial-head-checkpoint",
            str(checkpoint),
            "--spatial-checkpoint-use",
            "r2v3_blind_audit",
            "--no-map-restart",
            "--evidence-dir",
            str(evidence),
        ]
        template_id = str(case["template_id"])
        if template_id == "overtake_left":
            command.extend(["--v3-requested-overtake-side", "LEFT"])
        elif template_id == "overtake_right":
            command.extend(["--v3-requested-overtake-side", "RIGHT"])
        started = time.time()
        with (evidence / "runner.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=600.0,
            )
        row = {
            **case,
            "runner_exit_code": int(result.returncode),
            "wall_time_s": time.time() - started,
            "checkpoint_sha256": checkpoint_hash,
            "completed": False,
            "collision": False,
            "offroad": False,
            "wrong_exit": False,
            "evidence_dir": str(evidence),
        }
        if result.returncode == 0:
            try:
                acceptance = _validate_case(
                    fixture_id,
                    evidence,
                    family=str(case["family"]),
                    case_id=str(
                        case.get("case_id")
                        or case.get("pair_id")
                        or fixture_id
                    ),
                    conflict_side=_conflict_side(template_id),
                )
                route = dict(acceptance["route_completion"])
                reasons = {
                    str(reason)
                    for reason in route.get("reason_codes") or ()
                }
                row.update(
                    {
                        "completed": bool(acceptance["completed"]),
                        "collision": bool(route.get("collision")),
                        "offroad": bool(route.get("offroad")),
                        "wrong_exit": any(
                            reason.startswith("WRONG_EXIT")
                            for reason in reasons
                        ),
                        "reason_codes": list(
                            acceptance["reason_codes"]
                        ),
                        "route_completion": route,
                        "interaction_completion": dict(
                            acceptance["interaction_completion"]
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["validation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["reason_codes"] = ["RUNNER_FAILED"]
        rows.append(row)
        _write(
            root / "progress.json",
            {
                "schema_version": "safedrive.r2_v3.formal_long_progress.v1",
                "phase": str(manifest["phase"]),
                "map_name": map_name,
                "source_manifest_hash": manifest["manifest_hash"],
                "checkpoint_sha256": checkpoint_hash,
                "rows": rows,
            },
        )
        print(
            f"[formal-long] {index + 1}/{len(cases)} {fixture_id} "
            f"{'PASS' if row['completed'] else 'FAIL'}",
            flush=True,
        )
    report = {
        "schema_version": "safedrive.r2_v3.formal_long.v1",
        "phase": str(manifest["phase"]),
        "map_name": map_name,
        "source_manifest_hash": manifest["manifest_hash"],
        "checkpoint_sha256": checkpoint_hash,
        "n_run": len(rows),
        "completed": sum(bool(row["completed"]) for row in rows),
        "rows": rows,
    }
    report["passed"] = int(report["completed"]) == len(rows)
    _write(root / "long_campaign_report.json", report, exclusive=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
