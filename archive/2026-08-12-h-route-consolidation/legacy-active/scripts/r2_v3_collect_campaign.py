#!/usr/bin/env python3
"""Collect frozen K2 V3 campaign slots with resumable, immutable evidence."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/g3/run_g3_vla_mpc_stable.py"
MAP_LOADER = ROOT / "scripts/r2_v3_load_map.py"
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.navigation_contract import canonical_sha256  # noqa: E402


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    stored = str(body.pop("manifest_hash", ""))
    if stored != canonical_sha256(body):
        raise ValueError("campaign manifest hash mismatch")
    return value


def _completed_event_file(evidence: Path) -> Path | None:
    matches = sorted(
        path
        for path in evidence.glob("vla_events_*.json")
        if "partial" not in path.name
    )
    return matches[0] if len(matches) == 1 else None


def _scenario_id(slot: Mapping[str, Any]) -> str:
    return f"{slot['template_id']}_{slot['condition']}"


def _load_in_process_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "sdf_r2v3_stable_campaign_runner", RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import stable runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_in_process(
    runner_module: Any,
    *,
    runner_args: list[str],
    log_path: Path,
) -> int:
    old_argv = sys.argv
    sys.argv = [str(RUNNER), *runner_args, "--reuse-simlingo-runtime"]
    log_path.parent.mkdir(parents=True, exist_ok=False)
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(
                stream
            ):
                try:
                    return int(runner_module.main())
                except SystemExit as exc:
                    return int(exc.code or 0)
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--map", action="append", default=[])
    parser.add_argument("--mode", choices=("teacher", "learned"), default="teacher")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Keep one inference-only SimLingo backbone resident across slots.",
    )
    parser.add_argument(
        "--slot-timeout-s",
        type=float,
        default=0.0,
        help=(
            "Optional wall-clock timeout for one subprocess slot.  A timed-out "
            "runner is still accepted only when it left exactly one complete "
            "vla_events file; the report records timeout_after_evidence."
        ),
    )
    args = parser.parse_args()
    if args.mode == "learned" and not args.checkpoint:
        parser.error("--mode learned requires --checkpoint")

    manifest = _manifest(Path(args.manifest))
    selected_maps = set(str(value) for value in args.map)
    slots = [
        slot
        for slot in manifest["slots"]
        if not selected_maps or str(slot["map_name"]) in selected_maps
    ]
    if not slots:
        raise ValueError("no campaign slots selected")
    by_map: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for slot in slots:
        by_map[str(slot["map_name"])].append(slot)

    evidence_root = Path(args.evidence_root)
    registry_root = Path(args.registry_root)
    rows: list[dict[str, Any]] = []
    progress_path = evidence_root / "campaign_progress.json"
    in_process_runner = _load_in_process_runner() if args.in_process else None
    for map_name in sorted(by_map):
        load = subprocess.run(
            [
                sys.executable,
                str(MAP_LOADER),
                "--map",
                map_name,
                "--timeout",
                "300",
            ],
            cwd=ROOT,
            check=False,
        )
        if load.returncode != 0:
            raise RuntimeError(f"failed to load CARLA map {map_name}")
        for slot in sorted(
            by_map[map_name], key=lambda value: str(value["slot_id"])
        ):
            slot_id = str(slot["slot_id"])
            evidence = evidence_root / slot_id
            if evidence.exists():
                if args.resume and _completed_event_file(evidence) is not None:
                    row = {
                        "slot_id": slot_id,
                        "map_name": map_name,
                        "status": "COMPLETED_EXISTING",
                        "evidence_dir": str(evidence.as_posix()),
                    }
                    rows.append(row)
                    _write_json_atomic(
                        progress_path,
                        {
                            "schema_version": (
                                "safedrive.r2_v3.campaign_progress.v1"
                            ),
                            "manifest_hash": manifest["manifest_hash"],
                            "rows": rows,
                        },
                    )
                    continue
                raise RuntimeError(
                    f"refusing existing incomplete evidence: {evidence}"
                )
            registry = (
                registry_root
                / map_name
                / str(slot["lineage_id"])
                / "scenario_registry.toml"
            )
            if not registry.is_file():
                raise FileNotFoundError(
                    f"campaign registry missing: {registry}"
                )
            runner_args = [
                "--map",
                map_name,
                "--duration-s",
                "5.0",
                "--sim-dt",
                "0.05",
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
                str(args.mode),
                "--v3-scenario-registry",
                str(registry),
                "--v3-scenario-id",
                _scenario_id(slot),
                "--v3-seed-id",
                str(slot["seed_id"]),
                "--no-map-restart",
                "--evidence-dir",
                str(evidence),
            ]
            if args.mode == "teacher":
                runner_args.append("--v3-contract-auto-select")
            else:
                runner_args.extend(
                    [
                        "--spatial-head-checkpoint",
                        str(args.checkpoint),
                        "--spatial-checkpoint-use",
                        "development_live_smoke",
                    ]
                )
            template = str(slot["template_id"])
            if template == "overtake_left":
                runner_args.extend(
                    ["--v3-requested-overtake-side", "LEFT"]
                )
            elif template == "overtake_right":
                runner_args.extend(
                    ["--v3-requested-overtake-side", "RIGHT"]
                )
            start = time.time()
            print(f"[r2-v3-collect] {slot_id}", flush=True)
            timed_out = False
            if in_process_runner is not None:
                result_code = _run_in_process(
                    in_process_runner,
                    runner_args=runner_args,
                    log_path=evidence / "collector_runner.log",
                )
            else:
                try:
                    result = subprocess.run(
                        [sys.executable, str(RUNNER), *runner_args],
                        cwd=ROOT,
                        check=False,
                        timeout=(
                            None
                            if float(args.slot_timeout_s) <= 0.0
                            else float(args.slot_timeout_s)
                        ),
                    )
                    result_code = int(result.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    result_code = 124
            completed = (result_code == 0 or timed_out) and (
                _completed_event_file(evidence) is not None
            )
            row = {
                "slot_id": slot_id,
                "map_name": map_name,
                "lineage_id": str(slot["lineage_id"]),
                "status": "COMPLETED" if completed else "FAILED",
                "runner_exit_code": result_code,
                "wall_time_s": time.time() - start,
                "evidence_dir": str(evidence.as_posix()),
            }
            if timed_out:
                row["timeout_after_evidence"] = bool(completed)
            rows.append(row)
            _write_json_atomic(
                progress_path,
                {
                    "schema_version": (
                        "safedrive.r2_v3.campaign_progress.v1"
                    ),
                    "manifest_hash": manifest["manifest_hash"],
                    "rows": rows,
                },
            )
            if not completed and not args.continue_on_failure:
                raise RuntimeError(f"slot failed: {slot_id}")
    failed = [row for row in rows if row["status"] == "FAILED"]
    report = {
        "schema_version": "safedrive.r2_v3.campaign_report.v1",
        "manifest_hash": manifest["manifest_hash"],
        "mode": str(args.mode),
        "selected_maps": sorted(by_map),
        "slot_count": len(rows),
        "completed": len(rows) - len(failed),
        "failed": len(failed),
        "rows": rows,
    }
    report_path = evidence_root / "campaign_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
