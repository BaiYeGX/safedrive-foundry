#!/usr/bin/env python3
"""Run and independently validate the frozen R2 V3 long-smoke registry."""

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
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.long_horizon_observer import (  # noqa: E402
    InteractionBehavior,
    LongHorizonObserver,
)
from driving_vla.evaluation.maneuver_completion import route_projection  # noqa: E402
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    load_scenario_registry,
)
from driving_vla.model.navigation_contract import RouteContextV3  # noqa: E402


RUNNER = ROOT / "tests/g3/run_g3_vla_mpc_stable.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _behavior(
    scenario_id: str,
    *,
    family: str | None = None,
) -> InteractionBehavior:
    family_l = str(family or "").lower()
    if "lead" in family_l or "follow" in family_l:
        return InteractionBehavior.FOLLOW_STOP
    if "cut" in family_l:
        return InteractionBehavior.CUT_IN_AVOID
    if "cross" in family_l or "merge" in family_l:
        return InteractionBehavior.YIELD_WAIT
    if "obstruction" in family_l or "overtake" in family_l:
        return InteractionBehavior.OVERTAKE_REJOIN
    if "traffic" in family_l:
        return InteractionBehavior.TRAFFIC_CONTROL
    if family_l in {"clear", "empty"}:
        return InteractionBehavior.CLEAR
    if scenario_id == "follow_stop_resume":
        return InteractionBehavior.FOLLOW_STOP
    if scenario_id.startswith("cut_in_"):
        return InteractionBehavior.CUT_IN_AVOID
    if scenario_id in {
        "left_turn_crossing_yield",
        "right_turn_merge_yield",
    }:
        return InteractionBehavior.YIELD_WAIT
    if scenario_id.startswith("overtake_"):
        return InteractionBehavior.OVERTAKE_REJOIN
    if scenario_id == "red_green_resume_route":
        return InteractionBehavior.TRAFFIC_CONTROL
    return InteractionBehavior.CLEAR


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _conflict_point(
    route_context: RouteContextV3,
    rows: list[dict[str, Any]],
) -> float | None:
    junction = next((row for row in rows if bool(row.get("is_junction"))), None)
    if junction is not None:
        return route_projection(
            route_context.route_xy,
            float(junction["ego_x"]),
            float(junction["ego_y"]),
        )[0]
    active = [row for row in rows if bool(row.get("conflict_active"))]
    if not active:
        return None
    closest = min(
        active,
        key=lambda row: float(row.get("actor_clearance_m") or 1.0e9),
    )
    return route_projection(
        route_context.route_xy,
        float(closest["ego_x"]),
        float(closest["ego_y"]),
    )[0]


def _validate_case(
    scenario_id: str,
    evidence_dir: Path,
    *,
    family: str | None = None,
    case_id: str | None = None,
    conflict_side: str | None = None,
) -> dict[str, Any]:
    route_context = RouteContextV3.from_mapping(
        json.loads((evidence_dir / "route_context_v3.json").read_text(encoding="utf-8"))
    )
    rows = _load_jsonl(evidence_dir / "long_horizon_trace.jsonl")
    behavior = _behavior(scenario_id, family=family)
    resolved_conflict_side = (
        str(conflict_side)
        if conflict_side is not None
        else (
            "left"
            if scenario_id == "cut_in_left"
            else "right"
            if scenario_id == "cut_in_right"
            else ""
        )
    )
    observer = LongHorizonObserver(
        case_id=(
            str(case_id)
            if case_id is not None
            else f"r2v3-teacher_contract-{scenario_id}"
        ),
        route_context=route_context,
        behavior=behavior,
        conflict_side=resolved_conflict_side,
        conflict_point_s_m=(
            _conflict_point(route_context, rows)
            if behavior is InteractionBehavior.YIELD_WAIT
            else None
        ),
    )
    for row in rows:
        observer.observe(row)
    final_actor_lon = next(
        (
            float(value)
            for row in reversed(rows)
            if (value := row.get("actor_lon_m")) is not None
        ),
        None,
    )
    report = observer.finalize(final_actor_lon_m=final_actor_lon).to_dict()
    _write_json(
        evidence_dir / "long_horizon_report.json",
        report,
        exclusive=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--speed-cap-mps", type=float, default=6.0)
    parser.add_argument("--coarse-route-spacing-m", type=float, default=10.0)
    parser.add_argument(
        "--junction-duration-s",
        type=float,
        default=0.0,
        help=(
            "Optional V4 authoring duration for crossing junction fixtures. "
            "The frozen registry duration is left unchanged; this makes the "
            "physical exit window explicit rather than weakening completion."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("teacher", "learned"),
        default="teacher",
    )
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    if float(args.coarse_route_spacing_m) <= 0.0:
        parser.error("--coarse-route-spacing-m must be positive")
    if float(args.junction_duration_s) < 0.0:
        parser.error("--junction-duration-s must be non-negative")
    if args.mode == "learned" and not args.checkpoint:
        parser.error("--mode learned requires --checkpoint")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    registry = load_scenario_registry(args.registry)
    root = Path(args.evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    included = {str(value) for value in args.include}
    excluded = {str(value) for value in args.exclude}
    rows: list[dict[str, Any]] = []
    for fixture in registry.fixtures:
        scenario_id = fixture.scenario_id
        if included and scenario_id not in included:
            continue
        if scenario_id in excluded:
            continue
        evidence = root / scenario_id
        if evidence.exists():
            raise RuntimeError(
                f"refusing existing case directory; preserve result: {evidence}"
            )
        duration_s = float(fixture.duration_s)
        if (
            float(args.junction_duration_s) > 0.0
            and str(fixture.family).lower() == "crossing"
            and "left_turn_crossing" in scenario_id
        ):
            duration_s = max(duration_s, float(args.junction_duration_s))
        command = [
            sys.executable,
            str(RUNNER),
            "--map",
            fixture.map_name,
            "--duration-s",
            str(duration_s),
            "--sim-dt",
            str(float(fixture.sim_dt_s)),
            "--vla-period-s",
            "0.75",
            "--v-ref",
            str(float(args.speed_cap_mps)),
            "--speed-gain",
            "1.0",
            "--coarse-route-spacing-m",
            str(float(args.coarse_route_spacing_m)),
            "--vla-version",
            "v3",
            "--v3-mode",
            str(args.mode),
            "--v3-scenario-registry",
            str(Path(args.registry)),
            "--v3-scenario-id",
            scenario_id,
            "--v3-seed-id",
            fixture.seed_id,
            "--v3-contract-auto-select",
            "--no-map-restart",
            "--evidence-dir",
            str(evidence),
        ]
        if checkpoint_path is not None:
            command.extend(
                [
                    "--spatial-head-checkpoint",
                    str(checkpoint_path),
                    "--spatial-checkpoint-use",
                    "development_live_smoke",
                ]
            )
        if scenario_id == "overtake_left_rejoin":
            command.extend(["--v3-requested-overtake-side", "LEFT"])
        elif scenario_id == "overtake_right_rejoin":
            command.extend(["--v3-requested-overtake-side", "RIGHT"])
        evidence.mkdir(parents=True, exist_ok=False)
        start = time.time()
        print(f"[long-smoke] {scenario_id} ...", flush=True)
        with (evidence / "runner.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=600.0,
            )
        row: dict[str, Any] = {
            "case_id": scenario_id,
            "runner_exit_code": int(result.returncode),
            "wall_time_s": time.time() - start,
            "completed": False,
            "evidence_dir": str(evidence),
            "duration_s": duration_s,
        }
        if result.returncode == 0:
            try:
                report = _validate_case(scenario_id, evidence)
                row["completed"] = bool(report["completed"])
                row["reason_codes"] = list(report["reason_codes"])
                row["route_completion"] = dict(report["route_completion"])
                row["interaction_completion"] = dict(
                    report["interaction_completion"]
                )
            except Exception as exc:  # noqa: BLE001
                row["validation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["reason_codes"] = ["RUNNER_FAILED"]
        rows.append(row)
        _write_json(
            root / "campaign_progress.json",
            {
                "schema_version": "safedrive.r2_v3.long_smoke_progress.v1",
                "registry_hash": registry.compute_registry_sha256(),
                "rows": rows,
            },
        )
        print(
            f"  {'PASS' if row['completed'] else 'FAIL'} "
            f"exit={row['runner_exit_code']} "
            f"reasons={row.get('reason_codes') or row.get('validation_error')}",
            flush=True,
        )
    report = {
        "schema_version": "safedrive.r2_v3.long_smoke_campaign.v1",
        "registry_hash": registry.compute_registry_sha256(),
        "mode": str(args.mode),
        "checkpoint_sha256": (
            None
            if checkpoint_path is None
            else _sha256(checkpoint_path)
        ),
        "junction_duration_override_s": (
            None if float(args.junction_duration_s) <= 0.0 else float(args.junction_duration_s)
        ),
        "continue_policy": "continue_all",
        "included": sorted(included),
        "excluded": sorted(excluded),
        "n_run": len(rows),
        "n_pass": sum(bool(row["completed"]) for row in rows),
        "n_fail": sum(not bool(row["completed"]) for row in rows),
        "completed": sum(bool(row["completed"]) for row in rows),
        "rows": rows,
    }
    report["passed"] = bool(
        len(rows) == 16 and int(report["completed"]) == 16
    )
    _write_json(root / "campaign_report.json", report, exclusive=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["n_fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
