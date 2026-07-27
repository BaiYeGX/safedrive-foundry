#!/usr/bin/env python3
"""R2-F offline selection-space diagnostic (read-only on longitudinal R2 Evidence).

Does not re-run CARLA, rewrite pilot attempts, or alter Oracle thresholds.
Writes only to docs/runtime-evidence/r2x-selection-space-diagnostic/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

DEFAULT_PILOT = ROOT / "docs" / "runtime-evidence" / "r2-g4a-paired-pilot"
DEFAULT_OUT = ROOT / "docs" / "runtime-evidence" / "r2x-selection-space-diagnostic"

BOTTLENECKS = (
    "PROPOSAL_SAME_PATH",
    "PROPOSAL_TEMPORAL_ONLY",
    "SPEED_PLANNER_COMPRESSION",
    "PATH_MANAGER_MERGE",
    "MPC_CONTROL_COMPRESSION",
    "SCENE_INSENSITIVE",
    "ORACLE_DEADBAND_ONLY",
    "INCOMPLETE_EVIDENCE",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _xy_seps(pts0: list, pts1: list) -> dict[str, float | None]:
    if not pts0 or not pts1 or len(pts0) != len(pts1):
        return {"max_m": None, "mean_m": None, "final_m": None, "n": 0}
    seps = [
        math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
        for a, b in zip(pts0, pts1)
    ]
    return {
        "max_m": max(seps),
        "mean_m": sum(seps) / len(seps),
        "final_m": seps[-1],
        "n": len(seps),
    }


def _speed_gap(pts0: list, pts1: list) -> dict[str, float | None]:
    if not pts0 or not pts1 or len(pts0) != len(pts1):
        return {"mean_mps": None, "max_mps": None}
    gaps = [abs(float(a[3]) - float(b[3])) for a, b in zip(pts0, pts1)]
    return {"mean_mps": sum(gaps) / len(gaps), "max_mps": max(gaps)}


def _control_gap(ticks0: list, ticks1: list) -> dict[str, Any]:
    n = min(len(ticks0), len(ticks1))
    if n == 0:
        return {"n": 0, "steer_mae": None, "throttle_mae": None, "brake_mae": None, "first_div_tick": None}
    steers, thros, brakes = [], [], []
    first = None
    for i in range(n):
        a, b = ticks0[i], ticks1[i]
        ds = abs(float(a.get("steer", 0)) - float(b.get("steer", 0)))
        dt = abs(float(a.get("throttle", 0)) - float(b.get("throttle", 0)))
        db = abs(float(a.get("brake", 0)) - float(b.get("brake", 0)))
        steers.append(ds)
        thros.append(dt)
        brakes.append(db)
        if first is None and (ds > 1e-3 or dt > 1e-3 or db > 1e-3):
            first = int(a.get("tick", i))
    return {
        "n": n,
        "steer_mae": sum(steers) / n,
        "throttle_mae": sum(thros) / n,
        "brake_mae": sum(brakes) / n,
        "steer_max": max(steers),
        "throttle_max": max(thros),
        "brake_max": max(brakes),
        "first_div_tick": first,
    }


def _classify(
    *,
    same_path: bool | None,
    sep_max: float | None,
    speed_mean: float | None,
    ctrl: Mapping[str, Any],
    outcome_delta: Mapping[str, Any] | None,
    incomplete: bool,
) -> str:
    if incomplete:
        return "INCOMPLETE_EVIDENCE"
    if same_path is True:
        # Same execution spatial path: pure longitudinal proposal
        if (sep_max or 0) < 0.05 and (speed_mean or 0) < 0.05:
            return "PROPOSAL_SAME_PATH"
        return "PROPOSAL_TEMPORAL_ONLY"
    if same_path is False and (sep_max or 0) >= 0.5:
        # Spatial proposal differs — if control still tiny, executor/MPC compression
        if ctrl.get("n") and (ctrl.get("steer_mae") or 0) < 1e-3 and (ctrl.get("throttle_mae") or 0) < 1e-3:
            return "MPC_CONTROL_COMPRESSION"
    if ctrl.get("n") and (ctrl.get("steer_mae") or 0) < 1e-3 and (ctrl.get("throttle_mae") or 0) < 0.02:
        if (speed_mean or 0) >= 0.25:
            return "SPEED_PLANNER_COMPRESSION"
        return "MPC_CONTROL_COMPRESSION"
    # Closed-loop outcome: if continuous deltas exist but label TIE
    if outcome_delta:
        # presence of small continuous gaps under oracle deadband
        return "ORACLE_DEADBAND_ONLY"
    return "SCENE_INSENSITIVE"


def diagnose_pair(
    pairs_root: Path,
    *,
    index: int,
    pair_id: str,
    attempt_id: int,
    scenario_id: str,
    seed_id: str,
    family: str,
    pair_label: str | None,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    adir = pairs_root / pair_id / f"attempt_{int(attempt_id)}"
    missing: list[str] = []
    bundle = _read_json(adir / "anchor" / "anchor_bundle.json")
    if bundle is None:
        missing.append("anchor_bundle")
    b0s = _read_json(adir / "branch-0" / "branch_summary.json")
    b1s = _read_json(adir / "branch-1" / "branch_summary.json")
    c0 = _read_json(adir / "branch-0" / "control_seq.json")
    c1 = _read_json(adir / "branch-1" / "control_seq.json")
    oracle = _read_json(adir / "pair_oracle.json")

    proposal: dict[str, Any] = {"status": "ok"}
    same_path = None
    sep_max = None
    speed_mean = None
    if bundle:
        cands = bundle.get("candidates") or []
        if len(cands) >= 2:
            p0 = cands[0].get("spatial_path_xy") or []
            p1 = cands[1].get("spatial_path_xy") or []
            same_path = list(p0) == list(p1)
            pts0 = cands[0].get("points_xy_yaw_v_a_kappa") or []
            pts1 = cands[1].get("points_xy_yaw_v_a_kappa") or []
            seps = _xy_seps(pts0, pts1)
            speeds = _speed_gap(pts0, pts1)
            sep_max = seps["max_m"]
            speed_mean = speeds["mean_mps"]
            h0 = cands[0].get("native_path_hash")
            h1 = cands[1].get("native_path_hash")
            proposal = {
                "status": "ok",
                "same_spatial_path_xy": same_path,
                "native_path_hash_equal": h0 == h1,
                "native_path_hash": bundle.get("native_path_hash"),
                "timed_xy_separation": seps,
                "speed_gap": speeds,
                "diagnostics": bundle.get("diagnostics"),
                "guard_status": bundle.get("guard_status"),
                "branch_type": bundle.get("branch_type"),
                "selection_space_eligible": (bundle.get("diagnostics") or {}).get(
                    "selection_space_eligible"
                ),
                "path_speed_cap_active": (bundle.get("diagnostics") or {}).get(
                    "path_speed_cap_active"
                ),
            }
        else:
            missing.append("candidates")
            proposal = {"status": "missing_candidates"}
    else:
        proposal = {"status": "missing"}

    executor: dict[str, Any] = {"status": "partial"}
    # PathManager hashes not always persisted; use source_id / spatial equality proxy
    if b0s and b1s:
        executor = {
            "status": "ok",
            "branch0_candidate": b0s.get("candidate_id"),
            "branch1_candidate": b1s.get("candidate_id"),
            "note": "PathManager committed path hash not stored in R2 Evidence; "
            "proposal same_spatial_path_xy used as proxy for merge risk",
            "proposal_same_path_implies_pm_same_geometry": same_path,
        }
    else:
        missing.append("branch_summary")

    control: dict[str, Any] = {"status": "missing"}
    ctrl_gap: dict[str, Any] = {"n": 0}
    if c0 and c1:
        t0 = c0.get("ticks") or []
        t1 = c1.get("ticks") or []
        ctrl_gap = _control_gap(t0, t1)
        control = {"status": "ok", **ctrl_gap}
    else:
        missing.append("control_seq")

    outcome: dict[str, Any] = {"status": "partial"}
    outcome_delta = None
    if oracle:
        outcome_delta = oracle.get("outcome_delta") or row.get("outcome_delta")
        outcome = {
            "status": "ok",
            "pair_label": oracle.get("pair_label") or pair_label,
            "oracle_decision_level": oracle.get("oracle_decision_level"),
            "outcome_delta": outcome_delta,
            "both_bad": oracle.get("both_bad"),
        }
    elif b0s and b1s:
        m0 = b0s.get("metrics") or {}
        m1 = b1s.get("metrics") or {}
        outcome = {
            "status": "metrics_only",
            "pair_label": pair_label,
            "branch0": {
                "progress": m0.get("route_progress_delta_m"),
                "ttc": m0.get("minimum_ttc_s"),
                "clearance": m0.get("minimum_actor_clearance_m"),
                "jerk_p95": m0.get("jerk_abs_p95"),
                "offroad": m0.get("offroad_fraction"),
                "collision": m0.get("collision_episode_count"),
            },
            "branch1": {
                "progress": m1.get("route_progress_delta_m"),
                "ttc": m1.get("minimum_ttc_s"),
                "clearance": m1.get("minimum_actor_clearance_m"),
                "jerk_p95": m1.get("jerk_abs_p95"),
                "offroad": m1.get("offroad_fraction"),
                "collision": m1.get("collision_episode_count"),
            },
        }
    else:
        missing.append("oracle_or_metrics")

    incomplete = bool(missing)
    bottleneck = _classify(
        same_path=same_path,
        sep_max=float(sep_max) if sep_max is not None else None,
        speed_mean=float(speed_mean) if speed_mean is not None else None,
        ctrl=ctrl_gap,
        outcome_delta=outcome_delta if isinstance(outcome_delta, dict) else None,
        incomplete=incomplete,
    )

    return {
        "index": index,
        "pair_id": pair_id,
        "attempt_id": int(attempt_id),
        "scenario_id": scenario_id,
        "seed_id": seed_id,
        "family": family,
        "pair_label": pair_label,
        "attempt_dir_rel": f"{pair_id}/attempt_{int(attempt_id)}",
        "missing_fields": missing,
        "proposal": proposal,
        "executor": executor,
        "control": control,
        "outcome": outcome,
        "primary_bottleneck": bottleneck,
    }


def path_manager_threshold_hints() -> dict[str, Any]:
    """Read PathManager defaults used by paired live (no CARLA)."""
    try:
        from driving_vla.runtime.path_manager import PathManagerConfig

        cfg = PathManagerConfig()
        return {
            "max_switch_lateral_5m": float(cfg.max_switch_lateral_5m),
            "max_switch_heading_5m_deg": float(cfg.max_switch_heading_5m_deg),
            "max_abs_curvature": float(cfg.max_abs_curvature),
            "hard_max_abs_curvature": float(cfg.hard_max_abs_curvature),
            "min_path_length_m": float(cfg.min_path_length_m),
            "min_forward_ratio": float(cfg.min_forward_ratio),
            "note": "R2-G should freeze spatial residual envelopes within these limits",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run_diagnostic(
    *,
    pilot_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    pilot_dir = Path(pilot_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man = _read_json(pilot_dir / "run_set_manifest.json")
    rep = _read_json(pilot_dir / "run_set_report.json")
    if not man or not rep:
        raise SystemExit(f"missing run_set_manifest/report under {pilot_dir}")
    pairs_root = pilot_dir / "pairs"
    slots = man.get("pairs") or []
    rows = rep.get("pair_results") or []
    if len(slots) != 12 or len(rows) != 12:
        raise SystemExit(f"expected 12 slots/results, got {len(slots)}/{len(rows)}")

    diagnostics: list[dict[str, Any]] = []
    for slot, row in zip(slots, rows):
        comparable = bool(row.get("comparable"))
        status = str(row.get("status") or "")
        if not comparable and status not in ("COMPLETED", "COMPARABLE"):
            continue
        if status == "FAILED":
            continue
        diagnostics.append(
            diagnose_pair(
                pairs_root,
                index=int(slot.get("index", row.get("index", -1))),
                pair_id=str(row.get("pair_id") or slot.get("pair_id")),
                attempt_id=int(row.get("attempt_id", slot.get("planned_attempt_id", 0))),
                scenario_id=str(row.get("scenario_id") or slot.get("scenario_id")),
                seed_id=str(row.get("seed_id") or slot.get("seed_id")),
                family=str(row.get("family") or slot.get("family") or ""),
                pair_label=row.get("pair_label"),
                row=row,
            )
        )

    # counts
    bottleneck_counts: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for d in diagnostics:
        b = str(d["primary_bottleneck"])
        bottleneck_counts[b] = bottleneck_counts.get(b, 0) + 1
        fam = d.get("family") or "unknown"
        by_family.setdefault(fam, {})
        by_family[fam][b] = by_family[fam].get(b, 0) + 1

    same_path_n = sum(
        1
        for d in diagnostics
        if (d.get("proposal") or {}).get("same_spatial_path_xy") is True
    )
    pm_hints = path_manager_threshold_hints()

    aggregate = {
        "schema_version": "safedrive.r2x.selection_space_diagnostic.v1",
        "n_comparable_diagnosed": len(diagnostics),
        "n_expected": 11,
        "same_spatial_path_count": same_path_n,
        "bottleneck_counts": bottleneck_counts,
        "bottleneck_by_family": by_family,
        "path_manager_threshold_hints": pm_hints,
        "recommended_v2_envelopes": {
            "max_lateral_residual_m": min(
                1.0, float(pm_hints.get("max_switch_lateral_5m") or 1.0)
            ),
            "first_step_position_residual_m": 0.05,
            "near_field_0_5s_inter_candidate_lat_m": 0.20,
            "ambiguity_min_spatial_sep_m": 0.50,
            "source": "R2-F diagnostic + PathManagerConfig defaults; freeze in R2-G TOML",
        },
        "longitudinal_r2_conclusion_preserved": {
            "r2_status": "COMPLETED_WITH_LIMITS",
            "pilot_label": "NO_SELECTION_SPACE",
            "note": "R2-X must not rewrite this conclusion",
        },
        "primary_finding": (
            "PROPOSAL_TEMPORAL_ONLY_DOMINANT"
            if bottleneck_counts.get("PROPOSAL_TEMPORAL_ONLY", 0) >= len(diagnostics) // 2
            else "MIXED"
        ),
    }

    manifest = {
        "schema_version": "safedrive.r2x.selection_space_diagnostic.manifest.v1",
        "source_pilot": str(pilot_dir.as_posix()),
        "source_run_set_manifest_hash": man.get("manifest_content_hash"),
        "n_pairs_diagnosed": len(diagnostics),
        "read_only_source": True,
        "wrote_to": str(out_dir.as_posix()),
    }

    # write outputs
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "pair_diagnostics.jsonl").open("w", encoding="utf-8") as fh:
        for d in diagnostics:
            fh.write(json.dumps(d, sort_keys=True, default=str) + "\n")
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# R2-F Selection-space diagnostic",
        "",
        f"- Source pilot: `{pilot_dir.as_posix()}`",
        f"- Diagnosed comparable pairs: **{len(diagnostics)}** (expected 11)",
        f"- Same spatial_path_xy count: **{same_path_n}**",
        f"- Primary finding: **{aggregate['primary_finding']}**",
        "",
        "## Bottleneck counts",
        "",
    ]
    for k, v in sorted(bottleneck_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## By family",
        "",
        "```json",
        json.dumps(by_family, indent=2),
        "```",
        "",
        "## PathManager hints (for R2-G freeze)",
        "",
        "```json",
        json.dumps(pm_hints, indent=2),
        "```",
        "",
        "## Recommended V2 envelopes (proposal → freeze in TOML at R2-G)",
        "",
        "```json",
        json.dumps(aggregate["recommended_v2_envelopes"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "Longitudinal K2 uses **identical** `spatial_path_xy` for both candidates",
        "(`same_spatial_path_xy=true`). Timed T10 points can separate along-path",
        "(retiming) while execution geometry stays one polyline. Closed-loop pilot",
        "labels remain TIE → bottleneck is **proposal temporal-only / same path**,",
        "not a bad Oracle alone. R2-X must introduce candidate-specific geometry",
        "under Guard V2; do not shrink Oracle deadband to 'fix' this.",
        "",
        "Longitudinal R2 conclusion `NO_SELECTION_SPACE` is **preserved**.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"manifest": manifest, "aggregate": aggregate, "n": len(diagnostics)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    result = run_diagnostic(pilot_dir=args.pilot_dir, out_dir=args.out_dir)
    print(json.dumps({"status": "OK", **{k: result[k] for k in ("n",)}}, indent=2))
    print(json.dumps(result["aggregate"]["bottleneck_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
