#!/usr/bin/env python3
"""R2-X1 offline executor contract probe (no CARLA).

Answers: can two *legal* spatial paths produce different PathManager commits,
MPC reference paths, and control sequences?

Does NOT answer: whether the learned head is defensive.

Lineage is always ``contract_probe`` — never counts toward model quality / R2-K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.k2_spatial_artifact import (  # noqa: E402
    artifact_from_bundle_v2,
    bundle_from_artifact_v2,
    make_dummy_observation_fingerprint,
)
from driving_vla.evaluation.paired_contract import content_hash  # noqa: E402
from driving_vla.model.k2_spatial_builder import (  # noqa: E402
    build_spatial_k2_bundle_from_residuals,
    synthetic_diverse_residuals,
)
from driving_vla.model.k2_spatial_guard import attach_spatial_guard  # noqa: E402
from driving_vla.model.k2_spatial_types import stable_hash_xy  # noqa: E402
from driving_vla.runtime.k2_execution import apply_k2_to_executors, select_k2_spatial  # noqa: E402
from driving_vla.runtime.path_manager import (  # noqa: E402
    EgoPose,
    PathManagerConfig,
    VLAPathManager,
)
from driving_vla.runtime.vla_mpc_tracker import ConstrainedVLAMPC, VLAMPCConfig  # noqa: E402
from driving_vla.runtime.vla_speed_planner import VLASpeedPlanner  # noqa: E402

OUT = ROOT / "docs/runtime-evidence/r2x-executor-probe"

# Pre-freeze "meaningful control difference" thresholds (offline, not Oracle).
# Calibrated against R2-F engineering-identity (steer MAE ~1e-7).
STEER_MAE_MEANINGFUL = 1e-3  # rad
ACCEL_MAE_MEANINGFUL = 1e-2  # m/s^2
PATH_HASH_MUST_DIFFER = True


def _hash_array(arr: np.ndarray) -> str:
    payload = arr.astype(np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _path_ref_hash(committed) -> str:
    if committed is None:
        return "none"
    xy = np.stack([committed.x, committed.y], axis=-1)
    return _hash_array(xy)


def _run_branch(
    *,
    bundle,
    force_index: int,
    ego_xy: tuple[float, float],
    ego_v: float,
    n_ticks: int = 50,
    dt_s: float = 0.05,
) -> dict[str, Any]:
    sel = select_k2_spatial(bundle, mode="force", force_index=force_index)
    pm = VLAPathManager(PathManagerConfig(max_switch_lateral_5m=1.0))
    sp = VLASpeedPlanner()
    mpc = ConstrainedVLAMPC(VLAMPCConfig())
    ego = EgoPose(float(ego_xy[0]), float(ego_xy[1]), 0.0, float(ego_v))
    applied = apply_k2_to_executors(
        sel,
        speed_planner=sp,
        path_manager=pm,
        ego=ego,
        stamp_s=0.0,
        frame_id=f"probe-f{force_index}",
        dt_s=dt_s,
        ego_speed_mps=ego_v,
        nav_target_map_xy=(float(ego_xy[0]) + 40.0, float(ego_xy[1])),
    )
    raw = applied.path_update.raw
    committed = applied.path_update.committed
    raw_hash = _path_ref_hash(raw) if raw is not None else "none"
    committed_hash = _path_ref_hash(committed)
    proposal_hash = sel.execution_spec.spatial_path_hash

    controls: list[dict[str, float]] = []
    # Open-loop bicycle roll for N ticks on the committed path
    x, y, yaw, v = ego.x, ego.y, ego.yaw, ego.speed_mps
    steer = 0.0
    for i in range(n_ticks):
        pose = EgoPose(x, y, yaw, v)
        if pm.committed is None:
            break
        cmd = mpc.step(pm.committed, pose, measured_steer_rad=steer, now_s=float(i) * dt_s)
        steer = float(cmd.steer_rad)
        accel = float(cmd.accel_mps2)
        throttle = max(0.0, min(1.0, accel / 2.5))
        brake = max(0.0, min(1.0, -accel / 3.0))
        controls.append(
            {
                "t": float(i) * dt_s,
                "steer_rad": steer,
                "accel_mps2": accel,
                "throttle": throttle,
                "brake": brake,
                "mode": str(cmd.mode),
                "solver_status": str(cmd.solver_status),
            }
        )
        # integrate bicycle
        x = x + v * math.cos(yaw) * dt_s
        y = y + v * math.sin(yaw) * dt_s
        yaw = yaw + v / 2.7 * math.tan(steer) * dt_s
        v = max(0.0, v + accel * dt_s)

    ctrl_hash = content_hash(controls, nibble=16)
    return {
        "force_index": force_index,
        "candidate_id": sel.candidate_id,
        "proposal_path_hash": proposal_hash,
        "raw_path_hash": raw_hash,
        "committed_path_hash": committed_hash,
        "control_seq_hash": ctrl_hash,
        "n_controls": len(controls),
        "controls": controls,
        "accepted": bool(applied.path_update.accepted),
        "source_id": applied.source_id,
    }


def _mae(a: list[dict], b: list[dict], key: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(abs(float(a[i][key]) - float(b[i][key])) for i in range(n)) / n


def _first_div(a: list[dict], b: list[dict], *, steer_eps: float, accel_eps: float) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if abs(a[i]["steer_rad"] - b[i]["steer_rad"]) > steer_eps:
            return i
        if abs(a[i]["accel_mps2"] - b[i]["accel_mps2"]) > accel_eps:
            return i
    return None


def _temporal_only_baseline(native: list[tuple[float, float]], ego_v: float) -> dict[str, Any]:
    """Same path, different speed scale only — R2-F-like collapse baseline."""
    n = len(native)
    nom = {
        "raw_delta_s": [0.8] * n,
        "raw_d": [0.0] * n,
        "speed_scale": 1.0,
        "head_lineage": "contract_probe",
    }
    alt = {
        "raw_delta_s": [0.8] * n,
        "raw_d": [0.0] * n,
        "speed_scale": 0.65,
        "head_lineage": "contract_probe",
    }
    b = build_spatial_k2_bundle_from_residuals(
        native_path_xy=native,
        ego_xy=native[0],
        ego_v=ego_v,
        base_speed_mps=6.0,
        residual_nominal=nom,
        residual_defensive=alt,
        observation_identity={"probe": "temporal_baseline"},
        backbone_forward_id="probe-temporal",
        defensive_available=True,
    )
    # diversity not required for temporal-only
    g = attach_spatial_guard(
        replace(b, set_diagnostics={"eligible_for_diversity": False}),
        require_diversity_if_eligible=False,
    )
    r0 = _run_branch(bundle=g, force_index=0, ego_xy=native[0], ego_v=ego_v)
    r1 = _run_branch(bundle=g, force_index=1, ego_xy=native[0], ego_v=ego_v)
    return {
        "proposal_paths_differ": r0["proposal_path_hash"] != r1["proposal_path_hash"],
        "committed_paths_differ": r0["committed_path_hash"] != r1["committed_path_hash"],
        "control_seq_differ": r0["control_seq_hash"] != r1["control_seq_hash"],
        "steer_mae": _mae(r0["controls"], r1["controls"], "steer_rad"),
        "accel_mae": _mae(r0["controls"], r1["controls"], "accel_mps2"),
        "first_div_tick": _first_div(
            r0["controls"], r1["controls"], steer_eps=1e-6, accel_eps=1e-4
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ticks", type=int, default=50)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    native = [(float(i) * 1.2, 0.0) for i in range(20)]
    ego_v = 5.0
    nom, alt = synthetic_diverse_residuals(20, lateral_sign=1.0, lineage="contract_probe")
    alt["raw_d"] = [min(2.5, 0.4 * i) for i in range(20)]

    bundle = build_spatial_k2_bundle_from_residuals(
        native_path_xy=native,
        ego_xy=native[0],
        ego_v=ego_v,
        base_speed_mps=6.0,
        residual_nominal=nom,
        residual_defensive=alt,
        observation_identity={"probe": "spatial_contract"},
        backbone_forward_id="probe-spatial-v2",
        defensive_available=True,
    )
    bundle = replace(bundle, set_diagnostics={"eligible_for_diversity": True})
    guarded = attach_spatial_guard(bundle, require_diversity_if_eligible=True)
    if guarded.guard_status != "OK":
        report = {
            "schema_version": "safedrive.r2x.executor_probe.v1",
            "status": "GUARD_REJECT",
            "guard_reasons": list(guarded.guard_reasons),
            "evidence_lineage": "contract_probe",
        }
        (out / "executor_probe_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 2

    # Artifact round-trip (V2 cold rebuild)
    fp = make_dummy_observation_fingerprint(
        k2_bundle_hash=content_hash({"n": guarded.native_path_hash}, nibble=16)
    )
    art = artifact_from_bundle_v2(
        guarded,
        pair_id="r2x-executor-probe",
        scenario_id="contract_probe_spatial",
        seed_id="seed_a",
        anchor_run_id="anchor-probe",
        anchor_carla_frame=0,
        anchor_simulation_time_s=0.0,
        requested_initial_state_hash="0" * 64,
        measured_initial_state_hash="0" * 64,
        observation_fingerprint=fp,
        model_checkpoint_hash="unset",
        executor_config_hash="offline_probe",
        evidence_lineage="contract_probe",
    )
    art_path = out / "k2_anchor_v2.json"
    art_path.write_bytes(art.to_json_bytes())
    cold = bundle_from_artifact_v2(art)
    # re-attach guard status from artifact fields (already OK)
    cold = replace(
        cold,
        guard_status=art.guard_status,
        guard_reasons=art.guard_reasons,
        set_diagnostics=dict(art.set_diagnostics),
    )

    r0 = _run_branch(
        bundle=cold, force_index=0, ego_xy=native[0], ego_v=ego_v, n_ticks=args.n_ticks
    )
    r1 = _run_branch(
        bundle=cold, force_index=1, ego_xy=native[0], ego_v=ego_v, n_ticks=args.n_ticks
    )
    baseline = _temporal_only_baseline(native, ego_v)

    steer_mae = _mae(r0["controls"], r1["controls"], "steer_rad")
    accel_mae = _mae(r0["controls"], r1["controls"], "accel_mps2")
    first_div = _first_div(
        r0["controls"],
        r1["controls"],
        steer_eps=STEER_MAE_MEANINGFUL * 0.5,
        accel_eps=ACCEL_MAE_MEANINGFUL * 0.5,
    )

    proposal_differ = r0["proposal_path_hash"] != r1["proposal_path_hash"]
    committed_differ = r0["committed_path_hash"] != r1["committed_path_hash"]
    control_differ = r0["control_seq_hash"] != r1["control_seq_hash"]
    meaningful = (
        proposal_differ
        and committed_differ
        and (
            steer_mae >= STEER_MAE_MEANINGFUL
            or accel_mae >= ACCEL_MAE_MEANINGFUL
            or first_div is not None
        )
    )

    # strip heavy control arrays from branch summaries in report
    def slim(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "controls"}

    report = {
        "schema_version": "safedrive.r2x.executor_probe.v1",
        "status": "PASS" if meaningful else "FAIL_NO_CONTROL_DIVERGENCE",
        "evidence_lineage": "contract_probe",
        "note": (
            "Proves executor sensitivity to legal dual spatial paths. "
            "Not model quality; not R2-K pilot evidence."
        ),
        "thresholds": {
            "steer_mae_meaningful_rad": STEER_MAE_MEANINGFUL,
            "accel_mae_meaningful_mps2": ACCEL_MAE_MEANINGFUL,
        },
        "spatial_contract_probe": {
            "guard_status": guarded.guard_status,
            "proposal_paths_differ": proposal_differ,
            "raw_paths_differ": r0["raw_path_hash"] != r1["raw_path_hash"],
            "committed_paths_differ": committed_differ,
            "control_seq_differ": control_differ,
            "steer_mae_rad": steer_mae,
            "accel_mae_mps2": accel_mae,
            "first_div_tick": first_div,
            "branch_0": slim(r0),
            "branch_1": slim(r1),
        },
        "temporal_only_baseline": baseline,
        "gates": {
            "proposal_paths_differ": proposal_differ,
            "committed_paths_differ": committed_differ,
            "meaningful_control_diff": meaningful,
            "temporal_baseline_weaker_or_equal": (
                baseline["steer_mae"] <= steer_mae + 1e-9
                or not baseline["committed_paths_differ"]
            ),
        },
        "artifact_path": str(art_path.as_posix()),
        "artifact_content_hash": art.artifact_content_hash(),
        "native_path_hash": stable_hash_xy(native),
    }
    (out / "executor_probe_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    # store control sequences separately
    (out / "control_branch_0.jsonl").write_text(
        "\n".join(json.dumps(c) for c in r0["controls"]) + "\n", encoding="utf-8"
    )
    (out / "control_branch_1.jsonl").write_text(
        "\n".join(json.dumps(c) for c in r1["controls"]) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "spatial_contract_probe"}, indent=2))
    print("spatial:", json.dumps(report["spatial_contract_probe"], indent=2))
    print("temporal_baseline:", json.dumps(baseline, indent=2))
    print("status:", report["status"])
    return 0 if meaningful else 3


if __name__ == "__main__":
    raise SystemExit(main())
