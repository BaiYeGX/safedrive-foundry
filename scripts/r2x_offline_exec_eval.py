#!/usr/bin/env python3
"""R2-I offline execution eval for Spatial K2 V2 (no CARLA).

Metrics use **eligible** denominator (alternative_available=true samples).
pass_spatial requires eligible separation rate gate — never n_sep>0 alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.k2_spatial_builder import (  # noqa: E402
    build_spatial_k2_bundle_from_residuals,
)
from driving_vla.model.k2_spatial_guard import attach_spatial_guard  # noqa: E402
from driving_vla.model.spatial_mode_heads import (  # noqa: E402
    SpatialK2HeadRuntime,
    decoded_peak_lateral_separation,
)
from driving_vla.runtime.k2_execution import apply_k2_to_executors, select_k2_spatial  # noqa: E402
from driving_vla.runtime.path_manager import EgoPose, PathManagerConfig, VLAPathManager  # noqa: E402
from driving_vla.runtime.vla_speed_planner import VLASpeedPlanner  # noqa: E402

DATA_V4 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v4-real/samples.jsonl"
DATA_V3 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v3/samples.jsonl"
DATA_V2 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v2/samples.jsonl"
DATA_V1 = ROOT / "docs/runtime-evidence/r2x-training/dataset-v1/samples.jsonl"
DATA = (
    DATA_V4
    if DATA_V4.is_file()
    else (
        DATA_V3
        if DATA_V3.is_file()
        else (DATA_V2 if DATA_V2.is_file() else DATA_V1)
    )
)
CKPT = ROOT / "docs/runtime-evidence/r2x-training/checkpoints/spatial_heads_last.pt"
OUT = ROOT / "docs/runtime-evidence/r2x-training/offline_exec_report.json"

# Frozen offline gates (pre-outcome). Eligible-denominator only.
# X5F: slightly stricter than bootstrap (avail ≥0.80 when real-feature).
ELIGIBLE_GUARD_PASS_MIN = 0.90
ELIGIBLE_SPATIAL_SEP_MIN = 0.70
AVAIL_RECALL_MIN = 0.80
AVAIL_SPECIFICITY_MIN = 0.80


def _binary_prf(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = evaluate full val split (no silent truncation)",
    )
    ap.add_argument("--split", default="val", help="split_id to evaluate")
    ap.add_argument("--data", type=str, default=str(DATA))
    ap.add_argument("--ckpt", type=str, default=str(CKPT))
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument(
        "--checkpoint-use",
        default="offline_diagnostic",
        help="checkpoint contract use (default offline_diagnostic for trained heads)",
    )
    args = ap.parse_args()
    data_path = Path(args.data)
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    if not data_path.is_file():
        print("missing dataset")
        return 2
    if ckpt_path.is_file():
        from driving_vla.model.checkpoint_contract import (
            CheckpointContractError,
            require_checkpoint_for_use,
        )

        use = str(getattr(args, "checkpoint_use", None) or "offline_diagnostic")
        try:
            require_checkpoint_for_use(ckpt_path, use)
        except CheckpointContractError as exc:
            print(json.dumps({"status": "CHECKPOINT_CONTRACT_REJECT", "error": str(exc)}))
            return 4
    use = str(getattr(args, "checkpoint_use", None) or "offline_diagnostic")
    runtime = SpatialK2HeadRuntime(
        device=args.device,
        checkpoint_path=str(ckpt_path) if ckpt_path.is_file() else None,
        checkpoint_use=use,
        skip_checkpoint_contract=True,  # already validated above
    )
    rows = []
    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s.get("split_id") == args.split:
                rows.append(s)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        print(f"no samples for split={args.split}")
        return 2

    n_ok = 0
    n_sep = 0
    n_eligible = 0
    n_elig_ok = 0
    n_elig_sep = 0
    n_elig_proposal_valid = 0
    y_true_avail: list[bool] = []
    y_pred_avail: list[bool] = []
    details = []
    for s in rows:
        native = [tuple(p) for p in s["native_path_xy"]]
        o0, o1 = runtime.predict_modes(
            native,
            ego_v=float(s["ego_v"]),
            base_speed_mps=float(s["base_speed_mps"]),
            sample=s,
        )
        label_avail = bool(s.get("alternative_available"))
        pred_avail = bool(o1.available)
        residual_peak_sep = decoded_peak_lateral_separation(
            o0.raw_d, o1.raw_d
        )
        proposal_available = residual_peak_sep >= 0.5
        y_true_avail.append(label_avail)
        y_pred_avail.append(pred_avail)

        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=tuple(native[0]),
            ego_v=float(s["ego_v"]),
            base_speed_mps=float(s["base_speed_mps"]),
            residual_nominal={
                "raw_delta_s": o0.raw_delta_s,
                "raw_d": o0.raw_d,
                "speed_scale": o0.speed_scale,
                "head_lineage": o0.head_lineage,
            },
            residual_defensive={
                "raw_delta_s": o1.raw_delta_s,
                "raw_d": o1.raw_d,
                "speed_scale": o1.speed_scale,
                "head_lineage": o1.head_lineage,
            },
            observation_identity={"sample_id": s["sample_id"]},
            backbone_forward_id=f"offline-{s['sample_id']}",
            spatial_head_checkpoint_hash=runtime.spatial_head_checkpoint_hash,
            defensive_available=proposal_available,
            defensive_reason=(
                "PROPOSAL_SPATIALLY_DISTINCT"
                if proposal_available
                else "HEAD_COLLAPSE_SEP"
            ),
            nominal_probability=1.0 - float(o1.avail_prob),
            defensive_probability=float(o1.avail_prob),
            probability_source="learned_mode_confidence_non_blocking",
        )
        from dataclasses import replace

        bundle = replace(
            bundle,
            set_diagnostics={
                **dict(bundle.set_diagnostics),
                "eligible_for_diversity": proposal_available,
                "availability_semantics": "executability_only_v1",
                "learned_defensive_confidence": float(o1.avail_prob),
            },
        )
        guarded = attach_spatial_guard(
            bundle, require_diversity_if_eligible=proposal_available
        )
        ok = guarded.guard_status == "OK"
        proposal_valid = bool(proposal_available and ok)
        path_hashes = []
        for force in (0, 1):
            try:
                sel = select_k2_spatial(
                    guarded, mode="force", force_index=force, require_guard_ok=ok
                )
            except Exception as exc:
                if force == 1 and not proposal_available:
                    # expected fail-closed when candidate 1 unavailable
                    path_hashes.append(f"FORCE1_FAIL_CLOSED:{type(exc).__name__}")
                    continue
                if not ok:
                    break
                path_hashes.append(f"ERR:{type(exc).__name__}")
                continue
            pm = VLAPathManager(PathManagerConfig(max_switch_lateral_5m=1.0))
            sp = VLASpeedPlanner()
            ego = EgoPose(float(native[0][0]), float(native[0][1]), 0.0, float(s["ego_v"]))
            applied = apply_k2_to_executors(
                sel,
                speed_planner=sp,
                path_manager=pm,
                ego=ego,
                stamp_s=0.0,
                frame_id="offline",
                dt_s=0.05,
            )
            path_hashes.append(sel.execution_spec.spatial_path_hash)
            _ = applied
        lat = float((guarded.set_diagnostics or {}).get("max_lateral_separation_m") or 0.0)
        sep_ok = lat >= 0.5
        if sep_ok:
            n_sep += 1
        if ok:
            n_ok += 1
        if label_avail:
            n_eligible += 1
            if ok:
                n_elig_ok += 1
            if sep_ok:
                n_elig_sep += 1
            if proposal_valid:
                n_elig_proposal_valid += 1
        details.append(
            {
                "sample_id": s["sample_id"],
                "label_alternative_available": label_avail,
                "pred_alternative_available": pred_avail,
                "learned_defensive_confidence": float(o1.avail_prob),
                "proposal_available_pre_guard": proposal_available,
                "proposal_valid": proposal_valid,
                "availability_semantics": "executability_only_v1",
                "eligible": label_avail,
                "guard_ok": ok,
                "guard_reasons": list(guarded.guard_reasons),
                "path_hashes": path_hashes,
                "paths_differ": len({h for h in path_hashes if not str(h).startswith(("ERR", "FORCE"))})
                > 1
                if path_hashes
                else False,
                "max_lateral_separation_m": lat,
            }
        )

    # Pipeline contract proof using synthetic diverse residuals (independent of head fit).
    from driving_vla.model.k2_spatial_builder import synthetic_diverse_residuals

    pipe_ok = 0
    for s in rows[:4]:
        native = [tuple(p) for p in s["native_path_xy"]]
        nom, alt = synthetic_diverse_residuals(len(native), lateral_sign=1.0, lineage="contract_probe")
        alt["raw_d"] = [min(2.5, 0.4 * i) for i in range(len(native))]
        b = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=tuple(native[0]),
            ego_v=float(s["ego_v"]),
            base_speed_mps=float(s["base_speed_mps"]),
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={"sample_id": s["sample_id"]},
            backbone_forward_id=f"pipe-{s['sample_id']}",
            defensive_available=True,
        )
        from dataclasses import replace

        b = replace(b, set_diagnostics={"eligible_for_diversity": True})
        g = attach_spatial_guard(b, require_diversity_if_eligible=True)
        if g.guard_status == "OK":
            pipe_ok += 1
            sel0 = select_k2_spatial(g, mode="force", force_index=0)
            sel1 = select_k2_spatial(g, mode="force", force_index=1)
            assert sel0.execution_spec.spatial_path_hash != sel1.execution_spec.spatial_path_hash

    n = max(len(details), 1)
    n_elig = max(n_eligible, 1)
    guard_ok_rate = n_ok / n
    spatial_sep_rate_all = n_sep / n
    eligible_guard_ok_rate = n_elig_ok / n_elig if n_eligible else 0.0
    eligible_spatial_sep_rate = n_elig_sep / n_elig if n_eligible else 0.0
    eligible_proposal_valid_rate = (
        n_elig_proposal_valid / n_elig if n_eligible else 0.0
    )
    avail_stats = _binary_prf(y_true_avail, y_pred_avail)

    pass_quality = eligible_guard_ok_rate >= ELIGIBLE_GUARD_PASS_MIN if n_eligible else False
    # STRICT: eligible separation rate only — never "or n_sep > 0"
    pass_spatial = (
        eligible_spatial_sep_rate >= ELIGIBLE_SPATIAL_SEP_MIN if n_eligible else False
    )
    pass_confidence_diagnostic = (
        avail_stats["recall"] >= AVAIL_RECALL_MIN
        and avail_stats["specificity"] >= AVAIL_SPECIFICITY_MIN
    )
    pass_proposal_validity = (
        eligible_proposal_valid_rate >= ELIGIBLE_SPATIAL_SEP_MIN
        if n_eligible
        else False
    )
    head_ok = pass_quality and pass_spatial and pass_proposal_validity

    report = {
        "schema_version": "safedrive.r2x.offline_exec.v3",
        "split": args.split,
        "n": len(details),
        "n_eligible": n_eligible,
        "n_guard_ok": n_ok,
        "n_lat_sep_ge_0_5": n_sep,
        "n_eligible_guard_ok": n_elig_ok,
        "n_eligible_lat_sep_ge_0_5": n_elig_sep,
        "n_eligible_proposal_valid": n_elig_proposal_valid,
        "guard_ok_rate": guard_ok_rate,
        "spatial_sep_rate": spatial_sep_rate_all,
        "eligible_guard_ok_rate": eligible_guard_ok_rate,
        "eligible_spatial_sep_rate": eligible_spatial_sep_rate,
        "eligible_proposal_valid_rate": eligible_proposal_valid_rate,
        "learned_confidence_diagnostic": avail_stats,
        "availability_semantics": "executability_only_v1",
        "gates": {
            "eligible_guard_pass_min": ELIGIBLE_GUARD_PASS_MIN,
            "eligible_spatial_sep_min": ELIGIBLE_SPATIAL_SEP_MIN,
            "avail_recall_min": AVAIL_RECALL_MIN,
            "avail_specificity_min": AVAIL_SPECIFICITY_MIN,
        },
        "checkpoint": str(ckpt_path.as_posix()) if ckpt_path.is_file() else None,
        "data_path": str(data_path.as_posix()),
        "details": details,
        "pass_quality": pass_quality,
        "pass_spatial": pass_spatial,
        "pass_proposal_validity": pass_proposal_validity,
        "pass_confidence_diagnostic": pass_confidence_diagnostic,
        "pipeline_contract_ok": pipe_ok >= 3,
        "pipeline_contract_n_ok": pipe_ok,
        "pipeline_lineage": "contract_probe",
        "head_status": "OK" if head_ok else "IMPROVE_SPATIAL_K2",
        "note": (
            "v3 metrics: learned confidence is non-blocking; proposal validity "
            "comes from residual diversity + Guard; pipeline contract uses "
            "contract_probe lineage"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "details"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
