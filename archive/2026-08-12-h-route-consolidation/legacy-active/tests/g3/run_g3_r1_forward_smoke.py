#!/usr/bin/env python3
"""R1-C: one real SimLingo CUDA forward → real K2 bundle (no CARLA outcome claims)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "simlingo-main"))

from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.k2_builder import GUARD_OK, load_k2_config  # noqa: E402
from driving_vla.model.lineage import build_simlingo_manifest, write_manifest  # noqa: E402
from driving_vla.model.neural_policy import NeuralV1Policy  # noqa: E402
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 real forward K2 smoke")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "docs/runtime-evidence/r1-real-k2/forward",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence_dir = args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "status": "PLANNED",
        "branch_type": "longitudinal_temporal",
        "probability_source": "fixed_equal_prior_unscaled",
        "not_learned_k2": True,
    }

    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            report["status"] = "SKIPPED"
            report["error"] = "CUDA not available"
            _write_json(evidence_dir / "real_forward_report.json", report)
            print(json.dumps(report, indent=2))
            return 2

        runtime = SimLingoNeuralRuntime(device=args.device)
        load = runtime.load()
        if not load.ok:
            report["status"] = "FAILED"
            report["error"] = f"load_failed: {load.error}"
            _write_json(evidence_dir / "real_forward_report.json", report)
            print(json.dumps(report, indent=2))
            return 1

        # Lineage
        ckpt = Path(runtime.ckpt_path)
        manifest = build_simlingo_manifest(ckpt=ckpt, code_root=ROOT / "simlingo-main")
        write_manifest(evidence_dir / "lineage_manifest.json", manifest)
        # Also top-level r1 evidence
        parent = evidence_dir.parent
        write_manifest(parent / "lineage_manifest.json", manifest)

        policy = NeuralV1Policy(runtime=runtime, keep_on_gpu=True)
        policy.ensure_loaded()

        # Synthetic official-ish observation (not CARLA outcome)
        image = np.random.randint(0, 255, size=(512, 1024, 3), dtype=np.uint8)
        obs = ObservationBundle(
            run_id="r1_forward_smoke",
            frame_id="forward-0",
            scenario_id="synthetic_moving",
            simulation_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=4.0,
            route_xy=tuple((float(i), 0.0) for i in range(0, 80)),
            front_rgb=image,
            meta={
                "official_contract": True,
                "image_layout": "rgb",
                "target_ego_1": (8.0, 0.0),
                "target_ego_2": (15.0, 0.0),
                "command_text": None,
            },
        )

        t0 = time.perf_counter()
        bundle = policy.predict_bundle(obs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Second call must not be required; count is per-call 1
        forward_count = int(policy.last_forward_count)
        d = bundle.diagnostics
        peak = float(policy.last_peak_vram_mb)

        ok = (
            forward_count == 1
            and len(bundle.candidates) == 2
            and bundle.guard_status == GUARD_OK
            and not any(
                (not np.isfinite(v))
                for c in bundle.candidates
                for row in c.points_xy_yaw_v_a_kappa
                for v in row
            )
        )
        # Prefer a moving eligible non-collapse sample; if model outputs stop, record limit
        eligible_non_collapse = bool(
            d.selection_space_eligible and d.collapse_reason != "NUMERIC_COLLAPSE"
        )
        if d.selection_space_eligible and d.collapse_reason == "NUMERIC_COLLAPSE":
            ok = False

        cfg = load_k2_config()
        report.update(
            {
                "status": "PASS" if ok else "FAIL",
                "forward_count": forward_count,
                "latency_ms": latency_ms,
                "peak_vram_mb": peak,
                "candidate_count": len(bundle.candidates),
                "candidate_ids": list(bundle.candidate_ids()),
                "guard_status": bundle.guard_status,
                "guard_reasons": list(bundle.guard_reasons),
                "collapse_reason": d.collapse_reason,
                "collapsed": d.collapsed,
                "selection_space_eligible": d.selection_space_eligible,
                "eligible_non_collapse": eligible_non_collapse,
                "max_position_separation_m": d.max_position_separation_m,
                "mean_speed_gap_mps": d.mean_speed_gap_mps,
                "final_progress_gap_m": d.final_progress_gap_m,
                "native_path_hash": bundle.native_path_hash,
                "config_hash": bundle.config_hash,
                "retimer_version": bundle.retimer_version,
                "branch_type": bundle.branch_type,
                "top1_index": bundle.top1_index,
                "probability_source": bundle.probability_source,
                "probability_margin": bundle.probability_margin,
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "model_id": bundle.model_id,
                "k2_config_version": cfg.retimer_version,
            }
        )
        _write_json(evidence_dir / "real_forward_report.json", report)
        _write_json(parent / "real_forward_report.json", report)
        # config snapshot
        import shutil

        src_cfg = ROOT / "safedrive_foundry/config/vla/k2_v1.toml"
        if src_cfg.is_file():
            shutil.copy2(src_cfg, parent / "config_snapshot.toml")
            shutil.copy2(src_cfg, evidence_dir / "config_snapshot.toml")

        print(json.dumps(report, indent=2))
        if not ok:
            return 1
        if not eligible_non_collapse:
            # Still PASS if guard OK but mark limit — R1 wants at least one eligible
            # non-collapse; synthetic image may produce stop. Try second obs with assist.
            obs2 = ObservationBundle(
                run_id="r1_forward_smoke",
                frame_id="forward-1",
                scenario_id="synthetic_moving_assist",
                simulation_time_s=2.0,
                ego_x=0.0,
                ego_y=0.0,
                ego_yaw=0.0,
                ego_v=5.0,
                route_xy=tuple((float(i), 0.0) for i in range(0, 80)),
                front_rgb=image,
                meta={
                    "official_contract": True,
                    "image_layout": "rgb",
                    "target_ego_1": (12.0, 0.0),
                    "target_ego_2": (24.0, 0.0),
                    "vla_input_speed_mps": 5.0,
                    "command_text": None,
                },
            )
            b2 = policy.predict_bundle(obs2)
            d2 = b2.diagnostics
            report["second_sample"] = {
                "forward_count": policy.last_forward_count,
                "eligible": d2.selection_space_eligible,
                "collapse_reason": d2.collapse_reason,
                "max_position_separation_m": d2.max_position_separation_m,
                "mean_speed_gap_mps": d2.mean_speed_gap_mps,
                "guard_status": b2.guard_status,
            }
            if (
                d2.selection_space_eligible
                and d2.collapse_reason != "NUMERIC_COLLAPSE"
                and b2.guard_status == GUARD_OK
            ):
                report["eligible_non_collapse"] = True
                report["status"] = "PASS"
            else:
                report["status"] = "FAIL"
                report["error"] = "no_eligible_non_collapse_sample"
                _write_json(evidence_dir / "real_forward_report.json", report)
                _write_json(parent / "real_forward_report.json", report)
                print(json.dumps(report, indent=2))
                return 1
            _write_json(evidence_dir / "real_forward_report.json", report)
            _write_json(parent / "real_forward_report.json", report)
            print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        _write_json(evidence_dir / "real_forward_report.json", report)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
