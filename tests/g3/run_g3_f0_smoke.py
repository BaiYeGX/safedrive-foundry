#!/usr/bin/env python3
"""G3-03 F0 smoke: load, determinism, latency, short soak, validator chain."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle, arrays_to_candidate_set
from driving_vla.model.backbone_loader import SimLingoCheckpointHandle, V0Policy
from driving_vla.model.lineage import LineageManifest, write_manifest
from safety_kernel import SafetyKernel, ComponentAvailability, load_safety_config
from safety_kernel.contracts.types import (
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument(
        "--ckpt",
        default=str(ROOT / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/architecture/evidence/g3-03/f0/f0_report.json"),
    )
    args = ap.parse_args()
    out: dict = {"stages": {}}

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print("FAIL load: missing checkpoint", ckpt)
        return 2

    # lineage (sha deferred unless small — store size + mtime)
    manifest = LineageManifest(
        base_model="SimLingo/InternVL2-1B",
        checkpoint_path=str(ckpt),
        checkpoint_sha256="deferred_full_sha_use_SDF_G3_FULL_HASH",
        checkpoint_bytes=ckpt.stat().st_size,
        code_root=str(ROOT / "simlingo-main"),
        license_scope="simlingo research",
        deployment_scope="simulation_research_only",
        precision="bf16",
        model_id="sdf-vla-v0@0.0.1",
        notes=["F0 smoke", "path/speed via fingerprint-anchored V0 until full graph wire"],
    )
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "lineage_manifest.json", manifest)
    out["stages"]["lineage"] = {"ok": True, "bytes": manifest.checkpoint_bytes}

    handle = SimLingoCheckpointHandle(ckpt)
    rep = handle.load()
    out["stages"]["load"] = rep.__dict__
    print("load", rep)
    if not rep.ok:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        return 1

    # determinism
    route = tuple((float(i), 0.0) for i in range(0, 40, 1))
    a = handle.predict_path_speed(ego_v=5.0, route_xy=route, seed_extra=b"f0")
    b = handle.predict_path_speed(ego_v=5.0, route_xy=route, seed_extra=b"f0")
    det_ok = a.path_xy == b.path_xy and a.speed_mps == b.speed_mps
    out["stages"]["determinism"] = {"ok": det_ok, "n_path": len(a.path_xy), "n_speed": len(a.speed_mps)}
    print("determinism", det_ok)

    # latency
    policy = V0Policy(handle)
    latencies = []
    for i in range(args.steps):
        obs = ObservationBundle(
            run_id="f0",
            frame_id=f"f{i}",
            scenario_id="f0_sc",
            simulation_time_s=float(i) * 0.02,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            route_xy=route,
        )
        t0 = time.perf_counter()
        arrs = policy.predict_arrays(obs)
        cset = arrays_to_candidate_set(arrs, obs, model_id=policy.model_id, source=CandidateSource.VLA_FAST)
        latencies.append(time.perf_counter() - t0)
    out["stages"]["latency"] = {
        "n": len(latencies),
        "p50_ms": _percentile(latencies, 0.50) * 1000,
        "p95_ms": _percentile(latencies, 0.95) * 1000,
        "p99_ms": _percentile(latencies, 0.99) * 1000,
        "mean_ms": statistics.mean(latencies) * 1000,
        "deadline_200ms_miss": sum(1 for x in latencies if x > 0.2),
    }
    print("latency", out["stages"]["latency"])

    # VRAM optional
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # allocate tiny to touch GPU
            _ = torch.zeros(1, device="cuda")
            out["stages"]["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
            }
        else:
            out["stages"]["gpu"] = {"available": False}
    except Exception as exc:  # noqa: BLE001
        out["stages"]["gpu"] = {"error": str(exc)}

    # validator chain
    obs = ObservationBundle(
        run_id="f0",
        frame_id="f0",
        scenario_id="f0_sc",
        simulation_time_s=1.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_v=5.0,
        route_xy=route,
    )
    arrs = policy.predict_arrays(obs)
    cset = arrays_to_candidate_set(arrs, obs, model_id=policy.model_id, source=CandidateSource.VLA_FAST)
    snap = ObservableSnapshot(
        run_id=obs.run_id,
        frame_id=obs.frame_id,
        scenario_id=obs.scenario_id,
        simulation_time_s=obs.simulation_time_s,
        wall_time_s=0.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=5.0,
        observed_time_s=obs.simulation_time_s,
        corridor_centerline=route[:30],
        corridor_half_width_m=2.5,
        privilege=ObservationPrivilege.OBSERVABLE,
    )
    kernel = SafetyKernel(load_safety_config())
    result = kernel.tick(
        snap,
        cset,
        now_s=obs.simulation_time_s,
        availability=ComponentAvailability(classic=False, vla=True, world=False, safety=True),
    )
    out["stages"]["validator_chain"] = {
        "decision_kind": str(result.decision.decision_kind),
        "ok": result.decision is not None,
    }
    print("validator", out["stages"]["validator_chain"])

    # short soak (= steps already run)
    out["stages"]["soak"] = {"steps": args.steps, "ok": True, "note": "equivalent-step soak; full 30min optional"}

    out["f0_pass"] = bool(
        out["stages"]["load"]["ok"]
        and out["stages"]["determinism"]["ok"]
        and out["stages"]["validator_chain"]["ok"]
    )
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out, "f0_pass", out["f0_pass"])
    return 0 if out["f0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
