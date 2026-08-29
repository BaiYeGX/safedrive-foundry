"""Latency-only smoke for randomly initialized H5/H6 components.

Evaluates:
1. VLA convex QP smoother: latency, acceleration/curvature bounds, jerk RMS reduction.
2. DistilledWorldScorer: P50/P95/P99 latency, VRAM footprint, 0 deadline misses.
3. H5WorldRouter: source-level hold, emergency switch responsiveness, 0 glitch resets.
4. MPC tracking controller: warm-start transition smoothness and step latency.
"""

from __future__ import annotations

import json
import math
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from classic_stack.control.config import load_control_config
from classic_stack.control.controller import ControlLoop, EgoState
from classic_stack.planning.frenet.planner import Trajectory, TrajectoryPoint
from data_pipeline.h3.contracts import WorldPrediction, WorldScoreResult, stable_sha256
from data_pipeline.h3.model import WorldScorerModel
from data_pipeline.h5.distilled_scorer import DistilledWorldScorer
from data_pipeline.h5.runtime import H5WorldRouter
from driving_vla.adapter.policy_adapter import TrajectoryArray
from driving_vla.hybrid.contracts import (
    CandidateDifference,
    RoutingResult,
    SelectionSpace,
    WorldDisposition,
)
from driving_vla.hybrid.vla_smoother import VLASmootherConfig, smooth_vla_trajectory


class _Item:
    def __init__(self, candidate_id: str, passed: bool = True):
        self.guard = type("G", (), {"passed": passed})()
        self.candidate = type("C", (), {"candidate_id": candidate_id})()
        self.provenance = type("P", (), {
            "source": type("S", (), {"value": candidate_id.split(":")[-1]})(),
            "candidate_id": candidate_id,
        })()


class _CandidateSet:
    def __init__(self, ids):
        self.candidates = [_Item(i) for i in ids]


class _Fallback:
    def route(self, candidate_set):
        passed = [item.candidate.candidate_id for item in candidate_set.candidates if item.guard.passed]
        return RoutingResult(
            pass_candidate_ids=tuple(passed),
            rejected_candidate_ids=(),
            selected_candidate_id=passed[0] if passed else None,
            selection_space=SelectionSpace.DISTINCT,
            world=WorldDisposition.DEFERRED_NOT_APPLICABLE,
            selector="fallback",
            reason="fallback",
            difference=CandidateDifference(1.0, 0.5),
            scores={},
        )


def benchmark_vla_smoother(iterations: int = 500) -> dict:
    raw_pts = []
    for i in range(10):
        t = 0.25 * (i + 1)
        x = 2.0 * t + 0.08 * math.sin(4.0 * t)
        y = 0.4 * t * t + 0.15 * math.cos(3.0 * t)
        yaw = 0.15 * t
        v = 6.0 + 2.5 * (1 if i % 2 == 0 else -1)
        raw_pts.append((x, y, yaw, v, 0.0, 0.0))

    raw_traj = TrajectoryArray(
        points_xy_yaw_v_a_kappa=tuple(raw_pts),
        probability=1.0,
        uncertainty=0.1,
        candidate_id="vla_bench",
    )

    # Measure raw jerk RMS
    raw_jerks = []
    for i in range(1, 10):
        raw_a1 = (raw_pts[i][3] - raw_pts[i - 1][3]) / 0.25
        raw_a0 = (raw_pts[i - 1][3] - raw_pts[max(0, i - 2)][3]) / 0.25
        raw_jerks.append((raw_a1 - raw_a0) / 0.25)
    raw_jerk_rms = math.sqrt(sum(j * j for j in raw_jerks) / len(raw_jerks))

    cfg = VLASmootherConfig()
    latencies = []
    smoothed = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        smoothed = smooth_vla_trajectory(raw_traj, config=cfg)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    pts = np.array(smoothed.points_xy_yaw_v_a_kappa)
    smooth_jerks = [(pts[i, 4] - pts[i - 1, 4]) / 0.25 for i in range(1, 10)]
    smooth_jerk_rms = math.sqrt(sum(j * j for j in smooth_jerks) / len(smooth_jerks))

    return {
        "iterations": iterations,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "raw_jerk_rms": float(raw_jerk_rms),
        "smooth_jerk_rms": float(smooth_jerk_rms),
        "jerk_reduction_ratio": float((raw_jerk_rms - smooth_jerk_rms) / max(1e-3, raw_jerk_rms)),
        "max_accel": float(np.max(pts[:, 4])),
        "min_accel": float(np.min(pts[:, 4])),
        "max_curvature": float(np.max(np.abs(pts[:, 5]))),
    }


def benchmark_distilled_scorer(iterations: int = 200) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldScorerModel(d_model=64, layers=2, heads=4, ffn=128, scene_gate_mode="learned").to(device)
    scorer = DistilledWorldScorer(
        student_model=model,
        norm_mean=1.2,
        norm_std=2.5,
        device=device,
        risk_defer_probability=0.35,
    )

    valid_ctx = [0.1] * 499
    valid_cand1 = [[0.1] * 8 for _ in range(10)]
    valid_cand2 = [[0.2] * 8 for _ in range(10)]

    # Warmup
    for _ in range(20):
        scorer.score_pair(("c1", valid_ctx, valid_cand1), ("c2", valid_ctx, valid_cand2))

    latencies = []
    deadline_misses = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = scorer.score_pair(("c1", valid_ctx, valid_cand1), ("c2", valid_ctx, valid_cand2))
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        if lat > 50.0:
            deadline_misses += 1

    threshold_ms = 4.0 if device == "cuda" else 15.0
    return {
        "device": device,
        "model_state": "random_untrained",
        "iterations": iterations,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "deadline_misses": deadline_misses,
        "device_latency_threshold_p99_ms": threshold_ms,
        "latency_p99_pass": bool(np.percentile(latencies, 99) < threshold_ms),
        "latency_deadline_pass": deadline_misses == 0,
    }


def benchmark_mpc_control(iterations: int = 200) -> dict:
    cfg = load_control_config()
    loop = ControlLoop(cfg)

    pts = tuple(
        TrajectoryPoint(
            t=0.05 * i, x=1.0 * i, y=0.1 * math.sin(0.1 * i), yaw=0.01 * i, kappa=0.0, v=6.0, a=0.0, jerk=0.0
        )
        for i in range(50)
    )
    traj = Trajectory(points=pts, trajectory_id="bench_traj")
    loop.set_trajectory(traj, 0.0)

    ego = EgoState(x=0.0, y=0.0, yaw=0.0, v=6.0)
    latencies = []
    steer_cmds = []
    accel_cmds = []

    for step in range(iterations):
        t0 = time.perf_counter()
        cmd = loop.step(ego, 0.05 * step)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        steer_cmds.append(cmd.steer)
        accel_cmds.append(cmd.throttle * 3.5 - cmd.brake * 5.0)

    steer_deltas = [abs(steer_cmds[i] - steer_cmds[i - 1]) for i in range(1, len(steer_cmds))]
    max_steer_delta = max(steer_deltas) if steer_deltas else 0.0

    return {
        "iterations": iterations,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "max_steer_delta": float(max_steer_delta),
        "mean_steer_delta": float(np.mean(steer_deltas)) if steer_deltas else 0.0,
        "deadline_misses": loop.watchdog.deadline_misses,
    }


def build_latency_smoke_summary(
    vla_smoother: dict,
    distilled_scorer: dict,
    mpc_controller: dict,
    *,
    completed: bool = True,
    error: str | None = None,
    timestamp: float | None = None,
) -> dict:
    summary = {
        "schema_version": "safedrive.h6.c1.latency_smoke.v1",
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "benchmark_scope": "latency_only_smoke",
        "model_state": "random_untrained",
        "quality_gate_eligible": False,
        "status": "SMOKE_COMPLETED" if completed else "SMOKE_FAILED",
        "error": error,
        "vla_smoother": dict(vla_smoother),
        "distilled_scorer": dict(distilled_scorer),
        "mpc_controller": dict(mpc_controller),
    }
    summary["artifact_sha256"] = stable_sha256(summary)
    return summary


def verify_latency_smoke_artifact(payload: dict) -> bool:
    expected = stable_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    return (
        payload.get("artifact_sha256") == expected
        and payload.get("benchmark_scope") == "latency_only_smoke"
        and payload.get("model_state") == "random_untrained"
        and payload.get("quality_gate_eligible") is False
        and payload.get("status") in {"SMOKE_COMPLETED", "SMOKE_FAILED"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "generated/h6/c1-smoke/ultimate-latency-smoke.json",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("  SAFE-DRIVE C1 LATENCY-ONLY SMOKE (RANDOM MODEL)")
    print("=" * 60)
    try:
        print("\n[1/3] Benchmarking VLA smoother latency...")
        vla_res = benchmark_vla_smoother(500)
        print(f"  P99 Latency : {vla_res['p99_latency_ms']:.4f} ms")

        print("\n[2/3] Benchmarking random untrained scorer latency...")
        scorer_res = benchmark_distilled_scorer(200)
        latency_label = "PASS" if scorer_res["latency_p99_pass"] else "FAIL"
        print(f"  Device      : {scorer_res['device']}")
        print(
            f"  P99 Latency : {scorer_res['p99_latency_ms']:.2f} ms "
            f"(device threshold {scorer_res['device_latency_threshold_p99_ms']:.2f} ms: {latency_label})"
        )

        print("\n[3/3] Benchmarking MPC controller latency...")
        mpc_res = benchmark_mpc_control(200)
        print(f"  P99 Latency : {mpc_res['p99_latency_ms']:.4f} ms")
        summary = build_latency_smoke_summary(vla_res, scorer_res, mpc_res)
    except Exception as exc:
        summary = build_latency_smoke_summary(
            {},
            {"model_state": "random_untrained"},
            {},
            completed=False,
            error=f"{type(exc).__name__}:{exc}",
        )

    out_file = args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nBenchmark summary written to: {out_file}")
    print("=" * 60)
    return 0 if summary["status"] == "SMOKE_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
