"""H5 paired closed-loop metrics and acceptance gates.

These functions are pure and testable without CARLA.  They consume H5RunRecord
or simple dictionaries produced by the live runner.
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from .config import H5_CONFIG
from .contracts import H5RunRecord


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    frac = pos - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def summarize_run(run: H5RunRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(run, H5RunRecord):
        d = run.to_dict()
    else:
        d = dict(run)
    decisions = d.get("decisions", ())
    scorer_ms = [float(x["scorer_latency_ms"]) for x in decisions if x.get("scorer_latency_ms") is not None]
    return {
        "pair_id": d["pair_id"],
        "arm": d["arm"],
        "route_progress_m": float(d["route_progress_m"]),
        "route_completed": bool(d["route_completed"]),
        "collision_count": int(d["collision_count"]),
        "red_light_violation": bool(d["red_light_violation"]),
        "off_corridor_duration_s": float(d["off_corridor_duration_s"]),
        "switch_count": int(d["switch_count"]),
        "defer_count": int(d["defer_count"]),
        "fallback_count": int(d["fallback_count"]),
        "safety_fallback_count": int(d["safety_fallback_count"]),
        "scorer_deadline_misses": int(d["scorer_deadline_misses"]),
        "deadline_misses": int(d["deadline_misses"]),
        "p50_scorer_ms": _pct(scorer_ms, 0.50),
        "p95_scorer_ms": _pct(scorer_ms, 0.95),
        "p99_scorer_ms": _pct(scorer_ms, 0.99),
        "whole_gpu_peak_gb": float(d["whole_gpu_peak_gb"]),
        "ok": bool(d["ok"]),
        "cleanup_complete": bool(d["cleanup_complete"]),
        "ticks_executed": int(d["ticks_executed"]),
        "vla_forward_count": int(d["vla_forward_count"]),
    }


def _paired_values(
    runs: Sequence[H5RunRecord | Mapping[str, Any]],
    *,
    on_arm: str = "on",
    off_arm: str = "off",
) -> tuple[list[H5RunRecord | Mapping[str, Any]], list[H5RunRecord | Mapping[str, Any]]]:
    by_pair: dict[str, dict[str, Any]] = {}
    for run in runs:
        if isinstance(run, H5RunRecord):
            d = run.to_dict()
        else:
            d = dict(run)
        by_pair.setdefault(d["pair_id"], {})[d["arm"]] = d
    on_runs = []
    off_runs = []
    for pair_id in sorted(by_pair):
        if on_arm in by_pair[pair_id] and off_arm in by_pair[pair_id]:
            on_runs.append(by_pair[pair_id][on_arm])
            off_runs.append(by_pair[pair_id][off_arm])
    return on_runs, off_runs


def paired_progress_ci(
    runs: Sequence[H5RunRecord | Mapping[str, Any]],
    *,
    on_arm: str = "on",
    off_arm: str = "off",
    seed: int | None = None,
    rounds: int | None = None,
) -> dict[str, Any]:
    on_runs, off_runs = _paired_values(runs, on_arm=on_arm, off_arm=off_arm)
    if not on_runs:
        raise ValueError("no_paired_runs")
    diffs = [float(a["route_progress_m"]) - float(b["route_progress_m"]) for a, b in zip(on_runs, off_runs)]
    rng = random.Random(seed if seed is not None else H5_CONFIG["acceptance"]["bootstrap_seed"])
    n = len(diffs)
    rounds = int(rounds if rounds is not None else H5_CONFIG["acceptance"]["bootstrap_rounds"])
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower = means[max(0, int(rounds * 0.025) - 1)]
    upper = means[min(rounds - 1, int(rounds * 0.975) - 1)]
    return {
        "n": n,
        "mean_delta": sum(diffs) / n,
        "lower_95": lower,
        "upper_95": upper,
        "positive_pairs": sum(1 for d in diffs if d > 0),
        "zero_pairs": sum(1 for d in diffs if d == 0),
        "negative_pairs": sum(1 for d in diffs if d < 0),
    }


def paired_safety(
    runs: Sequence[H5RunRecord | Mapping[str, Any]],
    *,
    on_arm: str = "on",
    off_arm: str = "off",
) -> dict[str, Any]:
    on_runs, off_runs = _paired_values(runs, on_arm=on_arm, off_arm=off_arm)
    violations = []
    on_unsafe_total = 0
    off_unsafe_total = 0
    for a, b in zip(on_runs, off_runs):
        a_unsafe = int(a["collision_count"]) > 0 or bool(a["red_light_violation"]) or float(a["off_corridor_duration_s"]) > 0.25
        b_unsafe = int(b["collision_count"]) > 0 or bool(b["red_light_violation"]) or float(b["off_corridor_duration_s"]) > 0.25
        on_unsafe_total += int(a_unsafe)
        off_unsafe_total += int(b_unsafe)
        if a_unsafe and not b_unsafe:
            violations.append({"pair_id": a["pair_id"], "direction": "on_unsafe_off_safe"})
        elif b_unsafe and not a_unsafe:
            violations.append({"pair_id": a["pair_id"], "direction": "off_unsafe_on_safe"})
    return {
        "n": len(on_runs),
        "on_unsafe": on_unsafe_total,
        "off_unsafe": off_unsafe_total,
        "on_unsafe_off_safe": [v for v in violations if v["direction"] == "on_unsafe_off_safe"],
        "off_unsafe_on_safe": [v for v in violations if v["direction"] == "off_unsafe_on_safe"],
        "passed": not any(v["direction"] == "on_unsafe_off_safe" for v in violations),
    }


def paired_chattering(
    runs: Sequence[H5RunRecord | Mapping[str, Any]],
    *,
    on_arm: str = "on",
    off_arm: str = "off",
) -> dict[str, Any]:
    on_runs, off_runs = _paired_values(runs, on_arm=on_arm, off_arm=off_arm)
    deltas = [int(a["switch_count"]) - int(b["switch_count"]) for a, b in zip(on_runs, off_runs)]
    on_worse = [a["pair_id"] for a, b, d in zip(on_runs, off_runs, deltas) if d > 0]
    return {
        "n": len(on_runs),
        "on_switch_total": sum(int(a["switch_count"]) for a in on_runs),
        "off_switch_total": sum(int(b["switch_count"]) for b in off_runs),
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "on_worse_pairs": on_worse,
        "passed": not on_worse,
    }


def evaluate_gate(
    runs: Sequence[H5RunRecord | Mapping[str, Any]],
    *,
    on_arm: str = "on",
    off_arm: str = "off",
) -> dict[str, Any]:
    progress = paired_progress_ci(runs, on_arm=on_arm, off_arm=off_arm)
    safety = paired_safety(runs, on_arm=on_arm, off_arm=off_arm)
    chattering = paired_chattering(runs, on_arm=on_arm, off_arm=off_arm)
    # Use per-on-run summary values already computed by summarize_run when available.
    on_summaries = [summarize_run(r) for r in runs if (r["arm"] if isinstance(r, dict) else r.arm) == on_arm]
    p99s = [s["p99_scorer_ms"] for s in on_summaries if s["p99_scorer_ms"] is not None]
    scorer_deadline_misses = sum(s["scorer_deadline_misses"] for s in on_summaries)
    max_p99 = max(p99s) if p99s else 0.0
    resource_ok = bool(p99s) and max_p99 <= float(H5_CONFIG["runtime"]["scorer_deadline_ms"]) and scorer_deadline_misses == 0
    progress_ok = progress["lower_95"] > float(H5_CONFIG["acceptance"]["progress_ci_lower"])
    checks = {
        "protocol_all_ok": all(bool((r["ok"] if isinstance(r, dict) else r.ok)) for r in runs),
        "safety_noninferior": safety["passed"],
        "progress_net_benefit": progress_ok,
        "chattering_noninferior": chattering["passed"],
        "resource": resource_ok,
    }
    failures = [k for k, v in checks.items() if not v]
    return {
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "progress": progress,
        "safety": safety,
        "chattering": chattering,
        "resource": {
            "max_p99_ms": max_p99,
            "scorer_deadline_misses": scorer_deadline_misses,
        },
    }


__all__ = [
    "evaluate_gate",
    "paired_chattering",
    "paired_progress_ci",
    "paired_safety",
    "summarize_run",
]
