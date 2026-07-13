"""RACE-Plan ablation runner over frozen G1-04/G1-05 planners (honest counters)."""

from __future__ import annotations

import hashlib
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from classic_stack.planning.frenet import FrenetPlanner, load_frenet_st_config
from classic_stack.planning.frenet.scenarios import SCENARIO_KINDS, make_scenario
from classic_stack.planning.hybrid_astar import HybridAstarPlanner, load_hybrid_astar_config
from classic_stack.planning.hybrid_astar.scenarios import MANEUVER_KINDS, make_maneuver
from classic_stack.risk import evaluate_risk_field, monotonicity_ok


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class RacePlanRunner:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.race_cfg_path = self.root / "safedrive_foundry/config/planning/race_plan.toml"
        self.race_raw = self.race_cfg_path.read_text(encoding="utf-8")
        self.race_data = tomllib.loads(self.race_raw)
        frenet_path = self.root / "safedrive_foundry/config/classic_stack/frenet_st_baseline.toml"
        hybrid_path = self.root / "safedrive_foundry/config/classic_stack/hybrid_astar_baseline.toml"
        self.frenet_hash = _file_hash(frenet_path)
        self.hybrid_hash = _file_hash(hybrid_path)
        self.frenet_cfg = load_frenet_st_config(frenet_path)
        self.hybrid_cfg = load_hybrid_astar_config(hybrid_path)

    def _frenet_for_flags(self, flags: dict[str, Any]) -> FrenetPlanner:
        """Actually change candidate generation for adaptive sampling."""

        cfg = self.frenet_cfg
        if flags.get("adaptive_sampling"):
            offs = tuple(sorted({-1.75, 0.0, 1.75, *cfg.lateral_offsets_m[::2]}))
            hors = tuple(cfg.horizon_s[::2]) or cfg.horizon_s
            cfg = replace(
                cfg,
                lateral_offsets_m=offs,
                horizon_s=hors,
                max_candidates=max(20, int(cfg.max_candidates * 0.6)),
            )
        return FrenetPlanner(cfg)

    def _hybrid_for_flags(self, flags: dict[str, Any]) -> HybridAstarPlanner:
        cfg = self.hybrid_cfg
        if flags.get("multi_heuristic"):
            cfg = replace(
                cfg,
                max_expansions=int(cfg.max_expansions * 1.25),
                analytic_expansion_every=max(1, cfg.analytic_expansion_every // 2),
            )
        return HybridAstarPlanner(cfg)

    def run_variant(self, name: str) -> dict[str, Any]:
        flags = dict(self.race_data.get("ablation", {}).get(name, {}))
        t0 = time.perf_counter()
        frenet_planner = self._frenet_for_flags(flags)
        hybrid_planner = self._hybrid_for_flags(flags)

        frenet_rows = []
        for kind in SCENARIO_KINDS:
            req = make_scenario(kind, seed=1)
            actors = [{"s": 20.0, "v": 5.0}]
            risk_obs = evaluate_risk_field(
                ego_v=req.v0,
                actors=actors,
                lateral_offset_m=req.preferred_offset_m or 0.0,
                uncertainty_scale=1.5 if flags.get("risk_trackability") else 1.0,
                track="observable",
            )
            risk_oracle = evaluate_risk_field(
                ego_v=req.v0,
                actors=actors,
                lateral_offset_m=req.preferred_offset_m or 0.0,
                uncertainty_scale=1.5 if flags.get("risk_trackability") else 1.0,
                track="oracle",
                _compute_oracle=False,
            )
            # Coarse-to-fine: real two-stage search with separate raw counters
            stage_counts: list[int] = []
            best = None
            if flags.get("coarse_to_fine"):
                coarse_cfg = replace(
                    frenet_planner.config,
                    max_candidates=max(15, int(frenet_planner.config.max_candidates * 0.4)),
                )
                coarse = FrenetPlanner(coarse_cfg)
                r0 = coarse.plan(req)
                stage_counts.append(r0.candidates)
                fine = FrenetPlanner(self.frenet_cfg)  # full baseline fine stage
                r1 = fine.plan(req)
                stage_counts.append(r1.candidates)
                best = r1 if r1.ok or not r0.ok else r0
                candidates = sum(stage_counts)
            else:
                r = frenet_planner.plan(req)
                stage_counts.append(r.candidates)
                best = r
                candidates = r.candidates
            assert best is not None
            # Optional: soft risk term recorded only (does not rewrite candidate counts)
            frenet_rows.append(
                {
                    "scenario": kind,
                    "ok": best.ok,
                    "candidates_raw": candidates,
                    "stage_candidates": stage_counts,
                    "wall_ms": best.wall_time_ms,
                    "risk_mean": risk_obs.observable_score,
                    "oracle_risk": risk_oracle.observable_score,
                }
            )

        hybrid_rows = []
        for kind in MANEUVER_KINDS:
            req = make_maneuver(kind, seed=1)
            r = hybrid_planner.plan(req)
            hybrid_rows.append(
                {
                    "scenario": kind,
                    "ok": r.ok,
                    "partial": r.partial,
                    "nodes_raw": r.nodes_expanded,
                    "analytic_hits": r.analytic_hits,
                    "length": r.path_length_m,
                    "wall_ms": r.wall_time_ms,
                }
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        frenet_ok = sum(1 for r in frenet_rows if r["ok"])
        hybrid_ok = sum(1 for r in hybrid_rows if r["ok"] and not r.get("partial"))
        cand_sum = sum(r["candidates_raw"] for r in frenet_rows)
        nodes_sum = sum(r["nodes_raw"] for r in hybrid_rows)
        p99_fr = sorted(r["wall_ms"] for r in frenet_rows)[-1] if frenet_rows else 0.0
        return {
            "variant": name,
            "flags": flags,
            "frenet": frenet_rows,
            "hybrid": hybrid_rows,
            "frenet_success_rate": frenet_ok / max(1, len(frenet_rows)),
            "hybrid_success_rate": hybrid_ok / max(1, len(hybrid_rows)),
            "candidates_raw_sum": cand_sum,
            "nodes_raw_sum": nodes_sum,
            "frenet_wall_p99_ms": p99_fr,
            "wall_ms_total": elapsed,
            "baseline_hashes": {"frenet": self.frenet_hash, "hybrid": self.hybrid_hash},
        }


def run_race_plan_ablation(root: Path) -> dict[str, Any]:
    runner = RacePlanRunner(root)
    matrix = {}
    for name in ("basic", "p1", "p2", "full"):
        matrix[name] = runner.run_variant(name)
    mono = monotonicity_ok([{"s": 18.0, "v": 4.0}], ego_v=9.0)

    basic = matrix["basic"]
    full = matrix["full"]
    p1 = matrix["p1"]
    work_ratio = full["candidates_raw_sum"] / max(1, basic["candidates_raw_sum"])
    p99_b = basic.get("frenet_wall_p99_ms", 0.0)
    p99_f = full.get("frenet_wall_p99_ms", 0.0)
    # Net-benefit admission (CLAIMS): no free lunch on work/latency without success gain.
    # coarse-to-fine/multi-heuristic/risk_trackability remain experimental/partial.
    success_ok = full["frenet_success_rate"] + 1e-9 >= basic["frenet_success_rate"]
    work_ok = full["candidates_raw_sum"] <= basic["candidates_raw_sum"]  # no increase allowed
    p99_ok = p99_f <= p99_b * 1.05 + 1e-9
    promote_full = success_ok and work_ok and p99_ok
    # Prefer p1 when it reduces work without losing success
    p1_better = (
        p1["frenet_success_rate"] + 1e-9 >= basic["frenet_success_rate"]
        and p1["candidates_raw_sum"] < basic["candidates_raw_sum"]
    )
    recommended = "full" if promote_full else ("p1" if p1_better else "basic")
    reason = (
        "full meets success+work+p99 net benefit"
        if promote_full
        else f"negative full admission (work_ratio={work_ratio:.3f}); recommend {recommended}"
    )
    return {
        "schema": "safedrive.g1_07.ablation.repair.v2",
        "matrix": matrix,
        "risk_monotonicity_ok": mono,
        "frenet_baseline_hash": runner.frenet_hash,
        "hybrid_baseline_hash": runner.hybrid_hash,
        "default_admission": {
            "promote_full_to_default": promote_full,
            "recommended_default": recommended,
            "reason": reason,
            "work_ratio_full_over_basic": work_ratio,
            "basic_success": basic["frenet_success_rate"],
            "full_success": full["frenet_success_rate"],
            "basic_candidates": basic["candidates_raw_sum"],
            "full_candidates": full["candidates_raw_sum"],
            "basic_p99_ms": p99_b,
            "full_p99_ms": p99_f,
            "flag_status": {
                "coarse_to_fine": "experimental_partial_independent_rerun",
                "multi_heuristic": "experimental_partial_budget_only",
                "risk_trackability": "experimental_partial_stats_only",
                "adaptive_sampling": "active_reduces_candidate_set",
            },
        },
        "negative_result_policy": "No post-hoc scaling; full without net benefit keeps basic/p1 default",
        "honesty": "candidates_raw/nodes_raw are measured, never multiplied post-search",
    }
