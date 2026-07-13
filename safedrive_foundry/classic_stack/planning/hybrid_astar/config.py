"""Versioned Hybrid A* baseline configuration."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HybridAstarConfig:
    name: str
    schema_version: str
    analytic_expansion: str
    heuristic: str
    wheelbase_m: float
    max_steer_rad: float
    max_curvature_per_m: float
    width_m: float
    length_m: float
    min_turning_radius_m: float
    xy_resolution_m: float
    yaw_bins: int
    step_m: float
    steer_set: tuple[float, ...]
    allow_reverse: bool
    gear_switch_penalty: float
    steer_switch_penalty: float
    max_expansions: int
    goal_xy_tol_m: float
    goal_yaw_tol_rad: float
    analytic_expansion_every: int
    analytic_when_dist_m: float
    time_budget_ms: float
    partial_solution: bool
    w_length: float
    w_reverse: float
    w_gear: float
    w_steer: float
    source_path: str | None = None
    raw_toml: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "analytic_expansion": self.analytic_expansion,
            "xy_resolution_m": self.xy_resolution_m,
            "yaw_bins": self.yaw_bins,
            "step_m": self.step_m,
            "max_expansions": self.max_expansions,
            "time_budget_ms": self.time_budget_ms,
        }


def config_sha256(raw: str | bytes) -> str:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "config"
        / "classic_stack"
        / "hybrid_astar_baseline.toml"
    )


def load_hybrid_astar_config(path: str | Path | None = None) -> HybridAstarConfig:
    cfg_path = Path(path) if path is not None else default_config_path()
    raw = cfg_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    b = data.get("baseline", {})
    v = data.get("vehicle", {})
    d = data.get("discretization", {})
    s = data.get("search", {})
    c = data.get("cost", {})
    return HybridAstarConfig(
        name=str(b.get("name", "hybrid_astar")),
        schema_version=str(b.get("schema_version", "v1")),
        analytic_expansion=str(b.get("analytic_expansion", "reeds_shepp")),
        heuristic=str(b.get("heuristic", "nonholonomic_rs_plus_obstacle")),
        wheelbase_m=float(v.get("wheelbase_m", 2.8)),
        max_steer_rad=float(v.get("max_steer_rad", 0.6)),
        max_curvature_per_m=float(v.get("max_curvature_per_m", 0.25)),
        width_m=float(v.get("width_m", 1.9)),
        length_m=float(v.get("length_m", 4.5)),
        min_turning_radius_m=float(v.get("min_turning_radius_m", 4.0)),
        xy_resolution_m=float(d.get("xy_resolution_m", 0.5)),
        yaw_bins=int(d.get("yaw_bins", 16)),
        step_m=float(d.get("step_m", 0.7)),
        steer_set=tuple(float(x) for x in d.get("steer_set", [-0.5, 0.0, 0.5])),
        allow_reverse=bool(d.get("allow_reverse", True)),
        gear_switch_penalty=float(d.get("gear_switch_penalty", 2.0)),
        steer_switch_penalty=float(d.get("steer_switch_penalty", 0.3)),
        max_expansions=int(s.get("max_expansions", 5000)),
        goal_xy_tol_m=float(s.get("goal_xy_tol_m", 0.6)),
        goal_yaw_tol_rad=float(s.get("goal_yaw_tol_rad", 0.35)),
        analytic_expansion_every=int(s.get("analytic_expansion_every", 20)),
        analytic_when_dist_m=float(s.get("analytic_when_dist_m", 8.0)),
        time_budget_ms=float(s.get("time_budget_ms", 800.0)),
        partial_solution=bool(s.get("partial_solution", True)),
        w_length=float(c.get("w_length", 1.0)),
        w_reverse=float(c.get("w_reverse", 1.5)),
        w_gear=float(c.get("w_gear", 2.0)),
        w_steer=float(c.get("w_steer", 0.2)),
        source_path=str(cfg_path),
        raw_toml=raw,
    )
