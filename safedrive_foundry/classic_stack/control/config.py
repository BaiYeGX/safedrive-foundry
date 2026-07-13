"""Versioned control baseline configuration."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ControlConfig:
    name: str
    schema_version: str
    profile: str
    control_period_ms: float
    deadline_ms: float
    wheelbase_m: float
    max_speed_mps: float
    max_steer_rad: float
    max_accel_mps2: float
    max_brake_mps2: float
    steer_deadzone_rad: float
    throttle_deadzone: float
    mpc_horizon: int
    mpc_dt_s: float
    mpc_weights: dict[str, float]
    max_iterations: int
    solver_tolerance: float
    warm_start: bool
    wall_budget_ms: float
    fallback_chain: tuple[str, ...]
    stale_trajectory_s: float
    pid: dict[str, float]
    pure_pursuit: dict[str, float]
    lqr: dict[str, float]
    source_path: str | None = None
    raw_toml: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "profile": self.profile,
            "control_period_ms": self.control_period_ms,
            "deadline_ms": self.deadline_ms,
            "mpc_horizon": self.mpc_horizon,
            "warm_start": self.warm_start,
            "fallback_chain": list(self.fallback_chain),
            "wall_budget_ms": self.wall_budget_ms,
        }


def config_sha256(raw: str | bytes) -> str:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "control" / "mpc_pid_baseline.toml"


def load_control_config(path: str | Path | None = None) -> ControlConfig:
    cfg_path = Path(path) if path is not None else default_config_path()
    raw = cfg_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    b = data.get("baseline", {})
    v = data.get("vehicle", {})
    m = data.get("mpc", {})
    f = data.get("fallback", {})
    chain = f.get("chain", ["mpc", "lqr", "pure_pursuit", "pid", "brake"])
    return ControlConfig(
        name=str(b.get("name", "control")),
        schema_version=str(b.get("schema_version", "v1")),
        profile=str(b.get("profile", "control_50hz")),
        control_period_ms=float(b.get("control_period_ms", 20.0)),
        deadline_ms=float(b.get("deadline_ms", 20.0)),
        wheelbase_m=float(v.get("wheelbase_m", 2.8)),
        max_speed_mps=float(v.get("max_speed_mps", 15.0)),
        max_steer_rad=float(v.get("max_steer_rad", 0.6)),
        max_accel_mps2=float(v.get("max_accel_mps2", 2.5)),
        max_brake_mps2=float(v.get("max_brake_mps2", 4.0)),
        steer_deadzone_rad=float(v.get("steer_deadzone_rad", 0.01)),
        throttle_deadzone=float(v.get("throttle_deadzone", 0.02)),
        mpc_horizon=int(m.get("horizon", 10)),
        mpc_dt_s=float(m.get("dt_s", 0.02)),
        mpc_weights={
            "w_cte": float(m.get("w_cte", 4.0)),
            "w_heading": float(m.get("w_heading", 2.0)),
            "w_speed": float(m.get("w_speed", 1.0)),
            "w_steer": float(m.get("w_steer", 0.5)),
            "w_accel": float(m.get("w_accel", 0.3)),
            "w_dsteer": float(m.get("w_dsteer", 0.2)),
        },
        max_iterations=int(m.get("max_iterations", 15)),
        solver_tolerance=float(m.get("solver_tolerance", 1e-4)),
        warm_start=bool(m.get("warm_start", True)),
        wall_budget_ms=float(m.get("wall_budget_ms", 12.0)),
        fallback_chain=tuple(str(x) for x in chain),
        stale_trajectory_s=float(f.get("stale_trajectory_s", 0.2)),
        pid={k: float(v) for k, v in data.get("pid", {}).items()},
        pure_pursuit={k: float(v) for k, v in data.get("pure_pursuit", {}).items()},
        lqr={k: float(v) for k, v in data.get("lqr", {}).items()},
        source_path=str(cfg_path),
        raw_toml=raw,
    )
