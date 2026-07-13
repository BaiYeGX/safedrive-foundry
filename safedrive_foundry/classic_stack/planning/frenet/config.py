"""Versioned Frenet + S-T baseline configuration."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classic_stack.geometry import VehicleParams


@dataclass(frozen=True)
class FrenetSTConfig:
    name: str
    schema_version: str
    prediction_model: str
    speed_solver: str
    vehicle: VehicleParams
    ds_m: float
    horizon_s: tuple[float, ...]
    lateral_offsets_m: tuple[float, ...]
    target_speeds_mps: tuple[float, ...]
    max_candidates: int
    road_half_width_m: float
    costs: dict[str, float]
    pred_dt_s: float
    pred_horizon_s: float
    idm_time_gap_s: float
    idm_min_gap_m: float
    idm_max_accel: float
    idm_comf_decel: float
    actor_length_m: float
    actor_width_m: float
    st_ds_m: float
    st_dt_s: float
    st_s_horizon_m: float
    st_t_horizon_s: float
    st_inflation_m: float
    smooth_window: int
    source_path: str | None = None
    raw_toml: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "prediction_model": self.prediction_model,
            "speed_solver": self.speed_solver,
            "vehicle": self.vehicle.__dict__,
            "ds_m": self.ds_m,
            "horizon_s": list(self.horizon_s),
            "lateral_offsets_m": list(self.lateral_offsets_m),
            "target_speeds_mps": list(self.target_speeds_mps),
            "max_candidates": self.max_candidates,
            "road_half_width_m": self.road_half_width_m,
            "costs": dict(self.costs),
            "st_grid": {
                "ds_m": self.st_ds_m,
                "dt_s": self.st_dt_s,
                "s_horizon_m": self.st_s_horizon_m,
                "t_horizon_s": self.st_t_horizon_s,
                "inflation_m": self.st_inflation_m,
            },
        }


def config_sha256(raw: str | bytes) -> str:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "config"
        / "classic_stack"
        / "frenet_st_baseline.toml"
    )


def load_frenet_st_config(path: str | Path | None = None) -> FrenetSTConfig:
    cfg_path = Path(path) if path is not None else default_config_path()
    raw = cfg_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    base = data.get("baseline", {})
    veh = data.get("vehicle", {})
    sampling = data.get("sampling", {})
    cost = data.get("cost", {})
    pred = data.get("prediction", {})
    st = data.get("st_grid", {})
    smooth = data.get("smoothing", {})
    vehicle = VehicleParams(
        wheelbase_m=float(veh.get("wheelbase_m", 2.8)),
        max_speed_mps=float(veh.get("max_speed_mps", 15.0)),
        max_accel_mps2=float(veh.get("max_accel_mps2", 2.5)),
        max_decel_mps2=float(veh.get("max_decel_mps2", 4.0)),
        max_jerk_mps3=float(veh.get("max_jerk_mps3", 4.0)),
        max_curvature_per_m=float(veh.get("max_curvature_per_m", 0.25)),
        max_lateral_accel_mps2=float(veh.get("max_lateral_accel_mps2", 2.0)),
        width_m=float(veh.get("width_m", 1.9)),
        length_m=float(veh.get("length_m", 4.5)),
    )
    return FrenetSTConfig(
        name=str(base.get("name", "frenet_st")),
        schema_version=str(base.get("schema_version", "v1")),
        prediction_model=str(base.get("prediction_model", pred.get("model", "cv_ctrv_idm"))),
        speed_solver=str(base.get("speed_solver", "st_dp")),
        vehicle=vehicle,
        ds_m=float(sampling.get("ds_m", 2.0)),
        horizon_s=tuple(float(x) for x in sampling.get("horizon_s", [4.0, 5.0, 6.0])),
        lateral_offsets_m=tuple(float(x) for x in sampling.get("lateral_offsets_m", [0.0])),
        target_speeds_mps=tuple(float(x) for x in sampling.get("target_speeds_mps", [6.0])),
        max_candidates=int(sampling.get("max_candidates", 100)),
        road_half_width_m=float(sampling.get("road_half_width_m", 3.5)),
        costs={str(k): float(v) for k, v in cost.items()},
        pred_dt_s=float(pred.get("dt_s", 0.1)),
        pred_horizon_s=float(pred.get("horizon_s", 6.0)),
        idm_time_gap_s=float(pred.get("idm_desired_time_gap_s", 1.2)),
        idm_min_gap_m=float(pred.get("idm_min_gap_m", 3.0)),
        idm_max_accel=float(pred.get("idm_max_accel_mps2", 1.5)),
        idm_comf_decel=float(pred.get("idm_comfortable_decel_mps2", 2.5)),
        actor_length_m=float(pred.get("default_actor_length_m", 4.5)),
        actor_width_m=float(pred.get("default_actor_width_m", 1.9)),
        st_ds_m=float(st.get("ds_m", 1.0)),
        st_dt_s=float(st.get("dt_s", 0.2)),
        st_s_horizon_m=float(st.get("s_horizon_m", 60.0)),
        st_t_horizon_s=float(st.get("t_horizon_s", 6.0)),
        st_inflation_m=float(st.get("inflation_m", 0.5)),
        smooth_window=int(smooth.get("window", 5)),
        source_path=str(cfg_path),
        raw_toml=raw,
    )
