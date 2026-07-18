"""Temporal-vision single-trajectory baseline (lightweight, non-language) (G3-02).

Uses low-dim ego history (≤4) plus a short stack of image scalars (current +
optional history means stored in meta). Not a full visual backbone — honest
encoder_budget for fair V0 K1 comparison.
"""

from __future__ import annotations

from typing import Any, Sequence

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.baselines.route_ego import RouteEgoBaseline
from driving_vla.schema.trajectory_contract import DT_S, T_STEPS


def _image_scalar(front_rgb: Any) -> float:
    if front_rgb is None:
        return 0.5
    try:
        import numpy as np

        arr = np.asarray(front_rgb)
        if arr.size == 0:
            return 0.5
        return float(arr.mean()) / 255.0
    except Exception:
        return 0.5


def _temporal_image_scalars(obs: ObservationBundle) -> list[float]:
    """Stack current scalar + up to 3 history scalars from meta if present."""
    cur = _image_scalar(obs.front_rgb)
    hist = list(obs.meta.get("image_scalar_history") or [])
    # keep last 3 history + current → length 4
    hist = [float(x) for x in hist][-3:]
    while len(hist) < 3:
        hist.insert(0, cur)
    return hist + [cur]


def _history_speed_scale(history: Sequence[tuple[float, float, float, float]], ego_v: float) -> float:
    if not history:
        return 1.0
    vs = [float(h[3]) for h in history[-4:]]
    mean_v = sum(vs) / len(vs)
    base = max(ego_v, 0.5) if ego_v > 0.1 else 3.0
    return max(0.6, min(1.25, mean_v / base))


class VisionK1Baseline:
    """K=1 baseline with temporal ego + image-scalar stack modulating speed."""

    model_id = "baseline_vision_k1_v0"
    k = 1
    encoder_budget = {
        "params_m": 0.0,
        "kind": "temporal_scalar_stack",
        "history_steps": 4,
        "image_scalars": 4,
        "not_full_vision_backbone": True,
    }

    def __init__(self) -> None:
        self._route = RouteEgoBaseline()

    def _speed_scale(self, obs: ObservationBundle) -> float:
        scalars = _temporal_image_scalars(obs)
        # Temporal change in brightness + level
        mean_s = sum(scalars) / len(scalars)
        delta = scalars[-1] - scalars[0]
        img_scale = 0.7 + 0.45 * mean_s + 0.1 * max(-1.0, min(1.0, delta * 2.0))
        hist_scale = _history_speed_scale(obs.ego_history, obs.ego_v)
        return max(0.5, min(1.3, img_scale * hist_scale))

    def predict(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        scale = self._speed_scale(obs)
        scaled = ObservationBundle(
            run_id=obs.run_id,
            frame_id=obs.frame_id,
            scenario_id=obs.scenario_id,
            simulation_time_s=obs.simulation_time_s,
            wall_time_s=obs.wall_time_s,
            carla_frame=obs.carla_frame,
            ego_x=obs.ego_x,
            ego_y=obs.ego_y,
            ego_yaw=obs.ego_yaw,
            ego_v=max(0.5, obs.ego_v * scale if obs.ego_v > 0.1 else 3.0 * scale),
            route_xy=obs.route_xy,
            front_rgb=obs.front_rgb,
            ego_history=obs.ego_history,
            meta={**obs.meta, "vision_k1_scale": scale, "temporal_scalars": _temporal_image_scalars(obs)},
        )
        arrs = self._route.predict(scaled)
        out = []
        for a in arrs:
            out.append(
                TrajectoryArray(
                    points_xy_yaw_v_a_kappa=a.points_xy_yaw_v_a_kappa,
                    probability=1.0,
                    uncertainty=0.18,
                    candidate_id="vision_k1_0",
                    intended_action="nominal",
                    behavior="follow",
                )
            )
        return out
