"""K=2 multi-candidate interface (G3-02) — frozen shape, no random noise candidates."""

from __future__ import annotations

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.baselines.route_ego import RouteEgoBaseline
from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS, V1_K


class MultiCandidateK2Interface:
    """Produces nominal + conservative K=2 from geometry (fair non-language baseline).

    Not a learned multi-modal model — provides the K2 contract surface for V1/World.
    """

    model_id = "baseline_multi_k2_interface_v0"
    k = V1_K
    encoder_budget = {"params_m": 0.0, "kind": "dual_speed_route", "comparable_to_v1_k2": True}

    def __init__(self) -> None:
        self._route = RouteEgoBaseline()

    def predict(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        nominal_obs = obs
        cons_v = max(0.5, (obs.ego_v if obs.ego_v > 0.1 else 3.0) * 0.6)
        cons_obs = ObservationBundle(
            run_id=obs.run_id,
            frame_id=obs.frame_id,
            scenario_id=obs.scenario_id,
            simulation_time_s=obs.simulation_time_s,
            wall_time_s=obs.wall_time_s,
            carla_frame=obs.carla_frame,
            ego_x=obs.ego_x,
            ego_y=obs.ego_y,
            ego_yaw=obs.ego_yaw,
            ego_v=cons_v,
            route_xy=obs.route_xy,
            front_rgb=obs.front_rgb,
            ego_history=obs.ego_history,
            meta=obs.meta,
        )
        n = self._route.predict(nominal_obs)[0]
        c = self._route.predict(cons_obs)[0]
        return [
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=n.points_xy_yaw_v_a_kappa,
                probability=0.65,
                uncertainty=0.12,
                candidate_id="k2_nominal",
                intended_action="nominal",
            ),
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=c.points_xy_yaw_v_a_kappa,
                probability=0.35,
                uncertainty=0.20,
                candidate_id="k2_conservative",
                intended_action="conservative",
            ),
        ]

    def interface_frozen(self) -> dict:
        return {
            "k": self.k,
            "t_steps": T_STEPS,
            "dt_s": DT_S,
            "horizon_s": HORIZON_S,  # absolute end time 2.5s via t=(i+1)*dt
            "forbids_random_noise_candidates": True,
        }
