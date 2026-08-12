"""Nominal SimLingo policy used by the H route."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.model.canonicalizer import TrajectoryCanonicalizer, UpstreamPathSpeed
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime


def backend_name() -> str:
    return os.environ.get("SDF_VLA_BACKEND", "neural").strip().lower()


@dataclass(frozen=True)
class NativePathPrediction:
    """Unmodified SimLingo action heads in the project map frame."""

    path_map_xy: tuple[tuple[float, float], ...]
    speed_mps: tuple[float, ...]
    target_ego_1: tuple[float, float]
    target_ego_2: tuple[float, float]
    latency_s: float
    peak_vram_mb: float
    route_ego_xy: tuple[tuple[float, float], ...] = ()
    speed_wps_ego_xy: tuple[tuple[float, float], ...] = ()


class NominalVLAPolicy:
    """One real SimLingo forward producing one canonical nominal trajectory."""

    model_id = "sdf-vla-nominal@1.0.0"
    source = "neural_simlingo"

    def __init__(
        self,
        runtime: SimLingoNeuralRuntime | None = None,
        *,
        lazy: bool = True,
        keep_on_gpu: bool = False,
    ) -> None:
        self.runtime = runtime
        self._lazy = lazy
        self.keep_on_gpu = keep_on_gpu or os.environ.get(
            "SDF_VLA_KEEP_ON_GPU", ""
        ).strip() in {"1", "true", "TRUE", "yes"}
        self._gpu_pinned = False
        self.canonicalizer = TrajectoryCanonicalizer()
        self.last_latency_s = 0.0
        self.last_peak_vram_mb = 0.0

    def ensure_loaded(self) -> None:
        if self.runtime is None:
            self.runtime = SimLingoNeuralRuntime()
        if not self.runtime.load_report.ok:
            report = self.runtime.load()
            if not report.ok:
                raise RuntimeError(f"neural load failed: {report.error}")
        if self.keep_on_gpu and not self._gpu_pinned:
            self.runtime.keep_model_on_gpu()
            self._gpu_pinned = True

    @staticmethod
    def _map_to_ego(
        map_x: float,
        map_y: float,
        obs: ObservationBundle,
    ) -> tuple[float, float]:
        dx = map_x - obs.ego_x
        dy = map_y - obs.ego_y
        c, s = math.cos(-obs.ego_yaw), math.sin(-obs.ego_yaw)
        return float(c * dx - s * dy), float(s * dx + c * dy)

    def _route_targets(
        self,
        obs: ObservationBundle,
        *,
        d1: float = 15.0,
        d2: float = 30.0,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        from driving_vla.model.simlingo_contract import (
            SimLingoContractConfig,
            navigation_targets,
        )

        meta = obs.meta or {}
        if "target_ego_1" in meta and "target_ego_2" in meta:
            target_1 = meta["target_ego_1"]
            target_2 = meta["target_ego_2"]
            return (
                (float(target_1[0]), float(target_1[1])),
                (float(target_2[0]), float(target_2[1])),
            )
        if not obs.route_xy or len(obs.route_xy) < 2:
            return (float(d1), 0.0), (float(d2), 0.0)

        official = bool(meta.get("official_contract", True))
        config = SimLingoContractConfig(
            official_contract=official,
            legacy_target_d1_m=float(d1),
            legacy_target_d2_m=float(d2),
        )
        result = navigation_targets(
            list(obs.route_xy),
            ego_x=float(obs.ego_x),
            ego_y=float(obs.ego_y),
            ego_yaw=float(obs.ego_yaw),
            progress_hint_s=float(meta.get("route_progress_s", 0.0) or 0.0),
            config=config,
        )
        return result.target_ego_1, result.target_ego_2

    def _route_target(self, obs: ObservationBundle) -> tuple[float, float]:
        target_1, _ = self._route_targets(obs)
        return target_1

    @staticmethod
    def _ego_path_to_map(
        path_ego: np.ndarray,
        obs: ObservationBundle,
    ) -> tuple[tuple[float, float], ...]:
        c, s = math.cos(obs.ego_yaw), math.sin(obs.ego_yaw)
        return tuple(
            (
                float(obs.ego_x + c * float(x) - s * float(y)),
                float(obs.ego_y + s * float(x) + c * float(y)),
            )
            for x, y in path_ego
        )

    @staticmethod
    def resolve_vla_input_speed_mps(obs: ObservationBundle) -> float:
        """Return the observable ego speed passed to SimLingo."""

        ego_speed = max(0.0, float(obs.ego_v))
        meta = obs.meta or {}
        override = meta.get("vla_input_speed_mps")
        if override is not None:
            return max(0.0, float(override))
        startup_assist = meta.get("startup_speed_assist_mps")
        if (
            startup_assist is not None
            and not bool(meta.get("has_collided", False))
            and ego_speed <= 0.05
        ):
            return max(0.0, float(startup_assist))
        return ego_speed

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        native = self.predict_native(obs)
        trajectory = self.canonicalizer.canonicalize(
            UpstreamPathSpeed(
                path_xy=native.path_map_xy,
                speed_mps=native.speed_mps,
                frame="map",
            ),
            to_map=False,
        )
        return [
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=trajectory.points_xy_yaw_v_a_kappa,
                probability=1.0,
                uncertainty=0.15,
                candidate_id="vla_nominal",
                intended_action="nominal",
                behavior="follow",
            )
        ]

    def predict_native(self, obs: ObservationBundle) -> NativePathPrediction:
        if backend_name() in {"debug_geom", "fingerprint", "geom"}:
            raise RuntimeError(
                "debug geometry backend is disabled; unset SDF_VLA_BACKEND"
            )
        self.ensure_loaded()
        assert self.runtime is not None
        if obs.front_rgb is None:
            raise ValueError(
                "nominal VLA requires ObservationBundle.front_rgb (H,W,3 uint8)"
            )
        image = np.asarray(obs.front_rgb)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        meta = obs.meta or {}
        official = bool(meta.get("official_contract", True))
        if meta.get("image_layout"):
            image_layout = str(meta["image_layout"]).lower()
        elif meta.get("image_bgr"):
            image_layout = "bgr"
        else:
            image_layout = "rgb"

        target_1, target_2 = self._route_targets(obs)
        input_speed = self.resolve_vla_input_speed_mps(obs)
        if obs.meta is not None:
            obs.meta["resolved_target_ego_1"] = target_1
            obs.meta["resolved_target_ego_2"] = target_2
            obs.meta["resolved_vla_input_speed_mps"] = input_speed

        from driving_vla.model.simlingo_runtime import SIMLINGO_CAMERA_XYZ

        camera_xyz = tuple(meta.get("camera_mount_xyz") or SIMLINGO_CAMERA_XYZ)
        prompt_mode = str(
            meta.get("prompt_mode")
            or ("target_point" if official else "legacy_command_text")
        ).strip().lower()
        if official:
            if prompt_mode in {"command", "hlc_command"}:
                eval_route_as = "command"
            elif prompt_mode in {"target_point", "target_point_command"}:
                eval_route_as = "target_point"
            else:
                raise ValueError(f"unsupported official SimLingo prompt_mode={prompt_mode}")
        else:
            eval_route_as = "target_point"

        if "command_text" in meta:
            raw_command = meta.get("command_text")
            command_text = None if raw_command is None else str(raw_command)
        else:
            command_text = None if official else "Command: follow the road."
        if obs.meta is not None:
            obs.meta["resolved_command_text"] = command_text
            obs.meta["resolved_prompt_mode"] = (
                eval_route_as if official else "legacy_command_text"
            )

        result = self.runtime.forward_numpy(
            image,
            speed_mps=input_speed,
            target_point_xy=target_1,
            target_point2_xy=target_2,
            keep_on_gpu=self.keep_on_gpu,
            borrow_gpu=not self.keep_on_gpu,
            camera_mount_xyz=(
                float(camera_xyz[0]),
                float(camera_xyz[1]),
                float(camera_xyz[2]),
            ),
            image_layout=image_layout,
            official_contract=official,
            command_text=command_text,
            eval_route_as=eval_route_as,
        )
        self.last_latency_s = float(result.latency_s)
        self.last_peak_vram_mb = float(result.peak_vram_mb)
        route_ego = tuple(
            (float(x), float(y))
            for x, y in np.asarray(result.route_xy).reshape(-1, 2)
        )
        speed_waypoints = getattr(result, "speed_wps_xy", None)
        speed_ego = (
            ()
            if speed_waypoints is None
            else tuple(
                (float(x), float(y))
                for x, y in np.asarray(speed_waypoints).reshape(-1, 2)
            )
        )
        return NativePathPrediction(
            path_map_xy=self._ego_path_to_map(result.route_xy, obs),
            speed_mps=tuple(float(value) for value in result.speed_mps),
            target_ego_1=target_1,
            target_ego_2=target_2,
            latency_s=self.last_latency_s,
            peak_vram_mb=self.last_peak_vram_mb,
            route_ego_xy=route_ego,
            speed_wps_ego_xy=speed_ego,
        )


__all__ = ["NativePathPrediction", "NominalVLAPolicy", "backend_name"]
