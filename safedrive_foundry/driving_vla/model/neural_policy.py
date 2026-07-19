"""V0/V1 policies backed by real SimLingo neural forward (default)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.model.canonicalizer import TrajectoryCanonicalizer, UpstreamPathSpeed
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime
from driving_vla.model.v1_policy import _apply_residual, _history_features


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
    # Optional dual-head ego-frame dumps for debug-draw / evidence.
    route_ego_xy: tuple[tuple[float, float], ...] = ()
    speed_wps_ego_xy: tuple[tuple[float, float], ...] = ()


class NeuralV0Policy:
    model_id = "sdf-vla-v0-neural@0.1.0"
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
        self.keep_on_gpu = keep_on_gpu or os.environ.get("SDF_VLA_KEEP_ON_GPU", "").strip() in {
            "1",
            "true",
            "TRUE",
            "yes",
        }
        self._gpu_pinned = False
        self.canonicalizer = TrajectoryCanonicalizer()
        self.last_latency_s: float = 0.0
        self.last_peak_vram_mb: float = 0.0

    def ensure_loaded(self) -> None:
        if self.runtime is None:
            self.runtime = SimLingoNeuralRuntime()
        if not self.runtime.load_report.ok:
            rep = self.runtime.load()
            if not rep.ok:
                raise RuntimeError(f"neural load failed: {rep.error}")
        if self.keep_on_gpu and not self._gpu_pinned:
            self.runtime.keep_model_on_gpu()
            self._gpu_pinned = True

    def _map_to_ego(self, mx: float, my: float, obs: ObservationBundle) -> tuple[float, float]:
        dx = mx - obs.ego_x
        dy = my - obs.ego_y
        c, s = math.cos(-obs.ego_yaw), math.sin(-obs.ego_yaw)
        return (float(c * dx - s * dy), float(s * dx + c * dy))

    def _route_targets(
        self, obs: ObservationBundle, *, d1: float = 15.0, d2: float = 30.0
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Two ego-frame navigation targets.

        Prefer explicit meta targets from the runner (official RoutePlanner or
        legacy arc). Fallback: legacy +d1/+d2 arc along ``obs.route_xy``.
        """
        from driving_vla.model.simlingo_contract import (
            SimLingoContractConfig,
            navigation_targets,
        )

        meta = obs.meta or {}
        if "target_ego_1" in meta and "target_ego_2" in meta:
            t1 = meta["target_ego_1"]
            t2 = meta["target_ego_2"]
            return (float(t1[0]), float(t1[1])), (float(t2[0]), float(t2[1]))

        if not obs.route_xy or len(obs.route_xy) < 2:
            return (float(d1), 0.0), (float(d2), 0.0)

        official = bool(meta.get("official_contract", True))
        cfg = SimLingoContractConfig(
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
            config=cfg,
        )
        return result.target_ego_1, result.target_ego_2

    def _route_target(self, obs: ObservationBundle) -> tuple[float, float]:
        """Backward-compatible single target (first of dual advancing targets)."""
        t1, _t2 = self._route_targets(obs)
        return t1

    def _ego_path_to_map(self, path_ego: np.ndarray, obs: ObservationBundle) -> tuple[tuple[float, float], ...]:
        c, s = math.cos(obs.ego_yaw), math.sin(obs.ego_yaw)
        out = []
        for x, y in path_ego:
            mx = obs.ego_x + c * float(x) - s * float(y)
            my = obs.ego_y + s * float(x) + c * float(y)
            out.append((mx, my))
        return tuple(out)

    @staticmethod
    def resolve_vla_input_speed_mps(obs: ObservationBundle) -> float:
        """Speed value fed into SimLingo (not the commanded vehicle speed).

        Default: real non-negative ego speed, including true 0 when stopped.

        Historical bug: ``ego_v <= 0.05`` was replaced with ``3.0``, so after a
        first collision the model still saw "rolling at 3 m/s" while the car was
        parked and often kept outputting non-restartable stop intent.

        Optional first-start assist (distinct from post-collision stop):
        - set ``meta["startup_speed_assist_mps"]`` (e.g. 3.0) only while cold-starting
        - set ``meta["has_collided"]=True`` after the first collision episode; assist
          is then forbidden and real 0 m/s is always used when stopped
        - or force any value via ``meta["vla_input_speed_mps"]``
        """
        ego_v = max(0.0, float(obs.ego_v))
        meta = obs.meta or {}
        if "vla_input_speed_mps" in meta and meta["vla_input_speed_mps"] is not None:
            return max(0.0, float(meta["vla_input_speed_mps"]))
        has_collided = bool(meta.get("has_collided", False))
        assist = meta.get("startup_speed_assist_mps")
        if (
            assist is not None
            and not has_collided
            and ego_v <= 0.05
        ):
            return max(0.0, float(assist))
        return ego_v

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        native = self.predict_native(obs)
        upstream = UpstreamPathSpeed(
            path_xy=native.path_map_xy,
            speed_mps=native.speed_mps,
            frame="map",
        )
        tau = self.canonicalizer.canonicalize(upstream, to_map=False)
        return [
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=tau.points_xy_yaw_v_a_kappa,
                probability=1.0,
                uncertainty=0.15,
                candidate_id="tau0_neural",
                intended_action="nominal",
                behavior="follow",
            )
        ]

    def predict_native(self, obs: ObservationBundle) -> NativePathPrediction:
        """Return native spatial path + temporal speed without canonicalization."""
        if backend_name() in {"debug_geom", "fingerprint", "geom"}:
            raise RuntimeError("debug_geom backend disabled for production V0; unset SDF_VLA_BACKEND")
        self.ensure_loaded()
        assert self.runtime is not None
        rgb = obs.front_rgb
        if rgb is None:
            raise ValueError("neural V0 requires ObservationBundle.front_rgb (H,W,3 uint8)")
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        meta = obs.meta or {}
        official = bool(meta.get("official_contract", True))
        # Prefer explicit layout; image_bgr legacy flag ⇒ BGR.
        if meta.get("image_layout"):
            image_layout = str(meta["image_layout"]).lower()
        elif meta.get("image_bgr"):
            image_layout = "bgr"
        else:
            image_layout = "rgb"
        tp1, tp2 = self._route_targets(obs)
        # Expose for logging / debug demos
        if obs.meta is not None:
            try:
                obs.meta["resolved_target_ego_1"] = tp1
                obs.meta["resolved_target_ego_2"] = tp2
            except Exception:
                pass
        vla_speed = self.resolve_vla_input_speed_mps(obs)
        if obs.meta is not None:
            try:
                obs.meta["resolved_vla_input_speed_mps"] = vla_speed
            except Exception:
                pass
        from driving_vla.model.simlingo_runtime import SIMLINGO_CAMERA_XYZ

        cam_xyz = tuple(meta.get("camera_mount_xyz") or SIMLINGO_CAMERA_XYZ)
        # Official target_point mode: runtime uses command_text=None.
        # Do not invent "Command: follow the road" when meta explicitly has None.
        if "command_text" in meta:
            raw_cmd = meta.get("command_text")
            command_text = None if raw_cmd is None else str(raw_cmd)
        else:
            command_text = (
                None if official else "Command: follow the road."
            )
        if obs.meta is not None:
            try:
                obs.meta["resolved_command_text"] = command_text
                obs.meta["resolved_prompt_mode"] = (
                    "target_point" if official else "legacy_command_text"
                )
            except Exception:
                pass
        result = self.runtime.forward_numpy(
            arr,
            speed_mps=vla_speed,
            target_point_xy=tp1,
            target_point2_xy=tp2,
            keep_on_gpu=self.keep_on_gpu,
            borrow_gpu=not self.keep_on_gpu,
            camera_mount_xyz=(float(cam_xyz[0]), float(cam_xyz[1]), float(cam_xyz[2])),
            image_layout=image_layout,
            official_contract=official,
            command_text=command_text if command_text is not None else "Command: follow the road.",
        )
        self.last_latency_s = result.latency_s
        self.last_peak_vram_mb = result.peak_vram_mb
        path_map = self._ego_path_to_map(result.route_xy, obs)
        route_ego = tuple(
            (float(x), float(y)) for x, y in np.asarray(result.route_xy).reshape(-1, 2)
        )
        speed_wps = getattr(result, "speed_wps_xy", None)
        if speed_wps is None:
            speed_ego: tuple[tuple[float, float], ...] = ()
        else:
            speed_ego = tuple(
                (float(x), float(y)) for x, y in np.asarray(speed_wps).reshape(-1, 2)
            )
        return NativePathPrediction(
            path_map_xy=path_map,
            speed_mps=tuple(float(v) for v in result.speed_mps),
            target_ego_1=tp1,
            target_ego_2=tp2,
            latency_s=float(result.latency_s),
            peak_vram_mb=float(result.peak_vram_mb),
            route_ego_xy=route_ego,
            speed_wps_ego_xy=speed_ego,
        )


class NeuralV1Policy:
    model_id = "sdf-vla-v1-neural@0.1.0"
    source = "neural_simlingo"
    k = 2

    def __init__(self, runtime: SimLingoNeuralRuntime | None = None) -> None:
        self.v0 = NeuralV0Policy(runtime=runtime)

    def ensure_loaded(self) -> None:
        self.v0.ensure_loaded()

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        base = self.v0.predict_arrays(obs)[0]
        cur = (obs.ego_x, obs.ego_y, obs.ego_yaw, obs.ego_v)
        feats = _history_features(obs.ego_history, cur)
        hist_v = feats[3::4]
        mean_v = sum(hist_v) / len(hist_v)
        nom_scale = max(0.7, min(1.15, mean_v / max(obs.ego_v, 0.5) if obs.ego_v > 0.1 else 1.0))
        cons_scale = 0.65 * nom_scale
        nom = _apply_residual(base, nom_scale, 0.0)
        cons = _apply_residual(base, cons_scale, 0.0)
        return [
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=nom.points_xy_yaw_v_a_kappa,
                probability=0.62,
                uncertainty=0.12,
                candidate_id="v1_nominal",
                intended_action="nominal",
            ),
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=cons.points_xy_yaw_v_a_kappa,
                probability=0.38,
                uncertainty=0.22,
                candidate_id="v1_conservative",
                intended_action="conservative",
            ),
        ]
