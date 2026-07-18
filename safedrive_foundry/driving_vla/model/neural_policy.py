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
        """Two ego-frame navigation targets always *ahead* along route (~d1, ~d2 meters).

        Fixes the classic bug of fixed route_xy[10]: after the car passes that point,
        ego_x becomes negative (behind) and VLA is prompted to U-turn / circle.
        Override via obs.meta['target_ego_1'] / ['target_ego_2'] if provided.
        """
        meta = obs.meta or {}
        if "target_ego_1" in meta and "target_ego_2" in meta:
            t1 = meta["target_ego_1"]
            t2 = meta["target_ego_2"]
            return (float(t1[0]), float(t1[1])), (float(t2[0]), float(t2[1]))

        if not obs.route_xy or len(obs.route_xy) < 2:
            return (float(d1), 0.0), (float(d2), 0.0)

        # Arc-length table on map route
        poly = [(float(x), float(y)) for x, y in obs.route_xy]
        s_acc = [0.0]
        for i in range(1, len(poly)):
            s_acc.append(
                s_acc[-1] + math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1])
            )
        # Nearest s, then advance (prefer forward when ties)
        best_s, best_d = 0.0, float("inf")
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            vx, vy = x1 - x0, y1 - y0
            L2 = vx * vx + vy * vy
            if L2 < 1e-12:
                t = 0.0
            else:
                t = max(0.0, min(1.0, ((obs.ego_x - x0) * vx + (obs.ego_y - y0) * vy) / L2))
            px, py = x0 + t * vx, y0 + t * vy
            d = math.hypot(obs.ego_x - px, obs.ego_y - py)
            if d < best_d:
                best_d = d
                best_s = s_acc[i] + t * (s_acc[i + 1] - s_acc[i])

        def _at_s(s: float) -> tuple[float, float]:
            s = max(0.0, min(s, s_acc[-1]))
            if s <= 0:
                return poly[0]
            if s >= s_acc[-1]:
                return poly[-1]
            for i in range(1, len(s_acc)):
                if s_acc[i] >= s:
                    u = (s - s_acc[i - 1]) / max(s_acc[i] - s_acc[i - 1], 1e-9)
                    return (
                        poly[i - 1][0] + u * (poly[i][0] - poly[i - 1][0]),
                        poly[i - 1][1] + u * (poly[i][1] - poly[i - 1][1]),
                    )
            return poly[-1]

        # Always place goals ahead of current progress (not a fixed index)
        s1 = min(s_acc[-1], best_s + d1)
        s2 = min(s_acc[-1], best_s + d2)
        # If near end of route, keep targets at least slightly ahead of ego in ego frame
        m1 = _at_s(s1)
        m2 = _at_s(s2)
        e1 = self._map_to_ego(m1[0], m1[1], obs)
        e2 = self._map_to_ego(m2[0], m2[1], obs)
        # Clamp: if numerical glitch put target behind, fall back to ego-forward
        if e1[0] < 1.0:
            e1 = (float(d1), 0.0)
        if e2[0] <= e1[0] + 1.0:
            e2 = (float(e1[0] + max(5.0, d2 - d1)), float(e1[1]))
        return e1, e2

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
        # BGR carla often — if mean channel0 > channel2 slightly keep as is; convert if meta says bgr
        if obs.meta.get("image_bgr"):
            arr = arr[:, :, ::-1].copy()
        tp1, tp2 = self._route_targets(obs)
        # Expose for logging / debug demos
        if obs.meta is not None:
            try:
                obs.meta["resolved_target_ego_1"] = tp1
                obs.meta["resolved_target_ego_2"] = tp2
            except Exception:
                pass
        result = self.runtime.forward_numpy(
            arr,
            speed_mps=float(obs.ego_v if obs.ego_v > 0.05 else 3.0),
            target_point_xy=tp1,
            target_point2_xy=tp2,
            keep_on_gpu=self.keep_on_gpu,
            borrow_gpu=not self.keep_on_gpu,
        )
        self.last_latency_s = result.latency_s
        self.last_peak_vram_mb = result.peak_vram_mb
        path_map = self._ego_path_to_map(result.route_xy, obs)
        return NativePathPrediction(
            path_map_xy=path_map,
            speed_mps=tuple(float(v) for v in result.speed_mps),
            target_ego_1=tp1,
            target_ego_2=tp2,
            latency_s=float(result.latency_s),
            peak_vram_mb=float(result.peak_vram_mb),
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
