"""V0/V1 policies backed by real SimLingo neural forward (default)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.model.canonicalizer import TrajectoryCanonicalizer, UpstreamPathSpeed
from driving_vla.model.k2_builder import (
    K2Builder,
    K2BuilderConfig,
    K2PredictionBundle,
    attach_guard,
    load_k2_config,
)
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime


def backend_name() -> str:
    return os.environ.get("SDF_VLA_BACKEND", "neural").strip().lower()


@dataclass(frozen=True)
class NativePathPrediction:
    """Unmodified SimLingo action heads in the project map frame.

    X5A: same-forward driving features must be plumbed (not dropped).
    """

    path_map_xy: tuple[tuple[float, float], ...]
    speed_mps: tuple[float, ...]
    target_ego_1: tuple[float, float]
    target_ego_2: tuple[float, float]
    latency_s: float
    peak_vram_mb: float
    # Optional dual-head ego-frame dumps for debug-draw / evidence.
    route_ego_xy: tuple[tuple[float, float], ...] = ()
    speed_wps_ego_xy: tuple[tuple[float, float], ...] = ()
    # Same-forward Spatial K2 features (mean64 + lineage)
    driving_feature: tuple[float, ...] = ()
    driving_feature_hash: str = ""
    driving_feature_full_pool: tuple[float, ...] = ()
    driving_feature_full_pool_hash: str = ""
    driving_feature_raw_shape: tuple[int, ...] = ()
    driving_feature_raw_dtype: str = ""
    driving_feature_raw_hash: str = ""
    driving_feature_source: str = ""
    driving_feature_ok: bool = False
    driving_feature_error: str = ""


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
            driving_feature=tuple(
                float(x) for x in (getattr(result, "driving_feature", ()) or ())
            ),
            driving_feature_hash=str(getattr(result, "driving_feature_hash", "") or ""),
            driving_feature_full_pool=tuple(
                float(x) for x in (getattr(result, "driving_feature_full_pool", ()) or ())
            ),
            driving_feature_full_pool_hash=str(
                getattr(result, "driving_feature_full_pool_hash", "") or ""
            ),
            driving_feature_raw_shape=tuple(
                int(x) for x in (getattr(result, "driving_feature_raw_shape", ()) or ())
            ),
            driving_feature_raw_dtype=str(
                getattr(result, "driving_feature_raw_dtype", "") or ""
            ),
            driving_feature_raw_hash=str(
                getattr(result, "driving_feature_raw_hash", "") or ""
            ),
            driving_feature_source=str(
                getattr(result, "driving_feature_source", "") or ""
            ),
            driving_feature_ok=bool(getattr(result, "driving_feature_ok", False)),
            driving_feature_error=str(
                getattr(result, "driving_feature_error", "") or ""
            ),
        )


class NeuralV2Policy:
    """Spatial K2: one SimLingo forward → real driving feature → dual residual heads.

    Does **not** change V1 longitudinal semantics. Fail-closed on missing feature.
    """

    model_id = "sdf-vla-v2-spatial@0.2.0"
    source = "neural_simlingo_spatial_k2"
    k = 2
    branch_type = "learned_spatial_semantic"

    def __init__(
        self,
        runtime: SimLingoNeuralRuntime | None = None,
        *,
        spatial_head_checkpoint: str | Path | None = None,
        keep_on_gpu: bool = False,
        lazy: bool = True,
        require_driving_feature: bool = True,
        device: str = "cpu",
        checkpoint_use: str = "formal_offline",
        skip_checkpoint_contract: bool = False,
    ) -> None:
        from pathlib import Path as _Path

        from driving_vla.model.k2_spatial_builder import build_spatial_k2_bundle_from_residuals
        from driving_vla.model.k2_spatial_guard import attach_spatial_guard
        from driving_vla.model.spatial_mode_heads import SpatialK2HeadRuntime

        self.v0 = NeuralV0Policy(runtime=runtime, lazy=lazy, keep_on_gpu=keep_on_gpu)
        self.require_driving_feature = bool(require_driving_feature)
        ckpt = str(spatial_head_checkpoint) if spatial_head_checkpoint else None
        if ckpt and not _Path(ckpt).is_file():
            ckpt = None
        self.head = SpatialK2HeadRuntime(
            device=device,
            checkpoint_path=ckpt,
            checkpoint_use=checkpoint_use,
            skip_checkpoint_contract=skip_checkpoint_contract,
        )
        self._build = build_spatial_k2_bundle_from_residuals
        self._guard = attach_spatial_guard
        self.last_bundle = None
        self.last_forward_count: int = 0
        self.last_latency_s: float = 0.0
        self.last_peak_vram_mb: float = 0.0
        self.last_native: NativePathPrediction | None = None

    def ensure_loaded(self) -> None:
        self.v0.ensure_loaded()

    def predict_bundle(self, obs: ObservationBundle):
        """One neural forward + spatial heads + Guard V2."""
        from driving_vla.model.driving_feature import DrivingFeatureError

        native = self.v0.predict_native(obs)
        self.last_native = native
        self.last_forward_count = 1
        self.last_latency_s = float(native.latency_s)
        self.last_peak_vram_mb = float(native.peak_vram_mb)
        if self.require_driving_feature and not native.driving_feature_ok:
            raise DrivingFeatureError(
                native.driving_feature_error or "driving_feature_not_ok_for_v2"
            )
        if self.require_driving_feature and not native.driving_feature:
            raise DrivingFeatureError("driving_feature_empty_for_v2")

        path = list(native.path_map_xy)
        if len(path) < 2:
            raise RuntimeError("native_path_too_short_for_spatial_k2")
        ego_v = float(self.v0.resolve_vla_input_speed_mps(obs))
        base_speed = float(max(native.speed_mps)) if native.speed_mps else ego_v
        o0, o1 = self.head.predict_modes(
            path,
            ego_v=ego_v,
            base_speed_mps=base_speed,
            driving_feature=list(native.driving_feature),
        )
        # Contract (R2-X repair A): learned path is raw head residual only +
        # declared deterministic codec. NO runtime lattice/hump rescue.
        # Collapse → mark unavailable; Guard may still enforce diversity.
        import hashlib
        from driving_vla.model.spatial_mode_heads import (
            decoded_peak_lateral_separation,
        )

        # Candidate 0 is the exact native anchor.  Diversity is therefore the
        # defensive residual relative to zero, never two learned modes drifting
        # together in the same direction.
        nominal_anchor_d = [0.0] * len(o1.raw_d)
        peak_sep = decoded_peak_lateral_separation(nominal_anchor_d, o1.raw_d)
        d0_list = [float(x) for x in o0.raw_d]
        d1_list = [float(x) for x in o1.raw_d]
        # Fail-closed on spatial collapse: never invent lateral templates.
        # Frozen selection-space floor = Guard ambiguity_min_spatial_sep_m = 0.50 m.
        # Do NOT lower this to "manufacture" dual-force / selection-space claims.
        COLLAPSE_SEP_M = 0.50
        sep_ok = peak_sep >= COLLAPSE_SEP_M
        # Proposal validity is executability-only. Learned availability remains
        # a non-blocking confidence for downstream ranking and diagnostics.
        def_avail = bool(sep_ok)
        if def_avail:
            def_reason = "PROPOSAL_SPATIALLY_DISTINCT"
        else:
            def_reason = "HEAD_COLLAPSE_SEP"
        # Diversity eligibility tracks availability; Guard enforces SPATIAL_COLLAPSE
        # when eligible — do not mark available while disabling diversity checks.
        diversity_eligible = bool(def_avail)
        from driving_vla.model.k2_spatial_types import canonical_sha256

        raw_head_payload = {
            "d0": d0_list,
            "d1": d1_list,
            "ds0": [float(x) for x in o0.raw_delta_s],
            "ds1": [float(x) for x in o1.raw_delta_s],
            "sp0": float(o0.speed_scale),
            "sp1": float(o1.speed_scale),
            "avail": bool(o1.available),
            "avail_prob": float(o1.avail_prob),
        }
        raw_head_hash = canonical_sha256(raw_head_payload)
        _ = hashlib  # keep import used if needed
        fwd_id = f"v2-{native.driving_feature_hash}-{native.driving_feature_raw_hash}"
        bundle = self._build(
            native_path_xy=path,
            ego_xy=(float(path[0][0]), float(path[0][1])),
            ego_v=ego_v,
            base_speed_mps=base_speed,
            residual_nominal={
                "raw_delta_s": [0.0] * len(path),
                "raw_d": [0.0] * len(path),
                "speed_scale": 1.0,
                "head_lineage": "native_nominal_anchor",
                "raw_head_hash": raw_head_hash,
            },
            residual_defensive={
                "raw_delta_s": o1.raw_delta_s,
                "raw_d": d1_list,
                "speed_scale": o1.speed_scale,
                "head_lineage": "spatial_mode_head",
                "raw_head_hash": raw_head_hash,
            },
            observation_identity={
                "feature_hash": native.driving_feature_hash,
                "raw_feature_hash": native.driving_feature_raw_hash,
                "feature_source": native.driving_feature_source,
            },
            backbone_forward_id=fwd_id,
            model_id=self.model_id,
            spatial_head_checkpoint_hash=self.head.spatial_head_checkpoint_hash,
            defensive_available=def_avail,
            defensive_reason=def_reason,
            nominal_probability=1.0 - float(o1.avail_prob),
            defensive_probability=float(o1.avail_prob),
            probability_source="learned_mode_confidence_non_blocking",
        )
        from dataclasses import replace

        bundle = replace(
            bundle,
            set_diagnostics={
                **dict(bundle.set_diagnostics),
                "eligible_for_diversity": diversity_eligible,
                "peak_residual_lat_sep": peak_sep,
                "collapse_sep_m": COLLAPSE_SEP_M,
                "raw_head_hash": raw_head_hash,
                "learned_defensive_confidence": float(o1.avail_prob),
                "learned_availability_decision": bool(o1.available),
                "availability_semantics": "executability_only_v1",
                "learned_confidence_non_blocking": True,
                "runtime_rescue": False,
                "nominal_anchor_exact": True,
                "nominal_head_output_audit_only": True,
                "driving_feature_ok": native.driving_feature_ok,
                "driving_feature_hash": native.driving_feature_hash,
            },
        )
        guarded = self._guard(
            bundle, require_diversity_if_eligible=diversity_eligible
        )
        self.last_bundle = guarded
        return guarded

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        bundle = self.predict_bundle(obs)
        # V2 candidates are not TrajectoryArray; return empty for V1-compat callers
        _ = bundle
        return []


class NeuralV1Policy:
    """Real SimLingo K2: one ``predict_native`` then longitudinal path retiming.

    branch_type=longitudinal_temporal (deterministic; not learned multimodal K2).
    """

    model_id = "sdf-vla-v1-neural@0.1.0"
    source = "neural_simlingo"
    k = 2
    branch_type = "longitudinal_temporal"

    def __init__(
        self,
        runtime: SimLingoNeuralRuntime | None = None,
        *,
        k2_config: K2BuilderConfig | None = None,
        keep_on_gpu: bool = False,
        lazy: bool = True,
    ) -> None:
        self.v0 = NeuralV0Policy(runtime=runtime, lazy=lazy, keep_on_gpu=keep_on_gpu)
        self.k2_config = k2_config or load_k2_config()
        self._builder = K2Builder(self.k2_config)
        self.last_bundle: K2PredictionBundle | None = None
        self.last_forward_count: int = 0
        self.last_latency_s: float = 0.0
        self.last_peak_vram_mb: float = 0.0

    def ensure_loaded(self) -> None:
        self.v0.ensure_loaded()

    def predict_bundle(self, obs: ObservationBundle) -> K2PredictionBundle:
        """One neural forward → K2 bundle + Contract Guard."""
        native = self.v0.predict_native(obs)
        self.last_forward_count = 1
        self.last_latency_s = float(getattr(native, "latency_s", self.v0.last_latency_s))
        self.last_peak_vram_mb = float(
            getattr(native, "peak_vram_mb", self.v0.last_peak_vram_mb)
        )
        bundle = self._builder.build(native, obs, model_id=self.model_id)
        bundle = attach_guard(bundle, self.k2_config)
        self.last_bundle = bundle
        return bundle

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        """Compatibility wrapper: candidates from :meth:`predict_bundle`."""
        bundle = self.predict_bundle(obs)
        return list(bundle.candidates)
