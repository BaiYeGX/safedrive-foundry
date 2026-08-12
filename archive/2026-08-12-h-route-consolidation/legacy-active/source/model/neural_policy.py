"""V0/V1 policies backed by real SimLingo neural forward (default)."""

from __future__ import annotations

import math
import hashlib
import os
import tempfile
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
    driving_feature_raw_tensor_path: str = ""
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
        # The released checkpoint supports both target-point and HLC navigation.
        # Do not invent "Command: follow the road" when meta explicitly has None.
        raw_prompt_mode = str(
            meta.get("prompt_mode")
            or ("target_point" if official else "legacy_command_text")
        ).strip().lower()
        if official:
            if raw_prompt_mode in {"command", "hlc_command"}:
                eval_route_as = "command"
            elif raw_prompt_mode in {"target_point", "target_point_command"}:
                eval_route_as = "target_point"
            else:
                raise ValueError(
                    f"unsupported official SimLingo prompt_mode={raw_prompt_mode}"
                )
        else:
            eval_route_as = "target_point"
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
                    eval_route_as if official else "legacy_command_text"
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
            command_text=command_text,
            eval_route_as=eval_route_as,
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
            driving_feature_raw_tensor_path=str(
                getattr(result, "driving_feature_raw_tensor_path", "") or ""
            ),
            driving_feature_source=str(
                getattr(result, "driving_feature_source", "") or ""
            ),
            driving_feature_ok=bool(getattr(result, "driving_feature_ok", False)),
            driving_feature_error=str(
                getattr(result, "driving_feature_error", "") or ""
            ),
        )

    def dump_last_driving_tokens(self, path: str | Path) -> dict[str, Any]:
        """Persist V4 ordered tokens from the most recent same-forward."""
        self.ensure_loaded()
        assert self.runtime is not None
        return self.runtime.dump_last_driving_tokens(path)


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
            observable_scene=dict(
                (obs.meta or {}).get("observable_scene_v1") or {}
            ),
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


class NeuralV3Policy:
    """One SimLingo forward plus route-bound semantic teacher or learned head."""

    model_id = "sdf-vla-v3-mixed-semantic@0.1.0"
    source = "neural_simlingo_route_bound_semantic_k2"
    k = 2
    branch_type = "learned_route_bound_mixed_semantic"

    def __init__(
        self,
        runtime: SimLingoNeuralRuntime | None = None,
        *,
        semantic_head_checkpoint: str | Path | None = None,
        teacher_mode: bool = False,
        keep_on_gpu: bool = False,
        lazy: bool = True,
        device: str = "cpu",
        checkpoint_use: str = "development_live_smoke",
    ) -> None:
        self.v0 = NeuralV0Policy(
            runtime=runtime, lazy=lazy, keep_on_gpu=keep_on_gpu
        )
        self.teacher_mode = bool(teacher_mode)
        self.semantic_head_checkpoint = (
            str(semantic_head_checkpoint) if semantic_head_checkpoint else ""
        )
        if not self.teacher_mode:
            if not self.semantic_head_checkpoint:
                raise ValueError("K2 V3 learned mode requires semantic head checkpoint")
            from driving_vla.model.checkpoint_contract import (
                validate_checkpoint_for_use,
            )

            validate_checkpoint_for_use(
                self.semantic_head_checkpoint,
                checkpoint_use,
            )
            from driving_vla.model.semantic_mode_heads import (
                SpatialSemanticHeadRuntimeV3,
            )

            self.head = SpatialSemanticHeadRuntimeV3(
                device=device,
                checkpoint_path=self.semantic_head_checkpoint,
            )
        else:
            self.head = None
        self.last_bundle = None
        self.last_forward_count = 0
        self.last_latency_s = 0.0
        self.last_peak_vram_mb = 0.0
        self.last_native: NativePathPrediction | None = None

    def ensure_loaded(self) -> None:
        self.v0.ensure_loaded()

    def predict_bundle(self, obs: ObservationBundle):
        from driving_vla.model.k2_v3_guard import attach_k2_v3_guard
        from driving_vla.model.navigation_contract import (
            RouteContextV3,
            TargetLaneSide,
        )

        meta = dict(obs.meta or {})
        route_raw = meta.get("route_context_v3")
        if route_raw is None:
            raise ValueError("K2 V3 requires observable route_context_v3")
        route_context = (
            route_raw
            if isinstance(route_raw, RouteContextV3)
            else RouteContextV3.from_mapping(route_raw)
        )
        native = self.v0.predict_native(obs)
        self.last_native = native
        self.last_forward_count = 1
        self.last_latency_s = float(native.latency_s)
        self.last_peak_vram_mb = float(native.peak_vram_mb)
        ego_v = float(self.v0.resolve_vla_input_speed_mps(obs))
        base_speed = float(max(native.speed_mps)) if native.speed_mps else ego_v
        observation_identity = {
            "run_id": obs.run_id,
            "frame_id": obs.frame_id,
            "carla_frame": obs.carla_frame,
            "feature_content_hash": native.driving_feature_hash,
            "raw_feature_hash": native.driving_feature_raw_hash,
        }
        forward_id = (
            f"v3:{obs.frame_id}:{native.driving_feature_hash}:"
            f"{native.driving_feature_raw_hash}"
        )
        observable_scene = dict(meta.get("observable_scene_v1") or {})
        ego_route_error_m = (
            None
            if meta.get("ego_route_error_m") is None
            else float(meta["ego_route_error_m"])
        )
        overtake_actor_lon_m = (
            None
            if observable_scene.get("actor_lon_m") is None
            else float(observable_scene["actor_lon_m"])
        )
        overtake_phase_v3 = str(meta.get("overtake_phase_v3") or "")
        requested_overtake_side = TargetLaneSide(
            str(meta.get("requested_overtake_side") or "NONE")
        )
        conflict_active = (
            None
            if "conflict_active_v3" not in meta
            else bool(meta["conflict_active_v3"])
        )
        if self.teacher_mode:
            from driving_vla.model.semantic_k2_teacher import (
                build_semantic_teacher_bundle_v3,
                select_semantic_teacher_v3,
            )
            # A temporal yield is allowed to drive the native speed head to
            # zero.  Once the observable actor has cleared, the native
            # backbone can remain at that zero for several same-forward
            # frames, leaving the stateful speed planner's launch hysteresis
            # latched forever.  Match the learned V3 runtime's bounded resume
            # floor for interaction families; this changes only the temporal
            # execution speed after a proven clear, never the route or branch
            # kind/side.
            teacher_base_speed = base_speed
            teacher_family = str(meta.get("scenario_family_v3") or "").lower()
            if (
                conflict_active is False
                and bool(observable_scene.get("actor_present", False))
                and any(
                    token in teacher_family
                    for token in ("cut", "cross", "merge")
                )
            ):
                teacher_base_speed = max(teacher_base_speed, 2.0)
            label = select_semantic_teacher_v3(
                scenario_family=str(meta.get("scenario_family_v3") or "clear"),
                route_context=route_context,
                conflict_side=str(meta.get("conflict_side") or "none"),
                requested_overtake_side=requested_overtake_side,
                conflict_active=conflict_active,
            )
            bundle = build_semantic_teacher_bundle_v3(
                native_path_xy=native.path_map_xy,
                route_context=route_context,
                label=label,
                ego_v=ego_v,
                base_speed_mps=teacher_base_speed,
                backbone_forward_id=forward_id,
                observation_identity=observation_identity,
                ego_route_error_m=ego_route_error_m,
                overtake_actor_lon_m=overtake_actor_lon_m,
                overtake_phase_v3=overtake_phase_v3,
            )
        else:
            assert self.head is not None
            bundle = self.head.build_bundle(
                native_path_xy=native.path_map_xy,
                route_context=route_context,
                ego_v=ego_v,
                base_speed_mps=base_speed,
                driving_feature=native.driving_feature,
                observable_scene=observable_scene,
                observation_identity=observation_identity,
                backbone_forward_id=forward_id,
                base_checkpoint_hash=str(meta.get("base_checkpoint_hash") or "simlingo"),
                spatial_head_checkpoint_hash=str(
                    meta.get("semantic_head_checkpoint_hash")
                    or Path(self.semantic_head_checkpoint).name
                ),
                ego_route_error_m=ego_route_error_m,
                overtake_actor_lon_m=overtake_actor_lon_m,
                overtake_phase_v3=overtake_phase_v3,
                requested_overtake_side=requested_overtake_side,
                conflict_active=conflict_active,
                conflict_side=str(meta.get("conflict_side") or ""),
                scenario_family=str(meta.get("scenario_family_v3") or ""),
            )
        guarded = attach_k2_v3_guard(bundle)
        self.last_bundle = guarded
        return guarded

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        _ = self.predict_bundle(obs)
        return []


class NeuralV4Policy:
    """Scenario-blind ordered-token K2 V4 policy.

    One ``NeuralV0Policy.predict_native`` call supplies both the native path and
    the cached driving-adaptor tensor.  The V4 head consumes that exact tensor,
    writes an evidence-bound FP16 copy, and emits a V3-compatible bundle for
    the existing Guard/MPC boundary.  No scenario label or future observation
    is accepted on this path.
    """

    model_id = "sdf-vla-v4-spatial-k2@0.1.0"
    source = "neural_simlingo_ordered_tokens_v4"
    k = 2
    branch_type = "learned_route_bound_mixed_semantic_v4"

    def __init__(
        self,
        runtime: SimLingoNeuralRuntime | None = None,
        *,
        semantic_head_checkpoint: str | Path,
        keep_on_gpu: bool = False,
        lazy: bool = True,
        device: str = "cpu",
        availability_threshold: float | None = None,
        checkpoint_use: str = "development_live_smoke",
    ) -> None:
        from driving_vla.model.semantic_mode_heads_v4 import (
            SpatialSemanticHeadRuntimeV4,
        )

        checkpoint = Path(semantic_head_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        from driving_vla.model.checkpoint_contract import validate_checkpoint_for_use

        validate_checkpoint_for_use(checkpoint, checkpoint_use)
        self.v0 = NeuralV0Policy(
            runtime=runtime,
            lazy=lazy,
            keep_on_gpu=keep_on_gpu,
        )
        self.semantic_head_checkpoint = str(checkpoint)
        # Keep the checkpoint value authoritative when no explicit override is
        # supplied.  Calling ``float(None)`` here would make every collector
        # that relies on the frozen checkpoint threshold fail before CARLA is
        # even contacted.
        self.availability_threshold = (
            None if availability_threshold is None else float(availability_threshold)
        )
        self.head = SpatialSemanticHeadRuntimeV4(
            model=None,
            device=device,
            checkpoint_path=str(checkpoint),
        )
        if availability_threshold is not None and abs(
            float(availability_threshold) - float(self.head.availability_threshold)
        ) > 1.0e-9:
            raise ValueError(
                "V4 availability_threshold override does not match the frozen checkpoint"
            )
        self.availability_threshold = float(self.head.availability_threshold)
        self.collection_safe_stop = checkpoint_use in {
            "development_live_smoke",
            "collection_anchor",
        }
        self.last_bundle = None
        self.last_native: NativePathPrediction | None = None
        self.last_token_metadata: dict[str, Any] = {}
        self.last_forward_count = 0
        self.last_latency_s = 0.0
        self.last_peak_vram_mb = 0.0

    def ensure_loaded(self) -> None:
        self.v0.ensure_loaded()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _token_bundle_from_metadata(metadata: dict[str, Any]):
        from driving_vla.model.v4_token_features import DrivingTokenBundleV4

        return DrivingTokenBundleV4(
            ok=bool(metadata.get("ok", False)),
            schema_version=str(metadata.get("schema_version") or ""),
            adaptor_name=str(metadata.get("adaptor_name") or "driving"),
            raw_shape=tuple(int(value) for value in metadata.get("raw_shape") or ()),
            raw_dtype=str(metadata.get("raw_dtype") or ""),
            raw_content_hash=str(metadata.get("raw_content_hash") or ""),
            channel_dim=int(metadata.get("channel_dim") or 0),
            route_token_count=int(metadata.get("route_token_count") or 20),
            speed_token_count=int(metadata.get("speed_token_count") or 10),
            raw_tensor_path=str(metadata.get("raw_tensor_path") or ""),
            error=str(metadata.get("error") or ""),
        )

    def _token_path(self, obs: ObservationBundle) -> Path:
        meta = obs.meta or {}
        declared = meta.get("v4_raw_tensor_path")
        if declared:
            return Path(str(declared)).resolve()
        cache_root = Path(
            os.environ.get("SDF_V4_TOKEN_CACHE", tempfile.gettempdir())
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"sdf-v4-{obs.run_id}-{obs.frame_id}.npy"

    def predict_bundle(self, obs: ObservationBundle):
        from driving_vla.model.k2_v3_guard import attach_k2_v3_guard
        from driving_vla.model.navigation_contract import RouteContextV3

        meta = dict(obs.meta or {})
        if "scenario_family" in meta or "scenario_family_v3" in meta:
            raise ValueError("K2 V4 runtime forbids scenario_family conditioning")
        observable_scene_meta = dict(meta.get("observable_scene_v1") or {})
        if "family" in observable_scene_meta or "scenario_family" in observable_scene_meta:
            raise ValueError("K2 V4 observable scene forbids scenario_family fields")
        route_raw = meta.get("route_context_v3")
        if route_raw is None:
            raise ValueError("K2 V4 requires observable route_context_v3")
        route_context = (
            route_raw
            if isinstance(route_raw, RouteContextV3)
            else RouteContextV3.from_mapping(route_raw)
        )
        native = self.v0.predict_native(obs)
        self.last_native = native
        self.last_forward_count = 1
        self.last_latency_s = float(native.latency_s)
        self.last_peak_vram_mb = float(native.peak_vram_mb)
        if len(native.path_map_xy) < 2:
            raise RuntimeError("native_path_too_short_for_v4")
        token_path = self._token_path(obs)
        token_meta = self.v0.dump_last_driving_tokens(token_path)
        token_bundle = self._token_bundle_from_metadata(token_meta)
        token_bundle.require_ok()
        self.last_token_metadata = token_bundle.metadata()
        raw_head_hash = ""
        forward_id = (
            f"v4:{obs.frame_id}:{token_bundle.raw_content_hash}:"
            f"{native.driving_feature_raw_hash}"
        )
        base_checkpoint = str(meta.get("base_checkpoint_hash") or "simlingo")
        head_checkpoint_hash = str(
            meta.get("semantic_head_checkpoint_hash")
            or self._file_sha256(Path(self.semantic_head_checkpoint))
        )
        observation_identity = {
            "run_id": obs.run_id,
            "frame_id": obs.frame_id,
            "carla_frame": obs.carla_frame,
            "feature_content_hash": token_bundle.raw_content_hash,
            "raw_feature_hash": native.driving_feature_raw_hash,
            "v4_token_raw_tensor_path": str(token_path),
        }
        bundle = self.head.build_bundle(
            token_bundle=token_bundle,
            native_path_xy=native.path_map_xy,
            route_context=route_context,
            ego_v=float(self.v0.resolve_vla_input_speed_mps(obs)),
            base_speed_mps=float(max(native.speed_mps)) if native.speed_mps else float(obs.ego_v),
            nominal_target_speed_mps=(
                0.0
                if self.collection_safe_stop
                and str(route_context.traffic_signal_state.value)
                in {"RED", "STOP_SIGN"}
                and route_context.stop_line_distance_m is not None
                else None
            ),
            observable_scene=observable_scene_meta,
            observation_identity=observation_identity,
            backbone_forward_id=forward_id,
            base_checkpoint_hash=base_checkpoint,
            semantic_head_checkpoint_hash=head_checkpoint_hash,
            availability_threshold=self.availability_threshold,
        )
        raw_head_hash = str(bundle.observation_identity.get("raw_head_output_hash") or "")
        if not raw_head_hash:
            raise RuntimeError("v4 raw head output hash missing")
        guarded = attach_k2_v3_guard(bundle)
        self.last_bundle = guarded
        return guarded

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        _ = self.predict_bundle(obs)
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
