"""Token-aware, scenario-blind semantic head for R2 K2 V4.

The V4 head is additive to V3.  It consumes the ordered route/speed query
tokens from one SimLingo forward plus runtime-observable history/topology.  It
does not accept scenario-family labels or any oracle/future fields, and it
never mutates the upstream route.  Candidate decoding remains a separate
Guard/codec responsibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from driving_vla.model.driving_feature import observable_scene_vector
from driving_vla.model.k2_v3_codec import build_k2_v3_bundle
from driving_vla.model.k2_v3_types import K2PredictionBundleV3
from driving_vla.model.k2_v3_types import AlternativeKind
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    TargetLaneSide,
)
from driving_vla.model.semantic_mode_heads import (
    KIND_ORDER,
    NAV_CONTEXT_DIM,
    ROUTE_ORDER,
    SIDE_ORDER,
    navigation_context_vector,
)
from driving_vla.model.spatial_mode_heads import geometry_context_vector
from driving_vla.model.v4_token_features import (
    DrivingTokenBundleV4,
    TOTAL_TOKEN_COUNT,
)

V4_HEAD_SCHEMA = "safedrive.k2.semantic_head.v4"
TOKEN_EMBED_DIM = 128
AUX_DIM = 178
MANEUVER_PARAM_COUNT = 6
V4_KIND_ORDER = (
    AlternativeKind.NONE,
    AlternativeKind.SPATIAL_AVOID,
    AlternativeKind.SPATIAL_OVERTAKE,
    AlternativeKind.TEMPORAL_YIELD,
)
V4_SPATIAL_KINDS = frozenset(
    {AlternativeKind.SPATIAL_AVOID, AlternativeKind.SPATIAL_OVERTAKE}
)


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _bounded(value: Any, low: float, high: float) -> float:
    return max(float(low), min(float(high), _finite(value)))


def _history_point(row: Mapping[str, Any] | None) -> list[float]:
    if not row or not bool(row.get("valid", True)):
        return [0.0] * 11
    yaw = _finite(row.get("yaw_rad"))
    return [
        _bounded(row.get("x"), -50.0, 50.0) / 50.0,
        _bounded(row.get("y"), -20.0, 20.0) / 20.0,
        math.sin(yaw),
        math.cos(yaw),
        _bounded(row.get("vx"), -20.0, 20.0) / 20.0,
        _bounded(row.get("vy"), -20.0, 20.0) / 20.0,
        _bounded(row.get("ax"), -10.0, 10.0) / 10.0,
        _bounded(row.get("ay"), -10.0, 10.0) / 10.0,
        _bounded(row.get("yaw_rate"), -4.0, 4.0) / 4.0,
        1.0,
        _bounded(row.get("dt"), -2.0, 0.0) / 2.0,
    ]


def _current_actor(scene: Mapping[str, Any]) -> dict[str, Any] | None:
    if not bool(scene.get("actor_present")):
        return None
    lon = scene.get("actor_lon_m")
    lat = scene.get("actor_lat_m")
    speed = scene.get("actor_speed_mps")
    if lon is None or lat is None or speed is None:
        return None
    return {
        "x": _finite(lon),
        "y": _finite(lat),
        "vx": _finite(speed),
        "vy": 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "yaw_rad": 0.0,
        "yaw_rate": 0.0,
        "valid": True,
        "dt": 0.0,
    }


def _history_vector(scene: Mapping[str, Any]) -> list[float]:
    ego_rows = list(scene.get("ego_history") or [])[-5:]
    ego_rows = ([None] * (5 - len(ego_rows))) + ego_rows
    actor_groups = list(scene.get("actor_histories") or [])
    actor_group = actor_groups[0] if actor_groups else None
    actor_rows = list((actor_group or {}).get("history") or [])[-5:]
    if not actor_rows and (current := _current_actor(scene)) is not None:
        actor_rows = [current]
    actor_rows = ([None] * (5 - len(actor_rows))) + actor_rows
    actor_features = [item for row in actor_rows for item in _history_point(row)]
    actor_features.extend(
        [
            _bounded((actor_group or {}).get("length", 4.5), 0.0, 8.0) / 8.0,
            _bounded((actor_group or {}).get("width", 1.8), 0.0, 4.0) / 4.0,
        ]
    )
    return [item for row in ego_rows for item in _history_point(row)] + actor_features


def build_v4_aux_vector(
    *,
    native_path_xy: Sequence[tuple[float, float]],
    route_context: RouteContextV3,
    ego_v: float,
    base_speed_mps: float,
    observable_scene: Mapping[str, Any] | None,
) -> list[float]:
    """Build the fixed 178-dim observable/context branch for V4."""
    scene = dict(observable_scene or {})
    if "scenario_family" in scene or "family" in scene:
        raise ValueError("V4 observable scene forbids scenario_family conditioning")
    result = list(
        geometry_context_vector(
            native_path_xy,
            ego_v=float(ego_v),
            base_speed_mps=float(base_speed_mps),
            dim=32,
        )
    )
    result.extend(navigation_context_vector(route_context))
    result.extend(_history_vector(scene))
    result.extend(observable_scene_vector(scene, ego_v=float(ego_v)))
    result.extend(
        [
            _bounded(ego_v, 0.0, 20.0) / 20.0,
            _bounded(base_speed_mps, 0.0, 20.0) / 20.0,
        ]
    )
    if len(result) != AUX_DIM:
        raise AssertionError(f"V4 aux size mismatch: {len(result)} != {AUX_DIM}")
    return [float(value) for value in result]


class SpatialSemanticHeadV4(nn.Module):
    """Small ordered-token encoder with structured observable fusion."""

    def __init__(
        self,
        *,
        token_dim: int,
        aux_dim: int = AUX_DIM,
        embed_dim: int = TOKEN_EMBED_DIM,
        hidden: int = 192,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.token_dim = int(token_dim)
        self.aux_dim = int(aux_dim)
        self.embed_dim = int(embed_dim)
        self.token_type = nn.Embedding(2, embed_dim)
        self.token_position = nn.Embedding(TOTAL_TOKEN_COUNT, embed_dim)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, embed_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            # Keep the small CPU fallback deterministic and avoid PyTorch's
            # nested-tensor fast-path warning on every offline validation.
            norm_first=False,
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.aux_encoder = nn.Sequential(
            nn.LayerNorm(aux_dim),
            nn.Linear(aux_dim, embed_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.kind = nn.Linear(hidden, len(KIND_ORDER) - 1)  # no nominal output
        self.side = nn.Linear(hidden, len(SIDE_ORDER))
        self.available = nn.Linear(hidden, 1)
        self.maneuver = nn.Linear(hidden, MANEUVER_PARAM_COUNT)

    def forward(self, tokens: torch.Tensor, aux: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != TOTAL_TOKEN_COUNT:
            raise ValueError(f"V4 tokens must be [B,{TOTAL_TOKEN_COUNT},H]")
        if tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"V4 token channel mismatch: {tokens.shape[2]} != {self.token_dim}"
            )
        if aux.ndim != 2 or aux.shape[1] != self.aux_dim:
            raise ValueError(f"V4 aux must be [B,{self.aux_dim}]")
        positions = torch.arange(TOTAL_TOKEN_COUNT, device=tokens.device)
        types = torch.cat(
            [
                torch.zeros(20, dtype=torch.long, device=tokens.device),
                torch.ones(10, dtype=torch.long, device=tokens.device),
            ]
        )
        projected = self.token_projection(tokens)
        projected = projected + self.token_type(types)[None, :, :]
        projected = projected + self.token_position(positions)[None, :, :]
        if getattr(self, "_v4_gradient_checkpointing", False) and self.training:
            # The fallback LoRA path explicitly enables this flag.  Keeping
            # the loop here makes checkpointing cover exactly the same encoder
            # layers as the ordinary TransformerEncoder call.
            from torch.utils.checkpoint import checkpoint

            encoded = projected
            for layer in self.token_encoder.layers:
                encoded = checkpoint(layer, encoded, use_reentrant=False)
            if self.token_encoder.norm is not None:
                encoded = self.token_encoder.norm(encoded)
        else:
            encoded = self.token_encoder(projected)
        pooled = torch.cat(
            [encoded[:, :20].mean(dim=1), encoded[:, 20:].mean(dim=1), encoded.mean(dim=1)],
            dim=-1,
        )
        fused = self.fusion(torch.cat([pooled, self.aux_encoder(aux)], dim=-1))
        return {
            "kind_logits": self.kind(fused),
            "side_logits": self.side(fused),
            "avail_logit": self.available(fused).squeeze(-1),
            "maneuver_params": torch.sigmoid(self.maneuver(fused)),
        }


@dataclass(frozen=True)
class SemanticHeadOutputV4:
    alternative_kind: AlternativeKind
    target_lane_side: TargetLaneSide
    kind_probability: float
    side_probability: float
    availability_probability: float
    maneuver_params: tuple[float, ...]
    raw_head_output_hash: str
    semantic_rescue: bool = False

    @property
    def avoid_offset_m(self) -> float:
        return 0.50 + 0.50 * self.maneuver_params[0]

    @property
    def temporal_speed_scale(self) -> float:
        return 0.80 * self.maneuver_params[1]

    @property
    def departure_start(self) -> float:
        return 0.05 + 0.20 * self.maneuver_params[2]

    @property
    def departure_end(self) -> float:
        return self.departure_start + 0.15 + 0.15 * self.maneuver_params[3]

    @property
    def rejoin_start(self) -> float:
        return max(self.departure_end + 0.15, 0.50 + 0.20 * self.maneuver_params[4])

    @property
    def rejoin_end(self) -> float:
        return min(0.98, self.rejoin_start + 0.15 + 0.15 * self.maneuver_params[5])


class SpatialSemanticHeadRuntimeV4:
    """Pure runtime inference; no scenario-family or outcome override path."""

    def __init__(
        self,
        model: SpatialSemanticHeadV4 | None = None,
        *,
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        checkpoint: Mapping[str, Any] | None = None
        if checkpoint_path:
            loaded = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            if isinstance(loaded, Mapping):
                checkpoint = loaded
            else:
                checkpoint = {"model": loaded}
        if model is None:
            if checkpoint is None:
                raise ValueError("V4 runtime requires model or checkpoint_path")
            kwargs = dict(checkpoint.get("model_kwargs") or {})
            if "token_dim" not in kwargs:
                raise ValueError("V4 checkpoint missing model_kwargs.token_dim")
            model = SpatialSemanticHeadV4(**kwargs)
        lora_payload = checkpoint.get("lora") if checkpoint is not None else None
        if lora_payload:
            from driving_vla.model.v4_lora import V4LoRAConfig, apply_v4_lora_qv

            config = V4LoRAConfig(**dict(lora_payload.get("config") or lora_payload))
            apply_v4_lora_qv(model, config)
        self.model = model.to(self.device)
        self.model.eval()
        self.availability_threshold = 0.5
        self.aux_mean: torch.Tensor | None = None
        self.aux_std: torch.Tensor | None = None
        if checkpoint is not None:
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            normalization = checkpoint.get("normalization") or {}
            threshold = float(checkpoint.get("availability_threshold", 0.5))
            if not 0.0 < threshold < 1.0:
                raise ValueError("V4 checkpoint availability_threshold must be in (0,1)")
            self.availability_threshold = threshold
            if normalization:
                mean = torch.as_tensor(normalization.get("mean"), dtype=torch.float32)
                std = torch.as_tensor(normalization.get("std"), dtype=torch.float32)
                if tuple(mean.shape) != (AUX_DIM,) or tuple(std.shape) != (AUX_DIM,):
                    raise ValueError("V4 checkpoint normalization shape mismatch")
                self.aux_mean = mean.to(self.device)
                self.aux_std = torch.clamp(std.to(self.device), min=1.0e-4)
            self.model.eval()

    @staticmethod
    def _raw_hash(output: Mapping[str, torch.Tensor]) -> str:
        payload = {
            key: [[float(value) for value in row] for row in tensor.detach().cpu().tolist()]
            if tensor.ndim > 1
            else [float(value) for value in tensor.detach().cpu().tolist()]
            for key, tensor in sorted(output.items())
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @torch.inference_mode()
    def predict(
        self,
        *,
        token_bundle: DrivingTokenBundleV4,
        native_path_xy: Sequence[tuple[float, float]],
        route_context: RouteContextV3,
        ego_v: float,
        base_speed_mps: float,
        observable_scene: Mapping[str, Any] | None,
    ) -> SemanticHeadOutputV4:
        token_bundle.require_ok()
        tokens = token_bundle.load_tokens()
        aux = build_v4_aux_vector(
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            observable_scene=observable_scene,
        )
        token_tensor = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)[None]
        aux_tensor = torch.as_tensor(aux, dtype=torch.float32, device=self.device)[None]
        if self.aux_mean is not None and self.aux_std is not None:
            aux_tensor = (aux_tensor - self.aux_mean[None, :]) / self.aux_std[None, :]
        output = self.model(token_tensor, aux_tensor)
        kind_prob = torch.softmax(output["kind_logits"][0], dim=-1)
        side_prob = torch.softmax(output["side_logits"][0], dim=-1)
        kind_index = int(torch.argmax(kind_prob).item())
        side_index = int(torch.argmax(side_prob).item())
        kind_order = V4_KIND_ORDER
        predicted_kind = kind_order[kind_index]
        # Side is a spatial-only head.  The raw logits remain part of the
        # output hash for audit, but a non-spatial proposal can never carry a
        # lane side into the codec or Guard.
        predicted_side = (
            SIDE_ORDER[side_index]
            if predicted_kind in V4_SPATIAL_KINDS
            else TargetLaneSide.NONE
        )
        side_probability = (
            float(side_prob[side_index].item())
            if predicted_kind in V4_SPATIAL_KINDS
            else float(side_prob[SIDE_ORDER.index(TargetLaneSide.NONE)].item())
        )
        params = tuple(float(value) for value in output["maneuver_params"][0].tolist())
        return SemanticHeadOutputV4(
            alternative_kind=predicted_kind,
            target_lane_side=predicted_side,
            kind_probability=float(kind_prob[kind_index].item()),
            side_probability=side_probability,
            availability_probability=float(torch.sigmoid(output["avail_logit"][0]).item()),
            maneuver_params=params,
            raw_head_output_hash=self._raw_hash(output),
        )

    def build_bundle(
        self,
        *,
        token_bundle: DrivingTokenBundleV4,
        native_path_xy: Sequence[tuple[float, float]],
        route_context: RouteContextV3,
        ego_v: float,
        base_speed_mps: float,
        observable_scene: Mapping[str, Any] | None,
        observation_identity: Mapping[str, Any],
        backbone_forward_id: str,
        base_checkpoint_hash: str,
        semantic_head_checkpoint_hash: str,
        nominal_target_speed_mps: float | None = None,
        availability_threshold: float | None = None,
    ) -> K2PredictionBundleV3:
        """Decode one raw head output into the existing guarded K2 contract.

        This method is intentionally a thin codec boundary: the predicted kind
        and side are copied verbatim, and no scenario label, actor future, or
        rule-based rescue is consulted.  The caller must run Contract Guard
        afterwards; a rejected proposal is evidence, not a rewritten output.
        """
        token_bundle.require_ok()
        prediction = self.predict(
            token_bundle=token_bundle,
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            observable_scene=observable_scene,
        )
        aux_vector = build_v4_aux_vector(
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            observable_scene=observable_scene,
        )
        threshold = (
            self.availability_threshold
            if availability_threshold is None
            else float(availability_threshold)
        )
        if not 0.0 < threshold < 1.0:
            raise ValueError("V4 availability_threshold must be in (0,1)")
        available = bool(
            prediction.alternative_kind is not AlternativeKind.NONE
            and prediction.availability_probability
            >= threshold
        )
        identity = dict(observation_identity)
        identity.update(
            {
                "v4_token_schema": token_bundle.schema_version,
                "v4_token_raw_shape": list(token_bundle.raw_shape),
                "v4_token_raw_dtype": token_bundle.raw_dtype,
                "v4_token_raw_content_hash": token_bundle.raw_content_hash,
                "v4_token_route_count": token_bundle.route_token_count,
                "v4_token_speed_count": token_bundle.speed_token_count,
                "v4_aux": list(aux_vector),
                "raw_head_output_hash": prediction.raw_head_output_hash,
                "semantic_rescue_count": 0,
                "scenario_family_runtime_use": 0,
            }
        )
        bundle = build_k2_v3_bundle(
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=float(ego_v),
            base_speed_mps=float(base_speed_mps),
            nominal_target_speed_mps=nominal_target_speed_mps,
            alternative_kind=prediction.alternative_kind,
            alternative_available=available,
            alternative_reason=(
                "HEAD_AVAILABILITY_THRESHOLD"
                if available
                else "HEAD_UNAVAILABLE_OR_NONE"
            ),
            target_lane_side=prediction.target_lane_side,
            avoid_offset_m=prediction.avoid_offset_m,
            temporal_target_speed_mps=float(base_speed_mps)
            * prediction.temporal_speed_scale,
            departure_start=prediction.departure_start,
            departure_end=prediction.departure_end,
            rejoin_start=prediction.rejoin_start,
            rejoin_end=prediction.rejoin_end,
            observation_identity=identity,
            backbone_forward_id=str(backbone_forward_id),
            model_id="sdf-k2-v4@semantic-head",
            base_checkpoint_hash=str(base_checkpoint_hash),
            spatial_head_checkpoint_hash=str(semantic_head_checkpoint_hash),
            feature_content_hash=token_bundle.raw_content_hash,
            raw_head_output_hash=prediction.raw_head_output_hash,
            head_lineage="spatial_mode_head_v4",
            alternative_metadata={
                "v4_head_schema": V4_HEAD_SCHEMA,
                "availability_probability": prediction.availability_probability,
                "kind_probability": prediction.kind_probability,
                "side_probability": prediction.side_probability,
                "maneuver_params": list(prediction.maneuver_params),
                "semantic_rescue_count": 0,
                "scenario_family_runtime_use": 0,
            },
        )
        return bundle


__all__ = [
    "AUX_DIM",
    "MANEUVER_PARAM_COUNT",
    "SemanticHeadOutputV4",
    "SpatialSemanticHeadV4",
    "SpatialSemanticHeadRuntimeV4",
    "V4_KIND_ORDER",
    "V4_SPATIAL_KINDS",
    "V4_HEAD_SCHEMA",
    "build_v4_aux_vector",
]
