"""Small candidate-conditioned object/vector World-V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import (
    ACTOR_FEATURES,
    CANDIDATE_FEATURES,
    EGO_FEATURES,
    FUTURE_FEATURES,
    K,
    MAX_ACTORS,
    ROAD_FEATURES,
    T,
    WorldBatch,
    WorldPrediction,
)


@dataclass(frozen=True)
class WorldV0Config:
    d_model: int = 192
    n_heads: int = 6
    dim_feedforward: int = 768
    scene_layers: int = 3
    temporal_layers: int = 1
    candidate_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    max_history: int = 5
    max_road_points: int = 16
    future_steps: int = T
    no_action: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _encoder_layer(config: WorldV0Config) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        config: WorldV0Config,
        *,
        max_steps: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, max_steps, config.d_model))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(config),
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        steps = values.shape[-2]
        encoded = self.input(values) + self.position[:, :steps]
        flat_shape = encoded.shape
        encoded = encoded.reshape(-1, steps, flat_shape[-1])
        flat_mask = mask.reshape(-1, steps)
        all_missing = ~flat_mask.any(dim=-1)
        safe_mask = flat_mask.clone()
        if all_missing.any():
            safe_mask[all_missing, -1] = True
        encoded = self.encoder(encoded, src_key_padding_mask=~safe_mask)
        pooled = _masked_mean(encoded, safe_mask, dim=1)
        pooled[all_missing] = 0.0
        return self.output_norm(pooled).reshape(*flat_shape[:-2], flat_shape[-1])


class WorldV0(nn.Module):
    def __init__(self, config: WorldV0Config | None = None) -> None:
        super().__init__()
        self.config = config or WorldV0Config()
        cfg = self.config
        self.ego_encoder = TemporalEncoder(
            EGO_FEATURES,
            cfg,
            max_steps=cfg.max_history,
            layers=cfg.temporal_layers,
        )
        self.actor_encoder = TemporalEncoder(
            ACTOR_FEATURES,
            cfg,
            max_steps=cfg.max_history,
            layers=cfg.temporal_layers,
        )
        self.road_point = nn.Sequential(
            nn.Linear(ROAD_FEATURES, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.scene_type = nn.Parameter(torch.zeros(1, 1 + MAX_ACTORS + 3, cfg.d_model))
        nn.init.normal_(self.scene_type, std=0.02)
        self.scene_encoder = nn.TransformerEncoder(
            _encoder_layer(cfg),
            num_layers=cfg.scene_layers,
            enable_nested_tensor=False,
        )
        self.scene_norm = nn.LayerNorm(cfg.d_model)
        self.candidate_encoder = TemporalEncoder(
            CANDIDATE_FEATURES,
            cfg,
            max_steps=T,
            layers=cfg.candidate_layers,
        )
        self.no_action_query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.no_action_query, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.decoder_layers,
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        joint_dim = cfg.d_model * 2
        self.actor_condition = nn.Sequential(
            nn.Linear(joint_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.future_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Linear(cfg.dim_feedforward, T * 6),
        )
        # Forecast a residual over the observable CV baseline. Zero
        # initialization makes the untrained model exactly CV rather than an
        # arbitrary absolute-coordinate predictor.
        nn.init.zeros_(self.future_head[-1].weight)
        nn.init.zeros_(self.future_head[-1].bias)
        self.risk_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Linear(cfg.dim_feedforward, 6),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _as_tensor(self, value: Any, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

    def _prepare(self, batch: WorldBatch) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        return {
            "ego_history": self._as_tensor(batch.ego_history, dtype=torch.float32, device=device),
            "ego_history_mask": self._as_tensor(
                batch.ego_history_mask, dtype=torch.bool, device=device
            ),
            "actor_history": self._as_tensor(
                batch.actor_history, dtype=torch.float32, device=device
            ),
            "actor_history_mask": self._as_tensor(
                batch.actor_history_mask, dtype=torch.bool, device=device
            ),
            "road": self._as_tensor(batch.road, dtype=torch.float32, device=device),
            "road_mask": self._as_tensor(batch.road_mask, dtype=torch.bool, device=device),
            "candidates": self._as_tensor(
                batch.candidates, dtype=torch.float32, device=device
            ),
            "candidate_mask": self._as_tensor(
                batch.candidate_mask, dtype=torch.bool, device=device
            ),
        }

    def forward(self, batch: WorldBatch) -> WorldPrediction:
        x = self._prepare(batch)
        ego_token = self.ego_encoder(x["ego_history"], x["ego_history_mask"]).unsqueeze(1)
        actor_tokens = self.actor_encoder(x["actor_history"], x["actor_history_mask"])
        actor_present = x["actor_history_mask"].any(dim=-1)
        road_points = self.road_point(x["road"])
        road_present = x["road_mask"].any(dim=-1)
        road_tokens = _masked_mean(road_points, x["road_mask"], dim=2)
        scene_tokens = torch.cat((ego_token, actor_tokens, road_tokens), dim=1)
        scene_tokens = scene_tokens + self.scene_type[:, : scene_tokens.shape[1]]
        scene_mask = torch.cat(
            (
                torch.ones(
                    (scene_tokens.shape[0], 1), dtype=torch.bool, device=scene_tokens.device
                ),
                actor_present,
                road_present,
            ),
            dim=1,
        )
        scene = self.scene_encoder(scene_tokens, src_key_padding_mask=~scene_mask)
        scene = self.scene_norm(scene)

        batch_size = scene.shape[0]
        if self.config.no_action:
            candidate_tokens = self.no_action_query.expand(batch_size, K, -1)
        else:
            time_mask = x["candidate_mask"].unsqueeze(-1).expand(-1, -1, T)
            candidate_tokens = self.candidate_encoder(x["candidates"], time_mask)

        memory = (
            scene.unsqueeze(1)
            .expand(-1, K, -1, -1)
            .reshape(batch_size * K, scene.shape[1], scene.shape[2])
        )
        memory_mask = (
            scene_mask.unsqueeze(1)
            .expand(-1, K, -1)
            .reshape(batch_size * K, scene.shape[1])
        )
        query = candidate_tokens.reshape(batch_size * K, 1, -1)
        decoded = self.candidate_decoder(
            query,
            memory,
            memory_key_padding_mask=~memory_mask,
        )
        decoded = self.decoder_norm(decoded[:, 0]).reshape(batch_size, K, -1)

        actor_scene = scene[:, 1 : 1 + MAX_ACTORS]
        actor_joint = torch.cat(
            (
                actor_scene.unsqueeze(1).expand(-1, K, -1, -1),
                decoded.unsqueeze(2).expand(-1, -1, MAX_ACTORS, -1),
            ),
            dim=-1,
        )
        actor_latent = self.actor_condition(actor_joint)
        future_raw = self.future_head(actor_latent).reshape(
            batch_size, K, MAX_ACTORS, T, 6
        )
        current_actor = x["actor_history"][:, :, -1]
        current_xy = current_actor[..., :2]
        current_v = current_actor[..., 4:6]
        future_t = torch.arange(
            1,
            T + 1,
            dtype=current_xy.dtype,
            device=current_xy.device,
        ) * 0.25
        cv_xy = current_xy.unsqueeze(2) + current_v.unsqueeze(2) * future_t.view(
            1, 1, T, 1
        )
        cv_future = torch.cat(
            (
                cv_xy,
                current_v.unsqueeze(2).expand(-1, -1, T, -1),
            ),
            dim=-1,
        )
        actor_future_mean = (
            cv_future.unsqueeze(1).expand(-1, K, -1, -1, -1)
            + future_raw[..., :4]
        )
        actor_future_log_scale = future_raw[..., 4:].clamp(-5.0, 3.0)
        risk = self.risk_head(decoded)
        mask = x["candidate_mask"]
        masked_score = risk[..., 5].masked_fill(~mask, torch.finfo(risk.dtype).min)
        return WorldPrediction(
            actor_future_mean=actor_future_mean,
            actor_future_log_scale=actor_future_log_scale,
            collision_logit=risk[..., 0],
            offroad_logit=risk[..., 1],
            ttc_value=torch.nn.functional.softplus(risk[..., 2]),
            ttc_censored_logit=risk[..., 3],
            utility_score=masked_score,
            candidate_mask=mask,
        )

    @torch.no_grad()
    def predict_safe(self, batch: WorldBatch) -> WorldPrediction:
        try:
            prediction = self(batch)
            prediction.validate_finite()
            return prediction
        except (RuntimeError, ValueError) as exc:
            candidate_mask = np.asarray(batch.candidate_mask, dtype=bool)
            shape = candidate_mask.shape
            zeros = np.zeros(shape, dtype=np.float32)
            return WorldPrediction(
                actor_future_mean=np.zeros((*shape, MAX_ACTORS, T, 4), dtype=np.float32),
                actor_future_log_scale=np.zeros(
                    (*shape, MAX_ACTORS, T, 2), dtype=np.float32
                ),
                collision_logit=zeros,
                offroad_logit=zeros,
                ttc_value=zeros,
                ttc_censored_logit=zeros,
                utility_score=zeros,
                candidate_mask=candidate_mask,
                status="INVALID",
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def select_candidate(prediction: WorldPrediction, *, top1_index: int = 0) -> tuple[int, str]:
        mask = np.asarray(
            prediction.candidate_mask.detach().cpu()
            if hasattr(prediction.candidate_mask, "detach")
            else prediction.candidate_mask,
            dtype=bool,
        )
        if mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("select_candidate expects batch size 1")
        available = np.flatnonzero(mask[0])
        if prediction.status != "OK":
            return int(top1_index), "WORLD_INVALID_FALLBACK_TOP1"
        if len(available) == 0:
            return int(top1_index), "NO_VALID_CANDIDATE_FALLBACK_TOP1"
        if len(available) == 1:
            return int(available[0]), "NO_RANKING_NEEDED"
        score = (
            prediction.utility_score.detach().cpu().numpy()
            if hasattr(prediction.utility_score, "detach")
            else np.asarray(prediction.utility_score)
        )
        selected = int(available[np.argmax(score[0, available])])
        return selected, "WORLD_RANKED"
