"""Project-side spatial mode residual heads.

Context = geometry (32) + driving feature (64). Driving feature is either
pooled SimLingo adaptor output (preferred) or a runtime-matched scene proxy.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from driving_vla.model.driving_feature import (
    CONTEXT_DIM,
    DRIVING_FEAT_DIM,
    GEOM_DIM,
    build_context_vector,
    observable_scene_from_sample,
    scene_proxy_from_sample,
)


def geometry_context_vector(
    native_path_xy: Sequence[tuple[float, float]],
    *,
    ego_v: float,
    base_speed_mps: float,
    dim: int = GEOM_DIM,
) -> list[float]:
    """Deterministic context features from native polyline (no learned vision)."""
    pts = [(float(x), float(y)) for x, y in native_path_xy]
    feats = [float(ego_v), float(base_speed_mps), float(len(pts))]
    if len(pts) >= 2:
        total = 0.0
        for i in range(1, len(pts)):
            total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        feats.append(total)
        feats.append(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
        feats.append(
            math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
        )
        for i in (0, len(pts) // 2, len(pts) - 1):
            feats.append(pts[i][0] - pts[0][0])
            feats.append(pts[i][1] - pts[0][1])
    while len(feats) < dim:
        feats.append(0.0)
    return feats[:dim]


class SpatialModeResidualHead(nn.Module):
    """Shared MLP + mode embedding → residual Δs, d, speed_scale, avail logit.

    Mode slots are fixed-semantics (no WTA across slots):
      0 = nominal_progress
      1 = defensive_alternative
    """

    def __init__(
        self,
        *,
        context_dim: int = CONTEXT_DIM,
        hidden: int = 128,
        n_path: int = 20,
        n_modes: int = 2,
    ) -> None:
        super().__init__()
        self.n_path = int(n_path)
        self.context_dim = int(context_dim)
        self.mode_emb = nn.Embedding(n_modes, 16)
        # Small observable-scene adaptor.  Current actor state lives in
        # context[16:24]; giving it an explicit residual path prevents the
        # high-dimensional visual feature from memorizing a lineage while
        # ignoring the actor geometry that actually determines the manoeuvre.
        self.scene_adapter = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, context_dim),
        )
        self.backbone = nn.Sequential(
            nn.Linear(context_dim + 16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_s = nn.Linear(hidden, self.n_path)
        self.d = nn.Linear(hidden, self.n_path)
        self.speed_scale = nn.Linear(hidden, 1)
        self.avail = nn.Linear(hidden, 1)
        nn.init.normal_(self.d.weight, std=0.05)
        nn.init.zeros_(self.d.bias)
        nn.init.normal_(self.scene_adapter[-1].weight, std=0.01)
        nn.init.zeros_(self.scene_adapter[-1].bias)
        # bias defensive mode emb slightly so slots start distinct
        with torch.no_grad():
            if self.mode_emb.weight.shape[0] >= 2:
                self.mode_emb.weight[1].add_(0.1)

    def forward(self, context: torch.Tensor, mode_index: torch.Tensor) -> dict[str, torch.Tensor]:
        me = self.mode_emb(mode_index)
        scene = context[:, 16:24]
        context_enriched = context + self.scene_adapter(scene)
        h = self.backbone(torch.cat([context_enriched, me], dim=-1))
        return {
            "raw_delta_s": self.delta_s(h),
            "raw_d": self.d(h),
            "speed_scale": 0.5 + 0.5 * torch.sigmoid(self.speed_scale(h)).squeeze(-1),
            "avail_logit": self.avail(h).squeeze(-1),
        }


def decoded_lateral_separation(
    raw_d0: torch.Tensor,
    raw_d1: torch.Tensor,
    *,
    ramp_points: int = 6,
    max_lateral_m: float = 1.0,
) -> torch.Tensor:
    """Differentiable form of the Frenet codec's lateral separation."""
    if raw_d0.shape != raw_d1.shape:
        raise ValueError("mode residual tensors must have the same shape")
    if raw_d0.ndim != 2:
        raise ValueError("mode residual tensors must be [batch, path]")
    count = int(raw_d0.shape[1])
    envelope = torch.arange(
        count, dtype=raw_d0.dtype, device=raw_d0.device
    ) / float(max(1, int(ramp_points)))
    envelope = torch.clamp(envelope, max=1.0).unsqueeze(0)
    d0 = float(max_lateral_m) * envelope * torch.tanh(raw_d0)
    d1 = float(max_lateral_m) * envelope * torch.tanh(raw_d1)
    return torch.amax(torch.abs(d1 - d0), dim=1)


def decoded_lateral_smoothness_loss(
    raw_d: torch.Tensor,
    *,
    ramp_points: int = 6,
    max_lateral_m: float = 1.0,
) -> torch.Tensor:
    """Second-difference penalty on the runtime-decoded lateral residual.

    Penalizing decoded ``d`` rather than raw logits matches the Frenet codec's
    tanh/ramp envelope and discourages curvature spikes without shrinking a
    smooth, spatially distinct alternative back toward the nominal path.
    """
    if raw_d.ndim != 2:
        raise ValueError("lateral residual tensor must be [batch, path]")
    count = int(raw_d.shape[1])
    if count < 3:
        return raw_d.sum() * 0.0
    envelope = torch.arange(
        count, dtype=raw_d.dtype, device=raw_d.device
    ) / float(max(1, int(ramp_points)))
    envelope = torch.clamp(envelope, max=1.0).unsqueeze(0)
    decoded = float(max_lateral_m) * envelope * torch.tanh(raw_d)
    second_difference = decoded[:, 2:] - 2.0 * decoded[:, 1:-1] + decoded[:, :-2]
    return torch.mean(torch.abs(second_difference))


def decoded_diversity_floor_loss(
    raw_d0: torch.Tensor,
    raw_d1: torch.Tensor,
    *,
    target_m: float,
    ramp_points: int = 6,
    max_lateral_m: float = 1.0,
) -> torch.Tensor:
    """Per-sample diversity floor.

    Taking a batch mean *before* the hinge lets a highly separated sample hide
    another collapsed sample.  Apply the floor independently, then average.
    """
    separation = decoded_lateral_separation(
        raw_d0,
        raw_d1,
        ramp_points=ramp_points,
        max_lateral_m=max_lateral_m,
    )
    return torch.relu(float(target_m) - separation).mean()


def decoded_peak_lateral_separation(
    raw_d0: Sequence[float],
    raw_d1: Sequence[float],
    *,
    ramp_points: int = 6,
    max_lateral_m: float = 1.0,
) -> float:
    """Runtime-equivalent inter-candidate diversity.

    Absolute excursion from the native path is not candidate diversity.  In
    particular, two modes shifted 0.7 m and 0.8 m in the same direction are
    only 0.1 m apart and must not pass a 0.5 m diversity floor.
    """
    count = min(len(raw_d0), len(raw_d1))
    peak = 0.0
    for index in range(count):
        envelope = min(1.0, float(index) / float(max(1, int(ramp_points))))
        d0 = float(max_lateral_m) * envelope * math.tanh(float(raw_d0[index]))
        d1 = float(max_lateral_m) * envelope * math.tanh(float(raw_d1[index]))
        peak = max(peak, abs(d1 - d0))
    return peak


def defensive_speed_margin_loss(
    predicted_speed_scale: torch.Tensor,
    target_speed_scale: torch.Tensor,
    *,
    nominal_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize loss of a teacher-declared defensive slow/yield mode.

    The runtime nominal anchor uses ``nominal_scale``.  Samples whose teacher
    requests scale 1.0 add no artificial speed gap; samples labeled 0.85 keep
    that mode-conditioned longitudinal distinction.
    """
    if predicted_speed_scale.shape != target_speed_scale.shape:
        raise ValueError("predicted and target speed scales must have the same shape")
    required_gap = torch.clamp(
        float(nominal_scale) - target_speed_scale,
        min=0.0,
    )
    predicted_gap = float(nominal_scale) - predicted_speed_scale
    return torch.relu(required_gap - predicted_gap).mean()


@dataclass
class SpatialHeadOutput:
    raw_delta_s: list[float]
    raw_d: list[float]
    speed_scale: float
    available: bool
    avail_prob: float
    head_lineage: str = "spatial_mode_head"


class SpatialK2HeadRuntime:
    """Inference wrapper for dual-mode residual heads."""

    def __init__(
        self,
        model: SpatialModeResidualHead | None = None,
        *,
        device: str = "cpu",
        n_path: int = 20,
        context_dim: int = CONTEXT_DIM,
        checkpoint_path: str | None = None,
        checkpoint_use: str = "formal_offline",
        skip_checkpoint_contract: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.n_path = n_path
        self.context_dim = context_dim
        self.model = model or SpatialModeResidualHead(n_path=n_path, context_dim=context_dim)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_use = checkpoint_use
        if checkpoint_path and Path(checkpoint_path).is_file():
            # A2: fail-closed BEFORE torch.load for formal uses
            if not skip_checkpoint_contract:
                from driving_vla.model.checkpoint_contract import (
                    CheckpointContractError,
                    require_checkpoint_for_use,
                )

                try:
                    require_checkpoint_for_use(checkpoint_path, checkpoint_use)
                except CheckpointContractError:
                    raise
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            sd = state["model"] if isinstance(state, dict) and "model" in state else state
            try:
                self.model.load_state_dict(sd, strict=True)
            except RuntimeError:
                self.model.load_state_dict(sd, strict=False)
            if isinstance(state, dict) and "context_dim" in state:
                self.context_dim = int(state["context_dim"])
        self._ckpt_hash = self._file_hash(checkpoint_path) if checkpoint_path else "untrained"

    @staticmethod
    def _file_hash(path: str | None) -> str:
        if not path or not Path(path).is_file():
            return "missing"
        from driving_vla.model.checkpoint_contract import file_sha256

        return file_sha256(path, full=True)

    @property
    def spatial_head_checkpoint_hash(self) -> str:
        return self._ckpt_hash

    @torch.inference_mode()
    def predict_modes(
        self,
        native_path_xy: Sequence[tuple[float, float]],
        *,
        ego_v: float,
        base_speed_mps: float,
        driving_feature: Sequence[float] | None = None,
        observable_scene: dict[str, Any] | None = None,
        sample: dict[str, Any] | None = None,
    ) -> tuple[SpatialHeadOutput, SpatialHeadOutput]:
        if driving_feature is None and sample is not None:
            if sample.get("driving_feature") is not None:
                driving_feature = list(sample["driving_feature"])
            else:
                driving_feature = scene_proxy_from_sample(sample)
        if observable_scene is None and sample is not None:
            observable_scene = observable_scene_from_sample(sample)
        ctx = build_context_vector(
            native_path_xy,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            driving_feature=driving_feature,
            observable_scene=observable_scene,
        )
        # pad/truncate to model context_dim
        if len(ctx) < self.context_dim:
            ctx = ctx + [0.0] * (self.context_dim - len(ctx))
        ctx = ctx[: self.context_dim]
        x = torch.tensor([ctx], dtype=torch.float32, device=self.device)
        outs: list[SpatialHeadOutput] = []
        for mode_i in (0, 1):
            mi = torch.tensor([mode_i], dtype=torch.long, device=self.device)
            y = self.model(x, mi)
            avail_p = float(torch.sigmoid(y["avail_logit"][0]).item())
            available = True if mode_i == 0 else avail_p >= 0.5
            outs.append(
                SpatialHeadOutput(
                    raw_delta_s=[float(v) for v in y["raw_delta_s"][0].tolist()],
                    raw_d=[float(v) for v in y["raw_d"][0].tolist()],
                    speed_scale=float(y["speed_scale"][0].item()),
                    available=available,
                    avail_prob=avail_p if mode_i == 1 else 1.0,
                    head_lineage="spatial_mode_head",
                )
            )
        return outs[0], outs[1]
