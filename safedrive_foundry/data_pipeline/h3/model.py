"""Candidate-conditioned World scorer for H3v2.

The scorer is shared across candidates, contains no slot/source identity and is
permutation-equivariant by construction.  It consumes the frozen observable
feature contract from ``h3.contracts`` and additionally predicts a hard-risk
head when training data contains hard-unsafe positives.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import (
    H3_CANDIDATE_DIM,
    H3_CANDIDATE_STEPS,
    H3_CONFIG,
    H3_CONTEXT_DIM,
    H3_SCHEMA_VERSION,
    WorldPrediction,
)
from .dataset import PairExample


CONTEXT_DIM = H3_CONTEXT_DIM
CANDIDATE_STEPS = H3_CANDIDATE_STEPS
CANDIDATE_DIM = H3_CANDIDATE_DIM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class WorldScorerModel(nn.Module):
    """Structured context + candidate scorer with cross-attention."""

    def __init__(self, *, d_model: int = 128, layers: int = 2, heads: int = 4, ffn: int = 256, dropout: float = 0.1, scene_gate_mode: str = "hard") -> None:
        super().__init__()
        if scene_gate_mode not in {"hard", "learned"}:
            raise ValueError(f"unknown_scene_gate_mode:{scene_gate_mode}")
        self.scene_gate_mode = scene_gate_mode
        self.context_proj = nn.Sequential(nn.Linear(CONTEXT_DIM, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.candidate_proj = nn.Sequential(nn.Linear(CANDIDATE_DIM, d_model), nn.LayerNorm(d_model), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6),
        )
        self.register_buffer("position", torch.arange(CANDIDATE_STEPS, dtype=torch.float32), persistent=False)
        self.position_proj = nn.Linear(1, d_model)
        if scene_gate_mode == "learned":
            # Learned context gate for H3.1.  The legacy hard gate remains the
            # default only for loading frozen H3 checkpoints.
            self.scene_gate_proj = nn.Linear(d_model, 1)

    def forward(self, context: Tensor, candidate: Tensor, *, mask_context: bool = False, mask_candidate: bool = False) -> Tensor:
        if context.ndim != 2 or context.shape[-1] != CONTEXT_DIM:
            raise ValueError(f"context_shape:{tuple(context.shape)}")
        if candidate.ndim != 3 or candidate.shape[-2:] != (CANDIDATE_STEPS, CANDIDATE_DIM):
            raise ValueError(f"candidate_shape:{tuple(candidate.shape)}")
        if mask_context:
            context = torch.zeros_like(context)
        if mask_candidate:
            candidate = torch.zeros_like(candidate)

        context_token = self.context_proj(context).unsqueeze(1)  # (B,1,D)
        candidate_tokens = self.candidate_proj(candidate)       # (B,10,D)
        position = self.position[: candidate_tokens.shape[1]].to(candidate_tokens).view(1, -1, 1)
        candidate_tokens = candidate_tokens + self.position_proj(position)
        candidate_tokens = self.candidate_encoder(candidate_tokens)
        if self.scene_gate_mode == "hard":
            # Frozen H3 contract: all-zero context explicitly disables the
            # candidate path.  The H4/H5 runtime also checks this case before
            # inference and defers instead of relying on the model alone.
            presence = context.abs().sum(dim=-1, keepdim=True)
            scene_gate = (presence > 1e-6).to(candidate_tokens.dtype).unsqueeze(-1)
            candidate_tokens = candidate_tokens * scene_gate
        else:
            # H3.1 learned gate: context absence is a learned continuous gate,
            # not a hard-coded zero multiplier.
            gate = torch.sigmoid(self.scene_gate_proj(context_token[:, 0]))
            candidate_tokens = candidate_tokens * gate.unsqueeze(-1)
        attended, _ = self.cross_attention(candidate_tokens, context_token, context_token, need_weights=False)
        candidate_tokens = self.pool_norm(candidate_tokens + attended)
        pooled = torch.cat([candidate_tokens[:, 0], candidate_tokens.mean(dim=1), context_token[:, 0]], dim=-1)
        return self.head(pooled)


def _batch(examples: Sequence[PairExample], device: torch.device, *, swap: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    rows: list[tuple[list, list, list, list, list, list, bool]] = []
    for item in examples:
        if swap and random.random() < 0.5:
            left, right = item.candidates[1], item.candidates[0]
            winner = None if item.winner_index is None else 1 - item.winner_index
        else:
            left, right = item.candidates
            winner = item.winner_index
        rows.append((list(left.context), list(left.candidate), list(right.context), list(right.candidate), winner, item.tie,
                     left.progress_m, right.progress_m, left.jerk_rms_mps3, right.jerk_rms_mps3, left.risk, right.risk))
    contexts = torch.tensor([[row[0], row[2]] for row in rows], dtype=torch.float32, device=device)
    candidates = torch.tensor([[row[1], row[3]] for row in rows], dtype=torch.float32, device=device)
    # Outcome targets must follow the same left/right order as model inputs.
    progress = torch.tensor([[row[6], row[7]] for row in rows], dtype=torch.float32, device=device)
    jerk = torch.tensor([[math.log1p(max(0.0, row[8])), math.log1p(max(0.0, row[9]))] for row in rows], dtype=torch.float32, device=device)
    risk = torch.tensor([[float(row[10]), float(row[11])] for row in rows], dtype=torch.float32, device=device)
    winner = torch.tensor([(-1 if row[4] is None else row[4]) for row in rows], dtype=torch.long, device=device)
    ties = torch.tensor([row[5] for row in rows], dtype=torch.bool, device=device)
    return contexts, candidates, progress, jerk, risk, winner, ties


def scorer_loss(outputs: Tensor, progress: Tensor, jerk: Tensor, risk: Tensor, winner: Tensor, ties: Tensor, *, temperature: float = 1.0, risk_ranking_weight: float = 0.0) -> Tensor:
    utility = outputs[:, :, 0]
    progress_mean, progress_logvar = outputs[:, :, 1], outputs[:, :, 2].clamp(-6.0, 5.0)
    jerk_mean, jerk_logvar = outputs[:, :, 3], outputs[:, :, 4].clamp(-6.0, 5.0)
    risk_logit = outputs[:, :, 5]

    decisive = winner >= 0
    if bool(decisive.any()):
        target = (winner[decisive] == 0).float()
        risk_penalty = nn.functional.softplus(outputs[:, :, 5])
        ranking_utility = utility - float(risk_ranking_weight) * risk_penalty
        pair_loss = nn.functional.binary_cross_entropy_with_logits(
            (ranking_utility[decisive, 0] - ranking_utility[decisive, 1]) / max(0.05, temperature), target
        )
    else:
        pair_loss = utility.sum() * 0.0

    tie_loss = torch.relu(torch.abs(utility[ties, 0] - utility[ties, 1]) - float(H3_CONFIG["loss"]["tie_margin_value"])).mean() if bool(ties.any()) else utility.sum() * 0.0

    progress_nll = 0.5 * (((progress - progress_mean) ** 2) * torch.exp(-progress_logvar) + progress_logvar).mean()
    jerk_nll = 0.5 * (((jerk - jerk_mean) ** 2) * torch.exp(-jerk_logvar) + jerk_logvar).mean()

    risk_valid = risk.ge(0.5).any() & risk.lt(0.5).any()
    risk_loss = nn.functional.binary_cross_entropy_with_logits(risk_logit, risk) if bool(risk_valid) else risk_logit.sum() * 0.0

    cfg = H3_CONFIG["loss"]
    return (
        float(cfg["pairwise_bce"]) * pair_loss
        + float(cfg["progress_nll"]) * progress_nll
        + float(cfg["jerk_nll"]) * jerk_nll
        + float(cfg["risk_bce"]) * risk_loss
        + float(cfg["tie_margin"]) * tie_loss
    )


@dataclass(frozen=True)
class TrainResult:
    seed: int
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    checkpoint_sha256: str
    train_examples: int
    val_examples: int
    device: str


def _loss_for_examples(model: WorldScorerModel, examples: Sequence[PairExample], device: torch.device, *, temperature: float = 1.0, risk_ranking_weight: float = 0.0) -> float:
    if not examples:
        return float("nan")
    contexts, candidates, progress, jerk, risk, winner, ties = _batch(examples, device, swap=False)
    outputs = torch.stack([model(contexts[:, index], candidates[:, index]) for index in range(2)], dim=1)
    with torch.no_grad():
        return float(scorer_loss(outputs, progress, jerk, risk, winner, ties, temperature=temperature, risk_ranking_weight=risk_ranking_weight).detach().cpu())


def train_model(
    train_examples: Sequence[PairExample],
    val_examples: Sequence[PairExample],
    *,
    seed: int,
    checkpoint_path: Path,
    max_epochs: int | None = None,
    patience: int | None = None,
    device: str | torch.device = "cpu",
    scene_gate_mode: str = "hard",
    risk_ranking_weight: float = 0.0,
) -> TrainResult:
    if not train_examples:
        raise ValueError("empty_training_examples")
    seed_everything(seed)
    torch_device = torch.device(device)
    model_cfg = H3_CONFIG["model"]
    model = WorldScorerModel(**model_cfg, scene_gate_mode=scene_gate_mode).to(torch_device)
    optimizer_cfg = H3_CONFIG["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_cfg["lr"]), weight_decay=float(optimizer_cfg["weight_decay"]))
    epochs = int(max_epochs or optimizer_cfg["max_epochs"])
    wait_limit = int(patience if patience is not None else optimizer_cfg["patience"])
    temperature = float(H3_CONFIG["loss"]["train_temperature"])
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    best_state: dict[str, Tensor] | None = None
    batch_size = max(1, int(optimizer_cfg.get("batch_size", len(train_examples))))
    indices = list(range(len(train_examples)))
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_examples = [train_examples[index] for index in batch_indices]
            optimizer.zero_grad(set_to_none=True)
            contexts, candidates, progress, jerk, risk, winner, ties = _batch(batch_examples, torch_device, swap=True)
            outputs = torch.stack([model(contexts[:, index], candidates[:, index]) for index in range(2)], dim=1)
            loss = scorer_loss(outputs, progress, jerk, risk, winner, ties, temperature=temperature, risk_ranking_weight=risk_ranking_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non_finite_training_loss:{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimizer_cfg["gradient_clip"]))
            optimizer.step()
        model.eval()
        val_loss = _loss_for_examples(model, val_examples or train_examples, torch_device, temperature=temperature, risk_ranking_weight=risk_ranking_weight)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= wait_limit:
                break
    if best_state is None:
        raise RuntimeError("no_finite_checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": H3_SCHEMA_VERSION,
        "config": H3_CONFIG,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "temperature": temperature,
        "scene_gate_mode": scene_gate_mode,
        "risk_ranking_weight": float(risk_ranking_weight),
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return TrainResult(seed, best_epoch, best_loss, str(checkpoint_path), digest, len(train_examples), len(val_examples), str(torch_device))


def load_model(checkpoint_path: Path, *, device: str | torch.device = "cpu") -> tuple[WorldScorerModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scene_gate_mode = str(payload.get("metadata", {}).get("scene_gate_mode", "hard"))
    model = WorldScorerModel(**H3_CONFIG["model"], scene_gate_mode=scene_gate_mode).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))


def _mask_context_tensor(context: Tensor, mode: str) -> Tensor:
    """Apply a natural, non-global context ablation.

    Layout: history 0:140, route 140:344, actors 344:440, lights 440:494,
    ego-current 494:499.
    """
    masked = context.clone()
    if mode == "history":
        masked[:, :140] = 0.0
    elif mode == "route":
        masked[:, 140:344] = 0.0
    elif mode == "actors":
        masked[:, 344:440] = 0.0
    elif mode == "lights":
        masked[:, 440:494] = 0.0
    elif mode == "zero":
        masked.zero_()
    else:
        raise ValueError(f"unknown_context_mask_mode:{mode}")
    return masked


def predict_model(
    model: WorldScorerModel,
    example: PairExample,
    *,
    device: str | torch.device = "cpu",
    mask_context: bool = False,
    mask_candidate: bool = False,
    context_mask_mode: str | None = None,
) -> tuple[WorldPrediction, WorldPrediction]:
    device = torch.device(device)
    context = torch.tensor([list(item.context) for item in example.candidates], dtype=torch.float32, device=device)
    candidate = torch.tensor([list(item.candidate) for item in example.candidates], dtype=torch.float32, device=device)
    if mask_context:
        context = torch.zeros_like(context)
    elif context_mask_mode is not None:
        context = _mask_context_tensor(context, context_mask_mode)
    with torch.no_grad():
        outputs = model(context, candidate, mask_candidate=mask_candidate)
    predictions = []
    for index, item in enumerate(example.candidates):
        row = outputs[index].detach().cpu().tolist()
        predictions.append(WorldPrediction(item.candidate_key, row[0], row[1], row[2], row[3], row[4], row[5]))
    return predictions[0], predictions[1]


def ensemble_predict(
    models: Sequence[WorldScorerModel],
    example: PairExample,
    *,
    device: str | torch.device = "cpu",
    mask_context: bool = False,
    mask_candidate: bool = False,
    context_mask_mode: str | None = None,
) -> tuple[WorldPrediction, WorldPrediction, float]:
    if not models:
        raise ValueError("empty_model_ensemble")
    all_predictions = [
        predict_model(model, example, device=device, mask_context=mask_context, mask_candidate=mask_candidate, context_mask_mode=context_mask_mode)
        for model in models
    ]
    rows: list[WorldPrediction] = []
    utilities: list[list[float]] = [[], []]
    for index in range(2):
        selected = [item[index] for item in all_predictions]
        utilities[index] = [item.utility for item in selected]
        rows.append(
            WorldPrediction(
                selected[0].candidate_key,
                float(np.mean([item.utility for item in selected])),
                float(np.mean([item.progress_mean_m for item in selected])),
                float(np.mean([item.progress_logvar for item in selected])),
                float(np.mean([item.jerk_mean_log1p for item in selected])),
                float(np.mean([item.jerk_logvar for item in selected])),
                float(np.mean([item.risk_logit for item in selected])),
            )
        )
    uncertainty = float(np.mean([np.var(values) for values in utilities]))
    return rows[0], rows[1], uncertainty


__all__ = [
    "CANDIDATE_DIM",
    "CANDIDATE_STEPS",
    "CONTEXT_DIM",
    "TrainResult",
    "WorldScorerModel",
    "ensemble_predict",
    "load_model",
    "predict_model",
    "scorer_loss",
    "seed_everything",
    "train_model",
]
