"""Simple MLP baselines required by the H3 protocol.

Candidate-only MLP sees only the flattened candidate trajectory. Full-feature
MLP sees context+candidate. Both use the same pairwise BCE training objective
and independent shared scoring as the World model.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .dataset import PairExample
from .model import seed_everything


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def candidate_vector(item) -> list[float]:
    return [float(value) for row in item.candidate for value in row]


def full_vector(item) -> list[float]:
    return [float(value) for value in item.context] + candidate_vector(item)


@dataclass(frozen=True)
class BaselineTrainResult:
    name: str
    seed: int
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    checkpoint_sha256: str


def _pair_tensors(examples: Sequence[PairExample], kind: str, device: torch.device, *, swap: bool = False):
    rows = []
    for item in examples:
        if swap and random.random() < 0.5:
            left, right = item.candidates[1], item.candidates[0]
            winner = None if item.winner_index is None else 1 - item.winner_index
        else:
            left, right = item.candidates
            winner = item.winner_index
        lv = candidate_vector(left) if kind == "candidate_mlp" else full_vector(left)
        rv = candidate_vector(right) if kind == "candidate_mlp" else full_vector(right)
        rows.append((lv, rv, winner))
    x = torch.tensor([[row[0], row[1]] for row in rows], dtype=torch.float32, device=device)
    winner = torch.tensor([-1 if row[2] is None else row[2] for row in rows], dtype=torch.long, device=device)
    return x, winner


def _loss(model: BaselineMLP, x: Tensor, winner: Tensor) -> Tensor:
    utility = model(x)[:, :, 0]
    decisive = winner >= 0
    if not bool(decisive.any()):
        return utility.sum() * 0.0
    target = (winner[decisive] == 0).float()
    return nn.functional.binary_cross_entropy_with_logits(utility[decisive, 0] - utility[decisive, 1], target)


def train_baseline_model(
    kind: str,
    train_examples: Sequence[PairExample],
    val_examples: Sequence[PairExample],
    *,
    seed: int,
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
    max_epochs: int = 350,
    patience: int = 40,
) -> BaselineTrainResult:
    if kind not in {"candidate_mlp", "full_mlp"}:
        raise ValueError(f"unknown_baseline_kind:{kind}")
    seed_everything(seed)
    torch_device = torch.device(device)
    input_dim = 10 * 8 if kind == "candidate_mlp" else 499 + 10 * 8
    model = BaselineMLP(input_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        x, winner = _pair_tensors(train_examples, kind, torch_device, swap=True)
        loss = _loss(model, x, winner)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non_finite_baseline_loss:{kind}:{epoch}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        if val_examples:
            xv, wv = _pair_tensors(val_examples, kind, torch_device)
            with torch.no_grad():
                val_loss = float(_loss(model, xv, wv).detach().cpu())
        else:
            val_loss = float(loss.detach().cpu())
        if val_loss + 1e-8 < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"no_finite_baseline_checkpoint:{kind}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": {"kind": kind, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_loss}}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return BaselineTrainResult(kind, seed, best_epoch, best_loss, str(checkpoint_path), digest)


def load_baseline_model(checkpoint_path: Path, *, kind: str, device: str | torch.device = "cpu") -> tuple[BaselineMLP, dict]:
    input_dim = 10 * 8 if kind == "candidate_mlp" else 499 + 10 * 8
    model = BaselineMLP(input_dim).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model, dict(payload.get("metadata", {}))


def predict_baseline_pair(models: Sequence[BaselineMLP], example: PairExample, *, kind: str, device: str | torch.device = "cpu") -> int:
    left = example.candidates[0]; right = example.candidates[1]
    lv = candidate_vector(left) if kind == "candidate_mlp" else full_vector(left)
    rv = candidate_vector(right) if kind == "candidate_mlp" else full_vector(right)
    x = torch.tensor([[lv, rv]], dtype=torch.float32, device=device)
    with torch.no_grad():
        utilities = [float(model(x)[0, index, 0].detach().cpu()) for model in models for index in range(2)]
    # Sum over both candidates/models, then compare slot means.
    u0 = sum(model(x)[0, 0, 0].detach().cpu().item() for model in models) / len(models)
    u1 = sum(model(x)[0, 1, 0].detach().cpu().item() for model in models) / len(models)
    return 0 if u0 >= u1 else 1


def evaluate_baseline_models(models: Sequence[BaselineMLP], examples: Sequence[PairExample], *, kind: str, device: str = "cpu") -> dict:
    decisive = [item for item in examples if item.decisive]
    correct = sum(predict_baseline_pair(models, item, kind=kind, device=device) == item.winner_index for item in decisive)
    return {"name": kind, "n_decisive": len(decisive), "correct": correct, "accuracy": correct / len(decisive) if decisive else None,
            "mean_progress_regret_m": None, "mean_jerk_regret_mps3": None}


__all__ = [
    "BaselineMLP", "BaselineTrainResult", "candidate_vector", "evaluate_baseline_models",
    "full_vector", "load_baseline_model", "predict_baseline_pair", "train_baseline_model",
]
