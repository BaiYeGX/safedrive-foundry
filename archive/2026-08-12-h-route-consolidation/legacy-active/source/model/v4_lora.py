"""The single allowed R2 V4 head-generalization fallback.

This module is intentionally narrow.  It adapts only the query/value input
paths of the final two V4 self-attention blocks; the base head remains frozen.
The wrapper keeps the ordinary ``MultiheadAttention`` implementation as the
source of truth and disables its inference fast path so an adapter cannot be
silently skipped.  A checkpoint must record this exact configuration before it
can be used by the V4 runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class V4LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_blocks: tuple[int, ...] = (0, 1)
    target_modules: tuple[str, ...] = ("q", "v")
    bf16: bool = True
    gradient_checkpointing: bool = True

    def validate(self) -> "V4LoRAConfig":
        if int(self.rank) != 8:
            raise ValueError("R2 V4 LoRA rank is fixed at 8")
        if float(self.alpha) != 16.0:
            raise ValueError("R2 V4 LoRA alpha is fixed at 16")
        if float(self.dropout) != 0.05:
            raise ValueError("R2 V4 LoRA dropout is fixed at 0.05")
        if tuple(self.target_blocks) != (0, 1):
            raise ValueError("R2 V4 LoRA may target only the top two attention blocks")
        if tuple(self.target_modules) != ("q", "v"):
            raise ValueError("R2 V4 LoRA may target only q/v")
        if not bool(self.bf16):
            raise ValueError("R2 V4 LoRA must use bf16 adapter computation")
        if not bool(self.gradient_checkpointing):
            raise ValueError("R2 V4 LoRA must enable gradient checkpointing")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _LowRankInputAdapter(nn.Module):
    def __init__(self, dim: int, config: V4LoRAConfig) -> None:
        super().__init__()
        self.down = nn.Linear(dim, config.rank, bias=False)
        self.up = nn.Linear(config.rank, dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = float(config.alpha) / float(config.rank)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Adapter weights follow the base module dtype/device.  bf16 is a
        # training/runtime requirement for the fallback; CPU tests may use
        # float32 because CPU bf16 kernels are not available everywhere.
        return self.up(self.dropout(self.down(value))) * self.scale


class LoRAQVMultiheadAttention(nn.Module):
    """MHA wrapper with low-rank q/v input updates.

    ``batch_first`` is reported false only to force the parent Transformer
    layer off PyTorch's fused fast path.  The wrapped V4 encoder is batch-first
    and the actual base MHA still receives ``[B,T,H]`` tensors.
    """

    def __init__(self, base: nn.MultiheadAttention, config: V4LoRAConfig) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.q_lora = _LowRankInputAdapter(base.embed_dim, config)
        self.v_lora = _LowRankInputAdapter(base.embed_dim, config)

    @property
    def batch_first(self) -> bool:  # pragma: no cover - checked by parent layer
        return False

    @property
    def embed_dim(self) -> int:
        return self.base.embed_dim

    @property
    def num_heads(self) -> int:
        return self.base.num_heads

    @property
    def in_proj_weight(self) -> torch.Tensor | None:
        return self.base.in_proj_weight

    @property
    def in_proj_bias(self) -> torch.Tensor | None:
        return self.base.in_proj_bias

    @property
    def out_proj(self) -> nn.Module:
        return self.base.out_proj

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return self.base._qkv_same_embed_dim

    def merge_masks(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.merge_masks(*args, **kwargs)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any):
        query = query + self.q_lora(query)
        value = value + self.v_lora(value)
        return self.base(query, key, value, **kwargs)


def apply_v4_lora_qv(model: nn.Module, config: V4LoRAConfig | None = None) -> dict[str, Any]:
    """Freeze ``model`` and install the exact V4 q/v fallback adapters."""
    cfg = (config or V4LoRAConfig()).validate()
    encoder = getattr(model, "token_encoder", None)
    layers = list(getattr(encoder, "layers", ()))
    if len(layers) < 2:
        raise ValueError("V4 LoRA requires at least two Transformer blocks")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed: list[int] = []
    for index in cfg.target_blocks:
        if index < 0 or index >= len(layers):
            raise ValueError(f"V4 LoRA block out of range: {index}")
        layer = layers[index]
        if isinstance(layer.self_attn, LoRAQVMultiheadAttention):
            installed.append(index)
            continue
        layer.self_attn = LoRAQVMultiheadAttention(layer.self_attn, cfg)
        installed.append(index)
    if cfg.gradient_checkpointing:
        setattr(model, "_v4_gradient_checkpointing", True)
    setattr(model, "_v4_lora_config", cfg.to_dict())
    for parameter in model.parameters():
        if parameter.requires_grad:
            continue
        # Only adapters should be trainable after the surgery.
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "config": cfg.to_dict(),
        "installed_blocks": installed,
        "trainable_parameters": trainable,
        "base_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "q_lora" not in name and "v_lora" not in name
        ),
    }


__all__ = [
    "LoRAQVMultiheadAttention",
    "V4LoRAConfig",
    "apply_v4_lora_qv",
]
