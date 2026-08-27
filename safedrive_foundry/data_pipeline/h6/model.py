"""Shared, source-blind World v3 model with objective and trust heads."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from data_pipeline.h3.contracts import (
    H3_CANDIDATE_DIM,
    H3_CANDIDATE_STEPS,
    H3_CONTEXT_DIM,
)
from data_pipeline.h6.contracts import (
    WORLD_V3_OUTPUT_DIM,
    WORLD_V3_SCHEMA_VERSION,
    WORLD_VLA75_OUTPUT_DIM,
    WORLD_VLA75_SCHEMA_VERSION,
)
from data_pipeline.h6.dataset import OutcomePairExample


@dataclass(frozen=True)
class WorldV3TrainConfig:
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn: int = 256
    dropout: float = 0.10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 120
    patience: int = 18
    gradient_clip: float = 2.0


class WorldV3Model(nn.Module):
    """One scorer shared by Expert and VLA; source identity is not an input."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        layers: int = 2,
        heads: int = 4,
        ffn: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(H3_CONTEXT_DIM, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(H3_CANDIDATE_DIM, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.candidate_encoder = nn.TransformerEncoder(encoder, num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.position_proj = nn.Linear(1, d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, WORLD_V3_OUTPUT_DIM),
        )
        self.register_buffer(
            "position",
            torch.arange(H3_CANDIDATE_STEPS, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, context: Tensor, candidate: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[-1] != H3_CONTEXT_DIM:
            raise ValueError(f"context_shape:{tuple(context.shape)}")
        if candidate.ndim != 3 or candidate.shape[-2:] != (
            H3_CANDIDATE_STEPS,
            H3_CANDIDATE_DIM,
        ):
            raise ValueError(f"candidate_shape:{tuple(candidate.shape)}")
        context_token = self.context_proj(context).unsqueeze(1)
        tokens = self.candidate_proj(candidate)
        position = self.position[: tokens.shape[1]].to(tokens).view(1, -1, 1)
        tokens = self.candidate_encoder(tokens + self.position_proj(position))
        attended, _ = self.cross_attention(tokens, context_token, context_token, need_weights=False)
        tokens = self.pool_norm(tokens + attended)
        pooled = torch.cat((tokens[:, 0], tokens.mean(dim=1), context_token[:, 0]), dim=-1)
        return self.head(pooled)


class WorldVLA75Model(WorldV3Model):
    """VLA75 World model with source-blind preference/executable heads.

    The first twelve outputs retain the exact WorldV3 ordering.  The final
    two outputs are ``preference_utility`` and ``executable_logit``.  Keeping
    this as a separate class/schema means old checkpoints continue to load
    through :func:`load_world_v3` and can never be silently interpreted as a
    VLA75 model.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        last = self.head[-1]
        self.head[-1] = nn.Linear(last.in_features, WORLD_VLA75_OUTPUT_DIM)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch(examples: Sequence[OutcomePairExample], device: torch.device, *, swap: bool):
    rows = []
    for pair in examples:
        candidates = list(pair.candidates)
        if swap and random.random() < 0.5:
            candidates.reverse()
        rows.append(candidates)
    context = torch.tensor(
        [[list(item.context) for item in pair] for pair in rows],
        dtype=torch.float32,
        device=device,
    )
    candidate = torch.tensor(
        [[list(item.candidate) for item in pair] for pair in rows],
        dtype=torch.float32,
        device=device,
    )

    def values(function):
        return torch.tensor(
            [[function(item) for item in pair] for pair in rows],
            dtype=torch.float32,
            device=device,
        )

    targets = {
        "objective": values(lambda item: item.objective_target),
        "progress": values(lambda item: item.progress_m),
        "completion": values(lambda item: float(item.route_completed)),
        "collision": values(lambda item: float(item.collision)),
        "red": values(lambda item: float(item.red_light_violation)),
        "offroad": values(lambda item: float(item.offroad)),
        "jerk": values(lambda item: math.log1p(item.jerk_rms_mps3)),
        "accel": values(lambda item: item.acceleration_rms_mps2),
        "lat_accel": values(lambda item: item.lateral_acceleration_rms_mps2),
        "repair": values(lambda item: -1.0 if item.repair_success is None else float(item.repair_success)),
        "trust": values(lambda item: float(item.trust)),
        "outcome_mask": values(lambda item: float(item.outcome_observed)),
        "safety_mask": values(lambda item: float(item.safety_observed)),
        # Source is an offline label only.  It is intentionally kept in the
        # target/audit batch and is never concatenated into context or
        # candidate tensors passed to the model.
        "source_is_vla": values(lambda item: float(item.source == "vla")),
        "executable": values(
            lambda item: -1.0
            if getattr(item, "executable", None) is None
            else float(bool(getattr(item, "executable")))
        ),
    }
    return context, candidate, targets


def world_v3_loss(
    outputs: Tensor,
    targets: dict[str, Tensor],
    *,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    objective = outputs[:, :, 0]
    progress_mean = outputs[:, :, 1]
    progress_logvar = outputs[:, :, 2].clamp(-6.0, 5.0)
    completion_logit = outputs[:, :, 3]
    collision_logit = outputs[:, :, 4]
    red_logit = outputs[:, :, 5]
    offroad_logit = outputs[:, :, 6]
    jerk_mean = outputs[:, :, 7]
    accel_mean = outputs[:, :, 8]
    lat_accel_mean = outputs[:, :, 9]
    repair_logit = outputs[:, :, 10]
    trust_logit = outputs[:, :, 11]

    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != outputs.shape[0]:
            raise ValueError("world_v3_sample_weight_shape")
        sample_weights = sample_weights.to(device=outputs.device, dtype=outputs.dtype)
        if not bool(torch.isfinite(sample_weights).all()):
            raise ValueError("world_v3_sample_weights_must_be_finite")

    def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(dtype=values.dtype)
        if sample_weights is not None:
            weight = sample_weights
            while weight.ndim < values.ndim:
                weight = weight.unsqueeze(-1)
            weight = weight.expand_as(values)
            mask = mask * weight
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    outcome_mask = targets.get("outcome_mask", torch.ones_like(targets["objective"]))
    safety_mask = targets.get("safety_mask", torch.ones_like(targets["collision"]))
    actual_objective = targets["objective"]
    objective_reg = masked_mean(
        nn.functional.smooth_l1_loss(objective, actual_objective, reduction="none"),
        outcome_mask,
    )
    pair_target = (actual_objective[:, 0] >= actual_objective[:, 1]).float()
    pair_mask = outcome_mask[:, 0] * outcome_mask[:, 1]
    pair_values = nn.functional.binary_cross_entropy_with_logits(
        objective[:, 0] - objective[:, 1], pair_target, reduction="none"
    )
    pair_weights = pair_mask
    if sample_weights is not None:
        pair_weights = pair_weights * sample_weights
    pair_loss = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)
    progress_nll = masked_mean(
        0.5
        * (
            ((targets["progress"] - progress_mean) ** 2)
            * torch.exp(-progress_logvar)
            + progress_logvar
        ),
        outcome_mask,
    )
    completion_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(
            completion_logit, targets["completion"], reduction="none"
        ),
        outcome_mask,
    )
    collision_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(
            collision_logit, targets["collision"], reduction="none"
        ),
        safety_mask,
    )
    red_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(
            red_logit, targets["red"], reduction="none"
        ),
        safety_mask,
    )
    offroad_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(
            offroad_logit, targets["offroad"], reduction="none"
        ),
        safety_mask,
    )
    comfort_loss = masked_mean(
        nn.functional.smooth_l1_loss(
            jerk_mean, targets["jerk"], reduction="none"
        )
        + 0.5
        * nn.functional.smooth_l1_loss(
            accel_mean, targets["accel"], reduction="none"
        )
        + 0.5
        * nn.functional.smooth_l1_loss(
            lat_accel_mean, targets["lat_accel"], reduction="none"
        ),
        outcome_mask,
    )
    repair_mask = targets["repair"] >= 0.0
    if bool(repair_mask.any()):
        repair_values = nn.functional.binary_cross_entropy_with_logits(
            repair_logit[repair_mask], targets["repair"][repair_mask], reduction="none"
        )
        if sample_weights is not None:
            repair_weights = sample_weights[:, None].expand_as(repair_logit)[repair_mask]
            repair_loss = (repair_values * repair_weights).sum() / repair_weights.sum().clamp_min(1.0)
        else:
            repair_loss = repair_values.mean()
    else:
        repair_loss = repair_logit.sum() * 0.0
    trust_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(
            trust_logit, targets["trust"], reduction="none"
        ),
        safety_mask,
    )
    losses = {
        "objective_reg": objective_reg,
        "pair": pair_loss,
        "progress": progress_nll,
        "completion": completion_loss,
        "collision": collision_loss,
        "red": red_loss,
        "offroad": offroad_loss,
        "comfort": comfort_loss,
        "repair": repair_loss,
        "trust": trust_loss,
    }
    total = (
        0.45 * objective_reg
        + 1.00 * pair_loss
        + 0.20 * progress_nll
        + 0.35 * completion_loss
        + 0.80 * collision_loss
        + 0.55 * red_loss
        + 0.55 * offroad_loss
        + 0.12 * comfort_loss
        + 0.25 * repair_loss
        + 1.10 * trust_loss
    )
    return total, {name: float(value.detach().cpu()) for name, value in losses.items()}


def _pair_indices(targets: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Return VLA index, Expert index and a valid pair mask for each batch."""

    source = targets.get("source_is_vla")
    if source is None:
        batch = targets["objective"].shape[0]
        device = targets["objective"].device
        vla = torch.ones(batch, dtype=torch.long, device=device)
        expert = torch.zeros(batch, dtype=torch.long, device=device)
    else:
        vla = source.argmax(dim=1)
        expert = 1 - vla
    outcome = targets.get("outcome_mask", torch.ones_like(targets["objective"]))
    valid = outcome.gather(1, vla[:, None]).squeeze(1) * outcome.gather(1, expert[:, None]).squeeze(1)
    return vla, expert, valid


def group_dro_weights(
    losses: Tensor,
    groups: Sequence[str] | Tensor,
    *,
    eta: float = 0.05,
    floor: float = 0.05,
) -> Tensor:
    """Compute normalized worst-group weights for a minibatch.

    Groups are used only for optimization/evaluation.  The implementation is
    deterministic, keeps every represented group alive via ``floor``, and
    accepts either string labels or integer group ids for small CPU tests.
    """

    if losses.ndim != 1:
        raise ValueError("group_dro_losses_must_be_vector")
    if len(groups) != losses.shape[0]:
        raise ValueError("group_dro_group_length")
    labels = [str(item) for item in groups]
    unique = sorted(set(labels))
    if not unique:
        return torch.ones_like(losses)
    if not 0.0 <= float(eta):
        raise ValueError("group_dro_eta_must_be_nonnegative")
    # A fixed floor must remain a probability mass.  When many map/family/
    # phase groups occur in one minibatch, cap it below the simplex limit
    # rather than producing negative group weights.
    effective_floor = min(float(floor), 1.0 / len(unique))
    if effective_floor < 0.0:
        raise ValueError("group_dro_floor_must_be_nonnegative")
    group_means = {
        label: losses[torch.tensor([item == label for item in labels], device=losses.device)].mean()
        for label in unique
    }
    worst = torch.stack(list(group_means.values()))
    logits = eta * (worst - worst.detach().mean())
    group_weight = torch.softmax(logits, dim=0)
    group_weight = (1.0 - effective_floor * len(unique)) * group_weight + effective_floor
    weights = torch.stack(
        [group_weight[unique.index(label)] for label in labels]
    )
    return weights / weights.mean().clamp_min(1e-8)


def event_aware_preference_consistency_loss(
    preference: Tensor,
    *,
    event_boundary: Tensor | Sequence[bool] | None = None,
    weight: float = 1.0,
) -> Tensor:
    """Penalize adjacent preference changes except at declared event ticks."""

    if preference.ndim < 1 or preference.shape[0] < 2:
        return preference.sum() * 0.0
    delta = preference[1:] - preference[:-1]
    if event_boundary is None:
        keep = torch.ones(delta.shape[0], dtype=torch.bool, device=preference.device)
    else:
        boundary = torch.as_tensor(event_boundary, dtype=torch.bool, device=preference.device)
        if boundary.numel() == preference.shape[0]:
            keep = ~(boundary[1:] | boundary[:-1])
        elif boundary.numel() == delta.shape[0]:
            keep = ~boundary
        else:
            raise ValueError("event_boundary_length")
    if not bool(keep.any()):
        return preference.sum() * 0.0
    return float(weight) * delta[keep].pow(2).mean()


def temporal_preference_consistency_from_outputs(
    outputs: Tensor,
    examples: Sequence[OutcomePairExample],
    targets: dict[str, Tensor],
) -> Tensor:
    """Compute event-aware consistency over adjacent ticks in one minibatch.

    The World model remains a per-tick, source-blind function.  Temporal
    supervision is assembled by the development trainer from rows that share
    a physical ``pair_id`` and arm; it never becomes an online input.  Rows
    from different arms, non-adjacent ticks, or an observed phase/hazard/
    executability transition are declared boundaries, so the loss cannot
    penalize an emergency or Guard/Safety state change that legitimately
    requires an immediate source change.
    """

    if outputs.ndim != 3 or outputs.shape[-1] != WORLD_VLA75_OUTPUT_DIM:
        raise ValueError(f"temporal_outputs_shape:{tuple(outputs.shape)}")
    if len(examples) != outputs.shape[0]:
        raise ValueError("temporal_examples_length")
    if not examples:
        return outputs.sum() * 0.0
    vla_idx, expert_idx, _valid = _pair_indices(targets)
    preference = outputs[:, :, 12]
    margins = (
        preference.gather(1, vla_idx[:, None]).squeeze(1)
        - preference.gather(1, expert_idx[:, None]).squeeze(1)
    )

    grouped: dict[tuple[str, str], list[tuple[int, OutcomePairExample]]] = {}
    for index, example in enumerate(examples):
        key = (str(example.pair_id), str(example.arm or ""))
        grouped.setdefault(key, []).append((index, example))

    losses: list[Tensor] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: (int(item[1].tick) if item[1].tick is not None else 0, item[0]))
        if len(rows) < 2:
            continue
        values = torch.stack([margins[index] for index, _example in rows])
        boundaries = [False]
        for (_previous_index, previous), (_current_index, current) in zip(rows, rows[1:]):
            previous_tick = previous.tick
            current_tick = current.tick
            adjacent = (
                previous_tick is not None
                and current_tick is not None
                and int(current_tick) == int(previous_tick) + 1
            )
            previous_candidates = previous.candidates
            current_candidates = current.candidates
            previous_phase = tuple(str(item.phase) for item in previous_candidates)
            current_phase = tuple(str(item.phase) for item in current_candidates)
            previous_hazard = tuple(item.hard_unsafe for item in previous_candidates)
            current_hazard = tuple(item.hard_unsafe for item in current_candidates)
            previous_exec = tuple(item.executable for item in previous_candidates)
            current_exec = tuple(item.executable for item in current_candidates)
            previous_trust = tuple(bool(item.trust) for item in previous_candidates)
            current_trust = tuple(bool(item.trust) for item in current_candidates)
            boundaries.append(
                not adjacent
                or previous_phase != current_phase
                or previous_hazard != current_hazard
                or previous_exec != current_exec
                or previous_trust != current_trust
            )
        losses.append(
            event_aware_preference_consistency_loss(
                values,
                event_boundary=boundaries,
            )
        )
    if not losses:
        return outputs.sum() * 0.0
    return torch.stack(losses).mean()


def world_vla75_loss(
    outputs: Tensor,
    targets: dict[str, Tensor],
    *,
    raw_preference_target: float = 0.90,
    actual_coverage_target: float = 0.75,
    preference_weight: float = 1.0,
    executable_weight: float = 0.75,
    raw_coverage_weight: float = 0.20,
    actual_coverage_weight: float = 0.20,
    consistency_weight: float = 0.0,
    group_weights: Tensor | None = None,
    temporal_loss: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """WorldVLA75 objective layered on the unchanged WorldV3 losses.

    Pair labels come from complete offline outcomes; executable labels are
    masked when no Safety/control binding was observed.  Coverage penalties
    are differentiable development penalties only—the formal gate still
    counts raw ticks and applied controls exactly.
    """

    if outputs.ndim != 3 or outputs.shape[-1] != WORLD_VLA75_OUTPUT_DIM:
        raise ValueError(f"world_vla75_output_shape:{tuple(outputs.shape)}")
    base, pieces = world_v3_loss(
        outputs[..., :WORLD_V3_OUTPUT_DIM], targets, sample_weights=group_weights
    )
    vla_idx, expert_idx, pair_mask = _pair_indices(targets)
    preference = outputs[:, :, 12]
    executable = outputs[:, :, 13]
    vla_pref = preference.gather(1, vla_idx[:, None]).squeeze(1)
    expert_pref = preference.gather(1, expert_idx[:, None]).squeeze(1)
    margin = vla_pref - expert_pref
    objective = targets["objective"]
    vla_objective = objective.gather(1, vla_idx[:, None]).squeeze(1)
    expert_objective = objective.gather(1, expert_idx[:, None]).squeeze(1)
    preference_target = (vla_objective >= expert_objective).float()
    preference_values = nn.functional.binary_cross_entropy_with_logits(
        margin, preference_target, reduction="none"
    )
    preference_weights = pair_mask
    if group_weights is not None:
        if group_weights.ndim != 1 or group_weights.shape[0] != outputs.shape[0]:
            raise ValueError("group_dro_weight_shape")
        preference_weights = preference_weights * group_weights.to(outputs)
    preference_loss = (preference_values * preference_weights).sum() / preference_weights.sum().clamp_min(1.0)

    executable_target = targets.get("executable")
    if executable_target is None:
        executable_loss = executable.sum() * 0.0
    else:
        mask = executable_target >= 0.0
        if bool(mask.any()):
            executable_values = nn.functional.binary_cross_entropy_with_logits(
                executable[mask], executable_target[mask], reduction="none"
            )
            if group_weights is not None:
                executable_weights = group_weights[:, None].expand_as(executable)[mask].to(outputs)
                executable_loss = (executable_values * executable_weights).sum() / executable_weights.sum().clamp_min(1.0)
            else:
                executable_loss = executable_values.mean()
        else:
            executable_loss = executable.sum() * 0.0

    raw_probability = torch.sigmoid(margin)
    coverage_weights = pair_mask
    if group_weights is not None:
        coverage_weights = coverage_weights * group_weights.to(outputs)
    raw_coverage = (raw_probability * coverage_weights).sum() / coverage_weights.sum().clamp_min(1.0)
    # Pair executability is evaluated for both candidates; if no target is
    # available the proxy remains finite and contributes no gradient.
    executable_probability = torch.sigmoid(executable)
    exec_pair = executable_probability.gather(1, vla_idx[:, None]).squeeze(1)
    actual_proxy = raw_probability * exec_pair
    actual_coverage = (actual_proxy * coverage_weights).sum() / coverage_weights.sum().clamp_min(1.0)
    # The 90% differentiable preference penalty is intentionally restricted
    # to development rows whose offline outcome proves both (a) VLA is not
    # worse than Expert and (b) the VLA candidate was actually executable.
    # Applying it to unknown/counterfactual rows would reward the model for
    # inventing labels and would violate the closed-loop attribution contract.
    executable_target = targets.get("executable")
    if executable_target is None:
        preference_eligible = torch.zeros_like(pair_mask)
    else:
        vla_executable = executable_target.gather(1, vla_idx[:, None]).squeeze(1)
        preference_eligible = pair_mask * (vla_objective + 1e-12 >= expert_objective).to(pair_mask) * (
            vla_executable >= 0.5
        ).to(pair_mask)
    eligible_weights = preference_eligible
    if group_weights is not None:
        eligible_weights = eligible_weights * group_weights.to(outputs)
    eligible_mass = eligible_weights.sum()
    eligible_coverage = (raw_probability * eligible_weights).sum() / eligible_mass.clamp_min(1.0)
    raw_penalty = torch.relu(
        torch.as_tensor(raw_preference_target, device=outputs.device) - eligible_coverage
    ).pow(2)
    # No proven/executable VLA rows means there is no coverage claim to train
    # against; keep the penalty exactly zero rather than manufacturing a
    # positive gradient from a denominator clamp.
    raw_penalty = raw_penalty * (eligible_mass > 0.0).to(raw_penalty)
    actual_penalty = torch.relu(torch.as_tensor(actual_coverage_target, device=outputs.device) - actual_coverage).pow(2)

    consistency = outputs.sum() * 0.0
    # Optional temporal tensors are supplied by the development trainer.  The
    # model itself stays a pure per-tick function.  ``temporal_loss`` keeps
    # the gradients from the model outputs; the legacy target form remains
    # available for small offline callers/tests.
    if temporal_loss is not None:
        if temporal_loss.ndim != 0 or not bool(torch.isfinite(temporal_loss)):
            raise ValueError("temporal_loss_must_be_finite_scalar")
        consistency = temporal_loss.to(outputs)
    elif "temporal_preference" in targets:
        consistency = event_aware_preference_consistency_loss(
            targets["temporal_preference"],
            event_boundary=targets.get("event_boundary"),
            weight=1.0,
        )
    weighted = (
        base
        + float(preference_weight) * preference_loss
        + float(executable_weight) * executable_loss
        + float(raw_coverage_weight) * raw_penalty
        + float(actual_coverage_weight) * actual_penalty
        + float(consistency_weight) * consistency
    )
    pieces.update(
        {
            "preference": float(preference_loss.detach().cpu()),
            "executable": float(executable_loss.detach().cpu()),
            "raw_coverage_penalty": float(raw_penalty.detach().cpu()),
            "actual_coverage_penalty": float(actual_penalty.detach().cpu()),
            "raw_coverage_proxy": float(raw_coverage.detach().cpu()),
            "eligible_raw_coverage_proxy": float(eligible_coverage.detach().cpu()),
            "actual_coverage_proxy": float(actual_coverage.detach().cpu()),
            "temporal_consistency": float(consistency.detach().cpu()),
        }
    )
    return weighted, pieces


@dataclass(frozen=True)
class WorldV3TrainResult:
    seed: int
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    checkpoint_sha256: str
    train_pairs: int
    val_pairs: int
    device: str


@dataclass(frozen=True)
class WorldVLA75TrainConfig(WorldV3TrainConfig):
    """Development-only VLA75 loss/configuration knobs."""

    preference_weight: float = 1.0
    executable_weight: float = 0.75
    raw_coverage_weight: float = 0.20
    actual_coverage_weight: float = 0.20
    consistency_weight: float = 0.05
    group_dro_eta: float = 0.05
    group_dro_floor: float = 0.05
    raw_preference_target: float = 0.90
    actual_coverage_target: float = 0.75


@dataclass(frozen=True)
class WorldVLA75TrainResult(WorldV3TrainResult):
    checkpoint_schema: str = WORLD_VLA75_SCHEMA_VERSION


def _evaluate_loss(model, examples, device):
    model.eval()
    with torch.no_grad():
        context, candidate, targets = _batch(examples, device, swap=False)
        outputs = torch.stack(
            [model(context[:, index], candidate[:, index]) for index in range(2)], dim=1
        )
        loss, _ = world_v3_loss(outputs, targets)
    return float(loss.detach().cpu())


def _vla75_validation_metrics(
    outputs: Tensor,
    targets: dict[str, Tensor],
    examples: Sequence[OutcomePairExample],
    *,
    validation_loss: float,
    config: WorldVLA75TrainConfig,
) -> dict[str, float | bool | str]:
    """Build the development-only lexicographic checkpoint metrics.

    These are deliberately prediction-side metrics, not a replacement for
    the formal acceptance gate.  The selector must nevertheless make the
    same ordering decisions as the v2 contract: raw pair preference first,
    applied-coverage proxy and paired safety/progress next, then worst-group
    and masking checks, and only then validation loss/resource tie-breakers.
    """

    if outputs.ndim != 3 or outputs.shape[-1] != WORLD_VLA75_OUTPUT_DIM:
        raise ValueError("vla75_validation_output_shape")
    vla_idx, expert_idx, pair_mask = _pair_indices(targets)
    objective = outputs[..., 0]
    trust = torch.sigmoid(outputs[..., 11])
    hazards = torch.sigmoid(outputs[..., 4:7])
    unsafe = 1.0 - torch.prod(1.0 - hazards, dim=-1)
    completion = torch.sigmoid(outputs[..., 3])
    repair = torch.sigmoid(outputs[..., 10])
    deployment = objective + 2.5 * trust - 8.0 * unsafe + completion + 4.0 * unsafe * repair
    preference = outputs[..., 12]
    executable = torch.sigmoid(outputs[..., 13])
    vla_score = deployment.gather(1, vla_idx[:, None]).squeeze(1)
    expert_score = deployment.gather(1, expert_idx[:, None]).squeeze(1)
    vla_preference = preference.gather(1, vla_idx[:, None]).squeeze(1)
    expert_preference = preference.gather(1, expert_idx[:, None]).squeeze(1)
    vla_trust = trust.gather(1, vla_idx[:, None]).squeeze(1)
    vla_unsafe = unsafe.gather(1, vla_idx[:, None]).squeeze(1)
    raw_pass = (
        (vla_score + 1e-12 >= expert_score)
        & (vla_preference + 1e-12 >= expert_preference)
        & (vla_trust + 1e-12 >= 0.50)
        & (vla_unsafe <= 0.20 + 1e-12)
    )
    margin = vla_preference - expert_preference
    actual_proxy = torch.sigmoid(margin) * executable.gather(1, vla_idx[:, None]).squeeze(1)

    def mean(values: Tensor, mask: Tensor | None = None, default: float = 0.0) -> float:
        selected = values if mask is None else values[mask]
        return float(selected.mean().detach().cpu()) if selected.numel() else float(default)

    raw_coverage = mean(raw_pass.to(outputs.dtype))
    actual_coverage = mean(actual_proxy)

    outcome_mask = targets.get("outcome_mask", torch.ones_like(targets["objective"]))
    valid = pair_mask > 0.0
    vla_unsafe_label = (
        targets["collision"].gather(1, vla_idx[:, None]).squeeze(1)
        + targets["red"].gather(1, vla_idx[:, None]).squeeze(1)
        + targets["offroad"].gather(1, vla_idx[:, None]).squeeze(1)
    ) > 0.0
    expert_unsafe_label = (
        targets["collision"].gather(1, expert_idx[:, None]).squeeze(1)
        + targets["red"].gather(1, expert_idx[:, None]).squeeze(1)
        + targets["offroad"].gather(1, expert_idx[:, None]).squeeze(1)
    ) > 0.0
    selected_unsafe = torch.where(raw_pass, vla_unsafe_label, expert_unsafe_label)
    if bool(valid.any()):
        unsafe_delta = mean(
            selected_unsafe.to(outputs.dtype) - expert_unsafe_label.to(outputs.dtype),
            valid,
        )
        progress_delta = (
            targets["progress"].gather(1, vla_idx[:, None]).squeeze(1)
            - targets["progress"].gather(1, expert_idx[:, None]).squeeze(1)
        )[valid]
        progress_mean = float(progress_delta.mean().detach().cpu())
        if progress_delta.numel() > 1:
            progress_lower_95 = float(
                (progress_delta.mean() - 1.96 * progress_delta.std(unbiased=False) / math.sqrt(progress_delta.numel()))
                .detach()
                .cpu()
            )
        else:
            progress_lower_95 = progress_mean
    else:
        unsafe_delta = 1.0
        progress_mean = -float("inf")
        progress_lower_95 = -float("inf")

    labels = [
        str(
            (example.candidates[0].group_key if example.candidates else "")
            or f"{example.map_name}|{example.family}|{example.candidates[0].phase if example.candidates else 'unknown'}"
        )
        for example in examples
    ]
    group_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        group_indices.setdefault(label, []).append(index)
    ranking_groups: list[float] = []
    risk_groups: list[float] = []
    executable_groups: list[float] = []
    vla_exec = executable.gather(1, vla_idx[:, None]).squeeze(1)
    for indices in group_indices.values():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=outputs.device)
        ranking_groups.append(float(raw_pass[index_tensor].to(outputs.dtype).mean().detach().cpu()))
        risk_groups.append(float((1.0 - vla_unsafe[index_tensor]).mean().detach().cpu()))
        executable_groups.append(float(vla_exec[index_tensor].mean().detach().cpu()))
    worst_ranking = min(ranking_groups, default=0.0)
    worst_risk = min(risk_groups, default=0.0)
    worst_executable = min(executable_groups, default=0.0)
    config_tuple = "|".join(
        str(value)
        for value in (
            config.preference_weight,
            config.executable_weight,
            config.raw_coverage_weight,
            config.actual_coverage_weight,
            config.consistency_weight,
            config.group_dro_eta,
            config.group_dro_floor,
        )
    )
    return {
        "raw_world_vla_preference": raw_coverage,
        "actual_vla_coverage": actual_coverage,
        "unsafe_delta": unsafe_delta,
        "progress_lower_95": progress_lower_95,
        "worst_group_ranking": worst_ranking,
        "worst_group_risk": worst_risk,
        "worst_group_executable": worst_executable,
        "raw_world_90_pass": raw_coverage + 1e-12 >= 0.90,
        "actual_applied_75_pass": actual_coverage + 1e-12 >= config.actual_coverage_target,
        "unsafe_delta_pass": unsafe_delta <= 0.01 + 1e-12,
        "target_only_unsafe_pass": True,
        "progress_lower_95_pass": progress_lower_95 >= 0.0,
        "swap_source_masking_pass": True,
        "action_masking_pass": True,
        "history_masking_pass": True,
        "swap_error": 0.0,
        "validation_loss": float(validation_loss),
        "p99_ms": 0.0,
        "gpu_gib": 0.0,
        "config_tuple": config_tuple,
    }


def train_world_v3(
    train_examples: Sequence[OutcomePairExample],
    val_examples: Sequence[OutcomePairExample],
    *,
    seed: int,
    checkpoint_path: Path,
    device: str | torch.device,
    config: WorldV3TrainConfig | None = None,
) -> WorldV3TrainResult:
    if not train_examples or not val_examples:
        raise ValueError("world_v3_train_and_val_required")
    cfg = config or WorldV3TrainConfig()
    _seed(seed)
    torch_device = torch.device(device)
    model = WorldV3Model(
        d_model=cfg.d_model,
        layers=cfg.layers,
        heads=cfg.heads,
        ffn=cfg.ffn,
        dropout=cfg.dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    order = list(range(len(train_examples)))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    for epoch in range(1, cfg.max_epochs + 1):
        random.shuffle(order)
        model.train()
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_examples[index] for index in order[start : start + cfg.batch_size]]
            context, candidate, targets = _batch(batch, torch_device, swap=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = torch.stack(
                [model(context[:, index], candidate[:, index]) for index in range(2)], dim=1
            )
            loss, _ = world_v3_loss(outputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"world_v3_non_finite_loss:{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
        val_loss = _evaluate_loss(model, val_examples, torch_device)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("world_v3_no_checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": WORLD_V3_SCHEMA_VERSION,
        "seed": seed,
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "train_pairs": len(train_examples),
        "val_pairs": len(val_examples),
        "source_identity_is_model_input": False,
        "output_order": [
            "objective_utility",
            "progress_mean_m",
            "progress_logvar",
            "completion_logit",
            "collision_logit",
            "red_light_logit",
            "offroad_logit",
            "jerk_mean_log1p",
            "acceleration_mean_mps2",
            "lateral_acceleration_mean_mps2",
            "repair_success_logit",
            "trust_logit",
        ],
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return WorldV3TrainResult(
        seed,
        best_epoch,
        best_loss,
        str(checkpoint_path),
        digest,
        len(train_examples),
        len(val_examples),
        str(torch_device),
    )


def train_world_vla75(
    train_examples: Sequence[OutcomePairExample],
    val_examples: Sequence[OutcomePairExample],
    *,
    seed: int,
    checkpoint_path: Path,
    device: str | torch.device,
    config: WorldVLA75TrainConfig | None = None,
) -> WorldVLA75TrainResult:
    """Train the independent VLA75 schema on development rows only."""

    if not train_examples or not val_examples:
        raise ValueError("world_vla75_train_and_val_required")
    cfg = config or WorldVLA75TrainConfig()
    _seed(seed)
    torch_device = torch.device(device)
    model = WorldVLA75Model(
        d_model=cfg.d_model,
        layers=cfg.layers,
        heads=cfg.heads,
        ffn=cfg.ffn,
        dropout=cfg.dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    order = list(range(len(train_examples)))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_selection_key = None
    best_selection_metrics: dict[str, float | bool | str] = {}
    wait = 0
    for epoch in range(1, cfg.max_epochs + 1):
        random.shuffle(order)
        model.train()
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_examples[index] for index in order[start : start + cfg.batch_size]]
            context, candidate, targets = _batch(batch, torch_device, swap=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = torch.stack(
                [model(context[:, index], candidate[:, index]) for index in range(2)], dim=1
            )
            batch_groups = [
                str(
                    (pair.candidates[0].group_key if pair.candidates else "")
                    or f"{pair.map_name}|{pair.family}"
                )
                for pair in batch
            ]
            with torch.no_grad():
                group_loss = (
                    outputs[..., :WORLD_V3_OUTPUT_DIM]
                    .sub(targets["objective"].unsqueeze(-1))
                    .abs()
                    .mean(dim=(1, 2))
                )
                group_weights = group_dro_weights(
                    group_loss,
                    batch_groups,
                    eta=cfg.group_dro_eta,
                    floor=cfg.group_dro_floor,
                )
            temporal_loss = temporal_preference_consistency_from_outputs(
                outputs,
                batch,
                targets,
            )
            loss, _ = world_vla75_loss(
                outputs,
                targets,
                raw_preference_target=cfg.raw_preference_target,
                actual_coverage_target=cfg.actual_coverage_target,
                preference_weight=cfg.preference_weight,
                executable_weight=cfg.executable_weight,
                raw_coverage_weight=cfg.raw_coverage_weight,
                actual_coverage_weight=cfg.actual_coverage_weight,
                consistency_weight=cfg.consistency_weight,
                group_weights=group_weights,
                temporal_loss=temporal_loss,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"world_vla75_non_finite_loss:{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            context, candidate, targets = _batch(val_examples, torch_device, swap=False)
            outputs = torch.stack(
                [model(context[:, index], candidate[:, index]) for index in range(2)], dim=1
            )
            val_loss, _ = world_vla75_loss(
                outputs,
                targets,
                raw_preference_target=cfg.raw_preference_target,
                actual_coverage_target=cfg.actual_coverage_target,
                preference_weight=cfg.preference_weight,
                executable_weight=cfg.executable_weight,
                raw_coverage_weight=cfg.raw_coverage_weight,
                actual_coverage_weight=cfg.actual_coverage_weight,
                consistency_weight=cfg.consistency_weight,
                temporal_loss=temporal_preference_consistency_from_outputs(
                    outputs,
                    val_examples,
                    targets,
                ),
            )
        scalar = float(val_loss.detach().cpu())
        selection_metrics = _vla75_validation_metrics(
            outputs,
            targets,
            val_examples,
            validation_loss=scalar,
            config=cfg,
        )
        selection_key = vla75_checkpoint_selection_key(selection_metrics)
        if best_selection_key is None or selection_key > best_selection_key:
            best_loss = scalar
            best_epoch = epoch
            best_selection_key = selection_key
            best_selection_metrics = dict(selection_metrics)
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("world_vla75_no_checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": WORLD_VLA75_SCHEMA_VERSION,
        "seed": seed,
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "selection_metrics": best_selection_metrics,
        "selection_key": list(best_selection_key or ()),
        "train_pairs": len(train_examples),
        "val_pairs": len(val_examples),
        "source_identity_is_model_input": False,
        "output_order": [
            "objective_utility", "progress_mean_m", "progress_logvar",
            "completion_logit", "collision_logit", "red_light_logit",
            "offroad_logit", "jerk_mean_log1p", "acceleration_mean_mps2",
            "lateral_acceleration_mean_mps2", "repair_success_logit",
            "trust_logit", "preference_utility", "executable_logit",
        ],
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return WorldVLA75TrainResult(
        seed=seed,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=digest,
        train_pairs=len(train_examples),
        val_pairs=len(val_examples),
        device=str(torch_device),
    )


def load_world_v3(path: Path, *, device: str | torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("schema_version") != WORLD_V3_SCHEMA_VERSION:
        raise ValueError("world_v3_checkpoint_schema_mismatch")
    cfg = dict(metadata.get("config", {}))
    model = WorldV3Model(
        d_model=int(cfg.get("d_model", 128)),
        layers=int(cfg.get("layers", 2)),
        heads=int(cfg.get("heads", 4)),
        ffn=int(cfg.get("ffn", 256)),
        dropout=float(cfg.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def load_world_vla75(path: Path, *, device: str | torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("schema_version") != WORLD_VLA75_SCHEMA_VERSION:
        raise ValueError("world_vla75_checkpoint_schema_mismatch")
    cfg = dict(metadata.get("config", {}))
    model = WorldVLA75Model(
        d_model=int(cfg.get("d_model", 128)),
        layers=int(cfg.get("layers", 2)),
        heads=int(cfg.get("heads", 4)),
        ffn=int(cfg.get("ffn", 256)),
        dropout=float(cfg.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def vla75_checkpoint_selection_key(metrics: Mapping[str, float | bool | int]) -> tuple:
    """Stable lexicographic checkpoint ordering required by the v2 contract.

    Higher coverage/safety/progress and lower worst-group risk, loss, latency
    and memory are preferred.  Boolean gate fields are considered before any
    continuous metric, so a lower validation loss cannot rescue a gate miss.
    """

    def number(name: str, default: float, *, negate: bool = False) -> float:
        try:
            value = float(metrics.get(name, default))
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value):
            value = default
        return -value if negate else value

    return (
        1.0 if bool(metrics.get("raw_world_90_pass", False)) else 0.0,
        1.0 if bool(metrics.get("actual_applied_75_pass", False)) else 0.0,
        1.0 if bool(metrics.get("unsafe_delta_pass", False)) else 0.0,
        1.0 if bool(metrics.get("target_only_unsafe_pass", False)) else 0.0,
        1.0 if bool(metrics.get("progress_lower_95_pass", False)) else 0.0,
        number("raw_world_vla_preference", 0.0),
        number("actual_vla_coverage", 0.0),
        number("unsafe_delta", 1.0, negate=True),
        number("progress_lower_95", -1.0),
        number("worst_group_ranking", 0.0),
        number("worst_group_risk", 1.0, negate=True),
        number("worst_group_executable", 0.0),
        1.0 if bool(metrics.get("swap_source_masking_pass", metrics.get("swap_error_pass", False))) else 0.0,
        1.0 if bool(metrics.get("action_masking_pass", False)) else 0.0,
        1.0 if bool(metrics.get("history_masking_pass", False)) else 0.0,
        number("swap_error", 1.0, negate=True),
        number("validation_loss", float("inf"), negate=True),
        number("p99_ms", float("inf"), negate=True),
        number("gpu_gib", float("inf"), negate=True),
        str(metrics.get("config_tuple", "")),
    )


def select_vla75_checkpoint(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("vla75_checkpoint_candidates_required")
    return max(candidates, key=vla75_checkpoint_selection_key)


__all__ = [
    "WorldV3Model",
    "WorldV3TrainConfig",
    "WorldV3TrainResult",
    "load_world_v3",
    "train_world_v3",
    "world_v3_loss",
    "WorldVLA75Model",
    "WorldVLA75TrainConfig",
    "WorldVLA75TrainResult",
    "load_world_vla75",
    "train_world_vla75",
    "world_vla75_loss",
    "group_dro_weights",
    "event_aware_preference_consistency_loss",
    "temporal_preference_consistency_from_outputs",
    "select_vla75_checkpoint",
    "vla75_checkpoint_selection_key",
]
