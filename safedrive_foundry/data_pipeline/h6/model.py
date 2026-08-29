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
    stable_sha256,
)
from data_pipeline.h6.contracts import (
    WORLD_V3_OUTPUT_DIM,
    WORLD_V3_SCHEMA_VERSION,
    WORLD_VLA75_OUTPUT_DIM,
    WORLD_VLA75_SCHEMA_VERSION,
)
from data_pipeline.h6.dataset import OutcomePairExample, outcome_examples_lineage_sha256


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
        "repair_mask": values(lambda item: float(item.repair_success is not None)),
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
        "executable_mask": values(
            lambda item: float(getattr(item, "executable", None) is not None)
        ),
    }
    pair_mask = (targets["outcome_mask"][:, 0] > 0.0) & (
        targets["outcome_mask"][:, 1] > 0.0
    )
    targets["pair_mask"] = pair_mask.to(dtype=torch.float32)
    targets["pair_preference"] = (
        targets["objective"][:, 0] >= targets["objective"][:, 1]
    ).to(dtype=torch.float32)
    vla_index = targets["source_is_vla"].argmax(dim=1)
    expert_index = 1 - vla_index
    targets["vla_preference"] = (
        targets["objective"].gather(1, vla_index[:, None]).squeeze(1)
        >= targets["objective"].gather(1, expert_index[:, None]).squeeze(1)
    ).to(dtype=torch.float32)
    return context, candidate, targets


def _outcome_group(example: OutcomePairExample) -> str:
    declared = next(
        (
            str(candidate.group_key)
            for candidate in example.candidates
            if str(candidate.group_key or "").strip()
            and str(candidate.group_key).lower() != "unknown"
        ),
        "",
    )
    return declared or "|".join(
        (
            str(example.map_name),
            str(example.family),
            str(example.weather),
        )
    )


WORLD_V3_HEAD_WEIGHTS: dict[str, float] = {
    "objective": 0.45,
    "pair_preference": 1.00,
    "progress": 0.20,
    "completion": 0.35,
    "collision": 0.80,
    "red_light": 0.55,
    "offroad": 0.55,
    "comfort": 0.12,
    "repair": 0.25,
    "trust": 1.10,
}
WORLD_VLA75_EXTRA_HEAD_WEIGHTS: dict[str, float] = {
    "executable": 0.75,
}


@dataclass(frozen=True)
class PerSampleLossReport:
    """Unreduced supervised losses used by training, DRO and evaluation.

    ``head_losses`` and ``head_masks`` are one-dimensional tensors with one
    entry per outcome pair.  Candidate heads are reduced *within* each pair
    before they are stored here.  ``per_sample`` is therefore the frozen
    effective-head weighted mean, not a broadcast scalar objective.
    """

    head_losses: Mapping[str, Tensor]
    head_masks: Mapping[str, Tensor]
    head_weights: Mapping[str, float]
    per_sample: Tensor
    valid_samples: Tensor

    def detached_counts(self) -> dict[str, int]:
        return {
            name: int(mask.detach().to(dtype=torch.bool).sum().cpu())
            for name, mask in self.head_masks.items()
        }


def _candidate_to_sample(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("per_sample_candidate_head_shape")
    finite = torch.isfinite(values)
    valid = mask.to(dtype=torch.bool) & finite
    count = valid.sum(dim=1)
    collapsed = (torch.where(valid, values, torch.zeros_like(values))).sum(dim=1)
    collapsed = collapsed / count.clamp_min(1).to(values)
    return collapsed, count > 0


def _build_per_sample_report(
    head_values: Mapping[str, Tensor],
    head_masks: Mapping[str, Tensor],
    head_weights: Mapping[str, float],
) -> PerSampleLossReport:
    if set(head_values) != set(head_masks) or set(head_values) != set(head_weights):
        raise ValueError("per_sample_head_contract_mismatch")
    if not head_values:
        raise ValueError("per_sample_heads_required")
    batch = next(iter(head_values.values())).shape[0]
    reference = next(iter(head_values.values()))
    numerator = torch.zeros(batch, dtype=reference.dtype, device=reference.device)
    denominator = torch.zeros_like(numerator)
    normalized_losses: dict[str, Tensor] = {}
    normalized_masks: dict[str, Tensor] = {}
    for name, value in head_values.items():
        mask = head_masks[name].to(dtype=torch.bool, device=value.device)
        if value.ndim == 2:
            value, mask = _candidate_to_sample(value, mask)
        elif value.ndim != 1 or value.shape[0] != batch or mask.shape != value.shape:
            raise ValueError(f"per_sample_head_shape:{name}")
        finite = torch.isfinite(value)
        mask = mask & finite
        value = torch.where(mask, value, torch.zeros_like(value))
        weight = float(head_weights[name])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"per_sample_head_weight:{name}")
        normalized_losses[name] = value
        normalized_masks[name] = mask
        numerator = numerator + weight * value
        denominator = denominator + weight * mask.to(value)
    valid = denominator > 0.0
    per_sample = numerator / denominator.clamp_min(torch.finfo(reference.dtype).eps)
    per_sample = torch.where(valid, per_sample, torch.zeros_like(per_sample))
    return PerSampleLossReport(
        head_losses=normalized_losses,
        head_masks=normalized_masks,
        head_weights=dict(head_weights),
        per_sample=per_sample,
        valid_samples=valid,
    )


def world_v3_per_sample_loss_report(
    outputs: Tensor,
    targets: Mapping[str, Tensor],
) -> PerSampleLossReport:
    """Return the only supervised per-sample World-v3 loss definition."""

    if outputs.ndim != 3 or outputs.shape[-1] != WORLD_V3_OUTPUT_DIM:
        raise ValueError(f"world_v3_output_shape:{tuple(outputs.shape)}")
    objective = outputs[:, :, 0]
    progress_mean = outputs[:, :, 1]
    progress_logvar = outputs[:, :, 2].clamp(-6.0, 5.0)
    outcome_mask = targets.get("outcome_mask", torch.ones_like(targets["objective"])) > 0.0
    safety_mask = targets.get("safety_mask", torch.ones_like(targets["collision"])) > 0.0
    repair_mask = targets.get("repair_mask", targets["repair"] >= 0.0) > 0.0
    actual_objective = targets["objective"]
    pair_mask = targets.get(
        "pair_mask", (outcome_mask[:, 0] & outcome_mask[:, 1]).to(outputs)
    ) > 0.0
    pair_target = targets.get(
        "pair_preference",
        (actual_objective[:, 0] >= actual_objective[:, 1]).to(outputs),
    ).to(outputs)
    head_values = {
        "objective": nn.functional.smooth_l1_loss(
            objective, actual_objective, reduction="none"
        ),
        "pair_preference": nn.functional.binary_cross_entropy_with_logits(
            objective[:, 0] - objective[:, 1], pair_target, reduction="none"
        ),
        "progress": 0.5
        * (
            ((targets["progress"] - progress_mean) ** 2)
            * torch.exp(-progress_logvar)
            + progress_logvar
        ),
        "completion": nn.functional.binary_cross_entropy_with_logits(
            outputs[:, :, 3], targets["completion"], reduction="none"
        ),
        "collision": nn.functional.binary_cross_entropy_with_logits(
            outputs[:, :, 4], targets["collision"], reduction="none"
        ),
        "red_light": nn.functional.binary_cross_entropy_with_logits(
            outputs[:, :, 5], targets["red"], reduction="none"
        ),
        "offroad": nn.functional.binary_cross_entropy_with_logits(
            outputs[:, :, 6], targets["offroad"], reduction="none"
        ),
        "comfort": nn.functional.smooth_l1_loss(
            outputs[:, :, 7], targets["jerk"], reduction="none"
        )
        + 0.5
        * nn.functional.smooth_l1_loss(
            outputs[:, :, 8], targets["accel"], reduction="none"
        )
        + 0.5
        * nn.functional.smooth_l1_loss(
            outputs[:, :, 9], targets["lat_accel"], reduction="none"
        ),
        "repair": nn.functional.binary_cross_entropy_with_logits(
            outputs[:, :, 10], targets["repair"].clamp(0.0, 1.0), reduction="none"
        ),
        "trust": nn.functional.binary_cross_entropy_with_logits(
            outputs[:, :, 11], targets["trust"], reduction="none"
        ),
    }
    head_masks = {
        "objective": outcome_mask,
        "pair_preference": pair_mask,
        "progress": outcome_mask,
        "completion": outcome_mask,
        "collision": safety_mask,
        "red_light": safety_mask,
        "offroad": safety_mask,
        "comfort": outcome_mask,
        "repair": repair_mask,
        "trust": safety_mask,
    }
    return _build_per_sample_report(head_values, head_masks, WORLD_V3_HEAD_WEIGHTS)


def _reduce_per_sample_report(
    report: PerSampleLossReport,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    valid = report.valid_samples
    if not bool(valid.any()):
        raise ValueError("world_loss_batch_has_no_valid_supervision")
    weights = torch.ones_like(report.per_sample)
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape != report.per_sample.shape:
            raise ValueError("world_v3_sample_weight_shape")
        weights = sample_weights.to(report.per_sample)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
            raise ValueError("world_v3_sample_weights_must_be_finite_nonnegative")
    weights = weights * valid.to(weights)
    if not bool((weights > 0.0).any()):
        raise ValueError("world_loss_valid_sample_weight_is_zero")
    total = (report.per_sample * weights).sum() / weights.sum()
    pieces: dict[str, float | int] = {
        "valid_samples": int(valid.sum().detach().cpu()),
        "invalid_samples": int((~valid).sum().detach().cpu()),
    }
    for name, values in report.head_losses.items():
        mask = report.head_masks[name]
        head_weights = weights * mask.to(weights)
        if bool((head_weights > 0.0).any()):
            head_loss = (values * head_weights).sum() / head_weights.sum()
            pieces[name] = float(head_loss.detach().cpu())
        else:
            pieces[name] = float("nan")
        pieces[f"{name}_valid_count"] = int(mask.sum().detach().cpu())
    return total, pieces


def world_v3_loss(
    outputs: Tensor,
    targets: dict[str, Tensor],
    *,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    report = world_v3_per_sample_loss_report(outputs, targets)
    return _reduce_per_sample_report(report, sample_weights)


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


@dataclass(frozen=True)
class GroupDROBatchReport:
    sample_weights: Tensor
    groups: Mapping[str, Mapping[str, float | int | str | None]]


class GroupDROState:
    """Persistent exponentiated-gradient Group-DRO state.

    All train groups are registered before optimization.  Missing groups keep
    their previous probability mass and are reported as ``NOT_MEASURED``;
    invalid supervised rows receive zero sample weight.
    """

    def __init__(
        self,
        groups: Sequence[str],
        *,
        eta: float = 0.05,
        floor: float = 0.05,
    ) -> None:
        labels = sorted({str(item) for item in groups})
        if not labels:
            raise ValueError("group_dro_registered_groups_required")
        if not math.isfinite(float(eta)) or float(eta) < 0.0:
            raise ValueError("group_dro_eta_must_be_nonnegative")
        if not math.isfinite(float(floor)) or float(floor) < 0.0:
            raise ValueError("group_dro_floor_must_be_nonnegative")
        self.groups = tuple(labels)
        self.eta = float(eta)
        self.floor = min(float(floor), 1.0 / len(labels))
        self._weights = {label: 1.0 / len(labels) for label in labels}

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def evaluate_batch(
        self,
        losses: Tensor,
        valid_samples: Tensor,
        groups: Sequence[str] | Tensor,
        *,
        update: bool,
    ) -> GroupDROBatchReport:
        if losses.ndim != 1 or valid_samples.shape != losses.shape:
            raise ValueError("group_dro_losses_must_be_vector")
        labels = [str(item) for item in groups]
        if len(labels) != losses.shape[0]:
            raise ValueError("group_dro_group_length")
        unknown = sorted(set(labels) - set(self.groups))
        if unknown:
            raise ValueError(f"group_dro_unregistered_groups:{','.join(unknown)}")
        valid = valid_samples.to(device=losses.device, dtype=torch.bool) & torch.isfinite(losses)
        group_means: dict[str, float | None] = {}
        group_counts: dict[str, int] = {}
        for label in self.groups:
            selected = valid & torch.as_tensor(
                [item == label for item in labels], dtype=torch.bool, device=losses.device
            )
            count = int(selected.sum().detach().cpu())
            group_counts[label] = count
            group_means[label] = (
                float(losses[selected].detach().mean().cpu()) if count else None
            )
        if update:
            observed = [
                label for label in self.groups if group_means[label] is not None
            ]
            if observed:
                # Preserve every absent group's historical probability exactly.
                # Reweight only within the probability mass already owned by
                # groups measured in this batch.
                observed_mass = sum(self._weights[label] for label in observed)
                scores: dict[str, float] = {}
                for label in observed:
                    exponent = max(
                        -80.0,
                        min(80.0, self.eta * float(group_means[label])),
                    )
                    scores[label] = self._weights[label] * math.exp(exponent)
                score_total = sum(scores.values())
                if not math.isfinite(score_total) or score_total <= 0.0:
                    raise ValueError("group_dro_observed_weight_normalization")
                floor_mass = self.floor * len(observed)
                if observed_mass + 1e-12 < floor_mass:
                    raise ValueError("group_dro_observed_mass_below_floor")
                residual = max(0.0, observed_mass - floor_mass)
                for label in observed:
                    self._weights[label] = (
                        self.floor + residual * scores[label] / score_total
                    )
        sample_weights = torch.zeros_like(losses)
        for index, label in enumerate(labels):
            count = group_counts[label]
            if bool(valid[index]) and count > 0:
                # Dividing by the current batch count makes the reduced loss
                # a weighted mean of group means rather than a row-frequency
                # weighted objective.
                sample_weights[index] = self._weights[label] / count
        observed_mass = sample_weights.sum()
        if bool(valid.any()) and not bool(observed_mass > 0.0):
            raise ValueError("group_dro_no_weight_for_valid_samples")
        report = {
            label: {
                "status": "MEASURED" if group_counts[label] else "NOT_MEASURED",
                "loss": group_means[label],
                "count": group_counts[label],
                "weight": self._weights[label],
            }
            for label in self.groups
        }
        return GroupDROBatchReport(sample_weights=sample_weights, groups=report)


def group_dro_weights(
    losses: Tensor,
    groups: Sequence[str] | Tensor,
    *,
    eta: float = 0.05,
    floor: float = 0.05,
) -> Tensor:
    """Compatibility wrapper for one-batch callers.

    The trainer uses :class:`GroupDROState` directly so its group weights are
    persistent.  This wrapper still uses the real supplied per-sample loss.
    """

    state = GroupDROState([str(item) for item in groups], eta=eta, floor=floor)
    report = state.evaluate_batch(
        losses,
        torch.isfinite(losses),
        groups,
        update=True,
    )
    weights = report.sample_weights
    positive = weights > 0.0
    if bool(positive.any()):
        weights = weights / weights[positive].mean()
    return weights


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


def world_vla75_per_sample_loss_report(
    outputs: Tensor,
    targets: Mapping[str, Tensor],
    *,
    preference_weight: float = 1.0,
    executable_weight: float = 0.75,
) -> PerSampleLossReport:
    """Extend the v3 report with the VLA75 pair/executability heads."""

    if outputs.ndim != 3 or outputs.shape[-1] != WORLD_VLA75_OUTPUT_DIM:
        raise ValueError(f"world_vla75_output_shape:{tuple(outputs.shape)}")
    base = world_v3_per_sample_loss_report(
        outputs[..., :WORLD_V3_OUTPUT_DIM], targets
    )
    vla_idx, expert_idx, pair_mask = _pair_indices(dict(targets))
    preference = outputs[:, :, 12]
    vla_pref = preference.gather(1, vla_idx[:, None]).squeeze(1)
    expert_pref = preference.gather(1, expert_idx[:, None]).squeeze(1)
    objective = targets["objective"]
    vla_objective = objective.gather(1, vla_idx[:, None]).squeeze(1)
    expert_objective = objective.gather(1, expert_idx[:, None]).squeeze(1)
    preference_target = targets.get(
        "vla_preference", (vla_objective >= expert_objective).to(outputs)
    ).to(outputs)
    explicit_pair_mask = targets.get("pair_mask")
    if explicit_pair_mask is not None:
        pair_mask = pair_mask * (explicit_pair_mask > 0.0).to(pair_mask)
    executable_target = targets.get("executable")
    if executable_target is None:
        executable_target = torch.full_like(outputs[:, :, 13], -1.0)
    head_values = dict(base.head_losses)
    head_masks = dict(base.head_masks)
    head_weights = dict(base.head_weights)
    # VLA75's explicit preference output replaces the legacy objective-logit
    # pair term under the same semantic head; it is not double-counted.
    head_values["pair_preference"] = nn.functional.binary_cross_entropy_with_logits(
        vla_pref - expert_pref, preference_target, reduction="none"
    )
    head_masks["pair_preference"] = pair_mask > 0.0
    head_weights["pair_preference"] = float(preference_weight)
    head_values["executable"] = nn.functional.binary_cross_entropy_with_logits(
        outputs[:, :, 13], executable_target.clamp(0.0, 1.0), reduction="none"
    )
    head_masks["executable"] = (
        targets.get("executable_mask", executable_target >= 0.0) > 0.0
    )
    head_weights["executable"] = float(executable_weight)
    return _build_per_sample_report(head_values, head_masks, head_weights)


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
    supervised = world_vla75_per_sample_loss_report(
        outputs,
        targets,
        preference_weight=preference_weight,
        executable_weight=executable_weight,
    )
    base, pieces = _reduce_per_sample_report(supervised, group_weights)
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
        + float(raw_coverage_weight) * raw_penalty
        + float(actual_coverage_weight) * actual_penalty
        + float(consistency_weight) * consistency
    )
    pieces.update(
        {
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


def _evaluate_vla75_group_supervision(
    model: WorldVLA75Model,
    examples: Sequence[OutcomePairExample],
    device: torch.device,
    config: WorldVLA75TrainConfig,
    group_weights: Mapping[str, float],
) -> dict[str, dict[str, float | int | str | None]]:
    sums = {str(group): 0.0 for group in group_weights}
    counts = {str(group): 0 for group in group_weights}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            context, candidate, targets = _batch(batch, device, swap=False)
            outputs = torch.stack(
                [model(context[:, index], candidate[:, index]) for index in range(2)],
                dim=1,
            )
            report = world_vla75_per_sample_loss_report(
                outputs,
                targets,
                preference_weight=config.preference_weight,
                executable_weight=config.executable_weight,
            )
            for index, example in enumerate(batch):
                group = _outcome_group(example)
                sums.setdefault(group, 0.0)
                counts.setdefault(group, 0)
                if bool(report.valid_samples[index]):
                    sums[group] += float(report.per_sample[index].detach().cpu())
                    counts[group] += 1
    return {
        group: {
            "status": "MEASURED" if counts[group] else "NOT_MEASURED",
            "loss": sums[group] / counts[group] if counts[group] else None,
            "count": counts[group],
            "weight": float(group_weights.get(group, 0.0)),
        }
        for group in sorted(sums)
    }


def _vla75_validation_metrics(
    outputs: Tensor,
    swapped_outputs: Tensor,
    targets: dict[str, Tensor],
    examples: Sequence[OutcomePairExample],
    *,
    validation_loss: float,
    validation_lineage_sha256: str,
    config: WorldVLA75TrainConfig,
    group_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build source-neutral selection metrics from actual evaluator work."""

    if outputs.ndim != 3 or outputs.shape[-1] != WORLD_VLA75_OUTPUT_DIM:
        raise ValueError("vla75_validation_output_shape")
    if swapped_outputs.shape != outputs.shape:
        raise ValueError("vla75_validation_swap_output_shape")
    if len(examples) != outputs.shape[0]:
        raise ValueError("vla75_validation_example_count")
    if not validation_lineage_sha256:
        raise ValueError("vla75_validation_lineage_required")
    report = world_vla75_per_sample_loss_report(
        outputs,
        targets,
        preference_weight=config.preference_weight,
        executable_weight=config.executable_weight,
    )
    heads: dict[str, dict[str, float | int]] = {}
    for name, values in report.head_losses.items():
        mask = report.head_masks[name]
        count = int(mask.sum().detach().cpu())
        loss = (
            float(values[mask].mean().detach().cpu())
            if count
            else float("nan")
        )
        heads[name] = {"loss": loss, "count": count}

    pair_mask = report.head_masks["pair_preference"]
    vla_idx, expert_idx, _ = _pair_indices(targets)
    preference = outputs[:, :, 12]
    predicted_margin = (
        preference.gather(1, vla_idx[:, None]).squeeze(1)
        - preference.gather(1, expert_idx[:, None]).squeeze(1)
    )
    objective = targets["objective"]
    actual_margin = (
        objective.gather(1, vla_idx[:, None]).squeeze(1)
        - objective.gather(1, expert_idx[:, None]).squeeze(1)
    )
    pair_count = int(pair_mask.sum().detach().cpu())
    pair_accuracy = (
        float(
            ((predicted_margin[pair_mask] >= 0.0) == (actual_margin[pair_mask] >= 0.0))
            .to(outputs)
            .mean()
            .detach()
            .cpu()
        )
        if pair_count
        else float("nan")
    )
    # Regret is the objective lost by following the predicted pair ordering.
    pair_regret = (
        float(
            torch.where(
                predicted_margin[pair_mask] >= 0.0,
                torch.relu(-actual_margin[pair_mask]),
                torch.relu(actual_margin[pair_mask]),
            )
            .mean()
            .detach()
            .cpu()
        )
        if pair_count
        else float("nan")
    )

    labels = [_outcome_group(example) for example in examples]
    group_rows: dict[str, dict[str, float | int | str]] = {}
    for label in sorted(set(labels)):
        mask = torch.as_tensor(
            [item == label for item in labels], dtype=torch.bool, device=outputs.device
        ) & report.valid_samples
        count = int(mask.sum().detach().cpu())
        group_rows[label] = {
            "status": "MEASURED" if count else "NOT_MEASURED",
            "loss": (
                float(report.per_sample[mask].mean().detach().cpu())
                if count
                else float("nan")
            ),
            "count": count,
            "weight": (
                float(group_weights[label])
                if group_weights is not None and label in group_weights
                else 0.0
            ),
        }
    measured_groups = [item for item in group_rows.values() if item["count"]]
    worst_group = max(measured_groups, key=lambda item: float(item["loss"])) if measured_groups else None

    aligned_swapped = swapped_outputs.flip(1)
    swap_delta = (outputs - aligned_swapped).abs()
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
        "schema_version": "safedrive.world.vla75.selection_metrics.v1",
        "evaluator_lineage_sha256": validation_lineage_sha256,
        "validation_loss": float(validation_loss),
        "valid_samples": int(report.valid_samples.sum().detach().cpu()),
        "heads": heads,
        "pair_accuracy": pair_accuracy,
        "pair_regret": pair_regret,
        "pair_count": pair_count,
        "groups": group_rows,
        "worst_group_loss": (
            float(worst_group["loss"]) if worst_group is not None else float("nan")
        ),
        "worst_group_count": (
            int(worst_group["count"]) if worst_group is not None else 0
        ),
        "candidate_swap_error": float(swap_delta.max().detach().cpu()),
        "candidate_swap_count": int(swap_delta.numel()),
        # Diagnostic only.  Deliberately absent from the selection key.
        "source_usage": {
            "vla_rows": int(targets.get("source_is_vla", torch.zeros(())).sum().detach().cpu())
        },
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
    train_lineage_sha256 = outcome_examples_lineage_sha256(train_examples)
    validation_lineage_sha256 = outcome_examples_lineage_sha256(val_examples)
    dro_state = GroupDROState(
        [_outcome_group(example) for example in train_examples],
        eta=cfg.group_dro_eta,
        floor=cfg.group_dro_floor,
    )
    order = list(range(len(train_examples)))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_selection_key = None
    best_selection_metrics: dict[str, Any] = {}
    best_group_weights: dict[str, float] = {}
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
            batch_groups = [_outcome_group(pair) for pair in batch]
            with torch.no_grad():
                supervision = world_vla75_per_sample_loss_report(
                    outputs,
                    targets,
                    preference_weight=cfg.preference_weight,
                    executable_weight=cfg.executable_weight,
                )
                dro_batch = dro_state.evaluate_batch(
                    supervision.per_sample.detach(),
                    supervision.valid_samples,
                    batch_groups,
                    update=True,
                )
                group_weights = dro_batch.sample_weights
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
            swapped_outputs = torch.stack(
                [model(context[:, index], candidate[:, index]) for index in (1, 0)], dim=1
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
            swapped_outputs,
            targets,
            val_examples,
            validation_loss=scalar,
            validation_lineage_sha256=validation_lineage_sha256,
            config=cfg,
            group_weights=dro_state.weights,
        )
        selection_key = vla75_checkpoint_selection_key(selection_metrics)
        if best_selection_key is None or selection_key > best_selection_key:
            best_loss = scalar
            best_epoch = epoch
            best_selection_key = selection_key
            best_selection_metrics = dict(selection_metrics)
            best_group_weights = dro_state.weights
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
    model.load_state_dict(best_state)
    best_group_metrics = _evaluate_vla75_group_supervision(
        model,
        train_examples,
        torch_device,
        cfg,
        best_group_weights,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": WORLD_VLA75_SCHEMA_VERSION,
        "seed": seed,
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "train_lineage_sha256": train_lineage_sha256,
        "validation_lineage_sha256": validation_lineage_sha256,
        "selection_metrics": best_selection_metrics,
        "selection_metrics_sha256": stable_sha256(best_selection_metrics),
        "selection_key": list(best_selection_key or ()),
        "group_dro_weights": best_group_weights,
        "group_dro": best_group_metrics,
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


def vla75_checkpoint_selection_key(metrics: Mapping[str, Any]) -> tuple:
    """Fail-closed, source-neutral lexicographic checkpoint ordering."""

    if metrics.get("schema_version") != "safedrive.world.vla75.selection_metrics.v1":
        raise ValueError("vla75_selection_metrics_schema")
    lineage = metrics.get("evaluator_lineage_sha256")
    if not isinstance(lineage, str) or len(lineage) != 64:
        raise ValueError("vla75_selection_evaluator_lineage")

    def finite(name: str) -> float:
        if name not in metrics:
            raise ValueError(f"vla75_selection_metric_missing:{name}")
        try:
            value = float(metrics[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"vla75_selection_metric_invalid:{name}") from error
        if not math.isfinite(value):
            raise ValueError(f"vla75_selection_metric_nonfinite:{name}")
        return value

    def positive_count(name: str) -> int:
        if name not in metrics:
            raise ValueError(f"vla75_selection_metric_missing:{name}")
        try:
            value = int(metrics[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"vla75_selection_count_invalid:{name}") from error
        if value <= 0:
            raise ValueError(f"vla75_selection_count_zero:{name}")
        return value

    positive_count("valid_samples")
    positive_count("pair_count")
    positive_count("worst_group_count")
    positive_count("candidate_swap_count")
    heads = metrics.get("heads")
    required_heads = tuple(WORLD_V3_HEAD_WEIGHTS) + tuple(
        WORLD_VLA75_EXTRA_HEAD_WEIGHTS
    )
    if not isinstance(heads, Mapping):
        raise ValueError("vla75_selection_heads_missing")
    for name in required_heads:
        item = heads.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"vla75_selection_head_missing:{name}")
        try:
            count = int(item.get("count", 0))
            loss = float(item["loss"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"vla75_selection_head_invalid:{name}") from error
        if count <= 0:
            raise ValueError(f"vla75_selection_head_count_zero:{name}")
        if not math.isfinite(loss):
            raise ValueError(f"vla75_selection_head_nonfinite:{name}")
    groups = metrics.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("vla75_selection_groups_missing")
    for name, item in groups.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"vla75_selection_group_invalid:{name}")
        count = int(item.get("count", 0))
        if count <= 0 or item.get("status") != "MEASURED":
            raise ValueError(f"vla75_selection_group_not_measured:{name}")
        loss = float(item.get("loss", float("nan")))
        if not math.isfinite(loss):
            raise ValueError(f"vla75_selection_group_nonfinite:{name}")
    # Source usage and VLA quota diagnostics are intentionally excluded.
    return (
        -finite("candidate_swap_error"),
        finite("pair_accuracy"),
        -finite("pair_regret"),
        -finite("worst_group_loss"),
        -finite("validation_loss"),
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
    "world_v3_per_sample_loss_report",
    "PerSampleLossReport",
    "WORLD_V3_HEAD_WEIGHTS",
    "WorldVLA75Model",
    "WorldVLA75TrainConfig",
    "WorldVLA75TrainResult",
    "load_world_vla75",
    "train_world_vla75",
    "world_vla75_loss",
    "world_vla75_per_sample_loss_report",
    "WORLD_VLA75_EXTRA_HEAD_WEIGHTS",
    "group_dro_weights",
    "GroupDROState",
    "GroupDROBatchReport",
    "event_aware_preference_consistency_loss",
    "temporal_preference_consistency_from_outputs",
    "select_vla75_checkpoint",
    "vla75_checkpoint_selection_key",
]
