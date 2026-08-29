"""Measured, self-hashed C1 evaluator artifacts for the VLA75 World model."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from data_pipeline.h3.contracts import stable_sha256
from data_pipeline.h6.dataset import (
    OutcomePairExample,
    outcome_examples_lineage_sha256,
)
from data_pipeline.h6.model import (
    WORLD_V3_HEAD_WEIGHTS,
    WORLD_VLA75_EXTRA_HEAD_WEIGHTS,
    WORLD_VLA75_OUTPUT_DIM,
    WorldVLA75TrainConfig,
    _batch,
    _outcome_group,
    _pair_indices,
    _vla75_validation_metrics,
    load_world_vla75,
    temporal_preference_consistency_from_outputs,
    world_vla75_loss,
    world_vla75_per_sample_loss_report,
)

EVALUATOR_SCHEMA = "safedrive.world.vla75.evaluator.v1"
SUMMARY_SCHEMA = "safedrive.world.vla75.training_summary.v2"
MEASURED = "MEASURED"
NOT_MEASURED = "NOT_MEASURED"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("evaluator_latency_samples_required")
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _measurement(value: float | None, count: int) -> dict[str, Any]:
    if value is None:
        return {"status": NOT_MEASURED, "value": None, "count": int(count)}
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("evaluator_measurement_must_be_finite")
    return {"status": MEASURED, "value": numeric, "count": int(count)}


def _forward_pair(model, context: Tensor, candidate: Tensor) -> Tensor:
    return torch.stack(
        [model(context[:, index], candidate[:, index]) for index in range(2)], dim=1
    )


def _metadata_swap_examples(
    examples: Sequence[OutcomePairExample],
) -> list[OutcomePairExample]:
    swapped: list[OutcomePairExample] = []
    for example in examples:
        first, second = example.candidates
        swapped.append(
            replace(
                example,
                candidates=(
                    replace(first, source=second.source),
                    replace(second, source=first.source),
                ),
            )
        )
    return swapped


def _close_selection_metrics(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    """Compare evaluator metrics while tolerating device-level float noise."""

    if set(expected) != set(observed):
        return False
    for key in expected:
        left = expected[key]
        right = observed[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if not _close_selection_metrics(left, right):
                return False
        elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
            if not math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-6):
                return False
        elif left != right:
            return False
    return True


def build_vla75_evaluator(
    checkpoint_path: Path,
    validation_examples: Sequence[OutcomePairExample],
    *,
    device: str | torch.device,
    training_input_sha256: str,
    config_sha256: str,
    code_sha256: str,
    worktree_sha256: str,
    latency_iterations: int = 32,
) -> dict[str, Any]:
    """Run all C1 evaluator probes and return a self-hashed artifact."""

    if not validation_examples:
        raise ValueError("evaluator_validation_rows_required")
    if latency_iterations <= 0:
        raise ValueError("evaluator_latency_iterations_positive")
    torch_device = torch.device(device)
    model, metadata = load_world_vla75(checkpoint_path, device=torch_device)
    checkpoint_hash = file_sha256(checkpoint_path)
    validation_lineage = outcome_examples_lineage_sha256(validation_examples)
    if metadata.get("validation_lineage_sha256") != validation_lineage:
        raise ValueError("evaluator_checkpoint_validation_lineage_mismatch")
    if metadata.get("train_lineage_sha256") != training_input_sha256:
        raise ValueError("evaluator_checkpoint_training_lineage_mismatch")
    selection = metadata.get("selection_metrics")
    if not isinstance(selection, Mapping):
        raise ValueError("evaluator_checkpoint_selection_metrics_missing")
    if metadata.get("selection_metrics_sha256") != stable_sha256(dict(selection)):
        raise ValueError("evaluator_checkpoint_selection_metrics_hash")

    context, candidate, targets = _batch(validation_examples, torch_device, swap=False)
    cfg = WorldVLA75TrainConfig(**dict(metadata.get("config") or {}))
    model.eval()
    with torch.no_grad():
        outputs = _forward_pair(model, context, candidate)
        swapped_outputs = torch.stack(
            [model(context[:, index], candidate[:, index]) for index in (1, 0)], dim=1
        )
        temporal = temporal_preference_consistency_from_outputs(
            outputs, validation_examples, targets
        )
        validation_loss, _pieces = world_vla75_loss(
            outputs,
            targets,
            raw_preference_target=cfg.raw_preference_target,
            actual_coverage_target=cfg.actual_coverage_target,
            preference_weight=cfg.preference_weight,
            executable_weight=cfg.executable_weight,
            raw_coverage_weight=cfg.raw_coverage_weight,
            actual_coverage_weight=cfg.actual_coverage_weight,
            consistency_weight=cfg.consistency_weight,
            temporal_loss=temporal,
        )
        actual_selection = _vla75_validation_metrics(
            outputs,
            swapped_outputs,
            targets,
            validation_examples,
            validation_loss=float(validation_loss.detach().cpu()),
            validation_lineage_sha256=validation_lineage,
            config=cfg,
            group_weights=dict(metadata.get("group_dro_weights") or {}),
        )
        report = world_vla75_per_sample_loss_report(
            outputs,
            targets,
            preference_weight=cfg.preference_weight,
            executable_weight=cfg.executable_weight,
        )
        aligned_swap = swapped_outputs.flip(1)
        candidate_swap_error = float((outputs - aligned_swap).abs().max().cpu())

        metadata_context, metadata_candidate, _ = _batch(
            _metadata_swap_examples(validation_examples), torch_device, swap=False
        )
        metadata_outputs = _forward_pair(model, metadata_context, metadata_candidate)
        source_swap_error = float((outputs - metadata_outputs).abs().max().cpu())
        action_mask_sensitivity = float(
            (outputs - _forward_pair(model, context, torch.zeros_like(candidate)))
            .abs()
            .mean()
            .cpu()
        )
        context_mask_sensitivity = float(
            (outputs - _forward_pair(model, torch.zeros_like(context), candidate))
            .abs()
            .mean()
            .cpu()
        )

    if not _close_selection_metrics(dict(selection), actual_selection):
        raise ValueError("evaluator_checkpoint_selection_metrics_mismatch")

    head_metrics: dict[str, dict[str, Any]] = {}
    hazard_targets = {
        "collision": targets["collision"],
        "red_light": targets["red"],
        "offroad": targets["offroad"],
    }
    for name, values in report.head_losses.items():
        mask = report.head_masks[name]
        count = int(mask.sum().cpu())
        item: dict[str, Any] = {
            "status": MEASURED if count else NOT_MEASURED,
            "loss": float(values[mask].mean().cpu()) if count else None,
            "count": count,
            "weight": float(report.head_weights[name]),
        }
        if name in hazard_targets:
            raw_mask = (
                targets.get("safety_mask", torch.ones_like(hazard_targets[name])) > 0.0
            )
            item["positive_count"] = int(
                ((hazard_targets[name] > 0.5) & raw_mask).sum().cpu()
            )
        head_metrics[name] = item

    pair_mask = report.head_masks["pair_preference"]
    vla_idx, expert_idx, _ = _pair_indices(targets)
    predicted_margin = (
        outputs[:, :, 12].gather(1, vla_idx[:, None]).squeeze(1)
        - outputs[:, :, 12].gather(1, expert_idx[:, None]).squeeze(1)
    )
    actual_margin = (
        targets["objective"].gather(1, vla_idx[:, None]).squeeze(1)
        - targets["objective"].gather(1, expert_idx[:, None]).squeeze(1)
    )
    pair_count = int(pair_mask.sum().cpu())
    pair_accuracy = (
        float(
            ((predicted_margin[pair_mask] >= 0.0) == (actual_margin[pair_mask] >= 0.0))
            .float()
            .mean()
            .cpu()
        )
        if pair_count
        else None
    )
    pair_regret = (
        float(
            torch.where(
                predicted_margin[pair_mask] >= 0.0,
                torch.relu(-actual_margin[pair_mask]),
                torch.relu(actual_margin[pair_mask]),
            )
            .mean()
            .cpu()
        )
        if pair_count
        else None
    )

    labels = [_outcome_group(example) for example in validation_examples]
    checkpoint_group_weights = dict(metadata.get("group_dro_weights") or {})
    validation_groups = sorted(set(labels) | set(checkpoint_group_weights))
    fallback_weight = 1.0 / len(validation_groups)
    groups: dict[str, dict[str, Any]] = {}
    for label in validation_groups:
        mask = torch.as_tensor(
            [item == label for item in labels], dtype=torch.bool, device=torch_device
        ) & report.valid_samples
        count = int(mask.sum().cpu())
        groups[label] = {
            "status": MEASURED if count else NOT_MEASURED,
            "loss": float(report.per_sample[mask].mean().cpu()) if count else None,
            "count": count,
            "weight": float(checkpoint_group_weights.get(label, fallback_weight)),
        }

    # Measure the loaded model rather than copying a training placeholder.
    for _ in range(min(3, latency_iterations)):
        with torch.no_grad():
            _forward_pair(model, context[:1], candidate[:1])
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        baseline_gpu_bytes = int(torch.cuda.memory_allocated(torch_device))
        torch.cuda.reset_peak_memory_stats(torch_device)
    else:
        baseline_gpu_bytes = 0
    latencies: list[float] = []
    for _ in range(latency_iterations):
        started = time.perf_counter()
        with torch.no_grad():
            _forward_pair(model, context[:1], candidate[:1])
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    latency = {
        "status": MEASURED,
        "device": str(torch_device),
        "iterations": latency_iterations,
        "p50_ms": statistics.median(latencies),
        "p95_ms": _percentile(latencies, 0.95),
        "p99_ms": _percentile(latencies, 0.99),
        "max_ms": max(latencies),
    }
    if torch_device.type == "cuda":
        incremental = max(
            0, int(torch.cuda.max_memory_allocated(torch_device)) - baseline_gpu_bytes
        )
        gpu_peak = {
            "status": MEASURED,
            "device": str(torch_device),
            "incremental_peak_gib": incremental / (1024.0**3),
        }
    else:
        gpu_peak = {
            "status": NOT_MEASURED,
            "device": str(torch_device),
            "incremental_peak_gib": None,
        }

    artifact: dict[str, Any] = {
        "schema_version": EVALUATOR_SCHEMA,
        "evidence_state": MEASURED,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "seed": int(metadata.get("seed", -1)),
            "selection_metrics_sha256": metadata["selection_metrics_sha256"],
        },
        "inputs": {
            "training_lineage_sha256": str(training_input_sha256),
            "validation_lineage_sha256": validation_lineage,
            "config_sha256": str(config_sha256),
            "code_sha256": str(code_sha256),
            "worktree_sha256": str(worktree_sha256),
            "input_sha256": stable_sha256(
                {
                    "training": training_input_sha256,
                    "validation": validation_lineage,
                    "config": config_sha256,
                    "code": code_sha256,
                    "worktree": worktree_sha256,
                }
            ),
        },
        "validation": {
            "loss": _measurement(float(validation_loss.detach().cpu()), int(report.valid_samples.sum().cpu())),
            "heads": head_metrics,
            "pair": {
                "status": MEASURED if pair_count else NOT_MEASURED,
                "accuracy": pair_accuracy,
                "regret": pair_regret,
                "count": pair_count,
            },
            "groups": groups,
        },
        "probes": {
            "candidate_swap": _measurement(candidate_swap_error, int(outputs.numel())),
            "source_metadata_swap": _measurement(source_swap_error, int(outputs.numel())),
            "action_mask": _measurement(action_mask_sensitivity, len(validation_examples)),
            "context_history_mask": _measurement(context_mask_sensitivity, len(validation_examples)),
        },
        "latency": latency,
        "gpu_peak": gpu_peak,
        "selection_metrics": actual_selection,
        "artifact_verification": {
            "checkpoint_metadata": "VERIFIED",
            "selection_metrics": "VERIFIED",
        },
    }
    artifact["evaluator_sha256"] = stable_sha256(artifact)
    return artifact


def verify_vla75_evaluator(
    artifact: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    payload = {key: value for key, value in artifact.items() if key != "evaluator_sha256"}
    if artifact.get("schema_version") != EVALUATOR_SCHEMA:
        failures.append("schema")
    if artifact.get("evidence_state") != MEASURED:
        failures.append("evidence_state")
    if artifact.get("evaluator_sha256") != stable_sha256(payload):
        failures.append("self_hash")
    validation = artifact.get("validation")
    if not isinstance(validation, Mapping):
        failures.append("validation_missing")
    else:
        heads = validation.get("heads")
        required_heads = set(WORLD_V3_HEAD_WEIGHTS) | set(
            WORLD_VLA75_EXTRA_HEAD_WEIGHTS
        )
        if not isinstance(heads, Mapping):
            failures.append("heads_missing")
        else:
            for name in sorted(required_heads - set(heads)):
                failures.append(f"head_missing:{name}")
        for name in ("loss", "pair", "groups"):
            if not isinstance(validation.get(name), Mapping):
                failures.append(f"validation_field_missing:{name}")
    probes = artifact.get("probes")
    required_probes = {
        "candidate_swap",
        "source_metadata_swap",
        "action_mask",
        "context_history_mask",
    }
    if not isinstance(probes, Mapping):
        failures.append("probes_missing")
    else:
        for name in sorted(required_probes - set(probes)):
            failures.append(f"probe_missing:{name}")
    if not isinstance(artifact.get("latency"), Mapping):
        failures.append("latency_missing")
    if not isinstance(artifact.get("gpu_peak"), Mapping):
        failures.append("gpu_peak_missing")
    checkpoint = artifact.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        failures.append("checkpoint_missing")
    else:
        if not str(checkpoint.get("path", "")):
            failures.append("checkpoint_path")
        if not isinstance(checkpoint.get("sha256"), str) or len(
            str(checkpoint.get("sha256", ""))
        ) != 64:
            failures.append("checkpoint_digest")
        try:
            if int(checkpoint.get("seed", -1)) < 0:
                failures.append("checkpoint_seed")
        except (TypeError, ValueError):
            failures.append("checkpoint_seed")
    inputs = artifact.get("inputs")
    if not isinstance(inputs, Mapping):
        failures.append("inputs_missing")
    else:
        for name in (
            "training_lineage_sha256",
            "validation_lineage_sha256",
            "config_sha256",
            "code_sha256",
            "worktree_sha256",
            "input_sha256",
        ):
            value = inputs.get(name)
            if not isinstance(value, str) or len(value) != 64:
                failures.append(f"input_hash:{name}")
        expected_input = stable_sha256(
            {
                "training": inputs.get("training_lineage_sha256"),
                "validation": inputs.get("validation_lineage_sha256"),
                "config": inputs.get("config_sha256"),
                "code": inputs.get("code_sha256"),
                "worktree": inputs.get("worktree_sha256"),
            }
        )
        if inputs.get("input_sha256") != expected_input:
            failures.append("input_self_hash")
    if isinstance(checkpoint, Mapping) and root is not None:
        raw = Path(str(checkpoint.get("path", "")))
        path = raw if raw.is_absolute() else root / raw
        if not path.is_file():
            failures.append("checkpoint_file_missing")
        elif file_sha256(path) != checkpoint.get("sha256"):
            failures.append("checkpoint_hash")
        else:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            metadata = dict(loaded.get("metadata") or {})
            inputs = dict(artifact.get("inputs") or {})
            if metadata.get("validation_lineage_sha256") != inputs.get(
                "validation_lineage_sha256"
            ):
                failures.append("validation_lineage")
            if metadata.get("train_lineage_sha256") != inputs.get(
                "training_lineage_sha256"
            ):
                failures.append("training_lineage")
            selection = metadata.get("selection_metrics")
            if not isinstance(selection, Mapping) or stable_sha256(dict(selection)) != metadata.get(
                "selection_metrics_sha256"
            ):
                failures.append("checkpoint_selection_hash")
            if metadata.get("selection_metrics_sha256") != checkpoint.get(
                "selection_metrics_sha256"
            ):
                failures.append("evaluator_selection_binding")
            artifact_selection = artifact.get("selection_metrics")
            if not isinstance(artifact_selection, Mapping) or stable_sha256(
                dict(artifact_selection)
            ) != metadata.get("selection_metrics_sha256"):
                failures.append("checkpoint_selection_metrics")
            try:
                artifact_seed = int(checkpoint.get("seed", -1))
                metadata_seed = int(metadata.get("seed", -2))
            except (TypeError, ValueError, OverflowError):
                failures.append("checkpoint_seed_binding")
            else:
                if artifact_seed != metadata_seed:
                    failures.append("checkpoint_seed_binding")
    return {"valid": not failures, "failures": sorted(set(failures))}


def finalize_training_summary_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the C1 summary status contract and self hash."""

    summary = dict(payload)
    summary["schema_version"] = SUMMARY_SCHEMA
    summary["evidence_state"] = MEASURED
    summary["artifact_verification"] = "VERIFIED"
    summary["cora_algorithm_state"] = "NOT_VERIFIED"
    summary.pop("evidence_sha256", None)
    summary.pop("summary_sha256", None)
    summary["summary_sha256"] = stable_sha256(summary)
    return summary


__all__ = [
    "EVALUATOR_SCHEMA",
    "SUMMARY_SCHEMA",
    "MEASURED",
    "NOT_MEASURED",
    "build_vla75_evaluator",
    "verify_vla75_evaluator",
    "finalize_training_summary_v2",
    "file_sha256",
]
