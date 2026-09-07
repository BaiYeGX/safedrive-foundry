"""Existing-data C2 development release and a small World baseline.

This module deliberately consumes the recorded base roots and v3 correction
sidecars.  It does not start CARLA, copy raw rollout artifacts, or re-hash old
files.  New derived JSON/checkpoints are the release being measured here.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from data_pipeline.h3.contracts import H3_CANDIDATE_DIM, H3_CANDIDATE_STEPS, H3_CONTEXT_DIM
from data_pipeline.h3.dataset import H3DatasetError, _candidate_tensor, _context_vector
from .outcomes import PUBLIC_OUTCOME_HEADS
from .loader import load_cora_roots


SCHEMA = "safedrive.cora.dev_baseline.v1"
RELEASE_SCHEMA = "safedrive.cora.dev_release.v1"
TARGETS = (
    "route_progress_m",
    "acceleration_rms_mps2",
    "jerk_rms_mps3",
    "lateral_acceleration_rms_mps2",
)
ALL_SPLITS = ("coverage_pilot", "train", "validation", "calibration", "locked_development")
FEATURE_DIM = H3_CONTEXT_DIM + H3_CANDIDATE_STEPS * H3_CANDIDATE_DIM


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if target.suffix == ".toml":
        return tomllib.loads(target.read_text(encoding="utf-8"))
    return _json(target)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _finite_vector(value: Sequence[float], name: str) -> list[float]:
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise H3DatasetError(f"dev_baseline_non_finite:{name}")
    return result


def _physical_signature(anchor: Mapping[str, Any], scenario: Mapping[str, Any], physical: Mapping[str, Any] | None = None) -> str:
    """Normalize physical capture fields while excluding IDs, seeds and times."""
    snapshot = dict(anchor.get("observable_snapshot", {}))
    actors = []
    for actor in snapshot.get("actors", ()):
        actors.append({key: actor.get(key) for key in (
            "class_name", "x", "y", "yaw", "vx", "vy", "length_m", "width_m", "lost",
        )})
    lights = []
    for light in snapshot.get("traffic_lights", ()):
        lights.append({key: light.get(key) for key in (
            "state", "distance_m", "stop_line_distance_m", "controls_ego_lane",
        )})
    if physical is not None:
        payload = {
            "route": physical.get("route", anchor.get("route", ())),
            "ego_transform": physical.get("ego_transform", {}),
            "npc_actors": physical.get("npc_actors", ()),
            "weather": physical.get("weather", scenario.get("weather")),
            "script": physical.get("script", {}),
            "red_light": physical.get("red_light"),
            "observed_actors": actors,
            "observed_lights": lights,
        }
    else:
        payload = {
            "route": anchor.get("route", ()),
            "ego": {key: snapshot.get(key) for key in ("ego_x", "ego_y", "ego_yaw", "ego_v", "ego_a")},
            "actors": actors,
            "lights": lights,
            "weather": scenario.get("weather"),
        }
    return _canonical(payload)


def _nominal_branch(record: Mapping[str, Any], source: str) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    proposal = next((item for item in record.get("proposals", ())
                     if item.get("kind") == "nominal" and item.get("audit_source") == source), None)
    if proposal is None:
        return None
    branch = next((item for item in record.get("branches", ())
                   if item.get("proposal_id") == proposal.get("proposal_id")), None)
    if branch is None:
        return None
    return proposal, branch


def _label_path(repair_root: Path, root_id: str, proposal_sha: str) -> Path:
    return repair_root / "corrected-labels" / f"{root_id}__{proposal_sha}.v3.json"


def _audit_branch(branch: Mapping[str, Any], heads: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if not branch.get("outcome_valid"):
        failures.append("outcome_invalid")
    if not branch.get("identity_valid"):
        failures.append("identity_invalid")
    if not branch.get("cleanup_complete"):
        failures.append("cleanup_incomplete")
    reset = branch.get("reset")
    if not isinstance(reset, Mapping) or reset.get("comparable") is not True:
        failures.append("reset_not_comparable")
    if branch.get("auxiliary_only"):
        failures.append("auxiliary_only")
    for name in TARGETS:
        item = heads.get(name)
        if not isinstance(item, Mapping):
            failures.append(f"head_missing:{name}")
        elif item.get("unit") in (None, ""):
            failures.append(f"head_unit_missing:{name}")
        elif item.get("valid") and not isinstance(item.get("value"), (int, float)):
            failures.append(f"head_value_invalid:{name}")
    return failures


def _feature_pair(base_root: Path, record: Mapping[str, Any], proposals: Mapping[str, Mapping[str, Any]]) -> tuple[list[float], dict[str, list[list[float]]]]:
    anchor = _json(base_root / record["anchor_path"])
    required_snapshot = anchor.get("observable_snapshot", {})
    required = ("ego_x", "ego_y", "ego_yaw", "ego_v", "ego_a", "actors", "traffic_lights", "corridor_half_width_m")
    missing = [key for key in required if key not in required_snapshot]
    if missing or len(anchor.get("route", ())) < 2:
        raise H3DatasetError(f"dev_baseline_anchor_missing:{record['root_id']}:{missing}")
    feature_record = {"anchor": anchor, "observable_history": anchor.get("observable_history", ()), "route": anchor["route"]}
    context = _finite_vector(_context_vector(feature_record), "context")
    tensors: dict[str, list[list[float]]] = {}
    for source, proposal in proposals.items():
        tensor = _candidate_tensor(feature_record, proposal)
        tensor = [_finite_vector(row, f"candidate:{source}") for row in tensor]
        if len(tensor) != H3_CANDIDATE_STEPS or any(len(row) != H3_CANDIDATE_DIM for row in tensor):
            raise H3DatasetError(f"dev_baseline_candidate_shape:{record['root_id']}:{source}")
        tensors[source] = tensor
    return context, tensors


def _sample(record: Mapping[str, Any], base_root: Path, repair_root: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    selected: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    failures: list[str] = []
    for source in ("expert", "vla"):
        item = _nominal_branch(record, source)
        if item is None:
            failures.append(f"nominal_missing:{source}")
        else:
            selected[source] = item
    if failures:
        return None, [{"root_id": record.get("root_id"), "reason": item} for item in failures]
    proposals = {source: pair[0] for source, pair in selected.items()}
    context, tensors = _feature_pair(base_root, record, proposals)
    candidate_rows = []
    for source in ("expert", "vla"):
        proposal, branch = selected[source]
        sidecar = _label_path(repair_root, str(record["root_id"]), str(branch["proposal_sha256"]))
        if not sidecar.is_file():
            return None, [{"root_id": record["root_id"], "reason": f"label_missing:{source}"}]
        label = _json(sidecar)
        heads = label.get("heads", {})
        branch_failures = _audit_branch(branch, heads)
        core_failures = {
            "outcome_invalid", "identity_invalid", "cleanup_incomplete",
            "reset_not_comparable", "auxiliary_only",
        }
        if core_failures.intersection(branch_failures):
            return None, [{"root_id": record["root_id"], "reason": reason} for reason in branch_failures]
        targets: dict[str, Any] = {}
        for name in TARGETS:
            head = heads.get(name, {})
            targets[name] = {"value": head.get("value"), "mask": bool(head.get("valid", False)),
                             "unit": head.get("unit"), "derivation_version": head.get("derivation_version")}
        candidate_rows.append({
            "source": source,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_sha256": proposal.get("proposal_sha256"),
            "branch_id": branch.get("proposal_id"),
            "feature": {"context": context, "candidate": tensors[source]},
            "targets": targets,
            "audit": {
                "outcome_valid": bool(branch.get("outcome_valid")),
                "identity_valid": bool(branch.get("identity_valid")),
                "cleanup_complete": bool(branch.get("cleanup_complete")),
                "reset_comparable": branch.get("reset", {}).get("comparable") is True if isinstance(branch.get("reset"), Mapping) else False,
                "guard_verdict": branch.get("guard_verdict"),
                "errors": list(branch.get("errors", ())),
                "repair_success_valid": bool(heads.get("repair_success", {}).get("valid", False)),
            },
            "failures": branch_failures,
        })
    scenario = dict(record["scenario"])
    return {
        "pair_id": record["root_id"],
        "split": record["split"],
        "map": scenario.get("map_name"),
        "family": scenario.get("family"),
        "seed": scenario.get("seed"),
        "weather": scenario.get("weather"),
        "candidates": candidate_rows,
        "nominal_pair": True,
    }, []


def prepare_release(config_path: Path | str, project: Path) -> dict[str, Any]:
    config = _config(config_path)
    base = project / "generated/h6/cora" / str(config["base_dataset_id"])
    repair = project / "generated/h6/cora" / str(config["repair_dataset_id"])
    out = project / "generated/h6/cora" / str(config["dataset_id"])
    out.mkdir(parents=True, exist_ok=True)
    records = []
    exclusions = []
    signatures: dict[str, list[str]] = {}
    head_counts: dict[str, dict[str, int]] = {
        split: {name: 0 for name in PUBLIC_OUTCOME_HEADS} for split in ALL_SPLITS
    }
    manifest = _json(base / "scenario-manifest.json")
    physical_rows = {str(item["pair_id"]): item for item in manifest.get("rows", ())}
    for path in sorted((base / "pairs").glob("*.json")):
        record = _json(path)
        sample, errors = _sample(record, base, repair)
        if sample is None:
            exclusions.extend(errors)
            continue
        anchor = _json(base / record["anchor_path"])
        signature = _physical_signature(anchor, record["scenario"], physical_rows.get(str(record["root_id"])))
        sample["physical_signature"] = signature
        signatures.setdefault(signature, []).append(sample["pair_id"])
        for candidate in sample["candidates"]:
            # The release retains the full 29-head support audit even though
            # the baseline trains only four continuous heads.
            source_branch = next(
                item for item in record["branches"] if item.get("proposal_id") == candidate["branch_id"]
            )
            sidecar = _json(_label_path(repair, str(record["root_id"]), str(source_branch["proposal_sha256"])))
            for name in PUBLIC_OUTCOME_HEADS:
                if bool(sidecar.get("heads", {}).get(name, {}).get("valid", False)):
                    head_counts[str(record["split"])][name] += 1
        records.append(sample)
    by_signature = {}
    for sample in records:
        by_signature.setdefault(sample["physical_signature"], []).append(sample)
    retained: set[str] = set()
    for values in by_signature.values():
        values = sorted(values, key=lambda item: item["pair_id"])
        splits = {item["split"] for item in values}
        if len(splits) > 1:
            for item in values:
                exclusions.append({"root_id": item["pair_id"], "reason": "physical_duplicate_cross_split"})
            continue
        retained.add(values[0]["pair_id"])
        for item in values[1:]:
            exclusions.append({"root_id": item["pair_id"], "reason": "physical_duplicate_same_split"})
    records = [sample for sample in records if sample["pair_id"] in retained]
    split_counts = {split: sum(sample["split"] == split for sample in records) for split in ALL_SPLITS}
    index = {
        "schema_version": RELEASE_SCHEMA,
        "quality_profile": config["quality_profile"],
        "dataset_id": config["dataset_id"],
        "base_dataset_id": config["base_dataset_id"],
        "repair_dataset_id": config["repair_dataset_id"],
        "status": "DEV_DATA_READY" if split_counts.get("train", 0) and split_counts.get("validation", 0) else "DEV_DATA_FAILED",
        "raw_root_count": sum(1 for _ in (base / "pairs").glob("*.json")),
        "usable_root_count": len(records),
        "split_counts": split_counts,
        "training_splits": list(config["training_splits"]),
        "evaluation_splits": list(config["evaluation_splits"]),
        "target_heads": list(config["target_heads"]),
        "public_head_valid_candidate_counts": head_counts,
        "exclusion_count": len(exclusions),
        "exclusions": exclusions,
        "old_identity_policy": "REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN",
        "raw_data_references": {"base": str(base), "repair_sidecars": str(repair / "corrected-labels")},
        "samples": records,
    }
    _write(out / "release-index.json", index)
    _write(out / "data-card.json", {
        "schema_version": "safedrive.cora.dev_data_card.v1",
        "dataset_id": config["dataset_id"], "status": index["status"],
        "purpose": "candidate-conditioned short-horizon progress and comfort development",
        "root_counts": split_counts, "training_root_count": split_counts.get("train", 0),
        "validation_root_count": split_counts.get("validation", 0),
        "audit_only_splits": ["coverage_pilot", "calibration", "locked_development"],
        "learned_heads": list(config["target_heads"]),
        "unlearned_reported_heads": ["collision", "red_light_violation", "off_corridor_duration_s", "executable", "repair_success"],
        "vla_status": "FROZEN",
        "closed_loop_status": "NOT_MEASURED",
    })
    _write(out / "audit-report.json", {
        "schema_version": "safedrive.cora.dev_audit.v1",
        "dataset_id": config["dataset_id"],
        "status": index["status"],
        "raw_root_count": index["raw_root_count"],
        "usable_root_count": index["usable_root_count"],
        "split_counts": split_counts,
        "public_head_valid_candidate_counts": head_counts,
        "target_heads": list(config["target_heads"]),
        "exclusion_count": len(exclusions),
        "exclusions": exclusions,
        "duplicate_policy": "same_split_keep_first_sorted_root; cross_split_isolate_entire_cluster",
        "intervention_policy": "nominal_expert_vla_only; diagnostics_and_interventions_audit_only",
        "old_identity_policy": "REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN",
    })
    return {"ok": index["status"] == "DEV_DATA_READY", "dataset": str(out), "status": index["status"],
            "raw_roots": index["raw_root_count"], "usable_roots": len(records), "split_counts": split_counts,
            "exclusions": len(exclusions)}


def _load_release(path: Path, *, purpose: str, split: str) -> list[dict[str, Any]]:
    return list(load_cora_roots(
        path,
        splits=(split,),
        purpose=purpose,
        quality_profile="c2_dev_baseline_v1",
    ))


@dataclass
class _PairArrays:
    pair_ids: list[str]
    groups: list[tuple[str, str, str]]
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    source: list[tuple[str, str]]


def _arrays(samples: Sequence[Mapping[str, Any]], stats: Mapping[str, Any] | None = None) -> tuple[_PairArrays, dict[str, Any]]:
    ids, groups, sources, xs, ys, masks = [], [], [], [], [], []
    for sample in samples:
        candidates = sample["candidates"]
        ids.append(str(sample["pair_id"]))
        groups.append((str(sample["map"]), str(sample["family"]), str(sample["weather"])))
        sources.append((str(candidates[0]["source"]), str(candidates[1]["source"])))
        xs.append([list(c["feature"]["context"]) + [v for row in c["feature"]["candidate"] for v in row] for c in candidates])
        ys.append([[float(c["targets"][name]["value"]) if c["targets"][name]["mask"] else 0.0 for name in TARGETS] for c in candidates])
        masks.append([[bool(c["targets"][name]["mask"]) for name in TARGETS] for c in candidates])
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    mask = np.asarray(masks, dtype=bool)
    if x.ndim != 3 or x.shape[-1] != FEATURE_DIM:
        raise ValueError(f"dev_baseline_feature_shape:{x.shape}")
    if stats is None:
        mean, std = [], []
        for index in range(len(TARGETS)):
            values = y[:, :, index][mask[:, :, index]]
            if not len(values):
                raise ValueError(f"dev_baseline_no_train_target:{TARGETS[index]}")
            mean.append(float(values.mean())); spread = float(values.std()); std.append(1.0 if spread < 1e-6 else spread)
        stats = {"mean": mean, "std": std}
    return _PairArrays(ids, groups, x, y, mask, sources), dict(stats)


class DevWorldMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, output_dim: int = len(TARGETS)) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _loss(pred: Tensor, target: Tensor, mask: Tensor, stats: Mapping[str, Any], pair_progress_weight: float) -> Tensor:
    mean = torch.tensor(stats["mean"], dtype=pred.dtype, device=pred.device)
    std = torch.tensor(stats["std"], dtype=pred.dtype, device=pred.device)
    normalized = (target - mean) / std
    candidate = torch.nn.functional.smooth_l1_loss(pred, normalized, reduction="none")
    weights = mask.to(pred.dtype)
    row_loss = (candidate * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
    both_progress = mask[:, 0, 0] & mask[:, 1, 0]
    if bool(both_progress.any()):
        true_delta = normalized[:, 0, 0] - normalized[:, 1, 0]
        pred_delta = pred[:, 0, 0] - pred[:, 1, 0]
        pair_loss = torch.nn.functional.smooth_l1_loss(pred_delta[both_progress], true_delta[both_progress])
        return row_loss.mean() + pair_progress_weight * pair_loss
    return row_loss.mean()


def _predict(model: nn.Module, arrays: _PairArrays) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(arrays.x.reshape(-1, FEATURE_DIM))
        out = model(tensor).reshape(len(arrays.pair_ids), 2, len(TARGETS)).numpy()
    return out


def _unscale(pred: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    return pred * np.asarray(stats["std"], dtype=np.float32) + np.asarray(stats["mean"], dtype=np.float32)


def _fit_ridge(train: _PairArrays, stats: Mapping[str, Any], mode: str, reg: float) -> tuple[np.ndarray, np.ndarray]:
    if mode == "context":
        features = train.x[:, :, :H3_CONTEXT_DIM]
    elif mode == "candidate":
        features = train.x[:, :, H3_CONTEXT_DIM:]
    else:
        raise ValueError(f"dev_baseline_ridge_mode:{mode}")
    flat = features.reshape(-1, features.shape[-1])
    y = (train.y.reshape(-1, len(TARGETS)) - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    design = np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float32)], axis=1)
    gram = design.T @ design + float(reg) * np.eye(design.shape[1], dtype=np.float32)
    coef = np.linalg.solve(gram, design.T @ y)
    return coef[:-1], coef[-1]


def _ridge_predict(arrays: _PairArrays, coef: tuple[np.ndarray, np.ndarray], mode: str) -> np.ndarray:
    features = arrays.x[:, :, :H3_CONTEXT_DIM] if mode == "context" else arrays.x[:, :, H3_CONTEXT_DIM:]
    return features @ coef[0] + coef[1]


def _metrics(pred: np.ndarray, arrays: _PairArrays, *, label: str, bootstrap_rounds: int, seed: int) -> dict[str, Any]:
    pred_raw = pred
    per_head = {}
    for index, name in enumerate(TARGETS):
        valid = arrays.mask[:, :, index]
        errors = pred_raw[:, :, index][valid] - arrays.y[:, :, index][valid]
        per_head[name] = {"mae": float(np.mean(np.abs(errors))) if len(errors) else None,
                          "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                          "valid_candidate_count": int(valid.sum())}
    valid_progress = arrays.mask[:, 0, 0] & arrays.mask[:, 1, 0]
    actual_delta = arrays.y[:, 0, 0] - arrays.y[:, 1, 0]
    predicted_delta = pred_raw[:, 0, 0] - pred_raw[:, 1, 0]
    rank_valid = valid_progress & (np.abs(actual_delta) > 0.5)
    rank_acc = float(np.mean((predicted_delta[rank_valid] * actual_delta[rank_valid]) > 0)) if rank_valid.any() else None
    chosen = predicted_delta >= 0
    regret = np.maximum(arrays.y[:, 0, 0], arrays.y[:, 1, 0]) - np.where(chosen, arrays.y[:, 0, 0], arrays.y[:, 1, 0])
    regret = regret[valid_progress]
    metrics = {"label": label, "head_metrics": per_head, "progress_pair_count": int(valid_progress.sum()),
               "progress_ranking_accuracy": rank_acc, "progress_ranking_count": int(rank_valid.sum()),
               "progress_selection_regret_m": float(np.mean(regret)) if len(regret) else None}
    rng = random.Random(seed)
    boot_acc, boot_regret = [], []
    if valid_progress.any():
        indices = np.flatnonzero(valid_progress)
        for _ in range(bootstrap_rounds):
            sample = [indices[rng.randrange(len(indices))] for _ in indices]
            boot_regret.append(float(np.mean(regret[[int(np.flatnonzero(indices == item)[0]) for item in sample]])))
            if rank_valid.any():
                rank_indices = [item for item in sample if rank_valid[item]]
                if rank_indices:
                    boot_acc.append(float(np.mean((predicted_delta[rank_indices] * actual_delta[rank_indices]) > 0)))
    metrics["bootstrap_95"] = {
        "rounds": bootstrap_rounds,
        "selection_regret_m": [float(np.quantile(boot_regret, 0.025)), float(np.quantile(boot_regret, 0.975))] if boot_regret else None,
        "ranking_accuracy": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))] if boot_acc else None,
    }
    return metrics


def _group_metrics(pred: np.ndarray, arrays: _PairArrays) -> dict[str, Any]:
    out = {}
    for axis, index in (("map", 0), ("family", 1), ("weather", 2)):
        values = {}
        for value in sorted({group[index] for group in arrays.groups}):
            rows = np.asarray([i for i, group in enumerate(arrays.groups) if group[index] == value])
            valid = arrays.mask[rows, :, 0]
            errors = pred[rows, :, 0][valid] - arrays.y[rows, :, 0][valid]
            values[value] = {"progress_mae_m": float(np.mean(np.abs(errors))) if len(errors) else None, "valid_candidate_count": int(valid.sum())}
        out[axis] = values
    return out


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_baselines(config_path: Path | str, project: Path) -> dict[str, Any]:
    config = _config(config_path)
    release = project / "generated/h6/cora" / str(config["dataset_id"])
    train_samples = _load_release(release, purpose="training", split="train")
    valid_samples = _load_release(release, purpose="evaluation", split="validation")
    train, stats = _arrays(train_samples)
    valid, _ = _arrays(valid_samples, stats)
    out = release / "world-baseline"
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "normalization.json", stats)
    models: list[dict[str, Any]] = []
    deadline = time.perf_counter() + float(config["time_limit_s"])
    torch.set_num_threads(int(config["cpu_threads"]))
    for seed in config["seeds"]:
        if time.perf_counter() >= deadline:
            break
        _seed_all(int(seed))
        model = DevWorldMLP()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        best, best_state, stale = float("inf"), None, 0
        x_train = torch.from_numpy(train.x)
        y_train = torch.from_numpy(train.y)
        m_train = torch.from_numpy(train.mask)
        x_valid = torch.from_numpy(valid.x)
        y_valid = torch.from_numpy(valid.y)
        m_valid = torch.from_numpy(valid.mask)
        for epoch in range(int(config["max_epochs"])):
            model.train(); optimizer.zero_grad()
            pred = model(x_train.reshape(-1, FEATURE_DIM)).reshape(len(train.pair_ids), 2, len(TARGETS))
            loss = _loss(pred, y_train, m_train, stats, float(config["pair_progress_weight"]))
            if not torch.isfinite(loss):
                raise ValueError(f"dev_baseline_loss_nonfinite:{seed}:{epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(x_valid.reshape(-1, FEATURE_DIM)).reshape(len(valid.pair_ids), 2, len(TARGETS))
                val_loss = float(_loss(val_pred, y_valid, m_valid, stats, float(config["pair_progress_weight"])).item())
            if val_loss < best - 1e-8:
                best, best_state, stale = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            if stale >= int(config["patience"]):
                break
        if best_state is None:
            raise RuntimeError(f"dev_baseline_no_checkpoint:{seed}")
        model.load_state_dict(best_state)
        checkpoint = out / f"world_mlp_seed{seed}.pt"
        torch.save({"schema_version": SCHEMA, "seed": int(seed), "input_dim": FEATURE_DIM, "target_heads": list(TARGETS), "stats": stats, "state_dict": model.state_dict(), "best_validation_loss": best}, checkpoint)
        pred = _unscale(_predict(model, valid), stats)
        models.append({"seed": int(seed), "checkpoint": str(checkpoint), "best_validation_loss": best, "metrics": _metrics(pred, valid, label=f"world_mlp_seed{seed}", bootstrap_rounds=int(config["bootstrap_rounds"]), seed=int(seed)), "group_metrics": _group_metrics(pred, valid), "predictions": pred.tolist()})
    if len(models) != len(config["seeds"]):
        raise RuntimeError("dev_baseline_time_limit_before_all_seeds")
    ridges = {}
    for mode in ("context", "candidate"):
        coef = _fit_ridge(train, stats, mode, float(config["ridge_lambda"]))
        pred = _unscale(_ridge_predict(valid, coef, mode), stats)
        ridges[f"ridge_{mode}"] = {"metrics": _metrics(pred, valid, label=f"ridge_{mode}", bootstrap_rounds=int(config["bootstrap_rounds"]), seed=71), "predictions": pred.tolist()}
    mean_pred = np.broadcast_to(np.asarray(stats["mean"], dtype=np.float32), (len(valid.pair_ids), 2, len(TARGETS))).copy()
    baselines = {"train_mean": {"metrics": _metrics(mean_pred, valid, label="train_mean", bootstrap_rounds=int(config["bootstrap_rounds"]), seed=73), "predictions": mean_pred.tolist()}, **ridges}
    actual = valid.y
    valid_progress = valid.mask[:, 0, 0] & valid.mask[:, 1, 0]
    strategy = {}
    for source_index, source_name in ((0, "expert"), (1, "vla")):
        chosen = np.full((len(valid.pair_ids), 2, len(TARGETS)), np.nan, dtype=np.float32)
        chosen[:, 0, 0] = actual[:, source_index, 0]
        chosen[:, 1, 0] = actual[:, 1 - source_index, 0]
        strategy[source_name] = {"progress_pair_count": int(valid_progress.sum()), "progress_selection_regret_m": float(np.mean((np.maximum(actual[:, 0, 0], actual[:, 1, 0]) - actual[:, source_index, 0])[valid_progress])) if valid_progress.any() else None}
    predictions = {"pair_ids": valid.pair_ids, "groups": [list(group) for group in valid.groups], "targets": actual.tolist(), "world_models": models, "baselines": baselines, "fixed_source_strategies": strategy}
    _write(out / "validation-predictions.json", predictions)
    report = {"schema_version": SCHEMA, "dataset_id": config["dataset_id"], "status": "MEASURED", "train_roots": len(train.pair_ids), "validation_roots": len(valid.pair_ids), "models": [{"seed": item["seed"], "checkpoint": item["checkpoint"], "best_validation_loss": item["best_validation_loss"], "metrics": item["metrics"], "group_metrics": item["group_metrics"]} for item in models], "baselines": {name: item["metrics"] for name, item in baselines.items()}, "fixed_source_strategies": strategy, "swap_check": _swap_check(models[0], valid, stats), "gain_status": _gain_status(models, baselines), "target_heads": list(TARGETS), "calibration": "NOT_RUN", "closed_loop": "NOT_MEASURED"}
    _write(out / "baseline-report.json", report)
    return {"ok": True, "status": report["status"], "models": len(models), "train_roots": len(train.pair_ids), "validation_roots": len(valid.pair_ids), "report": str(out / "baseline-report.json")}


def _swap_check(model_record: Mapping[str, Any], valid: _PairArrays, stats: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(model_record["checkpoint"], map_location="cpu", weights_only=False)
    model = DevWorldMLP(); model.load_state_dict(checkpoint["state_dict"])
    swapped = _PairArrays(valid.pair_ids, valid.groups, valid.x[:, ::-1].copy(), valid.y[:, ::-1].copy(), valid.mask[:, ::-1].copy(), [(b, a) for a, b in valid.source])
    p = _predict(model, valid); q = _predict(model, swapped)
    expected = p[:, ::-1]
    return {"max_abs_output_difference": float(np.max(np.abs(q - expected))) if q.size else 0.0, "passed": bool(np.allclose(q, expected, atol=1e-6, rtol=1e-6)), "source_metadata_in_input": False}


def _gain_status(models: Sequence[Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]]) -> str:
    model_mae = float(np.mean([m["metrics"]["head_metrics"]["route_progress_m"]["mae"] for m in models]))
    base_mae = min(float(item["metrics"]["head_metrics"]["route_progress_m"]["mae"]) for item in baselines.values())
    return "DEMONSTRATED_GAIN" if model_mae < base_mae else "NO_DEMONSTRATED_GAIN"


def finalize_devbaseline(config_path: Path | str, project: Path) -> dict[str, Any]:
    config = _config(config_path)
    release = project / "generated/h6/cora" / str(config["dataset_id"])
    index = _json(release / "release-index.json")
    audit = release / "audit-report.json"
    baseline = release / "world-baseline" / "baseline-report.json"
    normalization = release / "world-baseline" / "normalization.json"
    tests = project / "docs/runtime-evidence/h6" / str(config["dataset_id"]) / "test-report.json"
    failures = []
    if index.get("status") != "DEV_DATA_READY": failures.append("data_not_ready")
    if not audit.is_file(): failures.append("audit_report_missing")
    if audit.is_file() and _json(audit).get("status") != "DEV_DATA_READY": failures.append("audit_not_ready")
    split_counts = index.get("split_counts", {})
    if index.get("training_splits") != list(config["training_splits"]): failures.append("training_split_contract")
    if index.get("evaluation_splits") != list(config["evaluation_splits"]): failures.append("evaluation_split_contract")
    if index.get("target_heads") != list(config["target_heads"]): failures.append("target_head_contract")
    if int(index.get("usable_root_count", -1)) != sum(int(value) for value in split_counts.values()): failures.append("root_count_mismatch")
    if int(index.get("exclusion_count", -1)) != len(index.get("exclusions", ())): failures.append("exclusion_count_mismatch")
    if not baseline.is_file():
        failures.append("baseline_report_missing")
        baseline_report = {}
    else:
        baseline_report = _json(baseline)
        if baseline_report.get("status") != "MEASURED": failures.append("baseline_not_measured")
        if baseline_report.get("train_roots") != split_counts.get("train"): failures.append("baseline_train_scope")
        if baseline_report.get("validation_roots") != split_counts.get("validation"): failures.append("baseline_validation_scope")
        if baseline_report.get("target_heads") != list(config["target_heads"]): failures.append("baseline_target_heads")
        if len(baseline_report.get("models", ())) != len(config["seeds"]): failures.append("baseline_seed_count")
        if baseline_report.get("swap_check", {}).get("passed") is not True: failures.append("swap_check")
        if baseline_report.get("swap_check", {}).get("source_metadata_in_input") is not False: failures.append("source_metadata_input")
        if not normalization.is_file():
            failures.append("normalization_missing")
        else:
            stats = _json(normalization)
            if len(stats.get("mean", ())) != len(TARGETS) or len(stats.get("std", ())) != len(TARGETS): failures.append("normalization_shape")
            if any((not math.isfinite(float(value)) or float(value) <= 0) for value in stats.get("std", ())): failures.append("normalization_std")
        for model in baseline_report.get("models", ()):
            checkpoint = Path(str(model.get("checkpoint", "")))
            if not checkpoint.is_file():
                failures.append(f"checkpoint_missing:{model.get('seed')}")
                continue
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                probe = DevWorldMLP()
                probe.load_state_dict(payload["state_dict"])
            except Exception:
                failures.append(f"checkpoint_unloadable:{model.get('seed')}")
    if not tests.is_file() or not _json(tests).get("passed"): failures.append("tests_not_passed")
    status = "DEV_BASELINE_GATE_PASSED" if not failures else "DEV_BASELINE_GATE_FAILED"
    report = {"schema_version": "safedrive.cora.dev_final.v1", "dataset_id": config["dataset_id"], "status": status, "passed": not failures, "original_quality_gate": "GATE_FAILED", "failures": failures, "data_release": index, "audit_report": _json(audit) if audit.is_file() else None, "baseline_report": baseline_report if baseline.is_file() else None, "tests": _json(tests) if tests.is_file() else None, "old_hash_policy": "REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN", "closed_loop": "NOT_MEASURED", "vla_finetune": "NOT_RUN"}
    _write(release / "final-delivery.json", report)
    evidence = project / "docs/runtime-evidence/h6" / str(config["dataset_id"])
    _write(evidence / "data-quality.json", report)
    return {"ok": True, "status": status, "passed": not failures, "failures": failures, "report": str(release / "final-delivery.json")}


__all__ = ["finalize_devbaseline", "prepare_release", "train_baselines"]
