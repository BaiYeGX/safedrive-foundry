#!/usr/bin/env python3
"""C3 native SimLingo SFT, baseline and evaluation entry point.

This command is intentionally independent from the online ScenarioRuntime.
It consumes the frozen C2 development release, calls the native differentiable
SimLingo model, and writes one immutable run directory under generated/.  The
heavy model imports happen only for commands that need a model; ``audit`` and
the metric helpers remain usable on CPU.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "safedrive_foundry", REPO / "simlingo-main"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from driving_vla.model.sft import (  # noqa: E402
    RELEASE_ID,
    EXECUTION_DT_S,
    ROUTE_STEPS,
    SPEED_DT_S,
    SPEED_STEPS,
    SFTSample,
    aggregate_metrics,
    build_sft_manifest,
    deterministic_order,
    masked_smooth_l1,
    root_metrics,
    sha256_file,
)


DEFAULT_RELEASE = REPO / "generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1"
DEFAULT_CKPT = REPO / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
DEFAULT_HYDRA = REPO / "models/simlingo/simlingo/.hydra/config.yaml"
DEFAULT_INTERNVL = REPO / "models/InternVL2-1B"
DEFAULT_OUTPUT = REPO / "generated/h6/cora"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha_object(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _seed_everything(seed: int) -> None:
    """Reset every RNG used by the native forward/training path."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _git_identity() -> dict[str, Any]:
    """Capture the source snapshot without changing or cleaning the worktree."""

    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return ""
        return result.stdout

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD").strip(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_lines": status.splitlines(),
    }


def _write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_root).expanduser().resolve() / str(args.run_id)


def _ensure_run_dir(args: argparse.Namespace) -> Path:
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _config_payload(args: argparse.Namespace, *, model_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "safedrive.c3.sft_run_config.v1",
        "run_id": str(args.run_id),
        "release_root": str(Path(args.release_root).resolve()),
        "release_id": RELEASE_ID,
        "seed": int(args.seed),
        "updates": int(args.updates),
        "microbatch": 1,
        "gradient_accumulation": int(args.accumulation),
        "lora_learning_rate": float(args.lora_lr),
        "driving_head_learning_rate": float(args.head_lr),
        "weight_decay": float(args.weight_decay),
        "route_loss": "valid_coordinate_element_smooth_l1_beta_1_mean",
        "speed_loss": "valid_coordinate_element_smooth_l1_beta_1_mean",
        "route_steps": ROUTE_STEPS,
        "route_spacing_m": 1.0,
        "speed_steps": SPEED_STEPS,
        "speed_dt_s": SPEED_DT_S,
        "execution_timeline_dt_s": float(EXECUTION_DT_S),
        "route_target_source": "anchor.route_with_ego_origin",
        "speed_target_source": "expert_execution_timeline",
        "warmup_fraction": 0.05,
        "gradient_clip": 1.0,
        "precision_policy": "bf16_preferred_verified_by_real_batch",
        "device": str(args.device),
        "max_hours": float(args.max_hours),
        "bootstrap_rounds": 1000,
        "bootstrap_seed": 71,
        "future_labels_in_input": False,
        "world_labels_in_sft": False,
        "teacher_contract": "expert_nominal_proposal_only",
        "carla": {
            "required": False,
            "installation": r"E:\CARLA_0.9.16\CarlaUE4.exe",
            "reason": "C2 release already contains complete valid expert timelines; no supplemental collection was needed",
        },
        "reproduction_commands": [
            f"python scripts/h6_cora_sft.py audit --run-id {args.run_id} --device cuda",
            f"python scripts/h6_cora_sft.py smoke --run-id {args.run_id} --device cuda",
            f"python scripts/h6_cora_sft.py baseline --run-id {args.run_id} --device cuda",
            f"python scripts/h6_cora_sft.py train --run-id {args.run_id} --device cuda --updates {int(args.updates)} --accumulation {int(args.accumulation)} --max-hours {float(args.max_hours):g}",
            f"python scripts/h6_cora_sft.py evaluate --run-id {args.run_id} --device cuda",
            f"python scripts/h6_cora_sft.py verify --run-id {args.run_id} --device cpu",
        ],
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "git": _git_identity(),
    }
    if model_identity is not None:
        payload["model"] = dict(model_identity)
    return payload


def _model_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _load_manifest(run_dir: Path) -> tuple[tuple[SFTSample, ...], dict[str, Any]]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run_requires_audit:{path}")
    payload = _read_json(path)
    digest = payload.get("manifest_sha256")
    check = dict(payload)
    check.pop("manifest_sha256", None)
    if digest != _sha_object(check):
        raise ValueError("manifest_sha256_mismatch")
    samples = tuple(SFTSample(**item) for item in payload.get("samples", ()))
    return samples, payload


def _torch_modules():
    import torch
    from PIL import Image
    from simlingo_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel, LanguageLabel

    return torch, Image, DrivingExample, DrivingInput, DrivingLabel, LanguageLabel


def _make_runtime(args: argparse.Namespace, *, ckpt_path: Path | None = None):
    from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

    requested = str(args.device).lower()
    if requested == "auto":
        import torch

        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("c3_cuda_required_but_unavailable")
    runtime = SimLingoNeuralRuntime(
        ckpt_path=ckpt_path or args.checkpoint,
        hydra_config=args.hydra_config,
        internvl_root=args.internvl_root,
        device=requested,
        predict_language=False,
    )
    report = runtime.load()
    if not report.ok:
        raise RuntimeError(f"c3_model_load_failed:{report.error}")
    return runtime, runtime.model, requested, report


def _freeze_trainable(model: Any) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """Freeze everything except native LoRA and the two driving heads."""

    lora: list[tuple[str, Any]] = []
    heads: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        if "lora_" in name and name.startswith("language_model."):
            parameter.requires_grad_(True)
            lora.append((name, parameter))
        elif name.startswith("adaptors.driving.route_head.") or name.startswith("adaptors.driving.speed_wps_head."):
            parameter.requires_grad_(True)
            heads.append((name, parameter))
    if not lora:
        raise RuntimeError("c3_no_lora_parameters")
    if not heads:
        raise RuntimeError("c3_no_driving_head_parameters")
    # Query embeddings, wp encoder and every vision/base parameter remain frozen.
    model.vision_model.eval()
    model.wp_encoder.eval()
    return lora, heads


def _set_lora_enabled(lora: Sequence[tuple[str, Any]], enabled: bool) -> None:
    for _, parameter in lora:
        parameter.requires_grad_(enabled)


def _make_example(runtime: Any, model: Any, sample: SFTSample, device: str):
    torch, Image, DrivingExample, DrivingInput, DrivingLabel, LanguageLabel = _torch_modules()
    image = np.asarray(Image.open(sample.image_path).convert("RGB"))
    driving_input = runtime.build_driving_input(
        image,
        speed_mps=sample.ego_speed_mps,
        target_point_xy=sample.target_ego_1,
        target_point2_xy=sample.target_ego_2,
        command_text=None,
        eval_route_as="target_point",
        image_layout="rgb",
        official_contract=True,
    )
    driving_input = runtime._to_device(driving_input, device=device)
    route = torch.tensor(np.asarray(sample.route_target), dtype=torch.float32, device=device).unsqueeze(0)
    speed = torch.tensor(np.asarray(sample.speed_target), dtype=torch.float32, device=device).unsqueeze(0)
    route_mask = torch.tensor(np.asarray(sample.route_mask), dtype=torch.bool, device=device).unsqueeze(0)
    speed_mask = torch.tensor(np.asarray(sample.speed_mask), dtype=torch.bool, device=device).unsqueeze(0)
    # DrivingAdaptor reads path for the route head and waypoints for speed_wps.
    answer = LanguageLabel(None, None, None, None, [""], None)
    label = DrivingLabel(
        waypoints=speed,
        path=route,
        answer=answer,
        image_ff_org=torch.empty((0,), device=device),
        eval_infos={"route_mask": route_mask, "speed_mask": speed_mask},
    )
    return DrivingExample(
        driving_input=driving_input,
        driving_label=label,
        run_id=[sample.root_id],
        qa_templates=None,
    )


def _forward_driving(model: Any, example: Any, *, train: bool):
    """Run image replacement without gradients, then native language/driver heads."""

    import torch

    driving_input = example.driving_input
    # Frozen vision and input embedding work should not retain a backward graph.
    with torch.no_grad():
        input_dict = model.adaptors(example, inference=True)
        input_dict = model.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=input_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            wp_encoder=model.wp_encoder,
        )
    # ``replace_placeholder_tokens`` updates the language portion in place and
    # also restores the driving query tokens into ``inputs``.  Feeding only
    # ``language_inputs`` would drop the 30 native route/speed query tokens and
    # make the upstream permutation index the wrong sequence length.
    embeddings = input_dict["inputs"].to(dtype=model.language_model.model.dtype)
    if train:
        outputs = model.language_model.model(
            inputs_embeds=embeddings,
            attention_mask=input_dict["inputs_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    else:
        with torch.inference_mode():
            outputs = model.language_model.model(
                inputs_embeds=embeddings,
                attention_mask=input_dict["inputs_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
    features = outputs.hidden_states[-1]
    logits = outputs[0]
    features_by = model.adaptors.split_outputs_by_adaptor(input_dict, features)
    logits_by = model.adaptors.split_outputs_by_adaptor(input_dict, logits)
    predictions = model.adaptors.driving.get_predictions(features_by["driving"], logits_by.get("driving"))
    return predictions, features_by["driving"], input_dict


def _losses(predictions: Mapping[str, Any], example: Any) -> tuple[Any, Any, Any]:
    import torch

    info = example.driving_label.eval_infos
    route_loss = (
        masked_smooth_l1(predictions["route"], example.driving_label.path, info["route_mask"])
        if bool(info["route_mask"].any().item())
        else predictions["route"].sum() * 0.0
    )
    speed_loss = (
        masked_smooth_l1(predictions["speed_wps"], example.driving_label.waypoints, info["speed_mask"])
        if bool(info["speed_mask"].any().item())
        else predictions["speed_wps"].sum() * 0.0
    )
    total = route_loss.float() + speed_loss.float()
    if not torch.isfinite(total):
        raise FloatingPointError("c3_non_finite_sft_loss")
    return total, route_loss.float(), speed_loss.float()


def _prediction_record(predictions: Mapping[str, Any], sample: SFTSample, latency_s: float, *, model_name: str) -> dict[str, Any]:
    route = predictions["route"].detach().float().cpu().numpy()[0]
    speed = predictions["speed_wps"].detach().float().cpu().numpy()[0]
    record = root_metrics({"route": route, "speed": speed}, sample)
    record.update(
        {
            "model": model_name,
            "route": route.tolist(),
            "speed": speed.tolist(),
            "latency_s": float(latency_s),
        }
    )
    return record


def _prediction_max_abs_diff(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Compare route and speed heads independently (their lengths differ)."""

    import torch

    route_diff = (first["route"].detach().float() - second["route"].detach().float()).abs().max()
    speed_diff = (first["speed_wps"].detach().float() - second["speed_wps"].detach().float()).abs().max()
    return float(torch.maximum(route_diff, speed_diff).item())


def _evaluate_model(runtime: Any, model: Any, samples: Sequence[SFTSample], *, device: str, model_name: str, repeat: bool = False):
    import torch

    model.eval()
    model.vision_model.eval()
    model.wp_encoder.eval()
    rows: list[dict[str, Any]] = []
    repeats: list[float] = []
    for sample in samples:
        example = _make_example(runtime, model, sample, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictions, _, _ = _forward_driving(model, example, train=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append(_prediction_record(predictions, sample, elapsed, model_name=model_name))
        if repeat:
            predictions_2, _, _ = _forward_driving(model, example, train=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            route_1 = predictions["route"].detach().float()
            speed_1 = predictions["speed_wps"].detach().float()
            repeats.append(_prediction_max_abs_diff(predictions, predictions_2))
        del example, predictions
    return rows, repeats


def _save_trainable_state(model: Any, path: Path) -> dict[str, Any]:
    import torch

    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(state, path)
    return {"keys": sorted(state), "parameter_count": int(sum(v.numel() for v in state.values())), "path": str(path)}


def _load_trainable_state(model: Any, path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("c3_adapter_state_not_mapping")
    names = {name for name, _ in model.named_parameters()}
    unknown = sorted(set(state) - names)
    if unknown:
        raise ValueError(f"c3_adapter_unknown_keys:{unknown[:5]}")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = sorted(expected - set(state))
    if missing:
        raise ValueError(f"c3_adapter_missing_keys:{missing[:5]}")
    with torch.no_grad():
        by_name = dict(model.named_parameters())
        for name, value in state.items():
            by_name[name].copy_(value.to(device=by_name[name].device, dtype=by_name[name].dtype))
    return {"keys": sorted(state), "parameter_count": int(sum(v.numel() for v in state.values()))}


def _frozen_fingerprints(model: Any) -> dict[str, str]:
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            selected.append((name, parameter))
    if not selected:
        return {}
    picks = [selected[0], selected[len(selected) // 2], selected[-1]]
    result: dict[str, str] = {}
    for name, parameter in picks:
        # NumPy has no bfloat16 dtype; the cast is only for a stable audit
        # fingerprint and never participates in the training computation.
        value = parameter.detach().float().cpu().contiguous().numpy()
        result[name] = hashlib.sha256(value.tobytes()).hexdigest()
    return result


def _optimizer_state_cpu(optimizer: Any) -> dict[str, Any]:
    state = optimizer.state_dict()
    def move(value: Any) -> Any:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, list):
            return [move(v) for v in value]
        return value
    return move(state)


def _set_schedule(optimizer: Any, update: int, *, total: int, head_lr: float, lora_lr: float, warmup: int, lora: Sequence[tuple[str, Any]]) -> dict[str, float]:
    if update <= warmup:
        head_scale = update / max(1, warmup)
        lora_scale = 0.0
    elif update <= 2 * warmup:
        head_scale = 1.0
        lora_scale = (update - warmup) / max(1, warmup)
    else:
        head_scale = 0.1 + 0.9 * (1.0 + math.cos(math.pi * (update - 2 * warmup) / max(1, total - 2 * warmup))) / 2.0
        lora_scale = head_scale
    values = {"head_lr": head_lr * head_scale, "lora_lr": lora_lr * lora_scale}
    for group in optimizer.param_groups:
        group_name = str(group.get("name", ""))
        group["lr"] = values["lora_lr"] if group_name == "lora" else values["head_lr"]
    _set_lora_enabled(lora, update > warmup)
    return values


def _gradient_gate_stats(sft_loss: Any, features: Any, lora_params: Sequence[tuple[str, Any]], device: str) -> dict[str, Any]:
    """Measure the C4 gate on a real shared hidden state during C3 smoke."""

    import torch
    from torch import nn

    if not lora_params:
        return {"status": "NO_LORA"}
    hidden = features.mean(dim=1).float()
    aux_head = nn.Sequential(nn.Linear(hidden.shape[-1], 32), nn.SiLU(), nn.Linear(32, 4)).to(device=device, dtype=torch.float32)
    target = torch.zeros_like(aux_head(hidden))
    aux_loss = torch.nn.functional.smooth_l1_loss(aux_head(hidden), target)
    params = [parameter for _, parameter in lora_params]
    grad_s = torch.autograd.grad(sft_loss, params, retain_graph=True, allow_unused=True)
    grad_a = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)
    flat_s = torch.cat([g.detach().float().reshape(-1) for g in grad_s if g is not None]) if any(g is not None for g in grad_s) else torch.zeros(1, device=device)
    flat_a = torch.cat([g.detach().float().reshape(-1) for g in grad_a if g is not None]) if any(g is not None for g in grad_a) else torch.zeros_like(flat_s)
    ns = torch.linalg.vector_norm(flat_s)
    na = torch.linalg.vector_norm(flat_a)
    cosine = torch.dot(flat_s, flat_a) / (ns * na + 1e-12)
    weight = torch.clamp(cosine, min=0.0)
    q = torch.clamp(0.25 * ns / (na + 1e-12), max=1.0)
    return {
        "status": "MEASURED",
        "shared_hidden_dim": int(hidden.shape[-1]),
        "sft_grad_norm": float(ns.item()),
        "aux_grad_norm": float(na.item()),
        "cosine": float(cosine.item()),
        "w": float(weight.item()),
        "q": float(q.item()),
        "aux_nonzero": bool(na.item() > 0.0),
    }


def cmd_audit(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"audit_already_exists:{run_dir}")
    samples, manifest = build_sft_manifest(args.release_root, splits=("train", "validation"))
    if manifest["split_counts"].get("train") != 158 or manifest["split_counts"].get("validation") != 53:
        raise ValueError(f"c3_expected_release_counts:{manifest['split_counts']}")
    model_identity = _model_identity(Path(args.checkpoint))
    config = _config_payload(args, model_identity=model_identity)
    _write_json(run_dir / "run_config.json", config)
    _write_json(run_dir / "manifest.json", manifest)
    audit = {
        "schema_version": "safedrive.c3.sft_audit.v1",
        "status": "AUDIT_PASSED",
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(samples),
        "split_counts": manifest["split_counts"],
        "teacher_sources": sorted({sample.teacher_source for sample in samples}),
        "teacher_generators": sorted({sample.teacher_generator for sample in samples}),
        "route_valid_point_counts": {
            split: int(sum(sum(sample.route_mask) for sample in samples if sample.split == split))
            for split in ("train", "validation")
        },
        "speed_valid_point_counts": {
            split: int(sum(sum(sample.speed_mask) for sample in samples if sample.split == split))
            for split in ("train", "validation")
        },
        "route_target_source": sorted({sample.route_source for sample in samples}),
        "speed_target_source": sorted({sample.speed_source for sample in samples}),
        "execution_timeline_rows": {
            "min": min(sample.execution_timeline_rows for sample in samples),
            "max": max(sample.execution_timeline_rows for sample in samples),
        },
        "execution_timeline_dt_s": sorted({sample.execution_timeline_dt_s for sample in samples}),
        "masked_route_samples": int(sum(not all(sample.route_mask) for sample in samples)),
        "masked_speed_samples": int(sum(not all(sample.speed_mask) for sample in samples)),
        "speed_missing_reason_counts": {
            reason: sum(reason in sample.speed_missing_reasons for sample in samples)
            for reason in sorted({reason for sample in samples for reason in sample.speed_missing_reasons})
        },
        "world_labels_in_sft": False,
        "future_labels_in_input": False,
        "release_index_sha256": sha256_file(Path(args.release_root) / "release-index.json"),
        "model_identity": model_identity,
    }
    audit["audit_sha256"] = _sha_object(audit)
    _write_json(run_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    output = run_dir / "smoke.json"
    if output.exists():
        raise FileExistsError(f"smoke_already_exists:{output}")
    samples, manifest = _load_manifest(run_dir)
    train = [sample for sample in samples if sample.split == "train"]
    if not train:
        raise ValueError("c3_smoke_no_train_samples")
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    lora, heads = _freeze_trainable(model)
    _set_lora_enabled(lora, True)
    model.train()
    model.vision_model.eval()
    model.wp_encoder.eval()
    sample = deterministic_order(train, args.seed)[0]
    example = _make_example(runtime, model, sample, device)
    frozen_before = _frozen_fingerprints(model)
    trainable_before = {name: parameter.detach().clone() for name, parameter in lora + heads}
    import torch
    optimizer = torch.optim.AdamW(
        [
            {"name": "lora", "params": [p for _, p in lora], "lr": args.lora_lr},
            {"name": "head", "params": [p for _, p in heads], "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    start = time.perf_counter()
    predictions, features, _ = _forward_driving(model, example, train=True)
    total, route_loss, speed_loss = _losses(predictions, example)
    gate = _gradient_gate_stats(total, features, lora, device)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    finite_grad = True
    nonzero_grad = 0
    for _, parameter in lora + heads:
        if parameter.grad is not None:
            finite_grad = finite_grad and bool(torch.isfinite(parameter.grad).all().item())
            nonzero_grad += int(torch.count_nonzero(parameter.grad).item())
    torch.nn.utils.clip_grad_norm_([p for _, p in lora + heads], 1.0)
    optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    changed = {
        name: float((parameter.detach() - trainable_before[name]).abs().max().item())
        for name, parameter in lora + heads
    }
    frozen_after = _frozen_fingerprints(model)
    adapter_path = run_dir / "smoke_adapter.pt"
    adapter_meta = _save_trainable_state(model, adapter_path)
    # Reload into an independent base model and compare the updated predictions.
    # The optimizer step above ran in train mode; recompute the reference in
    # deterministic eval mode before comparing it with the fresh model.
    model.eval()
    model.vision_model.eval()
    model.wp_encoder.eval()
    example_eval = _make_example(runtime, model, sample, device)
    pred_eval, _, _ = _forward_driving(model, example_eval, train=False)
    runtime_reload, model_reload, device_reload, reload_report = _make_runtime(args)
    _freeze_trainable(model_reload)
    _load_trainable_state(model_reload, adapter_path)
    model_reload.eval()
    example_reload = _make_example(runtime_reload, model_reload, sample, device_reload)
    pred_reload, _, _ = _forward_driving(model_reload, example_reload, train=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    reload_diff = _prediction_max_abs_diff(pred_eval, pred_reload)
    payload: dict[str, Any] = {
        "schema_version": "safedrive.c3.sft_smoke.v1",
        "status": "SMOKE_PASSED" if finite_grad and nonzero_grad > 0 and all(v > 0.0 for v in changed.values()) and frozen_before == frozen_after and reload_diff <= 1e-5 else "SMOKE_FAILED",
        "sample_root": sample.root_id,
        "device": device,
        "model_dtype": str(getattr(model.language_model.model, "dtype", "unknown")),
        "model_load": vars(report),
        "reload_model_load": vars(reload_report),
        "loss": {"total": float(total.detach().item()), "route": float(route_loss.detach().item()), "speed": float(speed_loss.detach().item())},
        "finite_grad": finite_grad,
        "nonzero_gradient_elements": nonzero_grad,
        "trainable_parameter_count": int(sum(p.numel() for _, p in lora + heads)),
        "lora_parameter_count": int(sum(p.numel() for _, p in lora)),
        "driving_head_parameter_count": int(sum(p.numel() for _, p in heads)),
        "changed_parameter_max_abs": changed,
        "frozen_fingerprint_before": frozen_before,
        "frozen_fingerprint_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "checkpoint_roundtrip_max_abs": reload_diff,
        "checkpoint_roundtrip_tolerance": 1e-5,
        "adapter": adapter_meta,
        "gradient_gate": gate,
        "wall_time_s": elapsed,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30) if device.startswith("cuda") else None,
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30) if device.startswith("cuda") else None,
    }
    payload["smoke_sha256"] = _sha_object(payload)
    _write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "SMOKE_PASSED":
        return 2
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    output = run_dir / "m0_predictions.json"
    if output.exists():
        raise FileExistsError(f"baseline_already_exists:{output}")
    samples, _ = _load_manifest(run_dir)
    validation = [sample for sample in samples if sample.split == "validation"]
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    rows, repeats = _evaluate_model(runtime, model, validation, device=device, model_name="m0", repeat=True)
    if len(rows) != 53:
        raise ValueError(f"c3_m0_validation_count:{len(rows)}")
    max_repeat = max(repeats) if repeats else 0.0
    baseline = {
        "schema_version": "safedrive.c3.m0_predictions.v1",
        "status": "M0_MEASURED",
        "model": vars(report),
        "rows": rows,
        "row_count": len(rows),
        "repeat_forward_max_abs": max_repeat,
        "eps_ADE_m": max(1e-5, 10.0 * max_repeat),
        "repeat_forward_count": len(repeats),
    }
    baseline["content_sha256"] = _sha_object(baseline)
    _write_json(output, baseline)
    _write_json(run_dir / "metrics-spec.json", {
        "schema_version": "safedrive.c3.metrics_spec.v1",
        "primary": "P-ADE.route_ade_m",
        "required_metrics": ["P-ADE", "P-FDE", "P-WP", "P-VALID", "P-FAIL"],
        "metric_mapping": {
            "P-ADE": "route_ade_m",
            "P-FDE": "route_fde_m",
            "P-WP": "speed_wp_ade_m",
            "P-VALID": "prediction_valid and valid point counts",
            "P-FAIL": "non-finite or failed prediction rows",
            "P-SPEED": "N/A without trusted speed ground truth conversion",
        },
        "aggregation": "root_equal",
        "mask": "same_validation_root_and_target_masks",
        "eps_ADE_m": baseline["eps_ADE_m"],
        "bootstrap_rounds": 1000,
        "bootstrap_seed": 71,
        "p_speed": "N/A_without_trusted_time_speed_ground_truth",
        "route_steps": ROUTE_STEPS,
        "speed_steps": SPEED_STEPS,
        "speed_dt_s": SPEED_DT_S,
        "execution_timeline_dt_s": EXECUTION_DT_S,
    })
    print(json.dumps({"status": baseline["status"], "row_count": len(rows), "eps_ADE_m": baseline["eps_ADE_m"]}, indent=2))
    return 0


def _save_training_checkpoint(path: Path, model: Any, optimizer: Any, *, update: int, sample_index: int, epoch: int, log: Sequence[Mapping[str, Any]]) -> None:
    import torch

    def is_update_parameter(name: str) -> bool:
        return (
            (name.startswith("language_model.") and "lora_" in name)
            or name.startswith("adaptors.driving.route_head.")
            or name.startswith("adaptors.driving.speed_wps_head.")
        )

    state = {
        "schema_version": "safedrive.c3.m1_resume.v1",
        "update": int(update),
        "sample_index": int(sample_index),
        "epoch": int(epoch),
        # Include the complete explicit update whitelist even during the first
        # head-only warmup, when LoRA tensors have requires_grad=False.  This
        # makes resume independent of the point at which the checkpoint was
        # written and prevents silently dropping the frozen-for-warmup LoRA
        # state.
        "model_state": {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if is_update_parameter(name)
        },
        "optimizer_state": _optimizer_state_cpu(optimizer),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "log_tail": list(log[-5:]),
    }
    if torch.cuda.is_available():
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def _restore_training_checkpoint(path: Path, model: Any, optimizer: Any) -> tuple[int, int, int, list[dict[str, Any]]]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != "safedrive.c3.m1_resume.v1":
        raise ValueError("c3_resume_schema")
    by_name = dict(model.named_parameters())
    expected = {
        name
        for name, parameter in model.named_parameters()
        if (name.startswith("language_model.") and "lora_" in name)
        or name.startswith("adaptors.driving.route_head.")
        or name.startswith("adaptors.driving.speed_wps_head.")
    }
    got = set(state.get("model_state", {}))
    if expected != got:
        raise ValueError(f"c3_resume_parameter_set:{sorted(expected-got)[:3]}:{sorted(got-expected)[:3]}")
    with torch.no_grad():
        for name, value in state["model_state"].items():
            by_name[name].copy_(value.to(device=by_name[name].device, dtype=by_name[name].dtype))
    optimizer.load_state_dict(state["optimizer_state"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.random.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return int(state["update"]), int(state["sample_index"]), int(state["epoch"]), list(state.get("log_tail", []))


def cmd_train(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    summary_path = run_dir / "m1_training_summary.json"
    if summary_path.exists() and not args.resume:
        raise FileExistsError(f"training_already_exists:{summary_path}")
    samples, manifest = _load_manifest(run_dir)
    if not (run_dir / "smoke.json").is_file():
        raise FileNotFoundError("c3_train_requires_smoke")
    if not (run_dir / "m0_predictions.json").is_file():
        raise FileNotFoundError("c3_train_requires_m0_baseline")
    train = [sample for sample in samples if sample.split == "train"]
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    lora, heads = _freeze_trainable(model)
    if args.resume:
        resume_path = run_dir / "checkpoint_latest.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    _set_lora_enabled(lora, False)
    import torch
    optimizer = torch.optim.AdamW(
        [
            {"name": "lora", "params": [p for _, p in lora], "lr": 0.0},
            {"name": "head", "params": [p for _, p in heads], "lr": 0.0},
        ],
        weight_decay=args.weight_decay,
    )
    total_updates = int(args.updates)
    warmup = max(1, math.ceil(total_updates * 0.05))
    start_update = 0
    sample_index = 0
    epoch = 0
    logs: list[dict[str, Any]] = []
    if args.resume:
        start_update, sample_index, epoch, logs = _restore_training_checkpoint(resume_path, model, optimizer)
    if start_update >= total_updates:
        raise ValueError("c3_resume_already_complete")
    if sample_index >= len(train):
        sample_index = 0
        epoch += 1
    exposures = {sample.root_id: 0 for sample in train}
    start_wall = time.perf_counter()
    max_allocated = 0.0
    max_reserved = 0.0
    current_order = deterministic_order(train, args.seed + epoch)
    # Restore index against the current deterministic epoch order. A checkpoint
    # stores the offset; the order itself is regenerated from the epoch seed.
    root_order = current_order
    while start_update < total_updates:
        update = start_update + 1
        if sample_index + args.accumulation > len(root_order):
            epoch += 1
            root_order = deterministic_order(train, args.seed + epoch)
            sample_index = 0
        window = root_order[sample_index : sample_index + args.accumulation]
        if not window:
            continue
        schedule = _set_schedule(optimizer, update, total=total_updates, head_lr=args.head_lr, lora_lr=args.lora_lr, warmup=warmup, lora=lora)
        optimizer.zero_grad(set_to_none=True)
        route_values: list[float] = []
        speed_values: list[float] = []
        total_values: list[float] = []
        route_valid_values: list[int] = []
        speed_valid_values: list[int] = []
        window_start = time.perf_counter()
        for sample in window:
            example = _make_example(runtime, model, sample, device)
            model.train()
            model.vision_model.eval()
            model.wp_encoder.eval()
            predictions, _, _ = _forward_driving(model, example, train=True)
            total, route_loss, speed_loss = _losses(predictions, example)
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f"c3_non_finite_loss:update={update}:root={sample.root_id}")
            total.backward()
            exposures[sample.root_id] += 1
            route_values.append(float(route_loss.detach().item()))
            speed_values.append(float(speed_loss.detach().item()))
            total_values.append(float(total.detach().item()))
            route_valid_values.append(int(sum(sample.route_mask)))
            speed_valid_values.append(int(sum(sample.speed_mask)))
            del example, predictions, total, route_loss, speed_loss
        # Normalize by the actual window size (the final short window is not
        # divided by the configured accumulation value).
        divisor = float(len(window))
        for _, parameter in lora + heads:
            if parameter.grad is not None:
                parameter.grad.div_(divisor)
        trainable = [p for _, p in lora + heads if p.requires_grad]
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()) for parameter in trainable):
            raise FloatingPointError(f"c3_non_finite_gradient:update={update}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()) if trainable else 0.0
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"c3_non_finite_gradient_norm:update={update}")
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            max_allocated = max(max_allocated, float(torch.cuda.max_memory_allocated() / 2**30))
            max_reserved = max(max_reserved, float(torch.cuda.max_memory_reserved() / 2**30))
        sample_index += len(window)
        elapsed = time.perf_counter() - window_start
        entry = {
            "update": update,
            "epoch": epoch,
            "root_ids": [sample.root_id for sample in window],
            "loss": float(np.mean(total_values)),
            "route_loss": float(np.mean(route_values)),
            "speed_loss": float(np.mean(speed_values)),
            "route_valid_points": int(sum(route_valid_values)),
            "speed_valid_points": int(sum(speed_valid_values)),
            "grad_norm_before_clip": grad_norm,
            "head_lr": schedule["head_lr"],
            "lora_lr": schedule["lora_lr"],
            "window_size": len(window),
            "window_time_s": elapsed,
            "wall_time_s": time.perf_counter() - start_wall,
        }
        logs.append(entry)
        with (run_dir / "training.jsonl").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
        if update % 10 == 0 or update == total_updates:
            _save_training_checkpoint(run_dir / "checkpoint_latest.pt", model, optimizer, update=update, sample_index=sample_index, epoch=epoch, log=logs)
        start_update = update
        if time.perf_counter() - start_wall > float(args.max_hours) * 3600.0:
            break
    completed = start_update >= total_updates
    # Persist the last finite update even when the loop stops between the
    # regular ten-update checkpoints (for example at the wall-time budget).
    # This keeps ``--resume`` exact with respect to model, optimizer, RNG and
    # sample position.
    _save_training_checkpoint(
        run_dir / "checkpoint_latest.pt",
        model,
        optimizer,
        update=start_update,
        sample_index=sample_index,
        epoch=epoch,
        log=logs,
    )
    adapter_path = run_dir / "m1_adapter.pt"
    adapter_meta = _save_trainable_state(model, adapter_path)
    payload = {
        "schema_version": "safedrive.c3.m1_training.v1",
        "status": "M1_COMPLETED" if completed else "M1_PARTIAL",
        "run_id": str(args.run_id),
        "model_load": vars(report),
        "target_updates": total_updates,
        "actual_updates": start_update,
        "warmup_head_updates": warmup,
        "lora_unfreeze_update": warmup + 1,
        "seed": int(args.seed),
        "train_root_count": len(train),
        "root_exposures": exposures,
        "root_exposure_min": min(exposures.values()) if exposures else 0,
        "root_exposure_max": max(exposures.values()) if exposures else 0,
        "adapter": adapter_meta,
        "checkpoint_latest": str(run_dir / "checkpoint_latest.pt"),
        "logs": logs,
        "resources": {
            "wall_time_s": time.perf_counter() - start_wall,
            "peak_allocated_gib": max_allocated if device.startswith("cuda") else None,
            "peak_reserved_gib": max_reserved if device.startswith("cuda") else None,
            "device": device,
        },
        "config_sha256": _sha_object(_read_json(run_dir / "run_config.json")),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    payload["content_sha256"] = _sha_object(payload)
    _write_json(summary_path, payload, overwrite=args.resume and summary_path.exists())
    resources = {
        "schema_version": "safedrive.c3.resource_ledger.v1",
        "run_id": str(args.run_id),
        "phase": "M1",
        "status": payload["status"],
        "device": device,
        "wall_time_s": payload["resources"]["wall_time_s"],
        "peak_allocated_gib": payload["resources"]["peak_allocated_gib"],
        "peak_reserved_gib": payload["resources"]["peak_reserved_gib"],
        "gpu_peak_limit_gib": 14.5,
        "max_hours": float(args.max_hours),
        "target_updates": total_updates,
        "actual_updates": start_update,
        "train_samples_per_update": int(args.accumulation),
        "adapter_bytes": int(adapter_path.stat().st_size),
        "checkpoint_bytes": int((run_dir / "checkpoint_latest.pt").stat().st_size),
        "gpu_optimization_budget_hours": 10.0,
        "content_sha256": payload["content_sha256"],
    }
    resources["ledger_sha256"] = _sha_object(resources)
    _write_json(run_dir / "resource-ledger.json", resources, overwrite=args.resume and (run_dir / "resource-ledger.json").exists())
    print(json.dumps({"status": payload["status"], "actual_updates": start_update, "target_updates": total_updates, "peak_allocated_gib": max_allocated}, indent=2))
    return 0 if completed else 2


def cmd_evaluate(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    output = run_dir / "m1_predictions.json"
    if output.exists():
        raise FileExistsError(f"evaluation_already_exists:{output}")
    samples, _ = _load_manifest(run_dir)
    validation = [sample for sample in samples if sample.split == "validation"]
    training = _read_json(run_dir / "m1_training_summary.json")
    if training.get("status") != "M1_COMPLETED":
        raise ValueError(f"c3_evaluation_requires_completed_m1:{training.get('status')}")
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    lora, heads = _freeze_trainable(model)
    _load_trainable_state(model, run_dir / "m1_adapter.pt")
    m1_rows, _ = _evaluate_model(runtime, model, validation, device=device, model_name="m1", repeat=False)
    if len(m1_rows) != 53:
        raise ValueError(f"c3_m1_validation_count:{len(m1_rows)}")
    m0 = _read_json(run_dir / "m0_predictions.json")
    m0_rows = list(m0.get("rows", ()))
    by_root = {row["root_id"]: row for row in m0_rows}
    if set(by_root) != {sample.root_id for sample in validation}:
        raise ValueError("c3_m0_m1_root_set_mismatch")
    metric_rows = [{**row, "model": "m0"} for row in m0_rows] + m1_rows
    aggregate = aggregate_metrics(metric_rows, bootstrap_rounds=1000, bootstrap_seed=71)
    aggregate["eps_ADE_m"] = float(m0["eps_ADE_m"])
    aggregate["primary_target"] = {
        "absolute_threshold_m": 0.01,
        "relative_target_percent": 2.0,
        "relative_stretch_percent": 3.0,
        "observed_delta_m0_minus_m1_m": aggregate["metrics"]["route_ade_m"]["delta_m0_minus_m1"],
        "observed_above_eps": (
            aggregate["metrics"]["route_ade_m"]["delta_m0_minus_m1"] is not None
            and aggregate["metrics"]["route_ade_m"]["delta_m0_minus_m1"] > float(m0["eps_ADE_m"])
        ),
    }
    payload = {
        "schema_version": "safedrive.c3.m1_predictions.v1",
        "status": "M1_MEASURED",
        "model": vars(report),
        "rows": m1_rows,
        "row_count": len(m1_rows),
        "content_sha256": _sha_object(m1_rows),
    }
    _write_json(output, payload)
    aggregate["m0_content_sha256"] = m0.get("content_sha256")
    aggregate["m1_content_sha256"] = payload["content_sha256"]
    aggregate["content_sha256"] = _sha_object(aggregate)
    _write_json(run_dir / "metrics.json", aggregate)
    report_payload = {
        "schema_version": "safedrive.c3.evaluation_report.v1",
        "status": "EVALUATION_MEASURED",
        "run_id": str(args.run_id),
        "validation_root_count": len(validation),
        "m0_prediction_path": str(run_dir / "m0_predictions.json"),
        "m1_prediction_path": str(output),
        "metrics_path": str(run_dir / "metrics.json"),
        "metrics_sha256": aggregate["content_sha256"],
        "metrics": aggregate["metrics"],
        "eps_ADE_m": aggregate["eps_ADE_m"],
        "p_speed": "N/A_without_trusted_time_speed_ground_truth",
        "reproduction_command": f"python scripts/h6_cora_sft.py evaluate --run-id {args.run_id} --device cuda",
    }
    report_payload["content_sha256"] = _sha_object(report_payload)
    _write_json(run_dir / "evaluation-report.json", report_payload)
    print(json.dumps({"status": payload["status"], "route_ade": aggregate["metrics"]["route_ade_m"], "speed_wp_ade": aggregate["metrics"]["speed_wp_ade_m"]}, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    required = [
        "run_config.json",
        "manifest.json",
        "audit.json",
        "smoke.json",
        "m0_predictions.json",
        "m1_training_summary.json",
        "m1_adapter.pt",
        "checkpoint_latest.pt",
        "training.jsonl",
        "m1_predictions.json",
        "metrics-spec.json",
        "metrics.json",
        "resource-ledger.json",
        "evaluation-report.json",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"c3_verify_missing:{missing}")
    samples, manifest = _load_manifest(run_dir)
    audit = _read_json(run_dir / "audit.json")
    smoke = _read_json(run_dir / "smoke.json")
    m0 = _read_json(run_dir / "m0_predictions.json")
    train = _read_json(run_dir / "m1_training_summary.json")
    m1 = _read_json(run_dir / "m1_predictions.json")
    metrics = _read_json(run_dir / "metrics.json")
    resources = _read_json(run_dir / "resource-ledger.json")
    evaluation = _read_json(run_dir / "evaluation-report.json")
    errors: list[str] = []
    if audit.get("status") != "AUDIT_PASSED":
        errors.append("audit_not_passed")
    if smoke.get("status") != "SMOKE_PASSED":
        errors.append("smoke_not_passed")
    if train.get("status") != "M1_COMPLETED":
        errors.append("m1_not_completed")
    if len(m0.get("rows", ())) != 53 or len(m1.get("rows", ())) != 53:
        errors.append("validation_rows_not_53")
    if {row.get("root_id") for row in m0.get("rows", ())} != {row.get("root_id") for row in m1.get("rows", ())}:
        errors.append("m0_m1_root_sets_differ")
    if manifest.get("world_labels_in_sft") is not False or manifest.get("future_labels_in_input") is not False:
        errors.append("leakage_flags")
    if train.get("actual_updates") != train.get("target_updates"):
        errors.append("update_budget_not_reached")
    if float(train.get("resources", {}).get("peak_allocated_gib") or 0.0) > 14.5:
        errors.append("gpu_peak_over_budget")
    if not metrics.get("metrics", {}).get("route_ade_m"):
        errors.append("route_metric_missing")
    if resources.get("status") != train.get("status"):
        errors.append("resource_ledger_status_mismatch")
    if resources.get("actual_updates") != train.get("actual_updates"):
        errors.append("resource_ledger_update_mismatch")
    if evaluation.get("status") != "EVALUATION_MEASURED" or evaluation.get("validation_root_count") != 53:
        errors.append("evaluation_report_incomplete")
    verification = {
        "schema_version": "safedrive.c3.verification.v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "run_id": str(args.run_id),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "m0_content_sha256": m0.get("content_sha256"),
        "m1_content_sha256": m1.get("content_sha256"),
        "metrics_content_sha256": metrics.get("content_sha256"),
        "m1_training_content_sha256": train.get("content_sha256"),
        "resource_ledger_sha256": resources.get("ledger_sha256"),
        "evaluation_report_sha256": evaluation.get("content_sha256"),
        "engineering_complete": not errors,
        "algorithm_gain_status": "MEASURED" if not errors else "NOT_VERIFIED",
    }
    verification["content_sha256"] = _sha_object(verification)
    _write_json(run_dir / "verify.json", verification)
    stage = {
        "schema_version": "safedrive.c3.stage_summary.v1",
        "stage": "C3",
        "status": verification["status"],
        "engineering_status": "COMPLETED" if not errors else "INCOMPLETE",
        "algorithm_status": verification["algorithm_gain_status"],
        "run_id": str(args.run_id),
        "manifest": str(run_dir / "manifest.json"),
        "m0": str(run_dir / "m0_predictions.json"),
        "m1": str(run_dir / "m1_predictions.json"),
        "metrics": str(run_dir / "metrics.json"),
        "resource_ledger": str(run_dir / "resource-ledger.json"),
        "evaluation_report": str(run_dir / "evaluation-report.json"),
        "verification": str(run_dir / "verify.json"),
        "reproduction_commands": _read_json(run_dir / "run_config.json").get("reproduction_commands", []),
        "source_identity": {
            "code_sha256": _read_json(run_dir / "run_config.json").get("code_sha256"),
            "git": _read_json(run_dir / "run_config.json").get("git", {}),
            "release_id": _read_json(run_dir / "run_config.json").get("release_id"),
            "model": _read_json(run_dir / "run_config.json").get("model", {}),
        },
        "next_stage": "C4" if not errors else None,
    }
    stage["content_sha256"] = _sha_object(stage)
    _write_json(run_dir / "stage-summary.json", stage)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "baseline", "train", "evaluate", "verify"))
    parser.add_argument("--run-id", default="c3-vla-sft-20260909")
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--hydra-config", type=Path, default=DEFAULT_HYDRA)
    parser.add_argument("--internvl-root", type=Path, default=DEFAULT_INTERNVL)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"audit": cmd_audit, "smoke": cmd_smoke, "baseline": cmd_baseline, "train": cmd_train, "evaluate": cmd_evaluate, "verify": cmd_verify}
    try:
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001 - CLI emits a clear failure and traceback for evidence
        print(f"C3 {args.command} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
