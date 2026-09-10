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
    native_route_target_with_report,
    sample_polyline,
)


DEFAULT_RELEASE = REPO / "generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1"
DEFAULT_CKPT = REPO / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
DEFAULT_HYDRA = REPO / "models/simlingo/simlingo/.hydra/config.yaml"
DEFAULT_INTERNVL = REPO / "models/InternVL2-1B"
DEFAULT_OUTPUT = REPO / "generated/h6/cora"
RUN_SCHEMA_VERSION = "safedrive.c3.repair.v2"
OPTIMIZATION_BUDGET_HOURS = 10.0
GPU_PEAK_LIMIT_GIB = 14.5


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
    if not str(args.run_id).startswith("c3-repair-"):
        raise ValueError("c3_repair_requires_unique_repair_run_id")
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _config_payload(args: argparse.Namespace, *, model_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "safedrive.c3.sft_run_config.v2",
        "repair_schema": RUN_SCHEMA_VERSION,
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
        "route_target_source": "native_expert_reference_path_projected",
        "speed_target_source": "expert_canonical_proposal",
        "warmup_fraction": 0.05,
        "gradient_clip": 1.0,
        "precision_policy": "bf16_preferred_verified_by_real_batch",
        "device": str(args.device),
        "max_hours": float(args.max_hours),
        "bootstrap_rounds": 1000,
        "bootstrap_seed": 71,
        "future_labels_in_input": False,
        "world_labels_in_sft": False,
        "teacher_contract": "native_expert_reference_path_plus_canonical_speed_proposal",
        "validation_input_freeze": "manifest_sample_order_and_masks_are_immutable_before_prediction",
        "m0_m1_start": "original_m0_checkpoint_each_model_load; no_m1_resume_for_m2",
        "resource_accounting": "historical_artifact_intervals_plus_current_phase_events",
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
    payload["content_sha256"] = _sha_object(payload)
    return payload


def _config_digest(config: Mapping[str, Any]) -> str:
    """Validate and return the non-circular run-config digest."""

    digest = str(config.get("content_sha256", ""))
    check = dict(config)
    check.pop("content_sha256", None)
    if not digest or digest != _sha_object(check):
        raise ValueError("c3_run_config_digest_mismatch")
    return digest


def _validate_run_config(run_dir: Path, args: argparse.Namespace, *, require_model_device: bool = True) -> dict[str, Any]:
    """Fail closed when a later phase is pointed at a different run contract."""

    config = _read_json(run_dir / "run_config.json")
    _config_digest(config)
    if config.get("schema_version") != "safedrive.c3.sft_run_config.v2" or config.get("repair_schema") != RUN_SCHEMA_VERSION:
        raise ValueError("c3_run_config_schema")
    if str(config.get("run_id")) != str(args.run_id):
        raise ValueError("c3_run_config_run_id_mismatch")
    if require_model_device and str(args.device).lower() != str(config.get("device", "")).lower():
        raise ValueError("c3_run_config_device_mismatch")
    if str(config.get("code_path")) != str(Path(__file__).resolve()) or str(config.get("code_sha256")) != sha256_file(Path(__file__).resolve()):
        raise ValueError("c3_run_config_code_identity_mismatch")
    return config


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
        if _is_lora_parameter(name):
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


def _is_lora_parameter(name: str) -> bool:
    return name.startswith("language_model.") and "lora_" in name


def _is_driving_head_parameter(name: str) -> bool:
    return name.startswith("adaptors.driving.route_head.") or name.startswith("adaptors.driving.speed_wps_head.")


def _is_update_parameter(name: str) -> bool:
    return _is_lora_parameter(name) or _is_driving_head_parameter(name)


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
    canonical_valid = bool(
        route.shape == (ROUTE_STEPS, 2)
        and speed.shape == (SPEED_STEPS, 2)
        and np.isfinite(route).all()
        and np.isfinite(speed).all()
    )
    record = root_metrics({"route": route, "speed": speed, "canonical_output_valid": canonical_valid}, sample)
    record.update(
        {
            "model": model_name,
            "route": route.tolist(),
            "speed": speed.tolist(),
            "latency_s": float(latency_s),
        }
    )
    return record


def _failed_prediction_record(sample: SFTSample, latency_s: float, *, model_name: str, reason: str) -> dict[str, Any]:
    """Keep a failed prediction in the denominator with its own reason."""

    record = root_metrics(
        {
            "route": np.empty((0, 2), dtype=np.float64),
            "speed": np.empty((0, 2), dtype=np.float64),
            "canonical_output_valid": False,
        },
        sample,
    )
    record.update(
        {
            "model": model_name,
            "route": [],
            "speed": [],
            "latency_s": float(latency_s),
            "prediction_exception": str(reason),
        }
    )
    record["failure_reasons"] = sorted(set(record.get("failure_reasons", [])) | {f"prediction_exception:{reason}"})
    return record


def _write_prediction_failures(
    run_dir: Path,
    *,
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Persist prediction exceptions separately while retaining full rows."""

    failures = [
        dict(row)
        for row in rows
        if row.get("prediction_exception")
        or any(str(reason).startswith(("prediction_exception:", "repeat_exception:", "repeat_output:")) for reason in row.get("failure_reasons", ()))
    ]
    payload: dict[str, Any] = {
        "schema_version": "safedrive.c3.prediction_failures.v1",
        "status": "FAILURES_RECORDED" if failures else "NO_FAILURES",
        "phase": str(phase),
        "manifest_sha256": manifest_sha256,
        "source_predictions_sha256": source_sha256,
        "row_count": len(failures),
        "rows": failures,
    }
    payload["content_sha256"] = _sha_object(payload)
    _write_json(run_dir / f"{phase}_prediction_failures.json", payload)
    return payload


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
        start = time.perf_counter()
        try:
            example = _make_example(runtime, model, sample, device)
        except Exception as exc:  # noqa: BLE001 - preserve input/IO failures too
            rows.append(_failed_prediction_record(sample, time.perf_counter() - start, model_name=model_name, reason=f"{type(exc).__name__}:{exc}"))
            continue
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        try:
            predictions, _, _ = _forward_driving(model, example, train=False)
        except Exception as exc:  # noqa: BLE001 - one bad root cannot hide the denominator
            rows.append(_failed_prediction_record(sample, time.perf_counter() - start, model_name=model_name, reason=f"{type(exc).__name__}:{exc}"))
            del example
            continue
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        try:
            rows.append(_prediction_record(predictions, sample, elapsed, model_name=model_name))
        except Exception as exc:  # noqa: BLE001 - malformed head output is a recorded failure
            rows.append(_failed_prediction_record(sample, elapsed, model_name=model_name, reason=f"output:{type(exc).__name__}:{exc}"))
            del example, predictions
            continue
        if repeat:
            try:
                predictions_2, _, _ = _forward_driving(model, example, train=False)
            except Exception as exc:  # noqa: BLE001 - repeat failure is part of M0 epsilon evidence
                rows[-1].setdefault("failure_reasons", []).append(f"repeat_exception:{type(exc).__name__}:{exc}")
                del example, predictions
                continue
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            try:
                repeats.append(_prediction_max_abs_diff(predictions, predictions_2))
            except Exception as exc:  # noqa: BLE001 - malformed repeat is a failed repeat, not missing truth
                rows[-1].setdefault("failure_reasons", []).append(f"repeat_output:{type(exc).__name__}:{exc}")
        del example, predictions
    return rows, repeats


def _save_trainable_state(model: Any, path: Path) -> dict[str, Any]:
    import torch

    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    fingerprint = _fingerprint_tensor_mapping(state)
    return {
        "keys": sorted(state),
        "parameter_count": int(sum(v.numel() for v in state.values())),
        "path": str(path),
        "sha256": sha256_file(path),
        "model_digest": fingerprint["model_digest"],
    }


def _save_update_state(model: Any, path: Path) -> dict[str, Any]:
    """Persist the complete whitelist snapshot used for formal change proof."""

    import torch

    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if _is_update_parameter(name)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return {"keys": sorted(state), "parameter_count": int(sum(v.numel() for v in state.values())), "path": str(path)}


def _load_update_state(path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("c3_initial_update_state_not_mapping")
    return state


def _parameter_change_max_abs(model: Any, initial_state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    by_name = dict(model.named_parameters())
    missing = sorted(set(initial_state) - set(by_name))
    if missing:
        raise ValueError(f"c3_initial_update_state_unknown:{missing[:5]}")
    for name, before in initial_state.items():
        current = by_name[name].detach().float().cpu()
        prior = before.detach().float().cpu()
        if current.shape != prior.shape:
            raise ValueError(f"c3_initial_update_state_shape:{name}")
        result[name] = float((current - prior).abs().max().item())
    return {
        "all": result,
        "lora_max_abs": max((value for name, value in result.items() if _is_lora_parameter(name)), default=0.0),
        "driving_head_max_abs": max((value for name, value in result.items() if _is_driving_head_parameter(name)), default=0.0),
        "lora_changed": any(value > 0.0 for name, value in result.items() if _is_lora_parameter(name)),
        "driving_head_changed": any(value > 0.0 for name, value in result.items() if _is_driving_head_parameter(name)),
    }


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


def _fingerprint_tensor_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic byte fingerprint for a named tensor mapping."""

    import torch

    result: dict[str, str] = {}
    aggregate = hashlib.sha256()
    element_count = 0
    for name in sorted(values):
        value = values[name]
        if not torch.is_tensor(value):
            raise TypeError(f"c3_fingerprint_non_tensor:{name}")
        tensor = value.detach().contiguous()
        raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(raw)
        digest_value = digest.hexdigest()
        result[name] = digest_value
        aggregate.update(name.encode("utf-8"))
        aggregate.update(digest_value.encode("ascii"))
        element_count += int(tensor.numel())
    return {
        "tensor_count": len(result),
        "element_count": element_count,
        "model_digest": aggregate.hexdigest(),
        "tensors": result,
    }


def _frozen_fingerprints(model: Any) -> dict[str, Any]:
    """Hash every frozen tensor, including query embeddings.

    The previous implementation sampled three tensors, so an accidental update
    to any other frozen parameter could pass smoke. Byte hashing preserves the
    stored dtype and moves one tensor at a time to CPU; it does not participate
    in the forward graph.
    """

    return _fingerprint_tensor_mapping(
        {
            name: parameter
            for name, parameter in model.named_parameters()
            if not _is_update_parameter(name)
        }
    )


def _trainable_fingerprints(model: Any) -> dict[str, Any]:
    """Hash all explicitly updateable LoRA and driving-head tensors."""

    return _fingerprint_tensor_mapping(
        {
            name: parameter
            for name, parameter in model.named_parameters()
            if _is_update_parameter(name)
        }
    )


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


def _c4_cost_smoke(runtime: Any, model: Any, samples: Sequence[SFTSample], *, device: str) -> dict[str, Any]:
    """Measure the four-output residual path on one complete accumulation window.

    This is a diagnostic only.  The residual is represented as
    ``head(h)-stop_gradient(head(h))`` so it starts exactly at zero while its
    nonzero, deterministic head weights expose the shared LoRA gradient path.
    No optimizer step is performed and no M2 checkpoint is produced.
    """

    import torch
    from torch import nn

    selected = list(samples[:4])
    if len(selected) < 4:
        return {"status": "INSUFFICIENT_WINDOW", "window_size": len(selected)}
    model.train()
    model.vision_model.eval()
    model.wp_encoder.eval()
    start = time.perf_counter()
    hidden_values: list[Any] = []
    targets: list[Any] = []
    for sample in selected:
        example = _make_example(runtime, model, sample, device)
        _, features, _ = _forward_driving(model, example, train=True)
        hidden_values.append(features.mean(dim=1).float())
        heads = dict(sample.outcome_label_heads)
        targets.append(
            torch.tensor(
                [
                    float(bool(heads.get("collision", {}).get("value", False))),
                    float(bool(heads.get("offroad", {}).get("value", False))),
                    float(bool(heads.get("mrm", {}).get("value", False))),
                    float(heads.get("progress", {}).get("value", 0.0)),
                ],
                dtype=torch.float32,
                device=device,
            )
        )
        del example
    hidden = torch.cat(hidden_values, dim=0)
    target = torch.stack(targets, dim=0)
    residual_head = nn.Linear(hidden.shape[-1], 4).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(71)
    with torch.no_grad():
        residual_head.weight.normal_(mean=0.0, std=0.01, generator=generator)
        residual_head.bias.zero_()
    raw = residual_head(hidden)
    residual = raw - raw.detach()
    aux_loss = torch.nn.functional.smooth_l1_loss(residual, target)
    lora = [(name, parameter) for name, parameter in model.named_parameters() if _is_lora_parameter(name)]
    shared_grads = torch.autograd.grad(aux_loss, [parameter for _, parameter in lora], retain_graph=True, allow_unused=True)
    head_grads = torch.autograd.grad(aux_loss, tuple(residual_head.parameters()), retain_graph=True, allow_unused=True)
    shared_norm = float(torch.linalg.vector_norm(torch.cat([g.float().reshape(-1) for g in shared_grads if g is not None])).item()) if any(g is not None for g in shared_grads) else 0.0
    head_norm = float(torch.linalg.vector_norm(torch.cat([g.float().reshape(-1) for g in head_grads if g is not None])).item()) if any(g is not None for g in head_grads) else 0.0
    def _group_norm(values: Sequence[Any]) -> float:
        finite = [value.float().reshape(-1) for value in values if value is not None]
        return float(torch.linalg.vector_norm(torch.cat(finite)).item()) if finite else 0.0

    shared_clip = _group_norm(shared_grads)
    head_clip = _group_norm(head_grads)
    def _clip_scale(norm: float, max_norm: float = 1.0) -> float:
        return float(min(1.0, max_norm / (norm + 1e-12))) if norm > 0.0 else 1.0

    shared_scale = _clip_scale(shared_clip)
    head_scale = _clip_scale(head_clip)
    elapsed = time.perf_counter() - start
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {
        "status": "MEASURED",
        "window_size": len(selected),
        "candidate_output_dim": 4,
        "zero_residual_max_abs": float(residual.detach().abs().max().item()),
        "aux_loss": float(aux_loss.detach().item()),
        "shared_lora_grad_norm": shared_norm,
        "shared_lora_grad_nonzero": bool(shared_norm > 0.0),
        "residual_head_grad_norm": head_norm,
        "shared_clip_norm_before": shared_clip,
        "residual_head_clip_norm_before": head_clip,
        "clip_max_norm": 1.0,
        "shared_clip_scale": shared_scale,
        "residual_head_clip_scale": head_scale,
        "shared_clip_norm_after": shared_clip * shared_scale,
        "residual_head_clip_norm_after": head_clip * head_scale,
        "wall_time_s": elapsed,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30) if device.startswith("cuda") else None,
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30) if device.startswith("cuda") else None,
        "label_source": "independent_c2_outcome_heads; diagnostic_only",
    }


def _diagnostic_baselines(train: Sequence[SFTSample], validation: Sequence[SFTSample]) -> dict[str, Any]:
    """Build fixed diagnostics without fitting on validation labels.

    ``train_mean`` exposes a label-template baseline.  ``navigation_geometry``
    uses only the two deployment navigation points and intentionally leaves
    unsupported route distances masked.  Neither is used for checkpoint or
    hyper-parameter selection.
    """

    train_routes = np.asarray([sample.route_target for sample in train], dtype=np.float64)
    train_speeds = np.asarray([sample.speed_target for sample in train], dtype=np.float64)
    train_route_masks = np.asarray([sample.route_mask for sample in train], dtype=bool)
    train_speed_masks = np.asarray([sample.speed_mask for sample in train], dtype=bool)
    route_mean = np.zeros((ROUTE_STEPS, 2), dtype=np.float64)
    speed_mean = np.zeros((SPEED_STEPS, 2), dtype=np.float64)
    for index in range(ROUTE_STEPS):
        valid = train_route_masks[:, index]
        if valid.any():
            route_mean[index] = train_routes[valid, index].mean(axis=0)
    for index in range(SPEED_STEPS):
        valid = train_speed_masks[:, index]
        if valid.any():
            speed_mean[index] = train_speeds[valid, index].mean(axis=0)
    rows: dict[str, list[dict[str, Any]]] = {"train_mean": [], "navigation_geometry": []}
    for sample in validation:
        mean_row = root_metrics(
            {"route": route_mean, "speed": speed_mean, "canonical_output_valid": True}, sample
        )
        mean_row.update({"model": "train_mean", "route": route_mean.tolist(), "speed": speed_mean.tolist(), "latency_s": 0.0})
        rows["train_mean"].append(mean_row)
        nodes = np.asarray([[0.0, 0.0], sample.target_ego_1, sample.target_ego_2], dtype=np.float64)
        geometry, geometry_mask = sample_polyline(nodes, np.arange(ROUTE_STEPS, dtype=np.float64))
        geometry_speed = np.zeros((SPEED_STEPS, 2), dtype=np.float64)
        # The navigation geometry only supports the segment between the two
        # deployment target points.  Score it with a derived diagnostic mask so
        # unsupported distances cannot be mistaken for a zero-valued route.
        geometry_sample = SFTSample(
            **{
                **sample.to_dict(),
                "route_mask": tuple(bool(a and b) for a, b in zip(sample.route_mask, geometry_mask)),
                "route_missing_reasons": tuple(sample.route_missing_reasons)
                + (f"navigation_geometry_support:{int(geometry_mask.sum())}/{ROUTE_STEPS}",),
            }
        )
        geometry_row = root_metrics(
            {"route": geometry, "speed": geometry_speed, "canonical_output_valid": True}, geometry_sample
        )
        geometry_row.update(
            {
                "model": "navigation_geometry",
                "route": geometry.tolist(),
                "speed": geometry_speed.tolist(),
                "geometry_route_mask": geometry_mask.tolist(),
                "latency_s": 0.0,
            }
        )
        rows["navigation_geometry"].append(geometry_row)
    return {
        "schema_version": "safedrive.c3.diagnostic_baselines.v2",
        "fit_split": "train",
        "selection_allowed": False,
        "rows": rows,
        "row_counts": {key: len(value) for key, value in rows.items()},
        "train_mean_source": "pointwise_mean_of_train_targets_under_train_masks",
        "navigation_geometry_source": "deployment_target_points_only; no_validation_labels",
        "content_sha256": _sha_object(rows),
    }


def _historical_resource_events(*, exclude_run_id: str | None = None) -> list[dict[str, Any]]:
    """Recover conservative timing for superseded C3 attempts.

    The old runs did not write phase timings for baseline/evaluation.  Their
    artifact mtime interval is a conservative, auditable upper bound for the
    complete attempt; an empty ``-r1`` directory is retained as a bounded failed
    attempt under the declared four-hour command budget.
    """

    events: list[dict[str, Any]] = []
    for path in sorted(DEFAULT_OUTPUT.glob("c3-vla-sft-20260909*")):
        audit = path / "audit.json"
        verify = path / "verify.json"
        if audit.is_file() and verify.is_file():
            start = int(audit.stat().st_mtime)
            end = int(verify.stat().st_mtime)
            summary_path = path / "m1_training_summary.json"
            summary = _read_json(summary_path) if summary_path.is_file() else {}
            known_optimization = float(summary.get("resources", {}).get("wall_time_s", 0.0) or 0.0)
            events.append(
                {
                    "run_id": path.name,
                    "phase": "historical_attempt_interval",
                    "status": "SUPERSEDED_INVALID_SUPERVISION",
                    "wall_time_s_upper_bound": float(max(0, end - start) + 1),
                    "optimization_wall_time_s": known_optimization,
                    # The interval includes idle time between commands and is
                    # not an optimization measurement.  The training summary
                    # is the authoritative upper bound for optimizer time.
                    "optimization_wall_time_s_upper_bound": float(known_optimization),
                    "peak_allocated_gib": summary.get("resources", {}).get("peak_allocated_gib"),
                    "measurement": "artifact_mtime_interval_1s_resolution",
                    "source": str(path),
                }
            )
        elif path.name.endswith("-r1"):
            events.append(
                {
                    "run_id": path.name,
                    "phase": "historical_failed_attempt",
                    "status": "FAILED_NO_ARTIFACT",
                    "wall_time_s_upper_bound": 4.0 * 3600.0,
                    "optimization_wall_time_s": 0.0,
                    "optimization_wall_time_s_upper_bound": 4.0 * 3600.0,
                    "peak_allocated_gib": None,
                    "measurement": "declared_max_hours_upper_bound",
                    "source": str(path),
                }
            )
    # Earlier repair attempts are evidence too.  A complete artifact interval
    # gives a conservative upper bound; an incomplete attempt is charged the
    # declared four-hour command bound instead of being silently treated as 0.
    for path in sorted(DEFAULT_OUTPUT.glob("c3-repair-*")):
        if not path.is_dir():
            continue
        if exclude_run_id is not None and path.name == str(exclude_run_id):
            continue
        audit = path / "audit.json"
        if not audit.is_file():
            continue
        start = int(audit.stat().st_mtime)
        end = int(max(item.stat().st_mtime for item in path.iterdir() if item.is_file()))
        interval = float(max(0, end - start) + 1)
        summary_path = path / "m1_training_summary.json"
        verify_path = path / "verify.json"
        if summary_path.is_file() and verify_path.is_file():
            summary = _read_json(summary_path)
            known_optimization = float(summary.get("resources", {}).get("wall_time_s", 0.0) or 0.0)
            events.append(
                {
                    "run_id": path.name,
                    "phase": "historical_repair_attempt",
                    "status": "SUPERSEDED_REPAIR_ATTEMPT",
                    "wall_time_s_upper_bound": interval,
                    "optimization_wall_time_s": known_optimization,
                    "optimization_wall_time_s_upper_bound": known_optimization,
                    "peak_allocated_gib": summary.get("resources", {}).get("peak_allocated_gib"),
                    "measurement": "artifact_mtime_interval_1s_resolution",
                    "source": str(path),
                }
            )
        else:
            events.append(
                {
                    "run_id": path.name,
                    "phase": "historical_repair_failed_attempt",
                    "status": "FAILED_OR_UNVERIFIED_REPAIR_ATTEMPT",
                    "wall_time_s_upper_bound": max(interval, 4.0 * 3600.0),
                    "optimization_wall_time_s": 0.0,
                    "optimization_wall_time_s_upper_bound": 4.0 * 3600.0,
                    "peak_allocated_gib": None,
                    "measurement": "incomplete_run_declared_four_hour_upper_bound",
                    "source": str(path),
                }
            )
    return events


def _independent_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    samples: Sequence[SFTSample],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Recompute per-root metrics from saved arrays, never from saved numbers."""

    by_root: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        root_id = str(row.get("root_id", ""))
        if not root_id or root_id in by_root:
            raise ValueError(f"c3_saved_prediction_duplicate_root:{root_id}")
        by_root[root_id] = row
    expected = {sample.root_id for sample in samples}
    if set(by_root) != expected:
        raise ValueError(f"c3_saved_prediction_root_set:{len(by_root)}:{len(expected)}")
    result: list[dict[str, Any]] = []
    compare_keys = (
        "split",
        "route_valid_points",
        "speed_valid_points",
        "route_shape_valid",
        "speed_shape_valid",
        "route_finite",
        "speed_finite",
        "canonical_output_valid",
        "route_full_valid",
        "speed_full_valid",
        "route_fde_target_valid",
        "route_fde_valid",
        "route_ade_m",
        "route_fde_m",
        "speed_wp_ade_m",
        "prediction_valid",
        "failure_reasons",
        "route_missing_reasons",
        "speed_missing_reasons",
        "route_source",
        "speed_source",
        "execution_timeline_rows",
        "execution_timeline_dt_s",
    )
    for sample in samples:
        stored = by_root[sample.root_id]
        route = np.asarray(stored.get("route", []), dtype=np.float64)
        speed = np.asarray(stored.get("speed", []), dtype=np.float64)
        recomputed = root_metrics(
            {"route": route, "speed": speed, "canonical_output_valid": bool(
                route.shape == (ROUTE_STEPS, 2)
                and speed.shape == (SPEED_STEPS, 2)
                and np.isfinite(route).all()
                and np.isfinite(speed).all()
            )},
            sample,
        )
        for key in compare_keys:
            expected_value = stored.get(key)
            actual_value = recomputed.get(key)
            if isinstance(expected_value, float) or isinstance(actual_value, float):
                if expected_value is None or actual_value is None:
                    if expected_value is not None or actual_value is not None:
                        raise ValueError(f"c3_saved_metric_mismatch:{sample.root_id}:{key}")
                elif not math.isclose(float(expected_value), float(actual_value), rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(f"c3_saved_metric_mismatch:{sample.root_id}:{key}")
            elif expected_value != actual_value:
                raise ValueError(f"c3_saved_metric_mismatch:{sample.root_id}:{key}")
        if (not recomputed["route_shape_valid"] or not recomputed["speed_shape_valid"]) and not stored.get("prediction_exception"):
            raise ValueError(f"c3_failed_prediction_reason_missing:{sample.root_id}")
        recomputed.update(
            {
                "model": model_name,
                "route": route.tolist(),
                "speed": speed.tolist(),
                "latency_s": float(stored.get("latency_s", 0.0) or 0.0),
            }
        )
        result.append(recomputed)
    return result


def _update_resource_ledger(run_dir: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "resource-ledger.json"
    if path.is_file():
        ledger = _read_json(path)
    else:
        ledger = {
            "schema_version": "safedrive.c3.resource_ledger.v2",
            "run_id": run_dir.name,
            "gpu_peak_limit_gib": GPU_PEAK_LIMIT_GIB,
            "optimization_budget_hours": OPTIMIZATION_BUDGET_HOURS,
            "historical_events": _historical_resource_events(exclude_run_id=run_dir.name),
            "events": [],
        }
    ledger.setdefault("events", []).append(dict(event))
    ledger["current_event_count"] = len(ledger["events"])
    ledger["known_current_wall_time_s"] = float(
        sum(float(item.get("wall_time_s", 0.0) or 0.0) for item in ledger["events"])
    )
    ledger["known_current_optimization_time_s"] = float(
        sum(float(item.get("optimization_wall_time_s", 0.0) or 0.0) for item in ledger["events"])
    )
    ledger["historical_wall_time_upper_bound_s"] = float(
        sum(float(item.get("wall_time_s_upper_bound", 0.0) or 0.0) for item in ledger.get("historical_events", []))
    )
    ledger["historical_optimization_time_s"] = float(
        sum(float(item.get("optimization_wall_time_s", 0.0) or 0.0) for item in ledger.get("historical_events", []))
    )
    ledger["historical_optimization_time_upper_bound_s"] = float(
        sum(float(item.get("optimization_wall_time_s_upper_bound", item.get("optimization_wall_time_s", 0.0)) or 0.0) for item in ledger.get("historical_events", []))
    )
    ledger["known_current_optimization_time_upper_bound_s"] = float(
        sum(float(item.get("optimization_wall_time_s_upper_bound", item.get("optimization_wall_time_s", 0.0)) or 0.0) for item in ledger["events"])
    )
    ledger["optimization_total_hours_upper_bound"] = (
        ledger["historical_optimization_time_upper_bound_s"] + ledger["known_current_optimization_time_upper_bound_s"]
    ) / 3600.0
    peaks = [
        float(item["peak_allocated_gib"])
        for item in ledger.get("historical_events", []) + ledger["events"]
        if item.get("peak_allocated_gib") is not None
    ]
    ledger["observed_peak_allocated_gib"] = max(peaks) if peaks else None
    reserved_peaks = [
        float(item["peak_reserved_gib"])
        for item in ledger.get("historical_events", []) + ledger["events"]
        if item.get("peak_reserved_gib") is not None
    ]
    ledger["observed_peak_reserved_gib"] = max(reserved_peaks) if reserved_peaks else None
    ledger["gpu_monitoring"] = {
        "method": "torch.cuda.max_memory_allocated and max_memory_reserved reset per model phase; host nvidia-smi diagnostic",
        "sampling_period": "phase boundary plus torch peak counters; nvidia-smi queried at GPU bring-up",
        "peak_limit_gib": GPU_PEAK_LIMIT_GIB,
    }
    ledger["status"] = str(ledger["events"][-1].get("status", "")) if ledger["events"] else "NO_EVENTS"
    # Keep the two hashes non-circular. ``content_sha256`` covers the ledger
    # payload, while ``ledger_sha256`` covers that payload plus the content
    # digest. A verifier can therefore recompute both hashes without trusting
    # a self-referential value.
    digest_payload = {
        key: value for key, value in ledger.items() if key not in {"content_sha256", "ledger_sha256"}
    }
    ledger["content_sha256"] = _sha_object(digest_payload)
    ledger["ledger_sha256"] = _sha_object({**digest_payload, "content_sha256": ledger["content_sha256"]})
    _write_json(path, ledger, overwrite=True)
    return ledger


def cmd_audit(args: argparse.Namespace) -> int:
    phase_start = time.perf_counter()
    run_dir = _ensure_run_dir(args)
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"audit_already_exists:{run_dir}")
    samples, manifest = build_sft_manifest(args.release_root, splits=("train", "validation"))
    if manifest["split_counts"].get("train") != 158 or manifest["split_counts"].get("validation") != 53:
        raise ValueError(f"c3_expected_release_counts:{manifest['split_counts']}")
    model_identity = _model_identity(Path(args.checkpoint))
    if any(sample.route_support_m < ROUTE_STEPS for sample in samples):
        raise ValueError("c3_route_support_short_for_formal_manifest")
    if any(sample.route_projection_ambiguous for sample in samples):
        raise ValueError("c3_route_projection_ambiguous")
    config = _config_payload(args, model_identity=model_identity)
    _write_json(run_dir / "run_config.json", config)
    _write_json(run_dir / "manifest.json", manifest)
    audit = {
        "schema_version": "safedrive.c3.sft_audit.v2",
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
        "route_projection_distance_m": {
            "max": max(sample.route_projection_distance_m for sample in samples),
            "p95": float(np.quantile([sample.route_projection_distance_m for sample in samples], 0.95)),
        },
        "route_support_m": {
            "min": min(sample.route_support_m for sample in samples),
            "max": max(sample.route_support_m for sample in samples),
        },
        "route_reference_revisions": sorted({sample.route_reference_revision for sample in samples}),
        "execution_quality_counts": {
            quality: sum(sample.execution_quality == quality for sample in samples)
            for quality in sorted({sample.execution_quality for sample in samples})
        },
        "execution_failure_reason_counts": {
            reason: sum(reason in sample.execution_failure_reasons for sample in samples)
            for reason in sorted({reason for sample in samples for reason in sample.execution_failure_reasons})
        },
        "execution_timeline_alignment_counts": {
            alignment: sum(sample.execution_timeline_alignment == alignment for sample in samples)
            for alignment in sorted({sample.execution_timeline_alignment for sample in samples})
        },
        "anchor_speed_input": {
            "source": "observable_snapshot.ego_v",
            "missing_count": sum(not math.isfinite(sample.ego_speed_mps) for sample in samples),
            "zero_count": sum(sample.ego_speed_mps == 0.0 for sample in samples),
            "unique_values": sorted({sample.ego_speed_mps for sample in samples}),
            "history_disagreement_count": sum(
                sample.anchor_history_speed_disagreement_mps is not None
                and abs(sample.anchor_history_speed_disagreement_mps) > 1e-8
                for sample in samples
            ),
        },
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
    _update_resource_ledger(
        run_dir,
        {
            "phase": "audit",
            "status": audit["status"],
            "wall_time_s": time.perf_counter() - phase_start,
            "optimization_wall_time_s": 0.0,
            "device": str(args.device),
            "peak_allocated_gib": None,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    run_dir = _ensure_run_dir(args)
    output = run_dir / "smoke.json"
    if output.exists():
        raise FileExistsError(f"smoke_already_exists:{output}")
    _validate_run_config(run_dir, args)
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
    trainable_before_fingerprint = _trainable_fingerprints(model)
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
    trainable_after_fingerprint = _trainable_fingerprints(model)
    lora_changed_any = any(changed[name] > 0.0 for name, _ in lora)
    head_changed_any = any(changed[name] > 0.0 for name, _ in heads)
    c4_cost = _c4_cost_smoke(runtime, model, train, device=device)
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
        "schema_version": "safedrive.c3.sft_smoke.v2",
        "status": "SMOKE_PASSED" if finite_grad and nonzero_grad > 0 and lora_changed_any and head_changed_any and frozen_before == frozen_after and reload_diff <= 1e-5 and c4_cost.get("status") == "MEASURED" and c4_cost.get("zero_residual_max_abs") == 0.0 and c4_cost.get("shared_lora_grad_nonzero") else "SMOKE_FAILED",
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
        "lora_changed": lora_changed_any,
        "driving_head_changed": head_changed_any,
        "trainable_fingerprint_before": trainable_before_fingerprint,
        "trainable_fingerprint_after": trainable_after_fingerprint,
        "frozen_fingerprint_before": frozen_before,
        "frozen_fingerprint_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "checkpoint_roundtrip_max_abs": reload_diff,
        "checkpoint_roundtrip_tolerance": 1e-5,
        "adapter": adapter_meta,
        "gradient_gate": gate,
        "c4_cost_smoke": c4_cost,
        "wall_time_s": elapsed,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30) if device.startswith("cuda") else None,
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30) if device.startswith("cuda") else None,
    }
    payload["smoke_sha256"] = _sha_object(payload)
    _write_json(output, payload)
    _update_resource_ledger(
        run_dir,
        {
            "phase": "smoke",
            "status": payload["status"],
            "wall_time_s": elapsed + float(c4_cost.get("wall_time_s", 0.0) or 0.0),
            "optimization_wall_time_s": 0.0,
            "device": device,
            "peak_allocated_gib": payload["peak_allocated_gib"],
            "peak_reserved_gib": payload["peak_reserved_gib"],
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "SMOKE_PASSED":
        return 2
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    phase_start = time.perf_counter()
    import torch

    run_dir = _ensure_run_dir(args)
    output = run_dir / "m0_predictions.json"
    if output.exists():
        raise FileExistsError(f"baseline_already_exists:{output}")
    _validate_run_config(run_dir, args)
    samples, manifest = _load_manifest(run_dir)
    validation = [sample for sample in samples if sample.split == "validation"]
    train = [sample for sample in samples if sample.split == "train"]
    if len(validation) != 53 or len(train) != 158:
        raise ValueError("c3_baseline_split_counts")
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    rows, repeats = _evaluate_model(runtime, model, validation, device=device, model_name="m0", repeat=True)
    if len(rows) != 53:
        raise ValueError(f"c3_m0_validation_count:{len(rows)}")
    max_repeat = max(repeats) if repeats else 0.0
    # Independent reload is part of the M0 numerical error budget.  Release
    # the first model before constructing the second copy to keep the 4080
    # peak bounded.
    rows_reload: list[dict[str, Any]] = []
    reload_report: Any = None
    try:
        del model, runtime
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        runtime_reload, model_reload, device_reload, reload_report = _make_runtime(args)
        rows_reload, _ = _evaluate_model(runtime_reload, model_reload, validation, device=device_reload, model_name="m0_reload", repeat=False)
    finally:
        try:
            del runtime_reload, model_reload
        except UnboundLocalError:
            pass
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    by_root = {row.get("root_id"): row for row in rows}
    by_reload = {row.get("root_id"): row for row in rows_reload}
    reload_diffs: list[float] = []
    reload_ade_diffs: list[float] = []
    for root_id, first in by_root.items():
        second = by_reload.get(root_id)
        if not second:
            continue
        try:
            route_a = np.asarray(first.get("route"), dtype=np.float64)
            route_b = np.asarray(second.get("route"), dtype=np.float64)
            speed_a = np.asarray(first.get("speed"), dtype=np.float64)
            speed_b = np.asarray(second.get("speed"), dtype=np.float64)
            if route_a.shape == route_b.shape and route_a.size:
                reload_diffs.append(float(np.max(np.abs(route_a - route_b))))
            if first.get("route_ade_m") is not None and second.get("route_ade_m") is not None:
                reload_ade_diffs.append(abs(float(first["route_ade_m"]) - float(second["route_ade_m"])))
            if speed_a.shape == speed_b.shape and speed_a.size:
                reload_diffs.append(float(np.max(np.abs(speed_a - speed_b))))
        except (TypeError, ValueError):
            continue
    reload_max_abs = max(reload_diffs) if reload_diffs else 0.0
    reload_ade_max_abs = max(reload_ade_diffs) if reload_ade_diffs else 0.0
    diagnostic = _diagnostic_baselines(train, validation)
    baseline = {
        "schema_version": "safedrive.c3.m0_predictions.v2",
        "status": "M0_MEASURED",
        "model": vars(report),
        "rows": rows,
        "row_count": len(rows),
        "repeat_forward_max_abs": max_repeat,
        "reload_max_abs": reload_max_abs,
        "reload_route_ade_max_abs": reload_ade_max_abs,
        "eps_ADE_m": max(1e-5, 10.0 * reload_ade_max_abs),
        "repeat_forward_count": len(repeats),
        "reload_row_count": len(rows_reload),
        "reload_model_load": vars(reload_report) if reload_report is not None else None,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    baseline["content_sha256"] = _sha_object(baseline)
    _write_json(output, baseline)
    _write_json(run_dir / "m0_reload_predictions.json", {
        "schema_version": "safedrive.c3.m0_reload_predictions.v2",
        "status": "M0_RELOAD_MEASURED",
        "rows": rows_reload,
        "row_count": len(rows_reload),
        "content_sha256": _sha_object(rows_reload),
    })
    _write_prediction_failures(
        run_dir,
        phase="m0",
        rows=rows,
        manifest_sha256=manifest["manifest_sha256"],
        source_sha256=baseline["content_sha256"],
    )
    m0_reload_payload = _read_json(run_dir / "m0_reload_predictions.json")
    _write_prediction_failures(
        run_dir,
        phase="m0_reload",
        rows=rows_reload,
        manifest_sha256=manifest["manifest_sha256"],
        source_sha256=m0_reload_payload["content_sha256"],
    )
    _write_json(run_dir / "diagnostic-baselines.json", diagnostic)
    metrics_spec = {
        "schema_version": "safedrive.c3.metrics_spec.v2",
        "primary": "P-ADE.route_ade_m",
        "required_metrics": ["P-ADE", "P-FDE", "P-WP", "P-VALID", "P-FAIL"],
        "metric_mapping": {
            "P-ADE": "route_ade_m",
            "P-FDE": "route_fde_m",
            "P-WP": "speed_wp_ade_m",
            "P-VALID": "canonical_output_valid plus at least one valid route and speed point",
            "P-FAIL": "all validation rows; failure_reasons retained per row",
            "P-SPEED": "N/A_without_trusted_speed_ground_truth_conversion",
        },
        "aggregation": "root_equal",
        "mask": "same_validation_root_and_target_masks",
        "eps_ADE_m": baseline["eps_ADE_m"],
        "bootstrap_rounds": 1000,
        "bootstrap_seed": 71,
        "p_speed": "N/A_without_trusted_time_speed_ground_truth",
        "fde": "fixed route point 20 only; missing point 20 is N/A",
        "failure_denominator": 53,
        "validation_input_manifest_sha256": manifest["manifest_sha256"],
        "route_steps": ROUTE_STEPS,
        "speed_steps": SPEED_STEPS,
        "speed_dt_s": SPEED_DT_S,
        "execution_timeline_dt_s": EXECUTION_DT_S,
    }
    metrics_spec["content_sha256"] = _sha_object(metrics_spec)
    _write_json(run_dir / "metrics-spec.json", metrics_spec)
    _update_resource_ledger(
        run_dir,
        {
            "phase": "baseline",
            "status": baseline["status"],
            "wall_time_s": time.perf_counter() - phase_start,
            "optimization_wall_time_s": 0.0,
            "device": device,
            "peak_allocated_gib": None,
            "validation_rows": len(rows),
            "reload_rows": len(rows_reload),
        },
    )
    print(json.dumps({"status": baseline["status"], "row_count": len(rows), "eps_ADE_m": baseline["eps_ADE_m"]}, indent=2))
    return 0


def _save_training_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    *,
    update: int,
    sample_index: int,
    epoch: int,
    log: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    import torch

    state = {
        "schema_version": "safedrive.c3.m1_resume.v2",
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
            if _is_update_parameter(name)
        },
        "optimizer_state": _optimizer_state_cpu(optimizer),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "log_tail": list(log[-5:]),
        "log_count": len(log),
    }
    if metadata:
        state.update(dict(metadata))
    if torch.cuda.is_available():
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _restore_training_checkpoint(path: Path, model: Any, optimizer: Any) -> tuple[int, int, int, list[dict[str, Any]]]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") not in {"safedrive.c3.m1_resume.v1", "safedrive.c3.m1_resume.v2"}:
        raise ValueError("c3_resume_schema")
    by_name = dict(model.named_parameters())
    expected = {
        name
        for name, parameter in model.named_parameters()
        if _is_update_parameter(name)
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


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("c3_checkpoint_state_not_mapping")
    return state


def _training_log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"c3_training_log_invalid_json:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"c3_training_log_row_not_object:{line_number}")
        rows.append(value)
    return rows


def _training_log_matches_summary(log_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> bool:
    """Bind the append-only JSONL log to the embedded training summary.

    The summary is deliberately redundant: it gives an independent verifier a
    complete record even when the human-readable JSONL is moved elsewhere.
    Comparing the two catches truncation, reordering, and edits to the log
    that would otherwise leave only the update numbers looking plausible.
    """

    recorded = summary.get("logs")
    return isinstance(recorded, list) and list(log_rows) == recorded


def cmd_train(args: argparse.Namespace) -> int:
    phase_start = time.perf_counter()
    run_dir = _ensure_run_dir(args)
    summary_path = run_dir / "m1_training_summary.json"
    if summary_path.exists() and not args.resume:
        raise FileExistsError(f"training_already_exists:{summary_path}")
    samples, manifest = _load_manifest(run_dir)
    audit = _read_json(run_dir / "audit.json") if (run_dir / "audit.json").is_file() else {}
    smoke = _read_json(run_dir / "smoke.json") if (run_dir / "smoke.json").is_file() else {}
    if audit.get("status") != "AUDIT_PASSED":
        raise ValueError("c3_train_requires_passed_audit")
    if smoke.get("status") != "SMOKE_PASSED":
        raise ValueError("c3_train_requires_passed_smoke")
    if not (run_dir / "m0_predictions.json").is_file():
        raise FileNotFoundError("c3_train_requires_m0_baseline")
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    if len(train) != 158 or len(validation) != 53:
        raise ValueError(f"c3_train_split_counts:{len(train)}:{len(validation)}")
    config = _validate_run_config(run_dir, args)
    config_sha = _config_digest(config)
    contract_values = {
        "seed": int(args.seed),
        "updates": int(args.updates),
        "gradient_accumulation": int(args.accumulation),
        "lora_learning_rate": float(args.lora_lr),
        "driving_head_learning_rate": float(args.head_lr),
        "weight_decay": float(args.weight_decay),
    }
    for key, expected in contract_values.items():
        actual = config.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"c3_train_config_conflict:{key}")
        elif actual != expected:
            raise ValueError(f"c3_train_config_conflict:{key}")
    resume_path = run_dir / "checkpoint_latest.pt"
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    if args.resume and not (run_dir / "m1_initial_parameters.pt").is_file():
        raise FileNotFoundError("c3_resume_initial_parameter_snapshot_missing")
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    lora, heads = _freeze_trainable(model)
    frozen_before = _frozen_fingerprints(model)
    trainable_before_fingerprint = _trainable_fingerprints(model)
    initial_state_path = run_dir / "m1_initial_parameters.pt"
    if args.resume:
        resume_meta = _checkpoint_metadata(resume_path)
        if resume_meta.get("schema_version") != "safedrive.c3.m1_resume.v2":
            raise ValueError("c3_resume_repair_schema_required")
        for key, expected in (("run_id", str(args.run_id)), ("config_sha256", config_sha), ("manifest_sha256", manifest["manifest_sha256"]), ("target_updates", int(args.updates)), ("accumulation", int(args.accumulation)), ("seed", int(args.seed))):
            if resume_meta.get(key) != expected:
                raise ValueError(f"c3_resume_identity_conflict:{key}")
        if resume_meta.get("initial_frozen_fingerprints") != frozen_before:
            raise ValueError("c3_resume_frozen_base_mismatch")
        if resume_meta.get("initial_trainable_fingerprints") != trainable_before_fingerprint:
            raise ValueError("c3_resume_initial_trainable_mismatch")
        initial_state = _load_update_state(initial_state_path)
    else:
        if initial_state_path.exists():
            raise FileExistsError(f"c3_initial_parameter_snapshot_exists:{initial_state_path}")
        initial_state_meta = _save_update_state(model, initial_state_path)
        initial_state = _load_update_state(initial_state_path)
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
    exposures = {sample.root_id: 0 for sample in train}
    prior_elapsed = 0.0
    max_allocated = 0.0
    max_reserved = 0.0
    if args.resume:
        resume_meta = _checkpoint_metadata(resume_path)
        required_checkpoint_keys = {
            "model_state",
            "optimizer_state",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "log_tail",
            "log_count",
            "root_exposures",
            "accumulated_wall_time_s",
            "training_log_sha256",
        }
        if not required_checkpoint_keys.issubset(resume_meta):
            raise ValueError("c3_resume_checkpoint_state_incomplete")
        start_update, sample_index, epoch, restored_tail = _restore_training_checkpoint(resume_path, model, optimizer)
        logs = _training_log_rows(run_dir / "training.jsonl")
        expected_log_count = int(resume_meta.get("log_count", start_update))
        if len(logs) != expected_log_count or len(logs) < start_update:
            raise ValueError("c3_resume_training_log_truncated")
        if list(resume_meta.get("log_tail", [])) != list(restored_tail) or list(restored_tail) != logs[-5:]:
            raise ValueError("c3_resume_training_log_boundary_mismatch")
        if str(resume_meta.get("training_log_sha256", "")) != sha256_file(run_dir / "training.jsonl"):
            raise ValueError("c3_resume_training_log_digest_mismatch")
        restored_exposures = resume_meta.get("root_exposures", {})
        if set(restored_exposures) != set(exposures):
            raise ValueError("c3_resume_root_exposure_set")
        exposures = {root_id: int(restored_exposures[root_id]) for root_id in exposures}
        prior_elapsed = float(resume_meta.get("accumulated_wall_time_s", 0.0) or 0.0)
        max_allocated = float(resume_meta.get("peak_allocated_gib", 0.0) or 0.0)
        max_reserved = float(resume_meta.get("peak_reserved_gib", 0.0) or 0.0)
    if start_update >= total_updates:
        raise ValueError("c3_resume_already_complete")
    if sample_index < 0 or sample_index > len(train):
        raise ValueError("c3_resume_sample_index_invalid")
    start_wall = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    root_order = deterministic_order(train, args.seed + epoch)

    def checkpoint_metadata() -> dict[str, Any]:
        current_elapsed = time.perf_counter() - start_wall
        return {
            "run_id": str(args.run_id),
            "config_sha256": config_sha,
            "manifest_sha256": manifest["manifest_sha256"],
            "target_updates": total_updates,
            "accumulation": int(args.accumulation),
            "seed": int(args.seed),
            "root_exposures": dict(exposures),
            "accumulated_wall_time_s": prior_elapsed + current_elapsed,
            "initial_frozen_fingerprints": frozen_before,
            "initial_trainable_fingerprints": trainable_before_fingerprint,
            "initial_parameter_snapshot_sha256": sha256_file(initial_state_path),
            "peak_allocated_gib": max_allocated,
            "peak_reserved_gib": max_reserved,
            "last_finite_update": int(start_update),
            "training_log_sha256": sha256_file(run_dir / "training.jsonl") if (run_dir / "training.jsonl").is_file() else "",
        }

    while start_update < total_updates:
        if sample_index >= len(root_order):
            epoch += 1
            root_order = deterministic_order(train, args.seed + epoch)
            sample_index = 0
        window = root_order[sample_index : min(sample_index + int(args.accumulation), len(root_order))]
        if not window:
            raise RuntimeError("c3_training_empty_window")
        update = start_update + 1
        schedule = _set_schedule(
            optimizer,
            update,
            total=total_updates,
            head_lr=args.head_lr,
            lora_lr=args.lora_lr,
            warmup=warmup,
            lora=lora,
        )
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
        start_update = update
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
            "wall_time_s": prior_elapsed + time.perf_counter() - start_wall,
        }
        logs.append(entry)
        with (run_dir / "training.jsonl").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
        if update % 10 == 0 or update == total_updates:
            _save_training_checkpoint(
                resume_path,
                model,
                optimizer,
                update=start_update,
                sample_index=sample_index,
                epoch=epoch,
                log=logs,
                metadata=checkpoint_metadata(),
            )
        if prior_elapsed + time.perf_counter() - start_wall > float(args.max_hours) * 3600.0:
            break
    completed = start_update >= total_updates
    _save_training_checkpoint(
        resume_path,
        model,
        optimizer,
        update=start_update,
        sample_index=sample_index,
        epoch=epoch,
        log=logs,
        metadata=checkpoint_metadata(),
    )
    _set_lora_enabled(lora, True)
    frozen_after = _frozen_fingerprints(model)
    trainable_after_fingerprint = _trainable_fingerprints(model)
    parameter_change = _parameter_change_max_abs(model, initial_state)
    adapter_path = run_dir / "m1_adapter.pt"
    adapter_meta = _save_trainable_state(model, adapter_path)
    total_wall = prior_elapsed + time.perf_counter() - start_wall
    window_sizes = [int(item.get("window_size", 0)) for item in logs]
    epoch_window_sizes: dict[str, list[int]] = {}
    for item in logs:
        epoch_window_sizes.setdefault(str(item.get("epoch", 0)), []).append(int(item.get("window_size", 0)))
    payload = {
        "schema_version": "safedrive.c3.m1_training.v2",
        "status": "M1_COMPLETED" if completed else "M1_PARTIAL",
        "run_id": str(args.run_id),
        "model_load": vars(report),
        "target_updates": total_updates,
        "actual_updates": start_update,
        "warmup_head_updates": warmup,
        "lora_unfreeze_update": warmup + 1,
        "seed": int(args.seed),
        "train_root_count": len(train),
        "validation_root_count": len(validation),
        "root_exposures": exposures,
        "root_exposure_min": min(exposures.values()) if exposures else 0,
        "root_exposure_max": max(exposures.values()) if exposures else 0,
        "samples_seen": int(sum(window_sizes)),
        "window_size_histogram": {str(size): window_sizes.count(size) for size in sorted(set(window_sizes))},
        "epoch_window_sizes": epoch_window_sizes,
        "tail_window_size": 2,
        "tail_window_count": int(sum(size == 2 for size in window_sizes)),
        "expected_first_epoch_windows": {"full_size_4": 39, "tail_size_2": 1, "roots": 158},
        "adapter": adapter_meta,
        "checkpoint_latest": str(resume_path),
        "logs": logs,
        "initial_frozen_fingerprints": frozen_before,
        "final_frozen_fingerprints": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "initial_trainable_fingerprints": trainable_before_fingerprint,
        "final_trainable_fingerprints": trainable_after_fingerprint,
        "parameter_change": parameter_change,
        "resources": {
            "wall_time_s": total_wall,
            "peak_allocated_gib": max_allocated if device.startswith("cuda") else None,
            "peak_reserved_gib": max_reserved if device.startswith("cuda") else None,
            "device": device,
        },
        "config_sha256": config_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "initial_parameter_snapshot_sha256": sha256_file(initial_state_path),
    }
    payload["content_sha256"] = _sha_object(payload)
    # Refresh the atomic checkpoint after the final fingerprints are known so
    # an independent verifier can bind the saved adapter and log boundary to
    # the exact checkpoint state.
    _save_training_checkpoint(
        resume_path,
        model,
        optimizer,
        update=start_update,
        sample_index=sample_index,
        epoch=epoch,
        log=logs,
        metadata={
            **checkpoint_metadata(),
            "final_frozen_fingerprints": frozen_after,
            "final_trainable_fingerprints": trainable_after_fingerprint,
            "parameter_change": parameter_change,
            "training_status": payload["status"],
        },
    )
    _write_json(summary_path, payload, overwrite=args.resume and summary_path.exists())
    ledger = _update_resource_ledger(
        run_dir,
        {
            "phase": "train_resume" if args.resume else "train",
            "status": payload["status"],
            "wall_time_s": time.perf_counter() - phase_start,
            "optimization_wall_time_s": time.perf_counter() - phase_start,
            "accumulated_optimization_wall_time_s": total_wall,
            "device": device,
            "peak_allocated_gib": payload["resources"]["peak_allocated_gib"],
            "peak_reserved_gib": payload["resources"]["peak_reserved_gib"],
            "target_updates": total_updates,
            "actual_updates": start_update,
            "window_size_histogram": payload["window_size_histogram"],
        },
    )
    print(json.dumps({"status": payload["status"], "actual_updates": start_update, "target_updates": total_updates, "peak_allocated_gib": max_allocated, "optimization_total_hours_upper_bound": ledger.get("optimization_total_hours_upper_bound")}, indent=2))
    return 0 if completed else 2


def cmd_evaluate(args: argparse.Namespace) -> int:
    phase_start = time.perf_counter()
    run_dir = _ensure_run_dir(args)
    output = run_dir / "m1_predictions.json"
    if output.exists():
        raise FileExistsError(f"evaluation_already_exists:{output}")
    config = _validate_run_config(run_dir, args)
    samples, manifest = _load_manifest(run_dir)
    validation = [sample for sample in samples if sample.split == "validation"]
    training = _read_json(run_dir / "m1_training_summary.json")
    if training.get("status") != "M1_COMPLETED":
        raise ValueError(f"c3_evaluation_requires_completed_m1:{training.get('status')}")
    if training.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("c3_evaluation_manifest_identity_mismatch")
    _seed_everything(args.seed)
    runtime, model, device, report = _make_runtime(args)
    lora, heads = _freeze_trainable(model)
    _load_trainable_state(model, run_dir / "m1_adapter.pt")
    m1_rows, _ = _evaluate_model(runtime, model, validation, device=device, model_name="m1", repeat=False)
    if len(m1_rows) != 53:
        raise ValueError(f"c3_m1_validation_count:{len(m1_rows)}")
    m0 = _read_json(run_dir / "m0_predictions.json")
    m0_rows = list(m0.get("rows", ()))
    independent_m0 = _independent_metric_rows(m0_rows, validation, model_name="m0")
    independent_m1 = _independent_metric_rows(m1_rows, validation, model_name="m1")
    metric_rows = independent_m0 + independent_m1
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
        "schema_version": "safedrive.c3.m1_predictions.v2",
        "status": "M1_MEASURED",
        "model": vars(report),
        "rows": m1_rows,
        "row_count": len(m1_rows),
        "content_sha256": _sha_object(m1_rows),
    }
    _write_json(output, payload)
    _write_prediction_failures(
        run_dir,
        phase="m1",
        rows=m1_rows,
        manifest_sha256=manifest["manifest_sha256"],
        source_sha256=payload["content_sha256"],
    )
    independent_payload = {
        "schema_version": "safedrive.c3.independent_recompute.v2",
        "status": "INDEPENDENT_RECOMPUTE_VERIFIED",
        "manifest_sha256": manifest["manifest_sha256"],
        "m0_source_sha256": m0.get("content_sha256"),
        "m1_source_sha256": payload["content_sha256"],
        "m0_rows": independent_m0,
        "m1_rows": independent_m1,
        "aggregate": aggregate,
    }
    independent_payload["content_sha256"] = _sha_object(independent_payload)
    _write_json(run_dir / "independent-recompute.json", independent_payload)
    aggregate["m0_content_sha256"] = m0.get("content_sha256")
    aggregate["m1_content_sha256"] = payload["content_sha256"]
    aggregate["content_sha256"] = _sha_object(aggregate)
    _write_json(run_dir / "metrics.json", aggregate)
    report_payload = {
        "schema_version": "safedrive.c3.evaluation_report.v2",
        "status": "EVALUATION_MEASURED",
        "run_id": str(args.run_id),
        "validation_root_count": len(validation),
        "m0_prediction_path": str(run_dir / "m0_predictions.json"),
        "m1_prediction_path": str(output),
        "metrics_path": str(run_dir / "metrics.json"),
        "metrics_sha256": aggregate["content_sha256"],
        "metrics": aggregate["metrics"],
        "eps_ADE_m": aggregate["eps_ADE_m"],
        "independent_recompute_path": str(run_dir / "independent-recompute.json"),
        "independent_recompute_sha256": independent_payload["content_sha256"],
        "validation_manifest_sha256": manifest["manifest_sha256"],
        "p_fail_denominator": len(validation),
        "p_speed": "N/A_without_trusted_time_speed_ground_truth",
        "reproduction_command": f"python scripts/h6_cora_sft.py evaluate --run-id {args.run_id} --device cuda",
    }
    report_payload["content_sha256"] = _sha_object(report_payload)
    _write_json(run_dir / "evaluation-report.json", report_payload)
    _update_resource_ledger(
        run_dir,
        {
            "phase": "evaluate",
            "status": report_payload["status"],
            "wall_time_s": time.perf_counter() - phase_start,
            "optimization_wall_time_s": 0.0,
            "device": device,
            "peak_allocated_gib": None,
            "validation_rows": len(validation),
        },
    )
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
        "m0_reload_predictions.json",
        "m0_prediction_failures.json",
        "m0_reload_prediction_failures.json",
        "diagnostic-baselines.json",
        "m1_initial_parameters.pt",
        "m1_training_summary.json",
        "m1_adapter.pt",
        "checkpoint_latest.pt",
        "training.jsonl",
        "m1_predictions.json",
        "m1_prediction_failures.json",
        "independent-recompute.json",
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
    m0_reload = _read_json(run_dir / "m0_reload_predictions.json")
    m0_failures = _read_json(run_dir / "m0_prediction_failures.json")
    m0_reload_failures = _read_json(run_dir / "m0_reload_prediction_failures.json")
    diagnostics = _read_json(run_dir / "diagnostic-baselines.json")
    train = _read_json(run_dir / "m1_training_summary.json")
    m1 = _read_json(run_dir / "m1_predictions.json")
    m1_failures = _read_json(run_dir / "m1_prediction_failures.json")
    independent = _read_json(run_dir / "independent-recompute.json")
    metrics_spec = _read_json(run_dir / "metrics-spec.json")
    metrics = _read_json(run_dir / "metrics.json")
    resources = _read_json(run_dir / "resource-ledger.json")
    evaluation = _read_json(run_dir / "evaluation-report.json")
    errors: list[str] = []
    try:
        config = _validate_run_config(run_dir, args, require_model_device=False)
    except Exception as exc:  # noqa: BLE001 - preserve a tamper result as verify evidence
        try:
            config = _read_json(run_dir / "run_config.json")
        except Exception:
            config = {}
        errors.append(f"run_config_validation:{type(exc).__name__}:{exc}")

    def digest_check(payload: Mapping[str, Any], field: str, *, mode: str = "payload") -> bool:
        digest = payload.get(field)
        if not digest:
            return False
        if mode == "rows":
            return digest == _sha_object(payload.get("rows", []))
        check = dict(payload)
        check.pop(field, None)
        return digest == _sha_object(check)

    if config.get("schema_version") != "safedrive.c3.sft_run_config.v2" or config.get("repair_schema") != RUN_SCHEMA_VERSION:
        errors.append("run_config_schema")
    if str(config.get("run_id")) != str(args.run_id):
        errors.append("run_id_identity")
    if not digest_check(manifest, "manifest_sha256"):
        errors.append("manifest_digest")
    if not digest_check(audit, "audit_sha256"):
        errors.append("audit_digest")
    if not digest_check(smoke, "smoke_sha256"):
        errors.append("smoke_digest")
    if not digest_check(m0, "content_sha256"):
        errors.append("m0_digest")
    if not digest_check(m0_reload, "content_sha256", mode="rows"):
        errors.append("m0_reload_digest")
    for failure_payload, name, source in (
        (m0_failures, "m0", m0),
        (m0_reload_failures, "m0_reload", m0_reload),
        (m1_failures, "m1", m1),
    ):
        if not digest_check(failure_payload, "content_sha256"):
            errors.append(f"{name}_failure_digest")
        if failure_payload.get("manifest_sha256") != manifest.get("manifest_sha256") or failure_payload.get("source_predictions_sha256") != source.get("content_sha256"):
            errors.append(f"{name}_failure_identity")
        failure_ids = [str(row.get("root_id", "")) for row in failure_payload.get("rows", [])]
        source_failures = {
            str(row.get("root_id", ""))
            for row in source.get("rows", [])
            if row.get("prediction_exception")
            or any(str(reason).startswith(("prediction_exception:", "repeat_exception:", "repeat_output:")) for reason in row.get("failure_reasons", ()))
        }
        if set(failure_ids) != source_failures or len(failure_ids) != len(set(failure_ids)):
            errors.append(f"{name}_failure_rows")
    if not digest_check(diagnostics, "content_sha256", mode="rows"):
        errors.append("diagnostic_digest")
    if not digest_check(train, "content_sha256"):
        errors.append("training_digest")
    if not digest_check(m1, "content_sha256", mode="rows"):
        errors.append("m1_digest")
    if not digest_check(independent, "content_sha256"):
        errors.append("independent_recompute_digest")
    if not digest_check(metrics, "content_sha256"):
        errors.append("metrics_digest")
    if not digest_check(evaluation, "content_sha256"):
        errors.append("evaluation_digest")
    resource_payload = {
        key: value for key, value in resources.items() if key not in {"content_sha256", "ledger_sha256"}
    }
    if resources.get("content_sha256") != _sha_object(resource_payload):
        errors.append("resource_content_digest")
    if resources.get("ledger_sha256") != _sha_object({**resource_payload, "content_sha256": resources.get("content_sha256")}):
        errors.append("resource_ledger_digest")
    if audit.get("status") != "AUDIT_PASSED":
        errors.append("audit_not_passed")
    if smoke.get("status") != "SMOKE_PASSED":
        errors.append("smoke_not_passed")
    if train.get("status") != "M1_COMPLETED":
        errors.append("m1_not_completed")
    if m0.get("schema_version") != "safedrive.c3.m0_predictions.v2":
        errors.append("m0_schema")
    if m0_reload.get("schema_version") != "safedrive.c3.m0_reload_predictions.v2":
        errors.append("m0_reload_schema")
    if diagnostics.get("schema_version") != "safedrive.c3.diagnostic_baselines.v2":
        errors.append("diagnostic_schema")
    if m1.get("schema_version") != "safedrive.c3.m1_predictions.v2":
        errors.append("m1_schema")
    if independent.get("schema_version") != "safedrive.c3.independent_recompute.v2":
        errors.append("independent_schema")
    if metrics.get("schema_version") != "safedrive.c3.metrics.v2" or metrics_spec.get("schema_version") != "safedrive.c3.metrics_spec.v2":
        errors.append("metrics_schema")
    if len(samples) != 211 or manifest.get("split_counts") != {"train": 158, "validation": 53}:
        errors.append("manifest_split_counts")
    sample_ids = [sample.root_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("manifest_duplicate_roots")
    if manifest.get("future_labels_in_input") is not False or manifest.get("world_labels_in_sft") is not False:
        errors.append("leakage_flags")
    for sample in samples:
        if len(sample.route_target) != ROUTE_STEPS or len(sample.route_mask) != ROUTE_STEPS:
            errors.append(f"route_shape:{sample.root_id}")
        if len(sample.speed_target) != SPEED_STEPS or len(sample.speed_mask) != SPEED_STEPS:
            errors.append(f"speed_shape:{sample.root_id}")
        if sample.route_source != "native_expert_reference_path_projected" or sample.teacher_generator != "classic-frenet-st@h1":
            errors.append(f"source_contract:{sample.root_id}")
        if sample.route_projection_ambiguous or sample.route_projection_distance_m > 2.5 or sample.route_support_m < ROUTE_STEPS:
            errors.append(f"route_projection:{sample.root_id}")
        if sample.route_reference_revision == "" or sample.outcome_label_sha256 == "":
            errors.append(f"label_identity:{sample.root_id}")
        if sample.speed_source != "expert_canonical_proposal":
            errors.append(f"speed_source:{sample.root_id}")
    validation = [sample for sample in samples if sample.split == "validation"]
    try:
        independent_m0 = _independent_metric_rows(m0.get("rows", []), validation, model_name="m0")
        independent_m1 = _independent_metric_rows(m1.get("rows", []), validation, model_name="m1")
        recomputed_aggregate = aggregate_metrics(independent_m0 + independent_m1, bootstrap_rounds=1000, bootstrap_seed=71)
        recomputed_aggregate["eps_ADE_m"] = float(m0.get("eps_ADE_m"))
        if recomputed_aggregate["metrics"] != metrics.get("metrics"):
            errors.append("metrics_independent_recompute_mismatch")
        if independent.get("status") != "INDEPENDENT_RECOMPUTE_VERIFIED":
            errors.append("independent_recompute_status")
        if independent.get("aggregate", {}).get("metrics") != metrics.get("metrics"):
            errors.append("independent_saved_aggregate_mismatch")
    except Exception as exc:  # noqa: BLE001 - verifier converts attacks to a failed evidence record
        errors.append(f"independent_recompute_failed:{type(exc).__name__}:{exc}")
    if len(m0.get("rows", [])) != 53 or len(m1.get("rows", [])) != 53 or len(m0_reload.get("rows", [])) != 53:
        errors.append("validation_rows_not_53")
    expected_validation_ids = {sample.root_id for sample in validation}
    for payload, name in ((m0, "m0"), (m0_reload, "m0_reload"), (m1, "m1")):
        ids = [row.get("root_id") for row in payload.get("rows", [])]
        if set(ids) != expected_validation_ids or len(ids) != len(set(ids)):
            errors.append(f"{name}_root_set")
    if not digest_check(metrics_spec, "content_sha256"):
        errors.append("metrics_spec_digest")
    if metrics_spec.get("primary") != "P-ADE.route_ade_m" or metrics_spec.get("failure_denominator") != 53:
        errors.append("metrics_spec_contract")
    if float(m0.get("eps_ADE_m", 0.0)) != max(1e-5, 10.0 * float(m0.get("reload_route_ade_max_abs", 0.0))):
        errors.append("eps_ADE_not_locked_from_reload")
    if train.get("actual_updates") != train.get("target_updates") or train.get("target_updates") != 200:
        errors.append("update_budget_not_reached")
    logs = _training_log_rows(run_dir / "training.jsonl")
    if not _training_log_matches_summary(logs, train):
        errors.append("training_log_summary_mismatch")
    if len(logs) != int(train.get("actual_updates", -1)) or [int(row.get("update", -1)) for row in logs] != list(range(1, len(logs) + 1)):
        errors.append("training_log_update_sequence")
    if any(int(row.get("window_size", 0)) not in {1, 2, 3, 4} for row in logs):
        errors.append("training_window_size")
    # The final window of every 158-root epoch has two roots.  Therefore the
    # exact sample count is derived from the recorded windows (200 updates
    # produce 790 samples), rather than incorrectly assuming 200*4 samples.
    expected_samples = int(sum(int(row.get("window_size", 0)) for row in logs))
    if int(train.get("samples_seen", -1)) != expected_samples or expected_samples != 790:
        errors.append("training_samples_seen")
    if train.get("expected_first_epoch_windows") != {"full_size_4": 39, "tail_size_2": 1, "roots": 158}:
        errors.append("training_tail_contract")
    first_epoch = [row for row in logs if int(row.get("epoch", -1)) == 0]
    if [int(row.get("window_size", 0)) for row in first_epoch] != [4] * 39 + [2]:
        errors.append("training_first_epoch_tail")
    expected_order = deterministic_order([sample for sample in samples if sample.split == "train"], int(train.get("seed", 17)))
    expected_epoch = 0
    expected_index = 0
    expected_exposures = {sample.root_id: 0 for sample in samples if sample.split == "train"}
    expected_schedule_samples = 0
    for row in logs:
        if expected_index >= len(expected_order):
            expected_epoch += 1
            expected_order = deterministic_order([sample for sample in samples if sample.split == "train"], int(train.get("seed", 17)) + expected_epoch)
            expected_index = 0
        expected_window = expected_order[expected_index : min(expected_index + int(config.get("gradient_accumulation", 4)), len(expected_order))]
        actual_ids = [str(item) for item in row.get("root_ids", [])]
        expected_ids = [sample.root_id for sample in expected_window]
        if int(row.get("epoch", -1)) != expected_epoch or actual_ids != expected_ids:
            errors.append("training_sample_order")
        expected_index += len(expected_window)
        expected_schedule_samples += len(expected_window)
        for sample in expected_window:
            expected_exposures[sample.root_id] += 1
    exposures = train.get("root_exposures", {})
    if expected_schedule_samples != expected_samples:
        errors.append("training_schedule_sample_count")
    if exposures != expected_exposures or set(exposures) != set(expected_exposures) or sum(int(value) for value in exposures.values()) != expected_samples:
        errors.append("training_root_exposures")
    if min((int(value) for value in exposures.values()), default=0) != 5 or max((int(value) for value in exposures.values()), default=0) != 5:
        errors.append("training_exposure_balance")
    if train.get("frozen_parameters_unchanged") is not True or train.get("initial_frozen_fingerprints") != train.get("final_frozen_fingerprints"):
        errors.append("frozen_parameters_changed")
    parameter_change = train.get("parameter_change", {})
    if parameter_change.get("lora_changed") is not True or parameter_change.get("driving_head_changed") is not True:
        errors.append("trainable_parameters_not_changed")
    if train.get("initial_trainable_fingerprints", {}).get("model_digest") == train.get("final_trainable_fingerprints", {}).get("model_digest"):
        errors.append("trainable_fingerprint_unchanged")
    c4_cost = smoke.get("c4_cost_smoke", {})
    if (
        c4_cost.get("status") != "MEASURED"
        or c4_cost.get("window_size") != 4
        or c4_cost.get("candidate_output_dim") != 4
        or c4_cost.get("zero_residual_max_abs") != 0.0
        or not c4_cost.get("shared_lora_grad_nonzero")
        or c4_cost.get("clip_max_norm") != 1.0
        or float(c4_cost.get("shared_clip_norm_after", float("inf"))) > 1.0 + 1e-6
        or float(c4_cost.get("residual_head_clip_norm_after", float("inf"))) > 1.0 + 1e-6
    ):
        errors.append("c4_cost_smoke_contract")
    try:
        checkpoint = _checkpoint_metadata(run_dir / "checkpoint_latest.pt")
        if checkpoint.get("schema_version") != "safedrive.c3.m1_resume.v2" or checkpoint.get("run_id") != args.run_id:
            errors.append("checkpoint_identity")
        if checkpoint.get("update") != train.get("actual_updates") or checkpoint.get("config_sha256") != train.get("config_sha256") or checkpoint.get("manifest_sha256") != manifest.get("manifest_sha256"):
            errors.append("checkpoint_position_identity")
        if checkpoint.get("log_count") != len(logs) or checkpoint.get("training_log_sha256") != sha256_file(run_dir / "training.jsonl"):
            errors.append("checkpoint_log_boundary")
        if checkpoint.get("root_exposures") != exposures or checkpoint.get("last_finite_update") != train.get("actual_updates"):
            errors.append("checkpoint_exposure_or_finite_update")
        if checkpoint.get("initial_parameter_snapshot_sha256") != sha256_file(run_dir / "m1_initial_parameters.pt"):
            errors.append("checkpoint_initial_snapshot_identity")
        if not {"python_random_state", "numpy_random_state", "torch_random_state", "optimizer_state", "model_state"}.issubset(checkpoint):
            errors.append("checkpoint_state_incomplete")
        initial_state = _load_update_state(run_dir / "m1_initial_parameters.pt")
        if set(checkpoint.get("model_state", {})) != set(initial_state):
            errors.append("checkpoint_whitelist")
        import torch

        adapter = torch.load(run_dir / "m1_adapter.pt", map_location="cpu", weights_only=True)
        if set(adapter) != set(initial_state):
            errors.append("adapter_whitelist")
        adapter_fp = _fingerprint_tensor_mapping(adapter)
        checkpoint_fp = _fingerprint_tensor_mapping(checkpoint.get("model_state", {}))
        final_fp = train.get("final_trainable_fingerprints", {})
        if adapter_fp.get("model_digest") != final_fp.get("model_digest") or checkpoint_fp.get("model_digest") != final_fp.get("model_digest"):
            errors.append("final_trainable_state_identity")
        if checkpoint.get("final_trainable_fingerprints") != final_fp:
            errors.append("checkpoint_final_trainable_fingerprint")
        if checkpoint.get("final_frozen_fingerprints") != train.get("final_frozen_fingerprints"):
            errors.append("checkpoint_final_frozen_fingerprint")
        if checkpoint.get("parameter_change") != train.get("parameter_change"):
            errors.append("checkpoint_parameter_change")
        adapter_meta = train.get("adapter", {})
        if adapter_meta.get("sha256") != sha256_file(run_dir / "m1_adapter.pt") or adapter_meta.get("model_digest") != final_fp.get("model_digest"):
            errors.append("adapter_file_identity")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"checkpoint_validation_failed:{type(exc).__name__}:{exc}")
    if float(train.get("resources", {}).get("peak_allocated_gib") or 0.0) > GPU_PEAK_LIMIT_GIB:
        errors.append("gpu_peak_over_budget")
    if resources.get("observed_peak_allocated_gib") is not None and float(resources["observed_peak_allocated_gib"]) > GPU_PEAK_LIMIT_GIB:
        errors.append("ledger_gpu_peak_over_budget")
    if resources.get("observed_peak_reserved_gib") is not None and float(resources["observed_peak_reserved_gib"]) > GPU_PEAK_LIMIT_GIB:
        errors.append("ledger_gpu_reserved_over_budget")
    if float(resources.get("optimization_total_hours_upper_bound", 1e9)) > OPTIMIZATION_BUDGET_HOURS:
        errors.append("optimization_budget_overrun")
    required_resource_phases = {"audit", "smoke", "baseline", "train", "evaluate"}
    if not resources.get("historical_events") or not resources.get("events") or resources.get("current_event_count") != len(resources.get("events", [])):
        errors.append("resource_history_missing")
    if not required_resource_phases.issubset({str(event.get("phase")) for event in resources.get("events", [])}):
        errors.append("resource_phase_coverage")
    if any(float(event.get("wall_time_s", 0.0) or 0.0) < 0.0 or float(event.get("optimization_wall_time_s", 0.0) or 0.0) < 0.0 for event in resources.get("events", [])):
        errors.append("resource_negative_duration")
    if any(
        float(event.get("optimization_wall_time_s_upper_bound", -1.0) or -1.0) < float(event.get("optimization_wall_time_s", 0.0) or 0.0)
        for event in resources.get("historical_events", [])
    ):
        errors.append("resource_historical_upper_bound_missing")
    if resources.get("status") != evaluation.get("status"):
        errors.append("resource_ledger_status_mismatch")
    if not any(event.get("status") == train.get("status") for event in resources.get("events", [])):
        errors.append("resource_ledger_status_mismatch")
    if evaluation.get("status") != "EVALUATION_MEASURED" or evaluation.get("validation_root_count") != 53 or evaluation.get("validation_manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append("evaluation_report_incomplete")
    verification = {
        "schema_version": "safedrive.c3.verification.v2",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": sorted(set(errors)),
        "run_id": str(args.run_id),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "m0_content_sha256": m0.get("content_sha256"),
        "m1_content_sha256": m1.get("content_sha256"),
        "metrics_content_sha256": metrics.get("content_sha256"),
        "m1_training_content_sha256": train.get("content_sha256"),
        "resource_ledger_sha256": resources.get("ledger_sha256"),
        "evaluation_report_sha256": evaluation.get("content_sha256"),
        "independent_recompute_sha256": independent.get("content_sha256"),
        "engineering_complete": not errors,
        "algorithm_gain_status": "MEASURED" if not errors else "NOT_VERIFIED",
        "verification_checks": {
            "data_semantics": not any(item.startswith(("route_", "speed_", "source_contract", "label_identity", "leakage", "manifest_")) for item in errors),
            "training_loop": not any(item.startswith(("training_", "update_", "trainable_", "frozen_", "checkpoint_", "adapter_")) for item in errors),
            "independent_metrics": not any(item.startswith(("metrics_", "m0_", "m1_", "independent_", "eps_", "evaluation_")) for item in errors),
            "resource_budget": not any(item.startswith(("gpu_", "ledger_gpu_", "optimization_", "resource_")) for item in errors),
        },
    }
    verification["content_sha256"] = _sha_object(verification)
    # Verification is deliberately repeatable so an evaluator can run the
    # tamper checks after editing a copied artifact; it never mutates source
    # predictions or training outputs.
    _write_json(run_dir / "verify.json", verification, overwrite=True)
    stage = {
        "schema_version": "safedrive.c3.stage_summary.v2",
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
        "reproduction_commands": config.get("reproduction_commands", []),
        "source_identity": {
            "code_sha256": config.get("code_sha256"),
            "git": config.get("git", {}),
            "release_id": config.get("release_id"),
            "model": config.get("model", {}),
        },
        "next_stage": "C4" if not errors else None,
    }
    stage["content_sha256"] = _sha_object(stage)
    _write_json(run_dir / "stage-summary.json", stage, overwrite=True)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "baseline", "train", "evaluate", "verify"))
    parser.add_argument("--run-id", default="")
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
    if not args.run_id:
        args.run_id = f"c3-repair-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
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
