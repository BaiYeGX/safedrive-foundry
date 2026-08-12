"""Auditable World-V0 checkpoint save/load/resume contract."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model_v0 import WorldV0, WorldV0Config

CHECKPOINT_SCHEMA = "safedrive.world_v0_checkpoint.v0"


class WorldCheckpointError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    directory: Path,
    *,
    model: WorldV0,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    data_hash: str,
    split_hash: str,
    code_hash: str,
    precision: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = directory / "world_v0.pt"
    state = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
    }
    torch.save(state, checkpoint_path)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config": model.config.to_dict(),
        "parameter_count": model.parameter_count,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "data_hash": str(data_hash),
        "split_hash": str(split_hash),
        "code_hash": str(code_hash),
        "precision": str(precision),
        "hardware": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "extra": dict(extra or {}),
    }
    (directory / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_checkpoint(
    directory: Path,
    *,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> tuple[WorldV0, dict[str, Any]]:
    directory = Path(directory)
    manifest = json.loads(
        (directory / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_path = directory / manifest["checkpoint_file"]
    if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise WorldCheckpointError("checkpoint SHA256 mismatch")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise WorldCheckpointError("unsupported checkpoint schema")
    model = WorldV0(WorldV0Config(**state["config"])).to(device)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        restore_rng_state(state["rng"])
    return model, {**manifest, "state": state}
