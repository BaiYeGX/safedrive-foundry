#!/usr/bin/env python3
"""Create the non-formal V4 checkpoint used to collect calibration anchors.

The first V4 dataset cannot be collected with a head trained on itself.  This
checkpoint is a deterministic neutral head: it preserves the same-forward
SimLingo token/observable artifact contract while proposing no semantic
alternative.  It is never eligible for formal, blind, R3, or World use.

The token channel dimension is read from an ABI sample produced by the driving
runtime, never hard-coded in the checkpoint payload.  The sample is an ABI
probe only and is not copied into the training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.checkpoint_contract import write_checkpoint_manifest  # noqa: E402
from driving_vla.model.semantic_mode_heads_v4 import AUX_DIM, SpatialSemanticHeadV4  # noqa: E402
from driving_vla.model.v4_token_features import DrivingTokenBundleV4  # noqa: E402


SCHEMA = "safedrive.k2.v4.semantic_head_checkpoint.v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_dim(path: Path) -> int:
    array = np.load(path, allow_pickle=False)
    bundle = DrivingTokenBundleV4.from_adaptor_output(array)
    bundle.require_ok()
    return int(bundle.raw_shape[-1])


def make_checkpoint(token_sample: Path, output_dir: Path) -> dict[str, object]:
    token_sample = token_sample.resolve()
    output_dir = output_dir.resolve()
    if not token_sample.is_file():
        raise FileNotFoundError(token_sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "k2_v4_collection_neutral.pt"
    status = output_dir / "CHECKPOINT_STATUS.json"
    if checkpoint.exists() or status.exists():
        raise FileExistsError(f"refusing to overwrite collection checkpoint directory: {output_dir}")
    token_dim = _token_dim(token_sample)
    model = SpatialSemanticHeadV4(token_dim=token_dim, dropout=0.0)
    # Deterministic neutral output.  The kind logits tie at index 0 (NONE),
    # availability is effectively zero, and all bounded controls are zero.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.available.bias.fill_(-20.0)
        model.maneuver.bias.fill_(-20.0)
    payload = {
        "schema_version": SCHEMA,
        "model": model.state_dict(),
        "model_kwargs": {
            "token_dim": token_dim,
            "aux_dim": AUX_DIM,
            "dropout": 0.0,
        },
        "normalization": {
            "mean": [0.0] * AUX_DIM,
            "std": [1.0] * AUX_DIM,
        },
        "class_order": {
            "kind": ["NONE", "SPATIAL_AVOID", "SPATIAL_OVERTAKE", "TEMPORAL_YIELD"],
            "side": ["NONE", "LEFT", "RIGHT"],
        },
        "token_abi_source": {
            "path": str(token_sample),
            "sha256": _sha(token_sample),
            "shape": list(np.load(token_sample, allow_pickle=False).shape),
            "dtype": str(np.load(token_sample, allow_pickle=False).dtype),
        },
        "training_config": {
            "kind": "deterministic_neutral_collection_only",
            "availability_threshold": 0.5,
        },
        "availability_threshold": 0.5,
    }
    torch.save(payload, checkpoint)
    manifest = write_checkpoint_manifest(
        status,
        checkpoint_path=checkpoint,
        status="HEAD_TRAINED_NOT_FORMAL",
        allowed_uses=["development_live_smoke", "collection_anchor"],
        forbidden_uses=[
            "r2v4_formal",
            "r2v4_blind_audit",
            "r3_final_head_formal",
            "world_campaign",
        ],
        reasons=[
            "deterministic neutral V4 checkpoint for same-forward calibration anchors only",
            "not trained on and not eligible to evaluate formal or blind outcomes",
        ],
        extra={
            "head_schema": SCHEMA,
            "token_dim": token_dim,
            "token_abi_source_sha256": _sha(token_sample),
            "normalization_frozen": True,
            "availability_threshold": 0.5,
        },
    )
    result = {
        "schema_version": "safedrive.r2_v4.collection_checkpoint.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "status": manifest["status"],
        "token_dim": token_dim,
        "token_abi_source_sha256": _sha(token_sample),
    }
    (output_dir / "COLLECTION_CHECKPOINT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-sample", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = make_checkpoint(Path(args.token_sample), Path(args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
