#!/usr/bin/env python3
"""Train K2 V3 semantic heads; never promotes a checkpoint to formal."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.r2_v3_dataset import (  # noqa: E402
    validate_v3_dataset_rows,
)
from driving_vla.model.checkpoint_contract import (  # noqa: E402
    STATUS_TRAINED_NOT_FORMAL,
    write_checkpoint_manifest,
)
from driving_vla.model.k2_v3_types import AlternativeKind  # noqa: E402
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteContextV3,
    TargetLaneSide,
)
from driving_vla.model.semantic_mode_heads import (  # noqa: E402
    KIND_ORDER,
    SEMANTIC_CONTEXT_DIM,
    SIDE_ORDER,
    SpatialSemanticHeadV3,
    semantic_context_vector,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dataset line {line_number} is not an object")
            rows.append(value)
    validate_v3_dataset_rows(rows)
    return rows


def _context(row: dict[str, Any]) -> list[float]:
    route = RouteContextV3.from_mapping(row["route_context"])
    return semantic_context_vector(
        native_path_xy=row["native_path_xy"],
        route_context=route,
        ego_v=float(row["ego_v"]),
        base_speed_mps=float(row["base_speed_mps"]),
        driving_feature=row.get("driving_feature"),
        observable_scene=row.get("observable_scene"),
    )


def _target_params(row: dict[str, Any]) -> list[float]:
    explicit = row.get("maneuver_params")
    if explicit is not None:
        if len(explicit) != 6:
            raise ValueError("maneuver_params must have six values")
        return [max(0.0, min(1.0, float(value))) for value in explicit]
    avoid_offset = float(row.get("avoid_offset_m", 0.5))
    temporal_scale = float(row.get("temporal_speed_scale", 0.0))
    depart_start = float(row.get("departure_start", 0.12))
    depart_end = float(row.get("departure_end", 0.35))
    rejoin_start = float(row.get("rejoin_start", 0.65))
    rejoin_end = float(row.get("rejoin_end", 0.90))
    return [
        max(0.0, min(1.0, (avoid_offset - 0.50) / 0.50)),
        max(0.0, min(1.0, temporal_scale / 0.80)),
        max(0.0, min(1.0, (depart_start - 0.05) / 0.20)),
        max(
            0.0,
            min(1.0, (depart_end - depart_start - 0.15) / 0.15),
        ),
        max(0.0, min(1.0, (rejoin_start - 0.50) / 0.20)),
        max(
            0.0,
            min(1.0, (rejoin_end - rejoin_start - 0.15) / 0.15),
        ),
    ]


def _batch(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "context": torch.tensor(
            [_context(row) for row in rows], dtype=torch.float32, device=device
        ),
        "kind": torch.tensor(
            [
                KIND_ORDER.index(AlternativeKind(str(row["alternative_kind"])))
                for row in rows
            ],
            dtype=torch.long,
            device=device,
        ),
        "side": torch.tensor(
            [
                SIDE_ORDER.index(TargetLaneSide(str(row["target_lane_side"])))
                for row in rows
            ],
            dtype=torch.long,
            device=device,
        ),
        "available": torch.tensor(
            [float(bool(row["alternative_available"])) for row in rows],
            dtype=torch.float32,
            device=device,
        ),
        "params": torch.tensor(
            [_target_params(row) for row in rows],
            dtype=torch.float32,
            device=device,
        ),
    }


@torch.inference_mode()
def evaluate(
    model: SpatialSemanticHeadV3,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0}
    model.eval()
    data = _batch(rows, device)
    output = model(data["context"])
    kind = output["kind_logits"].argmax(dim=-1)
    side = output["side_logits"].argmax(dim=-1)
    available = torch.sigmoid(output["avail_logit"]) >= 0.5
    target_available = data["available"] >= 0.5
    tp = int((available & target_available).sum().item())
    fn = int((~available & target_available).sum().item())
    tn = int((~available & ~target_available).sum().item())
    fp = int((available & ~target_available).sum().item())
    direction_mask = (data["side"] == SIDE_ORDER.index(TargetLaneSide.LEFT)) | (
        data["side"] == SIDE_ORDER.index(TargetLaneSide.RIGHT)
    )
    none_mask = data["kind"] == KIND_ORDER.index(AlternativeKind.NONE)
    return {
        "samples": len(rows),
        "semantic_accuracy": float((kind == data["kind"]).float().mean().item()),
        "direction_accuracy": float(
            (side[direction_mask] == data["side"][direction_mask])
            .float()
            .mean()
            .item()
        )
        if bool(direction_mask.any())
        else 0.0,
        "direction_denominator": int(direction_mask.sum().item()),
        "availability_recall": tp / max(tp + fn, 1),
        "availability_specificity": tn / max(tn + fp, 1),
        "none_closure_rate": float(
            (
                (kind[none_mask] == KIND_ORDER.index(AlternativeKind.NONE))
                & ~available[none_mask]
            )
            .float()
            .mean()
            .item()
        )
        if bool(none_mask.any())
        else 0.0,
        "none_denominator": int(none_mask.sum().item()),
        "parameter_mae": float(
            (output["maneuver_params"] - data["params"]).abs().mean().item()
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.data))
    dataset_audit = validate_v3_dataset_rows(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if args.overfit_32:
        if len(train_rows) < 32:
            raise ValueError("32-sample overfit requires at least 32 train rows")
        train_rows = train_rows[:32]
        dev_rows = train_rows
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model = SpatialSemanticHeadV3().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    positive = [row for row in train_rows if row["alternative_available"]]
    negative = [row for row in train_rows if not row["alternative_available"]]
    if not positive or not negative:
        raise ValueError("training requires available and unavailable examples")
    history: list[float] = []
    model.train()
    for step in range(args.steps):
        half = max(1, args.batch_size // 2)
        batch_rows = [
            positive[(step * half + index) % len(positive)]
            for index in range(half)
        ]
        batch_rows += [
            negative[(step * half + index) % len(negative)]
            for index in range(args.batch_size - half)
        ]
        data = _batch(batch_rows, device)
        output = model(data["context"])
        loss = 2.0 * F.cross_entropy(output["kind_logits"], data["kind"])
        loss = loss + 1.5 * F.cross_entropy(output["side_logits"], data["side"])
        loss = loss + 2.0 * F.binary_cross_entropy_with_logits(
            output["avail_logit"], data["available"]
        )
        available_mask = data["available"] >= 0.5
        if bool(available_mask.any()):
            loss = loss + F.smooth_l1_loss(
                output["maneuver_params"][available_mask],
                data["params"][available_mask],
            )
        none_mask = data["kind"] == KIND_ORDER.index(AlternativeKind.NONE)
        if bool(none_mask.any()):
            loss = loss + 0.5 * torch.sigmoid(
                output["avail_logit"][none_mask]
            ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.item()))

    tag = args.run_name or (
        ("overfit32" if args.overfit_32 else "development")
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir = Path(args.output_root) / tag
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "k2_v3_semantic_heads.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "context_dim": SEMANTIC_CONTEXT_DIM,
            "schema_version": "safedrive.k2_v3_semantic_head_checkpoint.v1",
            "dataset_sha256": _sha256(Path(args.data)),
            "steps": args.steps,
            "seed": args.seed,
        },
        checkpoint,
    )
    train_metrics = evaluate(model, train_rows, device)
    dev_metrics = evaluate(model, dev_rows, device)
    overfit_passed = (
        not args.overfit_32
        or (
            float(train_metrics["semantic_accuracy"]) >= 0.98
            and float(train_metrics["availability_recall"]) >= 0.98
            and float(train_metrics["availability_specificity"]) >= 0.98
        )
    )
    report = {
        "schema_version": "safedrive.k2_v3_training_report.v1",
        "status": STATUS_TRAINED_NOT_FORMAL,
        "formal_eligible": False,
        "run_name": tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset": str(Path(args.data)),
        "dataset_sha256": _sha256(Path(args.data)),
        "dataset_samples": int(dataset_audit["samples"]),
        "dataset_split_counts": dict(dataset_audit["split_counts"]),
        "dataset_lineages": int(dataset_audit["lineages"]),
        "dataset_routes": int(dataset_audit["routes"]),
        "dataset_actual_route_hashes": int(
            dataset_audit["actual_route_hashes"]
        ),
        "steps": args.steps,
        "final_loss": history[-1],
        "overfit_32": bool(args.overfit_32),
        "overfit_passed": overfit_passed,
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_manifest(
        output_dir / "CHECKPOINT_STATUS.json",
        checkpoint_path=checkpoint,
        status=STATUS_TRAINED_NOT_FORMAL,
        allowed_uses=[
            "offline_diagnostic",
            "development_live_smoke",
            "collection_anchor",
        ],
        forbidden_uses=[
            "formal_offline",
            "r2v3_formal",
            "world_campaign",
        ],
        reasons=[
            "training_does_not_promote",
            "requires_offline_long_smoke_core_blind_world_ready_audit",
        ],
        extra={
            "training_report": str(output_dir / "train_report.json"),
            "dataset_sha256": report["dataset_sha256"],
            "overfit_passed": overfit_passed,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overfit-32", action="store_true")
    args = parser.parse_args()
    report = train(args)
    return 0 if (not args.overfit_32 or report["overfit_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
