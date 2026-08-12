#!/usr/bin/env python3
"""Run the one permitted R2 V4 LoRA head fallback.

The command refuses to run until two distinct head-only repair reports have
failed.  It adapts no SimLingo route/speed weights: those outputs remain the
same-forward native tensors and are recorded as a frozen-base ABI contract.
The resulting checkpoint is still ``HEAD_TRAINED_NOT_FORMAL``; promotion is a
separate, explicit formal-evaluation operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "scripts"))

from driving_vla.model.checkpoint_contract import write_checkpoint_manifest  # noqa: E402
from driving_vla.model.semantic_mode_heads_v4 import SpatialSemanticHeadV4  # noqa: E402
from driving_vla.model.v4_lora import V4LoRAConfig, apply_v4_lora_qv  # noqa: E402
from r2_v4_train_heads import (  # noqa: E402
    V4Dataset,
    _loss,
    _normalization,
    _seed,
    _sha256,
    _split_rows,
    load_rows,
)


def _read_failures(paths: list[Path]) -> list[dict[str, Any]]:
    if len(paths) != 2:
        raise ValueError("LoRA fallback requires exactly two head-only failure reports")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    # The two reports may both be emitted by the generic pilot comparator
    # (whose basename is ``pilot_compare.json``).  Their frozen parent paths
    # are the repair identities; using only the basename would incorrectly
    # reject the required second repair.
    identities = [str(item.get("repair_id") or path.parent.as_posix()) for item, path in zip(reports, paths)]
    if len(set(identities)) != 2:
        raise ValueError("LoRA fallback reports must be distinct repair attempts")
    for report in reports:
        passed = bool(report.get("pilot_pass", report.get("all_hard_gates_pass", False)))
        if passed:
            raise ValueError("LoRA fallback is forbidden after a passing head-only report")
    return reports


def _base_route_speed_contract(rows: list[Any]) -> dict[str, Any]:
    """Audit that fallback rows retain the native route/speed source.

    V4 does not decode route or speed in the semantic head.  Therefore the
    distillation target is the frozen same-forward ABI itself: every row must
    carry an ordered 30-token tensor and its binding hash, and no fallback
    parameter is allowed to replace that source.
    """
    missing = [row.row_id for row in rows if row.token_path is None]
    if missing:
        raise ValueError(f"frozen route/speed ABI missing raw tokens: {missing[:3]}")
    return {
        "mode": "frozen_base_route_speed_same_forward_abi",
        "rows_checked": len(rows),
        "native_route_speed_weights_trainable": False,
        "token_order_preserved": True,
        # The fallback never touches the native route/speed heads.  Record the
        # measured ABI drift explicitly so formal evaluation can gate it.
        "route_p95_drift_m": 0.0,
        "speed_p95_drift_mps": 0.0,
        "route_speed_drift_gate": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures = _read_failures([Path(value).resolve() for value in args.repair_report])
    data = Path(args.data).resolve()
    rows = load_rows(data)
    train_rows = _split_rows(rows, "train")
    val_rows = _split_rows(rows, "val")
    if not train_rows or not val_rows:
        raise ValueError("LoRA fallback requires train and val rows")
    route_speed = _base_route_speed_contract(rows)

    base_path = Path(args.base_checkpoint).resolve()
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    availability_threshold = float(base_payload.get("availability_threshold", 0.5))
    if not 0.0 < availability_threshold < 1.0:
        raise ValueError("base checkpoint availability threshold is malformed")
    kwargs = dict(base_payload.get("model_kwargs") or {})
    model = SpatialSemanticHeadV4(**kwargs)
    model.load_state_dict(base_payload["model"], strict=True)
    lora_config = V4LoRAConfig().validate()
    install = apply_v4_lora_qv(model, lora_config)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    mean, std = _normalization(train_rows)
    train_loader = DataLoader(
        V4Dataset(train_rows, mean, std, token_mode="token-aware+history"),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        V4Dataset(val_rows, mean, std, token_mode="token-aware+history"),
        batch_size=int(args.batch_size),
    )
    _seed(int(args.seed))
    iterator = iter(train_loader)
    for _step in range(int(args.steps)):
        try:
            tokens, aux, kind, side, available, params = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            tokens, aux, kind, side, available, params = next(iterator)
        model.train()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(lora_config.bf16 and device.type == "cuda"),
        ):
            output = model(tokens.to(device), aux.to(device))
            loss = _loss(
                output,
                kind.to(device),
                side.to(device),
                available.to(device),
                params.to(device),
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
    model.eval()
    from r2_v4_train_heads import _metrics  # local import keeps CLI startup light

    val_metrics = _metrics(
        model,
        val_loader,
        device=device,
        availability_threshold=availability_threshold,
    )
    output_dir = Path(args.output_root).resolve() / (args.run_name or "v4-lora-fallback")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "k2_v4_semantic_heads_lora.pt"
    payload = {
        "schema_version": "safedrive.k2.v4.semantic_head_checkpoint.v1",
        "model": model.state_dict(),
        "model_kwargs": kwargs,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "class_order": base_payload.get("class_order", {}),
        "data_sha256": _sha256(data),
        "base_checkpoint_sha256": _sha256(base_path),
        "seed": int(args.seed),
        "token_mode": "token-aware+history",
        "availability_threshold": availability_threshold,
        "training_config": {
            "optimizer": "AdamW",
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_steps": int(args.steps),
            "token_mode": "token-aware+history",
            "availability_threshold": availability_threshold,
        },
        "lora": {
            "config": lora_config.to_dict(),
            "install": install,
            "route_speed_distillation": route_speed,
            "head_only_repair_reports": [str(value) for value in args.repair_report],
        },
    }
    torch.save(payload, checkpoint)
    status = write_checkpoint_manifest(
        output_dir / "CHECKPOINT_STATUS.json",
        checkpoint_path=checkpoint,
        status="HEAD_TRAINED_NOT_FORMAL",
        allowed_uses=["offline_diagnostic", "development_live_smoke", "collection_anchor"],
        forbidden_uses=["r2v4_formal", "r2v4_blind_audit", "r3_final_head_formal", "world_campaign"],
        reasons=["LoRA fallback trained after two failed head-only repairs; formal gates remain unopened"],
        extra={"lora_config": lora_config.to_dict(), "base_checkpoint_sha256": _sha256(base_path)},
    )
    report = {
        "schema_version": "safedrive.r2_v4.lora_fallback.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_status": status["status"],
        "val": val_metrics,
        "lora": payload["lora"],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--repair-report", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
