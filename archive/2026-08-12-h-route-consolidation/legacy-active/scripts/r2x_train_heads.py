#!/usr/bin/env python3
"""Heads-only Spatial K2 residual training with scene/driving features.

Fixed slot semantics (no WTA across nominal/defensive):
  mode 0 = nominal_progress
  mode 1 = defensive_alternative

Diversity loss uses decoded max |d1-d0| and is only applied on eligible samples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.driving_feature import (  # noqa: E402
    CONTEXT_DIM,
    build_context_vector,
    observable_scene_from_sample,
    scene_proxy_from_sample,
)
from driving_vla.model.frenet_codec import decode_frenet_residual_path  # noqa: E402
from driving_vla.model.spatial_mode_heads import (  # noqa: E402
    SpatialModeResidualHead,
    defensive_speed_margin_loss,
    decoded_diversity_floor_loss,
    decoded_lateral_smoothness_loss,
    decoded_lateral_separation,
)

DEFAULT_DATA = ROOT / "docs/runtime-evidence/r2x-training/dataset-v2/samples.jsonl"
CKPT_ROOT = ROOT / "docs/runtime-evidence/r2x-training/checkpoints"


def load_split(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s.get("split_id") == split:
                rows.append(s)
    return rows


def context_from_sample(sample: dict) -> list[float]:
    drive = sample.get("driving_feature")
    if drive is None:
        drive = scene_proxy_from_sample(sample)
    return build_context_vector(
        sample["native_path_xy"],
        ego_v=float(sample["ego_v"]),
        base_speed_mps=float(sample["base_speed_mps"]),
        driving_feature=drive,
        observable_scene=observable_scene_from_sample(sample),
    )


def sample_tensors(sample: dict, mode: int, n_path: int = 20):
    ctx = context_from_sample(sample)
    key = "nominal" if mode == 0 else "defensive"
    lab = sample[key]
    ds = list(lab["raw_delta_s"][:n_path])
    dd = list(lab["raw_d"][:n_path])
    while len(ds) < n_path:
        ds.append(0.5)
        dd.append(0.0)
    avail = 1.0 if (mode == 0 or sample.get("alternative_available")) else 0.0
    eligible = 1.0 if sample.get("alternative_available") else 0.0
    return (
        torch.tensor(ctx, dtype=torch.float32),
        torch.tensor(ds, dtype=torch.float32),
        torch.tensor(dd, dtype=torch.float32),
        torch.tensor([float(lab.get("speed_scale", 1.0))], dtype=torch.float32),
        torch.tensor([avail], dtype=torch.float32),
        torch.tensor([eligible], dtype=torch.float32),
    )


def decoded_max_lat_sep(
    native: list,
    raw_d0: torch.Tensor,
    raw_d1: torch.Tensor,
    raw_ds0: torch.Tensor,
    raw_ds1: torch.Tensor,
) -> float:
    """max |d1-d0| after Frenet envelope decode (eligible diversity metric)."""
    n = min(len(native), int(raw_d0.numel()), int(raw_d1.numel()))
    if n < 2:
        return 0.0
    try:
        _, _, d0 = decode_frenet_residual_path(
            native,
            [float(x) for x in raw_ds0[:n].tolist()],
            [float(x) for x in raw_d0[:n].tolist()],
            max_lateral_m=1.0,
        )
        _, _, d1 = decode_frenet_residual_path(
            native,
            [float(x) for x in raw_ds1[:n].tolist()],
            [float(x) for x in raw_d1[:n].tolist()],
            max_lateral_m=1.0,
        )
        return max(abs(float(a) - float(b)) for a, b in zip(d0, d1))
    except Exception:
        return float((raw_d0[:n] - raw_d1[:n]).abs().max().item())


def train(
    *,
    data_path: Path,
    steps: int = 800,
    lr: float = 1e-3,
    device: str = "cpu",
    overfit_n: int = 0,
    run_name: str = "",
    resume: str | None = None,
    seed: int = 29,
    batch_size: int = 16,
    diversity_target_m: float = 0.70,
    smoothness_weight: float = 0.25,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_rows = load_split(data_path, "train")
    val_rows = load_split(data_path, "val")
    assert train_rows, "no train samples"
    tag = run_name or (
        f"overfit{overfit_n}" if overfit_n > 0 else "full"
    ) + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ckpt_dir = CKPT_ROOT / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if overfit_n > 0:
        # prefer eligible samples for overfit
        elig = [s for s in train_rows if s.get("alternative_available")]
        pool = elig if len(elig) >= overfit_n else train_rows
        train_rows = pool[:overfit_n]

    model = SpatialModeResidualHead(n_path=20, context_dim=CONTEXT_DIM, hidden=128).to(device)
    if resume and Path(resume).is_file():
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    no_alt = [s for s in train_rows if not s.get("alternative_available")]
    has_alt = [s for s in train_rows if s.get("alternative_available")]
    batch_size = max(2, int(batch_size))
    model.train()
    hist = []
    for step in range(steps):
        n_positive = batch_size // 2
        n_negative = batch_size - n_positive
        if not has_alt:
            n_positive = 0
            n_negative = batch_size
        if not no_alt:
            n_positive = batch_size
            n_negative = 0
        batch = [
            has_alt[(step * max(n_positive, 1) + index) % len(has_alt)]
            for index in range(n_positive)
        ] + [
            no_alt[(step * max(n_negative, 1) + index) % len(no_alt)]
            for index in range(n_negative)
        ]
        # Deterministic rotation avoids a fixed positive/negative block order.
        rotation = step % len(batch)
        batch = batch[rotation:] + batch[:rotation]
        mode0 = [sample_tensors(sample, 0) for sample in batch]
        mode1 = [sample_tensors(sample, 1) for sample in batch]

        def _stack(items, index):
            return torch.stack([item[index] for item in items]).to(device)

        ctx0, ds0, dd0, sp0, av0, elig = (
            _stack(mode0, index) for index in range(6)
        )
        ctx1, ds1, dd1, sp1, av1, _ = (
            _stack(mode1, index) for index in range(6)
        )
        # Candidate-blind feature dropout forces the small scene adaptor to
        # learn current actor geometry instead of memorizing one visual
        # lineage.  Geometry + observable actor slots [0:32] are preserved.
        if step % 4 == 0:
            ctx0 = ctx0.clone()
            ctx1 = ctx1.clone()
            ctx0[:, 32:] = 0.0
            ctx1[:, 32:] = 0.0
        modes0 = torch.zeros(len(batch), dtype=torch.long, device=device)
        modes1 = torch.ones(len(batch), dtype=torch.long, device=device)
        o0 = model(ctx0, modes0)
        o1 = model(ctx1, modes1)

        loss = F.mse_loss(o0["raw_delta_s"], ds0)
        loss = loss + 5.0 * F.mse_loss(o0["raw_d"], dd0)
        # Runtime candidate 0 is the exact nominal anchor with speed scale 1.0.
        loss = loss + 1.0 * F.mse_loss(
            o0["speed_scale"], torch.ones_like(o0["speed_scale"])
        )
        # Nominal first samples must stay on path (near-field continuity)
        loss = loss + 4.0 * o0["raw_d"][:, :3].abs().mean()

        loss = loss + F.mse_loss(o1["raw_delta_s"], ds1)
        loss = loss + 5.0 * F.mse_loss(o1["raw_d"], dd1)
        loss = loss + 2.0 * F.mse_loss(o1["speed_scale"], sp1.squeeze(-1))
        # Defensive near-field: first 2 samples |d| ≤ ~0.15 (Guard inter-cand ≤0.20)
        loss = loss + 3.0 * torch.relu(o1["raw_d"][:, :2].abs() - 0.12).mean()
        # Balanced batches make availability supervision stable without letting
        # either the positive or the no-alternative class erase the other.
        loss = loss + 4.0 * F.binary_cross_entropy_with_logits(
            o1["avail_logit"], av1.squeeze(-1)
        )

        eligible_mask = elig.squeeze(-1) > 0.5
        no_alt_mask = ~eligible_mask
        if bool(eligible_mask.any()):
            # Preserve the teacher's committed spatial horizon.  Raw Δs is
            # decoded with softplus; matching only raw logits allowed the
            # student to shorten a 19 m route to ~13 m and then brake before
            # completing the lateral maneuver.
            predicted_horizon = F.softplus(
                o1["raw_delta_s"][eligible_mask, 1:]
            ).sum(dim=1)
            teacher_horizon = F.softplus(
                ds1[eligible_mask, 1:]
            ).sum(dim=1)
            loss = loss + 5.0 * torch.relu(
                0.90 * teacher_horizon - predicted_horizon
            ).mean()
            loss = loss + 2.0 * defensive_speed_margin_loss(
                o1["speed_scale"][eligible_mask],
                sp1.squeeze(-1)[eligible_mask],
            )
            # Match the runtime codec's tanh + near-field envelope.  The
            # 0.65 m target leaves margin above the frozen 0.50 m Guard floor.
            loss = loss + 8.0 * decoded_diversity_floor_loss(
                o0["raw_d"][eligible_mask],
                o1["raw_d"][eligible_mask],
                target_m=float(diversity_target_m),
                ramp_points=6,
            )
            loss = loss + 4.0 * F.mse_loss(
                torch.tanh(o1["raw_d"][eligible_mask]),
                torch.tanh(dd1[eligible_mask]),
            )
            # push defensive |d| scale (Huber-ish toward teacher)
            loss = loss + 1.5 * torch.relu(
                1.0 - o1["raw_d"][eligible_mask].abs().mean()
            )
            loss = loss + 2.0 * F.huber_loss(
                o1["raw_d"][eligible_mask], dd1[eligible_mask], delta=0.5
            )
            # Keep the learned alternative easy for PathManager/MPC to track.
            # This is a decoded Frenet curvature proxy, not a runtime rescue:
            # the neural head must itself emit the smoother residual.
            loss = loss + float(smoothness_weight) * decoded_lateral_smoothness_loss(
                o1["raw_d"][eligible_mask],
            )
            # push avail logit up for eligible
            loss = loss + 2.0 * torch.relu(
                1.5 - o1["avail_logit"][eligible_mask]
            ).mean()
        if bool(no_alt_mask.any()):
            loss = loss + 3.0 * o1["raw_d"][no_alt_mask].abs().mean()
            # mild push avail logit down when no-alt (specificity)
            loss = loss + 0.5 * torch.relu(
                o1["avail_logit"][no_alt_mask] + 0.5
            ).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        hist.append(float(loss.item()))
        if step % 50 == 0:
            print(f"step={step} loss={hist[-1]:.4f}", flush=True)

    model.eval()
    seps = []
    with torch.no_grad():
        for s in val_rows or train_rows[:8]:
            ctx = torch.tensor([context_from_sample(s)], dtype=torch.float32, device=device)
            o0 = model(ctx, torch.tensor([0], device=device))
            o1 = model(ctx, torch.tensor([1], device=device))
            seps.append(
                decoded_max_lat_sep(
                    s["native_path_xy"],
                    o0["raw_d"][0].cpu(),
                    o1["raw_d"][0].cpu(),
                    o0["raw_delta_s"][0].cpu(),
                    o1["raw_delta_s"][0].cpu(),
                )
            )
    mean_sep = sum(seps) / max(len(seps), 1)
    ckpt = ckpt_dir / "spatial_heads_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "steps": steps,
            "context_dim": CONTEXT_DIM,
            "n_train": len(train_rows),
            "data_path": str(data_path.as_posix()),
            "run_name": tag,
        },
        ckpt,
    )
    # also update pointer for offline eval default
    pointer = CKPT_ROOT / "spatial_heads_last.pt"
    pointer.write_bytes(ckpt.read_bytes())

    # lineage hashes
    import hashlib

    def _fsha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    ckpt_sha = _fsha(ckpt)
    data_sha = _fsha(data_path) if data_path.is_file() else ""
    # Training completion ≠ formal acceptance. Formal uses require offline gates
    # (and later X5H/R2-K contracts). Do not auto-promote to formal OK.
    report = {
        "status": "TRAINED",
        "run_name": tag,
        "steps": steps,
        "final_loss": hist[-1] if hist else None,
        "val_mean_decoded_lat_sep_m": mean_sep,
        "checkpoint": str(ckpt.as_posix()),
        "checkpoint_sha256": ckpt_sha,
        "pointer": str(pointer.as_posix()),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "device": device,
        "context_dim": CONTEXT_DIM,
        "data_path": str(data_path.as_posix()),
        "data_sha256": data_sha,
        "fixed_slot_supervision": True,
        "wta_across_slots": False,
        "seed": int(seed),
        "decoded_diversity_target_m": float(diversity_target_m),
        "decoded_lateral_smoothness_weight": float(smoothness_weight),
        "batch_size": int(batch_size),
        "balanced_availability_batches": True,
        "driving_feature_dropout_fraction": 0.25,
        "observable_scene_adaptor": True,
        "eligible_spatial_horizon_loss": True,
        "eligible_spatial_horizon_target_ratio": 0.90,
        "formal_eligible": False,
        "formal_note": (
            "TRAINED_NOT_FORMAL: offline gates / lineage / holdout isolation "
            "must pass before status=OK and formal uses"
        ),
    }
    (ckpt_dir / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (CKPT_ROOT / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    from driving_vla.model.checkpoint_contract import write_checkpoint_manifest

    # Diagnostic-only by default. Never auto-allow x5h_acceptance / r2k_pilot.
    diag_allowed = [
        "offline_diagnostic",
        "historical_comparison",
    ]
    forbidden_formal = [
        "formal_offline",
        "x5h_acceptance",
        "r2k_pilot",
    ]
    extra_lineage = {
        "train_report": str((ckpt_dir / "train_report.json").as_posix()),
        "data_path": str(data_path.as_posix()),
        "data_sha256": data_sha,
        "val_mean_decoded_lat_sep_m": mean_sep,
        "context_dim": CONTEXT_DIM,
        "run_name": tag,
        "source_checkpoint": str(ckpt.as_posix()),
        "source_sha256": ckpt_sha,
        "formal_eligible": False,
    }
    man = write_checkpoint_manifest(
        ckpt_dir / "CHECKPOINT_STATUS.json",
        checkpoint_path=ckpt,
        status="HEAD_TRAINED_NOT_FORMAL",
        allowed_uses=diag_allowed,
        forbidden_uses=forbidden_formal,
        reasons=[
            "heads_trained_sha256_bound",
            "formal_requires_offline_gates_and_lineage",
            "do_not_auto_promote_from_train_report",
        ],
        extra=extra_lineage,
    )
    # Keep last-trained pointer for diagnostics only (same bytes as run ckpt)
    last_diag = CKPT_ROOT / "spatial_heads_last.pt"
    last_diag.write_bytes(ckpt.read_bytes())
    # If a prior FORMAL_HEAD_REQUIRED alias exists, demote its manifest
    # (file may remain for historical SHA checks; uses blocked).
    alias = CKPT_ROOT / "FORMAL_HEAD_REQUIRED.pt"
    if alias.is_file() or True:
        # Always rewrite root status to not claim formal OK for last train.
        write_checkpoint_manifest(
            CKPT_ROOT / "CHECKPOINT_STATUS.json",
            checkpoint_path=alias if alias.is_file() else last_diag,
            status="HEAD_TRAINED_NOT_FORMAL",
            allowed_uses=diag_allowed,
            forbidden_uses=forbidden_formal,
            reasons=[
                "train_script_does_not_emit_formal_ok",
                f"source_run={tag}",
                "offline_gates_not_satisfied_or_not_checked_here",
            ],
            extra=extra_lineage,
        )
    report["checkpoint_status"] = man
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overfit-n", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", default="")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--run-name", default="")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--diversity-target-m", type=float, default=0.70)
    ap.add_argument("--smoothness-weight", type=float, default=0.25)
    args = ap.parse_args()
    data = Path(args.data)
    if not data.is_file():
        # fall back to v1
        alt = ROOT / "docs/runtime-evidence/r2x-training/dataset-v1/samples.jsonl"
        if alt.is_file():
            data = alt
        else:
            print("missing dataset; run r2x_scene_teacher_generate.py first", flush=True)
            return 2
    train(
        data_path=data,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        overfit_n=args.overfit_n,
        run_name=args.run_name,
        resume=args.resume or None,
        seed=args.seed,
        batch_size=args.batch_size,
        diversity_target_m=args.diversity_target_m,
        smoothness_weight=args.smoothness_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
