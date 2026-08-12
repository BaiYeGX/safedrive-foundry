#!/usr/bin/env python3
"""Train/evaluate the R2 K2 V4 ordered-token semantic head.

The input is a JSONL produced by the V4 calibration collector.  Rows contain
``v4_raw_tensor_path`` (a dumped [1,30,H] or [30,H] tensor), ``v4_aux`` (the
178-dim observable context), group/split identity and semantic labels.  Test
rows are not evaluated unless ``--evaluate-test`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.semantic_mode_heads_v4 import (  # noqa: E402
    AUX_DIM,
    SpatialSemanticHeadV4,
)
from driving_vla.model.v4_token_features import (  # noqa: E402
    DrivingTokenBundleV4,
    split_ordered_tokens,
)
from driving_vla.model.checkpoint_contract import write_checkpoint_manifest  # noqa: E402

SCHEMA = "safedrive.k2.v4.semantic_head_checkpoint.v1"
KIND_ORDER = ("NONE", "SPATIAL_AVOID", "SPATIAL_OVERTAKE", "TEMPORAL_YIELD")
SIDE_ORDER = ("NONE", "LEFT", "RIGHT")
SPATIAL_KINDS = {"SPATIAL_AVOID", "SPATIAL_OVERTAKE"}
TOKEN_MODES = (
    "structured-only",
    "mean64",
    "token-aware",
    "token-aware+history",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_list(value: Any, expected: int, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a numeric sequence")
    result = [float(x) for x in value]
    if len(result) != expected or not np.isfinite(np.asarray(result)).all():
        raise ValueError(f"{field} must contain exactly {expected} finite values")
    return result


@dataclass(frozen=True)
class V4Row:
    token_path: Path | None
    tokens: np.ndarray | None
    aux: tuple[float, ...]
    kind: int
    side: int
    available: float
    params: tuple[float, ...]
    split: str
    group: str
    row_id: str
    guard_mpc_executable: bool = False
    legal_route_target: bool = False


def load_rows(path: Path) -> list[V4Row]:
    rows: list[V4Row] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        token_raw = value.get("v4_raw_tensor_path") or value.get("raw_tensor_path")
        token_path = None
        tokens = None
        if token_raw:
            token_path = Path(str(token_raw))
            if not token_path.is_absolute():
                token_path = (path.parent / token_path).resolve()
            if not token_path.is_file():
                raise ValueError(f"line {line_no}: missing token file {token_path}")
            declared_hash = str(
                value.get("v4_token_raw_content_hash")
                or value.get("raw_content_hash")
                or ""
            ).lower()
            if len(declared_hash) != 64:
                raise ValueError(f"line {line_no}: V4 token hash binding is required")
            if declared_hash:
                bundle = DrivingTokenBundleV4.from_adaptor_output(
                    np.load(token_path, allow_pickle=False),
                    raw_tensor_path=token_path,
                )
                if bundle.raw_content_hash.lower() != declared_hash:
                    raise ValueError(
                        f"line {line_no}: V4 token hash mismatch for {token_path}"
                    )
        elif value.get("v4_tokens") is not None:
            all_tokens, _route, _speed = split_ordered_tokens(
                np.asarray(value["v4_tokens"], dtype=np.float32)
            )
            declared_hash = str(
                value.get("v4_token_raw_content_hash")
                or value.get("raw_content_hash")
                or ""
            ).lower()
            if len(declared_hash) != 64:
                raise ValueError(f"line {line_no}: inline V4 token hash binding is required")
            actual_hash = __import__("hashlib").sha256(
                np.ascontiguousarray(all_tokens, dtype=np.float16).tobytes()
            ).hexdigest()
            if actual_hash != declared_hash:
                raise ValueError(f"line {line_no}: inline V4 token hash mismatch")
            tokens = all_tokens.astype(np.float32, copy=False)
        else:
            raise ValueError(f"line {line_no}: v4 raw tokens are required")
        kind_name = str(
            value.get("target_kind")
            or value.get("alternative_kind")
            or value.get("kind")
        )
        if kind_name not in KIND_ORDER:
            raise ValueError(f"line {line_no}: unsupported kind {kind_name}")
        side_name = str(
            value.get("target_side")
            or value.get("target_lane_side")
            or value.get("side", "NONE")
        )
        if side_name not in SIDE_ORDER:
            raise ValueError(f"line {line_no}: unsupported side {side_name}")
        aux = _as_float_list(value.get("v4_aux"), AUX_DIM, "v4_aux")
        # V4's six bounded decoder outputs are not optional metadata.  A row
        # without an explicit label would silently train the decoder toward
        # zero while still looking schema-valid, which is exactly the dead
        # output path the V4 contract is meant to prevent.  Unused dimensions
        # (for example lateral controls on TEMPORAL_YIELD) may legitimately
        # be zero, but the complete six-value label must be present so the
        # loss mask and decoder binding remain auditable.
        if "maneuver_params" not in value:
            raise ValueError(f"line {line_no}: maneuver_params supervision is required")
        params = _as_float_list(value.get("maneuver_params"), 6, "maneuver_params")
        rows.append(
            V4Row(
                token_path=token_path,
                tokens=tokens,
                aux=tuple(aux),
                kind=KIND_ORDER.index(kind_name),
                side=SIDE_ORDER.index(side_name),
                available=1.0 if bool(value.get("available", value.get("alternative_available", False))) else 0.0,
                params=tuple(params),
                split=str(value.get("split", "train")),
                group=str(value.get("group_key") or value.get("lineage_id") or value.get("row_id") or f"line-{line_no}"),
                row_id=str(value.get("row_id") or value.get("sample_id") or f"row-{line_no}"),
                guard_mpc_executable=bool(value.get("guard_mpc_executable", False)),
                legal_route_target=bool(value.get("legal_route_target", False)),
            )
        )
    if not rows:
        raise ValueError("V4 dataset is empty")
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row.group, set()).add(row.split)
    overlap = {key: value for key, value in groups.items() if len(value) > 1}
    if overlap:
        raise ValueError(f"V4 root-lineage split overlap: {sorted(overlap)[:5]}")
    return rows


class V4Dataset(Dataset[tuple[torch.Tensor, ...]]):
    """Materialize one of the pre-registered V4 pilot representations.

    The runtime model always receives the fixed ``[30,H]`` ABI and 178-dim
    observable context.  Ablations are implemented at the dataset boundary so
    every comparison still uses the same head, normalization and labels:

    * ``structured-only`` removes token content;
    * ``mean64`` removes token order by repeating the channel mean;
    * ``token-aware`` keeps ordered tokens but masks the 112-dim history;
    * ``token-aware+history`` is the full V4 representation.
    """

    def __init__(
        self,
        rows: Sequence[V4Row],
        aux_mean: np.ndarray,
        aux_std: np.ndarray,
        *,
        token_mode: str = "token-aware+history",
    ):
        self.rows = list(rows)
        self.aux_mean = aux_mean
        self.aux_std = aux_std
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"unsupported V4 token mode: {token_mode}")
        self.token_mode = token_mode
        self._loaded: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _tokens(self, index: int, row: V4Row) -> np.ndarray:
        if index not in self._loaded:
            if row.token_path is not None:
                loaded = np.load(row.token_path, allow_pickle=False)
                tokens, _route, _speed = split_ordered_tokens(loaded)
                self._loaded[index] = tokens.astype(np.float32, copy=False)
            else:
                assert row.tokens is not None
                self._loaded[index] = row.tokens
        return self._loaded[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.rows[index]
        token_array = self._tokens(index, row)
        if self.token_mode == "structured-only":
            token_array = np.zeros_like(token_array)
        elif self.token_mode == "mean64":
            token_array = np.broadcast_to(
                token_array.mean(axis=0, keepdims=True), token_array.shape
            ).copy()
        tokens = torch.as_tensor(token_array, dtype=torch.float32)
        aux = (np.asarray(row.aux, dtype=np.float32) - self.aux_mean) / self.aux_std
        if self.token_mode == "token-aware":
            # geometry (32) + navigation (24) precede the 5-frame ego/actor
            # history (55 + 57) in build_v4_aux_vector.
            aux = aux.copy()
            aux[56:168] = 0.0
        return (
            tokens,
            torch.as_tensor(aux, dtype=torch.float32),
            torch.tensor(row.kind, dtype=torch.long),
            torch.tensor(row.side, dtype=torch.long),
            torch.tensor(row.available, dtype=torch.float32),
            torch.as_tensor(row.params, dtype=torch.float32),
        )


def _split_rows(rows: Sequence[V4Row], split: str) -> list[V4Row]:
    return [row for row in rows if row.split == split]


def _train_loader(
    dataset: V4Dataset,
    rows: Sequence[V4Row],
    *,
    batch_size: int,
    group_balanced: bool,
) -> DataLoader:
    if not group_balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.group] = counts.get(row.group, 0) + 1
    weights = torch.as_tensor(
        [1.0 / max(counts[row.group], 1) for row in rows], dtype=torch.double
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _normalization(rows: Sequence[V4Row]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([row.aux for row in rows], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != AUX_DIM:
        raise ValueError("V4 aux normalization shape mismatch")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1.0e-4).astype(np.float32)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(
    model: SpatialSemanticHeadV4,
    loader: DataLoader,
    *,
    device: torch.device,
    availability_threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    kind_true: list[int] = []
    kind_pred: list[int] = []
    side_true: list[int] = []
    side_pred: list[int] = []
    avail_true: list[int] = []
    avail_pred: list[int] = []
    kind_conf: list[float] = []
    kind_correct: list[int] = []
    avail_conf: list[float] = []
    avail_correct: list[int] = []
    with torch.inference_mode():
        for tokens, aux, kind, side, available, _params in loader:
            output = model(tokens.to(device), aux.to(device))
            kind_prob = torch.softmax(output["kind_logits"], dim=-1)
            kp_tensor = kind_prob.argmax(dim=-1)
            kp = kp_tensor.cpu().tolist()
            sp = output["side_logits"].argmax(dim=-1).cpu().tolist()
            avail_prob = torch.sigmoid(output["avail_logit"])
            ap = (avail_prob >= float(availability_threshold)).long().cpu().tolist()
            kind_true.extend(kind.tolist())
            kind_pred.extend(kp)
            side_true.extend(side.tolist())
            side_pred.extend(sp)
            avail_true.extend((available >= 0.5).long().tolist())
            avail_pred.extend(ap)
            kind_conf.extend(kind_prob.max(dim=-1).values.cpu().tolist())
            kind_correct.extend((kp_tensor == kind.to(device)).long().cpu().tolist())
            avail_conf.extend(torch.maximum(avail_prob, 1.0 - avail_prob).cpu().tolist())
            avail_correct.extend((torch.as_tensor(ap, device=device) == (available >= 0.5).long().to(device)).long().cpu().tolist())
    if not kind_true:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0}
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for cls, name in enumerate(KIND_ORDER):
        tp = sum(t == cls and p == cls for t, p in zip(kind_true, kind_pred))
        fp = sum(t != cls and p == cls for t, p in zip(kind_true, kind_pred))
        fn = sum(t == cls and p != cls for t, p in zip(kind_true, kind_pred))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-9)
        recalls[name] = recall
        f1_values.append(f1)
    spatial = [
        (t, p)
        for t, p, k in zip(side_true, side_pred, kind_true)
        if KIND_ORDER[k] in SPATIAL_KINDS
    ]
    side_accuracy = (
        sum(t == p for t, p in spatial) / len(spatial) if spatial else 1.0
    )
    avail_tp = sum(t == 1 and p == 1 for t, p in zip(avail_true, avail_pred))
    avail_fp = sum(t == 0 and p == 1 for t, p in zip(avail_true, avail_pred))
    avail_fn = sum(t == 1 and p == 0 for t, p in zip(avail_true, avail_pred))
    avail_tn = sum(t == 0 and p == 0 for t, p in zip(avail_true, avail_pred))
    none_true = [t == 0 for t in kind_true]
    none_pred = [p == 0 for p in kind_pred]
    none_tp = sum(t and p for t, p in zip(none_true, none_pred))
    none_fp = sum((not t) and p for t, p in zip(none_true, none_pred))
    none_fn = sum(t and not p for t, p in zip(none_true, none_pred))
    none_tn = sum((not t) and (not p) for t, p in zip(none_true, none_pred))

    def _ece(confidence: list[float], correct: list[int]) -> float:
        if not confidence:
            return 0.0
        values = np.asarray(confidence, dtype=np.float64)
        labels = np.asarray(correct, dtype=np.float64)
        result = 0.0
        for lo in np.linspace(0.0, 1.0, 11)[:-1]:
            hi = min(1.0, lo + 0.1)
            mask = (values >= lo) & (values <= hi if hi >= 1.0 else values < hi)
            if mask.any():
                result += float(mask.mean()) * abs(float(values[mask].mean()) - float(labels[mask].mean()))
        return result

    return {
        "n": len(kind_true),
        "accuracy": sum(t == p for t, p in zip(kind_true, kind_pred)) / len(kind_true),
        "macro_f1": sum(f1_values) / len(f1_values),
        "kind_recall": recalls,
        "side_accuracy_spatial": side_accuracy,
        "availability_precision": avail_tp / max(avail_tp + avail_fp, 1),
        "availability_recall": avail_tp / max(avail_tp + avail_fn, 1),
        "availability_specificity": avail_tn / max(avail_tn + avail_fp, 1),
        "none_recall": none_tp / max(none_tp + none_fn, 1),
        "none_specificity": none_tn / max(none_tn + none_fp, 1),
        # For the CLEAR/NONE closure gate, "specificity" means no false
        # semantic alternative on a true clear row (the NONE recall/closure
        # rate).  Keep conventional one-vs-rest specificity separately above.
        "clear_none_specificity": none_tp / max(none_tp + none_fn, 1),
        "ece": 0.5 * (_ece(kind_conf, kind_correct) + _ece(avail_conf, avail_correct)),
        "availability_threshold": float(availability_threshold),
    }


def _loss(
    output: Mapping[str, torch.Tensor],
    kind: torch.Tensor,
    side: torch.Tensor,
    available: torch.Tensor,
    params: torch.Tensor,
    *,
    kind_weights: torch.Tensor | None = None,
    side_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    kind_loss = F.cross_entropy(output["kind_logits"], kind, weight=kind_weights)
    avail_loss = F.binary_cross_entropy_with_logits(output["avail_logit"], available)
    spatial_mask = torch.tensor(
        [KIND_ORDER[int(value)] in SPATIAL_KINDS for value in kind.detach().cpu()],
        dtype=torch.bool,
        device=kind.device,
    )
    side_loss = (
        F.cross_entropy(
            output["side_logits"][spatial_mask],
            side[spatial_mask],
            weight=side_weights,
        )
        if bool(spatial_mask.any())
        else output["side_logits"].sum() * 0.0
    )
    param_mask = torch.zeros_like(params, dtype=torch.bool)
    for index, value in enumerate(kind.detach().cpu().tolist()):
        kind_name = KIND_ORDER[int(value)]
        if kind_name in SPATIAL_KINDS:
            param_mask[index, :] = True
        elif kind_name == "TEMPORAL_YIELD":
            # Only the speed-scale parameter is supervised for a temporal
            # yield; lateral departure/rejoin parameters are decoder-inert.
            param_mask[index, 1] = True
    param_loss = (
        F.smooth_l1_loss(output["maneuver_params"][param_mask], params[param_mask])
        if bool(param_mask.any())
        else output["maneuver_params"].sum() * 0.0
    )
    return kind_loss + 0.5 * avail_loss + 0.5 * side_loss + 0.2 * param_loss


def _balanced_weights(rows: Sequence[V4Row], *, field: str, classes: int) -> torch.Tensor:
    """Return auditable inverse-frequency weights for the first repair.

    The pilot intentionally has only a few spatial fixtures.  Weighting is
    computed from train rows only and is frozen into the checkpoint metadata;
    it changes optimization pressure, not labels or split membership.
    """
    counts = np.zeros(classes, dtype=np.float64)
    for row in rows:
        value = int(getattr(row, field))
        counts[value] += 1.0
    if not np.all(counts > 0.0):
        raise ValueError(f"cannot balance {field}: missing classes {counts.tolist()}")
    weights = counts.sum() / (classes * counts)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def train(args: argparse.Namespace) -> dict[str, Any]:
    _seed(int(args.seed))
    data_path = Path(args.data).resolve()
    rows = load_rows(data_path)
    if args.overfit_32:
        rows = [row for row in rows if row.split == "train"][:32]
        if len(rows) < 4:
            raise ValueError("--overfit-32 requires at least four train rows")
        rows = [V4Row(**{**row.__dict__, "split": "train"}) for row in rows]
    train_rows = _split_rows(rows, "train")
    val_rows = _split_rows(rows, "val")
    test_rows = _split_rows(rows, "test")
    if not train_rows:
        raise ValueError("V4 dataset has no train rows")
    if not args.overfit_32 and not val_rows:
        raise ValueError("V4 development training requires a val split")
    mean, std = _normalization(train_rows)
    probe = train_rows[0]
    if probe.token_path is not None:
        probe_tokens, _route, _speed = split_ordered_tokens(
            np.load(probe.token_path, allow_pickle=False)
        )
    else:
        assert probe.tokens is not None
        probe_tokens = probe.tokens
    token_dim = int(probe_tokens.shape[1])
    device = torch.device(args.device)
    model = SpatialSemanticHeadV4(token_dim=token_dim, dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    token_mode = str(getattr(args, "token_mode", "token-aware+history"))
    if token_mode not in TOKEN_MODES:
        raise ValueError(f"unsupported V4 token mode: {token_mode}")
    train_ds = V4Dataset(train_rows, mean, std, token_mode=token_mode)
    val_ds = V4Dataset(val_rows or train_rows, mean, std, token_mode=token_mode)
    group_balanced = bool(getattr(args, "group_balanced", False))
    class_balanced = bool(getattr(args, "class_balanced", False))
    availability_threshold = float(getattr(args, "availability_threshold", 0.5))
    if not 0.0 < availability_threshold < 1.0:
        raise ValueError("availability_threshold must be in (0,1)")
    train_loader = _train_loader(
        train_ds,
        train_rows,
        batch_size=int(args.batch_size),
        group_balanced=group_balanced,
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
    kind_weights = (
        _balanced_weights(train_rows, field="kind", classes=len(KIND_ORDER)).to(device)
        if class_balanced
        else None
    )
    side_weights = (
        _balanced_weights(train_rows, field="side", classes=len(SIDE_ORDER)).to(device)
        if class_balanced
        else None
    )
    best_metric = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    no_improve = 0
    iterator = iter(train_loader)
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        tokens, aux, kind, side, available, params = batch
        model.train()
        output = model(tokens.to(device), aux.to(device))
        loss = _loss(
            output,
            kind.to(device),
            side.to(device),
            available.to(device),
            params.to(device),
            kind_weights=kind_weights,
            side_weights=side_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(args.eval_every) != 0 and step != int(args.steps):
            continue
        metrics = _metrics(
            model,
            val_loader,
            device=device,
            availability_threshold=availability_threshold,
        )
        score = float(metrics.get("macro_f1", 0.0))
        if score > best_metric + 1.0e-8:
            best_metric = score
            best_step = step
            no_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= int(args.patience_evals):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = _metrics(
        model,
        val_loader,
        device=device,
        availability_threshold=availability_threshold,
    )
    test_metrics = None
    if args.evaluate_test and test_rows:
        test_metrics = _metrics(
            model,
            DataLoader(
                V4Dataset(test_rows, mean, std, token_mode=token_mode),
                batch_size=int(args.batch_size),
            ),
            device=device,
            availability_threshold=availability_threshold,
        )
    output_dir = Path(args.output_root) / (args.run_name or f"v4-seed-{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "k2_v4_semantic_heads.pt"
    payload = {
        "schema_version": SCHEMA,
        "model": model.state_dict(),
        "model_kwargs": {"token_dim": token_dim, "aux_dim": AUX_DIM, "dropout": float(args.dropout)},
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "class_order": {"kind": list(KIND_ORDER), "side": list(SIDE_ORDER)},
        "data_sha256": _sha256(data_path),
        "seed": int(args.seed),
        "best_step": int(best_step),
        "token_mode": token_mode,
        "group_balanced": group_balanced,
        "class_balanced": class_balanced,
        "loss_weights": {
            "kind": kind_weights.detach().cpu().tolist() if kind_weights is not None else None,
            "side": side_weights.detach().cpu().tolist() if side_weights is not None else None,
        },
        "availability_threshold": availability_threshold,
        "training_config": {
            "optimizer": "AdamW",
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "max_steps": int(args.steps),
            "eval_every": int(args.eval_every),
            "patience_evals": int(args.patience_evals),
            "group_balanced": group_balanced,
            "class_balanced": class_balanced,
            "token_mode": token_mode,
            "availability_threshold": availability_threshold,
        },
    }
    torch.save(payload, checkpoint)
    status_manifest = write_checkpoint_manifest(
        output_dir / "CHECKPOINT_STATUS.json",
        checkpoint_path=checkpoint,
        status="HEAD_TRAINED_NOT_FORMAL",
        allowed_uses=["offline_diagnostic", "development_live_smoke", "collection_anchor"],
        forbidden_uses=["r2v4_formal", "r2v4_blind_audit", "r3_final_head_formal", "world_campaign"],
        reasons=["offline V4 head training complete; formal gates not yet verified"],
        extra={
            "head_schema": SCHEMA,
            "normalization_frozen": True,
            "availability_threshold": availability_threshold,
            "training_config": payload["training_config"],
            "input_manifest_sha256": payload["data_sha256"],
        },
    )
    report = {
        "schema_version": SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "data_sha256": payload["data_sha256"],
        "seed": int(args.seed),
        "token_mode": token_mode,
        "best_step": int(best_step),
        "availability_threshold": availability_threshold,
        "training_config": payload["training_config"],
        "checkpoint_status": status_manifest["status"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "overfit_32": bool(args.overfit_32),
        "val": val_metrics,
        "test": test_metrics,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overfit-32", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--token-mode", choices=TOKEN_MODES, default="token-aware+history")
    parser.add_argument("--group-balanced", action="store_true")
    parser.add_argument("--class-balanced", action="store_true")
    parser.add_argument("--availability-threshold", type=float, default=0.5)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
