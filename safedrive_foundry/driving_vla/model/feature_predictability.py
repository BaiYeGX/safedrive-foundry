"""Leakage-safe linear probes for Spatial K2 driving features.

These probes answer a narrow question before changing the feature tap: does the
frozen feature contain linearly accessible scene information on a supported
train/validation distribution?  ``holdout_exam`` is counted for audit only and
is never used for fitting, threshold selection, or metrics.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

EXPECTED_CONFLICT_CLASSES = ("left", "right", "center", "none")


def canonical_conflict_side(value: Any) -> str:
    side = str(value or "none").strip().lower()
    if side in {"empty", "no_conflict", "no-alternative", "none"}:
        return "none"
    return side


def defensive_direction(row: Mapping[str, Any]) -> str | None:
    if not bool(row.get("alternative_available", False)):
        return None
    defensive = row.get("defensive")
    if not isinstance(defensive, Mapping):
        return None
    raw_d = defensive.get("raw_d")
    if not isinstance(raw_d, Sequence) or isinstance(raw_d, (str, bytes)):
        return None
    values = [float(value) for value in raw_d]
    if not values:
        return None
    peak = max(values, key=lambda value: abs(value))
    if abs(peak) < 1.0e-6:
        return "zero"
    return "positive_d" if peak > 0.0 else "negative_d"


def _feature(row: Mapping[str, Any]) -> np.ndarray:
    raw = row.get("driving_feature")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("sample missing driving_feature sequence")
    out = np.asarray([float(value) for value in raw], dtype=np.float64)
    if out.ndim != 1 or out.size < 1 or not np.isfinite(out).all():
        raise ValueError("invalid driving_feature")
    return out


def _linear_ridge_probe(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    *,
    label_fn: Callable[[Mapping[str, Any]], str | None],
    classes: Sequence[str],
    ridge: float = 1.0,
) -> dict[str, Any]:
    train = [(row, label_fn(row)) for row in train_rows]
    val = [(row, label_fn(row)) for row in val_rows]
    train = [(row, label) for row, label in train if label is not None]
    val = [(row, label) for row, label in val if label is not None]
    class_to_index = {label: index for index, label in enumerate(classes)}
    if any(label not in class_to_index for _row, label in train + val):
        unknown = sorted(
            {
                str(label)
                for _row, label in train + val
                if label not in class_to_index
            }
        )
        raise ValueError(f"unexpected probe labels: {unknown}")

    x_train = np.stack([_feature(row) for row, _label in train])
    x_val = np.stack([_feature(row) for row, _label in val])
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("train/val feature dimension mismatch")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1.0e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_train = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    x_val = np.concatenate([x_val, np.ones((len(x_val), 1))], axis=1)

    y_train = np.zeros((len(train), len(classes)), dtype=np.float64)
    for index, (_row, label) in enumerate(train):
        y_train[index, class_to_index[str(label)]] = 1.0
    reg = ridge * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + reg) @ x_train.T @ y_train
    pred = np.argmax(x_val @ weights, axis=1)
    truth = np.asarray(
        [class_to_index[str(label)] for _row, label in val], dtype=np.int64
    )
    recalls: dict[str, float] = {}
    for label, class_index in class_to_index.items():
        mask = truth == class_index
        recalls[label] = float(np.mean(pred[mask] == class_index))
    accuracy = float(np.mean(pred == truth))
    balanced_accuracy = float(np.mean(list(recalls.values())))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall_by_class": recalls,
        "n_train": len(train),
        "n_val": len(val),
        "feature_dim": int(x_train.shape[1] - 1),
        "fit_split": "train",
        "metric_split": "val",
        "holdout_exam_used": False,
        "method": "standardized_linear_ridge",
    }


def _task(
    *,
    name: str,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    label_fn: Callable[[Mapping[str, Any]], str | None],
    required_classes: Sequence[str],
    min_train_per_class: int,
    min_val_per_class: int,
    min_balanced_accuracy: float,
) -> dict[str, Any]:
    train_counts = Counter(
        label
        for row in train_rows
        if (label := label_fn(row)) is not None
    )
    val_counts = Counter(
        label for row in val_rows if (label := label_fn(row)) is not None
    )
    reasons: list[str] = []
    for label in required_classes:
        if int(train_counts.get(label, 0)) < int(min_train_per_class):
            reasons.append(
                f"train[{label}]={train_counts.get(label, 0)}"
                f"<{min_train_per_class}"
            )
        if int(val_counts.get(label, 0)) < int(min_val_per_class):
            reasons.append(
                f"val[{label}]={val_counts.get(label, 0)}<{min_val_per_class}"
            )
    result: dict[str, Any] = {
        "task": name,
        "required_classes": list(required_classes),
        "train_counts": dict(sorted(train_counts.items())),
        "val_counts": dict(sorted(val_counts.items())),
        "support_ok": not reasons,
        "support_reasons": reasons,
        "min_balanced_accuracy": float(min_balanced_accuracy),
    }
    if reasons:
        result["status"] = "BLOCKED_LABEL_SUPPORT"
        result["probe_ran"] = False
        return result
    metrics = _linear_ridge_probe(
        train_rows,
        val_rows,
        label_fn=label_fn,
        classes=required_classes,
    )
    result["probe_ran"] = True
    result["metrics"] = metrics
    result["status"] = (
        "PASS"
        if metrics["balanced_accuracy"] >= min_balanced_accuracy
        else "FEATURE_NOT_PREDICTIVE"
    )
    return result


def build_feature_predictability_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_train_per_class: int = 5,
    min_val_per_class: int = 3,
) -> dict[str, Any]:
    train = [row for row in rows if row.get("split_id") == "train"]
    val = [row for row in rows if row.get("split_id") == "val"]
    holdout = [row for row in rows if row.get("split_id") == "holdout_exam"]
    if not train or not val:
        raise ValueError("feature probe requires non-empty train and val splits")

    dims = {int(_feature(row).size) for row in train + val}
    if len(dims) != 1:
        raise ValueError(f"inconsistent train/val feature dimensions: {sorted(dims)}")

    tasks = [
        _task(
            name="alternative_available",
            train_rows=train,
            val_rows=val,
            label_fn=lambda row: (
                "available"
                if bool(row.get("alternative_available", False))
                else "unavailable"
            ),
            required_classes=("available", "unavailable"),
            min_train_per_class=min_train_per_class,
            min_val_per_class=min_val_per_class,
            min_balanced_accuracy=0.80,
        ),
        _task(
            name="conflict_side",
            train_rows=train,
            val_rows=val,
            label_fn=lambda row: canonical_conflict_side(row.get("conflict_side")),
            required_classes=EXPECTED_CONFLICT_CLASSES,
            min_train_per_class=min_train_per_class,
            min_val_per_class=min_val_per_class,
            min_balanced_accuracy=0.80,
        ),
        _task(
            name="defensive_direction",
            train_rows=train,
            val_rows=val,
            label_fn=defensive_direction,
            required_classes=("negative_d", "positive_d"),
            min_train_per_class=min_train_per_class,
            min_val_per_class=min_val_per_class,
            min_balanced_accuracy=0.80,
        ),
    ]
    if any(task["status"] == "BLOCKED_LABEL_SUPPORT" for task in tasks):
        status = "DATA_SUPPORT_BLOCKED"
    elif any(task["status"] == "FEATURE_NOT_PREDICTIVE" for task in tasks):
        status = "FEATURE_TAP_REPAIR_REQUIRED"
    else:
        status = "FEATURE_PROBE_PASS"
    return {
        "schema_version": "safedrive.r2x.feature_predictability.v1",
        "status": status,
        "feature_dim": next(iter(dims)),
        "split_counts": {
            "train": len(train),
            "val": len(val),
            "holdout_exam_audit_only": len(holdout),
        },
        "holdout_exam_policy": (
            "counted for isolation audit only; never fit, thresholded, or scored"
        ),
        "tasks": tasks,
    }
