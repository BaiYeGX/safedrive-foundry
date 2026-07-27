"""Runtime-observable driving features for Spatial K2 heads (X5A contract).

Three representations from the **same** SimLingo driving adaptor output:

1. **raw_tokens** — meta + content hash of pre-pool tensor (optional path to FP16 dump)
2. **full_pool** — mean over token/time dims, **keep all channels** (no truncate to 64)
3. **mean64** — full_pool truncated/padded to 64 (legacy baseline; may lose L/R)

Contract:
- Prefer same-forward ``outputs_by_adaptor['driving']``; never re-forward for features.
- **Fail-closed**: missing/empty features raise ``DrivingFeatureError`` when
  ``require=True`` (default for V2 / probe / collect). Silent zeros are forbidden
  on the real-feature path.
- Privileged actor futures may select teacher labels only; never enter head input.
- ``scene_proxy_feature`` remains bootstrap-only; must not be used as live SimLingo
  substitute without explicit ``source=scene_proxy_*`` tag.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

GEOM_DIM = 32
DRIVING_FEAT_DIM = 64
CONTEXT_DIM = GEOM_DIM + DRIVING_FEAT_DIM  # 96 (mean64 path)
FEATURE_PIPELINE_VERSION = "safedrive.driving_feature.v2"

SOURCE_MEAN64 = "simlingo_driving_mean64_v1"
SOURCE_FULL_POOL = "simlingo_driving_full_pool_v1"
SOURCE_RAW_META = "simlingo_driving_raw_tokens_v1"
SOURCE_SCENE_PROXY = "scene_proxy_v1"
SOURCE_EMPTY = "empty"


class DrivingFeatureError(RuntimeError):
    """Fail-closed feature extraction / binding error."""


def feature_vector_hash(vec: Sequence[float]) -> str:
    payload = ",".join(f"{float(v):.6f}" for v in vec)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tensor_content_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _to_numpy(features: Any) -> np.ndarray:
    if features is None:
        raise DrivingFeatureError("driving_adaptor_output_is_none")
    try:
        import torch

        if isinstance(features, torch.Tensor):
            return features.detach().float().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(features, dtype=np.float64)
    if arr.size == 0:
        raise DrivingFeatureError("driving_adaptor_output_empty")
    return arr


def mean_pool_tokens(arr: np.ndarray) -> np.ndarray:
    """Mean over all leading dims except the last channel dim.

    [B,T,C] → [C], [B,C] → [C], [C] → [C].
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0:
        raise DrivingFeatureError("driving_adaptor_scalar")
    while a.ndim > 1:
        a = a.mean(axis=0)
    return a.reshape(-1)


def truncate_or_pad(vec: np.ndarray, out_dim: int) -> list[float]:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size >= out_dim:
        out = v[:out_dim]
    else:
        out = np.concatenate([v, np.zeros(out_dim - int(v.size), dtype=np.float64)])
    return [float(x) if math.isfinite(float(x)) else 0.0 for x in out]


@dataclass(frozen=True)
class DrivingFeatureBundle:
    """Same-forward feature package for Spatial K2 (X5A)."""

    ok: bool
    pipeline_version: str = FEATURE_PIPELINE_VERSION
    adaptor_name: str = "driving"
    raw_shape: tuple[int, ...] = ()
    raw_dtype: str = ""
    raw_content_hash: str = ""
    full_pool: tuple[float, ...] = ()
    full_pool_hash: str = ""
    full_pool_dim: int = 0
    mean64: tuple[float, ...] = ()
    mean64_hash: str = ""
    source_mean64: str = SOURCE_EMPTY
    source_full_pool: str = SOURCE_EMPTY
    error: str = ""
    # Optional path if raw tensor was dumped to disk (collect script)
    raw_tensor_path: str = ""

    def require_ok(self) -> "DrivingFeatureBundle":
        if not self.ok:
            raise DrivingFeatureError(self.error or "driving_feature_not_ok")
        if not self.mean64 or not self.mean64_hash:
            raise DrivingFeatureError("mean64_missing")
        if not self.full_pool or not self.full_pool_hash:
            raise DrivingFeatureError("full_pool_missing")
        if not self.raw_content_hash:
            raise DrivingFeatureError("raw_hash_missing")
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # keep lists for JSON
        d["raw_shape"] = list(self.raw_shape)
        d["full_pool"] = list(self.full_pool)
        d["mean64"] = list(self.mean64)
        return d


def dump_raw_tokens_fp16(
    features: Any,
    path: str | Any,
) -> tuple[str, tuple[int, ...], str]:
    """Write raw driving tokens as float16 ``.npy`` for later re-pooling.

    Returns ``(raw_content_hash, shape, dtype_str)``.
    """
    from pathlib import Path as _Path

    arr = _to_numpy(features)
    fp16 = np.asarray(arr, dtype=np.float16)
    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), fp16)
    return tensor_content_hash(fp16), tuple(int(x) for x in fp16.shape), str(fp16.dtype)


def extract_driving_feature_bundle(
    features: Any,
    *,
    adaptor_name: str = "driving",
    mean64_dim: int = DRIVING_FEAT_DIM,
    require: bool = True,
    raw_tensor_path: str = "",
) -> DrivingFeatureBundle:
    """Build raw/full_pool/mean64 package from driving adaptor tensor."""
    try:
        arr = _to_numpy(features)
        if not np.isfinite(arr).all():
            # replace non-finite for hash stability then still fail if all bad
            if not np.isfinite(arr).any():
                raise DrivingFeatureError("driving_adaptor_all_non_finite")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        raw_hash = tensor_content_hash(arr.astype(np.float16, copy=False))
        path_str = str(raw_tensor_path or "")
        if path_str:
            dumped_hash, _, _ = dump_raw_tokens_fp16(arr, path_str)
            if dumped_hash != raw_hash:
                # recompute from same cast path for consistency
                raw_hash = dumped_hash
        full = mean_pool_tokens(arr)
        if full.size == 0:
            raise DrivingFeatureError("full_pool_empty")
        full_list = [float(x) if math.isfinite(float(x)) else 0.0 for x in full.tolist()]
        mean64 = truncate_or_pad(full, mean64_dim)
        # Reject pure zeros (silent null) when require
        if require and all(abs(x) < 1e-12 for x in mean64):
            raise DrivingFeatureError("mean64_all_zeros")
        return DrivingFeatureBundle(
            ok=True,
            adaptor_name=str(adaptor_name),
            raw_shape=tuple(int(x) for x in arr.shape),
            raw_dtype=str(arr.dtype),
            raw_content_hash=raw_hash,
            full_pool=tuple(full_list),
            full_pool_hash=feature_vector_hash(full_list),
            full_pool_dim=len(full_list),
            mean64=tuple(mean64),
            mean64_hash=feature_vector_hash(mean64),
            source_mean64=SOURCE_MEAN64,
            source_full_pool=SOURCE_FULL_POOL,
            raw_tensor_path=path_str,
        )
    except DrivingFeatureError as exc:
        if require:
            raise
        return DrivingFeatureBundle(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        if require:
            raise DrivingFeatureError(f"extract_failed:{type(exc).__name__}:{exc}") from exc
        return DrivingFeatureBundle(ok=False, error=f"{type(exc).__name__}:{exc}")


def pool_driving_adaptor_output(
    features: Any,
    *,
    out_dim: int = DRIVING_FEAT_DIM,
    require: bool = False,
) -> list[float]:
    """Legacy mean64 helper. Prefer ``extract_driving_feature_bundle``.

    When ``require=False`` (legacy default for V1 path), returns zeros on failure
    for backward compatibility — **V2/probe must use require=True via bundle**.
    """
    try:
        b = extract_driving_feature_bundle(
            features, mean64_dim=out_dim, require=True
        )
        return list(b.mean64)
    except DrivingFeatureError:
        if require:
            raise
        return [0.0] * out_dim


def scene_proxy_feature(
    *,
    conflict_side: str = "none",
    actor_rel_x_m: float = 0.0,
    actor_rel_y_m: float = 0.0,
    actor_speed_mps: float = 0.0,
    clearance_m: float = 20.0,
    ttc_s: float = 30.0,
    ego_v: float = 5.0,
    scenario_family: str = "synthetic",
    out_dim: int = DRIVING_FEAT_DIM,
) -> list[float]:
    """Bootstrap-only scene proxy (not a SimLingo substitute for live V2)."""
    side = str(conflict_side or "none").lower()
    side_oh = [
        1.0 if side == "left" else 0.0,
        1.0 if side == "right" else 0.0,
        1.0 if side == "center" else 0.0,
        1.0 if side in {"none", "empty", ""} else 0.0,
    ]
    fam = str(scenario_family or "synthetic").lower()
    fam_oh = [
        1.0 if "cut" in fam else 0.0,
        1.0 if "lead" in fam or "brake" in fam else 0.0,
        1.0 if "cross" in fam else 0.0,
        1.0 if "synth" in fam else 0.0,
    ]
    rx = max(-30.0, min(30.0, float(actor_rel_x_m))) / 30.0
    ry = max(-10.0, min(10.0, float(actor_rel_y_m))) / 10.0
    av = max(0.0, min(20.0, float(actor_speed_mps))) / 20.0
    cl = max(0.0, min(40.0, float(clearance_m))) / 40.0
    ttc = max(0.0, min(30.0, float(ttc_s))) / 30.0
    ev = max(0.0, min(20.0, float(ego_v))) / 20.0
    close = 1.0 if float(clearance_m) < 8.0 else 0.0
    urgent = 1.0 if float(ttc_s) < 4.0 else 0.0
    feats = side_oh + fam_oh + [rx, ry, av, cl, ttc, ev, close, urgent]
    base = list(feats)
    for k in (1, 2, 3):
        for v in base[:8]:
            feats.append(math.sin(k * math.pi * v))
            feats.append(math.cos(k * math.pi * v))
    while len(feats) < out_dim:
        feats.append(0.0)
    return [float(x) for x in feats[:out_dim]]


def scene_proxy_from_sample(sample: Mapping[str, Any]) -> list[float]:
    obs = sample.get("observables") or sample.get("scene_observables") or {}
    return scene_proxy_feature(
        conflict_side=str(obs.get("conflict_side") or sample.get("conflict_side") or "none"),
        actor_rel_x_m=float(obs.get("actor_rel_x_m", sample.get("actor_rel_x_m", 0.0)) or 0.0),
        actor_rel_y_m=float(obs.get("actor_rel_y_m", sample.get("actor_rel_y_m", 0.0)) or 0.0),
        actor_speed_mps=float(obs.get("actor_speed_mps", sample.get("actor_speed_mps", 0.0)) or 0.0),
        clearance_m=float(obs.get("clearance_m", sample.get("clearance_m", 20.0)) or 20.0),
        ttc_s=float(obs.get("ttc_s", sample.get("ttc_s", 30.0)) or 30.0),
        ego_v=float(sample.get("ego_v", 5.0) or 5.0),
        scenario_family=str(sample.get("scenario_family") or "synthetic"),
    )


def build_context_vector(
    native_path_xy: Sequence[tuple[float, float]],
    *,
    ego_v: float,
    base_speed_mps: float,
    driving_feature: Sequence[float] | None = None,
) -> list[float]:
    """Geometry (32) + driving feature (64) → 96-d head input (mean64 path)."""
    from driving_vla.model.spatial_mode_heads import geometry_context_vector

    geom = geometry_context_vector(
        native_path_xy, ego_v=ego_v, base_speed_mps=base_speed_mps, dim=GEOM_DIM
    )
    if driving_feature is None:
        raise DrivingFeatureError("build_context_requires_driving_feature")
    drive = [float(x) for x in driving_feature]
    if len(drive) < DRIVING_FEAT_DIM:
        drive = drive + [0.0] * (DRIVING_FEAT_DIM - len(drive))
    drive = drive[:DRIVING_FEAT_DIM]
    return geom + drive


def linear_probe_labels_accuracy(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    train_ratio: float = 0.7,
    seed: int = 0,
) -> dict[str, float]:
    """Tiny ridge linear probe for X5A signal check (numpy only)."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(y) != x.shape[0] or x.shape[0] < 4:
        return {"n": float(len(y)), "accuracy": 0.0, "n_classes": 0.0, "ok": 0.0}
    rng = np.random.RandomState(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    n_tr = max(2, int(train_ratio * len(idx)))
    tr, te = idx[:n_tr], idx[n_tr:]
    if te.size == 0:
        te = tr
    # one-vs-rest ridge
    classes = sorted(set(int(v) for v in y.tolist()))
    x_tr = x[tr]
    y_tr = y[tr]
    x_te = x[te]
    y_te = y[te]
    # standardize train
    mu = x_tr.mean(axis=0)
    sig = x_tr.std(axis=0) + 1e-6
    x_tr_n = (x_tr - mu) / sig
    x_te_n = (x_te - mu) / sig
    scores = []
    for c in classes:
        yt = (y_tr == c).astype(np.float64) * 2.0 - 1.0
        # ridge: (X'X + λI)^{-1} X'y
        xtx = x_tr_n.T @ x_tr_n + 1e-2 * np.eye(x_tr_n.shape[1])
        xty = x_tr_n.T @ yt
        try:
            w = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        scores.append(x_te_n @ w)
    pred = np.array(classes)[np.argmax(np.stack(scores, axis=1), axis=1)]
    acc = float((pred == y_te).mean()) if y_te.size else 0.0
    return {
        "n": float(x.shape[0]),
        "n_train": float(tr.size),
        "n_test": float(te.size),
        "n_classes": float(len(classes)),
        "accuracy": acc,
        "chance": 1.0 / max(len(classes), 1),
        "ok": 1.0 if acc >= (1.0 / max(len(classes), 1)) + 0.25 else 0.0,
    }
