"""Lossless same-forward SimLingo token contract for R2 K2 V4.

V3's mean64 representation is intentionally kept for backward compatibility.
V4 preserves the ordered route/speed query tokens emitted by the driving
adaptor.  The module is deliberately independent from runtime candidate
selection: it only validates, hashes and packages an observable forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from driving_vla.model.driving_feature import (
    DrivingFeatureError,
    _to_numpy,
    dump_raw_tokens_fp16,
)

V4_TOKEN_SCHEMA = "safedrive.driving_tokens.v4"
ROUTE_TOKEN_COUNT = 20
SPEED_TOKEN_COUNT = 10
TOTAL_TOKEN_COUNT = ROUTE_TOKEN_COUNT + SPEED_TOKEN_COUNT


def _v4_tensor_hash(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float16))
    return hashlib.sha256(arr.tobytes()).hexdigest()


class DrivingTokenError(DrivingFeatureError):
    """Raised when the V4 ordered-token ABI cannot be proven."""


def _finite_fp16(value: Any) -> np.ndarray:
    arr = np.asarray(_to_numpy(value))
    if arr.ndim == 3:
        if int(arr.shape[0]) != 1:
            raise DrivingTokenError(
                f"V4 requires one anchor forward, got batch={arr.shape[0]}"
            )
        arr = arr[0]
    if arr.ndim != 2:
        raise DrivingTokenError(f"V4 requires [tokens,channels], got shape={arr.shape}")
    if int(arr.shape[0]) != TOTAL_TOKEN_COUNT:
        raise DrivingTokenError(
            "V4 adaptor token count mismatch: "
            f"expected {TOTAL_TOKEN_COUNT}, got {arr.shape[0]}"
        )
    if int(arr.shape[1]) <= 0:
        raise DrivingTokenError("V4 adaptor channel dimension is empty")
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise DrivingTokenError("V4 adaptor tokens contain non-finite values")
    return arr


def split_ordered_tokens(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(all_tokens, route_tokens, speed_tokens)`` in adaptor order."""
    arr = _finite_fp16(value)
    return (
        arr,
        np.ascontiguousarray(arr[:ROUTE_TOKEN_COUNT]),
        np.ascontiguousarray(arr[ROUTE_TOKEN_COUNT:]),
    )


@dataclass(frozen=True)
class DrivingTokenBundleV4:
    """Immutable metadata plus ordered tokens from one SimLingo forward."""

    ok: bool
    schema_version: str = V4_TOKEN_SCHEMA
    adaptor_name: str = "driving"
    raw_shape: tuple[int, ...] = ()
    raw_dtype: str = ""
    raw_content_hash: str = ""
    channel_dim: int = 0
    route_token_count: int = ROUTE_TOKEN_COUNT
    speed_token_count: int = SPEED_TOKEN_COUNT
    raw_tensor_path: str = ""
    error: str = ""

    @classmethod
    def from_adaptor_output(
        cls,
        features: Any,
        *,
        raw_tensor_path: str | Path = "",
        require: bool = True,
    ) -> "DrivingTokenBundleV4":
        try:
            raw = _to_numpy(features)
            all_tokens, _route, _speed = split_ordered_tokens(raw)
            fp16 = np.asarray(raw, dtype=np.float16)
            path = str(raw_tensor_path or "")
            if path:
                _dumped_hash, shape, dtype = dump_raw_tokens_fp16(raw, path)
                if tuple(shape) != tuple(int(x) for x in raw.shape):
                    raise DrivingTokenError("V4 raw dump shape mismatch")
                # V4 uses a full SHA256 binding (legacy feature hashes remain
                # 16-hex for V0/V3 compatibility).
                raw_hash = _v4_tensor_hash(np.load(path, allow_pickle=False))
                raw_dtype = str(dtype)
            else:
                raw_hash = _v4_tensor_hash(fp16)
                raw_dtype = str(fp16.dtype)
            return cls(
                ok=True,
                raw_shape=tuple(int(x) for x in raw.shape),
                raw_dtype=raw_dtype,
                raw_content_hash=raw_hash,
                channel_dim=int(all_tokens.shape[1]),
                raw_tensor_path=path,
            )
        except Exception as exc:  # noqa: BLE001
            if require:
                if isinstance(exc, DrivingTokenError):
                    raise
                raise DrivingTokenError(f"V4 token extraction failed: {exc}") from exc
            return cls(ok=False, error=f"{type(exc).__name__}:{exc}")

    def require_ok(self) -> "DrivingTokenBundleV4":
        if not self.ok:
            raise DrivingTokenError(self.error or "V4 token bundle is not ok")
        if self.schema_version != V4_TOKEN_SCHEMA:
            raise DrivingTokenError("V4 token schema mismatch")
        if self.route_token_count != ROUTE_TOKEN_COUNT or self.speed_token_count != SPEED_TOKEN_COUNT:
            raise DrivingTokenError(
                "V4 token split metadata mismatch: expected "
                f"{ROUTE_TOKEN_COUNT}+{SPEED_TOKEN_COUNT}, got "
                f"{self.route_token_count}+{self.speed_token_count}"
            )
        if len(self.raw_shape) not in {2, 3}:
            raise DrivingTokenError(
                f"V4 raw shape must be [30,H] or [1,30,H], got {self.raw_shape}"
            )
        token_axis = 1 if len(self.raw_shape) == 3 else 0
        if len(self.raw_shape) == 3 and self.raw_shape[0] != 1:
            raise DrivingTokenError(
                f"V4 raw shape batch must be one, got {self.raw_shape}"
            )
        if self.raw_shape[token_axis] != TOTAL_TOKEN_COUNT:
            raise DrivingTokenError(
                f"V4 raw shape token count mismatch: {self.raw_shape}"
            )
        try:
            if np.dtype(self.raw_dtype) != np.dtype(np.float16):
                raise DrivingTokenError(
                    f"V4 raw tensor must be persisted as float16, got {self.raw_dtype}"
                )
        except TypeError as exc:
            raise DrivingTokenError(f"V4 raw dtype is malformed: {self.raw_dtype}") from exc
        if self.raw_shape[-1:] != (self.channel_dim,):
            raise DrivingTokenError("V4 token channel metadata mismatch")
        if not self.raw_content_hash:
            raise DrivingTokenError("V4 raw token hash missing")
        return self

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_shape"] = list(self.raw_shape)
        return payload

    def load_tokens(self) -> np.ndarray:
        """Load and revalidate the dumped [1,30,H] or [30,H] raw tensor."""
        self.require_ok()
        if not self.raw_tensor_path:
            raise DrivingTokenError("V4 raw tensor path is required for reload")
        arr = np.load(self.raw_tensor_path, allow_pickle=False)
        all_tokens, _route, _speed = split_ordered_tokens(arr)
        if tuple(int(x) for x in np.asarray(arr).shape) != self.raw_shape:
            raise DrivingTokenError("V4 raw token shape metadata mismatch")
        if np.asarray(arr).dtype != np.dtype(np.float16):
            raise DrivingTokenError("V4 raw token dtype metadata mismatch")
        actual = _v4_tensor_hash(np.load(self.raw_tensor_path, allow_pickle=False))
        if actual != self.raw_content_hash:
            raise DrivingTokenError("V4 raw token content hash mismatch")
        return all_tokens

    def token_type_ids(self) -> tuple[int, ...]:
        return (0,) * ROUTE_TOKEN_COUNT + (1,) * SPEED_TOKEN_COUNT

    def position_ids(self) -> tuple[int, ...]:
        return tuple(range(TOTAL_TOKEN_COUNT))


__all__ = [
    "DrivingTokenBundleV4",
    "DrivingTokenError",
    "ROUTE_TOKEN_COUNT",
    "SPEED_TOKEN_COUNT",
    "TOTAL_TOKEN_COUNT",
    "V4_TOKEN_SCHEMA",
    "split_ordered_tokens",
]
