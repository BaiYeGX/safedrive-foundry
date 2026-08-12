"""Frozen tensor and identity contracts for ActionBranchDatasetV0/World-V0."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "safedrive.action_branch_dataset.v0"
SCHEMA_VERSION_V1 = "safedrive.action_branch_dataset.v1"
WORLD_BATCH_SCHEMA = "safedrive.world_batch.v0"
WORLD_PREDICTION_SCHEMA = "safedrive.world_prediction.v0"

K = 2
T = 10
DT_S = 0.25
HISTORY = 5
MAX_ACTORS = 8
MAX_ROAD_POLYLINES = 3
MAX_ROAD_POINTS = 16

EGO_FEATURES = 11
ACTOR_FEATURES = 14
ROAD_FEATURES = 6
CANDIDATE_FEATURES = 8
FUTURE_FEATURES = 6
OUTCOME_FEATURES = 8


class WorldContractError(ValueError):
    """Raised when a dataset or World runtime payload violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type = np.float32,
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != shape:
        raise WorldContractError(f"{name} shape {arr.shape} != {shape}")
    return np.ascontiguousarray(arr)


def _binary_mask(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = _array(value, name=name, shape=shape, dtype=np.bool_)
    return arr


def _finite_on_mask(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
    expand_dims: int = 1,
) -> None:
    expanded = mask
    for _ in range(expand_dims):
        expanded = np.expand_dims(expanded, axis=-1)
    if not np.isfinite(values[expanded.repeat(values.shape[-1], axis=-1)]).all():
        raise WorldContractError(f"{name} has non-finite values on valid mask")


def _require_hash(value: str, name: str) -> str:
    text = str(value or "").strip()
    if len(text) < 16 or any(c not in "0123456789abcdef" for c in text.lower()):
        raise WorldContractError(f"{name} must be a lowercase hexadecimal content hash")
    return text.lower()


@dataclass(frozen=True)
class SampleIdentity:
    sample_id: str
    pair_id: str
    scenario_id: str
    seed_id: str
    group_key: str
    family: str
    map_name: str
    initial_state_hash: str
    observation_hash: str
    anchor_artifact_hash: str
    model_hash: str
    guard_hash: str
    executor_hash: str
    source_manifest_hash: str

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "pair_id",
            "scenario_id",
            "seed_id",
            "group_key",
            "family",
            "map_name",
        ):
            if not str(getattr(self, name)).strip():
                raise WorldContractError(f"identity.{name} must be non-empty")
        for name in (
            "initial_state_hash",
            "observation_hash",
            "anchor_artifact_hash",
            "model_hash",
            "guard_hash",
            "executor_hash",
            "source_manifest_hash",
        ):
            _require_hash(str(getattr(self, name)), f"identity.{name}")

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleIdentity":
        return cls(**{name: str(value[name]) for name in cls.__dataclass_fields__})


@dataclass
class ActionBranchSample:
    """One anchor with up to two paired branches.

    Oracle-only arrays are labels and never accepted by :class:`WorldBatch`.
    """

    identity: SampleIdentity
    ego_history: np.ndarray
    ego_history_mask: np.ndarray
    actor_history: np.ndarray
    actor_history_mask: np.ndarray
    road: np.ndarray
    road_mask: np.ndarray
    candidates: np.ndarray
    candidate_mask: np.ndarray
    actor_future: np.ndarray
    actor_future_mask: np.ndarray
    outcomes: np.ndarray
    outcome_mask: np.ndarray
    rank_target: float
    rank_mask: bool
    rank_weight: float
    tie_target: bool
    comparable: bool
    unavailable_reasons: tuple[str | None, str | None] = (None, None)
    audit: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.ego_history = _array(
            self.ego_history, name="ego_history", shape=(HISTORY, EGO_FEATURES)
        )
        self.ego_history_mask = _binary_mask(
            self.ego_history_mask, name="ego_history_mask", shape=(HISTORY,)
        )
        self.actor_history = _array(
            self.actor_history,
            name="actor_history",
            shape=(MAX_ACTORS, HISTORY, ACTOR_FEATURES),
        )
        self.actor_history_mask = _binary_mask(
            self.actor_history_mask,
            name="actor_history_mask",
            shape=(MAX_ACTORS, HISTORY),
        )
        self.road = _array(
            self.road,
            name="road",
            shape=(MAX_ROAD_POLYLINES, MAX_ROAD_POINTS, ROAD_FEATURES),
        )
        self.road_mask = _binary_mask(
            self.road_mask,
            name="road_mask",
            shape=(MAX_ROAD_POLYLINES, MAX_ROAD_POINTS),
        )
        self.candidates = _array(
            self.candidates, name="candidates", shape=(K, T, CANDIDATE_FEATURES)
        )
        self.candidate_mask = _binary_mask(
            self.candidate_mask, name="candidate_mask", shape=(K,)
        )
        self.actor_future = _array(
            self.actor_future,
            name="actor_future",
            shape=(K, MAX_ACTORS, T, FUTURE_FEATURES),
        )
        self.actor_future_mask = _binary_mask(
            self.actor_future_mask,
            name="actor_future_mask",
            shape=(K, MAX_ACTORS, T),
        )
        self.outcomes = _array(
            self.outcomes, name="outcomes", shape=(K, OUTCOME_FEATURES)
        )
        self.outcome_mask = _binary_mask(
            self.outcome_mask, name="outcome_mask", shape=(K,)
        )

        _finite_on_mask(
            self.ego_history,
            self.ego_history_mask,
            name="ego_history",
        )
        _finite_on_mask(
            self.actor_history,
            self.actor_history_mask,
            name="actor_history",
        )
        _finite_on_mask(self.road, self.road_mask, name="road")
        if self.candidate_mask.any():
            valid_candidates = self.candidates[self.candidate_mask]
            if not np.isfinite(valid_candidates).all():
                raise WorldContractError("candidates have non-finite values on valid mask")
        _finite_on_mask(
            self.actor_future,
            self.actor_future_mask,
            name="actor_future",
        )
        if self.outcome_mask.any() and not np.isfinite(
            self.outcomes[self.outcome_mask]
        ).all():
            raise WorldContractError("outcomes have non-finite values on valid mask")
        if self.rank_mask:
            if not self.comparable or not bool(self.candidate_mask.all()):
                raise WorldContractError(
                    "rank supervision requires comparable pair and two available candidates"
                )
            if not math.isfinite(float(self.rank_target)) or not (
                -1.0 <= float(self.rank_target) <= 1.0
            ):
                raise WorldContractError("rank_target must be finite in [-1,1]")
            if not math.isfinite(float(self.rank_weight)) or not (
                0.0 < float(self.rank_weight) <= 1.0
            ):
                raise WorldContractError("rank_weight must be finite in (0,1]")
        elif float(self.rank_weight) != 0.0:
            raise WorldContractError("rank_weight must be 0 when rank_mask is false")
        if self.tie_target and not self.rank_mask:
            raise WorldContractError("tie_target requires rank_mask")
        for index, available in enumerate(self.candidate_mask.tolist()):
            reason = self.unavailable_reasons[index]
            if available and reason:
                raise WorldContractError(
                    f"candidate {index} is available but has unavailable_reason"
                )
            if not available and not reason:
                raise WorldContractError(
                    f"candidate {index} is unavailable without exact reason"
                )
        forbidden = {"oracle_winner", "true_future_feature", "privileged_intent"}
        overlap = forbidden.intersection(str(k) for k in self.audit)
        if overlap:
            raise WorldContractError(f"runtime audit contains oracle keys: {sorted(overlap)}")

    def source_record(self, *, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
        # Keep the supervision masks explicit in the immutable source index.
        # The NPZ tensors carry the dense candidate/outcome masks, while these
        # scalar labels make TIE/BOTH_BAD/incomparable filtering auditable
        # without reconstructing an Oracle decision from the raw report.
        both_bad = bool(self.audit.get("both_bad", False))
        return {
            "schema_version": str(schema_version),
            "identity": self.identity.to_dict(),
            "candidate_mask": self.candidate_mask.astype(int).tolist(),
            "candidate_unavailable_mask": (~self.candidate_mask).astype(int).tolist(),
            "outcome_mask": self.outcome_mask.astype(int).tolist(),
            "rank_target": float(self.rank_target),
            "rank_mask": bool(self.rank_mask),
            "rank_weight": float(self.rank_weight),
            "tie_target": bool(self.tie_target),
            "tie_mask": bool(self.tie_target),
            "both_bad_mask": both_bad,
            "comparable": bool(self.comparable),
            "incomparable_mask": not bool(self.comparable),
            "unavailable_reasons": list(self.unavailable_reasons),
            "audit": dict(self.audit),
        }


@dataclass(frozen=True)
class WorldBatch:
    """Runtime-safe observable batch. It intentionally has no label fields."""

    ego_history: Any
    ego_history_mask: Any
    actor_history: Any
    actor_history_mask: Any
    road: Any
    road_mask: Any
    candidates: Any
    candidate_mask: Any
    sample_ids: tuple[str, ...] = ()

    @classmethod
    def from_samples(cls, samples: list[ActionBranchSample]) -> "WorldBatch":
        if not samples:
            raise WorldContractError("WorldBatch requires at least one sample")
        for sample in samples:
            sample.validate()
        return cls(
            ego_history=np.stack([x.ego_history for x in samples]),
            ego_history_mask=np.stack([x.ego_history_mask for x in samples]),
            actor_history=np.stack([x.actor_history for x in samples]),
            actor_history_mask=np.stack([x.actor_history_mask for x in samples]),
            road=np.stack([x.road for x in samples]),
            road_mask=np.stack([x.road_mask for x in samples]),
            candidates=np.stack([x.candidates for x in samples]),
            candidate_mask=np.stack([x.candidate_mask for x in samples]),
            sample_ids=tuple(x.identity.sample_id for x in samples),
        )


@dataclass(frozen=True)
class WorldPrediction:
    actor_future_mean: Any
    actor_future_log_scale: Any
    collision_logit: Any
    offroad_logit: Any
    ttc_value: Any
    ttc_censored_logit: Any
    utility_score: Any
    candidate_mask: Any
    status: str = "OK"
    error: str | None = None

    def validate_finite(self) -> None:
        if self.status != "OK":
            return
        mask = (
            self.candidate_mask.detach().cpu().numpy().astype(bool, copy=False)
            if hasattr(self.candidate_mask, "detach")
            else np.asarray(self.candidate_mask, dtype=bool)
        )
        for name in (
            "actor_future_mean",
            "actor_future_log_scale",
            "collision_logit",
            "offroad_logit",
            "ttc_value",
            "ttc_censored_logit",
            "utility_score",
        ):
            value = getattr(self, name)
            arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if arr.shape[:2] != mask.shape:
                raise WorldContractError(f"{name} leading shape {arr.shape[:2]} != {mask.shape}")
            if not np.isfinite(arr[mask]).all():
                raise WorldContractError(f"{name} contains non-finite valid output")
