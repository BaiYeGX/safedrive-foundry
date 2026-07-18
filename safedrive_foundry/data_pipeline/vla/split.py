"""Frozen train/val/test/regression split assignment (G3-01)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from data_pipeline.vla.schema import FrameIdentity, SampleRecord


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    REGRESSION = "regression"


@dataclass(frozen=True)
class SplitSpec:
    """Deterministic hash-bucket split with optional regression pinning."""

    seed: int = 11
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Scenario families always pinned to regression (never train).
    regression_families: tuple[str, ...] = ("regression_core", "holdout_gate")

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


class SplitAssigner:
    def __init__(self, spec: SplitSpec | None = None) -> None:
        self.spec = spec or SplitSpec()
        self._regression_keys: set[str] = set()

    def pin_regression(self, identity: FrameIdentity) -> None:
        self._regression_keys.add(identity.key())

    def _bucket(self, identity: FrameIdentity) -> float:
        material = (
            f"{self.spec.seed}|{identity.town}|{identity.route_id}|"
            f"{identity.scenario_family}|{identity.weather}|{identity.scenario_id}|"
            f"{identity.failure_cluster}"
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        # Map first 8 hex chars to [0, 1).
        return int(digest[:8], 16) / 0x100000000

    def assign(self, identity: FrameIdentity) -> SplitName:
        if identity.key() in self._regression_keys:
            return SplitName.REGRESSION
        if identity.scenario_family in self.spec.regression_families:
            return SplitName.REGRESSION
        u = self._bucket(identity)
        if u < self.spec.train_ratio:
            return SplitName.TRAIN
        if u < self.spec.train_ratio + self.spec.val_ratio:
            return SplitName.VAL
        return SplitName.TEST

    def assign_sample(self, sample: SampleRecord) -> SplitName:
        return self.assign(sample.identity)

    def partition(
        self, samples: Iterable[SampleRecord]
    ) -> dict[SplitName, list[SampleRecord]]:
        out: dict[SplitName, list[SampleRecord]] = {s: [] for s in SplitName}
        for sample in samples:
            out[self.assign_sample(sample)].append(sample)
        return out
