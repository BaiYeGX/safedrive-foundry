"""Convert the VLA speed head into a bounded, executable speed reference.

The VLA remains the source of the desired speed.  This module only applies a
calibration gain, an absolute cap and asymmetric slew limiting: braking is
accepted immediately while acceleration is introduced gradually.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VLASpeedConfig:
    max_speed_mps: float = 15.0
    calibration_gain: float = 1.50
    stop_threshold_mps: float = 0.35
    max_accel_mps2: float = 2.0


@dataclass(frozen=True)
class VLASpeedDecision:
    raw_speed_mps: float
    calibrated_speed_mps: float
    target_speed_mps: float
    stop_requested: bool
    valid: bool


class VLASpeedPlanner:
    """Stateful speed calibration with brake-first temporal filtering."""

    def __init__(self, config: VLASpeedConfig | None = None) -> None:
        self.config = config or VLASpeedConfig()
        if self.config.max_speed_mps < 0.0:
            raise ValueError("max_speed_mps must be non-negative")
        if self.config.calibration_gain < 0.0:
            raise ValueError("calibration_gain must be non-negative")
        if self.config.max_accel_mps2 <= 0.0:
            raise ValueError("max_accel_mps2 must be positive")
        self._target_speed_mps = 0.0

    @property
    def target_speed_mps(self) -> float:
        return self._target_speed_mps

    def reset(self, *, target_speed_mps: float = 0.0) -> None:
        self._target_speed_mps = float(
            np.clip(target_speed_mps, 0.0, self.config.max_speed_mps)
        )

    @staticmethod
    def _robust_raw_speed(speed_samples_mps: Sequence[float]) -> tuple[float, bool]:
        values = np.asarray(tuple(speed_samples_mps), dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, False
        # SimLingo predicts a short speed sequence.  The near-field median is
        # less sensitive to a single decoding spike than the first element.
        raw = float(np.median(finite[:5]))
        return max(0.0, raw), math.isfinite(raw)

    def update(self, speed_samples_mps: Sequence[float], *, dt_s: float) -> VLASpeedDecision:
        raw, valid = self._robust_raw_speed(speed_samples_mps)
        cfg = self.config
        stop_requested = (not valid) or raw <= cfg.stop_threshold_mps
        calibrated = 0.0 if stop_requested else min(cfg.max_speed_mps, raw * cfg.calibration_gain)

        if calibrated <= self._target_speed_mps:
            # Never delay a semantic VLA brake request.
            self._target_speed_mps = calibrated
        else:
            rise = cfg.max_accel_mps2 * max(0.0, float(dt_s))
            self._target_speed_mps = min(calibrated, self._target_speed_mps + rise)

        return VLASpeedDecision(
            raw_speed_mps=raw,
            calibrated_speed_mps=calibrated,
            target_speed_mps=self._target_speed_mps,
            stop_requested=stop_requested,
            valid=valid,
        )
