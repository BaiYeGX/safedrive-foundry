"""Convert the VLA speed head into a bounded, executable speed reference.

The VLA remains the source of the desired speed.  This module only applies a
calibration gain, an absolute cap, asymmetric slew limiting, stop/launch
hysteresis, and a short post-stale recovery assist (execution layer only — never
fakes the speed fed *into* the VLA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VLASpeedConfig:
    max_speed_mps: float = 15.0
    # Official agent_simlingo has no 1.5× cruise gain; default 1.0.
    calibration_gain: float = 1.0
    # Enter latched stop when raw <= this (moving → stop).
    stop_threshold_mps: float = 0.35
    # Leave latched stop only when raw >= this for launch_confirm_frames (hysteresis).
    launch_threshold_mps: float = 0.50
    launch_confirm_frames: int = 4
    max_accel_mps2: float = 2.0
    # After execution-layer stale stop / reanchor: allow restart without fake VLA input.
    recovery_raw_min_mps: float = 0.20
    recovery_launch_floor_mps: float = 1.00
    recovery_confirm_frames: int = 3
    recovery_window_s: float = 4.0
    recovery_ego_speed_max_mps: float = 1.20


@dataclass(frozen=True)
class VLASpeedDecision:
    raw_speed_mps: float
    calibrated_speed_mps: float
    target_speed_mps: float
    stop_requested: bool
    valid: bool
    # none | vla_head | invalid | launch_hysteresis | execution_recovery_pending
    stop_source: str = "none"
    launch_confirm_count: int = 0
    recovery_active: bool = False
    recovery_timer_s: float = 0.0


class VLASpeedPlanner:
    """Stateful speed calibration with brake-first filtering and launch hysteresis."""

    def __init__(self, config: VLASpeedConfig | None = None) -> None:
        self.config = config or VLASpeedConfig()
        if self.config.max_speed_mps < 0.0:
            raise ValueError("max_speed_mps must be non-negative")
        if self.config.calibration_gain < 0.0:
            raise ValueError("calibration_gain must be non-negative")
        if self.config.max_accel_mps2 <= 0.0:
            raise ValueError("max_accel_mps2 must be positive")
        if self.config.launch_threshold_mps < self.config.stop_threshold_mps:
            raise ValueError("launch_threshold_mps must be >= stop_threshold_mps")
        self._target_speed_mps = 0.0
        # Cold start: not latched until first low VLA sample (tests/motion assume free rise).
        self._stop_latched = False
        self._stop_source = "none"
        self._launch_pos_frames = 0
        self._recovery_timer_s = 0.0
        # Last known execution-layer stale stop (for evidence / recovery).
        self._execution_stale_latched = False

    @property
    def target_speed_mps(self) -> float:
        return self._target_speed_mps

    @property
    def stop_latched(self) -> bool:
        return self._stop_latched

    @property
    def stop_source(self) -> str:
        return self._stop_source

    @property
    def execution_stale_latched(self) -> bool:
        return self._execution_stale_latched

    def reset(self, *, target_speed_mps: float = 0.0) -> None:
        self._target_speed_mps = float(
            np.clip(target_speed_mps, 0.0, self.config.max_speed_mps)
        )
        self._stop_latched = self._target_speed_mps <= self.config.stop_threshold_mps
        self._stop_source = "none" if not self._stop_latched else "vla_head"
        self._launch_pos_frames = 0
        self._recovery_timer_s = 0.0
        self._execution_stale_latched = False

    def notify_execution_stale_stop(self) -> None:
        """MPC/path freshness forced a stop while path was still the reference."""
        self._execution_stale_latched = True
        self._stop_latched = True
        self._stop_source = "execution_freshness"
        self._launch_pos_frames = 0
        self._target_speed_mps = 0.0

    def notify_path_accepted(self, *, reanchor: bool, path_age_s: float) -> None:
        """Arm recovery only after a proven execution-freshness stop.

        ``reanchor`` is retained for call-site compatibility and evidence, but
        must never by itself override a semantic VLA stop (for example a red
        light).  Recovery authority comes solely from
        :meth:`notify_execution_stale_stop` followed by a fresh valid path.
        """
        fresh = float(path_age_s) <= 1.0  # aligns with MPC soft stale window
        _ = bool(reanchor)
        if self._execution_stale_latched and fresh:
            self._recovery_timer_s = float(self.config.recovery_window_s)
            self._execution_stale_latched = False
            # Do not clear stop_latched here — still need VLA intent + confirm.
            if self._stop_source == "execution_freshness":
                self._stop_source = "execution_recovery_pending"

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

    def update(
        self,
        speed_samples_mps: Sequence[float],
        *,
        dt_s: float,
        ego_speed_mps: float = 0.0,
    ) -> VLASpeedDecision:
        raw, valid = self._robust_raw_speed(speed_samples_mps)
        cfg = self.config
        dt = max(0.0, float(dt_s))
        ego_v = max(0.0, float(ego_speed_mps))
        recovery_active = self._recovery_timer_s > 0.0

        if not valid:
            self._stop_latched = True
            self._stop_source = "invalid"
            self._launch_pos_frames = 0
        elif self._stop_latched:
            # Already stopped: only *leave* via sustained launch/recovery intent.
            # Do NOT re-apply stop_threshold here — recovery raw (0.20–0.35) must
            # be able to climb out after execution-layer stale stop + fresh path.
            if recovery_active:
                need_raw = cfg.recovery_raw_min_mps
                need_frames = max(1, int(cfg.recovery_confirm_frames))
            else:
                need_raw = cfg.launch_threshold_mps
                need_frames = max(1, int(cfg.launch_confirm_frames))
            # Strong VLA intent (well above launch bar) releases in one frame.
            if raw >= max(cfg.launch_threshold_mps, 1.0):
                need_frames = 1
            if raw >= need_raw:
                self._launch_pos_frames += 1
            else:
                self._launch_pos_frames = 0
            if self._launch_pos_frames >= need_frames:
                self._stop_latched = False
                self._stop_source = "none"
                self._launch_pos_frames = 0
            else:
                self._stop_source = (
                    "execution_recovery_pending"
                    if recovery_active
                    else "launch_hysteresis"
                )
        elif raw <= cfg.stop_threshold_mps:
            # Moving → stop: semantic VLA brake.
            self._stop_latched = True
            self._stop_source = "vla_head"
            self._launch_pos_frames = 0
        else:
            self._launch_pos_frames = 0
            self._stop_source = "none"

        if self._stop_latched:
            calibrated = 0.0
            stop_requested = True
        else:
            calibrated = min(cfg.max_speed_mps, raw * cfg.calibration_gain)
            # Execution-layer launch floor after stale-stop recovery — not VLA input spoof.
            if (
                recovery_active
                and ego_v <= cfg.recovery_ego_speed_max_mps
                and raw >= cfg.recovery_raw_min_mps
            ):
                calibrated = max(calibrated, float(cfg.recovery_launch_floor_mps))
            stop_requested = False

        if calibrated <= self._target_speed_mps:
            # Never delay a semantic VLA brake request.
            self._target_speed_mps = calibrated
        else:
            rise = cfg.max_accel_mps2 * dt
            self._target_speed_mps = min(calibrated, self._target_speed_mps + rise)

        if recovery_active:
            self._recovery_timer_s = max(0.0, self._recovery_timer_s - dt)

        return VLASpeedDecision(
            raw_speed_mps=raw,
            calibrated_speed_mps=calibrated,
            target_speed_mps=self._target_speed_mps,
            stop_requested=stop_requested,
            valid=valid,
            stop_source=self._stop_source,
            launch_confirm_count=int(self._launch_pos_frames),
            recovery_active=self._recovery_timer_s > 0.0,
            recovery_timer_s=float(self._recovery_timer_s),
        )
