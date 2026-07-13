"""Vehicle parameter helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class VehicleParams:
    wheelbase_m: float = 2.8
    max_speed_mps: float = 15.0
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 4.0
    max_jerk_mps3: float = 4.0
    max_curvature_per_m: float = 0.25
    max_lateral_accel_mps2: float = 2.0
    width_m: float = 1.9
    length_m: float = 4.5
