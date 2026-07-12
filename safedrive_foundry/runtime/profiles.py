"""Runtime Profile parsing and CARLA fixed-step legality gates."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ProfileViolation(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    fixed_delta_seconds: float
    max_substep_delta_time: float
    max_substeps: int
    control_period_ms: int
    synchronous_mode: bool = True
    substepping: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.name not in {"throughput_20hz", "control_50hz"}:
            errors.append("unsupported runtime profile")
        if not self.synchronous_mode:
            errors.append("synchronous_mode must be true")
        if not self.substepping:
            errors.append("substepping must be true")
        if not math.isfinite(self.fixed_delta_seconds) or self.fixed_delta_seconds <= 0:
            errors.append("fixed_delta_seconds must be finite and positive")
        if not math.isfinite(self.max_substep_delta_time) or not 0 < self.max_substep_delta_time <= 0.01:
            errors.append("max_substep_delta_time must be in (0, 0.01]")
        if not isinstance(self.max_substeps, int) or not 1 <= self.max_substeps <= 10:
            errors.append("max_substeps must be an integer in [1, 10]")
        if self.fixed_delta_seconds > self.max_substep_delta_time * self.max_substeps + 1e-12:
            errors.append("fixed_delta_seconds exceeds CARLA substep capacity")
        expected_period_ms = round(self.fixed_delta_seconds * 1000)
        if self.control_period_ms != expected_period_ms:
            errors.append("control_period_ms must equal fixed_delta_seconds")
        expected_delta = {"throughput_20hz": 0.05, "control_50hz": 0.02}.get(self.name)
        if expected_delta is not None and abs(self.fixed_delta_seconds - expected_delta) > 1e-12:
            errors.append(f"{self.name} requires fixed_delta_seconds={expected_delta}")
        return errors

    def assert_valid(self) -> None:
        if errors := self.validate():
            raise ProfileViolation("; ".join(errors))

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Any]) -> "RuntimeProfile":
        profile = cls(
            name=name,
            fixed_delta_seconds=float(values["fixed_delta_seconds"]),
            max_substep_delta_time=float(values["max_substep_delta_time"]),
            max_substeps=int(values["max_substeps"]),
            control_period_ms=int(values["control_period_ms"]),
            synchronous_mode=bool(values.get("synchronous_mode", True)),
            substepping=bool(values.get("substepping", True)),
        )
        profile.assert_valid()
        return profile


def load_runtime_profiles(path: Path) -> dict[str, RuntimeProfile]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profile")
    if not isinstance(profiles, dict):
        raise ProfileViolation("[profile] table is required")
    parsed = {name: RuntimeProfile.from_mapping(name, values) for name, values in profiles.items()}
    if set(parsed) != {"throughput_20hz", "control_50hz"}:
        raise ProfileViolation("profiles must define exactly throughput_20hz and control_50hz")
    return parsed
