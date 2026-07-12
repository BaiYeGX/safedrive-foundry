"""Validated compatibility adapter from G0 JSON status to formal contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .identity import RunIdentity


class ContractViolation(ValueError):
    pass


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ContractViolation(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f"{field} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractViolation(f"{field} is outside its valid range")
    return result


@dataclass(frozen=True)
class StatusFrame:
    identity: RunIdentity
    carla_frame: int
    simulation_time: float
    wall_time: float
    producer_version: str
    schema_version: str


class StatusJsonAdapter:
    """Consumes only known G0 fields and emits a typed runtime status frame."""

    legacy_schemas = frozenset({"safedrive.carla.status.v2"})

    def parse(self, payload: Mapping[str, Any], identity: RunIdentity) -> StatusFrame:
        if not isinstance(payload, Mapping):
            raise ContractViolation("status payload must be an object")
        if payload.get("schema") not in self.legacy_schemas:
            raise ContractViolation("unsupported legacy status schema")
        frame = payload.get("carla_frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ContractViolation("carla_frame must be a non-negative integer")
        legacy_episode = payload.get("episode_id")
        if not isinstance(legacy_episode, str) or not legacy_episode:
            raise ContractViolation("legacy episode_id is required for compatibility validation")
        simulation_time = _number(payload.get("simulation_seconds"), "simulation_seconds", minimum=0)
        wall_time = _number(payload.get("publisher_wall_time"), "publisher_wall_time", minimum=0)
        return StatusFrame(
            identity=identity,
            carla_frame=frame,
            simulation_time=simulation_time,
            wall_time=wall_time,
            producer_version=identity.producer_version,
            schema_version=identity.schema_version,
        )
