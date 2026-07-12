"""Stable identities owned by the runtime, never by a bridge process."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping


SCHEMA_VERSION = "safedrive.runtime.v1"


def _require(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    run_id: str
    scenario_id: str
    attempt_id: int
    server_epoch: str
    producer_version: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("experiment_id", "run_id", "scenario_id", "server_epoch", "producer_version"):
            _require(getattr(self, name), name)
        if isinstance(self.attempt_id, bool) or self.attempt_id < 0:
            raise ValueError("attempt_id must be a non-negative integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RuntimeIdentityFactory:
    """Build reproducible run IDs from explicit runtime inputs.

    The same inputs produce the same ID in any process.  Changing attempt or
    server epoch isolates a restarted run without relying on a bridge UUID.
    """

    @staticmethod
    def create(inputs: Mapping[str, object]) -> RunIdentity:
        required = ("experiment_id", "scenario_id", "server_epoch", "producer_version")
        normalized = {name: _require(str(inputs.get(name, "")), name) for name in required}
        attempt_id = int(inputs.get("attempt_id", 0))
        if attempt_id < 0:
            raise ValueError("attempt_id must be a non-negative integer")
        explicit_run_id = inputs.get("run_id")
        if explicit_run_id is None:
            seed = {**normalized, "attempt_id": attempt_id, "schema_version": SCHEMA_VERSION}
            encoded = json.dumps(seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
            run_id = "run-" + hashlib.sha256(encoded).hexdigest()[:20]
        else:
            run_id = _require(str(explicit_run_id), "run_id")
        return RunIdentity(run_id=run_id, attempt_id=attempt_id, **normalized)
