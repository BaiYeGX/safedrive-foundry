"""Deterministic candidate-blind reactive actor controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


class ReactiveActorContractError(RuntimeError):
    pass


FORBIDDEN_REACTIVE_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_index",
        "oracle",
        "oracle_winner",
        "future",
        "scripted_future",
    }
)


@dataclass(frozen=True)
class ReactiveActorConfig:
    desired_speed_mps: float = 5.0
    yield_ttc_s: float = 2.5
    stop_distance_m: float = 5.0
    max_throttle: float = 0.45
    max_brake: float = 0.80
    gain_speed: float = 0.15


def assert_candidate_blind_observation(observation: Mapping[str, Any]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key)
                if token in FORBIDDEN_REACTIVE_KEYS or token.startswith("oracle_"):
                    raise ReactiveActorContractError(
                        f"forbidden reactive input {path}{token}"
                    )
                walk(child, f"{path}{token}.")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}{index}.")

    walk(observation, "")


def deterministic_reactive_control(
    observation: Mapping[str, Any],
    config: ReactiveActorConfig = ReactiveActorConfig(),
) -> dict[str, Any]:
    """Return longitudinal control from current relative state only."""
    assert_candidate_blind_observation(observation)
    distance = max(0.0, float(observation.get("relative_distance_m", math.inf)))
    closing = max(0.0, float(observation.get("closing_speed_mps", 0.0)))
    speed = max(0.0, float(observation.get("actor_speed_mps", 0.0)))
    conflict = bool(observation.get("lane_conflict", False))
    has_priority = bool(observation.get("actor_has_priority", False))
    ttc = distance / closing if closing > 0.05 else math.inf
    should_yield = conflict and not has_priority and (
        distance <= config.stop_distance_m or ttc <= config.yield_ttc_s
    )
    if should_yield:
        severity = max(
            0.25,
            min(
                1.0,
                max(
                    (config.stop_distance_m - distance) / max(config.stop_distance_m, 1e-6),
                    (config.yield_ttc_s - ttc) / max(config.yield_ttc_s, 1e-6),
                ),
            ),
        )
        return {
            "throttle": 0.0,
            "brake": min(config.max_brake, 0.25 + severity * config.max_brake),
            "steer": 0.0,
            "reason": "YIELD_CURRENT_STATE",
        }
    throttle = min(
        config.max_throttle,
        max(0.0, (config.desired_speed_mps - speed) * config.gain_speed),
    )
    return {
        "throttle": throttle,
        "brake": 0.0,
        "steer": 0.0,
        "reason": "TRACK_SPEED_CURRENT_STATE",
    }
