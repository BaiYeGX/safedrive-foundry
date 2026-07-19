"""Stationary re-query of VLA when stop feedback may be self-reinforcing.

Does not force throttle.  Only injects a one-shot low-speed *hint* into the
VLA observation so the model can be re-asked; recovery still requires a normal
accepted path and a positive VLA speed head.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationaryRequeryConfig:
    min_stop_frames: int = 4
    max_ego_speed_mps: float = 0.50
    hint_speed_mps: float = 1.50
    cooldown_s: float = 8.0


@dataclass(frozen=True)
class StationaryRequeryDecision:
    trigger: bool
    reason: str
    hint_speed_mps: float = 0.0


def evaluate_stationary_requery(
    *,
    ego_speed_mps: float,
    stop_requested: bool,
    consecutive_stop_frames: int,
    has_collided: bool,
    navigation_valid: bool,
    path_nav_aligned: bool,
    already_active: bool,
    last_trigger_sim_s: float | None,
    sim_s: float,
    config: StationaryRequeryConfig | None = None,
) -> StationaryRequeryDecision:
    """Return whether to re-query VLA with a small speed hint this frame."""
    cfg = config or StationaryRequeryConfig()
    if already_active:
        return StationaryRequeryDecision(False, "already_active")
    if has_collided:
        return StationaryRequeryDecision(False, "has_collided")
    if not navigation_valid:
        return StationaryRequeryDecision(False, "navigation_invalid")
    if not path_nav_aligned:
        return StationaryRequeryDecision(False, "path_not_nav_aligned")
    if max(0.0, float(ego_speed_mps)) > float(cfg.max_ego_speed_mps):
        return StationaryRequeryDecision(False, "not_stationary")
    if not stop_requested:
        return StationaryRequeryDecision(False, "vla_not_stop")
    if int(consecutive_stop_frames) < int(cfg.min_stop_frames):
        return StationaryRequeryDecision(False, "stop_frames_insufficient")
    if last_trigger_sim_s is not None:
        if float(sim_s) - float(last_trigger_sim_s) < float(cfg.cooldown_s):
            return StationaryRequeryDecision(False, "cooldown")
    return StationaryRequeryDecision(
        True,
        "stationary_stop_requery",
        hint_speed_mps=float(cfg.hint_speed_mps),
    )


def requery_success(
    *,
    path_accepted: bool,
    stop_requested_after: bool,
    target_speed_after: float,
    min_positive_mps: float = 0.35,
) -> bool:
    """True only when re-query yields a gated path and a positive speed head."""
    return (
        bool(path_accepted)
        and (not bool(stop_requested_after))
        and float(target_speed_after) >= float(min_positive_mps)
    )
