"""Observation-only traffic-light sampling and terminal-stop classification.

Evidence / acceptance helpers only.  Never feed into VLA, PathManager, MPC, or
speed control.
"""

from __future__ import annotations

from typing import Any, Sequence

# Default tail window matches existing _tail_motion_metrics.
DEFAULT_TAIL_WINDOW_S = 20.0
DEFAULT_MOVING_SPEED_MPS = 0.50
DEFAULT_MOVING_FRACTION_THRESHOLD = 0.50
# After red→green, allow brief reaction before calling green_light_stuck.
DEFAULT_GREEN_GRACE_S = 5.0

TERMINAL_MOVING = "moving"
TERMINAL_EXPECTED_RED = "expected_red_light_stop"
TERMINAL_UNEXPLAINED = "unexplained_stop"
TERMINAL_GREEN_STUCK = "green_light_stuck"

_EXPECTED_STOP_STATES = frozenset({"red", "yellow"})


def observe_traffic_light(
    vehicle: Any,
    *,
    carla_module: Any | None = None,
) -> dict[str, Any]:
    """Read CARLA traffic-light association for evidence only.

    Always sets ``oracle_only=True``.
    """
    out: dict[str, Any] = {
        "oracle_only": True,
        "is_at_traffic_light": None,
        "traffic_light_state": None,
        "traffic_light_id": None,
        "ok": False,
    }
    try:
        is_at = bool(vehicle.is_at_traffic_light())
        out["is_at_traffic_light"] = is_at
        light = None
        try:
            light = vehicle.get_traffic_light()
        except Exception:
            light = None
        if light is not None:
            try:
                out["traffic_light_id"] = int(light.id)
            except Exception:
                out["traffic_light_id"] = None
            try:
                state = light.get_state()
                # CARLA enum: Red/Yellow/Green/Off/Unknown
                name = getattr(state, "name", None)
                out["traffic_light_state"] = str(name if name is not None else state)
            except Exception as exc:
                out["traffic_light_state"] = None
                out["state_error"] = f"{type(exc).__name__}:{exc}"
        elif is_at and carla_module is not None:
            # is_at true but no light object — still record association.
            out["traffic_light_state"] = "Unknown"
        out["ok"] = True
    except Exception as exc:  # pragma: no cover - live only
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _normalize_state(state: str | None) -> str | None:
    if state is None:
        return None
    s = str(state).strip().lower()
    if not s:
        return None
    # Accept "TrafficLightState.Red" style.
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s


def classify_terminal_stop(
    *,
    speed_samples_mps: Sequence[float],
    traffic_samples: Sequence[dict[str, Any]] | None = None,
    sim_dt_s: float,
    window_s: float = DEFAULT_TAIL_WINDOW_S,
    moving_speed_mps: float = DEFAULT_MOVING_SPEED_MPS,
    moving_fraction_threshold: float = DEFAULT_MOVING_FRACTION_THRESHOLD,
    green_grace_s: float = DEFAULT_GREEN_GRACE_S,
) -> dict[str, Any]:
    """Classify end-of-run motion using optional traffic-light oracle samples.

    Returns classification, raw tail motion numbers, and whether the tail
    acceptance gate should pass.
    """
    dt = max(float(sim_dt_s), 1e-6)
    win = max(float(window_s), 0.0)
    n = max(1, int(round(win / dt))) if speed_samples_mps else 0
    speeds = [float(v) for v in speed_samples_mps[-n:]] if n else []
    if not speeds:
        return {
            "oracle_only": True,
            "terminal_stop_classification": TERMINAL_UNEXPLAINED,
            "tail_acceptance_ok": False,
            "moving_fraction": 0.0,
            "mean_speed_mps": 0.0,
            "window_s": 0.0,
            "reason": "no_speed_samples",
            "traffic_light_at_end": None,
        }

    moving_frac = sum(1 for v in speeds if v >= float(moving_speed_mps)) / len(speeds)
    mean_speed = sum(speeds) / len(speeds)
    window_actual = len(speeds) * dt

    base = {
        "oracle_only": True,
        "moving_fraction": float(moving_frac),
        "mean_speed_mps": float(mean_speed),
        "window_s": float(window_actual),
        "moving_speed_mps": float(moving_speed_mps),
        "moving_fraction_threshold": float(moving_fraction_threshold),
        "green_grace_s": float(green_grace_s),
    }

    if moving_frac >= float(moving_fraction_threshold):
        return {
            **base,
            "terminal_stop_classification": TERMINAL_MOVING,
            "tail_acceptance_ok": True,
            "reason": "tail_moving_fraction_ok",
            "traffic_light_at_end": None,
        }

    # Align traffic samples to the same tail window (by count if same rate).
    tl = list(traffic_samples or [])
    if len(tl) >= len(speeds):
        tl_tail = tl[-len(speeds) :]
    else:
        tl_tail = tl

    # Zip speeds with traffic samples from the end.
    pairs: list[tuple[float, dict[str, Any]]] = []
    for i, spd in enumerate(speeds):
        if tl_tail:
            # Map speed index into tl_tail if lengths differ slightly.
            j = int(round(i * (len(tl_tail) - 1) / max(len(speeds) - 1, 1))) if len(speeds) > 1 else 0
            j = max(0, min(len(tl_tail) - 1, j))
            pairs.append((spd, tl_tail[j]))
        else:
            pairs.append((spd, {}))

    last_tl = pairs[-1][1] if pairs else {}
    end_state = _normalize_state(last_tl.get("traffic_light_state"))
    end_at = last_tl.get("is_at_traffic_light")
    end_info = {
        "is_at_traffic_light": end_at,
        "traffic_light_state": last_tl.get("traffic_light_state"),
        "traffic_light_id": last_tl.get("traffic_light_id"),
        "oracle_only": True,
    }

    # Continuous green while stationary at end → green_light_stuck after grace.
    green_streak_s = 0.0
    for spd, sample in reversed(pairs):
        if spd >= float(moving_speed_mps):
            break
        state = _normalize_state(sample.get("traffic_light_state"))
        is_at = bool(sample.get("is_at_traffic_light"))
        if is_at and state == "green":
            green_streak_s += dt
        else:
            break
    if green_streak_s > float(green_grace_s):
        return {
            **base,
            "terminal_stop_classification": TERMINAL_GREEN_STUCK,
            "tail_acceptance_ok": False,
            "reason": "stopped_through_green_beyond_grace",
            "green_while_stopped_s": float(green_streak_s),
            "traffic_light_at_end": end_info,
        }

    # Stationary stretch at end under red/yellow → expected legal stop.
    red_streak_s = 0.0
    saw_expected = False
    for spd, sample in reversed(pairs):
        if spd >= float(moving_speed_mps):
            break
        state = _normalize_state(sample.get("traffic_light_state"))
        is_at = bool(sample.get("is_at_traffic_light"))
        if is_at and state in _EXPECTED_STOP_STATES:
            saw_expected = True
            red_streak_s += dt
        elif is_at and state == "green":
            # Entered green after red — already handled by green streak if long.
            break
        else:
            break

    if saw_expected and (
        end_state in _EXPECTED_STOP_STATES or bool(end_at and end_state in _EXPECTED_STOP_STATES)
    ):
        return {
            **base,
            "terminal_stop_classification": TERMINAL_EXPECTED_RED,
            "tail_acceptance_ok": True,
            "reason": "stationary_at_red_or_yellow_light",
            "red_yellow_while_stopped_s": float(red_streak_s),
            "traffic_light_at_end": end_info,
            "acceptance_note": (
                "tail_moving_fraction below threshold is an expected traffic-light "
                "stop false-negative if acceptance only checked motion"
            ),
        }

    # Also accept if terminal sample is clearly red/yellow at light even if
    # streak is short (test ended mid-stop).
    if bool(end_at) and end_state in _EXPECTED_STOP_STATES:
        return {
            **base,
            "terminal_stop_classification": TERMINAL_EXPECTED_RED,
            "tail_acceptance_ok": True,
            "reason": "terminal_sample_red_or_yellow",
            "traffic_light_at_end": end_info,
            "acceptance_note": (
                "tail_moving_fraction below threshold is an expected traffic-light stop"
            ),
        }

    # No usable light association while stopped.
    any_at = any(bool(s.get("is_at_traffic_light")) for _, s in pairs if s)
    if not any_at and not bool(end_at):
        return {
            **base,
            "terminal_stop_classification": TERMINAL_UNEXPLAINED,
            "tail_acceptance_ok": False,
            "reason": "stationary_without_traffic_light",
            "traffic_light_at_end": end_info,
        }

    return {
        **base,
        "terminal_stop_classification": TERMINAL_UNEXPLAINED,
        "tail_acceptance_ok": False,
        "reason": "stationary_not_explained_by_red_light",
        "traffic_light_at_end": end_info,
        "green_while_stopped_s": float(green_streak_s),
    }


__all__ = [
    "DEFAULT_TAIL_WINDOW_S",
    "DEFAULT_MOVING_SPEED_MPS",
    "DEFAULT_MOVING_FRACTION_THRESHOLD",
    "DEFAULT_GREEN_GRACE_S",
    "TERMINAL_MOVING",
    "TERMINAL_EXPECTED_RED",
    "TERMINAL_UNEXPLAINED",
    "TERMINAL_GREEN_STUCK",
    "observe_traffic_light",
    "classify_terminal_stop",
]
