"""NORMAL / DEGRADED / MINIMAL_RISK / EMERGENCY with debounce, dwell, hysteresis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from safety_kernel.config import SafetyKernelConfig, load_safety_config
from safety_kernel.contracts.types import (
    ComponentAvailability,
    DecisionKind,
    SafetyDecision,
    SafetyMode,
)

_LOG = logging.getLogger("safedrive.safety_kernel.state_machine")

_SEVERITY = {
    SafetyMode.NORMAL: 0,
    SafetyMode.DEGRADED: 1,
    SafetyMode.MINIMAL_RISK: 2,
    SafetyMode.EMERGENCY: 3,
}


@dataclass(frozen=True)
class StateTransition:
    from_state: SafetyMode
    to_state: SafetyMode
    reason: str
    simulation_time_s: float
    frame_id: str


def desired_mode_from_decision(
    decision: SafetyDecision,
    availability: ComponentAvailability,
) -> SafetyMode:
    if decision.decision_kind is DecisionKind.EMERGENCY:
        return SafetyMode.EMERGENCY
    if decision.decision_kind is DecisionKind.MINIMAL_RISK:
        return SafetyMode.MINIMAL_RISK
    if decision.decision_kind is DecisionKind.HARD_REJECT:
        return SafetyMode.MINIMAL_RISK
    if decision.decision_kind is DecisionKind.CLASSIC_FALLBACK:
        return SafetyMode.DEGRADED
    if decision.decision_kind is DecisionKind.ACCEPT:
        # Learning off is not automatically DEGRADED if Classic accepted.
        return SafetyMode.NORMAL
    if decision.decision_kind is DecisionKind.QP:
        # Successful longitudinal repair is a legal executed trajectory.
        return SafetyMode.NORMAL
    if decision.decision_kind is DecisionKind.RATO:
        # Reserved G2-03; successful RATO also stays NORMAL.
        return SafetyMode.NORMAL
    if not availability.classic:
        return SafetyMode.EMERGENCY
    return SafetyMode.DEGRADED


class SafetyStateMachine:
    """Deterministic safety mode machine; errors are logged, never swallowed."""

    def __init__(self, config: SafetyKernelConfig | None = None) -> None:
        self.config = config or load_safety_config()
        self.mode = SafetyMode.NORMAL
        self._entered_at_s = 0.0
        self._pending_target: SafetyMode | None = None
        self._pending_count = 0
        self._clear_count = 0
        self.history: list[StateTransition] = []
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def reset(self, mode: SafetyMode = SafetyMode.NORMAL, *, now_s: float = 0.0) -> None:
        self.mode = mode
        self._entered_at_s = now_s
        self._pending_target = None
        self._pending_count = 0
        self._clear_count = 0
        self._last_error = None

    def step(
        self,
        decision: SafetyDecision,
        availability: ComponentAvailability,
        *,
        now_s: float,
        frame_id: str,
        state_floor: SafetyMode | None = None,
    ) -> StateTransition | None:
        desired = desired_mode_from_decision(decision, availability)
        # Defense-in-depth: observation-state floor cannot be undercut by ACCEPT/QP/RATO.
        if state_floor is not None and _SEVERITY[state_floor] > _SEVERITY[desired]:
            desired = state_floor
        # Immediate emergency escalate (no debounce for life-critical).
        if desired is SafetyMode.EMERGENCY and self.mode is not SafetyMode.EMERGENCY:
            return self._commit(SafetyMode.EMERGENCY, reason="emergency_immediate", now_s=now_s, frame_id=frame_id)

        dwell_ok = (now_s - self._entered_at_s) >= self.config.min_dwell_s

        if _SEVERITY[desired] > _SEVERITY[self.mode]:
            # Escalate path with debounce (except already handled EMERGENCY).
            if self._pending_target is desired:
                self._pending_count += 1
            else:
                self._pending_target = desired
                self._pending_count = 1
            self._clear_count = 0
            need = self.config.escalate_debounce_frames
            if self._pending_count >= need and dwell_ok:
                return self._commit(desired, reason=f"escalate:{decision.decision_kind.value}", now_s=now_s, frame_id=frame_id)
            return None

        if desired is self.mode:
            self._pending_target = None
            self._pending_count = 0
            self._clear_count += 1
            return None

        # Recovery / de-escalate: require hysteresis clear frames + dwell.
        if _SEVERITY[desired] < _SEVERITY[self.mode]:
            self._pending_target = desired
            self._clear_count += 1
            need = self.config.recover_clear_frames
            if self._clear_count >= need and dwell_ok:
                return self._commit(desired, reason=f"recover:{decision.decision_kind.value}", now_s=now_s, frame_id=frame_id)
            return None

        return None

    def _commit(self, to_state: SafetyMode, *, reason: str, now_s: float, frame_id: str) -> StateTransition:
        transition = StateTransition(
            from_state=self.mode,
            to_state=to_state,
            reason=reason,
            simulation_time_s=now_s,
            frame_id=frame_id,
        )
        _LOG.info(
            "safety_mode_transition from=%s to=%s reason=%s frame=%s t=%.4f",
            self.mode.value,
            to_state.value,
            reason,
            frame_id,
            now_s,
        )
        self.mode = to_state
        self._entered_at_s = now_s
        self._pending_target = None
        self._pending_count = 0
        self._clear_count = 0
        self.history.append(transition)
        return transition

    def record_error(self, message: str) -> None:
        """Surface errors explicitly — never silently drop."""
        self._last_error = message
        _LOG.error("safety_state_machine_error: %s", message)

    def apply_batch(
        self,
        decisions: Sequence[SafetyDecision],
        availability: ComponentAvailability,
        *,
        times_s: Sequence[float],
        frame_ids: Sequence[str],
    ) -> list[StateTransition]:
        out: list[StateTransition] = []
        for decision, t, fid in zip(decisions, times_s, frame_ids, strict=True):
            tr = self.step(decision, availability, now_s=t, frame_id=fid)
            if tr is not None:
                out.append(tr)
        return out
