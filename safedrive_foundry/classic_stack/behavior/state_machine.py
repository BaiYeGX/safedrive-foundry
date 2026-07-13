"""Auditable behavior state machine for classic expert goals.

The machine never emits vehicle controls or local trajectories. It only
produces behavior goals with enter/hold/exit/timeout/suppress reasons that
downstream planners (G1-04+) can consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class BehaviorState(str, Enum):
    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    STOP = "STOP"
    YIELD = "YIELD"
    LANE_CHANGE = "LANE_CHANGE"
    AVOID = "AVOID"
    MIN_RISK = "MIN_RISK"


class BehaviorEvent(str, Enum):
    TICK = "TICK"
    LEAD_VEHICLE_DETECTED = "LEAD_VEHICLE_DETECTED"
    LEAD_VEHICLE_CLEARED = "LEAD_VEHICLE_CLEARED"
    RED_LIGHT = "RED_LIGHT"
    GREEN_LIGHT = "GREEN_LIGHT"
    UNPROTECTED_LEFT = "UNPROTECTED_LEFT"
    CROSS_TRAFFIC_CLEAR = "CROSS_TRAFFIC_CLEAR"
    LANE_CHANGE_REQUEST = "LANE_CHANGE_REQUEST"
    LANE_CHANGE_COMPLETE = "LANE_CHANGE_COMPLETE"
    LANE_CHANGE_BLOCKED = "LANE_CHANGE_BLOCKED"
    OBSTACLE_AHEAD = "OBSTACLE_AHEAD"
    OBSTACLE_CLEARED = "OBSTACLE_CLEARED"
    TIMEOUT = "TIMEOUT"
    SUPPRESS = "SUPPRESS"
    FORCE_MIN_RISK = "FORCE_MIN_RISK"
    ROUTE_STRAIGHT = "ROUTE_STRAIGHT"
    ROUTE_TURN = "ROUTE_TURN"


@dataclass(frozen=True)
class BehaviorGoal:
    state: BehaviorState
    reason: str
    route_id: str | None = None
    target_lane_id: int | None = None
    target_speed_mps: float | None = None
    hold_seconds: float | None = None
    oracle_inputs: tuple[str, ...] = ()
    observable_inputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


@dataclass(frozen=True)
class BehaviorTransition:
    timestamp_s: float
    event: BehaviorEvent
    from_state: BehaviorState
    to_state: BehaviorState
    phase: str  # enter | hold | exit | timeout | suppress
    reason: str
    suppressed: bool = False
    goal: BehaviorGoal | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "event": self.event.value,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "phase": self.phase,
            "reason": self.reason,
            "suppressed": self.suppressed,
            "goal": None if self.goal is None else self.goal.to_dict(),
        }


@dataclass
class BehaviorConfig:
    follow_timeout_s: float = 8.0
    stop_timeout_s: float = 60.0
    yield_timeout_s: float = 10.0
    lane_change_timeout_s: float = 6.0
    avoid_timeout_s: float = 5.0
    min_risk_hold_s: float = 2.0
    default_cruise_speed_mps: float = 12.0
    suppressed_events: tuple[BehaviorEvent, ...] = ()


class BehaviorStateMachine:
    """Deterministic behavior layer with explicit enter/hold/exit/timeout/suppress."""

    def __init__(self, config: BehaviorConfig | None = None, *, route_id: str | None = None) -> None:
        self.config = config or BehaviorConfig()
        self.route_id = route_id
        self.state = BehaviorState.CRUISE
        self.state_entered_s = 0.0
        self.history: list[BehaviorTransition] = []
        self._suppressed = set(self.config.suppressed_events)
        self._active_goal = self._goal_for(self.state, reason="initial_enter")

    @property
    def goal(self) -> BehaviorGoal:
        return self._active_goal

    def suppress(self, event: BehaviorEvent, *, reason: str, timestamp_s: float) -> BehaviorTransition:
        self._suppressed.add(event)
        transition = BehaviorTransition(
            timestamp_s=timestamp_s,
            event=event,
            from_state=self.state,
            to_state=self.state,
            phase="suppress",
            reason=reason,
            suppressed=True,
            goal=self._active_goal,
        )
        self.history.append(transition)
        return transition

    def unsuppress(self, event: BehaviorEvent) -> None:
        self._suppressed.discard(event)

    def handle(self, event: BehaviorEvent, *, timestamp_s: float, context: Mapping[str, Any] | None = None) -> BehaviorTransition:
        context = context or {}
        if event in self._suppressed and event not in {BehaviorEvent.TIMEOUT, BehaviorEvent.FORCE_MIN_RISK, BehaviorEvent.SUPPRESS}:
            transition = BehaviorTransition(
                timestamp_s=timestamp_s,
                event=event,
                from_state=self.state,
                to_state=self.state,
                phase="suppress",
                reason=f"event_suppressed:{event.value}",
                suppressed=True,
                goal=self._active_goal,
            )
            self.history.append(transition)
            return transition

        if event is BehaviorEvent.SUPPRESS:
            target = context.get("suppress_event")
            if not isinstance(target, BehaviorEvent):
                raise ValueError("SUPPRESS requires context['suppress_event'] as BehaviorEvent")
            return self.suppress(target, reason=str(context.get("reason", "manual_suppress")), timestamp_s=timestamp_s)

        previous = self.state
        dwell = timestamp_s - self.state_entered_s
        next_state, phase, reason = self._resolve(event, dwell=dwell, context=context)

        if next_state != previous:
            # explicit exit then enter records
            exit_transition = BehaviorTransition(
                timestamp_s=timestamp_s,
                event=event,
                from_state=previous,
                to_state=previous,
                phase="exit",
                reason=f"exit_for:{reason}",
                goal=self._active_goal,
            )
            self.history.append(exit_transition)
            self.state = next_state
            self.state_entered_s = timestamp_s
            self._active_goal = self._goal_for(next_state, reason=reason, context=context)
            transition = BehaviorTransition(
                timestamp_s=timestamp_s,
                event=event,
                from_state=previous,
                to_state=next_state,
                phase="enter",
                reason=reason,
                goal=self._active_goal,
            )
            self.history.append(transition)
            return transition

        # same state: hold or timeout no-op hold
        if phase == "timeout":
            # timeout without available transition stays and records timeout hold
            transition = BehaviorTransition(
                timestamp_s=timestamp_s,
                event=event,
                from_state=previous,
                to_state=previous,
                phase="timeout",
                reason=reason,
                goal=self._active_goal,
            )
        else:
            transition = BehaviorTransition(
                timestamp_s=timestamp_s,
                event=event,
                from_state=previous,
                to_state=previous,
                phase="hold",
                reason=reason,
                goal=self._active_goal,
            )
        self.history.append(transition)
        return transition

    def timeline(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.history]

    def _resolve(self, event: BehaviorEvent, *, dwell: float, context: Mapping[str, Any]) -> tuple[BehaviorState, str, str]:
        state = self.state

        if event is BehaviorEvent.FORCE_MIN_RISK:
            return BehaviorState.MIN_RISK, "enter", "force_min_risk"

        if event is BehaviorEvent.TIMEOUT or self._timed_out(state, dwell):
            timed = self._timeout_transition(state)
            if timed is not None:
                return timed, "enter", f"timeout_from_{state.value.lower()}"
            if event is BehaviorEvent.TIMEOUT:
                return state, "timeout", f"timeout_hold_{state.value.lower()}"

        if state is BehaviorState.CRUISE:
            if event is BehaviorEvent.LEAD_VEHICLE_DETECTED:
                return BehaviorState.FOLLOW, "enter", "lead_vehicle_detected"
            if event is BehaviorEvent.RED_LIGHT:
                return BehaviorState.STOP, "enter", "red_light"
            if event is BehaviorEvent.UNPROTECTED_LEFT:
                return BehaviorState.YIELD, "enter", "unprotected_left"
            if event is BehaviorEvent.LANE_CHANGE_REQUEST:
                return BehaviorState.LANE_CHANGE, "enter", "lane_change_request"
            if event is BehaviorEvent.OBSTACLE_AHEAD:
                return BehaviorState.AVOID, "enter", "obstacle_ahead"
            if event in {BehaviorEvent.ROUTE_STRAIGHT, BehaviorEvent.ROUTE_TURN, BehaviorEvent.TICK, BehaviorEvent.GREEN_LIGHT}:
                return state, "hold", f"cruise_hold:{event.value.lower()}"

        if state is BehaviorState.FOLLOW:
            if event is BehaviorEvent.LEAD_VEHICLE_CLEARED:
                return BehaviorState.CRUISE, "enter", "lead_vehicle_cleared"
            if event is BehaviorEvent.RED_LIGHT:
                return BehaviorState.STOP, "enter", "red_light_while_follow"
            if event is BehaviorEvent.OBSTACLE_AHEAD:
                return BehaviorState.AVOID, "enter", "obstacle_while_follow"
            if event is BehaviorEvent.FORCE_MIN_RISK:
                return BehaviorState.MIN_RISK, "enter", "force_min_risk"
            return state, "hold", "follow_hold"

        if state is BehaviorState.STOP:
            if event is BehaviorEvent.GREEN_LIGHT:
                return BehaviorState.CRUISE, "enter", "green_light"
            if event is BehaviorEvent.FORCE_MIN_RISK:
                return BehaviorState.MIN_RISK, "enter", "force_min_risk"
            return state, "hold", "stop_hold"

        if state is BehaviorState.YIELD:
            if event is BehaviorEvent.CROSS_TRAFFIC_CLEAR:
                return BehaviorState.CRUISE, "enter", "cross_traffic_clear"
            if event is BehaviorEvent.RED_LIGHT:
                return BehaviorState.STOP, "enter", "red_light_while_yield"
            if event is BehaviorEvent.FORCE_MIN_RISK:
                return BehaviorState.MIN_RISK, "enter", "force_min_risk"
            return state, "hold", "yield_hold"

        if state is BehaviorState.LANE_CHANGE:
            if event is BehaviorEvent.LANE_CHANGE_COMPLETE:
                return BehaviorState.CRUISE, "enter", "lane_change_complete"
            if event is BehaviorEvent.LANE_CHANGE_BLOCKED:
                return BehaviorState.CRUISE, "enter", "lane_change_blocked_abort"
            if event is BehaviorEvent.OBSTACLE_AHEAD:
                return BehaviorState.AVOID, "enter", "obstacle_while_lane_change"
            if event is BehaviorEvent.FORCE_MIN_RISK:
                return BehaviorState.MIN_RISK, "enter", "force_min_risk"
            return state, "hold", "lane_change_hold"

        if state is BehaviorState.AVOID:
            if event is BehaviorEvent.OBSTACLE_CLEARED:
                return BehaviorState.CRUISE, "enter", "obstacle_cleared"
            if event is BehaviorEvent.FORCE_MIN_RISK:
                return BehaviorState.MIN_RISK, "enter", "force_min_risk"
            return state, "hold", "avoid_hold"

        if state is BehaviorState.MIN_RISK:
            if event is BehaviorEvent.TICK and dwell >= self.config.min_risk_hold_s and context.get("clear_to_cruise"):
                return BehaviorState.CRUISE, "enter", "min_risk_release"
            return state, "hold", "min_risk_hold"

        return state, "hold", "noop"

    def _timed_out(self, state: BehaviorState, dwell: float) -> bool:
        limit = {
            BehaviorState.FOLLOW: self.config.follow_timeout_s,
            BehaviorState.STOP: self.config.stop_timeout_s,
            BehaviorState.YIELD: self.config.yield_timeout_s,
            BehaviorState.LANE_CHANGE: self.config.lane_change_timeout_s,
            BehaviorState.AVOID: self.config.avoid_timeout_s,
        }.get(state)
        return limit is not None and dwell >= limit

    def _timeout_transition(self, state: BehaviorState) -> BehaviorState | None:
        return {
            BehaviorState.FOLLOW: BehaviorState.CRUISE,
            BehaviorState.YIELD: BehaviorState.MIN_RISK,
            BehaviorState.LANE_CHANGE: BehaviorState.CRUISE,
            BehaviorState.AVOID: BehaviorState.MIN_RISK,
            # STOP times out into MIN_RISK rather than running the light.
            BehaviorState.STOP: BehaviorState.MIN_RISK,
        }.get(state)

    def _goal_for(self, state: BehaviorState, *, reason: str, context: Mapping[str, Any] | None = None) -> BehaviorGoal:
        context = context or {}
        target_lane = context.get("target_lane_id")
        target_speed = context.get("target_speed_mps")
        if target_speed is None:
            target_speed = {
                BehaviorState.CRUISE: self.config.default_cruise_speed_mps,
                BehaviorState.FOLLOW: min(self.config.default_cruise_speed_mps, 8.0),
                BehaviorState.STOP: 0.0,
                BehaviorState.YIELD: 0.0,
                BehaviorState.LANE_CHANGE: self.config.default_cruise_speed_mps,
                BehaviorState.AVOID: 4.0,
                BehaviorState.MIN_RISK: 0.0,
            }[state]
        hold = {
            BehaviorState.STOP: self.config.stop_timeout_s,
            BehaviorState.YIELD: self.config.yield_timeout_s,
            BehaviorState.LANE_CHANGE: self.config.lane_change_timeout_s,
            BehaviorState.MIN_RISK: self.config.min_risk_hold_s,
        }.get(state)
        return BehaviorGoal(
            state=state,
            reason=reason,
            route_id=self.route_id,
            target_lane_id=None if target_lane is None else int(target_lane),
            target_speed_mps=float(target_speed),
            hold_seconds=hold,
            oracle_inputs=(
                "signal_state",
                "true_lead_vehicle",
                "true_cross_traffic",
                "route_maneuver",
                "obstacle_polygon",
            ),
            observable_inputs=(
                "detected_lead",
                "perceived_signal",
                "navigation_instruction",
                "detected_obstacle",
                "timeout_clock",
            ),
            metadata={
                "behavior_layer": "classic_g1_03",
                "emits_controls": False,
                "emits_local_trajectory": False,
            },
        )
