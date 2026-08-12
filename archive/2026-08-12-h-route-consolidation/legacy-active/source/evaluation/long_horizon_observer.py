"""15–20 second rolling R2 V3 manoeuvre observer.

This module owns no CARLA tick and performs no cleanup.  A live runner feeds
one observable record per synchronous tick, continues replanning in the same
fixture session, and calls :meth:`finalize` only before the runner performs its
single end-of-session cleanup.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from driving_vla.model.navigation_contract import RouteContextV3, TargetLaneSide

from .maneuver_completion import (
    evaluate_cut_in_completion,
    evaluate_follow_stop_completion,
    evaluate_overtake_completion,
    evaluate_route_maneuver_completion,
    evaluate_traffic_control_completion,
    evaluate_yield_wait_completion,
    signed_route_projection,
)


class InteractionBehavior(str, Enum):
    FOLLOW_STOP = "FOLLOW_STOP"
    CUT_IN_AVOID = "CUT_IN_AVOID"
    YIELD_WAIT = "YIELD_WAIT"
    OVERTAKE_REJOIN = "OVERTAKE_REJOIN"
    CLEAR = "CLEAR"
    TRAFFIC_CONTROL = "TRAFFIC_CONTROL"


class BehaviorPhase(str, Enum):
    APPROACH = "APPROACH"
    WAIT = "WAIT"
    STOPPED = "STOPPED"
    DEPART = "DEPART"
    AVOID = "AVOID"
    PASS = "PASS"
    REJOIN = "REJOIN"
    RESUME = "RESUME"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class LongHorizonAcceptanceReport:
    case_id: str
    behavior: str
    completed: bool
    reason_codes: tuple[str, ...]
    duration_s: float
    tick_count: int
    replan_count: int
    mpc_acceptance_rate: float
    route_completion: Mapping[str, Any]
    interaction_completion: Mapping[str, Any]
    phase_history: tuple[str, ...]
    execution_binding_ok: bool
    spectator_follow_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(tick: Any, name: str, default: Any = None) -> Any:
    return tick.get(name, default) if isinstance(tick, Mapping) else getattr(tick, name, default)


class LongHorizonObserver:
    """Deterministic state machine plus independent completion evaluators."""

    def __init__(
        self,
        *,
        case_id: str,
        route_context: RouteContextV3,
        behavior: InteractionBehavior | str,
        conflict_side: str = "",
        conflict_point_s_m: float | None = None,
        minimum_duration_s: float = 15.0,
        minimum_mpc_acceptance_rate: float = 0.90,
    ) -> None:
        self.case_id = str(case_id)
        self.route_context = route_context
        self.behavior = InteractionBehavior(behavior)
        self.conflict_side = str(conflict_side or "").lower()
        self.conflict_point_s_m = conflict_point_s_m
        self.minimum_duration_s = float(minimum_duration_s)
        self.minimum_mpc_acceptance_rate = float(minimum_mpc_acceptance_rate)
        self._ticks: list[Any] = []
        self._phases: list[BehaviorPhase] = [BehaviorPhase.APPROACH]
        self._stop_run = 0

    @property
    def phase(self) -> BehaviorPhase:
        return self._phases[-1]

    @property
    def ticks(self) -> tuple[Any, ...]:
        return tuple(self._ticks)

    def _transition(self, phase: BehaviorPhase) -> None:
        if phase is not self.phase:
            self._phases.append(phase)

    def observe(self, tick: Any) -> None:
        time_s = float(_field(tick, "simulation_time_s", 0.0))
        if self._ticks and time_s <= float(
            _field(self._ticks[-1], "simulation_time_s", 0.0)
        ):
            raise ValueError("long-horizon tick time must be strictly increasing")
        self._ticks.append(tick)
        if bool(_field(tick, "collision", False)) or bool(
            _field(tick, "offroad", False)
        ):
            self._transition(BehaviorPhase.FAILED)
            return
        speed = float(_field(tick, "ego_v", 0.0))
        _route_s, signed_d = signed_route_projection(
            self.route_context.route_xy,
            float(_field(tick, "ego_x", 0.0)),
            float(_field(tick, "ego_y", 0.0)),
        )
        stopped = speed <= 0.35
        self._stop_run = self._stop_run + 1 if stopped else 0

        if self.behavior is InteractionBehavior.FOLLOW_STOP:
            if self._stop_run >= 10:
                self._transition(BehaviorPhase.STOPPED)
            elif self.phase is BehaviorPhase.STOPPED and speed >= 0.75:
                self._transition(BehaviorPhase.RESUME)
        elif self.behavior is InteractionBehavior.YIELD_WAIT:
            if bool(_field(tick, "conflict_active", False)) and stopped:
                self._transition(BehaviorPhase.WAIT)
            elif (
                self.phase is BehaviorPhase.WAIT
                and not bool(_field(tick, "conflict_active", False))
                and speed >= 0.75
            ):
                self._transition(BehaviorPhase.RESUME)
        elif self.behavior is InteractionBehavior.CUT_IN_AVOID:
            away = (
                signed_d if self.conflict_side == "left" else -signed_d
            )
            if away >= 0.50:
                self._transition(BehaviorPhase.AVOID)
            elif self.phase is BehaviorPhase.AVOID and abs(signed_d) <= 0.60:
                self._transition(BehaviorPhase.REJOIN)
        elif self.behavior is InteractionBehavior.OVERTAKE_REJOIN:
            if abs(signed_d) >= 0.50 and self.phase is BehaviorPhase.APPROACH:
                self._transition(BehaviorPhase.DEPART)
            actor_lon = _field(tick, "actor_lon_m")
            if (
                actor_lon is not None
                and float(actor_lon) <= -2.0
                and self.phase in {BehaviorPhase.DEPART, BehaviorPhase.PASS}
            ):
                self._transition(BehaviorPhase.PASS)
            if self.phase is BehaviorPhase.PASS and abs(signed_d) <= 0.60:
                self._transition(BehaviorPhase.REJOIN)
        elif self.behavior is InteractionBehavior.TRAFFIC_CONTROL:
            signal = str(
                _field(tick, "traffic_signal_state", "UNKNOWN") or "UNKNOWN"
            )
            if signal in {"RED", "STOP_SIGN"} and self._stop_run >= 10:
                self._transition(BehaviorPhase.STOPPED)
            elif self.phase is BehaviorPhase.STOPPED and signal == "GREEN" and speed >= 0.75:
                self._transition(BehaviorPhase.RESUME)

    def _interaction_report(
        self, *, final_actor_lon_m: float | None
    ) -> Mapping[str, Any]:
        route = self.route_context.route_xy
        if self.behavior is InteractionBehavior.FOLLOW_STOP:
            return evaluate_follow_stop_completion(
                route_xy=route, ticks=self._ticks
            ).to_dict()
        if self.behavior is InteractionBehavior.CUT_IN_AVOID:
            side = self.conflict_side.lower()
            away = (
                TargetLaneSide.RIGHT
                if side == "left"
                else TargetLaneSide.LEFT
                if side == "right"
                else TargetLaneSide.NONE
            )
            # A topology-only lane permission is not enough to reinterpret a
            # spatial cut-in trace as a temporal yield.  Synthetic/offline
            # traces may deliberately exercise the spatial state machine even
            # when the route context has no adjacent authorized lane.  Enable
            # the fallback only when the executable trace actually selected a
            # temporal candidate, as the live runner records on each VLA tick.
            temporal_observed = any(
                str(_field(tick, "alternative_kind", "")).upper()
                == "TEMPORAL_YIELD"
                for tick in self._ticks
            )
            return evaluate_cut_in_completion(
                route_xy=route,
                ticks=self._ticks,
                conflict_side=self.conflict_side,
                allow_temporal_fallback=(
                    away is not TargetLaneSide.NONE
                    and not self.route_context.lane(away).authorized
                    and temporal_observed
                ),
            ).to_dict()
        if self.behavior is InteractionBehavior.YIELD_WAIT:
            if self.conflict_point_s_m is None:
                return {
                    "completed": False,
                    "reason_codes": ("CONFLICT_POINT_REQUIRED",),
                }
            return evaluate_yield_wait_completion(
                route_xy=route,
                ticks=self._ticks,
                conflict_point_s_m=float(self.conflict_point_s_m),
            ).to_dict()
        if self.behavior is InteractionBehavior.OVERTAKE_REJOIN:
            return evaluate_overtake_completion(
                family="obstruction",
                route_xy=route,
                ticks=self._ticks,
                final_actor_lon_m=final_actor_lon_m,
            ).to_dict()
        if self.behavior is InteractionBehavior.TRAFFIC_CONTROL:
            return evaluate_traffic_control_completion(
                ticks=self._ticks
            ).to_dict()

        # Clear is a singleton/no-intervention state machine.  Snake detection
        # is relative to the selected executable path, not the coarse map
        # mission centerline: a valid native VLA turn may be offset from that
        # centerline while MPC tracks it smoothly.
        tracking_errors = [
            float(
                _field(
                    tick,
                    "path_tracking_error_m",
                    signed_route_projection(
                        route,
                        float(_field(tick, "ego_x", 0.0)),
                        float(_field(tick, "ego_y", 0.0)),
                    )[1],
                )
            )
            for tick in self._ticks
        ]
        ordered_abs = sorted(abs(value) for value in tracking_errors)
        p95 = (
            ordered_abs[
                max(
                    0,
                    min(
                        len(ordered_abs) - 1,
                        int(math.ceil(0.95 * len(ordered_abs))) - 1,
                    ),
                )
            ]
            if ordered_abs
            else math.inf
        )
        significant_flips = sum(
            first * second < 0.0
            and max(abs(first), abs(second)) >= 0.25
            for first, second in zip(tracking_errors, tracking_errors[1:])
        )
        tail = self._ticks[-min(10, len(self._ticks)) :]
        tail_speed = (
            sum(float(_field(tick, "ego_v", 0.0)) for tick in tail)
            / len(tail)
            if tail
            else 0.0
        )
        reasons: list[str] = []
        if any(bool(_field(tick, "candidate1_available", False)) for tick in self._ticks):
            reasons.append("CLEAR_CREATED_MEANINGLESS_CANDIDATE")
        if p95 > 0.85 or significant_flips >= 4:
            reasons.append("CLEAR_SNAKE_OR_LATERAL_EXCURSION")
        if tail_speed < 0.75:
            reasons.append("CLEAR_ABNORMAL_SLOWDOWN")
        return {
            "behavior": "CLEAR",
            "completed": not reasons,
            "reason_codes": tuple(reasons),
            "path_tracking_p95_m": float(p95),
            "significant_tracking_error_flips": int(significant_flips),
            "tail_mean_speed_mps": float(tail_speed),
        }

    def finalize(
        self,
        *,
        final_actor_lon_m: float | None = None,
    ) -> LongHorizonAcceptanceReport:
        route_report = evaluate_route_maneuver_completion(
            route_context=self.route_context,
            ticks=self._ticks,
        ).to_dict()
        interaction = self._interaction_report(
            final_actor_lon_m=final_actor_lon_m
        )
        duration = (
            float(_field(self._ticks[-1], "simulation_time_s", 0.0))
            - float(_field(self._ticks[0], "simulation_time_s", 0.0))
            if len(self._ticks) >= 2
            else 0.0
        )
        replan_ids = {
            str(value)
            for tick in self._ticks
            if (value := _field(tick, "replan_id")) not in (None, "")
        }
        mpc_values = [
            str(_field(tick, "mpc_status", "")).lower()
            for tick in self._ticks
            if str(_field(tick, "mpc_status", ""))
        ]
        mpc_rate = (
            sum("solved" in value for value in mpc_values) / len(mpc_values)
            if mpc_values
            else 0.0
        )
        execution_binding_ok = all(
            (
                not str(_field(tick, "selected_candidate_id", ""))
                or (
                    str(_field(tick, "selected_candidate_id", ""))
                    == str(_field(tick, "executed_candidate_id", ""))
                    and str(_field(tick, "selected_candidate_id", ""))
                    in str(_field(tick, "source_id", ""))
                )
            )
            for tick in self._ticks
        )
        spectator_follow_ok = all(
            bool(_field(tick, "spectator_follow_ok", True))
            for tick in self._ticks
        )
        reasons: list[str] = []
        if duration + 1.0e-9 < self.minimum_duration_s:
            reasons.append("LONG_HORIZON_LT_15S")
        if len(replan_ids) < 2:
            reasons.append("ROLLING_REPLAN_NOT_OBSERVED")
        if mpc_rate + 1.0e-9 < self.minimum_mpc_acceptance_rate:
            reasons.append("MPC_ACCEPTANCE_LT_90PCT")
        if not execution_binding_ok:
            reasons.append("EXECUTION_BINDING_FAILURE")
        if not spectator_follow_ok:
            reasons.append("SPECTATOR_FOLLOW_FAILURE")
        if not bool(route_report["completed"]):
            reasons.extend(
                f"ROUTE:{reason}" for reason in route_report["reason_codes"]
            )
        if not bool(interaction.get("completed")):
            reasons.extend(
                f"INTERACTION:{reason}"
                for reason in interaction.get("reason_codes", ("FAILED",))
            )
        completed = not reasons
        if completed:
            self._transition(BehaviorPhase.COMPLETE)
        else:
            self._transition(BehaviorPhase.FAILED)
        return LongHorizonAcceptanceReport(
            case_id=self.case_id,
            behavior=self.behavior.value,
            completed=completed,
            reason_codes=tuple(dict.fromkeys(reasons)),
            duration_s=duration,
            tick_count=len(self._ticks),
            replan_count=len(replan_ids),
            mpc_acceptance_rate=mpc_rate,
            route_completion=route_report,
            interaction_completion=interaction,
            phase_history=tuple(phase.value for phase in self._phases),
            execution_binding_ok=execution_binding_ok,
            spectator_follow_ok=spectator_follow_ok,
        )


__all__ = [
    "BehaviorPhase",
    "InteractionBehavior",
    "LongHorizonAcceptanceReport",
    "LongHorizonObserver",
]
