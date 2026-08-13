"""Typed Safety contracts (frozen dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class SafetyMode(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    MINIMAL_RISK = "MINIMAL_RISK"
    EMERGENCY = "EMERGENCY"


class DecisionKind(str, Enum):
    ACCEPT = "ACCEPT"
    HARD_REJECT = "HARD_REJECT"
    QP = "QP"  # G2-02 longitudinal repair
    RATO = "RATO"  # reserved for G2-03
    CLASSIC_FALLBACK = "CLASSIC_FALLBACK"
    MINIMAL_RISK = "MINIMAL_RISK"
    EMERGENCY = "EMERGENCY"


class FallbackTarget(str, Enum):
    CLASSIC = "CLASSIC"
    MINIMAL_RISK = "MINIMAL_RISK"
    EMERGENCY = "EMERGENCY"
    HOLD_LAST = "HOLD_LAST"


class CandidateSource(str, Enum):
    CLASSIC = "classic"
    VLA_FAST = "vla_fast"
    VLA_SLOW = "vla_slow"
    SHADOW = "shadow"
    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    UNKNOWN = "unknown"


class ObservationPrivilege(str, Enum):
    """Runtime validator may only consume OBSERVABLE."""

    OBSERVABLE = "observable"
    ORACLE = "oracle"  # offline labels / evaluation only


class EventPhase(str, Enum):
    NORMAL = "normal"
    PRE_EVENT = "pre_event"
    INTERVENTION = "intervention"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float
    kappa: float
    v: float
    a: float
    jerk: float = 0.0


@dataclass(frozen=True)
class PolicyCandidate:
    candidate_id: str
    source: CandidateSource
    generated_time_s: float
    valid_until_s: float
    probability: float
    points: tuple[TrajectoryPoint, ...]
    behavior: str = ""
    critical_actor: str | None = None
    conflict_type: str | None = None
    risk_horizon_s: float = 0.0
    intended_action: str = ""
    uncertainty: float = 0.0
    availability: bool = True
    dynamics_meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trajectory_id(self) -> str:
        return self.candidate_id

    @property
    def horizon_s(self) -> float:
        if not self.points:
            return 0.0
        return float(self.points[-1].t - self.points[0].t)


@dataclass(frozen=True)
class PolicyCandidateSet:
    run_id: str
    frame_id: str
    scenario_id: str
    model_id: str
    carla_frame: int
    simulation_time_s: float
    wall_time_s: float
    candidates: tuple[PolicyCandidate, ...]
    schema_version: str
    coordinate_frame: str = "map"


@dataclass(frozen=True)
class TrackedObject:
    actor_id: str
    class_name: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length_m: float
    width_m: float
    observed_time_s: float
    lost: bool = False
    source: str = "observable"
    cov_xx: float = 0.25
    cov_yy: float = 0.25


@dataclass(frozen=True)
class TrafficLightObs:
    light_id: str
    state: str  # red|yellow|green|unknown
    distance_m: float
    observed_time_s: float
    # Additive H2 geometry: distance from the ego projection to the trigger/stop
    # line along the active route.  Older producers leave these unset and retain
    # the legacy Euclidean-distance semantics above.
    stop_line_distance_m: float | None = None
    controls_ego_lane: bool | None = None


@dataclass(frozen=True)
class ObservableSnapshot:
    """Runtime observation bundle subset for Safety (Observable only)."""

    run_id: str
    frame_id: str
    scenario_id: str
    simulation_time_s: float
    wall_time_s: float
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_v: float
    ego_a: float = 0.0
    observed_time_s: float = 0.0
    freshness_s: float = 0.0
    speed_limit_mps: float | None = None
    actors: tuple[TrackedObject, ...] = ()
    traffic_lights: tuple[TrafficLightObs, ...] = ()
    # Axis-aligned corridor in map frame for road checks (optional polyline half-width model).
    corridor_centerline: tuple[tuple[float, float], ...] = ()
    corridor_half_width_m: float = 1.75
    privilege: ObservationPrivilege = ObservationPrivilege.OBSERVABLE
    # Explicit oracle fields must not be read by runtime validator paths.
    oracle_fields: Mapping[str, Any] = field(default_factory=dict)
    # Identity fields aligned with PolicyCandidateSet for hard contract checks.
    schema_version: str = "safedrive.safety.contracts.v1"
    coordinate_frame: str = "map"


@dataclass(frozen=True)
class ComponentAvailability:
    classic: bool = True
    vla: bool = False
    world: bool = False
    safety: bool = True
    detail: Mapping[str, str] = field(default_factory=dict)

    @property
    def learning_all_failed(self) -> bool:
        return (not self.vla) and (not self.world)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classic": self.classic,
            "vla": self.vla,
            "world": self.world,
            "safety": self.safety,
            "detail": dict(self.detail),
            "learning_all_failed": self.learning_all_failed,
        }


@dataclass(frozen=True)
class ConstraintMargin:
    name: str
    margin: float
    hard: bool
    first_violation_time_s: float | None = None
    actor_id: str | None = None
    rule_id: str | None = None
    message: str = ""

    @property
    def violated(self) -> bool:
        return self.margin < 0.0


@dataclass(frozen=True)
class FallbackRequest:
    reason_code: str
    target: FallbackTarget
    from_state: SafetyMode
    to_state: SafetyMode
    urgency: float
    source_candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "target": self.target.value,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "urgency": self.urgency,
            "source_candidate_id": self.source_candidate_id,
        }


@dataclass(frozen=True)
class SafetyDecision:
    decision_id: str
    run_id: str
    frame_id: str
    prefilter_candidate_ids: tuple[str, ...]
    final_candidate_id: str | None
    pre_repair_trajectory_id: str | None
    post_repair_trajectory_id: str | None
    executed_trajectory_id: str | None
    constraint_margins: tuple[ConstraintMargin, ...]
    decision_kind: DecisionKind
    modification_norm: float
    slack: float
    progress_loss: float
    solver_status: str
    latency_ms: float
    state_before: SafetyMode
    state_after: SafetyMode
    recovery_conditions: tuple[str, ...]
    fallback_request: FallbackRequest | None
    reject_reasons: tuple[str, ...]
    learning_modules_required: bool = False
    accepted_candidate: PolicyCandidate | None = None


@dataclass(frozen=True)
class SafetyEvent:
    event_id: str
    run_id: str
    frame_id: str
    phase: EventPhase
    decision: SafetyDecision | None
    availability: ComponentAvailability
    privilege: ObservationPrivilege
    message: str
    simulation_time_s: float
    wall_time_s: float
