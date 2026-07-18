"""Bind SafetyDecision to vehicle controls without bypassing the kernel (G3-05).

Hard rules:
- Only track a trajectory when Safety provides an executable decision + resolvable id.
- Never fall back to "first available" candidate.
- EMERGENCY / MINIMAL_RISK / HARD_REJECT / missing exec id → brake, no learning throttle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from classic_stack.control.controller import EgoState
from classic_stack.planning.frenet.planner import Trajectory as CtrlTraj
from classic_stack.planning.frenet.planner import TrajectoryPoint as CtrlPt
from safety_kernel.contracts.types import (
    DecisionKind,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    TrajectoryPoint,
)


class AppliedMode(str, Enum):
    TRACK_APPROVED = "TRACK_APPROVED"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    MINIMAL_RISK_BRAKE = "MINIMAL_RISK_BRAKE"
    HOLD_NO_EXEC = "HOLD_NO_EXEC"
    EXEC_ID_ORPHAN = "EXEC_ID_ORPHAN"


# Decisions that may drive MPC tracking when executed_trajectory_id resolves.
_EXECUTABLE_KINDS = frozenset(
    {
        DecisionKind.ACCEPT,
        DecisionKind.QP,
        DecisionKind.RATO,
        DecisionKind.CLASSIC_FALLBACK,
    }
)


class SupportsControlLoop(Protocol):
    def set_trajectory(self, traj: CtrlTraj, now_s: float) -> None: ...

    def step(self, ego: EgoState, now_s: float, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class AppliedControl:
    """One control tick after Safety arbitration."""

    throttle: float
    brake: float
    steer: float
    applied_mode: AppliedMode
    executed_id: str | None
    decision_kind: str
    notes: tuple[str, ...] = ()

    @property
    def is_track_approved(self) -> bool:
        return self.applied_mode is AppliedMode.TRACK_APPROVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "throttle": self.throttle,
            "brake": self.brake,
            "steer": self.steer,
            "applied_mode": self.applied_mode.value,
            "executed_id": self.executed_id,
            "decision_kind": self.decision_kind,
            "notes": list(self.notes),
        }


def safety_points_to_ctrl(points: tuple[TrajectoryPoint, ...] | list[TrajectoryPoint]) -> CtrlTraj:
    cps = tuple(
        CtrlPt(
            t=p.t,
            x=p.x,
            y=p.y,
            yaw=p.yaw,
            kappa=p.kappa,
            v=p.v,
            a=p.a,
            jerk=getattr(p, "jerk", 0.0),
        )
        for p in points
    )
    return CtrlTraj(points=cps, trajectory_id="exec", source="safety_bind")


def resolve_executable_candidate(
    decision: SafetyDecision,
    candidate_set: PolicyCandidateSet | None,
) -> tuple[PolicyCandidate | None, tuple[str, ...]]:
    """Resolve the single trajectory Safety authorized. No first-available fallback."""
    notes: list[str] = []
    # Prefer explicit exec id; fall back to final/post-repair ids from SafetyDecision.
    tid = (
        decision.executed_trajectory_id
        or decision.post_repair_trajectory_id
        or decision.final_candidate_id
    )
    accepted = decision.accepted_candidate

    if accepted is not None:
        if tid is None or accepted.candidate_id == tid:
            notes.append("from_accepted_candidate")
            return accepted, tuple(notes)
        notes.append("accepted_candidate_id_mismatch")

    if not tid:
        notes.append("no_executed_trajectory_id")
        return None, tuple(notes)

    if candidate_set is not None:
        for c in candidate_set.candidates:
            if c.candidate_id == tid:
                notes.append("from_candidate_set")
                return c, tuple(notes)

    notes.append("exec_id_orphan")
    return None, tuple(notes)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _brake_command(
    *,
    applied_mode: AppliedMode,
    decision: SafetyDecision,
    hold_steer: float,
    brake: float,
    notes: tuple[str, ...],
) -> AppliedControl:
    return AppliedControl(
        throttle=0.0,
        brake=_clamp(brake, 0.0, 1.0),
        steer=_clamp(hold_steer, -1.0, 1.0),
        applied_mode=applied_mode,
        executed_id=decision.executed_trajectory_id,
        decision_kind=str(getattr(decision.decision_kind, "value", decision.decision_kind)),
        notes=notes,
    )


def apply_safety_control(
    decision: SafetyDecision,
    candidate_set: PolicyCandidateSet | None,
    control: SupportsControlLoop,
    ego: EgoState,
    sim_t: float,
    *,
    emergency_brake: float = 1.0,
    minimal_risk_brake: float = 0.75,
    hold_brake: float = 0.5,
) -> AppliedControl:
    """Map SafetyDecision → throttle/brake/steer without demo bypasses.

    Does not force throttle when Safety withholds an executable trajectory.
    Does not select an arbitrary available candidate when exec id is missing.
    """
    kind = decision.decision_kind
    kind_s = str(getattr(kind, "value", kind))
    hold_steer = float(getattr(ego, "steer", 0.0) or 0.0)

    # Non-executable / emergency path: never track learning candidates.
    if kind is DecisionKind.EMERGENCY:
        return _brake_command(
            applied_mode=AppliedMode.EMERGENCY_BRAKE,
            decision=decision,
            hold_steer=hold_steer,
            brake=emergency_brake,
            notes=("decision_emergency",),
        )
    if kind is DecisionKind.MINIMAL_RISK:
        return _brake_command(
            applied_mode=AppliedMode.MINIMAL_RISK_BRAKE,
            decision=decision,
            hold_steer=hold_steer,
            brake=minimal_risk_brake,
            notes=("decision_minimal_risk",),
        )
    if kind is DecisionKind.HARD_REJECT:
        return _brake_command(
            applied_mode=AppliedMode.HOLD_NO_EXEC,
            decision=decision,
            hold_steer=hold_steer,
            brake=hold_brake,
            notes=("decision_hard_reject",),
        )

    if kind not in _EXECUTABLE_KINDS:
        return _brake_command(
            applied_mode=AppliedMode.HOLD_NO_EXEC,
            decision=decision,
            hold_steer=hold_steer,
            brake=hold_brake,
            notes=(f"non_executable_kind:{kind_s}",),
        )

    cand, resolve_notes = resolve_executable_candidate(decision, candidate_set)
    if cand is None:
        mode = (
            AppliedMode.EXEC_ID_ORPHAN
            if "exec_id_orphan" in resolve_notes
            else AppliedMode.HOLD_NO_EXEC
        )
        return _brake_command(
            applied_mode=mode,
            decision=decision,
            hold_steer=hold_steer,
            brake=hold_brake if mode is AppliedMode.HOLD_NO_EXEC else emergency_brake,
            notes=resolve_notes,
        )

    if not cand.points:
        return _brake_command(
            applied_mode=AppliedMode.HOLD_NO_EXEC,
            decision=decision,
            hold_steer=hold_steer,
            brake=hold_brake,
            notes=resolve_notes + ("empty_points",),
        )

    control.set_trajectory(safety_points_to_ctrl(cand.points), sim_t)
    cmd = control.step(ego, sim_t)
    thr = float(getattr(cmd, "throttle", 0.0))
    brk = float(getattr(cmd, "brake", 0.0))
    ste = float(getattr(cmd, "steer", 0.0))
    return AppliedControl(
        throttle=_clamp(thr, 0.0, 1.0),
        brake=_clamp(brk, 0.0, 1.0),
        steer=_clamp(ste, -1.0, 1.0),
        applied_mode=AppliedMode.TRACK_APPROVED,
        executed_id=cand.candidate_id,
        decision_kind=kind_s,
        notes=resolve_notes,
    )


def evaluate_episode_status(
    *,
    steps: int,
    max_steps: int,
    min_steps: int,
    distance_m: float,
    decisions: list[str],
    n_track_approved: int,
    n_emergency: int,
    sources_seen: list[str] | set[str],
    camera_frames: int,
    fault: str,
    mode: str,
    classic_current_frame: bool,
) -> tuple[str, list[str]]:
    """Return (COMPLETED|FAILED, reasons). Replaces steps-only success."""
    reasons: list[str] = []
    need = min(min_steps, max_steps)
    if steps < need:
        reasons.append(f"steps_short:{steps}<{need}")

    if mode == "VLA_SAFETY" and classic_current_frame:
        reasons.append("classic_current_frame_under_vla_safety")

    fault_s = (fault or "").strip().lower()
    if fault_s == "timeout":
        # Fault path: must not hang; must not claim VLA-driven long motion without sources.
        if steps < need:
            pass  # already recorded
        # long distance with zero VLA sources is invalid for "timeout survived" narrative
        if distance_m > 30.0 and not sources_seen:
            reasons.append("timeout_long_distance_without_sources")
        if not reasons:
            return "COMPLETED", []
        return "FAILED", reasons

    # Nominal path
    if camera_frames < 1:
        reasons.append("camera_frames_zero")

    src = set(sources_seen)
    if mode == "VLA_SAFETY" and not any(s.startswith("vla") for s in src):
        reasons.append("no_vla_sources_seen")

    if steps > 0:
        if n_track_approved < max(1, int(0.15 * steps)):
            reasons.append(f"track_approved_low:{n_track_approved}/{steps}")
        if n_emergency / max(steps, 1) >= 0.25:
            reasons.append(f"emergency_ratio_high:{n_emergency}/{steps}")

    if distance_m > 20.0 and decisions and all(d == "EMERGENCY" for d in decisions):
        reasons.append("all_emergency_long_run")

    if distance_m > 20.0 and steps >= 15:
        tail = decisions[-15:] if len(decisions) >= 15 else decisions
        if tail and all(d == "EMERGENCY" for d in tail) and n_track_approved == 0:
            reasons.append("decision_tail_all_emergency_no_track")

    if reasons:
        return "FAILED", reasons
    return "COMPLETED", []
