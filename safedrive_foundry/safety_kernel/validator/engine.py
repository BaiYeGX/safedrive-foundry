"""Trajectory Validator engine: prefilter + full checks, no learning dependency."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from safety_kernel.config import SafetyKernelConfig, load_safety_config
from safety_kernel.contracts.types import (
    CandidateSource,
    ComponentAvailability,
    ConstraintMargin,
    DecisionKind,
    EventPhase,
    FallbackRequest,
    FallbackTarget,
    ObservableSnapshot,
    ObservationPrivilege,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyEvent,
    SafetyMode,
)
from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.validator.checks import (
    check_set_identity_and_contract,
    hard_violations,
    run_full_checks,
    run_prefilter_checks,
    run_state_checks,
)


class ValidationStage(str, Enum):
    """PREFILTER = hard numeric/schema/freshness; FINAL = full hard safety checks."""

    PREFILTER = "prefilter"
    FINAL = "final"


@dataclass
class ValidatorResult:
    decision: SafetyDecision
    events: list[SafetyEvent] = field(default_factory=list)
    accepted: PolicyCandidate | None = None
    prefilter_passed: list[PolicyCandidate] = field(default_factory=list)
    state_margins: list[ConstraintMargin] = field(default_factory=list)
    deadline_miss: bool = False
    stage: ValidationStage = ValidationStage.FINAL


def _decision_id() -> str:
    return uuid.uuid4().hex[:16]


def _pick_preferred(
    passed: Sequence[PolicyCandidate],
    *,
    prefer_classic: bool = True,
) -> PolicyCandidate | None:
    if not passed:
        return None
    # Prefer Classic when learning may be unavailable; then highest probability.
    if prefer_classic:
        classic = [c for c in passed if c.source is CandidateSource.CLASSIC]
        pool = classic if classic else list(passed)
    else:
        pool = list(passed)
    return max(pool, key=lambda c: (c.probability, -c.uncertainty, c.candidate_id))


class TrajectoryValidator:
    """Independent validator usable with Classic-only stack (learning modules off)."""

    def __init__(self, config: SafetyKernelConfig | None = None) -> None:
        self.config = config or load_safety_config()
        self._events: list[SafetyEvent] = []
        self._failure_samples: list[dict] = []
        self._latencies_state_ms: list[float] = []
        self._latencies_candidate_ms: list[float] = []
        self._deadline_misses = 0

    @property
    def events(self) -> list[SafetyEvent]:
        return list(self._events)

    @property
    def failure_samples(self) -> list[dict]:
        return list(self._failure_samples)

    @property
    def latency_state_ms(self) -> list[float]:
        return list(self._latencies_state_ms)

    @property
    def latency_candidate_ms(self) -> list[float]:
        return list(self._latencies_candidate_ms)

    @property
    def deadline_misses(self) -> int:
        return self._deadline_misses

    def check_state(
        self,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        availability: ComponentAvailability | None = None,
        mode: SafetyMode = SafetyMode.NORMAL,
    ) -> ValidatorResult:
        t0 = time.perf_counter()
        now = obs.simulation_time_s if now_s is None else now_s
        avail = availability or ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        margins = run_state_checks(obs, self.config, now_s=now)
        viol = hard_violations(margins)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies_state_ms.append(latency_ms)
        deadline_miss = latency_ms > self.config.state_check_deadline_ms
        if deadline_miss:
            self._deadline_misses += 1

        if viol:
            # Integrity failures (ego/actor non-finite, Oracle privilege) escalate to EMERGENCY.
            integrity = {"ego_numeric", "privilege", "actor_numeric"}
            kind = (
                DecisionKind.EMERGENCY
                if any(v.name in integrity for v in viol)
                else DecisionKind.MINIMAL_RISK
            )
            target = FallbackTarget.EMERGENCY if kind is DecisionKind.EMERGENCY else FallbackTarget.MINIMAL_RISK
            to_state = SafetyMode.EMERGENCY if kind is DecisionKind.EMERGENCY else SafetyMode.MINIMAL_RISK
            fallback = FallbackRequest(
                reason_code="state_hard_violation",
                target=target,
                from_state=mode,
                to_state=to_state,
                urgency=1.0,
            )
            decision = SafetyDecision(
                decision_id=_decision_id(),
                run_id=obs.run_id,
                frame_id=obs.frame_id,
                prefilter_candidate_ids=(),
                final_candidate_id=None,
                pre_repair_trajectory_id=None,
                post_repair_trajectory_id=None,
                executed_trajectory_id=None,
                constraint_margins=tuple(margins),
                decision_kind=kind,
                modification_norm=0.0,
                slack=min(m.margin for m in margins),
                progress_loss=0.0,
                solver_status="n/a",
                latency_ms=latency_ms,
                state_before=mode,
                state_after=to_state,
                recovery_conditions=("state_clear", "obs_fresh"),
                fallback_request=fallback,
                reject_reasons=tuple(f"{v.name}:{v.message}" for v in viol),
                learning_modules_required=False,
            )
        else:
            decision = SafetyDecision(
                decision_id=_decision_id(),
                run_id=obs.run_id,
                frame_id=obs.frame_id,
                prefilter_candidate_ids=(),
                final_candidate_id=None,
                pre_repair_trajectory_id=None,
                post_repair_trajectory_id=None,
                executed_trajectory_id=None,
                constraint_margins=tuple(margins),
                decision_kind=DecisionKind.ACCEPT,
                modification_norm=0.0,
                slack=min(m.margin for m in margins),
                progress_loss=0.0,
                solver_status="n/a",
                latency_ms=latency_ms,
                state_before=mode,
                state_after=mode,
                recovery_conditions=(),
                fallback_request=None,
                reject_reasons=(),
                learning_modules_required=False,
            )
        events = self._maybe_event(decision, avail, obs, phase=EventPhase.NORMAL if not viol else EventPhase.INTERVENTION)
        return ValidatorResult(
            decision=decision,
            events=events,
            state_margins=list(margins),
            deadline_miss=deadline_miss,
        )

    def validate_candidates(
        self,
        candidate_set: PolicyCandidateSet,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        availability: ComponentAvailability | None = None,
        mode: SafetyMode = SafetyMode.NORMAL,
        stage: ValidationStage = ValidationStage.FINAL,
    ) -> ValidatorResult:
        t0 = time.perf_counter()
        now = candidate_set.simulation_time_s if now_s is None else now_s
        avail = availability or ComponentAvailability(
            classic=True,
            vla=False,
            world=False,
            safety=True,
            detail={"note": "default_classic_only"},
        )

        # Runtime must not depend on learning modules.
        if not avail.safety:
            raise RuntimeError("Safety component marked unavailable — cannot validate")

        # Hard identity / contract consistency before any ACCEPT path.
        identity_margins = check_set_identity_and_contract(
            candidate_set,
            obs,
            expected_schema_version=SCHEMA_VERSION,
        )
        identity_viol = hard_violations(identity_margins)
        if identity_viol:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._latencies_candidate_ms.append(latency_ms)
            deadline_miss = latency_ms > self.config.candidate_check_deadline_ms
            if deadline_miss:
                self._deadline_misses += 1
            reasons = tuple(f"identity:{v.message}" for v in identity_viol)
            fallback = FallbackRequest(
                reason_code="identity_contract_mismatch",
                target=FallbackTarget.MINIMAL_RISK,
                from_state=mode,
                to_state=SafetyMode.MINIMAL_RISK,
                urgency=1.0,
            )
            decision = SafetyDecision(
                decision_id=_decision_id(),
                run_id=candidate_set.run_id,
                frame_id=candidate_set.frame_id,
                prefilter_candidate_ids=tuple(c.candidate_id for c in candidate_set.candidates),
                final_candidate_id=None,
                pre_repair_trajectory_id=None,
                post_repair_trajectory_id=None,
                executed_trajectory_id=None,
                constraint_margins=tuple(identity_margins),
                decision_kind=DecisionKind.HARD_REJECT,
                modification_norm=0.0,
                slack=-1.0,
                progress_loss=1.0,
                solver_status="n/a",
                latency_ms=latency_ms,
                state_before=mode,
                state_after=SafetyMode.MINIMAL_RISK,
                recovery_conditions=("identity_aligned", "schema_match"),
                fallback_request=fallback,
                reject_reasons=reasons,
                learning_modules_required=False,
            )
            events = self._maybe_event(decision, avail, obs, phase=EventPhase.INTERVENTION)
            return ValidatorResult(
                decision=decision,
                events=events,
                accepted=None,
                prefilter_passed=[],
                deadline_miss=deadline_miss,
                stage=stage,
            )

        passed: list[PolicyCandidate] = []
        all_margins: list[ConstraintMargin] = list(identity_margins)
        reject_reasons: list[str] = []
        seen_ids: list[str] = []
        check_fn = run_prefilter_checks if stage is ValidationStage.PREFILTER else run_full_checks
        hard_fail_count = 0
        source_drop_count = 0

        for cand in candidate_set.candidates:
            seen_ids.append(cand.candidate_id)
            if not cand.candidate_id:
                reject_reasons.append(":missing_candidate_id")
                hard_fail_count += 1
                continue
            if not cand.availability:
                reject_reasons.append(f"{cand.candidate_id}:unavailable")
                source_drop_count += 1
                continue
            # If VLA/World sources but modules failed, reject those sources; Classic still allowed.
            if cand.source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW} and not avail.vla:
                reject_reasons.append(f"{cand.candidate_id}:vla_unavailable")
                source_drop_count += 1
                continue
            if cand.source is CandidateSource.CLASSIC and not avail.classic:
                reject_reasons.append(f"{cand.candidate_id}:classic_unavailable")
                source_drop_count += 1
                continue
            margins = check_fn(cand, obs, self.config, now_s=now)
            all_margins.extend(margins)
            viol = hard_violations(margins)
            if viol:
                reason = f"{cand.candidate_id}:" + ",".join(f"{v.name}:{v.message}" for v in viol)
                reject_reasons.append(reason)
                hard_fail_count += 1
                self._record_failure(cand.candidate_id, reason, margins)
                continue
            passed.append(cand)

        prefer_classic = avail.learning_all_failed or (not avail.vla)
        chosen = _pick_preferred(passed, prefer_classic=prefer_classic)
        # Classic fallback selection: if learning sources failed hard but classic passed.
        classic_accepted_after_learning_drop = (
            chosen is not None
            and chosen.source is CandidateSource.CLASSIC
            and any("vla_unavailable" in r or "vla_" in r for r in reject_reasons)
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies_candidate_ms.append(latency_ms)
        deadline_miss = latency_ms > self.config.candidate_check_deadline_ms
        if deadline_miss:
            self._deadline_misses += 1

        if chosen is not None:
            kind = DecisionKind.ACCEPT
            # Explicitly record that Classic was selected after learning drop (still ACCEPT).
            if classic_accepted_after_learning_drop:
                reject_reasons = list(reject_reasons) + ["selected_classic_after_learning_drop"]
            decision = SafetyDecision(
                decision_id=_decision_id(),
                run_id=candidate_set.run_id,
                frame_id=candidate_set.frame_id,
                prefilter_candidate_ids=tuple(seen_ids),
                final_candidate_id=chosen.candidate_id,
                pre_repair_trajectory_id=chosen.candidate_id,
                post_repair_trajectory_id=chosen.candidate_id,
                executed_trajectory_id=chosen.candidate_id,
                constraint_margins=tuple(all_margins),
                decision_kind=kind,
                modification_norm=0.0,
                slack=min((m.margin for m in all_margins), default=0.0),
                progress_loss=0.0,
                solver_status="n/a",
                latency_ms=latency_ms,
                state_before=mode,
                state_after=mode if mode is not SafetyMode.EMERGENCY else SafetyMode.EMERGENCY,
                recovery_conditions=(),
                fallback_request=None,
                reject_reasons=tuple(reject_reasons),
                learning_modules_required=False,
                accepted_candidate=chosen,
            )
            phase = EventPhase.NORMAL
        else:
            kind, fallback, to_state = self._no_legal_resolution(
                avail=avail,
                mode=mode,
                had_candidates=bool(candidate_set.candidates),
                hard_fail_count=hard_fail_count,
                source_drop_count=source_drop_count,
            )
            decision = SafetyDecision(
                decision_id=_decision_id(),
                run_id=candidate_set.run_id,
                frame_id=candidate_set.frame_id,
                prefilter_candidate_ids=tuple(seen_ids),
                final_candidate_id=None,
                pre_repair_trajectory_id=None,
                post_repair_trajectory_id=None,
                executed_trajectory_id=None,
                constraint_margins=tuple(all_margins),
                decision_kind=kind,
                modification_norm=0.0,
                slack=min((m.margin for m in all_margins), default=-1.0),
                progress_loss=1.0,
                solver_status="n/a",
                latency_ms=latency_ms,
                state_before=mode,
                state_after=to_state,
                recovery_conditions=("legal_candidate", "obs_fresh"),
                fallback_request=fallback,
                reject_reasons=tuple(reject_reasons) if reject_reasons else ("no_candidates",),
                learning_modules_required=False,
            )
            phase = EventPhase.INTERVENTION

        events = self._maybe_event(decision, avail, obs, phase=phase)
        return ValidatorResult(
            decision=decision,
            events=events,
            accepted=chosen,
            prefilter_passed=passed,
            deadline_miss=deadline_miss,
            stage=stage,
        )

    def _no_legal_resolution(
        self,
        *,
        avail: ComponentAvailability,
        mode: SafetyMode,
        had_candidates: bool,
        hard_fail_count: int,
        source_drop_count: int,
    ) -> tuple[DecisionKind, FallbackRequest, SafetyMode]:
        """Map empty/illegal candidate outcomes to decision + fallback (no silent drop)."""
        if not avail.classic and avail.learning_all_failed:
            return (
                DecisionKind.EMERGENCY,
                FallbackRequest(
                    reason_code="no_classic_no_learning",
                    target=FallbackTarget.EMERGENCY,
                    from_state=mode,
                    to_state=SafetyMode.EMERGENCY,
                    urgency=1.0,
                ),
                SafetyMode.EMERGENCY,
            )
        if not avail.classic:
            return (
                DecisionKind.EMERGENCY,
                FallbackRequest(
                    reason_code="no_legal_candidate_no_classic",
                    target=FallbackTarget.EMERGENCY,
                    from_state=mode,
                    to_state=SafetyMode.EMERGENCY,
                    urgency=1.0,
                ),
                SafetyMode.EMERGENCY,
            )
        if had_candidates and hard_fail_count > 0:
            # All provided candidates failed hard checks → MR + classic replan request.
            return (
                DecisionKind.HARD_REJECT if source_drop_count == 0 else DecisionKind.MINIMAL_RISK,
                FallbackRequest(
                    reason_code="all_candidates_hard_rejected" if source_drop_count == 0 else "classic_candidates_rejected",
                    target=FallbackTarget.MINIMAL_RISK,
                    from_state=mode,
                    to_state=SafetyMode.MINIMAL_RISK,
                    urgency=0.95,
                ),
                SafetyMode.MINIMAL_RISK,
            )
        if avail.classic:
            # Empty set or only source drops: request classic fallback, degrade.
            to_state = SafetyMode.DEGRADED if mode is SafetyMode.NORMAL else mode
            if mode in {SafetyMode.MINIMAL_RISK, SafetyMode.EMERGENCY}:
                to_state = mode
            return (
                DecisionKind.CLASSIC_FALLBACK,
                FallbackRequest(
                    reason_code="no_legal_candidate",
                    target=FallbackTarget.CLASSIC,
                    from_state=mode,
                    to_state=to_state,
                    urgency=0.8,
                ),
                to_state,
            )
        return (
            DecisionKind.EMERGENCY,
            FallbackRequest(
                reason_code="unresolved",
                target=FallbackTarget.EMERGENCY,
                from_state=mode,
                to_state=SafetyMode.EMERGENCY,
                urgency=1.0,
            ),
            SafetyMode.EMERGENCY,
        )

    def emit_decision_event(
        self,
        decision: SafetyDecision,
        availability: ComponentAvailability,
        obs: ObservableSnapshot,
        *,
        phase: EventPhase,
    ) -> list[SafetyEvent]:
        """Public event emission for kernel / arbitration (no private API needed)."""
        return self._maybe_event(decision, availability, obs, phase=phase)

    def record_candidate_latency_ms(self, latency_ms: float) -> None:
        """Record candidate-path latency and deadline misses from external pipelines."""
        self._latencies_candidate_ms.append(float(latency_ms))
        if latency_ms > self.config.candidate_check_deadline_ms:
            self._deadline_misses += 1

    def _maybe_event(
        self,
        decision: SafetyDecision,
        availability: ComponentAvailability,
        obs: ObservableSnapshot,
        *,
        phase: EventPhase,
    ) -> list[SafetyEvent]:
        if not self.config.emit_safety_events:
            return []
        # Never silent on hard reject / fallback / emergency.
        if decision.decision_kind is DecisionKind.ACCEPT and phase is EventPhase.NORMAL:
            # Still log optional heartbeat only when reject reasons present.
            if not decision.reject_reasons:
                return []
        message = (
            f"kind={decision.decision_kind.value} "
            f"rejects={list(decision.reject_reasons)} "
            f"fallback={decision.fallback_request.reason_code if decision.fallback_request else None}"
        )
        event = SafetyEvent(
            event_id=_decision_id(),
            run_id=decision.run_id,
            frame_id=decision.frame_id,
            phase=phase,
            decision=decision,
            availability=availability,
            privilege=ObservationPrivilege.OBSERVABLE,
            message=message,
            simulation_time_s=obs.simulation_time_s,
            wall_time_s=obs.wall_time_s,
        )
        self._events.append(event)
        return [event]

    def _record_failure(self, candidate_id: str, reason: str, margins: list[ConstraintMargin]) -> None:
        if not self.config.record_failure_samples:
            return
        if len(self._failure_samples) >= self.config.max_failure_samples:
            return
        self._failure_samples.append(
            {
                "candidate_id": candidate_id,
                "reason": reason,
                "hard_margins": [
                    {"name": m.name, "margin": m.margin, "message": m.message}
                    for m in margins
                    if m.hard and m.margin < 0.0
                ],
            }
        )
