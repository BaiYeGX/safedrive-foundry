"""SafetyKernel runtime facade: state + arbitration + validator + QP/RATO + mode machine.

Learning modules are never required. Pipeline (G2-04):
  hard precheck → soft score → arbitration → final Validator → QP/RATO → fallback
Classic Shadow is compare-only (no control / no tick ownership).

State hard violations (stale obs, privilege, non-finite ego, overspeed) lock the
tick: no candidate validation, soft score, or QP/RATO on untrusted observations.
Collision checks use constant-velocity actors + circular envelope (documented limit).
"""

from __future__ import annotations

import copy

from dataclasses import dataclass, field
from typing import Any

from safety_kernel.arbitration import ArbitrationPipeline, ArbitrationRecord
from safety_kernel.arbitration.types import PipelineStage
from safety_kernel.config import SafetyKernelConfig, config_sha256, load_safety_config
from safety_kernel.contracts.schema import SCHEMA_VERSION, contracts_schema_hash
from safety_kernel.contracts.types import (
    ComponentAvailability,
    DecisionKind,
    EventPhase,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyEvent,
    SafetyMode,
    ObservableSnapshot,
)
from safety_kernel.metrics import build_latency_report
from safety_kernel.repair import RepairInterface, RepairMode, RepairResult
from safety_kernel.repair.corridor import has_legal_lateral_corridor, is_rato_eligible_hints
from safety_kernel.repair.longitudinal_qp import is_longitudinally_repairable
from safety_kernel.repair.types import SolverStatus
from safety_kernel.state_machine import SafetyStateMachine, StateTransition
from safety_kernel.validator import TrajectoryValidator, ValidatorResult
from safety_kernel.validator.engine import ValidationStage, _decision_id
from safety_kernel.validator.checks import hard_violations, run_full_checks


@dataclass
class KernelTickResult:
    mode: SafetyMode
    state_result: ValidatorResult
    candidate_result: ValidatorResult | None
    transition: StateTransition | None
    decision: SafetyDecision
    events: list[SafetyEvent] = field(default_factory=list)
    repair_result: RepairResult | None = None
    arbitration: ArbitrationRecord | None = None

    @property
    def accepted_trajectory_id(self) -> str | None:
        return self.decision.executed_trajectory_id


class SafetyKernel:
    """Independent safety boundary usable with Classic-only stacks."""

    def __init__(self, config: SafetyKernelConfig | None = None) -> None:
        self.config = config or load_safety_config()
        self.validator = TrajectoryValidator(self.config)
        self.state_machine = SafetyStateMachine(self.config)
        self.repair = RepairInterface(self.config)
        self.arbitration = ArbitrationPipeline(self.config, self.validator)
        self._ticks = 0
        self._repair_traces: list[dict[str, Any]] = []

    @property
    def mode(self) -> SafetyMode:
        return self.state_machine.mode

    @property
    def repair_traces(self) -> tuple[dict[str, Any], ...]:
        """Read-only snapshot of actual solver attempts, including failures."""
        return tuple(copy.deepcopy(self._repair_traces))

    def reset(self, mode: SafetyMode = SafetyMode.NORMAL, *, now_s: float = 0.0) -> None:
        self.state_machine.reset(mode, now_s=now_s)
        self._ticks = 0
        self._repair_traces.clear()
        self.repair.clear_warm_starts()
        self.arbitration.clear()

    def tick_state(
        self,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        availability: ComponentAvailability | None = None,
    ) -> KernelTickResult:
        """50Hz-class state monitoring tick (no candidate set)."""
        avail = availability or ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        now = obs.simulation_time_s if now_s is None else now_s
        state_res = self.validator.check_state(obs, now_s=now, availability=avail, mode=self.mode)
        driving = state_res.decision
        transition = self.state_machine.step(driving, avail, now_s=now, frame_id=obs.frame_id)
        self._ticks += 1
        return KernelTickResult(
            mode=self.mode,
            state_result=state_res,
            candidate_result=None,
            transition=transition,
            decision=state_res.decision,
            events=list(state_res.events),
        )

    def tick(
        self,
        obs: ObservableSnapshot,
        candidate_set: PolicyCandidateSet | None,
        *,
        now_s: float | None = None,
        availability: ComponentAvailability | None = None,
        stage: ValidationStage = ValidationStage.FINAL,
        repair_mode: RepairMode = RepairMode.LONGITUDINAL,
    ) -> KernelTickResult:
        """Full safety tick: state → optional arbitration pipeline → mode update.

        When state monitoring reports hard violations (elevated to MINIMAL_RISK or
        EMERGENCY), the tick is state-locked: no candidate validation, soft score,
        or repair is run on the untrusted observation.
        """
        avail = availability or ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        now = obs.simulation_time_s if now_s is None else now_s
        events: list[SafetyEvent] = []
        repair_result: RepairResult | None = None
        arb_record: ArbitrationRecord | None = None

        state_res = self.validator.check_state(obs, now_s=now, availability=avail, mode=self.mode)
        events.extend(state_res.events)
        state_hard = hard_violations(state_res.state_margins or list(state_res.decision.constraint_margins))
        state_floor = state_res.decision.state_after
        state_locked = bool(state_hard) and state_floor in {
            SafetyMode.EMERGENCY,
            SafetyMode.MINIMAL_RISK,
        }

        cand_res: ValidatorResult | None = None
        if state_locked:
            # Observation integrity failure: never ACCEPT/QP/RATO on untrusted obs.
            driving = self._state_locked_decision(state_res.decision)
            lock_events = self.validator.emit_decision_event(
                driving, avail, obs, phase=EventPhase.INTERVENTION
            )
            events.extend(lock_events)
            arb_record = self._state_lock_audit(
                decision=driving,
                mode_before=self.mode,
            )
        elif candidate_set is not None:
            use_pipeline = (
                self.config.arbitration.enabled
                and stage is ValidationStage.FINAL
            )
            if use_pipeline:

                def _repair_fn(**kwargs):
                    return self._try_repair_cascade(
                        preferred_mode=repair_mode,
                        **kwargs,
                    )

                pipe = self.arbitration.run_candidate_pipeline(
                    obs=obs,
                    candidate_set=candidate_set,
                    availability=avail,
                    mode=self.mode,
                    now_s=now,
                    repair_fn=_repair_fn,
                    state_emergency=False,
                )
                cand_res = pipe.candidate_result
                driving = pipe.decision
                repair_result = pipe.repair_result
                arb_record = pipe.arbitration
                events.extend(pipe.events)
            else:
                cand_res = self.validator.validate_candidates(
                    candidate_set,
                    obs,
                    now_s=now,
                    availability=avail,
                    mode=self.mode,
                    stage=stage,
                )
                events.extend(cand_res.events)
                driving = cand_res.decision

                if (
                    stage is ValidationStage.FINAL
                    and cand_res.accepted is None
                    and driving.decision_kind
                    in {
                        DecisionKind.HARD_REJECT,
                        DecisionKind.MINIMAL_RISK,
                        DecisionKind.CLASSIC_FALLBACK,
                    }
                ):
                    repaired_decision, repair_result, repair_events = self._try_repair_cascade(
                        candidate_set=candidate_set,
                        obs=obs,
                        now_s=now,
                        availability=avail,
                        mode=self.mode,
                        reject_reasons=list(driving.reject_reasons),
                        prefilter_ids=driving.prefilter_candidate_ids,
                        all_margins=list(driving.constraint_margins),
                        base_latency_ms=driving.latency_ms,
                        preferred_mode=repair_mode,
                    )
                    if repaired_decision is not None:
                        driving = repaired_decision
                        events.extend(repair_events)
                        cand_res.accepted = repaired_decision.accepted_candidate
                        cand_res.decision = repaired_decision
        else:
            driving = state_res.decision

        transition = self.state_machine.step(
            driving,
            avail,
            now_s=now,
            frame_id=obs.frame_id,
            state_floor=state_floor if state_hard else None,
        )
        self._ticks += 1
        return KernelTickResult(
            mode=self.mode,
            state_result=state_res,
            candidate_result=cand_res,
            transition=transition,
            decision=driving,
            events=events,
            repair_result=repair_result,
            arbitration=arb_record,
        )

    @staticmethod
    def _state_locked_decision(state_decision: SafetyDecision) -> SafetyDecision:
        """Clone state decision with explicit lock reason; no executable trajectory."""
        extra = "state_locked_no_candidate_override"
        reasons = tuple(state_decision.reject_reasons)
        if extra not in reasons:
            reasons = reasons + (extra,)
        return SafetyDecision(
            decision_id=state_decision.decision_id,
            run_id=state_decision.run_id,
            frame_id=state_decision.frame_id,
            prefilter_candidate_ids=(),
            final_candidate_id=None,
            pre_repair_trajectory_id=None,
            post_repair_trajectory_id=None,
            executed_trajectory_id=None,
            constraint_margins=state_decision.constraint_margins,
            decision_kind=state_decision.decision_kind,
            modification_norm=0.0,
            slack=state_decision.slack,
            progress_loss=1.0,
            solver_status="n/a",
            latency_ms=state_decision.latency_ms,
            state_before=state_decision.state_before,
            state_after=state_decision.state_after,
            recovery_conditions=state_decision.recovery_conditions,
            fallback_request=state_decision.fallback_request,
            reject_reasons=reasons,
            learning_modules_required=False,
            accepted_candidate=None,
        )

    def _state_lock_audit(
        self,
        *,
        decision: SafetyDecision,
        mode_before: SafetyMode,
    ) -> ArbitrationRecord | None:
        if not self.config.arbitration.enabled:
            return None
        rec = ArbitrationRecord(
            run_id=decision.run_id,
            frame_id=decision.frame_id,
            stages=(
                PipelineStage.STATE.value,
                "state_locked",
                PipelineStage.DONE.value,
            ),
            ranked_ids=(),
            selected_id=None,
            decision_kind=decision.decision_kind.value,
            mode_before=mode_before.value,
            mode_after=decision.state_after.value,
            latency_ms=decision.latency_ms,
            arbitration_latency_ms=0.0,
            fallback=decision.fallback_request,
            shadow=None,
            audits=(),
            learning_required=False,
            emergency_locked=decision.decision_kind is DecisionKind.EMERGENCY,
            notes=("state_locked_no_candidate_override",),
        )
        self.arbitration.record_audit(rec)
        return rec

    def _select_repair_seed(
        self,
        candidate_set: PolicyCandidateSet,
        reject_reasons: list[str],
        *,
        for_rato: bool = False,
    ) -> tuple[PolicyCandidate | None, list[str]]:
        """Pick preferred classic-first candidate that is repairable."""
        by_id: dict[str, list[str]] = {}
        for reason in reject_reasons:
            cid = reason.split(":", 1)[0] if ":" in reason else ""
            if cid:
                by_id.setdefault(cid, []).append(reason)

        ordered = sorted(
            candidate_set.candidates,
            key=lambda c: (
                0 if c.source.value == "classic" else 1,
                -c.probability,
                c.candidate_id,
            ),
        )
        for cand in ordered:
            hints = by_id.get(cand.candidate_id, list(reject_reasons))
            if for_rato:
                if is_rato_eligible_hints(hints) or not hints:
                    return cand, hints
            else:
                if is_longitudinally_repairable(hints) or not hints:
                    if hints and not is_longitudinally_repairable(hints):
                        continue
                    return cand, hints
        for cand in ordered:
            joined = " ".join(reject_reasons).lower()
            tokens = (
                (
                    "collision",
                    "road",
                    "offroad",
                    "cut_in",
                    "static",
                    "lateral",
                    "lat_accel",
                    "curvature",
                    "trackability",
                    "teleport",
                    "yaw_rate",
                )
                if for_rato
                else ("collision", "rules", "dynamics", "red_light", "speed")
            )
            if any(tok in joined for tok in tokens):
                return cand, reject_reasons
        return None, reject_reasons

    def _should_try_rato(
        self,
        *,
        obs: ObservableSnapshot,
        reject_reasons: list[str],
        qp_result: RepairResult | None,
    ) -> bool:
        rato = self.config.rato
        if not rato.enabled or not rato.repair_on_hard_reject:
            return False
        if not has_legal_lateral_corridor(
            obs, self.config, min_clearance_m=rato.min_lateral_clearance_m
        ):
            return False
        # Direct eligibility on reject reasons (lateral tokens) or QP failure / low progress.
        if is_rato_eligible_hints(reject_reasons):
            if qp_result is None:
                return True
            if not qp_result.success:
                return True
            if qp_result.metrics.progress_ratio + 1e-9 < rato.min_qp_progress_to_skip:
                return True
            return False
        # Longitudinal-only rejects: only escalate if QP failed or progress collapsed.
        if qp_result is None:
            return False
        if not qp_result.success:
            return True
        return qp_result.metrics.progress_ratio + 1e-9 < rato.min_qp_progress_to_skip

    def _try_repair_cascade(
        self,
        *,
        candidate_set: PolicyCandidateSet,
        obs: ObservableSnapshot,
        now_s: float,
        availability: ComponentAvailability,
        mode: SafetyMode,
        reject_reasons: list[str],
        prefilter_ids: tuple[str, ...],
        all_margins: list,
        base_latency_ms: float,
        preferred_mode: RepairMode,
    ) -> tuple[SafetyDecision | None, RepairResult | None, list[SafetyEvent]]:
        """QP first (default), then optional restricted RATO under frozen triggers."""
        qp_result: RepairResult | None = None
        latency_acc = base_latency_ms

        # Stage-1 longitudinal QP unless caller explicitly requests RATO-only.
        if preferred_mode is RepairMode.RATO:
            pass
        elif self.config.qp.enabled and self.config.qp.repair_on_hard_reject:
            qp_decision, qp_result, qp_events = self._try_repair(
                candidate_set=candidate_set,
                obs=obs,
                now_s=now_s,
                availability=availability,
                mode=mode,
                reject_reasons=reject_reasons,
                prefilter_ids=prefilter_ids,
                all_margins=all_margins,
                base_latency_ms=latency_acc,
                repair_mode=RepairMode.LONGITUDINAL,
            )
            if qp_result is not None:
                latency_acc = (
                    base_latency_ms + qp_result.solver_trace.latency_ms
                    if qp_decision is None
                    else (qp_decision.latency_ms)
                )
            # Accept QP when successful and progress is adequate (skip RATO).
            if qp_decision is not None and qp_result is not None:
                if qp_result.metrics.progress_ratio + 1e-9 >= self.config.rato.min_qp_progress_to_skip:
                    return qp_decision, qp_result, qp_events
                # Low progress: keep QP as fallback if RATO fails.
                kept_qp = (qp_decision, qp_result, qp_events)
            else:
                kept_qp = None
        else:
            kept_qp = None

        # Stage-2 restricted RATO-SCP.
        try_rato = preferred_mode is RepairMode.RATO or self._should_try_rato(
            obs=obs,
            reject_reasons=reject_reasons,
            qp_result=qp_result,
        )
        if try_rato:
            rato_decision, rato_result, rato_events = self._try_repair(
                candidate_set=candidate_set,
                obs=obs,
                now_s=now_s,
                availability=availability,
                mode=mode,
                reject_reasons=reject_reasons,
                prefilter_ids=prefilter_ids,
                all_margins=all_margins,
                base_latency_ms=latency_acc,
                repair_mode=RepairMode.RATO,
                for_rato=True,
            )
            if rato_decision is not None and rato_result is not None:
                # Prefer RATO when it improves progress over a low-progress QP, or when QP failed.
                if kept_qp is None:
                    return rato_decision, rato_result, rato_events
                qp_dec, qp_res, qp_ev = kept_qp
                if rato_result.metrics.progress_ratio + 1e-3 >= qp_res.metrics.progress_ratio:
                    return rato_decision, rato_result, rato_events
                return qp_dec, qp_res, qp_ev
            if kept_qp is not None:
                return kept_qp
            return None, rato_result, []

        if kept_qp is not None:
            return kept_qp
        return None, qp_result, []

    def _try_repair(
        self,
        *,
        candidate_set: PolicyCandidateSet,
        obs: ObservableSnapshot,
        now_s: float,
        availability: ComponentAvailability,
        mode: SafetyMode,
        reject_reasons: list[str],
        prefilter_ids: tuple[str, ...],
        all_margins: list,
        base_latency_ms: float,
        repair_mode: RepairMode,
        for_rato: bool = False,
    ) -> tuple[SafetyDecision | None, RepairResult | None, list[SafetyEvent]]:
        seed, hints = self._select_repair_seed(candidate_set, reject_reasons, for_rato=for_rato)
        if seed is None:
            return None, None, []

        result = self.repair.repair(
            seed,
            obs,
            mode=repair_mode,
            now_s=now_s,
            reject_hints=hints,
        )
        self._repair_traces.append(result.to_dict())

        if not result.success or result.candidate is None:
            return None, result, []

        # Re-validate repaired trajectory with full checks.
        margins = run_full_checks(result.candidate, obs, self.config, now_s=now_s)
        viol = hard_violations(margins)
        total_latency = base_latency_ms + result.solver_trace.latency_ms
        if viol:
            fail = RepairResult(
                mode=result.mode,
                success=False,
                candidate=None,
                pre_repair_id=result.pre_repair_id,
                post_repair_id=result.post_repair_id,
                solver_trace=result.solver_trace,
                metrics=result.metrics,
                constraint_margins=tuple(margins),
                reason="revalidate_fail:" + ",".join(f"{v.name}:{v.message}" for v in viol),
            )
            self._repair_traces[-1] = fail.to_dict()
            return None, fail, []

        if repair_mode is RepairMode.LONGITUDINAL:
            kind = DecisionKind.QP
        elif repair_mode is RepairMode.RATO:
            kind = DecisionKind.RATO
        else:
            kind = DecisionKind.ACCEPT
        decision = SafetyDecision(
            decision_id=_decision_id(),
            run_id=candidate_set.run_id,
            frame_id=candidate_set.frame_id,
            prefilter_candidate_ids=prefilter_ids,
            final_candidate_id=result.candidate.candidate_id,
            pre_repair_trajectory_id=result.pre_repair_id,
            post_repair_trajectory_id=result.post_repair_id,
            executed_trajectory_id=result.candidate.candidate_id,
            constraint_margins=tuple(list(all_margins) + list(margins)),
            decision_kind=kind,
            modification_norm=result.metrics.modification_norm,
            slack=result.metrics.slack_used_max,
            progress_loss=result.metrics.progress_loss,
            solver_status=result.solver_trace.status.value,
            latency_ms=total_latency,
            state_before=mode,
            state_after=mode if mode is not SafetyMode.EMERGENCY else SafetyMode.EMERGENCY,
            recovery_conditions=(),
            fallback_request=None,
            reject_reasons=tuple(reject_reasons) + (f"repaired_via_{repair_mode.value}",),
            learning_modules_required=False,
            accepted_candidate=result.candidate,
        )
        events = self.validator.emit_decision_event(
            decision,
            availability,
            obs,
            phase=EventPhase.INTERVENTION,
        )
        return decision, result, events

    def metrics_snapshot(self) -> dict[str, Any]:
        state_report = build_latency_report(
            self.validator.latency_state_ms,
            deadline_ms=self.config.state_check_deadline_ms,
        )
        cand_report = build_latency_report(
            self.validator.latency_candidate_ms,
            deadline_ms=self.config.candidate_check_deadline_ms,
        )
        qp_latencies = [
            float(t.get("solver_trace", {}).get("latency_ms", 0.0))
            for t in self._repair_traces
            if t.get("mode") == RepairMode.LONGITUDINAL.value
        ]
        rato_latencies = [
            float(t.get("solver_trace", {}).get("latency_ms", 0.0))
            for t in self._repair_traces
            if t.get("mode") == RepairMode.RATO.value
        ]
        # Fallback: count all repair traces if mode missing (legacy).
        if not qp_latencies and not rato_latencies:
            all_lat = [
                float(t.get("solver_trace", {}).get("latency_ms", 0.0)) for t in self._repair_traces
            ]
            qp_latencies = all_lat
        qp_report = build_latency_report(qp_latencies, deadline_ms=self.config.qp.deadline_ms)
        rato_report = build_latency_report(rato_latencies, deadline_ms=self.config.rato.deadline_ms)
        arb_report = build_latency_report(
            self.arbitration.latencies_ms, deadline_ms=self.config.arbitration.deadline_ms
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "contracts_schema_hash": contracts_schema_hash(),
            "config_hash": config_sha256(self.config.raw_toml),
            "config_name": self.config.name,
            "ticks": self._ticks,
            "mode": self.mode.value,
            "state_latency": state_report.to_dict(),
            "candidate_latency": cand_report.to_dict(),
            "qp_latency": qp_report.to_dict(),
            "rato_latency": rato_report.to_dict(),
            "arbitration_latency": arb_report.to_dict(),
            "qp_repair_count": sum(1 for t in self._repair_traces if t.get("mode") == "longitudinal"),
            "qp_success_count": sum(
                1 for t in self._repair_traces if t.get("mode") == "longitudinal" and t.get("success")
            ),
            "rato_repair_count": sum(1 for t in self._repair_traces if t.get("mode") == "rato"),
            "rato_success_count": sum(
                1 for t in self._repair_traces if t.get("mode") == "rato" and t.get("success")
            ),
            "arbitration_tick_count": len(self.arbitration.records),
            "deadline_miss_total": self.validator.deadline_misses,
            "failure_sample_count": len(self.validator.failure_samples),
            "event_count": len(self.validator.events),
            "transitions": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "reason": t.reason,
                    "t": t.simulation_time_s,
                    "frame_id": t.frame_id,
                }
                for t in self.state_machine.history
            ],
        }
