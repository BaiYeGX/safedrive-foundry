"""Deterministic arbitration pipeline (G2-04).

Order (frozen):
  hard precheck → soft score → rank → final Validator (all ranked) → QP/RATO → fallback → Shadow

Soft scores never override hard rejects, Emergency, or frozen fallback authority.
Caller must state-lock elevated observation faults before invoking this pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from safety_kernel.arbitration.degradation import degrade_candidate_set
from safety_kernel.arbitration.shadow import run_classic_shadow
from safety_kernel.arbitration.soft_score import rank_candidates, score_candidate
from safety_kernel.arbitration.types import (
    ArbitrationRecord,
    CandidateAudit,
    DegradationReason,
    PipelineStage,
    SoftScore,
    source_version_hint,
)
from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    CandidateSource,
    ComponentAvailability,
    DecisionKind,
    EventPhase,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyEvent,
    SafetyMode,
)
from safety_kernel.repair.types import RepairMode, RepairResult
from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.validator.checks import (
    check_set_identity_and_contract,
    hard_violations,
    run_full_checks,
    run_prefilter_checks,
)
from safety_kernel.validator.engine import TrajectoryValidator, ValidationStage, ValidatorResult


@dataclass
class PipelineTickResult:
    decision: SafetyDecision
    events: list[SafetyEvent] = field(default_factory=list)
    repair_result: RepairResult | None = None
    arbitration: ArbitrationRecord | None = None
    candidate_result: ValidatorResult | None = None
    accepted: PolicyCandidate | None = None


class ArbitrationPipeline:
    """World-ready arbitration wrapper used by SafetyKernel when enabled."""

    def __init__(self, config: SafetyKernelConfig, validator: TrajectoryValidator) -> None:
        self.config = config
        self.validator = validator
        self._records: list[dict[str, Any]] = []
        self._latencies_ms: list[float] = []

    def clear(self) -> None:
        self._records.clear()
        self._latencies_ms.clear()

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @property
    def latencies_ms(self) -> list[float]:
        return list(self._latencies_ms)

    def record_audit(self, rec: ArbitrationRecord) -> None:
        """Append a pre-built arbitration audit (e.g. state-lock short-circuit)."""
        self._records.append(rec.to_dict())

    def run_candidate_pipeline(
        self,
        *,
        obs: ObservableSnapshot,
        candidate_set: PolicyCandidateSet,
        availability: ComponentAvailability,
        mode: SafetyMode,
        now_s: float,
        repair_fn,
        state_emergency: bool,
    ) -> PipelineTickResult:
        """Execute arbitration stages. repair_fn(cset, obs, ...) → (decision|None, RepairResult|None, events)."""
        t0 = time.perf_counter()
        stages: list[str] = [PipelineStage.STATE.value]
        notes: list[str] = []
        audits_map: dict[str, CandidateAudit] = {}
        arb_cfg = self.config.arbitration

        # Hard identity / contract consistency (same gate as TrajectoryValidator).
        identity_margins = check_set_identity_and_contract(
            candidate_set,
            obs,
            expected_schema_version=SCHEMA_VERSION,
        )
        identity_viol = hard_violations(identity_margins)
        if identity_viol:
            from safety_kernel.contracts.types import FallbackRequest, FallbackTarget
            from safety_kernel.validator.engine import _decision_id

            stages.extend([PipelineStage.PREFILTER.value, PipelineStage.FALLBACK.value, PipelineStage.DONE.value])
            reasons = tuple(f"identity:{v.message}" for v in identity_viol)
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
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                state_before=mode,
                state_after=SafetyMode.MINIMAL_RISK,
                recovery_conditions=("identity_aligned", "schema_match"),
                fallback_request=FallbackRequest(
                    reason_code="identity_contract_mismatch",
                    target=FallbackTarget.MINIMAL_RISK,
                    from_state=mode,
                    to_state=SafetyMode.MINIMAL_RISK,
                    urgency=1.0,
                ),
                reject_reasons=reasons,
                learning_modules_required=False,
            )
            events = self.validator.emit_decision_event(
                decision, availability, obs, phase=EventPhase.INTERVENTION
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._latencies_ms.append(latency_ms)
            self.validator.record_candidate_latency_ms(latency_ms)
            rec = self._record(
                candidate_set=candidate_set,
                stages=stages,
                ranked_ids=(),
                selected_id=None,
                decision=decision,
                mode_before=mode,
                latency_ms=latency_ms,
                arb_latency_ms=latency_ms,
                shadow=None,
                audits=(),
                notes=("identity_contract_mismatch",),
            )
            return PipelineTickResult(
                decision=decision,
                events=events,
                arbitration=rec,
                candidate_result=ValidatorResult(
                    decision=decision,
                    events=list(events),
                    accepted=None,
                    prefilter_passed=[],
                    stage=ValidationStage.FINAL,
                ),
            )

        if state_emergency or mode is SafetyMode.EMERGENCY:
            # Emergency is never overridden by scores/language/hot config.
            stages.append(PipelineStage.DONE.value)
            # Use validator path for empty accept to produce structured reject.
            empty_res = self.validator.validate_candidates(
                candidate_set,
                obs,
                now_s=now_s,
                availability=availability,
                mode=SafetyMode.EMERGENCY,
                stage=ValidationStage.FINAL,
            )
            # Force emergency kind if not already.
            dec = empty_res.decision
            if dec.decision_kind is not DecisionKind.EMERGENCY:
                from safety_kernel.contracts.types import FallbackRequest, FallbackTarget
                from safety_kernel.validator.engine import _decision_id

                dec = SafetyDecision(
                    decision_id=_decision_id(),
                    run_id=candidate_set.run_id,
                    frame_id=candidate_set.frame_id,
                    prefilter_candidate_ids=tuple(c.candidate_id for c in candidate_set.candidates),
                    final_candidate_id=None,
                    pre_repair_trajectory_id=None,
                    post_repair_trajectory_id=None,
                    executed_trajectory_id=None,
                    constraint_margins=dec.constraint_margins,
                    decision_kind=DecisionKind.EMERGENCY,
                    modification_norm=0.0,
                    slack=-1.0,
                    progress_loss=1.0,
                    solver_status="n/a",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    state_before=mode,
                    state_after=SafetyMode.EMERGENCY,
                    recovery_conditions=("emergency_clear",),
                    fallback_request=FallbackRequest(
                        reason_code="emergency_locked",
                        target=FallbackTarget.EMERGENCY,
                        from_state=mode,
                        to_state=SafetyMode.EMERGENCY,
                        urgency=1.0,
                    ),
                    reject_reasons=tuple(dec.reject_reasons) + ("emergency_locked_no_score_override",),
                    learning_modules_required=False,
                )
            rec = self._record(
                candidate_set=candidate_set,
                stages=stages,
                ranked_ids=(),
                selected_id=None,
                decision=dec,
                mode_before=mode,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                arb_latency_ms=0.0,
                shadow=None,
                audits=(),
                notes=("emergency_locked",),
            )
            return PipelineTickResult(
                decision=dec,
                events=list(empty_res.events),
                arbitration=rec,
                candidate_result=empty_res,
            )

        # --- DEGRADE ---
        stages.append(PipelineStage.DEGRADE.value)
        # Learning + classic both gone → Emergency is locked; scores cannot rescue.
        if not availability.classic and availability.learning_all_failed:
            from safety_kernel.contracts.types import FallbackRequest, FallbackTarget
            from safety_kernel.validator.engine import _decision_id

            stages.extend(
                [
                    PipelineStage.PREFILTER.value,
                    PipelineStage.FALLBACK.value,
                    PipelineStage.DONE.value,
                ]
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
                constraint_margins=(),
                decision_kind=DecisionKind.EMERGENCY,
                modification_norm=0.0,
                slack=-1.0,
                progress_loss=1.0,
                solver_status="n/a",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                state_before=mode,
                state_after=SafetyMode.EMERGENCY,
                recovery_conditions=("classic_available",),
                fallback_request=FallbackRequest(
                    reason_code="no_classic_no_learning",
                    target=FallbackTarget.EMERGENCY,
                    from_state=mode,
                    to_state=SafetyMode.EMERGENCY,
                    urgency=1.0,
                ),
                reject_reasons=tuple(
                    f"{c.candidate_id}:source_unavailable" for c in candidate_set.candidates
                )
                + ("no_classic_no_learning", "emergency_locked_no_score_override"),
                learning_modules_required=False,
            )
            events = self.validator.emit_decision_event(
                decision, availability, obs, phase=EventPhase.INTERVENTION
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._latencies_ms.append(latency_ms)
            self.validator.record_candidate_latency_ms(latency_ms)
            rec = self._record(
                candidate_set=candidate_set,
                stages=stages,
                ranked_ids=(),
                selected_id=None,
                decision=decision,
                mode_before=mode,
                latency_ms=latency_ms,
                arb_latency_ms=latency_ms,
                shadow=None,
                audits=(),
                notes=("emergency_no_classic_no_learning",),
            )
            cand_res = ValidatorResult(
                decision=decision,
                events=list(events),
                accepted=None,
                prefilter_passed=[],
                stage=ValidationStage.FINAL,
            )
            return PipelineTickResult(
                decision=decision,
                events=events,
                arbitration=rec,
                candidate_result=cand_res,
            )

        deg_set, deg_reasons = degrade_candidate_set(
            candidate_set, availability, now_s=now_s, cfg=self.config
        )
        for cand in candidate_set.candidates:
            deg = deg_reasons.get(cand.candidate_id, DegradationReason.NONE)
            if deg is DegradationReason.NONE:
                rej: tuple[str, ...] = ()
            elif deg is DegradationReason.SOURCE_UNAVAILABLE and cand.source in {
                CandidateSource.VLA_FAST,
                CandidateSource.VLA_SLOW,
            }:
                # Compatibility token expected by G2-01/G2-04 audit consumers.
                rej = (f"{cand.candidate_id}:vla_unavailable", f"degrade:{deg.value}")
            else:
                rej = (f"degrade:{deg.value}",)
            audits_map[cand.candidate_id] = CandidateAudit(
                candidate_id=cand.candidate_id,
                source=cand.source.value,
                version_hint=source_version_hint(cand),
                prefilter_ok=False,
                final_ok=None,
                soft_score=None,
                reject_reasons=rej,
                degradation=deg,
                repaired=False,
                selected=False,
            )

        # --- PREFILTER ---
        stages.append(PipelineStage.PREFILTER.value)
        prefilter_passed: list[PolicyCandidate] = []
        prefilter_margins: dict[str, list] = {}
        reject_pool: list[str] = []
        for cand in deg_set.candidates:
            if not cand.availability:
                deg = cand.dynamics_meta.get("degradation", "unavailable")
                if deg == DegradationReason.SOURCE_UNAVAILABLE.value and cand.source in {
                    CandidateSource.VLA_FAST,
                    CandidateSource.VLA_SLOW,
                }:
                    reject_pool.append(f"{cand.candidate_id}:vla_unavailable")
                reject_pool.append(f"{cand.candidate_id}:degrade:{deg}")
                continue
            margins = run_prefilter_checks(cand, obs, self.config, now_s=now_s)
            prefilter_margins[cand.candidate_id] = margins
            viol = hard_violations(margins)
            if viol:
                reason = f"{cand.candidate_id}:" + ",".join(f"{v.name}:{v.message}" for v in viol)
                reject_pool.append(reason)
                a = audits_map[cand.candidate_id]
                audits_map[cand.candidate_id] = CandidateAudit(
                    candidate_id=a.candidate_id,
                    source=a.source,
                    version_hint=a.version_hint,
                    prefilter_ok=False,
                    final_ok=None,
                    soft_score=None,
                    reject_reasons=tuple(a.reject_reasons) + (reason,),
                    degradation=a.degradation,
                    repaired=False,
                    selected=False,
                )
                continue
            prefilter_passed.append(cand)
            a = audits_map[cand.candidate_id]
            audits_map[cand.candidate_id] = CandidateAudit(
                candidate_id=a.candidate_id,
                source=a.source,
                version_hint=a.version_hint,
                prefilter_ok=True,
                final_ok=None,
                soft_score=None,
                reject_reasons=a.reject_reasons,
                degradation=a.degradation,
                repaired=False,
                selected=False,
            )

        # --- SOFT SCORE + RANK ---
        stages.append(PipelineStage.SOFT_SCORE.value)
        scores: list[SoftScore] = []
        for cand in prefilter_passed:
            # Soft score uses light margins from prefilter only (no full cost before rank).
            sc = score_candidate(cand, obs, self.config, now_s=now_s, margins=prefilter_margins.get(cand.candidate_id))
            scores.append(sc)
            a = audits_map[cand.candidate_id]
            audits_map[cand.candidate_id] = CandidateAudit(
                candidate_id=a.candidate_id,
                source=a.source,
                version_hint=a.version_hint,
                prefilter_ok=True,
                final_ok=None,
                soft_score=sc,
                reject_reasons=a.reject_reasons,
                degradation=a.degradation,
                repaired=False,
                selected=False,
            )
        stages.append(PipelineStage.ARBITRATE.value)
        # A World router may provide an explicit preference order.  Safety
        # validates in that order, while retaining its own soft scores only for
        # audit.  Missing/duplicate ids are ignored and every remaining
        # candidate is appended deterministically.
        preferred_ids = tuple(dict.fromkeys(candidate_set.preference_order))
        if preferred_ids:
            preference_rank = {
                candidate_id: index for index, candidate_id in enumerate(preferred_ids)
            }
            ranked = sorted(
                prefilter_passed,
                key=lambda candidate: (
                    preference_rank.get(candidate.candidate_id, len(preference_rank)),
                    candidate.candidate_id,
                ),
            )
            notes.append("external_preference_order")
        else:
            ranked = rank_candidates(prefilter_passed, scores)
        ranked_ids = tuple(c.candidate_id for c in ranked)
        # max_final_candidates is a primary preference window for audit only;
        # hard-legal candidates beyond top-K are still checked before repair.
        primary_k = max(1, arb_cfg.max_final_candidates)

        # --- FINAL VALIDATOR (winner-first over full ranked list) ---
        stages.append(PipelineStage.FINAL.value)
        chosen: PolicyCandidate | None = None
        all_margins: list = []
        chosen_rank_idx = -1
        repair_result: RepairResult | None = None
        preferred_repair_decision: SafetyDecision | None = None
        events: list[SafetyEvent] = []
        for rank_idx, cand in enumerate(ranked):
            margins = run_full_checks(cand, obs, self.config, now_s=now_s)
            if chosen is not None:
                # The winner is already fixed.  Validate every remaining
                # candidate for offline supervision/audit only; this must not
                # change the decision, trigger repair, or contaminate the
                # selected trajectory's margins/reject reasons.
                viol = hard_violations(margins)
                a = audits_map[cand.candidate_id]
                reason = (
                    f"{cand.candidate_id}:"
                    + ",".join(f"{v.name}:{v.message}" for v in viol)
                    if viol
                    else None
                )
                audits_map[cand.candidate_id] = CandidateAudit(
                    candidate_id=a.candidate_id,
                    source=a.source,
                    version_hint=a.version_hint,
                    prefilter_ok=True,
                    final_ok=not bool(viol),
                    soft_score=a.soft_score,
                    reject_reasons=(
                        tuple(a.reject_reasons) + (reason,)
                        if reason is not None
                        else a.reject_reasons
                    ),
                    degradation=a.degradation,
                    repaired=False,
                    selected=False,
                )
                continue
            all_margins.extend(margins)
            viol = hard_violations(margins)
            a = audits_map[cand.candidate_id]
            if viol:
                reason = f"{cand.candidate_id}:" + ",".join(f"{v.name}:{v.message}" for v in viol)
                reject_pool.append(reason)
                audits_map[cand.candidate_id] = CandidateAudit(
                    candidate_id=a.candidate_id,
                    source=a.source,
                    version_hint=a.version_hint,
                    prefilter_ok=True,
                    final_ok=False,
                    soft_score=a.soft_score,
                    reject_reasons=tuple(a.reject_reasons) + (reason,),
                    degradation=a.degradation,
                    repaired=False,
                    selected=False,
                )
                # When World explicitly asks for VLA first, give a repairable
                # VLA one bounded QP/RATO attempt before falling through to the
                # Expert from the same tick.  The repaired trajectory is still
                # fully revalidated inside repair_fn.
                if (
                    rank_idx == 0
                    and bool(preferred_ids)
                    and cand.source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW}
                ):
                    if PipelineStage.REPAIR.value not in stages:
                        stages.append(PipelineStage.REPAIR.value)
                    single_set = replace(
                        candidate_set,
                        candidates=(cand,),
                        preference_order=(cand.candidate_id,),
                    )
                    repaired_decision, attempted_repair, repair_events = repair_fn(
                        candidate_set=single_set,
                        obs=obs,
                        now_s=now_s,
                        availability=availability,
                        mode=mode,
                        reject_reasons=[reason],
                        prefilter_ids=(cand.candidate_id,),
                        all_margins=list(margins),
                        base_latency_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    events.extend(repair_events)
                    repair_result = attempted_repair
                    if (
                        repaired_decision is not None
                        and attempted_repair is not None
                        and attempted_repair.success
                    ):
                        preferred_repair_decision = repaired_decision
                        chosen = repaired_decision.accepted_candidate
                        chosen_rank_idx = rank_idx
                        audits_map[cand.candidate_id] = CandidateAudit(
                            candidate_id=a.candidate_id,
                            source=a.source,
                            version_hint=a.version_hint,
                            prefilter_ok=True,
                            final_ok=False,
                            soft_score=a.soft_score,
                            reject_reasons=tuple(a.reject_reasons) + (reason,),
                            degradation=a.degradation,
                            repaired=True,
                            selected=True,
                            repair_attempted=True,
                            repair_success=True,
                        )
                        notes.append(f"preferred_vla_repaired_via_{attempted_repair.mode.value}")
                        continue
                    audits_map[cand.candidate_id] = CandidateAudit(
                        candidate_id=a.candidate_id,
                        source=a.source,
                        version_hint=a.version_hint,
                        prefilter_ok=True,
                        final_ok=False,
                        soft_score=a.soft_score,
                        reject_reasons=tuple(a.reject_reasons) + (reason,),
                        degradation=a.degradation,
                        repaired=False,
                        selected=False,
                        repair_attempted=attempted_repair is not None,
                        repair_success=(
                            None
                            if attempted_repair is None
                            else bool(attempted_repair.success)
                        ),
                    )
                    notes.append("preferred_vla_repair_failed_try_expert")
                continue
            chosen = cand
            chosen_rank_idx = rank_idx
            audits_map[cand.candidate_id] = CandidateAudit(
                candidate_id=a.candidate_id,
                source=a.source,
                version_hint=a.version_hint,
                prefilter_ok=True,
                final_ok=True,
                soft_score=a.soft_score,
                reject_reasons=a.reject_reasons,
                degradation=a.degradation,
                repaired=False,
                selected=True,
            )
            continue
        if chosen is not None and chosen_rank_idx >= primary_k:
            notes.append("final_sweep_beyond_topk")

        decision: SafetyDecision

        if chosen is not None and preferred_repair_decision is not None:
            decision = preferred_repair_decision
        elif chosen is not None:
            from safety_kernel.validator.engine import _decision_id

            latency_ms = (time.perf_counter() - t0) * 1000.0
            used_same_tick_expert_fallback = (
                chosen.source is CandidateSource.CLASSIC
                and "preferred_vla_repair_failed_try_expert" in notes
            )
            accept_reasons = list(reject_pool)
            if (
                chosen.source is CandidateSource.CLASSIC
                and any("vla_unavailable" in r for r in reject_pool)
            ):
                accept_reasons.append("selected_classic_after_learning_drop")
            decision = SafetyDecision(
                decision_id=_decision_id(),
                run_id=candidate_set.run_id,
                frame_id=candidate_set.frame_id,
                prefilter_candidate_ids=tuple(c.candidate_id for c in prefilter_passed),
                final_candidate_id=chosen.candidate_id,
                pre_repair_trajectory_id=chosen.candidate_id,
                post_repair_trajectory_id=chosen.candidate_id,
                executed_trajectory_id=chosen.candidate_id,
                constraint_margins=tuple(all_margins),
                decision_kind=(
                    DecisionKind.CLASSIC_FALLBACK
                    if used_same_tick_expert_fallback
                    else DecisionKind.ACCEPT
                ),
                modification_norm=0.0,
                slack=min((m.margin for m in all_margins), default=0.0),
                progress_loss=0.0,
                solver_status="n/a",
                latency_ms=latency_ms,
                state_before=mode,
                state_after=mode,
                recovery_conditions=(),
                fallback_request=None,
                reject_reasons=tuple(accept_reasons),
                learning_modules_required=False,
                accepted_candidate=chosen,
            )
            events = self.validator.emit_decision_event(
                decision, availability, obs, phase=EventPhase.NORMAL
            )
        else:
            # --- REPAIR ---
            if PipelineStage.REPAIR.value not in stages:
                stages.append(PipelineStage.REPAIR.value)
            repaired_decision, repair_result, repair_events = repair_fn(
                candidate_set=candidate_set,
                obs=obs,
                now_s=now_s,
                availability=availability,
                mode=mode,
                reject_reasons=list(reject_pool),
                prefilter_ids=tuple(c.candidate_id for c in prefilter_passed),
                all_margins=all_margins,
                base_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
            events.extend(repair_events)
            if repaired_decision is not None and repair_result is not None and repair_result.success:
                decision = repaired_decision
                chosen = repaired_decision.accepted_candidate
                if repair_result.pre_repair_id and repair_result.pre_repair_id in audits_map:
                    a = audits_map[repair_result.pre_repair_id]
                    audits_map[repair_result.pre_repair_id] = CandidateAudit(
                        candidate_id=a.candidate_id,
                        source=a.source,
                        version_hint=a.version_hint,
                        prefilter_ok=a.prefilter_ok,
                        final_ok=False,
                        soft_score=a.soft_score,
                        reject_reasons=a.reject_reasons,
                        degradation=a.degradation,
                        repaired=True,
                        selected=True,
                    )
                notes.append(f"repaired_via_{repair_result.mode.value}")
            else:
                stages.append(PipelineStage.FALLBACK.value)
                fallback_res = self.validator.validate_candidates(
                    candidate_set,
                    obs,
                    now_s=now_s,
                    availability=availability,
                    mode=mode,
                    stage=ValidationStage.FINAL,
                )
                if fallback_res.accepted is not None:
                    decision = fallback_res.decision
                    chosen = fallback_res.accepted
                    events.extend(fallback_res.events)
                else:
                    decision = fallback_res.decision
                    decision = SafetyDecision(
                        decision_id=decision.decision_id,
                        run_id=decision.run_id,
                        frame_id=decision.frame_id,
                        prefilter_candidate_ids=tuple(c.candidate_id for c in prefilter_passed),
                        final_candidate_id=None,
                        pre_repair_trajectory_id=None,
                        post_repair_trajectory_id=None,
                        executed_trajectory_id=None,
                        constraint_margins=tuple(list(decision.constraint_margins) + all_margins),
                        decision_kind=decision.decision_kind,
                        modification_norm=0.0,
                        slack=decision.slack,
                        progress_loss=1.0,
                        solver_status=repair_result.solver_trace.status.value if repair_result else "n/a",
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        state_before=mode,
                        state_after=decision.state_after,
                        recovery_conditions=decision.recovery_conditions,
                        fallback_request=decision.fallback_request,
                        reject_reasons=tuple(
                            dict.fromkeys(list(reject_pool) + list(decision.reject_reasons))
                        ),
                        learning_modules_required=False,
                    )
                    events.extend(fallback_res.events)
                    notes.append("fallback_after_repair_fail")

        # --- SHADOW ---
        stages.append(PipelineStage.SHADOW.value)
        classic_seed = next(
            (c for c in candidate_set.candidates if c.source is CandidateSource.CLASSIC and c.availability),
            None,
        )
        # Also allow degraded-flagged classic from original for shadow compare if points exist.
        if classic_seed is None:
            classic_seed = next(
                (c for c in candidate_set.candidates if c.source is CandidateSource.CLASSIC),
                None,
            )
        shadow = run_classic_shadow(
            classic=classic_seed,
            executed=chosen,
            obs=obs,
            cfg=self.config,
            now_s=now_s,
        )
        stages.append(PipelineStage.DONE.value)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies_ms.append(latency_ms)
        # Share candidate-path latency with validator metrics (G2-01 snapshot contract).
        self.validator.record_candidate_latency_ms(latency_ms)
        arb_latency = latency_ms
        rec = self._record(
            candidate_set=candidate_set,
            stages=stages,
            ranked_ids=ranked_ids,
            selected_id=None if chosen is None else chosen.candidate_id,
            decision=decision,
            mode_before=mode,
            latency_ms=latency_ms,
            arb_latency_ms=arb_latency,
            shadow=shadow,
            audits=tuple(audits_map[cid] for cid in audits_map),
            notes=tuple(notes),
        )
        # Synthetic ValidatorResult for callers.
        cand_res = ValidatorResult(
            decision=decision,
            events=list(events),
            accepted=chosen,
            prefilter_passed=list(prefilter_passed),
            stage=ValidationStage.FINAL,
        )
        return PipelineTickResult(
            decision=decision,
            events=events,
            repair_result=repair_result,
            arbitration=rec,
            candidate_result=cand_res,
            accepted=chosen,
        )

    def _record(
        self,
        *,
        candidate_set: PolicyCandidateSet,
        stages: list[str],
        ranked_ids: tuple[str, ...],
        selected_id: str | None,
        decision: SafetyDecision,
        mode_before: SafetyMode,
        latency_ms: float,
        arb_latency_ms: float,
        shadow,
        audits: tuple[CandidateAudit, ...],
        notes: tuple[str, ...],
    ) -> ArbitrationRecord:
        rec = ArbitrationRecord(
            run_id=candidate_set.run_id,
            frame_id=candidate_set.frame_id,
            stages=tuple(stages),
            ranked_ids=ranked_ids,
            selected_id=selected_id,
            decision_kind=decision.decision_kind.value,
            mode_before=mode_before.value,
            mode_after=decision.state_after.value,
            latency_ms=latency_ms,
            arbitration_latency_ms=arb_latency_ms,
            fallback=decision.fallback_request,
            shadow=shadow,
            audits=audits,
            learning_required=decision.learning_modules_required,
            emergency_locked=decision.decision_kind is DecisionKind.EMERGENCY,
            notes=notes,
        )
        self._records.append(rec.to_dict())
        return rec
