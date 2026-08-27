"""Frozen degradation gates for unavailable / timeout / OOD / overconfident sources."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from safety_kernel.arbitration.types import DegradationReason
from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    CandidateSource,
    ComponentAvailability,
    PolicyCandidate,
    PolicyCandidateSet,
)


def apply_source_availability(
    cand: PolicyCandidate,
    availability: ComponentAvailability,
) -> DegradationReason:
    if not cand.availability:
        return DegradationReason.CANDIDATE_UNAVAILABLE
    if cand.source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW} and not availability.vla:
        return DegradationReason.SOURCE_UNAVAILABLE
    if cand.source is CandidateSource.CLASSIC and not availability.classic:
        return DegradationReason.SOURCE_UNAVAILABLE
    # World is not a candidate source enum yet; future-proof via dynamics_meta.
    if str(cand.dynamics_meta.get("ranked_by", "")).startswith("world") and not availability.world:
        return DegradationReason.SOURCE_UNAVAILABLE
    return DegradationReason.NONE


def apply_quality_gates(
    cand: PolicyCandidate,
    *,
    now_s: float,
    cfg: SafetyKernelConfig,
) -> DegradationReason:
    """Quality gates: soft-stale / OOD / overconfident for learning sources only.

    Classic / Shadow / Replay / Synthetic are NOT soft-staled here — hard freshness
    is enforced only by the Validator (max_candidate_age_s / valid_until). Explicit
    dynamics_meta[\"degrade\"] injection still works for any source (fault matrix).
    """
    arb = cfg.arbitration
    is_learning = cand.source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW}
    meta_flag = str(cand.dynamics_meta.get("degrade", "")).lower()
    # Explicit injection (fault matrix / red-team) — any source.
    if meta_flag == "timeout":
        return DegradationReason.TIMEOUT
    if meta_flag == "ood":
        return DegradationReason.OOD
    if meta_flag == "overconfident":
        return DegradationReason.OVERCONFIDENT
    if meta_flag == "soft_stale":
        return DegradationReason.SOFT_STALE

    if not is_learning:
        return DegradationReason.NONE

    age = now_s - cand.generated_time_s
    if age > arb.soft_stale_age_s:
        return DegradationReason.SOFT_STALE
    if cand.valid_until_s < now_s:
        return DegradationReason.TIMEOUT

    if cand.uncertainty >= arb.ood_uncertainty_min:
        return DegradationReason.OOD
    if (
        cand.probability >= arb.overconfident_prob_min
        and cand.uncertainty <= arb.overconfident_uncertainty_max
    ):
        return DegradationReason.OVERCONFIDENT
    return DegradationReason.NONE


def degrade_candidate_set(
    candidate_set: PolicyCandidateSet,
    availability: ComponentAvailability,
    *,
    now_s: float,
    cfg: SafetyKernelConfig,
) -> tuple[PolicyCandidateSet, dict[str, DegradationReason]]:
    """Return a copy with degraded candidates marked availability=False + reason map."""
    reasons: dict[str, DegradationReason] = {}
    out: list[PolicyCandidate] = []
    for cand in candidate_set.candidates:
        reason = apply_source_availability(cand, availability)
        if reason is DegradationReason.NONE:
            reason = apply_quality_gates(cand, now_s=now_s, cfg=cfg)
        reasons[cand.candidate_id] = reason
        if reason is DegradationReason.NONE:
            out.append(cand)
        else:
            # Mark unavailable so downstream hard precheck drops deterministically.
            out.append(
                PolicyCandidate(
                    candidate_id=cand.candidate_id,
                    source=cand.source,
                    generated_time_s=cand.generated_time_s,
                    valid_until_s=cand.valid_until_s,
                    probability=cand.probability,
                    points=cand.points,
                    behavior=cand.behavior,
                    critical_actor=cand.critical_actor,
                    conflict_type=cand.conflict_type,
                    risk_horizon_s=cand.risk_horizon_s,
                    intended_action=cand.intended_action,
                    uncertainty=cand.uncertainty,
                    availability=False,
                    dynamics_meta={
                        **dict(cand.dynamics_meta),
                        "degradation": reason.value,
                    },
                )
            )
    degraded_set = PolicyCandidateSet(
        run_id=candidate_set.run_id,
        frame_id=candidate_set.frame_id,
        scenario_id=candidate_set.scenario_id,
        model_id=candidate_set.model_id,
        carla_frame=candidate_set.carla_frame,
        simulation_time_s=candidate_set.simulation_time_s,
        wall_time_s=candidate_set.wall_time_s,
        candidates=tuple(out),
        schema_version=candidate_set.schema_version,
        coordinate_frame=candidate_set.coordinate_frame,
        preference_order=candidate_set.preference_order,
    )
    return degraded_set, reasons


def fallback_for_degradation(
    reason: DegradationReason,
    *,
    availability: ComponentAvailability,
) -> str:
    """Human-readable frozen mapping (does not execute Emergency from scores)."""
    if reason is DegradationReason.SOURCE_UNAVAILABLE:
        return "classic_if_available" if availability.classic else "minimal_risk"
    if reason in {DegradationReason.TIMEOUT, DegradationReason.SOFT_STALE}:
        return "hold_last_or_classic"
    if reason is DegradationReason.OOD:
        return "classic_or_repair"
    if reason is DegradationReason.OVERCONFIDENT:
        return "downrank_then_classic"
    return "none"
